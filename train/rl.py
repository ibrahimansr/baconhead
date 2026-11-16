"""RL pathway, separate from imitation (train/sft.py + train/dagger.py): Backplay curriculum, end-only rewards.

Reward is 1 if the policy beats the milestone (the game's own score() goes up: next obby stage, next door, a won
round, a sale, a survived night) before its horizon runs out, else 0. Nothing else: no shaping, no potential, so
nothing to hack by standing still (v3) or holding W (v4).

Sparse end rewards are only learnable if success is sometimes reachable, so episodes start from the expert's own run,
k decisions (0.2 s each) before it beat the milestone: the expert played the prefix, the policy plays the rest
(Backplay, Resnick et al. 2018; Salimans & Chen 2018). Per game, k starts at 2 and grows by 50% whenever the policy
beats the milestone in >60% of its last 24 episodes: it gets harder over time until the policy plays whole levels.

Update (GRPO over whole episodes): G rollouts from the same start; advantage A = (r - mean) / std per group, groups
where all G tie are skipped. Every token the policy generated in the episode gets its episode's A. Loss per token:
    -min(rho * A, clip(rho, 1-eps, 1+eps) * A) + beta * KL,   rho = pi(token) / pi_old(token)
with EPOCHS passes over each batch (so the clip binds from the second pass), KL (k3 estimator) to the frozen start
model (LoRA disabled). LoRA r=64 on all linear layers.

    python -m train.rl --model Qwen/Qwen3.5-4B --out runs/rl_v1 --steps 300
    python -m train.rl --model runs/sft_plan_v1 --out runs/rl_from_il --steps 300     # same RL, IL start
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time

import numpy as np
import torch

import sim
from model.vlm import CHUNK, EXEC, OBS_H, OBS_W, decode, image, processor, prompt, split_plan
from train import runlog

GAMES = [g for g in sim.ENVS if g != "brainrot"]   # brainrot's first score change is ~45 s in: no milestones in reach


def milestones(game: str, seed: int, ticks: int = 3000) -> list[tuple[int, list[bytes]]]:
    """The expert plays one episode; for every time its score() went up, the snapshots of the decisions before it.
    → [(milestone tick, [state bytes at tick m - EXEC, m - 2*EXEC, ...])]"""
    env = sim.make(game, seed=seed, max_steps=10 ** 9)
    snaps, out, best = {}, [], env.score()
    for t in range(ticks):
        if t % EXEC == 0:
            snaps[t] = pickle.dumps(env)
        env.step(env.expert())
        if env.score() > best + 1e-9:
            best = env.score()
            m = (t // EXEC) * EXEC
            back = [snaps[m - j * EXEC] for j in range(m // EXEC + 1) if m - j * EXEC in snaps]
            out.append((t, back))
        if env.done:
            break
    return out


class Curriculum:
    """Per game: k = decisions of the level the policy plays itself. Grows when the policy wins >60% at this k."""

    def __init__(self, games, k0=2, grow=1.5, window=24, bar=0.6, kmax=150):
        self.k = {g: k0 for g in games}
        self.hist = {g: [] for g in games}
        self.grow, self.window, self.bar, self.kmax = grow, window, bar, kmax

    def update(self, game, wins: list[float]):
        h = self.hist[game]
        h += wins
        del h[:-self.window]
        if len(h) >= self.window and np.mean(h) > self.bar:
            self.k[game] = min(self.kmax, math.ceil(self.k[game] * self.grow))
            h.clear()

    def state(self):
        return {"k": self.k, "recent": {g: float(np.mean(h)) if h else None for g, h in self.hist.items()}}


def rollout(policy, proc, game: str, start: bytes, k: int, G: int, device: str, temperature: float = 1.0):
    """G copies of the start state played by the policy in lockstep for up to horizon(k) decisions.
    → per env: reward (1 = beat the milestone), [(model inputs for the prompt, completion token ids)]"""
    from sim.render import render
    envs = [pickle.loads(start) for _ in range(G)]
    score0 = envs[0].score()
    horizon = k + max(4, k // 2)                       # slack: the policy may be slower than the expert
    plans, alive = [""] * G, [True] * G
    wins, steps = [0.0] * G, [[] for _ in range(G)]
    prev = None
    for _ in range(horizon):
        idx = [i for i in range(G) if alive[i]]
        if not idx:
            break
        frames = render([envs[i].scene() for i in idx], OBS_H, OBS_W, device=device).cpu().numpy()
        now = [image(f) for f in frames]
        before = [prev.get(i, n) if prev else n for i, n in zip(idx, now)]   # 0.2 s ago (the decision before)
        texts = [proc.apply_chat_template(prompt(game, plans[i]), add_generation_prompt=True, tokenize=False) for i in idx]
        proc.tokenizer.padding_side = "left"
        inp = proc(text=texts, images=[[b, n] for b, n in zip(before, now)], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = policy.generate(**inp, max_new_tokens=96, do_sample=True, temperature=temperature,
                                  eos_token_id=[proc.tokenizer.convert_tokens_to_ids("<|im_end|>"), proc.tokenizer.eos_token_id])
        comp = out[:, inp["input_ids"].shape[1]:]
        prev = {}
        for j, i in enumerate(idx):
            ids = comp[j]
            stop = (ids == proc.tokenizer.pad_token_id).nonzero()
            ids = ids[: stop[0, 0]] if len(stop) else ids
            text = proc.decode(ids, skip_special_tokens=True)
            single = proc(text=[texts[j]], images=[[before[j], now[j]]], return_tensors="pt")   # unpadded, for training
            steps[i].append(({k_: v.cpu() for k_, v in single.items()}, ids.cpu()))
            plans[i] = split_plan(text)[0] or plans[i]
            acts, _ = decode(text, envs[i].cursor)
            for a in acts[:EXEC]:
                envs[i].step(a)
            if envs[i].score() > score0 + 1e-9:        # beat the milestone: the only reward
                wins[i], alive[i] = 1.0, False
            elif envs[i].done:
                alive[i] = False
            prev[i] = now[j]
    return wins, steps


def token_logprobs(model, inputs: dict, comp: torch.Tensor, device: str) -> torch.Tensor:
    """log pi(each completion token | prompt with images, earlier completion tokens) → (len(comp),)"""
    ids = torch.cat([inputs["input_ids"][0], comp]).unsqueeze(0).to(device)
    L = inputs["input_ids"].shape[1]
    kw = {}
    for k, v in inputs.items():
        if k in ("input_ids", "attention_mask"):
            continue
        if v.dim() == 2 and v.shape == (1, L):   # per-token side inputs (Qwen3.5's mm_token_type_ids): completion = text = 0
            v = torch.cat([v, torch.zeros(1, len(comp), dtype=v.dtype)], 1)
        kw[k] = v.to(device)
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), **kw).logits[0, inputs["input_ids"].shape[1] - 1:-1]
    return torch.log_softmax(logits.float(), -1).gather(-1, comp.to(device)[:, None])[:, 0]   # (T,)


def main(argv=None, git=None):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForImageTextToText
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B", help="base for the RL pathway; an IL checkpoint for the comparison run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--games", default=",".join(GAMES))
    ap.add_argument("--group", type=int, default=8, help="G rollouts per start state")
    ap.add_argument("--starts", type=int, default=2, help="start states per step")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=2, help="passes per batch; >1 makes the PPO clip bind")
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--k0", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=4, help="expert episodes per game for start states")
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    runlog.start(args.out, vars(args), git)
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    games = args.games.split(",")

    proc = processor(args.model)
    base = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16 if cuda else torch.float32)
    last = os.path.join(args.out, "last")
    if os.path.exists(os.path.join(last, "adapter_config.json")):   # resume: adapter + curriculum + step
        model = PeftModel.from_pretrained(base, last, is_trainable=True)
        state = json.load(open(os.path.join(last, "rl_state.json")))
    else:
        model = get_peft_model(base, LoraConfig(r=64, lora_alpha=128, target_modules="all-linear", task_type="CAUSAL_LM"))
        state = {"step": 0, "curriculum": None}
    model.to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    cur = Curriculum(games, k0=args.k0)
    if state["curriculum"]:
        cur.k.update(state["curriculum"]["k"])

    print("collecting expert milestones...", flush=True)
    pool = {g: [m for s in range(args.seeds) for m in milestones(g, 50_000 + s)] for g in games}
    games = [g for g in games if pool[g]]
    print({g: len(pool[g]) for g in games}, flush=True)
    logf = open(os.path.join(args.out, "rl_log.jsonl"), "a")

    for step in range(state["step"], args.steps):
        t0 = time.time()
        model.eval()
        batch, log = [], {"step": step + 1, "games": {}}
        for _ in range(args.starts):
            g = str(rng.choice(games))
            m_tick, back = pool[g][int(rng.integers(len(pool[g])))]
            k = min(cur.k[g], len(back) - 1)
            wins, steps = rollout(model, proc, g, back[k], k, args.group, device)
            cur.update(g, wins)
            log["games"].setdefault(g, []).append({"k": k, "win": float(np.mean(wins))})
            r = np.array(wins)
            if r.std() < 1e-6:            # all tied: no advantage, no gradient
                continue
            adv = (r - r.mean()) / (r.std() + 1e-6)
            batch += [(inp, comp, float(a)) for a, ep in zip(adv, steps) for inp, comp in ep if len(comp)]
        log["tokens"] = sum(len(c) for _, c, _ in batch)
        if batch:
            model.train()
            with torch.no_grad():         # pi_old and the reference (LoRA off) for every sampled token
                old = [token_logprobs(model, i, c, device).detach() for i, c, _ in batch]
                with model.disable_adapter():
                    ref = [token_logprobs(model, i, c, device).detach() for i, c, _ in batch]
            stats = []
            for _ in range(args.epochs):
                for (inp, comp, a), lo, lr_ in zip(batch, old, ref):
                    lp = token_logprobs(model, inp, comp, device)
                    rho = torch.exp(lp - lo)
                    pg = -torch.min(rho * a, torch.clamp(rho, 1 - args.eps, 1 + args.eps) * a)
                    kl = torch.exp(lr_ - lp) - (lr_ - lp) - 1          # k3: unbiased, >= 0
                    loss = (pg + args.beta * kl).sum() / log["tokens"]
                    loss.backward()
                    stats.append((float(kl.mean()), float(((rho - 1).abs() > args.eps).float().mean())))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
            log["kl"] = float(np.mean([s[0] for s in stats]))
            log["clip_frac"] = float(np.mean([s[1] for s in stats]))
        log |= {"curriculum": cur.state(), "sec": round(time.time() - t0, 1)}
        logf.write(json.dumps(log) + "\n")
        logf.flush()
        print(json.dumps(log), flush=True)
        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            model.save_pretrained(last)
            proc.save_pretrained(last)
            json.dump({"step": step + 1, "curriculum": cur.state()}, open(os.path.join(last, "rl_state.json"), "w"))
            if (step + 1) % (4 * args.save_every) == 0:
                model.save_pretrained(os.path.join(args.out, f"step{step + 1}"))


if __name__ == "__main__":
    main()
