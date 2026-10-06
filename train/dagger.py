"""DAgger: the model drives, the scripted expert labels every state the model actually reaches, retrain on the pile.

Why: SFT only ever saw states the expert visits. In closed loop the model drifts off them (walks onto the kill brick,
faces a wall) and has never seen how to recover, so errors compound (Ross et al. 2011). The expert reads privileged sim
state, so it can label any state, including the model's mistakes: this is on-policy distillation with a deterministic
teacher, the cheapest dense signal there is. GRPO only goes on top of a policy that already survives closed loop.

    python -m train.dagger --model models/sft --base-data data/sim_sft --out runs/dagger --rounds 3

Round k: roll out the current model on N envs per game in lockstep (one batched generate per EXEC ticks), with
probability beta_k = 0.5^k the expert drives a tick instead (so early rounds still reach later parts of each game),
label each replan point with the expert's next CHUNK ticks from a copy of that exact state, then LoRA-SFT the current
model on base data + every round so far and merge. The rollouts double as a closed-loop eval: each round logs
the mean score per game with beta=0 on held-out seeds.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import numpy as np

import sim
from model.vlm import CHUNK, VLMRunner, encode, prompt
from train.dataset import DECAY, _completion, _kw, keep_rows, save
from train.eval import bench, rollouts

LABEL_EVERY = 4   # ticks between labelled states; neighbours are near-duplicates and the expert lookahead is the cost


def expert_chunk(env) -> str:
    """The expert's next CHUNK inputs from exactly this state, played out on a copy."""
    e = pickle.loads(pickle.dumps(env))
    start, acts = e.cursor, []
    for _ in range(CHUNK):
        acts.append(e.expert())
        e.step(acts[-1])
        if e.done:
            break
    acts += [acts[-1]] * (CHUNK - len(acts))
    return encode(acts, start)


def closed_loop(path: str, games: list[str], per_game: int, ticks: int, device: str) -> dict:
    """The benchmark (train.eval.bench) on the current model: same seeds and metric as any other model."""
    return bench(VLMRunner(path, device=device), games, per_game, ticks, device)


def collect(model: str, games: list[str], per_game: int, ticks: int, beta: float, seed: int, device: str) -> list[dict]:
    """The model drives (the expert takes a tick with prob beta); the expert labels where it actually went."""
    runner = VLMRunner(model, device=device)
    _, labels = rollouts(runner, games, per_game, ticks, seed, device, env_kw=lambda g, rng: {**_kw(g, rng), "max_steps": 10 ** 9},
                         beta=beta, label_fn=expert_chunk, label_every=LABEL_EVERY)
    rng = np.random.default_rng(seed)
    rows = []
    for env_labels in labels:
        texts = [t for t, _ in env_labels]
        rows += [{"prompt": prompt(), "images": pair, "completion": _completion(t)}
                 for (t, pair), k in zip(env_labels, keep_rows(texts, rng, DECAY)) if k]
    return rows


def main(argv=None, git=None):
    from train import runlog, sft
    from train.grpo import merge
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the SFT model to start from (full weights, not an adapter)")
    ap.add_argument("--base-data", required=True, help="the SFT data; every round trains on it plus all DAgger rows")
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", default=",".join(sim.ENVS))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--per-game", type=int, default=4, help="envs per game per round")
    ap.add_argument("--ticks", type=int, default=600)
    ap.add_argument("--eval-per-game", type=int, default=2)
    ap.add_argument("--eval-ticks", type=int, default=300)
    ap.add_argument("--sft-args", default="", help="passed through to train.sft, e.g. '--max-steps 500'")
    ap.add_argument("--eval-only", default="", help="comma-separated model dirs: just the closed-loop eval of each, into <out>/eval.json")
    args = ap.parse_args(argv)
    import shlex
    import torch
    from sim.render import best_device
    device = "cuda" if torch.cuda.is_available() else best_device()
    runlog.start(args.out, vars(args), git)
    games = args.games.split(",")
    if args.eval_only:
        res = {m: closed_loop(m, games, args.eval_per_game, args.eval_ticks, device) for m in args.eval_only.split(",")}
        json.dump(res, open(os.path.join(args.out, "eval.json"), "w"), indent=1)
        print(json.dumps(res), flush=True)
        return
    # Every stage leaves its output on disk and is skipped if it's already there, so a preempted container picks up
    # where it stopped (Modal restarted an L4 mid-rollout once and the first version lost 50 min).
    logf = os.path.join(args.out, "dagger.json")
    log = json.load(open(logf)) if os.path.exists(logf) else []
    model, data = args.model, [args.base_data]
    for k in range(args.rounds + 1):
        if len(log) <= k:
            t0 = time.time()
            ev = closed_loop(model, games, args.eval_per_game, args.eval_ticks, device)
            log.append({"round": k, "model": model, "eval": ev, "eval_s": round(time.time() - t0)})
            print(json.dumps(log[-1]), flush=True)
            json.dump(log, open(logf, "w"), indent=1)
        if k == args.rounds:
            break
        d = os.path.join(args.out, f"round{k}_data")
        if not os.path.exists(os.path.join(d, "dataset_info.json")):
            t0 = time.time()
            rows = collect(model, games, args.per_game, args.ticks, 0.5 ** (k + 1), 1000 * (k + 1), device)
            save(rows, d)
            log[k] |= {"rollout_rows": len(rows), "rollout_s": round(time.time() - t0)}
            json.dump(log, open(logf, "w"), indent=1)
        data.append(d)
        adapter, nxt = os.path.join(args.out, f"round{k}_lora"), os.path.join(args.out, f"round{k}")
        if not os.path.exists(os.path.join(nxt, "config.json")):
            sft.main(["--model", model, "--data", ",".join(data), "--lora", "--out", adapter] + shlex.split(args.sft_args), git)
            merge(adapter, nxt)
        model = nxt


if __name__ == "__main__":
    main()
