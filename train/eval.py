"""The benchmark: any policy, closed loop, on fixed held-out seeds of all ten sims, next to noop / random / expert.

    python -m train.eval --model runs/sft_v2                         # a trained model dir (or LoRA dir)
    python -m train.eval --model Qwen/Qwen2.5-VL-7B-Instruct         # any HF VLM, zero-shot
    python -m train.eval --api qwen/qwen3-vl-8b-instruct             # any OpenRouter VLM, zero-shot (.env key)
    python -m train.eval --table                                     # leaderboard of everything in results/bench/

Every policy gets the same prompt (model.vlm.SYSTEM + two frames) and must answer in the action text format; the
episode runs in lockstep with the sim (generation time doesn't cost game time, latency is reported separately).
Per game: score() is the game's own number (stages, rooms, net worth...) but moves slowly, so the headline is the
summed env reward, which includes partial progress, normalized as (policy - noop) / (expert - noop): 0 = does
nothing useful, 1 = the scripted expert. "walk" (hold W) is listed too: in short episodes walking forward alone
scores well on doors and rivals, so that's the bar a model has to clear. brainrot pays nothing in the first 30 s even for the expert, so it has no
norm at the default length. Seeds EVAL_SEED.. are never used for training data.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time
from collections import deque

import numpy as np

import sim
from core.actions import HEADS, Action
from model.vlm import EXEC, FRAME_GAP, OBS_H, OBS_W, VLMRunner, decode, encode, image

EVAL_SEED = 999_000   # training seeds never reach this range
BENCH_DIR = "results/bench"


def _random_action(rng) -> Action:
    ids = {k: (rng.random(n) < 0.15).astype(np.float32) if k == "keys" else int(rng.integers(n)) for k, n in HEADS.items()}
    ids["click"] = int(rng.random() < 0.05)
    return Action.from_ids(ids)


def rollouts(policy, games: list[str], per_game: int, ticks: int, seed: int, device: str, env_kw=None,
             beta: float = 0.0, label_fn=None, label_every: int = 4, gen_batch: int = 16):
    """All envs step in lockstep; every EXEC ticks one batched policy call for all of them.
    beta: per-tick chance the expert drives instead (DAgger). label_fn(env) -> text is recorded every label_every
    ticks with the frame pair the policy saw. → ({game: [(score, summed reward)]}, [[(text, pair)] per env])."""
    from sim.render import render
    rng = np.random.default_rng(seed)
    envs = [(g, sim.make(g, seed=seed + i, **(env_kw(g, rng) if env_kw else {"max_steps": 10 ** 9, **sim.ENVS[g].eval_kw})))
            for g in games for i in range(per_game)]
    hist = [deque(maxlen=FRAME_GAP + 1) for _ in envs]
    queues: list[list] = [[] for _ in envs]
    labels: list[list] = [[] for _ in envs]
    ret = [0.0] * len(envs)
    for t in range(ticks):
        frames = render([e.scene() for _, e in envs], OBS_H, OBS_W, device=device).cpu().numpy()   # (B, H, W, 3)
        for h, f in zip(hist, frames):
            h.append(f)
        if t % EXEC == 0:
            pairs = [[image(h[0]), image(h[-1])] for h in hist]
            out = []
            for i in range(0, len(pairs), gen_batch):
                out += policy.generate_batch(pairs[i:i + gen_batch])
            for i, ((g, e), text) in enumerate(zip(envs, out)):
                acts, ok = decode(text, e.cursor)
                policy.parse_fail += not ok
                policy.calls += 1
                queues[i] = acts[:EXEC]
                if label_fn and t % label_every == 0:
                    labels[i].append((label_fn(e), pairs[i]))
        for i, (g, e) in enumerate(envs):
            a = queues[i].pop(0) if queues[i] else None
            if a is None or (beta and rng.random() < beta):
                a = e.expert()
            ret[i] += e.step(a)[0]
    scores: dict[str, list] = {}
    for (g, e), r in zip(envs, ret):
        scores.setdefault(g, []).append((float(e.score()), r))
    return scores, labels


def scripted(game: str, who: str, per_game: int, ticks: int) -> list:
    """(score, summed reward) per seed for noop / random / walk (hold W) / expert. No rendering needed."""
    out = []
    for i in range(per_game):
        env = sim.make(game, seed=EVAL_SEED + i, max_steps=10 ** 9, **sim.ENVS[game].eval_kw)
        rng = np.random.default_rng(i)
        pick = {"expert": lambda: env.expert(), "random": lambda: _random_action(rng), "noop": Action,
                "walk": lambda: Action(keys=frozenset({"w"}))}[who]
        r = sum(env.step(pick())[0] for _ in range(ticks))
        out.append((float(env.score()), r))
    return out


def bench(policy, games: list[str], per_game: int = 2, ticks: int = 300, device: str = "cpu") -> dict:
    """The policy next to noop / random / expert on the same seeds; one row per game plus the mean norm."""
    t0 = time.time()
    pol, _ = rollouts(policy, games, per_game, ticks, EVAL_SEED, device)
    wall = time.time() - t0
    res = {}
    for g in games:
        p, n, r, w, e = (np.mean(x, axis=0) for x in (pol[g], *(scripted(g, b, per_game, ticks) for b in ("noop", "random", "walk", "expert"))))
        span = e[1] - n[1]
        nz = lambda x: float((x - n[1]) / span) if abs(span) > 1e-6 else None
        res[g] = {"score": float(p[0]), "score_noop": float(n[0]), "score_random": float(r[0]), "score_expert": float(e[0]),
                  "ret": float(p[1]), "ret_noop": float(n[1]), "ret_random": float(r[1]), "ret_expert": float(e[1]),
                  "ret_walk": float(w[1]), "norm": nz(p[1]), "norm_random": nz(r[1]), "norm_walk": nz(w[1]), "n": per_game}
    mean = lambda k: float(np.mean([v[k] for v in res.values() if v[k] is not None]))
    res["_summary"] = {"mean_norm": mean("norm"), "mean_norm_random": mean("norm_random"), "mean_norm_walk": mean("norm_walk"),
                       "games": sum(v["norm"] is not None for v in res.values()),
                       "parse_fail": policy.parse_fail / max(1, policy.calls), "calls": policy.calls,
                       "api_latency_s": getattr(policy, "latency", 0.0) / max(1, policy.calls), "api_errors": getattr(policy, "errors", 0),
                       "wall_s": round(wall),
                       "per_game": per_game, "ticks": ticks}
    return res


def openloop(policy, games: list[str], ticks: int = 400) -> dict:
    """On held-out *expert* states: does the first tick's held-key set match the expert's? Split into rows where the
    expert keeps the previous keys and rows where it changes them (jump, turn, stop, fire). A model that's fine on the
    first and bad on the second can't see the decision, which DAgger won't fix; good on both but bad closed loop
    is compounding error, which DAgger does fix."""
    from train.dataset import sim_rows
    keys = lambda text: decode(text)[0][0].keys
    res = {}
    for g in games:
        rows = sim_rows(g, EVAL_SEED + 3, ticks, noise=0.0, decay=1.0)
        labs = [r["completion"][0]["content"][0]["text"] for r in rows]
        outs = []
        for i in range(0, len(rows), 16):
            outs += policy.generate_batch([r["images"] for r in rows[i:i + 16]])
        hit = {"same": [], "change": []}
        for i in range(1, len(rows)):
            hit["same" if keys(labs[i]) == keys(labs[i - 1]) else "change"].append(keys(outs[i]) == keys(labs[i]))
        res[g] = {k: float(np.mean(v)) if v else None for k, v in hit.items()} | {"n_change": len(hit["change"])}
    for k in ("same", "change"):
        res["_" + k] = float(np.mean([v[k] for g, v in res.items() if not g.startswith("_") and v[k] is not None]))
    return res


class APIRunner:
    """The same prompt, answered by a hosted VLM through OpenRouter: a zero-shot baseline."""

    def __init__(self, model: str, workers: int = 16):
        from model.vlm import SYSTEM
        self.model, self.workers, self.system = model, workers, SYSTEM
        self.key = os.environ.get("OPENROUTER_API_KEY") or _dotenv().get("OPENROUTER_API_KEY")
        assert self.key, "OPENROUTER_API_KEY missing from env and .env"
        self.calls, self.parse_fail, self.latency, self.errors = 0, 0, 0.0, 0
        self._post({"model": model, "max_tokens": 5, "messages": [{"role": "user", "content": "ok"}]})   # bad key/model: fail now, not score 3000 no-ops

    def _post(self, body: dict) -> str:
        import urllib.request
        req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)["choices"][0]["message"]["content"] or ""

    def generate_batch(self, batch: list[list]) -> list[str]:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(self._one, batch))

    def _one(self, imgs) -> str:
        import base64
        import io

        def url(im):
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=90)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        body = {"model": self.model, "max_tokens": 80, "temperature": 0,
                "messages": [{"role": "system", "content": self.system},
                             {"role": "user", "content": [*({"type": "image_url", "image_url": {"url": url(im)}} for im in imgs),
                                                          {"type": "text", "text": "Your next 4 ticks. Reply with only the 4 ticks."}]}]}
        for attempt in range(4):   # rate limits and provider hiccups: back off, then give up to a no-op chunk
            t0 = time.time()
            try:
                out = self._post(body)
                self.latency += time.time() - t0
                return out
            except Exception:
                time.sleep(2 ** attempt)
        self.errors += 1
        return ""


def _dotenv() -> dict:
    try:
        return dict(line.strip().split("=", 1) for line in open(".env") if "=" in line and not line.startswith("#"))
    except OSError:
        return {}


def table(d: str = BENCH_DIR) -> str:
    """Leaderboard: one row per benchmarked policy, mean norm then per-game norms."""
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(p))
        rows.append((r["_summary"]["mean_norm"], os.path.basename(p)[:-5], r))
    if not rows:
        return "no results in " + d
    games = [g for g in rows[0][2] if not g.startswith("_")]
    fmt = lambda x: "  -  " if x is None else f"{x:5.2f}"
    head = f"{'policy':38s} {'mean':>5s} " + " ".join(f"{g[:7]:>7s}" for g in games) + "  parse-fail  open-loop same/change"
    lines = [head, "-" * len(head)]
    r0 = rows[0][2]
    for b in ("random", "walk"):   # scripted baselines, same seeds: a model has to beat holding W, not just noop
        lines.append(f"{'(' + b + ')':38s} {r0['_summary']['mean_norm_' + b]:5.2f} " + " ".join(f"{fmt(r0[g]['norm_' + b]):>7s}" for g in games))
    for m, name, r in sorted(rows, key=lambda x: -x[0]):
        lines.append(f"{name[:38]:38s} {m:5.2f} " + " ".join(f"{fmt(r[g]['norm']):>7s}" for g in games)
                     + f"  {r['_summary']['parse_fail']:9.0%}" + (f"  {r['_openloop']['_same']:.2f}/{r['_openloop']['_change']:.2f}" if "_openloop" in r else ""))
    lines.append("norm: 0 = noop, 1 = scripted expert, on summed env reward over the same held-out seeds")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="model dir, LoRA dir, or HF id")
    ap.add_argument("--api", default=None, help="OpenRouter model id, e.g. qwen/qwen3-vl-8b-instruct")
    ap.add_argument("--games", default=",".join(sim.ENVS))
    ap.add_argument("--per-game", type=int, default=2)
    ap.add_argument("--ticks", type=int, default=300)
    ap.add_argument("--name", default=None, help="leaderboard row name (default: from --model/--api)")
    ap.add_argument("--out", default=BENCH_DIR)
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--openloop", action="store_true", help="also key accuracy on expert states, split same/changed keys")
    args = ap.parse_args(argv)
    if args.table:
        print(table(args.out))
        return
    from sim.render import best_device
    dev = best_device()
    policy = APIRunner(args.api) if args.api else VLMRunner(args.model, device=dev)
    res = bench(policy, args.games.split(","), args.per_game, args.ticks, dev)
    if args.openloop:
        res["_openloop"] = openloop(policy, args.games.split(","))
    name = args.name or (args.api or os.path.basename(os.path.normpath(args.model))).replace("/", "_")
    os.makedirs(args.out, exist_ok=True)
    json.dump(res | {"_policy": args.api or args.model}, open(os.path.join(args.out, name + ".json"), "w"), indent=1)
    print(table(args.out))
    return res


if __name__ == "__main__":
    main()
