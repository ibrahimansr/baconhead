"""Evaluation: goal completion on held-out levels (the headline), the reward benchmark, and open-loop accuracy.

    python -m train.eval --model runs/il --play 1500 --seeds 5 --openloop     # a model dir (or LoRA dir)
    python -m train.eval --table                                              # leaderboard of results/bench/
    python -m train.eval --video results/bench/play_il                        # mp4s from saved inputs (local)

--play: the model plays every game's gold goal (sim.gold: "reach room 10 alive", "catch a fish and sell it") on
`seeds` held-out levels (seed 0 is the gold level itself) for `ticks` ticks, all envs in lockstep; the result is the
fraction of levels where the goal was reached, plus every input and generation saved for videos. One level per
game can't tell 1.5/10 from 3/10, so use 5 seeds for anything you'll compare. API models (gpt-6.1-sol, Qwen3.5
zero-shot) play the same levels through train/frontier.py.

Every policy gets the same prompt (model.vlm.prompt + two frames) and must answer in the action text format; the
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
from model.vlm import EXEC, FRAME_GAP, OBS_H, OBS_W, VLMRunner, decode, encode, image, split_plan

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
    plans = [""] * len(envs)   # each env's last PLAN line from the policy: its memory, fed back in the next prompt
    ret = [0.0] * len(envs)
    for t in range(ticks):
        frames = render([e.scene() for _, e in envs], OBS_H, OBS_W, device=device).cpu().numpy()   # (B, H, W, 3)
        for h, f in zip(hist, frames):
            h.append(f)
        if t % EXEC == 0:
            pairs = [[image(h[0]), image(h[-1])] for h in hist]
            ctx = [(g, pl) for (g, _), pl in zip(envs, plans)]
            out = []
            for i in range(0, len(pairs), gen_batch):
                out += policy.generate_batch(pairs[i:i + gen_batch], ctx[i:i + gen_batch])
            for i, ((g, e), text) in enumerate(zip(envs, out)):
                acts, ok = decode(text, e.cursor)
                policy.parse_fail += not ok
                policy.calls += 1
                queues[i] = acts[:EXEC]
                if label_fn and t % label_every == 0:
                    labels[i].append((label_fn(e), pairs[i], ctx[i]))
                plans[i] = split_plan(text)[0] or plans[i]
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


def play(policy, games: list[str], ticks: int, device: str, out: str, seeds: int = 1) -> dict:
    """The model on each game's gold goal and settings (sim.gold) on `seeds` held-out levels, all envs in lockstep.
    Saves <out>/<game>_s<k>.json in the gold format (the played inputs as text, plus every generation), so
    `python -m train.eval --video <out>` replays them deterministically into mp4s. Records the first tick the goal held.
    → {game: {"success": fraction of levels, "levels": [per level]}}"""
    from sim.gold import PLANS, SEED
    from sim.render import render
    from model.vlm import encode_tick
    runs = [(g, k) for g in games for k in range(seeds)]
    envs = [sim.make(g, seed=SEED + k, max_steps=10 ** 9, **PLANS[g][0]) for g, k in runs]
    games = [g for g, _ in runs]
    hist = [deque(maxlen=FRAME_GAP + 1) for _ in envs]
    queues, texts, first = [[] for _ in envs], [[] for _ in envs], [None] * len(envs)
    raw = [[] for _ in envs]   # every generation as (tick, text): the trace, including ticks it never got to play
    plans = [""] * len(envs)
    for t in range(ticks):
        frames = render([e.scene() for e in envs], OBS_H, OBS_W, device=device).cpu().numpy()
        for h, f in zip(hist, frames):
            h.append(f)
        if t % EXEC == 0:
            outs = policy.generate_batch([[image(h[0]), image(h[-1])] for h in hist], list(zip(games, plans)))
            for i, (e, o) in enumerate(zip(envs, outs)):
                plans[i] = split_plan(o)[0] or plans[i]
                acts, ok = decode(o, e.cursor)
                policy.parse_fail += not ok
                policy.calls += 1
                queues[i] = acts[:EXEC]
                raw[i].append((t, o))
        for i, (g, e) in enumerate(zip(games, envs)):
            a = queues[i].pop(0) if queues[i] else Action(cursor=e.cursor)
            texts[i].append(encode_tick(a, e.cursor))
            e.step(decode(texts[i][-1], e.cursor)[0][0])   # exactly what replaying the text does
            if first[i] is None and PLANS[g][2](e):
                first[i] = t + 1
    os.makedirs(out, exist_ok=True)
    res = {}
    for (g, k), e, tx, f, rw in zip(runs, envs, texts, first, raw):
        lv = {"seed": SEED + k, "reached_at": f, "score": float(e.score()), "deaths": int(e.deaths), "info": e.info()}
        res.setdefault(g, {"goal": PLANS[g][1], "levels": []})["levels"].append(lv)
        json.dump({"game": g, "kw": PLANS[g][0], "ticks": len(tx), "inputs": tx, "raw": rw, "goal": PLANS[g][1]} | lv,
                  open(os.path.join(out, f"{g}_s{k}.json"), "w"))
    for v in res.values():
        v["success"] = float(np.mean([lv["reached_at"] is not None for lv in v["levels"]]))
    res["_success"] = float(np.mean([v["success"] for v in res.values()]))
    return res


def videos(d: str) -> None:
    """Every <d>/<game>.json from play() → <d>/<game>.mp4, replayed and rendered locally."""
    from sim.gold import render_video
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(p))
        if "inputs" in r:
            name = os.path.basename(p)[:-5]
            print(render_video(r["game"], r["inputs"], r["kw"], out=d, seed=r.get("seed"), name=name), "goal",
                  "REACHED at tick %s" % r["reached_at"] if r["reached_at"] else "not reached")


def openloop(policy, games: list[str], ticks: int = 400) -> dict:
    """On held-out *expert* states: does the first tick's held-key set match the expert's? Split into rows where the
    expert keeps the previous keys and rows where it changes them (jump, turn, stop, fire). A model that's fine on the
    first and bad on the second can't see the decision, which DAgger won't fix; good on both but bad closed loop
    is compounding error, which DAgger does fix."""
    from train.dataset import game_rows
    keys = lambda text: decode(text)[0][0].keys
    prev = lambda row: row["prompt"][1]["content"][2]["text"].split("Your previous plan: ")[1].split("\n")[0]
    res = {}
    for g in games:
        rows = game_rows(g, EVAL_SEED + 3, ticks, noise=0.0, decay=1.0)
        labs = [r["completion"][0]["content"][0]["text"] for r in rows]
        outs = []
        for i in range(0, len(rows), 16):
            outs += policy.generate_batch([r["images"] for r in rows[i:i + 16]],
                                          [(g, "" if prev(r) == "none yet" else prev(r)) for r in rows[i:i + 16]])
        hit = {"same": [], "change": []}
        for i in range(1, len(rows)):
            hit["same" if keys(labs[i]) == keys(labs[i - 1]) else "change"].append(keys(outs[i]) == keys(labs[i]))
        plan_hit = [split_plan(o)[0] == split_plan(lab)[0] for o, lab in zip(outs, labs)]
        # the copycat test: rows where the expert's plan differs from the previous one (the prompt shows the old plan)
        flip = [h for h, r, lab in zip(plan_hit, rows, labs) if split_plan(lab)[0] != prev(r)]
        res[g] = {k: float(np.mean(v)) if v else None for k, v in hit.items()} | {"n_change": len(hit["change"]),
                  "plan": float(np.mean(plan_hit)), "plan_switch": float(np.mean(flip)) if flip else None, "n_switch": len(flip)}
    for k in ("same", "change", "plan", "plan_switch"):
        res["_" + k] = float(np.mean([v[k] for g, v in res.items() if not g.startswith("_") and v[k] is not None]))
    return res


def table(d: str = BENCH_DIR) -> str:
    """Leaderboard, one row per evaluated policy: gold-goal success per game (the headline), then the reward
    benchmark's mean norm and open-loop key accuracy where they were run."""
    rows = [(os.path.basename(p)[:-5], json.load(open(p))) for p in sorted(glob.glob(os.path.join(d, "*.json")))]
    if not rows:
        return "no results in " + d
    games = list(sim.ENVS)
    pct = lambda x: "    -" if x is None else f"{x:5.0%}"
    head = f"{'policy':30s} {'goals':>5s} " + " ".join(f"{g[:7]:>7s}" for g in games) + "   norm  open-loop same/change/plan-switch"
    lines = [head, "-" * len(head)]
    goal = lambda r: r.get("_play", {}).get("_success")
    for name, r in sorted(rows, key=lambda x: -(goal(x[1]) or -1)):
        pl, ol = r.get("_play", {}), r.get("_openloop", {})
        norm = r.get("_summary", {}).get("mean_norm")
        lines.append(f"{name[:30]:30s} {pct(goal(r))} " + " ".join(f"{pct(pl.get(g, {}).get('success')):>7s}" for g in games)
                     + f"  {'   -  ' if norm is None or norm != norm else f'{norm:5.2f} '}"
                     + (f"  {ol['_same']:.2f}/{ol['_change']:.2f}/{ol.get('_plan_switch', float('nan')):.2f}" if ol else ""))
    lines.append("goals: fraction of --play levels where the gold goal was reached; norm: reward benchmark, 0 = noop, 1 = expert")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="model dir, LoRA dir, or HF id")
    ap.add_argument("--games", default=",".join(sim.ENVS))
    ap.add_argument("--per-game", type=int, default=2)
    ap.add_argument("--ticks", type=int, default=300)
    ap.add_argument("--name", default=None, help="leaderboard row name (default: from --model)")
    ap.add_argument("--out", default=BENCH_DIR)
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--openloop", action="store_true", help="also key accuracy on expert states, split same/changed keys")
    ap.add_argument("--play", type=int, default=0, help="also play every gold goal this many ticks; inputs → <out>/play_<name>/")
    ap.add_argument("--seeds", type=int, default=5, help="--play levels per game (seed 0 = the gold level)")
    ap.add_argument("--no-bench", action="store_true", help="skip the reward benchmark (gameplay/open-loop only)")
    ap.add_argument("--video", default=None, help="render a play dir's inputs to mp4s (local, no model)")
    args = ap.parse_args(argv)
    if args.table:
        print(table(args.out))
        return
    if args.video:
        return videos(args.video)
    from sim.render import best_device
    dev = best_device()
    policy = VLMRunner(args.model, device=dev)
    games = args.games.split(",")
    res = {} if args.no_bench else bench(policy, games, args.per_game, args.ticks, dev)
    if args.openloop:
        res["_openloop"] = openloop(policy, games)
    name = args.name or os.path.basename(os.path.normpath(args.model)).replace("/", "_")
    if args.play:
        res["_play"] = play(policy, games, args.play, dev, os.path.join(args.out, "play_" + name), args.seeds)
        print("goal success:", {g: v["success"] for g, v in res["_play"].items() if not g.startswith("_")}, flush=True)
    res.setdefault("_summary", {"mean_norm": float("nan"), "parse_fail": policy.parse_fail / max(1, policy.calls)})
    os.makedirs(args.out, exist_ok=True)
    json.dump(res | {"_policy": args.model}, open(os.path.join(args.out, name + ".json"), "w"), indent=1)
    print(table(args.out))
    return res


if __name__ == "__main__":
    main()
