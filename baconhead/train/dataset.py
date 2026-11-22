"""Imitation data: TRL {prompt, images, completion} rows from the scripted experts, two kinds.

    python -m train.dataset games  --episodes 2 --ticks 1500 --out data/games      # whole episodes
    python -m train.dataset skills --clips 40 --out data/skills                    # short go-to-target clips
    (on Modal: modal run train/modal_app.py --cmd data --args "games ..." --name ...)

games:  the expert plays whole episodes from spawn. Most ticks are "keep walking" or "wait".
skills: the expert first plays a random stretch of the episode unrecorded (so clips start mid-game: shops open,
        fish hooked, Rush coming), then the camera is spun to a random heading and the expert is recorded until it
        finishes its current subgoal (its plan changes or the score goes up), at most CLIP ticks. Clips where it
        doesn't finish are dropped: only verified successes are kept (Script2Play, arXiv 2611.11136). Spinning the
        camera means most clips open with "find the target, turn, walk to it, act": the exact chain the v4 models
        never learned (they walked into walls with the right plan in their heads).

Each row holds two frames FRAME_GAP ticks apart (so motion is visible) and is taken every EXEC ticks, the rate the
model replans at. Labels are the expert's PLAN line (with where its target is on screen) and its next CHUNK ticks.
The prompt carries the game's objective and the expert's plan from the previous replan, which is what the policy's
own previous plan stands in for at test time. No action history (see model.vlm.prompt).

Most consecutive labels repeat (walk, walk, walk), so the k-th repeat in a row is kept with probability decay^(k-1):
Game-TARS's "decaying continual loss" (arXiv 2510.23691), done as subsampling since SFTTrainer has no per-row
weights. Injected noise: with prob `noise` a tick executes a perturbed action, but the label is always what the expert
would do from where you actually are, so the model sees recoveries from mistakes instead of only perfect play.
"""

from __future__ import annotations

import argparse
import math
import pickle
import re

import numpy as np

import sim
from core.actions import Action
from model.vlm import CHUNK, EXEC, FRAME_GAP, OBS_H, OBS_W, encode, image, prompt, target

DECAY = 0.5
CLIP = 100          # skill clip cap: 10 s, the paper's episodes are ~5 s


def _kw(game, rng):
    kw = {"max_steps": sim.ENVS[game].train_steps}
    if game == "obby":
        kw |= {"difficulty": float(rng.uniform(0, 1)), "respawn": 1.0}   # uniform difficulty is the curriculum
    return kw


def _perturb(a: Action, rng) -> Action:
    keys = set(a.keys) ^ {str(rng.choice(["w", "a", "s", "d", "space"]))}
    return Action(keys=frozenset(keys), look_x=a.look_x + float(rng.normal(0, 15)), look_y=a.look_y, cursor=a.cursor)


def _completion(text: str) -> list[dict]:
    return [{"role": "assistant", "content": [{"type": "text", "text": text}]}]


def with_dir(env, a: Action, plan: str) -> str:
    """Append where the expert's target is on screen ("…; target slightly left"), when the expert is moving toward
    it. v4 plans said what but not where: the model planned "walk to the next door" and walked into walls, while
    gpt-6.1-sol wrote "door 4 is just right of center" and got there. Direction relative to the camera is visible."""
    g = getattr(env, "goal", None)
    if g is None or not plan or not env.player.alive or not ("w" in a.keys or abs(a.look_x) >= 7):
        return plan
    d = np.asarray(g, float) - env.player.pos
    if math.hypot(d[0], d[2]) < 2.5:
        return plan
    rel = math.degrees((math.atan2(d[0], d[2]) - env.cam.yaw + math.pi) % (2 * math.pi) - math.pi)   # >0: right
    side = "right" if rel > 0 else "left"
    word = "ahead" if abs(rel) < 20 else f"slightly {side}" if abs(rel) < 60 else f"to your {side}" if abs(rel) < 135 else "behind you"
    return f"{plan}; target {word}"


def keep_rows(labels: list[str], rng, decay: float = DECAY) -> list[bool]:
    """Which rows survive the repeat decay: the k-th label in a row with the same keys and clicks (camera degrees
    ignored, the expert jitters those) is kept with prob decay^(k-1)."""
    sig = [re.sub(r" -?\d+ -?\d+", "", lab) for lab in labels]
    out, k = [], 0
    for i, lab in enumerate(sig):
        k = k + 1 if i and lab == sig[i - 1] else 1
        out.append(bool(rng.random() < decay ** (k - 1)))
    return out


def record(game: str, env, rng, ticks: int, noise: float, device: str, decay: float = DECAY,
           stop=None) -> tuple[list[dict], bool]:
    """The expert plays `env` from where it is for up to `ticks` ticks, recorded. stop(env, first_plan) → True ends
    the recording early and marks it a success. → (rows, stopped)"""
    from sim.render import render
    frames, labels, cursors, plans = {}, [], [], []
    first, stopped = None, False
    for t in range(ticks + CHUNK):
        if t % EXEC == 0 or (t + FRAME_GAP) % EXEC == 0:   # only frames some row will use
            frames[t] = image(render([env.scene()], OBS_H, OBS_W, device=device)[0].cpu().numpy())
        cursors.append(env.cursor)
        e = env.expert()
        first = env.plan if first is None else first
        plans.append(with_dir(env, e, env.plan))
        labels.append(e)
        env.step(_perturb(e, rng) if rng.random() < noise else e)
        if env.done:
            break
        if stop is not None and t >= CHUNK and stop(env, first):
            stopped = True
            ticks = min(ticks, t + 1)            # label the last ticks too, then end
    ts = list(range(0, min(ticks, len(labels) - CHUNK + 1), EXEC))
    texts = [target(plans[t], encode(labels[t:t + CHUNK], cursors[t])) for t in ts]
    rows = []
    for t, text, k in zip(ts, texts, keep_rows(texts, rng, decay)):
        if k:   # previous plan = the expert's at the previous replan, what the policy itself wrote EXEC ticks ago at test time
            rows.append({"prompt": prompt(game, plans[t - EXEC] if t >= EXEC else ""),
                         "images": [frames.get(t - FRAME_GAP, frames[t]), frames[t]], "completion": _completion(text)})
    return rows, stopped


def game_rows(game: str, seed: int, ticks: int, noise: float = 0.2, device: str = "cpu", decay: float = DECAY) -> list[dict]:
    """A whole episode from spawn."""
    rng = np.random.default_rng(seed)
    return record(game, sim.make(game, seed=seed, **_kw(game, rng)), rng, ticks, noise, device, decay)[0]


def skill_rows(game: str, seed: int, clips: int, noise: float = 0.1, device: str = "cpu", decay: float = DECAY) -> tuple[list[dict], int]:
    """`clips` go-to-target clips from one episode: play on unrecorded to a random point, spin the camera, record the
    expert until it finishes its subgoal. Unfinished clips are dropped. → (rows, clips kept)"""
    rng = np.random.default_rng(seed)
    env = sim.make(game, seed=seed, **_kw(game, rng))
    base = lambda p: p.split("; target")[0]
    done = lambda e, p0: base(e.plan) != base(p0) or e.score() > score0 + 1e-9
    rows, kept = [], 0
    for _ in range(clips):
        for _ in range(int(rng.integers(20, 200))):   # unrecorded stretch: later phases of the game
            env.step(env.expert())
            if env.done:
                return rows, kept
        clip_env = pickle.loads(pickle.dumps(env))
        clip_env.cam.yaw += math.radians(float(rng.uniform(-180, 180)))   # target is now anywhere: turn first
        score0 = clip_env.score()
        r, ok = record(game, clip_env, rng, CLIP, noise, device, decay, stop=done)
        if ok:
            rows += r
            kept += 1
    return rows, kept


def load(spec: str):
    """Comma-separated dataset dirs; a dir of shard_* subdirs (from modal_app data) loads as one."""
    import glob
    import os
    from datasets import concatenate_datasets, load_from_disk
    dirs = []
    for d in spec.split(","):
        dirs += [d] if os.path.exists(os.path.join(d, "dataset_info.json")) else sorted(glob.glob(os.path.join(d, "shard_*")))
    return concatenate_datasets([load_from_disk(d) for d in dirs])


def save(rows: list[dict], out: str):
    import datasets
    if not rows:
        raise SystemExit("no rows")
    ds = datasets.Dataset.from_list(rows)
    ds.save_to_disk(out)
    print(f"{len(ds)} rows → {out}")
    return ds


def _check():
    rng = np.random.default_rng(0)
    labs = ["w 1 0 | w 0 0"] * 2000 + ["j 0 0 | . 0 0"]
    k = keep_rows(labs, rng)
    assert k[0] and k[-1] and sum(k) < 10, sum(k)          # 1 + 0.5 + 0.25 + ... ≈ 2 of the 2000 repeats survive


def main(argv=None):
    from sim.render import best_device
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["games", "skills"])
    ap.add_argument("--games", default=",".join(sim.ENVS))
    ap.add_argument("--episodes", type=int, default=2, help="per game")
    ap.add_argument("--ticks", type=int, default=1500, help="games: episode length")
    ap.add_argument("--clips", type=int, default=40, help="skills: clips attempted per game per episode")
    ap.add_argument("--noise", type=float, default=None, help="default 0.2 games, 0.1 skills")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decay", type=float, default=DECAY, help="keep prob for the k-th repeated label is decay^(k-1); 1 keeps all")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    dev, rows = best_device(), []
    for g in args.games.split(","):
        for ep in range(args.episodes):
            seed = args.seed * 1000 + ep
            if args.kind == "games":
                rows += game_rows(g, seed, args.ticks, 0.2 if args.noise is None else args.noise, dev, args.decay)
            else:
                r, kept = skill_rows(g, seed, args.clips, 0.1 if args.noise is None else args.noise, dev, args.decay)
                rows += r
                print(f"{g} seed {seed}: kept {kept}/{args.clips} clips, {len(r)} rows", flush=True)
    save(rows, args.out)


if __name__ == "__main__":
    _check()
    main()
