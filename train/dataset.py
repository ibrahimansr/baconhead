"""Build TRL datasets: {prompt, image, completion} for SFT, {prompt, image, state} for GRPO.

    python -m train.dataset sim   --games obby,brainrot,adoptme --episodes 4 --ticks 1200 --out data/sim_sft
    python -m train.dataset sim   --states --episodes 4 --out data/sim_grpo           # GRPO start states
    python -m train.dataset demos --dirs demos/tower_of_hell --out data/demos_toh      # your recorded play

Each row holds two frames FRAME_GAP ticks apart (so motion is visible) and is taken every EXEC ticks, the rate the
model replans at. No action history goes in the prompt (see model.vlm.prompt). Most consecutive labels repeat (walk,
walk, walk; or wait), so a row whose label is the k-th repeat in a row is kept with probability decay^(k-1): Game-TARS's
"decaying continual loss" (arXiv 2510.23691), done as subsampling since SFTTrainer has no per-row weights. Sim rows come from the scripted expert with injected noise: with prob `noise` a tick executes a perturbed
action, but the label is always what the expert would do from where you actually are, so the model sees
recoveries from mistakes (DAgger-style) instead of only perfect trajectories.
"""

from __future__ import annotations

import argparse
import pickle
import re

import numpy as np

import sim
from core.actions import Action
from model.vlm import CHUNK, EXEC, FRAME_GAP, OBS_H, OBS_W, encode, image, prompt

DECAY = 0.5

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


def keep_rows(labels: list[str], rng, decay: float = DECAY) -> list[bool]:
    """Which rows survive the repeat decay: the k-th label in a row with the same keys and clicks (camera degrees
    ignored, the expert jitters those) is kept with prob decay^(k-1)."""
    sig = [re.sub(r" -?\d+ -?\d+", "", lab) for lab in labels]
    out, k = [], 0
    for i, lab in enumerate(sig):
        k = k + 1 if i and lab == sig[i - 1] else 1
        out.append(bool(rng.random() < decay ** (k - 1)))
    return out


def sim_rows(game: str, seed: int, ticks: int, noise: float = 0.2, states: bool = False, device: str = "cpu",
             stride: int = EXEC, decay: float = DECAY) -> list[dict]:
    """One row every EXEC ticks: frames FRAME_GAP ticks apart and the expert's next CHUNK inputs from here (SFT) or a
    pickled copy of the env (GRPO start state)."""
    from sim.render import render
    rng = np.random.default_rng(seed)
    env = sim.make(game, seed=seed, **_kw(game, rng))
    frames, labels, cursors, snaps = {}, [], [], {}
    for t in range(ticks + CHUNK):
        if t % stride == 0 or (t + FRAME_GAP) % stride == 0:   # only frames some row will use
            frames[t] = image(render([env.scene()], OBS_H, OBS_W, device=device)[0].cpu().numpy())
        if states and t % stride == 0 and t < ticks:
            snaps[t] = pickle.dumps(env)
        cursors.append(env.cursor)
        e = env.expert()
        a = _perturb(e, rng) if rng.random() < noise else e
        labels.append(e)
        env.step(a)
        if env.done:
            break
    ts = list(range(0, min(ticks, len(labels) - CHUNK + 1), stride))
    texts = [encode(labels[t:t + CHUNK], cursors[t]) for t in ts]
    keep = keep_rows(texts, rng, decay) if not states else [True] * len(ts)
    rows = []
    for t, text, k in zip(ts, texts, keep):
        if not k:
            continue
        row = {"prompt": prompt(), "images": [frames.get(t - FRAME_GAP, frames[t]), frames[t]]}
        if states:
            row["state"] = snaps[t]
        else:
            row["completion"] = _completion(text)
        rows.append(row)
    return rows


def demo_rows(dirs: list[str], device: str = "cpu", decay: float = DECAY) -> list[dict]:
    """Recorded play (real Roblox episodes or play.py .npz files) → SFT rows, one every EXEC ticks."""
    import glob
    import os
    from train.data import label_real, replay_sim
    rows = []
    for d in dirs:
        eps = []
        for ep in sorted({os.path.dirname(p) for p in glob.glob(os.path.join(d, "**", "meta.json"), recursive=True)}):
            fi, acts = label_real(ep)
            raw = np.load(os.path.join(ep, "frames.npy"), mmap_mode="r")
            eps.append(([raw[i] for i in fi], acts))
        for p in sorted(glob.glob(os.path.join(d, "**", "*.npz"), recursive=True)):
            eps.append(replay_sim(p, device))
        rng = np.random.default_rng(0)
        for frames, acts in eps:
            ts = list(range(CHUNK, len(acts) - CHUNK + 1, EXEC))
            texts = [encode(acts[t:t + CHUNK], acts[t - 1].cursor) for t in ts]
            for t, text, k in zip(ts, texts, keep_rows(texts, rng, decay)):
                if k:
                    rows.append({"prompt": prompt(), "images": [image(frames[max(0, t - FRAME_GAP)]), image(frames[t])],
                                 "completion": _completion(text)})
    return rows


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
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["sim", "demos"])
    ap.add_argument("--games", default=",".join(sim.ENVS))
    ap.add_argument("--episodes", type=int, default=2, help="per game")
    ap.add_argument("--ticks", type=int, default=1200)
    ap.add_argument("--noise", type=float, default=0.2)
    ap.add_argument("--states", action="store_true", help="GRPO start states instead of SFT labels")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stride", type=int, default=EXEC, help="ticks between rows; neighbours are near-duplicates, so >EXEC buys more episodes per row")
    ap.add_argument("--decay", type=float, default=DECAY, help="keep prob for the k-th repeated label is decay^(k-1); 1 keeps all")
    ap.add_argument("--dirs", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.kind == "sim":
        from sim.render import best_device
        dev = best_device()
        rows = [r for g in args.games.split(",") for ep in range(args.episodes)
                for r in sim_rows(g, args.seed * 1000 + ep, args.ticks, args.noise, args.states, dev, args.stride, args.decay)]
    else:
        rows = demo_rows([d for d in args.dirs.split(",") if d], decay=args.decay)
    save(rows, args.out)


if __name__ == "__main__":
    _check()
    main()
