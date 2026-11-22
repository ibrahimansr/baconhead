"""Train the world model, the imitation prior, and both heads in one pass over your demos."""

from __future__ import annotations

import argparse
import glob
import json
import os

import torch

from wm.actions import N_ACTIONS
from wm.data import episode_to_ticks, load_hf, sample_batch
from wm.model import WorldModel, goal_vec


def collect_sources(demos: str, hf: dict | None = None) -> list[dict]:
    """Every episode directory under `demos`, plus a Hugging Face set if one is configured."""
    sources = []
    for d in sorted(glob.glob(os.path.join(demos, "*"))):
        if os.path.isfile(os.path.join(d, "meta.json")):   # meta.json is what marks a directory as a finished episode
            t = episode_to_ticks(d)
            print(f"{d}: {t['actions'].shape[0]} ticks", flush=True)
            if t["actions"].shape[0]:
                sources.append(t)
    if hf and hf.get("dataset"):
        t = load_hf(hf["dataset"], hf.get("split", "train"), hf.get("frame_column", "image"),
                    hf.get("action_column", "action"), hf.get("action_map") or {})
        print(f"{hf['dataset']}: {t['actions'].shape[0]} ticks", flush=True)
        sources.append(t)
    return sources


def train(sources: list[dict], epochs: int, seq_len: int, batch: int, lr: float, out: str,
          device: str = "cpu") -> str:
    model = WorldModel(N_ACTIONS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    steps_per_epoch = max(1, sum(s["actions"].shape[0] for s in sources) // (seq_len * batch))

    for ep in range(epochs):
        agg = {}
        for _ in range(steps_per_epoch):
            for src in sources:
                got = sample_batch(src, seq_len, batch)
                if got is None:
                    continue
                frames, actions, alive, progress = got
                total, parts = model.loss(
                    frames.to(device), actions.to(device), goal_vec(src.get("goal", "")).to(device),
                    alive.to(device), progress.to(device),
                )
                opt.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # the GRU will diverge without this
                opt.step()
                for k, v in parts.items():
                    agg[k] = agg.get(k, 0.0) + v
        n = max(1, steps_per_epoch * len(sources))
        print(f"epoch {ep + 1}/{epochs}  " + "  ".join(f"{k}={v / n:.4f}" for k, v in sorted(agg.items())), flush=True)

    torch.save(model.state_dict(), out)
    print(f"saved → {out}")
    return out


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"   # MPS is the Apple GPU, roughly 5x faster than CPU here


def main() -> None:
    p = argparse.ArgumentParser(description="Train the latent world model")
    p.add_argument("--demos", default="demos")
    p.add_argument("--hf", default=None, help="Hugging Face dataset for imitation pretraining")
    p.add_argument("--hf-split", default="train")
    p.add_argument("--hf-frame", default="image")
    p.add_argument("--hf-action", default="action")
    p.add_argument("--hf-map", default="{}", help='JSON mapping that dataset\'s labels onto wm.actions names')
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--out", default="wm.pt")
    args = p.parse_args()

    hf = {"dataset": args.hf, "split": args.hf_split, "frame_column": args.hf_frame,
          "action_column": args.hf_action, "action_map": json.loads(args.hf_map)} if args.hf else None
    sources = collect_sources(args.demos, hf)
    if not sources:
        raise SystemExit("No data. Record a demo first: python -m recorder.record --seconds 120")

    train(sources, args.epochs, args.seq_len, args.batch, args.lr, args.out, pick_device())


if __name__ == "__main__":
    main()
