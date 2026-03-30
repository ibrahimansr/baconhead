#!/usr/bin/env python3
"""
run_dpo.py — Fine-tune GameSense with Direct Preference Optimization.

Reads gameplay data (data.npz) collected by vision.collect, builds
(chosen, rejected) screenshot pairs from observed game-state transitions,
and fine-tunes the GameSense ViT in-place using the DPO loss.

After training, the updated .pt file contains BOTH policy weights and a
frozen reference copy — no separate reward model is needed at inference time.

Usage:
  # Basic — fine-tune from episode data
  python run_dpo.py --data episode_data --model game_sense.pt --epochs 3

  # Custom output (keep original untouched)
  python run_dpo.py --data episode_data --model game_sense.pt \\
      --out game_sense_dpo.pt --epochs 5 --beta 0.1

  # Quick smoke test (one forward pass, no save)
  python run_dpo.py --data episode_data --model game_sense.pt --dry-run

  # Use policy_data directory
  python run_dpo.py --data policy_data --model game_sense.pt --epochs 3
"""

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune GameSense with DPO from gameplay pairs"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="episode_data",
        help=(
            "Directory containing data.npz from `python -m vision.collect`. "
            "Defaults to episode_data."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="game_sense.pt",
        help=(
            "Path to existing GameSense checkpoint to fine-tune. "
            "If the file does not exist, a fresh model is initialised."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Output path for the DPO-fine-tuned checkpoint. "
            "Defaults to --model (overwrites in-place)."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of DPO training epochs (default: 3).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Pairs per gradient step (default: 8).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help=(
            "DPO temperature β — controls how much the policy can deviate from "
            "the reference. Lower = more conservative (default: 0.1)."
        ),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-6,
        help=(
            "Learning rate. Keep small — this is fine-tuning on top of a "
            "pre-trained ViT (default: 5e-6)."
        ),
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=4000,
        help="Maximum (chosen, rejected) pairs to extract from data.npz (default: 4000).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Build pairs and run one forward pass to verify the setup, then exit "
            "without saving. Useful for testing."
        ),
    )
    args = parser.parse_args()

    out_path = args.out or args.model

    print(
        f"[run_dpo] data={args.data!r}  model={args.model!r}  "
        f"out={out_path!r}  epochs={args.epochs}  beta={args.beta}",
        flush=True,
    )

    # Validate data dir early to give a clear error message
    npz_path = os.path.join(args.data, "data.npz")
    if not os.path.isfile(npz_path):
        print(
            f"[run_dpo] ERROR: No data.npz found at {npz_path!r}.\n"
            f"Run first:  python -m vision.collect --seconds 120 --out {args.data}",
            file=sys.stderr,
        )
        sys.exit(1)

    from dpo.train_dpo import train_dpo

    train_dpo(
        data_dir=args.data,
        policy_path=args.model,
        out_path=out_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        beta=args.beta,
        lr=args.lr,
        max_pairs=args.max_pairs,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
