"""
collect_pairs.py — convert existing GameSense data into DPO pairs.

This is a thin CLI wrapper around dpo.dataset.build_pairs_from_collect.
It reads data.npz produced by `python -m vision.collect` and writes
dpo_pairs.npz that can be inspected or passed to train_dpo.

Usage:
  python -m dpo.collect_pairs --data episode_data --out dpo_pairs.npz
"""

import argparse

from dpo.dataset import build_pairs_from_collect, save_pairs


def main():
    parser = argparse.ArgumentParser(
        description="Build DPO (chosen, rejected) pairs from GameSense data.npz"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="episode_data",
        help="Directory containing data.npz (from vision.collect)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="dpo_pairs.npz",
        help="Output path for the pairs file",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=4000,
        help="Maximum number of (chosen, rejected) pairs to extract",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible pair sampling",
    )
    args = parser.parse_args()

    chosen, rejected = build_pairs_from_collect(
        args.data, max_pairs=args.max_pairs, seed=args.seed
    )
    save_pairs(chosen, rejected, args.out)
    print(f"Saved {len(chosen)} pairs → {args.out}", flush=True)


if __name__ == "__main__":
    main()
