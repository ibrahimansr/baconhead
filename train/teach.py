"""Teach one game: recorded play → dataset → short LoRA on the base model. Deliberately quick: 2 passes
over a few minutes of play (~750 chunks per 5 min, ~90 optimizer steps).

    python -m train.teach --game 2153167977 --demos demos/2153167977 --base models/base --out models/2153167977
"""

from __future__ import annotations

import argparse
import shutil
import tempfile


def teach(game: str, demos: str, base: str, out: str, git=None, extra: list[str] | None = None) -> str:
    from train import dataset, sft
    data = tempfile.mkdtemp(prefix=f"teach_{game}_")
    dataset.save(dataset.demo_rows([demos]), data)
    tmp = out + ".training"
    sft.main(["--model", base, "--data", data, "--lora", "--epochs", "2", "--out", tmp] + (extra or []), git)
    shutil.rmtree(out, ignore_errors=True)
    shutil.move(tmp, out)            # swap in atomically, so a server never loads a half-written adapter
    shutil.rmtree(data, ignore_errors=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--demos", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(teach(a.game, a.demos, a.base, a.out))
