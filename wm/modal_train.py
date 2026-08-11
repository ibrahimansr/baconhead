"""Train on a cloud GPU instead of your laptop: modal run wm/modal_train.py --demos demos --epochs 300

Your Mac has to run Roblox, capture the screen, and inject input, so it is the wrong place to also be
grinding gradients. This ships the demos to Modal, trains on an A10G, and writes wm.pt back locally.
"""

from __future__ import annotations

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "pillow", "datasets")   # no pyobjc here, the GPU box never touches macOS input
    .add_local_python_source("wm", "recorder", "control", "core")
)

app = modal.App("baconhead-wm", image=image)
vol = modal.Volume.from_name("baconhead-demos", create_if_missing=True)   # persists between runs so re-training is instant


@app.function(gpu="A10G", volumes={"/data": vol}, timeout=60 * 60 * 4)
def train_remote(epochs: int, seq_len: int, batch: int, lr: float, hf: dict | None) -> bytes:
    import tempfile

    from wm.train import collect_sources, train

    sources = collect_sources("/data/demos", hf)
    if not sources:
        raise RuntimeError("No episodes on the volume — did the upload run?")

    with tempfile.TemporaryDirectory() as d:
        out = f"{d}/wm.pt"
        train(sources, epochs, seq_len, batch, lr, out, device="cuda")
        return open(out, "rb").read()   # returned as bytes so nothing has to be downloaded from the volume separately


@app.local_entrypoint()
def main(demos: str = "demos", epochs: int = 300, seq_len: int = 16, batch: int = 64,
         lr: float = 3e-4, out: str = "wm.pt", hf: str = "", hf_map: str = "{}",
         hf_frame: str = "image", hf_action: str = "action") -> None:
    import json
    import os

    if not os.path.isdir(demos):
        raise SystemExit(f"No {demos}/ directory. Record a demo first: python -m recorder.record")

    with vol.batch_upload(force=True) as up:   # force=True so a re-record replaces the old episode
        up.put_directory(demos, "/demos")
    print(f"uploaded {demos}/ → volume")

    cfg = {"dataset": hf, "split": "train", "frame_column": hf_frame,
           "action_column": hf_action, "action_map": json.loads(hf_map)} if hf else None
    blob = train_remote.remote(epochs, seq_len, batch, lr, cfg)
    open(out, "wb").write(blob)
    print(f"trained on A10G → {out}  ({len(blob) / 1e6:.1f} MB)")
