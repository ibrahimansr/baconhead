"""Modal: imitation data on CPUs; SFT, DAgger and RL on a training GPU; evals on a cheap GPU.
The pipeline and its commands: README. Set GPUs at deploy time and redeploy after code changes (sft/dagger/rl/eval
are spawned on the deployed app so they outlive the client):

    BH_GPU=H100 BH_EVAL_GPU=A10 modal deploy train/modal_app.py

Starter workspaces can't schedule H100s ("H100 unavailable on starter plan"): use BH_GPU=A100-80GB there. With the
vision tower frozen it ran SFT at 1.8-2.0 s/step vs the H100's 1.5, at ~60% of the price per hour.
"""

from __future__ import annotations

import os

import modal

app = modal.App("baconhead")
vol = modal.Volume.from_name("baconhead", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential")                          # torch.compile needs a C compiler for the CPU renderer
    .pip_install("trl[vllm]==1.14.0", "transformers==5.17.0", "peft==0.21.0", "datasets==5.0.1", "accelerate==1.15.0",
                 "pillow", "numpy",
                 "flash-linear-attention")   # Qwen3.5's 24 linear-attention layers: ~38 s per 10-game step without it
    .env({"HF_HOME": "/vol/hf", "HF_HUB_DISABLE_XET": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})  # xet downloads stalled in testing
    .add_local_python_source("core", "sim", "model", "train")
)
GPU = os.environ.get("BH_GPU", "H100")
EVAL_GPU = os.environ.get("BH_EVAL_GPU", "A10")   # gameplay evals are generation-bound; an A10 does 10 games x 5 seeds


def _long(m, argv, git):
    """Commit the volume every 10 min while training, so checkpoints survive a cancel or crash."""
    import threading
    import time

    def commit():
        while True:
            time.sleep(600)
            vol.commit()
    threading.Thread(target=commit, daemon=True).start()
    m.main(argv, git)
    vol.commit()


@app.function(image=image, cpu=4, memory=8192, timeout=4 * 3600, volumes={"/vol": vol})
def gen_shard(argv: list[str]):
    from train import dataset
    dataset.main(argv)
    vol.commit()


@app.function(image=image, gpu=GPU, cpu=16, memory=65536, timeout=24 * 3600, volumes={"/vol": vol})
def sft(argv: list[str], git: dict):
    from train import sft as m
    _long(m, argv, git)


@app.function(image=image, gpu=GPU, cpu=16, memory=65536, timeout=24 * 3600, volumes={"/vol": vol})
def dagger(argv: list[str], git: dict):
    from train import dagger as m
    _long(m, argv, git)


@app.function(image=image, gpu=GPU, cpu=16, memory=65536, timeout=24 * 3600, volumes={"/vol": vol})
def rl(argv: list[str], git: dict):
    from train import rl as m
    _long(m, argv, git)


@app.function(image=image, gpu=EVAL_GPU, cpu=8, memory=32768, timeout=6 * 3600, volumes={"/vol": vol})
def evaluate(argv: list[str]) -> dict:
    """train.eval on a GPU: --model <dir on /vol or HF id>. Results land in /vol/results/bench/<name>.json and
    play_<name>/; `modal volume get baconhead /results/bench results/` to compare and render videos locally."""
    from train import eval as E
    res = E.main(argv + ["--out", "/vol/results/bench"])
    vol.commit()
    return res


@app.local_entrypoint()
def main(cmd: str, args: str = "", name: str = ""):
    import shlex
    from train.runlog import _git
    argv, git = shlex.split(args), _git()
    if cmd == "data":   # --args "games ..." or "skills ...", --shards N fans out over N CPU containers → /vol/data/<name>/shard_i
        n = int(argv[argv.index("--shards") + 1])
        rest = [a for i, a in enumerate(argv) if a != "--shards" and (i == 0 or argv[i - 1] != "--shards")]
        list(gen_shard.map([rest + ["--seed", str(i), "--out", f"/vol/data/{name}/shard_{i:04d}"] for i in range(n)]))
    elif cmd in ("sft", "dagger", "rl", "evaluate"):
        # Spawned on the *deployed* app: a `modal run` call dies with its client (a laptop crash cancelled GRPO v3 at
        # step 292), a spawned call doesn't. Watch: modal app logs baconhead.
        fn = modal.Function.from_name("baconhead", cmd)
        call = fn.spawn(argv) if cmd == "evaluate" else fn.spawn(argv, git)
        print(f"spawned {cmd}: {call.object_id}  (cancel: python -c \"import modal; modal.FunctionCall.from_id('{call.object_id}').cancel()\")")
    else:
        raise SystemExit(f"unknown --cmd {cmd}")
