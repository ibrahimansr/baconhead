"""Modal: sim data on CPUs; SFT, DAgger, GRPO, the benchmark and per-game LoRAs on GPUs; vLLM serving.
Commands and costs: README "Train the base model" and "Benchmark". Set the GPU at deploy time:
BH_GPU=H100 modal deploy train/modal_app.py (L4 is plenty for the 256M proxy). Redeploy after code changes:
sft/dagger/grpo/eval are spawned on the deployed app so they outlive the client.
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
                 "pillow", "numpy")   # the versions the local smoke tests passed with; vllm/torch resolved within trl's range
    .env({"HF_HOME": "/vol/hf", "HF_HUB_DISABLE_XET": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})  # xet downloads stalled in testing
    .add_local_python_source("core", "sim", "model", "train", "recorder")
)
GPU = os.environ.get("BH_GPU", "H100")
BASE = "/vol/models/base"


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
def grpo(argv: list[str], git: dict):
    from train import grpo as m
    _long(m, argv, git)


@app.function(image=image, gpu=GPU, cpu=16, memory=65536, timeout=24 * 3600, volumes={"/vol": vol})
def dagger(argv: list[str], git: dict):
    from train import dagger as m
    _long(m, argv, git)


@app.function(image=image, gpu=GPU, cpu=8, memory=32768, timeout=4 * 3600, volumes={"/vol": vol})
def finetune(game: str, argv: list[str], git: dict) -> str:
    """LoRA on everything you've recorded for `game`, on top of the base model."""
    from train.teach import teach
    vol.reload()   # see the demos the client just uploaded
    out = teach(game, f"/vol/demos/{game}", BASE, f"/vol/models/{game}", git, argv)
    vol.commit()
    return out.removeprefix("/vol")


@app.function(image=image, gpu=GPU, cpu=16, memory=65536, timeout=6 * 3600, volumes={"/vol": vol})
def evaluate(argv: list[str]) -> dict:
    """The benchmark (train.eval) on a GPU: --model <dir on /vol or HF id> or --api <OpenRouter id>.
    Results land in /vol/results/bench/<name>.json; `modal volume get baconhead /results/bench results/` to compare."""
    from train import eval as E
    res = E.main(argv + ["--out", "/vol/results/bench"])
    vol.commit()
    return res


@app.function(image=image, gpu=GPU, memory=65536, timeout=3600, volumes={"/vol": vol})
def promote(src: str):
    import shutil
    from train.grpo import merge
    if os.path.exists(os.path.join(src, "adapter_config.json")):
        merge(src, BASE)
    else:
        shutil.copytree(src, BASE, dirs_exist_ok=True)
    vol.commit()


@app.cls(image=image, gpu="L4", volumes={"/vol": vol}, scaledown_window=600, timeout=3600)
class Serve:
    """vLLM on the base model; per-game LoRAs hot-load by path, and a retrained adapter gets a fresh id."""

    @modal.enter()
    def load(self):
        from vllm import LLM
        from model.vlm import processor
        self.llm = LLM(BASE, enable_lora=True, max_lora_rank=64, limit_mm_per_prompt={"image": 2}, max_model_len=2048)
        self.proc = processor(BASE)

    @modal.method()
    def generate(self, jpegs: list, adapter: str | None = None) -> str:
        import io
        from PIL import Image
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest
        from model.vlm import prompt
        lora = None
        if adapter:
            vol.reload()
            path = "/vol" + adapter
            if os.path.exists(path):
                lora = LoRARequest(adapter, int(os.path.getmtime(path)) % 2 ** 31, path)
        text = self.proc.apply_chat_template(prompt(), add_generation_prompt=True, tokenize=False)
        imgs = [Image.open(io.BytesIO(j)).convert("RGB") for j in jpegs]   # 0.2 s ago, now
        out = self.llm.generate({"prompt": text, "multi_modal_data": {"image": imgs}}, SamplingParams(temperature=0, max_tokens=64),
                                lora_request=lora)
        return out[0].outputs[0].text


@app.local_entrypoint()
def main(cmd: str, args: str = "", name: str = ""):
    import shlex
    from train.runlog import _git
    argv, git = shlex.split(args), _git()
    if cmd == "data":   # --shards N fans out over N CPU containers, each writing /vol/data/<name>/shard_i
        n = int(argv[argv.index("--shards") + 1])
        rest = [a for i, a in enumerate(argv) if a != "--shards" and (i == 0 or argv[i - 1] != "--shards")]
        kind = ["sim"]
        list(gen_shard.map([kind + rest + ["--seed", str(i), "--out", f"/vol/data/{name}/shard_{i:04d}"] for i in range(n)]))
    elif cmd in ("sft", "dagger", "grpo"):
        # Spawned on the *deployed* app: a `modal run` call dies with its client (a laptop crash cancelled GRPO v3 at
        # step 292), a spawned call doesn't. Redeploy after code changes. Watch: modal app logs baconhead.
        call = modal.Function.from_name("baconhead", cmd).spawn(argv, git)
        print(f"spawned {cmd}: {call.object_id}  (cancel: python -c \"import modal; modal.FunctionCall.from_id('{call.object_id}').cancel()\")")
    elif cmd == "promote":
        promote.remote(argv[0])
    elif cmd == "eval":   # --args "--model /vol/runs/sft_v2" or "--model Qwen/Qwen2.5-VL-7B-Instruct" or "--api <openrouter id>"
        call = modal.Function.from_name("baconhead", "evaluate").spawn(argv)
        print(f"spawned eval: {call.object_id}")
    else:
        raise SystemExit(f"unknown --cmd {cmd}")
