import json
import os
import time

import modal

from baconhead import agent as A

ROOT = "/vol/robloxagent"


app = modal.App("robloxagent-train")
vol = modal.Volume.from_name("baconhead", create_if_missing=True)
train_image = (modal.Image.debian_slim(python_version="3.12").apt_install("build-essential")
               .pip_install("transformers==5.20.1", "trl==1.14.0", "peft==0.21.0", "datasets==5.0.1", "accelerate==1.15.0",
                            "torch", "torchvision", "pillow", "numpy", "flash-linear-attention", "requests")
               .env({"HF_HOME": "/vol/hf", "HF_HUB_DISABLE_XET": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
               .add_local_python_source("baconhead"))
rl_image = (modal.Image.from_registry("radixark/miles:latest")
            .env({"HF_HOME": "/vol/hf", "HF_HUB_DISABLE_XET": "1"})
            .add_local_python_source("baconhead"))


def _commit_every(minutes=8):
    import threading

    def loop():
        while True:
            time.sleep(60 * minutes)
            vol.commit()
    threading.Thread(target=loop, daemon=True).start()


@app.function(image=train_image, cpu=4, memory=16384, timeout=3 * 3600, volumes={"/vol": vol})
def gen(game, seed, decisions, out):
    from baconhead.train.bc import expert_rows
    from baconhead.train.dataset import save
    rows = expert_rows(game, seed, decisions)
    if rows:
        save(rows, out)
        vol.commit()
    return len(rows)


# loRA SFT, merged into <out>
@app.function(image=train_image, gpu="H100", cpu=16, memory=65536, timeout=6 * 3600, volumes={"/vol": vol})
def sft(model, data, out, steps):
    from baconhead.train import sft as T
    _commit_every()
    vol.reload()
    T.main(["--model", model, "--data", ",".join(data), "--lora", "--out", out + "_lora", "--max-steps", str(steps),
            "--batch", "8", "--grad-accum", "2", "--workers", "12"])
    T.merge(out + "_lora", out)
    vol.commit()
    return out


# dagger collection into <out>, or the held-out eval when out is None
@app.function(image=A.image, gpu="H100", cpu=16, memory=65536, timeout=4 * 3600, volumes={"/vol": vol})
def rollout(model, games, n_envs, decisions, beta, seed, out):
    from vllm import LLM
    from baconhead.train import dagger
    from baconhead.train.dataset import save
    vol.reload()
    llm = LLM(model, trust_remote_code=True, max_model_len=8192, limit_mm_per_prompt={"image": A.FRAMES, "video": 0},
              gpu_memory_utilization=0.6, max_num_seqs=64)   # leaves room for the renderer
    res, rows = dagger.rollout(llm, games, n_envs, decisions, beta, seed, evaluating=out is None)
    if out:
        save(rows, out)
        res["rows"] = len(rows)
        vol.commit()
    return res


# async GRPO from the DAgger policy; -> the last HF snapshot
@app.function(image=rl_image, gpu="H100:2", cpu=32, memory=262144, timeout=24 * 3600, volumes={"/vol": vol})
def rl(model, games, rollouts, out):
    import glob
    import shutil
    from baconhead.train import rl as R
    from huggingface_hub import snapshot_download
    _commit_every()
    vol.reload()
    model = model if os.path.isdir(model) else snapshot_download(model)
    R.train(model, R.prepare(f"{out}/data", games), out, rollouts)
    hf = max(glob.glob(f"{out}/hf/step_*"), key=lambda p: int(p.rsplit("_", 1)[1]))
    for f in glob.glob(f"{model}/*"):   # tokenizer, processor and chat template ride along with the weights
        if not f.endswith(".safetensors") and not os.path.exists(f"{hf}/{os.path.basename(f)}"):
            shutil.copy(f, hf)
    vol.commit()
    return hf


# BC on expert play -> `rounds` of DAgger -> async RL, evaluated after every stage. --resume <model> --start k picks up at DAgger round k (k = rounds + 1: RL only)
@app.local_entrypoint()
def main(games: str = "obby", base: str = "Qwen/Qwen3.5-4B", rounds: int = 3, rl_rollouts: int = 500, sft_steps: int = 700,
         dagger_steps: int = 400, seeds: int = 12, eval_levels: int = 5, envs_per_game: int = 8,
         resume: str = "", start: int = 1, root: str = ROOT):
    games = games.split(",")
    os.makedirs("results/training", exist_ok=True)
    logp = os.path.join("results/training", time.strftime("%Y%m%d_%H%M%S") + ".jsonl")

    def log(d):
        open(logp, "a").write(json.dumps(d) + "\n")
        print(json.dumps({k: v for k, v in d.items() if k != "traces"}), flush=True)

    def evaluate(model, tag):
        log({"eval": tag, "model": model, **rollout.remote(model, games, eval_levels, 300, 0.0, 7, None)})

    t0 = time.time()
    data = [f"{root}/expert/{g}_{s}" for g in games for s in range(seeds)]
    if resume:
        model = resume
        data += [f"{root}/dagger_{j}" for j in range(1, start)]
    else:   # behaviour cloning
        n = list(gen.starmap([(g, s, 700, f"{root}/expert/{g}_{s}") for g in games for s in range(seeds)]))
        data = [d for d, k in zip(data, n) if k]
        log({"stage": "expert data", "rows": int(sum(n)), "min": round((time.time() - t0) / 60, 1)})
        evaluate(base, "base")
        model = sft.remote(base, data, f"{root}/r0", sft_steps)
        evaluate(model, "bc")
    for k in range(start, rounds + 1):   # DAgger: the model drives, the expert relabels
        d = rollout.remote(model, games, envs_per_game, 300, 0.5 ** k, 100 + k, f"{root}/dagger_{k}")
        log({"stage": f"dagger {k}", "rows": d["rows"], "traces": d["traces"]})
        data.append(f"{root}/dagger_{k}")
        model = sft.remote(model, [*data[::3], *[x for x in data if "dagger" in x]], f"{root}/r{k}", dagger_steps)
        evaluate(model, f"dagger {k}")
    if rl_rollouts:
        model = rl.remote(model, games, rl_rollouts, f"{root}/rl")
        evaluate(model, "rl")
    log({"done": model, "min": round((time.time() - t0) / 60, 1)})
