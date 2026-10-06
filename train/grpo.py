"""RL with TRL's GRPOTrainer. Each prompt is a saved sim state; the model samples G action chunks for it;
each chunk is scored by playing it in a copy of that state, then letting the scripted expert continue for
CONT ticks (0.8 s), so a chunk that leaves you on the edge of a platform scores worse than one that lands you safely.
Run it only after DAgger (train/dagger.py): on the SFT copycat model, GRPO sharpened "stand still" (see README).
This is one-step improvement over the expert: it learns which chunk to play now given competent play afterwards.
Full multi-chunk rollouts by the model itself would need TRL's multi-turn VLM support; not done.
GRPO ranks the G chunks against each other, so the three games' reward scales don't need matching.

    python -m train.grpo --model models/base --data data/sim_grpo --out models/base_grpo
"""

from __future__ import annotations

import argparse
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import torch

from model.vlm import CHUNK, MODEL_ID, decode, processor
from train import runlog
from train.dataset import load

CONT = 8         # expert ticks after the sampled chunk. Was 30: the expert then repaired almost any chunk, 30-77% of
                 # groups tied, and "stand still" won once sampling collapsed (v3, steps 200-600). Short tail = the
                 # chunk itself decides the reward.
GAMMA = 0.98     # per tick
_POOL = None


def _text(c) -> str:
    if isinstance(c, str):
        return c
    content = c[0]["content"]
    return content if isinstance(content, str) else "".join(p.get("text") or "" for p in content)


def _score(args) -> float:
    """Discounted reward over the whole sampled chunk (every generated token gets credit, though live play only uses
    its first EXEC ticks before replanning), then CONT expert ticks, plus the change in
    the game's progress potential (γ^T·Φ(end) − Φ(start), which shapes without changing the optimal policy).
    Without Φ, brainrot and adoptme rewards were too sparse: every sampled chunk scored 0."""
    state, text = args
    env = pickle.loads(state)
    phi0 = env.potential()
    acts, _ = decode(text, env.cursor)
    r, g = 0.0, 1.0
    for t in range(CHUNK + CONT):
        r += g * env.step(acts[t] if t < CHUNK else env.expert())[0]
        g *= GAMMA
        if env.done:
            return r - phi0
    return r + g * env.potential() - phi0


def sim_reward(completions, state, **_) -> list[float]:
    global _POOL
    if _POOL is None:   # spawn, not fork: the parent holds a CUDA context the workers must not inherit
        import multiprocessing as mp
        _POOL = ProcessPoolExecutor(os.cpu_count(), mp_context=mp.get_context("spawn"))
    return list(_POOL.map(_score, zip(state, [_text(c) for c in completions])))


def format_reward(completions, **_) -> list[float]:
    return [0.0 if decode(_text(c))[1] else -1.0 for c in completions]


def main(argv=None, git=None):   # device_map=None off-CUDA: transformers hangs loading straight onto MPS
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beta", type=float, default=0.04, help="KL to the start model; 0 (TRL's default) let v3 collapse to entropy 0.016 by step 600")
    ap.add_argument("--entropy-target", type=float, default=0.2, help="TRL adaptive entropy bonus holds token entropy near this; 0 disables")
    ap.add_argument("--vllm", action="store_true", help="vLLM generation; its logprobs drifted from the trainer's in v1/v2 (cause unconfirmed)")
    ap.add_argument("--resume", action="store_true", help="continue from the last checkpoint in --out")
    args = ap.parse_args(argv)
    runlog.start(args.out, vars(args), git)
    cuda = torch.cuda.is_available()
    ds = load(args.data).shuffle(seed=args.seed)
    cfg = GRPOConfig(output_dir=args.out, num_generations=args.generations, max_completion_length=64,
                     per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.grad_accum,
                     max_steps=args.max_steps, learning_rate=args.lr, bf16=cuda, gradient_checkpointing=cuda,
                     beta=args.beta, use_adaptive_entropy=args.entropy_target > 0, entropy_target=args.entropy_target or 0.2,
                     use_vllm=cuda and args.vllm, vllm_mode="colocate", vllm_gpu_memory_utilization=0.5,
                     vllm_max_model_length=2048,   # vLLM sampling logprobs drifted from the trainer's (IS ratio → 0 by ~step 200)
                     # with and without regex decoding; cause unconfirmed, so it's opt-in.
                     model_init_kwargs={"dtype": torch.bfloat16 if cuda else torch.float32, **({} if cuda else {"device_map": None})},
                     logging_steps=5, save_steps=200, save_total_limit=2, report_to="none", seed=args.seed,
                     use_cpu=not cuda and not torch.backends.mps.is_available())
    trainer = GRPOTrainer(model=args.model, reward_funcs=[sim_reward, format_reward], args=cfg, train_dataset=ds, processing_class=processor(args.model),
                          peft_config=LoraConfig(r=64, lora_alpha=128, target_modules="all-linear", task_type="CAUSAL_LM"))
    # Resume whenever a checkpoint exists: Modal retries a preempted container with the same args, and without this
    # the 3B SFT restarted from step 0 twice.
    import glob
    trainer.train(resume_from_checkpoint=True if (args.resume or glob.glob(os.path.join(args.out, "checkpoint-*"))) else None)
    trainer.save_model(args.out)
    trainer.processing_class.save_pretrained(args.out)


def merge(adapter: str, out: str) -> None:
    """Fold a LoRA into its base so it can be the next base model (and served without an adapter)."""
    import json
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor
    base = json.load(open(os.path.join(adapter, "adapter_config.json")))["base_model_name_or_path"]
    m = PeftModel.from_pretrained(AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16), adapter)
    m.merge_and_unload().save_pretrained(out)
    AutoProcessor.from_pretrained(adapter).save_pretrained(out)


if __name__ == "__main__":
    main()
