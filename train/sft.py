"""Imitation (behavior cloning) with TRL's SFTTrainer on expert data from train.dataset; train.dagger reuses it
for its LoRA rounds.

    python -m train.sft --data data/games,data/skills --out runs/il                 # full finetune, frozen vision
    python -m train.sft --model runs/il --data data/dagger_r0 --lora --out runs/il_r0 # adapter on top
"""

from __future__ import annotations

import argparse
import os

import torch

from model.vlm import MODEL_ID, processor
from train import runlog
from train.dataset import load


def main(argv=None, git=None):   # device_map=None off-CUDA: transformers hangs loading straight onto MPS
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--data", required=True, help="comma-separated dataset dirs from train.dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=None, help="default 2e-5 full (Script2Play), 1e-4 LoRA")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int, default=8, help="dataloader processes: image decoding/preprocessing is the bottleneck")
    ap.add_argument("--ckpt", action="store_true", help="gradient checkpointing (only needed if memory runs out)")
    ap.add_argument("--train-vision", action="store_true", help="also train the vision tower; frozen by default (Script2Play freezes it)")
    args = ap.parse_args(argv)
    runlog.start(args.out, vars(args), git)
    cuda = torch.cuda.is_available()
    ds = load(args.data).shuffle(seed=args.seed)
    cfg = SFTConfig(output_dir=args.out, max_length=None, per_device_train_batch_size=args.batch,
                    gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs, max_steps=args.max_steps,
                    learning_rate=args.lr or (1e-4 if args.lora else 2e-5), warmup_steps=0.05, lr_scheduler_type="cosine_with_min_lr",
                    lr_scheduler_kwargs={"min_lr_rate": 0.1}, bf16=cuda, gradient_checkpointing=args.ckpt,
                    dataloader_num_workers=args.workers if cuda else 0, dataloader_persistent_workers=cuda and args.workers > 0,
                    model_init_kwargs={"dtype": torch.bfloat16 if cuda else torch.float32, **({} if cuda else {"device_map": None})}, logging_steps=10,
                    save_steps=500, save_total_limit=2, report_to="none", seed=args.seed, use_cpu=not cuda and not torch.backends.mps.is_available())
    peft = LoraConfig(r=64, lora_alpha=128, target_modules="all-linear", task_type="CAUSAL_LM") if args.lora else None
    model = args.model
    if not args.lora and not args.train_vision:   # frozen ViT + merger: no vision backward, and the pretrained features stay intact
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16 if cuda else torch.float32)
        frozen = 0
        for n, p in model.named_parameters():
            if "visual" in n:
                p.requires_grad, frozen = False, frozen + p.numel()
        print(f"froze {frozen / 1e6:.0f}M vision params", flush=True)
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=peft, processing_class=processor(args.model))
    # Resume whenever a checkpoint exists: Modal retries a preempted container with the same args, and without this
    # the 3B SFT restarted from step 0 twice.
    import glob
    trainer.train(resume_from_checkpoint=True if (args.resume or glob.glob(os.path.join(args.out, "checkpoint-*"))) else None)
    trainer.save_model(args.out)
    trainer.processing_class.save_pretrained(args.out)


def merge(adapter: str, out: str) -> None:
    """Fold a LoRA into its base so the result is a plain model dir (the next DAgger round's start, RL's start)."""
    import json
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText
    base = json.load(open(os.path.join(adapter, "adapter_config.json")))["base_model_name_or_path"]
    m = PeftModel.from_pretrained(AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16), adapter)
    m.merge_and_unload().save_pretrained(out)
    processor(adapter).save_pretrained(out)


if __name__ == "__main__":
    main()
