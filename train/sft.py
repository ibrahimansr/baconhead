"""Imitation with TRL's SFTTrainer: sim expert data for the base model, your recorded play for per-game LoRAs.

    python -m train.sft --data data/sim_sft --out models/base                             # full finetune
    python -m train.sft --model models/base --data data/demos_toh --lora --out models/toh # per-game adapter
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
    ap.add_argument("--lr", type=float, default=None, help="default 1e-5 full, 1e-4 LoRA")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int, default=8, help="dataloader processes: image decoding/preprocessing is the bottleneck")
    ap.add_argument("--ckpt", action="store_true", help="gradient checkpointing (only needed if memory runs out)")
    args = ap.parse_args(argv)
    runlog.start(args.out, vars(args), git)
    cuda = torch.cuda.is_available()
    ds = load(args.data).shuffle(seed=args.seed)
    cfg = SFTConfig(output_dir=args.out, max_length=None, per_device_train_batch_size=args.batch,
                    gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs, max_steps=args.max_steps,
                    learning_rate=args.lr or (1e-4 if args.lora else 1e-5), bf16=cuda, gradient_checkpointing=args.ckpt,
                    dataloader_num_workers=args.workers if cuda else 0, dataloader_persistent_workers=cuda and args.workers > 0,
                    model_init_kwargs={"dtype": torch.bfloat16 if cuda else torch.float32, **({} if cuda else {"device_map": None})}, logging_steps=10,
                    save_steps=500, save_total_limit=2, report_to="none", seed=args.seed, use_cpu=not cuda and not torch.backends.mps.is_available())
    peft = LoraConfig(r=64, lora_alpha=128, target_modules="all-linear", task_type="CAUSAL_LM") if args.lora else None
    trainer = SFTTrainer(model=args.model, args=cfg, train_dataset=ds, peft_config=peft, processing_class=processor(args.model))
    # Resume whenever a checkpoint exists: Modal retries a preempted container with the same args, and without this
    # the 3B SFT restarted from step 0 twice.
    import glob
    trainer.train(resume_from_checkpoint=True if (args.resume or glob.glob(os.path.join(args.out, "checkpoint-*"))) else None)
    trainer.save_model(args.out)
    trainer.processing_class.save_pretrained(args.out)


if __name__ == "__main__":
    main()
