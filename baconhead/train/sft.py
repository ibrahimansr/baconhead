import argparse
import os

import torch

from baconhead.model.vlm import MODEL_ID, processor
from baconhead.train import runlog
from baconhead.train.dataset import load


def main(argv=None, git=None):
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--data", required=True, help="comma-separated dataset dirs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=None, help="default 2e-5 full, 1e-4 LoRA")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int, default=8, help="dataloader processes")
    ap.add_argument("--ckpt", action="store_true", help="gradient checkpointing")
    ap.add_argument("--train-vision", action="store_true", help="also train the vision tower (frozen by default)")
    args = ap.parse_args(argv)
    runlog.start(args.out, vars(args), git)
    cuda = torch.cuda.is_available()
    ds = load(args.data).shuffle(seed=args.seed)
    cfg = SFTConfig(output_dir=args.out, max_length=None, per_device_train_batch_size=args.batch,
                    gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs, max_steps=args.max_steps,
                    learning_rate=args.lr or (1e-4 if args.lora else 2e-5), warmup_steps=0.05, lr_scheduler_type="cosine_with_min_lr",
                    lr_scheduler_kwargs={"min_lr_rate": 0.1}, bf16=cuda, gradient_checkpointing=args.ckpt,
                    dataloader_num_workers=args.workers if cuda else 0, dataloader_persistent_workers=cuda and args.workers > 0,
                    model_init_kwargs={"dtype": torch.bfloat16 if cuda else torch.float32, **({} if cuda else {"device_map": None})},  # MPS hangs on device_map
                    logging_steps=10,
                    save_steps=500, save_total_limit=2, report_to="none", seed=args.seed, use_cpu=not cuda and not torch.backends.mps.is_available())
    peft = LoraConfig(r=64, lora_alpha=128, target_modules="all-linear", task_type="CAUSAL_LM") if args.lora else None
    model = args.model
    if not args.lora and not args.train_vision:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16 if cuda else torch.float32)
        frozen = 0
        for n, p in model.named_parameters():
            if "visual" in n:
                p.requires_grad, frozen = False, frozen + p.numel()
        print(f"froze {frozen / 1e6:.0f}M vision params", flush=True)
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=peft, processing_class=processor(args.model))
    import glob   # resume if a checkpoint exists (a preempted Modal container reruns with the same args)
    trainer.train(resume_from_checkpoint=True if (args.resume or glob.glob(os.path.join(args.out, "checkpoint-*"))) else None)
    trainer.save_model(args.out)
    trainer.processing_class.save_pretrained(args.out)


# fold a LoRA into its base: a plain model dir vLLM can serve
def merge(adapter, out):
    import json
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText
    base = json.load(open(os.path.join(adapter, "adapter_config.json")))["base_model_name_or_path"]
    m = PeftModel.from_pretrained(AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16), adapter)
    m.merge_and_unload().save_pretrained(out)
    processor(adapter).save_pretrained(out)


if __name__ == "__main__":
    main()
