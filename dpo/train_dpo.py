"""
DPO training loop for GameSense.

Implements Direct Preference Optimization (Rafailov et al., 2023) on top of
the GameSense ViT classifier.

The policy logits for a frame are the raw classification logits.
We use the log-sum-exp of logits as a scalar "log-probability" for DPO:
    log π(x) = logsumexp(logits(x))   ← represents overall model confidence

This is a lightweight adaptation: we don't need a token-level LM.
The key insight is that DPO simply needs a scalar score per sample so that:
  - chosen frames get higher scores than rejected frames
  - relative to the frozen reference model

DPO loss:
  L = -log σ(β * [(log π_θ(w) - log π_ref(w)) - (log π_θ(l) - log π_ref(l))])

where w = chosen (winner), l = rejected (loser).

Usage:
  from dpo.train_dpo import train_dpo
  train_dpo(data_dir="episode_data", policy_path="game_sense.pt", out_path="game_sense.pt")
"""

import copy
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from vision.game_sense import GameSense, load_game_sense, save_game_sense
from dpo.dataset import build_pairs_from_collect, DPOPairDataset


# ── DPO Loss ──────────────────────────────────────────────────────────────────


def _model_log_score(model: GameSense, pixel_values: torch.Tensor) -> torch.Tensor:
    """
    Compute a scalar log-score for each frame in the batch.

    We take logsumexp(logits) as the model's "log partition function" —
    a scalar that is higher when the model is more confidently assigning
    probability mass to *any* state. This gives DPO a per-sample scalar
    that is differentiable and monotonically related to model confidence.

    Shape: (B,)
    """
    logits = model(pixel_values)           # (B, num_states)
    return torch.logsumexp(logits, dim=-1) # (B,)


def dpo_loss(
    policy: GameSense,
    ref: GameSense,
    chosen_pv: torch.Tensor,
    rejected_pv: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    Compute the DPO loss for a batch of (chosen, rejected) pairs.

    Args:
        policy:      the model being trained (gradients flow through)
        ref:         frozen reference model (no gradients)
        chosen_pv:   pixel_values for chosen (positive) frames (B, 3, 224, 224)
        rejected_pv: pixel_values for rejected (negative) frames (B, 3, 224, 224)
        beta:        KL penalty coefficient (higher = stay closer to ref)

    Returns:
        scalar loss tensor
    """
    # Policy scores (differentiable)
    log_pi_chosen = _model_log_score(policy, chosen_pv)
    log_pi_rejected = _model_log_score(policy, rejected_pv)

    # Reference scores (no gradient)
    with torch.no_grad():
        log_ref_chosen = _model_log_score(ref, chosen_pv)
        log_ref_rejected = _model_log_score(ref, rejected_pv)

    # Implicit reward differences
    pi_logratios = log_pi_chosen - log_pi_rejected
    ref_logratios = log_ref_chosen - log_ref_rejected

    loss = -F.logsigmoid(beta * (pi_logratios - ref_logratios))
    return loss.mean()


# ── Training ──────────────────────────────────────────────────────────────────


def train_dpo(
    data_dir: str,
    policy_path: str,
    out_path: str,
    epochs: int = 3,
    batch_size: int = 8,
    beta: float = 0.1,
    lr: float = 5e-6,
    val_frac: float = 0.1,
    max_pairs: int = 4000,
    dry_run: bool = False,
):
    """
    Fine-tune *policy_path* GameSense model with DPO on gameplay pairs from
    *data_dir/data.npz*.

    After training, saves the updated model + embedded reference to *out_path*.

    Args:
        data_dir:    directory containing data.npz from vision.collect
        policy_path: path to existing game_sense.pt (used as init for both
                     policy and reference)
        out_path:    where to write the DPO-fine-tuned checkpoint
        epochs:      number of DPO training epochs
        batch_size:  pairs per gradient step
        beta:        DPO temperature (0.05–0.5 typical)
        lr:          learning rate (keep small — fine-tuning only)
        val_frac:    fraction of pairs held out for validation
        max_pairs:   cap on pairs extracted from data.npz
        dry_run:     if True, build pairs + one forward pass, print loss, exit
    """
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[dpo] Device: {device}", flush=True)

    # ── Build pairs ───────────────────────────────────────────────────────────
    chosen_frames, rejected_frames = build_pairs_from_collect(data_dir, max_pairs=max_pairs)
    n = len(chosen_frames)
    print(f"[dpo] {n} pairs total", flush=True)

    # ── Load policy model ─────────────────────────────────────────────────────
    if os.path.isfile(policy_path):
        print(f"[dpo] Loading policy from {policy_path}", flush=True)
        policy = load_game_sense(policy_path, device=device)
    else:
        print(
            f"[dpo] No checkpoint at {policy_path} — initialising fresh GameSense",
            flush=True,
        )
        policy = GameSense().to(device)

    # ── Freeze reference copy ─────────────────────────────────────────────────
    ref = copy.deepcopy(policy)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False
    print("[dpo] Reference model frozen.", flush=True)

    # ── Dataset / DataLoader ──────────────────────────────────────────────────
    full_ds = DPOPairDataset(chosen_frames, rejected_frames, policy.processor)

    n_val = max(1, int(n * val_frac))
    n_train = n - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    if dry_run:
        print("[dpo] --dry-run: one forward pass only", flush=True)
        policy.train()
        batch = next(iter(train_loader))
        chosen_pv = batch["chosen"].to(device)
        rejected_pv = batch["rejected"].to(device)
        loss = dpo_loss(policy, ref, chosen_pv, rejected_pv, beta=beta)
        print(f"[dpo] dry-run loss = {loss.item():.4f}", flush=True)
        print("[dpo] dry-run complete — no checkpoint written.", flush=True)
        return

    # ── Optimiser ─────────────────────────────────────────────────────────────
    # Fine-tune the full model (backbone + head) with a very small LR
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        policy.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            chosen_pv = batch["chosen"].to(device)
            rejected_pv = batch["rejected"].to(device)

            opt.zero_grad()
            loss = dpo_loss(policy, ref, chosen_pv, rejected_pv, beta=beta)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train = total_loss / max(n_batches, 1)

        # ── Validate ───────────────────────────────────────────────────────
        val_loss = _eval_val_loss(policy, ref, val_loader, device, beta)
        print(
            f"[dpo] epoch {epoch}/{epochs}  "
            f"train_loss={avg_train:.4f}  val_loss={val_loss:.4f}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save policy weights + embedded reference
            _save_dpo_checkpoint(policy, ref, out_path)
            print(f"[dpo]   ↳ saved best → {out_path}", flush=True)

    print(
        f"[dpo] Training complete. Best val_loss={best_val_loss:.4f}. "
        f"Checkpoint: {out_path}",
        flush=True,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


@torch.no_grad()
def _eval_val_loss(
    policy: GameSense,
    ref: GameSense,
    loader: DataLoader,
    device: torch.device,
    beta: float,
) -> float:
    policy.eval()
    total, n = 0.0, 0
    for batch in loader:
        chosen_pv = batch["chosen"].to(device)
        rejected_pv = batch["rejected"].to(device)
        loss = dpo_loss(policy, ref, chosen_pv, rejected_pv, beta=beta)
        total += loss.item()
        n += 1
    return total / max(n, 1)


def _save_dpo_checkpoint(policy: GameSense, ref: GameSense, path: str):
    """
    Save the DPO-trained policy.  Embeds the reference model state so
    future DPO fine-tuning rounds can recover it.  Compatible with
    load_game_sense() — the 'model_state' key is identical.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state": policy.state_dict(),
            "ref_model_state": ref.state_dict(),
            "backbone": policy._backbone,
            "num_states": policy._num_states,
            "dpo": True,
        },
        path,
    )
