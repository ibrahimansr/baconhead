"""
DPO pair dataset for GameSense.

Builds (chosen, rejected) frame pairs from existing GameSense collect output.

Preference mapping (higher = more preferred):
  playing  (0)  → rank 3  ← chosen when paired with lower-ranked states
  dead     (1)  → rank 0  ← always rejected against anything else
  menu     (2)  → rank 1
  danger   (3)  → rank 2

build_pairs_from_collect(data_dir)
    Reads data.npz (frames, labels) and returns lists of
    (chosen_frame, rejected_frame) numpy arrays (H, W, 3) uint8.

DPOPairDataset
    PyTorch Dataset wrapping the pairs + ViT processor.
"""

import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# Rank of each state index: higher rank = preferred
# STATE_LABELS = ["playing", "dead", "menu", "danger"]
_RANK = {
    0: 3,  # playing  — most preferred
    1: 0,  # dead     — least preferred
    2: 1,  # menu
    3: 2,  # danger
}

# Minimum frames apart for a pair to be considered a meaningful contrast
_MIN_DIFF = 1
# Max pairs to extract per data file (keep memory manageable)
_MAX_PAIRS = 4000


def build_pairs_from_collect(
    data_dir: str,
    max_pairs: int = _MAX_PAIRS,
    seed: int = 42,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load frames from *data_dir/data.npz* and build (chosen, rejected) pairs.

    Auto-detects two schemas:

    Schema A — vision.collect output (keys: frames, labels)
        labels follow STATE_LABELS: 0=playing, 1=dead, 2=menu, 3=danger
        Preference rank: playing(3) > danger(2) > menu(1) > dead(0)
        Pairs are built between all state combinations where rank(A) > rank(B).

    Schema B — episode_data output (keys: frames, survived, flow_mag, ...)
        survived: 1 = episode ended with survival, 0 = fell/died
        chosen  = survived==1 frames
        rejected = survived==0 frames + N frames preceding each death (danger signal)

    Returns:
        chosen_frames : list of np.ndarray (H, W, 3) uint8
        rejected_frames: list of np.ndarray (H, W, 3) uint8
    """
    npz_path = os.path.join(data_dir, "data.npz")
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(
            f"No data at {npz_path}. "
            f"Run: python -m vision.collect --seconds 120 --out {data_dir}"
        )

    data = np.load(npz_path, allow_pickle=False)
    frames: np.ndarray = data["frames"]  # (N, H, W, 3)
    rng = np.random.default_rng(seed)

    # ── Schema detection ──────────────────────────────────────────────────────
    if "labels" in data.files:
        return _pairs_from_labels(frames, data["labels"].astype(int), rng, max_pairs)
    elif "survived" in data.files:
        return _pairs_from_survived(frames, data["survived"].astype(int), rng, max_pairs)
    elif "actions" in data.files:
        return _pairs_from_actions(frames, data["actions"].astype(int), rng, max_pairs)
    else:
        raise ValueError(
            f"Unrecognised data schema in {npz_path}. "
            f"Expected 'labels', 'survived', or 'actions' key. Found: {list(data.files)}"
        )


def _pairs_from_labels(
    frames: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    max_pairs: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Build DPO pairs from vision.collect 4-class labels."""
    # Group indices by label
    groups: dict = {}
    for idx in range(4):
        groups[idx] = np.where(labels == idx)[0]

    state_pairs = [
        (a, b)
        for a in range(4)
        for b in range(4)
        if _RANK[a] > _RANK[b] and len(groups[a]) > 0 and len(groups[b]) > 0
    ]

    budget = max_pairs // max(len(state_pairs), 1)
    chosen_frames: List[np.ndarray] = []
    rejected_frames: List[np.ndarray] = []

    for higher_state, lower_state in state_pairs:
        higher_idxs = groups[higher_state]
        lower_idxs = groups[lower_state]
        n = min(len(higher_idxs), len(lower_idxs), budget)
        if n == 0:
            continue
        chosen_sel = rng.choice(higher_idxs, size=n, replace=False)
        rejected_sel = rng.choice(lower_idxs, size=n, replace=False)
        for c_idx, r_idx in zip(chosen_sel, rejected_sel):
            chosen_frames.append(frames[c_idx])
            rejected_frames.append(frames[r_idx])

    if not chosen_frames:
        raise ValueError(
            "Could not build DPO pairs — need at least two different game states "
            "in the data. Collect more gameplay data first."
        )

    print(
        f"[dpo/dataset] Built {len(chosen_frames)} pairs (label schema) "
        f"from {len(labels)} frames",
        flush=True,
    )
    from vision.game_sense import STATE_LABELS
    for a, b in state_pairs:
        n = min(len(groups[a]), len(groups[b]), budget)
        if n > 0:
            print(
                f"  {STATE_LABELS[a]:8s} (chosen) vs {STATE_LABELS[b]:8s} (rejected): {n}",
                flush=True,
            )
    return chosen_frames, rejected_frames


def _pairs_from_survived(
    frames: np.ndarray,
    survived: np.ndarray,
    rng: np.random.Generator,
    max_pairs: int,
    danger_lookback: int = 5,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Build DPO pairs from episode_data binary survived signal.

    chosen   = frames where survived==1
    rejected = frames where survived==0, plus *danger_lookback* frames
               immediately preceding each death (these are near-death states
               that should be avoided — analogous to the 'danger' label).
    """
    survived_idxs = np.where(survived == 1)[0]
    fell_idxs = np.where(survived == 0)[0]

    # Expand rejection set: include frames just before each death
    rejected_set = set(fell_idxs.tolist())
    for fi in fell_idxs:
        for k in range(1, danger_lookback + 1):
            if fi - k >= 0:
                rejected_set.add(int(fi - k))
    # Don't include survived frames in rejected set
    rejected_set -= set(survived_idxs.tolist())
    rejected_idxs = np.array(sorted(rejected_set), dtype=int)

    n_chosen = len(survived_idxs)
    n_rejected = len(rejected_idxs)

    if n_chosen == 0 or n_rejected == 0:
        raise ValueError(
            "Could not build DPO pairs from survived labels — need both survived "
            "and fell episodes in the data."
        )

    n = min(n_chosen, n_rejected, max_pairs)
    chosen_sel = rng.choice(survived_idxs, size=n, replace=n_chosen < n)
    rejected_sel = rng.choice(rejected_idxs, size=n, replace=n_rejected < n)

    chosen_frames = [frames[i] for i in chosen_sel]
    rejected_frames = [frames[i] for i in rejected_sel]

    print(
        f"[dpo/dataset] Built {n} pairs (survived schema): "
        f"chosen={n_chosen} survived frames, rejected={len(rejected_idxs)} frames "
        f"(fell + {danger_lookback}-frame lookback)",
        flush=True,
    )
    return chosen_frames, rejected_frames


def _pairs_from_actions(
    frames: np.ndarray,
    actions: np.ndarray,
    rng: np.random.Generator,
    max_pairs: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Build DPO pairs from policy_data action labels.

    action_names (from config.json): W(0) A(1) S(2) D(3) space(4) none(5) look_left(6) look_right(7)

    chosen   = frames where a movement action was taken (W/A/S/D/space = 0-4)
               these represent frames where the bot was actively making progress.
    rejected = frames where action was 'none' (5) — the bot was idle/stuck.
               look actions (6,7) are neutral and excluded from both groups.
    """
    # Movement: W(0) A(1) S(2) D(3) space(4)
    movement_idxs = np.where((actions >= 0) & (actions <= 4))[0]
    # Idle: none(5)
    idle_idxs = np.where(actions == 5)[0]

    n_chosen = len(movement_idxs)
    n_rejected = len(idle_idxs)

    if n_chosen == 0 or n_rejected == 0:
        raise ValueError(
            "Could not build DPO pairs from action labels — need both movement "
            "and idle frames in policy_data. The bot must have been active and idle."
        )

    n = min(n_chosen, n_rejected, max_pairs)
    chosen_sel = rng.choice(movement_idxs, size=n, replace=n_chosen < n)
    rejected_sel = rng.choice(idle_idxs, size=n, replace=n_rejected < n)

    chosen_frames = [frames[i] for i in chosen_sel]
    rejected_frames = [frames[i] for i in rejected_sel]

    print(
        f"[dpo/dataset] Built {n} pairs (actions schema): "
        f"chosen={n_chosen} movement frames, rejected={n_rejected} idle frames",
        flush=True,
    )
    return chosen_frames, rejected_frames


def save_pairs(
    chosen: List[np.ndarray],
    rejected: List[np.ndarray],
    out_path: str,
) -> None:
    """Save DPO pairs to a .npz file for later reuse."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    chosen_arr = np.stack(chosen, axis=0)
    rejected_arr = np.stack(rejected, axis=0)
    np.savez(out_path, chosen=chosen_arr, rejected=rejected_arr)
    print(f"[dpo/dataset] Saved {len(chosen)} pairs → {out_path}", flush=True)


def load_pairs(
    pairs_path: str,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load DPO pairs from a .npz file."""
    data = np.load(pairs_path, allow_pickle=False)
    chosen = list(data["chosen"])
    rejected = list(data["rejected"])
    print(f"[dpo/dataset] Loaded {len(chosen)} pairs ← {pairs_path}", flush=True)
    return chosen, rejected


class DPOPairDataset(Dataset):
    """
    Torch Dataset yielding (chosen_pixel_values, rejected_pixel_values) tensors.

    Each item is a dict with keys:
        "chosen"   : FloatTensor (3, 224, 224)
        "rejected" : FloatTensor (3, 224, 224)
    """

    def __init__(self, chosen: List[np.ndarray], rejected: List[np.ndarray], processor):
        assert len(chosen) == len(rejected), "chosen/rejected must be same length"
        self.chosen = chosen
        self.rejected = rejected
        self.processor = processor

    def __len__(self) -> int:
        return len(self.chosen)

    def __getitem__(self, i: int) -> dict:
        from PIL import Image

        def _process(arr: np.ndarray) -> torch.Tensor:
            pil = Image.fromarray(arr.astype(np.uint8))
            inputs = self.processor(images=pil, return_tensors="pt")
            return inputs["pixel_values"].squeeze(0)

        return {
            "chosen": _process(self.chosen[i]),
            "rejected": _process(self.rejected[i]),
        }
