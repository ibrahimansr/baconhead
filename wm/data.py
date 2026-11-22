"""Training data: your recorded demos re-labelled onto the tick grid, plus optional Hugging Face gameplay."""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from recorder.episode import Episode
from wm.actions import STEP_MS, label_from_state
from wm.model import IMG, preprocess


def _resize(frames: np.ndarray) -> torch.Tensor:
    return torch.cat([preprocess(f) for f in frames])


def episode_to_ticks(path: str) -> dict:
    """One recorded episode to frames, actions, death labels, and progress labels on the 200ms grid."""
    ep = Episode.load(path)
    if len(ep.frame_pts) < 2:
        return {"frames": torch.zeros(0, 3, IMG, IMG), "actions": torch.zeros(0, dtype=torch.long)}

    step_ns = STEP_MS * 1_000_000
    t0, t1 = int(ep.frame_pts[0]), int(ep.frame_pts[-1])
    ticks = list(range(t0, t1 - step_ns, step_ns))   # the last partial tick is dropped so every window is full length

    keys: set[str] = set()
    events = sorted((e for e in ep.events if e.get("source") == "human"), key=lambda e: e["t_ns"])   # our own injected input is not a label
    ei = 0
    idxs, actions = [], []
    for t in ticks:
        look_dx = 0
        touched = set(keys)   # a key counts for this tick if it was held at any point in it, so a quick jump still registers
        while ei < len(events) and events[ei]["t_ns"] < t + step_ns:
            e = events[ei]
            if e["kind"] == "key_down" and e.get("key"):
                keys.add(e["key"])
                touched.add(e["key"])
            elif e["kind"] == "key_up" and e.get("key"):
                keys.discard(e["key"])
            elif e["kind"] in ("mouse_move", "mouse_drag"):
                look_dx += int(e.get("dx") or 0)   # relative deltas are the camera channel, not cursor position
            ei += 1
        idxs.append(ep.frame_index_at(t))   # the shared clock is what makes this frame the right one for this tick
        actions.append(label_from_state(touched, look_dx))

    n = len(idxs)
    alive = torch.full((n,), float("nan"))   # NaN means unlabelled, and the loss skips those ticks entirely
    for tag in ep.tags:
        if tag.get("kind") in ("death", "dead"):
            j = min(n - 1, max(0, (int(tag["t_ns"]) - t0) // step_ns))
            alive[j] = 0.0
            for back in range(1, 6):   # the second before a death was survivable, which is the contrast the head needs
                if j - back >= 0 and torch.isnan(alive[j - back]):
                    alive[j - back] = 1.0

    progress = torch.full((n,), float("nan"))
    goal = ""
    for rec in _read_jsonl(os.path.join(path, "labels.jsonl")):
        j = min(n - 1, max(0, (int(rec["t_ns"]) - t0) // step_ns))
        progress[j] = float(rec["score"]) / 10.0   # Claude scores 0-10, the value head regresses 0-1
        goal = rec.get("goal", goal)

    return {
        "frames": _resize(ep.frames[idxs]),
        "actions": torch.tensor(actions, dtype=torch.long),
        "alive": alive,
        "progress": progress,
        "goal": goal,
    }


def load_hf(name: str, split: str, frame_col: str, action_col: str, action_map: dict,
            limit: int = 5_000) -> dict:
    """A Hugging Face gameplay dataset in the same shape, for teaching the prior how a body moves in 3D."""
    from datasets import load_dataset

    from wm.actions import ID

    ds = load_dataset(name, split=split, streaming=True)   # streamed so a huge dataset never lands on disk
    frames, actions = [], []
    for row in ds:
        if len(actions) >= limit:
            break
        img = row[frame_col]
        frames.append(preprocess(np.asarray(img.convert("RGB") if hasattr(img, "convert") else img)))
        actions.append(ID.get(action_map.get(str(row[action_col]), "idle"), 0))   # anything we can't map becomes idle
    n = len(actions)
    return {
        "frames": torch.cat(frames) if frames else torch.zeros(0, 3, IMG, IMG),
        "actions": torch.tensor(actions, dtype=torch.long),
        "alive": torch.full((n,), float("nan")),   # someone else's gameplay has no death or progress labels of ours
        "progress": torch.full((n,), float("nan")),
        "goal": "",
    }


def sample_batch(tick_dict: dict, seq_len: int = 16, batch: int = 8):
    """Random contiguous windows, since dynamics can only be learned from consecutive ticks."""
    n = tick_dict["actions"].shape[0]
    if n < seq_len + 1:
        return None
    starts = torch.randint(0, n - seq_len, (batch,))
    sl = lambda key: torch.stack([tick_dict[key][s: s + seq_len] for s in starts])
    return sl("frames"), sl("actions"), sl("alive"), sl("progress")


def append_label(episode_dir: str, t_ns: int, goal: str, score: float) -> None:
    """Save one of Claude's progress judgements, which is the value head's entire training set."""
    with open(os.path.join(episode_dir, "labels.jsonl"), "a") as f:
        f.write(json.dumps({"t_ns": int(t_ns), "goal": goal, "score": float(score)}) + "\n")


def _read_jsonl(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def _demo() -> None:
    """Self-check that a synthetic episode re-labels onto the tick grid the way we think it does."""
    import tempfile

    from recorder.episode import EpisodeWriter
    from recorder.events import InputEvent
    from wm.actions import ID

    step_ns = STEP_MS * 1_000_000
    with tempfile.TemporaryDirectory() as d:
        w = EpisodeWriter(out_dir=d, store_size=64)
        for i in range(30):
            w.add_frame(np.full((64, 64, 3), i * 8, dtype=np.uint8), i * (step_ns // 5))   # five frames per tick
        w.add_events([
            InputEvent(t_ns=step_ns + 1000, kind="key_down", source="human", key="w"),
            InputEvent(t_ns=2 * step_ns + 1000, kind="key_down", source="human", key="space"),
            InputEvent(t_ns=2 * step_ns + 50_000_000, kind="key_up", source="human", key="space"),
            InputEvent(t_ns=3 * step_ns + 1000, kind="key_up", source="human", key="w"),
            InputEvent(t_ns=500, kind="key_down", source="agent", key="d"),
        ])
        w.add_tag(4 * step_ns, "death")
        w.close()

        t = episode_to_ticks(d)
        acts = t["actions"].tolist()
        assert len(acts) >= 5, acts
        assert acts[0] == ID["idle"], f"agent input leaked into the labels: {acts}"
        assert acts[1] == ID["w"], acts
        assert acts[2] == ID["w_jump"], acts   # space was pressed and released inside one tick and must still count
        assert acts[4] == ID["idle"], acts
        assert t["frames"].shape[1:] == (3, IMG, IMG)
        assert t["frames"].shape[0] == len(acts)
        assert t["alive"][4] == 0.0 and t["alive"][3] == 1.0
        assert torch.isnan(t["progress"]).all()

        append_label(d, 2 * step_ns, "climb", 8.0)
        t2 = episode_to_ticks(d)
        assert t2["progress"][2] == 0.8 and t2["goal"] == "climb"

    b = sample_batch({"frames": torch.rand(20, 3, IMG, IMG), "actions": torch.zeros(20, dtype=torch.long),
                      "alive": torch.full((20,), float("nan")), "progress": torch.full((20,), float("nan"))},
                     seq_len=8, batch=3)
    assert b[0].shape == (3, 8, 3, IMG, IMG) and b[1].shape == (3, 8)
    assert sample_batch({"actions": torch.zeros(3)}, seq_len=8, batch=2) is None   # too short to form one window
    print("wm.data ok")


if __name__ == "__main__":
    _demo()
