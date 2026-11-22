"""The on-disk format for a session: frames, input events, and semantic tags fused on one timeline."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
from PIL import Image

if TYPE_CHECKING:   # only a type annotation, and importing it for real would drag macOS Quartz onto the training box
    from recorder.events import InputEvent


@dataclass
class EpisodeWriter:
    out_dir: str
    region: Optional[dict] = None
    store_size: int = 256   # frames go to disk downscaled, so a few minutes of play stays a manageable array
    _frames: list = field(default_factory=list, init=False)
    _pts: list = field(default_factory=list, init=False)
    _events_fp: object = field(default=None, init=False)
    _tags_fp: object = field(default=None, init=False)

    def __post_init__(self):
        os.makedirs(self.out_dir, exist_ok=True)
        self._events_fp = open(os.path.join(self.out_dir, "events.jsonl"), "w")
        self._tags_fp = open(os.path.join(self.out_dir, "tags.jsonl"), "w")

    def add_frame(self, rgb: np.ndarray, t_ns: int) -> None:
        small = np.asarray(
            Image.fromarray(rgb).resize((self.store_size, self.store_size), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )
        self._frames.append(small)
        self._pts.append(int(t_ns))

    def add_event(self, ev: InputEvent) -> None:
        self._events_fp.write(json.dumps(ev.to_dict()) + "\n")

    def add_events(self, evs: list[InputEvent]) -> None:
        for ev in evs:
            self._events_fp.write(json.dumps(ev.to_dict()) + "\n")
        self._events_fp.flush()   # flushed as we go so a Ctrl+C still leaves a readable episode

    def add_tag(self, t_ns: int, kind: str, **data) -> None:
        """Mark something meaningful on the timeline: a death, a checkpoint, a respawn."""
        rec = {"t_ns": int(t_ns), "kind": kind, **data}
        self._tags_fp.write(json.dumps(rec) + "\n")
        self._tags_fp.flush()

    def close(self, extra_meta: Optional[dict] = None) -> str:
        from core.clock import _NUMER, _DENOM

        frames = (
            np.stack(self._frames, axis=0)
            if self._frames
            else np.empty((0, self.store_size, self.store_size, 3), dtype=np.uint8)
        )
        pts = np.array(self._pts, dtype=np.int64)
        np.save(os.path.join(self.out_dir, "frames.npy"), frames)
        np.save(os.path.join(self.out_dir, "frame_pts.npy"), pts)

        meta = {
            "n_frames": int(frames.shape[0]),
            "store_size": self.store_size,
            "region": self.region,
            "clock_timebase": {"numer": _NUMER, "denom": _DENOM},   # recorded so timestamps stay interpretable on another machine
            "duration_ns": int(pts[-1] - pts[0]) if len(pts) > 1 else 0,
        }
        if extra_meta:
            meta.update(extra_meta)
        with open(os.path.join(self.out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)   # written last, which is what marks the episode as complete

        self._events_fp.flush(); self._events_fp.close()
        self._tags_fp.flush(); self._tags_fp.close()
        return self.out_dir


@dataclass
class Episode:
    """A recorded episode read back for training."""
    frames: np.ndarray
    frame_pts: np.ndarray
    events: list[dict]
    tags: list[dict]
    meta: dict

    @classmethod
    def load(cls, path: str) -> "Episode":
        frames = np.load(os.path.join(path, "frames.npy"))
        pts = np.load(os.path.join(path, "frame_pts.npy"))
        events = _read_jsonl(os.path.join(path, "events.jsonl"))
        tags = _read_jsonl(os.path.join(path, "tags.jsonl"))
        with open(os.path.join(path, "meta.json")) as f:
            meta = json.load(f)
        return cls(frames=frames, frame_pts=pts, events=events, tags=tags, meta=meta)

    def frame_index_at(self, t_ns: int) -> int:
        """The frame nearest a timestamp, which is how a tag or a tick boundary finds its picture."""
        return int(np.argmin(np.abs(self.frame_pts - t_ns)))

    def events_between(self, t0_ns: int, t1_ns: int) -> list[dict]:
        return [e for e in self.events if t0_ns <= e["t_ns"] <= t1_ns]


def _read_jsonl(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def next_run_dir(root: str, prefix: str = "run") -> str:
    """The first unused demos/run_000 style directory, so recordings never overwrite each other."""
    os.makedirs(root, exist_ok=True)
    i = 0
    while os.path.exists(os.path.join(root, f"{prefix}_{i:03d}")):
        i += 1
    return os.path.join(root, f"{prefix}_{i:03d}")
