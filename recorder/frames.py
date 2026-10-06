"""Screen capture that stamps each frame on the shared clock. Needs Screen Recording permission for real pixels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mss
import numpy as np

from core.clock import now_ns
from core.window import get_roblox_region, Region


@dataclass(slots=True)
class Frame:
    t_ns: int
    seq: int
    rgb: np.ndarray


class FrameSource:
    """Captures one fixed region, reusing a single mss handle because opening one per grab is slow."""

    def __init__(self, region: Optional[Region] = None, monitor: int = 1):
        self._region = region
        self._monitor = monitor
        self._sct = mss.MSS()
        self._seq = 0

    @classmethod
    def for_roblox(cls, wait_s: float = 0.0) -> "FrameSource":
        """Bind to the Roblox window, optionally waiting for it to appear."""
        import time
        region = get_roblox_region()
        deadline = time.monotonic() + wait_s
        while region is None and time.monotonic() < deadline:
            time.sleep(0.5)
            region = get_roblox_region()
        return cls(region=region)

    @property
    def region(self) -> Optional[Region]:
        return self._region

    def grab(self) -> Frame:
        box = self._region
        if box is None:
            mon = self._sct.monitors[self._monitor]
            box = {"left": mon["left"], "top": mon["top"], "width": mon["width"], "height": mon["height"]}
        raw = self._sct.grab(box)
        t = now_ns()   # stamped right after the grab returns, so the frame lines up with the input stream
        rgb = np.array(raw)[:, :, :3][:, :, ::-1]   # mss hands back BGRA, so alpha is dropped and the channels flipped
        frame = Frame(t_ns=t, seq=self._seq, rgb=np.ascontiguousarray(rgb))
        self._seq += 1
        return frame

    def close(self) -> None:
        try:
            self._sct.close()
        except Exception:
            pass
