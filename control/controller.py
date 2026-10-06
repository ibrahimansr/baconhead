"""Plays one Action per 100ms tick on the real Roblox window, through the Quartz injector.

Keys are held across ticks and only the differences are posted, exactly like hands. The camera turn is a
right-drag sized at 2 px/degree and squeezed into the tick; in shift lock it becomes a
plain mouse-move delta, because that's what Roblox reads then.
"""

from __future__ import annotations

from core.actions import Action
from control import inject

LOOK_PX_PER_DEGREE = 2.0   # measured on the actually-works branch: 180px of right-drag = 90° in Roblox
LOOK_BUDGET_MS = 85   # calibration knob: drag time per tick, leaving headroom inside the 100ms tick


class Controller:
    def __init__(self, region: dict):
        self.region = region
        self.held: set[str] = set()
        self.shiftlock = False
        self.cursor = None

    def _xy(self, u: float, v: float) -> tuple[float, float]:
        r = self.region
        return r["left"] + u * r["width"], r["top"] + v * r["height"]

    def apply(self, a: Action) -> None:
        center = self._xy(0.5, 0.5)
        x, y = self._xy(*a.cursor)
        for k in sorted(a.keys - self.held):
            if k == "lmb":
                inject.mouse_button(*((x, y) if not self.shiftlock else center), down=True)
                continue
            inject.key_down("escape" if k == "esc" else k)
            if k == "shift":
                self.shiftlock = not self.shiftlock   # Roblox toggles shift lock on press
        for k in sorted(self.held - a.keys):
            if k == "lmb":
                inject.mouse_button(*((x, y) if not self.shiftlock else center), down=False)
                continue
            inject.key_up("escape" if k == "esc" else k)
        self.held = set(a.keys)
        dx, dy = int(round(a.look_x * LOOK_PX_PER_DEGREE)), int(round(a.look_y * LOOK_PX_PER_DEGREE))
        if dx or dy:
            if self.shiftlock:
                inject.look_locked(dx, dy, LOOK_BUDGET_MS, center)
            else:
                inject.look(dx, LOOK_BUDGET_MS, center=(x, y), dy=dy)   # drag from where the cursor is, so it doesn't jump
        elif not self.shiftlock and (x, y) != self.cursor:
            inject.move_mouse(x, y)
        self.cursor = (x, y)
        if a.click:
            inject.click(*((x, y) if not self.shiftlock else center))
        if a.scroll:
            inject.scroll(-a.scroll)

    def release(self) -> None:
        if "lmb" in self.held and self.cursor:
            inject.mouse_button(*self.cursor, down=False)
        inject.release_keys(*("escape" if k == "esc" else k for k in self.held if k != "lmb"))
        self.held.clear()
