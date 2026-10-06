"""Synthetic keyboard, camera, and click input via Quartz. Needs Accessibility permission to reach other apps."""

from __future__ import annotations

import time

import Quartz

from recorder.events import AGENT_USERDATA_MAGIC, _KEYCODE_NAMES

_NAME_TO_KEYCODE = {v: k for k, v in _KEYCODE_NAMES.items()}   # the recorder's map read backwards, so we can press by name

RDOWN_SETTLE_S = 0.01   # calibration knob: the old 200ms-tick code waited 80ms here; raise it if Roblox misses short drags

_SOURCE = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)


def _tag(event) -> None:
    Quartz.CGEventSetIntegerValueField(event, Quartz.kCGEventSourceUserData, AGENT_USERDATA_MAGIC)   # marks the event ours, so the recorder never mistakes it for you


def _post(event) -> None:
    _tag(event)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)


def key_down(name: str) -> None:
    code = _NAME_TO_KEYCODE.get(name.lower())
    if code is None:
        raise ValueError(f"Unknown key name: {name!r}")
    _post(Quartz.CGEventCreateKeyboardEvent(_SOURCE, code, True))


def key_up(name: str) -> None:
    code = _NAME_TO_KEYCODE.get(name.lower())
    if code is None:
        raise ValueError(f"Unknown key name: {name!r}")
    _post(Quartz.CGEventCreateKeyboardEvent(_SOURCE, code, False))


def release_keys(*names: str) -> None:
    """Best-effort release of everything, for aborts and shutdown."""
    for n in names:
        try:
            key_up(n)
        except Exception:
            pass


def look(dx: int, duration_ms: int, center: tuple[float, float] = (960.0, 540.0), dy: int = 0) -> None:
    """Rotate the camera by (dx, dy) pixels of right-drag, positive meaning right / down."""
    cx, cy = center
    steps = max(6, max(abs(dx), abs(dy)) // 6)   # enough small steps that Roblox reads a smooth drag rather than one teleport
    sx, sy = dx / steps, dy / steps
    step_dur = (duration_ms / 1000.0) / steps
    x, y = cx, cy

    down = Quartz.CGEventCreateMouseEvent(_SOURCE, Quartz.kCGEventRightMouseDown, Quartz.CGPoint(x, y), 1)
    _post(down)   # third-person camera only rotates while the right button is held
    time.sleep(RDOWN_SETTLE_S)

    ax = ay = 0.0
    for _ in range(steps):
        ax += sx
        ay += sy
        ix, iy = int(round(ax)), int(round(ay))   # carry the rounding remainder so the total drag is exact
        ax -= ix
        ay -= iy
        x += ix
        y += iy
        drag = Quartz.CGEventCreateMouseEvent(_SOURCE, Quartz.kCGEventRightMouseDragged, Quartz.CGPoint(x, y), 1)
        Quartz.CGEventSetIntegerValueField(drag, Quartz.kCGMouseEventDeltaX, ix)   # Roblox reads this delta field, not the cursor position
        Quartz.CGEventSetIntegerValueField(drag, Quartz.kCGMouseEventDeltaY, iy)
        _post(drag)
        time.sleep(step_dur)

    up = Quartz.CGEventCreateMouseEvent(_SOURCE, Quartz.kCGEventRightMouseUp, Quartz.CGPoint(x, y), 0)
    _post(up)


def look_locked(dx: int, dy: int, duration_ms: int, center: tuple[float, float]) -> None:
    """Shift-lock camera: Roblox turns on plain mouse-move deltas, no button held."""
    steps = max(4, max(abs(dx), abs(dy)) // 6)
    ax = ay = 0.0
    for _ in range(steps):
        ax += dx / steps
        ay += dy / steps
        ix, iy = int(round(ax)), int(round(ay))
        ax -= ix
        ay -= iy
        ev = Quartz.CGEventCreateMouseEvent(_SOURCE, Quartz.kCGEventMouseMoved, Quartz.CGPoint(*center), 0)
        Quartz.CGEventSetIntegerValueField(ev, Quartz.kCGMouseEventDeltaX, ix)
        Quartz.CGEventSetIntegerValueField(ev, Quartz.kCGMouseEventDeltaY, iy)
        _post(ev)
        time.sleep(duration_ms / 1000.0 / steps)


def move_mouse(x: float, y: float) -> None:
    _post(Quartz.CGEventCreateMouseEvent(_SOURCE, Quartz.kCGEventMouseMoved, Quartz.CGPoint(x, y), 0))


def scroll(lines: int) -> None:
    """Positive scrolls up, which zooms the Roblox camera in."""
    _post(Quartz.CGEventCreateScrollWheelEvent(_SOURCE, Quartz.kCGScrollEventUnitLine, 1, int(lines)))


def mouse_button(x: float, y: float, down: bool) -> None:
    """Press or release the left button without the other half: holding it is how Roblox tools fire and reel."""
    kind = Quartz.kCGEventLeftMouseDown if down else Quartz.kCGEventLeftMouseUp
    _post(Quartz.CGEventCreateMouseEvent(_SOURCE, kind, Quartz.CGPoint(x, y), Quartz.kCGMouseButtonLeft))


def click(x: float, y: float, button: str = "left") -> None:
    """A single click at absolute screen coordinates, which is most of what simulators and tycoons need."""
    if button == "left":
        down, up = Quartz.kCGEventLeftMouseDown, Quartz.kCGEventLeftMouseUp
        btn = Quartz.kCGMouseButtonLeft
    else:
        down, up = Quartz.kCGEventRightMouseDown, Quartz.kCGEventRightMouseUp
        btn = Quartz.kCGMouseButtonRight
    _post(Quartz.CGEventCreateMouseEvent(_SOURCE, down, Quartz.CGPoint(x, y), btn))
    time.sleep(0.04)   # a press and release in the same instant is often swallowed by the UI
    _post(Quartz.CGEventCreateMouseEvent(_SOURCE, up, Quartz.CGPoint(x, y), btn))
