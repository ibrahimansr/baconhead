"""Synthetic keyboard, camera, and click input via Quartz. Needs Accessibility permission to reach other apps."""

from __future__ import annotations

import time

import Quartz

from recorder.events import AGENT_USERDATA_MAGIC, _KEYCODE_NAMES
from control.physics import LOOK_PX_PER_MS

_NAME_TO_KEYCODE = {v: k for k, v in _KEYCODE_NAMES.items()}   # the recorder's map read backwards, so we can press by name

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


def hold_key(name: str, duration_ms: int) -> None:
    key_down(name)
    try:
        time.sleep(max(0, duration_ms) / 1000.0)
    finally:
        key_up(name)   # in a finally block because an escaped key-down means the character runs forever


def tap_key(name: str, press_ms: int = 50) -> None:
    key_down(name)
    time.sleep(press_ms / 1000.0)
    key_up(name)


def release_keys(*names: str) -> None:
    """Best-effort release of everything, for aborts and shutdown."""
    for n in names:
        try:
            key_up(n)
        except Exception:
            pass


def look(dx: int, duration_ms: int, center: tuple[float, float] = (960.0, 540.0)) -> None:
    """Rotate the camera by dx pixels, positive meaning right."""
    cx, cy = center
    steps = max(20, abs(dx) // 6)   # enough small steps that Roblox reads a smooth drag rather than one teleport
    dx_step = dx / steps
    step_dur = (duration_ms / 1000.0) / steps
    x, y = cx, cy

    down = Quartz.CGEventCreateMouseEvent(_SOURCE, Quartz.kCGEventRightMouseDown, Quartz.CGPoint(x, y), 1)
    _post(down)   # third-person camera only rotates while the right button is held
    time.sleep(0.08)

    for _ in range(steps):
        x += dx_step
        drag = Quartz.CGEventCreateMouseEvent(_SOURCE, Quartz.kCGEventRightMouseDragged, Quartz.CGPoint(x, y), 1)
        Quartz.CGEventSetIntegerValueField(drag, Quartz.kCGMouseEventDeltaX, int(round(dx_step)))   # Roblox reads this delta field, not the cursor position
        Quartz.CGEventSetIntegerValueField(drag, Quartz.kCGMouseEventDeltaY, 0)
        _post(drag)
        time.sleep(step_dur)

    up = Quartz.CGEventCreateMouseEvent(_SOURCE, Quartz.kCGEventRightMouseUp, Quartz.CGPoint(x, y), 0)
    _post(up)
    time.sleep(0.05)


def look_degrees(degrees: float, center: tuple[float, float] = (960.0, 540.0)) -> None:
    from control.physics import degrees_to_px

    dx = degrees_to_px(degrees) * (1 if degrees >= 0 else -1)
    duration_ms = max(100, int(abs(dx) / LOOK_PX_PER_MS))
    look(dx, duration_ms, center=center)


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
