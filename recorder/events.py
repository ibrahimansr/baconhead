"""Every keyboard and mouse event, timestamped on the same clock as the frames. Needs Input Monitoring permission."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, asdict
from typing import Callable, Deque, Optional

import Quartz

from core.clock import mach_ticks_to_ns

AGENT_USERDATA_MAGIC = 0x7BAC04   # the injector stamps this into every event it posts, so agent input never gets read as yours

_KEYCODE_NAMES = {   # Apple virtual keycodes, named for the keys obbies and simulators actually use
    0: "a", 1: "s", 2: "d", 13: "w", 49: "space", 56: "shift", 60: "rshift",
    14: "e", 12: "q", 15: "r", 17: "t", 9: "v", 8: "c", 6: "z", 7: "x",
    48: "tab", 53: "escape", 36: "return", 51: "delete",
    123: "left", 124: "right", 125: "down", 126: "up",
    18: "1", 19: "2", 20: "3", 21: "4", 23: "5", 22: "6", 26: "7", 28: "8", 25: "9", 29: "0",
}


def keycode_name(code: int) -> str:
    return _KEYCODE_NAMES.get(int(code), f"key:{int(code)}")   # unknown keys are still logged rather than dropped


@dataclass(slots=True)
class InputEvent:
    t_ns: int
    kind: str
    source: str            # "human" means it is a free action label, "agent" means we caused it
    key: Optional[str] = None
    button: Optional[str] = None
    x: float = 0.0
    y: float = 0.0
    dx: int = 0            # relative mouse delta, which is the camera-look channel
    dy: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


_KEY_TYPES = {
    Quartz.kCGEventKeyDown: "key_down",
    Quartz.kCGEventKeyUp: "key_up",
    Quartz.kCGEventFlagsChanged: "flags",
}
_BUTTON_TYPES = {
    Quartz.kCGEventLeftMouseDown: ("mouse_down", "left"),
    Quartz.kCGEventLeftMouseUp: ("mouse_up", "left"),
    Quartz.kCGEventRightMouseDown: ("mouse_down", "right"),
    Quartz.kCGEventRightMouseUp: ("mouse_up", "right"),
}
_MOVE_TYPES = {
    Quartz.kCGEventMouseMoved: "mouse_move",
    Quartz.kCGEventLeftMouseDragged: "mouse_drag",
    Quartz.kCGEventRightMouseDragged: "mouse_drag",
}

_EVENT_MASK = (
    Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown)
    | Quartz.CGEventMaskBit(Quartz.kCGEventKeyUp)
    | Quartz.CGEventMaskBit(Quartz.kCGEventFlagsChanged)
    | Quartz.CGEventMaskBit(Quartz.kCGEventLeftMouseDown)
    | Quartz.CGEventMaskBit(Quartz.kCGEventLeftMouseUp)
    | Quartz.CGEventMaskBit(Quartz.kCGEventRightMouseDown)
    | Quartz.CGEventMaskBit(Quartz.kCGEventRightMouseUp)
    | Quartz.CGEventMaskBit(Quartz.kCGEventMouseMoved)
    | Quartz.CGEventMaskBit(Quartz.kCGEventLeftMouseDragged)
    | Quartz.CGEventMaskBit(Quartz.kCGEventRightMouseDragged)
    | Quartz.CGEventMaskBit(Quartz.kCGEventScrollWheel)
)


class InputRecorder:
    """A listen-only tap on the whole system's input, running its own run loop on a daemon thread."""

    def __init__(self, maxlen: int = 200_000, on_event: Optional[Callable[[InputEvent], None]] = None):
        self._buf: Deque[InputEvent] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._on_event = on_event
        self._tap = None
        self._runloop = None
        self._thread: Optional[threading.Thread] = None
        self._source = None

    def start(self) -> None:
        if self._thread is not None:
            return
        ready = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(ready,), daemon=True)
        self._thread.start()
        ready.wait(timeout=5.0)
        if self._tap is None:
            raise PermissionError(
                "CGEventTapCreate returned None — grant Input Monitoring permission "
                "to your terminal: System Settings → Privacy & Security → Input Monitoring."
            )

    def _run(self, ready: threading.Event) -> None:
        tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap,
            Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionListenOnly,   # listen-only, so we can never swallow or delay your real input
            _EVENT_MASK,
            self._callback,
            None,
        )
        self._tap = tap
        if tap is None:
            ready.set()
            return
        self._source = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
        self._runloop = Quartz.CFRunLoopGetCurrent()
        Quartz.CFRunLoopAddSource(self._runloop, self._source, Quartz.kCFRunLoopCommonModes)
        Quartz.CGEventTapEnable(tap, True)
        ready.set()
        Quartz.CFRunLoopRun()   # blocks this thread forever, which is why it is a daemon

    def stop(self) -> None:
        if self._runloop is not None:
            Quartz.CFRunLoopStop(self._runloop)
        self._thread = None
        self._tap = None

    def snapshot(self) -> list[InputEvent]:
        with self._lock:
            return list(self._buf)

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()

    def recent(self, window_ns: int) -> list[InputEvent]:
        """The last window of events, for answering what exactly preceded a death."""
        from core.clock import now_ns
        cutoff = now_ns() - window_ns
        with self._lock:
            return [e for e in self._buf if e.t_ns >= cutoff]

    def _callback(self, proxy, type_, event, refcon):
        try:
            ev = self._decode(type_, event)
            if ev is not None:
                with self._lock:
                    self._buf.append(ev)
                if self._on_event is not None:
                    self._on_event(ev)
        except Exception:
            pass   # an exception escaping into the run loop would tear the tap down, so the event is dropped instead
        return event

    def _decode(self, type_, event) -> Optional[InputEvent]:
        if type_ in (Quartz.kCGEventTapDisabledByTimeout, Quartz.kCGEventTapDisabledByUserInput):
            if self._tap is not None:
                Quartz.CGEventTapEnable(self._tap, True)   # macOS disables a slow tap, so it has to be turned back on
            return None

        t_ns = mach_ticks_to_ns(Quartz.CGEventGetTimestamp(event))   # the same units as our frame timestamps
        user_data = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGEventSourceUserData)
        source = "agent" if user_data == AGENT_USERDATA_MAGIC else "human"

        if type_ in _KEY_TYPES:
            kind = _KEY_TYPES[type_]
            code = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
            return InputEvent(t_ns=t_ns, kind=kind, source=source, key=keycode_name(code))

        if type_ in _BUTTON_TYPES:
            kind, button = _BUTTON_TYPES[type_]
            loc = Quartz.CGEventGetLocation(event)
            return InputEvent(t_ns=t_ns, kind=kind, source=source, button=button, x=loc.x, y=loc.y)

        if type_ in _MOVE_TYPES:
            loc = Quartz.CGEventGetLocation(event)
            dx = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGMouseEventDeltaX)
            dy = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGMouseEventDeltaY)
            return InputEvent(
                t_ns=t_ns, kind=_MOVE_TYPES[type_], source=source,
                x=loc.x, y=loc.y, dx=dx, dy=dy,
            )

        if type_ == Quartz.kCGEventScrollWheel:
            dy = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGScrollWheelEventDeltaAxis1)
            return InputEvent(t_ns=t_ns, kind="scroll", source=source, dy=dy)

        return None


if __name__ == "__main__":
    import time
    print("Starting input recorder for 6s — type/move the mouse in any window.")
    rec = InputRecorder()
    rec.start()
    time.sleep(6)
    evs = rec.snapshot()
    rec.stop()
    print(f"Captured {len(evs)} events.")
    for e in evs[:20]:
        print(f"  +{e.t_ns/1e6:10.2f}ms  {e.kind:11s} {e.source:5s} "
              f"key={e.key} btn={e.button} d=({e.dx},{e.dy})")
