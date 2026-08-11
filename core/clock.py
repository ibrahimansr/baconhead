"""One monotonic timebase for everything, so a frame and the keypresses around it line up exactly."""

from __future__ import annotations

import ctypes
import time as _time

_libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)


class _MachTimebaseInfo(ctypes.Structure):
    _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]


_libc.mach_absolute_time.restype = ctypes.c_uint64
_libc.mach_timebase_info.argtypes = [ctypes.POINTER(_MachTimebaseInfo)]
_libc.mach_timebase_info.restype = ctypes.c_int


def _read_timebase() -> tuple[int, int]:
    info = _MachTimebaseInfo()
    _libc.mach_timebase_info(ctypes.byref(info))
    return info.numer or 1, info.denom or 1   # the fallbacks keep the conversion from dividing by zero


_NUMER, _DENOM = _read_timebase()


def mach_ticks() -> int:
    """Raw host ticks since boot, which is the unit macOS actually counts in."""
    return _libc.mach_absolute_time()


def mach_ticks_to_ns(ticks: int) -> int:
    return ticks * _NUMER // _DENOM   # required on Apple Silicon, where one tick is not one nanosecond


def now_ns() -> int:
    """The canonical timestamp for frames and events, directly comparable to any CGEvent timestamp."""
    return mach_ticks_to_ns(_libc.mach_absolute_time())


def ns_to_s(ns: int) -> float:
    return ns / 1e9


def clocks_agree(tolerance_ns: int = 5_000_000) -> bool:
    """A sanity check that our tick maths is right for this machine, since bad timestamps are silent poison."""
    a0, b0 = now_ns(), _time.monotonic_ns()
    _time.sleep(0.02)
    a1, b1 = now_ns(), _time.monotonic_ns()
    return abs((a1 - a0) - (b1 - b0)) < tolerance_ns   # both clocks are monotonic, so they must advance at one rate


if __name__ == "__main__":
    print(f"timebase numer/denom = {_NUMER}/{_DENOM}")
    t0 = now_ns()
    _time.sleep(0.1)
    print(f"100ms sleep measured as {ns_to_s(now_ns() - t0) * 1000:.3f} ms")
    print(f"clocks_agree() = {clocks_agree()}")
