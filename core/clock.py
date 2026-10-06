"""One monotonic timebase for everything, so a frame and the keypresses around it line up exactly."""

from __future__ import annotations

import ctypes

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


def mach_ticks_to_ns(ticks: int) -> int:
    return ticks * _NUMER // _DENOM   # required on Apple Silicon, where one tick is not one nanosecond


def now_ns() -> int:
    """The canonical timestamp for frames and events, directly comparable to any CGEvent timestamp."""
    return mach_ticks_to_ns(_libc.mach_absolute_time())
