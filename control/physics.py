"""Motion constants measured against live Roblox. Re-run calibration before changing any number here."""

from __future__ import annotations

LOOK_PX_PER_DEGREE: float = 180 / 90   # 200px per 90° overshot by ~10° each turn, which compounded to 400° over four turns
LOOK_PX_PER_MS: float = 0.55           # drag speed, about 220px in 400ms

WALK_MS_PER_STUD: float = 1000 / 16    # Roblox default walk speed is 16 studs per second
MAX_MOVEMENT_MS: int = 3000            # a ceiling on any single key hold, so one bad plan can't run for a minute


def degrees_to_px(degrees: float) -> int:
    """Camera rotation in degrees to the Quartz drag distance that produces it."""
    return max(10, int(abs(degrees) * LOOK_PX_PER_DEGREE))


def degrees_to_ms(degrees: float) -> int:
    px = degrees_to_px(degrees)
    return max(100, min(2000, int(px / LOOK_PX_PER_MS)))   # too fast and Roblox drops drag events, too slow and it drifts


def studs_to_ms(studs: float) -> int:
    """Forward distance in studs to how long W must be held: 3 studs enters a door, 16 crosses a courtyard."""
    return max(100, min(MAX_MOVEMENT_MS, int(studs * WALK_MS_PER_STUD)))
