"""Find, focus, and click the Roblox window on macOS."""

from __future__ import annotations

import subprocess
import sys
from typing import Optional

Region = dict


def get_roblox_region() -> Optional[Region]:
    """The game window as {left, top, width, height}, in the top-left origin pixels mss expects."""
    if sys.platform != "darwin":
        return None
    return _quartz_region() or _applescript_region()


def _quartz_region() -> Optional[Region]:
    try:
        from Quartz import (
            CGWindowListCopyWindowInfo,
            kCGWindowListExcludeDesktopElements,
            kCGNullWindowID,
        )
    except Exception:
        return None
    try:
        wl = CGWindowListCopyWindowInfo(kCGWindowListExcludeDesktopElements, kCGNullWindowID)   # sees Roblox even though System Events usually can't
        best, best_area = None, 0
        for w in wl or []:
            owner = (w.get("kCGWindowOwnerName") or "")
            if "roblox" not in owner.lower():
                continue
            b = w.get("kCGWindowBounds")
            if not b:
                continue
            try:
                x, y = int(b.get("X", 0)), int(b.get("Y", 0))
                width, height = int(b.get("Width", 0)), int(b.get("Height", 0))
            except (TypeError, ValueError):
                continue
            if width < 100 or height < 100:
                continue   # skips the menubar item and other helper windows Roblox also owns
            area = width * height
            if area > best_area:
                best_area = area
                best = {"left": x, "top": y, "width": width, "height": height}   # already top-left origin, so no Y flip
        return best
    except Exception:
        return None


def _applescript_region() -> Optional[Region]:
    try:
        for proc in ("Roblox", "RobloxPlayer"):
            script = f'''
            tell application "System Events"
                set wins to every window of every process whose name contains "{proc}"
                if (count of wins) > 0 then
                    set w to item 1 of wins
                    set b to position of w
                    set s to size of w
                    return (item 1 of b) & "," & (item 2 of b) & "," & (item 1 of s) & "," & (item 2 of s)
                end if
            end tell'''
            out = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=2)
            if out.returncode == 0 and out.stdout.strip():
                parts = [int(x.strip()) for x in out.stdout.strip().split(",")]
                if len(parts) == 4:
                    return {"left": parts[0], "top": parts[1], "width": parts[2], "height": parts[3]}
    except Exception:
        pass
    return None


def focus_roblox() -> bool:
    """Bring Roblox to the front, since synthetic keys go to whichever app has focus."""
    if sys.platform != "darwin":
        return False
    script = '''
    tell application "System Events"
        set procs to every process whose name contains "Roblox"
        if (count of procs) > 0 then
            set frontmost of item 1 of procs to true
            return "ok"
        end if
    end tell'''
    try:
        out = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=3)
        return out.returncode == 0 and "ok" in (out.stdout or "")
    except Exception:
        return False
