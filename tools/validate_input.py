"""Prove Roblox honours our synthetic input before trusting anything else. Run it with Roblox open.

An action passes only if the recorder logged agent events AND the screen visibly changed, because
logged-but-no-change is the classic symptom of input not actually reaching the game.

    python -m tools.validate_input
"""

from __future__ import annotations

import time

import numpy as np
from PIL import Image

from core.window import get_roblox_region
from recorder.events import InputRecorder
from recorder.frames import FrameSource
from control import inject
from control.physics import degrees_to_px, degrees_to_ms, LOOK_PX_PER_MS


def _similarity(a: np.ndarray, b: np.ndarray) -> float:
    """1.0 means identical, and anything lower means the screen moved."""
    sz = (112, 112)   # downscaled first, so tiny animations and noise don't read as movement
    ga = np.asarray(Image.fromarray(a).resize(sz).convert("L"), np.float32)
    gb = np.asarray(Image.fromarray(b).resize(sz).convert("L"), np.float32)
    return max(0.0, 1.0 - np.abs(ga - gb).mean() / 60.0)


def _focus_roblox(region) -> None:
    """Front the window and click inside it, since Roblox ignores keys until it has been clicked."""
    import subprocess
    subprocess.run(
        ["osascript", "-e",
         'tell application "System Events" to set frontmost of '
         '(first process whose name contains "Roblox") to true'],
        capture_output=True,
    )
    time.sleep(0.4)
    if region:
        inject.click(region["left"] + region["width"] / 2, region["top"] + region["height"] / 2)
        time.sleep(0.3)


def main():
    region = get_roblox_region()
    if region is None:
        print("Roblox window not found. Open Roblox and try again.")
        return
    print(f"Roblox region: {region}")
    cx = region["left"] + region["width"] / 2
    cy = region["top"] + region["height"] / 2

    rec = InputRecorder()
    try:
        rec.start()
    except PermissionError as e:
        print(f"\n{e}\n")
        return
    src = FrameSource(region=region)

    print("\nSwitch to Roblox now — starting in 5s...")
    time.sleep(5)
    _focus_roblox(region)

    look_ms = degrees_to_ms(90)
    actions = [
        ("W forward (600ms)", lambda: inject.hold_key("w", 600), "key", "w"),
        ("look right 90°", lambda: inject.look(degrees_to_px(90), look_ms, center=(cx, cy)), "drag", None),
        ("look left 90°", lambda: inject.look(-degrees_to_px(90), look_ms, center=(cx, cy)), "drag", None),
        ("jump (space)", lambda: inject.tap_key("space"), "key", "space"),
    ]

    results = []
    for label, fn, ev_kind, ev_key in actions:
        rec.clear()
        before = src.grab().rgb
        fn()
        time.sleep(0.25)
        after = src.grab().rgb

        evs = rec.snapshot()
        agent_evs = [e for e in evs if e.source == "agent"]
        if ev_kind == "key":
            matched = [e for e in agent_evs if e.key == ev_key]
        else:
            matched = [e for e in agent_evs if e.kind in ("mouse_drag",) and abs(e.dx) > 0]

        sim = _similarity(before, after)
        changed = sim < 0.97
        logged = len(matched) > 0
        ok = logged and changed
        results.append((label, ok, logged, len(agent_evs), sim))
        print(f"\n[{label}]")
        print(f"  agent events logged: {len(agent_evs)} (matching: {len(matched)}) -> {'OK' if logged else 'NONE'}")
        print(f"  frame similarity before/after: {sim:.3f} -> {'CHANGED' if changed else 'NO CHANGE'}")
        print(f"  {'PASS' if ok else 'FAIL'}")
        time.sleep(0.4)

    inject.release_keys("w", "a", "s", "d", "space")
    rec.stop()
    src.close()

    print("\n" + "=" * 52)
    n_pass = sum(1 for _, ok, *_ in results if ok)
    for label, ok, logged, n, sim in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {label:22s} events={n:3d} sim={sim:.3f}")
    print("=" * 52)
    print(f"{n_pass}/{len(results)} actions verified.")
    if n_pass == len(results):
        print("I/O substrate works: Roblox honors synthetic input, recorder sees it on the shared clock.")
    elif any(logged and not ok for _, ok, logged, *_ in results):
        print("Events logged but screen didn't change — input may not be reaching Roblox "
              "(focus, or pointer-lock camera needing a virtual HID device).")
    else:
        print("No agent events logged — check Input Monitoring permission and the event tap.")


if __name__ == "__main__":
    main()
