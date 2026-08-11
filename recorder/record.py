"""Record yourself playing. Your keypresses become the action labels the world model trains on.

    python -m recorder.record --seconds 180
"""

from __future__ import annotations

import argparse
import sys
import time

from core.clock import now_ns, ns_to_s
from recorder.events import InputRecorder
from recorder.frames import FrameSource
from recorder.episode import EpisodeWriter, next_run_dir


def record(out_dir: str, seconds: float | None, fps_cap: float | None, store_size: int) -> str:
    src = FrameSource.for_roblox(wait_s=20.0)
    if src.region is None:
        print("Roblox window not found — capturing the primary monitor instead.", file=sys.stderr)
    print(f"Region: {src.region}", flush=True)

    rec = InputRecorder()
    try:
        rec.start()
    except PermissionError as e:
        print(f"\n{e}\n", file=sys.stderr)
        src.close()
        sys.exit(1)

    writer = EpisodeWriter(out_dir=out_dir, region=src.region, store_size=store_size)
    min_interval = (1.0 / fps_cap) if fps_cap else 0.0

    print(f"Recording → {out_dir}")
    print("Play normally. Ctrl+C to stop.\n", flush=True)

    n_frames = 0
    events_written = 0
    t_start = now_ns()
    last_report = time.monotonic()

    try:
        while True:
            tick = time.monotonic()

            f = src.grab()
            writer.add_frame(f.rgb, f.t_ns)
            n_frames += 1

            snap = rec.snapshot()
            if len(snap) > events_written:
                writer.add_events(snap[events_written:])   # only the events we have not written yet
                events_written = len(snap)

            if seconds is not None and ns_to_s(now_ns() - t_start) >= seconds:
                break

            now = time.monotonic()
            if now - last_report >= 2.0:
                dur = ns_to_s(now_ns() - t_start)
                print(f"  {n_frames} frames | {events_written} events | "
                      f"{n_frames / max(dur, 1e-6):.1f} FPS | {dur:.0f}s", flush=True)
                last_report = now

            if min_interval:
                sleep = min_interval - (time.monotonic() - tick)
                if sleep > 0:
                    time.sleep(sleep)   # capping FPS leaves the machine some headroom for the game itself

    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        snap = rec.snapshot()
        if len(snap) > events_written:
            writer.add_events(snap[events_written:])   # one last drain, so the final keypresses are not lost
            events_written = len(snap)
        rec.stop()
        src.close()
        path = writer.close(extra_meta={"kind": "human_demo"})

    dur = ns_to_s(now_ns() - t_start)
    print(f"\nSaved episode → {path}")
    print(f"  {n_frames} frames, {events_written} input events, {dur:.1f}s "
          f"({n_frames / max(dur, 1e-6):.1f} FPS)")
    return path


def main():
    p = argparse.ArgumentParser(description="Record a human demo as a fused episode")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--seconds", type=float, default=None)
    p.add_argument("--fps", type=float, default=None, help="cap capture rate, uncapped by default")
    p.add_argument("--store-size", type=int, default=256)
    args = p.parse_args()

    out_dir = args.out or next_run_dir("demos")
    record(out_dir, seconds=args.seconds, fps_cap=args.fps, store_size=args.store_size)


if __name__ == "__main__":
    main()
