#!/usr/bin/env python3
"""Play the game: the world model runs every tick locally, Claude only when the world model is out of its depth.

    python run.py --model wm.pt          takes over once you go idle
    python run.py --model wm.pt --now    start immediately
    python run.py --now                  no checkpoint yet, Claude picks moves directly
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque

import numpy as np
import torch
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

from control import inject
from core.idle import HumanIdle
from core.window import focus_roblox_and_click
from recorder.episode import EpisodeWriter, next_run_dir
from recorder.events import InputRecorder
from recorder.frames import FrameSource
from wm import actions as A
from wm.data import append_label
from wm.model import WorldModel, goal_vec, preprocess
from wm.plan import DEFAULT_W, LatentBank, plan, surprise

DEAD_P_ALIVE = 0.25   # below this the death head thinks we are gone
DEAD_TICKS = 3        # required in a row, because an undertrained head reads near zero on any unfamiliar frame
SURPRISE_HI = 0.6     # latent prediction error past which imagination is worthless and Claude gets woken


def main() -> None:
    p = argparse.ArgumentParser(description="Play with a latent world model + Claude")
    p.add_argument("--model", default="wm.pt")
    p.add_argument("--idle", type=float, default=3.0, help="take over after N seconds of no human input")
    p.add_argument("--now", action="store_true")
    p.add_argument("--horizon", type=int, default=8, help="how many ticks ahead to imagine")
    p.add_argument("--candidates", type=int, default=32, help="how many futures to imagine per replan")
    p.add_argument("--replan", type=int, default=4, help="ticks executed before replanning")
    p.add_argument("--label-every", type=int, default=25, help="ticks between Claude progress scores")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    src = FrameSource.for_roblox(wait_s=20.0)
    if src.region is None:
        print("Roblox window not found — capturing the primary monitor.", file=sys.stderr)
    center = ((src.region["left"] + src.region["width"] / 2,
               src.region["top"] + src.region["height"] / 2) if src.region else (960.0, 540.0))

    model = None
    if os.path.isfile(args.model):
        model = WorldModel(A.N_ACTIONS)
        model.load_state_dict(torch.load(args.model, map_location="cpu"))
        model.eval()
        print(f"world model: {args.model}")
    else:
        print(f"no checkpoint at {args.model} — Claude-only bootstrap "
              f"(record demos, then: python -m wm.train)")

    brain = None
    try:
        from agent.brain import Brain

        brain = Brain()
    except Exception as e:
        print(f"no brain ({e}) — world model will run on default weights", file=sys.stderr)
    if brain is None and model is None:
        raise SystemExit("Need either a world model checkpoint or ANTHROPIC_API_KEY.")

    idle = HumanIdle()
    rec = InputRecorder(on_event=idle.observe)   # the same tap that labels demos also tells us when you come back
    try:
        rec.start()
    except PermissionError as e:
        print(f"\n{e}\n", file=sys.stderr)

    out_dir = args.out or next_run_dir("demos", prefix="agent")
    writer = EpisodeWriter(out_dir=out_dir, region=src.region)   # agent runs are recorded exactly like your demos
    bank = LatentBank()

    goal, weights, suggested = "", dict(DEFAULT_W), []
    queue: deque = deque()
    memory: deque = deque(maxlen=6)
    recent_frames: deque = deque(maxlen=8)
    recent_actions: deque = deque(maxlen=8)
    z_pred = None
    tick = 0
    dead_streak = 0
    events_written = 0
    active = args.now

    print(f"→ {out_dir}   Ctrl+C to stop.\n", flush=True)

    try:
        while True:
            f = src.grab()
            writer.add_frame(f.rgb, f.t_ns)
            snap = rec.snapshot()
            if len(snap) > events_written:
                writer.add_events(snap[events_written:])   # flushed as we go, so Ctrl+C still leaves a usable episode
                events_written = len(snap)

            if not args.now and idle.seconds() < args.idle:
                if active:
                    print("[user active] pausing")
                    active, queue, z_pred = False, deque(), None   # the plan is abandoned, since you have moved us
                time.sleep(0.2)
                continue
            if not active:
                print("[idle] taking over")
                focus_roblox_and_click()   # most games ignore synthetic keys until something clicks inside the window
                active = True
                time.sleep(0.3)
                continue

            x = preprocess(f.rgb)
            if model is None:
                if not queue:
                    out = brain.set_goal(f.rgb, "\n".join(memory))
                    goal, queue = out["goal"], deque(out["suggested"] or [A.ID["w"]])
                    print(f"[goal] {goal} — {out['reason']}")
                a = queue.popleft()
            else:
                with torch.no_grad():
                    z = model.encode(x)
                    bank.add(z, np.asarray(Image.fromarray(f.rgb).resize((96, 96))))   # every real frame becomes preview material
                    s = surprise(model, z_pred, z) if z_pred is not None else 0.0
                    _, log_alive, _ = model.heads(z, goal_vec(goal))
                p_alive = float(log_alive.exp())

                dead_streak = dead_streak + 1 if p_alive < DEAD_P_ALIVE else 0
                if dead_streak >= DEAD_TICKS:
                    writer.add_tag(f.t_ns, "death", p_alive=p_alive)   # this tag is what trains the death head next time
                    if brain:
                        lesson = brain.explain_death(list(recent_frames), list(recent_actions), goal)
                        memory.append(f"DIED: {lesson}")
                        print(f"[death] {lesson}")
                    goal, queue, z_pred, dead_streak = "", deque(), None, 0
                    time.sleep(3.0)   # respawn
                    continue

                if not queue or s > SURPRISE_HI:
                    new_goal = False
                    if brain and (not goal or s > SURPRISE_HI):
                        out = brain.set_goal(f.rgb, "\n".join(memory))
                        goal = out["goal"]
                        weights = {**DEFAULT_W, **out["weights"]}
                        suggested = out["suggested"]
                        new_goal = True
                        print(f"[goal] {goal} (surprise={s:.2f}) — {out['reason']}")
                    plans = plan(model, z, goal, horizon=args.horizon, k=args.candidates,
                                 weights=weights, suggested=suggested)
                    pick = 0
                    if brain and new_goal and len(bank.thumbs) > 50:   # only at a plan boundary, or the API would stall every tick
                        top = plans[:3]
                        pick = brain.choose_plan(f.rgb, [bank.strip(p_.latents) for p_ in top], top, goal)
                    chosen = plans[pick]
                    queue = deque(chosen.actions[: args.replan])   # only the first few actions run before we look again
                    print(f"[plan] {[A.NAMES[a] for a in chosen.actions]} "
                          f"P(alive)={chosen.p_alive:.2f} progress={chosen.value:.2f}")

                a = queue.popleft()
                with torch.no_grad():
                    z_pred = model.step(z, torch.tensor([a]))   # remembered so the next tick can measure how wrong we were

            A.execute(a, center=center)
            recent_frames.append(f.rgb)
            recent_actions.append(a)
            tick += 1

            if brain and goal and tick % args.label_every == 0:
                score = brain.score_progress(f.rgb, goal)
                if score is not None:
                    append_label(out_dir, f.t_ns, goal, score)
                    memory.append(f"progress {score:.0f}/10 on '{goal}'")
                    print(f"[label] {score:.0f}/10 — {goal}")
                    if score >= 9.0:
                        goal, queue = "", deque()   # goal met, so retire it and ask for the next one

    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        inject.release_keys("w", "a", "s", "d", "space")   # never leave the character sprinting into a wall
        snap = rec.snapshot()
        if len(snap) > events_written:
            writer.add_events(snap[events_written:])
        rec.stop()
        src.close()
        path = writer.close(extra_meta={"kind": "agent_run", "goal": goal,
                                        "brain_calls": brain.calls if brain else 0})
        print(f"episode → {path}\n  retrain with: python -m wm.train --demos demos")


if __name__ == "__main__":
    main()
