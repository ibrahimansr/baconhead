"""Recorded play → 10 Hz (frame, action) ticks in the shared action space.

Two kinds of recording:
  real Roblox  recorder/ episodes (frames.npy, frame_pts.npy, events.jsonl, meta.json). Each tick's label
               is what your hands did in (t, t+100ms]; its frame is the last one captured at or before t.
  sim play     play.py --record .npz files (seed + per-substep inputs), replayed exactly and rendered here.
"""

from __future__ import annotations

import json

import numpy as np

from core.actions import KEYS, Action, TICK_S
from model.vlm import OBS_H, OBS_W
from sim.core import PX_PER_DEG, SUBSTEPS

TICK_NS = int(TICK_S * 1e9)


def label_real(ep_dir: str):
    """→ frame index per tick (K,), list of Action (K,)"""
    from recorder.episode import Episode
    ep = Episode.load(ep_dir)
    pts = ep.frame_pts
    if len(pts) < 2:
        return np.zeros(0, int), []
    reg = ep.meta.get("region") or {"left": 0, "top": 0, "width": 1920, "height": 1080}
    evs = sorted(ep.events, key=lambda e: e["t_ns"])
    K = int((pts[-1] - pts[0]) // TICK_NS)
    held, right, left, shiftlock = set(), False, False, False
    cur = (0.5, 0.5)
    j = 0
    fidx, acts = [], []
    for k in range(K):
        t0, t1 = pts[0] + k * TICK_NS, pts[0] + (k + 1) * TICK_NS
        pressed, lx, ly, click, scroll = set(), 0.0, 0.0, False, 0
        held_start = set(held)   # a key counts for the tick if it was down at the start or pressed during it
        left_start = left
        while j < len(evs) and evs[j]["t_ns"] <= t1:
            e = evs[j]
            j += 1
            if e["source"] == "agent":   # only your hands are labels
                continue
            kind = e["kind"]
            key = "esc" if e.get("key") == "escape" else e.get("key")
            if kind == "key_down" and key in KEYS + ("up", "down"):
                held.add(key)
                pressed.add(key)
            elif kind == "key_up":
                held.discard(key)
            elif kind == "flags" and e["key"] in ("shift", "rshift"):
                if "shift" in held:
                    held.discard("shift")
                else:
                    held.add("shift")
                    pressed.add("shift")
                    shiftlock = not shiftlock   # Roblox toggles shift lock on each press
            elif kind == "mouse_down" and e["button"] == "right":
                right = True
            elif kind == "mouse_up" and e["button"] == "right":
                right = False
            elif kind == "mouse_down" and e["button"] == "left":
                click, left = True, True
            elif kind == "mouse_up" and e["button"] == "left":
                left = False
            elif kind in ("mouse_drag", "mouse_move"):
                if right or shiftlock:
                    lx += e["dx"] / PX_PER_DEG
                    ly += e["dy"] / PX_PER_DEG
                if not shiftlock:
                    cur = ((e["x"] - reg["left"]) / reg["width"], (e["y"] - reg["top"]) / reg["height"])
            elif kind == "scroll":
                scroll += -1 if e["dy"] > 0 else 1   # wheel up zooms in
        keys = {"w" if k_ == "up" else "s" if k_ == "down" else k_ for k_ in held_start | held | pressed}
        if left_start and left:
            keys.add("lmb")   # held through the tick, not just clicked in it
        acts.append(Action(keys=frozenset(keys), look_x=lx, look_y=ly, cursor=(float(np.clip(cur[0], 0, 1)), float(np.clip(cur[1], 0, 1))),
                           click=click, scroll=int(np.sign(scroll))))
        fidx.append(int(np.searchsorted(pts, t0, side="right") - 1))
    return np.array(fidx), acts


def replay_sim(path: str, device="cpu"):
    """play.py recording → rendered frames (K,252,448,3) uint8 and Actions."""
    import sim
    from sim.core import Input
    from sim.render import render
    z = np.load(path, allow_pickle=True)
    kw = eval(str(z["kw"]))    # written by play.py as repr(dict) of plain numbers
    env = sim.make(str(z["game"]), seed=int(z["seed"]), **kw)
    inp = z["inputs"]
    nk = len(KEYS)
    frames, acts = [], []
    for k in range(len(inp) // SUBSTEPS):
        frames.append(render([env.scene()], OBS_H, OBS_W, device=device)[0].cpu().numpy())
        chunk = inp[k * SUBSTEPS:(k + 1) * SUBSTEPS]
        keys = frozenset(KEYS[i] for i in range(nk) if chunk[:, i].max() > 0.5)
        acts.append(Action(keys=keys, look_x=float(chunk[:, nk].sum()), look_y=float(chunk[:, nk + 1].sum()),
                           cursor=(float(chunk[-1, nk + 2]), float(chunk[-1, nk + 3])), click=bool(chunk[:, nk + 4].max() > 0.5),
                           scroll=int(np.sign(chunk[:, nk + 5].sum()))))
        for row in chunk:
            env.substep(Input(keys=frozenset(KEYS[i] for i in range(nk) if row[i] > 0.5), dyaw=float(row[nk]), dpitch=float(row[nk + 1]),
                              cursor=(float(row[nk + 2]), float(row[nk + 3])), click=bool(row[nk + 4] > 0.5), scroll=int(row[nk + 5])))
    return frames, acts


if __name__ == "__main__":
    # label a synthetic recording: hold W for 0.25s, right-drag 60px, click once, scroll up
    import tempfile
    from recorder.episode import EpisodeWriter
    d = tempfile.mkdtemp()
    wr = EpisodeWriter(out_dir=d, region={"left": 0, "top": 0, "width": 1000, "height": 500})
    t0 = 1_000_000_000
    for i in range(12):
        wr.add_frame(np.zeros((144, 256, 3), np.uint8), t0 + i * 50_000_000)
    E = lambda t, kind, **kw: wr._events_fp.write(json.dumps({"t_ns": t0 + t, "kind": kind, "source": "human", "key": None, "button": None,
                                                           "x": 0.0, "y": 0.0, "dx": 0, "dy": 0, **kw}) + "\n")
    E(10_000_000, "key_down", key="w")
    E(260_000_000, "key_up", key="w")
    E(120_000_000, "mouse_down", button="right")
    E(150_000_000, "mouse_drag", dx=60, x=500.0, y=250.0)
    E(190_000_000, "mouse_up", button="right")
    E(330_000_000, "mouse_move", x=250.0, y=125.0)
    E(340_000_000, "mouse_down", button="left", x=250.0, y=125.0)
    E(420_000_000, "scroll", dy=3)
    wr.close()
    fi, acts = label_real(d)
    assert "w" in acts[0].keys and "w" in acts[2].keys and "w" not in acts[3].keys, [a.keys for a in acts]
    assert acts[1].look_x == 30.0 and acts[0].look_x == 0.0
    assert acts[3].click and abs(acts[3].cursor[0] - 0.25) < 1e-6 and abs(acts[3].cursor[1] - 0.25) < 1e-6
    assert acts[4].scroll == -1
    print("labeler ok:", len(acts), "ticks", [sorted(a.keys) for a in acts])
