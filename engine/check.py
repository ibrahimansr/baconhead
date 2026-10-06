"""Does this game need teaching? Score the model's predictions against what you actually did.

While you play (the model only watching), each 4-tick chunk of your recording is replayed to the model:
it sees the two frames, predicts the next 4 ticks, and is scored against what you did.
The same score is computed for "repeat your last input", which is hard to beat over 0.4 s. Teaching is
recommended unless the model beats both that baseline and an absolute floor.

Per-tick agreement, 0..1: 0.5·Jaccard(held keys) + 0.3·(camera turn direction matches, ±2° deadzone)
+ 0.2·(clicks match: neither clicked, or both clicked within 4 grid cells).
"""

from __future__ import annotations

import os

import numpy as np

from core.actions import CUR_X, CUR_Y, Action
from model.vlm import CHUNK, FRAME_GAP, decode, image

FLOOR = 0.6          # calibration knob: below this the model is too unlike you to trust untaught
MARGIN = 0.05        # and it must beat "repeat last input" by this much


def tick_agree(p: Action, t: Action) -> float:
    keys = 1.0 if not (p.keys | t.keys) else len(p.keys & t.keys) / len(p.keys | t.keys)
    sgn = lambda x: 0 if abs(x) < 2 else (1 if x > 0 else -1)
    look = float(sgn(p.look_x) == sgn(t.look_x))
    if p.click or t.click:
        near = abs(p.cursor[0] - t.cursor[0]) * CUR_X <= 4 and abs(p.cursor[1] - t.cursor[1]) * CUR_Y <= 4
        click = float(p.click and t.click and near)
    else:
        click = 1.0
    return 0.5 * keys + 0.3 * look + 0.2 * click


def run(ep_dir: str, generate) -> dict:
    """generate([img 0.2 s ago, img now]) -> reply text, i.e. the backend with the base model."""
    from train.data import label_real
    fi, acts = label_real(ep_dir)
    frames = np.load(os.path.join(ep_dir, "frames.npy"), mmap_mode="r")
    model, repeat = [], []
    for t in range(CHUNK, len(acts) - CHUNK + 1, CHUNK):
        truth = acts[t:t + CHUNK]
        if not any(a.keys or a.look_x or a.click for a in truth):
            continue   # idle chunks say nothing about skill
        imgs = [image(frames[fi[max(0, t - FRAME_GAP)]]), image(frames[fi[t]])]
        pred, _ = decode(generate(imgs), acts[t - 1].cursor)
        model += [tick_agree(p, x) for p, x in zip(pred, truth)]
        repeat += [tick_agree(acts[t - 1], x) for x in truth]
    if not model:
        return {"verdict": "unknown", "reason": "you barely moved while it was watching", "model": None, "repeat": None, "n": 0}
    m, r = float(np.mean(model)), float(np.mean(repeat))
    teach = m < max(FLOOR, r + MARGIN)
    return {"verdict": "teach" if teach else "ok", "model": m, "repeat": r, "n": len(model),
            "reason": (f"base model matches your play {m:.0%} vs {r:.0%} for just repeating your last input"
                       + ("; teaching recommended" if teach else "; it already plays like you, no teaching needed"))}


if __name__ == "__main__":
    a, b = Action(keys=frozenset({"w", "space"}), look_x=10.0), Action(keys=frozenset({"w"}), look_x=5.0)
    assert abs(tick_agree(a, b) - (0.5 * 0.5 + 0.3 + 0.2)) < 1e-9
    assert tick_agree(Action(), Action()) == 1.0
    assert tick_agree(Action(click=True, cursor=(0.5, 0.5)), Action()) == 0.5 + 0.3
    print("agreement ok")
