import pickle

import numpy as np

from baconhead import agent as A
from baconhead.agent import chunk_text

KEYS = {"w": "W", "a": "A", "s": "S", "d": "D", "space": "Space", "shift": "Shift", "e": "E", "q": "Q", "r": "R",
        "f": "F", "1": "one", "2": "two", "3": "three", "lmb": "LB"}


# the expert's next chunk from a copy of this state: 3 of its 100 ms ticks, each held for two 50 ms steps
def expert_chunk(env):
    e = pickle.loads(pickle.dumps(env))
    acts = []
    for _ in range(A.TICKS):
        acts.append(e.expert())
        e.step(acts[-1])
        if e.done:
            break
    acts += [acts[-1]] * (A.TICKS - len(acts))
    ticks = [[KEYS[k] for k in sorted(a.keys) if k in KEYS] + (["LB"] if a.click and "lmb" not in a.keys else []) for a in acts]
    dx = round(sum(a.look_x for a in acts) / A.DEG_PER_UNIT)
    dy = round(sum(a.look_y for a in acts) / A.DEG_PER_UNIT)
    dz = int(np.sign(sum(a.scroll for a in acts)))
    return int(dx), int(dy), dz, [t for t in ticks for _ in range(A.STEPS // A.TICKS)]


# what the expert executes on noisy steps: camera off by ~10 degrees, one key flipped
def perturb(c, rng):
    dx, dy, dz, groups = c
    flip = str(rng.choice(["W", "A", "S", "D", "Space"]))
    return dx + int(rng.normal(0, 100)), dy, dz, [sorted(set(g) ^ {flip}) for g in groups]


# rendered at the model's frame size and passed through JPEG q88, as at inference
def frame(env, device="cpu"):
    import io
    from PIL import Image
    from baconhead.sim.render import render
    f = render([env.scene()], A.H, A.W, device=device)[0].cpu().numpy()  # (H, W, 3)
    buf = io.BytesIO()
    Image.fromarray(f).save(buf, format="JPEG", quality=88)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


# one SFT row: previous + current frame in, the chunk out
def row(game, prev, img, label):
    return {"prompt": A.prompt(game, [{"type": "image"}, {"type": "image"}]), "images": [prev, img],
            "completion": [{"role": "assistant", "content": chunk_text(label)}]}


# the k-th identical chunk in a row (camera ignored) is kept w.p. decay^(k-1)
def keep(texts, rng, decay=0.5):
    sig = [t.split("|", 1)[-1] for t in texts]
    out, k = [], 0
    for i, s in enumerate(sig):
        k = k + 1 if i and s == sig[i - 1] else 1
        out.append(bool(rng.random() < decay ** (k - 1)))
    return out


# the expert drives with 20% perturbed chunks; labels are always the expert's
def expert_rows(game, seed, decisions, noise=0.2):
    from baconhead import sim
    from baconhead.train.dataset import _kw
    rng = np.random.default_rng(seed)
    env = sim.make(game, seed=seed, **_kw(game, rng))
    rows, texts, prev = [], [], None
    for _ in range(decisions):
        label = expert_chunk(env)
        img = frame(env)
        rows.append(row(game, prev or img, img, label))
        texts.append(chunk_text(label))
        prev = img
        A.play_chunk(env, *(perturb(label, rng) if rng.random() < noise else label))
        if env.done:
            break
    return [r for r, k in zip(rows, keep(texts, rng)) if k]
