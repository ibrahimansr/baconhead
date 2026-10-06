"""The policy is a vision-language model (Qwen2.5-VL-3B by default) that sees two frames (0.2 s ago and now)
and writes its next 4 ticks as text. Only the first EXEC=2 are played before it looks again (receding horizon),
so it reacts every 0.2 s while still being trained to plan 0.4 s ahead.

One tick: `<keys> <look_x> <look_y>[ c<col>,<row>][ z<scroll>]`, ticks joined by ` | `.
  keys  letters for held keys: w a s d, j = Space, h = Shift, e q r f 1 2 3, l = left mouse held, x = Esc; `.` = none
  look  camera degrees this tick (yaw, pitch), snapped to core.actions bins
  c     left click at cursor grid cell (64x36)
  m     move the cursor to a grid cell without clicking (hovering, aiming before a click)
  z     scroll notch (-1 zoom in, +1 out)
e.g. `wj 15 0 | w 0 0 | . 0 0 c32,18 | w -7 0` ≈ 25 tokens for 0.4 s of play.
"""

from __future__ import annotations

import os
import re

import numpy as np
from PIL import Image

from core.actions import CUR_X, CUR_Y, KEYS, Action, quantize

MODEL_ID = os.environ.get("BH_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
OBS_W, OBS_H = 448, 252      # multiples of 28, so Qwen2.5-VL sees exactly 16x9 = 144 image tokens
CHUNK = 4         # ticks predicted per call (the training target)
EXEC = 2          # ticks played before replanning
FRAME_GAP = 2     # the earlier frame is this many ticks back, so motion is visible

LETTER = {"w": "w", "a": "a", "s": "s", "d": "d", "space": "j", "shift": "h", "e": "e", "q": "q", "r": "r", "f": "f",
          "1": "1", "2": "2", "3": "3", "lmb": "l", "esc": "x"}
KEY_OF = {v: k for k, v in LETTER.items()}
SYSTEM = ("You are playing a Roblox game on a Mac. You see two screenshots: 0.2 seconds ago, then now. "
          "Output your inputs for the next 4 ticks "
          "(100 ms each), separated by ' | '. Each tick: held keys (w a s d, j=space, h=shift, e q r f 1 2 3, "
          "l=hold left mouse, x=esc, or '.'), "
          "camera yaw and pitch in degrees, then optionally c<col>,<row> to left-click or m<col>,<row> to move the "
          "cursor to that cell of a 64x36 grid, and z<n> to scroll.")
NOOP = ". 0 0"
_TICK = re.compile(r"^\s*([wasdjheqrf123lx.]+)\s+(-?\d+)\s+(-?\d+)(?:\s+([cm])(\d+),(\d+))?(?:\s+z(-?\d+))?\s*$")


def _cell(cursor) -> tuple[int, int]:
    return min(CUR_X - 1, int(cursor[0] * CUR_X)), min(CUR_Y - 1, int(cursor[1] * CUR_Y))


def encode_tick(a: Action, prev_cursor=None) -> str:
    a = quantize(a)
    keys = "".join(LETTER[k] for k in KEYS if k in a.keys) or "."
    s = f"{keys} {a.look_x:.0f} {a.look_y:.0f}"
    cx, cy = _cell(a.cursor)
    if a.click:
        s += f" c{cx},{cy}"
    elif prev_cursor is not None and (cx, cy) != _cell(prev_cursor):
        s += f" m{cx},{cy}"
    if a.scroll:
        s += f" z{a.scroll}"
    return s


def encode(actions: list[Action], prev_cursor=(0.5, 0.5)) -> str:
    out = []
    for a in actions:
        out.append(encode_tick(a, prev_cursor))
        prev_cursor = a.cursor
    return " | ".join(out)


def decode(text: str, cursor=(0.5, 0.5)) -> tuple[list[Action], bool]:
    """→ exactly CHUNK actions and whether every tick parsed. Unparseable ticks become no-ops."""
    out, ok = [], True
    for part in text.strip().split("|")[:CHUNK]:
        m = _TICK.match(part)
        if not m:
            out.append(Action(cursor=cursor))
            ok = False
            continue
        keys, lx, ly, kind, cx, cy, z = m.groups()
        click = kind == "c"
        if kind:
            cursor = ((min(int(cx), CUR_X - 1) + 0.5) / CUR_X, (min(int(cy), CUR_Y - 1) + 0.5) / CUR_Y)
        out.append(quantize(Action(keys=frozenset(KEY_OF[c] for c in keys if c in KEY_OF), look_x=float(lx), look_y=float(ly),
                                   cursor=cursor, click=click, scroll=int(z or 0))))
    ok &= len(out) == CHUNK
    out += [Action(cursor=cursor)] * (CHUNK - len(out))
    return out, ok


def processor(path: str):
    """AutoProcessor, with Idefics3/SmolVLM's image tiling off: tiling made 2 frames ~1900 tokens instead of ~130."""
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(path)
    if hasattr(proc.image_processor, "do_image_splitting"):
        proc.image_processor.do_image_splitting = False
    return proc


def image(frame) -> Image.Image:
    """Any (H,W,3) uint8 frame → the 448x252 image the model sees (box filter, like a downscaled screenshot)."""
    im = Image.fromarray(np.asarray(frame))
    return im if im.size == (OBS_W, OBS_H) else im.resize((OBS_W, OBS_H), Image.Resampling.BOX)


def prompt() -> list[dict]:
    """Two image slots: the frame FRAME_GAP ticks ago, then now. Datasets put both in `images`, in that order.
    No action history: with it the SFT model learned to repeat its last input (key accuracy never beat "copy the
    previous tick", and from an all-no-op history at spawn it stood still forever). The two frames show the motion."""
    return [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": "Your next 4 ticks:"}]}]


class VLMRunner:
    """Frames in, one Action per tick out.

    Every EXEC ticks it asks the model for a fresh 4-tick chunk from the last two frames. With background=True
    (live play) the request runs on a thread while the current chunk keeps playing: when the answer lands, the
    ticks that already went by are skipped, and if the queue runs dry first the held keys simply stay held."""

    def __init__(self, path: str = MODEL_ID, device: str = "cpu", greedy: bool = True, background: bool = False):
        import torch
        from transformers import AutoModelForImageTextToText
        dtype = torch.bfloat16 if device != "cpu" else torch.float32
        self.model = AutoModelForImageTextToText.from_pretrained(path, dtype=dtype).to(device).eval()   # a LoRA dir loads base + adapter
        base = getattr(self.model, "peft_config", None) and next(iter(self.model.peft_config.values())).base_model_name_or_path
        if base:   # an unmerged r=64 LoRA doubled step time on a 16 GB Mac; merged it costs nothing per call
            self.model = self.model.merge_and_unload()
        self.proc = processor(base or path)
        self.device, self.greedy, self.background = device, greedy, background
        self.reset()

    def reset(self):
        from collections import deque
        self.queue: list[Action] = []
        self.frames: deque = deque(maxlen=FRAME_GAP + 1)
        self.done: list[Action] = []          # the last CHUNK actions actually played
        self.cursor = (0.5, 0.5)
        self.parse_fail, self.calls = 0, 0
        self.pending, self.sent_at, self.tick = None, 0, 0

    def generate(self, imgs: list) -> str:
        return self.generate_batch([imgs])[0]

    def generate_batch(self, batch: list[list]) -> list[str]:
        """One forward pass for many envs' frame pairs (DAgger rollouts step 32 envs in lockstep)."""
        import torch
        text = self.proc.apply_chat_template(prompt(), add_generation_prompt=True, tokenize=False)
        self.proc.tokenizer.padding_side = "left"
        inp = self.proc(text=[text] * len(batch), images=batch, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=64, do_sample=not self.greedy)
        return self.proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)

    def _request(self):
        return [image(self.frames[0]), image(self.frames[-1])], self.cursor

    def _accept(self, text: str, cursor, skip: int):
        q, ok = decode(text, cursor)
        self.parse_fail += not ok
        self.queue = q[min(skip, CHUNK - 1):]

    def act(self, frame) -> Action:
        self.frames.append(frame)
        self.tick += 1
        if self.background:
            if self.pending is not None and self.pending.done():
                self._accept(self.pending.result(), self._pending_cursor, self.tick - self.sent_at)
                self.pending = None
            if self.pending is None and len(self.queue) <= CHUNK - EXEC:
                from concurrent.futures import ThreadPoolExecutor
                self._pool = getattr(self, "_pool", None) or ThreadPoolExecutor(1)
                imgs, self._pending_cursor = self._request()
                self.pending, self.sent_at = self._pool.submit(self.generate, imgs), self.tick
        elif len(self.queue) <= CHUNK - EXEC:
            imgs, cur = self._request()
            self._accept(self.generate(imgs), cur, 0)
        if self.queue:
            a = self.queue.pop(0)
        else:   # still waiting on the model: keep holding what was held, no new camera turn or click
            last = self.done[-1] if self.done else Action(cursor=self.cursor)
            a = Action(keys=last.keys, cursor=last.cursor)
        self.cursor = a.cursor
        self.done = (self.done + [a])[-CHUNK:]
        return a


if __name__ == "__main__":
    acts = [Action(keys=frozenset({"w", "space"}), look_x=14.0), Action(keys=frozenset({"w"}), cursor=(0.2, 0.3)),
            Action(cursor=(0.51, 0.52), click=True), Action(keys=frozenset({"d", "1"}), look_x=-6.0, look_y=3.0, cursor=(0.51, 0.52), scroll=1)]
    t = encode(acts)
    back, ok = decode(t)
    assert ok and encode(back) == t, (t, encode(back))
    assert decode("garbage")[1] is False and len(decode("garbage")[0]) == CHUNK

    # replanning: a stub model that numbers its chunks; every EXEC ticks a new chunk replaces the queue
    class Stub(VLMRunner):
        def __init__(self, background=False):
            self.greedy, self.background, self.n = True, background, 0
            self.reset()

        def generate(self, imgs):
            import time
            self.n += 1
            assert len(imgs) == 2
            if self.background:
                time.sleep(0.05)
            return " | ".join([f"w {self.n % 30} 0"] * CHUNK)
    r = Stub()
    looks = [r.act(np.zeros((4, 4, 3), np.uint8)).look_x for _ in range(6)]
    assert r.n == 3 and looks == [1.0, 1.0, 1.0, 1.0, 3.0, 3.0], (r.n, looks)   # chunk n yaws n°, snapped to bins (2 → 1)
    import time
    rb = Stub(background=True)
    out = []
    for _ in range(12):
        out.append(rb.act(np.zeros((4, 4, 3), np.uint8)))
        time.sleep(0.02)
    first = next(i for i, a in enumerate(out) if "w" in a.keys)
    assert rb.n >= 2 and all("w" in a.keys for a in out[first:]), rb.n   # once something's held, it stays held while waiting
    print("codec ok:", t, "| replans every", EXEC, "ticks; background runner keeps keys held while waiting")
