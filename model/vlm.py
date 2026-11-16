"""The policy is a vision-language model (Qwen3.5-4B by default) that sees two frames (0.2 s ago and now), the
game's objective and its own previous plan, and writes a PLAN line then its next 4 ticks as text. Only the first
EXEC=2 ticks are played before it looks again (receding horizon), so it reacts every 0.2 s while trained to plan
0.4 s ahead.

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

MODEL_ID = os.environ.get("BH_MODEL", "Qwen/Qwen3.5-4B")   # Script2Play's base (arXiv 2611.11136); was Qwen2.5-VL-3B
OBS_W, OBS_H = 512, 288      # multiples of 32 (16 px patches, 2x2 merge): exactly 16x9 = 144 image tokens per frame
CHUNK = 4         # ticks predicted per call (the training target)
EXEC = 2          # ticks played before replanning
FRAME_GAP = 2     # the earlier frame is this many ticks back, so motion is visible

LETTER = {"w": "w", "a": "a", "s": "s", "d": "d", "space": "j", "shift": "h", "e": "e", "q": "q", "r": "r", "f": "f",
          "1": "1", "2": "2", "3": "3", "lmb": "l", "esc": "x"}
KEY_OF = {v: k for k, v in LETTER.items()}
SYSTEM = ("You are playing a Roblox game on a Mac. You see two screenshots: 0.2 seconds ago, then now. "
          "First write 'PLAN: ' and one short line saying what you see and what you are doing next, then on a new line "
          "'ACT: ' and your inputs for the next 4 ticks "
          "(100 ms each), separated by ' | '. Each tick: held keys (w a s d, j=space, h=shift, e q r f 1 2 3, "
          "l=hold left mouse, x=esc, or '.'), "
          "camera yaw and pitch in degrees, then optionally c<col>,<row> to left-click or m<col>,<row> to move the "
          "cursor to that cell of a 64x36 grid, and z<n> to scroll.")
NOOP = ". 0 0"
OBJECTIVE = {   # the instruction g (Script2Play, arXiv 2611.11136): what the game is about, given every decision
    "obby": "Obby: get through the obstacle course checkpoint by checkpoint without falling.",
    "doors": "DOORS: go through the numbered doors one by one and survive the monsters.",
    "rivals": "RIVALS: win 1v1 rounds by finding and shooting the enemy.",
    "mm2": "Murder Mystery 2: survive the round; as Sheriff shoot the Murderer, as Murderer eliminate everyone.",
    "disaster": "Natural Disaster Survival: survive each disaster until it ends.",
    "forest": "99 Nights in the Forest: gather logs, keep the campfire burning, eat, and survive the nights.",
    "fisch": "Fisch: cast, reel in fish, and sell them for money.",
    "garden": "Grow a Garden: buy seeds, plant them, harvest the crops, and sell them for profit.",
    "adoptme": "Adopt Me!: take care of your pet's needs shown in its bubbles.",
    "brainrot": "Steal a Brainrot: buy brainrots from the carpet and collect the cash they earn in your base.",
}
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


def split_plan(text: str) -> tuple[str, str]:
    """'PLAN: ...\nACT: ticks' → (plan, ticks). Text without a PLAN line is all ticks (old models, recorded demos)."""
    if "ACT:" not in text:
        return "", text
    head, acts = text.split("ACT:", 1)
    return head.replace("PLAN:", "").strip(), acts.split("\n")[0].strip()


def target(plan: str, ticks: str) -> str:
    return f"PLAN: {plan}\nACT: {ticks}" if plan else ticks


def decode(text: str, cursor=(0.5, 0.5)) -> tuple[list[Action], bool]:
    """→ exactly CHUNK actions and whether every tick parsed. Unparseable ticks become no-ops. A PLAN line is skipped."""
    out, ok = [], True
    text = split_plan(text)[1]
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


NO_THINK = "enable_thinking is defined and enable_thinking is false"


def processor(path: str):
    """AutoProcessor, with Idefics3/SmolVLM's image tiling off (tiling made 2 frames ~1900 tokens instead of ~130), and
    Qwen3.5's thinking forced off in the chat template itself: by default it opens `<think>` and reasons before
    answering, which at 10 Hz is all latency. Patching the template (saved with every checkpoint) covers every
    caller at once: TRL's SFT formatting, VLMRunner and train.rl."""
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(path)
    if hasattr(proc.image_processor, "do_image_splitting"):
        proc.image_processor.do_image_splitting = False
    for obj in (proc, getattr(proc, "tokenizer", None)):
        tpl = getattr(obj, "chat_template", None)
        if isinstance(tpl, str) and NO_THINK in tpl:
            obj.chat_template = tpl.replace(NO_THINK, "true")
    return proc


def image(frame) -> Image.Image:
    """Any (H,W,3) uint8 frame → the 448x252 image the model sees (box filter, like a downscaled screenshot)."""
    im = Image.fromarray(np.asarray(frame))
    return im if im.size == (OBS_W, OBS_H) else im.resize((OBS_W, OBS_H), Image.Resampling.BOX)


def prompt(game: str | None = None, prev_plan: str = "") -> list[dict]:
    """Two image slots (the frame FRAME_GAP ticks ago, then now; datasets put both in `images`), the game's objective,
    and the policy's own previous PLAN line as memory. No action history: with it the SFT model learned to repeat its
    last input and stood still forever. The plan carries the phase (hiding, reeling, heading to the shop), which
    two frames can't show; v3 without it froze in 4/10 games and hid in a closet for 2 minutes."""
    text = (f"Game: {OBJECTIVE.get(game, 'a Roblox game; play it well.')}\n"
            f"Your previous plan: {prev_plan or 'none yet'}\nYour plan and next 4 ticks:")
    return [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": text}]}]


class VLMRunner:
    """A trained model as a policy: frame pairs in, PLAN + 4-tick chunks out, many envs per forward pass.
    The env loops (train.eval.rollouts/play, train.dagger) feed it two frames and each env's (game, previous plan)
    every EXEC ticks and play the first EXEC ticks of each answer."""

    def __init__(self, path: str = MODEL_ID, device: str = "cpu", greedy: bool = True):
        import json
        import torch
        from transformers import AutoModelForImageTextToText
        dtype = torch.bfloat16 if device != "cpu" else torch.float32
        cfg = os.path.join(path, "adapter_config.json")
        base = json.load(open(cfg))["base_model_name_or_path"] if os.path.exists(cfg) else None
        if base:   # a LoRA dir: base + adapter, merged (an unmerged r=64 LoRA doubled step time on a 16 GB Mac)
            from peft import PeftModel
            m = AutoModelForImageTextToText.from_pretrained(base, dtype=dtype)
            self.model = PeftModel.from_pretrained(m, path).merge_and_unload().to(device).eval()
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(path, dtype=dtype).to(device).eval()
        self.proc = processor(base or path)
        self.device, self.greedy = device, greedy
        self.parse_fail, self.calls = 0, 0

    def generate_batch(self, batch: list[list], ctx: list | None = None) -> list[str]:
        """One forward pass for many envs' frame pairs (DAgger rollouts step 40 envs in lockstep).
        ctx: per item (game, previous plan) for the prompt; None = unknown game, no plan yet."""
        import torch
        ctx = ctx or [(None, "")] * len(batch)
        texts = [self.proc.apply_chat_template(prompt(g, pl), add_generation_prompt=True, tokenize=False) for g, pl in ctx]
        self.proc.tokenizer.padding_side = "left"
        inp = self.proc(text=texts, images=batch, return_tensors="pt", padding=True).to(self.device)
        stop = self.proc.tokenizer.convert_tokens_to_ids("<|im_end|>")   # v3 ran on past 4 ticks without it
        with torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=96, do_sample=not self.greedy,
                                      eos_token_id=[stop, self.proc.tokenizer.eos_token_id])
        return self.proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)


if __name__ == "__main__":
    acts = [Action(keys=frozenset({"w", "space"}), look_x=14.0), Action(keys=frozenset({"w"}), cursor=(0.2, 0.3)),
            Action(cursor=(0.51, 0.52), click=True), Action(keys=frozenset({"d", "1"}), look_x=-6.0, look_y=3.0, cursor=(0.51, 0.52), scroll=1)]
    t = encode(acts)
    back, ok = decode(t)
    assert ok and encode(back) == t, (t, encode(back))
    tp = target("door 4 is locked, search the dresser", t)
    assert split_plan(tp) == ("door 4 is locked, search the dresser", t) and encode(decode(tp)[0]) == t, tp
    assert decode("garbage")[1] is False and len(decode("garbage")[0]) == CHUNK
    print("codec ok:", tp.replace("\n", " / "))
