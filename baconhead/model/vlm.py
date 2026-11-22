import os
import re

from baconhead.core.actions import CUR_X, CUR_Y, KEYS, Action, quantize

MODEL_ID = os.environ.get("BH_MODEL", "Qwen/Qwen3.5-4B")
OBS_W, OBS_H = 512, 288   # video render size
CHUNK = 4

LETTER = {"w": "w", "a": "a", "s": "s", "d": "d", "space": "j", "shift": "h", "e": "e", "q": "q", "r": "r", "f": "f",
          "1": "1", "2": "2", "3": "3", "lmb": "l", "esc": "x"}
KEY_OF = {v: k for k, v in LETTER.items()}
_TICK = re.compile(r"^\s*([wasdjheqrf123lx.]+)\s+(-?\d+)\s+(-?\d+)(?:\s+([cm])(\d+),(\d+))?(?:\s+z(-?\d+))?\s*$")


def _cell(cursor):
    return min(CUR_X - 1, int(cursor[0] * CUR_X)), min(CUR_Y - 1, int(cursor[1] * CUR_Y))


def encode_tick(a, prev_cursor=None):
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


def encode(actions, prev_cursor=(0.5, 0.5)):
    out = []
    for a in actions:
        out.append(encode_tick(a, prev_cursor))
        prev_cursor = a.cursor
    return " | ".join(out)


# 'PLAN: ... ACT: ticks' → (plan, ticks). Text without a PLAN line is all ticks (old models, recorded demos)
def split_plan(text):
    if "ACT:" not in text:
        return "", text
    head, acts = text.split("ACT:", 1)
    return head.replace("PLAN:", "").strip(), acts.split("\n")[0].strip()


# → exactly CHUNK actions and whether every tick parsed. Unparseable ticks become no-ops. A PLAN line is skipped
def decode(text, cursor=(0.5, 0.5)):
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


# autoProcessor with the chat template's thinking forced off (saved with every checkpoint)
def processor(path):
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(path)
    for obj in (proc, getattr(proc, "tokenizer", None)):
        tpl = getattr(obj, "chat_template", None)
        if isinstance(tpl, str) and NO_THINK in tpl:
            obj.chat_template = tpl.replace(NO_THINK, "true")
    return proc


if __name__ == "__main__":
    acts = [Action(keys=frozenset({"w", "space"}), look_x=14.0), Action(keys=frozenset({"w"}), cursor=(0.2, 0.3)),
            Action(cursor=(0.51, 0.52), click=True), Action(keys=frozenset({"d", "1"}), look_x=-6.0, look_y=3.0, cursor=(0.51, 0.52), scroll=1)]
    t = encode(acts)
    back, ok = decode(t)
    assert ok and encode(back) == t, (t, encode(back))
    assert decode("garbage")[1] is False and len(decode("garbage")[0]) == CHUNK
    print("codec ok:", t)
