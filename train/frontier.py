"""Can a frontier model play these games? A hosted model plays each gold level (sim.gold: same seed, settings, goal)
with everything our 3B policy lacks: it's told the game and its goal, it thinks before acting, and it sees its own
previous plan and inputs. Every decision (thinking summary, plan, inputs) is saved; videos render locally after.

    python -m train.frontier --ticks 600                       # all 10 games, 60 s each
    python -m train.frontier --games obby,doors --ticks 300
    python -m train.eval --video results/frontier/<model>      # mp4s from the saved inputs

Keys come from .env: ANTHROPIC_API_KEY first, then OPENAI_API_KEY; the first that answers a ping is used.
The sim waits for the model (lockstep), so latency costs no game time. One decision plays all CHUNK ticks.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import sim
from core.actions import Action
from model.vlm import CHUNK, FRAME_GAP, SYSTEM, decode, encode_tick
from sim.gold import PLANS, SEED

TITLES = {"obby": "an obby (obstacle course)", "doors": "DOORS", "rivals": "RIVALS (1v1 shooter)", "mm2": "Murder Mystery 2",
          "disaster": "Natural Disaster Survival", "forest": "99 Nights in the Forest", "fisch": "Fisch (fishing)",
          "garden": "Grow a Garden", "adoptme": "Adopt Me!", "brainrot": "Steal a Brainrot"}
W, H = 672, 378   # what the frontier model sees: 1.5x our policy's 448x252, ~340 image tokens each on Claude
GUIDE = SYSTEM + """

Facts: you walk 16 studs/s (1.6 studs per tick); a jump (j) rises 7.2 studs and clears gaps up to ~8 studs at a run.
Camera yaw is degrees per tick, positive turns right; 30 is a big turn, a full turn is 360. Keys are held for the whole
tick; to keep walking, repeat w every tick. e interacts with the on-screen prompt when one is shown. Clicks use a
64x36 grid over the screen: c32,18 is the centre.

Reply in exactly this format, nothing else:
PLAN: <one short sentence: what you see and what you're doing next>
ACTIONS: <tick> | <tick> | <tick> | <tick>"""


def _jpeg(frame) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(frame).resize((W, H), Image.Resampling.BOX).save(buf, format="JPEG", quality=88)
    return base64.standard_b64encode(buf.getvalue()).decode()


def _dotenv() -> dict:
    try:
        return dict(line.strip().split("=", 1) for line in open(".env") if "=" in line and not line.lstrip().startswith("#"))
    except OSError:
        return {}


class Claude:
    def __init__(self, key: str, model: str = "claude-opus-5-5"):
        import anthropic
        self.client, self.model = anthropic.Anthropic(api_key=key.strip().strip('"')), model

    def __call__(self, user_text: str, imgs: list[str]) -> tuple[str, str]:
        content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": d}} for d in imgs]
        r = self.client.beta.messages.create(
            model=self.model, max_tokens=8000, system=GUIDE, betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            thinking={"type": "adaptive", "display": "summarized"}, output_config={"effort": "low"},
            messages=[{"role": "user", "content": content + [{"type": "text", "text": user_text}]}])
        if r.stop_reason == "refusal":
            return "", "refusal"
        text = "".join(b.text for b in r.content if b.type == "text")
        think = "".join(b.thinking for b in r.content if b.type == "thinking")
        return text, think


class OpenAI:   # also OpenRouter (OpenAI-compatible): see OpenRouter below
    def __init__(self, key: str, model: str = "", base_url: str | None = None):
        import openai
        self.client = openai.OpenAI(api_key=key.strip().strip('"'), base_url=base_url)
        self.model = model or os.environ.get("BH_OPENAI_MODEL", "gpt-5")

    def __call__(self, user_text: str, imgs: list[str]) -> tuple[str, str]:
        content = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + d}} for d in imgs]
        r = self.client.chat.completions.create(model=self.model, messages=[
            {"role": "system", "content": GUIDE}, {"role": "user", "content": content + [{"type": "text", "text": user_text}]}])
        return r.choices[0].message.content or "", ""


def OpenRouter(key: str):
    return OpenAI(key, os.environ.get("BH_OPENROUTER_MODEL", "qwen/qwen3.5-9b"), "https://openrouter.ai/api/v1")


def connect(prefer: str = ""):
    """First provider in .env that answers a one-line ping."""
    env = {**_dotenv(), **os.environ}
    tries = [("ANTHROPIC_API_KEY", Claude), ("OPENAI_API_KEY", OpenAI), ("OPENROUTER_API_KEY", OpenRouter)]
    if prefer:
        tries.sort(key=lambda t: prefer not in t[0].lower())
    errs = []
    for var, cls in tries:
        if not env.get(var):
            continue
        try:
            m = cls(env[var])
            m("Reply with: PLAN: ok\nACTIONS: . 0 0 | . 0 0 | . 0 0 | . 0 0", [])
            print(f"using {var} → {m.model}", flush=True)
            return m
        except Exception as e:   # bad key, no access, quota: try the next one
            errs.append(f"{var}: {type(e).__name__}: {str(e)[:200]}")
    raise SystemExit("no working API key in .env\n" + "\n".join(errs))


def _lenient(tick: str) -> str:
    """Small models' near-misses → our tick format: 'w' → 'w 0 0', 'w,0,-5' → 'w 0 -5', 'w0,0' → 'w 0 0'."""
    tick = re.sub(r"\b([cm])(\d+),(\d+)", r"\1\2;\3", tick.strip())     # protect c32,18 / m32,18
    t = tick.replace(",", " ").replace(";", ",").split()
    if not t:
        return ". 0 0"
    m = re.fullmatch(r"([a-z.]+)(-?\d+)", t[0])                              # 'w0' → 'w', '0'
    if m:
        t = [m.group(1), m.group(2)] + t[1:]
    nums = []
    while len(nums) < 2 and len(t) > 1 + len(nums) and re.fullmatch(r"-?\d+", t[1 + len(nums)]):
        nums.append(t[1 + len(nums)])
    return " ".join([t[0]] + (nums + ["0", "0"])[:2] + t[1 + len(nums):])


def parse(text: str, lenient: bool = False) -> tuple[str, str]:
    plan = re.search(r"PLAN:\s*(.*)", text)
    acts = re.search(r"ACTIONS:\s*(.*)", text)
    a = acts.group(1).strip() if acts else ""
    if lenient and a:
        a = " | ".join(_lenient(x) for x in a.split("|"))
    return (plan.group(1).strip() if plan else ""), a


def run(model, games: list[str], ticks: int, out: str, lenient: bool = False) -> dict:
    from sim.render import best_device, render
    dev = best_device()
    envs = [sim.make(g, seed=SEED, max_steps=10 ** 9, **PLANS[g][0]) for g in games]
    hist = [deque(maxlen=FRAME_GAP + 1) for _ in envs]
    texts, logs, first = [[] for _ in envs], [[] for _ in envs], [None] * len(envs)
    plans, fails, cost_calls = [""] * len(envs), [0] * len(envs), 0
    pool = ThreadPoolExecutor(len(envs))

    def decide(i, t):
        g = games[i]
        user = (f"Game: {TITLES[g]}. Your goal: {PLANS[g][1]}.\n"
                f"Time: {t / 10:.1f} s. Your previous plan: {plans[i] or '(none, you just spawned)'}\n"
                f"Your last inputs: {' | '.join(texts[i][-8:]) or '(none)'}\n"
                "Image 1 is 0.2 s ago, image 2 is now. Your next 4 ticks:")
        for attempt in range(4):
            try:
                return model(user, [_jpeg(hist[i][-2]), _jpeg(hist[i][-1])])   # mid-chunk frame (0.2 s ago), now
            except Exception as e:
                err = e
                time.sleep(2 ** attempt)
        return "", f"error: {err}"

    t0 = time.time()
    for t in range(0, ticks, CHUNK):
        frames = render([e.scene() for e in envs], 252 * 2, 448 * 2, device=dev).cpu().numpy()   # 896x504, downscaled to WxH
        for h, f in zip(hist, frames):
            h.append(f)
            while len(h) < FRAME_GAP + 1:   # spawn: no motion yet, both images are the first frame
                h.append(f)
        outs = list(pool.map(lambda i: decide(i, t), range(len(envs))))
        cost_calls += len(envs)
        for i, ((text, think), g, e) in enumerate(zip(outs, games, envs)):
            plan, acts_text = parse(text, lenient)
            acts, ok = decode(acts_text, e.cursor)
            fails[i] += not ok
            plans[i] = plan or plans[i]
            logs[i].append({"t": t, "plan": plan, "actions": acts_text, "parsed": ok, "thinking": think[:1500], "raw": text[:500]})
            for k in range(CHUNK):   # frames FRAME_GAP ticks apart: render the in-between ticks too
                texts[i].append(encode_tick(acts[k], e.cursor))
                e.step(decode(texts[i][-1], e.cursor)[0][0])
                if k == FRAME_GAP - 1:
                    hist[i].append(render([e.scene()], 252 * 2, 448 * 2, device=dev)[0].cpu().numpy())
                if first[i] is None and PLANS[g][2](e):
                    first[i] = len(texts[i])
        print(f"t={t / 10:5.1f}s  {time.time() - t0:5.0f}s wall  " +
              "  ".join(f"{g}:{'✓' if f else '·'}" for g, f in zip(games, first)), flush=True)
    os.makedirs(out, exist_ok=True)
    res = {}
    for g, e, tx, lg, f, nf in zip(games, envs, texts, logs, first, fails):
        res[g] = {"goal": PLANS[g][1], "reached_at": f, "score": float(e.score()), "deaths": int(e.deaths), "info": e.info(),
                  "parse_fail": nf / max(1, len(lg))}
        json.dump({"game": g, "seed": SEED, "kw": PLANS[g][0], "ticks": len(tx), "inputs": tx, "decisions": lg} | res[g],
                  open(os.path.join(out, g + ".json"), "w"), indent=1)
    json.dump(res | {"_model": model.model, "_calls": cost_calls}, open(os.path.join(out, "summary.json"), "w"), indent=1)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default=",".join(PLANS))
    ap.add_argument("--ticks", type=int, default=600)
    ap.add_argument("--prefer", default="", help="anthropic or openai")
    args = ap.parse_args(argv)
    model = connect(args.prefer)
    out = os.path.join("results/frontier", model.model.replace("/", "_"))
    for g, r in run(model, args.games.split(","), args.ticks, out).items():
        print(f"{g:9s} {'REACHED at tick ' + str(r['reached_at']) if r['reached_at'] else 'not reached':24s} score {r['score']:.2f}  "
              f"deaths {r['deaths']}  parse-fail {r['parse_fail']:.0%}  {r['info']}")


if __name__ == "__main__":
    main()
