import argparse
import base64
import io
import json
import os
import re

import modal
import numpy as np

MODEL = "ibrahimansr/robloxagent"
W, H = 640, 360        # each frame the model sees
FRAMES = 2             # the frame from one chunk ago, then the current one
STEPS = 6              # 50 ms key steps per chunk, so a chunk is 300 ms
TICKS = 3              # 100 ms sim ticks per chunk
STEP_SUBSTEPS = 3      # 50 ms at the sims' 60 Hz
DEG_PER_UNIT = 0.1     # camera units are tenths of a degree
ARROW_DEG = 6.0        # left/right arrow keys: 120 deg/s
TITLES = {"obby": "Obby", "doors": "DOORS", "rivals": "RIVALS", "mm2": "Murder Mystery 2",
          "disaster": "Natural Disaster Survival", "forest": "99 Nights in the Forest", "fisch": "Fisch",
          "garden": "Grow a Garden", "adoptme": "Adopt Me!", "brainrot": "Steal a Brainrot"}
SYSTEM = ("You are playing Roblox {game} with a keyboard and mouse. You see the frame from 300 ms ago and the current frame. "
          "Reply with one line for the next 300 ms: camera yaw, pitch (tenths of a degree) and scroll, "
          "then the keys to hold in each of six 50 ms steps, all separated by |.\nTask: {task}")
KEY_OF = {"W": "w", "A": "a", "S": "s", "D": "d", "Up": "w", "Down": "s", "Space": "space", "Shift": "shift",
          "E": "e", "Q": "q", "R": "r", "F": "f", "one": "1", "two": "2", "three": "3", "LB": "lmb", "Esc": "esc"}
TASKS = {"obby": "Get through the obby: jump across the platforms from checkpoint to checkpoint without falling.",
         "disaster": "Survive each disaster until it ends.",
         "forest": "Gather logs, keep the campfire burning, eat, and survive the nights.",
         "doors": "Go through the numbered doors one by one and survive the monsters.",
         "brainrot": "Buy brainrots from the carpet and collect the cash they earn in your base.",
         "rivals": "Win 1v1 rounds by finding and shooting the enemy.",
         "mm2": "Survive the round; as Sheriff shoot the Murderer, as Murderer eliminate everyone."}


# capital first letter, closing punctuation
def task(text):
    text = text.strip()
    return text[0].upper() + text[1:] + ("" if text[-1] in ".!?" else ".")


# system + user turn; images are chat content parts, oldest frame first
def prompt(game, images):
    return [{"role": "system", "content": SYSTEM.format(game=TITLES.get(game, game), task=TASKS.get(game, game))},
            {"role": "user", "content": images}]


# -> (yaw, pitch, scroll, STEPS key groups, well-formed). Never raises
def parse(text):
    s = text.split("</think>")[-1].split("<|im_end|>")[0].strip().split("\n")[0]
    fields = [f.strip() for f in s.split("|")]
    nums = fields[0].split()
    num_ok = len(nums) == 3 and all(re.fullmatch(r"-?\d+", n) for n in nums)
    dx, dy, dz = (max(-999, min(999, int(n))) for n in nums) if num_ok else (0, 0, 0)
    groups = [f.split()[:4] for f in fields[1:STEPS + 1]]
    return dx, dy, dz, groups + [[] for _ in range(STEPS - len(groups))], num_ok and len(fields) == STEPS + 1


# a chunk back as model text
def chunk_text(c):
    dx, dy, dz, groups = c
    return " | ".join([f"{dx} {dy} {dz}"] + [" ".join(g) for g in groups])


# A window is open (shop, backpack, Esc menu): the mouse moves the cursor instead of the camera
def cursor_mode(env):
    return bool(getattr(env, "menu", None) or getattr(env, "ui", None))


# one 300 ms chunk on a sim: each 50 ms key group held for 3 substeps, the camera turn spread over all 18
# part=0/1/2 plays only that 100 ms, with the same substeps (for rendering between them)
def play_chunk(env, dx, dy, dz, groups, part=None):
    from baconhead.sim.core import Input
    n = STEPS * STEP_SUBSTEPS
    cur = cursor_mode(env)
    env.reward = 0.0
    for gi, names in enumerate(groups):
        if part is not None and gi // 2 != part:
            continue
        keys = frozenset(KEY_OF[k] for k in names if k in KEY_OF)
        arrow = ARROW_DEG * (("Right" in names) - ("Left" in names)) / STEP_SUBSTEPS
        for k in range(STEP_SUBSTEPS):
            scroll = int(np.sign(dz)) if gi == 0 and k == 0 else 0
            if cur:   # a menu is open: the same units move the cursor, in frame pixels
                c = (float(np.clip(env.cursor[0] + dx / W / n, 0, 1)), float(np.clip(env.cursor[1] + dy / H / n, 0, 1)))
                env.substep(Input(keys=keys, cursor=c, scroll=scroll))
            else:
                env.substep(Input(keys=keys, dyaw=dx * DEG_PER_UNIT / n + arrow, dpitch=dy * DEG_PER_UNIT / n,
                                  cursor=env.cursor, scroll=scroll))
        if gi % 2 == 1:
            env.steps += 1


def jpeg_url(frame):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# each game's gold goal on `seeds` held-out levels, all envs in lockstep, one batched call per chunk
def run(llm, games, ticks, seeds, out):
    from baconhead import sim
    from baconhead.sim.gold import PLANS, SEED
    from baconhead.sim.render import render
    from vllm import SamplingParams
    stop = [llm.get_tokenizer().convert_tokens_to_ids("<|im_end|>")]
    runs = [(g, k) for g in games for k in range(seeds)]
    envs = [sim.make(g, seed=SEED + k, max_steps=10 ** 9, **PLANS[g][0]) for g, k in runs]
    logs, first, bad, prev = [[] for _ in runs], [None] * len(runs), [0] * len(runs), None
    for d in range(ticks // TICKS):
        frames = np.concatenate([render([e.scene() for e in envs[i:i + 10]], H, W, device="cuda").cpu().numpy()
                                 for i in range(0, len(envs), 10)])  # [N, H, W, 3]
        prev = frames if prev is None else prev
        msgs = [prompt(g, [{"type": "image_url", "image_url": {"url": jpeg_url(p)}}, {"type": "image_url", "image_url": {"url": jpeg_url(f)}}])
                for (g, _), p, f in zip(runs, prev, frames)]
        params = [SamplingParams(temperature=0.8, top_p=0.99, max_tokens=64, stop_token_ids=stop,
                                 skip_special_tokens=True, seed=d * 1000 + i) for i in range(len(runs))]
        outs = llm.chat(msgs, params, chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
        prev = frames
        for i, ((g, k), e, o) in enumerate(zip(runs, envs, outs)):
            text = o.outputs[0].text
            dx, dy, dz, groups, ok = parse(text)
            bad[i] += not ok
            logs[i].append({"t": TICKS * d, "text": text, "chunk": [dx, dy, dz, groups]})
            play_chunk(e, dx, dy, dz, groups)
            if first[i] is None and PLANS[g][2](e):
                first[i] = TICKS * (d + 1)
        if d % 20 == 0:
            print(f"t={d * TICKS / 10:5.1f}s goals " + " ".join(f"{g}{k}:{'✓' if f else '·'}" for (g, k), f in zip(runs, first)), flush=True)
    os.makedirs(out, exist_ok=True)
    res = {}
    for (g, k), e, lg, f, nb in zip(runs, envs, logs, first, bad):
        lv = {"seed": SEED + k, "reached_at": f, "score": float(e.score()), "deaths": int(e.deaths), "info": e.info(),
              "malformed": nb / max(1, len(lg))}
        res.setdefault(g, {"goal": PLANS[g][1], "levels": []})["levels"].append(lv)
        json.dump({"game": g, "kw": PLANS[g][0], "goal": PLANS[g][1], "decisions": lg} | lv,
                  open(os.path.join(out, f"{g}_s{k}.json"), "w"))
    for v in res.values():
        v["success"] = float(np.mean([lv["reached_at"] is not None for lv in v["levels"]]))
    res["_success"] = float(np.mean([v["success"] for v in res.values()]))
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=1)
    return res


# every RobloxAgent run <d>/*.json -> .mp4, replayed on a fresh level and rendered every 100 ms
def videos(d):
    import glob
    import subprocess
    from baconhead import sim
    from baconhead.model.vlm import OBS_H, OBS_W
    from baconhead.sim.render import best_device, render
    dev = best_device()
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(p))
        if not r.get("decisions") or "chunk" not in r["decisions"][0]:   # not a RobloxAgent run
            continue
        env = sim.make(r["game"], seed=r["seed"], max_steps=10 ** 9, **r["kw"])
        mp4 = p[:-5] + ".mp4"
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OBS_W}x{OBS_H}",
                               "-r", "10", "-i", "-", "-pix_fmt", "yuv420p", mp4], stdin=subprocess.PIPE)
        for c in r["decisions"]:
            for part in range(TICKS):
                ff.stdin.write(render([env.scene()], OBS_H, OBS_W, device=dev)[0].cpu().numpy().tobytes())
                play_chunk(env, *c["chunk"], part=part)
        ff.stdin.close()
        ff.wait()
        print(mp4, "goal", f"reached at tick {r['reached_at']}" if r["reached_at"] else "not reached")


app = modal.App("robloxagent-eval")   # modal run baconhead/agent.py --games obby; python -m baconhead.agent --video <dir>
vol = modal.Volume.from_name("baconhead", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("build-essential")
         .pip_install("vllm==0.26.0", "transformers==5.20.1", "pillow", "numpy", "datasets==5.0.1")  # 5.23 breaks vLLM image profiling
         .env({"HF_HOME": "/vol/hf", "HF_HUB_DISABLE_XET": "1",
               "VLLM_USE_FLASHINFER_SAMPLER": "0"})  # its sampler JIT-compiles and the image has no nvcc
         .add_local_python_source("baconhead"))


@app.function(image=image, gpu="A10", cpu=8, memory=32768, timeout=3 * 3600, volumes={"/vol": vol})
def remote(games, ticks, seeds, model=MODEL, name="robloxagent"):
    from vllm import LLM
    vol.reload()
    llm = LLM(model, trust_remote_code=True, max_model_len=8192, limit_mm_per_prompt={"image": FRAMES, "video": 0},
              gpu_memory_utilization=0.6, max_num_seqs=64)  # leaves room for the renderer on a 24 GB card
    res = run(llm, games, ticks, seeds, f"/vol/results/{name}")
    vol.commit()
    return res


@app.local_entrypoint()
def main(ticks: int = 600, seeds: int = 3, games: str = "obby", model: str = MODEL, name: str = "robloxagent"):
    res = remote.remote(games.split(","), ticks, seeds, model, name)
    for g, v in res.items():
        if not g.startswith("_"):
            print(f"{g:9s} success {v['success']:.0%}  " + "  ".join(
                f"s{lv['seed'] % 1000}:{'✓@' + str(lv['reached_at']) if lv['reached_at'] else '·'}(d{lv['deaths']})" for lv in v["levels"]))
    print(f"overall {res['_success']:.0%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="a results dir of saved runs")
    videos(ap.parse_args().video)
