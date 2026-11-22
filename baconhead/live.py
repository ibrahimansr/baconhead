import argparse
import base64
import io
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import modal

MODEL = os.environ.get("BH_LIVE_MODEL", "ibrahimansr/robloxagent")
PX_SCALE = 0.2   # model px -> Roblox drag px; depends on Roblox's camera sensitivity (--calibrate)
STEP_S = 0.05    # one key group

app = modal.App("robloxagent-live")   # modal deploy baconhead/live.py; python -m baconhead.live --url <url>
vol = modal.Volume.from_name("baconhead", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("vllm==0.26.0", "transformers==5.20.1")
         .env({"HF_HOME": "/vol/hf", "HF_HUB_DISABLE_XET": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"}))


@app.function(image=image, gpu="H100", volumes={"/vol": vol}, timeout=3600, scaledown_window=120)  # 0 GPUs after 2 idle min
@modal.concurrent(max_inputs=8)
@modal.web_server(8000, startup_timeout=900)
def serve():
    subprocess.Popen(["vllm", "serve", MODEL, "--served-model-name", "robloxagent", "--host", "0.0.0.0", "--port", "8000",
                      "--trust-remote-code", "--max-model-len", "8192", "--max-num-batched-tokens", "8192",
                      "--limit-mm-per-prompt", '{"image":2,"video":0}', "--gpu-memory-utilization", "0.9", "--max-num-seqs", "8"])


KEYCODE = {"W": 13, "A": 0, "S": 1, "D": 2, "Space": 49, "Shift": 56, "E": 14, "Q": 12, "R": 15, "F": 3,
           "one": 18, "two": 19, "three": 20, "four": 21, "five": 23, "Up": 126, "Down": 125, "Left": 123, "Right": 124}
# esc and Tab are deliberately not sent: they open Roblox's menu / leaderboard and would strand the policy.


# the frontmost Roblox window: its bounds, a screenshot, and Quartz input that only flows while it's in front
class Roblox:

    def __init__(self, camera="drag", px_scale=PX_SCALE):
        import Quartz
        self.Q, self.camera, self.px_scale = Quartz, camera, px_scale
        self.src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
        self.held = set()
        self.lb = False

    def front(self):
        from AppKit import NSWorkspace
        return "roblox" in (NSWorkspace.sharedWorkspace().frontmostApplication().localizedName() or "").lower()

    def bounds(self):
        Q = self.Q
        wins = Q.CGWindowListCopyWindowInfo(Q.kCGWindowListOptionOnScreenOnly | Q.kCGWindowListExcludeDesktopElements, Q.kCGNullWindowID)
        rob = [w for w in wins if "roblox" in str(w.get("kCGWindowOwnerName", "")).lower() and w.get("kCGWindowLayer", 1) == 0]
        if not rob:
            return None
        b = max(rob, key=lambda w: w["kCGWindowBounds"]["Width"] * w["kCGWindowBounds"]["Height"])["kCGWindowBounds"]
        top = 28   # the macOS title bar isn't game
        return {"left": int(b["X"]), "top": int(b["Y"]) + top, "width": int(b["Width"]), "height": int(b["Height"]) - top}

    # the game view at the model's frame size, bicubic, JPEG q88 (the training input)
    def grab(self, sct, box):
        from PIL import Image
        from baconhead.agent import H, W
        shot = sct.grab(box)
        img = Image.frombytes("RGB", shot.size, shot.rgb).resize((W, H), Image.BICUBIC)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
        return buf.getvalue()

    def _post(self, ev):
        self.Q.CGEventPost(self.Q.kCGHIDEventTap, ev)

    def key(self, name, down):
        if name == "LB":
            Q = self.Q
            c = self.center
            self._post(Q.CGEventCreateMouseEvent(self.src, Q.kCGEventLeftMouseDown if down else Q.kCGEventLeftMouseUp,
                                                 Q.CGPoint(*c), Q.kCGMouseButtonLeft))
            self.lb = down
        elif name in KEYCODE:
            self._post(self.Q.CGEventCreateKeyboardEvent(self.src, KEYCODE[name], down))
        (self.held.add if down else self.held.discard)(name)

    def release(self):
        for k in list(self.held):
            self.key(k, False)

    # camera turn by (dx, dy) Roblox px over dur seconds: right-drag, or plain moves in shift-lock
    def turn(self, dx, dy, dur):
        Q, (cx, cy) = self.Q, self.center
        n = max(2, int(max(abs(dx), abs(dy)) // 6))
        drag = self.camera == "drag"
        if drag:
            self._post(Q.CGEventCreateMouseEvent(self.src, Q.kCGEventRightMouseDown, Q.CGPoint(cx, cy), Q.kCGMouseButtonRight))
        ax = ay = 0.0
        for _ in range(n):
            ax, ay = ax + dx / n, ay + dy / n
            ix, iy = int(round(ax)), int(round(ay))
            ax, ay = ax - ix, ay - iy
            ev = Q.CGEventCreateMouseEvent(self.src, Q.kCGEventRightMouseDragged if drag else Q.kCGEventMouseMoved,
                                           Q.CGPoint(cx, cy), Q.kCGMouseButtonRight)
            Q.CGEventSetIntegerValueField(ev, Q.kCGMouseEventDeltaX, ix)   # roblox reads the delta field, not the position
            Q.CGEventSetIntegerValueField(ev, Q.kCGMouseEventDeltaY, iy)
            self._post(ev)
            time.sleep(dur / n)
        if drag:
            self._post(Q.CGEventCreateMouseEvent(self.src, Q.kCGEventRightMouseUp, Q.CGPoint(cx, cy), Q.kCGMouseButtonRight))

    # one 300 ms chunk: per 50 ms group, press what's new, release what's gone, turn the camera a sixth
    def play(self, chunk):
        dx, dy, _dz, groups = chunk
        n = len(groups)
        for names in groups:
            want = {n for n in names if n in KEYCODE or n == "LB"}
            for k in self.held - want:
                self.key(k, False)
            for k in want - self.held:
                self.key(k, True)
            t0 = time.time()
            if dx or dy:
                self.turn(dx * self.px_scale / n, dy * self.px_scale / n, STEP_S * 0.8)
            time.sleep(max(0.0, STEP_S - (time.time() - t0)))


# system prompt with the task; the previous and current frame as the user turn. Stops at <|im_end|> (the model's eos)
def request(url, prev, cur, system, seed):
    import requests
    img = lambda j: {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(j).decode()}}
    body = {"model": "robloxagent", "temperature": 0.8, "top_p": 0.99, "max_tokens": 64, "skip_special_tokens": True,
            "chat_template_kwargs": {"enable_thinking": False}, "seed": seed,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": [img(prev), img(cur)]}]}
    t0 = time.time()
    r = requests.post(url.rstrip("/") + "/v1/chat/completions", json=body, timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or "", time.time() - t0


def run(args):
    import mss
    from baconhead.agent import SYSTEM, TASKS, TITLES, parse, task as as_task
    rob = Roblox(args.camera, args.px_scale)
    out = os.path.join("results/live", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    log = open(os.path.join(out, "log.jsonl"), "w")
    system = SYSTEM.format(game=TITLES.get(args.game, args.game), task=as_task(args.task or TASKS[args.game]))
    pool = ThreadPoolExecutor(1)
    print(f"waiting for Roblox to be the frontmost window... ({args.seconds}s of play, logs → {out})", flush=True)
    with mss.mss() as sct:
        while not (rob.front() and rob.bounds()):
            time.sleep(0.2)
        box = rob.bounds()
        rob.center = (box["left"] + box["width"] / 2, box["top"] + box["height"] / 2)
        frame = rob.grab(sct, box)
        fut = pool.submit(request, args.url, frame, frame, system, 0)
        t_end, i, lat = time.time() + args.seconds, 0, []
        try:
            while time.time() < t_end:
                text, dt = fut.result()
                lat.append(dt)
                chunk = parse(text)
                i += 1
                if not rob.front():   # you switched away: let go of everything and wait
                    rob.release()
                    print("paused (Roblox not in front)", flush=True)
                    while not rob.front():
                        time.sleep(0.2)
                prev, frame = frame, rob.grab(sct, box)                         # the next decision's frames,
                fut = pool.submit(request, args.url, prev, frame, system, i)   # in flight while this chunk plays
                with open(os.path.join(out, f"{i:05d}.jpg"), "wb") as f:
                    f.write(frame)
                log.write(json.dumps({"i": i, "text": text, "latency_s": round(dt, 3)}) + "\n")
                if i % 25 == 0:
                    print(f"{i * 0.3:5.1f}s  latency p50 {sorted(lat)[len(lat) // 2] * 1000:.0f} ms  last: {text!r}", flush=True)
                rob.play(chunk[:4])
        finally:
            rob.release()
            log.close()
    lat.sort()
    print(f"done: {i} decisions, round trip p50 {lat[len(lat) // 2] * 1000:.0f} ms, p90 {lat[int(len(lat) * 0.9)] * 1000:.0f} ms "
          f"(real time needs < 300). Frames + outputs in {out}")


# turns the camera by 300 drag px after 3 s: note how many degrees Roblox turned (D), then --px-scale 30/D
def calibrate(args):
    rob = Roblox(args.camera)
    print("focus Roblox; turning in 3 s", flush=True)
    time.sleep(3)
    box = rob.bounds()
    rob.center = (box["left"] + box["width"] / 2, box["top"] + box["height"] / 2)
    if rob.front():
        rob.turn(300, 0, 0.5)
    print("if Roblox turned D degrees for those 300 px, use --px-scale 30/D (a model unit is 0.1 deg: 0.1 * 300 / D);"
          " e.g. a 90° turn → 0.33, a 150° turn → 0.2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="the deployed server's URL (modal deploy baconhead/live.py prints it)")
    ap.add_argument("--game", default="obby", help="a trained game (obby, doors, rivals, ...) or any title")
    ap.add_argument("--task", default=None, help="defaults to the game's training task")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--camera", choices=["drag", "locked"], default="drag", help="locked: shift-lock / first person")
    ap.add_argument("--px-scale", type=float, default=PX_SCALE)
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()
    calibrate(args) if args.calibrate else run(args)


if __name__ == "__main__":
    main()
