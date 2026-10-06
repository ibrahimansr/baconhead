"""The model server: base VLM + one LoRA per game, over HTTP. The same script runs on this Mac (local backend)
or on your GPU box (SSH backend, reached through a tunnel), so the engine never cares where it runs.

    python -m engine.server --port 8771 --root .

    GET  /health                 → {"ok", "device", "base"}
    GET  /adapters               → {"games": [ids with a trained LoRA]}
    POST /generate               {"images": [b64 jpeg 0.2 s ago, b64 jpeg now], "adapter": id|null} → {"text"}
    POST /train                  {"game": id, "demos": dir on this machine} → {"job"}
    GET  /train/<job>            → {"state": running|done|failed, "log": tail}
    POST /forget                 {"game": id} → deletes that game's LoRA (back to the base model)
Adapters live in <root>/models/<game>/; the base is <root>/models/base if it exists, else the HF id.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Model:
    def __init__(self, base: str, root: str, device: str):
        import torch
        from transformers import AutoModelForImageTextToText
        from model.vlm import processor
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        self.model = AutoModelForImageTextToText.from_pretrained(base, dtype=dtype).to(device).eval()   # CPU first: loading straight onto MPS hangs
        self.proc = processor(base)
        self.base, self.root, self.device = base, root, device
        self.loaded: dict[str, float] = {}           # game → adapter mtime when loaded
        self.lock = threading.Lock()

    def adapter_dir(self, game: str) -> str:
        return os.path.join(self.root, "models", game)

    def games(self) -> list[str]:
        d = os.path.join(self.root, "models")
        return sorted(g for g in os.listdir(d) if os.path.exists(os.path.join(d, g, "adapter_config.json"))) if os.path.isdir(d) else []

    def _use(self, game: str | None) -> None:
        """Add the game's LoRA (reloading it if it was retrained), or take all adapters off for the base model."""
        path = self.adapter_dir(game) if game else None
        if not path or not os.path.exists(os.path.join(path, "adapter_config.json")):
            if self.loaded:
                self.model.disable_adapters()
            return
        mtime = os.path.getmtime(path)
        if self.loaded.get(game) != mtime:
            if game in self.loaded:
                self.model.delete_adapter(game)
            if self.device == "mps":   # transformers' loader hangs writing onto MPS; load on CPU and move back (~seconds, once per game)
                self.model.to("cpu")
                self.model.load_adapter(path, adapter_name=game)
                self.model.to(self.device)
            else:
                self.model.load_adapter(path, adapter_name=game)
            self.loaded[game] = mtime
        self.model.set_adapter(game)
        self.model.enable_adapters()

    def forget(self, game: str) -> None:
        with self.lock:
            if game in self.loaded:
                self.model.delete_adapter(game)
                del self.loaded[game]
            shutil.rmtree(self.adapter_dir(game), ignore_errors=True)

    def generate(self, imgs: list, game: str | None) -> str:
        import torch
        from model.vlm import prompt
        with self.lock:
            self._use(game)
            text = self.proc.apply_chat_template(prompt(), add_generation_prompt=True, tokenize=False)
            inp = self.proc(text=[text], images=imgs, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.generate(**inp, max_new_tokens=64, do_sample=False)
            return self.proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def serve(port: int, root: str, base: str | None, device: str | None):
    from PIL import Image
    from sim.render import best_device
    base = base or (os.path.join(root, "models", "base") if os.path.isdir(os.path.join(root, "models", "base")) else None)
    if base is None:
        from model.vlm import MODEL_ID
        base = MODEL_ID
    device = device or best_device()
    m = Model(base, root, device)
    jobs: dict[str, dict] = {}

    def train(job: str, game: str, demos: str):
        log = os.path.join(root, "logs", f"teach_{game}_{job}.log")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, "w") as f:   # a subprocess, so a crash in training can't take the server down
            rc = subprocess.run([sys.executable, "-m", "train.teach", "--game", game, "--demos", os.path.join(root, demos),
                                 "--base", base, "--out", m.adapter_dir(game)], stdout=f, stderr=subprocess.STDOUT, cwd=CODE).returncode
        jobs[job]["state"] = "done" if rc == 0 else "failed"

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self._send({"ok": True, "device": device, "base": base})
            if self.path == "/adapters":
                return self._send({"games": m.games()})
            if self.path.startswith("/train/"):
                j = jobs.get(self.path[7:])
                if not j:
                    return self._send({"error": "no such job"}, 404)
                tail = open(j["log"]).read()[-2000:] if os.path.exists(j["log"]) else ""
                return self._send({"state": j["state"], "log": tail})
            self._send({"error": "not found"}, 404)

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            try:
                if self.path == "/generate":
                    imgs = [Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB") for b in req["images"]]
                    return self._send({"text": m.generate(imgs, req.get("adapter"))})
                if self.path == "/train":
                    job = uuid.uuid4().hex[:8]
                    jobs[job] = {"state": "running", "log": os.path.join(root, "logs", f"teach_{req['game']}_{job}.log")}
                    threading.Thread(target=train, args=(job, req["game"], req["demos"]), daemon=True).start()
                    return self._send({"job": job})
                if self.path == "/forget":
                    m.forget(req["game"])
                    return self._send({"ok": True})
            except Exception as e:   # a bad request must not kill the server
                return self._send({"error": repr(e)}, 500)
            self._send({"error": "not found"}, 404)

    print(f"model server on 127.0.0.1:{port}, base={base}, device={device}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()   # localhost only; remote access goes through the SSH tunnel


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8771)
    ap.add_argument("--root", default=".")
    ap.add_argument("--base", default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    serve(a.port, os.path.abspath(a.root), a.base, a.device)
