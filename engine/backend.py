"""Where the model runs. Pick one; the engine talks to all of them the same way.

  local   engine.server as a subprocess on this Mac (MPS). Free; the 3B model is slow here.
  ssh     your own GPU box. We rsync this repo there, install requirements-server.txt into a venv on first
          connect, start engine.server, and tunnel its port back to 127.0.0.1. Nothing is exposed publicly.
  modal   Modal's GPUs (train/modal_app.py, must be deployed). Costs money; only used if you pick it.

Config lives in ~/.baconhead/config.json, e.g.
  {"backend": "ssh", "host": "me@gpu-box", "port": 22, "dir": "~/baconhead", "python": "python3"}
"""

from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
import time
import urllib.request

from model.vlm import VLMRunner

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.expanduser("~/.baconhead/config.json")
SERVER_PORT = 8771


def load_config() -> dict:
    try:
        return json.load(open(CONFIG))
    except (OSError, ValueError):
        return {"backend": "local"}


def save_config(cfg: dict) -> None:
    os.makedirs(os.path.dirname(CONFIG), exist_ok=True)
    json.dump(cfg, open(CONFIG, "w"), indent=2)


class HTTPBackend:
    """A model server reachable at 127.0.0.1:port (directly, or through an SSH tunnel)."""

    kind = "local"

    def __init__(self, port: int, procs: list):
        self.url, self.procs = f"http://127.0.0.1:{port}", procs

    def _call(self, path, body=None, timeout=120):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)

    def wait(self, timeout=900) -> dict:
        """Model loading takes a while (minutes for 3B on first download)."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                return self._call("/health", timeout=5)
            except OSError:
                if any(p.poll() is not None for p in self.procs):
                    raise RuntimeError("model server exited; see its log")
                time.sleep(2)
        raise TimeoutError("model server didn't come up")

    def generate(self, imgs: list, adapter: str | None) -> str:
        return self._call("/generate", {"images": [base64.b64encode(_jpeg(im)).decode() for im in imgs], "adapter": adapter})["text"]

    def adapters(self) -> list[str]:
        return self._call("/adapters")["games"]

    def upload(self, local_dir: str, game: str) -> str:
        return os.path.abspath(local_dir)

    def train(self, game: str, local_dir: str) -> str:
        return self._call("/train", {"game": game, "demos": self.upload(local_dir, game)})["job"]

    def job(self, job: str) -> dict:
        return self._call(f"/train/{job}")

    def forget(self, game: str) -> None:
        self._call("/forget", {"game": game})

    def close(self) -> None:
        for p in self.procs:
            p.terminate()


def local() -> HTTPBackend:
    os.makedirs(os.path.join(REPO, "logs"), exist_ok=True)
    log = open(os.path.join(REPO, "logs", "server.log"), "a")
    p = subprocess.Popen([sys.executable, "-m", "engine.server", "--port", str(SERVER_PORT), "--root", REPO],
                         cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    return HTTPBackend(SERVER_PORT, [p])


class SSHBackend(HTTPBackend):
    kind = "ssh"

    def __init__(self, host: str, port: int = 22, dir: str = "~/baconhead", python: str = "python3", **_):
        self.host, self.dir = host, dir
        self.ssh = ["ssh", "-p", str(port), "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", host]
        self.rsh = f"ssh -p {port} -o BatchMode=yes"
        subprocess.run(self.ssh + [f"mkdir -p {dir}"], check=True, timeout=30)
        subprocess.run(["rsync", "-az", "--delete", "-e", self.rsh, "--exclude", ".venv", "--exclude", "demos", "--exclude", "models",
                        "--exclude", "data", "--exclude", "logs", "--exclude", "runs", "--exclude", ".git",
                        REPO + "/", f"{host}:{dir}/"], check=True)
        setup = (f"cd {dir} && ( test -x .venv/bin/python || ( {python} -m venv .venv && .venv/bin/pip install -q -r requirements-server.txt ) )"
                 f" && mkdir -p logs && ( pkill -f '[e]ngine.server --port {SERVER_PORT}' ; true )"   # [e]: don't match this shell
                 f" && nohup .venv/bin/python -m engine.server --port {SERVER_PORT} --root . < /dev/null > logs/server.log 2>&1 &")
        subprocess.run(self.ssh + [setup], check=True, timeout=1800)
        tunnel = subprocess.Popen(self.ssh[:-1] + ["-N", "-L", f"{SERVER_PORT}:127.0.0.1:{SERVER_PORT}", host])   # the server only listens on the box's localhost
        super().__init__(SERVER_PORT, [tunnel])

    def upload(self, local_dir: str, game: str) -> str:
        remote = f"{self.dir}/demos/{game}/"
        subprocess.run(self.ssh + [f"mkdir -p {remote}"], check=True, timeout=30)
        subprocess.run(["rsync", "-az", "-e", self.rsh, local_dir.rstrip("/"), f"{self.host}:{remote}"], check=True)
        return f"demos/{game}"          # relative to the server's --root

    def close(self) -> None:
        super().close()
        subprocess.run(self.ssh + [f"pkill -f '[e]ngine.server --port {SERVER_PORT}'"], timeout=30)


class ModalBackend:
    kind = "modal"

    def __init__(self):
        import modal
        self.modal = modal
        self.srv = modal.Cls.from_name("baconhead", "Serve")()
        self.jobs: dict[str, object] = {}

    def wait(self, timeout=0):
        return {"ok": True, "device": "modal"}

    def generate(self, imgs, adapter):
        return self.srv.generate.remote([_jpeg(im) for im in imgs], f"/models/{adapter}" if adapter else None)

    def adapters(self) -> list[str]:
        from train.modal_app import vol
        try:
            return [os.path.basename(e.path) for e in vol.listdir("/models") if os.path.basename(e.path) != "base"]
        except Exception:
            return []

    def train(self, game: str, local_dir: str) -> str:
        from train.modal_app import vol
        from train.runlog import _git
        with vol.batch_upload(force=True) as b:
            b.put_directory(local_dir, f"/demos/{game}/{os.path.basename(local_dir.rstrip('/'))}")
        call = self.modal.Function.from_name("baconhead", "finetune").spawn(game, [], _git())
        self.jobs[call.object_id] = call
        return call.object_id

    def job(self, job: str) -> dict:
        try:
            self.jobs[job].get(timeout=0)
            return {"state": "done", "log": ""}
        except TimeoutError:
            return {"state": "running", "log": ""}
        except Exception as e:
            return {"state": "failed", "log": repr(e)}

    def forget(self, game: str) -> None:
        from train.modal_app import vol
        vol.remove_file(f"/models/{game}", recursive=True)

    def close(self):
        pass


def connect(cfg: dict):
    kind = cfg.get("backend", "local")
    if kind == "ssh":
        return SSHBackend(**{k: v for k, v in cfg.items() if k != "backend"})
    if kind == "modal":
        return ModalBackend()
    return local()


class BackendRunner(VLMRunner):
    """VLMRunner whose generate step runs on the chosen backend with the current game's adapter. It runs in the
    background, so the game keeps getting inputs while the next chunk is being generated."""

    def __init__(self, backend, adapter: str | None):
        self.backend, self.adapter, self.greedy, self.background = backend, adapter, True, True
        self.reset()

    def generate(self, imgs):
        return self.backend.generate(imgs, self.adapter)


def _jpeg(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()
