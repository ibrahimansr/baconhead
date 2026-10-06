"""The local engine: run it, play Roblox, it handles the rest. Compute comes from ~/.baconhead/config.json.

    .venv/bin/python run.py                         # this Mac
    .venv/bin/python run.py --backend ssh --host me@gpu-box --dir ~/baconhead
    .venv/bin/python run.py --teach 5 --drive       # teach the next game you join, then let it play

Type into its terminal while it runs:  teach [minutes] · drive on|off · forget · check · status · quit

What it does on its own:
  Roblox home screen / closed   nothing runs, no adapter, the model never touches input
  join a game it was taught     that game's LoRA goes on; leaving takes it off
  join a new game               it watches the first minute of your play and decides whether the base model
                                already plays like you (engine/check.py) or whether you should teach it
  teach (command / --teach)     records N minutes of your in-game play, trains the LoRA on the chosen backend,
                                then a pop-up says it's ready
  AI play on                    after --idle seconds without your input the model drives; touch anything to take over
macOS permissions for whoever launches it: Screen Recording, Input Monitoring, Accessibility.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import threading
import time

from core.actions import TICK_S
from core.idle import HumanIdle
from core.window import focus_roblox
from engine import backend as B
from engine import check
from engine.roblox import RobloxWatcher
from recorder.episode import EpisodeWriter, next_run_dir
from recorder.events import InputRecorder
from recorder.frames import FrameSource

CHECK_S = 60.0                                  # seconds of your play the check watches
GAMES = os.path.expanduser("~/.baconhead/games.json")


def notify(msg: str, dialog: bool = False) -> None:
    msg = msg.replace('"', "'")
    script = (f'display dialog "{msg}" with title "baconhead" buttons {{"OK"}} default button "OK"' if dialog
              else f'display notification "{msg}" with title "baconhead"')
    threading.Thread(target=subprocess.run, args=(["osascript", "-e", script],), kwargs={"capture_output": True}, daemon=True).start()


class Engine:
    def __init__(self, cfg: dict, idle_s: float):
        self.cfg, self.idle_s = cfg, idle_s
        self.watch = RobloxWatcher()
        self.idle = HumanIdle()
        self.rec = InputRecorder(on_event=self.idle.observe)
        self.perm_error = None
        try:
            self.rec.start()
        except PermissionError:   # surfaced in the app instead of crashing the engine
            self.perm_error = "Grant Input Monitoring (System Settings → Privacy & Security), then restart baconhead"
        self.lock = threading.RLock()
        self.games = _load(GAMES)
        self.backend, self.trained = None, set()
        self.bstate = {"kind": cfg.get("backend", "local"), "status": "connecting", "error": None, "device": None}
        self.roblox = self.watch.poll()
        self.adapter = None
        self.mode = "idle"                       # idle | watching | checking | teaching | training
        self.drive, self.driving = False, False
        self.message = "Waiting for Roblox"
        self.src = self.ctl = self.runner = None
        self.writer, self.writer_dir, self.written = None, None, 0
        self.rec_s, self.rec_goal = 0.0, 0.0
        self.pending_teach = None
        self._polled = 0.0
        self._closed = False
        threading.Thread(target=self._connect, daemon=True).start()

    # ── compute ──
    def _connect(self):
        with self.lock:
            old, self.backend = self.backend, None
            self.bstate = {"kind": self.cfg.get("backend", "local"), "status": "connecting", "error": None, "device": None}
        if old:
            old.close()
        try:
            b = self._starting = B.connect(self.cfg)   # tracked so quitting mid-startup still shuts the server down
            if self._closed:
                b.close()
                return
            h = b.wait()
            trained = set(b.adapters())
            with self.lock:
                self.backend, self.trained = b, trained
                self.bstate.update(status="ready", device=h.get("device"))
                self._enter(self.roblox, force=True)
        except Exception as e:
            with self.lock:
                self.bstate.update(status="error", error=str(e)[:300])

    def set_backend(self, cfg: dict):
        self.cfg = cfg
        B.save_config(cfg)
        threading.Thread(target=self._connect, daemon=True).start()

    # ── roblox state ──
    def _enter(self, st, force=False):
        """Called when the Roblox state changes: put the right adapter on (or none) and decide what to do."""
        prev = self.roblox
        self.roblox = st
        if st.status != "in_game":
            if self.driving:
                self._stop_driving()
            self.adapter = None
            if self.mode == "watching":
                self._drop_recording()
                self.mode = "idle"
            self.message = "Roblox is closed" if st.status == "closed" else "On the Roblox home screen" if st.status == "home" else "Joining a game…"
            return
        if not force and prev.status == "in_game" and prev.game_id == st.game_id:
            return
        g = st.game_id
        name = st.name or g
        self.adapter = g if g in self.trained else None
        self.src = FrameSource.for_roblox(wait_s=5.0)
        from control.controller import Controller
        self.ctl = Controller(self.src.region) if self.src.region else None
        info = self.games.setdefault(g, {"name": name})
        info["name"] = name
        _save(GAMES, self.games)
        if self.pending_teach:
            self.start_teach(self.pending_teach)
            self.pending_teach = None
        elif self.adapter:
            self.message = f"{name}: using the model you taught"
        elif info.get("verdict") == "ok":
            self.message = f"{name}: the base model already plays like you"
        elif info.get("verdict") == "teach":
            self.message = f"{name}: teach it for better play ({info.get('reason', '')})"
        elif self.backend and self.mode == "idle":
            self._start_recording(CHECK_S, keep=False)
            self.mode = "watching"
            self.message = f"{name}: watching you for {CHECK_S:.0f}s to see if it needs teaching"
        else:
            self.message = f"{name}: base model"

    # ── recording ──
    def _start_recording(self, seconds: float, keep: bool):
        g = self.roblox.game_id
        d = next_run_dir(os.path.join("demos", g)) if keep else tempfile.mkdtemp(prefix="check_")
        self.writer, self.writer_dir = EpisodeWriter(out_dir=d, region=self.src.region), d
        self.rec.clear()   # each recording starts a fresh event buffer, so indices never drift
        self.written = 0
        self.rec_s, self.rec_goal = 0.0, seconds

    def _drop_recording(self):
        if self.writer:
            self.writer.close()
        self.writer = None

    def _record_tick(self, f):
        self.writer.add_frame(f.rgb, f.t_ns)
        snap = self.rec.snapshot()
        if len(snap) > self.written:
            self.writer.add_events(snap[self.written:])
            self.written = len(snap)
        self.rec_s += TICK_S
        if self.rec_s >= self.rec_goal:
            path = self.writer.close(extra_meta={"kind": "human_demo", "game": self.roblox.game_id})
            self.writer = None
            if self.mode == "watching":
                self.mode = "checking"
                self.message = f"{self.roblox.name}: checking whether it needs teaching…"
                threading.Thread(target=self._check, args=(self.roblox.game_id, path), daemon=True).start()
            elif self.mode == "teaching":
                self.mode = "training"
                self.message = f"{self.roblox.name}: training on your {self.rec_goal / 60:g} minutes of play…"
                threading.Thread(target=self._train, args=(self.roblox.game_id, path), daemon=True).start()

    def _check(self, g, path):
        try:
            res = check.run(path, lambda imgs: self.backend.generate(imgs, None))
        except Exception as e:
            res = {"verdict": "unknown", "reason": f"check failed: {e!r}"[:200]}
        with self.lock:
            self.games.setdefault(g, {}).update(res)
            _save(GAMES, self.games)
            self.mode = "idle"
            name = self.games[g].get("name", g)
            self.message = f"{name}: {res['reason']}"
            notify(self.message)

    def start_teach(self, minutes: float):
        if self.roblox.status != "in_game":
            self.pending_teach = minutes
            self.message = f"Join a game and it'll start recording {minutes:g} minutes of your play"
            return
        if self.mode in ("teaching", "training"):
            return
        if self.writer:
            self._drop_recording()
        self._start_recording(minutes * 60, keep=True)
        self.mode = "teaching"
        self.message = f"{self.roblox.name}: recording your play for {minutes:g} minutes. Just play normally."

    def _train(self, g, path):
        try:
            job = self.backend.train(g, os.path.dirname(path))
            while (s := self.backend.job(job))["state"] == "running":
                time.sleep(5)
            ok = s["state"] == "done"
        except Exception as e:
            ok, s = False, {"log": repr(e)}
        with self.lock:
            self.mode = "idle"
            name = self.games.get(g, {}).get("name", g)
            if ok:
                self.trained.add(g)
                self.games.setdefault(g, {}).update(verdict="taught")
                _save(GAMES, self.games)
                if self.roblox.game_id == g:
                    self.adapter = g
                self.message = f"{name}: your model is trained. Type `drive on` to let it play."
                notify(f"Your model for {name} is trained. Type drive on in the baconhead terminal to let it play.", dialog=True)
            else:
                self.message = f"{name}: training failed: {s.get('log', '')[-200:]}"
                notify(f"Training failed for {name}. See logs/.")

    def forget(self):
        g = self.roblox.game_id
        if g and self.backend:
            self.backend.forget(g)
            self.trained.discard(g)
            self.adapter = None
            self.games.get(g, {}).pop("verdict", None)
            _save(GAMES, self.games)
            self.message = f"{self.roblox.name}: back to the base model"

    # ── driving ──
    def _stop_driving(self):
        self.driving = False
        if self.ctl:
            self.ctl.release()

    def tick(self):
        with self.lock:
            now = time.monotonic()
            st = self.roblox
            if now - self._polled >= 1.0:   # the log only needs reading once a second
                st, self._polled = self.watch.poll(), now
            if (st.status, st.game_id) != (self.roblox.status, self.roblox.game_id):
                self._enter(st)
            in_game = self.roblox.status == "in_game" and self.src is not None and self.src.region is not None
            want_drive = (in_game and self.drive and self.backend is not None and self.ctl is not None
                          and self.mode not in ("teaching",) and self.idle.seconds() >= self.idle_s)
            if not (in_game and (self.writer or want_drive)):
                if self.driving and not want_drive:
                    self._stop_driving()
                return
            f = self.src.grab()
            if self.writer and not self.driving:
                self._record_tick(f)
            if want_drive:
                if not self.driving:
                    self.driving = True
                    focus_roblox()   # synthetic keys go to whichever app is in front
                    self.runner = B.BackendRunner(self.backend, self.adapter)
                self.runner.adapter = self.adapter
                a = self.runner.act(f.rgb)
            elif self.driving:
                self._stop_driving()
                return
            else:
                return
        self.ctl.apply(a)   # outside the lock: the camera drag takes most of the tick

    def state(self) -> dict:
        with self.lock:
            g = self.roblox.game_id
            return {"roblox": {"status": self.roblox.status, "game_id": g, "name": self.roblox.name},
                    "backend": self.bstate, "adapter": self.adapter, "trained": g in self.trained if g else False,
                    "game": self.games.get(g, {}) if g else {}, "mode": self.mode, "drive": self.drive, "driving": self.driving,
                    "message": self.perm_error or self.message,
                    "recording": {"seconds": round(self.rec_s), "goal": round(self.rec_goal)} if self.writer else None,
                    "config": {k: v for k, v in self.cfg.items()}}

    def close(self):
        self._closed = True
        self._stop_driving()
        for b in {id(x): x for x in (self.backend, getattr(self, "_starting", None)) if x}.values():
            b.close()
        self.rec.stop()


def _load(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return {}


def _save(p, obj):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(obj, open(p, "w"), indent=2)


def commands(eng: Engine, stop: threading.Event):
    """Read commands typed into the engine's terminal."""
    def loop():
        while not stop.is_set():
            try:
                parts = input().strip().lower().split()
            except EOFError:   # no terminal attached (e.g. launched in the background): automatic behaviour only
                return
            if not parts:
                continue
            cmd, arg = parts[0], parts[1:]
            with eng.lock:
                if cmd == "teach":
                    eng.start_teach(float(arg[0]) if arg else 5.0)
                elif cmd == "drive":
                    eng.drive = not arg or arg[0] in ("on", "1", "yes")
                    print(f"AI play {'on' if eng.drive else 'off'}", flush=True)
                elif cmd == "forget":
                    eng.forget()
                elif cmd == "check":
                    if eng.roblox.game_id:
                        eng.games.get(eng.roblox.game_id, {}).pop("verdict", None)
                        eng._enter(eng.roblox, force=True)
                elif cmd == "status":
                    print(json.dumps(eng.state(), indent=2), flush=True)
                elif cmd == "quit":
                    stop.set()
                else:
                    print("commands: teach [minutes] · drive on|off · forget · check · status · quit", flush=True)
    threading.Thread(target=loop, daemon=True).start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["local", "ssh", "modal"], default=None)
    ap.add_argument("--host", default=None, help="ssh: user@host")
    ap.add_argument("--port", type=int, default=None, help="ssh port")
    ap.add_argument("--dir", default=None, help="ssh: repo dir on the GPU box")
    ap.add_argument("--teach", type=float, default=None, help="minutes: record and teach the next game you join")
    ap.add_argument("--drive", action="store_true", help="turn AI play on")
    ap.add_argument("--idle", type=float, default=3.0)
    args = ap.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cfg = B.load_config()
    if args.backend:
        cfg = {"backend": args.backend, **({k: v for k, v in (("host", args.host), ("port", args.port), ("dir", args.dir)) if v})}
        B.save_config(cfg)
    eng = Engine(cfg, args.idle)
    eng.drive = args.drive
    if args.teach:
        eng.start_teach(args.teach)
    stop = threading.Event()
    commands(eng, stop)
    print(f"engine running, backend={cfg.get('backend', 'local')}. Commands: teach [minutes] · drive on|off · forget · check · status · quit", flush=True)
    last = None
    try:
        while not stop.is_set():
            t = time.monotonic()
            eng.tick()
            if eng.message != last:
                print(eng.message, flush=True)
                last = eng.message
            time.sleep(max(0.0, TICK_S - (time.monotonic() - t)))
    except KeyboardInterrupt:
        pass
    finally:
        eng.close()


if __name__ == "__main__":
    main()
