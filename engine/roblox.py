"""Is Roblox closed, on the home screen, or in a game, and which one? Read from Roblox's own client log.

The Mac client writes ~/Library/Logs/Roblox/<version>_<time>_Player_<id>_last.log. The lines we key on
(checked against real logs on this machine, client 0.739):
  [FLog::Output] ! Joining game '<jobId>' place <placeId> at <ip>      → joining that place
  [FLog::Network] serverId: <ip>|<port>                                 → in the game
  [FLog::Network] Time to disconnect replication data                   → left the server (teleport or leave)
  [FLog::SingleSurfaceApp] leaveUGCGameInternal                         → back on the home screen
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import urllib.request
from dataclasses import dataclass, field
from functools import lru_cache

LOG_DIR = os.path.expanduser("~/Library/Logs/Roblox")
_JOIN = re.compile(r"! Joining game '[^']*' place (\d+)")


@dataclass
class RobloxState:
    status: str = "closed"          # closed | home | joining | in_game
    place_id: str | None = None
    game_id: str | None = None      # universe id: one game's sub-places (lobby → round) share it, so adapters key on this
    name: str | None = None


@dataclass
class RobloxWatcher:
    log_dir: str = LOG_DIR
    state: RobloxState = field(default_factory=RobloxState)
    _path: str | None = None
    _pos: int = 0

    def poll(self) -> RobloxState:
        if not _running():
            self.state, self._path = RobloxState(), None
            return self.state
        logs = [p for p in glob.glob(os.path.join(self.log_dir, "*_Player_*.log"))]
        if not logs:
            self.state = RobloxState("home")
            return self.state
        newest = max(logs, key=os.path.getmtime)
        if newest != self._path:   # a new client session starts a new log; read it from the top
            self._path, self._pos = newest, 0
            self.state = RobloxState("home")
        with open(newest, "rb") as f:
            f.seek(self._pos)
            chunk = f.read()
            self._pos = f.tell()
        for line in chunk.decode("utf-8", "replace").splitlines():
            self._line(line)
        return self.state

    def _line(self, line: str) -> None:
        m = _JOIN.search(line)
        if m:
            self.state = RobloxState("joining", m.group(1), *game_info(m.group(1)))
        elif "[FLog::Network] serverId:" in line and self.state.place_id:
            self.state = RobloxState("in_game", self.state.place_id, self.state.game_id, self.state.name)
        elif "Time to disconnect replication data" in line or "leaveUGCGameInternal" in line:
            self.state = RobloxState("home")


def _running() -> bool:
    return subprocess.run(["pgrep", "-x", "RobloxPlayer"], capture_output=True).returncode == 0


@lru_cache(maxsize=256)
def game_info(place_id: str) -> tuple[str, str | None]:
    """Public Roblox APIs, no auth: place → (universe id, game name). Offline falls back to the place id."""
    try:
        with urllib.request.urlopen(f"https://apis.roblox.com/universes/v1/places/{place_id}/universe", timeout=3) as r:
            uni = str(json.load(r)["universeId"])
    except Exception:
        return place_id, None
    try:
        with urllib.request.urlopen(f"https://games.roblox.com/v1/games?universeIds={uni}", timeout=3) as r:
            return uni, json.load(r)["data"][0]["name"]
    except Exception:
        return uni, None


if __name__ == "__main__":
    # replay a synthetic log: join → in game → teleport → home
    import tempfile
    d = tempfile.mkdtemp()
    p = os.path.join(d, "0.739_x_Player_1_last.log")
    w = RobloxWatcher(log_dir=d)
    globals()["_running"] = lambda: True
    globals()["game_info"] = lambda pid: (f"u{pid}", f"game{pid}")
    lines = ["[FLog::Output] ! Joining game 'abc' place 123 at 1.2.3.4", "[FLog::Network] serverId: 1.2.3.4|5",
             "[FLog::Network] Time to disconnect replication data: 1.0", "[FLog::Output] ! Joining game 'def' place 456 at 1.2.3.4",
             "[FLog::Network] serverId: 1.2.3.4|6", "[FLog::SingleSurfaceApp] leaveUGCGameInternal"]
    want = ["joining", "in_game", "home", "joining", "in_game", "home"]
    open(p, "w").close()
    for line, st in zip(lines, want):
        with open(p, "a") as f:
            f.write(line + "\n")
        s = w.poll()
        assert s.status == st, (line, s)
    print("watcher ok")
    if os.path.isdir(LOG_DIR):
        print("this Mac right now:", RobloxWatcher().poll())
