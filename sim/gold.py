"""Gold solutions: one fixed level per game, a concrete goal, and a hand-written plan that reaches it.

    python -m sim.gold --check            # replay every results/gold/<game>.json and assert its goal
    python -m sim.gold --make obby        # re-run a plan and rewrite its gold file (+ an mp4 with --video)

The plan for each game is written for that level: which platform, prompt, button or enemy, in what order. Moves are
built from a few helpers (walk to a point, hold a key, click a spot, wait), and the few timing-critical moves (an
obby jump onto a fading part) are found by trying candidates from a savestate, like a TAS. The scripted experts are
not used. Every tick is round-tripped through the action-text codec before the game sees it, so a gold file is
exactly what a model would have to output, and replaying it reproduces the run tick for tick.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle

import numpy as np

import sim
from core.actions import Action
from model.vlm import decode, encode_tick

SEED = 999_000          # the first held-out eval seed: nothing was trained on it
OUT = "results/gold"


class Run:
    """An env plus the inputs played so far. Helpers append ticks; snapshot()/restore() are savestates."""

    def __init__(self, game: str, kw: dict):
        self.game, self.kw = game, kw
        self.env = sim.make(game, seed=SEED, max_steps=10 ** 9, **kw)
        self.texts: list[str] = []
        self.react = None          # plan-specific reflex checked every walking tick (hide from Rush, ...); True = it acted

    def act(self, a: Action) -> None:
        prev = self.env.cursor
        text = encode_tick(a, prev)
        self.env.step(decode(text, prev)[0][0])   # exactly what replaying the text does
        self.texts.append(text)

    def snapshot(self):
        return pickle.dumps((self.env, len(self.texts)))

    def restore(self, snap) -> None:
        self.env, n = pickle.loads(snap)
        del self.texts[n:]

    # ── moves ──
    def wait(self, n: int) -> None:
        for _ in range(n):
            self.act(Action(cursor=self.env.cursor))

    def hold(self, keys, n: int) -> None:
        for _ in range(n):
            self.act(Action(keys=frozenset(keys), cursor=self.env.cursor))

    def tap(self, key: str) -> None:
        self.hold({key}, 1)
        self.wait(1)

    def go(self, x: float, z: float, stop: float = 1.2, limit: int = 400, jump: bool = False) -> bool:
        """Walk to (x, z): turn the camera toward it and hold W, like a player."""
        e = self.env
        for _ in range(limit):
            e = self.env
            p = e.player.pos
            if math.hypot(x - p[0], z - p[2]) < stop:
                return True
            if self.react and self.react():
                continue
            self.act(e.steer_to(np.array([x, 0.0, z]), jump=jump, stop_dist=stop))
        return False

    def face(self, x: float, z: float, limit: int = 20) -> None:
        """Turn to look at (x, z) without walking."""
        for _ in range(limit):
            a = self.env.steer_to(np.array([x, 0.0, z]), keys=(), stop_dist=0)
            if abs(a.look_x) < 1:
                return
            self.act(a)

    def click(self, u: float, v: float) -> None:
        self.act(Action(cursor=(u, v), click=True))
        self.wait(1)

    def button(self, id_: str) -> bool:
        """Move the cursor onto the on-screen button with this id, then click it (a hover tick, like a player)."""
        for el in self.env.hud() + self.env.ui_layers():
            if el.get("id") == id_:
                x, y, w, h = el["rect"]
                uv = (x + w / 2, y + h / 2)
                self.act(Action(cursor=uv))
                self.click(*uv)
                return True
        return False

    def click_world(self, x: float, y: float, z: float, limit: int = 30) -> bool:
        """Turn toward a spot in the world, tilt the camera until it's on screen, put the cursor on it, click."""
        e = self.env
        for _ in range(limit):
            a = e.steer_to(np.array([x, 0.0, z]), keys=(), stop_dist=99)
            uv = e.click_at_world(np.array([x, y, z]))
            if uv is None or abs(a.look_x) > 3:
                a.look_y = -6.0 if uv is None else 0.0
                self.act(a)
                continue
            if abs(uv[0] - e.cursor[0]) < 0.01 and abs(uv[1] - e.cursor[1]) < 0.02:
                self.click(*e.cursor)
                return True
            self.act(Action(cursor=uv))
        return False

    def prompt(self, fn: str, limit: int = 80) -> bool:
        """Hold E on the prompt in front of you until the one that runs `fn` fires."""
        for _ in range(limit):
            pr = self.env.active_prompt()
            if pr is None or pr.on_trigger[0] != fn:
                return False
            self.hold({"e"}, 1)
            if self.env.active_prompt() is not pr or pr.progress < 0:
                self.wait(1)
                return True
        return False


# ── one plan per game ──

def plan_obby(r: Run) -> None:
    """Hop from platform to platform along this course. Each hop is timed from a savestate: wait 0-4 s, then walk at
    the next platform jumping on tick j, and keep it if it lands on top and is still standing 0.3 s later. A chain of
    fading parts can be a trap (land just before yours fades while the next is still gone), so when a hop has no
    timing that works the plan backs up and takes the previous hop later, like a player waiting on the checkpoint."""
    def options(k: int):
        w = r.env.world
        fading = lambda part: getattr(w.motions.get(part), "kind", None) == "vanish"
        moving = r.env.waypoints[k][0] in w.motions or bool(r.env._near_kill())
        # wait only where it's safe to stand (not on a fading part), and only if the next part moves: then up to a full
        # fade/slide cycle (6 s). On a fading chain you hop straight on, choosing just the jump tick - that's how you
        # play it, and it keeps backtracking to "wait longer on the last safe platform" instead of a blow-up.
        waits = range(0, 61, 2) if moving and not fading(r.env.waypoints[k - 1][0]) else (0,)
        for wait in waits:
            for j in (None, "hop", 0, 1, 2, 3, 4, 5, 6, 7, 8):   # "hop": hold Space, bouncing over a spinner's bar
                yield wait, j

    def hop(k: int, wait: int, j) -> bool:
        e = r.env
        part = e.waypoints[k][0]
        r.wait(wait)
        for t in range(40):
            r.act(e.steer_to(e.wp_point(k), jump=(j == "hop" or (j is not None and t == j)), stop_dist=0.3))
            if not e.player.alive:
                return False
            near = np.linalg.norm((e.wp_point(k) - e.player.pos)[[0, 2]]) < 2.5   # a spinner disc holds 3 waypoints:
            if e.player.grounded and e.player.ground == part and e._on_top(e.player, part) and near:   # reach each one
                r.wait(3)       # still there 0.3 s later. (1 s made a fading chain impossible: at ~1.6 s per hop
                return e.player.alive and e.player.ground == part   # no start time works; at ~0.7 s a 1.6 s window does)
        return False

    n = len(r.env.waypoints)
    stack = [(r.snapshot(), options(1))]         # stack[i]: state before hop i+1 and its untried timings
    deepest = 0
    while len(stack) < n:
        k = len(stack)
        snap, opts = stack[-1]
        for wait, j in opts:
            r.restore(snap)
            if hop(k, wait, j):
                stack.append((r.snapshot(), options(k + 1) if k + 1 < n else iter(())))
                if k > deepest:
                    deepest = k
                    print(f"  obby: waypoint {k}/{n - 1} (stage {r.env.stage}) at t={r.env.t:.1f}s", flush=True)
                break
        else:
            stack.pop()
            if len(stack) <= 1 or deepest - len(stack) > 8:
                raise RuntimeError(f"obby: no timing gets past waypoint {deepest + 1} — make this part easier")


def plan_doors(r: Run) -> None:
    """This hotel: rooms 0-2 straight through; door 4 (room 3, dark) is locked, its key is in the dresser by the far
    wall at (32.5, 45); door 8 (room 7) is locked, key in the dresser at (117, 75.5); then on to room 10. Walk entry →
    room centre → exit door → next room. Reflexes on the way: when the lights flicker (Rush), hide in the nearest
    wardrobe and come out once it has passed; when Screech says "Psst!", turn and look at it."""
    def react() -> bool:
        e = r.env
        if e.screech_pos is not None:
            r.face(e.screech_pos[0], e.screech_pos[2], limit=6)
            return True
        if (e.rush_warn > 0 or e.rush_on) and e.hidden < 0:
            room = [c for c in e.closets if c["room"] == e.room]
            c = min(room, key=lambda c: np.linalg.norm((c["front"] - e.player.pos)[[0, 2]]))
            r.react = None                          # walking to the wardrobe mustn't re-trigger this
            r.go(c["front"][0], c["front"][2], stop=0.8)
            r.prompt("_hide")
            while e.rush_warn > 0 or e.rush_on:
                r.wait(1)
            r.wait(5)
            r.prompt("_unhide")
            r.react = react
            return True
        return False

    def through(num: int) -> None:
        e = r.env
        d = e.doors[num]
        if d["locked"]:
            k = e.keys[num]
            r.go(k["front"][0], k["front"][2], stop=0.8)
            r.prompt("_open")
            r.prompt("_take_key")
            u = d["center"] - np.r_[d["n"][0], 0, d["n"][1]] * 2.0
            r.go(u[0], u[2], stop=1.0)
            r.prompt("_unlock")
        c = d["center"]
        n = d["n"]
        r.go(c[0] - n[0] * 3, c[2] - n[1] * 3, stop=1.0)        # line up in front of the door
        nxt = e.rooms[num]["c"]
        r.go(nxt[0], nxt[1], stop=3.0)                          # through it, to the next room's centre

    r.react = react
    for num in range(1, 11):
        r.go(*r.env.rooms[num - 1]["c"], stop=3.0)
        through(num)


def plan_rivals(r: Run) -> None:
    """Round 1: the bot spawns 88 studs east behind cover. Walk east until it's in sight and within ~35 studs, then
    stand and fight: put the crosshair on its chest (head when close), hold fire only while on target, strafe A/D
    every 0.6 s so its shots miss, reload on empty, and push forward again if it hides."""
    e = r.env
    start_wins = e.wins
    for t in range(1500):
        if e.wins > start_wins:
            return
        p, b = e.player, e.bot
        if not p.alive or not b.alive or e.round_over_t is not None:
            r.wait(1)
            continue
        eye = e._eye()
        flat = float(np.linalg.norm((b.pos - p.pos)[[0, 2]]))
        aim = b.pos + np.array([0.0, 4.6 if flat < 18 else 2.8, 0.0])
        rel = aim - eye
        yaw = math.degrees((math.atan2(rel[0], rel[2]) - e.cam.yaw + math.pi) % (2 * math.pi) - math.pi)
        pitch = math.degrees(-math.asin(float(np.clip(rel[1] / np.linalg.norm(rel), -1, 1))) - e.cam.pitch)
        a = Action(look_x=float(np.clip(yaw, -30, 30)), look_y=float(np.clip(pitch, -10, 10)), cursor=(0.5, 0.5))
        keys = set()
        seen = e._los(eye, aim)
        if e.ammo == 0:
            keys.add("r")
        if seen and flat < 35:
            keys.add("a" if (t // 6) % 2 == 0 else "d")
            if abs(yaw) < 4.0 and abs(pitch) < 5.0 and e.ammo > 0:
                keys.add("lmb")
        else:
            keys.add("w")
        a.keys = frozenset(keys)
        r.act(a)


def plan_mm2(r: Run) -> None:
    """You're the Sheriff (pinned). Nobody's role is visible, so: wait out the lobby, take the gun out (1), keep near
    the other players (the Murderer hunts whoever's alone), and the moment someone has a knife out where you can
    see them, turn to them, put the cursor on their chest and click. Shooting an innocent would kill you too,
    so only the knife counts as proof."""
    e = r.env
    for t in range(3000):
        if e.rounds:
            return
        p = e.player
        if e.phase != "round" or not p.alive:
            r.wait(1)
            continue
        if e.equipped != "gun":
            r.tap("1")
            continue
        armed = [b for b in e.bots if b.h.alive and b.armed and b.role == "murderer" and e._los(p, b.h)]   # the knife is visible
        if armed:
            m = armed[0].h
            chest = m.pos + [0, 2.5, 0]
            a = e.steer_to(m.pos, keys=(), stop_dist=99)
            uv = e.cam.project(chest, e.aspect)
            if uv is not None and 0.05 < uv[0] < 0.95 and 0.1 < uv[1] < 0.85:
                on = abs(uv[0] - e.cursor[0]) < 0.03 and abs(uv[1] - e.cursor[1]) < 0.04
                a.cursor, a.click = (float(uv[0]), float(uv[1])), bool(on and e.gun_cd <= 0)
            r.act(a)
            continue
        others = [b.h.pos for b in e.bots if b.h.alive]
        near = min(others, key=lambda q: np.linalg.norm((q - p.pos)[[0, 2]]))
        r.act(e.steer_to(near, stop_dist=6.0))


def plan_disaster(r: Run) -> None:
    """Round 1 here is Acid Rain (t=15 s), and you spawn at (7, 19), next to the building at (-5, 5) whose door faces
    you. Wait for the warning, walk to the doorway (-5, 16.9), step in (-5, 9.9), stand in the middle of the room
    under the roof until the round is over."""
    e = r.env
    while e.disaster is None:
        r.wait(1)
    r.go(-5, 16.9, stop=1.0)
    r.go(-5, 9.9, stop=1.0)
    r.go(-5, 5, stop=1.5)
    while e.phase == "disaster":
        r.wait(1)
    r.wait(5)


def plan_forest(r: Run) -> None:
    """Night 1 starts at t=60 and lasts 45 s; the fire has 70 fuel and burns 1/s, so it needs 2+ logs (22 each) before
    dark. Axe out (1); chop the tree 25 studs north at (0.9, 29.3) (4 swings), pick up its log; go south of the two
    rocks at (-8, 24-28) via (-4, 19) and (-14, 17) to the tree at (-31.8, 11.5), chop, pick up; carry both to the
    fire and feed it; one more from (17.7, -36.7) if there's daylight left; then stand in the firelight until dawn."""
    e = r.env
    r.tap("1")

    def chop(x, z, via=()):
        for vx, vz in via:
            r.go(vx, vz, stop=1.5)
        r.go(x, z, stop=3.2)
        r.face(x, z)
        for _ in range(30):
            if e.logs:
                break
            r.hold({"lmb"}, 1)
        lid = min(e.logs, key=lambda i: np.hypot(*(e.logs[i] - e.player.pos)[[0, 2]]))
        r.go(e.logs[lid][0], e.logs[lid][2], stop=1.5)
        r.prompt("_pick_log")

    def feed():
        r.go(0, 0, stop=2.5)
        r.prompt("_add_fuel")

    chop(0.9, 29.3)
    chop(-31.8, 11.5, via=((-4, 19), (-14, 17)))
    feed()
    if e.t < 40:
        chop(17.7, -36.7)
        feed()
    r.go(2.5, 2.5, stop=1.0)
    while e.nights < 1:
        r.wait(1)


def plan_fisch(r: Run) -> None:
    """Onto the pier through its base (0, -4) - it's a plank over water - out to the end (0, 28), face the open
    water, rod out (1). Cast: hold the mouse while the meter rises, let go near the top. Wait for the bite. Shake:
    click the Shake button wherever it jumps. Reel: hold to push the white zone right, let go to let it drift left,
    keeping it over the fish. Then back off the pier the same way to the Merchant (9, -6): E, "Sell All"."""
    from sim.fisch import CHARGE_PERIOD, RODS
    e = r.env
    r.go(0, -4, stop=0.8)
    r.go(0, 28, stop=0.8)
    r.face(0, 60)
    r.tap("1")
    while e.phase in ("idle", "charge"):
        u = ((e.t - e.charge_t) / CHARGE_PERIOD) % 1.0 if e.phase == "charge" else 0.0
        r.act(Action(keys=frozenset() if (e.phase == "charge" and e.power >= 0.9 and u < 0.5) else {"lmb"}, cursor=(0.5, 0.45)))
    while e.phase == "wait":
        r.wait(1)
    while e.phase == "shake":
        x, y, w, h = e.shake_rect
        uv = (x + w / 2, y + h / 2)
        on = abs(e.cursor[0] - uv[0]) < 0.012 and abs(e.cursor[1] - uv[1]) < 0.02
        r.act(Action(cursor=uv, click=on))
    while e.phase == "reel":
        centre = e.zone + RODS[e.rod][2] / 2
        r.act(Action(keys=frozenset({"lmb"}) if e.fpos - centre - 0.3 * e.zone_v + 0.02 > 0 else frozenset(), cursor=(0.5, 0.45)))
    r.wait(3)
    r.go(0, -4, stop=0.8)
    r.go(9, -2.5, stop=1.0)
    r.prompt("_open")
    r.wait(2)
    r.button("sell_all")
    r.wait(2)


def plan_garden(r: Run) -> None:
    """20 Sheckles to start. Talk to Sam at (-22, -24), buy two Carrot seeds (10 each), close. Carrot is hotbar slot 0:
    equip it, walk to the front of the plot (0, -3), click soil cells (2, 0) at (-2, 2) and (3, 0) at (2, 2). Carrots
    take 15 s: wait, then walk to each one and Collect. Sell both to Steven at (22, -24): 2 x 18 = 36 > 20."""
    e = r.env
    r.go(-22, -25.5, stop=1.5)
    r.prompt("_open")
    r.wait(2)
    r.button("buy:Carrot")
    r.button("buy:Carrot")
    r.button("close")
    r.button("slot:0")
    r.go(0, -3, stop=1.0)
    for x, z in ((-2, 2), (2, 2)):
        r.click_world(x, 0.3, z)
        r.wait(2)
    while not any(pl.age >= f.ripe_at for pl in e.plants.values() for f in pl.fruits):
        r.wait(1)
    r.wait(10)
    for _ in range(2):
        fruits = [f for pl in e.plants.values() for f in pl.fruits if pl.age >= f.ripe_at]
        if not fruits:
            break
        f = min(fruits, key=lambda f: np.linalg.norm((f.pos - e.player.pos)[[0, 2]]))
        r.go(f.pos[0], f.pos[2], stop=1.5)
        r.prompt("_collect")
    r.go(22, -25.5, stop=1.5)
    r.prompt("_sell")
    r.wait(2)


def plan_adoptme(r: Run) -> None:
    """The pet's first need here is "hungry" (t=4 s). Map → teleport to the Pizza Shop, click the counter, pick an
    Apple (2 Bucks) → Buy, Backpack → Apple to hold it, click the pet → "Use", and stay while it eats. Pop-ups
    (daily reward, offers) get closed whenever they show up, since they block every other click."""
    e = r.env

    def popups():
        while e.ui in ("daily", "offer"):
            r.button("claim" if e.ui == "daily" else "close")

    while "hungry" not in e.needs:
        r.wait(1)
    popups()
    r.button("map")
    r.button("tp:Pizza Shop")
    popups()
    c = e.world.c[e.pizza]
    r.go(c[0], c[2], stop=6.0)
    r.click_world(c[0], c[1], c[2])
    popups()
    r.button("shopitem:apple")
    r.button("buy")
    popups()
    r.button("backpack")
    r.button("item:apple")
    popups()
    q = e.pet_pos
    r.click_world(q[0], q[1] + 1.2, q[2])
    r.button("use")
    start = e.pet_xp
    for _ in range(80):
        if e.pet_xp > start:
            break
        popups()
        r.wait(1)


def plan_brainrot(r: Run) -> None:
    """$100 to start. Watch the red carpet for the first brainrot you can afford, walk alongside it and hold E
    ("Purchase", 0.5 s); it walks itself home to a slot in your base and starts earning. Wait ~10 s, then step on
    that slot's green pad to collect the cash."""
    from sim.brainrot import KINDS
    e = r.env
    bought = None
    for _ in range(600):
        mine = [b for b in e.brainrots.values() if b.owner == 0]
        if mine:
            bought = mine[0]
            break
        ok = [b for b in e.brainrots.values() if b.state == "conveyor" and KINDS[b.kind][2] <= e.cash]
        if not ok:
            r.wait(1)
            continue
        b = min(ok, key=lambda b: np.linalg.norm((b.pos - e.player.pos)[[0, 2]]))
        a = e.steer_to(b.pos + [1.5, 0, -2.5], stop_dist=1.0)
        pr = e.active_prompt()
        if pr is not None and pr.on_trigger[:2] == ("_buy", b.id):
            a.keys = a.keys | {"e"}
        r.act(a)
    while bought.state != "slot":
        r.wait(1)
    r.wait(100)
    pad = e.world.c[e.plots[0]["pads"][bought.slot]]
    r.go(pad[0], pad[2], stop=0.4)
    r.wait(3)


PLANS = {
    "obby": (sim.ENVS["obby"].eval_kw, "reach the final checkpoint (all stages)",
             lambda e: e.best_stage >= e.n_stages, plan_obby),
    "doors": ({}, "reach room 10 alive", lambda e: e.best_room >= 10 and e.player.alive, plan_doors),
    "rivals": ({}, "win the first round without dying", lambda e: e.wins >= 1 and e.losses == 0, plan_rivals),
    "mm2": ({"role": "sheriff"}, "as Sheriff, shoot the Murderer and win the first round",
            lambda e: e.rounds[:1] == [1.0] and e.deaths == 0, plan_mm2),
    "disaster": ({}, "survive the first disaster (Acid Rain)", lambda e: e.wins >= 1 and e.deaths == 0, plan_disaster),
    "forest": ({}, "survive night 1", lambda e: e.nights >= 1 and e.deaths == 0, plan_forest),
    "fisch": ({}, "catch a fish and sell it", lambda e: e.caught >= 1 and e.money > 0, plan_fisch),
    "garden": ({}, "grow carrots and sell them for more than you started with (20)", lambda e: e.sheckles > 20, plan_garden),
    "adoptme": ({}, "feed the hungry pet (one task done)", lambda e: e.pet_xp >= 1, plan_adoptme),
    "brainrot": ({}, "buy a brainrot and collect its earnings (net worth above the starting $100)",
                 lambda e: any(b.owner == 0 for b in e.brainrots.values()) and e._networth() > 100, plan_brainrot),
}


def make(game: str, video: bool = False) -> dict:
    kw, goal, check, plan = PLANS[game]
    r = Run(game, kw)
    plan(r)
    ok = bool(check(r.env))
    out = {"game": game, "seed": SEED, "kw": kw, "goal": goal, "ticks": len(r.texts), "score": float(r.env.score()),
           "reached": ok, "inputs": r.texts}
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, game + ".json"), "w"))
    if video:
        render_video(game, r.texts, kw)
    return out


def replay(gold: dict):
    env = sim.make(gold["game"], seed=gold["seed"], max_steps=10 ** 9, **gold["kw"])
    for text in gold["inputs"]:
        env.step(decode(text, env.cursor)[0][0])
    return env


def render_video(game: str, texts: list[str], kw: dict, out: str = OUT, seed: int | None = None, name: str | None = None) -> str:
    import subprocess
    from model.vlm import OBS_H, OBS_W
    from sim.render import best_device, render
    env, dev = sim.make(game, seed=SEED if seed is None else seed, max_steps=10 ** 9, **kw), best_device()
    path = os.path.join(out, (name or game) + ".mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OBS_W}x{OBS_H}",
                           "-r", "10", "-i", "-", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    for text in texts:
        ff.stdin.write(render([env.scene()], OBS_H, OBS_W, device=dev)[0].cpu().numpy().tobytes())
        env.step(decode(text, env.cursor)[0][0])
    ff.stdin.close()
    ff.wait()
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make", default=None, help="comma-separated games, or 'all'")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--video", action="store_true")
    args = ap.parse_args()
    if args.make:
        for g in (PLANS if args.make == "all" else args.make.split(",")):
            o = make(g, args.video)
            print(f"{g:9s} {'REACHED' if o['reached'] else 'FAILED '}  {o['goal']}  in {o['ticks']} ticks ({o['ticks'] / 10:.0f} s), score {o['score']:.2f}")
    if args.check:
        bad = 0
        for g, (kw, goal, check, _) in PLANS.items():
            path = os.path.join(OUT, g + ".json")
            if not os.path.exists(path):
                print(f"{g:9s} MISSING")
                bad += 1
                continue
            ok = bool(check(replay(json.load(open(path)))))
            bad += not ok
            print(f"{g:9s} {'ok    ' if ok else 'FAILED'}  {goal}")
        raise SystemExit(bad)


if __name__ == "__main__":
    main()
