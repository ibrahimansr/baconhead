"""Murder Mystery 2: a short lobby, then rounds in a mansion with 8 players and secret roles.

Real-game rules kept: one Murderer (knife, kills on contact + click), one Sheriff (gun, click to shoot at the
cursor), everyone else Innocent. Shooting the Murderer ends the round; a Sheriff who shoots an Innocent dies
too. A dead Sheriff drops the gun, which any Innocent can pick up by touching it (they become the Hero).
Coins spawn around the map and are picked up by touch. Innocents win if the Murderer dies or time runs out;
the Murderer wins by killing everyone else. Your role is shown on a card at the start of each round.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from core.actions import Action
from sim.core import DT, SUBSTEPS, Env, Humanoid, avatar_boxes, hotbar_hud, npc_walk, ray_humanoids

LOBBY_S, ROUND_S, END_S, CARD_S = 6.0, 150.0, 4.0, 3.0
KNIFE_RANGE, KNIFE_CD = 3.8, 1.0
GUN_CD = 1.5
MAX_COINS = 20
NAMES = ["Builderman", "xX_Noob_Xx", "coolkid22", "Pixel_Pro", "mm2_god", "Lunaa", "bloxxer7"]
ROLE_COL = {"innocent": (0.3, 0.85, 0.35), "sheriff": (0.3, 0.55, 1.0), "murderer": (1.0, 0.25, 0.25), "hero": (1.0, 0.8, 0.2)}
XS, ZS = (-39.0, -13.0, 13.0, 39.0), (-30.0, 0.0, 30.0)   # mansion: 3x2 rooms


@dataclass
class Bot:
    h: Humanoid
    name: str
    role: str = "innocent"
    armed: bool = False          # knife or gun out
    goal: np.ndarray = field(default_factory=lambda: np.zeros(3))
    target: int = -1
    react: float = 0.0
    cd: float = 0.0
    coins: int = 0


def _cell(p):
    i = 0 if p[0] < XS[1] else (1 if p[0] < XS[2] else 2)
    return i, (0 if p[2] < ZS[1] else 1)


def _center(i, j):
    return np.array([(XS[i] + XS[i + 1]) / 2, 0.0, (ZS[j] + ZS[j + 1]) / 2])


def nav(p, goal) -> np.ndarray:
    """Next point toward goal through the mansion's doorways (x first, then z)."""
    (i, j), (gi, gj) = _cell(p), _cell(goal)
    if (i, j) == (gi, gj):
        return np.asarray(goal, float)
    if i != gi:
        ni = i + (1 if gi > i else -1)
        door = np.array([XS[max(i, ni)], 0.0, (ZS[j] + ZS[j + 1]) / 2])
    else:
        door = np.array([(XS[i] + XS[i + 1]) / 2, 0.0, ZS[1]])
    return door if np.linalg.norm((door - p)[[0, 2]]) > 1.0 else np.asarray(goal, float)


class MurderMystery2(Env):
    name = "mm2"
    train_steps = 3000
    eval_kw = {}

    def build(self) -> None:
        w, r = self.world, self.rng
        self.max_steps = int(self.kw.get("max_steps", self.train_steps))
        self.respawn_s = 1e9                          # dead players spectate until the next round
        w.add((0, -0.6, 0), (400, 1, 400), (0.35, 0.55, 0.3))                              # grass outside
        w.add((0, -0.5, 0), (80, 1, 62), (0.55, 0.38, 0.25))                               # wooden floor
        wall = (0.85, 0.82, 0.72)
        for z in (ZS[0], ZS[2]):
            w.add((0, 5, z), (80, 10, 1), wall)
        for x in (XS[0], XS[3]):
            w.add((x, 5, 0), (1, 10, 62), wall)
        for x in (XS[1], XS[2]):                       # interior x-walls, a 6-stud doorway in each row
            for j in range(2):
                zc = (ZS[j] + ZS[j + 1]) / 2
                for z0, z1 in ((ZS[j], zc - 3), (zc + 3, ZS[j + 1])):
                    w.add((x, 5, (z0 + z1) / 2), (1, 10, z1 - z0), wall)
        for i in range(3):                             # the z=0 wall, a doorway in each column
            xc = (XS[i] + XS[i + 1]) / 2
            for x0, x1 in ((XS[i], xc - 3), (xc + 3, XS[i + 1])):
                w.add(((x0 + x1) / 2, 5, 0), (x1 - x0, 10, 1), wall)
        self.furniture = []
        palette = [(0.45, 0.25, 0.15), (0.6, 0.15, 0.15), (0.2, 0.3, 0.55), (0.3, 0.3, 0.3)]
        for i in range(3):
            for j in range(2):
                c = _center(i, j)
                for _ in range(2):
                    sx, sz = float(r.uniform(3, 7)), float(r.uniform(2, 4))
                    ox, oz = float(r.uniform(-6, 6)), float(r.uniform(-7, 7))
                    if abs(ox) < 4 and abs(oz) < 4:
                        ox += 6
                    p = (c[0] + ox, 1.5, c[2] + oz)
                    w.add(p, (sx, 3, sz), palette[r.integers(len(palette))])
                    self.furniture.append((p[0], p[2], sx / 2 + 1.2, sz / 2 + 1.2))
        self.lobby = np.array([0.0, 50.0, 150.0])
        w.add(self.lobby - [0, 0.5, 0], (30, 1, 30), (0.7, 0.7, 0.75))

        self.bots = [Bot(h=Humanoid(pos=self.lobby + [r.uniform(-10, 10), 0, r.uniform(-10, 10)],
                                    shirt=tuple(r.uniform(0.1, 0.9, 3)), pants=tuple(r.uniform(0.1, 0.5, 3))), name=n)
                     for n in NAMES]
        self.role, self.coins, self.gun_cd, self.knife_cd = "lobby", 0, 0.0, 0.0
        self.coin_pos: list[np.ndarray] = []
        self.gun_drop = None
        self.phase, self.phase_t = "lobby", 0.0
        self.rounds: list[float] = []
        self.result = ""
        self.goal = None
        self.cam.pitch = math.radians(20)

    def respawn_point(self):
        return self.lobby.copy() if getattr(self, "phase", "lobby") == "lobby" else self._floor_point()

    def _floor_point(self) -> np.ndarray:
        r = self.rng
        for _ in range(50):
            i, j = int(r.integers(3)), int(r.integers(2))
            p = np.array([r.uniform(XS[i] + 3, XS[i + 1] - 3), 0.0, r.uniform(ZS[j] + 3, ZS[j + 1] - 3)])
            if not any(abs(p[0] - fx) < hx and abs(p[2] - fz) < hz for fx, fz, hx, hz in self.furniture):
                return p
        return _center(int(r.integers(3)), int(r.integers(2)))

    # ── everyone in one list: index 0 is you ──
    def chars(self):
        out = [(0, self.player, self.role, self.equipped is not None)]
        return out + [(k + 1, b.h, b.role, b.armed) for k, b in enumerate(self.bots)]

    def _murderer(self):
        for idx, h, role, armed in self.chars():
            if role == "murderer" and h.alive:
                return idx, h, armed
        return None

    def _los(self, a, b) -> bool:
        o, e = a.pos + [0, 4.5, 0], b.pos + [0, 3.0, 0]
        d = e - o
        dist = float(np.linalg.norm(d))
        t, _ = self.world.raycast(o, d / max(dist, 1e-6), max_t=dist, mask=self.world.collide)
        return not math.isfinite(t)

    # ── rounds ──
    def _start_round(self):
        r = self.rng
        forced = self.kw.get("role")
        roles = ["murderer", "sheriff"] + ["innocent"] * 6
        r.shuffle(roles)
        if forced:   # tests can pin your role; swap it into slot 0
            k = roles.index(forced)
            roles[0], roles[k] = roles[k], roles[0]
        self.role, self.equipped = roles[0], None
        self.phase, self.phase_t = "round", 0.0
        self.respawn()
        for b, role in zip(self.bots, roles[1:]):
            b.role, b.armed, b.target, b.react, b.cd = role, False, -1, 0.0, 0.0
            b.h.pos, b.h.vel, b.h.alive, b.h.stun = self._floor_point(), np.zeros(3), True, 0.0
            b.goal = b.h.pos.copy()
        self.coin_pos, self.gun_drop, self._next_coin = [], None, 0.0

    def _end_round(self, innocents_win: bool):
        if self.phase != "round":
            return
        me_innocent = self.role != "murderer"
        alive = self.player.alive
        win = (innocents_win and me_innocent and alive) or (not innocents_win and not me_innocent)
        self.rounds.append(1.0 if win else 0.0)
        if win:
            self.reward += 2.0 if self.role == "murderer" else 1.0
        elif innocents_win and me_innocent:
            self.reward += 0.3                    # your team won, but you didn't make it
        self.result = "Innocents win!" if innocents_win else "The Murderer wins!"
        self.toast(self.result, END_S)
        self.phase, self.phase_t = "end", 0.0

    def _kill(self, idx: int, by_sheriff_mistake=False):
        h = self.player if idx == 0 else self.bots[idx - 1].h
        if not h.alive:
            return
        role = self.role if idx == 0 else self.bots[idx - 1].role
        if idx == 0:
            self.die()
            self.equipped = None
            self.reward -= 1.0
        else:
            h.alive, h.dead_t = False, self.t
            self.bots[idx - 1].armed = False
        if role in ("sheriff", "hero"):
            self.gun_drop = h.pos + [0, 1.0, 0]      # the gun falls where its holder died
            if self.phase == "round" and idx != 0:
                self.toast("The Sheriff died! The gun dropped.", 3)
        if role == "murderer":
            self._end_round(True)
        elif not any(hh.alive for i, hh, rr, _ in self.chars() if rr != "murderer"):
            self._end_round(False)

    def _shoot(self, shooter: int, o, d):
        """A gunshot from o along d: first character hit (if no wall is in front of it)."""
        others = [(i, h) for i, h, _, _ in self.chars() if i != shooter]
        hit = ray_humanoids(o, d, [h for _, h in others], max_t=200)
        if hit is None:
            return
        tw, _ = self.world.raycast(o, d, max_t=hit[1], mask=self.world.collide)
        if math.isfinite(tw):
            return                                    # the bullet hit a wall first
        victim = others[hit[0]][0]
        vrole = self.role if victim == 0 else self.bots[victim - 1].role
        self._kill(victim)
        if vrole == "murderer":
            if shooter == 0:
                self.reward += 2.0
                self.toast("You shot the Murderer!")
        elif self.phase == "round":                   # shot an innocent: the shooter dies too
            if shooter == 0:
                self.toast("You shot an Innocent!")
            self._kill(shooter)

    # ── your inputs ──
    def on_key(self, k: str) -> None:
        if k == "1":
            self.on_ui("slot:0")

    def on_ui(self, id_: str) -> None:
        if id_ == "slot:0" and self._item():
            self.equipped = None if self.equipped else self._item()

    def _item(self):
        return {"murderer": "knife", "sheriff": "gun", "hero": "gun"}.get(self.role) if self.phase == "round" else None

    def on_tool_click(self) -> bool:
        if not self.equipped or any(el.get("id") and _over(self.cursor, el["rect"]) for el in self.hud()):
            return False
        if self.equipped == "gun" and self.gun_cd <= 0:
            self.gun_cd = GUN_CD
            self._shoot(0, self.cam.pos, self.cam.ray(self.cursor[0], self.cursor[1], self.aspect))
        elif self.equipped == "knife" and self.knife_cd <= 0:
            self.knife_cd = KNIFE_CD
            p = self.player
            fx, fz = math.sin(p.face), math.cos(p.face)
            for k, b in enumerate(self.bots):
                d = b.h.pos - p.pos
                dist = float(np.hypot(d[0], d[2]))
                if b.h.alive and dist < KNIFE_RANGE and (d[0] * fx + d[2] * fz) > -0.3 * dist:
                    self._kill(k + 1)
                    self.reward += 0.5
                    break
        return True

    # ── world ──
    def on_substep(self) -> None:
        self.phase_t += DT
        self.gun_cd = max(0.0, self.gun_cd - DT)
        self.knife_cd = max(0.0, self.knife_cd - DT)
        if self.phase == "lobby":
            for k, b in enumerate(self.bots):   # everyone mills around their own spot in the lobby
                a = 2 * math.pi * k / len(self.bots)
                npc_walk(b.h, self.world, self.lobby + [9 * math.cos(a), 0, 9 * math.sin(a)], stop=1.0)
            if self.phase_t >= LOBBY_S:
                self._start_round()
        elif self.phase == "end":
            if self.phase_t >= END_S:
                self.phase, self.phase_t = "lobby", 0.0
                self.respawn()
                for b in self.bots:
                    b.h.pos, b.h.alive, b.role, b.armed = self.lobby + [self.rng.uniform(-8, 8), 0, self.rng.uniform(-8, 8)], True, "innocent", False
                self.role, self.equipped = "lobby", None
        else:
            self._round_substep()
        if self.steps >= self.max_steps:
            self.done = True

    def _round_substep(self):
        if self.phase_t >= self._next_coin and len(self.coin_pos) < MAX_COINS:
            self.coin_pos.append(self._floor_point() + [0, 1.0, 0])
            self._next_coin = self.phase_t + 2.0
        for idx, h, role, _ in self.chars():   # coins and the dropped gun are picked up by touch
            if not h.alive:
                continue
            for ci, c in enumerate(self.coin_pos):
                if np.linalg.norm((c - h.pos)[[0, 2]]) < 2.0:
                    del self.coin_pos[ci]
                    if idx == 0:
                        self.coins += 1
                        self.reward += 0.05
                    else:
                        self.bots[idx - 1].coins += 1
                    break
            if self.gun_drop is not None and role == "innocent" and np.linalg.norm((self.gun_drop - h.pos)[[0, 2]]) < 2.5:
                self.gun_drop = None
                if idx == 0:
                    self.role = "hero"
                    self.toast("You picked up the gun!")
                else:
                    self.bots[idx - 1].role = "hero"
        think = int(round(self.phase_t / DT)) % SUBSTEPS == 0   # bots decide 10x a second, walk every substep
        m = self._murderer()
        for k, b in enumerate(self.bots):
            if not b.h.alive:
                continue
            b.cd = max(0.0, b.cd - DT)
            if think:
                self._bot_think(k + 1, b, m)
            npc_walk(b.h, self.world, nav(b.h.pos, b.goal), stop=0.8)
            if b.role == "murderer" and b.armed and b.cd <= 0 and b.target >= 0:
                th = self.player if b.target == 0 else self.bots[b.target - 1].h
                if th.alive and np.linalg.norm((th.pos - b.h.pos)[[0, 2]]) < KNIFE_RANGE - 0.5:
                    b.cd = KNIFE_CD
                    if b.target == 0:
                        self.toast("The Murderer got you!", 3)
                    self._kill(b.target)
        if self.phase == "round" and self.phase_t >= ROUND_S:
            self._end_round(True)

    def _bot_think(self, idx: int, b: Bot, m):
        r = self.rng
        h = b.h
        if b.role == "murderer":
            if b.target < 0 or not self._char(b.target).alive or r.random() < 1 / 30:   # rethinks its victim every ~3 s
                b.target = self._isolated(idx) if self.phase_t > 15 else -1
            if b.target >= 0:
                th = self._char(b.target)
                dist = float(np.linalg.norm((th.pos - h.pos)[[0, 2]]))
                b.armed = dist < 12                    # the knife only comes out up close
                b.goal = th.pos.copy()
            else:
                b.armed = False
                b.goal = self._nearest_coin(h.pos)
            return
        armed_m = m is not None and m[2] and m[0] != idx and self._los(h, m[1]) and np.linalg.norm((m[1].pos - h.pos)[[0, 2]]) < 50
        if b.role in ("sheriff", "hero"):
            b.armed = armed_m
            if armed_m:
                b.react += 1 / 10
                b.goal = h.pos.copy()
                if b.react > 0.8 and b.cd <= 0:        # reaction time, then an imperfect shot
                    b.cd = GUN_CD
                    dist = float(np.linalg.norm(m[1].pos - h.pos))
                    if r.random() < float(np.clip(1 - dist / 60, 0.2, 0.75)):
                        self._kill(m[0])
                return
            b.react = 0.0
        if armed_m and np.linalg.norm((m[1].pos - h.pos)[[0, 2]]) < 35:
            b.goal = self._far_room(m[1].pos)
        elif self.gun_drop is not None and b.role == "innocent" and np.linalg.norm((self.gun_drop - h.pos)[[0, 2]]) < 30:
            b.goal = self.gun_drop.copy()
        elif np.linalg.norm((b.goal - h.pos)[[0, 2]]) < 1.5 or r.random() < 0.02:
            b.goal = self._nearest_coin(h.pos) if r.random() < 0.7 else self._floor_point()

    def _char(self, idx):
        return self.player if idx == 0 else self.bots[idx - 1].h

    def _isolated(self, me: int) -> int:
        """The living non-murderer with the fewest people near them."""
        best, key = -1, None
        alive = [(i, h) for i, h, rr, _ in self.chars() if h.alive and i != me]
        for i, h in alive:
            near = sum(np.linalg.norm((h2.pos - h.pos)[[0, 2]]) < 15 for j, h2 in alive if j != i)
            d = float(np.linalg.norm((h.pos - self._char(me).pos)[[0, 2]]))
            if key is None or (near, d) < key:
                best, key = i, (near, d)
        return best

    def _nearest_coin(self, p):
        if not self.coin_pos:
            return _center(*_cell(p))          # no RNG here: the expert calls this, and asking it must not change the game
        return min(self.coin_pos, key=lambda c: np.linalg.norm((c - p)[[0, 2]])).copy()

    def _crowd_goal(self, p):
        """A coin near other players, or the middle of the nearest group: the murderer hunts whoever is alone."""
        others = [b.h.pos for b in self.bots if b.h.alive]
        if not others:
            return self._nearest_coin(p)
        near = lambda c: min(np.linalg.norm((o - c)[[0, 2]]) for o in others)
        if self.coin_pos:
            return min(self.coin_pos, key=lambda c: np.linalg.norm((c - p)[[0, 2]]) + 2 * max(0.0, near(c) - 10)).copy()
        close = sorted(others, key=lambda o: np.linalg.norm((o - p)[[0, 2]]))[:3]
        return np.mean(close, axis=0)

    def _charging(self):
        """Anyone within 14 studs sprinting straight at you: the tell real players watch for."""
        p = self.player.pos
        for b in self.bots:
            d = (p - b.h.pos)[[0, 2]]
            dist = float(np.linalg.norm(d))
            v = b.h.vel[[0, 2]]
            if b.h.alive and 1 < dist < 14 and float(v @ d) > 0.75 * 16 * dist:
                return b.h
        return None

    def _refuge(self, threat):
        """Run to whichever other player is farthest from the threat: safety in numbers, away from danger."""
        others = [b.h.pos for b in self.bots if b.h.alive and b.h is not threat]
        if not others:
            return self._far_room(threat.pos)
        return max(others, key=lambda o: np.linalg.norm((o - threat.pos)[[0, 2]]) - 0.3 * np.linalg.norm((o - self.player.pos)[[0, 2]])).copy()

    def _gun_sees(self, h) -> bool:
        return any(b.h.alive and b.role in ("sheriff", "hero") and np.linalg.norm((b.h.pos - h.pos)[[0, 2]]) < 50 and self._los(b.h, h)
                   for b in self.bots)

    def _far_room(self, threat):
        cs = [_center(i, j) for i in range(3) for j in range(2)]
        return max(cs, key=lambda c: np.linalg.norm((c - threat)[[0, 2]])).copy()

    # ── rendering ──
    def dynamic_boxes(self) -> list:
        out = []
        for b in self.bots:
            if b.h.alive or self.t - b.h.dead_t < 0.4:
                out += avatar_boxes(b.h, dead_for=0 if b.h.alive else self.t - b.h.dead_t)
                if b.h.alive and b.armed:
                    out.append(_in_hand(b.h, "knife" if b.role == "murderer" else "gun"))
        if self.player.alive and self.equipped:
            out.append(_in_hand(self.player, self.equipped))
        spin = self.t * 3
        for c in self.coin_pos:
            out.append((tuple(c), (0.9, 0.9, 0.25), spin, (1.0, 0.85, 0.1), 1))
        if self.gun_drop is not None:
            out.append((tuple(self.gun_drop + [0, 0.4, 0]), (0.8, 1.2, 2.6), spin, (1.0, 0.85, 0.2), 1))
        return out

    def hud(self) -> list:
        els = []
        rows = [["You" + ("" if self.player.alive else " (dead)"), self.coins]] + \
               [[b.name[:10] + ("" if b.h.alive or self.phase != "round" else " (dead)"), b.coins] for b in self.bots]
        els.append(dict(kind="leaderboard", cols=["Coins"], rows=rows))
        if self.phase == "lobby":
            els.append(dict(kind="text", rect=(0.3, 0.05, 0.4, 0.06), text=f"Next round in {LOBBY_S - self.phase_t:.0f}", size=28, align="center"))
        elif self.phase == "round":
            left = max(0.0, ROUND_S - self.phase_t)
            els.append(dict(kind="text", rect=(0.4, 0.02, 0.2, 0.06), text=f"{int(left // 60)}:{int(left % 60):02d}", size=30, align="center"))
            if self.phase_t < CARD_S:
                col = ROLE_COL.get(self.role, (1, 1, 1))
                els += [dict(kind="panel", rect=(0.33, 0.3, 0.34, 0.24), color=(0.08, 0.08, 0.1), alpha=0.85),
                        dict(kind="text", rect=(0.33, 0.33, 0.34, 0.08), text="You are the", size=22, align="center"),
                        dict(kind="text", rect=(0.33, 0.41, 0.34, 0.1), text=self.role.title(), size=44, align="center", color=col)]
            item = self._item()
            if item:
                els += hotbar_hud([item.title()], 0 if self.equipped else None)
            els.append(dict(kind="text", rect=(0.02, 0.9, 0.25, 0.06), text=f"Coins: {self.coins}", size=24, color=(1, 0.85, 0.2)))
            els.append(dict(kind="text", rect=(0.02, 0.84, 0.3, 0.05), text=self.role.title(), size=20, color=ROLE_COL.get(self.role, (1, 1, 1))))
            if not self.player.alive:
                els.append(dict(kind="text", rect=(0.3, 0.12, 0.4, 0.06), text="You died. Spectating…", size=24, align="center"))
        else:
            els.append(dict(kind="text", rect=(0.25, 0.1, 0.5, 0.08), text=self.result, size=40, align="center"))
        if self.gun_drop is not None and self.phase == "round":
            bb = self.billboard(self.gun_drop + [0, 3, 0], "GUN", color=(1, 0.85, 0.2), size=1.3, max_dist=120)
            if bb:
                els.append(bb)
        return els

    def info(self) -> dict:
        return dict(role=self.role, phase=self.phase, coins=self.coins, rounds=len(self.rounds), wins=int(sum(self.rounds)),
                    alive=sum(h.alive for _, h, _, _ in self.chars()), deaths=self.deaths)

    def score(self) -> float:
        return (float(np.mean(self.rounds)) if self.rounds else 0.0) + 0.01 * self.coins

    def potential(self) -> float:
        self.goal = None
        self.expert()
        if self.goal is None or not self.player.alive:
            return 0.0
        return -0.01 * float(np.linalg.norm((self.goal - self.player.pos)[[0, 2]]))

    # ── expert ──
    def expert(self) -> Action:
        p = self.player
        if self.phase != "round" or not p.alive:
            return Action(cursor=self.cursor)
        m = self._murderer()
        armed_m = m is not None and m[0] != 0 and m[2] and self._los(p, m[1])
        if self.role in ("sheriff", "hero"):
            if self.equipped != "gun":
                return self._tap("1")
            if armed_m:
                return self._aim_and_fire(m[1])
            return self._go(self._crowd_goal(p.pos))
        if self.role == "murderer":
            gun = [k + 1 for k, b in enumerate(self.bots) if b.h.alive and b.role in ("sheriff", "hero")]
            tgt = gun[0] if gun else self._isolated(0)   # the gun-holder first: they're the only one who can stop you
            if tgt < 0:
                return Action(cursor=self.cursor)
            th = self._char(tgt)
            dist = float(np.linalg.norm((th.pos - p.pos)[[0, 2]]))
            seen = self._gun_sees(p)
            # knife out only for the kill, away right after; striking the gun-holder beats its reaction time
            want = dist < 5 and (bool(gun) or not seen) and self.knife_cd <= 0.3
            if want != (self.equipped == "knife"):
                return self._tap("1")
            a = self._go(th.pos, stop=1.5)
            if dist < KNIFE_RANGE - 0.4 and self.equipped == "knife" and self.knife_cd <= 0:
                a.click, a.cursor = True, (0.5, 0.5)
            return a
        threat = m[1] if armed_m and np.linalg.norm((m[1].pos - p.pos)[[0, 2]]) < 50 else self._charging()
        if threat is not None:
            return self._go(self._refuge(threat))
        if self.gun_drop is not None:
            return self._go(self.gun_drop)
        return self._go(self._crowd_goal(p.pos))

    def _tap(self, k: str) -> Action:
        """Press k, releasing first if it's still held: a held key isn't a new press."""
        return Action(cursor=self.cursor) if k in self.held else Action(keys=frozenset({k}), cursor=self.cursor)

    def _go(self, goal, stop=1.0) -> Action:
        self.goal = np.asarray(goal, float)
        nxt = nav(self.player.pos, self.goal)
        return self.steer_to(nxt, stop_dist=stop if np.allclose(nxt, self.goal) else 0.5)

    def _aim_and_fire(self, target) -> Action:
        """Turn toward the Murderer, put the cursor on their chest, fire once the cursor is already on it."""
        self.goal = target.pos.copy()
        a = self.steer_to(target.pos, keys=(), stop_dist=99)
        chest = target.pos + [0, 2.5, 0]
        uv = self.cam.project(chest, self.aspect)
        if uv is None or not (0.03 < uv[0] < 0.97 and 0.1 < uv[1] < 0.85):
            return a
        on = abs(uv[0] - self.cursor[0]) < 0.03 and abs(uv[1] - self.cursor[1]) < 0.04
        d = self.cam.ray(uv[0], uv[1], self.aspect)
        clear = not math.isfinite(self.world.raycast(self.cam.pos, d, max_t=float(np.linalg.norm(chest - self.cam.pos)) - 1.5,
                                                     mask=self.world.collide)[0])
        a.cursor, a.click = (float(uv[0]), float(uv[1])), bool(on and clear and self.gun_cd <= 0)
        return a


def _in_hand(h: Humanoid, item: str):
    fx, fz = math.sin(h.face), math.cos(h.face)
    rx, rz = math.cos(h.face), -math.sin(h.face)
    p = h.pos + [rx * 1.5 + fx * 1.6, 2.4, rz * 1.5 + fz * 1.6]   # held out in front of the right hand
    return (tuple(p), (0.35, 0.5, 2.4) if item == "knife" else (0.45, 0.9, 1.8), h.face,
            (0.85, 0.85, 0.9) if item == "knife" else (0.12, 0.12, 0.14))


def _over(p, rect) -> bool:
    x, y, w, h = rect
    return x <= p[0] <= x + w and y <= p[1] <= y + h


if __name__ == "__main__":
    import pickle
    import time
    from core.actions import KEYS

    def rand_action(rng):
        return Action(keys=frozenset(k for k in KEYS if k not in ("esc",) and rng.random() < 0.12), look_x=float(rng.normal(0, 12)),
                      cursor=tuple(rng.random(2)), click=rng.random() < 0.1)

    for role in ("innocent", "sheriff", "murderer"):
        for seed in range(5):
            res = {}
            for who in ("expert", "random"):
                env = MurderMystery2(seed=seed, role=role, max_steps=1800)
                rng = np.random.default_rng(seed)
                t0 = time.time()
                while not env.done:
                    env.step(env.expert() if who == "expert" else rand_action(rng))
                res[who] = (env.score(), env.info(), (time.time() - t0) / env.steps * 1000)
            e, rn = res["expert"], res["random"]
            print(f"{role:9s} seed {seed}: expert score {e[0]:.2f} (wins {e[1]['wins']}/{e[1]['rounds']}, coins {e[1]['coins']}, {e[2]:.1f} ms/tick)"
                  f"  | random {rn[0]:.2f} (wins {rn[1]['wins']}/{rn[1]['rounds']}, coins {rn[1]['coins']})")
    env = MurderMystery2(seed=5)
    for _ in range(300):
        env.step(env.expert())
    twin = pickle.loads(pickle.dumps(env))
    for _ in range(100):
        a = env.expert()
        r1, r2 = env.step(a)[0], twin.step(a)[0]
        assert r1 == r2 and np.allclose(env.player.pos, twin.player.pos)
    print("pickle ok")
