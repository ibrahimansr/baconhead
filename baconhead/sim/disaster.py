import math

import numpy as np

from baconhead.core.actions import Action
from baconhead.sim.core import DT, Env, Humanoid, Room, avatar_boxes, health_hud, npc_walk, route

LOBBY = np.array([0.0, 120.0, 0.0])        # the floating lobby, well above the map
LOBBY_S, PRE_S, DISASTER_S, END_S = 10.0, 5.0, 50.0, 3.0
DISASTERS = ("flood", "meteor", "acid", "sandstorm", "tornado", "earthquake")
TITLE = {"flood": "Flash Flood", "meteor": "Meteor Shower", "acid": "Acid Rain", "sandstorm": "Sandstorm",
         "tornado": "Tornado", "earthquake": "Earthquake"}
TIP = {"flood": "Get to high ground!", "meteor": "Find shelter!", "acid": "Get under a roof!", "sandstorm": "Get inside!",
       "tornado": "Stay away from the tornado!", "earthquake": "Get away from buildings!"}
SAFETY = {"flood": "high", "meteor": "indoors", "acid": "indoors", "sandstorm": "indoors", "tornado": "away", "earthquake": "open"}
# plan labels (say()): moving toward safety vs. already safe, phrased via the HUD disaster title and what's visible
GO = {"flood_roof": "flash flood, climb the stairs onto the nearest roof", "flood_hill": "flash flood, climb to the top of the hill",
      "meteor": "meteor shower, get inside the nearest building", "acid": "acid rain is falling, get under the nearest roof",
      "sandstorm": "sandstorm, get inside the nearest building", "tornado": "tornado, run far away from the funnel",
      "earthquake": "earthquake, run into the open away from buildings"}
STAY = {"flood": "flash flood, stay up high above the water", "meteor": "meteor shower, stay inside under the roof",
        "acid": "acid rain, stay inside under the roof", "sandstorm": "sandstorm, stay inside under the roof",
        "tornado": "tornado is far away, stay put and watch it", "earthquake": "earthquake, stay in the open away from buildings"}
FLOOD_TOP = 14.0
ROOF_TOP = 13.0
N_NPC = 4


class NaturalDisaster(Env):
    name = "disaster"
    train_steps = 3000
    eval_kw = {}

    def build(self):
        r, w = self.rng, self.world
        self.max_steps = int(self.kw.get("max_steps", 3000))
        w.add((0, -0.5, 0), (200, 1, 200), (0.36, 0.62, 0.3))                         # the island
        w.add(LOBBY - [0, 0.5, 0], (30, 1, 30), (0.6, 0.6, 0.65))                     # lobby floor
        for dx, dz, sx, sz in ((15, 0, 1, 30), (-15, 0, 1, 30), (0, 15, 30, 1), (0, -15, 30, 1)):
            w.add(LOBBY + [dx, 1.5, dz], (sx, 3, sz), (0.45, 0.45, 0.5))
        self.water = w.add((0, -100, 0), (220, 40, 220), (0.2, 0.45, 0.85), collide=False)

        slots = [(-55, -45), (10, -55), (55, -5), (-50, 35), (25, 50), (-5, 5)]
        order = r.permutation(len(slots))
        self.buildings = [self._building(*slots[k]) for k in order[:5]]
        self.rooms = [b["room"] for b in self.buildings]
        hx, hz = slots[order[5]]
        self.hill = np.array([hx, 0.0, hz])
        for i, s in enumerate((26, 22, 18, 14, 10, 6, 4, 3)):                       # stepped hill, 1.8-stud tiers
            w.add((hx, 0.9 + 1.8 * i, hz), (s, 1.8, s), (0.42, 0.55, 0.3) if i % 2 else (0.5, 0.45, 0.35))
        self.hill_top = 1.8 * 8
        for _ in range(8):                                                             # trees and rocks for cover/obstacles
            x, z = r.uniform(-90, 90, 2)
            if min(np.hypot(x - b["cx"], z - b["cz"]) for b in self.buildings) > 20 and np.hypot(x - hx, z - hz) > 20:
                w.add((x, 3, z), (1.5, 6, 1.5), (0.4, 0.28, 0.15))
                w.add((x, 7, z), (6, 4, 6), (0.2, 0.55, 0.2))
        self.opens = [np.array([x, 0.0, z]) for x in (-80, -30, 30, 80) for z in (-80, -30, 30, 80)]

        self.npcs = [Humanoid(pos=LOBBY + [r.uniform(-8, 8), 0, r.uniform(-8, 8)], shirt=tuple(r.uniform(0.1, 0.9, 3)),
                              pants=tuple(r.uniform(0.1, 0.6, 3))) for _ in range(N_NPC)]
        self.npc_goal = [LOBBY.copy() for _ in range(N_NPC)]
        self.npc_smart = [bool(r.random() < 0.6) for _ in range(N_NPC)]   # the rest panic and wander
        self.phase, self.phase_t = "lobby", 0.0
        self.disaster = None
        self.order = [str(d) for d in r.permutation(DISASTERS)]
        self.round = 0
        self.wins = 0
        self.water_y = -100.0
        self.meteors = []
        self.falling = {}       # part → fall speed, during earthquakes
        self.hit_by = set()
        self.tornado = np.zeros(3)
        self.tornado_goal = np.zeros(3)
        self.drops = []
        self.in_round = False
        self.cam.pitch = math.radians(20)

    def _building(self, cx, cz):
        r, w = self.rng, self.world
        sx, sz = float(r.uniform(14, 20)), float(r.uniform(15, 20))
        side = int(r.choice([-1, 1]))
        h = ROOF_TOP - 1
        col = tuple(r.uniform(0.5, 0.95, 3))
        zd = cz + side * sz / 2
        rest = (sx - 6) / 2
        parts = [w.add((cx, h / 2, cz - side * sz / 2), (sx, h, 1), col),
                 w.add((cx - sx / 2, h / 2, cz), (1, h, sz), col),
                 w.add((cx + sx / 2, h / 2, cz), (1, h, sz), col),
                 w.add((cx - 3 - rest / 2, h / 2, zd), (rest, h, 1), col),
                 w.add((cx + 3 + rest / 2, h / 2, zd), (rest, h, 1), col),
                 w.add((cx, h - 2, zd), (6, 4, 1), col),
                 w.add((cx, h + 0.5, cz), (sx, 1, sz), tuple(np.array(col) * 0.7))]   # roof, top at ROOF_TOP
        w.add((cx, 0.05, cz), (sx - 1, 0.1, sz - 1), (0.7, 0.6, 0.45))
        # exterior staircase up the +x wall: 1.5-stud rises, under the 2-stud hip height so it walks
        xs = cx + sx / 2 + 1.5
        z0 = cz - sz / 2 + 0.75
        n = math.ceil(ROOF_TOP / 1.5)
        for k in range(n):
            top = min(ROOF_TOP, 1.5 * (k + 1)) if k < n - 1 else ROOF_TOP
            w.add((xs, top / 2, z0 + 1.5 * k), (2, top, 1.5), (0.55, 0.55, 0.6))
        room = Room(cx, cz, sx / 2 - 0.5, sz / 2 - 0.5, np.array([cx, 0, zd - side * 3.0]), np.array([cx, 0, zd + side * 4.0]))
        return dict(cx=cx, cz=cz, sx=sx, sz=sz, room=room, parts=parts, xs=xs, z0=z0, zt=z0 + 1.5 * (n - 1),
                    orig={i: self.world._rows[i]["c"] for i in parts})

    # ── round flow ──
    def respawn_point(self):
        return LOBBY + [self.rng.uniform(-6, 6), 0.1, self.rng.uniform(-6, 6)]

    def _all(self):
        return [self.player] + self.npcs

    def _spawn_on_map(self, h):
        a = self.rng.uniform(0, 2 * math.pi)
        h.pos = np.array([math.cos(a) * 20, 0.1, math.sin(a) * 20])
        h.vel[:] = 0

    def _set_phase(self, ph):
        self.phase, self.phase_t = ph, 0.0
        w = self.world
        if ph == "pre":            # teleport everyone in the lobby down to the map
            self.round += 1
            for b in self.buildings:   # repair last round's earthquake damage
                for i, c in b["orig"].items():
                    w.c[i] = np.asarray(c, float)
            self.falling.clear()
            for h in self._all():
                if h.alive:
                    self._spawn_on_map(h)
            self.in_round = self.player.alive
            self.toast(f"Round {self.round}: Disaster in 5…", 3)
        elif ph == "disaster":
            self.disaster = self.order[(self.round - 1) % len(self.order)]
            self.toast(f"{TITLE[self.disaster]}! {TIP[self.disaster]}", 4)
            self.hit_by.clear()
            if self.disaster == "tornado":
                self.tornado = np.array([self.rng.uniform(-60, 60), 0, self.rng.uniform(-60, 60)])
                self.tornado_goal = self.tornado.copy()
            if self.disaster == "earthquake":
                parts = [i for b in self.buildings for i in b["parts"]]
                k = len(parts) // 2
                self.quake_plan = sorted((float(self.rng.uniform(2, DISASTER_S - 5)), int(i)) for i in self.rng.choice(parts, k, replace=False))
        elif ph == "end":
            survived = self.player.alive and self.in_round and self.player.pos[1] < LOBBY[1] - 20
            if survived:
                self.wins += 1
                self.reward += 1.0
                self.toast("You survived!", 3)
            self.disaster, self.water_y, self.meteors, self.drops = None, -100.0, [], []
            w.c[self.water] = [0, -100, 0]
        elif ph == "lobby":
            for h in self._all():
                if h.alive:
                    h.pos = LOBBY + [self.rng.uniform(-6, 6), 0.1, self.rng.uniform(-6, 6)]
                    h.vel[:] = 0
                    h.health = 100.0

    def on_substep(self):
        self.phase_t += DT
        dur = {"lobby": LOBBY_S, "pre": PRE_S, "disaster": DISASTER_S, "end": END_S}[self.phase]
        if self.phase_t >= dur:
            self._set_phase({"lobby": "pre", "pre": "disaster", "disaster": "end", "end": "lobby"}[self.phase])
        if self.phase == "disaster":
            self._disaster_step()
        self._npcs()
        if self.steps >= self.max_steps:
            self.done = True

    def on_death(self):
        self.reward -= 1.0
        self.in_round = False

    # ── hazards ──
    def _covered(self, h):
        t, _ = self.world.raycast(h.pos + [0, 5.2, 0], np.array([0.0, 1.0, 0.0]), max_t=40, mask=self.world.collide)
        return math.isfinite(t)

    def _hurt(self, h, amount, why):
        if h is self.player:
            self.damage(amount, why)
        elif h.alive:
            h.health -= amount
            if h.health <= 0:
                h.alive, h.dead_t = False, self.t

    def _disaster_step(self):
        d, w, r, t = self.disaster, self.world, self.rng, self.phase_t
        on_map = [h for h in self._all() if h.alive and h.pos[1] < LOBBY[1] - 20]
        if d == "flood":
            self.water_y = min(FLOOD_TOP, -2 + (FLOOD_TOP + 2) * t / 30)
            w.c[self.water] = [0, self.water_y - 20, 0]
            for h in on_map:
                wet = h.pos[1] < self.water_y
                h.speed_mult = 0.6 if wet else 1.0
                if h.pos[1] + 4.5 < self.water_y:
                    self._hurt(h, 20 * DT, "You drowned")
        elif d == "meteor":
            if r.random() < DT / 0.35:
                tgt = (self.player.pos + r.normal(0, 12, 3)) if r.random() < 0.35 else np.array([r.uniform(-95, 95), 0, r.uniform(-95, 95)])
                hit_t, _ = w.raycast(np.array([tgt[0], 100.0, tgt[2]]), np.array([0.0, -1.0, 0.0]), max_t=200, mask=w.collide)
                y = 100 - hit_t if math.isfinite(hit_t) else -60
                self.meteors.append({"p": np.array([tgt[0], y, tgt[2]]), "t": self.t + 1.4})
            for m in [m for m in self.meteors if m["t"] <= self.t]:
                self.meteors.remove(m)
                for h in on_map:
                    if np.linalg.norm((h.pos - m["p"])[[0, 2]]) < 5 and abs(h.pos[1] - m["p"][1]) < 5:
                        self._hurt(h, 100, "A meteor hit you")
        elif d in ("acid", "sandstorm"):
            for h in on_map:
                if not self._covered(h):
                    self._hurt(h, (10 if d == "acid" else 6) * DT, "The acid rain got you" if d == "acid" else "The sandstorm got you")
                    if d == "sandstorm" and h.grounded:
                        h.pos[0] += 5 * DT       # the wind shoves you while you're outside
            if round(self.phase_t / DT) % 6 == 0:   # once per tick: drops only where the sky is open
                p, up = self.player.pos, np.array([0.0, 1.0, 0.0])
                pts = [np.array([p[0] + r.uniform(-15, 15), p[1] + r.uniform(0, 14), p[2] + r.uniform(-15, 15)]) for _ in range(30)]
                self.drops = [tuple(q) for q in pts if not math.isfinite(w.raycast(q, up, max_t=40, mask=w.collide)[0])]
        elif d == "tornado":
            if np.linalg.norm(self.tornado_goal - self.tornado) < 3:
                near = self.player.pos if r.random() < 0.4 else np.array([r.uniform(-80, 80), 0, r.uniform(-80, 80)])
                self.tornado_goal = np.array([near[0], 0, near[2]])
            v = self.tornado_goal - self.tornado
            self.tornado += v / max(1e-6, np.linalg.norm(v)) * 9 * DT
            for h in on_map:
                rel = (h.pos - self.tornado)[[0, 2]]
                dist = float(np.linalg.norm(rel))
                if dist < 14:
                    tang = np.array([-rel[1], rel[0]]) / max(dist, 1e-6)
                    h.vel[0] += tang[0] * 60 * DT - rel[0] / max(dist, 1) * 40 * DT
                    h.vel[2] += tang[1] * 60 * DT - rel[1] / max(dist, 1) * 40 * DT
                    if dist < 8:
                        h.vel[1] = max(h.vel[1], 45)
                        h.grounded = False
                        self._hurt(h, 25 * DT, "The tornado got you")
        elif d == "earthquake":
            while self.quake_plan and self.quake_plan[0][0] <= t:
                _, i = self.quake_plan.pop(0)
                self.falling[i] = 0.0
            for i in list(self.falling):
                v = self.falling[i] + 120 * DT
                c, hh = w.c[i], w.h[i]
                ny = max(hh[1], c[1] - v * DT)            # comes to rest on the ground
                w.c[i] = [c[0], ny, c[2]]
                self.falling[i] = v
                for k, h in enumerate(self._all()):
                    if not h.alive or (i, k) in self.hit_by:
                        continue
                    lx, lz = abs(h.pos[0] - c[0]), abs(h.pos[2] - c[2])
                    if lx < hh[0] + 1 and lz < hh[2] + 1 and h.pos[1] < ny + hh[1] and h.pos[1] + 5 > ny - hh[1]:
                        self.hit_by.add((i, k))
                        self._hurt(h, 45, "Falling debris hit you")
                if ny <= hh[1]:
                    del self.falling[i]

    # ── NPC players ──
    def _npcs(self):
        r = self.rng
        for k, h in enumerate(self.npcs):
            if not h.alive:
                if self.t - h.dead_t > 3 and self.phase in ("lobby", "end"):
                    h.alive, h.health = True, 100.0
                    h.pos = LOBBY + [r.uniform(-6, 6), 0.1, r.uniform(-6, 6)]
                continue
            if self.phase == "disaster" and self.npc_smart[k]:
                goal = self._next(h, self._goal(h))
            elif np.linalg.norm((self.npc_goal[k] - h.pos)[[0, 2]]) < 2 or r.random() < DT / 6:
                base = LOBBY if h.pos[1] > LOBBY[1] - 20 else np.zeros(3)
                self.npc_goal[k] = base + [r.uniform(-60, 60) if base[1] == 0 else r.uniform(-10, 10), 0, r.uniform(-60, 60) if base[1] == 0 else r.uniform(-10, 10)]
                goal = self.npc_goal[k]
            else:
                goal = self.npc_goal[k]
            if npc_walk(h, self.world, goal, stop=0.8):
                h.alive, h.dead_t = False, self.t

    # ── where's safe ──
    # (kind, point) a sensible player heads for right now
    def _goal(self, h):
        p, d = h.pos, self.disaster
        if d is None:
            return ("walk", p.copy())
        s = SAFETY[d]
        if s == "high":
            cand = [("roof", b) for b in self.buildings] + [("hill", None)]
            def cost(c):
                if c[0] == "hill":
                    return float(np.linalg.norm((self.hill - p)[[0, 2]]))
                b = c[1]
                return float(np.linalg.norm((np.array([b["xs"], 0, b["z0"] - 3.5]) - p)[[0, 2]]))
            return min(cand, key=cost)
        if s == "indoors":
            b = min(self.buildings, key=lambda b: np.hypot(b["cx"] - p[0], b["cz"] - p[2]))
            return ("indoors", b)
        if s == "away":
            pts = self.opens + [np.array([b["cx"], 0, b["cz"]]) for b in self.buildings]
            score = lambda q: np.linalg.norm((q - self.tornado)[[0, 2]]) - 0.3 * np.linalg.norm((q - p)[[0, 2]])
            return ("point", max(pts, key=score))
        far = lambda q: min(np.hypot(q[0] - b["cx"], q[2] - b["cz"]) for b in self.buildings)
        return ("point", min([q for q in self.opens if far(q) > 25] or self.opens, key=lambda q: np.linalg.norm((q - p)[[0, 2]])))

    # the next point to walk at for a goal, handling doors, stairs, and walking around buildings
    def _next(self, h, goal):
        kind, g = goal
        p = h.pos
        if kind == "indoors":
            nxt = route(p, np.array([g["cx"], 0, g["cz"]]), self.rooms)
            return nxt if any(rm.inside(p) for rm in self.rooms) else self._around(p, nxt)
        if kind == "hill":
            return self._around(p, self.hill)
        if kind == "roof":
            b = g
            on_roof = abs(p[0] - b["cx"]) < b["sx"] / 2 and abs(p[2] - b["cz"]) < b["sz"] / 2 and p[1] > ROOF_TOP - 0.5
            # the stairs plus the lane in front of them: once in it, climb. (Was: within 1.2 of the foot → climb,
            # else → the foot; one step out of that circle flipped the target 180° and it dithered through floods.)
            on_stairs = abs(p[0] - b["xs"]) < 1.3 and b["z0"] - 4.5 < p[2] < b["zt"] + 1
            if on_roof:
                return np.array([b["cx"], ROOF_TOP, b["cz"]])
            if on_stairs:
                if p[1] > ROOF_TOP - 0.3:
                    return np.array([b["cx"], ROOF_TOP, b["zt"]])
                return np.array([b["xs"], 0, b["zt"]])
            foot = np.array([b["xs"], 0, b["z0"] - 3.5])
            for rm in self.rooms:
                if rm.inside(p):
                    return route(p, foot, self.rooms)
            return self._around(p, foot)
        if any(rm.inside(p) for rm in self.rooms):
            return route(p, g, self.rooms)
        return self._around(p, g)

    # straight at g unless a building is in the way; then via the best corner of its footprint
    def _around(self, p, g):
        for b in sorted(self.buildings, key=lambda b: np.hypot(b["cx"] - p[0], b["cz"] - p[2])):
            x0, x1 = b["cx"] - b["sx"] / 2 - 1.5, b["xs"] + 2.5
            z0, z1 = b["cz"] - b["sz"] / 2 - 1.5, b["cz"] + b["sz"] / 2 + 1.5
            if x0 < g[0] < x1 and z0 < g[2] < z1:
                continue          # heading into this building (its doorway or stairs): route()/the stair logic handles it
            if _seg_hits_rect(p, g, x0, x1, z0, z1):
                corners = [np.array([x, 0, z]) for x in (x0 - 0.5, x1 + 0.5) for z in (z0 - 0.5, z1 + 0.5)]
                # reaching a corner is checked against the real footprint, not the padding: hugging a wall puts you
                # in (or on the edge of) the padding, every line then "crossed" it, no corner qualified, and the
                # player walked straight into the wall through a whole flood or sandstorm
                r = (x0 + 1.3, x1 - 1.2, z0 + 1.3, z1 - 1.3)
                ok = [c for c in corners if not _seg_hits_rect(p, c, *r) and np.linalg.norm((c - p)[[0, 2]]) > 1.5]
                if ok:   # corners with a clear line on to the goal first, then shortest total
                    return min(ok, key=lambda c: (_seg_hits_rect(c, g, x0, x1, z0, z1),
                                                  np.linalg.norm((c - p)[[0, 2]]) + np.linalg.norm((g - c)[[0, 2]])))
        return np.asarray(g, float)

    def _safe(self, h):
        d = self.disaster
        if d is None:
            return True
        s, p = SAFETY[d], h.pos
        if s == "high":
            return p[1] + 4.5 > FLOOD_TOP + 0.5
        if s == "indoors":
            return any(rm.inside(p) for rm in self.rooms) and self._covered(h)
        if s == "away":
            return float(np.linalg.norm((p - self.tornado)[[0, 2]])) > 30
        return min(np.hypot(p[0] - b["cx"], p[2] - b["cz"]) for b in self.buildings) > 20

    # ── interface ──
    def expert(self):
        p = self.player
        if not p.alive:
            self.say("you died, wait to respawn in the lobby")
            return Action(cursor=self.cursor)
        if p.pos[1] > LOBBY[1] - 20:
            self.say("intermission, wait in the lobby" if self.phase == "lobby" else "round in progress, wait in the lobby for the next one")
            return Action(cursor=self.cursor)
        goal = self._goal(p)
        self.goal = goal[1] if isinstance(goal[1], np.ndarray) else (
            np.array([goal[1]["cx"], 0, goal[1]["cz"]]) if goal[1] is not None else self.hill)
        nxt = self._next(p, goal)
        if self._safe(p) and np.linalg.norm((nxt - p.pos)[[0, 2]]) < 2.5:
            self.say(STAY[self.disaster] if self.disaster else
                     "disaster incoming, wait for it to be announced" if self.phase == "pre" else "round is over, wait to go back to the lobby")
            return Action(cursor=self.cursor)
        idle = "disaster incoming, wait for it to be announced" if self.phase == "pre" else "round is over, wait to go back to the lobby"
        self.say(GO["flood_" + goal[0]] if self.disaster == "flood" else GO.get(self.disaster, idle))   # none: between disasters
        return self.steer_to(nxt, stop_dist=1.0)

    def potential(self):
        p = self.player
        if not p.alive or self.disaster is None:
            return 0.0
        self.goal = None
        self.expert()
        dist = float(np.linalg.norm((self.goal - p.pos)[[0, 2]])) if self.goal is not None else 0.0
        return 0.5 * p.health / 100 + 0.5 * self._safe(p) - 0.003 * dist

    def score(self):
        return float(self.wins)

    def info(self):
        return dict(wins=self.wins, rounds=self.round, phase=self.phase, disaster=self.disaster,
                    health=round(self.player.health), deaths=self.deaths)

    def scene_theme(self):
        d, t = self.disaster, dict(self.theme)
        if d == "acid":
            t.update(sky_top=np.array([0.35, 0.5, 0.25]), sky_hor=np.array([0.55, 0.7, 0.4]), sun_k=0.5, fog=250)
        elif d == "sandstorm":
            t.update(sky_top=np.array([0.7, 0.6, 0.4]), sky_hor=np.array([0.8, 0.7, 0.5]), sun_k=0.5, fog=60)
        elif d in ("tornado", "flood"):
            t.update(sky_top=np.array([0.4, 0.45, 0.5]), sky_hor=np.array([0.6, 0.63, 0.66]), sun_k=0.45, fog=300)
        elif d == "meteor":
            t.update(sky_top=np.array([0.35, 0.2, 0.2]), sky_hor=np.array([0.75, 0.45, 0.3]), sun_k=0.6)
        return t

    def dynamic_boxes(self):
        out = []
        for h in self.npcs:
            if h.alive:
                out += avatar_boxes(h)
        for m in self.meteors:
            left = m["t"] - self.t
            out.append(((m["p"][0], m["p"][1] + 0.08, m["p"][2]), (9, 0.15, 9), 0.0, (1.0, 0.15, 0.1), 1))   # warning ring
            out.append(((m["p"][0], m["p"][1] + 2 + left * 60, m["p"][2]), (3, 3, 3), left, (1.0, 0.5, 0.1), 1))
        if self.disaster == "tornado":
            x, _, z = self.tornado
            for i in range(7):
                s = 3 + i * 2.6
                out.append(((x + math.sin(self.t * 3 + i) * 0.8, 2 + i * 5, z), (s, 5, s), self.t * 4 + i, (0.45, 0.45, 0.48)))
        if self.disaster in ("acid", "sandstorm"):
            col = (0.4, 0.95, 0.3) if self.disaster == "acid" else (0.85, 0.75, 0.5)
            out += [((x, y, z), (0.12, 1.2, 0.12), 0.0, col, 1) for x, y, z in self.drops]
        return out

    def hud(self):
        els = [dict(kind="leaderboard", cols=["Wins"], rows=[["You", self.wins]] + [[f"Player{k + 2}", 0] for k in range(N_NPC)])]
        if self.phase == "lobby":
            msg = f"Intermission  {max(0, LOBBY_S - self.phase_t):.0f}"
        elif self.phase == "pre":
            msg = f"Disaster in {max(0, PRE_S - self.phase_t):.0f}…"
        elif self.phase == "disaster":
            msg = f"{TITLE[self.disaster]}   {max(0, DISASTER_S - self.phase_t):.0f}"
        else:
            msg = "Survivors: " + ("You" if self.player.alive and self.in_round else "…")
        els.append(dict(kind="text", rect=(0.3, 0.02, 0.4, 0.07), text=msg, size=30, align="center",
                        color=(1, 0.85, 0.3) if self.phase == "disaster" else (1, 1, 1)))
        if self.phase == "disaster":
            els.append(dict(kind="text", rect=(0.3, 0.09, 0.4, 0.05), text=TIP[self.disaster], size=18, align="center"))
        return els + health_hud(self.player.health)


# does the xz segment a→b cross the rectangle? Slab test on the segment's parameter
def _seg_hits_rect(a, b, x0, x1, z0, z1):
    ax, az, dx, dz = a[0], a[2], b[0] - a[0], b[2] - a[2]
    lo, hi = 0.0, 1.0
    for o, d, mn, mx in ((ax, dx, x0, x1), (az, dz, z0, z1)):
        if abs(d) < 1e-9:
            if not (mn <= o <= mx):
                return False
            continue
        t1, t2 = (mn - o) / d, (mx - o) / d
        lo, hi = max(lo, min(t1, t2)), min(hi, max(t1, t2))
        if lo > hi:
            return False
    return True


if __name__ == "__main__":
    import pickle
    import time

    from baconhead.core.actions import KEYS
    TICKS = 2400   # 4 minutes: ~3.5 rounds
    for seed in range(4):
        env = NaturalDisaster(seed=seed, max_steps=TICKS)
        t0 = time.time()
        log = []
        while not env.done:
            env.step(env.expert())
            if env.phase == "end" and (not log or log[-1][0] != env.round):
                log.append((env.round, env.order[(env.round - 1) % 6], env.player.alive and env.in_round))
        ms = (time.time() - t0) / env.steps * 1000
        rnd = NaturalDisaster(seed=seed, max_steps=TICKS)
        rng = np.random.default_rng(seed)
        while not rnd.done:
            rnd.step(Action(keys=frozenset(k for k in KEYS if k not in ("esc", "lmb") and rng.random() < 0.12), look_x=float(rng.normal(0, 10))))
        print(f"seed {seed}: expert wins {env.score():.0f}/{env.round} deaths {env.deaths} {log}  |  random wins {rnd.score():.0f} deaths {rnd.deaths}  |  {ms:.1f} ms/tick")

    env = NaturalDisaster(seed=1)
    for _ in range(300):
        env.step(env.expert())
    clone = pickle.loads(pickle.dumps(env))
    for _ in range(50):
        a = env.expert()
        r1, r2 = env.step(a)[0], clone.step(a)[0]
        assert r1 == r2 and np.allclose(env.player.pos, clone.player.pos)
    print("pickle round-trip steps identically")
