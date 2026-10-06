"""99 Nights in the Forest: keep the campfire alive, stay fed, and stay in its light when The Deer comes.

Loop, as in the real game: by day, equip the axe (slot 1), chop trees (click or hold left mouse next to a
trunk; 4 hits fell it and drop a log), pick logs up into your sack (3 max), and hold E "Add Fuel" at the
campfire. Fuel drains constantly; the fire's light radius scales with fuel. Eat berries (E) before hunger
runs out. At night The Deer stalks anyone outside the light and mauls on contact; inside the light it
can only circle. Survive the night → +1.
"""

from __future__ import annotations

import math

import numpy as np

from core.actions import Action
from sim.core import DT, Env, Humanoid, Prompt, health_hud, hotbar_hud, night_theme, npc_walk

DAY_S, NIGHT_S = 60.0, 45.0
CYCLE = DAY_S + NIGHT_S
FUEL_MAX, FUEL_PER_LOG = 100.0, 22.0
FUEL_DRAIN = 1.0                 # per second; a full fire lasts ~100 s
HUNGER_DRAIN = 0.55              # per second; full to empty in ~3 minutes
SACK = 3
TREE_HITS, AXE_CD, AXE_RANGE = 4, 0.45, 4.5
REGROW_S = 150.0                 # ponytail: felled trees regrow; the real map is just bigger
DEER_SPEED, DEER_HIT, DEER_CD = 0.85, 35.0, 2.0   # speed as a fraction of WalkSpeed: you can outrun it, barely
FIRE = np.array([0.0, 0.0, 0.0])


def light_radius(fuel: float) -> float:
    return 0.0 if fuel <= 0 else 6.0 + 0.3 * fuel


class NinetyNineNights(Env):
    name = "forest"
    train_steps = 3000
    eval_kw: dict = {}

    def build(self) -> None:
        r, w = self.rng, self.world
        self.max_steps = int(self.kw.get("max_steps", self.train_steps))
        w.add((0, -0.5, 0), (320, 1, 320), (0.22, 0.42, 0.18))                                     # forest floor
        w.add((0, 0.02, 0), (16, 0.04, 16), (0.45, 0.36, 0.25))                                  # trampled camp clearing
        for ang in np.linspace(0, 2 * math.pi, 8, endpoint=False):                                # stone ring
            w.add((math.cos(ang) * 2.2, 0.3, math.sin(ang) * 2.2), (0.9, 0.6, 0.9), (0.5, 0.5, 0.52), collide=False)
        self.flame = w.add((0, 0.8, 0), (1.6, 1.6, 1.6), (1.0, 0.55, 0.1), neon=True, collide=False)
        w.add((5, 0.6, 3), (3, 1.2, 1), (0.45, 0.3, 0.18))                                        # a bench log
        for x in (-160, 160):
            w.add((x, 20, 0), (1, 40, 320), (0, 0, 0), visible=False)
        for z in (-160, 160):
            w.add((0, 20, z), (320, 40, 1), (0, 0, 0), visible=False)

        self.trees = []          # [trunk, canopy, hp, pos, regrow time]
        while len(self.trees) < 70:
            p = r.uniform(-130, 130, 2)
            if np.hypot(*p) < 22 or any(np.hypot(*(p - t[3][[0, 2]])) < 6 for t in self.trees):
                continue
            hgt = float(r.uniform(12, 18))
            trunk = w.add((p[0], hgt / 2, p[1]), (1.8, hgt, 1.8), (0.4, 0.27, 0.15))
            leaf = tuple(np.clip(np.array([0.15, 0.45, 0.15]) + r.normal(0, 0.05, 3), 0, 1))
            canopy = w.add((p[0], hgt + 2, p[1]), (8, 6, 8), leaf, collide=False)
            self.trees.append([trunk, canopy, TREE_HITS, np.array([p[0], 0.0, p[1]]), 0.0])
        for _ in range(25):                                                                         # rocks
            p = r.uniform(-140, 140, 2)
            if np.hypot(*p) > 15:
                s = float(r.uniform(1.5, 4))
                w.add((p[0], s / 3, p[1]), (s, s / 1.5, s), (0.45, 0.45, 0.47), yaw=float(r.uniform(0, 3)))

        self.bushes = []         # [berries part, regrow time, pos]
        for k in range(12):
            ang, dist = r.uniform(0, 2 * math.pi), r.uniform(14, 70)
            p = np.array([math.cos(ang) * dist, 0.0, math.sin(ang) * dist])
            w.add((p[0], 1.0, p[2]), (3, 2, 3), (0.12, 0.38, 0.14))
            berries = w.add((p[0], 2.1, p[2]), (2.2, 0.4, 2.2), (0.8, 0.1, 0.25), collide=False)
            self.bushes.append([berries, 0.0, p])
        self.prompts = [Prompt(pos=FIRE + [0, 1.5, 0], action="Add Fuel", obj="Campfire", hold=0.3,
                               on_trigger=("_add_fuel",), enabled=("_has_logs",))]
        self.prompts += [Prompt(pos=b[2] + [0, 2, 0], action="Eat", obj="Berry Bush", hold=0.5,
                                on_trigger=("_eat", k), enabled=("_bush_ready", k)) for k, b in enumerate(self.bushes)]
        self.logs: dict[int, np.ndarray] = {}          # id → position of a log lying on the ground
        self._log_id = 0
        self.sack = 0
        self.fuel = 70.0
        self.hunger = 100.0
        self.axe_cd = 0.0
        self.swing_t = -9.0
        self.nights = 0
        self.night_now = False
        self.died_tonight = False
        self.deer = Humanoid(pos=np.array([140.0, 0.0, 140.0]), color=(0.1, 0.08, 0.08), shirt=(0.1, 0.08, 0.08),
                             pants=(0.1, 0.08, 0.08), speed_mult=DEER_SPEED)
        self.deer_cd = 0.0
        self.goal = None
        self.cam.yaw = math.pi / 4
        self.cam.pitch = math.radians(20)

    def respawn_point(self):
        return np.array([4.0, 0.0, 4.0])

    # ── time of day ──
    def phase(self) -> float:
        return self.t % CYCLE

    def is_night(self) -> bool:
        return self.phase() >= DAY_S

    def day_num(self) -> int:
        return int(self.t // CYCLE) + 1

    def _dark(self) -> float:
        """0 day → 1 night, with 6 s dusk and dawn."""
        ph = self.phase()
        if ph < DAY_S - 6:
            return 0.0
        if ph < DAY_S:
            return (ph - (DAY_S - 6)) / 6
        if ph > CYCLE - 6:
            return (CYCLE - ph) / 6
        return 1.0

    def in_light(self, p) -> bool:
        return float(np.hypot(*(p - FIRE)[[0, 2]])) < light_radius(self.fuel)

    def scene_theme(self) -> dict:
        lr = light_radius(self.fuel)
        lights = [(0.0, 2.0, 0.0, lr * 1.3, 1.6, 0.85, 0.35)] if lr > 0 else []
        return night_theme(self.theme, 0.97 * self._dark(), lights=lights)

    # ── prompts ──
    def _has_logs(self) -> bool:
        return self.sack > 0

    def _add_fuel(self) -> None:
        self.fuel = min(FUEL_MAX, self.fuel + FUEL_PER_LOG * self.sack)
        self.toast(f"Added {self.sack} log{'s' * (self.sack > 1)} to the fire")
        self.reward += 0.05 * self.sack
        self.sack = 0

    def _bush_ready(self, k: int) -> bool:
        return self.t >= self.bushes[k][1]

    def _eat(self, k: int) -> None:
        self.hunger = min(100.0, self.hunger + 40)
        self.bushes[k][1] = self.t + 45.0
        self.world.visible0[self.bushes[k][0]] = self.world.visible[self.bushes[k][0]] = False
        self.reward += 0.05

    def _pick_log(self, lid: int) -> None:
        if lid in self.logs and self.sack < SACK:
            del self.logs[lid]
            self.sack += 1

    def _sack_room(self) -> bool:
        return self.sack < SACK

    # ── axe ──
    def on_key(self, k: str) -> None:
        if k == "1":
            self.equipped = None if self.equipped == "axe" else "axe"

    def on_ui(self, id_: str) -> None:
        if id_ == "slot:0":
            self.equipped = None if self.equipped == "axe" else "axe"

    def on_tool_click(self) -> bool:
        if self.equipped != "axe":
            return False
        self._swing()
        return True

    def _swing(self) -> None:
        if self.axe_cd > 0:
            return
        self.axe_cd, self.swing_t = AXE_CD, self.t
        tr = self.target_tree()
        if tr is None:
            return
        tr[2] -= 1
        if tr[2] > 0:
            return
        tr[4] = self.t + REGROW_S
        w = self.world
        for i in tr[:2]:
            w.visible0[i] = w.visible[i] = w.collide0[i] = w.collide[i] = False
        p = tr[3] + self.rng.uniform(-1.5, 1.5, 3) * [1, 0, 1]
        lid = self._log_id
        self._log_id += 1
        self.logs[lid] = p
        self.prompts.append(Prompt(pos=p + [0, 1, 0], action="Pick Up", obj="Log", hold=0.0,
                                   on_trigger=("_pick_log", lid), enabled=("_sack_room",)))
        self.toast("Timber!", 1.2)

    def target_tree(self):
        """The standing tree in reach, in front of you (body facing or camera)."""
        p = self.player.pos
        best, bd = None, AXE_RANGE
        for tr in self.trees:
            if tr[2] <= 0:
                continue
            d = (tr[3] - p)[[0, 2]]
            dist = float(np.hypot(*d)) - 0.9
            if dist < bd:
                best, bd = tr, dist
        return best

    # ── simulation ──
    def on_substep(self) -> None:
        p = self.player
        self.axe_cd = max(0.0, self.axe_cd - DT)
        if self.mouse_held and self.equipped == "axe" and p.alive:   # holding the button keeps swinging
            self._swing()
        self.fuel = max(0.0, self.fuel - FUEL_DRAIN * DT)
        self.hunger = max(0.0, self.hunger - HUNGER_DRAIN * DT)
        if self.hunger <= 0:
            self.damage(3.0 * DT, "You starved")
        elif self.hunger > 60 and p.alive:
            p.health = min(100.0, p.health + 0.6 * DT)
        for tr in self.trees:
            if tr[2] <= 0 and self.t >= tr[4]:
                tr[2] = TREE_HITS
                for i in tr[:2]:
                    self.world.visible0[i] = self.world.visible[i] = True
                    self.world.collide0[i] = self.world.collide[i] = i == tr[0]
        for b in self.bushes:
            if self.t >= b[1]:
                self.world.visible0[b[0]] = self.world.visible[b[0]] = True
        self.prompts = [pr for pr in self.prompts if pr.on_trigger[0] != "_pick_log" or pr.on_trigger[1] in self.logs]
        fl = self.world
        s = 0.4 + 1.4 * self.fuel / FUEL_MAX
        fl.h[self.flame] = np.array([s, s, s]) * 0.6
        fl.c[self.flame] = fl.c0[self.flame] = np.array([0, 0.3 + 0.6 * s, 0])
        fl.visible[self.flame] = fl.visible0[self.flame] = self.fuel > 0

        night = self.is_night()
        if night and not self.night_now:
            self.toast(f"Night {self.day_num()} falls. Stay in the light!", 3)
            self.died_tonight = not p.alive
        if night and not p.alive:
            self.died_tonight = True
        if self.night_now and not night:
            if self.died_tonight:   # a night only counts if you lived through all of it
                self.toast("You didn't make it through the night", 3)
            else:
                self.nights += 1
                self.reward += 1.0
                self.toast(f"You survived Night {self.day_num() - 1}!", 3)
        self.night_now = night
        self._deer()
        if self.steps >= self.max_steps:
            self.done = True

    def _deer(self) -> None:
        d, p = self.deer, self.player
        self.deer_cd = max(0.0, self.deer_cd - DT)
        if not self.is_night():   # by day it lurks at the far edge of the forest
            far = np.array([135.0, 0.0, 135.0])
            if np.hypot(*(d.pos - far)[[0, 2]]) > 30:
                d.pos = far.copy()
                d.vel[:] = 0
            return
        lr = light_radius(self.fuel)
        if p.alive and not self.in_light(p.pos):
            goal = p.pos
        else:   # circle just outside the light, waiting
            ang = math.atan2(*(d.pos - FIRE)[[0, 2]][::-1]) + 0.4
            goal = FIRE + np.array([math.cos(ang), 0, math.sin(ang)]) * (lr + 4)
        npc_walk(d, self.world, goal, stop=0.5)
        dd = float(np.hypot(*(d.pos - FIRE)[[0, 2]]))
        if dd < lr:   # the light pushes it back out
            n = (d.pos - FIRE)[[0, 2]] / max(dd, 1e-6)
            d.pos[[0, 2]] = FIRE[[0, 2]] + n * lr
        if p.alive and self.deer_cd <= 0 and np.hypot(*(d.pos - p.pos)[[0, 2]]) < 2.6:
            self.deer_cd = DEER_CD
            self.damage(DEER_HIT, "The Deer got you")
            n = (p.pos - d.pos)[[0, 2]]
            n = n / max(np.hypot(*n), 1e-6)
            p.vel[0], p.vel[2], p.vel[1] = n[0] * 35, n[1] * 35, 25
            p.grounded = False
            self.toast("The Deer attacked you!", 1.5)

    def on_death(self) -> None:
        self.reward -= 1.0
        self.sack = 0

    def respawn(self) -> None:
        super().respawn()
        self.hunger = max(self.hunger, 60.0)

    # ── looks ──
    def dynamic_boxes(self) -> list:
        out = []
        for p in self.logs.values():
            out.append(((p[0], 0.45, p[2]), (0.9, 0.9, 3.2), 0.6, (0.48, 0.3, 0.16)))
        d = self.deer
        if self.is_night() or self._dark() > 0:
            f = d.face
            fx, fz = math.sin(f), math.cos(f)
            rx, rz = math.cos(f), -math.sin(f)
            x, z = d.pos[0], d.pos[2]
            dark = (0.07, 0.06, 0.06)
            for sx, sf in ((-0.7, 0.9), (0.7, 0.9), (-0.7, -0.9), (0.7, -0.9)):
                out.append(((x + rx * sx + fx * sf, 2.2, z + rz * sx + fz * sf), (0.45, 4.4, 0.45), f, dark))
            out.append(((x, 5.2, z), (2.0, 1.8, 3.2), f, dark))
            out.append(((x + fx * 2.0, 7.0, z + fz * 2.0), (1.1, 2.6, 1.1), f, dark))
            out.append(((x + fx * 2.3, 8.6, z + fz * 2.3), (1.3, 1.1, 1.6), f, dark))
            for s in (-1, 1):
                out.append(((x + rx * s * 0.9 + fx * 2.1, 10.0, z + rz * s * 0.9 + fz * 2.1), (0.25, 2.2, 0.25), f, (0.85, 0.82, 0.75)))
                out.append(((x + rx * s * 0.35 + fx * 3.0, 8.8, z + rz * s * 0.35 + fz * 3.0), (0.2, 0.2, 0.1), f, (1.0, 0.1, 0.1), 1))
        if self.equipped == "axe":
            h = self.player
            u = (self.t - self.swing_t) / 0.3
            ang = h.face + (0.9 - 1.8 * u if 0 <= u < 1 else 0.5)
            ox, oz = math.sin(ang), math.cos(ang)
            out.append(((h.pos[0] + ox * 2.0, h.pos[1] + 3.0, h.pos[2] + oz * 2.0), (0.3, 0.3, 2.2), ang, (0.5, 0.35, 0.2)))
            out.append(((h.pos[0] + ox * 3.0, h.pos[1] + 3.0, h.pos[2] + oz * 3.0), (0.9, 0.7, 0.2), ang, (0.7, 0.7, 0.75)))
        return out

    def hud(self) -> list:
        night = self.is_night()
        left = (CYCLE - self.phase()) if night else (DAY_S - self.phase())
        els = [dict(kind="text", rect=(0.35, 0.02, 0.3, 0.06), text=f"{'Night' if night else 'Day'} {self.day_num()}", size=32, align="center",
                    color=(0.75, 0.8, 1.0) if night else (1.0, 0.95, 0.7)),
               dict(kind="text", rect=(0.35, 0.075, 0.3, 0.04), text=f"{'dawn' if night else 'night'} in {left:.0f}s", size=16, align="center"),
               dict(kind="leaderboard", cols=["Nights"], rows=[["You", self.nights]]),
               dict(kind="text", rect=(0.02, 0.74, 0.2, 0.04), text="Hunger", size=16),
               dict(kind="bar", rect=(0.02, 0.785, 0.18, 0.022), frac=self.hunger / 100, color=(0.95, 0.6, 0.2)),
               dict(kind="text", rect=(0.02, 0.62, 0.2, 0.04), text=f"Sack: {self.sack}/{SACK} logs", size=18)]
        els += health_hud(self.player.health)
        els += hotbar_hud(["Axe"], 0 if self.equipped == "axe" else None)
        b = self.billboard(FIRE + [0, 4.5, 0], f"Fuel {self.fuel:.0f}%", color=(1, 0.75, 0.4), max_dist=120)
        if b:
            els.append(b)
        return els

    def info(self) -> dict:
        return dict(nights=self.nights, day=self.day_num(), fuel=round(self.fuel), hunger=round(self.hunger),
                    health=round(self.player.health), deaths=self.deaths, sack=self.sack)

    def score(self) -> float:
        return float(self.nights)

    def potential(self) -> float:
        p = self.player
        phi = 0.6 * self.fuel / FUEL_MAX + 0.3 * self.hunger / 100 + 0.5 * p.health / 100 + 0.1 * self.sack
        if self.is_night():
            phi += 0.5 * self.in_light(p.pos)
        self.goal = None
        self.expert()
        if self.goal is not None:
            phi -= 0.004 * float(np.hypot(*(self.goal - p.pos)[[0, 2]]))
        return phi

    # ── expert: chop, haul, feed the fire, eat, and never be outside the light at night ──
    def expert(self) -> Action:
        p = self.player
        if not p.alive:
            return Action(cursor=self.cursor)
        ph = self.phase()
        night_soon = self.is_night() or ph > DAY_S - 8
        fire_d = float(np.hypot(*(p.pos - FIRE)[[0, 2]]))
        lr = light_radius(self.fuel)
        # deliver logs whenever the sack is full, the fire is low, or night is coming
        if self.sack > 0 and (self.sack >= SACK or self.fuel < 60 or night_soon):
            return self._go_prompt(FIRE, "_add_fuel")
        if night_soon:   # wait it out near the fire
            if fire_d > max(2.5, 0.4 * lr):
                return self._go(FIRE + [2.5, 0, 2.5], stop=1.0)
            return Action(cursor=self.cursor)
        if self.hunger < 50:
            ready = [k for k, b in enumerate(self.bushes) if self.t >= b[1]]
            if ready:
                k = min(ready, key=lambda k: np.hypot(*(self.bushes[k][2] - p.pos)[[0, 2]]))
                return self._go_prompt(self.bushes[k][2], "_eat")
        if self.logs and self.sack < SACK:
            lid = min(self.logs, key=lambda i: np.hypot(*(self.logs[i] - p.pos)[[0, 2]]))
            return self._go_prompt(self.logs[lid], "_pick_log")
        standing = [tr for tr in self.trees if tr[2] > 0]
        if not standing:
            return self._go(FIRE + [2.5, 0, 2.5])
        tr = min(standing, key=lambda tr: np.hypot(*(tr[3] - p.pos)[[0, 2]]))
        if self.equipped != "axe":
            a = self._go(tr[3], stop=2.6)
            a.keys = a.keys | {"1"} if "1" not in self.held else a.keys
            return a
        a = self._go(tr[3], stop=2.6)
        if np.hypot(*(tr[3] - p.pos)[[0, 2]]) - 0.9 < AXE_RANGE - 0.5:
            a.keys = frozenset({"lmb"})
        return a

    def _go(self, goal, stop=1.5) -> Action:
        """Walk at goal, detouring around trunks and rocks a straight line would push into."""
        self.goal = np.asarray(goal, float)
        p = self.player.pos
        d = (self.goal - p)[[0, 2]]
        dist = float(np.hypot(*d))
        if dist > stop + 1:
            base = math.atan2(d[0], d[1])
            o = p + [0, 1.2, 0]
            for off in (0.0, 0.6, -0.6, 1.1, -1.1, 1.6, -1.6):
                ang = base + off
                ray = np.array([math.sin(ang), 0.0, math.cos(ang)])
                t, _ = self.world.raycast(o, ray, max_t=min(3.5, dist), mask=self.world.collide)
                if not math.isfinite(t):
                    if off:
                        return self.steer_to(p + ray * 4, stop_dist=0.5)
                    break
        return self.steer_to(self.goal, stop_dist=stop)

    def _go_prompt(self, where, fn: str) -> Action:
        """Walk up to a prompt and hold E on it, letting go once it fires so the next one can."""
        a = self._go(where, stop=2.0)
        pr = self.active_prompt()
        if pr is not None and pr.on_trigger[0] == fn:
            a.keys = frozenset({"e"}) if pr.progress >= 0 else frozenset()
        return a


if __name__ == "__main__":
    import pickle
    import sys
    import time
    from core.actions import KEYS
    ticks = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    for seed in (0, 1, 2):
        env = NinetyNineNights(seed=seed, max_steps=ticks)
        t0 = time.time()
        while not env.done:
            env.step(env.expert())
            if env.steps == ticks // 2:   # pickling mid-episode must reproduce the run exactly
                twin = pickle.loads(pickle.dumps(env))
                a = env.expert()
                r1, _, i1 = env.step(a)
                r2, _, i2 = twin.step(a)
                assert r1 == r2 and i1 == i2, (i1, i2)
        ms = (time.time() - t0) / env.steps * 1000
        print(f"seed {seed} expert: score {env.score():.0f}  {env.info()}  {ms:.1f} ms/tick")
        env = NinetyNineNights(seed=seed, max_steps=ticks)
        rng = np.random.default_rng(seed)
        while not env.done:
            env.step(Action(keys=frozenset(k for k in KEYS if k != "esc" and rng.random() < 0.1), look_x=float(rng.normal(0, 10)),
                            cursor=tuple(rng.random(2)), click=rng.random() < 0.05))
        print(f"seed {seed} random: score {env.score():.0f}  {env.info()}")
