"""Rivals: 1v1 first-person duels. First to 5 rounds wins the match; a new match starts right after.

You: first-person assault rifle. Hold left mouse for automatic fire (10 rounds/s), 30-round magazine, R to
reload (2 s), 20 body / 40 head, walls stop bullets, spread grows when you move or jump. The bot strafes
between cover, peeks, and shoots back after a reaction delay with accuracy that falls off with distance
and with your movement, so standing still in the open loses.
"""

from __future__ import annotations

import math

import numpy as np

from core.actions import LOOK_X, LOOK_Y, Action, _nearest
from sim.core import DT, HEAD_Y, Env, Humanoid, avatar_boxes, hotbar_hud, humanoid_step, ray_humanoids

MAG, RPS, RELOAD_S = 30, 10.0, 2.0
BODY_DMG, HEAD_DMG = 20.0, 40.0
BOT_RPS, BOT_REACT, BOT_MAG = 8.0, 0.35, 30
ROUNDS_TO_WIN, ROUND_GAP = 5, 2.0
SPAWN_X = 44.0


def _dir(yaw: float, pitch: float) -> np.ndarray:
    cp = math.cos(pitch)
    return np.array([math.sin(yaw) * cp, -math.sin(pitch), math.cos(yaw) * cp])


class Rivals(Env):
    name = "rivals"
    train_steps = 3000

    def build(self) -> None:
        r, w = self.rng, self.world
        self.max_steps = int(self.kw.get("max_steps", self.train_steps))
        self.respawn_s = 1e9   # rounds reset both players themselves
        w.add((0, -0.5, 0), (104, 1, 74), (0.55, 0.58, 0.62))
        for x in (-52, 52):
            w.add((x, 7, 0), (1, 14, 74), (0.3, 0.32, 0.36))
        for z in (-37, 37):
            w.add((0, 7, z), (104, 14, 1), (0.3, 0.32, 0.36))
        w.add((-SPAWN_X, 0.05, 0), (6, 0.1, 6), (1.0, 0.5, 0.1), neon=True, collide=False)
        w.add((SPAWN_X, 0.05, 0), (6, 0.1, 6), (0.2, 0.5, 1.0), neon=True, collide=False)
        # the raised centre platform with stairs on both z sides
        w.add((0, 2.5, 0), (16, 5, 16), (0.85, 0.55, 0.25))
        for s in (-1, 1):
            for k in range(1, 5):
                h = 1.25 * k
                w.add((0, h / 2, s * (8 + (4 - k) * 2 + 1)), (6, h, 2), (0.75, 0.48, 0.22))
        # cover, mirrored across x = 0 so neither spawn is favoured; jittered per seed
        layout = [(-30, 10, (4, 4, 4)), (-30, -12, (6, 3, 2)), (-20, 0, (2, 6, 8)), (-36, 22, (4, 4, 4)),
                  (-14, -22, (8, 3, 2)), (-24, -26, (4, 6, 4)), (-12, 18, (4, 4, 4)), (-40, -18, (2, 6, 6))]
        self.cover = []
        for x, z, size in layout:
            jx, jz = r.uniform(-3, 3), r.uniform(-3, 3)
            col = (0.55, 0.4, 0.25) if size[1] <= 4 else (0.42, 0.44, 0.48)
            for sx in (1, -1):
                cx, cz = sx * (x + jx), z + jz
                w.add((cx, size[1] / 2, cz), size, col)
                # where a fighter stands to use this cover: on the side facing their own spawn
                self.cover.append(np.array([cx + sx * (size[0] / 2 + 2.0), 0.0, cz]))

        self.bot = Humanoid(pos=np.array([SPAWN_X, 0.0, 0.0]), shirt=(0.8, 0.15, 0.15), pants=(0.15, 0.15, 0.18))
        self.wins = self.losses = 0                 # rounds, across all matches
        self.round_you = self.round_bot = 0         # this match
        self.matches_won = self.matches_lost = 0
        self.round_over_t = None
        self.banner = ""
        self.cam.first_person = True
        self._new_round()

    def respawn_point(self):
        return np.array([-SPAWN_X, 0.0, 0.0])

    def score(self) -> float:
        return float(self.wins - self.losses)

    # ── rounds ──
    def _new_round(self) -> None:
        p, b = self.player, self.bot
        p.pos, p.vel = self.respawn_point(), np.zeros(3)
        p.alive, p.health, p.stun, p.grounded = True, 100.0, 0.0, False
        b.pos, b.vel = np.array([SPAWN_X, 0.0, self.rng.uniform(-6, 6)]), np.zeros(3)
        b.alive, b.health, b.grounded = True, 100.0, False
        b.face = -math.pi / 2
        self.cam.yaw, self.cam.pitch = math.pi / 2, math.radians(3)
        p.face = self.cam.yaw
        self.ammo, self.reload_until, self.fire_cd = MAG, 0.0, 0.0
        self.bot_ammo, self.bot_reload_until, self.bot_fire_cd = BOT_MAG, 0.0, 0.0
        self.bot_seen, self.bot_react = 0.0, BOT_REACT + float(self.rng.uniform(0, 0.2))
        self.bot_goal, self.bot_retarget = b.pos.copy(), 0.0
        self.bot_phase = float(self.rng.uniform(0, 2 * math.pi))
        self.bot_stuck, self.bot_last = 0.0, b.pos.copy()
        self.hit_t = self.hurt_t = self.flash_t = -9.0
        self.hurt_side, self.kill_marker = 0.0, False
        self.round_over_t, self.banner = None, ""
        self._prev_r = False
        self._exp_stuck, self._exp_last = 0, p.pos.copy()

    def _end_round(self, you_won: bool) -> None:
        if self.round_over_t is not None:
            return
        self.round_over_t = self.t + ROUND_GAP
        if you_won:
            self.wins += 1
            self.round_you += 1
            self.reward += 1.0
            self.banner = "Round won!"
        else:
            self.losses += 1
            self.round_bot += 1
            self.reward -= 1.0
            self.banner = "Eliminated"
        if self.round_you >= ROUNDS_TO_WIN or self.round_bot >= ROUNDS_TO_WIN:
            won = self.round_you >= ROUNDS_TO_WIN
            self.matches_won += won
            self.matches_lost += not won
            self.banner = "VICTORY" if won else "DEFEAT"
            self.round_you = self.round_bot = 0

    def on_death(self) -> None:
        self._end_round(False)

    # ── shooting ──
    def _los(self, a: np.ndarray, b: np.ndarray) -> bool:
        d = b - a
        dist = float(np.linalg.norm(d))
        t, _ = self.world.raycast(a, d / max(dist, 1e-6), max_t=dist, mask=self.world.collide & self.world.visible)
        return not math.isfinite(t)

    def _eye(self) -> np.ndarray:
        return self.player.pos + np.array([0.0, HEAD_Y, 0.0])

    def _shoot(self) -> None:
        if self.fire_cd > 0 or self.ammo <= 0 or self.menu or self.round_over_t is not None:
            return
        p = self.player
        self.fire_cd = 1.0 / RPS
        self.ammo -= 1
        self.flash_t = self.t
        speed = math.hypot(p.vel[0], p.vel[2])
        spread = math.radians(4.0 if not p.grounded else (2.5 if speed > 2 else 0.6))
        yaw = self.cam.yaw + float(self.rng.normal(0, spread / 2))
        pitch = self.cam.pitch + float(self.rng.normal(0, spread / 2))
        o, d = self._eye(), _dir(yaw, pitch)
        wall, _ = self.world.raycast(o, d, max_t=300, mask=self.world.collide & self.world.visible)
        hit = ray_humanoids(o, d, [self.bot], max_t=wall)
        if hit is None:
            return
        dmg = HEAD_DMG if hit[2] else BODY_DMG
        b = self.bot
        b.health = max(0.0, b.health - dmg)
        self.hit_t, self.kill_marker = self.t, b.health <= 0
        self.reward += 0.1
        if b.health <= 0:
            b.alive, b.dead_t = False, self.t
            self._end_round(True)

    def on_tool_click(self) -> bool:
        self._shoot()
        return True

    # ── per substep ──
    def on_substep(self) -> None:
        if self.round_over_t is not None and self.t >= self.round_over_t:
            self._new_round()
        self.fire_cd = max(0.0, self.fire_cd - DT)
        r_now = "r" in self.held and not self.menu
        if r_now and not self._prev_r and self.ammo < MAG and self.t >= self.reload_until:
            self.reload_until = self.t + RELOAD_S
            self.ammo = -self.ammo - 1   # negative = mid-reload; refilled when the timer runs out
        self._prev_r = r_now
        if self.ammo < 0 and self.t >= self.reload_until:
            self.ammo = MAG
        if self.mouse_held and self.player.alive:
            self._shoot()
        if self.bot.alive and self.round_over_t is None:
            self._bot_step()
        if self.steps >= self.max_steps:
            self.done = True

    def _bot_step(self) -> None:
        b, p, r = self.bot, self.player, self.rng
        head = b.pos + np.array([0.0, HEAD_Y, 0.0])
        target = p.pos + np.array([0.0, 3.0, 0.0])
        los = p.alive and self._los(head, target)
        self.bot_seen = self.bot_seen + DT if los else 0.0
        to_p = (p.pos - b.pos)[[0, 2]]
        dist = float(np.linalg.norm(to_p))

        # move: hold a cover spot near the right range, peek sideways from it, push if it hasn't seen you in a while
        if self.t >= self.bot_retarget:
            def cost(c):
                return abs(float(np.linalg.norm((c - p.pos)[[0, 2]])) - (22 if self.bot_seen == 0 else 30)) + float(r.uniform(0, 8))
            self.bot_goal = min(self.cover, key=cost).copy()
            self.bot_retarget = self.t + float(r.uniform(2.5, 4.5))
        perp = np.array([-to_p[1], to_p[0]]) / max(dist, 1e-6)
        goal = self.bot_goal[[0, 2]] + perp * 4.0 * math.sin(2 * math.pi * self.t / 1.8 + self.bot_phase)
        move = goal - b.pos[[0, 2]]
        if np.linalg.norm(move) < 0.6:
            move = np.zeros(2)
        moved = float(np.linalg.norm((b.pos - self.bot_last)[[0, 2]]))
        self.bot_stuck = self.bot_stuck + DT if (np.linalg.norm(move) > 0 and moved < 0.05) else 0.0
        self.bot_last = b.pos.copy()
        face = math.atan2(to_p[0], to_p[1]) if los else None
        humanoid_step(b, self.world, move, self.bot_stuck > 0.3, face_to=face)

        # shoot
        self.bot_fire_cd = max(0.0, self.bot_fire_cd - DT)
        if self.bot_ammo == 0:
            self.bot_ammo, self.bot_reload_until = -1, self.t + RELOAD_S
        elif self.bot_ammo < 0 and self.t >= self.bot_reload_until:
            self.bot_ammo = BOT_MAG
        if los and self.bot_seen >= self.bot_react and self.bot_ammo > 0 and self.bot_fire_cd <= 0 and self.t >= self.bot_reload_until:
            self.bot_fire_cd = 1.0 / BOT_RPS
            self.bot_ammo -= 1
            p_hit = float(np.clip(0.75 - dist / 120, 0.12, 0.75))
            if math.hypot(p.vel[0], p.vel[2]) > 4:
                p_hit *= 0.55   # strafing works
            if np.linalg.norm(move) > 0:
                p_hit *= 0.8
            if r.random() < p_hit:
                dmg = HEAD_DMG if r.random() < 0.12 else BODY_DMG
                self.reward -= 0.05
                self.hurt_t = self.t
                self.hurt_side = math.atan2(-to_p[0], -to_p[1]) - self.cam.yaw   # where the shot came from, camera-relative
                self.damage(dmg, "Eliminated")

    # ── rendering ──
    def dynamic_boxes(self) -> list:
        out = []
        b = self.bot
        if b.alive or self.t - b.dead_t < 0.4:
            out += avatar_boxes(b, dead_for=0 if b.alive else self.t - b.dead_t)
            if b.alive:
                fx, fz, rx, rz = math.sin(b.face), math.cos(b.face), math.cos(b.face), -math.sin(b.face)
                out.append(((b.pos[0] + rx * 1.5 + fx * 1.2, b.pos[1] + 3.2, b.pos[2] + rz * 1.5 + fz * 1.2), (0.25, 0.3, 1.6), b.face, (0.1, 0.1, 0.1)))
        if self.player.alive:   # the first-person view model: rifle low and to the right
            c = self.cam
            f = _dir(c.yaw, c.pitch)
            right = np.array([math.cos(c.yaw), 0.0, -math.sin(c.yaw)])
            g = self._eye() + f * 1.3 + right * 0.7 + np.array([0.0, -0.6, 0.0])
            out.append((tuple(g), (0.16, 0.2, 0.8), c.yaw, (0.12, 0.12, 0.14)))
            if self.t - self.flash_t < 0.05:
                m = g + f * 0.5
                out.append((tuple(m), (0.25, 0.25, 0.25), c.yaw, (1.0, 0.85, 0.3), 1))
        return out

    def hud(self) -> list:
        p, b = self.player, self.bot
        els = [dict(kind="panel", rect=(0.38, 0.015, 0.24, 0.1), color=(0.08, 0.08, 0.1), alpha=0.55),
               dict(kind="text", rect=(0.38, 0.02, 0.24, 0.05), text=f"You {self.round_you}  -  {self.round_bot} Enemy", size=26, align="center"),
               dict(kind="bar", rect=(0.395, 0.075, 0.1, 0.02), frac=p.health / 100, color=(0.3, 0.9, 0.35)),
               dict(kind="bar", rect=(0.505, 0.075, 0.1, 0.02), frac=b.health / 100, color=(0.95, 0.25, 0.25))]
        # crosshair: four short ticks around the centre
        for x, y, w_, h_ in ((0.488, 0.4985, 0.008, 0.003), (0.504, 0.4985, 0.008, 0.003), (0.4985, 0.478, 0.0017, 0.014), (0.4985, 0.508, 0.0017, 0.014)):
            els.append(dict(kind="panel", rect=(x, y, w_, h_), color=(1, 1, 1), alpha=0.9))
        if self.t - self.hit_t < 0.15:
            els.append(dict(kind="text", rect=(0.47, 0.465, 0.06, 0.07), text="X", size=30, align="center",
                            color=(1, 0.2, 0.2) if self.kill_marker else (1, 1, 1)))
        if self.t - self.hurt_t < 0.4:   # red flash on the side the shot came from
            a = (self.hurt_side + math.pi) % (2 * math.pi) - math.pi
            if abs(a) < math.pi / 4:
                rect = (0.3, 0.12, 0.4, 0.04)
            elif abs(a) > 3 * math.pi / 4:
                rect = (0.3, 0.84, 0.4, 0.04)
            else:
                rect = (0.93, 0.25, 0.05, 0.5) if a > 0 else (0.02, 0.25, 0.05, 0.5)
            els.append(dict(kind="panel", rect=rect, color=(0.9, 0.05, 0.05), alpha=0.55))
        reloading = self.ammo < 0
        els.append(dict(kind="text", rect=(0.78, 0.86, 0.2, 0.07), text="Reloading…" if reloading else f"{self.ammo} / {MAG}",
                        size=30, align="right", color=(1, 0.8, 0.3) if self.ammo <= 5 else (1, 1, 1)))
        els += hotbar_hud(["Assault Rifle"], 0)
        if self.banner:
            els.append(dict(kind="text", rect=(0.3, 0.3, 0.4, 0.1), text=self.banner, size=44, align="center",
                            color=(0.4, 1, 0.5) if self.banner in ("Round won!", "VICTORY") else (1, 0.35, 0.35)))
        return els

    def info(self) -> dict:
        return dict(rounds_won=self.wins, rounds_lost=self.losses, matches_won=self.matches_won, matches_lost=self.matches_lost,
                    health=self.player.health, enemy_health=self.bot.health, ammo=max(self.ammo, 0))

    # ── shaping and expert ──
    def potential(self) -> float:
        p, b = self.player, self.bot
        if not (p.alive and b.alive) or self.round_over_t is not None:
            return 0.0
        phi = 0.005 * (p.health - b.health)
        tgt = b.pos + np.array([0.0, 3.0, 0.0])
        rel = tgt - self._eye()
        ang = math.degrees(math.acos(float(np.clip(_dir(self.cam.yaw, self.cam.pitch) @ rel / np.linalg.norm(rel), -1, 1))))
        if ang < 5 and self._los(self._eye(), tgt):
            phi += 0.2
        return phi

    def expert(self) -> Action:
        p, b = self.player, self.bot
        if self.menu:
            self.say("menu is open, close it")
            return Action(keys=frozenset({"esc"}))
        if not p.alive or not b.alive or self.round_over_t is not None:
            self.say("eliminated, wait for the next round" if not p.alive else "round won, wait for the next round")
            return Action()
        eye = self._eye()
        flat = float(np.linalg.norm((b.pos - p.pos)[[0, 2]]))
        tgt = b.pos + np.array([0.0, 4.6 if flat < 18 else 2.8, 0.0]) + b.vel * 0.06   # head up close, chest at range, small lead
        rel = tgt - eye
        dist = float(np.linalg.norm(rel))
        yaw_t = math.atan2(rel[0], rel[2])
        pitch_t = -math.asin(float(np.clip(rel[1] / dist, -1, 1)))
        ey = math.degrees((yaw_t - self.cam.yaw + math.pi) % (2 * math.pi) - math.pi)
        ep = math.degrees(pitch_t - self.cam.pitch)
        lx = LOOK_X[_nearest(LOOK_X, float(np.clip(ey, -30, 30)))]
        ly = LOOK_Y[_nearest(LOOK_Y, float(np.clip(ep, -10, 10)))] if abs(ep) > 2.5 else 0.0
        los = self._los(eye, tgt)
        tol = math.degrees(math.atan(0.9 / max(dist, 1.0)))
        on = abs(ey - lx) < tol and abs(ep - ly) < max(tol * 2, 2.5)
        keys = set()
        reloading = self.ammo < 0
        if not reloading and (self.ammo == 0 or (not los and self.ammo < 15)):
            keys.add("r")
        if los:
            keys.add("a" if int(self.t / 0.6) % 2 == 0 else "d")   # strafe while shooting
            if flat > 35:
                keys.add("w")
            if on and self.ammo > 0 and not reloading:
                keys.add("lmb")
        else:
            keys.add("w")
        moved = float(np.linalg.norm((p.pos - self._exp_last)[[0, 2]]))
        self._exp_stuck = self._exp_stuck + 1 if ("w" in keys and moved < 0.3) else 0
        self._exp_last = p.pos.copy()
        if self._exp_stuck >= 3:
            keys.add("space")
            if self._exp_stuck >= 8:   # still stuck behind cover: sidestep around it
                keys.add("d" if (self._exp_stuck // 10) % 2 == 0 else "a")
        if "r" in keys and self.ammo == 0:
            self.say("out of ammo, reload")
        elif self._exp_stuck >= 3:
            self.say("stuck on cover, jump and go around it")
        elif los and reloading:
            self.say("reloading, strafe until the magazine is full")
        elif los and "lmb" in keys:
            self.say("enemy in sight, strafe and fire")
        elif los:
            self.say("enemy in sight, strafe and aim at them")
        elif "r" in keys:
            self.say("enemy out of sight, reload while pushing")
        else:
            self.say("enemy out of sight, push forward to find them")
        return Action(keys=frozenset(keys), look_x=lx, look_y=ly, cursor=(0.5, 0.5))


if __name__ == "__main__":
    import os
    import pickle
    import time
    from train.eval import _random_action

    ticks = 1800   # 3 minutes
    for seed in (0, 1, 2):
        env = Rivals(seed=seed, max_steps=10 ** 9)
        t0 = time.time()
        for _ in range(ticks):
            env.step(env.expert())
        ms = (time.time() - t0) / ticks * 1000
        rnd = Rivals(seed=seed, max_steps=10 ** 9)
        rng = np.random.default_rng(seed)
        for _ in range(ticks):
            rnd.step(_random_action(rng))
        print(f"seed {seed}: expert score {env.score():+.0f} {env.info()} {ms:.1f} ms/tick | random score {rnd.score():+.0f} "
              f"rounds {rnd.wins}-{rnd.losses}")
        assert env.score() > rnd.score() and env.wins > env.losses

    # pickle mid-episode, both copies step identically
    env = Rivals(seed=5, max_steps=10 ** 9)
    for _ in range(150):
        env.step(env.expert())
    twin = pickle.loads(pickle.dumps(env))
    for _ in range(100):
        a = env.expert()
        assert twin.expert() == a
        env.step(a)
        twin.step(a)
    assert np.allclose(env.player.pos, twin.player.pos) and env.info() == twin.info()
    print("pickle ok")

    if os.environ.get("RENDER"):
        from PIL import Image
        from sim.render import render
        os.makedirs("/tmp/bh/rivals", exist_ok=True)
        env = Rivals(seed=0, max_steps=10 ** 9)
        shots = {}
        for i in range(900):
            env.step(env.expert())
            if i == 3:
                shots["spawn"] = env.scene()
            if "fight" not in shots and env.t - env.hit_t < 0.1 and env.bot.alive:
                shots["fight"] = env.scene()
            if "hurt" not in shots and env.t - env.hurt_t < 0.1:
                shots["hurt"] = env.scene()
            if "banner" not in shots and env.banner:
                shots["banner"] = env.scene()
        for k, s in shots.items():
            Image.fromarray(render([s], 360, 640, device="mps").cpu().numpy()[0]).save(f"/tmp/bh/rivals/{k}.png")
        print("rendered", list(shots))
