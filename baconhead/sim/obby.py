import copy
import math

import numpy as np

from baconhead.core.actions import Action
from baconhead.sim.core import (CHAR_RADIUS, DT, GRAVITY, JUMP_V, SUBSTEPS, WALK_SPEED, Env, Motion,
                      humanoid_step, forward_xz, to_local, to_world)

PALETTE = [(0.77, 0.16, 0.11), (0.05, 0.41, 0.67), (0.96, 0.8, 0.19), (0.29, 0.59, 0.29), (0.85, 0.52, 0.25),
           (0.42, 0.2, 0.6), (0.97, 0.97, 0.97), (0.07, 0.07, 0.07), (0.0, 0.56, 0.61), (0.86, 0.36, 0.67)]
KILL = (1.0, 0.1, 0.05)
CP_GREY = (0.64, 0.64, 0.66)


# max edge-to-edge gap when landing dh studs higher, times a safety margin, plus both hitbox radii
def reach(dh, margin=0.8):
    disc = JUMP_V ** 2 - 2 * GRAVITY * dh
    if disc <= 0:
        return 0.0
    t = (JUMP_V + math.sqrt(disc)) / GRAVITY
    return margin * WALK_SPEED * t + 2 * CHAR_RADIUS * 0.6


class Obby(Env):
    name = "obby"
    train_steps = 1500
    eval_kw = {"difficulty": 0.5, "stages": 20, "respawn": 2.0}

    def score(self):
        return float(self.best_stage)       # stages cleared

    def build(self):
        r, w = self.rng, self.world
        self.difficulty = float(self.kw.get("difficulty", 0.5))
        self.n_stages = int(self.kw.get("stages", 20))
        self.respawn_s = float(self.kw.get("respawn", 2.0))     # most obbies shorten RespawnTime; Roblox default is 5
        self.max_steps = int(self.kw.get("max_steps", 3000))
        self.waypoints = []            # (part, local xz offset on its top face)
        self.cps = []                                # checkpoint part per stage
        self.cp_wp = []                              # waypoint index of each checkpoint

        if r.random() < 0.5:   # half the obbies float over lava, half over void
            w.add((0, -30, 0), (4000, 2, 4000), (1.0, 0.35, 0.05), kill=True, neon=True)
        spawn = w.add((0, -0.5, 0), (14, 1, 14), CP_GREY)
        w.add((0, 0.05, 0), (5, 0.1, 5), (0.2, 0.8, 0.3), neon=True, collide=False)
        self._cp(spawn)
        pos, heading = np.array([0.0, 0.0, 7.0]), 0.0     # pos = top-centre of the far edge of the last surface
        for s in range(1, self.n_stages + 1):
            d = min(1.0, self.difficulty * (0.6 + 0.8 * s / self.n_stages))   # ramps within a run, like real obbies
            kinds = ["jumps", "stairs", "lava"] + (["beam", "vanish"] if d > 0.25 else []) + (["moving", "spinner"] if d > 0.4 else [])
            kind = kinds[r.integers(len(kinds))]
            pos = getattr(self, "_seg_" + kind)(pos, heading, d)
            pos = pos + np.r_[forward_xz(heading)[0], 0, forward_xz(heading)[1]] * reach(0, 0.5) * r.uniform(0.3, 0.7)
            cp = w.add(self._at(pos, heading, 0, 3, -0.5), (6, 1, 6), CP_GREY, yaw=heading)
            w.add(self._at(pos, heading, 0, 3, 0.05), (4, 0.1, 4), (0.2, 0.8, 0.3) if s < self.n_stages else (1, 0.85, 0.1),
                  neon=True, collide=False, yaw=heading)
            self._cp(cp)
            pos = self._at(pos, heading, 0, 6, 0)
            if r.random() < 0.35:   # a corner makes you use the camera; headings stay within ±90° so the course never crosses itself
                heading = 0.0 if heading != 0.0 else float(r.choice([-1, 1]) * math.pi / 2)

        self.stage = 0
        self.best_wp = 0
        self.best_stage = 0
        self.plan = []
        self.cam.yaw = 0.0
        self.cam.pitch = math.radians(25)
        self._dist_prev = None

    # ── geometry helpers ──
    def _at(self, pos, heading, side, fwd, up):
        f = forward_xz(heading)
        r = np.array([math.cos(heading), -math.sin(heading)])
        return np.array([pos[0] + f[0] * fwd + r[0] * side, pos[1] + up, pos[2] + f[1] * fwd + r[1] * side])

    def _plat(self, top_center, size, heading, color=None, **kw):
        c = np.array(top_center, float) - [0, size[1] / 2, 0]
        col = color if color is not None else PALETTE[self.rng.integers(len(PALETTE))]
        i = self.world.add(c, size, col, yaw=heading, **kw)
        return i

    def _wp(self, part, off=(0.0, 0.0)):
        self.waypoints.append((part, off))

    def _cp(self, part):
        self.cps.append(part)
        self._wp(part)
        self.cp_wp.append(len(self.waypoints) - 1)

    def _gap(self, d, dh=0.0):
        g = self.rng.uniform(2.0 + 3 * d, 3.0 + 6.5 * d)
        return min(g, reach(dh))

    # ── segments; each takes the far-edge point of the previous surface and returns its own ──
    def _seg_jumps(self, pos, hd, d):
        r = self.rng
        for _ in range(r.integers(3, 7)):
            s = r.uniform(max(2.0, 6 - 4 * d), 7 - 3 * d)
            dh = r.uniform(-2, 1 + 2.5 * d)
            g = self._gap(d, max(dh, 0))
            lat = r.uniform(-3, 3) * d
            top = self._at(pos, hd, lat, g + s / 2, dh)
            self._wp(self._plat(top, (s, 1, s), hd))
            pos = self._at(top, hd, 0, s / 2, 0)
        return pos

    def _seg_stairs(self, pos, hd, d):
        r = self.rng
        for _ in range(r.integers(4, 7)):
            dh = r.uniform(1.5, 2.5 + 1.5 * d)
            g = min(self._gap(d * 0.7, dh), reach(dh))
            top = self._at(pos, hd, 0, g + 2, dh)
            self._wp(self._plat(top, (4, 1, 4), hd))
            pos = self._at(top, hd, 0, 2, 0)
        return pos

    # A kill-brick floor with stepping stones, the most common obby stage there is
    def _seg_lava(self, pos, hd, d):
        r = self.rng
        n = r.integers(3, 6)
        length = 0.0
        stones = []
        for _ in range(n):
            g = self._gap(d * 0.8)
            s = r.uniform(max(1.5, 4 - 2.5 * d), 5 - 2 * d)
            stones.append((length + g + s / 2, s, r.uniform(-2, 2) * d))
            length += g + s
        self._gap(d * 0.8)   # drawn but unused, so every other course keeps its exact layout
        floor = self._at(pos, hd, 0, length / 2, -0.5)
        self.world.add(floor + [0, -0.25, 0], (10, 1, length), KILL, kill=True, neon=True, yaw=hd)
        for z, s, lat in stones:
            self._wp(self._plat(self._at(pos, hd, lat, z, 0.8), (s, 1.3, s), hd))
        # end at the last stone's far edge like every segment: ending a jump-gap past it, plus the checkpoint's own
        # gap, made 9.7-12 stud jumps (max is ~10, safe ~8.3); 2 of 16 eval courses were impossible there
        return self._at(pos, hd, 0, length, 0)

    def _seg_beam(self, pos, hd, d):
        r = self.rng
        g = self._gap(d * 0.6)
        L = r.uniform(12, 18 + 10 * d)
        wid = max(1.0, 3 - 2 * d)
        top = self._at(pos, hd, 0, g + L / 2, 0)
        b = self._plat(top, (wid, 1, L), hd)
        self._wp(b, (0, -L / 2 + 1.5))
        self._wp(b, (0, L / 2 - 1.5))
        return self._at(top, hd, 0, L / 2, 0)

    def _seg_vanish(self, pos, hd, d):
        r = self.rng
        period = r.uniform(3.5 - 1.5 * d, 5.0 - 1.5 * d)
        for k in range(r.integers(3, 6)):
            g = self._gap(d * 0.7)
            top = self._at(pos, hd, 0, g + 2, 0)
            ph = r.uniform(0, 1)
            if k == 0:
                ph0 = ph
            # each part turns solid 0.8 s after the one before, a wave you can run with. Independent random phases made
            # some chains impossible (seed 999000 stage 16: three parts fading together, ~0.16 s short at full speed)
            m = Motion("vanish", period=period, phase=(ph0 - k * 0.8 / period) % 1.0, on_frac=0.65)
            self._wp(self._plat(top, (4, 1, 4), hd, motion=m))
            pos = self._at(top, hd, 0, 2, 0)
        return pos

    def _seg_moving(self, pos, hd, d):
        r = self.rng
        for _ in range(r.integers(2, 4)):
            g = self._gap(d * 0.5)
            s = r.uniform(4, 6)
            top = self._at(pos, hd, 0, g + s / 2, 0)
            f = forward_xz(hd)
            lateral = (math.cos(hd), 0.0, -math.sin(hd)) if r.random() < 0.6 else (f[0], 0.0, f[1])
            m = Motion("slide", axis=lateral, amp=r.uniform(3, 4 + 5 * d), period=r.uniform(4.5 - 1.5 * d, 6 - 1.5 * d),
                       phase=r.uniform(0, 2 * math.pi))
            self._wp(self._plat(top, (s, 1, s), hd, motion=m))
            pos = self._at(top, hd, 0, s / 2, 0)
        return pos

    def _seg_spinner(self, pos, hd, d):
        r = self.rng
        g = self._gap(d * 0.5)
        R = r.uniform(9, 12)
        top = self._at(pos, hd, 0, g + R, 0)
        base = self._plat(top, (2 * R, 1, 2 * R), hd)
        omega = r.uniform(1.0 + 1.5 * d, 1.5 + 2.0 * d) * r.choice([-1, 1])
        self.world.add(top + [0, 0.9, 0], (2 * R - 1, 0.8, 0.8), KILL, kill=True, neon=True,
                       motion=Motion("spin", omega=omega), yaw=hd)
        self.world.add(top + [0, 1.5, 0], (1.2, 3, 1.2), (0.2, 0.2, 0.2))
        self._wp(base, (0, -R + 1.5))
        self._wp(base, (R - 1.5 if r.random() < 0.5 else -(R - 1.5), 0))
        self._wp(base, (0, R - 1.5))
        return self._at(top, hd, 0, R, 0)

    # ── game logic ──
    def respawn_point(self):
        i = self.cps[getattr(self, "stage", 0)]
        return self.world.c[i] + [0, self.world.h[i, 1], 0]

    def wp_point(self, k):
        part, (ox, oz) = self.waypoints[min(k, len(self.waypoints) - 1)]
        w = self.world
        wx, wz = to_world(ox, oz, w.yaw[part])
        return np.array([w.c[part, 0] + wx, w.c[part, 1] + w.h[part, 1], w.c[part, 2] + wz])

    # index of the waypoint we're standing on, preferring the furthest along
    def _wp_on(self):
        p = self.player
        if not p.grounded:
            return -1
        best = -1
        for k in range(max(0, self.best_wp - 3), min(len(self.waypoints), self.best_wp + 4)):
            part, _ = self.waypoints[k]
            if part == p.ground and np.linalg.norm((self.wp_point(k) - p.pos)[[0, 2]]) < 4.0:
                best = k
        return best

    def on_substep(self):
        p = self.player
        if not p.alive:
            return
        if p.grounded and p.ground in self.cps:
            s = self.cps.index(p.ground)
            if s > self.stage:
                self.stage = s
                if s > self.best_stage:
                    self.best_stage = s
                    self.reward += 1.0
                    self.toast(f"Stage {s}!" if s < self.n_stages else "You beat the obby!")
                if s == self.n_stages:
                    self.done = True
        k = self._wp_on()
        if k > self.best_wp:
            self.reward += 0.2 * (k - self.best_wp)
            self.best_wp = k
            self._dist_prev = None
        nxt = self.wp_point(self.best_wp + 1)
        dist = float(np.linalg.norm((nxt - p.pos)[[0, 2]]))
        if self._dist_prev is not None:
            self.reward += 0.02 * (self._dist_prev - dist)   # dense pull toward the next platform
        self._dist_prev = dist
        if self.steps >= self.max_steps:
            self.done = True

    def on_death(self):
        self.reward -= 1.0
        self._dist_prev = None
        self.best_wp = max(self.best_wp, self.cp_wp[self.stage])

    def respawn(self):
        super().respawn()
        self.best_wp = max(self.cp_wp[self.stage], min(self.best_wp, self.cp_wp[self.stage]))
        i = self.cps[self.stage]
        self.cam.yaw = math.atan2(*(self.wp_point(self.cp_wp[self.stage] + 1) - self.world.c[i])[[0, 2]])
        self._dist_prev = None

    def hud(self):
        els = [dict(kind="leaderboard", cols=["Stage"], rows=[["You", self.stage]]),
               dict(kind="text", rect=(0.4, 0.02, 0.2, 0.06), text=f"Stage {self.stage}", size=30, align="center")]
        for s in range(self.stage, min(self.stage + 2, len(self.cps))):
            i = self.cps[s]
            b = self.billboard(self.world.c[i] + [0, 4, 0], str(s), size=1.4)
            if b:
                els.append(b)
        return els

    def info(self):
        return dict(stage=self.stage, best_stage=self.best_stage, deaths=self.deaths, progress=self.best_wp / max(1, len(self.waypoints) - 1))

    # ── expert: pick the jump timing whose short simulated future lands on the next platform ──
    def expert(self):
        p = self.player
        if not p.alive:
            self.say("fell off, wait to respawn at the last checkpoint")
            return Action(cursor=self.cursor)
        k = self._wp_on()
        base = k if k >= 0 else self.best_wp
        tgt_k = min(base + 1, len(self.waypoints) - 1)
        best, best_score = None, -1e9
        timed = any(self.waypoints[j][0] in self.world.motions or np.any(self.world.kill[self._near_kill()])
                    for j in (tgt_k,))
        for wait in ((0, 3, 6) if timed else (0,)):   # waiting only helps when something moves
            for jump_at in (None, 0, 1, 2, 3, 4):
                s = self._rollout(tgt_k, wait, jump_at)
                if s > best_score:
                    best, best_score = (wait, jump_at), s
        wait, jump_at = best
        if wait > 0:
            self.say("the next platform is moving or blocked, wait for the right moment")
        elif jump_at == 0:
            self.say("at the edge, jump to the next platform now")
        elif jump_at is not None:
            self.say("run toward the edge, then jump to the next platform")
        else:
            self.say("walk onto the next platform")
        tgt = self.wp_point(tgt_k)
        self.goal = tgt
        a = self.steer_to(tgt, keys=() if wait > 0 else ("w",), jump=(jump_at == 0 and wait == 0), stop_dist=0.3)
        pitch_err = math.degrees(math.radians(30) - self.cam.pitch)
        a.look_y = float(np.clip(pitch_err, -10, 10))
        return a

    # centre over the part's top face with a margin, i.e. a landing that won't slide off
    def _on_top(self, h, part):
        w = self.world
        lx, lz = to_local(h.pos[0] - w.c[part, 0], h.pos[2] - w.c[part, 2], w.yaw[part])
        return abs(lx) < w.h[part, 0] - 0.2 and abs(lz) < w.h[part, 2] - 0.2

    # moving kill parts (spinners) within reach, which also make timing matter
    def _near_kill(self):
        w = self.world
        idx = [i for i in w.motions if w.kill[i]]
        return [i for i in idx if np.linalg.norm((w.c[i] - self.player.pos)[[0, 2]]) < 20]

    def _rollout(self, tgt_k, wait, jump_at, horizon=12):
        w, p0 = self.world, self.player
        h = copy.deepcopy(p0)
        yaw, t = self.cam.yaw, self.t
        tgt_part = self.waypoints[tgt_k][0]
        score = None
        for tick in range(horizon):
            for sub in range(SUBSTEPS):
                w.update(t + DT)
                tgt = self.wp_point(tgt_k)
                want = math.atan2(tgt[0] - h.pos[0], tgt[2] - h.pos[2])
                err = (want - yaw + math.pi) % (2 * math.pi) - math.pi
                yaw += max(-math.radians(5), min(math.radians(5), err))
                go = tick >= wait and abs(math.degrees(err)) < 70 and np.linalg.norm((tgt - h.pos)[[0, 2]]) > 0.3
                move = forward_xz(yaw) if go else np.zeros(2)
                jump = jump_at is not None and tick == wait + jump_at
                t += DT
                if humanoid_step(h, w, move, jump) or h.pos[1] < tgt[1] - 12:
                    score = -1000 + tick
                    break
                if h.grounded and h.ground == tgt_part and np.linalg.norm((tgt - h.pos)[[0, 2]]) < 3.0 and self._on_top(h, tgt_part):
                    score = 500 - 10 * tick
                    break
            if score is not None:
                break
        if score is None:
            tgt = self.wp_point(tgt_k)
            score = -float(np.linalg.norm(tgt - h.pos)) - (5 if not h.grounded else 0) - (2 if jump_at is not None else 0)
        w.update(self.t)
        return score


if __name__ == "__main__":
    import time
    for diff in (0.2, 0.6, 1.0):
        env = Obby(seed=1, difficulty=diff, stages=8)
        t0 = time.time()
        R = 0.0
        while not env.done:
            r, done, info = env.step(env.expert())
            R += r
        print(f"difficulty {diff}: {info}, return {R:.1f}, {env.steps} ticks, {(time.time() - t0) / env.steps * 1000:.1f} ms/tick")
