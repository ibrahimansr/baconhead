"""Roblox-faithful humanoid physics, parts, camera, prompts, and the base Env every game extends.

Units are studs and seconds. Numbers are Roblox defaults unless marked as a calibration knob.
y is up. Camera yaw 0 looks down +z; turning right increases yaw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core.actions import Action, TICK_S

# ── Roblox defaults ───────────────────────────────────────────────────────────
DT = 1 / 60                        # Roblox steps humanoids at 60 Hz (physics itself is 240 Hz; 60 is enough here)
SUBSTEPS = round(TICK_S / DT)      # 6 per agent tick
GRAVITY = 196.2                    # workspace.Gravity
WALK_SPEED = 16.0                  # Humanoid.WalkSpeed
JUMP_HEIGHT = 7.2                  # Humanoid.JumpHeight (UseJumpPower=false, the new-place default)
JUMP_V = math.sqrt(2 * GRAVITY * JUMP_HEIGHT)   # 53.15 studs/s → 0.54s airtime, ~8.7 studs flat at walk speed
HIP_HEIGHT = 2.0                   # R15 default; the root floats this high, so ledges under it are stepped up
FOV_DEG = 70.0                     # Camera.FieldOfView (vertical)
ZOOM_DEFAULT, ZOOM_MIN, ZOOM_MAX = 12.5, 0.5, 128.0
PROMPT_DIST = 10.0                 # ProximityPrompt.MaxActivationDistance
CLICK_DIST = 32.0                  # ClickDetector.MaxActivationDistance
RESPAWN_S = 5.0                    # Players.RespawnTime

# ── calibration knobs: not published by Roblox, set from feel. Change them if play feels off. ────────
CHAR_RADIUS = 1.0                  # R15 root/torso collision half-width
CHAR_HEIGHT = 5.0
SUPPORT_R = 0.5                    # a part holds you up only while your centre is within this of its top face; past it you slide off
HEAD_Y = 4.5                       # camera focus above feet
GROUND_ACCEL = 400.0               # studs/s², "more or less instant" on the ground
AIR_ACCEL = 90.0                   # strong air control; releasing W mid-jump bleeds speed in ~0.2s
JUMP_COOLDOWN = 0.05               # holding space bunny-hops, with this gap after each landing
TURN_RATE = 12.0                   # rad/s the body turns toward its movement (AutoRotate)
PX_PER_DEG = 2.0                   # control/controller.LOOK_PX_PER_DEGREE, so a sim drag equals a real drag
SCROLL_ZOOM = 1.25                 # zoom multiplier per scroll notch
VOID_Y = -60.0                     # ponytail: real FallenPartsDestroyHeight is -500; -60 keeps the same outcome, 2s sooner
SHIFTLOCK_OFFSET = 1.75            # studs the camera shifts right in shift lock


def to_local(x, z, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return c * x - s * z, s * x + c * z


def to_world(x, z, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return c * x + s * z, -s * x + c * z


def forward_xz(yaw: float) -> np.ndarray:
    return np.array([math.sin(yaw), math.cos(yaw)])


def right_xz(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw), -math.sin(yaw)])


# ── parts ─────────────────────────────────────────────────────────────────────
@dataclass
class Motion:
    kind: str                    # "slide" (sine along axis), "spin" (around y), "vanish" (on/off cycle)
    axis: tuple = (1.0, 0.0, 0.0)
    amp: float = 0.0
    period: float = 4.0
    phase: float = 0.0
    omega: float = 0.0           # spin rad/s
    on_frac: float = 0.6         # vanish: fraction of the period it is solid


class World:
    """Every part as flat numpy arrays, so collision and rendering are single vector ops."""

    def __init__(self):
        self._rows: list[dict] = []
        self.motions: dict[int, Motion] = {}
        self.t = 0.0

    def add(self, center, size, color, *, kill=False, neon=False, collide=True, yaw=0.0,
            motion: Optional[Motion] = None, tag: Optional[str] = None, visible=True) -> int:
        i = len(self._rows)
        self._rows.append(dict(c=center, h=np.asarray(size, float) / 2, col=color, kill=kill, neon=neon,
                               collide=collide, yaw=yaw, tag=tag, visible=visible))
        if motion is not None:
            self.motions[i] = motion
        return i

    def build(self) -> None:
        r = self._rows
        self.c0 = np.array([x["c"] for x in r], float).reshape(-1, 3)
        self.h = np.array([x["h"] for x in r], float).reshape(-1, 3)
        self.col = np.array([x["col"] for x in r], float).reshape(-1, 3)
        self.yaw0 = np.array([x["yaw"] for x in r], float)
        self.kill = np.array([x["kill"] for x in r], bool)
        self.neon = np.array([x["neon"] for x in r], bool)
        self.collide0 = np.array([x["collide"] for x in r], bool)
        self.visible0 = np.array([x["visible"] for x in r], bool)
        self.tags = [x["tag"] for x in r]
        self.c, self.yaw = self.c0.copy(), self.yaw0.copy()
        self.collide, self.visible = self.collide0.copy(), self.visible0.copy()
        self.blink = np.zeros(len(r), bool)
        self.update(0.0)

    def __len__(self):
        return len(self._rows)

    def pose_at(self, i: int, t: float):
        m = self.motions.get(i)
        c, yaw, solid = self.c0[i].copy(), self.yaw0[i], True
        if m is None:
            return c, yaw, solid, False
        if m.kind == "slide":
            c += np.asarray(m.axis) * m.amp * math.sin(2 * math.pi * t / m.period + m.phase)
        elif m.kind == "spin":
            yaw = self.yaw0[i] + m.omega * t
        elif m.kind == "vanish":
            u = ((t / m.period + m.phase) % 1.0)
            solid = u < m.on_frac
            return c, yaw, solid, solid and u > m.on_frac - 0.25   # blinks for the last quarter-cycle, like fading obby tiles
        return c, yaw, solid, False

    def update(self, t: float) -> None:
        """Pose every moving part at time t in three vector ops, one per motion kind."""
        self.t = t
        if not self.motions:
            return
        if not hasattr(self, "_mv"):
            ms = self.motions
            pick = lambda kind: np.array([i for i, m in ms.items() if m.kind == kind], int)
            sl, sp, va = pick("slide"), pick("spin"), pick("vanish")
            g = lambda idx, f: np.array([f(ms[i]) for i in idx], float)
            self._mv = dict(sl=sl, ax=g(sl, lambda m: np.asarray(m.axis) * m.amp).reshape(-1, 3), sl_p=g(sl, lambda m: m.period),
                            sl_ph=g(sl, lambda m: m.phase), sp=sp, om=g(sp, lambda m: m.omega), va=va,
                            va_p=g(va, lambda m: m.period), va_ph=g(va, lambda m: m.phase), va_on=g(va, lambda m: m.on_frac))
        m = self._mv
        if len(m["sl"]):
            self.c[m["sl"]] = self.c0[m["sl"]] + m["ax"] * np.sin(2 * np.pi * t / m["sl_p"] + m["sl_ph"])[:, None]
        if len(m["sp"]):
            self.yaw[m["sp"]] = self.yaw0[m["sp"]] + m["om"] * t
        if len(m["va"]):
            u = (t / m["va_p"] + m["va_ph"]) % 1.0
            solid = u < m["va_on"]
            self.collide[m["va"]] = self.collide0[m["va"]] & solid
            self.visible[m["va"]] = self.visible0[m["va"]] & solid
            self.blink[m["va"]] = solid & (u > m["va_on"] - 0.25)

    def raycast(self, o, d, max_t=1e9, mask=None):
        """Nearest part hit by a ray, as (t, index) or (inf, -1). Same slab test the renderer uses."""
        m = self.collide | self.visible if mask is None else mask
        rel = o[None, :] - self.c
        ox, oz = to_local(rel[:, 0], rel[:, 2], self.yaw)
        dx, dz = to_local(d[0], d[2], self.yaw)
        O = np.stack([ox, rel[:, 1], oz], 1)
        D = np.stack([dx, np.full_like(dx, d[1]), dz], 1)
        D = np.where(np.abs(D) < 1e-9, 1e-9, D)
        t1, t2 = (-self.h - O) / D, (self.h - O) / D
        tn, tf = np.minimum(t1, t2).max(1), np.maximum(t1, t2).min(1)
        hit = m & (tf >= tn) & (tn > 0) & (tn < max_t)
        if not hit.any():
            return math.inf, -1
        tt = np.where(hit, tn, np.inf)
        i = int(tt.argmin())
        return float(tt[i]), i


# ── humanoid ──────────────────────────────────────────────────────────────────
@dataclass
class Humanoid:
    pos: np.ndarray                       # feet
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    face: float = 0.0                     # body yaw
    grounded: bool = False
    ground: int = -1
    jump_cd: float = 0.0
    speed_mult: float = 1.0
    stun: float = 0.0                     # seconds left knocked silly (bonked, tackled): no walking or jumping
    health: float = 100.0                 # Humanoid.Health / MaxHealth
    alive: bool = True
    dead_t: float = 0.0
    color: tuple = (0.95, 0.8, 0.2)      # the noob-yellow head/arms
    shirt: tuple = (0.05, 0.45, 0.8)
    pants: tuple = (0.2, 0.6, 0.2)


def humanoid_step(h: Humanoid, w: World, move: np.ndarray, jump: bool, dt: float = DT,
                  face_to: Optional[float] = None) -> bool:
    """One 1/60 s humanoid update. Returns True if it touched a kill part."""
    if not h.alive:
        return False

    # ride whatever we stood on last step, the way Roblox friction carries you on moving parts
    if h.grounded and h.ground >= 0 and h.ground in w.motions:
        c1, y1, _, _ = w.pose_at(h.ground, w.t + dt)
        c0, y0 = w.c[h.ground], w.yaw[h.ground]
        lx, lz = to_local(h.pos[0] - c0[0], h.pos[2] - c0[2], y0)
        wx, wz = to_world(lx, lz, y1)
        h.pos = h.pos + np.array([c1[0] + wx - h.pos[0], c1[1] - c0[1], c1[2] + wz - h.pos[2]])
        h.face += y1 - y0

    if h.stun > 0:
        h.stun = max(0.0, h.stun - dt)
        move, jump = np.zeros(2), False
    n = float(np.hypot(move[0], move[1]))
    target = (move / n) * WALK_SPEED * h.speed_mult if n > 1e-6 else np.zeros(2)
    acc = GROUND_ACCEL if h.grounded else AIR_ACCEL
    dv = target - h.vel[[0, 2]]
    step = acc * dt
    dvn = float(np.hypot(*dv))
    if dvn > step:
        dv *= step / dvn
    h.vel[0] += dv[0]
    h.vel[2] += dv[1]

    want = face_to if face_to is not None else (math.atan2(move[0], move[1]) if n > 1e-6 else None)
    if want is not None:
        d = (want - h.face + math.pi) % (2 * math.pi) - math.pi
        h.face += max(-TURN_RATE * dt, min(TURN_RATE * dt, d))

    h.jump_cd = max(0.0, h.jump_cd - dt)
    if jump and h.grounded and h.jump_cd <= 0:
        h.vel[1] = JUMP_V
        h.grounded = False
    prev_feet = h.pos[1]
    h.pos = h.pos + h.vel * dt
    h.pos[1] -= 0.5 * GRAVITY * dt * dt   # exact under constant gravity, so apex and airtime match Roblox at 60 Hz
    h.vel[1] -= GRAVITY * dt
    was_grounded = h.grounded
    touched_kill = _collide(h, w, prev_feet)
    if h.grounded and not was_grounded:
        h.jump_cd = JUMP_COOLDOWN
    if h.pos[1] < VOID_Y:
        touched_kill = True
    return touched_kill


def _overlaps(h: Humanoid, w: World, margin: float):
    rel = h.pos - w.c
    lx, lz = to_local(rel[:, 0], rel[:, 2], w.yaw)
    qx, qz = np.clip(lx, -w.h[:, 0], w.h[:, 0]), np.clip(lz, -w.h[:, 2], w.h[:, 2])
    ex, ez = lx - qx, lz - qz
    dist = np.hypot(ex, ez)
    top, bot = w.c[:, 1] + w.h[:, 1], w.c[:, 1] - w.h[:, 1]
    hit = (dist < CHAR_RADIUS + margin) & (h.pos[1] < top + margin) & (h.pos[1] + CHAR_HEIGHT > bot - margin)
    return hit, lx, lz, ex, ez, dist, top, bot


def _collide(h: Humanoid, w: World, prev_feet: float) -> bool:
    """Roblox R15 rules: legs don't collide, so a part whose top is under HIP_HEIGHT above your feet is no wall;
    it holds you up once your centre is over it (SUPPORT_R). Anything taller pushes you sideways or bumps your head."""
    h.grounded, h.ground = False, -1
    skip = np.zeros(len(w.c), bool)
    for _ in range(8):
        hit, lx, lz, ex, ez, dist, top, bot = _overlaps(h, w, 0.0)
        hit &= w.collide & ~skip
        idx = np.flatnonzero(hit)
        if len(idx) == 0:
            break
        i = idx[np.argmax(top[idx])]   # resolve the highest surface first, which is the one we'd stand on
        up = top[i] - h.pos[1]
        down = h.pos[1] + CHAR_HEIGHT - bot[i]
        landing = prev_feet >= top[i] - 1e-3 and h.vel[1] <= 0   # swept test, so a fast fall can't tunnel a thin part
        supported = dist[i] <= SUPPORT_R
        if supported and (landing or (up <= HIP_HEIGHT and h.vel[1] <= 0.5)):
            h.pos[1] = top[i]
            h.vel[1] = 0.0
            h.grounded, h.ground = True, int(i)
            continue
        if up <= HIP_HEIGHT and (h.vel[1] <= 0.5 or landing):
            skip[i] = True             # under the hips and not under our centre: the legs pass through it
            continue
        if dist[i] > 1e-9:
            side = CHAR_RADIUS - dist[i]
            nx, nz = ex[i] / dist[i], ez[i] / dist[i]
        else:   # centre inside the footprint: push out through the nearest edge
            e = np.array([w.h[i, 0] - lx[i], w.h[i, 0] + lx[i], w.h[i, 2] - lz[i], w.h[i, 2] + lz[i]])
            k = int(e.argmin())
            side = e[k] + CHAR_RADIUS
            nx, nz = [(1, 0), (-1, 0), (0, 1), (0, -1)][k]
        if h.vel[1] > 0 and down < side and down < 1.5:
            h.pos[1] -= down
            h.vel[1] = 0.0
        else:
            wx, wz = to_world(nx, nz, w.yaw[i])
            h.pos[0] += wx * side
            h.pos[2] += wz * side
            vn = h.vel[0] * wx + h.vel[2] * wz
            if vn < 0:
                h.vel[0] -= vn * wx
                h.vel[2] -= vn * wz
        skip[i] = True   # each part is resolved once per substep, so a wall can't starve the floor under us
    hit = _overlaps(h, w, 0.05)[0] & w.kill & w.collide
    return bool(hit.any())


# ── camera ────────────────────────────────────────────────────────────────────
@dataclass
class Camera:
    yaw: float = 0.0
    pitch: float = math.radians(15)      # + looks down
    zoom: float = ZOOM_DEFAULT
    fov: float = FOV_DEG
    shiftlock: bool = False
    first_person: bool = False           # LockFirstPerson games (shooters): camera in the head, mouse turns it
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    fwd: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))

    def turn(self, dyaw_deg: float, dpitch_deg: float) -> None:
        self.yaw = (self.yaw + math.radians(dyaw_deg)) % (2 * math.pi)
        self.pitch = float(np.clip(self.pitch + math.radians(dpitch_deg), math.radians(-80), math.radians(80)))

    def follow(self, h: Humanoid, w: World) -> None:
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        self.fwd = np.array([math.sin(self.yaw) * cp, -sp, math.cos(self.yaw) * cp])
        focus = h.pos + np.array([0, HEAD_Y, 0])
        if self.first_person:
            self.pos = focus
            return
        if self.shiftlock:
            r = right_xz(self.yaw)
            focus = focus + np.array([r[0], 0, r[1]]) * SHIFTLOCK_OFFSET
        back = -self.fwd
        t, _ = w.raycast(focus, back, max_t=self.zoom, mask=w.collide & w.visible & ~w.neon)
        dist = min(self.zoom, t - 0.5) if math.isfinite(t) else self.zoom   # Roblox's Popper pulls the camera in front of walls
        self.pos = focus + back * max(0.5, dist)

    def ray(self, u: float, v: float, aspect: float) -> np.ndarray:
        """World direction through normalized screen point (u, v), (0,0) top-left."""
        th = math.tan(math.radians(self.fov) / 2)
        f = self.fwd
        r = np.array([math.cos(self.yaw), 0.0, -math.sin(self.yaw)])
        up = np.cross(f, r)
        d = f + (2 * u - 1) * th * aspect * r - (2 * v - 1) * th * up
        return d / np.linalg.norm(d)

    def project(self, p: np.ndarray, aspect: float):
        """World point to normalized screen (u, v, depth), or None if behind the camera."""
        rel = p - self.pos
        f = self.fwd
        r = np.array([math.cos(self.yaw), 0.0, -math.sin(self.yaw)])
        up = np.cross(f, r)
        z = rel @ f
        if z < 0.1:
            return None
        th = math.tan(math.radians(self.fov) / 2)
        return (0.5 + (rel @ r) / (z * th * aspect) / 2, 0.5 - (rel @ up) / (z * th) / 2, z)


# ── proximity prompts and click detectors ────────────────────────────────────
@dataclass
class Prompt:
    pos: np.ndarray
    action: str                     # "Purchase", "Steal"
    obj: str = ""                   # "Noobini Pizzanini  $25"
    hold: float = 0.0               # HoldDuration seconds
    on_trigger: tuple = ()          # (env method name, *args): plain data, so an env pickles for GRPO rollouts
    enabled: tuple = ()             # same form; empty = always enabled
    progress: float = 0.0


# ── base env ──────────────────────────────────────────────────────────────────
@dataclass
class Input:
    """Raw hands for one substep; the agent's Action spreads over SUBSTEPS of these."""
    keys: frozenset = frozenset()
    dyaw: float = 0.0                # degrees this substep
    dpitch: float = 0.0
    cursor: tuple = (0.5, 0.5)
    click: bool = False
    scroll: int = 0


class Env:
    name = "base"
    aspect = 16 / 9
    respawn_s = RESPAWN_S
    train_steps = 3000                # ticks per training episode
    eval_kw: dict = {}                # fixed settings for train.eval

    def __init__(self, seed: int = 0, **kw):
        self.seed = seed
        self.kw = kw
        self.reset(seed)

    # subclasses fill these in
    def build(self) -> None: ...
    def on_substep(self) -> None: ...
    def dynamic_boxes(self) -> list: return []
    def hud(self) -> list: return []
    def on_ui(self, id_: str) -> None: ...
    def on_world_click(self, part: int) -> None: ...
    def respawn_point(self) -> np.ndarray: return np.zeros(3)
    def expert(self) -> Action: return Action()
    def say(self, plan: str) -> None:
        """Experts call this on every decision path: what a player would say they're doing now, in terms of what's on
        screen (never coordinates or hidden state). It's the PLAN line the policy learns to write before its inputs."""
        self.plan = plan
    def potential(self) -> float: return 0.0   # dense progress score for reward shaping; games with sparse rewards override
    def info(self) -> dict: return {}
    def score(self) -> float: return 0.0      # the one eval number, higher is better

    def reset(self, seed: Optional[int] = None) -> None:
        self.seed = self.seed if seed is None else seed
        self.rng = np.random.default_rng(self.seed)
        self.t = 0.0
        self.steps = 0
        self.reward = 0.0
        self.done = False
        self.deaths = 0
        self.plan = ""                              # the expert's current subgoal in words (see say())
        self.world = World()
        self.prompts: list[Prompt] = []
        self.cam = Camera()
        self.player = Humanoid(pos=np.zeros(3))
        self.cursor = (0.5, 0.5)
        self.held: frozenset = frozenset()
        self.mouse_held = False
        self.menu: Optional[str] = None             # Roblox's own Esc menu: None | "main" | "reset"
        self.equipped: Optional[str] = None
        self.toasts: list[tuple[float, str]] = []    # (expire time, text) Roblox-style notifications
        self.theme = self._theme()
        self.build()
        self.world.build()
        self.player.pos = self.respawn_point().astype(float)
        self.cam.follow(self.player, self.world)

    def _theme(self) -> dict:
        """Per-episode lighting and colour randomization, so the policy never overfits one sky."""
        r = self.rng
        return dict(
            sky_top=np.clip(np.array([0.36, 0.62, 0.95]) + r.normal(0, 0.06, 3), 0, 1),
            sky_hor=np.clip(np.array([0.78, 0.88, 0.98]) + r.normal(0, 0.04, 3), 0, 1),
            sun=_unit(np.array([r.uniform(-0.6, 0.6), r.uniform(0.55, 0.95), r.uniform(-0.6, 0.6)])),
            ambient=float(r.uniform(0.45, 0.6)),
            fog=float(r.uniform(350, 700)),
            tint=np.clip(1 + r.normal(0, 0.04, 3), 0.85, 1.15),
        )

    def toast(self, text: str, secs: float = 2.5) -> None:
        self.toasts.append((self.t + secs, text))

    # ── stepping ──
    def step(self, a: Action) -> tuple[float, bool, dict]:
        """One 100ms agent tick. The camera drag is spread across the tick like the real injector's."""
        self.reward = 0.0
        for k in range(SUBSTEPS):
            self.substep(Input(keys=a.keys, dyaw=a.look_x / SUBSTEPS, dpitch=a.look_y / SUBSTEPS,
                               cursor=a.cursor, click=a.click and k == 0, scroll=a.scroll if k == 0 else 0))
        self.steps += 1
        return self.reward, self.done, self.info()

    def substep(self, inp: Input) -> None:
        pressed = inp.keys - self.held
        self.held = inp.keys
        self.cursor = inp.cursor
        self.mouse_held = "lmb" in inp.keys
        click = inp.click or "lmb" in pressed   # a fresh press of a held button is also a click
        if "esc" in pressed:
            self.menu = None if self.menu else "main"
        if self.menu:   # Roblox's menu swallows game input, the world keeps running
            if self.menu == "main" and "r" in pressed:
                self.menu = "reset"
            if click:
                self._click(inp.cursor)
            inp = Input(cursor=inp.cursor)
            click = False
            pressed = frozenset()
        if "shift" in pressed and not self.cam.first_person:
            self.cam.shiftlock = not self.cam.shiftlock
        for k in ("1", "2", "3"):
            if k in pressed:
                self.on_key(k)
        self.cam.turn(inp.dyaw, inp.dpitch)
        if inp.scroll:
            self.cam.zoom = float(np.clip(self.cam.zoom * SCROLL_ZOOM ** inp.scroll, 4.0, 40.0))

        self.world.update(self.t + DT)
        p = self.player
        if p.alive:
            f, r = forward_xz(self.cam.yaw), right_xz(self.cam.yaw)
            ks = inp.keys
            fb = ("w" in ks or "up" in ks) - ("s" in ks or "down" in ks)
            lr = ("d" in ks) - ("a" in ks)
            move = f * fb + r * lr
            locked = self.cam.shiftlock or self.cam.first_person
            if humanoid_step(p, self.world, move, "space" in ks, face_to=self.cam.yaw if locked else None):
                self.die()
        elif self.t - p.dead_t >= self.respawn_s:
            self.respawn()

        self._prompts("e" in inp.keys)
        if click and not (self.player.alive and self.on_tool_click()):
            self._click(inp.cursor)
        self.t += DT
        self.on_substep()
        self.cam.follow(p, self.world)

    def on_key(self, k: str) -> None: ...
    def on_tool_click(self) -> bool: return False   # an equipped tool (bat, gun, rod) eats the click before UI/world clicks

    def die(self) -> None:
        self.player.alive = False
        self.player.dead_t = self.t
        self.deaths += 1
        self.on_death()

    def on_death(self) -> None: ...

    def respawn(self) -> None:
        p = self.player
        p.pos = self.respawn_point().astype(float)
        p.vel[:] = 0
        p.alive, p.grounded, p.health, p.stun = True, False, 100.0, 0.0

    def damage(self, amount: float, why: str = "") -> None:
        """Take health off the player; at 0 the character dies like a Roblox Humanoid does."""
        p = self.player
        if not p.alive or amount <= 0:
            return
        p.health = max(0.0, p.health - amount)
        if p.health <= 0:
            if why:
                self.toast(why, 3)
            self.die()

    def scene_theme(self) -> dict:
        """The lighting this frame. Games with day/night or dark rooms override this."""
        return self.theme

    def active_prompt(self) -> Optional[Prompt]:
        """Roblox shows only the closest enabled prompt in range."""
        if not self.player.alive:
            return None
        best, bd = None, PROMPT_DIST
        for pr in self.prompts:
            d = float(np.linalg.norm(pr.pos - (self.player.pos + [0, 2.5, 0])))
            if d < bd and (not pr.enabled or getattr(self, pr.enabled[0])(*pr.enabled[1:])):
                best, bd = pr, d
        return best

    def _prompts(self, e_held: bool) -> None:
        act = self.active_prompt()
        for pr in self.prompts:
            if pr is not act:
                pr.progress = 0.0
        if act is None:
            return
        if not e_held:
            act.progress = 0.0
            return
        if act.progress < 0:      # already fired on this hold; Roblox needs a release before re-triggering
            return
        act.progress += DT
        if act.progress >= act.hold:
            act.progress = -1.0
            getattr(self, act.on_trigger[0])(*act.on_trigger[1:])

    def _click(self, cursor) -> None:
        for el in reversed(self.ui_layers()):    # topmost UI wins, exactly like Roblox GUI input sinking
            if el.get("id") and _inside(cursor, el["rect"]):
                if el["id"].startswith("rbx:"):
                    self._menu_click(el["id"][4:])
                else:
                    self.on_ui(el["id"])
                return
        if self.menu:
            return
        d = self.cam.ray(cursor[0], cursor[1], self.aspect)
        t, i = self.world.raycast(self.cam.pos, d, max_t=500)
        if i >= 0 and np.linalg.norm(self.cam.pos + d * t - self.player.pos) <= CLICK_DIST:
            self.on_world_click(i)

    # ── Roblox's own menu (the logo button top-left, or Esc) ──
    def ui_layers(self) -> list:
        """Everything clickable, bottom to top: the Roblox menu button, the game's UI, then the menu if open."""
        return [dict(kind="hit", id="rbx:menu", rect=MENU_BUTTON)] + self.hud() + self._menu_hud()

    def _menu_click(self, id_: str) -> None:
        if id_ == "menu":
            self.menu = None if self.menu else "main"
        elif id_ == "resume":
            self.menu = None
        elif id_ == "reset":
            self.menu = "reset"
        elif id_ == "reset_yes":
            self.menu = None
            if self.player.alive:
                self.die()            # Roblox resets by killing the character; it respawns at the checkpoint
        elif id_ == "reset_no":
            self.menu = "main"
        elif id_ == "leave":
            self.menu = None
            self.done = True

    def _menu_hud(self) -> list:
        if not self.menu:
            return []
        els = [dict(kind="panel", id="rbx:block", rect=(0, 0, 1, 1), color=(0.05, 0.05, 0.07), alpha=0.75)]
        for j, tab in enumerate(("Players", "Settings", "Report", "Help")):
            els.append(dict(kind="text", rect=(0.3 + 0.1 * j, 0.1, 0.1, 0.05), text=tab, size=20, align="center",
                            color=(1, 1, 1) if j == 0 else (0.7, 0.7, 0.7)))
        els.append(dict(kind="text", rect=(0.35, 0.3, 0.3, 0.05), text="You", size=22, align="center"))
        if self.menu == "main":
            for j, (id_, txt) in enumerate((("reset", "Reset Character  (R)"), ("leave", "Leave  (L)"), ("resume", "Resume  (Esc)"))):
                els.append(dict(kind="button", id=f"rbx:{id_}", rect=(0.27 + 0.16 * j, 0.82, 0.14, 0.08), text=txt, size=15,
                                color=(0.25, 0.25, 0.3)))
        else:
            els += [dict(kind="panel", rect=(0.33, 0.35, 0.34, 0.3), color=(0.15, 0.15, 0.18), alpha=0.95),
                    dict(kind="text", rect=(0.33, 0.38, 0.34, 0.1), text="Are you sure you want to\nreset your character?", size=20, align="center"),
                    dict(kind="button", id="rbx:reset_yes", rect=(0.36, 0.53, 0.13, 0.08), text="Reset", color=(0.2, 0.5, 1.0)),
                    dict(kind="button", id="rbx:reset_no", rect=(0.51, 0.53, 0.13, 0.08), text="Don't Reset", color=(0.35, 0.35, 0.4), size=16)]
        return els

    def open_reset(self) -> Action:
        """Expert helper: Esc → R → click Reset, like a stuck player does."""
        if self.menu is None:
            return Action(keys=frozenset({"esc"}), cursor=self.cursor)
        if self.menu == "main":
            return Action(keys=frozenset({"r"}), cursor=self.cursor)
        x, y, w, h = next(el["rect"] for el in self._menu_hud() if el.get("id") == "rbx:reset_yes")
        on = _inside(self.cursor, (x, y, w, h))
        return Action(cursor=(x + w / 2, y + h / 2), click=on)

    # ── what the renderer needs ──
    def scene(self, k_max: int = 128) -> dict:
        w = self.world
        vis = np.flatnonzero(w.visible)
        c, h, yaw, col = w.c[vis], w.h[vis], w.yaw[vis], w.col[vis].copy()
        neon = w.neon[vis].copy()
        col[w.blink[vis]] = np.minimum(1, col[w.blink[vis]] * 0.5 + 0.5) if (self.t * 6) % 1 < 0.5 else col[w.blink[vis]]
        dyn = self.dynamic_boxes()
        if (self.player.alive or self.t - self.player.dead_t < 0.4) and not self.cam.first_person:
            dyn = dyn + avatar_boxes(self.player, dead_for=0 if self.player.alive else self.t - self.player.dead_t)
        if dyn:
            d = np.array([list(b[0]) + list(b[1]) + [b[2]] + list(b[3]) + [b[4] if len(b) > 4 else 0] for b in dyn], float)
            c = np.concatenate([c, d[:, 0:3]])
            h = np.concatenate([h, d[:, 3:6] / 2])
            yaw = np.concatenate([yaw, d[:, 6]])
            col = np.concatenate([col, d[:, 7:10]])
            neon = np.concatenate([neon, d[:, 10] > 0])
        # frustum-ish cull: drop parts behind the camera, keep the nearest k_max by surface distance
        rel = c - self.cam.pos
        rad = np.linalg.norm(h, axis=1)
        ahead = rel @ self.cam.fwd > -rad - 2
        dist = np.maximum(0, np.linalg.norm(rel, axis=1) - rad)
        cand = np.flatnonzero(ahead & (dist < self.theme["fog"]))
        keep = cand[np.argsort(dist[cand])[:k_max]]
        return dict(cam_pos=self.cam.pos.copy(), cam_fwd=self.cam.fwd.copy(), cam_yaw=self.cam.yaw, fov=self.cam.fov,
                    c=c[keep], h=h[keep], yaw=yaw[keep], col=col[keep] * self.theme["tint"], neon=neon[keep],
                    theme=self.scene_theme(), hud=self._hover(self.ui_layers()) + self._toast_hud() + ([] if self.menu else self._prompt_hud()))

    def _hover(self, els: list) -> list:
        """Mark the topmost clickable element under the cursor, which the HUD draws lit up."""
        for el in reversed(els):
            if el.get("id") and el["kind"] in ("button", "icon") and _inside(self.cursor, el["rect"]):
                el["hover"] = True
                break
        return els

    def _prompt_hud(self) -> list:
        pr = self.active_prompt()
        if pr is None:
            return []
        uv = self.cam.project(pr.pos, self.aspect)
        if uv is None:
            return []
        return [dict(kind="prompt", at=(uv[0], uv[1]), text=pr.action, sub=pr.obj,
                     progress=(max(pr.progress, 0) / pr.hold) if pr.hold > 0 else 0.0)]

    def _toast_hud(self) -> list:
        self.toasts = [x for x in self.toasts if x[0] > self.t]
        return [dict(kind="toast", rect=(0.35, 0.16 + 0.06 * i, 0.3, 0.05), text=txt) for i, (_, txt) in enumerate(self.toasts[-3:])]

    def billboard(self, p, text, color=(1, 1, 1), size=1.0, max_dist=80) -> Optional[dict]:
        uv = self.cam.project(np.asarray(p, float), self.aspect)
        if uv is None or uv[2] > max_dist:
            return None
        return dict(kind="billboard", at=(uv[0], uv[1]), text=text, color=color, size=size * min(1.5, 14 / max(uv[2], 4)))

    # ── expert helpers shared by the games ──
    def steer_to(self, target, keys=("w",), jump=False, stop_dist=1.5) -> Action:
        """Turn the camera toward target and walk: what a human does with mouse + W."""
        p = self.player.pos
        dx, dz = target[0] - p[0], target[2] - p[2]
        want = math.atan2(dx, dz)
        err = math.degrees((want - self.cam.yaw + math.pi) % (2 * math.pi) - math.pi)
        look = float(np.clip(err, -30, 30))
        ks = set(keys) if (math.hypot(dx, dz) > stop_dist and abs(err) < 70) else set()
        if jump:
            ks.add("space")
        pitch_err = math.degrees(math.radians(20) - self.cam.pitch)
        return Action(keys=frozenset(ks), look_x=look, look_y=float(np.clip(pitch_err, -10, 10)), cursor=self.cursor)

    def click_at_world(self, p) -> Optional[tuple]:
        uv = self.cam.project(np.asarray(p, float), self.aspect)
        if uv is None or not (0.02 < uv[0] < 0.98 and 0.02 < uv[1] < 0.98):
            return None
        return (uv[0], uv[1])


@dataclass
class Room:
    """A walled building you enter through one doorway."""
    cx: float
    cz: float
    hx: float
    hz: float
    door_in: np.ndarray
    door_out: np.ndarray

    def inside(self, p) -> bool:
        return abs(p[0] - self.cx) < self.hx and abs(p[2] - self.cz) < self.hz


def route(p, goal, rooms: list) -> np.ndarray:
    """Next point toward goal, going through doorways instead of walls."""
    for rm in rooms:
        if rm.inside(p) != rm.inside(goal):
            a, b = (rm.door_in, rm.door_out) if rm.inside(p) else (rm.door_out, rm.door_in)
            past_a = np.linalg.norm((a - p)[[0, 2]]) < 2.0 or (
                np.linalg.norm((b - p)[[0, 2]]) < np.linalg.norm((b - a)[[0, 2]]) and np.linalg.norm(np.cross(
                    np.r_[(b - a)[[0, 2]], 0], np.r_[(p - a)[[0, 2]], 0])) / max(1e-6, np.linalg.norm((b - a)[[0, 2]])) < 2.5)
            return b if past_a else a   # once lined up in the doorway, head straight through it
    return np.asarray(goal, float)


def npc_walk(h: Humanoid, w: World, goal, stop: float = 0.8, jump: bool = False) -> bool:
    """Walk an NPC straight at goal for one substep. Returns True if it touched a kill part."""
    d = (np.asarray(goal, float) - h.pos)[[0, 2]]
    return humanoid_step(h, w, d if np.linalg.norm(d) > stop else np.zeros(2), jump)


def ray_humanoids(o, d, hs: list, max_t: float = 1e9):
    """First humanoid a ray hits: (index, distance, headshot) or None. Body is the torso+legs box, head on top."""
    best = None
    for i, h in enumerate(hs):
        if not h.alive:
            continue
        for head, (cy, half) in ((False, (2.0, np.array([1.0, 2.0, 0.6]))), (True, (4.6, np.array([0.6, 0.6, 0.6])))):
            c = h.pos + [0, cy, 0]
            rel = o - c
            lx, lz = to_local(rel[0], rel[2], h.face)
            dx, dz = to_local(d[0], d[2], h.face)
            O, D = np.array([lx, rel[1], lz]), np.array([dx, d[1], dz])
            D = np.where(np.abs(D) < 1e-9, 1e-9, D)
            t1, t2 = (-half - O) / D, (half - O) / D
            tn, tf = np.minimum(t1, t2).max(), np.maximum(t1, t2).min()
            if tf >= tn and 0 < tn < max_t and (best is None or tn < best[1]):
                best = (i, float(tn), head)
    return best


def health_hud(health: float) -> list:
    """Roblox's health bar: green to red, top right under the player list."""
    f = max(0.0, min(1.0, health / 100))
    col = (min(1, 2 * (1 - f)), min(1, 2 * f), 0.15)
    return [dict(kind="bar", rect=(0.80, 0.30, 0.18, 0.022), frac=f, color=col)]


def hotbar_hud(names: list, equipped: Optional[int]) -> list:
    """The default Roblox backpack hotbar: numbered slots along the bottom; click one or press its number."""
    n = len(names)
    w, gap = 0.07, 0.008
    x0 = 0.5 - (n * w + (n - 1) * gap) / 2
    out = []
    for i, name in enumerate(names):
        r = (x0 + i * (w + gap), 0.86, w, 0.115)
        out.append(dict(kind="button", id=f"slot:{i}", rect=r, text=f"{i + 1}\n{name}", size=13,
                        color=(0.25, 0.5, 0.95) if equipped == i else (0.15, 0.15, 0.17)))
    return out


def night_theme(theme: dict, k: float, lights=()) -> dict:
    """Blend a day theme toward night by k in [0,1]: dark blue sky, weak sun, low ambient."""
    t = dict(theme)
    t["sky_top"] = theme["sky_top"] * (1 - k) + np.array([0.02, 0.03, 0.08]) * k
    t["sky_hor"] = theme["sky_hor"] * (1 - k) + np.array([0.05, 0.06, 0.12]) * k
    t["ambient"] = theme["ambient"] * (1 - k) + 0.12 * k
    t["sun_k"] = 1 - 0.9 * k
    t["fog"] = theme["fog"] * (1 - k) + 140 * k
    t["lights"] = list(lights)
    return t


def avatar_boxes(h: Humanoid, dead_for: float = 0.0) -> list:
    """A blocky R15-ish avatar: legs, torso, arms, head. Each entry is (center, size, yaw, colour)."""
    f = h.face
    fx, fz = math.sin(f), math.cos(f)
    rx, rz = math.cos(f), -math.sin(f)
    p = h.pos
    spread = dead_for * 8          # on death the parts fly apart like a Roblox "oof"
    walk = math.sin(p[0] * 0.9 + p[2] * 0.9) * 0.35 if h.grounded and np.hypot(h.vel[0], h.vel[2]) > 1 else 0.0
    def at(side, up, fwd=0.0):
        return (p[0] + rx * side * (1 + spread) + fx * fwd, p[1] + up, p[2] + rz * side * (1 + spread) + fz * fwd)
    return [
        (at(-0.5, 1.0, walk), (1, 2, 1), f, h.pants),
        (at(0.5, 1.0, -walk), (1, 2, 1), f, h.pants),
        (at(0, 3.0 + spread * 0.3), (2, 2, 1), f, h.shirt),
        (at(-1.5, 3.0, -walk), (1, 2, 1), f, h.color),
        (at(1.5, 3.0, walk), (1, 2, 1), f, h.color),
        (at(0, 4.6 + spread, 0), (1.2, 1.2, 1.2), f, h.color),
    ]


MENU_BUTTON = (12 / 1280, 12 / 720, 44 / 1280, 44 / 720)   # the Roblox logo button hud._topbar draws top-left


def _inside(p, rect) -> bool:
    x, y, w, h = rect
    return x <= p[0] <= x + w and y <= p[1] <= y + h


def _unit(v):
    return v / np.linalg.norm(v)


if __name__ == "__main__":
    # the numbers a Roblox player feels: jump apex, airtime, and flat jump distance at walk speed
    w = World()
    w.add((0, -0.5, 0), (2000, 1, 2000), (0.4, 0.6, 0.3))
    w.build()
    h = Humanoid(pos=np.array([0.0, 0.0, 0.0]))
    for _ in range(10):
        humanoid_step(h, w, np.zeros(2), False)
    assert h.grounded
    apex, t, z0, left = 0.0, 0.0, None, False
    for i in range(200):
        humanoid_step(h, w, np.array([0.0, 1.0]), jump=(i == 30))
        if i == 30:
            z0 = h.pos[2]
        if i > 30:
            apex = max(apex, h.pos[1])
            if not h.grounded:
                t += DT
                left = True
            elif left:
                break
    dist = h.pos[2] - z0
    print(f"jump apex {apex:.2f} studs (Roblox 7.2), airtime {t:.2f}s (0.54), flat distance {dist:.2f} studs (~8.7)")
    assert abs(apex - JUMP_HEIGHT) < 0.4 and abs(t - 0.54) < 0.06 and 7.5 < dist < 9.5

    # a 1.5-stud step is walked up, a 3-stud wall is not
    w = World()
    w.add((0, -0.5, 0), (200, 1, 200), (0.4, 0.6, 0.3))
    w.add((0, 0.75, 30), (10, 1.5, 44), (1, 0, 0))
    w.add((20, 1.5, 10), (10, 3, 4), (1, 0, 0))
    w.build()
    a = Humanoid(pos=np.array([0.0, 0.0, 0.0]))
    b = Humanoid(pos=np.array([20.0, 0.0, 0.0]))
    for _ in range(90):
        humanoid_step(a, w, np.array([0.0, 1.0]), False)
        humanoid_step(b, w, np.array([0.0, 1.0]), False)
    assert a.pos[1] > 1.4 and b.pos[1] < 0.1, (a.pos, b.pos)
    print("step-up ok, wall blocks ok")
