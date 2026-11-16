"""Steal a Brainrot: buy brainrots off the red carpet, collect their cash, lock your base, steal from rivals.

Mechanics follow the real game: $100 start, first lock 30s then 60s, laser door blocks everyone but the
owner, buying is an E prompt on a walking brainrot, stealing is a held E prompt that slows you while you
carry it home, and the owner (or anyone with a bat) can knock it out of your hands.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core.actions import Action
from sim.core import DT, Env, Humanoid, Prompt, Room, avatar_boxes, humanoid_step, route

# (name, rarity, price, $/s, spawn weight)
KINDS = [
    ("Noobini Pizzanini", "Common", 25, 1, 30), ("Lirili Larila", "Common", 250, 3, 25), ("Tim Cheese", "Common", 500, 5, 20),
    ("Trippi Troppi", "Rare", 2_000, 15, 12), ("Tung Tung Sahur", "Rare", 3_000, 25, 10),
    ("Cappuccino Assassino", "Epic", 10_000, 75, 6), ("Brr Brr Patapim", "Epic", 15_000, 100, 5),
    ("Chimpanzini Bananini", "Legendary", 50_000, 300, 2.5), ("Ballerina Cappuccina", "Legendary", 100_000, 500, 1.5),
    ("Frigo Camelo", "Mythic", 300_000, 1_400, 0.6), ("Tralalero Tralala", "Brainrot God", 2_000_000, 15_000, 0.1),
]
RARITY_COL = {"Common": (0.85, 0.85, 0.85), "Rare": (0.2, 0.55, 1.0), "Epic": (0.7, 0.3, 1.0), "Legendary": (1.0, 0.8, 0.1),
              "Mythic": (1.0, 0.2, 0.2), "Brainrot God": (0.2, 1.0, 0.9)}
BODY_COL = [(0.9, 0.6, 0.3), (0.5, 0.8, 0.4), (0.95, 0.85, 0.3), (0.3, 0.6, 0.9), (0.6, 0.4, 0.25), (0.35, 0.25, 0.2),
            (0.4, 0.7, 0.3), (0.95, 0.9, 0.3), (0.95, 0.6, 0.8), (0.8, 0.9, 1.0), (0.3, 0.6, 1.0)]
NPC_NAMES = ["xX_Slayer_Xx", "noob1234", "BrainrotKing", "coolkid_77", "Tralalover", "bacon_hair", "SAB_Grinder"]
CARPET_X = (-135.0, 135.0)
CONVEYOR_SPEED = 5.0
SPAWN_EVERY = 2.5
FIRST_LOCK, LOCK_S = 30.0, 60.0
CARRY_SLOW = 0.7          # the real game slows thieves; exact factor unpublished, so a knob
BAT_RANGE, BAT_CD = 6.0, 0.8
SLOTS = 8


@dataclass
class Rot:
    id: int
    kind: int
    pos: np.ndarray
    state: str = "conveyor"      # conveyor | walking | slot | carried
    owner: int = -1
    slot: int = -1
    acc: float = 0.0             # uncollected cash sitting on its pad
    carrier: int = -1            # who holds it: 0 = you, 1.. = npc index + 1
    path: list = field(default_factory=list)


@dataclass
class Npc:
    h: Humanoid
    plot: int
    name: str
    cash: float = 100.0
    state: str = "idle"
    target: Optional[int] = None
    timer: float = 0.0
    carrying: int = -1
    bat_cd: float = 0.0
    greed: float = 0.5


class StealABrainrot(Env):
    name = "brainrot"

    def score(self) -> float:
        return math.log10(max(1.0, self._networth()))   # log10 net worth; you start at 2.0

    def build(self) -> None:
        r, w = self.rng, self.world
        self.max_steps = int(self.kw.get("max_steps", 6000))
        n_npc = int(self.kw.get("npcs", r.integers(3, 7)))
        w.add((0, -0.5, 0), (400, 1, 240), (0.35, 0.6, 0.3))                       # grass baseplate
        w.add((0, 0.1, 0), (CARPET_X[1] - CARPET_X[0], 0.2, 10), (0.75, 0.1, 0.1))   # the red carpet
        w.add((CARPET_X[0] - 6, 7, 0), (8, 14, 18), (0.25, 0.22, 0.2))              # the cave they walk out of
        w.add((CARPET_X[1] + 6, 7, 0), (8, 14, 18), (0.25, 0.22, 0.2))
        for x in (-200, 200):   # invisible map edge walls, which every Roblox map has
            w.add((x, 20, 0), (1, 40, 240), (0, 0, 0), visible=False)
        for z in (-120, 120):
            w.add((0, 20, z), (400, 40, 1), (0, 0, 0), visible=False)

        self.plots = []
        xs = (-90, -30, 30, 90)
        for i in range(8):
            px, s = xs[i % 4], (-1 if i < 4 else 1)
            pz = 40.0 if i < 4 else -40.0
            self.plots.append(self._plot(i, px, pz, s))
        self.rooms = [Room(P["px"], P["pz"], 17.5, 21.5, P["door_in"], P["door_out"]) for P in self.plots]

        self.player.shirt = (0.1, 0.4, 0.9)
        self.me = 0
        order = [p for p in range(1, 8)]
        r.shuffle(order)
        self.npcs: list[Npc] = []
        for k in range(n_npc):
            pl = order[k]
            hh = Humanoid(pos=self.plots[pl]["home"].copy(), shirt=tuple(r.uniform(0.1, 0.9, 3)), pants=tuple(r.uniform(0.1, 0.6, 3)))
            self.npcs.append(Npc(h=hh, plot=pl, name=NPC_NAMES[k], greed=float(r.uniform(0.2, 0.9)),
                                 cash=float(r.uniform(100, 3000))))
        self.owner_of = {0: 0}
        for k, n in enumerate(self.npcs):
            self.owner_of[n.plot] = k + 1
        for pl in range(8):
            self.plots[pl]["lock_until"] = FIRST_LOCK if pl in self.owner_of else 0.0

        self.cash = 100.0
        self.brainrots: dict[int, Rot] = {}
        self._next_id = 0
        self._next_spawn = 0.0
        self.br_prompts: dict[int, Prompt] = {}
        self.carrying = -1
        self.bat_cd = 0.0
        self.swing_t = -9.0
        self.networth_prev = self._networth()
        # rivals start with a couple of brainrots, so stealing is possible from minute one
        for k, n in enumerate(self.npcs):
            for s in range(int(r.integers(1, 4))):
                kind = int(r.choice(len(KINDS), p=self._weights()))
                b = self._new_br(kind, self.plots[n.plot]["slot_pos"][s].copy())
                b.state, b.owner, b.slot = "slot", n.plot, s
        self.cam.yaw = math.pi   # plot 0 faces the carpet toward -z
        self.cam.pitch = math.radians(20)

    def _weights(self):
        w = np.array([k[4] for k in KINDS], float)
        return w / w.sum()

    def _plot(self, i, px, pz, s):
        w = self.world
        col = [(0.9, 0.9, 0.9), (0.75, 0.85, 1.0), (1.0, 0.85, 0.75), (0.85, 1.0, 0.8),
               (1.0, 0.95, 0.7), (0.9, 0.8, 1.0), (0.8, 0.95, 0.95), (1.0, 0.8, 0.85)][i]
        zd, zb = pz + s * 22, pz - s * 22
        w.add((px, 0.2, pz), (36, 0.4, 44), col)
        w.add((px, 5, zb), (36, 10, 1), (0.55, 0.55, 0.6))
        w.add((px - 18, 5, pz), (1, 10, 44), (0.55, 0.55, 0.6))
        w.add((px + 18, 5, pz), (1, 10, 44), (0.55, 0.55, 0.6))
        w.add((px - 11.5, 5, zd), (13, 10, 1), (0.55, 0.55, 0.6))
        w.add((px + 11.5, 5, zd), (13, 10, 1), (0.55, 0.55, 0.6))
        w.add((px, 10.5, zd), (36, 1, 1), (0.45, 0.45, 0.5))
        door = w.add((px, 5, zd), (10, 10, 0.6), (1.0, 0.1, 0.1), visible=False)   # the collider; the lasers are just looks
        beams = [w.add((px + x, 5, zd), (0.25, 10, 0.25), (1.0, 0.1, 0.1), neon=True, collide=False) for x in (-4, -2, 0, 2, 4)]
        beams += [w.add((px, y, zd), (10, 0.25, 0.25), (1.0, 0.1, 0.1), neon=True, collide=False) for y in (2, 5, 8)]
        lock = w.add((px + 13, 0.5, zd - s * 4), (4, 0.2, 4), (0.2, 0.4, 1.0), neon=True)
        slots, pads, slot_pos = [], [], []
        for row, dz in enumerate((6, 16)):
            for dx in (-12, -4, 4, 12):
                sz = pz - s * dz
                slots.append(w.add((px + dx, 0.9, sz), (4, 1, 4), (0.3, 0.3, 0.35)))
                pads.append(w.add((px + dx, 0.5, sz + s * 3.8), (3, 0.2, 2), (0.2, 0.9, 0.3), neon=True))
                slot_pos.append(np.array([px + dx, 1.4, sz]))
        return dict(px=px, pz=pz, s=s, door=door, beams=beams, lock=lock, slots=slots, pads=pads, slot_pos=slot_pos,
                    door_out=np.array([px, 0, zd + s * 5.0]), door_in=np.array([px, 0, zd - s * 5.0]),
                    home=np.array([px, 0.4, pz]), lock_until=0.0)

    def respawn_point(self):
        return self.plots[0]["home"] + [0, 0.1, 0]

    # ── brainrot bookkeeping ──
    def _new_br(self, kind, pos) -> Rot:
        b = Rot(id=self._next_id, kind=kind, pos=np.asarray(pos, float))
        self.brainrots[b.id] = b
        self._next_id += 1
        return b

    def _free_slot(self, plot) -> int:
        used = {b.slot for b in self.brainrots.values() if b.owner == plot and b.state in ("slot", "walking")}
        for s in range(SLOTS):
            if s not in used:
                return s
        return -1

    def _inside(self, plot, p) -> bool:
        return self.rooms[plot].inside(p)

    def _locked(self, plot) -> bool:
        return self.t < self.plots[plot]["lock_until"]

    def _route(self, h: Humanoid, goal) -> np.ndarray:
        return route(h.pos, goal, self.rooms)

    def _buy(self, bid: int, buyer: int) -> bool:
        """buyer is a plot index."""
        b = self.brainrots.get(bid)
        if b is None or b.state != "conveyor":
            return False
        price = KINDS[b.kind][2]
        cash = self.cash if buyer == 0 else self.npcs[self.owner_of[buyer] - 1].cash
        slot = self._free_slot(buyer)
        if cash < price or slot < 0:
            if buyer == 0:
                self.toast("Not enough cash!" if cash < price else "Your base is full!")
            return False
        if buyer == 0:
            self.cash -= price
            self.toast(f"Bought {KINDS[b.kind][0]}!")
        else:
            self.npcs[self.owner_of[buyer] - 1].cash -= price
        b.state, b.owner, b.slot = "walking", buyer, slot
        P = self.plots[buyer]
        b.path = [P["door_out"] + [0, 0.2, 0], P["door_in"] + [0, 0.2, 0], P["slot_pos"][slot]]
        self.br_prompts.pop(bid, None)
        return True

    def _drop(self, b: Rot) -> None:
        """Knocked out of a thief's hands: it walks straight back to its owner's podium."""
        if b.carrier == 0:
            self.carrying = -1
            self.player.speed_mult = 1.0
        elif b.carrier > 0:
            n = self.npcs[b.carrier - 1]
            n.carrying, n.h.speed_mult, n.state = -1, 1.0, "idle"
        b.carrier = -1
        P = self.plots[b.owner]
        b.state = "walking"
        b.path = [P["door_out"] + [0, 0.2, 0], P["door_in"] + [0, 0.2, 0], P["slot_pos"][b.slot]] if not self._inside(b.owner, b.pos) else [P["slot_pos"][b.slot]]

    def _steal_id(self, bid: int, thief: int) -> None:
        self._steal(self.brainrots[bid], thief)

    def _steal(self, b: Rot, thief: int) -> None:
        b.state, b.carrier = "carried", thief
        if thief == 0:
            self.carrying = b.id
            self.player.speed_mult = CARRY_SLOW
        else:
            n = self.npcs[thief - 1]
            n.carrying, n.h.speed_mult, n.state = b.id, CARRY_SLOW, "escape"
        if b.owner == 0:
            self.toast("Someone is stealing your brainrot!", 3)

    def _land(self, b: Rot, new_owner: int) -> None:
        """A carried brainrot reaching its thief's base becomes theirs."""
        slot = self._free_slot(new_owner)
        if slot < 0:
            self._drop(b)
            return
        if b.carrier == 0:
            self.carrying, self.player.speed_mult = -1, 1.0
            self.toast(f"You stole {KINDS[b.kind][0]}!", 3)
        else:
            n = self.npcs[b.carrier - 1]
            n.carrying, n.h.speed_mult, n.state = -1, 1.0, "idle"
            if b.owner == 0:
                self.toast(f"{n.name} stole your {KINDS[b.kind][0]}!", 3)
        b.owner, b.slot, b.carrier, b.acc = new_owner, slot, -1, 0.0
        b.state = "slot"
        b.pos = self.plots[new_owner]["slot_pos"][slot].copy()

    def _networth(self) -> float:
        owned = sum(KINDS[b.kind][2] for b in self.brainrots.values() if b.owner == 0 and b.state in ("slot", "walking"))
        return self.cash + owned

    def income(self, plot=0) -> float:
        return sum(KINDS[b.kind][3] for b in self.brainrots.values() if b.owner == plot and b.state == "slot")

    # ── per-substep world ──
    def _set_doors(self, who_plot: int) -> None:
        w = self.world
        for k, P in enumerate(self.plots):
            on = self._locked(k)
            w.visible[P["beams"]] = on
            w.collide[P["door"]] = on and k != who_plot

    def on_key(self, k: str) -> None:
        if k == "1":
            self.equipped = None if self.equipped == "bat" else "bat"

    def on_substep(self) -> None:
        r = self.rng
        # conveyor spawns
        if self.t >= self._next_spawn:
            self._next_spawn = self.t + SPAWN_EVERY
            kind = int(r.choice(len(KINDS), p=self._weights()))
            b = self._new_br(kind, (CARPET_X[0], 0.2, r.uniform(-2, 2)))
            name, rar, price, inc, _ = KINDS[kind]
            self.br_prompts[b.id] = Prompt(pos=b.pos, action="Purchase", obj=f"{name}  ${_fmt(price)}", hold=0.5,
                                           on_trigger=("_buy", b.id, 0))
        # brainrot motion and income
        for b in list(self.brainrots.values()):
            if b.state == "conveyor":
                b.pos[0] += CONVEYOR_SPEED * DT
                if b.pos[0] > CARPET_X[1]:
                    del self.brainrots[b.id]
                    self.br_prompts.pop(b.id, None)
                    continue
            elif b.state == "walking":
                tgt = b.path[0]
                d = tgt - b.pos
                n = np.linalg.norm(d[[0, 2]])
                if n < 0.5:
                    b.path.pop(0)
                    if not b.path:
                        b.state, b.pos = "slot", self.plots[b.owner]["slot_pos"][b.slot].copy()
                else:
                    b.pos[[0, 2]] += d[[0, 2]] / n * min(n, 9.0 * DT)
                    b.pos[1] = tgt[1]
            elif b.state == "slot":
                b.acc += KINDS[b.kind][3] * DT
            elif b.state == "carried":
                ch = self.player if b.carrier == 0 else self.npcs[b.carrier - 1].h
                b.pos = ch.pos + [0, 5.6, 0]
            if b.id in self.br_prompts:
                self.br_prompts[b.id].pos = b.pos + [0, 2.5, 0]
        self._steal_prompts()
        self.prompts = list(self.br_prompts.values())

        # the player's own base: lock pad, collect pads, delivering a stolen brainrot
        p = self.player
        P0 = self.plots[0]
        if p.grounded and p.ground == P0["lock"] and not self._locked(0):
            P0["lock_until"] = self.t + LOCK_S
            self.toast("Base locked for 60s")
        if p.grounded and p.ground in P0["pads"]:
            s = P0["pads"].index(p.ground)
            for b in self.brainrots.values():
                if b.owner == 0 and b.slot == s and b.state == "slot" and b.acc >= 1:
                    self.cash += b.acc
                    b.acc = 0.0
        if self.carrying >= 0 and self._inside(0, p.pos):
            self._land(self.brainrots[self.carrying], 0)
        self.bat_cd = max(0.0, self.bat_cd - DT)

        for k, n in enumerate(self.npcs):
            self._npc_step(k, n)
        self._set_doors(0)

        nw = self._networth()
        self.reward += math.log1p(max(nw, 0)) - math.log1p(max(self.networth_prev, 0))
        self.networth_prev = nw
        if self.steps >= self.max_steps:
            self.done = True

    def _steal_prompts(self) -> None:
        for b in self.brainrots.values():
            ok = b.state == "slot" and b.owner != 0
            if ok and b.id not in self.br_prompts:
                self.br_prompts[b.id] = Prompt(pos=b.pos + [0, 2.5, 0], action="Steal", obj=KINDS[b.kind][0], hold=1.0,
                                               on_trigger=("_steal_id", b.id, 0), enabled=("_can_steal", b.id))
            elif not ok and b.id in self.br_prompts and b.state != "conveyor":
                del self.br_prompts[b.id]

    def _can_steal(self, bid) -> bool:
        b = self.brainrots.get(bid)
        return b is not None and b.state == "slot" and self.carrying < 0 and self._inside(b.owner, self.player.pos)

    def on_ui(self, id_: str) -> None: ...

    def on_world_click(self, part: int) -> None: ...

    def on_tool_click(self) -> bool:
        if self.equipped != "bat":
            return False
        self._swing(self.player, 0)
        return True

    def _swing(self, h: Humanoid, who: int) -> None:
        cd = self.bat_cd if who == 0 else self.npcs[who - 1].bat_cd
        if cd > 0:
            return
        if who == 0:
            self.bat_cd, self.swing_t = BAT_CD, self.t
        else:
            self.npcs[who - 1].bat_cd = BAT_CD
        fx, fz = math.sin(h.face), math.cos(h.face)
        targets = [(0, self.player)] + [(k + 1, n.h) for k, n in enumerate(self.npcs)]
        for tid, th in targets:
            if tid == who:
                continue
            d = th.pos - h.pos
            dist = float(np.hypot(d[0], d[2]))
            if dist < BAT_RANGE and (d[0] * fx + d[2] * fz) / max(dist, 1e-6) > 0.2:
                th.vel[0] += d[0] / max(dist, 1e-6) * 40
                th.vel[2] += d[2] / max(dist, 1e-6) * 40
                th.vel[1] = 20
                th.grounded = False
                th.stun = 1.5
                carried = self.carrying if tid == 0 else self.npcs[tid - 1].carrying
                if carried >= 0:
                    owner = self.brainrots[carried].owner
                    self._drop(self.brainrots[carried])
                    if who == 0 and owner == 0:
                        self.toast("They dropped your brainrot!")
                if who == 0:
                    self.toast(f"Hit {self.npcs[tid - 1].name}!", 1.5)
                elif tid == 0:
                    self.toast("You got bonked!", 1.5)

    # ── rivals ──
    def _npc_step(self, k: int, n: Npc) -> None:
        h, me = n.h, k + 1
        n.bat_cd = max(0.0, n.bat_cd - DT)
        n.timer -= DT
        P = self.plots[n.plot]
        goal, jump = None, False

        # defend: chase anyone carrying my brainrot
        thief = None
        for b in self.brainrots.values():
            if b.state == "carried" and b.owner == n.plot:
                thief = self.player if b.carrier == 0 else self.npcs[b.carrier - 1].h
        if thief is not None and np.linalg.norm(thief.pos - h.pos) < 60:
            goal = thief.pos
            if np.linalg.norm(thief.pos - h.pos) < BAT_RANGE - 1:
                h.face = math.atan2(thief.pos[0] - h.pos[0], thief.pos[2] - h.pos[2])
                self._swing(h, me)
        elif n.state == "escape" and n.carrying >= 0:
            goal = P["home"]
            if self._inside(n.plot, h.pos):
                self._land(self.brainrots[n.carrying], n.plot)
        else:
            if n.timer <= 0 or n.state == "idle":
                self._npc_choose(k, n)
            if n.state == "collect":
                goal = self.world.c[P["pads"][n.target]]
                if np.linalg.norm((goal - h.pos)[[0, 2]]) < 1.2:
                    for b in self.brainrots.values():
                        if b.owner == n.plot and b.slot == n.target and b.state == "slot":
                            n.cash += b.acc
                            b.acc = 0.0
                    n.state = "idle"
            elif n.state == "lock":
                goal = self.world.c[P["lock"]]
                if np.linalg.norm((goal - h.pos)[[0, 2]]) < 1.5:
                    P["lock_until"] = self.t + LOCK_S
                    n.state = "idle"
            elif n.state == "buy":
                b = self.brainrots.get(n.target)
                if b is None or b.state != "conveyor":
                    n.state = "idle"
                else:
                    goal = b.pos
                    if np.linalg.norm(b.pos - h.pos) < 6:
                        self._buy(b.id, n.plot)
                        n.state = "idle"
            elif n.state == "steal":
                b = self.brainrots.get(n.target)
                if b is None or b.state != "slot" or self._locked(b.owner) and not self._inside(b.owner, h.pos):
                    n.state = "idle"
                else:
                    goal = b.pos
                    if np.linalg.norm((b.pos - h.pos)[[0, 2]]) < 4:
                        n.timer = min(n.timer, 1.5)
                        n.state = "stealing"
            elif n.state == "stealing":
                b = self.brainrots.get(n.target)
                if b is None or b.state != "slot":
                    n.state = "idle"
                elif n.timer <= 0:
                    self._steal(b, me)
            elif n.state == "wander":
                goal = n.h.pos + [math.sin(n.greed * 7 + self.t * 0.1), 0, math.cos(self.t * 0.13)]
        move = np.zeros(2)
        if goal is not None and n.state != "stealing":
            nxt = self._route(h, goal)
            d = (nxt - h.pos)[[0, 2]]
            if np.linalg.norm(d) > 0.8:
                move = d
        self._set_doors(n.plot)
        humanoid_step(h, self.world, move, jump)
        if h.pos[1] < -20:
            h.pos, h.vel = P["home"].copy(), np.zeros(3)

    def _npc_choose(self, k: int, n: Npc) -> None:
        r = self.rng
        n.timer = float(r.uniform(4, 10))
        mine = [b for b in self.brainrots.values() if b.owner == n.plot and b.state == "slot"]
        if not self._locked(n.plot) and r.random() < 0.35:
            n.state = "lock"
            return
        rich = [b for b in mine if b.acc > 30]
        if rich and r.random() < 0.5:
            n.state, n.target = "collect", max(rich, key=lambda b: b.acc).slot
            return
        buyable = [b for b in self.brainrots.values() if b.state == "conveyor" and KINDS[b.kind][2] <= n.cash and b.pos[0] < 60]
        if buyable and self._free_slot(n.plot) >= 0 and r.random() < 0.7:
            n.state, n.target = "buy", max(buyable, key=lambda b: KINDS[b.kind][3]).id
            return
        if r.random() < n.greed * 0.4:
            victims = [b for b in self.brainrots.values() if b.state == "slot" and b.owner != n.plot and not self._locked(b.owner)]
            if victims:
                n.state, n.target = "steal", max(victims, key=lambda b: KINDS[b.kind][3] + r.uniform(0, 5)).id
                n.timer = 25.0
                return
        n.state = "wander"

    # ── rendering ──
    def dynamic_boxes(self) -> list:
        out = []
        for n in self.npcs:
            out += avatar_boxes(n.h)
        for b in self.brainrots.values():
            out += _br_boxes(b)
        if self.equipped == "bat":   # a wide horizontal arc across the front for 0.3 s after each swing
            h = self.player
            u = (self.t - self.swing_t) / 0.3
            ang = h.face + (1.4 - 2.8 * u if 0 <= u < 1 else 0.6)
            ox, oz = math.sin(ang), math.cos(ang)
            out.append(((h.pos[0] + ox * 2.6, h.pos[1] + 3.0, h.pos[2] + oz * 2.6), (0.5, 0.5, 3.5), ang, (0.55, 0.35, 0.15)))
        return out

    def hud(self) -> list:
        rows = [["You", "$" + _fmt(self.cash)]] + [[n.name[:10], "$" + _fmt(n.cash)] for n in self.npcs]
        els = [dict(kind="leaderboard", cols=["Cash"], rows=rows),
               dict(kind="text", rect=(0.02, 0.9, 0.3, 0.06), text=f"${_fmt(self.cash)}   +${_fmt(self.income())}/s", size=28, color=(0.4, 1.0, 0.4)),
               dict(kind="button", rect=(0.45, 0.9, 0.1, 0.07), text="1  Bat", color=(0.25, 0.25, 0.25) if self.equipped != "bat" else (0.2, 0.55, 1.0), size=18)]
        near = sorted(self.brainrots.values(), key=lambda b: np.linalg.norm(b.pos - self.player.pos))[:8]
        for b in near:
            name, rar, price, inc, _ = KINDS[b.kind]
            txt = f"{name}\n{rar}  ${_fmt(inc)}/s" + (f"\n${_fmt(price)}" if b.state == "conveyor" else (f"\n${_fmt(b.acc)}" if b.state == "slot" else ""))
            bb = self.billboard(b.pos + [0, 4.5, 0], txt, color=RARITY_COL[rar], size=0.8, max_dist=45)
            if bb:
                els.append(bb)
        for k, P in enumerate(self.plots):
            label = ("YOUR BASE" if k == 0 else (self.npcs[self.owner_of[k] - 1].name if k in self.owner_of else "Empty"))
            lt = P["lock_until"] - self.t
            txt = label + (f"\nLocked {lt:.0f}s" if lt > 0 else "\nUnlocked")
            bb = self.billboard(P["door_out"] + [0, 12, 0], txt, color=(1, 1, 0.4) if k == 0 else (1, 1, 1), max_dist=90)
            if bb:
                els.append(bb)
        return els

    def potential(self) -> float:
        """Closer to wherever the expert would go next (a brainrot to buy, a pad to collect, home) is progress."""
        self.goal = None
        self.expert()
        return 0.0 if self.goal is None else -0.01 * float(np.linalg.norm((self.goal - self.player.pos)[[0, 2]]))

    def info(self) -> dict:
        return dict(cash=self.cash, income=self.income(), owned=sum(1 for b in self.brainrots.values() if b.owner == 0),
                    networth=self._networth())

    # ── expert ──
    def expert(self) -> Action:
        p, P0 = self.player, self.plots[0]
        cur = self.cursor
        if self.carrying >= 0:
            self.say("carrying a stolen brainrot, run back to my base")
            return self._go(P0["home"])
        for b in self.brainrots.values():   # defend
            if b.state == "carried" and b.owner == 0 and b.carrier > 0:
                th = self.npcs[b.carrier - 1].h
                if self.equipped != "bat":
                    self.say("someone is stealing my brainrot, equip the bat")
                    return Action(keys=frozenset({"1"}), cursor=cur)
                self.say("someone is stealing my brainrot, chase and hit them")
                a = self._go(th.pos, stop=2.5)
                if np.linalg.norm(th.pos - p.pos) < BAT_RANGE - 1.5:
                    a.click = True
                    a.cursor = (0.5, 0.5)
                return a
        if self.equipped == "bat":
            self.say("bat is out, put it away")
            return Action(keys=frozenset({"1"}), cursor=cur)
        if not self._locked(0) and any(np.linalg.norm(n.h.pos - P0["home"]) < 45 for n in self.npcs):
            self.say("base is unlocked and rivals are near, step on the lock")
            return self._go(self.world.c[P0["lock"]], stop=0.5)
        mine = [b for b in self.brainrots.values() if b.owner == 0 and b.state == "slot"]
        unc = sum(b.acc for b in mine)
        if mine and unc > max(40, 0.25 * self.cash):
            b = max(mine, key=lambda b: b.acc)
            self.say("cash piled up on my pads, step on them to collect")
            return self._go(self.world.c[P0["pads"][b.slot]], stop=0.4)
        free = self._free_slot(0) >= 0
        worst = min((KINDS[b.kind][3] for b in mine), default=0)
        cands = [b for b in self.brainrots.values() if b.state == "conveyor" and KINDS[b.kind][2] <= self.cash
                 and (free or KINDS[b.kind][3] > worst) and b.pos[0] < 110]
        if cands and free:
            b = max(cands, key=lambda b: KINDS[b.kind][3] - 0.02 * np.linalg.norm(b.pos - p.pos))
            self.say("affordable brainrot on the carpet, walk beside it and buy it")
            a = self._go(b.pos + [CONVEYOR_SPEED * 0.6, 0, 0], stop=3.0)
            pr = self.active_prompt()
            if pr is self.br_prompts.get(b.id):
                a.keys = a.keys | {"e"}
            return a
        best_inc = max((KINDS[b.kind][3] for b in mine), default=0)
        targets = [b for b in self.brainrots.values() if b.state == "slot" and b.owner != 0 and not self._locked(b.owner)
                   and KINDS[b.kind][3] > best_inc and free]
        if targets:
            b = max(targets, key=lambda b: KINDS[b.kind][3] - 0.05 * np.linalg.norm(b.pos - p.pos))
            self.say("rival base is unlocked with a better brainrot, steal it")
            a = self._go(b.pos, stop=2.5)
            pr = self.active_prompt()
            if pr is not None and pr is self.br_prompts.get(b.id):
                a.keys = frozenset({"e"})
            return a
        self.say("nothing worth buying, wait at my base")
        return self._go(P0["home"], stop=4.0)

    def _go(self, goal, stop=1.5) -> Action:
        self.goal = np.asarray(goal, float)
        nxt = self._route(self.player, np.asarray(goal, float))
        return self.steer_to(nxt, stop_dist=stop if np.allclose(nxt, goal) else 1.0)


def _br_boxes(b: Rot) -> list:
    k = b.kind
    s = 1.0 + 0.06 * k
    c = BODY_COL[k]
    acc = RARITY_COL[KINDS[k][1]]
    x, y, z = b.pos
    return [((x, y + 1.3 * s, z), (2.2 * s, 1.6 * s, 1.8 * s), 0.0, c),
            ((x, y + 2.7 * s, z + 0.3), (1.5 * s, 1.3 * s, 1.5 * s), 0.0, acc, k >= 7),
            ((x - 0.6 * s, y + 0.25 * s, z), (0.6 * s, 0.5 * s + 0.5, 0.6 * s), 0.0, c),
            ((x + 0.6 * s, y + 0.25 * s, z), (0.6 * s, 0.5 * s + 0.5, 0.6 * s), 0.0, c)]


def _fmt(v: float) -> str:
    v = float(v)
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(v) >= div:
            return f"{v / div:.1f}{unit}"
    return f"{v:.0f}"



if __name__ == "__main__":
    import time
    env = StealABrainrot(seed=0, max_steps=3000)
    t0 = time.time()
    R = 0.0
    while not env.done:
        r, done, info = env.step(env.expert())
        R += r
    print(f"expert: {info}, return {R:.2f}, {(time.time() - t0) / env.steps * 1000:.1f} ms/tick")
    env = StealABrainrot(seed=0, max_steps=3000)
    rng = np.random.default_rng(0)
    from core.actions import KEYS
    while not env.done:
        env.step(Action(keys=frozenset(k for k in KEYS if rng.random() < 0.1), look_x=float(rng.normal(0, 10))))
    print(f"random: {env.info()}")
