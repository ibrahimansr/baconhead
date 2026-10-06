"""Adopt Me-style pet care: the cursor-heavy one. Almost every task is a chain of clicks:

  hungry/thirsty  click the shop counter → shop panel → click an item → confirm Buy → Backpack → click the
                  item to equip it → click your pet → pet menu → "Use"
  sleepy/dirty    click the bed/bathtub in your house (ClickDetectors), stay near it
  sick            click the doctor → confirm
  bored/camping   stand in the Playground / Campsite
  travel          Map button → click a location to teleport, like Adopt Me's map
Buttons light up under the cursor. Modal pop-ups (daily reward, offers) sink clicks until closed.
Bucks per task follow the wiki: food/drink 6, sleepy 14, dirty 7.
"""

from __future__ import annotations

import math

import numpy as np

from core.actions import Action
from sim.core import DT, Env, Room, route

TASKS = {   # name: (how, bucks, seconds, where)
    "hungry": ("food:hungry", 6, 3.0, "Pizza Shop"),
    "thirsty": ("food:thirsty", 6, 3.0, "Lemonade Stand"),
    "sleepy": ("click:bed", 14, 6.0, "your house"),
    "dirty": ("click:tub", 7, 5.0, "your house"),
    "bored": ("zone:playground", 8, 8.0, "the Playground"),
    "camping": ("zone:campsite", 8, 8.0, "the Campsite"),
    "sick": ("click:doctor", 10, 2.0, "the Hospital"),
    "attention": ("pet:petit", 3, 0.5, "your pet (click it)"),
}
TASK_COL = {"hungry": (0.95, 0.55, 0.2), "thirsty": (0.3, 0.6, 1.0), "sleepy": (0.5, 0.4, 0.9), "dirty": (0.55, 0.4, 0.25),
            "bored": (0.95, 0.8, 0.2), "camping": (0.3, 0.7, 0.3), "sick": (0.9, 0.3, 0.4), "choose": (0.6, 0.2, 0.8),
            "attention": (1.0, 0.45, 0.7)}
FOODS = {"pizza": ("hungry", 4, (1.0, 0.75, 0.3)), "apple": ("hungry", 2, (0.9, 0.15, 0.15)), "donut": ("hungry", 3, (0.95, 0.6, 0.8)),
         "lemonade": ("thirsty", 3, (1.0, 1.0, 0.5)), "water": ("thirsty", 1, (0.5, 0.8, 1.0)), "tea": ("thirsty", 2, (0.6, 0.4, 0.2))}
SHOPS = {"pizza_shop": ("Pizza Shop", ("pizza", "apple", "donut")), "stand": ("Lemonade Stand", ("lemonade", "water", "tea"))}
SHOP_OF = {"hungry": "pizza_shop", "thirsty": "stand"}
TELEPORTS = {"Home": (0.0, 0.0, 40.0), "Pizza Shop": (-70.0, 0.0, -5.0), "Lemonade Stand": (70.0, 0.0, -12.0),
             "Playground": (-70.0, 0.0, -100.0), "Campsite": (70.0, 0.0, -100.0), "Hospital": (0.0, 0.0, -133.0)}
TP_DIST = 45.0   # the expert takes the map for trips longer than this
PET_FOLLOW = 3.5


def _building(w, cx, cz, sx, sz, col, door_w=6.0, h=12.0, side=-1):
    """Four walls, a roof, and a doorway on the `side` z face (-1 or +1). Returns the Room for routing."""
    zd = cz + side * sz / 2
    w.add((cx, h / 2, cz - side * sz / 2), (sx, h, 1), col)
    w.add((cx - sx / 2, h / 2, cz), (1, h, sz), col)
    w.add((cx + sx / 2, h / 2, cz), (1, h, sz), col)
    rest = (sx - door_w) / 2
    w.add((cx - door_w / 2 - rest / 2, h / 2, zd), (rest, h, 1), col)
    w.add((cx + door_w / 2 + rest / 2, h / 2, zd), (rest, h, 1), col)
    w.add((cx, h - 2, zd), (door_w, 4, 1), col)
    w.add((cx, h + 0.5, cz), (sx + 2, 1, sz + 2), tuple(np.array(col) * 0.7))
    w.add((cx, 0.05, cz), (sx - 1, 0.1, sz - 1), (0.75, 0.6, 0.45))                     # wood floor
    return Room(cx, cz, sx / 2 - 0.5, sz / 2 - 0.5, np.array([cx, 0, zd - side * 3.0]), np.array([cx, 0, zd + side * 4.0]))


class AdoptMe(Env):
    name = "adoptme"

    def score(self) -> float:
        return float(self.pet_xp)           # pet tasks completed

    def build(self) -> None:
        r, w = self.rng, self.world
        self.max_steps = int(self.kw.get("max_steps", 6000))
        w.add((0, -0.5, 0), (500, 1, 500), (0.4, 0.68, 0.33))
        for x in (-30, 30):
            w.add((x, 0.02, -40), (10, 0.05, 260), (0.3, 0.3, 0.32))
        w.add((0, 0.03, -55), (160, 0.05, 10), (0.3, 0.3, 0.32))
        for x in (-250, 250):
            w.add((x, 20, 0), (1, 40, 500), (0, 0, 0), visible=False)
        for z in (-250, 250):
            w.add((0, 20, z), (500, 40, 1), (0, 0, 0), visible=False)

        self.rooms = []
        self.rooms.append(_building(w, 0, 60, 34, 26, (0.95, 0.85, 0.7)))                  # your house
        self.bed = w.add((-10, 1.0, 66), (6, 2, 9), (0.4, 0.5, 0.95), tag="bed")
        w.add((-10, 1.5, 70), (6, 3, 1), (0.5, 0.3, 0.2))
        self.tub = w.add((10, 1.0, 66), (5, 2, 8), (0.95, 0.95, 1.0), tag="tub")
        w.add((10, 1.9, 66), (4, 0.2, 7), (0.5, 0.8, 1.0), neon=True, collide=False)
        w.add((0, 1.5, 71), (6, 3, 3), (0.1, 0.1, 0.1))                                     # piano, decorative
        self.rooms.append(_building(w, -70, -20, 26, 20, (0.95, 0.4, 0.3), side=1))         # pizza shop, door toward spawn
        w.add((-70, 1.5, -26), (14, 3, 3), (0.6, 0.4, 0.25))
        self.pizza = w.add((-70, 3.3, -26), (2.5, 0.6, 2.5), (1.0, 0.75, 0.3), tag="shop:pizza_shop")
        self.stand = w.add((70, 1.5, -20), (10, 3, 4), (1.0, 0.95, 0.4))                    # lemonade stand
        w.add((70, 6.5, -20), (12, 0.6, 6), (1.0, 0.9, 0.2))
        for x in (65, 75):
            w.add((x, 3.5, -21.5), (0.5, 7, 0.5), (0.9, 0.9, 0.9))
        self.lemonade = w.add((70, 3.6, -20), (1.2, 1.2, 1.2), (1.0, 1.0, 0.5), tag="shop:stand")
        self.zones = {"playground": (np.array([-70.0, 0, -100]), 14.0), "campsite": (np.array([70.0, 0, -100]), 12.0)}
        for i, (dx, dz, sz, c) in enumerate([(-6, 0, (4, 6, 4), (1, 0.3, 0.3)), (5, 3, (3, 3, 10), (0.3, 0.6, 1)),
                                             (0, -7, (8, 1, 2), (1, 0.85, 0.2)), (-8, 8, (2, 8, 2), (0.3, 0.9, 0.4))]):
            w.add((-70 + dx, sz[1] / 2, -100 + dz), sz, c)
        w.add((-70, 0.03, -100), (28, 0.05, 28), (0.85, 0.75, 0.55))
        w.add((70, 0.5, -100), (2.5, 1, 2.5), (1.0, 0.5, 0.1), neon=True)                  # campfire
        for dx, dz in ((-7, 4), (6, 6), (0, -8)):
            w.add((70 + dx, 2, -100 + dz), (5, 4, 5), (0.3, 0.55, 0.3), yaw=0.4)
        self.rooms.append(_building(w, 0, -150, 30, 22, (0.95, 0.95, 0.97), side=1))       # hospital
        w.add((0, 7, -138.4), (8, 2, 0.5), (0.9, 0.2, 0.2), neon=True)
        self.doctor = w.add((0, 2.5, -155), (2, 5, 1), (0.95, 0.95, 0.95), tag="doctor")
        w.add((0, 5.6, -155), (1.2, 1.2, 1.2), (0.95, 0.8, 0.6))
        self.signs = [((0, 14, 46), "Your House"), ((-70, 14, -10), "Pizza Shop"), ((70, 9, -20), "Lemonade Stand"),
                      ((-70, 10, -100), "Playground"), ((70, 8, -100), "Campsite"), ((0, 14, -139), "Hospital")]

        # the pet: world parts moved by hand every substep, so clicks and the camera both see it
        self.pet_parts = [w.add((0, 0.8, 16), (1.6, 1.2, 2.4), (0.85, 0.6, 0.35), collide=False, tag="pet"),
                          w.add((0, 1.7, 17.4), (1.2, 1.1, 1.2), (0.85, 0.6, 0.35), collide=False, tag="pet"),
                          w.add((0, 2.4, 17.4), (1.4, 0.5, 0.4), (0.5, 0.35, 0.2), collide=False, tag="pet")]
        self.pet_pos = np.array([0.0, 0.0, 16.0])
        self.pet_face = 0.0
        self.pet_xp = 0

        self.bucks = 50
        self.inventory = {k: 0 for k in FOODS}
        self.needs: list[str] = []
        self.next_need = 3.0
        self.activity = None          # (task, seconds left, anchor, radius)
        self.ui = None                # None | "backpack" | "map" | "pet" | ("shop", s) | ("buy", item) | "doctor" | "choose" | "daily" | "offer"
        self.choices: list[str] = []
        self.next_popup = float(r.uniform(30, 60))
        self.cam.yaw = math.pi
        self.cam.pitch = math.radians(20)

    def respawn_point(self):
        return np.array([0.0, 0.0, 20.0])

    # ── clicks ──
    def on_ui(self, id_: str) -> None:
        if id_ == "modal_bg":
            return
        if id_ in ("backpack", "map"):
            self.ui = None if self.ui == id_ else id_
        elif id_.startswith("tp:"):
            p = np.array(TELEPORTS[id_[3:]])
            self.player.pos, self.player.vel = p.copy(), np.zeros(3)
            self.pet_pos = p + [0, 0, -2]
            self.ui = None
        elif id_.startswith("item:"):
            it = id_[5:]
            if self.inventory[it] > 0:
                self.equipped, self.ui = it, None
                self.toast("Click your pet to use it")
        elif id_.startswith("shopitem:"):
            self.ui = ("buy", id_[9:])
        elif id_ == "buy" and isinstance(self.ui, tuple):
            it = self.ui[1]
            if self.bucks >= FOODS[it][1]:
                self.bucks -= FOODS[it][1]
                self.inventory[it] += 1
                self.toast(f"Bought {it.title()}! Open your Backpack")
            else:
                self.toast("Not enough Bucks!")
            self.ui = None
        elif id_ == "use" and self.ui == "pet":
            self.ui = None
            it = self.equipped
            need = FOODS[it][0] if it else None
            if need in self.needs and self.activity is None:
                self.inventory[it] -= 1
                if self.inventory[it] == 0:
                    self.equipped = None
                self._start(need, self.pet_pos, 1e9)
            else:
                self.toast(f"Your pet isn't {need}")
        elif id_ == "petit":
            self.ui = None
            self.toast("Your pet loves you!")
            if "attention" in self.needs and self.activity is None:
                self._start("attention", self.pet_pos, 1e9)
        elif id_ in ("cancel", "close"):
            self.ui = None
        elif id_ == "heal":
            self.ui = None
            if "sick" in self.needs:
                self._start("sick", self.world.c[self.doctor], 30)
        elif id_ == "claim":
            self.bucks += 10
            self.reward += 0.5
            self.ui = None
        elif id_.startswith("task:"):
            k = int(id_[5:])
            if k < len(self.needs):
                n = self.needs[k]
                if n == "choose":
                    self.ui = "choose"
                    self.choices = list(self.rng.choice([t for t in TASKS if t not in self.needs], 3, replace=False))
                else:
                    self.toast(f"Your pet is {n}! Go to {TASKS[n][3]}")
        elif id_.startswith("pick:"):
            n = self.choices[int(id_[5:])]
            if "choose" in self.needs:
                self.needs[self.needs.index("choose")] = n
            self.ui = None

    def on_world_click(self, part: int) -> None:
        if self.ui not in (None, "backpack"):
            return
        tag = self.world.tags[part] or ""
        if tag.startswith("shop:"):
            self.ui = ("shop", tag[5:])
        elif tag == "doctor":
            self.ui = "doctor"
        elif tag == "pet":
            self.ui = "pet"
        elif tag in ("bed", "tub"):
            need = {"bed": "sleepy", "tub": "dirty"}[tag]
            if need in self.needs and self.activity is None:
                self._start(need, self.world.c[part], 25)
            else:
                self.toast("Your pet isn't " + need)

    def _start(self, need, anchor, radius):
        self.activity = [need, TASKS[need][2], np.asarray(anchor, float).copy(), radius]

    # ── simulation ──
    def on_substep(self) -> None:
        r = self.rng
        self.next_need -= DT
        if self.next_need <= 0 and len(self.needs) < 4:
            pool = [t for t in TASKS if t not in self.needs] + (["choose"] if "choose" not in self.needs else [])
            self.needs.append(str(r.choice(pool)))
            self.next_need = float(r.uniform(8, 20))
        if self.t >= self.next_popup and self.ui is None:
            self.ui = "daily" if r.random() < 0.5 else "offer"
            self.next_popup = self.t + float(r.uniform(40, 90))

        # zone needs start themselves when you're standing in the zone
        for need in ("bored", "camping"):
            c, rad = self.zones[TASKS[need][0][5:]]
            if need in self.needs and self.activity is None and np.linalg.norm((self.player.pos - c)[[0, 2]]) < rad:
                self._start(need, c, rad)
        if self.activity is not None:
            need, left, anchor, rad = self.activity
            if np.linalg.norm((self.player.pos - anchor)[[0, 2]]) > rad + (2 if rad < 1e8 else 0):
                self.activity = None
                self.toast("Task cancelled, you walked away")
            else:
                self.activity[1] -= DT
                if self.activity[1] <= 0:
                    self.needs.remove(need)
                    b = TASKS[need][1]
                    self.bucks += b
                    self.pet_xp += 1
                    self.reward += 0.1 * b
                    self.toast(f"+{b} Bucks!")
                    self.activity = None
        self._pet()
        if self.steps >= self.max_steps:
            self.done = True

    def _pet(self) -> None:
        p = self.player
        if self.activity and self.activity[0] in ("sleepy", "dirty"):
            goal = self.activity[2] + [0, 0, 0]
        else:
            goal = p.pos - np.array([math.sin(p.face), 0, math.cos(p.face)]) * PET_FOLLOW
        d = (goal - self.pet_pos)[[0, 2]]
        n = float(np.linalg.norm(d))
        if n > 40:
            self.pet_pos = goal.copy()      # pets teleport to you when left far behind
        elif n > 0.3:
            step = min(n, 18 * DT)
            self.pet_pos[[0, 2]] += d / n * step
            self.pet_face = math.atan2(d[0], d[1])
        y = 2.3 if self.activity and self.activity[0] in ("sleepy", "dirty") else 0.0
        fx, fz = math.sin(self.pet_face), math.cos(self.pet_face)
        w = self.world
        for i, (off, up) in zip(self.pet_parts, ((0, 0.8), (1.4, 1.7), (1.4, 2.4))):
            w.c[i] = w.c0[i] = self.pet_pos + [fx * off, up + y, fz * off]
            w.yaw[i] = w.yaw0[i] = self.pet_face

    # ── UI ──
    def hud(self) -> list:
        els = [dict(kind="panel", rect=(0.42, 0.015, 0.16, 0.06), color=(0.1, 0.35, 0.1), alpha=0.8),
               dict(kind="text", rect=(0.42, 0.015, 0.16, 0.06), text=f"$ {self.bucks}", size=26, color=(0.5, 1.0, 0.4), align="center"),
               dict(kind="button", id="backpack", rect=(0.37, 0.88, 0.12, 0.09), text="Backpack", color=(0.2, 0.55, 0.95)),
               dict(kind="button", id="map", rect=(0.51, 0.88, 0.12, 0.09), text="Map", color=(0.25, 0.7, 0.45)),
               dict(kind="bar", rect=(0.02, 0.93, 0.18, 0.03), frac=(self.pet_xp % 5) / 5, color=(0.9, 0.5, 0.9)),
               dict(kind="text", rect=(0.02, 0.88, 0.18, 0.05), text=["Newborn", "Junior", "Pre-Teen", "Teen", "Post-Teen", "Full Grown"][min(5, self.pet_xp // 5)], size=16)]
        for k, n in enumerate(self.needs):
            els.append(dict(kind="icon", id=f"task:{k}", rect=(0.015, 0.3 + 0.13 * k, 0.065, 0.115), color=TASK_COL[n],
                            text=n.title() if n != "choose" else "?", size=13))
        if self.equipped:
            els.append(dict(kind="text", rect=(0.35, 0.8, 0.3, 0.06), text=f"Holding: {self.equipped.title()} x{self.inventory[self.equipped]}",
                            size=20, align="center"))
        for pos, txt in self.signs:
            b = self.billboard(pos, txt, size=1.3, max_dist=150)
            if b:
                els.append(b)
        if self.activity:
            need, left, _, _ = self.activity
            uv = self.cam.project(self.pet_pos + [0, 4, 0], self.aspect)
            if uv:
                els.append(dict(kind="bar", rect=(uv[0] - 0.05, uv[1], 0.1, 0.02), frac=1 - left / TASKS[need][2], color=TASK_COL[need]))
        els += self._ui_els()
        return els

    def _ui_els(self) -> list:
        u = self.ui
        X = lambda x, y: dict(kind="button", id="close", rect=(x, y, 0.035, 0.06), text="X", color=(0.85, 0.2, 0.2))
        if u is None:
            return []
        if u == "backpack":
            els = [dict(kind="panel", rect=(0.28, 0.4, 0.44, 0.46), color=(0.12, 0.12, 0.15), alpha=0.85),
                   dict(kind="text", rect=(0.28, 0.41, 0.44, 0.06), text="Backpack", size=24, align="center"), X(0.675, 0.41)]
            for j, it in enumerate(FOODS):
                els.append(dict(kind="button", id=f"item:{it}", rect=(0.3 + 0.14 * (j % 3), 0.49 + 0.18 * (j // 3), 0.12, 0.15),
                                text=f"{it.title()}\nx{self.inventory[it]}", size=15,
                                color=(0.3, 0.3, 0.35) if self.inventory[it] == 0 else FOODS[it][2]))
            return els
        if u == "map":
            els = [dict(kind="panel", rect=(0.25, 0.2, 0.5, 0.55), color=(0.15, 0.3, 0.2), alpha=0.9),
                   dict(kind="text", rect=(0.25, 0.21, 0.5, 0.07), text="Map", size=26, align="center"), X(0.705, 0.21)]
            for j, loc in enumerate(TELEPORTS):
                els.append(dict(kind="button", id=f"tp:{loc}", rect=(0.28 + 0.15 * (j % 3), 0.32 + 0.2 * (j // 3), 0.13, 0.15),
                                text=loc.replace(" ", "\n"), color=(0.35, 0.55, 0.9), size=15))
            return els
        if u == "pet":
            els = [dict(kind="panel", rect=(0.38, 0.3, 0.24, 0.36), color=(0.12, 0.12, 0.15), alpha=0.85),
                   dict(kind="text", rect=(0.38, 0.31, 0.24, 0.06), text="Your Pet", size=22, align="center")]
            if self.equipped:
                els.append(dict(kind="button", id="use", rect=(0.4, 0.39, 0.2, 0.07), text=f"Use {self.equipped.title()}", color=(0.2, 0.75, 0.3), size=16))
            els += [dict(kind="button", id="petit", rect=(0.4, 0.48, 0.2, 0.07), text="Pet", color=(0.9, 0.5, 0.9), size=16),
                    dict(kind="button", id="close", rect=(0.4, 0.57, 0.2, 0.07), text="Close", color=(0.5, 0.5, 0.55), size=16)]
            return els
        if isinstance(u, tuple) and u[0] == "shop":
            title, items = SHOPS[u[1]]
            els = [dict(kind="panel", id="modal_bg", rect=(0, 0, 1, 1), color=(0, 0, 0), alpha=0.25),
                   dict(kind="panel", rect=(0.27, 0.28, 0.46, 0.42), color=(0.95, 0.9, 0.8), alpha=0.97),
                   dict(kind="text", rect=(0.27, 0.3, 0.46, 0.08), text=title, size=26, align="center", color=(0.3, 0.15, 0.05)), X(0.685, 0.29)]
            for j, it in enumerate(items):
                els.append(dict(kind="button", id=f"shopitem:{it}", rect=(0.3 + 0.14 * j, 0.42, 0.12, 0.2),
                                text=f"{it.title()}\n${FOODS[it][1]}", color=FOODS[it][2], fg=(0.1, 0.1, 0.1), size=16))
            return els
        els = [dict(kind="panel", id="modal_bg", rect=(0, 0, 1, 1), color=(0, 0, 0), alpha=0.25),
               dict(kind="panel", rect=(0.3, 0.3, 0.4, 0.36), color=(0.95, 0.95, 0.97), alpha=0.97)]
        if isinstance(u, tuple):
            it = u[1]
            els += [dict(kind="text", rect=(0.3, 0.33, 0.4, 0.1), text=f"Buy {it.title()} for ${FOODS[it][1]}?", size=24, align="center"),
                    dict(kind="button", id="buy", rect=(0.34, 0.52, 0.14, 0.09), text="Buy", color=(0.2, 0.75, 0.3)),
                    dict(kind="button", id="cancel", rect=(0.52, 0.52, 0.14, 0.09), text="Cancel", color=(0.8, 0.25, 0.25))]
        elif u == "doctor":
            els += [dict(kind="text", rect=(0.3, 0.33, 0.4, 0.1), text="Doctor: Heal your pet?", size=24, align="center"),
                    dict(kind="button", id="heal", rect=(0.34, 0.52, 0.14, 0.09), text="Yes", color=(0.2, 0.75, 0.3)),
                    dict(kind="button", id="cancel", rect=(0.52, 0.52, 0.14, 0.09), text="No", color=(0.8, 0.25, 0.25))]
        elif u == "choose":
            els += [dict(kind="text", rect=(0.3, 0.32, 0.4, 0.08), text="What should your pet do?", size=22, align="center")]
            for j, n in enumerate(self.choices):
                els.append(dict(kind="button", id=f"pick:{j}", rect=(0.32 + 0.125 * j, 0.45, 0.11, 0.15), text=n.title(), color=TASK_COL[n], size=15))
        elif u == "daily":
            els += [dict(kind="text", rect=(0.3, 0.33, 0.4, 0.1), text="Daily Reward!  +10 Bucks", size=24, align="center"),
                    dict(kind="button", id="claim", rect=(0.42, 0.52, 0.16, 0.09), text="Claim", color=(0.2, 0.75, 0.3)), X(0.655, 0.31)]
        elif u == "offer":
            els += [dict(kind="text", rect=(0.3, 0.36, 0.4, 0.14), text="LIMITED OFFER!\nLegendary Egg 1,450 R$", size=22, align="center", color=(0.9, 0.5, 0.1)),
                    X(0.655, 0.31)]
        return els

    def dynamic_boxes(self) -> list:
        if not self.equipped:
            return []
        h = self.player
        fx, fz = math.sin(h.face), math.cos(h.face)
        rx, rz = math.cos(h.face), -math.sin(h.face)
        return [((h.pos[0] + rx * 1.5 + fx * 0.8, h.pos[1] + 2.2, h.pos[2] + rz * 1.5 + fz * 0.8), (1, 0.4, 1), h.face, FOODS[self.equipped][2])]

    def potential(self) -> float:
        """Sub-steps of the most urgent need (item bought → equipped → activity running → its progress),
        closeness to where it gets done, and a penalty while a pop-up blocks the screen."""
        self.goal = None
        self.expert()
        phi = -0.3 if self.ui in ("daily", "offer", "choose") else 0.0
        if self.activity is not None:
            need, left, _, _ = self.activity
            return phi + 0.9 + 0.5 * (1 - left / TASKS[need][2])
        if self.goal is not None:
            phi -= 0.01 * float(np.linalg.norm((self.goal - self.player.pos)[[0, 2]]))
        food = [n for n in self.needs if n in SHOP_OF]
        if food:
            items = [x for x in FOODS if FOODS[x][0] == food[0]]
            phi += 0.3 * any(self.inventory[x] > 0 for x in items) + 0.3 * (self.equipped in items)
        return phi

    def info(self) -> dict:
        return dict(bucks=self.bucks, tasks_done=self.pet_xp, needs=len(self.needs))

    # ── expert: what a practised Adopt Me player does, in order ──
    def expert(self) -> Action:
        u = self.ui
        if u in ("daily", "offer"):
            return self._click_ui("claim" if u == "daily" else "close")
        if u == "choose":
            return self._click_ui("pick:0")
        if u == "doctor":
            return self._click_ui("heal")
        need = self._top_need()
        if isinstance(u, tuple) and u[0] == "buy":
            return self._click_ui("buy" if self.bucks >= FOODS[u[1]][1] else "cancel")
        if isinstance(u, tuple) and u[0] == "shop":
            want = [x for x in SHOPS[u[1]][1] if need and FOODS[x][0] == need and FOODS[x][1] <= self.bucks]
            return self._click_ui(f"shopitem:{min(want, key=lambda x: FOODS[x][1])}" if want else "close")
        if u == "pet":
            if need == "attention":
                return self._click_ui("petit")
            ok = self.equipped and need == FOODS[self.equipped][0]
            return self._click_ui("use" if ok else "close")
        if u == "map":
            return self._click_ui(f"tp:{self._tp}" if getattr(self, "_tp", None) else "close")
        if self.activity is not None:
            need_, _, anchor, rad = self.activity
            return self._walk(anchor, stop=min(rad * 0.5, 4.0)) if rad < 1e8 else Action(cursor=self.cursor)
        if need is None:
            return self._close_or(lambda: self._walk(np.array([0.0, 0, 30]), stop=6))
        if need == "choose":
            return self._click_ui(f"task:{self.needs.index('choose')}")
        how = TASKS[need][0]
        if how.startswith("pet:"):
            return self._close_or(lambda: self._click_world(self.pet_pos + [0, 1.2, 0], stop=8))
        if how.startswith("food:"):
            items = [x for x in FOODS if FOODS[x][0] == need]
            have = [x for x in items if self.inventory[x] > 0]
            if self.equipped in items:
                return self._close_or(lambda: self._click_world(self.pet_pos + [0, 1.2, 0], stop=6))
            if have:
                return self._click_ui(f"item:{have[0]}") if u == "backpack" else self._click_ui("backpack")
            if self.bucks < min(FOODS[x][1] for x in items):
                return self._close_or(lambda: self._walk(np.array([0.0, 0, 30]), stop=6))
            part = self.pizza if SHOP_OF[need] == "pizza_shop" else self.lemonade
            return self._close_or(lambda: self._travel(self.world.c[part], click=True))
        if how.startswith("click:"):
            part = {"bed": self.bed, "tub": self.tub, "doctor": self.doctor}[how[6:]]
            return self._close_or(lambda: self._travel(self.world.c[part], click=True))
        c, rad = self.zones[how[5:]]
        return self._close_or(lambda: self._travel(c, stop=rad * 0.4))

    def _top_need(self):
        if "choose" in self.needs:
            return "choose"
        order = ("attention", "sick", "hungry", "thirsty", "sleepy", "dirty", "bored", "camping")
        return min(self.needs, key=order.index) if self.needs else None

    def _close_or(self, then):
        return self._click_ui("close") if self.ui == "backpack" else then()

    def _travel(self, goal, click=False, stop=5.0) -> Action:
        """Far away and outside: open the Map and teleport to the nearest stop, like players do."""
        goal = np.asarray(goal, float)
        far = float(np.linalg.norm((goal - self.player.pos)[[0, 2]])) > TP_DIST
        if far and not any(rm.inside(self.player.pos) for rm in self.rooms):
            self._tp = min(TELEPORTS, key=lambda k: np.linalg.norm((np.array(TELEPORTS[k]) - goal)[[0, 2]]))
            self.goal = goal
            return self._click_ui("map")
        return self._click_world(goal, stop=stop) if click else self._walk(goal, stop=stop)

    def _walk(self, goal, stop=1.5) -> Action:
        self.goal = np.asarray(goal, float)
        nxt = route(self.player.pos, np.asarray(goal, float), self.rooms)
        return self.steer_to(nxt, stop_dist=stop if np.allclose(nxt, goal) else 1.0)

    def _point(self, uv, look=None) -> Action:
        """Move the cursor onto the target first, click on the next tick once it's there: how hands do it."""
        a = look or Action()
        on = abs(self.cursor[0] - uv[0]) < 0.012 and abs(self.cursor[1] - uv[1]) < 0.02
        a.cursor, a.click = tuple(uv), on
        return a

    def _click_ui(self, id_) -> Action:
        for el in self.hud():
            if el.get("id") == id_:
                x, y, w, h = el["rect"]
                return self._point((x + w / 2, y + h / 2))
        return Action(cursor=self.cursor)

    def _click_world(self, p, stop=5.0) -> Action:
        self.goal = np.asarray(p, float)
        d = float(np.linalg.norm((p - self.player.pos)[[0, 2]]))
        if d > stop or not np.allclose(route(self.player.pos, p, self.rooms), p):
            return self._walk(p, stop=stop * 0.6)
        a = self.steer_to(p, keys=(), stop_dist=99)
        a.look_y = float(np.clip(math.degrees(math.atan2(self.cam.pos[1] - p[1], max(1.0, d + self.cam.zoom * 0.8)) - self.cam.pitch), -10, 10))
        uv = self.click_at_world(p)
        if uv is None or abs(a.look_x) >= 10:
            a.cursor = self.cursor
            return a
        return self._point(uv, look=Action(look_x=a.look_x, look_y=a.look_y))


if __name__ == "__main__":
    import time
    env = AdoptMe(seed=0, max_steps=3000)
    t0 = time.time()
    R = 0.0
    while not env.done:
        r, done, info = env.step(env.expert())
        R += r
    print(f"expert: {info}, return {R:.2f}, {(time.time() - t0) / env.steps * 1000:.1f} ms/tick")
    env = AdoptMe(seed=0, max_steps=3000)
    rng = np.random.default_rng(0)
    from core.actions import KEYS
    R = 0.0
    while not env.done:
        r, _, _ = env.step(Action(keys=frozenset(k for k in KEYS if rng.random() < 0.1), look_x=float(rng.normal(0, 10)),
                                  cursor=tuple(rng.random(2)), click=rng.random() < 0.2))
        R += r
    print(f"random: {env.info()}, return {R:.2f}")
