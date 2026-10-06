"""Grow a Garden: buy seeds from Sam, plant them in your plot, water, collect the fruit, sell it to Steven.

Mechanics follow the real game, with growth scaled to minutes: you start with 20 Sheckles; Sam's Seed Shop
has limited stock that restocks every 5 minutes; bought seeds land in the hotbar; with a seed equipped,
clicking your soil plants it in that cell; single-harvest crops (Carrot, Watermelon) are collected once,
multi-harvest crops (Strawberry, Blueberry, Tomato) keep regrowing fruit; ripe fruit has a Collect prompt
and can roll a mutation (Gold 20x, Rainbow 50x); Steven buys your whole inventory; the gear shop sells a
Watering Can that speeds up a plant when you click it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from core.actions import Action
from sim.core import DT, Env, Humanoid, Prompt, avatar_boxes, hotbar_hud, _inside

# kind: (price, grow s, regrow s or 0 for single-harvest, fruit value, max fruits, restock range, colour)
SEEDS = {
    "Carrot": (10, 15.0, 0.0, 18, 1, (8, 20), (1.0, 0.5, 0.1)),
    "Strawberry": (50, 25.0, 12.0, 14, 3, (3, 6), (0.95, 0.15, 0.2)),
    "Blueberry": (400, 35.0, 15.0, 40, 3, (1, 3), (0.25, 0.35, 0.95)),
    "Tomato": (800, 45.0, 18.0, 60, 3, (0, 2), (0.9, 0.2, 0.1)),
    "Watermelon": (2500, 70.0, 0.0, 2700, 1, (0, 1), (0.2, 0.6, 0.2)),
}
MUTATIONS = (("Rainbow", 50.0, 0.005), ("Gold", 20.0, 0.03))   # (name, value multiplier, chance per fruit)
RESTOCK_S = 300.0
CAN_PRICE, CAN_USES, CAN_BOOST = 60, 10, 6.0
GRID, CELL = 6, 4.0
PLOT_C = np.array([0.0, 0.0, 12.0])
SAM, STEVEN, GEAR = np.array([-22.0, 0, -24.0]), np.array([22.0, 0, -24.0]), np.array([0.0, 0, -34.0])


@dataclass
class Fruit:
    id: int
    pos: np.ndarray
    ripe_at: float          # plant-age seconds when it's ripe
    mutation: str = ""


@dataclass
class Plant:
    kind: str
    cell: tuple
    age: float = 0.0        # seconds of growth, watering adds to it
    fruits: list = field(default_factory=list)
    next_fruit: float = 0.0


class GrowAGarden(Env):
    name = "garden"
    train_steps = 3000

    def build(self) -> None:
        w, r = self.world, self.rng
        self.max_steps = int(self.kw.get("max_steps", 6000))
        w.add((0, -0.5, 0), (300, 1, 240), (0.35, 0.65, 0.3))
        w.add((0, 0.02, -28), (120, 0.05, 14), (0.55, 0.5, 0.45))                    # the street
        for x in (-150, 150):
            w.add((x, 20, 0), (1, 40, 240), (0, 0, 0), visible=False)
        for z in (-120, 120):
            w.add((0, 20, z), (300, 40, 1), (0, 0, 0), visible=False)
        self.soil = w.add(PLOT_C + [0, 0.15, 0], (GRID * CELL, 0.3, GRID * CELL), (0.45, 0.3, 0.18))
        for dx in (-GRID * CELL / 2 - 0.5, GRID * CELL / 2 + 0.5):                    # little fence
            w.add(PLOT_C + [dx, 0.6, 0], (0.6, 1.2, GRID * CELL + 1.6), (0.85, 0.75, 0.55))
        w.add(PLOT_C + [0, 0.6, GRID * CELL / 2 + 0.5], (GRID * CELL + 1.6, 1.2, 0.6), (0.85, 0.75, 0.55))
        self.neighbours = []
        for x in (-40.0, 40.0, -80.0, 80.0):   # cosmetic NPC gardens
            w.add((x, 0.15, PLOT_C[2]), (GRID * CELL, 0.3, GRID * CELL), (0.45, 0.3, 0.18))
            for _ in range(int(r.integers(4, 10))):
                k = str(r.choice(list(SEEDS)))
                px, pz = x + r.uniform(-10, 10), PLOT_C[2] + r.uniform(-10, 10)
                self.neighbours.append((k, px, pz, float(r.uniform(0.6, 1.0))))
        for c, col, label in ((SAM, (0.3, 0.75, 0.3), "Sam's Seed Shop"), (STEVEN, (0.9, 0.7, 0.2), "Steven - Sell Fruit"),
                              (GEAR, (0.4, 0.5, 0.9), "Gear Shop")):
            w.add(c + [0, 1.5, -3], (10, 3, 2), (0.6, 0.4, 0.25))                     # counter
            w.add(c + [0, 6.5, -3], (12, 0.6, 5), col)                                 # awning
            for dx in (-5.5, 5.5):
                w.add(c + [dx, 3.2, -4.5], (0.5, 6.4, 0.5), (0.85, 0.85, 0.85))
        self.labels = [(SAM + [0, 8.5, -3], "Sam's Seed Shop"), (STEVEN + [0, 8.5, -3], "Steven - Sell"), (GEAR + [0, 8.5, -3], "Gear Shop")]
        self.keepers = [Humanoid(pos=c + np.array([0, 0, -5.5]), shirt=col, face=0.0) for c, col in
                        ((SAM, (0.2, 0.6, 0.2)), (STEVEN, (0.8, 0.6, 0.1)), (GEAR, (0.3, 0.4, 0.8)))]

        self.sheckles = 20.0
        self.seeds = {k: 0 for k in SEEDS}
        self.can_uses = 0
        self.inventory: list[tuple[str, float, str]] = []      # (kind, value, mutation)
        self.plants: dict[tuple, Plant] = {}
        self.stock: dict[str, int] = {}
        self._restock()
        self.next_restock = RESTOCK_S
        self.fruit_prompts: dict[int, Prompt] = {}
        self._fid = 0
        self.statics = [Prompt(pos=SAM + [0, 2.5, -2], action="Talk", obj="Sam", on_trigger=("_open", "seeds")),
                        Prompt(pos=STEVEN + [0, 2.5, -2], action="Sell Inventory", obj="Steven", hold=0.5, on_trigger=("_sell",)),
                        Prompt(pos=GEAR + [0, 2.5, -2], action="Talk", obj="Gear Shop", on_trigger=("_open", "gear"))]
        self.prompts = list(self.statics)
        self.ui = None
        self.goal = None
        self.nw_prev = self._networth()
        self.cam.yaw = 0.0
        self.cam.pitch = math.radians(28)

    def respawn_point(self):
        return np.array([0.0, 0.0, -14.0])

    # ── economy ──
    def _restock(self) -> None:
        self.stock = {k: int(self.rng.integers(v[5][0], v[5][1] + 1)) for k, v in SEEDS.items()}

    def _networth(self) -> float:
        nw = self.sheckles + sum(v for _, v, _ in self.inventory) + sum(SEEDS[k][0] * n for k, n in self.seeds.items())
        for p in self.plants.values():
            nw += SEEDS[p.kind][0] + sum(self._fruit_value(p.kind, f) for f in p.fruits if p.age >= f.ripe_at)
        return nw

    @staticmethod
    def _fruit_value(kind: str, f: Fruit) -> float:
        mult = next((m for n, m, _ in MUTATIONS if n == f.mutation), 1.0)
        return SEEDS[kind][3] * mult

    def _open(self, what: str) -> None:
        self.ui = what

    def _sell(self) -> None:
        if not self.inventory:
            self.toast("You have no fruit to sell")
            return
        total = sum(v for _, v, _ in self.inventory)
        self.sheckles += total
        self.toast(f"Sold {len(self.inventory)} fruit for ${total:,.0f}")
        self.inventory = []

    def _collect(self, cell: tuple, fid: int) -> None:
        p = self.plants.get(cell)
        f = next((f for f in p.fruits if f.id == fid), None) if p else None
        if f is None or p.age < f.ripe_at:
            return
        self.inventory.append((p.kind, self._fruit_value(p.kind, f), f.mutation))
        self.toast(f"+1 {(f.mutation + ' ') if f.mutation else ''}{p.kind}", 1.5)
        p.fruits.remove(f)
        if SEEDS[p.kind][2] == 0:
            del self.plants[cell]          # single-harvest: the plant is gone
        else:
            p.next_fruit = max(p.next_fruit, p.age + SEEDS[p.kind][2])

    def _new_fruit(self, p: Plant, ripe_at: float) -> None:
        cx, cz = self._cell_center(p.cell)
        n = len(p.fruits)
        ang = self._fid * 2.1
        off = (0.0, 0.0) if SEEDS[p.kind][4] == 1 else (math.cos(ang) * 0.9, math.sin(ang) * 0.9)
        mut = ""
        for name, _, chance in MUTATIONS:
            if self.rng.random() < chance:
                mut = name
                break
        h = 1.2 if p.kind != "Watermelon" else 1.0
        p.fruits.append(Fruit(id=self._fid, pos=np.array([cx + off[0], 0.3 + h + 0.25 * n, cz + off[1]]), ripe_at=ripe_at, mutation=mut))
        self._fid += 1

    def _cell_center(self, cell) -> tuple:
        x0, z0 = PLOT_C[0] - GRID * CELL / 2, PLOT_C[2] - GRID * CELL / 2
        return x0 + (cell[0] + 0.5) * CELL, z0 + (cell[1] + 0.5) * CELL

    def _cell_at(self, p) -> tuple | None:
        x0, z0 = PLOT_C[0] - GRID * CELL / 2, PLOT_C[2] - GRID * CELL / 2
        i, j = int((p[0] - x0) // CELL), int((p[2] - z0) // CELL)
        return (i, j) if 0 <= i < GRID and 0 <= j < GRID else None

    # ── items and clicks ──
    def items(self) -> list[str]:
        out = [k for k in SEEDS if self.seeds[k] > 0]
        return out + (["Watering Can"] if self.can_uses > 0 else [])

    def _item_label(self, it: str) -> str:
        return f"{it[:5]} x{self.can_uses}" if it == "Watering Can" else f"{it[:6]} x{self.seeds[it]}"

    def on_key(self, k: str) -> None:
        self._equip(int(k) - 1)

    def _equip(self, i: int) -> None:
        its = self.items()
        if i < len(its):
            self.equipped = None if self.equipped == its[i] else its[i]

    def on_ui(self, id_: str) -> None:
        if id_.startswith("slot:"):
            self._equip(int(id_[5:]))
        elif id_ == "close":
            self.ui = None
        elif id_.startswith("buy:"):
            k = id_[4:]
            price = SEEDS[k][0]
            if self.stock[k] <= 0:
                self.toast("Out of stock!")
            elif self.sheckles < price:
                self.toast("Not enough Sheckles!")
            else:
                self.sheckles -= price
                self.stock[k] -= 1
                self.seeds[k] += 1
                self.toast(f"Bought {k} Seed", 1.5)
        elif id_ == "buy_can":
            if self.sheckles >= CAN_PRICE:
                self.sheckles -= CAN_PRICE
                self.can_uses += CAN_USES
                self.toast("Bought Watering Can")
            else:
                self.toast("Not enough Sheckles!")

    def on_world_click(self, part: int) -> None:
        if part != self.soil or not self.equipped:
            return
        d = self.cam.ray(self.cursor[0], self.cursor[1], self.aspect)
        t, _ = self.world.raycast(self.cam.pos, d, max_t=500)
        cell = self._cell_at(self.cam.pos + d * t)
        if cell is None:
            return
        if self.equipped == "Watering Can":
            p = self.plants.get(cell)
            if p is not None and self.can_uses > 0:
                p.age += CAN_BOOST
                self.can_uses -= 1
                self.toast("Watered!", 1.0)
        elif cell not in self.plants and self.seeds.get(self.equipped, 0) > 0:
            self.seeds[self.equipped] -= 1
            self.plants[cell] = Plant(kind=self.equipped, cell=cell)
        if self.equipped not in self.items():
            self.equipped = None

    # ── simulation ──
    def on_substep(self) -> None:
        if self.t >= self.next_restock:
            self._restock()
            self.next_restock += RESTOCK_S
            self.toast("Seed shop restocked!", 3)
        for p in self.plants.values():
            p.age += DT
            price, grow, regrow, _, maxf, _, _ = SEEDS[p.kind]
            if regrow == 0:
                if not p.fruits:
                    self._new_fruit(p, grow)
            elif p.age >= grow and len(p.fruits) < maxf and p.age >= p.next_fruit:
                self._new_fruit(p, p.age + regrow * 0.5)
                p.next_fruit = p.age + regrow * 0.5
        ripe = {f.id: (cell, f) for cell, p in self.plants.items() for f in p.fruits if p.age >= f.ripe_at}
        for fid in list(self.fruit_prompts):
            if fid not in ripe:
                del self.fruit_prompts[fid]
        for fid, (cell, f) in ripe.items():
            if fid not in self.fruit_prompts:
                kind = self.plants[cell].kind
                self.fruit_prompts[fid] = Prompt(pos=f.pos, action="Collect", obj=f"{f.mutation} {kind}".strip(), on_trigger=("_collect", cell, fid))
        self.prompts = self.statics + list(self.fruit_prompts.values())
        nw = self._networth()
        self.reward += math.log1p(nw) - math.log1p(self.nw_prev)
        self.nw_prev = nw
        if self.steps >= self.max_steps:
            self.done = True

    # ── rendering ──
    def dynamic_boxes(self) -> list:
        out = []
        for h in self.keepers:
            out += avatar_boxes(h)
        for cell, p in self.plants.items():
            cx, cz = self._cell_center(cell)
            out += self._plant_boxes(p.kind, cx, cz, min(1.0, p.age / SEEDS[p.kind][1]))
            for f in p.fruits:
                g = min(1.0, max(0.25, p.age / max(f.ripe_at, 1e-3)))
                out.append(self._fruit_box(p.kind, f, g))
        for k, px, pz, g in self.neighbours:
            out += self._plant_boxes(k, px, pz, g)
        return out

    @staticmethod
    def _plant_boxes(kind, cx, cz, g) -> list:
        h = 0.3 + 1.2 * g
        if SEEDS[kind][4] == 1:   # carrots and melons are just leaves above the soil
            return [((cx, 0.3 + h * 0.4, cz), (0.6, h * 0.8, 0.6), 0.3, (0.25, 0.7, 0.2))]
        return [((cx, 0.3 + h / 2, cz), (0.4, h, 0.4), 0.0, (0.2, 0.55, 0.15)),
                ((cx, 0.3 + h, cz), (1.8 * g + 0.3, 0.8 * g + 0.2, 1.8 * g + 0.3), 0.6, (0.25, 0.65, 0.25))]

    @staticmethod
    def _fruit_box(kind, f: Fruit, g):
        s = (0.5 if SEEDS[kind][4] > 1 else (0.7 if kind == "Carrot" else 1.8)) * g
        col = {"Gold": (1.0, 0.8, 0.1), "Rainbow": (1.0, 0.3, 0.9)}.get(f.mutation, SEEDS[kind][6])
        return (tuple(f.pos), (s, s, s), 0.0, col, bool(f.mutation))

    def hud(self) -> list:
        els = [dict(kind="leaderboard", cols=["Sheckles"], rows=[["You", f"{self.sheckles:,.0f}"]]),
               dict(kind="panel", rect=(0.02, 0.12, 0.2, 0.07), color=(0.1, 0.25, 0.1), alpha=0.75),
               dict(kind="text", rect=(0.03, 0.12, 0.19, 0.07), text=f"{self.sheckles:,.0f} Sheckles", size=24, color=(0.5, 1.0, 0.45)),
               dict(kind="text", rect=(0.03, 0.2, 0.3, 0.05), text=f"Fruit: {len(self.inventory)} (${sum(v for _, v, _ in self.inventory):,.0f})", size=18)]
        its = self.items()
        if its:
            els += hotbar_hud([self._item_label(i) for i in its], its.index(self.equipped) if self.equipped in its else None)
        for c, txt in self.labels:
            b = self.billboard(c, txt, size=1.2, max_dist=120)
            if b:
                els.append(b)
        near = sorted(((c, f) for c, p in self.plants.items() for f in p.fruits if p.age >= f.ripe_at and f.mutation),
                      key=lambda cf: np.linalg.norm(cf[1].pos - self.player.pos))[:4]
        for c, f in near:
            b = self.billboard(f.pos + [0, 1.2, 0], f"{f.mutation} {self.plants[c].kind}", color=(1, 0.85, 0.2), size=0.8, max_dist=30)
            if b:
                els.append(b)
        if self.ui == "seeds":
            els += self._shop_ui()
        elif self.ui == "gear":
            els += [dict(kind="panel", rect=(0.33, 0.3, 0.34, 0.3), color=(0.15, 0.2, 0.35), alpha=0.92),
                    dict(kind="text", rect=(0.33, 0.31, 0.34, 0.07), text="Gear Shop", size=24, align="center"),
                    dict(kind="button", id="close", rect=(0.625, 0.31, 0.035, 0.06), text="X", color=(0.85, 0.2, 0.2)),
                    dict(kind="text", rect=(0.35, 0.42, 0.18, 0.07), text=f"Watering Can ({CAN_USES} uses)", size=16),
                    dict(kind="button", id="buy_can", rect=(0.54, 0.42, 0.1, 0.07), text=f"${CAN_PRICE}", color=(0.2, 0.7, 0.3))]
        return els

    def _shop_ui(self) -> list:
        left = self.next_restock - self.t
        X = 0.04   # left of centre, so Sam's Talk prompt (over his head, mid-screen) never covers the buttons
        els = [dict(kind="panel", rect=(X, 0.25, 0.4, 0.62), color=(0.32, 0.22, 0.12), alpha=0.95),
               dict(kind="text", rect=(X, 0.26, 0.4, 0.06), text="Seed Shop", size=26, align="center"),
               dict(kind="text", rect=(X, 0.315, 0.4, 0.04), text=f"Restock in {int(left // 60)}:{int(left % 60):02d}", size=16, align="center",
                    color=(0.9, 0.9, 0.6)),
               dict(kind="button", id="close", rect=(X + 0.355, 0.26, 0.035, 0.06), text="X", color=(0.85, 0.2, 0.2))]
        for j, (k, v) in enumerate(SEEDS.items()):
            y = 0.37 + 0.098 * j
            els += [dict(kind="panel", rect=(X + 0.02, y, 0.36, 0.085), color=v[6], alpha=0.35),
                    dict(kind="text", rect=(X + 0.035, y, 0.2, 0.085), text=f"{k} Seed", size=18),
                    dict(kind="text", rect=(X + 0.2, y, 0.07, 0.085), text=f"x{self.stock[k]}", size=16,
                         color=(1, 1, 1) if self.stock[k] else (1, 0.4, 0.4)),
                    dict(kind="button", id=f"buy:{k}", rect=(X + 0.28, y + 0.01, 0.09, 0.065), text=f"${v[0]:,}", size=15,
                         color=(0.2, 0.7, 0.3) if self.stock[k] else (0.4, 0.4, 0.4))]
        return els

    def info(self) -> dict:
        return dict(sheckles=round(self.sheckles), networth=round(self._networth()), plants=len(self.plants),
                    fruit=len(self.inventory))

    def score(self) -> float:
        return math.log10(max(1.0, self._networth()))   # you start at log10(20) = 1.3

    def potential(self) -> float:
        """Closer to the expert's next goal, and fewer ripe fruits left uncollected."""
        self.goal = None
        self.expert()
        phi = -0.01 * float(np.linalg.norm((self.goal - self.player.pos)[[0, 2]])) if self.goal is not None else 0.0
        return phi - 0.02 * min(20, len(self.fruit_prompts))

    # ── expert: buy → plant → water → collect → sell ──
    def expert(self) -> Action:
        p = self.player
        if self.ui == "seeds":
            free = GRID * GRID - len(self.plants) - sum(self.seeds.values())
            k = self._best_seed(free)
            return self._click_ui(f"buy:{k}" if k else "close")
        if self.ui == "gear":
            return self._click_ui("buy_can" if self.can_uses == 0 and self.sheckles >= CAN_PRICE + 20 else "close")
        ripe = [f for c, pl in self.plants.items() for f in pl.fruits if pl.age >= f.ripe_at]
        if ripe:
            f = min(ripe, key=lambda f: np.linalg.norm(f.pos - p.pos))
            a = self._go(f.pos, stop=1.8)
            if self.active_prompt() is self.fruit_prompts.get(f.id):
                a.keys = a.keys | {"e"}
            return a
        has_seed = [k for k in SEEDS if self.seeds[k] > 0]
        free_cells = [(i, j) for j in range(GRID) for i in range(GRID) if (i, j) not in self.plants]
        if has_seed and free_cells:
            k = max(has_seed, key=lambda k: SEEDS[k][0])
            return self._use_on_cell(k, min(free_cells, key=lambda c: np.linalg.norm(np.array(self._cell_center(c)) - p.pos[[0, 2]])))
        inv_value = sum(v for _, v, _ in self.inventory)
        if self.inventory and (inv_value >= 40 or not self.plants or self._best_seed(len(free_cells), self.sheckles + inv_value)):
            return self._prompt_at(STEVEN, self.statics[1])
        if free_cells and self._best_seed(len(free_cells)):
            return self._prompt_at(SAM, self.statics[0])
        growing = [pl for pl in self.plants.values() if not any(pl.age >= f.ripe_at for f in pl.fruits)]
        if growing and self.can_uses > 0:
            pl = max(growing, key=lambda pl: SEEDS[pl.kind][3])
            return self._use_on_cell("Watering Can", pl.cell)
        if self.can_uses == 0 and self.sheckles >= CAN_PRICE + 40 and len(self.plants) >= 4:
            return self._prompt_at(GEAR, self.statics[2])
        if self.equipped:
            return self._click_ui(f"slot:{self.items().index(self.equipped)}") if self.equipped in self.items() else Action(cursor=self.cursor)
        return self._go(PLOT_C + [0, 0, -GRID * CELL / 2 - 3], stop=3)

    def _best_seed(self, free: int, cash: float | None = None):
        cash = self.sheckles if cash is None else cash
        if free <= 0:
            return None
        ok = [k for k, v in SEEDS.items() if self.stock[k] > 0 and v[0] <= cash]
        return max(ok, key=lambda k: SEEDS[k][0]) if ok else None

    def _go(self, goal, stop=1.5) -> Action:
        self.goal = np.asarray(goal, float)
        a = self.steer_to(self.goal, stop_dist=stop)
        a.look_y = float(np.clip(math.degrees(math.radians(28) - self.cam.pitch), -10, 10))
        return a

    def _prompt_at(self, where, pr: Prompt) -> Action:
        a = self._go(where + [0, 0, -1.5], stop=2.5)
        if self.active_prompt() is pr:
            a.keys = a.keys | {"e"}
        return a

    def _point(self, uv, base: Action | None = None) -> Action:
        """Cursor onto the target first, click on the next tick once it's there."""
        a = base or Action()
        on = abs(self.cursor[0] - uv[0]) < 0.012 and abs(self.cursor[1] - uv[1]) < 0.02
        a.cursor, a.click = tuple(uv), on
        return a

    def _click_ui(self, id_) -> Action:
        for el in self.hud():
            if el.get("id") == id_:
                x, y, w, h = el["rect"]
                return self._point((x + w / 2, y + h / 2))
        return Action(cursor=self.cursor)

    def _use_on_cell(self, item: str, cell) -> Action:
        """Equip the item from the hotbar, walk close, look at the cell, click it."""
        if self.equipped != item:
            return self._click_ui(f"slot:{self.items().index(item)}")
        cx, cz = self._cell_center(cell)
        tgt = np.array([cx, 0.3, cz])
        d = float(np.linalg.norm((tgt - self.player.pos)[[0, 2]]))
        if d > 7:
            return self._go(tgt, stop=4)
        self.goal = tgt
        a = self.steer_to(tgt, keys=(), stop_dist=99)
        a.look_y = float(np.clip(math.degrees(math.atan2(self.cam.pos[1] - tgt[1], max(1.0, d + self.cam.zoom * 0.8)) - self.cam.pitch), -10, 10))
        uv = self.click_at_world(tgt)
        if uv is None or abs(a.look_x) >= 12 or any(_inside(uv, el["rect"]) for el in self.hud() if el.get("id")):
            a.cursor = self.cursor
            return a
        return self._point(uv, Action(look_x=a.look_x, look_y=a.look_y))


if __name__ == "__main__":
    import pickle
    import time
    from core.actions import KEYS
    TICKS = 3000   # 5 minutes
    for seed in (0, 1, 2):
        env = GrowAGarden(seed=seed, max_steps=TICKS)
        t0 = time.time()
        while not env.done:
            env.step(env.expert())
            if env.steps == 1000:   # pickle round trip mid-episode must step identically
                twin = pickle.loads(pickle.dumps(env))
                a = env.expert()
                r1, r2 = env.step(a)[0], twin.step(a)[0]
                assert r1 == r2 and env.sheckles == twin.sheckles
        ms = (time.time() - t0) / env.steps * 1000
        rnd = GrowAGarden(seed=seed, max_steps=TICKS)
        rng = np.random.default_rng(seed)
        while not rnd.done:
            rnd.step(Action(keys=frozenset(k for k in KEYS if k != "esc" and rng.random() < 0.1), look_x=float(rng.normal(0, 10)),
                            cursor=tuple(rng.random(2)), click=rng.random() < 0.2))
        print(f"seed {seed}: expert score {env.score():.2f} {env.info()}  |  random score {rnd.score():.2f} {rnd.info()}  |  {ms:.1f} ms/tick")
