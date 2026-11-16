"""Fisch: a fishing island. The loop is the real game's:

  equip the rod (1) at the water's edge → hold left mouse to charge the cast (the meter rises and falls;
  release near the top for a Perfect cast) → wait for a bite (the bobber gets a "!") → SHAKE: click the
  Shake button, which jumps somewhere new after each click → REEL: hold left mouse to push your white zone
  right, release and it drifts left; keep the fish inside it until the progress bar fills (empty = it gets away)
  → sell at the Merchant ("Sell All") → buy better rods at the Rod Shop board (bigger zone, faster lure, luck).
Rarer fish dart harder. Currency is C$.
"""

from __future__ import annotations

import math

import numpy as np

from core.actions import Action
from sim.core import DT, Env, Humanoid, Prompt, avatar_boxes, hotbar_hud, right_xz, forward_xz

# (tier, spawn weight, agility, colour, [(name, base C$, base kg)])
TIERS = [
    ("Trash", 15.0, 0.6, (0.55, 0.5, 0.4), [("Old Boot", 3, 1.0), ("Seaweed", 2, 0.3), ("Tin Can", 4, 0.2)]),
    ("Common", 50.0, 0.9, (0.85, 0.85, 0.85), [("Carp", 18, 2.0), ("Bass", 25, 1.5), ("Perch", 20, 0.8)]),
    ("Uncommon", 22.0, 1.2, (0.4, 0.9, 0.4), [("Trout", 60, 1.8), ("Salmon", 85, 4.0)]),
    ("Rare", 9.0, 1.6, (0.3, 0.6, 1.0), [("Pike", 220, 5.0), ("Tuna", 320, 30.0)]),
    ("Legendary", 3.2, 2.1, (1.0, 0.75, 0.1), [("Sturgeon", 900, 60.0), ("Swordfish", 1300, 90.0)]),
    ("Mythical", 0.8, 2.7, (1.0, 0.25, 0.6), [("Great White Shark", 4500, 700.0)]),
]
# (name, price, catch-zone width, lure speed, luck)
RODS = [("Training Rod", 0, 0.22, 1.0, 0.0), ("Plastic Rod", 900, 0.26, 1.15, 0.1), ("Carbon Rod", 3000, 0.30, 1.3, 0.25),
        ("Steady Rod", 8000, 0.36, 1.2, 0.3), ("Fortune Rod", 20000, 0.30, 1.45, 0.7)]
BAG = 20
CHARGE_PERIOD = 2.0          # the cast meter rises for 1 s then falls for 1 s while held
ZONE_ACCEL = 2.5             # reel: catch zone acceleration (screen widths / s²), right while held, left otherwise
ZONE_VMAX = 1.2
FILL, DRAIN = 0.28, 0.20     # reel progress per second with the fish inside / outside the zone
SHAKE_TIME = 7.0
REEL_BAR = (0.25, 0.68, 0.5, 0.05)
SPOT = np.array([0.0, 0.5, 28.0])          # end of the pier, facing open water (+z)
MERCHANT = np.array([9.0, 0.0, -6.0])
BOARD = np.array([-9.0, 0.0, -6.0])


def _on_land(x, z) -> bool:
    return (abs(x) < 30 and -40 < z < 0) or (abs(x) < 3 and 0 <= z < 30)   # island or pier


class Fisch(Env):
    name = "fisch"
    train_steps = 3000

    def build(self) -> None:
        w = self.world
        self.max_steps = int(self.kw.get("max_steps", self.train_steps))
        w.add((0, -0.4, 0), (500, 0.2, 500), (0.15, 0.45, 0.75), collide=False)       # the water surface
        w.add((0, -8.5, 0), (500, 1, 500), (0.75, 0.68, 0.45))                         # sea bed
        w.add((0, -0.5, -20), (60, 1, 40), (0.35, 0.65, 0.3))                          # the island
        w.add((0, -0.2, -20), (64, 0.4, 44), (0.9, 0.82, 0.6))                         # beach rim
        w.add((0, 0.25, 15), (6, 0.5, 30), (0.55, 0.38, 0.22))                         # the pier
        for z in (4, 14, 24):
            for x in (-3.2, 3.2):
                w.add((x, -1.5, z), (0.6, 4, 0.6), (0.4, 0.28, 0.16))
        w.add((MERCHANT[0], 1.0, MERCHANT[2] - 2.5), (6, 2, 2), (0.6, 0.42, 0.25))    # merchant's counter
        w.add((MERCHANT[0], 4.5, MERCHANT[2] - 3.5), (7, 0.4, 4), (0.85, 0.25, 0.2))  # awning
        w.add((BOARD[0], 3.0, BOARD[2]), (6, 4, 0.5), (0.5, 0.33, 0.18))              # rod shop board
        w.add((BOARD[0] - 2.6, 1.0, BOARD[2]), (0.4, 2, 0.4), (0.4, 0.28, 0.16))
        w.add((BOARD[0] + 2.6, 1.0, BOARD[2]), (0.4, 2, 0.4), (0.4, 0.28, 0.16))
        for x, z in ((-20, -30), (18, -32), (-24, -12), (22, -14)):                   # palm-ish trees
            w.add((x, 3, z), (1.2, 6, 1.2), (0.45, 0.3, 0.18))
            w.add((x, 6.5, z), (6, 1.2, 6), (0.25, 0.6, 0.25))
        for x in (-250, 250):
            w.add((x, 20, 0), (1, 40, 500), (0, 0, 0), visible=False)
        for z in (-250, 250):
            w.add((0, 20, z), (500, 40, 1), (0, 0, 0), visible=False)
        self.merchant = Humanoid(pos=MERCHANT.copy(), face=math.pi, shirt=(0.8, 0.6, 0.2), pants=(0.3, 0.2, 0.15))
        self.prompts = [Prompt(pos=MERCHANT + [0, 2.5, 0], action="Talk", obj="Merchant", hold=0.3, on_trigger=("_open", "merchant")),
                        Prompt(pos=BOARD + [0, 2.5, 0.8], action="Browse", obj="Rod Shop", hold=0.3, on_trigger=("_open", "shop"))]
        self.money = 0.0
        self.bag: list[tuple] = []                 # (name, tier index, kg, value)
        self.rods_owned = [0]
        self.rod = 0
        self.equipped = None
        self.ui = None                             # None | "merchant" | "shop"
        self.phase = "idle"                        # idle | charge | wait | shake | reel
        self.power = 0.0
        self.perfect = False
        self.charge_t = 0.0
        self.bob = None
        self.wait_until = 0.0
        self.fish = None                           # (tier, name, base C$, base kg, kg) on the line
        self.shakes_left, self.shake_rect, self.shake_until = 0, (0.45, 0.4, 0.1, 0.08), 0.0
        self.zone, self.zone_v, self.fpos, self.ftarget, self.fnext, self.prog = 0.4, 0.0, 0.5, 0.5, 0.0, 0.25
        self._lmb_prev = False
        self.caught = 0
        self.nw_prev = self._networth()
        self.cam.yaw = 0.0
        self.cam.pitch = math.radians(22)

    def respawn_point(self):
        return np.array([0.0, 0.0, -15.0])

    # ── economy ──
    def _networth(self) -> float:
        return self.money + sum(v for *_, v in self.bag) + sum(RODS[i][1] for i in self.rods_owned)

    def score(self) -> float:
        return math.log10(max(1.0, self._networth()))

    def info(self) -> dict:
        return dict(money=round(self.money), bag=len(self.bag), caught=self.caught, rod=RODS[self.rod][0],
                    networth=round(self._networth()))

    # ── input ──
    def on_key(self, k: str) -> None:
        if k == "1":
            self._toggle_rod()

    def _toggle_rod(self):
        self.equipped = None if self.equipped else "rod"
        if not self.equipped:
            self._reset_line()

    def _reset_line(self):
        self.phase, self.bob, self.fish = "idle", None, None

    def _open(self, which: str):
        self.ui = which

    def on_ui(self, id_: str) -> None:
        if id_ == "slot:0":
            self._toggle_rod()
        elif id_ == "close":
            self.ui = None
        elif id_ == "sell_all" and self.bag:
            total = sum(v for *_, v in self.bag)
            self.money += total
            self.toast(f"Sold {len(self.bag)} fish for C${total:.0f}")
            self.bag = []
        elif id_.startswith("buy:"):
            i = int(id_[4:])
            if i in self.rods_owned:
                self.rod = i
                self.toast(f"Equipped {RODS[i][0]}")
            elif self.money >= RODS[i][1]:
                self.money -= RODS[i][1]
                self.rods_owned.append(i)
                self.rod = i
                self.toast(f"Bought {RODS[i][0]}!")
            else:
                self.toast("Not enough C$")
        elif id_ == "shake" and self.phase == "shake":
            self.shakes_left -= 1
            if self.shakes_left <= 0:
                self._start_reel()
            else:
                self._move_shake()

    def _over_ui(self) -> bool:
        x, y = self.cursor
        return any(el.get("id") and el["rect"][0] <= x <= el["rect"][0] + el["rect"][2] and el["rect"][1] <= y <= el["rect"][1] + el["rect"][3]
                   for el in self.ui_layers())

    # ── the fishing state machine ──
    def on_substep(self) -> None:
        p = self.player
        held = self.mouse_held and not self.menu
        press = held and not self._lmb_prev
        self._lmb_prev = held
        if p.alive and p.pos[1] < -1.5:   # fell in: you'd swim back to the pier
            p.pos, p.vel = np.array([0.0, 0.5, 2.0]), np.zeros(3)
            self._reset_line()
            self.toast("You fell in the water!")
        if self.ui and not (np.linalg.norm((p.pos - (MERCHANT if self.ui == "merchant" else BOARD))[[0, 2]]) < 14):
            self.ui = None   # walked away from the shop

        if self.equipped == "rod" and p.alive:
            if self.phase == "idle" and press and not self._over_ui():
                self.phase, self.charge_t = "charge", self.t
            elif self.phase == "charge":
                u = ((self.t - self.charge_t) / CHARGE_PERIOD) % 1.0
                self.power = 1 - abs(2 * u - 1)          # 0 → 1 → 0 while held
                if not held:
                    self._cast()
            elif self.phase == "wait":
                self.bob[1] = -0.25 + 0.05 * math.sin(self.t * 3)
                if self.t >= self.wait_until:
                    self._bite()
            elif self.phase == "shake":
                self.bob[1] = -0.25 + 0.15 * math.sin(self.t * 25)
                if self.t >= self.shake_until:
                    self.toast("The fish got away...")
                    self._reset_line()
            elif self.phase == "reel":
                self._reel_step(held)

        nw = self._networth()
        self.reward += math.log1p(nw) - math.log1p(self.nw_prev)
        self.nw_prev = nw
        if self.steps >= self.max_steps:
            self.done = True

    def _cast(self):
        pw = self.power
        f = forward_xz(self.cam.yaw)
        land = self.player.pos[[0, 2]] + f * (6 + 24 * pw)
        if pw < 0.05 or _on_land(land[0], land[1]):
            self.toast("Can't cast there")
            self._reset_line()
            return
        self.perfect = pw >= 0.95
        if self.perfect:
            self.toast("Perfect!", 1.5)
        rod = RODS[self.rod]
        self.bob = np.array([land[0], -0.25, land[1]])
        self.wait_until = self.t + float(self.rng.uniform(2.5, 6.0)) / rod[3] * (0.6 if self.perfect else 1.0 - 0.3 * pw)
        self.phase = "wait"

    def _bite(self):
        luck = RODS[self.rod][4] + (0.15 if self.perfect else 0.0)
        w = np.array([t[1] * (1 + luck) ** (0.6 * i) for i, t in enumerate(TIERS)])
        tier = int(self.rng.choice(len(TIERS), p=w / w.sum()))
        name, base, kg = TIERS[tier][4][int(self.rng.integers(len(TIERS[tier][4])))]
        self.fish = (tier, name, base, kg, kg * float(self.rng.lognormal(0, 0.35)))   # base kg, this one's kg
        self.phase = "shake"
        self.shakes_left = int(self.rng.integers(3, 7)) - (1 if self.perfect else 0)
        self.shake_until = self.t + SHAKE_TIME
        self._move_shake()

    def _move_shake(self):
        self.shake_rect = (float(self.rng.uniform(0.15, 0.75)), float(self.rng.uniform(0.2, 0.62)), 0.1, 0.08)

    def _start_reel(self):
        self.phase = "reel"
        w = RODS[self.rod][2]
        self.zone, self.zone_v, self.prog = 0.5 - w / 2, 0.0, 0.25
        self.fpos = self.ftarget = 0.5
        self.fnext = self.t

    def _reel_step(self, held: bool):
        w = RODS[self.rod][2]
        agility = TIERS[self.fish[0]][2]
        self.zone_v = float(np.clip(self.zone_v + (ZONE_ACCEL if held else -ZONE_ACCEL) * DT, -ZONE_VMAX, ZONE_VMAX))
        self.zone += self.zone_v * DT
        if self.zone <= 0 or self.zone >= 1 - w:
            self.zone = float(np.clip(self.zone, 0, 1 - w))
            self.zone_v = 0.0
        if self.t >= self.fnext:   # the fish picks a new spot to dart to; rarer fish more often
            self.ftarget = float(self.rng.uniform(0, 1))
            self.fnext = self.t + float(self.rng.uniform(0.4, 1.4)) / agility
        step = 0.35 * agility * DT
        self.fpos += float(np.clip(self.ftarget - self.fpos, -step, step))
        inside = self.zone <= self.fpos <= self.zone + w
        self.prog += (FILL if inside else -DRAIN) * DT
        if self.prog >= 1:
            tier, name, base, kg0, kg = self.fish
            value = round(base * (0.6 + 0.4 * kg / kg0))   # heavier than usual is worth more
            if len(self.bag) < BAG:
                self.bag.append((name, tier, kg, value))
                self.caught += 1
                self.toast(f"Caught a {kg:.1f}kg {name} ({TIERS[tier][0]})!", 3)
            else:
                self.toast("Your backpack is full!")
            self._reset_line()
        elif self.prog <= 0:
            self.toast(f"The {self.fish[1]} got away...")
            self._reset_line()

    # ── drawing ──
    def dynamic_boxes(self) -> list:
        out = avatar_boxes(self.merchant)
        p = self.player
        if self.equipped == "rod" and p.alive:
            f, r = forward_xz(self.cam.yaw), right_xz(self.cam.yaw)
            tip = p.pos + [r[0] * 1.9 + f[0] * 2.2, 3.6, r[1] * 1.9 + f[1] * 2.2]
            out.append((tuple(tip), (0.3, 0.3, 4.5), self.cam.yaw, (0.95, 0.85, 0.3) if self.rod else (0.55, 0.35, 0.18)))
        if self.bob is not None:
            out.append((tuple(self.bob), (0.9, 0.7, 0.9), 0.0, (0.95, 0.15, 0.15), True))
            out.append((tuple(self.bob + [0, 0.55, 0]), (0.6, 0.4, 0.6), 0.0, (1, 1, 1)))
        return out

    def hud(self) -> list:
        val = sum(v for *_, v in self.bag)
        els = [dict(kind="leaderboard", cols=["C$"], rows=[["You", f"{self.money:.0f}"]]),
               dict(kind="text", rect=(0.02, 0.1, 0.25, 0.05), text=f"C$ {self.money:.0f}", size=26, color=(1.0, 0.85, 0.3)),
               dict(kind="text", rect=(0.02, 0.16, 0.35, 0.04), text=f"Backpack {len(self.bag)}/{BAG}  (C${val:.0f})", size=16),
               dict(kind="text", rect=(0.02, 0.21, 0.35, 0.04), text=RODS[self.rod][0], size=16, color=(0.8, 0.9, 1.0))]
        els += hotbar_hud([RODS[self.rod][0].replace(" Rod", "\nRod")], 0 if self.equipped else None)
        for pos, txt in ((MERCHANT + [0, 6.5, 0], "Merchant"), (BOARD + [0, 6, 0], "Rod Shop")):
            b = self.billboard(pos, txt, size=1.2, max_dist=90)
            if b:
                els.append(b)
        if self.phase == "charge":
            col = (0.3, 1.0, 0.4) if self.power >= 0.95 else (1.0, 0.85, 0.2)
            els += [dict(kind="text", rect=(0.6, 0.3, 0.2, 0.04), text="Cast power", size=16),
                    dict(kind="bar", rect=(0.6, 0.35, 0.2, 0.03), frac=self.power, color=col)]
        elif self.phase in ("wait", "shake") and self.bob is not None:
            b = self.billboard(self.bob + [0, 2.0, 0], "!" if self.phase == "shake" else "...", color=(1, 0.9, 0.2), size=2.0, max_dist=200)
            if b:
                els.append(b)
        if self.phase == "shake":
            els.append(dict(kind="button", id="shake", rect=self.shake_rect, text="Shake", color=(0.25, 0.55, 1.0), size=18))
        elif self.phase == "reel":
            x, y, bw, bh = REEL_BAR
            w = RODS[self.rod][2]
            els += [dict(kind="panel", rect=REEL_BAR, color=(0.08, 0.1, 0.15), alpha=0.8),
                    dict(kind="panel", rect=(x + self.zone * bw, y, w * bw, bh), color=(1, 1, 1), alpha=0.55),
                    dict(kind="icon", rect=(x + self.fpos * bw - 0.012, y - 0.004, 0.024, bh + 0.008), color=(0.2, 0.3, 0.45), text="><>", size=10),
                    dict(kind="bar", rect=(x, y + bh + 0.012, bw, 0.018), frac=self.prog, color=(0.3, 0.9, 0.4)),
                    dict(kind="text", rect=(x, y - 0.05, bw, 0.04), text="Hold to reel", size=16, align="center")]
        els += self._ui_els()
        return els

    def _ui_els(self) -> list:
        if self.ui == "merchant":
            val = sum(v for *_, v in self.bag)
            return [dict(kind="panel", rect=(0.32, 0.3, 0.36, 0.34), color=(0.95, 0.92, 0.85), alpha=0.97),
                    dict(kind="text", rect=(0.32, 0.32, 0.36, 0.08), text="Merchant", size=26, align="center", color=(0.3, 0.2, 0.1)),
                    dict(kind="text", rect=(0.32, 0.4, 0.36, 0.06), text=f"{len(self.bag)} fish worth C${val:.0f}", size=18, align="center", color=(0.2, 0.2, 0.2)),
                    dict(kind="button", id="sell_all", rect=(0.35, 0.5, 0.14, 0.09), text="Sell All", color=(0.2, 0.75, 0.3)),
                    dict(kind="button", id="close", rect=(0.51, 0.5, 0.14, 0.09), text="Close", color=(0.6, 0.6, 0.65))]
        if self.ui == "shop":
            els = [dict(kind="panel", rect=(0.22, 0.18, 0.56, 0.62), color=(0.12, 0.14, 0.18), alpha=0.92),
                   dict(kind="text", rect=(0.22, 0.2, 0.56, 0.07), text="Rod Shop", size=26, align="center"),
                   dict(kind="button", id="close", rect=(0.735, 0.2, 0.035, 0.06), text="X", color=(0.85, 0.2, 0.2))]
            for i, (nm, price, zone, lure, luck) in enumerate(RODS):
                y = 0.29 + 0.095 * i
                own = i in self.rods_owned
                els += [dict(kind="text", rect=(0.25, y, 0.3, 0.07), text=f"{nm}   zone {zone:.2f}  lure {lure:.2f}x  luck {luck:.1f}", size=14),
                        dict(kind="button", id=f"buy:{i}", rect=(0.62, y + 0.005, 0.13, 0.065), size=15,
                             text=("Equipped" if i == self.rod else "Equip") if own else f"C${price}",
                             color=(0.3, 0.3, 0.35) if i == self.rod else (0.2, 0.6, 0.95) if own else (0.2, 0.75, 0.3))]
            return els
        return []

    # ── expert ──
    def _next_rod(self):
        nxt = self.rod + 1
        return nxt if nxt < len(RODS) else None

    def expert(self) -> Action:
        self.goal = None
        p = self.player
        nxt = self._next_rod()
        val = sum(v for *_, v in self.bag)
        if self.ui == "merchant":
            self.say("merchant is open, click sell all" if self.bag else "nothing to sell, close the merchant")
            return self._click_ui("sell_all" if self.bag else "close")
        if self.ui == "shop":
            if nxt is not None and self.money >= RODS[nxt][1]:
                self.say("rod shop is open, buy the next rod")
                return self._click_ui(f"buy:{nxt}")
            self.say("done shopping, close the rod shop")
            return self._click_ui("close")
        if self.phase == "shake":
            self.say("fish is biting, click the shake button")
            x, y, w, h = self.shake_rect
            return self._point((x + w / 2, y + h / 2))
        if self.phase == "reel":
            self.say("fish is hooked, hold click to keep it in the zone")
            w = RODS[self.rod][2]
            lead = 0.75 * self.fpos + 0.25 * self.ftarget         # aim a little toward where it's darting
            err = lead - (self.zone + w / 2)
            hold = err - 0.3 * self.zone_v + 0.02 > 0             # PD on the zone centre; +bias fights the leftward drift
            return Action(keys=frozenset({"lmb"} if hold else ()), cursor=(0.5, 0.45))
        if self.phase == "charge":
            u = ((self.t - self.charge_t) / CHARGE_PERIOD) % 1.0
            rising = u < 0.5
            release = self.power >= 0.9 and rising
            self.say("charging the cast, release near the top of the meter")
            return Action(keys=frozenset() if release else frozenset({"lmb"}), cursor=(0.5, 0.45))
        if self.phase == "wait":
            self.say("bobber is in the water, wait for a bite")
            return Action(cursor=(0.5, 0.45))
        # idle: sell, upgrade, or go fish
        if len(self.bag) >= BAG or (self.bag and nxt is not None and self.money + val >= RODS[nxt][1] and self.money < RODS[nxt][1]):
            self.say("backpack is full, go sell fish to the merchant" if len(self.bag) >= BAG else
                     "fish are worth enough for a better rod, go sell them")
            return self._visit(MERCHANT + [0, 0, 3.5], 0)
        if nxt is not None and self.money >= RODS[nxt][1]:
            self.say("enough money for a better rod, go to the rod shop")
            return self._visit(BOARD + [0, 0, 3.0], 1)
        d = float(np.linalg.norm((SPOT - p.pos)[[0, 2]]))
        if d > 1.2:
            self.say("walk to the end of the pier to fish")
            return self._walk(SPOT, stop=0.8)
        yaw_err = math.degrees((0.0 - self.cam.yaw + math.pi) % (2 * math.pi) - math.pi)   # face the open water (+z)
        pitch = float(np.clip(math.degrees(math.radians(22) - self.cam.pitch), -10, 10))
        if abs(yaw_err) > 5:
            self.say("at the end of the pier, turn to face the water")
            return Action(look_x=float(np.clip(yaw_err, -30, 30)), look_y=pitch, cursor=(0.5, 0.45))
        if self.equipped != "rod":
            self.say("press 1 to equip the rod")
            return Action(keys=frozenset({"1"}), cursor=(0.5, 0.45))
        if self.mouse_held:   # still holding from the reel: a cast needs a fresh press
            self.say("let go of the mouse before the next cast")
            return Action(look_y=pitch, cursor=(0.5, 0.45))
        self.say("facing the water, hold click to start a cast")
        return Action(keys=frozenset({"lmb"}), look_y=pitch, cursor=(0.5, 0.45))

    def _walk(self, at, stop=1.0) -> Action:
        """The pier is a narrow plank over water: get on or off it through its base, never diagonally."""
        self.goal = at
        p = self.player.pos
        crossing = (p[2] > -1) != (at[2] > -1)
        lined_up = abs(p[0]) < 1.8 and p[2] > -6            # in the corridor the pier continues along
        if crossing and not lined_up:
            return self.steer_to(np.array([0.0, 0, -4.0]), stop_dist=0.5)
        return self.steer_to(at, stop_dist=stop)

    def _visit(self, at, prompt_i) -> Action:
        a = self._walk(at)
        if self.active_prompt() is self.prompts[prompt_i]:
            a.keys = a.keys | {"e"}
        return a

    def _point(self, uv) -> Action:
        """Cursor onto the target first, click on the next tick once it's there."""
        on = abs(self.cursor[0] - uv[0]) < 0.012 and abs(self.cursor[1] - uv[1]) < 0.02
        return Action(cursor=tuple(uv), click=on)

    def _click_ui(self, id_) -> Action:
        for el in self.hud():
            if el.get("id") == id_:
                x, y, w, h = el["rect"]
                return self._point((x + w / 2, y + h / 2))
        return Action(cursor=self.cursor)

    def potential(self) -> float:
        """Line progress (charge → bite → shake → reel), +1 per fish caught, plus closeness to wherever the expert is
        headed. Finishing a step never lowers Φ: a landed fish (reel 1.0 → idle 0) is paid back by `caught`, and
        charge = wait at the same power, so releasing a weak cast early earns nothing. (Was: catching dropped Φ by ~1
        and an early release jumped +0.2, so GRPO's noop beat the expert's chunk on fisch on average.)"""
        self.expert()
        cast = 0.15 + 0.15 * self.power
        phase = {"idle": 0.0, "charge": cast, "wait": cast,
                 "shake": 0.4 + 0.2 * (1 - self.shakes_left / 6), "reel": 0.6 + 0.4 * self.prog}[self.phase]
        dist = 0.0 if self.goal is None else float(np.linalg.norm((self.goal - self.player.pos)[[0, 2]]))
        return phase + self.caught - 0.01 * dist


if __name__ == "__main__":
    import pickle
    import time
    from core.actions import KEYS
    T = 3000   # 5 minutes
    for seed in (0, 1, 2):
        env = Fisch(seed=seed, max_steps=T)
        t0 = time.time()
        while not env.done:
            env.step(env.expert())
        ms = (time.time() - t0) / env.steps * 1000
        print(f"seed {seed} expert: score {env.score():.2f} {env.info()}  {ms:.2f} ms/tick")
        env = Fisch(seed=seed, max_steps=T)
        rng = np.random.default_rng(seed)
        while not env.done:
            env.step(Action(keys=frozenset(k for k in KEYS if rng.random() < 0.1), look_x=float(rng.normal(0, 10)),
                            cursor=tuple(rng.random(2)), click=rng.random() < 0.2))
        print(f"seed {seed} random: score {env.score():.2f} {env.info()}")
    # pickles mid-episode and replays identically
    env = Fisch(seed=5)
    for _ in range(400):
        env.step(env.expert())
    twin = pickle.loads(pickle.dumps(env))
    for _ in range(200):
        a = env.expert()
        r1, r2 = env.step(a)[0], twin.step(a)[0]
        assert r1 == r2 and env.phase == twin.phase
    print("pickle ok")
