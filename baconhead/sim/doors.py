import math

import numpy as np

from baconhead.core.actions import Action
from baconhead.sim.core import DT, HEAD_Y, Env, Prompt, health_hud, night_theme

H = 13.0               # room half-size (interior is 26x26)
WALL_H = 10.0
DOOR_W, DOOR_H = 5.0, 7.0
PITCH = 2 * H + 1      # centre-to-centre distance between rooms
RUSH_SPEED = 60.0      # studs/s; real Rush is very fast
RUSH_WARN = 3.5        # seconds of flicker before it arrives
RUSH_CHANCE = 0.22     # per new room, from room 3
SCREECH_TIME = 2.0
SCREECH_DMG = 40.0
COL = dict(wall=(0.55, 0.42, 0.30), floor=(0.45, 0.12, 0.12), ceil=(0.33, 0.28, 0.23), door=(0.30, 0.18, 0.10),
           wardrobe=(0.38, 0.24, 0.14), dresser=(0.52, 0.34, 0.20), key=(1.0, 0.85, 0.2), lamp=(1.0, 0.85, 0.55))


def _dir(a):
    return np.array([math.sin(a), math.cos(a)])


class Doors(Env):
    name = "doors"
    train_steps = 3000
    eval_kw = {}
    respawn_s = 1e9       # death ends the run

    def build(self):
        r = self.rng
        self.n_rooms = int(self.kw.get("rooms", 50))
        self.max_steps = int(self.kw.get("max_steps", self.train_steps))
        self.rooms, self.doors, self.closets, self.drawers, self.keys = [], {}, [], [], {}
        self.room_prompts = []
        c, heading, ent = np.zeros(2), 0.0, None
        for k in range(self.n_rooms):
            exit_a = heading if k < self.n_rooms - 1 else None
            self._room(k, c, ent, exit_a, dark=k >= 3 and r.random() < 0.2)
            if exit_a is None:
                break
            c = c + _dir(exit_a) * PITCH
            ent = exit_a + math.pi
            if r.random() < 0.35:   # turns stay within ±90° of forward, so the hotel never folds back on itself
                heading = 0.0 if heading != 0.0 else float(r.choice([-1, 1]) * math.pi / 2)

        self.room, self.best_room, self.room_t = 0, 0, 0.0
        self.hidden = -1
        self.e_lock = False
        self.key_held = set()
        self.rush_warn, self.rush_on, self.rush_end = 0.0, False, -9.0
        self.rush_path, self.rush_s, self.rush_pos = [], 0.0, None
        self.screech_pos, self.screech_t, self.screech_done = None, 0.0, set()
        self.goal = None
        self.prompts = self._near_prompts()
        self.cam.yaw, self.cam.pitch = 0.0, math.radians(15)

    # ── geometry ──
    # wall on side a of a room centred at c (2D), with a door-sized gap if asked
    def _wall(self, c, a, gap):
        w = self.world
        n = _dir(a)
        t = np.array([n[1], -n[0]])
        L = 2 * H + 2
        mid = c + n * (H + 0.5)
        along_x = abs(n[1]) > 0.5
        size = lambda length, hgt: (length, hgt, 1.0) if along_x else (1.0, hgt, length)
        if not gap:
            w.add((mid[0], WALL_H / 2, mid[1]), size(L, WALL_H), COL["wall"])
            return
        seg = (L - DOOR_W) / 2
        for s in (-1, 1):
            p = mid + t * s * (DOOR_W / 2 + seg / 2)
            w.add((p[0], WALL_H / 2, p[1]), size(seg, WALL_H), COL["wall"])
        w.add((mid[0], (DOOR_H + WALL_H) / 2, mid[1]), size(DOOR_W, WALL_H - DOOR_H), COL["wall"])

    def _room(self, k, c, ent, exit_a, dark):
        r, w = self.rng, self.world
        w.add((c[0], -0.5, c[1]), (2 * H + 2, 1, 2 * H + 2), COL["floor"])
        w.add((c[0], WALL_H + 0.5, c[1]), (2 * H + 2, 1, 2 * H + 2), COL["ceil"])
        sides = [i * math.pi / 2 for i in range(4)]
        same = lambda a, b: b is not None and abs((a - b + math.pi) % (2 * math.pi) - math.pi) < 0.1
        free = []
        for a in sides:
            if same(a, ent):
                continue                                   # the previous room built this wall, door and all
            self._wall(c, a, gap=same(a, exit_a))
            if not same(a, exit_a):
                free.append(a)
        if k == 0 and ent is None:
            free = [a for a in free if not same(a, math.pi)] or free   # keep the wall behind spawn bare
        lamp = w.add((c[0], WALL_H - 0.3, c[1]), (1.6, 0.4, 1.6), COL["lamp"] if not dark else (0.3, 0.3, 0.3), neon=not dark, collide=False)
        prompts = []
        room = dict(c=c.copy(), exit=exit_a, dark=dark, lamp=lamp)
        self.rooms.append(room)

        # furniture against the free walls: a wardrobe one side of centre, a dresser the other
        drawers_here = []
        for j, a in enumerate(free[:2]):
            n = _dir(a)
            t = np.array([n[1], -n[0]])
            s = float(r.choice([-1, 1])) * 5.5
            if j == 0 or r.random() < 0.7:                 # every room has at least one wardrobe
                depth = 2.6
                ctr = c + n * (H - depth / 2) + t * s
                part = w.add((ctr[0], 4.0, ctr[1]), (4.0, 8.0, depth) if abs(n[1]) > 0.5 else (depth, 8.0, 4.0), COL["wardrobe"])
                for dy in (2.5, 5.5):                      # door seam and handles, so it reads as a wardrobe
                    hp = c + n * (H - depth - 0.05) + t * (s + 0.4)
                    w.add((hp[0], dy, hp[1]), (0.2, 0.6, 0.2), (0.85, 0.7, 0.3), collide=False)
                front = c + n * (H - depth - 2.0) + t * s
                i = len(self.closets)
                self.closets.append(dict(part=part, room=k, front=np.r_[front[0], 0, front[1]], spot=np.r_[ctr[0], 0, ctr[1]],
                                         out_yaw=math.atan2(-n[0], -n[1])))
                prompts.append(Prompt(pos=np.r_[front[0] + n[0] * 1.2, 3.0, front[1] + n[1] * 1.2], action="Hide", obj="Wardrobe",
                                      on_trigger=("_hide", i), enabled=("_hide_ok", i)))
                prompts.append(Prompt(pos=np.r_[ctr[0], 3.0, ctr[1]], action="Exit", obj="Wardrobe",
                                      on_trigger=("_unhide",), enabled=("_exit_ok", i)))
            depth = 2.0
            ctr = c + n * (H - depth / 2) - t * s
            part = w.add((ctr[0], 1.5, ctr[1]), (4.0, 3.0, depth) if abs(n[1]) > 0.5 else (depth, 3.0, 4.0), COL["dresser"])
            front = c + n * (H - depth - 2.0) - t * s
            jd = len(self.drawers)
            self.drawers.append(dict(part=part, room=k, front=np.r_[front[0], 0, front[1]], top=np.r_[ctr[0], 3.15, ctr[1]],
                                     key=None, opened=False))
            drawers_here.append(jd)
            prompts.append(Prompt(pos=np.r_[front[0] + n[0] * 1.0, 2.0, front[1] + n[1] * 1.0], action="Open", obj="Dresser",
                                  hold=0.3, on_trigger=("_open", jd), enabled=("_open_ok", jd)))
        # a painting on a free wall, for the hotel look
        if free:
            n = _dir(free[-1])
            pp = c + n * (H - 0.05)
            w.add((pp[0], 5.5, pp[1]), (3.0, 2.0, 0.1) if abs(n[1]) > 0.5 else (0.1, 2.0, 3.0), tuple(r.uniform(0.2, 0.7, 3)), collide=False)

        if exit_a is not None:
            num = k + 1
            n = _dir(exit_a)
            dc = c + n * (H + 0.5)
            along_x = abs(n[1]) > 0.5
            door = w.add((dc[0], DOOR_H / 2, dc[1]), (DOOR_W, DOOR_H, 0.5) if along_x else (0.5, DOOR_H, DOOR_W), COL["door"])
            locked = num >= 2 and bool(drawers_here) and r.random() < 0.25
            lock = -1
            if locked:
                lp = dc - n * 0.4
                lock = w.add((lp[0], 3.5, lp[1]), (0.6, 0.8, 0.6), COL["key"], neon=True, collide=False)
                jd = drawers_here[int(r.integers(len(drawers_here)))]
                kp = self.drawers[jd]["top"] + [0, 0.2, 0]
                part = w.add(kp, (0.5, 0.2, 1.0), COL["key"], neon=True, collide=False, visible=False)
                self.drawers[jd]["key"] = num
                self.keys[num] = dict(part=part, pos=kp, front=self.drawers[jd]["front"], revealed=False, taken=False, room=k)
                prompts.append(Prompt(pos=kp, action="Pick up", obj=f"Key {num:04d}", on_trigger=("_take_key", num), enabled=("_key_ok", num)))
                prompts.append(Prompt(pos=np.r_[dc[0] - n[0] * 1.0, 3.5, dc[1] - n[1] * 1.0], action="Unlock", obj=f"Door {num:04d}",
                                      hold=0.6, on_trigger=("_unlock", num), enabled=("_unlock_ok", num)))
            self.doors[num] = dict(part=door, lock=lock, center=np.r_[dc[0], 0, dc[1]], n=n, locked=locked, open=False, room=k)
        self.room_prompts.append(prompts)

    # ── prompt callbacks (plain method names, so the env pickles) ──
    def _free(self):
        return not self.e_lock and self.player.alive

    def _hide_ok(self, i):
        return self._free() and self.hidden < 0 and self.closets[i]["room"] == self.room

    def _exit_ok(self, i):
        return self._free() and self.hidden == i

    def _open_ok(self, j):
        return self._free() and self.hidden < 0 and not self.drawers[j]["opened"] and self.drawers[j]["room"] == self.room

    def _key_ok(self, num):
        k = self.keys[num]
        return self._free() and k["revealed"] and not k["taken"]

    def _unlock_ok(self, num):
        return self._free() and self.doors[num]["locked"] and num in self.key_held

    def _fired(self):
        self.e_lock = True    # a prompt needs E released before the next one fires, like Roblox's

    def _hide(self, i):
        self._fired()
        c = self.closets[i]
        self.hidden = i
        self.world.collide[c["part"]] = False
        self.player.pos, self.player.vel = c["spot"].copy(), np.zeros(3)
        self.cam.first_person = True
        self.cam.yaw, self.cam.pitch = c["out_yaw"] % (2 * math.pi), 0.0

    def _unhide(self):
        self._fired()
        c = self.closets[self.hidden]
        self.world.collide[c["part"]] = True
        self.player.pos, self.player.vel = c["front"].copy(), np.zeros(3)
        self.cam.first_person = False
        self.cam.pitch = math.radians(15)
        self.hidden = -1

    def _open(self, j):
        self._fired()
        d = self.drawers[j]
        d["opened"] = True
        self.world.col[d["part"]] = np.array(COL["dresser"]) * 0.7
        if d["key"] is not None:
            k = self.keys[d["key"]]
            k["revealed"] = True
            self.world.visible[k["part"]] = True
        else:
            self.toast("Nothing here", 1.5)

    def _take_key(self, num):
        self._fired()
        k = self.keys[num]
        k["taken"] = True
        self.world.visible[k["part"]] = False
        self.key_held.add(num)
        self.toast(f"Picked up Key {num:04d}", 2)

    def _unlock(self, num):
        self._fired()
        d = self.doors[num]
        d["locked"] = False
        self.key_held.discard(num)
        if d["lock"] >= 0:
            self.world.visible[d["lock"]] = False

    # ── simulation ──
    def _room_of(self, p):
        for k in range(max(0, self.room - 2), min(len(self.rooms), self.room + 3)):
            c = self.rooms[k]["c"]
            if abs(p[0] - c[0]) < H + 0.4 and abs(p[2] - c[1]) < H + 0.4:
                return k
        return -1

    def _near_prompts(self):
        out = []
        for k in range(max(0, self.room - 1), min(len(self.room_prompts), self.room + 2)):
            out += self.room_prompts[k]
        return out

    def on_substep(self):
        p = self.player
        if "e" not in self.held:
            self.e_lock = False
        if self.hidden >= 0:   # pinned in the wardrobe: no walking, the camera can still look around
            p.pos, p.vel, p.stun = self.closets[self.hidden]["spot"].copy(), np.zeros(3), 0.2
        k = self._room_of(p.pos)
        if k >= 0 and k != self.room:
            self.room, self.room_t = k, 0.0
            self.prompts = self._near_prompts()
            if k > self.best_room:
                self.reward += 0.5 * (k - self.best_room)
                self.best_room = k
                if k >= 3 and self.rush_warn <= 0 and not self.rush_on and self.rng.random() < RUSH_CHANCE:
                    self.rush_warn, self.rush_room = RUSH_WARN, k
                if k == len(self.rooms) - 1:
                    self.reward += 2.0
                    self.toast("You escaped the hotel!", 4)
                    self.done = True
        self.room_t += DT
        # doors swing open as you walk up to them
        for num in (self.room, self.room + 1):
            d = self.doors.get(num)
            if d and not d["open"] and not d["locked"] and np.linalg.norm((d["center"] - p.pos)[[0, 2]]) < 5.0:
                d["open"] = True
                self.world.collide[d["part"]] = self.world.visible[d["part"]] = False
        self._rush()
        self._screech()
        if self.steps >= self.max_steps:
            self.done = True

    def _rush(self):
        if self.rush_warn > 0:
            self.rush_warn -= DT
            if self.rush_warn <= 0:
                a, b = max(0, self.rush_room - 3), min(len(self.rooms) - 1, self.rush_room + 3)
                path = []
                for k in range(a, b + 1):
                    path.append(np.r_[self.rooms[k]["c"][0], 2.5, self.rooms[k]["c"][1]])
                    if k < b:
                        path.append(self.doors[k + 1]["center"] + [0, 2.5, 0])
                self.rush_path, self.rush_s, self.rush_on = path, 0.0, True
        if not self.rush_on:
            return
        self.rush_s += RUSH_SPEED * DT
        s, pos = self.rush_s, None
        for a, b in zip(self.rush_path, self.rush_path[1:]):
            seg = float(np.linalg.norm(b - a))
            if s <= seg:
                pos = a + (b - a) * (s / max(seg, 1e-6))
                break
            s -= seg
        if pos is None:
            self.rush_on, self.rush_pos, self.rush_end = False, None, self.t
            return
        self.rush_pos = pos
        if self.player.alive and self.hidden < 0 and self._room_of(pos) == self.room >= 0:
            self.damage(999, "Rush got you")

    def _screech(self):
        p = self.player
        if self.screech_pos is not None:
            v = self.screech_pos - self.cam.pos
            if v @ self.cam.fwd / max(np.linalg.norm(v), 1e-6) > math.cos(math.radians(30)):
                self.toast("You scared Screech off", 1.5)
                self.screech_pos = None
                return
            self.screech_t -= DT
            if self.screech_t <= 0:
                self.screech_pos = None
                self.damage(SCREECH_DMG, "Screech got you")
                if p.alive:
                    self.toast("Screech bit you! (-40)", 2)
            return
        r = self.room
        if (self.rooms[r]["dark"] and r not in self.screech_done and self.hidden < 0 and p.alive and self.room_t > 0.8
                and self.rng.random() < 0.8 * DT):
            self.screech_done.add(r)
            # off-screen relative to the camera (FOV is ±54° wide), at about the player's distance: you have to turn
            a = self.cam.yaw + math.radians(float(self.rng.uniform(70, 150)) * float(self.rng.choice([-1, 1])))
            dist = max(3.0, float(np.linalg.norm((p.pos - self.cam.pos)[[0, 2]])))
            self.screech_pos = np.r_[self.cam.pos[0] + math.sin(a) * dist, p.pos[1] + HEAD_Y + 0.5, self.cam.pos[2] + math.cos(a) * dist]
            self.screech_t = SCREECH_TIME
            self.toast("Psst!", 2)

    def on_death(self):
        self.reward -= 2.0
        self.done = True

    # ── rendering ──
    def scene_theme(self):
        flick = self.rush_warn > 0 and (int(self.t * 12) % 3 == 0 or int(self.t * 7) % 4 == 1)
        lights = []
        if not flick:
            for k in range(max(0, self.room - 1), min(len(self.rooms), self.room + 2)):
                rm = self.rooms[k]
                if not rm["dark"]:
                    lights.append((rm["c"][0], WALL_H - 1.5, rm["c"][1], 19.0, 1.1, 0.85, 0.55))   # reaches the corners, spills only near doorways
        return night_theme(self.theme, 0.97, lights)

    def dynamic_boxes(self):
        out = []
        if self.rush_pos is not None:   # a black mass with a pale, stretched face
            x, y, z = self.rush_pos
            out += [((x, y + 1, z), (5, 7, 5), 0.0, (0.03, 0.03, 0.03)),
                    ((x - 1.2, y + 2.4, z), (1.0, 1.4, 5.3), 0.0, (0.95, 0.95, 0.9), 1),
                    ((x + 1.2, y + 2.4, z), (1.0, 1.4, 5.3), 0.0, (0.95, 0.95, 0.9), 1),
                    ((x, y + 0.2, z), (3.0, 0.8, 5.3), 0.0, (0.9, 0.9, 0.85), 1)]
        if self.screech_pos is not None:
            x, y, z = self.screech_pos
            out += [((x, y, z), (0.9, 0.9, 0.9), 0.0, (0.02, 0.02, 0.02)),
                    ((x, y + 0.05, z), (0.4, 0.15, 0.95), self.cam.yaw, (0.9, 0.9, 0.9), 1)]
        return out

    def hud(self):
        els = [dict(kind="text", rect=(0.4, 0.02, 0.2, 0.06), text=f"Room {self.room:04d}", size=26, align="center",
                    color=(0.95, 0.85, 0.6))]
        els += health_hud(self.player.health)
        if self.key_held:
            els.append(dict(kind="text", rect=(0.02, 0.9, 0.2, 0.05), text=f"Key {max(self.key_held):04d}", size=20, color=COL["key"]))
        for num in (self.room + 1, self.room + 2):
            d = self.doors.get(num)
            if d and (num == self.room + 1 or self.doors[self.room + 1]["open"]):   # billboards ignore walls; only show what you could see
                b = self.billboard(d["center"] + np.r_[-d["n"][0] * 0.6, DOOR_H + 0.9, -d["n"][1] * 0.6], f"{num:04d}",
                                   color=(0.95, 0.85, 0.55), size=1.2, max_dist=60)
                if b:
                    els.append(b)
        if self.hidden >= 0:   # inside the wardrobe you see out through a slit
            els += [dict(kind="panel", rect=(0, 0, 1, 0.42), color=(0.02, 0.015, 0.01), alpha=0.95),
                    dict(kind="panel", rect=(0, 0.58, 1, 0.42), color=(0.02, 0.015, 0.01), alpha=0.95)]
        return els

    def info(self):
        return dict(room=self.room, best_room=self.best_room, health=self.player.health, alive=self.player.alive,
                    hidden=self.hidden >= 0, deaths=self.deaths)

    def score(self):
        return float(self.best_room)

    # ── expert ──
    def expert(self):
        p = self.player
        self.goal = None
        if not p.alive:
            self.say("dead, the run is over")
            return Action(cursor=self.cursor)
        threat = self.rush_warn > 0 or self.rush_on
        if self.hidden >= 0:
            if not threat and self.t - self.rush_end > 0.6:
                self.say("rush has passed, get out of the wardrobe")
                return self._press(("_unhide",))
            self.say("hiding in the wardrobe, wait for rush to pass")
            return Action(cursor=self.cursor)
        if self.screech_pos is not None:
            self.say("psst, turn around and look at screech")
            return self._look_at(self.screech_pos)
        r = self.room
        if threat:
            self.say("lights are flickering, hide in the wardrobe")
            cl = [i for i, c in enumerate(self.closets) if c["room"] == r]
            i = min(cl, key=lambda i: np.linalg.norm((self.closets[i]["front"] - p.pos)[[0, 2]]))
            return self._reach(self.closets[i]["front"], ("_hide", i))
        num = r + 1
        d = self.doors.get(num)
        if d is None:
            self.say("last room reached, nothing left to do")
            return Action(cursor=self.cursor)
        if d["locked"] and num not in self.key_held:
            k = self.keys[num]
            if k["revealed"] and not k["taken"]:
                self.say("the key is in the open dresser, pick it up")
                return self._reach(k["front"], ("_take_key", num))
            self.say("door is locked, search the dressers for the key")
            jd = next(j for j, dr in enumerate(self.drawers) if dr["key"] == num)
            return self._reach(self.drawers[jd]["front"], ("_open", jd))
        n3 = np.r_[d["n"][0], 0, d["n"][1]]
        inside, outside = d["center"] - n3 * 2.5, d["center"] + n3 * 4.0
        if d["locked"]:
            self.say("got the key, unlock the door")
            return self._reach(inside, ("_unlock", num))
        c = self.rooms[r]["c"]
        cc = np.r_[c[0], 0, c[1]]
        rel = p.pos - inside
        along = rel[0] * n3[0] + rel[2] * n3[2]
        lateral = abs(rel[0] * n3[2] - rel[2] * n3[0])
        if along > -1.0 and lateral < 1.5:
            self.say("lined up with the door, walk through it")
            return self._go(outside)                  # lined up in the doorway: commit and walk through
        pd = np.linalg.norm((inside - p.pos)[[0, 2]])
        if pd > np.linalg.norm((inside - cc)[[0, 2]]) + 1:
            self.say("cut through the middle of the room to the door")
            return self._go(cc)                       # cut through the middle of the room, clear of furniture
        self.say("walk to the next door")
        return self._go(inside, stop=0.3)

    def _go(self, goal, stop=0.6):
        self.goal = np.asarray(goal, float)
        return self.steer_to(self.goal, stop_dist=stop)

    # hold E only while the right prompt is the active one, and let go between prompts
    def _press(self, trigger):
        pr = self.active_prompt()
        if self.e_lock or pr is None or pr.on_trigger[:len(trigger)] != tuple(trigger):
            return Action(cursor=self.cursor)
        return Action(keys=frozenset({"e"}), cursor=self.cursor)

    def _reach(self, front, trigger):
        p = self.player
        if np.linalg.norm((front - p.pos)[[0, 2]]) > 1.2:
            a = self._go(front, stop=0.5)
            pr = self.active_prompt()
            if pr is not None and pr.on_trigger == tuple(trigger) and not self.e_lock:
                a.keys = frozenset({"e"})            # already in range: no need to stop first
            return a
        self.goal = np.asarray(front, float)
        return self._press(trigger)

    def _look_at(self, q):
        v = q - self.cam.pos
        want_yaw = math.atan2(v[0], v[2])
        want_pitch = -math.atan2(v[1], math.hypot(v[0], v[2]))
        ey = math.degrees((want_yaw - self.cam.yaw + math.pi) % (2 * math.pi) - math.pi)
        ep = math.degrees(want_pitch - self.cam.pitch)
        return Action(look_x=float(np.clip(ey, -30, 30)), look_y=float(np.clip(ep, -10, 10)), cursor=self.cursor)

    # closeness to the expert's next target, and being hidden while Rush is coming
    def potential(self):
        self.expert()
        threat = self.rush_warn > 0 or self.rush_on
        phi = 0.5 if (threat and self.hidden >= 0) else 0.0
        if self.goal is not None:
            phi -= 0.01 * float(np.linalg.norm((self.goal - self.player.pos)[[0, 2]]))
        return phi


if __name__ == "__main__":
    import os
    import pickle
    import time
    from baconhead.core.actions import KEYS
    ticks = 3000
    for seed in (0, 1, 2):
        env = Doors(seed=seed, max_steps=ticks)
        t0 = time.time()
        R = 0.0
        events = dict(rush=0, screech=0, keys=0)
        prev = (False, False)
        while not env.done:
            if env.rush_warn > 0 and not prev[0]:
                events["rush"] += 1
            if env.screech_pos is not None and not prev[1]:
                events["screech"] += 1
            prev = (env.rush_warn > 0, env.screech_pos is not None)
            R += env.step(env.expert())[0]
        events["keys"] = sum(k["taken"] for k in env.keys.values())
        ms = (time.time() - t0) / env.steps * 1000
        print(f"seed {seed} expert: score {env.score():.0f}/{len(env.rooms) - 1}  {env.info()}  return {R:.1f}  {events}  "
              f"{env.steps} ticks, {ms:.1f} ms/tick")
        env = Doors(seed=seed, max_steps=ticks)
        rng = np.random.default_rng(seed)
        while not env.done:
            env.step(Action(keys=frozenset(k for k in KEYS if k not in ("esc",) and rng.random() < 0.12),
                            look_x=float(rng.normal(0, 10)), cursor=tuple(rng.random(2)), click=rng.random() < 0.05))
        print(f"seed {seed} random: score {env.score():.0f}  {env.info()}")

    # pickle mid-episode: a copy must step identically
    env = Doors(seed=3)
    for _ in range(300):
        env.step(env.expert())
    twin = pickle.loads(pickle.dumps(env))
    for _ in range(200):
        a = env.expert()
        r1, r2 = env.step(a)[0], twin.step(a)[0]
        assert r1 == r2 and np.allclose(env.player.pos, twin.player.pos)
    print("pickle ok")

    # renders of the key moments
    from PIL import Image
    from baconhead.sim.render import render
    os.makedirs("/tmp/bh/doors", exist_ok=True)
    shots, want = {}, ("start", "locked", "warn", "rush", "hidden", "screech", "dark")
    for seed in range(12):
        env = Doors(seed=seed, max_steps=3000)
        while not env.done and len(shots) < len(want):
            if "start" not in shots and env.steps == 5:
                shots["start"] = env.scene()
            d = env.doors.get(env.room + 1)
            if "locked" not in shots and d and d["locked"] and env.keys[env.room + 1]["revealed"]:
                shots["locked"] = env.scene()
            if "warn" not in shots and env.rush_warn > 0 and env.hidden < 0:
                shots["warn"] = env.scene()
            if "rush" not in shots and env.rush_pos is not None and env.hidden >= 0:
                shots["rush"] = env.scene()
            if "hidden" not in shots and env.hidden >= 0 and env.rush_warn > 0:
                shots["hidden"] = env.scene()
            if "screech" not in shots and env.screech_pos is not None:
                shots["screech"] = env.scene()
            if "dark" not in shots and env.rooms[env.room]["dark"] and env.room_t > 0.5:
                shots["dark"] = env.scene()
            env.step(env.expert())
    keys = [k for k in want if k in shots]
    imgs = render([shots[k] for k in keys], 360, 640, device="mps").cpu().numpy()
    for k, im in zip(keys, imgs):
        Image.fromarray(im).save(f"/tmp/bh/doors/{k}.png")
    print("rendered:", keys)
