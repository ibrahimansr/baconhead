from dataclasses import dataclass

import numpy as np

TICK_S = 0.1   # 10 Hz decisions, 6 physics substeps of 1/60 s each

KEYS = ("w", "a", "s", "d", "space", "shift", "e", "q", "r", "f", "1", "2", "3", "lmb", "esc")   # lmb = left mouse held (tools, guns, fishing)
# 30 deg/tick cap = 60px of drag in ~100ms, the most the old calibration found Roblox reliably reads
LOOK_X = (-30.0, -15.0, -7.0, -3.0, -1.0, 0.0, 1.0, 3.0, 7.0, 15.0, 30.0)   # camera yaw degrees per tick, + is right
LOOK_Y = (-10.0, -4.0, 0.0, 4.0, 10.0)                                     # camera pitch degrees per tick, + is looking down
CUR_X, CUR_Y = 64, 36                                                      # cursor grid, ~30px cells on a 1920x1080 window
SCROLL = (-1, 0, 1)                                                        # -1 zooms in, +1 zooms out

# number of choices per action field
HEADS = {"keys": len(KEYS), "look_x": len(LOOK_X), "look_y": len(LOOK_Y),
         "cur_x": CUR_X, "cur_y": CUR_Y, "click": 2, "scroll": len(SCROLL)}


@dataclass
class Action:
    keys: frozenset = frozenset()
    look_x: float = 0.0            # degrees
    look_y: float = 0.0
    cursor: tuple = (0.5, 0.5)     # normalized window coords, (0,0) top-left
    click: bool = False
    scroll: int = 0

    def to_ids(self):
        return {
            "keys": np.array([k in self.keys for k in KEYS], dtype=np.float32),
            "look_x": _nearest(LOOK_X, self.look_x),
            "look_y": _nearest(LOOK_Y, self.look_y),
            "cur_x": min(CUR_X - 1, max(0, int(self.cursor[0] * CUR_X))),
            "cur_y": min(CUR_Y - 1, max(0, int(self.cursor[1] * CUR_Y))),
            "click": int(self.click),
            "scroll": SCROLL.index(int(np.sign(self.scroll))),
        }

    @classmethod
    def from_ids(cls, ids):
        keys = frozenset(k for k, on in zip(KEYS, ids["keys"]) if on > 0.5)
        return cls(
            keys=keys,
            look_x=LOOK_X[int(ids["look_x"])],
            look_y=LOOK_Y[int(ids["look_y"])],
            cursor=((int(ids["cur_x"]) + 0.5) / CUR_X, (int(ids["cur_y"]) + 0.5) / CUR_Y),
            click=bool(ids["click"]),
            scroll=SCROLL[int(ids["scroll"])],
        )


def _nearest(bins, v):
    return int(np.argmin([abs(b - v) for b in bins]))


# what the action looks like after a round trip through the policy's heads
def quantize(a):
    return Action.from_ids(a.to_ids())


if __name__ == "__main__":
    a = Action(keys=frozenset({"w", "space"}), look_x=12.0, look_y=-3.0, cursor=(0.21, 0.9), click=True, scroll=1)
    b = quantize(a)
    assert b.keys == a.keys and b.look_x == 15.0 and b.look_y == -4.0 and b.click and b.scroll == 1
    assert abs(b.cursor[0] - 0.21) < 1 / CUR_X and abs(b.cursor[1] - 0.9) < 1 / CUR_Y
    print("actions ok", HEADS)
