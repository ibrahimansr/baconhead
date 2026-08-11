"""The discrete action vocabulary shared by the world model, the planner, and the injector."""

from __future__ import annotations

from control.physics import degrees_to_px

STEP_MS = 200      # every action lasts exactly one tick, which is what lets the world model learn z' = f(z, a)
TURN_DEG = 30.0    # how far the camera swings in one tick, small enough to finish inside STEP_MS

ACTIONS: list[tuple[str, tuple[str, ...], float, bool]] = [
    ("idle",         (),             0.0,       False),
    ("w",            ("w",),         0.0,       False),
    ("w_jump",       ("w", "space"), 0.0,       False),   # the obby workhorse: run and jump together
    ("a",            ("a",),         0.0,       False),
    ("d",            ("d",),         0.0,       False),
    ("s",            ("s",),         0.0,       False),
    ("jump",         ("space",),     0.0,       False),
    ("look_left",    (),            -TURN_DEG,  False),
    ("look_right",   (),             TURN_DEG,  False),
    ("w_look_left",  ("w",),        -TURN_DEG,  False),   # turning while moving is how you follow a curved path
    ("w_look_right", ("w",),         TURN_DEG,  False),
    ("click",        (),             0.0,       True),    # simulators and tycoons are mostly UI clicks
]
N_ACTIONS = len(ACTIONS)
NAMES = [a[0] for a in ACTIONS]
ID = {n: i for i, n in enumerate(NAMES)}


def execute(action_id: int, center: tuple[float, float] | None = None) -> None:
    """Run one tick of an action, blocking for STEP_MS, always releasing its keys."""
    import time

    from control import inject   # imported here so training on Linux never touches macOS Quartz

    name, keys, look_deg, do_click = ACTIONS[action_id % N_ACTIONS]
    cx, cy = center or (960.0, 540.0)
    for k in keys:
        inject.key_down(k)
    try:
        if do_click:
            inject.click(cx, cy)
        if look_deg:
            dx = degrees_to_px(look_deg) * (1 if look_deg > 0 else -1)
            inject.look(dx, STEP_MS, center=(cx, cy))   # the drag itself takes the whole tick, so no extra sleep
        else:
            time.sleep(STEP_MS / 1000.0)
    finally:
        inject.release_keys(*keys)   # a held key that escapes here would make the character run forever


def label_from_state(keys_down: set[str], look_dx: int) -> int:
    """Turn one tick of observed human input into the closest action id."""
    fwd = "w" in keys_down
    turn = 0 if abs(look_dx) < degrees_to_px(TURN_DEG) // 2 else (1 if look_dx > 0 else -1)   # ignore small mouse jitter
    if turn and fwd:
        return ID["w_look_right"] if turn > 0 else ID["w_look_left"]
    if turn:
        return ID["look_right"] if turn > 0 else ID["look_left"]
    if fwd and "space" in keys_down:
        return ID["w_jump"]
    if fwd:
        return ID["w"]
    for k in ("a", "d", "s"):   # checked after forward because W dominates when combined
        if k in keys_down:
            return ID[k]
    if "space" in keys_down:
        return ID["jump"]
    return ID["idle"]
