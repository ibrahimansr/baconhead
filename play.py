"""Play the training games yourself, with Roblox's Mac controls.

    python play.py obby --difficulty 0.7
    python play.py brainrot
    python play.py adoptme --record          # your play becomes BC data (demos/sim/*.npz)

WASD/arrows move (left/right arrows turn the camera), Space jumps (hold to bunny-hop), right-drag turns
the camera, scroll zooms, Shift toggles shift lock, E holds prompts, left-click hits UI and 3D objects,
1 equips a tool.
Esc opens the Roblox menu (Reset Character with R), or click the Roblox logo top-left.
F1 agent view (the exact 448x252 frame the model sees) · F2 hand control to the scripted expert ·
F3 hand control to --policy · F5 new level · F12 quit.
"""

from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import pygame
import torch

import sim
from core.actions import KEYS
from sim.core import DT, PX_PER_DEG, SUBSTEPS, Input
from sim.hud import draw as draw_hud
from sim.render import best_device, render

KEYMAP = {pygame.K_w: "w", pygame.K_a: "a", pygame.K_s: "s", pygame.K_d: "d", pygame.K_UP: "w", pygame.K_DOWN: "s",
          pygame.K_SPACE: "space", pygame.K_LSHIFT: "shift", pygame.K_RSHIFT: "shift", pygame.K_e: "e", pygame.K_q: "q",
          pygame.K_r: "r", pygame.K_f: "f", pygame.K_1: "1", pygame.K_2: "2", pygame.K_3: "3",
          pygame.K_ESCAPE: "esc"}
ARROW_TURN_DEG_S = 120.0   # calibration knob: Roblox's arrow-key camera turn rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("game", choices=list(sim.ENVS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--difficulty", type=float, default=0.5, help="obby only, 0..1")
    ap.add_argument("--res", default="640x360", help="3D render resolution; the window is --scale times this")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--policy", default=None, help="model dir / LoRA dir / HF id to watch with F3")
    ap.add_argument("--no-shadows", action="store_true")
    ap.add_argument("--crisp-hud", action="store_true", help="draw UI at window res (sharper, ~10ms slower)")
    args = ap.parse_args()

    W, H = map(int, args.res.split("x"))
    WW, WH = W * args.scale, H * args.scale
    dev = best_device()
    kw = {"difficulty": args.difficulty} if args.game == "obby" else {}
    seed = args.seed
    env = sim.make(args.game, seed=seed, **kw)

    pygame.init()
    screen = pygame.display.set_mode((WW, WH))
    font = pygame.font.Font(None, 22)
    clock = pygame.time.Clock()

    runner = None
    driver = "human"
    agent_view = False
    dragging = False
    drag_anchor = (0, 0)
    look = [0.0, 0.0]
    click = False
    scroll = 0
    rec = []

    def save_rec():   # one file per level, since replay needs the seed the inputs were played on
        if args.record and rec:
            os.makedirs("demos/sim", exist_ok=True)
            path = f"demos/sim/{args.game}_s{seed}_{int(time.time())}.npz"
            np.savez_compressed(path, game=args.game, seed=seed, kw=repr(kw), inputs=np.array(rec, np.float32))
            print(f"saved {len(rec)} substeps → {path}")
        rec.clear()

    pending = []          # substeps of the current agent tick
    last = time.perf_counter()
    acc = 0.0
    t_render = 0.0

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_F12):
                running = False
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_F1:
                agent_view = not agent_view
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_F2:
                driver = "expert" if driver != "expert" else "human"
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_F3 and args.policy:
                if runner is None:
                    from model.vlm import VLMRunner
                    runner = VLMRunner(args.policy, device=dev)
                runner.reset()
                driver = "policy" if driver != "policy" else "human"
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_F5:
                save_rec()
                seed += 1
                env = sim.make(args.game, seed=seed, **kw)
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 3:
                dragging, drag_anchor = True, ev.pos
                pygame.mouse.set_visible(False)
                pygame.event.set_grab(True)
                pygame.mouse.get_rel()
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 3:
                dragging = False
                pygame.event.set_grab(env.cam.shiftlock or env.cam.first_person)
                pygame.mouse.set_visible(not (env.cam.shiftlock or env.cam.first_person))
                pygame.mouse.set_pos(drag_anchor)   # Roblox leaves the cursor where the drag started
            elif ev.type == pygame.MOUSEMOTION and (dragging or ((env.cam.shiftlock or env.cam.first_person) and not env.menu)):
                look[0] += ev.rel[0] / PX_PER_DEG
                look[1] += ev.rel[1] / PX_PER_DEG
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                click = True
            elif ev.type == pygame.MOUSEWHEEL:
                scroll += -1 if ev.y > 0 else 1

        now = time.perf_counter()
        acc = min(acc + now - last, 4 * DT)   # never more than 4 substeps of catch-up, so a stall doesn't teleport you
        last = now
        pressed = pygame.key.get_pressed()
        keys = frozenset(n for k, n in KEYMAP.items() if pressed[k]) | (
            {"lmb"} if pygame.mouse.get_pressed()[0] else set())
        if pressed[pygame.K_LEFT]:
            look[0] -= ARROW_TURN_DEG_S * acc
        if pressed[pygame.K_RIGHT]:
            look[0] += ARROW_TURN_DEG_S * acc
        mx, my = pygame.mouse.get_pos()
        locked = (env.cam.shiftlock or env.cam.first_person) and not env.menu   # Roblox frees the mouse while its menu is open
        cursor = (0.5, 0.5) if locked else (mx / WW, my / WH)

        was_shiftlock = (env.cam.shiftlock or env.cam.first_person) and not env.menu
        while acc >= DT:
            acc -= DT
            if driver == "human":
                inp = Input(keys=keys, dyaw=look[0], dpitch=look[1], cursor=cursor, click=click, scroll=scroll)
                look, click, scroll = [0.0, 0.0], False, 0
            else:
                if not pending:
                    if driver == "expert":
                        a = env.expert()
                    else:
                        from model.vlm import OBS_H, OBS_W
                        a = runner.act(render([env.scene()], OBS_H, OBS_W, device=dev)[0].cpu().numpy())
                    pending = [Input(keys=a.keys, dyaw=a.look_x / SUBSTEPS, dpitch=a.look_y / SUBSTEPS, cursor=a.cursor,
                                     click=a.click and k == 0, scroll=a.scroll if k == 0 else 0) for k in range(SUBSTEPS)]
                inp = pending.pop(0)
            env.substep(inp)
            if args.record:
                rec.append(_pack(inp))
            if env.done:
                print("episode done:", env.info())
                save_rec()
                env = sim.make(args.game, seed=seed, **kw)
        locked = (env.cam.shiftlock or env.cam.first_person) and not env.menu
        if locked != was_shiftlock and not dragging:
            pygame.event.set_grab(locked)
            pygame.mouse.set_visible(not locked)

        t0 = time.perf_counter()
        sc = env.scene()
        if agent_view:
            from model.vlm import OBS_H, OBS_W
            img = render([sc], OBS_H, OBS_W, device=dev, shadows=not args.no_shadows).cpu().numpy()[0]
            surf = pygame.transform.scale(pygame.image.frombuffer(img.tobytes(), (OBS_W, OBS_H), "RGB"), (WW, WH))
        elif args.crisp_hud:
            img = render([sc], H, W, device=dev, hud=False, shadows=not args.no_shadows)
            big = torch.nn.functional.interpolate(img.permute(0, 3, 1, 2).float(), size=(WH, WW), mode="bilinear")
            big = draw_hud(sc["hud"], big.clamp(0, 255).byte().permute(0, 2, 3, 1)[0].cpu().numpy())
            surf = pygame.image.frombuffer(big.tobytes(), (WW, WH), "RGB")
        else:
            img = render([sc], H, W, device=dev, shadows=not args.no_shadows).cpu().numpy()[0]
            surf = pygame.transform.smoothscale(pygame.image.frombuffer(img.tobytes(), (W, H), "RGB"), (WW, WH))
        screen.blit(surf, (0, 0))
        t_render = 0.9 * t_render + 0.1 * (time.perf_counter() - t0)
        p = env.player
        hs = math.hypot(p.vel[0], p.vel[2])
        info = "  ".join(f"{k}={v:.0f}" if isinstance(v, float) else f"{k}={v}" for k, v in env.info().items())
        txt = f"{clock.get_fps():4.0f} fps  render {t_render * 1000:3.0f}ms  [{driver}]  speed {hs:4.1f}  y {p.pos[1]:5.1f}  " \
              f"{'grounded' if p.grounded else 'air'}  t {env.t:5.0f}s  {info}" + ("  REC" if args.record else "")
        screen.blit(font.render(txt, True, (255, 255, 255), (0, 0, 0)), (4, WH - 20))
        pygame.display.flip()
        clock.tick(60)

    save_rec()
    pygame.quit()


def _pack(inp: Input) -> list:
    """One substep of input as floats: key bits, look, cursor, click, scroll. data.replay_sim unpacks it."""
    return [float(k in inp.keys) for k in KEYS] + [inp.dyaw, inp.dpitch, inp.cursor[0], inp.cursor[1], float(inp.click), float(inp.scroll)]


if __name__ == "__main__":
    main()
