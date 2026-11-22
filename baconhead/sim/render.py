"""Batched box raycaster in plain torch: every Roblox part is a box, so one slab test renders the world.

Same code on CPU, MPS, and CUDA. The player's window and the agent's observations come from this one
function, so what you play is what the model trains on.

Two passes:
  1. primary rays, tile-culled: each TILE×TILE block of pixels only tests the M boxes whose screen
     bounding rect touches it (M is typically 10-25 of K=64-128), so cost ≈ B·H·W·M.
  2. sun shadow rays at 1/3 resolution against every box, since shadow casters can be off-screen.
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch

from sim import hud as hudlib

TILE = 16
SHADOW_STRIDE = 3   # shadow rays every 3rd pixel each way: 1/9 the cost, upsampled bilinearly
_SIGNS = torch.tensor([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=torch.float32)


def _pack(scenes, device, bucket=16):
    B = len(scenes)
    K = -(-max(1, max(len(s["c"]) for s in scenes)) // bucket) * bucket   # bucketed so compile sees few shapes
    c = np.full((B, K, 3), 1e6, np.float32)   # padding boxes sit a million studs away with zero size, so nothing hits them
    h = np.zeros((B, K, 3), np.float32)
    yaw = np.zeros((B, K), np.float32)
    col = np.zeros((B, K, 3), np.float32)
    neon = np.zeros((B, K), bool)
    for b, s in enumerate(scenes):
        n = len(s["c"])
        c[b, :n], h[b, :n], yaw[b, :n], col[b, :n], neon[b, :n] = s["c"], s["h"], s["yaw"], s["col"], s["neon"]
    T = lambda x: torch.from_numpy(x).to(device)
    return T(c), T(h), T(yaw), T(col), T(neon)


def _trace(o, d, c, h, cy, sy, want_normal=True):
    """Nearest box along each ray. o (N,P,3) or (N,1,3), d (N,P,3); boxes c,h (N,K,3), cy/sy (N,K).
    Returns t (N,P) with inf for misses, box index, and the hit's local-frame normal and point."""
    rel = o[:, :, None, :] - c[:, None, :, :]                                   # (N,P|1,K,3)
    cyk, syk = cy[:, None, :], sy[:, None, :]                                   # (N,1,K)
    lo = torch.stack([cyk * rel[..., 0] - syk * rel[..., 2], rel[..., 1].expand_as(cyk * rel[..., 0]),
                      syk * rel[..., 0] + cyk * rel[..., 2]], -1)
    dx, dy, dz = d[..., 0:1], d[..., 1:2], d[..., 2:3]                          # (N,P,1)
    ld = torch.stack([cyk * dx - syk * dz, dy.expand_as(cyk * dx), syk * dx + cyk * dz], -1)   # (N,P,K,3)
    ld = torch.where(ld.abs() < 1e-8, torch.full_like(ld, 1e-8), ld)
    inv = 1.0 / ld
    hk = h[:, None, :, :]
    t1, t2 = (-hk - lo) * inv, (hk - lo) * inv
    tmin_ax = torch.minimum(t1, t2)                                              # (N,P,K,3)
    tn = tmin_ax.amax(-1)
    tf = torch.maximum(t1, t2).amin(-1)
    hit = (tf >= tn) & (tn > 1e-3)
    tt = torch.where(hit, tn, torch.full_like(tn, float("inf")))
    tbest, ib = tt.min(-1)                                                       # (N,P)
    ib = ib.clamp(min=0)   # MPS returns -1 for all-miss rows; those pixels are sky anyway
    if not want_normal:
        return tbest, ib, None, None
    g = ib[..., None, None].expand(-1, -1, 1, 3)
    ax = tmin_ax.gather(2, g).squeeze(2).argmax(-1)                              # which slab we entered through
    ldb = ld.gather(2, g).squeeze(2)
    lob = lo.expand(-1, g.shape[1], -1, -1).gather(2, g).squeeze(2)
    n_l = torch.zeros_like(ldb).scatter_(-1, ax[..., None], 1.0) * -torch.sign(ldb.gather(-1, ax[..., None]))
    p_l = lob + ldb * tbest.clamp(max=1e6)[..., None]
    return tbest, ib, n_l, p_l


def _primary(cam, d, c, h, yaw, col, neon):
    """Per tile: hit distance, world normal, albedo, neon flag, and distance to the nearest face edge."""
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    t, ib, n_l, p_l = _trace(cam, d, c, h, cy, sy)
    cyb, syb = cy.gather(1, ib), sy.gather(1, ib)
    n = torch.stack([cyb * n_l[..., 0] + syb * n_l[..., 2], n_l[..., 1], -syb * n_l[..., 0] + cyb * n_l[..., 2]], -1)
    alb = col.gather(1, ib[..., None].expand(-1, -1, 3))
    is_neon = neon.gather(1, ib)
    hb = h.gather(1, ib[..., None].expand(-1, -1, 3))
    e = (hb - p_l.abs()).masked_fill(n_l != 0, 1e9).amin(-1)
    return t, n, alb, is_neon, e


def _shadow(p, sun, c, h, yaw):
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    ts, _, _, _ = _trace(p, sun.expand_as(p), c, h, cy, sy, want_normal=False)
    return torch.isfinite(ts)


if os.environ.get("BH_COMPILE", "1") == "1":   # ~5x on CPU; BH_COMPILE=0 to debug in eager
    _primary = torch.compile(_primary)
    _shadow = torch.compile(_shadow)


def _tile_lists(cam, f, r, up, th, aspect, c, h, yaw, TY, TX):
    """Which boxes can appear in each screen tile, from each box's projected corner bounds."""
    B, K = c.shape[:2]
    sg = _SIGNS.to(c.device)
    loc = sg[None, None] * h[:, :, None, :]                                     # (B,K,8,3)
    cy, sy = torch.cos(yaw)[..., None], torch.sin(yaw)[..., None]
    corners = c[:, :, None, :] + torch.stack([cy * loc[..., 0] + sy * loc[..., 2], loc[..., 1],
                                              -sy * loc[..., 0] + cy * loc[..., 2]], -1)
    rel = corners - cam[:, None, None, :]
    z = (rel * f[:, None, None]).sum(-1)
    x = (rel * r[:, None, None]).sum(-1)
    y = (rel * up[:, None, None]).sum(-1)
    near = z < 0.05
    zc = z.clamp(min=0.05)
    u = 0.5 + x / (zc * th[:, None, None] * aspect) / 2
    v = 0.5 - y / (zc * th[:, None, None]) / 2
    u0, u1, v0, v1 = u.amin(-1), u.amax(-1), v.amin(-1), v.amax(-1)
    straddle = near.any(-1) & ~near.all(-1)                                     # crosses the camera plane: could be anywhere
    u0, v0 = torch.where(straddle, torch.zeros_like(u0), u0), torch.where(straddle, torch.zeros_like(v0), v0)
    u1, v1 = torch.where(straddle, torch.ones_like(u1), u1), torch.where(straddle, torch.ones_like(v1), v1)
    valid = ~near.all(-1) & (h.abs().sum(-1) > 0)
    tu = torch.arange(TX + 1, device=c.device, dtype=torch.float32) / TX
    tv = torch.arange(TY + 1, device=c.device, dtype=torch.float32) / TY
    ou = (u0[..., None] <= tu[None, None, 1:]) & (u1[..., None] >= tu[None, None, :-1])    # (B,K,TX)
    ov = (v0[..., None] <= tv[None, None, 1:]) & (v1[..., None] >= tv[None, None, :-1])    # (B,K,TY)
    m = valid[..., None, None] & ov[..., :, None] & ou[..., None, :]                         # (B,K,TY,TX)
    m = m.reshape(B, K, TY * TX).transpose(1, 2)                                             # (B,T,K)
    order = (m.float() + torch.linspace(0.5, 0, K, device=c.device)).argsort(-1, descending=True)   # overlapping boxes first
    return order, m.sum(-1)


@torch.no_grad()
def render(scenes: list, H: int, W: int, device="cpu", shadows=True, hud=True, chrome=True, augment=None) -> torch.Tensor:
    """scenes from Env.scene() → (B, H, W, 3) uint8 on device."""
    B = len(scenes)
    c, h, yaw, col, neon = _pack(scenes, device)
    K = c.shape[1]
    F = lambda xs: torch.tensor(np.stack(xs), dtype=torch.float32, device=device)
    cam = F([s["cam_pos"] for s in scenes])                                     # (B,3)
    f = F([s["cam_fwd"] for s in scenes])
    cyaw = F([s["cam_yaw"] for s in scenes])
    r = torch.stack([torch.cos(cyaw), torch.zeros_like(cyaw), -torch.sin(cyaw)], -1)
    up = torch.linalg.cross(f, r)
    th = F([math.tan(math.radians(s["fov"]) / 2) for s in scenes])
    aspect = W / H

    TY, TX = -(-H // TILE), -(-W // TILE)
    Hp, Wp = TY * TILE, TX * TILE
    vs = (torch.arange(Hp, device=device, dtype=torch.float32) + 0.5) / H
    us = (torch.arange(Wp, device=device, dtype=torch.float32) + 0.5) / W
    vv, uu = torch.meshgrid(vs, us, indexing="ij")
    tile = lambda g: g.reshape(TY, TILE, TX, TILE).permute(0, 2, 1, 3).reshape(TY * TX, TILE * TILE)   # tile-major pixels
    uu, vv = tile(uu), tile(vv)
    d = (f[:, None, None] + (2 * uu[None, ..., None] - 1) * th[:, None, None, None] * aspect * r[:, None, None]
         - (2 * vv[None, ..., None] - 1) * th[:, None, None, None] * up[:, None, None])       # (B,T,Pt,3)
    d = d / d.norm(dim=-1, keepdim=True)

    order, counts = _tile_lists(cam, f, r, up, th, aspect, c, h, yaw, TY, TX)            # (B,T,K), (B,T)
    T, Pt = TY * TX, TILE * TILE
    order, counts = order.reshape(B * T, K), counts.reshape(B * T)
    env_of = torch.arange(B, device=device).repeat_interleave(T)
    d = d.reshape(B * T, Pt, 3)
    t = torch.full((B * T, Pt), float("inf"), device=device)
    n = torch.zeros(B * T, Pt, 3, device=device)
    alb = torch.zeros(B * T, Pt, 3, device=device)
    is_neon = torch.zeros(B * T, Pt, dtype=torch.bool, device=device)
    e = torch.zeros(B * T, Pt, device=device)
    lo = 0
    for Mg in (8, 16, 32, 64, 128, 256, 512):   # tiles grouped by how many boxes touch them, so a busy tile doesn't pad the rest
        Mg = min(Mg, K)
        sel = torch.nonzero((counts > lo) & (counts <= Mg)).squeeze(1)
        lo = Mg
        if sel.numel() > 0:
            eb, ix = env_of[sel], order[sel, :Mg]                                          # (G,), (G,Mg)
            okm = torch.arange(Mg, device=device)[None] < counts[sel][:, None]
            gat = lambda x: x[eb[:, None], ix]
            ct = torch.where(okm[..., None], gat(c), torch.full((1,), 1e6, device=device))
            ht = torch.where(okm[..., None], gat(h), torch.zeros((1,), device=device))
            r_ = _primary(cam[eb][:, None], d[sel], ct, ht, gat(yaw), gat(col), gat(neon))
            t[sel], n[sel], alb[sel], is_neon[sel], e[sel] = r_
        if lo >= K:
            break

    def untile(x):   # (B*T, TILE*TILE, ...) → (B, H, W, ...)
        tail = x.shape[2:]
        x = x.reshape(B, TY, TX, TILE, TILE, *tail).transpose(2, 3)
        return x.reshape(B, Hp, Wp, *tail)[:, :H, :W]
    t, n, alb, is_neon, e = untile(t), untile(n), untile(alb), untile(is_neon), untile(e)
    d = untile(d)

    th_ = [s["theme"] for s in scenes]
    sun = F([x["sun"] for x in th_])[:, None, None]                           # (B,1,1,3)
    sky_top = F([x["sky_top"] for x in th_])[:, None, None]
    sky_hor = F([x["sky_hor"] for x in th_])[:, None, None]
    amb = F([x["ambient"] for x in th_])[:, None, None]
    fog = F([x["fog"] for x in th_])[:, None, None]

    hitm = torch.isfinite(t)
    lam = (n * sun).sum(-1).clamp(min=0)
    if shadows:
        tc = t.clamp(max=1e5)[..., None]
        pw = (cam[:, None, None] + d * tc + n * 0.02)[:, 1::SHADOW_STRIDE, 1::SHADOW_STRIDE]   # sparse shadow rays
        hh, ww = pw.shape[1:3]
        sh = _shadow(pw.reshape(B, hh * ww, 3), sun[:, 0], c, h, yaw).reshape(B, 1, hh, ww).float()
        sh = torch.nn.functional.interpolate(sh, size=(H, W), mode="bilinear", align_corners=False)[:, 0]   # soft edges, like Roblox's
        lam = lam * (1 - 0.75 * sh)
    sun_k = F([x.get("sun_k", 1.0) for x in th_])[:, None, None]             # <1 at dusk and night
    a = amb * (0.85 + 0.15 * n[..., 1])
    shade = a + (1 - a) * lam * sun_k
    edge = 0.82 + 0.18 * (e / (0.004 * t.clamp(max=1e4) + 0.04)).clamp(0, 1)   # darker near part edges, which is what makes bricks read as bricks
    lit = alb * (shade * edge)[..., None]
    L = max(len(x.get("lights", ())) for x in th_)
    if L:   # point lights (campfires, lamps): smooth falloff to their radius, no shadows
        arr = np.zeros((B, L, 7), np.float32)
        arr[..., 3] = 1e-3
        for b, x in enumerate(th_):
            for j, lt in enumerate(x.get("lights", ())):
                arr[b, j] = lt
        lt = torch.from_numpy(arr).to(device)                                   # (B,L,[x,y,z,radius,r,g,b])
        pw = cam[:, None, None] + d * t.clamp(max=1e5)[..., None] + n * 0.05    # (B,H,W,3)
        rel = lt[:, None, None, :, :3] - pw[..., None, :]                        # (B,H,W,L,3)
        dist = rel.norm(dim=-1).clamp(min=1e-3)
        fall = (1 - dist / lt[:, None, None, :, 3]).clamp(0, 1) ** 2
        lam_l = ((rel / dist[..., None]) * n[..., None, :]).sum(-1).clamp(min=0)
        add = ((fall * (0.35 + 0.65 * lam_l))[..., None] * lt[:, None, None, :, 4:]).sum(-2)
        lit = lit + alb * add
    lit = torch.where(is_neon[..., None], alb * 1.15, lit)
    sky = sky_hor + (sky_top - sky_hor) * (d[..., 1:2] * 2.5).clamp(0, 1)
    sky = torch.where(((d * sun).sum(-1, keepdim=True) > 0.9985), torch.ones_like(sky), sky)
    fogk = (1 - torch.exp(-t.clamp(max=1e5)[..., None] / fog[..., None])) * 0.8
    img = torch.where(hitm[..., None], lit * (1 - fogk) + sky_hor * fogk, sky)

    if augment is not None:   # photometric jitter for training, from a CPU torch Generator so runs replay
        g = augment
        bri = 1 + 0.15 * (torch.rand(B, 1, 1, 1, generator=g) - 0.5).to(device)
        con = 1 + 0.2 * (torch.rand(B, 1, 1, 1, generator=g) - 0.5).to(device)
        m = img.mean((1, 2, 3), keepdim=True)
        img = ((img - m) * con + m) * bri + 0.02 * torch.randn(img.shape, generator=g).to(device)
    img = (img.clamp(0, 1) * 255).to(torch.uint8)
    if hud:   # PIL draws the UI on CPU; at 256x144 that's well under a millisecond per frame
        cpu = img.cpu().numpy()
        img = torch.from_numpy(np.stack([hudlib.draw(s["hud"], fr, chrome=chrome) for s, fr in zip(scenes, cpu)])).to(device)
    return img


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
