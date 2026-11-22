from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@lru_cache(maxsize=64)
def font(px):
    return ImageFont.load_default(size=max(6, px))


def _c(col, a=1.0):
    return tuple(int(255 * max(0, min(1, x))) for x in col[:3]) + (int(255 * a),)


# draws the UI onto an (H, W, 3) uint8 frame. Draw(RGB, "RGBA") alpha-blends each shape as it lands
def draw(elements, rgb, chrome=True):
    im = Image.fromarray(rgb)
    W, H = im.size
    d = ImageDraw.Draw(im, "RGBA")
    s = H / 720   # every pixel size below is authored at 720p
    if chrome:
        _topbar(d, W, H, s)
    for el in elements:
        _element(d, el["kind"], el, W, H, s)
    return np.asarray(im)


def _element(d, k, el, W, H, s):
    if k == "panel":
        x, y, w, h = _px(el["rect"], W, H)
        d.rounded_rectangle((x, y, x + w, y + h), radius=int(10 * s), fill=_c(el.get("color", (0.1, 0.1, 0.1)), el.get("alpha", 0.6)))
    elif k == "button":
        x, y, w, h = _px(el["rect"], W, H)
        col = el.get("color", (0.2, 0.6, 1.0))
        col = tuple(min(1.0, c * 0.7 + 0.35) for c in col) if el.get("hover") else col   # lit under the cursor
        d.rounded_rectangle((x, y, x + w, y + h), radius=int(10 * s), fill=_c(col),
                            outline=_c((0, 0, 0), 0.5), width=max(1, int(2 * s)))
        _text_center(d, (x + w / 2, y + h / 2), el.get("text", ""), int(el.get("size", 20) * s), el.get("fg", (1, 1, 1)), s)
    elif k == "text":
        x, y, w, h = _px(el["rect"], W, H)
        f = font(int(el.get("size", 20) * s))
        align = el.get("align", "left")
        tx = x if align == "left" else (x + w / 2 if align == "center" else x + w)
        anchor = {"left": "lm", "center": "mm", "right": "rm"}[align]
        d.text((tx, y + h / 2), el["text"], font=f, fill=_c(el.get("color", (1, 1, 1))), anchor=anchor,
               stroke_width=max(1, int(1.5 * s)), stroke_fill=(0, 0, 0, 200))
    elif k == "bar":
        x, y, w, h = _px(el["rect"], W, H)
        d.rounded_rectangle((x, y, x + w, y + h), radius=int(h / 2), fill=_c((0.1, 0.1, 0.1), 0.7))
        fw = max(0, min(1, el["frac"])) * w
        if fw > 1:
            d.rounded_rectangle((x, y, x + fw, y + h), radius=int(h / 2), fill=_c(el.get("color", (0.3, 0.9, 0.3))))
    elif k == "icon":
        x, y, w, h = _px(el["rect"], W, H)
        d.ellipse((x, y, x + w, y + h), fill=_c(el.get("color", (0.9, 0.9, 0.9))), outline=_c((1, 1, 1)), width=max(1, int(2 * s)))
        _text_center(d, (x + w / 2, y + h / 2), el.get("text", ""), int(el.get("size", 18) * s), (1, 1, 1), s)
    elif k == "leaderboard":
        _leaderboard(d, W, H, s, el["cols"], el["rows"])
    elif k == "prompt":
        _prompt(d, W, H, s, el)
    elif k == "billboard":
        u, v = el["at"]
        _text_center(d, (u * W, v * H), el["text"], int(18 * s * el.get("size", 1.0)), el.get("color", (1, 1, 1)), s)
    elif k == "toast":
        x, y, w, h = _px(el["rect"], W, H)
        d.rounded_rectangle((x, y, x + w, y + h), radius=int(8 * s), fill=(20, 20, 20, 170))
        _text_center(d, (x + w / 2, y + h / 2), el["text"], int(18 * s), (1, 1, 1), s)


def _px(rect, W, H):
    x, y, w, h = rect
    return x * W, y * H, w * W, h * H


def _text_center(d, xy, text, px, col, s):
    lines = text.split("\n")
    for i, line in enumerate(lines):   # PIL can't centre-anchor multiline text, so lines go one at a time
        y = xy[1] + (i - (len(lines) - 1) / 2) * px * 1.15
        d.text((xy[0], y), line, font=font(px), fill=_c(col), anchor="mm", stroke_width=max(1, int(1.5 * s)), stroke_fill=(0, 0, 0, 220))


# the two rounded buttons every Roblox experience has top-left: the menu and chat
def _topbar(d, W, H, s):
    r, pad, sz = int(10 * s), int(12 * s), int(44 * s)
    for i in range(2):
        x = pad + i * (sz + int(8 * s))
        d.rounded_rectangle((x, pad, x + sz, pad + sz), radius=r, fill=(20, 20, 20, 150))
        m = int(12 * s)
        if i == 0:
            d.rounded_rectangle((x + m, pad + m, x + sz - m, pad + sz - m), radius=int(3 * s), outline=(255, 255, 255, 230), width=max(1, int(3 * s)))
        else:
            d.rounded_rectangle((x + m, pad + m, x + sz - m, pad + sz - m - int(4 * s)), radius=int(5 * s), fill=(255, 255, 255, 230))


# top-right player list with leaderstats, which is where obbies show Stage and simulators show Cash
def _leaderboard(d, W, H, s, cols, rows):
    rw, rh = int(90 * s) * (len(cols) + 1), int(26 * s)
    x0, y0 = W - rw - int(10 * s), int(10 * s)
    d.rounded_rectangle((x0, y0, x0 + rw, y0 + rh * (len(rows) + 1)), radius=int(6 * s), fill=(20, 20, 20, 150))
    cw = rw // (len(cols) + 1)
    f = font(int(14 * s))
    for j, name in enumerate(["People"] + list(cols)):
        d.text((x0 + cw * j + cw / 2, y0 + rh / 2), name, font=f, fill=(255, 255, 255, 200), anchor="mm")
    for i, row in enumerate(rows[:8]):
        y = y0 + rh * (i + 1) + rh / 2
        for j, val in enumerate(row):
            d.text((x0 + cw * j + cw / 2, y), str(val), font=f, fill=(255, 255, 255, 255), anchor="mm")


# A ProximityPrompt: key cap with the hold ring on the left, object and action text on the right
def _prompt(d, W, H, s, el):
    u, v = el["at"]
    x, y = u * W + 20 * s, v * H - 36 * s
    w, h = 190 * s, 72 * s
    d.rounded_rectangle((x, y, x + w, y + h), radius=int(10 * s), fill=(15, 15, 15, 180))
    kc = (x + 36 * s, y + h / 2)
    rr = 22 * s
    d.ellipse((kc[0] - rr, kc[1] - rr, kc[0] + rr, kc[1] + rr), fill=(40, 40, 40, 230), outline=(255, 255, 255, 120), width=max(1, int(2 * s)))
    if el.get("progress", 0) > 0:
        d.arc((kc[0] - rr, kc[1] - rr, kc[0] + rr, kc[1] + rr), -90, -90 + 360 * min(1, el["progress"]), fill=(255, 255, 255, 255), width=max(2, int(4 * s)))
    d.text(kc, "E", font=font(int(22 * s)), fill=(255, 255, 255, 255), anchor="mm")
    d.text((x + 66 * s, y + h * 0.33), el.get("sub", ""), font=font(int(14 * s)), fill=(220, 220, 220, 255), anchor="lm")
    d.text((x + 66 * s, y + h * 0.66), el["text"], font=font(int(20 * s)), fill=(255, 255, 255, 255), anchor="lm")
