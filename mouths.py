"""口の種類。元の絵から取り出した口（kind 0、口を結んだ微笑み）のほかに、同じ絵柄（線＋唇の影）で描く 4 種。
どの顔にも同じ形で置く（顔の幅 w に対する割合で描く）。口を描く位置は土台の口の目印。"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

KINDS = ("smile", "toothy", "flat", "smirk", "small")  # 0 は元の絵の口
INK = (62, 33, 28)
SS = 4  # 描くときの拡大

# 口の線の半分の幅（w に対する割合）
HALF = {"toothy": 0.20, "flat": 0.16, "smirk": 0.17, "small": 0.11}


def _curve(x0, x1, f, n=48):
    xs = np.linspace(x0, x1, n)
    return [(x, f(x)) for x in xs]


def draw_mouth(kind: int, gender: str, center, w: float, size, skin) -> Image.Image:
    """kind（1〜4）の口を center（元の絵の座標）に、顔の幅 w に合わせて描いた RGBA（size の大きさ）。skin は 0〜1 の RGB。"""
    name = KINDS[kind]
    out = Image.new("RGBA", size, (0, 0, 0, 0))
    pw, ph = int(w * 0.8), int(w * 0.5)
    ox, oy = int(center[0] - pw / 2), int(center[1] - ph / 2)
    patch = Image.new("RGBA", (pw * SS, ph * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(patch)
    skin = np.asarray(skin)
    shade = tuple(int(c * 255) for c in np.clip(skin * (0.90, 0.80, 0.78), 0, 1))  # 下唇の影
    lip = tuple(int(c * 255) for c in np.clip(skin * (1.0, 0.84, 0.80), 0, 1))     # 女の唇
    teeth = (250, 246, 240)
    inside = (112, 42, 40)

    def P(x, y):  # w 単位（中心が原点）→ 拡大した patch の座標
        return ((x * w + pw / 2) * SS, (y * w + ph / 2) * SS)

    def poly(pts, **kw):
        d.polygon([P(x, y) for x, y in pts], **kw)

    def ell(cx, cy, rx, ry, fill):
        (x0, y0), (x1, y1) = P(cx - rx, cy - ry), P(cx + rx, cy + ry)
        d.ellipse([x0, y0, x1, y1], fill=fill + (255,))

    def stroke(pts, t):
        d.line([P(x, y) for x, y in pts], fill=INK + (255,), width=max(1, int(t * w * SS)), joint="curve")
        for x, y in (pts[0], pts[-1]):  # 端を丸く
            r = t * w * SS / 2
            cx, cy = P(x, y)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=INK + (255,))

    half = HALF[name]
    female = gender == "female"
    if name == "toothy":
        top = lambda x: 0.004 - 0.026 * (x / half) ** 2
        bot = lambda x: 0.004 + 0.062 * (1 - (x / half) ** 2) ** 0.7 - 0.026 * (x / half) ** 2 + 0.004
        up, lo = _curve(-half, half, top), _curve(half, -half, bot)
        poly(up + lo, fill=inside + (255,))
        # 歯: 開きの上の半分を白く
        tl = lambda x: top(x) + 0.5 * (bot(x) - top(x))
        poly(_curve(-half * 0.94, half * 0.94, top) + _curve(half * 0.94, -half * 0.94, tl), fill=teeth + (255,))
        if female:
            poly(_curve(-half * 0.7, half * 0.7, lambda x: bot(x) + 0.004) +
                 _curve(half * 0.7, -half * 0.7, lambda x: bot(x) + 0.028 * (1 - (x / half) ** 2)), fill=lip + (255,))
        else:
            ell(0, 0.085, half * 0.42, 0.016, shade)
        stroke(up, 0.0075)
        stroke(lo, 0.005)
    elif name == "flat":
        line = _curve(-half, half, lambda x: 0.002 + 0.006 * (x / half) ** 2)
        if female:
            poly(_curve(-half * 0.85, half * 0.85, lambda x: 0.002 - 0.02 * (1 - (x / half) ** 2) ** 0.5 + 0.006 * (x / half) ** 2) +
                 _curve(half * 0.85, -half * 0.85, lambda x: 0.004), fill=lip + (255,))
            poly(_curve(-half * 0.8, half * 0.8, lambda x: 0.006) +
                 _curve(half * 0.8, -half * 0.8, lambda x: 0.006 + 0.04 * (1 - (x / half) ** 2) ** 0.6), fill=lip + (255,))
        else:
            ell(0, 0.045, half * 0.38, 0.016, shade)
        stroke(line, 0.0075)
    elif name == "smirk":
        # 右の口角を上げ、左は少し下げる。右の端に小さな笑いじわの弧
        line = _curve(-half, half, lambda x: 0.012 - 0.030 * ((x + half) / (2 * half)) ** 2.2 + 0.004 * (1 - (x / half) ** 2))
        if female:
            poly(_curve(-half * 0.8, half * 0.8, lambda x: 0.014 - 0.02 * ((x + half) / (2 * half)) ** 2.2) +
                 _curve(half * 0.8, -half * 0.8, lambda x: 0.014 + 0.04 * (1 - (x / half) ** 2) ** 0.6), fill=lip + (255,))
        else:
            ell(0.03, 0.05, half * 0.4, 0.016, shade)
        stroke(line, 0.0075)
        stroke(_curve(0.012, 0.045, lambda x: -0.030 - 0.020 * ((x - 0.012) / 0.033) ** 2 + 0.014 + 0.012), 0.004)
    else:  # small: 小さめのやわらかい微笑み
        line = _curve(-half, half, lambda x: 0.006 - 0.014 * (x / half) ** 2)
        if female:
            poly(_curve(-half * 0.85, half * 0.85, lambda x: 0.002 - 0.014 * (1 - (x / half) ** 2) ** 0.5 + 0.006 - 0.014 * (x / half) ** 2) +
                 _curve(half * 0.85, -half * 0.85, lambda x: 0.006 - 0.014 * (x / half) ** 2), fill=lip + (255,))
            poly(_curve(-half * 0.8, half * 0.8, lambda x: 0.006 - 0.014 * (x / half) ** 2) +
                 _curve(half * 0.8, -half * 0.8, lambda x: 0.006 - 0.014 * (x / half) ** 2 + 0.036 * (1 - (x / half) ** 2) ** 0.6),
                 fill=lip + (255,))
        else:
            ell(0, 0.042, half * 0.45, 0.014, shade)
        stroke(line, 0.0075)
    patch = patch.resize((pw, ph), Image.Resampling.LANCZOS)
    out.alpha_composite(patch, (max(ox, 0), max(oy, 0)))
    return out
