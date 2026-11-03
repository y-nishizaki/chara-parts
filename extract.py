"""1 体ずつ丸ごと描いた顔（whole/raw/<id>.png）と髪を消した版（<id>_bald.png）から部品を取り出す。

使い方:
  python3 extract.py marks     手で決めた目印（landmarks.manual.json）に、頭の上・顔の幅・あご先を測って足し、landmarks.json に書く
  python3 extract.py           部品を assets/parts_whole/<id>/ に書く（landmarks.json を使う）

部品（どれも元の絵と同じ大きさの RGBA。座標は元の絵のまま）:
  base        顔の土台（髪を消した版の頭と首。目・眉・鼻・口を肌で埋めた物）
  eyes/brows/nose/mouth  髪を消した版から、肌の色との差で抜いた物（縁は柔らかく、色は肌から戻す）
  hair_front  元の絵と髪を消した版の差のうち、髪を消した版の絵（頭・首・服）の上に乗る所
  hair_back   差のうち、髪を消した版では背景だった所（頭の外の後ろ髪・横髪）
meta.json に、髪を消した版を元の絵に合わせたずれ・肌と髪の地の色を書く。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

import ages

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "whole" / "raw"
OUT = ROOT / "assets" / "parts_whole"
IDS = ages.IDS
# 髪を消した版の服だけを描き替えた絵（首と肩の持ち主, 服の名前, 絵の名前の後ろ）。render_whole.OUTFITS と同じ並び。
# 女の服は、首を細くし首の筋の線を消した版（_slim）
BODIES = [("m0", "jacket", ""), ("m1", "knit", ""), ("m2", "cardigan", ""), ("m3", "vest", ""),
          ("f0", "cardigan", "_slim"), ("f1", "jacket", "_slim"), ("f2", "knit", "_slim"), ("f3", "blouse", "_slim")]
# 年代ごとの服。女の服は首を細くし首の筋を消した版（_slim）から取る
BODIES += [(owner, name, "")
           for outfits in ages.NEW_OUTFITS.values() for name, gender, owner in outfits]

# 部品を抜く楕円の半径（顔の幅に対する割合）。landmarks.json の "<part>_r" で体ごとに上書きできる
WINDOW_R = {"eye": (0.19, 0.12), "brow": (0.23, 0.08), "nose": (0.11, 0.22), "mouth": (0.22, 0.11)}
WINDOW_R_INFANT = {"eye": (0.24, 0.185)}  # 乳幼児の目は顔の幅に対して大きいので、窓を広げる（小さいと目の下の縁が土台に三日月形に残る。0y の b0m3）
WINDOW_OFFSET = {"nose": (0.0, -0.14), "mouth": (0.0, 0.02)}  # 鼻の窓は鼻先より上（鼻筋の線を含める）
KEY_LO, KEY_HI = 0.035, 0.20   # 肌の色との差がこれより小さい所は透明、大きい所は不透明
HAIR_DIFF = 0.16              # 元の絵と髪を消した版の差がこれを超えたら髪の候補
BACK_MARGIN = 0.03            # 後ろ髪を塗り広げるのは、頭のこれだけ外まで（顔の幅に対する割合）
HAIR_SKIN_DIST = 0.15        # 肌の色からこれより近い画素は髪に入れない
BACK_SPREAD = 0.12          # 後ろ髪を頭の内側へ塗り広げる幅（顔の幅に対する割合）
HAIR_THIN = 4               # 顔の上の毛のうち、これより細い線（元の絵の px の半分）は消す
HAIR_LINE_NEAR = 4          # 髪の色の画素からこれだけ（px）の内の暗い画素は、毛束の線として髪に入れる
NECK_HALF = 0.45            # あご先より下は、首の幅（真ん中からこれだけ。顔の幅に対する割合）だけ残す（肩の線を入れない。首の輪郭線は残す）
HAIR_COLOR_FRACTION = 0.45  # かけらのうち髪の色の画素の割合がこれ以下なら髪にしない
HAIR_MIN_PIECE = 1500         # 髪の候補のかけらのうち、これより小さい物は消す（描き直しのずれの線）
SEAM_UNDER = 4                # 後ろ髪を前髪の下へ入れる幅（元の絵の px）
RIM_CLOSE = 2                 # 頭の外の髪の外形の、幅がこれの 2 倍以下の隙間を埋める（px）
HAIR_HOLE_LUM = 0.45          # 髪の中の穴のうち、元の絵の明るさがこれ未満の所だけ埋める（耳・頬の影は埋めない）
HAIR_HOLE_MAX = 600             # 髪に囲まれた穴をこの px 未満なら髪に戻す
HAIR_CLOSE = 11                 # 髪の内側のハイライトの抜けを閉じる幅（元の絵の px。抜けの幅がこの 2 倍以下）
HAIR_HOLE_MAX = 3000          # 髪の中の閉じた穴のうち、これより小さい物は埋める（顔の形の大きな穴は残す）
LINE_CLEAN = {"m1", "m11"}    # 土台から老けて見える線を消す体
LINE_CLEAN_INSET = 10         # 輪郭線からこれだけ（元の絵の px）内側だけを消す（輪郭線は残す）
SPECK_MAX = 400               # 土台に残った線のかけらとして埋める大きさの上限（元の絵の px）
FACE_LINE = 8                 # 顔の輪郭線の太さの分（元の絵の px）
EAR_BOTTOM = 0.06             # 耳たぶの下の端（鼻先からの高さ。顔の幅に対する割合）
NECK_CUT = 0.30               # 首はあご先から顔の幅のこれだけ下まで（その下は服なので、描き直した体を使う）
NECK_FEATHER = 0.10           # 首の下の端をぼかす幅（顔の幅に対する割合）


def load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA")).astype(np.float64) / 255


def save(px: np.ndarray, path: Path) -> None:
    Image.fromarray(np.clip(px * 255 + 0.5, 0, 255).astype(np.uint8), "RGBA").save(path, optimize=False)


def shift_image(px: np.ndarray, dx: float, dy: float, s: float, cx: float, cy: float) -> np.ndarray:
    """(cx, cy) を中心に s 倍し、(dx, dy) ずらす（髪を消した版を元の絵に合わせる）。"""
    img = Image.fromarray(np.clip(px * 255 + 0.5, 0, 255).astype(np.uint8), "RGBA").convert("RGBa")
    # 出力の (x, y) ← 入力の ((x - cx - dx) / s + cx, ...)
    a, e = 1 / s, 1 / s
    c = cx - (cx + dx) / s
    f = cy - (cy + dy) / s
    out = img.transform(img.size, Image.Transform.AFFINE, (a, 0, c, 0, e, f), resample=Image.Resampling.BICUBIC)
    return np.asarray(out.convert("RGBA")).astype(np.float64) / 255


def premul(px):
    return px[..., :3] * px[..., 3:4]


def register(orig, bald, marks):
    """顔の下半分（目の線より下・あごより上・頬の内側）が一番合うずれと倍率を探す（粗く探してから 1px で詰める）。"""
    w = marks["faceR"][0] - marks["faceL"][0]
    x0, x1 = round(marks["faceL"][0] + 0.12 * w), round(marks["faceR"][0] - 0.12 * w)
    y0, y1 = round(marks["eyeL"][1] - 0.05 * w), round(marks["chin"][1] + 0.02 * w)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ref = premul(orig)[y0:y1, x0:x1]

    def err(dx, dy, s, f=1):
        """f 分の 1 に縮めた絵で比べる（粗い探しを速くする）。"""
        b = small[f][1] if f > 1 else bald
        r = small[f][0] if f > 1 else ref
        got = premul(shift_image(b, dx / f, dy / f, s, cx / f, cy / f))[round(y0 / f):round(y0 / f) + r.shape[0],
                                                                       round(x0 / f):round(x0 / f) + r.shape[1]]
        return float(np.abs(got - r).mean())

    def shrink(px, f):
        img = Image.fromarray(np.clip(px * 255 + 0.5, 0, 255).astype(np.uint8), "RGBA")
        return np.asarray(img.resize((img.width // f, img.height // f), Image.BOX)).astype(np.float64) / 255

    small = {4: (premul(shrink(orig, 4))[round(y0 / 4):round(y1 / 4), round(x0 / 4):round(x1 / 4)], shrink(bald, 4))}
    best = min((err(dx, dy, s, 4), dx, dy, s) for s in np.arange(0.94, 1.061, 0.03)
               for dy in range(-32, 33, 4) for dx in range(-32, 33, 4))
    best = (err(*best[1:]),) + best[1:]
    for step, ss in ((2, 0.015), (1, 0.005)):
        _, bx, by, bs = best
        best = min([best] + [(err(bx + i * step, by + j * step, bs + k * ss), bx + i * step, by + j * step,
                              bs + k * ss) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
    e, dx, dy, s = best
    return {"dx": dx, "dy": dy, "s": round(float(s), 4), "cx": cx, "cy": cy, "err": round(e, 4)}


def skin_color(bald, marks):
    """頬（目の下・鼻の横）の肌の色の中央値。"""
    w = marks["faceR"][0] - marks["faceL"][0]
    y = round((marks["eyeL"][1] + marks["mouth"][1]) / 2)
    if marks["mouth"][1] - marks["eyeL"][1] < 0.62 * (marks["eyeR"][0] - marks["eyeL"][0]):  # 赤ちゃんは目が大きく下まであるので、目と口の間の真ん中が瞳に当たる（r0*）。口寄りで測る
        y = round((marks["eyeL"][1] + 3 * marks["mouth"][1]) / 4)
    pts = []
    for ex in (marks["eyeL"][0], marks["eyeR"][0]):
        x = round(ex)
        pts.append(bald[y - round(0.03 * w):y + round(0.03 * w), x - round(0.03 * w):x + round(0.03 * w), :3]
                   .reshape(-1, 3))
    return np.median(np.concatenate(pts), axis=0)


def ellipse(shape, center, r):
    rows, cols = np.indices(shape)
    d = ((cols - center[0]) / r[0]) ** 2 + ((rows - center[1]) / r[1]) ** 2
    # 内側は 1、外側の 2 割でなめらかに 0 へ
    return np.clip((1.2 - np.sqrt(d)) / 0.2, 0, 1)


def windows(shape, marks, w, infant=False):
    """目・眉・鼻・口の窓。重なる所は、楕円の中心に一番近い部品だけに入れる（同じ画素を 2 度描かない）。"""
    rows, cols = np.indices(shape)
    dist = {}
    for part, keys in (("eye", ("eyeL", "eyeR")), ("brow", ("browL", "browR")), ("nose", ("nose",)),
                       ("mouth", ("mouth",))):
        rx, ry = marks.get(f"{part}_r", WINDOW_R_INFANT.get(part, WINDOW_R[part]) if infant else WINDOW_R[part])
        ox, oy = WINDOW_OFFSET.get(part, (0, 0))
        dd = []
        for k in keys:
            cx, cy = marks[k][0] + ox * w, marks[k][1] + oy * w
            dd.append(np.sqrt(((cols - cx) / (rx * w)) ** 2 + ((rows - cy) / (ry * w)) ** 2))
        dist[part] = np.minimum.reduce(dd)
    stack = np.stack(list(dist.values()))
    nearest = stack.argmin(axis=0)
    return {part: np.clip((1.15 - d) / 0.15, 0, 1) * (nearest == i) for i, (part, d) in enumerate(dist.items())}


def contour(bald, marks):
    """頭の輪郭線（顔の幅の目印の所を通る暗い線のつながり）を少し太らせた物。"""
    dark = (bald[..., :3].mean(axis=2) < 0.4) & (bald[..., 3] > 0.5)
    lab, _ = ndimage.label(dark)
    ids = set()
    for k in ("faceL", "faceR"):
        x, y = (round(v) for v in marks[k])
        win = lab[y - 4:y + 5, x - 4:x + 5]
        ids |= set(np.unique(win[win > 0]).tolist())
    return ndimage.binary_dilation(np.isin(lab, list(ids)), iterations=3)


def keyed(px, skin):
    """肌の色との差から不透明度を出し、色を肌から戻す（肌の上に重ねると元の色になる）。"""
    d = np.sqrt(((px[..., :3] - skin) ** 2).sum(axis=2))
    a = np.clip((d - KEY_LO) / (KEY_HI - KEY_LO), 0, 1) * px[..., 3]
    with np.errstate(invalid="ignore", divide="ignore"):
        rgb = np.where(a[..., None] > 0.02, (px[..., :3] - (1 - a[..., None]) * skin) / a[..., None], skin)
    return np.concatenate([np.clip(rgb, 0, 1), a[..., None]], axis=2)


def fill(px, hole, sigma):
    """穴（hole が 1 の所）を、周りの色のぼかしで埋める（肌は平塗りなので、これで足りる）。"""
    known = (1 - hole) * px[..., 3] * (px[..., :3].mean(axis=2) > 0.55)  # 線（暗い所）の色は広げない
    out = px.copy()
    num = np.stack([ndimage.gaussian_filter(px[..., c] * known, sigma) for c in range(3)], axis=2)
    den = ndimage.gaussian_filter(known, sigma)[..., None]
    with np.errstate(invalid="ignore", divide="ignore"):
        filled = np.where(den > 1e-4, num / den, 0)
    out[..., :3] = px[..., :3] * (1 - hole[..., None]) + filled * hole[..., None]
    return out


BODY_SPECK = 0.01  # 体の層に残る、顔の幅の二乗のこの割合より小さな離れた塊は消す
WRINKLE_BANDS = ("mid", "senior")  # しわ・ほうれい線の層は 40〜50 代と 60 代以上だけ
WRINKLE_DARK = (0.03, 0.30)   # 肌の明るさとの差がこの範囲の細い線をしわとして取る
WRINKLE_THICK = 3             # これより太い影（鼻筋の影・目のまわりの影）は、しわに入れない（開く処理の回数）
WRINKLE_EDGE = 0.07           # 顔の左右の端からこれだけ（顔の幅に対する割合）内側だけ
WRINKLE_MIN = 80              # しわの線のかけらのうち、これより小さい物（px）は消す


def wrinkle_layer(bald, face, skin, marks, w):
    """髪を消した版から、顔の窓（目・眉・鼻・口）の外の細い薄茶の線（しわ・ほうれい線）を取る。窓の中の線は
    目・眉・鼻・口の部品に入っている（土台の線は埋めてあるので、年代の目印が片側だけに出ていた）。"""
    lum = bald[..., :3].mean(axis=2)
    skin_l = float(np.mean(skin))
    shape = lum.shape
    inside = ndimage.binary_erosion(face, iterations=12)
    cols = np.indices(shape)[1]
    inside &= (cols > marks["faceL"][0] + WRINKLE_EDGE * w) & (cols < marks["faceR"][0] - WRINKLE_EDGE * w)  # 耳のそばの線（片側だけ見えた）は取らない
    win = np.zeros(shape, bool)
    for key, m in windows(shape, marks, w).items():
        win |= m > 0.02
    win = ndimage.binary_dilation(win, iterations=3)
    ink = ndimage.binary_dilation(lum < 0.35, iterations=7)
    white = ndimage.binary_dilation(lum > skin_l + 0.05, iterations=5)
    rgb = bald[..., :3]
    skinny = (rgb[..., 0] >= rgb[..., 1]) & (rgb[..., 1] >= rgb[..., 2])
    shade = skin_l - lum
    m = (shade > WRINKLE_DARK[0]) & (shade < WRINKLE_DARK[1]) & skinny & inside & ~win & ~ink & ~white & (bald[..., 3] > 0.5)
    broad = ndimage.binary_dilation(ndimage.binary_opening(m, iterations=WRINKLE_THICK), iterations=3)
    thin = m & ~broad
    lab, n = ndimage.label(ndimage.binary_dilation(thin, iterations=1))
    sizes = ndimage.sum(thin, lab, range(1, n + 1))
    thin &= np.isin(lab, 1 + np.nonzero(sizes >= WRINKLE_MIN)[0])
    alpha = np.clip(ndimage.gaussian_filter(thin.astype(float), 0.8) * 1.6, 0, 1) * np.clip(shade / 0.12, 0, 1)
    out = np.zeros(shape + (4,))
    out[..., :3] = rgb
    out[..., 3] = alpha
    return out


def extract_one(pid, marks):
    orig = load(RAW / f"{pid}.png")
    bald0 = load(RAW / f"{pid}_bald.png")
    if bald0.shape != orig.shape:
        bald0 = np.asarray(Image.fromarray((bald0 * 255).astype(np.uint8), "RGBA")
                           .resize(orig.shape[1::-1], Image.LANCZOS)).astype(np.float64) / 255
    reg = json.loads((OUT / "register.json").read_text(encoding="utf-8"))[pid]
    bald = shift_image(bald0, reg["dx"], reg["dy"], reg["s"], reg["cx"], reg["cy"])
    shape = orig.shape[:2]
    w = marks["faceR"][0] - marks["faceL"][0]
    skin = skin_color(bald, marks)
    parts = {}
    k = keyed(bald, skin)
    k[..., 3] *= ~contour(bald, marks)  # 頭の輪郭線は土台に残す（目尻が輪郭に近い顔がある）
    holes = np.zeros(shape, bool)
    for name, key in (("eyes", "eye"), ("brows", "brow"), ("nose", "nose"), ("mouth", "mouth")):
        win = windows(shape, marks, w, ages.BAND_OF[pid] == "infant")[key]
        part = k.copy()
        part[..., 3] *= win
        parts[name] = part
        holes |= part[..., 3] > 0.03
    holes = ndimage.binary_dilation(holes, iterations=3)
    holes = np.clip(ndimage.gaussian_filter(holes.astype(float), 1.2) * 2, 0, 1)
    base = fill(bald, holes, sigma=0.03 * w)
    base = fill(base, holes * (base[..., 3] > 0), sigma=0.08 * w)  # 大きな穴の真ん中の残り
    # 窓の外に残った小さな線のかけら（眉の端・鼻筋の点）も肌で埋める（頭の輪郭線は除く）
    d = np.sqrt(((base[..., :3] - skin) ** 2).sum(axis=2))
    lab, n = ndimage.label((d > 0.12) & (base[..., 3] > 0.5) & ~contour(bald, marks))
    sizes = ndimage.sum(np.ones(shape), lab, range(1, n + 1))
    specks = ndimage.binary_dilation(np.isin(lab, 1 + np.nonzero(sizes < SPECK_MAX)[0]), iterations=2)
    base = fill(base, specks.astype(float), sigma=0.02 * w)
    # 土台は頭だけ（首は描き直す: 絵ごとに首の太さが違い、描き直した体の襟と合わなかった）。
    # 耳の下より下は、顔の輪郭線の内側（鼻から塗りつぶして届く所）と輪郭線だけを残す
    rows = np.indices(shape)[0]
    light = (base[..., :3].mean(axis=2) > 0.45) & (base[..., 3] > 0.5)
    lab, _ = ndimage.label(light)
    nx, ny = (round(v) for v in marks["nose"])
    near = lab[ny - 25:ny + 26, nx - 25:nx + 26]  # 鼻の目印が鼻の穴の点（暗い）に当たると背景の塊を顔と取り違える（0 歳の r0*）。近くの最も多い塊にする
    face = lab == (lab[ny, nx] or np.bincount(near[near > 0]).argmax())
    face = ndimage.binary_dilation(face, iterations=FACE_LINE)
    keep = (rows < marks["nose"][1] + EAR_BOTTOM * w) | face
    base[..., 3] *= keep
    if pid not in ages.ADULT_IDS:  # 背景にうっすら残った透明度の点（d40m1 の頭の外 4 万点。透明度 1〜3/255）を消す
        base[..., 3] *= base[..., 3] > 0.04
    if pid in LINE_CLEAN or pid not in ages.ADULT_IDS:  # 20 代の 23 体より後に描いた物（30 代など）も。新しい年代は全部（40〜50 代・60 代以上は、しわを別の層で左右対称に足す）。# 老けて見える線（頬のくぼみ・あごの割れ線・ほうれい線）を、目より下の輪郭の内側で肌に
        inside = ndimage.binary_erosion(face, iterations=LINE_CLEAN_INSET)
        below = rows > max(marks["eyeL"][1], marks["eyeR"][1]) + 0.08 * w
        cols = np.indices(shape)[1]
        below &= (cols > marks["faceL"][0] + 0.03 * w) & (cols < marks["faceR"][0] - 0.03 * w)  # 耳は除く
        d = np.sqrt(((base[..., :3] - skin) ** 2).sum(axis=2))
        lines = ndimage.binary_dilation(inside & below & (d > 0.10), iterations=3) & inside
        base = fill(base, lines.astype(float), sigma=0.03 * w)
    parts["base"] = base
    if ages.BAND_OF[pid] in WRINKLE_BANDS:
        parts["wrinkles"] = wrinkle_layer(bald, face, skin, marks, w)

    # 髪: 元の絵と髪を消した版の差
    d = np.abs(premul(orig) - premul(bald)).max(axis=2)
    d = np.maximum(d, np.abs(orig[..., 3] - bald[..., 3]))
    hair = d > HAIR_DIFF
    hair = ndimage.binary_opening(hair, iterations=1)  # 2 にすると耳に掛かる細い毛束まで消えた
    lab, n = ndimage.label(hair)
    sizes = ndimage.sum(hair, lab, range(1, n + 1))
    hair = np.isin(lab, 1 + np.nonzero(sizes >= HAIR_MIN_PIECE)[0])
    hair = ndimage.binary_closing(hair, iterations=3)
    # 小さな穴だけ埋める（顔を囲む髪で顔まで埋めないよう、大きな穴は残す）
    holes_in = ndimage.binary_fill_holes(hair) & ~hair
    lab, n = ndimage.label(holes_in)
    sizes = ndimage.sum(holes_in, lab, range(1, n + 1))
    hair |= np.isin(lab, 1 + np.nonzero(sizes < 3000)[0])
    hair &= orig[..., 3] > 0.02
    # 髪の色の画素が少ないかけら（耳の線のずれなど）は髪にしない
    hsv_s = np.ptp(orig[..., :3], axis=2) / np.maximum(orig[..., :3].max(axis=2), 1e-3)
    vmax = orig[..., :3].max(axis=2)
    browny = (hsv_s > 0.3) & (vmax > 0.25) & (vmax < 0.82) & (orig[..., 0] > orig[..., 2])
    lab, n = ndimage.label(hair)
    sizes = ndimage.sum(hair, lab, range(1, n + 1))
    brown = ndimage.sum(browny & hair, lab, range(1, n + 1))
    hair = np.isin(lab, 1 + np.nonzero(brown / np.maximum(sizes, 1) > HAIR_COLOR_FRACTION)[0])
    # 髪の範囲の中でも、髪の色の画素と、髪の色のすぐ近くの暗い線だけを残す（毛束の隙間に見える
    # 元の顔の肌・輪郭線・耳・服を髪に入れない。別の顔に載せると、頬の上の線や白い塊に見えた）
    lum = orig[..., :3].mean(axis=2)
    hair &= browny | ((lum < 0.35) & ndimage.binary_dilation(browny, iterations=HAIR_LINE_NEAR))
    hair = ndimage.binary_opening(hair, iterations=1)
    filled = np.zeros_like(hair)
    if pid not in ages.ADULT_IDS:  # 髪の内側の明るいハイライト（雷形。肌色に近く、髪の色の判定から外れる）は、髪の外形を小さく閉じて髪に戻す。穴にすると、下の肌や背景が橙・白の点に見えた
        filled = ndimage.binary_closing(np.pad(hair, HAIR_CLOSE), iterations=HAIR_CLOSE)[HAIR_CLOSE:-HAIR_CLOSE, HAIR_CLOSE:-HAIR_CLOSE] & ~hair
        filled &= ndimage.binary_erosion(ndimage.binary_fill_holes(ndimage.binary_closing(hair, iterations=HAIR_CLOSE)), iterations=HAIR_CLOSE)  # 外形の縁の毛先の間（肌・首が見える所）は埋めない
        hair = hair | filled
        sealed = ndimage.binary_closing(hair, iterations=3)  # 細い隙間でつながった穴（雷形の切れ込み）も囲まれた穴として数える
        holes, nh = ndimage.label(ndimage.binary_fill_holes(sealed) & ~hair)  # 閉じても残る、髪に囲まれた小さな穴（肌色に近いハイライトの中心）も髪に戻す
        small = np.isin(holes, 1 + np.nonzero(ndimage.sum(holes > 0, holes, range(1, nh + 1)) < HAIR_HOLE_MAX)[0]) if nh else np.zeros_like(hair)
        filled |= small
        hair = hair | small
        hole_px = filled & (orig[..., 3] <= 0.5)  # 元の絵そのものが透明に抜けた小さな点（菱形）は、近くの髪の色で埋める
        if hole_px.any():
            orig = orig.copy()
            _, (jy, jx) = ndimage.distance_transform_edt(~(hair & ~hole_px), return_indices=True)
            orig[hole_px, :3] = orig[jy[hole_px], jx[hole_px], :3]
            orig[hole_px, 3] = 1.0
    # 顔の上に掛かる細い毛の線（太さ HAIR_THIN px 未満）は hair_thin に分ける。持ち主の顔では毛に見えるが、
    # 別の顔に載せると頬の傷・ひびに見えた。render_whole は持ち主の顔のときだけ描く
    over = bald[..., 3] > 0.5
    lab, n = ndimage.label(hair)
    sizes = ndimage.sum(hair, lab, range(1, n + 1))
    hair = np.isin(lab, 1 + np.nonzero(sizes >= HAIR_MIN_PIECE)[0])  # 離れた小さな点（256px で黒い粒に見えた）
    thick = ndimage.binary_opening(hair & over, iterations=HAIR_THIN)
    thin = hair & over & ~ndimage.binary_dilation(thick, iterations=HAIR_THIN + 1)
    soft = np.clip(ndimage.gaussian_filter(hair.astype(float), 0.7), 0, 1)
    alpha = soft * ndimage.binary_dilation(hair) * orig[..., 3]
    # 縁の肌の色・明るい色の画素（元の顔・耳との境目のにじみ）は髪に入れない。別の顔では耳の周りの白い線に見えた
    not_skin = ((np.sqrt(((orig[..., :3] - skin) ** 2).sum(axis=2)) > HAIR_SKIN_DIST) & (lum < 0.72)) | filled
    alpha *= not_skin
    hair_px = np.concatenate([orig[..., :3], alpha[..., None]], axis=2)
    thin_soft = np.clip(ndimage.gaussian_filter(thin.astype(float), 0.7), 0, 1) * ndimage.binary_dilation(thin)
    # 前と後ろは、分けてから縁をぼかす（ぼかしてから分けると、持ち主の頭の形の細い線が前髪に残り、
    # 別の頭に載せると耳の周りの線に見えた）
    edge = not_skin * orig[..., 3]

    def soft_of(mask):
        return np.clip(ndimage.gaussian_filter(mask.astype(float), 0.7), 0, 1) * ndimage.binary_dilation(mask) * edge

    front = hair_px.copy()
    front[..., 3] = soft_of(hair & over) * (1 - thin_soft)
    parts["hair_thin"] = hair_px.copy()
    parts["hair_thin"][..., 3] *= over * thin_soft
    back = hair_px.copy()
    # 後ろ髪は頭の線の内側へ SEAM_UNDER px 入れる（前髪の下に隠れる）。境目で前と後ろの縁のぼかしが重なり、
    # 半透明の細い弧が髪の上に見えた。前髪の下全体に暗い色を敷く案は、平らな色の継ぎ目と
    # 前髪の縁の明るい色が髪の上の模様に見えたのでやめた
    back[..., 3] = soft_of(hair & ~ndimage.binary_erosion(over, iterations=SEAM_UNDER))
    # 後ろ髪の頭の形の穴を、頭の内側へ少し塗り広げる（暗い髪の色）。別の人の小さい頭に載せたとき、
    # 穴の縁と頭の間に背景が見えて白い縁取りになった
    hair_back_mask = back[..., 3] > 0.5
    spread = ndimage.binary_dilation(hair_back_mask, iterations=round(BACK_SPREAD * w)) & \
        ndimage.binary_dilation(parts["base"][..., 3] > 0.5, iterations=round(BACK_MARGIN * w)) & \
        ~hair_back_mask & ~(front[..., 3] > 0.5)  # 頭（首を除く）の裏と、その少し外（あごの線の外の隙間）
    dark_hair = np.median(orig[hair & (lum > 0.15) & (lum < 0.35), :3], axis=0)
    back[spread, :3] = dark_hair
    back[spread, 3] = 1
    # 髪の中の閉じた小さな穴を元の絵の色で埋める。元の絵の髪の暗い影は髪の色の判定（browny）から外れ、
    # 透明の穴になって、別の体に載せると背景が白い点に見えた
    # 書き出した PNG の値（0〜255）で判定し、細い隙間で外とつながった穴も閉じて数える
    u = (np.round(front[..., 3] * 255) > 127) | (np.round(back[..., 3] * 255) > 127)
    h = ndimage.binary_fill_holes(ndimage.binary_closing(u, iterations=2)) & ~u
    lab, n = ndimage.label(h)
    sizes = ndimage.sum(h, lab, range(1, n + 1))
    # 埋めるのは暗い髪の色の所だけ（耳・頬の影の肌は明るさ 0.6〜0.7 で、埋めると別の顔の上で薄い点線に見えた）
    hole = np.isin(lab, 1 + np.nonzero(sizes < HAIR_HOLE_MAX)[0]) & (edge > 0.5) & (lum < HAIR_HOLE_LUM)
    for px, where in ((front, hole & over), (back, hole & ~over)):
        px[where, :3] = orig[where, :3]
        px[where, 3] = 1
    # 頭の外の、髪の外形の細い隙間（幅 4px 以下）を、いちばん近い髪の色で後ろ髪に埋める。元の絵でも外側の輪郭線と
    # 毛束の間に透明の点が並び、黒髪では髪の外側の白いふちに見えた
    u = (np.round(front[..., 3] * 255) > 127) | (np.round(back[..., 3] * 255) > 127)
    rim = ndimage.binary_closing(u, iterations=RIM_CLOSE) & ~u & ~over
    _, (iy, ix) = ndimage.distance_transform_edt(~u, return_indices=True)
    color = np.where((front[..., 3] > 0.5)[..., None], front[..., :3], back[..., :3])
    back[rim, :3] = color[iy[rim], ix[rim]]
    back[rim, 3] = 1
    parts["hair_front"] = front
    parts["hair_back"] = back
    hair_base = np.median(orig[hair & (orig[..., :3].mean(axis=2) > 0.25) & (orig[..., :3].mean(axis=2) < 0.6), :3], axis=0)
    meta = {"register": reg, "skin": [round(float(v), 4) for v in skin],
            "hair": [round(float(v), 4) for v in hair_base], "size": list(shape[::-1])}
    return parts, meta


REGISTER = {}  # complete_marks が測った、髪を消した版を元の絵に合わせるずれ（register.json に書く）


def complete_marks(pid, manual):
    """目・眉・鼻先・口の中心（画像を見て手で決めた物）から、頭の上・顔の幅・あご先を髪を消した版で測って足す。"""
    bald = load(RAW / f"{pid}_bald.png")
    dark = (bald[..., :3].mean(axis=2) < 0.38) & (bald[..., 3] > 0.5)
    m = dict(manual)
    ex = (m["eyeL"][0] + m["eyeR"][0]) / 2
    ey = round((m["eyeL"][1] + m["eyeR"][1]) / 2)
    eyed = m["eyeR"][0] - m["eyeL"][0]
    cx = round(ex)
    m["top"] = [cx, int(np.argmax(bald[:, cx, 3] > 0.5))]
    # 顔の幅: 目と鼻先の間の高さで、目の中心から外へ最初の暗い線（耳の付け根の手前の輪郭）
    y = round((ey + m["nose"][1]) / 2)
    if ages.BAND_OF[pid] != "adult":  # 大きな目（子ども）では、目と鼻先の間の高さが目の線に掛かるので、鼻先の高さで測る
        y = round(m["nose"][1])
    x = m["eyeL"][0]
    while not dark[y, x]:
        x -= 1
    m["faceL"] = m.get("faceL", [x, y])  # faceR と同じく、手で決めた faceL があればそれを使う
    x = m["eyeR"][0]
    while not dark[y, x]:
        x += 1
    m["faceR"] = m.get("faceR", [x, y])  # 手で決めた faceR があればそれを使う
    # あご先: 口の下で、口の x の列の一番下の暗い線（首の線より上）
    mx = round(m["mouth"][0])
    col = dark[:, mx - 2:mx + 3].any(axis=1)
    reach = 0.55 if ages.BAND_OF[pid] == "infant" else 0.9  # 乳幼児は口からあごまでが近く、0.9 目幅先まで見ると首の襟の線をあご先と取り違える
    for reach in (reach, 0.9, 1.3):  # 探す幅の中に線が無い（あごが遠い）ときは広げる
        ys = [yy for yy in range(m["mouth"][1] + 40 if reach > 0.6 else m["mouth"][1] + 25, min(col.shape[0], m["mouth"][1] + round(reach * eyed))) if col[yy]]
        if ys:
            break
    else:
        raise ValueError(f"{pid}: あご先の線が見つからない（口の y={m['mouth'][1]}）")
    m["chin"] = m.get("chin", [mx, ys[-1]])  # 手で決めた chin があればそれを使う
    # 目印は髪を消した版で測ったので、元の絵に合わせたときの位置に移す（部品は元の絵の座標で持つ）
    reg = register(load(RAW / f"{pid}.png"), bald, m)
    REGISTER[pid] = reg
    s, c, d = reg["s"], (reg["cx"], reg["cy"]), (reg["dx"], reg["dy"])
    return {k: [round((v[i] - c[i]) * s + c[i] + d[i], 1) for i in (0, 1)] for k, v in m.items()}


def overlay(pid, marks, path):
    from PIL import ImageDraw
    img = Image.open(RAW / f"{pid}_bald.png").convert("RGBA")
    d = ImageDraw.Draw(img)
    for k, (x, y) in marks.items():
        if not isinstance(x, (int, float)):
            continue
        d.ellipse((x - 6, y - 6, x + 6, y + 6), outline=(255, 0, 0, 255), width=3)
        d.text((x + 8, y - 8), k, fill=(255, 0, 0, 255))
    img.crop((250, 100, 1000, 850)).save(path)


NECK_UP = 0.5        # 体の首を、あご先からこれだけ上まで伸ばす（顔の幅に対する割合。細いあごの頭を載せても隙間が出ない）
NECK_PROBE = 0.10    # 首の輪郭線を測る高さ（あご先から下）
NECK_LINE = 0.016    # 伸ばした首の輪郭線の太さ
INK = (0.17, 0.11, 0.09)
MALE_NECK_SLIM = {"knit": 0.76}  # 男の服のうち、首を横に細くする物と倍率（ニットの m1 の首が太かった）
NECK_SLIM_IN = 0.6     # 首を細くする幅（真ん中からの距離。顔の幅に対する割合）。ここまでは同じ倍率
NECK_SLIM_OUT = 1.1     # ここで 1 倍に戻る（肩）
NECK_SLIM_BLEND = 0.12  # 襟のこれだけ上から襟までで 1 倍に戻す
NECK_CLEAN_INSET = 6    # 首の線を消すのは、肌の輪郭線からこれだけ（px）内側
NECK_LINE_DARK = 0.10   # 肌の明るさの中央値よりこれだけ暗い所を線とみなす
NECK_KEEP = 0.12        # 襟のこれだけ上より下の線（鎖骨）は残す（顔の幅に対する割合）
UPPER_FILL = 0.3        # 首の上のほうを肌の平塗りにする高さ（あご先から下。顔の幅に対する割合）
UPPER_FEATHER = 0.1     # 平塗りと元の絵の境目をぼかす幅


def is_skin(px):
    """肌と肌の影の色（橙の色合い）。服（青緑・白）と線（暗い）は入らない。"""
    rgb = px[..., :3]
    mx, mn = rgb.max(axis=2), rgb.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1e-3)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (r >= g) & (g >= b) & (sat > 0.12) & (sat < 0.7) & (mx > 0.45) & (px[..., 3] > 0)


def collar_row(px, cx, chin, w):
    """首の真ん中を下へたどって、肌でなくなる（襟の）高さ。"""
    sk = is_skin(px)
    y = round(chin + 0.2 * w)
    while y < px.shape[0] - 1 and sk[y, cx]:
        y += 1
    return y


def slim_neck(px, cx, chin, w, k):
    """首を横に k 倍へ細くする（ニットの m1 の首が太かった）。真ん中から顔の幅の NECK_SLIM_IN までは k 倍、
    NECK_SLIM_OUT へ向けてなめらかに 1 倍へ戻す。襟の NECK_SLIM_BLEND 上から襟までも 1 倍へ戻す（襟の形は変えない）。"""
    h, wd = px.shape[:2]
    collar = collar_row(px, cx, chin, w)
    rows, cols = np.indices((collar, wd)).astype(float)
    u = np.abs(cols - cx)
    t_x = np.clip((u - NECK_SLIM_IN * w) / ((NECK_SLIM_OUT - NECK_SLIM_IN) * w), 0, 1)
    t_y = np.clip((rows - (collar - NECK_SLIM_BLEND * w)) / (NECK_SLIM_BLEND * w), 0, 1)
    s = 3 * t_x ** 2 - 2 * t_x ** 3
    kk = k + (1 - k) * np.maximum(s, t_y)
    src_x = cx + (cols - cx) / kk
    out = px.copy()
    for c in range(4):
        out[:collar, :, c] = ndimage.map_coordinates(px[..., c], [rows, src_x], order=1, mode="nearest")
    return out


def clean_neck_lines(px, chin, w, cx):
    """首から胸元の肌の内側の線と影を肌で埋める。襟の NECK_KEEP 上より上（首筋）は全部、それより下は縦長の物と、
    横長の物（鎖骨）の大きい 2 本より後の物。"""
    sk = is_skin(px) | ((px[..., :3].mean(axis=2) < 0.45) & (px[..., 3] > 0.5))
    region = ndimage.binary_erosion(ndimage.binary_fill_holes(ndimage.binary_closing(is_skin(px), iterations=4)),
                                    iterations=NECK_CLEAN_INSET)
    region &= np.arange(px.shape[0])[:, None] > chin + 0.05 * w
    lum = px[..., :3].mean(axis=2)
    line = region & sk & (lum < np.median(lum[region & is_skin(px)]) - NECK_LINE_DARK)
    neck = np.arange(px.shape[0])[:, None] < collar_row(px, cx, chin, w) - NECK_KEEP * w
    drop = line & neck
    lab, n = ndimage.label(line & ~neck)
    across = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        piece = lab == i
        if sl[0].stop - sl[0].start > sl[1].stop - sl[1].start:
            drop |= piece
        else:
            across.append((int(piece[sl].sum()), i))
    for _, i in sorted(across, reverse=True)[2:]:
        drop |= lab == i
    drop = ndimage.binary_dilation(drop, iterations=2) & region
    return fill(px, drop.astype(float), sigma=0.03 * w)


def extract_body(pid, name, suffix, marks, base_alpha, skin):
    """服を描き替えた絵から、首・鎖骨・肩・服を取り出す（頭は除き、首は上へ伸ばす）。肌の層と服の層に分ける。"""
    px = load(RAW / f"{pid}_bald_{name}{suffix}.png")
    bald = load(RAW / f"{pid}_bald.png")
    reg = register(bald, px, marks)
    px = shift_image(px, reg["dx"], reg["dy"], reg["s"], reg["cx"], reg["cy"])
    shape = px.shape[:2]
    w = marks["faceR"][0] - marks["faceL"][0]
    cx, chin = round(marks["chin"][0]), marks["chin"][1]
    px[..., 3] *= ~ndimage.binary_dilation(base_alpha > 0.5, iterations=3)
    if ages.BAND_OF[pid] != "adult":  # 頭を除いた後に残る、あごの輪郭の切れ端（首の横に浮く小さな線）を消す。体につながった大きな塊だけ残す
        lab, n = ndimage.label(px[..., 3] > 0.5)
        if n > 1:
            size = ndimage.sum(px[..., 3] > 0.5, lab, range(1, n + 1))
            for i in range(1, n + 1):
                if size[i - 1] < BODY_SPECK * w * w:
                    px[lab == i, 3] = 0
    dark = (px[..., :3].mean(axis=2) < 0.4) & (px[..., 3] > 0.5)
    y0 = round(chin + NECK_PROBE * w)
    xs = []
    for step in (-1, 1):
        x = cx + step * round(0.12 * w)
        while abs(x - cx) < 0.45 * w and not dark[y0, x]:
            x += step
        xs.append(x)
    xl, xr = xs
    lw = max(2, round(NECK_LINE * w))
    for y in range(round(chin - NECK_UP * w), y0):
        row = px[y]
        empty = row[:, 3] < 0.5
        cols = np.arange(shape[1])
        inside = (cols > xl + lw // 2) & (cols < xr - lw // 2) & empty
        row[inside, :3] = skin
        row[inside, 3] = 1
        line = ((abs(cols - xl) <= lw // 2) | (abs(cols - xr) <= lw // 2)) & empty
        row[line, :3] = INK
        row[line, 3] = 1
    # 首の上のほうの影（元の持ち主のあごの形の影）は肌に戻す。あごの下の影は組むときに頭の形から描く
    rows, cols = np.indices(shape)
    # 境目をなめらかに（肌の平塗りと元の絵の影の間に横の線が見えた）
    blend = np.clip((chin + (UPPER_FILL + UPPER_FEATHER / 2) * w - rows) / (UPPER_FEATHER * w), 0, 1)
    upper = (blend > 0) & (cols > xl) & (cols < xr) & is_skin(px)
    px[upper, :3] = px[upper, :3] * (1 - blend[upper, None]) + np.asarray(skin) * blend[upper, None]
    if ages.GENDER_OF[pid] == "male":  # 男の首の線が多く太い首に見えた
        if name in MALE_NECK_SLIM and ages.BAND_OF[pid] == "adult":
            px = slim_neck(px, cx, chin, w, MALE_NECK_SLIM[name])
        px = clean_neck_lines(px, chin, w, cx)
    sk = is_skin(px)
    skin_layer = px.copy()
    skin_layer[..., 3] *= sk
    clothes = px.copy()
    clothes[..., 3] *= ~sk
    rgb = px[..., :3]
    mx, mn = rgb.max(axis=2), rgb.min(axis=2)
    teal = (rgb[..., 2] > rgb[..., 0] + 0.08) & ((mx - mn) / np.maximum(mx, 1e-3) > 0.2) & (px[..., 3] > 0.5)
    return {"skin": skin_layer, "clothes": clothes}, np.median(rgb[teal], axis=0), reg


def main():
    OUT.mkdir(exist_ok=True)
    args = sys.argv[1:]
    only = next((a for a in args if a in ages.BANDS), None)  # 年代を渡すと、その年代の体だけ（ほかの年代の部品は触らない）
    ids = [i for i in IDS if only is None or ages.BAND_OF[i] == only]
    if "marks" in args:
        manual = json.loads((OUT / "landmarks.manual.json").read_text(encoding="utf-8"))
        # 測り終えた体はそのまま（1 体 20 秒ほどかかる）。手で決めた目印を直した体は landmarks.json から消して測り直す
        res = json.loads((OUT / "landmarks.json").read_text(encoding="utf-8")) if (OUT / "landmarks.json").exists() else {}
        res = {k: v for k, v in res.items() if k in IDS}  # 一覧から外した古い元の絵の目印は持ち越さない
        REGISTER.update(json.loads((OUT / "register.json").read_text(encoding="utf-8")) if (OUT / "register.json").exists() else {})
        for pid in ids:
            if pid not in res:
                res[pid] = complete_marks(pid, manual[pid])
        (OUT / "landmarks.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
        (OUT / "register.json").write_text(json.dumps(REGISTER, indent=1) + "\n", encoding="utf-8")
        for pid in ids:
            overlay(pid, res[pid], ROOT / "scratch" / "marks" / f"marks_{pid}.png")
        print(json.dumps(res))
        return
    marks = json.loads((OUT / "landmarks.json").read_text(encoding="utf-8"))
    metas = json.loads((OUT / "meta.json").read_text(encoding="utf-8")) if only and (OUT / "meta.json").exists() else {}
    metas = {k: v for k, v in metas.items() if k in IDS or k == "clothes_base"}
    for pid in ids:
        parts, meta = extract_one(pid, marks[pid])
        d = OUT / pid
        d.mkdir(exist_ok=True)
        for name, px in parts.items():
            save(px, d / f"{name}.png")
        metas[pid] = meta
        print(pid, meta)
    teals = []
    for pid, name, suffix in BODIES:
        if pid not in ids:
            continue
        base_alpha = load(OUT / pid / "base.png")[..., 3]
        layers, teal, reg = extract_body(pid, name, suffix, marks[pid], base_alpha, np.asarray(metas[pid]["skin"]))
        gender = ages.GENDER_OF[pid]
        d = OUT / "bodies" / (f"{gender}_{name}" if ages.BAND_OF[pid] == "adult" else f"{ages.BAND_OF[pid]}_{gender}_{name}")
        d.mkdir(parents=True, exist_ok=True)
        for k, px in layers.items():
            save(px, d / f"{k}.png")
        teals.append(teal)
        print(pid, name, reg)
    if not only:  # 上着の青緑の地の色。年代ごとに回すときは変えない
        metas["clothes_base"] = [round(float(v), 4) for v in np.median(teals, axis=0)]
    (OUT / "meta.json").write_text(json.dumps(metas, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
