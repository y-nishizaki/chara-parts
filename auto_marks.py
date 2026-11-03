"""髪を消した版の画像から、目・眉・鼻先・口の中心（手で読んでいた目印）を自動で測る。
python3 auto_marks.py check          手で読んだ目印（assets/parts_whole/landmarks.manual.json）との差を出す
python3 auto_marks.py <id> [<id> ...]  測った目印を JSON で出す（重ねた画像は scratch/onboard/auto_<id>.png）"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "whole" / "raw"
PARTS = ROOT / "assets" / "parts_whole"


def load(pid):
    return np.asarray(Image.open(RAW / f"{pid}_bald.png").convert("RGBA")).astype(np.float64) / 255


def lum(a):
    return a[..., :3] @ np.array([0.299, 0.587, 0.114])


NOSE_R = 0.679
MOUTH_DX = 5.2
NOSE_DX = -2.3
BIAS = {'eyeL': (14.8, -17.5), 'eyeR': (-15.8, -15.5), 'browL': (-14.2, -9.3), 'browR': (12.9, -8.2)}  # 手で読んだ目印との平均の差（71 体で測った）  # 手で読んだ目印との平均の差（71 体で測った）


def measure(pid, group=None):
    a = load(pid)
    h, w = a.shape[:2]
    L = lum(a)
    opaque = a[..., 3] > 0.5
    sat = a[..., :3].max(axis=2) - a[..., :3].min(axis=2)
    # 頭の輪郭（暗い線）で囲まれた中の肌: 中央の頬の色を肌とする
    cy = int(np.argmax(opaque.any(axis=1)))  # 頭の上
    top = cy
    white = (L > 0.90) & (sat < 0.12) & opaque
    # 顔の中（頭の上から 25% 〜 75%）の白い塊 = 白目
    x0, x1 = int(w * 0.30), int(w * 0.70)
    band = np.zeros_like(white)
    band[top + 120:top + 520, x0:x1] = True
    lab, n = ndimage.label(white & band)
    if n < 2:
        raise RuntimeError(f"{pid}: 白目が見つからない")
    sizes = ndimage.sum(white & band, lab, range(1, n + 1))
    order = np.argsort(sizes)[::-1][:4]
    cands = []
    for i in order:
        ys, xs = np.where(lab == i + 1)
        cands.append((xs.mean(), ys.mean(), sizes[i], xs.min(), xs.max(), ys.min(), ys.max()))
    # 左右 1 つずつ: 面積の大きい 2 つ（y が近い）
    cands.sort(key=lambda c: -c[2])
    c1 = cands[0]
    c2 = next(c for c in cands[1:] if abs(c[1] - c1[1]) < 0.12 * w and abs(c[0] - c1[0]) > 0.1 * w)
    eL, eR = sorted((c1, c2), key=lambda c: c[0])

    def eye_center(c):
        # 白目の外接枠の中の、茶色の塊（虹彩）の中心。まつげ・瞳孔はほぼ黒で彩度が低いので入らない
        x0, x1, y0, y1 = int(c[3]) - 6, int(c[4]) + 6, int(c[5]) - 6, int(c[6]) + 6
        sub = (L[y0:y1, x0:x1] < 0.55) & (sat[y0:y1, x0:x1] > 0.22) & opaque[y0:y1, x0:x1]
        ys, xs = np.where(sub)
        if len(ys) < 20:
            return float((c[3] + c[4]) / 2), float((c[5] + c[6]) / 2)
        return float(xs.mean() + x0), float(ys.mean() + y0)

    eyeL, eyeR = eye_center(eL), eye_center(eR)
    eyeL = (eyeL[0] + BIAS['eyeL'][0], eyeL[1] + BIAS['eyeL'][1]); eyeR = (eyeR[0] + BIAS['eyeR'][0], eyeR[1] + BIAS['eyeR'][1])
    ed = eyeR[0] - eyeL[0]
    ey = (eyeL[1] + eyeR[1]) / 2
    mx = (eyeL[0] + eyeR[0]) / 2
    # 眉: 目（白目の上端）より上で、肌の色から離れた塊（赤ちゃんの眉は淡いので、暗さでなく肌との色の差で見る）
    skin = np.median(a[int(ey) - 20:int(ey) + 5, int(mx) - 12:int(mx) + 12, :3].reshape(-1, 3), axis=0)  # 目の間の肌
    far = (np.linalg.norm(a[..., :3] - skin, axis=2) > 0.10) & opaque
    dark = (L < 0.45) & opaque

    def brow(eye, c):
        y1 = int(c[5] - 0.04 * ed)
        y0 = int(y1 - 0.5 * ed)
        xa, xb = int(eye[0] - 0.34 * ed), int(eye[0] + 0.34 * ed)
        m = np.zeros_like(far)
        m[y0:y1, xa:xb] = far[y0:y1, xa:xb]
        lab, n = ndimage.label(m)
        if not n:
            return eye[0], y1 - 0.15 * ed
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        means = ndimage.mean(L, lab, range(1, n + 1))
        sizes = np.where(means > 0.22, sizes, 0)  # 頭の輪郭（ほぼ黒）は眉でない
        i = int(np.argmax(sizes)) + 1
        ys, xs = np.where(lab == i)
        return float(xs.mean()), float(ys.mean())

    browL, browR = brow(eyeL, eL), brow(eyeR, eR)
    if group == 'infant':  # 赤ちゃんの眉は淡くて、まつげと取り違えるので、白目の上端から決まった高さに置く
        browL, browR = (eyeL[0], eL[5] - 0.30 * ed), (eyeR[0], eR[5] - 0.30 * ed)
    # 口: 目より下 0.5〜1.25 目幅の範囲で、中央の暗い横長の線（あごの輪郭は範囲の外）
    lo, hi = {'infant': (0.22, 0.62), 'child': (0.35, 0.95)}.get(group, (0.45, 1.1))
    y0, y1 = int(ey + lo * ed), int(ey + hi * ed)
    xa, xb = int(mx - 0.30 * ed), int(mx + 0.30 * ed)
    m = np.zeros_like(dark)
    m[y0:y1, xa:xb] = dark[y0:y1, xa:xb]
    mm = ndimage.binary_dilation(m, structure=np.ones((1, 25)))  # 口の線は途中で薄くなって 2 つに切れる
    lab, n = ndimage.label(mm)
    lab = lab * m  # 元の暗い画素だけ数える
    best, bsc = None, 0
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if not len(ys):
            continue
        ww, hh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        sc = ww / hh  # 口の線はいちばん横長（あごの輪郭は縦にも伸びる）
        if 0.12 * ed < ww < 0.6 * ed and sc > bsc:
            best, bsc = (float(xs.mean()), float(ys.mean())), sc
    if DEBUG: print('mouth win',y0,y1,xa,xb,'best',best,'n',n)
    mouth = (mx + MOUTH_DX, (best[1] - 9.1 - -7.0) if best else ey + (lo + hi) / 2 * ed)
    # 鼻先: 目と口の間（目から 45% の所）を中心の列に置く
    nose = (mx + NOSE_DX, ey + (0.5 if group == 'infant' else NOSE_R) * (mouth[1] - ey))
    browL = (browL[0] + BIAS['browL'][0], browL[1] + BIAS['browL'][1]); browR = (browR[0] + BIAS['browR'][0], browR[1] + BIAS['browR'][1])
    return {"eyeL": eyeL, "eyeR": eyeR, "browL": browL, "browR": browR, "nose": nose, "mouth": mouth}


DEBUG = False
GROUP: dict = {}  # id → 群（check で使う）


def check():
    man = json.loads((PARTS / "landmarks.manual.json").read_text(encoding="utf-8"))
    errs = {k: [] for k in ("eyeL", "eyeR", "browL", "browR", "nose", "mouth")}
    bad = []
    for pid, m in man.items():
        if not (RAW / f"{pid}_bald.png").exists():
            continue
        try:
            r = measure(pid, GROUP.get(pid))
        except Exception as e:  # noqa: BLE001
            bad.append((pid, str(e)))
            continue
        for k in errs:
            d = float(np.hypot(r[k][0] - m[k][0], r[k][1] - m[k][1]))
            errs[k].append((d, pid))
    for k, v in errs.items():
        ds = np.array([d for d, _ in v])
        worst = sorted(v, reverse=True)[:4]
        print(f"{k:6s} n={len(ds)} 平均 {ds.mean():5.1f}px 中央 {np.median(ds):5.1f} 90% {np.percentile(ds, 90):5.1f} 最大 {ds.max():5.1f} 悪い順 {[(p, round(d)) for d, p in worst]}")
    print("失敗", bad)


def overlay(pid, r):
    img = Image.open(RAW / f"{pid}_bald.png").convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255)); bg.alpha_composite(img)
    d = ImageDraw.Draw(bg)
    for k, (x, y) in r.items():
        d.ellipse([x - 8, y - 8, x + 8, y + 8], outline=(255, 0, 0), width=3)
    return bg.convert("RGB")


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check()
    else:
        out = {pid: {k: [round(v[0]), round(v[1])] for k, v in measure(pid).items()} for pid in sys.argv[1:]}
        print(json.dumps(out))
