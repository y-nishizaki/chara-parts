"""変な絵（背景の抜け・顔と髪のずれ・頭の大きさの外れ・離れた断片）を、描いた絵から機械で数える。
python3 lint_faces.py [<1 段あたりの体数>=20] [<seed>=1]   全段の乱数の絵を描いて、外れた体の番号と理由を出す。
生成器へは random_specs(..., ok=lint_faces.ok) で引き直しを頼む（clean_specs）。閾値は「直しを外した絵」と「今の絵」を分けるように合わせた（完全には分かれない）。"""
from __future__ import annotations

import sys

import numpy as np
from scipy import ndimage

import render_whole as rw

DOWN = 4            # 体の持ち主の元の絵（1024px 前後）を 1/4 にして数える
GAP_MAX = 0.0095    # 頭のそばの細い隙間の最大の塊（顔の幅の 2 乗に対する割合）。「直しを外した絵」と「今の絵」で合わせた
OFFSET_MAX = 0.10   # 頭と髪の上の部分の中心の差（顔の幅に対する割合）
FRAGMENT_AREA = 40  # 離れた断片とみなす最小の面積（1/4 にした絵の px。6〜14px の毛先の飛び毛は元の絵どおりで、断片にしない）
SIZE_MAD = 4.0      # 段の中央値からの外れ（MAD の倍数）。lint_stage だけ


def _down(a: np.ndarray) -> np.ndarray:
    h, w = (n // DOWN for n in a.shape)
    return a[:h * DOWN, :w * DOWN].reshape(h, DOWN, w, DOWN).mean(axis=(1, 3))


def hole_pixels(solid: np.ndarray) -> int:
    return int((ndimage.binary_fill_holes(solid) & ~solid).sum())


def notch_pixels(solid: np.ndarray, region: np.ndarray | None = None, r: int = 2) -> int:
    """幅 2r px 以下の細い隙間（囲まれていない切れ込み）。囲まれた穴は hole_pixels が数えるので除く。"""
    closed = ndimage.binary_closing(np.pad(solid, r), iterations=r)[r:-r, r:-r]
    gap = closed & ~solid & ~(ndimage.binary_fill_holes(solid) & ~solid)
    return int((gap if region is None else gap & region).sum())


def fragments(solid: np.ndarray, min_area: int = FRAGMENT_AREA) -> int:
    """いちばん大きい塊のほかの、min_area 以上の離れた塊の数。"""
    lab, n = ndimage.label(solid)
    if n < 2:
        return 0
    sizes = np.sort(ndimage.sum(solid, lab, range(1, n + 1)))
    return int((sizes[:-1] >= min_area).sum())


def _center_x(alpha: np.ndarray, rows: slice) -> float | None:
    cols = np.nonzero((alpha[rows] > 0.5).any(axis=0))[0]
    return None if cols.size == 0 else (cols.min() + cols.max()) / 2


def hair_offset(lay: dict, src: int) -> float:
    """頭（土台）と髪の、上の部分の中心の差。髪の上端から髪の高さの 4 割までの行で比べる（ポニーテールなど下の偏りは見ない）。"""
    hair = np.maximum(rw.arr(lay["back"])[..., 3], rw.arr(lay["front"])[..., 3])
    ys = np.nonzero((hair > 0.5).any(axis=1))[0]
    head = rw.arr(lay["head"])[..., 3]
    hy = np.nonzero((head > 0.5).any(axis=1))[0]
    if ys.size == 0 or hy.size == 0:
        return 0.0
    rows = slice(ys.min(), ys.min() + max(1, round(0.4 * (hy.max() - ys.min()))))
    hc, bc = _center_x(hair, rows), _center_x(head, rows)
    return 0.0 if hc is None or bc is None else abs(hc - bc) / rw.face_w(src)


def head_gap(canvas, lay: dict, src: int) -> float:
    """頭のまわり（頭の輪郭から顔の幅の 2 割、あごより上）の、幅が顔の幅の 4.5% 以下の細い隙間のいちばん大きい塊（頭と髪の間の白い抜け）。"""
    solid = _down(rw.arr(canvas)[..., 3]) > 0.5
    head = _down(rw.arr(lay["head"])[..., 3]) > 0.5
    w, chin = rw.face_w(src) / DOWN, rw.marks(src)["chin"][1] / DOWN
    r = max(1, round(rw.EAR_GAP_CLOSE * w))
    closed = ndimage.binary_closing(np.pad(solid, r), iterations=r)[r:-r, r:-r]
    ring = ndimage.binary_dilation(head, iterations=round(0.2 * w)) & (np.arange(solid.shape[0])[:, None] < chin)
    lab, n = ndimage.label(closed & ~solid & ring)
    return max(ndimage.sum(lab > 0, lab, range(1, n + 1)), default=0) / (w * w)


def analyze(spec: dict, lay: dict | None = None) -> tuple[list[str], tuple[float, float]]:
    """1 体の仕様を描いて、外れた理由（無ければ空）と、頭の大きさの手がかり（髪の上端からあご先までの高さの、顔の幅に対する比）を返す。"""
    lay = lay or rw.placed_in_body(spec)
    src = rw.OUTFITS[spec["outfit"]]["source"]
    canvas = rw.compose_full(spec, lay)
    solid = rw.arr(canvas)[..., 3] > 0.5
    why = []
    gap = head_gap(canvas, lay, src)
    if gap > GAP_MAX:
        why.append(f"頭のそばの白い隙間 {gap:.4f}")
    off = hair_offset(lay, src)
    if off > OFFSET_MAX:
        why.append(f"顔と髪の中心のずれ {off:.2f}")
    if fragments(_down(rw.arr(canvas)[..., 3]) > 0.5):
        why.append("離れた断片")
    w, m = rw.face_w(src), rw.marks(src)
    ys = np.nonzero(solid.any(axis=1))[0]
    xs = np.nonzero(solid[round(m["eyeL"][1])])[0]
    return why, ((m["chin"][1] - ys.min()) / w, (xs.max() - xs.min()) / w if xs.size else 0.0)


def lint(spec: dict, lay: dict | None = None) -> list[str]:
    return analyze(spec, lay)[0]


def ok(spec: dict) -> bool:
    return not lint(spec)


def clean_specs(seed: int, count: int, age_band: str | None = None) -> list[dict]:
    """lint で外れた組を引き直す random_specs。"""
    return rw.random_specs(seed, count, age_band=age_band, ok=ok)


def lint_stage(specs: list[dict]) -> dict[int, list[str]]:
    """同じ段の体の中で、頭の大きさが中央値から大きく外れた体（と、各体の lint の理由）。"""
    res = [analyze(s) for s in specs]
    bad = {i: r[0] for i, r in enumerate(res)}
    f = np.array([r[1] for r in res])
    med = np.median(f, axis=0)
    mad = np.maximum(np.median(np.abs(f - med), axis=0), 0.04)
    for i, row in enumerate(f):
        z = np.abs(row - med) / mad
        if z.max() > SIZE_MAD:
            bad[i] = bad[i] + [f"頭の大きさ・位置が段の中央値から外れ（頭の高さ {row[0]:.2f}、中央値 {med[0]:.2f}）"]
    return {i: r for i, r in bad.items() if r}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    total = 0
    for stage in rw.STAGES_AVAILABLE:
        specs = rw.random_specs(seed, n, age_band=stage) if n <= 20 else sum((rw.random_specs(seed + k, 20, age_band=stage) for k in range((n + 19) // 20)), [])[:n]
        for i, r in lint_stage(specs).items():
            total += 1
            print(f"{stage} #{i}: {'; '.join(r)}")
    print(f"外れた体: {total}")
