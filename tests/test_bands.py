"""年代（群）の試験。20〜30 代の基本の試験は test_render_whole.py。"""
import colorsys
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import render_whole as rw  # noqa: E402

N = len(rw.IDS)


def lum(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def spec_for(j, hair_color, skin):
    g = "male" if j in rw.GENDER_IDS["male"] else "female"
    return {"gender": g, "face": j, "eyes": j, "brows": j, "nose": j, "mouth": j, "hair": j, "hairColor": hair_color,
            "skin": skin, "outfit": rw.OUTFIT_IDS[g][0], "clothesColor": 0, "eyeColor": 0}


# ---- 不具合 1: 眉の色 ----
@pytest.mark.parametrize("j", range(N))
def test_brow_part_keeps_one_stroke_per_side(j):
    # 目の下のまぶたの線・額の細い線の切れ端は眉の部品に入れない（おでこの斜線）
    img = rw.brow_part(j)
    a = rw.arr(img)
    ink = (a[..., 3] > 0.3) & (lum(a[..., :3]) < 0.42)
    mj = rw.marks(j)
    mid = round((mj["browL"][0] + mj["browR"][0]) / 2)
    for side in (ink[:, :mid], ink[:, mid:]):
        _, n = ndimage.label(ndimage.binary_dilation(side, iterations=3))
        assert n == 1, (rw.IDS[j], n)


@pytest.mark.parametrize("j", range(N))
def test_brow_part_has_no_bright_highlight(j):
    # 眉の上の明るい線をやめる: 肌より明るい画素は部品に入れない
    a = rw.arr(rw.brow_part(j))
    skin_l = lum(np.asarray(rw.meta()[rw.IDS[j]]["skin"]))
    bright = (a[..., 3] > 0.3) & (lum(a[..., :3]) > skin_l + 0.02)
    assert int(bright.sum()) == 0, (rw.IDS[j], int(bright.sum()))


@pytest.mark.parametrize("hair_color", range(len(rw.HAIR)))
@pytest.mark.parametrize("skin", range(len(rw.SKIN)))
def test_brow_color_is_dark_brown_and_darker_than_skin(hair_color, skin):
    # 髪色を暗くして彩度を落とした茶系。必ず肌より暗い。緑・オリーブ・橙にならない（金髪×濃い肌、赤毛×濃い肌）
    j = 0
    s = spec_for(j, hair_color, skin)
    a = rw.arr(rw.tinted_brows(rw.brow_part(j), s))
    ink = (a[..., 3] > 0.9) & (lum(a[..., :3]) < 0.42 * lum(np.asarray(rw.SKIN[skin])) / 0.76 + 0.1)
    assert ink.sum() > 200
    px = a[ink][:, :3]
    skin_l = lum(np.asarray(rw.SKIN[skin]))
    assert lum(px).mean() <= 0.55 * skin_l + 1e-6
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    # 緑・青みにならない（白髪の眉は灰がかるので、縁の線の画素のごく一部は許す）
    assert (g > r + 0.03).mean() < 0.01 and (b > g + 0.03).mean() < 0.01
    hsv = np.array([colorsys.rgb_to_hsv(*p) for p in px[lum(px) > 0.08]])
    assert 0.02 <= np.median(hsv[:, 0]) <= 0.10, np.median(hsv[:, 0])  # 中心の色は茶（緑・橙・黄にならない）
    assert np.median(hsv[:, 1]) <= 0.65


# ---- 年代（age_band） ----
import hashlib  # noqa: E402
import json  # noqa: E402

import ages  # noqa: E402

BANDS = rw.AVAILABLE_BANDS if hasattr(rw, "AVAILABLE_BANDS") else None
GOLDEN = json.loads((ROOT / "tests" / "golden_adult.json").read_text(encoding="utf-8")) if (ROOT / "tests" / "golden_adult.json").exists() else {}


def test_available_bands_include_adult():
    assert "adult" in rw.AVAILABLE_BANDS


def test_spec_without_age_band_is_adult():
    s = rw.random_specs(40, 5, age_band="adult")[0]
    t = {k: v for k, v in s.items() if k != "age_band"}
    assert rw.render_png(t, 96) == rw.render_png({**t, "age_band": "adult"}, 96)


def test_unknown_age_band_is_rejected():
    s = {k: v for k, v in rw.random_specs(40, 1, age_band="adult")[0].items()}
    with pytest.raises(rw.SpecError):
        rw.validate({**s, "age_band": "baby"})


@pytest.mark.parametrize("band", [b for b in rw.AVAILABLE_BANDS if b != "adult"])
def test_parts_of_other_band_are_rejected(band):
    # 輪郭・目・眉・鼻・口・髪・服は年代ごと。20〜30 代の部品を別の年代の仕様に入れると弾く
    if band not in rw.AVAILABLE_BANDS:
        pytest.skip("まだ描いていない年代")
    s = rw.random_specs(40, 1, age_band="adult")[0]
    with pytest.raises(rw.SpecError):
        rw.validate({**s, "age_band": band})


@pytest.mark.parametrize("band", ages.BANDS)
def test_random_specs_stay_inside_the_band(band):
    if band not in rw.AVAILABLE_BANDS:
        pytest.skip("まだ描いていない年代")
    for seed in (40, 41, 42):
        for s in rw.random_specs(seed, 20, age_band=band):
            assert s["age_band"] == band
            for k in rw.FACE_KEYS:
                if k != "hair":  # 髪型だけは、ほかの年代の物を借りられる（HAIR_BORROW_FROM）
                    assert ages.BAND_OF[ages.IDS[s[k]]] == band, (band, k, s)
                assert ages.GENDER_OF[ages.IDS[s[k]]] == s["gender"], (band, k, s)
            assert s["hair"] in rw.band_hair_ids(band, s["gender"])
            assert rw.OUTFITS[s["outfit"]]["band"] == band and rw.OUTFITS[s["outfit"]]["gender"] == s["gender"]


def test_random_specs_without_band_draw_from_all_available_bands():
    seen = {s["age_band"] for s in rw.random_specs(40, 60)}
    assert seen == set(rw.AVAILABLE_BANDS)


@pytest.mark.parametrize("band", ages.BANDS)
def test_same_spec_same_bytes_per_band(band):
    if band not in rw.AVAILABLE_BANDS:
        pytest.skip("まだ描いていない年代")
    s = rw.random_specs(40, 3, age_band=band)[1]
    assert rw.render_png(s, 96) == rw.render_png(dict(s), 96)


def test_grey_hair_and_wrinkles_only_for_elders():
    # 白髪・しわ・ほうれい線は 40〜50 代と 60 代以上だけ。濃さも年代で分ける（40〜50 代は弱・60 代以上は中）
    for j, name in enumerate(ages.IDS):
        band = ages.BAND_OF[name]
        level = rw.WRINKLE_LEVEL.get(name, 0)
        assert level == {"mid": 1, "senior": 2}.get(band, 0), (name, level)
    for band in rw.AVAILABLE_BANDS:
        for seed in range(40, 46):
            for s in rw.random_specs(seed, 20, age_band=band):
                if s["hairColor"] in rw.GREY_COLORS:
                    assert band in ("mid", "senior"), (band, s)


@pytest.mark.parametrize("seed", (40, 41, 42))
def test_adult_output_is_frozen(seed):
    # 同じ seed の 20〜30 代は同じ画素（不具合 3 つを直した後の出力で固定した golden。年代を足しても変えない）。PNG の圧縮は環境で変わるので、RGBA の画素のバイト列で固定する
    specs = rw.random_specs(seed, 20, age_band="adult")
    bare = [{k: v for k, v in s.items() if k != "age_band"} for s in specs]  # golden は age_band を足す前の仕様の形
    assert hashlib.sha256(json.dumps(bare, sort_keys=True).encode()).hexdigest() == GOLDEN[str(seed)]["specs_sha"]
    for s, h in zip(specs, GOLDEN[str(seed)]["rgba256"]):
        assert hashlib.sha256(rw.render(s, 256).tobytes()).hexdigest() == h


@pytest.mark.parametrize("band", [b for b in rw.AVAILABLE_BANDS if b != "adult"])
def test_hair_is_not_cut_at_the_top_edge(band):
    # 髪が枠の上の端で切れない。20〜30 代は golden で固定なので対象外
    if band not in rw.AVAILABLE_BANDS:
        pytest.skip("まだ描いていない年代")
    for seed in (40, 41):
        for i, s in enumerate(rw.random_specs(seed, 20, age_band=band)):
            top_row = np.asarray(rw.render(s, 128))[0, :, 3]
            assert not (top_row > 20).any(), (band, seed, i)


# ---- 2 回目: 髪型の共有・白髪・口の種類 ----
def test_hair_pools_follow_the_borrow_table():
    # 髪型は仕様上「年代ごと」に入っていないので、描いた髪を年代をまたいで使える（20〜30 代を 10 代・40〜50 代・子どもへ、など）。
    # ただし薄毛の髪（em1・sm1・sm2）は 40〜50 代と 60 代以上だけ
    thinning = {ages.IDS.index(n) for n in rw.THINNING_HAIR}
    for band in rw.AVAILABLE_BANDS:
        for g in rw.GENDERS:
            pool = set(rw.band_hair_ids(band, g))
            assert set(rw.BAND_IDS[band][g]) <= pool
            allowed = set(rw.BAND_IDS[band][g]) | set(rw.RETIRED_IDS[band][g])  # 古い元の絵の髪型も使える
            for b in rw.HAIR_BORROW_FROM.get(band, ()):
                allowed |= set(rw.BAND_IDS[b][g])
            assert pool <= allowed, (band, g)
            if band not in ("mid", "senior"):
                assert not (pool & thinning), (band, g)


@pytest.mark.parametrize("band", [b for b in rw.AVAILABLE_BANDS if b != "adult"])
def test_random_hair_uses_borrowed_styles(band):
    if band not in rw.AVAILABLE_BANDS:
        pytest.skip("まだ描いていない年代")
    if band not in rw.HAIR_BORROW_FROM:
        pytest.skip("髪型を借りない群（乳幼児は自分の元の絵の髪だけ）")
    own = {i for g in rw.GENDERS for i in rw.BAND_IDS[band][g]}
    used = {s["hair"] for seed in (40, 41, 42) for s in rw.random_specs(seed, 20, age_band=band)}
    assert used - own, band  # 借りた髪型も出る
    assert len(used) >= 10, (band, len(used))


def test_senior_hair_is_grey_only():
    # 60 代以上は地毛の色（茶・黒）が残ると 30〜40 代に見えた。髪色は白髪の 4 色だけ
    for seed in range(40, 46):
        for s in rw.random_specs(seed, 20, age_band="senior"):
            assert s["hairColor"] in rw.GREY_COLORS, s


def test_child_mouths_are_never_flat_or_tiny():
    # 眉が前髪で隠れ口が一直線・短い線だけの子どもは睨んで見えた
    for seed in range(40, 46):
        for s in rw.random_specs(seed, 20, age_band="child"):
            assert s["mouthKind"] in (0, 1, 3), s


@pytest.mark.parametrize("band", [b for b in rw.AVAILABLE_BANDS if b != "adult"])
def test_no_recolored_twins_in_small_pools(band):
    # 同じ年代・性別で、同じ土台・同じ目・同じ髪型の系統の組（同じ人の色違い）を同じ 20 体に出さない
    if band not in rw.AVAILABLE_BANDS:
        pytest.skip("まだ描いていない年代")
    for seed in (40, 41, 42):
        specs = rw.random_specs(seed, 20, age_band=band)
        for i, a in enumerate(specs):
            for b in specs[i + 1:]:
                if a["gender"] != b["gender"]:
                    continue
                same = a["face"] == b["face"] and a["eyes"] == b["eyes"] and \
                    rw.family(rw.HAIR_FAMILY, a["hair"]) == rw.family(rw.HAIR_FAMILY, b["hair"])
                assert not same, (band, seed, a, b)
                assert not (a["face"] == b["face"] and a["hair"] == b["hair"]) or rw.BAND_IDS[band][a["gender"]] and \
                    sum((s["face"], s["hair"]) == (a["face"], a["hair"]) for s in specs if s["gender"] == a["gender"]) <= 2


# ---- 3 回目: 借りた髪の継ぎ目（顔の線・こめかみの棒・首の縦の切り口） ----
import numpy as np  # noqa: E402,F811
from scipy import ndimage  # noqa: E402,F811

NEW_BANDS = [b for b in rw.AVAILABLE_BANDS if b != "adult"]


def _borrowed_specs(band, seeds=(40, 41)):
    return [s for seed in seeds for s in rw.random_specs(seed, 20, age_band=band) if s["hair"] not in rw.BAND_IDS[band][s["gender"]]]


@pytest.mark.parametrize("band", NEW_BANDS)
def test_borrowed_hair_leaves_no_thin_lines_on_the_face(band):
    # 借りた髪の、顔（眉より下の輪郭の内側）に掛かる細い線は、太い毛束につながっていても残さない
    for s in _borrowed_specs(band)[:12]:
        p = rw.placed_parts(s)
        a = rw.arr(p["hair_front"])
        strong = ndimage.binary_dilation(ndimage.binary_opening(a[..., 3] > 0.5, iterations=rw.FACE_STRAND_THIN), iterations=2)
        left = (a[..., 3] > 0.04) & rw.lower_face(s["face"]) & ~strong
        assert int(left.sum()) <= 6, (band, s["face"], s["hair"], int(left.sum()))


@pytest.mark.parametrize("band", NEW_BANDS)
def test_borrowed_hair_does_not_leave_a_skin_bar_at_the_temples(band):
    # 持ち主の頭より外の土台は、目の高さまで消す（眉の高さで切ると、こめかみに四角い肌の棒が見えた）
    for s in _borrowed_specs(band)[:12]:
        p = rw.placed_parts(s)
        mb = rw.marks(s["face"])
        eye_row = int(max(mb["eyeL"][1], mb["eyeR"][1]))
        h = s["hair"]
        anchor_h, anchor_b = (rw.face_cx(h), rw.marks(h)["top"][1]), (rw.face_cx(s["face"]), mb["top"][1])
        owner = rw.arr(rw.moved(rw.part(h, "base"), anchor_h, anchor_b, rw.face_w(s["face"]) / rw.face_w(h),
                                rw.part(s["face"], "base").size))[..., 3] > 0.5
        owner = ndimage.binary_erosion(owner, iterations=rw.CLIP_INSET)
        base = rw.arr(p["base"])[..., 3] > 0.01
        above = np.arange(base.shape[0])[:, None] < eye_row - 2
        assert int((base & ~owner & above).sum()) == 0, (band, s["face"], s["hair"])


@pytest.mark.parametrize("band", ["mid", "senior"])
def test_long_hair_on_a_wide_neck_is_not_cut_by_a_vertical_edge(band):
    # 首の前の髪を切るのは首の肌の上だけ。首の太い体（40〜50 代・60 代以上）で、髪が縦に切られて肌の三角が出た
    long_hair = rw.IDS.index("f0")
    o = rw.BAND_OUTFIT_IDS[band]["female"][-1]
    src = rw.OUTFITS[o]["source"]
    face = rw.BAND_IDS[band]["female"][3]
    s = dict(rw.identity_spec(face), hair=long_hair, hairColor=12 if band == "senior" else 3, skin=1, clothesColor=1, outfit=o, age_band=band)
    lay = rw.placed_in_body(s)
    front = rw.arr(lay["front"])[..., 3] > 0.5
    m, w, cx = rw.marks(src), rw.face_w(src), rw.face_cx(src)
    rows, cols = np.indices(front.shape)
    on_clothes = rw.arr(lay["clothes"])[..., 3] > 0.5
    zone = (rows >= m["chin"][1] + 0.05 * w) & (np.abs(cols - cx) <= rw.NECK_CLEAR * w) & on_clothes
    assert int((front & zone).sum()) > 50, band  # 服の上の髪は残る（縦に切られていない。旗を外すと 0）


# ---- 5 回目: 濃い肌のしわ ----
def _wrinkle_pixels(band, skin, monkeypatch=None):
    s = [x for x in rw.random_specs(40, 20, age_band=band) if x["skin"] == skin][0]
    with_wr = rw.arr(rw.compose_head(s, shadow=False))
    saved, saved_stage = dict(rw.WRINKLE_GAIN), dict(rw.WRINKLE_GAIN_STAGE)
    rw.WRINKLE_GAIN[band] = 0.0
    rw.WRINKLE_GAIN_STAGE.update({st: 0.0 for st in saved_stage})  # 段ごとの強さも 0 に
    try:
        no_wr = rw.arr(rw.compose_head(s, shadow=False))
    finally:
        rw.WRINKLE_GAIN.update(saved)
        rw.WRINKLE_GAIN_STAGE.update(saved_stage)
    d = rw.lum(no_wr[..., :3]) - rw.lum(with_wr[..., :3])
    return int(((d > 0.08) & (no_wr[..., 3] > 0.9)).sum())


@pytest.mark.parametrize("skin", sorted(rw.DARK_SKINS))
def test_wrinkles_are_visible_on_dark_skin(skin):
    # 濃い肌でしわを弱めるとしわが消え、60 代以上が 30〜40 代に見えた。
    # しわの線が肌より明るさで 0.08 以上暗い画素が、一定数ある
    assert _wrinkle_pixels("senior", skin) >= 250


# ---- 5 回目: 古い元の絵・借りた髪の制限・描き足した元の絵 ----
RETIRED_NAMES = [n for names in ages.RETIRED.values() for n in names]


def test_retired_originals_are_not_used_for_face_parts():
    # 40〜50 代の古い元の絵（若く見えた）は、顔・目・眉・鼻・口に使わない。髪型は使ってよい
    retired = {ages.IDS.index(n) for n in RETIRED_NAMES}
    assert retired
    for seed in range(40, 44):
        for s in rw.random_specs(seed, 20, age_band="mid"):
            for k in ("face", "eyes", "brows", "nose", "mouth"):
                assert s[k] not in retired, (seed, k, ages.IDS[s[k]])


@pytest.mark.parametrize("band", [b for b in rw.AVAILABLE_BANDS if b != "adult"])
def test_m8_hair_is_not_borrowed_by_other_ages(band):
    # m8 のもみあげは、顔の幅が違うとこめかみで平らな塊になる
    m8 = ages.IDS.index("m8")
    for seed in range(40, 44):
        assert all(s["hair"] != m8 for s in rw.random_specs(seed, 20, age_band=band))


def test_added_originals_are_in_their_bands():
    for n in ("em4", "em7", "ef4", "ef7", "tm4", "tm6", "sm4", "sm6"):
        if n in ages.IDS:
            assert ages.BAND_OF[n] == {"e": "mid", "t": "teen", "s": "senior", "c": "child"}[n[0]]
            assert rw.part(ages.IDS.index(n), "base").size[0] > 0
