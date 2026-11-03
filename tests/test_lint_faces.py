"""lint_faces: 変な絵（背景の抜け・顔と髪のずれ・頭の大きさの外れ・離れた断片）を機械で捕まえる。"""
import numpy as np
from PIL import ImageChops

import lint_faces as lf
import render_whole as rw


def spec_at(stage, n=0, seed=7):
    return rw.random_specs(seed, 20, age_band=stage)[n]


def test_enclosed_hole_and_fragment_are_counted():
    a = np.zeros((100, 100), bool)
    a[20:80, 20:80] = True
    assert lf.hole_pixels(a) == 0 and lf.fragments(a) == 0
    a[40:44, 40:44] = False  # 髪に囲まれた小さな穴（背景が見える）
    a[3:13, 3:13] = True  # 離れた小さな断片（面積 40px 以上）
    assert lf.hole_pixels(a) == 16
    assert lf.fragments(a) == 1


def test_narrow_notch_is_counted_as_a_gap():
    a = np.zeros((100, 100), bool)
    a[20:80, 20:80] = True
    a[20:50, 49:51] = False  # 上から入る細い切れ込み（頭のそばの隙間）
    assert lf.hole_pixels(a) == 0
    assert lf.notch_pixels(a) > 0


def test_a_clean_render_passes_and_a_shifted_hair_fails():
    s = spec_at("0y", 3)
    lay = rw.placed_in_body(s)
    assert lf.lint(s, lay) == []
    for k in ("back", "front"):  # 髪だけを横へ大きくずらす（顔と髪のずれ）
        lay[k] = ImageChops.offset(lay[k], int(0.25 * rw.face_w(rw.OUTFITS[s["outfit"]]["source"])), 0)
    assert any("ずれ" in r for r in lf.lint(s, lay))


def test_clean_specs_of_every_stage_pass_lint():
    bad = {}
    for stage in rw.STAGES_AVAILABLE:
        for i, s in enumerate(lf.clean_specs(11, 4, age_band=stage)):
            r = lf.lint(s)
            if r:
                bad[(stage, i)] = r
    assert not bad, bad


def test_generator_redraws_specs_that_fail_lint():
    seen = []

    def ok(spec):
        seen.append(spec["hair"])
        return len(seen) > 5  # 最初の 5 本は不合格
    specs = rw.random_specs(3, 3, age_band="30s", ok=ok)
    assert len(specs) == 3 and len(seen) > 5
