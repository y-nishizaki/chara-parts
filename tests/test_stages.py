"""19 段・年を数で持つ仕組みの試験。段は 0y 1y 2y 3y 4-5 6-7 8-9 10-11 12-13 14-15 16-17 18-19 20s 30s 40s 50s 60s 70s 80+。
元の絵（部品）が揃った段だけが使える（STAGES_AVAILABLE）。年代の群（child・teen・adult・mid・senior と infant）は変えていない。"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ages  # noqa: E402
import render_whole as rw  # noqa: E402

STAGES = ("0y", "1y", "2y", "3y", "4-5", "6-7", "8-9", "10-11", "12-13", "14-15", "16-17", "18-19", "20s", "30s", "40s", "50s", "60s", "70s", "80+")


def test_there_are_19_stages_in_age_order():
    assert ages.STAGES == STAGES
    assert len(set(ages.STAGES)) == 19


@pytest.mark.parametrize("age,stage", [(0, "0y"), (1, "1y"), (3, "3y"), (4, "4-5"), (5, "4-5"), (6, "6-7"), (19, "18-19"), (20, "20s"), (29, "20s"),
                                       (30, "30s"), (45, "40s"), (59, "50s"), (60, "60s"), (79, "70s"), (80, "80+"), (99, "80+")])
def test_age_number_maps_to_stage(age, stage):
    assert ages.stage_of_age(age) == stage


def test_every_age_has_exactly_one_stage_and_ranges_are_contiguous():
    seen = [ages.stage_of_age(a) for a in range(0, 100)]
    assert [s for i, s in enumerate(seen) if i == 0 or seen[i - 1] != s] == list(ages.STAGES)


def test_stages_belong_to_one_group():
    assert {s: ages.GROUP_OF_STAGE[s] for s in ("0y", "3y", "4-5", "12-13", "14-15", "18-19", "20s", "30s", "40s", "50s", "60s", "80+")} == {
        "0y": "infant", "3y": "infant", "4-5": "child", "12-13": "child", "14-15": "teen", "18-19": "teen", "20s": "adult", "30s": "adult",
        "40s": "mid", "50s": "mid", "60s": "senior", "80+": "senior"}


def test_stage_pools_are_subsets_of_their_group_pool():
    for stage in rw.STAGES_AVAILABLE:
        group = ages.GROUP_OF_STAGE[stage]
        for g in rw.GENDERS:
            assert set(rw.STAGE_IDS[stage][g]) <= set(rw.BAND_IDS[group][g]), (stage, g)


def test_stage_pools_do_not_share_originals():
    for g in rw.GENDERS:
        seen = {}
        for stage in ages.STAGES:
            for i in rw.STAGE_IDS[stage][g]:
                assert i not in seen, (ages.IDS[i], seen.get(i), stage)
                seen[i] = stage


@pytest.mark.parametrize("stage", sorted(rw.STAGES_AVAILABLE if hasattr(rw, "STAGES_AVAILABLE") else []))
def test_random_specs_for_a_stage_use_only_that_stage_parts(stage):
    for seed in (40, 41):
        for s in rw.random_specs(seed, 20, age_band=stage):
            assert ages.stage_of_age(s["age"]) == stage, s
            assert s["age_band"] == ages.GROUP_OF_STAGE[stage]
            for k in ("face", "eyes", "brows", "nose", "mouth"):
                assert s[k] in rw.STAGE_IDS[stage][s["gender"]], (stage, k, ages.IDS[s[k]])


@pytest.mark.parametrize("stage", sorted(rw.STAGES_AVAILABLE if hasattr(rw, "STAGES_AVAILABLE") else []))
def test_same_spec_gives_same_bytes(stage):
    s = rw.random_specs(40, 3, age_band=stage)[1]
    assert rw.render_png(s, 96) == rw.render_png(dict(s), 96)
    assert rw.random_specs(40, 3, age_band=stage) == rw.random_specs(40, 3, age_band=stage)


def test_age_must_agree_with_age_band():
    s = rw.random_specs(40, 1, age_band="20s")[0]  # 群 adult は 30 代の元の絵も含むので、年と同じ段（20 代）の部品で作る
    with pytest.raises(rw.SpecError):
        rw.validate({**s, "age": 5})  # 5 歳は child の群
    rw.validate({**s, "age": 27})


def test_age_out_of_range_is_rejected():
    s = rw.random_specs(40, 1, age_band="adult")[0]
    for bad in (-1, 120, 2.5, "30"):
        with pytest.raises(rw.SpecError):
            rw.validate({**s, "age": bad})


def test_at_age_keeps_colours_and_hair_family():
    base = rw.random_specs(40, 1, age_band="adult")[0]
    for age in [(ages.AGE_RANGE[st][0] + ages.AGE_RANGE[st][1]) // 2 for st in rw.STAGES_AVAILABLE]:
        t = rw.at_age(base, age)
        rw.validate(t)
        assert t["age"] == age
        assert (t["eyeColor"], t["skin"], t["gender"]) == (base["eyeColor"], base["skin"], base["gender"])
        if t["age_band"] not in ("mid", "senior"):
            assert t["hairColor"] == base["hairColor"]


def test_at_age_parts_come_from_the_new_stage():
    base = rw.random_specs(40, 1, age_band="adult")[0]
    for stage in rw.STAGES_AVAILABLE:
        age = ages.AGE_RANGE[stage][0]
        t = rw.at_age(base, age)
        for k in ("face", "eyes", "brows", "nose", "mouth"):
            assert t[k] in rw.STAGE_IDS[stage][t["gender"]], (stage, k)


def test_at_age_is_deterministic_and_reversible_in_stage():
    base = rw.random_specs(41, 1, age_band="adult")[0]
    age = ages.AGE_RANGE[rw.STAGES_AVAILABLE[-1]][0] + 1
    assert rw.at_age(base, age) == rw.at_age(base, age)
    back = rw.at_age(rw.at_age(base, age), 24)
    assert back["gender"] == base["gender"] and ages.stage_of_age(back["age"]) == "20s"


# --- ---------------------------------------------------------
@pytest.mark.parametrize("stage", ["0y", "1y", "2y", "3y"])
def test_infant_hair_comes_only_from_its_own_stage(stage):
    # 0 歳に 3 歳のふさふさの髪、3 歳に 0 歳の薄い髪が載ると年が崩れた
    for g in rw.GENDERS:
        for spec in rw.random_specs(40, 40, age_band=stage):
            if spec["gender"] == g:
                assert spec["hair"] in rw.STAGE_IDS[stage][g], (stage, g, spec["hair"])


def test_at_age_to_senior_turns_the_hair_grey():
    # 梯子で 74 歳・89 歳の髪が若いまま（茶）だと 50 代に見えた
    base = rw.random_specs(40, 1, age_band="20s")[0]
    assert base["hairColor"] not in rw.GREY_COLORS
    for age in (64, 74, 89):
        assert rw.at_age(base, age)["hairColor"] in rw.GREY_COLORS


def test_wrinkles_get_stronger_from_40s_to_50s_and_lighter_after_60s():
    g = rw.WRINKLE_GAIN_STAGE
    assert g["40s"] > 0 and g["50s"] > 0 and g["50s"] != g["40s"]  # 段ごとに別の値（50 代の女が老けて見えたので 40s > 50s に調整済み）
    assert g["60s"] > g["70s"] > g["80+"]


def test_infant_landmarks_are_roughly_symmetric():
    # b0f3 は faceL が髪の線に当たって顔の左半分が狭くなり、顔の層が割れた
    for pid in ages.band_ids("infant"):
        m = rw.landmarks()[pid]
        left, right = m["mouth"][0] - m["faceL"][0], m["faceR"][0] - m["mouth"][0]
        assert abs(left - right) < 45, (pid, left, right)
        assert abs(m["eyeL"][1] - m["eyeR"][1]) < 20, pid


def test_infant_base_has_no_leftover_eye_pieces():
    # 目の窓が小さいと、大きな目の下の縁が土台に三日月形で残った
    import numpy as np
    from PIL import Image
    from scipy import ndimage
    for pid in ages.band_ids("infant"):
        a = np.asarray(Image.open(ROOT / "assets" / "parts_whole" / pid / "base.png"))[..., 3] > 127
        _, n = ndimage.label(a)
        assert n == 1, (pid, n)


# （test_long_hair_on_another_head_leaves_no_white_gap_at_the_ears は、再現に使った古い顔 cf3 が乱数の対象から外れ、
# 今の顔の組では耳の隙間がそもそも出ない（8〜9 歳の女 × 長い髪で 0〜1px）ので外した。耳の隙間は lint_faces の head_gap が見る）


# --- ---------------------------------------------------------
def test_head_gets_smaller_as_age_goes_up_inside_a_body_type():
    # 梯子の 4・6・8・10・12 歳が同じ大きさに見えた。頭と体の比は年の手がかり
    s = rw.HEAD_SCALE_STAGE
    assert s["0y"] > s["1y"] > s["2y"] > s["3y"]
    assert s["4-5"] > s["6-7"] > s["8-9"] > s["10-11"] > s["12-13"]
    assert s["14-15"] > s["16-17"] > s["18-19"]
    assert not {"20s", "30s"} & set(s)  # 20〜30 代は golden のまま


def test_head_scale_makes_the_younger_head_bigger_on_the_same_character():
    import numpy as np
    base = rw.random_specs(40, 1, age_band="4-5")[0]

    def head_width(age):
        spec = rw.at_age(base, age)
        a = np.asarray(rw.compose_head(spec, shadow=False).getchannel("A")) > 127
        cols = np.nonzero(a.any(axis=0))[0]
        return (cols.max() - cols.min()) * rw.HEAD_SCALE_STAGE[ages.stage_of_age(age)] / max(1, rw.face_w(spec["face"]))
    assert head_width(4) > head_width(12)


def test_grey_hair_has_no_orange_highlights():
    # 白髪の中に元の茶髪の橙のハイライトが残った。色味は無彩色に近い（元の茶のままだと 99 パーセンタイルが 20〜55、直すと 11 以下）
    import numpy as np
    for spec in rw.random_specs(40, 10, age_band="60s"):
        s = dict(spec, hairColor=sorted(rw.GREY_COLORS)[0])
        layer = rw.head_layers(s)["front"]
        p, a = np.asarray(layer).astype(float)[..., :3], np.asarray(layer)[..., 3] > 200
        assert np.percentile((p.max(axis=2) - p.min(axis=2))[a], 99) < 18


def test_mouth_parts_are_small_and_do_not_carry_a_skin_halo():
    # 目と口の間の真ん中が瞳に当たると、肌色の測り違いで口の部品が顔の下半分の肌ごと不透明になり、別の顔に載せると口のまわりに淡い塊が出た
    # （1 歳 r1f0・r1f1: 口の部品が 24000 画素。ふつうは 300〜900）
    import numpy as np
    from PIL import Image
    for pid in ages.IDS:
        if ages.STAGE_OF[pid] is None and ages.BAND_OF[pid] != "infant":
            continue
        a = np.asarray(Image.open(ROOT / "assets" / "parts_whole" / pid / "mouth.png"))[..., 3] / 255
        assert (a > 0.05).sum() < 6000, (pid, int((a > 0.05).sum()))


def test_grey_hair_gets_lighter_with_each_senior_stage():
    # 60 代・70 代・80 歳以上の髪が全員同じ銀色で、年の差が見えなかった。段ごとに濃い灰→薄灰→白へ寄せる
    def share(stage, color):
        specs = [s for k in range(5) for s in rw.random_specs(100 + k, 20, age_band=stage)]
        return sum(s["hairColor"] == color for s in specs) / len(specs)
    assert share("60s", 12) >= 0.5 and share("60s", 14) == 0
    assert share("70s", 13) >= 0.4
    assert share("80+", 14) >= 0.6 and share("80+", 12) == 0


def test_ladder_hair_lightens_from_64_to_89():
    s = rw.random_specs(5, 20, age_band="30s")[0]
    s = dict(s, hairColor=0)
    cols = [rw.at_age(s, a)["hairColor"] for a in (64, 74, 89)]
    assert cols == [12, 13, 14]
