"""年代（群）・段と、元の絵の id（render_whole.py と extract.py が共有する）。
群（BANDS）: infant・child・teen・adult・mid・senior。体の型・しわ・白髪・服はこの群で決まる。
段（STAGES）: 0y 1y 2y 3y 4-5 6-7 ... 20s 30s 40s 50s 60s 70s 80+ の 19 段。各段は 1 つの群に属し、部品（元の絵）は段ごとに分かれる。
元の絵の一覧は ages_registry.json（id・群・段。段が null の id は乱数の顔の対象から外した古い絵）。id の並びは 20〜30 代の 23 体が先頭で変えない。"""
from __future__ import annotations

import json
from pathlib import Path

BANDS = ("infant", "child", "teen", "adult", "mid", "senior")  # 群
DEFAULT_BAND = "adult"
STAGES = ("0y", "1y", "2y", "3y", "4-5", "6-7", "8-9", "10-11", "12-13", "14-15", "16-17", "18-19", "20s", "30s", "40s", "50s", "60s", "70s", "80+")
AGE_RANGE = {"0y": (0, 0), "1y": (1, 1), "2y": (2, 2), "3y": (3, 3), "4-5": (4, 5), "6-7": (6, 7), "8-9": (8, 9), "10-11": (10, 11),
             "12-13": (12, 13), "14-15": (14, 15), "16-17": (16, 17), "18-19": (18, 19), "20s": (20, 29), "30s": (30, 39), "40s": (40, 49),
             "50s": (50, 59), "60s": (60, 69), "70s": (70, 79), "80+": (80, 99)}
GROUP_OF_STAGE = {"0y": "infant", "1y": "infant", "2y": "infant", "3y": "infant", "4-5": "child", "6-7": "child", "8-9": "child",
                  "10-11": "child", "12-13": "child", "14-15": "teen", "16-17": "teen", "18-19": "teen", "20s": "adult", "30s": "adult",
                  "40s": "mid", "50s": "mid", "60s": "senior", "70s": "senior", "80+": "senior"}
MAX_AGE = 99


def stage_of_age(age: int) -> str:
    for stage, (lo, hi) in AGE_RANGE.items():
        if lo <= age <= hi:
            return stage
    raise ValueError(f"age は 0〜{MAX_AGE}")


_REG = json.loads((Path(__file__).resolve().parent / "ages_registry.json").read_text(encoding="utf-8"))["ids"]
ORDER = ("adult", "child", "teen", "mid", "senior", "infant")  # id の並び順（20〜30 代が先頭）
IDS = [i for g in ORDER for i, grp, _ in _REG if grp == g]
BAND_OF = {i: g for i, g, _ in _REG}
STAGE_OF = {i: s for i, _, s in _REG}
GENDER_OF = {i: "male" if i.rstrip("0123456789").endswith("m") else "female" for i in IDS}
ADULT_IDS = [i for i in IDS if BAND_OF[i] == "adult"][:23]
# 乱数の顔の対象から外した古い元の絵（髪型と、体の首・肩の持ち主としては残す）
RETIRED = {b: tuple(i for i, g, s in _REG if g == b and s is None) for b in BANDS if any(g == b and s is None for _, g, s in _REG)}


def band_ids(band: str) -> list[str]:
    return [i for i in IDS if BAND_OF[i] == band]


# 群ごとの服（名前, 性別, 首と肩の持ち主）。描いた群だけここに足す。extract.py の BODIES と render_whole.py の OUTFITS が使う
NEW_OUTFITS: dict[str, list[tuple[str, str, str]]] = {
    "teen": [("blazer", "male", "tm0"), ("knit", "male", "tm0"), ("cardigan", "female", "tf0"), ("blouse", "female", "tf0")],
    "mid": [("jacket", "male", "em0"), ("knit", "male", "em0"), ("jacket", "female", "ef0"), ("blouse", "female", "ef0")],
    "senior": [("cardigan", "male", "sm0"), ("jacket", "male", "sm0"), ("cardigan", "female", "sf0"), ("knit", "female", "sf0")],
    "infant": [("jacket", "male", "b0m0"), ("knit", "male", "b0m0"), ("cardigan", "female", "b0f0"), ("blouse", "female", "b0f0"),
               ("romper", "male", "r0m0"), ("bodysuit", "male", "r0m0"), ("romper", "female", "r0f0"), ("bodysuit", "female", "r0f0")],  # 赤ちゃんの服（体の持ち主は新しい 0 歳）
    "child": [("jacket", "male", "cm0"), ("knit", "male", "cm0"), ("cardigan", "female", "cf0"), ("blouse", "female", "cf0")],
}
