"""仕様 JSON → 正面のバストアップの PNG（1 体ずつ丸ごと描いた顔から取り出した部品を組み直す）。

部品は extract.py が assets/parts_whole/ に書いた物（元の絵の座標のまま）と、目印 assets/parts_whole/landmarks.json。
顔の土台（face）の目印に、ほかの部品を合わせて置く:
  目は左右それぞれを土台の目の中心へ、眉は土台の元の「目と眉の間」を保つ高さへ、鼻は鼻先へ、口は口の中心へ、
  髪は頭の上へ。大きさは顔の幅（faceL〜faceR）の比で合わせる。
首・肩・服: 髪を消した版の服だけを描き替えた絵から取り出した体（assets/parts_whole/bodies/）。頭は体の元の
  持ち主のあご先と顔の幅に合わせて載せるので、頭と肩幅の比は元の絵のまま。描画に乱数は入れない。


使い方: python3 render_whole.py spec.json out.png [--size 256]
"""
from __future__ import annotations

import colorsys
import io
import json
import random
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

import ages
import mouths

ROOT = Path(__file__).resolve().parent
PARTS = ROOT / "assets" / "parts_whole"
IDS = ages.IDS  # 20〜30 代の 23 体が先頭、後ろに年代ごとの元の絵
GENDERS = ("male", "female")
GENDER_IDS = {"male": range(0, 12), "female": range(12, 23)}  # 20〜30 代
BAND_IDS = {b: {g: [i for i, n in enumerate(IDS) if ages.BAND_OF[n] == b and ages.GENDER_OF[n] == g and n not in ages.RETIRED.get(b, ())]
                for g in GENDERS} for b in ages.BANDS}
RETIRED_IDS = {b: {g: [IDS.index(n) for n in ages.RETIRED.get(b, ()) if ages.GENDER_OF[n] == g] for g in GENDERS} for b in ages.BANDS}
# 段（19 段）ごとの元の絵。部品は段ごとに分かれ、段の元の絵が男女そろった段だけ使える
STAGE_IDS = {st: {g: [i for i, n in enumerate(IDS) if ages.STAGE_OF[n] == st and ages.GENDER_OF[n] == g] for g in GENDERS} for st in ages.STAGES}
MIN_STAGE_POOL = 3  # 乱数の組が成り立つ最小の元の絵の数（男女それぞれ）。これ未満の段は、元の絵を描き足すまで使わない

# 髪型は仕様上「年代ごと」に入っていない（輪郭・目・首・体格だけが年代ごと）ので、ほかの年代の髪型を借りられる。
# 薄毛の髪は 40〜50 代と 60 代以上だけ。借りて合わない髪は HAIR_BLOCKED に入れる（年代ごとに見て決めた）
HAIR_BORROW_FROM = {"child": ("teen", "adult"), "teen": ("adult",), "mid": ("adult",), "senior": ("mid", "adult")}
THINNING_HAIR = ("em1", "sm1", "sm2")
# m8 のもみあげは、顔の幅が違うとこめかみで目の高さまで平らな塊になるので、20〜30 代以外には載せない
HAIR_BLOCKED: dict[str, set] = {"child": {IDS.index("tm0"), IDS.index("tm1"), IDS.index("m8"), IDS.index("n16m0")},  # n16m0: 背の高い髪で、大きな頭の子ども（12〜13 歳）に載せると頭頂が枠の上で切れた  # tm0 tm1: 頬に黒い切れ端の房が下がる（子どもの頭に載せると目立った）
                                "teen": {IDS.index("m8")}, "mid": {IDS.index("m8")}, "senior": {IDS.index("m8")}, "infant": {IDS.index("m8")}}
# 年代ごとに乱数の候補から外す。ef2 は角ばった輪郭・短髪、短い髪は女に見えにくい
BAND_EXCLUDED = {("mid", "female"): {"face": {IDS.index("ef2")}, "hair": {IDS.index("ef2"), 13, 16, 20}}}


def band_hair_ids(band: str, gender: str) -> list[int]:
    """その年代・性別で使える髪型（自分の年代の物と、借りられる年代の物）。"""
    ids = list(BAND_IDS[band][gender]) + RETIRED_IDS[band][gender]  # 古い元の絵の髪型は使える
    for b in HAIR_BORROW_FROM.get(band, ()):
        ids += [i for i in BAND_IDS[b][gender] if IDS[i] not in THINNING_HAIR or band in GREY_BANDS]
    return [i for i in ids if i not in HAIR_BLOCKED.get(band, set())]
STAGE_OUTFIT_NAMES = {"0y": ("romper", "bodysuit"), "1y": ("romper", "bodysuit", "knit"), "3y": ("jacket", "knit", "cardigan", "blouse")}  # 0〜1 歳は赤ちゃんの服（大人と同じ襟付きの上着だと小さな大人に見えた）


def outfit_pool(group: str, gender: str, stage: str | None) -> list[int]:
    pool = BAND_OUTFIT_IDS[group][gender]
    names = STAGE_OUTFIT_NAMES.get(stage) if stage else None
    return [o for o in pool if OUTFITS[o]["name"] in names] if names else pool


INFANT_STAGES = tuple(st for st, grp in ages.GROUP_OF_STAGE.items() if grp == "infant")


def stage_hair_ids(group: str, gender: str, stage: str | None) -> list[int]:
    """乱数・at_age が使う髪型。乳幼児（0〜3 歳）は、その段で描いた髪型だけ（0 歳に 3 歳のふさふさの髪、3 歳に 0 歳の薄い髪が載ると年が崩れる）。"""
    if stage in INFANT_STAGES and stage in STAGES_AVAILABLE:
        return list(STAGE_IDS[stage][gender])
    return band_hair_ids(group, gender)


FACE_KEYS = ("face", "eyes", "brows", "nose", "mouth", "hair")
# 服: 髪を消した版の服だけを描き替えた絵（source は首と肩の持ち主）。上着は青緑 1 色で描かせ、色を塗り替える。
# 女の服は首を細くし首の筋の線を消した版（*_slim）から取る
OUTFITS = [
    {"name": "jacket", "gender": "male", "source": 0}, {"name": "knit", "gender": "male", "source": 1},
    {"name": "cardigan", "gender": "male", "source": 2}, {"name": "vest", "gender": "male", "source": 3},
    {"name": "cardigan", "gender": "female", "source": 12}, {"name": "jacket", "gender": "female", "source": 13},
    {"name": "knit", "gender": "female", "source": 14}, {"name": "blouse", "gender": "female", "source": 15},
]
for _o in OUTFITS:
    _o.setdefault("band", ages.DEFAULT_BAND)
for _band, _list in ages.NEW_OUTFITS.items():  # 年代ごとの服（髪を消した版の服だけを描き替えた絵。source は体の持ち主）
    for _name, _gender, _owner in _list:
        OUTFITS.append({"name": _name, "gender": _gender, "source": IDS.index(_owner), "band": _band})
OUTFIT_IDS = {g: [i for i, o in enumerate(OUTFITS) if o["gender"] == g and o["band"] == ages.DEFAULT_BAND] for g in GENDERS}
BAND_OUTFIT_IDS = {b: {g: [i for i, o in enumerate(OUTFITS) if o["gender"] == g and o["band"] == b] for g in GENDERS}
                   for b in ages.BANDS}
# 使える群: 元の絵と服が男女そろった群。使える段: その群に属し、段の元の絵が男女それぞれ MIN_STAGE_POOL 体以上の段
AVAILABLE_BANDS = [b for b in ages.BANDS if all(BAND_IDS[b][g] and BAND_OUTFIT_IDS[b][g] for g in GENDERS)]
STAGES_AVAILABLE = [st for st in ages.STAGES if ages.GROUP_OF_STAGE[st] in AVAILABLE_BANDS
                    and all(len(STAGE_IDS[st][g]) >= MIN_STAGE_POOL for g in GENDERS)]
# 大人の髪の色。絵の髪の地の色に掛ける倍率で塗る
# 明るい茶と、はちみつ色の金髪を足した
HAIR = [(0.16, 0.11, 0.09), (0.29, 0.20, 0.16), (0.39, 0.27, 0.19), (0.46, 0.29, 0.19),
        (0.40, 0.22, 0.17), (0.55, 0.38, 0.24), (0.27, 0.13, 0.11), (0.64, 0.46, 0.29), (0.74, 0.58, 0.36),
        (0.58, 0.24, 0.15), (0.84, 0.68, 0.40), (0.52, 0.45, 0.36),  # 赤・金・灰がかった茶を足した
        (0.62, 0.62, 0.65), (0.78, 0.78, 0.81), (0.91, 0.91, 0.93), (0.42, 0.42, 0.46)]  # 白髪（銀・薄灰・白・濃い灰）
MOUTH_KINDS_BAND = {"child": [0, 1, 3], "infant": [0, 4]}  # 赤ちゃんは元の口か小さな口だけ（にやけ口の笑いじわの弧が口の横に浮いて、傷に見えた）。子どもは、真顔の一文字（2）と小さな口（4）にしない（眉が隠れると睨んで見えた）
GREY_COLORS = {12, 13, 14, 15}  # 白髪は 40〜50 代と 60 代以上だけ
GREY_BANDS = ("mid", "senior")
# 40〜50 代の女は白髪にしない（銀髪の若い顔・銀の編み込みが年代も性別も読めなかった）。60 代以上の濃い灰（c15）も外した（40 代に見えた）
NO_GREY = {("mid", "female")}
WRINKLE_LEVEL = {n: {"mid": 1, "senior": 2}.get(ages.BAND_OF[n], 0) for n in IDS}  # しわ・ほうれい線の濃さ（元の絵に描いた物）
HAIR_WEIGHTS_BAND = {  # 年代ごとの髪色の重み
    "adult": None, "infant": [5, 3, 3, 3, 3, 2, 2, 2, 3, 2, 2, 0], "child": [5, 3, 3, 3, 3, 2, 2, 2, 3, 2, 2, 0], "teen": [5, 3, 3, 3, 3, 2, 2, 2, 3, 2, 2, 0],
    "mid": [4, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 0, 4, 3, 0, 1], "senior": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 0]}
def _grey_weights(**w):
    return [w.get(f"c{i}", 0) for i in range(len(HAIR))]


# 60 代以上は段ごとに白髪の明るさを変える
HAIR_WEIGHTS_STAGE = {"60s": _grey_weights(c12=6, c13=2), "70s": _grey_weights(c12=1, c13=5, c14=2), "80+": _grey_weights(c13=1, c14=6)}
HAIR_WEIGHTS = [5, 3, 3, 3, 3, 2, 2, 2, 3, 2, 2, 0]  # 茶系（1・2・3・5・6・7）は 30 のうち 15。c11 は白髪に見えたので 0
# 髪色の系統（似た組の判定と、茶系の偏りの上限に使う）
HAIR_GROUP = {0: "black", 1: "brown", 2: "brown", 3: "brown", 4: "red", 5: "brown", 6: "brown", 7: "brown",
              8: "blonde", 9: "red", 10: "blonde", 11: "ash", 12: "grey", 13: "grey", 14: "grey", 15: "grey"}
# 瞳の色。自然な範囲: 濃い茶・こげ茶・ヘーゼル・灰青
EYE_COLORS = [None, (0.30, 0.18, 0.12), (0.20, 0.13, 0.10), (0.50, 0.40, 0.20), (0.38, 0.48, 0.58)]
EYE_COLOR_WEIGHTS = [6, 3, 3, 2, 1]
SKIN = [(0.97, 0.81, 0.67), (1, 0.78, 0.60), (0.95, 0.71, 0.50),
        (0.85, 0.61, 0.41), (0.74, 0.50, 0.34), (0.63, 0.41, 0.28)]  # 肌の 6 色
# 上着の色（0 は描いたままの青緑）
CLOTHES = [None, (0.20, 0.25, 0.42), (0.25, 0.42, 0.30), (0.50, 0.22, 0.22), (0.72, 0.55, 0.25),
           (0.45, 0.32, 0.22), (0.40, 0.30, 0.50), (0.32, 0.35, 0.40)]
COUNTS = {"hairColor": len(HAIR), "skin": len(SKIN), "clothesColor": len(CLOTHES), "eyeColor": len(EYE_COLORS)}
KEYS = {"gender", "outfit", *FACE_KEYS, *COUNTS}
OPTIONAL_KEYS = {"age_band", "age", "mouthKind"}  # 省くと 20〜30 代・元の絵の口
# まぶたの重い目。乱数の仕様では使わない
HEAVY_LID_EYES = {1, 10, 15}  # m10 の目は眠そうに見えた
# 乱数の仕様で使わない部品: 髪を隠すと同じ人に見えた男の土台（m2・m4・m6）、女の面長の土台
# （f3・f4・f6）、女の太く角ばった眉（f3）、男の子に見えかねない女の短髪（f5 の髪）
# 眠そうに見えた m10 も外す
# m10 の髪（h10）も外す
# f6 の髪（h18）も外す（元の絵の毛束の線の間が透明で、左の毛束に白い点の列が残った）
EXCLUDED = {"male": {"face": {2, 4, 6, 10}, "hair": {10}}, "female": {"face": {15, 16, 18}, "brows": {15}, "hair": {17, 18}}}
# 描いた髪のうち、元の絵のハイライトが雷形に抜けて（肌色の欠片・穴）、白髪や別の色にすると橙・白の点に見えた物
EXCLUDED["female"]["hair"] |= {IDS.index(n) for n in ("a30f0", "a30f2", "a30f3", "d50f3", "o70f2", "n14f1", "tf3")}
EXCLUDED["male"]["hair"] |= {IDS.index("n18m2")}
# 似た組の判定に使う、男の土台と目の系統（元の絵を描いたときの指示から）
FACE_FAMILY = {0: "oval", 1: "square", 2: "round", 3: "long", 4: "long", 5: "diamond", 6: "round", 7: "heart",
               8: "square", 9: "round", 10: "narrow", 11: "square"}
EYE_FAMILY = {0: "up", 1: "narrow", 2: "down", 3: "round", 4: "almond", 5: "up", 6: "down", 7: "round",
              8: "narrow", 9: "down", 10: "up", 11: "almond"}
DARK_SKINS = {4, 5}          # 目と眉の差がつぶれる濃い肌
YELLOW_EYE_COLORS = {3}      # 赤ちゃんには出さない黄の瞳
PALE_EYE_COLORS = {4}        # 濃い肌と組ませない薄い瞳（灰青）
# 双子に見える組（同じ性別で、髪型の系統・髪の色の濃さ・肌の濃さの段がそろう）の判定に使う表
HAIR_FAMILY = {0: "tousled", 2: "tousled", 4: "tousled", 5: "tousled", 6: "tousled", 9: "tousled", 1: "up", 8: "up",
               11: "neat", 3: "center", 10: "center", 7: "tied",
               12: "long", 21: "long", 14: "tied", 22: "tied", 15: "wavy", 19: "wavy", 13: "bob", 18: "bob",
               16: "short", 20: "short", 17: "short"}
HAIR_NEAR = 0.1     # 髪の色の明るさ（RGB の平均）の差がこれ未満なら近い
LOOK_ALIKE_MIN = 3  # look_alike の重なりの数
BROWN_MAX = 10      # 20 体の中で茶系の髪はこれだけまで
SKIN_NEAR = 2       # 肌の色の段（SKIN の並び）の差がこれ以下なら近い
SKIN_BAND = {0: "light", 1: "light", 2: "light", 3: "dark", 4: "dark", 5: "dark"}
RELAX_AFTER = 1000   # 引き直しがこれだけ続いたら、部品の繰り返しの上限を 1 つ増やす（部品の少ない年代で合う組が尽きたとき。20〜30 代はここまで来ない）
SENIOR_TIGHT = ("senior",)  # 髪色が白髪（灰）だけで服も 2 種の群は、元の絵が多くても似た組を避けきれない（小さい山札と同じ扱い）
SMALL_POOL = 8      # 同じ性別の元の絵がこれ未満の年代では、双子・似た組の規則を緩める
PER_GENDER = 10      # 20 体の組の、同じ性別の人数（部品の繰り返しの上限を部品の数に応じて決める）
PART_MAX = 3        # 20 体の中で、同じ目・眉・髪型はこれだけまで（土台は 2 回まで）
SKIN_BAND_MAX = 11  # 20 体の中で、同じ肌の濃さの段はこれだけまで
# 組み合わせの規則に使う部品の性質（元の絵を見て決めた）
LONGER_HAIR = {2, 3, 7, 10}             # 男の、耳・襟足に掛かる長めの髪（ほかの男の髪は短い）
EYE_BANG_HAIR = {2, 3, 5, 7, 9, 10}     # 男の、目に掛かる前髪・毛束の髪
NARROW_EYES = {8, 9, 11}                # 男の細目・半眼（小さい目）
THICK_BROWS = {1, 8, 10, 16, 21}        # 太くまっすぐな眉（男 b1・b8・b10、女 f4・f9）
SHORT_FEMALE_HAIR = {13, 16, 20}        # 女の短い髪（f1・f4・f8。f5 の髪は使わない）
LASH_EYES = {12, 13, 14, 17, 19, 20}    # 女の、まつげの強い目
SOFT_FACES = {12, 13, 14, 19, 20}       # 女の、柔らかい（丸い・細いあごの）輪郭。f5・f9・f10 は角ばる
SIMILAR_KEYS = ("brows", "nose", "skin", "hairColor", "outfit")
SIMILAR_MAX = 2              # 同じ性別で SIMILAR_KEYS のうち同じ物がこれより多い組は引き直す
NECK_CLEAR = 0.40            # あご先より下で、真ん中からこれだけ（顔の幅に対する割合）の前髪は描かない（首の前の毛の輪）
FACE_STRAND_THIN = 5         # 別の人の顔の眉より下に掛かる前髪のうち、これより細い線（元の絵の px の半分）は描かない
BROW_DARKEN = 0.8     # 眉の色は髪の色のこれだけ倍（少し濃く）
HAIR_MIN_SHARE = 0.03  # 髪のうち、一番大きい塊のこれより小さい塊（本体から離れた切れ端）は消す

FRAME = 3.0         # 出力の 1 辺（体の持ち主の顔の幅の倍数）。肩幅（顔の幅の 2.5〜3 倍）が枠に収まる
CHIN_AT = 0.70      # あご先の高さ（出力の上から、1 辺に対する割合）
CHIN_AT_BAND = {"infant": 0.78, "teen": 0.78, "mid": 0.78, "senior": 0.78}  # 借りた髪型で高い髪が出るので mid・senior も 0.78   # 年代ごとの上書き（髪が枠の上で切れる年代だけ、あご先を下げる）
BANG_SHADE = 0.75   # 前髪の落ち影の濃さ
BANG_DROP = (0.03, 0.008)  # 落ち影のずれ（下・右。顔の幅に対する割合）
NOTCH_FILL = 0.025  # 頭のそばの、幅が顔の幅の 2 倍これ未満の切れ込みを後ろ髪の色で埋める
BACK_BRIDGE = 0.06  # 後ろ髪の塊どうしの間の、幅が顔の幅の 2 倍これ未満の切れ込みも埋める
BACK_BRIDGE_NEAR = 0.15  # 後ろ髪の間を埋めるのは頭からこれだけの内
NOTCH_NEAR = 0.06   # 埋めるのは頭（土台）からこれだけ（顔の幅に対する割合）の内
NECK_SHADE = 0.06   # あごの下の影（頭の形を下へずらした所。顔の幅に対する割合）
BORROWED_CUT_AT_EYES = True  # 借りた髪（非 20〜30 代）で、持ち主の頭より外の土台を消す高さを目の線にする
CLEAN_FACE_LINES = True     # 借りた髪（非 20〜30 代）の、顔に掛かる細い線を全部消す
NECK_CLEAR_BY_SKIN = True   # 首の前の髪を切るのは首の肌の上だけ（非 20〜30 代）
EYE_CUT = 0.0      # 借りた髪で、持ち主の頭より外の土台を消す高さ（目の線から、顔の幅に対する割合だけ上）
CLIP_INSET = 3      # 土台の頭のうち、髪の持ち主の頭の線からこれだけ（元の絵の px）内側より外を、眉より上で消す（以前は
                    # 4px 外まで残し、前髪の縁と後ろ髪の間に肌の細い帯が白い弧に見えた。後ろ髪は頭の線の 4px 内側まである）
SHADE_RGB = (0.76, 0.60, 0.58)  # 肌に掛ける影の色（頭と体の影の色をそろえる）


class SpecError(ValueError):
    pass


def validate(spec: dict) -> dict:
    if not isinstance(spec, dict) or not KEYS <= set(spec) or set(spec) - KEYS - OPTIONAL_KEYS:
        raise SpecError(f"キーは {sorted(KEYS)}（と省いてよい {sorted(OPTIONAL_KEYS)}）ちょうど")
    mk = spec.get("mouthKind", 0)
    if type(mk) is not int or not 0 <= mk < len(mouths.KINDS):
        raise SpecError(f"mouthKind は 0〜{len(mouths.KINDS) - 1}")
    if spec["gender"] not in GENDERS:
        raise SpecError("gender は male か female")
    g = spec["gender"]
    band = spec.get("age_band", ages.DEFAULT_BAND)
    if band not in ages.BANDS:
        raise SpecError(f"age_band は {list(ages.BANDS)} のどれか")
    age, stage = spec.get("age"), None
    if "age" in spec:  # 年は数。段は年で決まり、段の部品だけを使う
        if type(age) is not int or not 0 <= age <= ages.MAX_AGE:
            raise SpecError(f"age は 0〜{ages.MAX_AGE} の整数")
        stage = ages.stage_of_age(age)
        if ages.GROUP_OF_STAGE[stage] != band:
            raise SpecError(f"age {age} は {ages.GROUP_OF_STAGE[stage]} の群で、age_band {band} と合わない")
    for k in FACE_KEYS:
        v = spec[k]
        pool = band_hair_ids(band, g) if k == "hair" else (STAGE_IDS[stage][g] if stage else BAND_IDS[band][g])
        if type(v) is not int or v not in pool:
            raise SpecError(f"{k} は {band} の {g} の部品 {pool} のどれか")
    if type(spec["outfit"]) is not int or spec["outfit"] not in BAND_OUTFIT_IDS[band][g]:
        raise SpecError(f"outfit は {band} の {g} の服 {BAND_OUTFIT_IDS[band][g]} のどれか")
    if spec["hairColor"] in GREY_COLORS and band not in GREY_BANDS:
        raise SpecError("白髪の髪色は 40〜50 代と 60 代以上だけ")
    for k, n in COUNTS.items():
        v = spec[k]
        if v is None and k in ("skin", "hairColor"):  # None は絵のままの色（組み直しの試験用）
            continue
        if type(v) is not int or not 0 <= v < n:
            raise SpecError(f"{k} は 0〜{n - 1}")
    return spec


# ---- 部品と目印 ----
@lru_cache(maxsize=None)
def landmarks() -> dict:
    return json.loads((PARTS / "landmarks.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def meta() -> dict:
    return json.loads((PARTS / "meta.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=64)  # 1254x1254 RGBA の画像（約 6 MiB）を持つので、数に上限を付ける
def part(i: int, name: str) -> Image.Image:
    return Image.open(PARTS / IDS[i] / f"{name}.png").convert("RGBA")


DARK_WRINKLES = True      # 濃い肌のしわを、肌より暗い 1 色の線で描く
DARK_WRINKLE_LINE = 0.55  # その線の明るさ（肌の色に対する倍率）
WRINKLE_GAIN_SEX = {("40s", "male"): 1.3, ("50s", "male"): 1.25, ("40s", "female"): 0.8, ("50s", "female"): 0.6, ("80+", "female"): 0.8}  # 男は 40・50 代が若く、女は 50 代が老けて見えた。段ごとの強さへの倍率
WRINKLE_GAIN_STAGE = {"40s": 3.2, "50s": 2.5, "60s": 1.0, "70s": 0.8, "80+": 0.55}  # 段ごとの強さ（批評: 40 代と 50 代が見分けにくい・80 歳以上のしわが多く濃くてやつれて見える）。表に無い段は下の群の値
WRINKLE_GAIN = {"mid": 3.0, "senior": 1.0}  # しわの層の強さ（40〜50 代は元の線が薄いので強め、60 代以上は土台の線と重なるので弱め）。40〜50 代 2.2 → 3.0


@lru_cache(maxsize=64)  # 同じく画像を持つ
def wrinkle_part(i: int):
    """40〜50 代・60 代以上の元の絵から取ったしわ・ほうれい線の層（土台と同じ座標）。無い年代は None。"""
    f = PARTS / IDS[i] / "wrinkles.png"
    return Image.open(f).convert("RGBA") if f.exists() else None


@lru_cache(maxsize=16)  # 同じく画像を 2 枚持つ
def body(o: int) -> dict:
    b = OUTFITS[o]["band"]
    d = PARTS / "bodies" / (f"{OUTFITS[o]['gender']}_{OUTFITS[o]['name']}" if b == ages.DEFAULT_BAND
                            else f"{b}_{OUTFITS[o]['gender']}_{OUTFITS[o]['name']}")
    return {k: Image.open(d / f"{k}.png").convert("RGBA") for k in ("skin", "clothes")}


@lru_cache(maxsize=None)
def silhouette(i: int) -> np.ndarray:
    """元の絵の輪郭（透明度が 0.5 より上の所）。元の絵そのものは載せていない（大きさはこの配列の形）。"""
    return np.asarray(Image.open(PARTS / "silhouette" / f"{IDS[i]}.png"))


def marks(i):
    return landmarks()[IDS[i]]


def face_w(i):
    m = marks(i)
    return m["faceR"][0] - m["faceL"][0]


def face_cx(i):
    m = marks(i)
    return (m["faceL"][0] + m["faceR"][0]) / 2


def arr(img):
    return np.asarray(img).astype(np.float64) / 255


def to_img(px):
    return Image.fromarray(np.clip(px * 255 + 0.5, 0, 255).astype(np.uint8), "RGBA")


def moved(img: Image.Image, src, dst, s, size) -> Image.Image:
    """src の点が dst に来るよう s 倍して置く。同じ点・等倍なら何もしない（組み直しで元に戻るため）。"""
    if s == 1 and tuple(src) == tuple(dst) and img.size == size:
        return img
    inv = 1 / s
    coeffs = (inv, 0, src[0] - dst[0] * inv, 0, inv, src[1] - dst[1] * inv)
    out = img.convert("RGBa").transform(size, Image.Transform.AFFINE, coeffs, resample=Image.Resampling.BICUBIC)
    return out.convert("RGBA")


def halves(img: Image.Image, x: float):
    """左右に分ける（目・眉を 1 つずつ置くため）。"""
    a = np.asarray(img).copy()
    b = a.copy()
    a[:, round(x):] = 0
    b[:, :round(x)] = 0
    return Image.fromarray(a, "RGBA"), Image.fromarray(b, "RGBA")


def tinted(img: Image.Image, ratio, where=None) -> Image.Image:
    """色に倍率を掛ける。where を渡すと、その画素だけ。"""
    a = arr(img)
    if where is None:
        where = np.ones(a.shape[:2], bool)
    a[where, :3] = np.clip(a[where, :3] * np.asarray(ratio), 0, 1)
    return to_img(a)


GREY_KEEP = 0.2  # 白髪にしたとき、元の髪のハイライトの橙・黄の色味を残す割合（0.0 で無彩色）


def desaturated(img: Image.Image, keep: float) -> Image.Image:
    """色味を keep 倍に近づける（明るさは保つ）。白髪の中に元の茶髪のハイライトの橙の点が残る（60 代・70 代で見えた）。"""
    a = arr(img)
    l = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2])[..., None]
    chroma = a[..., :3].max(axis=2) - a[..., :3].min(axis=2)
    k = keep * np.clip(1 - (chroma - 0.08) / 0.2, 0, 1)[..., None]  # 色味の濃い画素（ハイライトの橙）ほど無彩色へ。薄い色味だけ残す
    a[..., :3] = np.clip(l + k * (a[..., :3] - l), 0, 1)
    return to_img(a)


def skin_like(img: Image.Image) -> np.ndarray:
    """部品の中の、肌と肌の影の色の画素（目のまわり・眉のまわりの影）。肌の色を掛ける所。"""
    a = arr(img)
    rgb = a[..., :3]
    mx, mn = rgb.max(axis=2), rgb.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1e-3)
    return (a[..., 3] > 0) & (rgb[..., 0] >= rgb[..., 1]) & (rgb[..., 1] >= rgb[..., 2]) & (sat > 0.1) & \
        (sat < 0.7) & (mx >= 0.7)


def shade_layer(mask: np.ndarray, skin) -> Image.Image:
    layer = np.zeros(mask.shape + (4,))
    layer[..., :3] = np.asarray(skin) * np.asarray(SHADE_RGB)
    layer[..., 3] = mask
    return to_img(layer)


# ---- 頭 ----
def brow_dy(spec: dict) -> dict:
    """眉を、土台の元の「目の上の端と眉の下の端の間」を保つ高さに置くための、土台の眉の目印からのずれ（左右）。"""
    b, mb, w = spec["face"], marks(spec["face"]), face_w(spec["face"])
    e, br = extent(spec["eyes"], "eyes", "eye"), extent(spec["brows"], "brows", "brow")
    eo, bo = extent(b, "eyes", "eye"), extent(b, "brows", "brow")
    se, sb = w / face_w(spec["eyes"]), w / face_w(spec["brows"])
    out = {}
    for side in ("L", "R"):
        own_gap = (mb["eye" + side][1] - eo[side]["up"]) - (mb["brow" + side][1] + bo[side]["down"])
        y = mb["eye" + side][1] - e[side]["up"] * se - own_gap - br[side]["down"] * sb
        out[side] = y - mb["brow" + side][1]
    return out


def placed_parts(spec: dict) -> dict:
    """部品を土台の座標に置いた画像（重ねる前）。"""
    b = spec["face"]
    mb = marks(b)
    size = part(b, "base").size
    out: dict = {}

    def scale(j):
        return face_w(b) / face_w(j)

    dy = brow_dy(spec)
    for name, key in (("eyes", "eye"), ("brows", "brow")):
        j = spec[name]
        mj = marks(j)
        left, right = halves(brow_part(j) if name == "brows" else part(j, name), (mj[key + "L"][0] + mj[key + "R"][0]) / 2)
        sides = []
        for side, img in (("L", left), ("R", right)):
            dst = mb[key + side] if key == "eye" else (mb[key + side][0], mb[key + side][1] + dy[side])
            if j == b:  # 自分の部品は元の位置のまま（組み直しで元に戻る）
                dst = mb[key + side]
            sides.append(moved(img, mj[key + side], dst, scale(j), size))
        img = sides[0].copy()
        img.alpha_composite(sides[1])
        out[name] = img
    for name in ("nose", "mouth"):
        j = spec[name]
        out[name] = moved(part(j, name), marks(j)[name], mb[name], scale(j), size)
    h = spec["hair"]
    anchor_h, anchor_b = (face_cx(h), marks(h)["top"][1]), (face_cx(b), mb["top"][1])
    for name in ("hair_front", "hair_back"):
        out[name] = moved(part(h, name), anchor_h, anchor_b, scale(h), size)
    if h != b:  # 頬と目尻を横切る細い毛の線は、別の人の顔では描かない
        a = arr(out["hair_front"])
        for _ in range(3):  # 消した残りの縁がまた細い線になるので、残らなくなるまで繰り返す
            on = (a[..., 3] > 0.5) & lower_face(b)
            thick = ndimage.binary_dilation(ndimage.binary_opening(on, iterations=FACE_STRAND_THIN),
                                            iterations=FACE_STRAND_THIN + 1)
            thin = on & ~thick
            if not thin.any():
                break
            a[..., 3] *= ~(ndimage.binary_dilation(thin, iterations=1) & ~thick)
        out["hair_front"] = to_img(a)
    if CLEAN_FACE_LINES and h != b and spec.get("age_band", ages.DEFAULT_BAND) != ages.DEFAULT_BAND:
        # 借りた髪の、顔（眉より下の輪郭の内側）に掛かる細い線は、太い毛束につながっていても全部消す
        #
        a = arr(out["hair_front"])
        on = a[..., 3] > 0.04  # 縮小で薄くなった線も（半透明の線が頬の薄い傷に見えた）
        strong = ndimage.binary_dilation(ndimage.binary_opening(a[..., 3] > 0.5, iterations=FACE_STRAND_THIN), iterations=2)
        kill = ndimage.binary_dilation(on & lower_face(b) & ~strong, iterations=1) & ~strong
        a[..., 3] *= ~kill
        out["hair_front"] = to_img(a)
    drop_hair_fragments(out)  # 細い線を消した後に（消した残りが小さな切れ端になる）
    # 顔に掛かる細い毛の線は、持ち主の顔のときだけ（別の顔では頬の傷に見えた）
    out["hair_thin"] = part(h, "hair_thin") if h == b else None
    base = part(b, "base")
    # 土台の頭のうち、髪の持ち主の頭の線より外（と線の上）を、眉より上だけ消す。以前は別の人の髪のときだけ
    # 消していた。持ち主の髪でも、土台の頭の線（茶色の輪郭線）が前髪の縁から透けて黒髪の上の弧に見えた
    owner = arr(moved(part(h, "base"), anchor_h, anchor_b, scale(h), size))[..., 3] > 0.5
    owner = ndimage.binary_erosion(owner, iterations=CLIP_INSET)  # 内側へ（後ろ髪が頭の線の内側 4px まである）
    rows = np.indices(owner.shape)[0]
    cut = min(mb["browL"][1], mb["browR"][1])
    if BORROWED_CUT_AT_EYES and spec.get("age_band", ages.DEFAULT_BAND) != ages.DEFAULT_BAND and h != b:
        # 借りた髪では、髪の持ち主の頭より外の土台が眉の高さまで四角く突き出し、こめかみの毛束のすき間に肌の棒が見えた。
        # 切る高さを目の高さまで下げる
        cut = max(mb["eyeL"][1], mb["eyeR"][1]) - EYE_CUT * face_w(b)
    keep = owner | (rows > cut)
    a = arr(base)
    a[..., 3] *= keep
    out["base"] = to_img(a)
    return out


@lru_cache(maxsize=None)
def lower_face(b: int) -> np.ndarray:
    """土台の顔のうち、眉より下で輪郭の内側の所（土台の座標）。"""
    m = marks(b)
    face = ndimage.binary_erosion(arr(part(b, "base"))[..., 3] > 0.5, iterations=6)
    rows = np.indices(face.shape)[0]
    return face & (rows > min(m["browL"][1], m["browR"][1]))


def drop_hair_fragments(p: dict) -> None:
    """前髪と後ろ髪を合わせた髪のうち、本体から離れた小さな塊を消す（頬の横・襟の上の切れ端）。"""
    front, back = arr(p["hair_front"]), arr(p["hair_back"])
    a = np.maximum(front[..., 3], back[..., 3]) > 0.5
    lab, n = ndimage.label(a)
    if n < 2:
        return
    sizes = ndimage.sum(a, lab, range(1, n + 1))
    small = np.isin(lab, 1 + np.nonzero(sizes < HAIR_MIN_SHARE * sizes.max())[0])
    small = ndimage.binary_dilation(small, iterations=2) & ~np.isin(lab, 1 + np.nonzero(
        sizes >= HAIR_MIN_SHARE * sizes.max())[0])
    for name, px in (("hair_front", front), ("hair_back", back)):
        px[..., 3] *= ~small
        p[name] = to_img(px)

INK_LUM = 0.42        # 眉の部品の「線・塗り」（暗い画素）の明るさの上限
BROW_VALUE = 0.65     # 眉の色の明るさは髪の色のこれだけ倍（上限 BROW_VALUE_MAX）
BROW_VALUE_MAX = 0.42
BROW_SAT = (0.3, 0.55)  # 眉の色の彩度の範囲（髪の彩度を 0.8 倍して収める）
BROW_HUE = (0.03, 0.09)  # 眉の色相（暗い茶の範囲。金髪の黄・赤毛の赤に引かれると緑・橙の帯になった）
BROW_SKIN_MAX = 0.55  # 眉の色の明るさは肌の明るさのこれ以下（必ず肌より暗く）


def lum(rgb):
    rgb = np.asarray(rgb)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


@lru_cache(maxsize=64)  # 同じく画像を持つ
def brow_part(j: int) -> Image.Image:
    """眉の部品から、左右それぞれ一番大きい 1 本（線と塗り）とそのまわりの肌の影だけを残す。目の下のまぶたの線や
    額の細い線の切れ端（おでこの斜線に見えた）と、肌より明るい筋（眉の上の明るい線）は消す。"""
    img = part(j, "brows")
    a = arr(img)
    mj = marks(j)
    mid = round((mj["browL"][0] + mj["browR"][0]) / 2)
    skin_l = lum(np.asarray(meta()[IDS[j]]["skin"]))
    ink = (a[..., 3] > 0.3) & (lum(a[..., :3]) < INK_LUM)
    cols = np.arange(ink.shape[1])
    keep = np.zeros(ink.shape, bool)
    kept_ink = np.zeros(ink.shape, bool)
    for in_side in (cols < mid, cols >= mid):
        side = ink & in_side[None, :]
        lab, n = ndimage.label(ndimage.binary_dilation(side, iterations=3))
        if n == 0:
            continue
        big = lab == (1 + int(np.argmax(ndimage.sum(side, lab, range(1, n + 1)))))
        keep |= ndimage.binary_dilation(big, iterations=6) & in_side[None, :]
        kept_ink |= big
    bright = (a[..., 3] > 0.3) & (lum(a[..., :3]) > skin_l + 0.02) & ~ink
    a[..., 3] *= keep & ~bright & (~ink | kept_ink)  # 線・塗りは一番大きい 1 本だけ（近くの 1px のごみも消す）
    return to_img(a)


def brow_color(hair_rgb, skin_rgb) -> np.ndarray:
    """眉の色: 髪の色を暗くして彩度を落とした茶系。必ず肌より暗い。"""
    h, sat, v = colorsys.rgb_to_hsv(*hair_rgb)
    h = min(max(h, BROW_HUE[0]), BROW_HUE[1])
    sat = min(max(sat * 0.8, BROW_SAT[0]), BROW_SAT[1])
    c = np.asarray(colorsys.hsv_to_rgb(h, sat, min(v * BROW_VALUE, BROW_VALUE_MAX)))
    if colorsys.rgb_to_hsv(*hair_rgb)[1] < 0.12:  # 白髪の眉は、茶に髪の灰色を半分混ぜた灰がかった茶（真っ黒の眉が浮かない）
        c = 0.5 * c + 0.5 * v * 0.6
    cap = BROW_SKIN_MAX * lum(skin_rgb)
    return c * min(1.0, cap / lum(c))


def tinted_brows(img: Image.Image, spec: dict) -> Image.Image:
    """眉の部品に色を掛ける。線・塗りの画素は眉の色へ、肌の画素は肌の色へ、その間は明るさで混ぜる
    （縁の画素に髪の色の倍率を掛けると、金髪×濃い肌で緑、赤毛×濃い肌で橙の帯になった）。"""
    j = spec["brows"]
    a = arr(img)
    src_skin = np.asarray(meta()[IDS[j]]["skin"])
    src_hair = np.asarray(meta()[IDS[j]]["hair"])
    skin = np.asarray(SKIN[spec["skin"]]) if spec["skin"] is not None else src_skin
    skin_ratio = skin / src_skin
    brow_ratio = (brow_color(HAIR[spec["hairColor"]], skin) / src_hair) if spec["hairColor"] is not None \
        else np.ones(3)
    hi, lo = 0.9 * lum(src_skin), 1.1 * lum(src_hair)
    t = np.clip((hi - lum(a[..., :3])) / (hi - lo), 0, 1)[..., None]
    a[..., :3] = np.clip(a[..., :3] * ((1 - t) * skin_ratio + t * brow_ratio), 0, 1)
    return to_img(a)


def tinted_mouth(img: Image.Image, spec: dict) -> Image.Image:
    """唇の色を肌の色に掛ける（濃い肌に明るいピンクの唇が浮いた）。"""
    if spec["skin"] is None:
        return img
    return tinted(img, np.asarray(SKIN[spec["skin"]]) / np.asarray(meta()[IDS[spec["mouth"]]]["skin"]))


def recolor_irises(img: Image.Image, target) -> Image.Image:
    """瞳（茶色の画素）を、明るさの比を保って target の色へ。白目・線・まつげはそのまま。"""
    if target is None:
        return img
    a = arr(img)
    rgb = a[..., :3]
    mx, mn = rgb.max(axis=2), rgb.min(axis=2)
    iris = (a[..., 3] > 0.05) & ((mx - mn) / np.maximum(mx, 1e-3) > 0.25) & (mx > 0.18) & (mx < 0.7) & \
        (rgb[..., 0] > rgb[..., 2])
    if not iris.any():
        return img
    k = (rgb.mean(axis=2) / np.median(rgb[iris].mean(axis=1)))[..., None]
    a[iris, :3] = np.clip(np.asarray(target) * k[iris], 0, 1)
    return to_img(a)


# 段ごとの頭の大きさ（体に対する倍率。あご先は動かさない）。年が下がるほど頭が大きい。同じ群（体の型）の中で、段の年の差を見せる
HEAD_SCALE_STAGE = {"0y": 1.10, "1y": 1.05, "2y": 1.0, "3y": 0.95, "4-5": 1.12, "6-7": 1.07, "8-9": 1.03, "10-11": 1.0, "12-13": 0.96,
                    "14-15": 1.04, "16-17": 1.0, "18-19": 0.97}
TOP_MARGIN = 0.02  # 髪の上の端から枠の上の端までの空き（枠の高さに対する割合）
EAR_GAP_CLOSE = 0.045  # 耳のまわりの隙間を閉じる幅（顔の幅に対する割合）


def fill_ear_gaps(back: Image.Image, base: Image.Image, b: int) -> Image.Image:
    """別の人の頭に長い髪（耳を覆う髪）を載せると、後ろ髪の穴（元の人の顔と耳の形）と頭の間に背景が見えて、耳の位置に白い欠片が出た
    （8〜9 歳と 6〜7 歳で見えた）。後ろ髪と頭を合わせた形を小さく閉じて、目から顎の高さに空いた隙間だけを、
    後ろ髪の暗い色で埋める（大きく埋めると耳の横に板が出た）。"""
    B, base_a = arr(back), arr(base)[..., 3] > 0.5
    hair = B[..., 3] > 0.5
    if not hair.any():
        return back
    m, w = marks(b), face_w(b)
    r = round(EAR_GAP_CLOSE * w)
    solid = hair | base_a
    closed = ndimage.binary_closing(np.pad(solid, r), iterations=r)[r:-r, r:-r]
    rows = np.indices(hair.shape)[0]
    gap = closed & ~solid & (rows > m["eyeL"][1] - 0.1 * w) & (rows < m["chin"][1])
    if not gap.any():
        return back
    lum = B[..., :3].mean(axis=2)
    dark = np.median(B[hair & (lum <= np.median(lum[hair])), :3], axis=0)
    B[gap, :3] = dark
    B[gap, 3] = 1
    return to_img(B)


def head_layers(spec: dict, tint: bool = True, shadow: bool = True) -> dict:
    """頭を土台の元の絵の座標で、後ろ髪・頭（土台と目鼻口眉）・前髪の 3 層に組む。
    tint=False は絵のままの色で影も足さない（組み直しの試験）。shadow=False は前髪の落ち影を足さない。"""
    validate(spec)
    p = placed_parts(spec)
    b, h = spec["face"], spec["hair"]
    skin = np.asarray(meta()[IDS[b]]["skin"])
    if tint:  # 瞳は肌の色を掛ける前に（濃い肌に掛けた後の目のまわりの影を瞳と取り違えた）
        p["eyes"] = recolor_irises(p["eyes"], EYE_COLORS[spec["eyeColor"]])
    if tint and spec["skin"] is not None:
        skin = np.asarray(SKIN[spec["skin"]])
        p["base"] = tinted(p["base"], skin / np.asarray(meta()[IDS[b]]["skin"]))
        p["nose"] = tinted(p["nose"], skin / np.asarray(meta()[IDS[spec["nose"]]]["skin"]))
        p["mouth"] = tinted_mouth(p["mouth"], spec)
        # 目と眉のまわりの肌の影も肌の色に
        p["eyes"] = tinted(p["eyes"], skin / np.asarray(meta()[IDS[spec["eyes"]]]["skin"]), skin_like(p["eyes"]))
    if spec.get("mouthKind", 0):  # 描いた口（kind 0 は元の絵の口）
        p["mouth"] = mouths.draw_mouth(spec["mouthKind"], spec["gender"], marks(b)["mouth"], face_w(b),
                                       p["base"].size, skin)
    if tint:  # 眉は髪色を暗くした茶系（肌より暗い）。縁の画素は肌の色へ
        p["brows"] = tinted_brows(p["brows"], spec)
    if p["hair_thin"] is not None:
        p["hair_front"] = p["hair_front"].copy()
        p["hair_front"].alpha_composite(p["hair_thin"])
    if tint and spec["hairColor"] is not None:
        target = np.asarray(HAIR[spec["hairColor"]])
        for name, j in (("hair_front", h), ("hair_back", h)):
            p[name] = tinted(p[name], target / np.asarray(meta()[IDS[j]]["hair"]))
            if spec["hairColor"] in GREY_COLORS:
                p[name] = desaturated(p[name], GREY_KEEP)

    if tint and ages.BAND_OF[IDS[b]] != ages.DEFAULT_BAND:  # 20〜30 代の出力は変えない
        p["hair_back"] = fill_ear_gaps(p["hair_back"], p["base"], b)
    head = p["base"].copy()
    wr = wrinkle_part(b)
    if wr is not None and tint:  # 年代の目印（しわ・ほうれい線）。土台の上、目鼻口眉の下
        a = arr(wr)
        gain = WRINKLE_GAIN_STAGE.get(ages.STAGE_OF[IDS[b]], WRINKLE_GAIN[ages.BAND_OF[IDS[b]]]) * WRINKLE_GAIN_SEX.get((ages.STAGE_OF[IDS[b]], ages.GENDER_OF[IDS[b]]), 1.0)
        dark = DARK_WRINKLES and spec["skin"] in DARK_SKINS
        if spec["skin"] in DARK_SKINS and not DARK_WRINKLES:  # 濃い肌では明るい線が引っかき傷に見えた
            gain *= 0.5
        a[..., 3] = np.clip(a[..., 3] * gain, 0, 1)
        if dark:  # 濃い肌のしわは、肌より暗い 1 色の線で描く
            a[..., :3] = np.clip(np.asarray(skin) * DARK_WRINKLE_LINE, 0, 1)
            wr = to_img(a)
        else:
            wr = to_img(a)
            if spec["skin"] is not None:
                wr = tinted(wr, skin / np.asarray(meta()[IDS[b]]["skin"]))
        head.alpha_composite(wr)
    if tint and shadow:  # 前髪の落ち影（額に）。はっきりした影
        hair = arr(p["hair_front"])[..., 3]
        face = arr(p["base"])[..., 3]
        w = face_w(b)
        sh = ndimage.shift(hair, (BANG_DROP[0] * w, BANG_DROP[1] * w), order=1)
        sh = ndimage.gaussian_filter(sh, 0.004 * w) * (1 - hair) * face * BANG_SHADE
        head.alpha_composite(shade_layer(sh, skin))
    for name in ("nose", "mouth", "eyes", "brows"):
        head.alpha_composite(p[name])
    return {"back": p["hair_back"], "head": head, "front": p["hair_front"]}


def compose_head(spec: dict, tint: bool = True, shadow: bool = True) -> Image.Image:
    """頭（後ろ髪・頭・前髪）だけを土台の座標で重ねた物。"""
    layers = head_layers(spec, tint, shadow)
    out = layers["back"].copy()
    out.alpha_composite(layers["head"])
    out.alpha_composite(layers["front"])
    return out


def identity_spec(i: int) -> dict:
    """元の体を自分の部品だけで組む仕様（色は絵のまま）。"""
    g, band = ages.GENDER_OF[IDS[i]], ages.BAND_OF[IDS[i]]
    out = {"gender": g, "face": i, "eyes": i, "brows": i, "nose": i, "mouth": i, "hair": i,
           "hairColor": None, "skin": None, "outfit": BAND_OUTFIT_IDS[band][g][0], "clothesColor": 0, "eyeColor": 0}
    return out if band == ages.DEFAULT_BAND else {**out, "age_band": band}


def head_region(i: int) -> np.ndarray:
    """組み直しを比べる所: 土台と髪。首と服は比べない。"""
    base = arr(part(i, "base"))[..., 3] > 0.99
    hair = (arr(part(i, "hair_front"))[..., 3] > 0.5) | (arr(part(i, "hair_back"))[..., 3] > 0.5)
    return (base | hair) & silhouette(i)


# ---- 首・肩・服 ----
def recolor_clothes(img: Image.Image, target) -> Image.Image:
    """青緑の上着の画素を、明るさの比を保って target の色へ。シャツ（白）と線はそのまま。"""
    if target is None:
        return img
    a = arr(img)
    rgb = a[..., :3]
    mx, mn = rgb.max(axis=2), rgb.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1e-3)
    teal = (rgb[..., 2] > rgb[..., 0] + 0.08) & (sat > 0.2) & (a[..., 3] > 0)
    base = np.asarray(meta()["clothes_base"])
    k = (rgb.mean(axis=2) / base.mean())[..., None]
    a[teal, :3] = np.clip(np.asarray(target) * k[teal], 0, 1)
    return to_img(a)


def placed_in_body(spec: dict) -> dict:
    """頭の層と体の層を、体の持ち主の元の絵の座標に置いた物（重ねる前）。"""
    validate(spec)
    o = spec["outfit"]
    src = OUTFITS[o]["source"]
    b = spec["face"]
    layers = head_layers(spec)
    size = silhouette(src).shape[::-1]
    k = face_w(src) / face_w(b)
    at = ((face_cx(b), marks(b)["chin"][1]), (face_cx(src), marks(src)["chin"][1]))
    k *= HEAD_SCALE_STAGE.get(ages.STAGE_OF[IDS[b]], 1.0)
    if OUTFITS[o]["band"] != ages.DEFAULT_BAND:  # 背の高い髪・大きな頭が枠の上で切れないよう、あご先を動かさずに頭を少し小さく置く（20〜30 代は golden のまま）
        top = min(np.nonzero(np.asarray(img.getchannel("A")) > 20)[0].min() for img in layers.values() if img.getchannel("A").getbbox())
        side = FRAME * face_w(src)  # render() が切り出す枠の高さ。あご先は枠の上から CHIN_AT の所
        room = (CHIN_AT_BAND.get(OUTFITS[o]["band"], CHIN_AT) - TOP_MARGIN) * side
        if (at[0][1] - top) * k > room:
            k = room / (at[0][1] - top)
    lay: dict = {n: moved(img, at[0], at[1], k, size) for n, img in layers.items()}
    bd = body(o)
    # 首の前を横切る前髪は描かない
    m, w, cx = marks(src), face_w(src), face_cx(src)
    front = arr(lay["front"])
    rows, cols = np.indices(front.shape[:2])
    clear = (rows >= m["chin"][1] + 0.05 * w - 1) & (np.abs(cols - cx) <= NECK_CLEAR * w + 1)
    if NECK_CLEAR_BY_SKIN and OUTFITS[o]["band"] != ages.DEFAULT_BAND:
        # 年代を足した体の首は太い（60 代以上は顔の幅の 0.8 倍）。真っ直ぐな縦の切り口が首の肌の脇に残るので、
        # 切るのは首の肌の上だけ（首の輪郭に沿って）。20〜30 代は変えない（golden）
        neck = (arr(bd["skin"])[..., 3] > 0.5) & ~(arr(bd["clothes"])[..., 3] > 0.5)
        clear = (rows >= m["chin"][1] + 0.05 * w - 1) & ndimage.binary_dilation(neck, iterations=3)
    front[..., 3] *= ~clear
    lay["front"] = to_img(front)
    skin = np.asarray(SKIN[spec["skin"]] if spec["skin"] is not None else meta()[IDS[src]]["skin"])
    lay["skin"] = tinted(bd["skin"], skin / np.asarray(meta()[IDS[src]]["skin"]))
    lay["clothes"] = recolor_clothes(bd["clothes"], CLOTHES[spec["clothesColor"]])
    lay["skin_rgb"] = skin
    return lay


def compose_full(spec: dict, lay: dict | None = None) -> Image.Image:
    """頭と体を、体の持ち主の元の絵の座標で重ねる。後ろ髪は首の肌の後ろ・服の上（長い髪の下の端で服が透けた）。lay は placed_in_body の結果（lint が層も見るので使い回す）。"""
    lay = lay or placed_in_body(spec)
    src = OUTFITS[spec["outfit"]]["source"]
    canvas = lay["back"].copy()
    canvas.alpha_composite(lay["skin"])
    canvas.alpha_composite(lay["clothes"])
    back = arr(lay["back"])
    back[..., 3] *= arr(lay["clothes"])[..., 3]
    canvas.alpha_composite(to_img(back))
    # あごの下の影: 頭の形を下へずらし、首の肌の上だけに
    head_a = arr(lay["head"])[..., 3]
    neck = arr(lay["skin"])[..., 3] * (1 - arr(lay["clothes"])[..., 3])
    sh = ndimage.shift(head_a, (NECK_SHADE * face_w(src), 0), order=1) * neck * (1 - head_a)
    canvas.alpha_composite(shade_layer(np.clip(sh, 0, 1), lay["skin_rgb"]))
    canvas.alpha_composite(lay["head"])
    canvas.alpha_composite(lay["front"])
    return fill_notches(canvas, lay, src)


def fill_notches(canvas: Image.Image, lay: dict, src: int) -> Image.Image:
    """頭のそば（あご先より上）で、絵の外形の細い切れ込みを、後ろ髪の暗い色で後ろから埋める（m10 の髪を別の顔に
    載せると、耳の下の後ろ髪と頬・あごの間に白い斜めの隙間が出た）。"""
    a = arr(canvas)
    solid = a[..., 3] > 0.5
    w, m = face_w(src), marks(src)
    r = round(NOTCH_FILL * w)
    head = arr(lay["head"])[..., 3] > 0.5
    closed = ndimage.binary_closing(np.pad(solid, r), iterations=r)[r:-r, r:-r]
    closed &= ndimage.binary_dilation(head, iterations=round(NOTCH_NEAR * w))
    # 後ろ髪の塊どうしの間の広い切れ込み（m10 の髪の、横髪と耳の下の後ろ髪の間）も、後ろ髪だけで閉じて埋める
    rb = round(BACK_BRIDGE * w)
    back_solid = arr(lay["back"])[..., 3] > 0.5
    bridged = ndimage.binary_closing(np.pad(back_solid, rb), iterations=rb)[rb:-rb, rb:-rb]
    closed |= bridged & ndimage.binary_dilation(head, iterations=round(BACK_BRIDGE_NEAR * w))
    gap = closed & ~solid & (np.arange(a.shape[0])[:, None] < m["chin"][1])
    if not gap.any():
        return canvas
    back = arr(lay["back"])
    hair = back[..., 3] > 0.9
    lum = back[..., :3].mean(axis=2)
    dark = hair & (lum <= np.percentile(lum[hair], 30))
    under = np.zeros_like(a)
    under[gap, :3] = np.median(back[dark, :3], axis=0)
    under[gap, 3] = 1
    out = to_img(under)
    out.alpha_composite(canvas)
    return out


def render(spec: dict, size: int = 256, lay: dict | None = None) -> Image.Image:
    img = compose_full(spec, lay)
    src = OUTFITS[spec["outfit"]]["source"]
    side = FRAME * face_w(src)
    x0 = face_cx(src) - side / 2
    y0 = marks(src)["chin"][1] - CHIN_AT_BAND.get(OUTFITS[spec["outfit"]]["band"], CHIN_AT) * side
    box = (round(x0), round(y0), round(x0 + side), round(y0 + side))
    return img.crop(box).resize((size, size), Image.Resampling.LANCZOS, reducing_gap=None)


def render_png(spec: dict, size: int = 256) -> bytes:
    buf = io.BytesIO()
    render(spec, size).save(buf, "PNG", optimize=False)
    return buf.getvalue()


# ---- 合わない組み合わせ（仕様を作るときに出さない）----
# 部品の端は抜いた窓で決まるので、土台の顔が自分の部品で持っていた値（元の絵）と比べる（顔の幅に対する割合）
BROW_SHIFT = 0.05     # 眉を目との間を保つ高さに置いたとき、土台の元の眉からずれる量の上限（額の上・目の際）
EYE_OUTER = 0.02      # 目の外の端が、元の絵の目より外へ出る量の上限（細い顔に大きい目・離れた目）
EYE_INNER = 0.03      # 左右の目の内側の端の間が、元の絵より狭くなる量の上限（寄り目）
MOUTH_EDGE = 0.06     # 口の端は、口の高さのあごの輪郭よりこれだけ内側
NOSE_LONGER = 0.06    # 鼻の上の端が、元の絵の鼻より上へ出る量の上限


@lru_cache(maxsize=None)
def extent(i: int, name: str, key: str) -> dict:
    """部品の左右それぞれの、目印から見た上下左右の端（元の絵の px）。"""
    m = marks(i)
    a = arr(part(i, name))[..., 3] > 0.5
    out = {}
    mid = round(m[key + "L"][0] / 2 + m[key + "R"][0] / 2) if key in ("eye", "brow") else None
    sides = (("L", slice(0, mid)), ("R", slice(mid, None))) if mid else (("", slice(None)),)
    for side, cols in sides:
        sub = np.zeros_like(a)
        sub[:, cols] = a[:, cols]
        ys, xs = np.nonzero(sub)
        cx, cy = m[key + side]
        out[side] = {"left": cx - xs.min(), "right": xs.max() - cx, "up": cy - ys.min(), "down": ys.max() - cy}
    return out


@lru_cache(maxsize=None)
def jaw_half(i: int) -> float:
    """口の高さでの、顔の輪郭の内側の半分の幅。"""
    m = marks(i)
    base = arr(part(i, "base"))
    y = round(m["mouth"][1])
    dark = base[y, :, :3].mean(axis=1) < 0.4
    cx = round(m["mouth"][0])
    left = cx
    while left > 0 and not dark[left]:
        left -= 1
    right = cx
    while right < len(dark) - 1 and not dark[right]:
        right += 1
    return min(cx - left, right - cx)


def measures(spec: dict) -> dict:
    """土台に置いたときの、目の外と内の端・口の幅・鼻の上の端（元の絵の px）。"""
    mb, w = marks(spec["face"]), face_w(spec["face"])
    e = extent(spec["eyes"], "eyes", "eye")
    se = w / face_w(spec["eyes"])
    m = extent(spec["mouth"], "mouth", "mouth")[""]
    if spec.get("mouthKind", 0):  # 描いた口は顔の幅に対する割合で決まる
        m = {"left": mouths.HALF[mouths.KINDS[spec["mouthKind"]]] * face_w(spec["mouth"]), "right": 0}
    n = extent(spec["nose"], "nose", "nose")[""]
    return {
        "outerL": mb["eyeL"][0] - e["L"]["left"] * se, "outerR": mb["eyeR"][0] + e["R"]["right"] * se,
        "inner": (mb["eyeR"][0] - e["R"]["left"] * se) - (mb["eyeL"][0] + e["L"]["right"] * se),
        "mouth": max(m["left"], m["right"]) * w / face_w(spec["mouth"]),
        "noseTop": mb["nose"][1] - n["up"] * w / face_w(spec["nose"]),
    }


def issues(spec: dict) -> list[str]:
    b, w = spec["face"], face_w(spec["face"])
    got = measures(spec)
    own = measures({**spec, **{k: b for k in ("eyes", "brows", "mouth", "nose")}})
    out = []
    dy = brow_dy(spec)
    if max(abs(dy["L"]), abs(dy["R"])) / w > BROW_SHIFT:
        out.append(f"眉の高さが土台の顔に合わない（{dy['L'] / w:+.3f}, {dy['R'] / w:+.3f}）")
    if (own["outerL"] - got["outerL"]) / w > EYE_OUTER or (got["outerR"] - own["outerR"]) / w > EYE_OUTER:
        out.append("目が顔の幅に対して外へ出る（細い顔に大きい目・離れた目）")
    if (own["inner"] - got["inner"]) / w > EYE_INNER:
        out.append("左右の目が寄りすぎ")
    if got["mouth"] > jaw_half(b) - MOUTH_EDGE * w:
        out.append("口があごの幅に対して広すぎる")
    if (own["noseTop"] - got["noseTop"]) / w > NOSE_LONGER:
        out.append("鼻が土台の顔に対して長すぎる")
    return out


def clashes(spec: dict) -> list[str]:
    """合わない部品の組み合わせ。issues と違い、部品の性質の表で決める。"""
    face, eyes, brows, hair = spec["face"], spec["eyes"], spec["brows"], spec["hair"]
    out = []
    if spec["gender"] == "male":
        if (face, eyes) == (11, 11):
            if SKIN_BAND[spec["skin"]] != "dark" or hair not in LONGER_HAIR:
                out.append("m11 の土台と目は、濃い肌と長めの髪とだけ組む")
        elif eyes in NARROW_EYES and (hair in EYE_BANG_HAIR or face == 11):
            out.append("細目・半眼は、目に掛かる前髪・幅広の輪郭 m11 と組まない")
        if brows in THICK_BROWS and hair not in LONGER_HAIR and eyes in NARROW_EYES:
            out.append("太くまっすぐな眉は、短い髪と小さい目と同時に出さない")
    elif hair in SHORT_FEMALE_HAIR and (eyes not in LASH_EYES or face not in SOFT_FACES or brows in THICK_BROWS):
        out.append("女の短い髪は、まつげの強い目・柔らかい輪郭とだけ組み、太い眉と組まない")
    return out


def family(table: dict, i: int):
    """系統の表。年代を足した部品は表に無いので、1 つずつ別の系統として扱う。"""
    return table.get(i, ("id", i))


def twins(a: dict, b: dict) -> bool:
    """同じ性別で、髪型の系統・髪の色の明るさ・肌の濃さが近い組（遠目に双子に見えた）。"""
    lum = [sum(c) / 3 for c in HAIR]
    return a["gender"] == b["gender"] and family(HAIR_FAMILY, a["hair"]) == family(HAIR_FAMILY, b["hair"]) and \
        abs(lum[a["hairColor"]] - lum[b["hairColor"]]) < HAIR_NEAR and abs(a["skin"] - b["skin"]) <= SKIN_NEAR


def look_alike(a: dict, b: dict) -> bool:
    """同じ性別で、同じ土台・同じ目・同じ髪型の系統・同じ髪色の系統・同じ服の種類・近い肌（1 段差まで）のうち
    LOOK_ALIKE_MIN 個以上が重なる組（遠目に似て見えた。土台が同じで髪型・髪色・服のうち 2 つが重なる組を含む）。"""
    if a["gender"] != b["gender"]:
        return False
    same = [a["face"] == b["face"], a["eyes"] == b["eyes"],
            family(HAIR_FAMILY, a["hair"]) == family(HAIR_FAMILY, b["hair"]), HAIR_GROUP[a["hairColor"]] == HAIR_GROUP[b["hairColor"]],
            OUTFITS[a["outfit"]]["name"] == OUTFITS[b["outfit"]]["name"], abs(a["skin"] - b["skin"]) <= 1]
    return sum(same) >= LOOK_ALIKE_MIN


def similar_to(spec: dict, others: list[dict], small_pool: bool = False) -> bool:
    """同じ性別の誰かと、眉・鼻・肌の色・髪の色・服の型のうち 3 つ以上が同じか。濃い肌の男どうしで、土台と目が
    どちらも同じ系統か。双子に見える組か。似て見える組か。"""
    for o in others:
        if sum(spec[k] == o[k] for k in SIMILAR_KEYS) > SIMILAR_MAX + small_pool:
            return True
        if twins(spec, o) or (not small_pool and look_alike(spec, o)):  # 部品の少ない年代は、似た組（3 つ重なり）は避けきれない
            return True
        if small_pool and spec["face"] == o["face"] and spec["eyes"] == o["eyes"] and \
                family(HAIR_FAMILY, spec["hair"]) == family(HAIR_FAMILY, o["hair"]):  # 同じ土台・目・髪型の系統は同じ人に見える
            return True
        if spec["gender"] == "male" and spec["skin"] in DARK_SKINS and o["skin"] in DARK_SKINS and \
                family(FACE_FAMILY, spec["face"]) == family(FACE_FAMILY, o["face"]) and family(EYE_FAMILY, spec["eyes"]) == family(EYE_FAMILY, o["eyes"]):
            return True
    return False


def random_specs(seed: int, count: int, age_band: str | None = None, ok=None) -> list[dict]:
    """乱数の仕様。性別と各部品を山札から引き（土台と目も別々に。20 体で偏らない）、合わない組み合わせと、
    20 体の中で「土台・髪型」が同じ組・3 回目の土台・似た組・双子に見える組・埋まった肌の段・4 回目の目・眉・髪型・同じ髪型の系統＋同じ髪色（肌が違っても）・似て見える組・11 体目の茶髪は引き直す。合わない組み合わせ
    （issues・clashes）も出さない。使わない部品（EXCLUDED）は引かない。
    age_band を指定するとその年代だけ、省くと描いてある全年代から引く（20〜30 代だけの乱数列は変えない）。
    ok（lint_faces）: 仕様を受け取り、描いて変な絵なら False を返す関数。False の組は引き直す。"""
    stage = age_band if age_band in ages.STAGES else None  # 段を渡すと、その段の元の絵だけで引き、年（段の中の数）を持たせる
    if age_band is not None and age_band not in AVAILABLE_BANDS and stage not in STAGES_AVAILABLE:
        raise SpecError(f"age_band は {AVAILABLE_BANDS} か {STAGES_AVAILABLE} のどれか")
    age_bands = [ages.GROUP_OF_STAGE[stage]] if stage else ([age_band] if age_band else list(AVAILABLE_BANDS))
    rng = random.Random(f"{seed}:age2:{stage}" if stage else (f"{seed}:x5" if age_bands == [ages.DEFAULT_BAND] else f"{seed}:age1:{age_band or 'all'}"))
    decks: dict = {}

    def draw(key, values):
        d = decks.setdefault(key, [])
        if not d:
            d.extend(values)
            rng.shuffle(d)
        return d.pop()

    out: list[dict] = []
    for n in range(count):
        gender = draw("gender", list(GENDERS))
        band = draw("band", age_bands) if len(age_bands) > 1 else age_bands[0]
        ids = list(STAGE_IDS[stage][gender]) if stage else list(BAND_IDS[band][gender])
        weights = list((HAIR_WEIGHTS_STAGE.get(stage) if stage else None) or HAIR_WEIGHTS_BAND[band] or HAIR_WEIGHTS)
        if (band, gender) in NO_GREY:
            weights = [0 if c in GREY_COLORS else wt for c, wt in enumerate(weights)]
        excluded = EXCLUDED.get(gender, {})
        cand = {k: [i for i in ids if i not in excluded.get(k, set())] for k in FACE_KEYS}
        cand["hair"] = [i for i in stage_hair_ids(band, gender, stage) if i not in excluded.get("hair", set())]
        for k, v in BAND_EXCLUDED.get((band, gender), {}).items():
            cand[k] = [i for i in cand[k] if i not in v]
        cand["eyes"] = [i for i in cand["eyes"] if i not in HEAVY_LID_EYES]
        group = out[n - n % 20:]  # 同じ 20 体の組（20 体ずつの区切り）
        same_gender = [s for s in group if s["gender"] == gender]
        small_pool = len(ids) < SMALL_POOL or band in SENIOR_TIGHT
        quota = -(-PER_GENDER // len(ids))  # 部品の少ない年代（子ども 3 体ずつ）では、同じ土台・部品が繰り返す上限を増やす
        used = [(s["face"], s["hair"]) for s in group]  # 土台と髪型が同じ組（色違いの同じ人に見えた）。部品の少ない年代は数回まで
        pair_max = max(1, -(-PER_GENDER // (len(ids) ** 2)))
        face_max, part_max = max(2, quota), max(PART_MAX, quota + 1)
        faces = [s["face"] for s in group]
        bands = [SKIN_BAND[s["skin"]] for s in group]
        hair_colors = {(family(HAIR_FAMILY, s["hair"]), s["hairColor"]) for s in group}  # 同じ髪型（系統）＋同じ髪色は 1 回まで
        browns = sum(HAIR_GROUP[s["hairColor"]] == "brown" for s in group)  # 茶系は BROWN_MAX まで
        spec = {k: draw((gender, band, k), cand[k]) for k in FACE_KEYS}
        spec.update(gender=gender, outfit=draw((gender, band, "outfit"), outfit_pool(band, gender, stage)),
                    mouthKind=draw((gender, band, "mouthKind"), MOUTH_KINDS_BAND.get(band, list(range(len(mouths.KINDS))))),
                    hairColor=rng.choices(range(len(weights)), weights)[0],
                    skin=rng.randrange(len(SKIN)), clothesColor=rng.randrange(len(CLOTHES)),
                    eyeColor=rng.choices(range(len(EYE_COLORS)), EYE_COLOR_WEIGHTS)[0])
        for attempt in range(2000):
            if spec["skin"] in DARK_SKINS and spec["eyeColor"] in PALE_EYE_COLORS:
                spec["eyeColor"] = 0
            if band == "infant" and spec["eyeColor"] in YELLOW_EYE_COLORS:
                spec["eyeColor"] = 0
            if not clashes(spec) and not issues(spec) and used.count((spec["face"], spec["hair"])) < pair_max \
                    and (not small_pool or band == ages.DEFAULT_BAND or attempt >= RELAX_AFTER
                         or not any(s["face"] == spec["face"] and s["eyes"] == spec["eyes"] for s in same_gender)) \
                    and faces.count(spec["face"]) < face_max + (attempt >= RELAX_AFTER) and not similar_to(spec, same_gender, small_pool) \
                    and bands.count(SKIN_BAND[spec["skin"]]) < SKIN_BAND_MAX \
                    and (family(HAIR_FAMILY, spec["hair"]), spec["hairColor"]) not in hair_colors \
                    and not (HAIR_GROUP[spec["hairColor"]] == "brown" and browns >= BROWN_MAX) \
                    and all(sum(s[k] == spec[k] for s in group) < part_max + (attempt >= RELAX_AFTER)
                            for k in ("eyes", "brows", "hair")) \
                    and (ok is None or ok(dict(spec, age_band=band, **({"age": ages.AGE_RANGE[stage][0]} if stage else {})))):
                break
            if bands.count(SKIN_BAND[spec["skin"]]) >= SKIN_BAND_MAX:  # 肌の濃さの段が埋まった
                spec["skin"] = rng.randrange(len(SKIN))
            if (family(HAIR_FAMILY, spec["hair"]), spec["hairColor"]) in hair_colors or \
                    (HAIR_GROUP[spec["hairColor"]] == "brown" and browns >= BROWN_MAX):
                spec["hairColor"] = rng.choices(range(len(weights)), weights)[0]
            if attempt % 5 == 4:  # 似た組: 肌・髪の色・服の型も引き直す
                spec.update(skin=rng.randrange(len(SKIN)), hairColor=rng.choices(range(len(weights)), weights)[0],
                            outfit=rng.choice(outfit_pool(band, gender, stage)))
            if attempt % 20 == 19 or faces.count(spec["face"]) >= face_max:  # 合う目が無い土台・3 回目の土台は引き直す
                spec["face"] = rng.choice(cand["face"])
            spec["eyes"] = rng.choice(cand["eyes"])
            for k in ("brows", "mouth", "nose"):
                spec[k] = rng.choice(cand[k])
            if rng.random() < 0.2 or sum(s["hair"] == spec["hair"] for s in group) >= part_max:
                spec["hair"] = rng.choice(cand["hair"])
        else:
            why = {"clashes": bool(clashes(spec)), "issues": bool(issues(spec)),
                   "pair": used.count((spec["face"], spec["hair"])) >= pair_max, "face": faces.count(spec["face"]) >= face_max,
                   "similar": similar_to(spec, same_gender, small_pool),
                   "skin_band": bands.count(SKIN_BAND[spec["skin"]]) >= SKIN_BAND_MAX,
                   "hair_color": (family(HAIR_FAMILY, spec["hair"]), spec["hairColor"]) in hair_colors,
                   "brown": HAIR_GROUP[spec["hairColor"]] == "brown" and browns >= BROWN_MAX,
                   "part": [k for k in ("eyes", "brows", "hair") if sum(s[k] == spec[k] for s in group) >= part_max],
                   "counts": {k: [sum(s[k] == i for s in group) for i in cand[k]] for k in ("face", "eyes", "brows", "hair")}}
            raise RuntimeError(f"合う組み合わせが見つからない（{n} 体目 {band} {gender}）: {why}")
        spec["age_band"] = band
        if stage:
            spec["age"] = rng.randint(*ages.AGE_RANGE[stage])
        out.append(validate(spec))
    return out


def at_age(spec: dict, age: int) -> dict:
    """同じ髪・肌・目の色のまま、年だけを変えた仕様。部品は段ごとなので、顔・目・眉・鼻・口の番号は「その段の同じ並び順」に写す
    （同じ並び順 = 段の元の絵の何番目か。段の絵の数が違えば割り算の余り）。髪型は新しい群で使えれば残す。
    白髪の髪色は 40〜50 代と 60 代以上だけなので、若い群へ移すときは髪色 0 に戻す。"""
    validate(spec)
    stage = ages.stage_of_age(age) if type(age) is int and 0 <= age <= ages.MAX_AGE else None
    if stage is None or stage not in STAGES_AVAILABLE:
        raise SpecError(f"age は、元の絵が男女そろった段の年 {[ages.AGE_RANGE[s] for s in STAGES_AVAILABLE]}")
    g, group = spec["gender"], ages.GROUP_OF_STAGE[stage]
    old_band = spec.get("age_band", ages.DEFAULT_BAND)
    old_stage = ages.stage_of_age(spec["age"]) if "age" in spec else None
    old_pool = STAGE_IDS[old_stage][g] if old_stage in STAGES_AVAILABLE else BAND_IDS[old_band][g]
    new_pool = STAGE_IDS[stage][g]
    out = dict(spec, age=age, age_band=group)
    for k in ("face", "eyes", "brows", "nose", "mouth"):
        out[k] = new_pool[(old_pool.index(spec[k]) if spec[k] in old_pool else 0) % len(new_pool)]
    hair_pool = stage_hair_ids(group, g, stage)
    if spec["hair"] not in hair_pool:
        old_hair = stage_hair_ids(old_band, g, old_stage)
        out["hair"] = hair_pool[(old_hair.index(spec["hair"]) if spec["hair"] in old_hair else 0) % len(hair_pool)]
    outfits = outfit_pool(group, g, stage)
    old_outfits = BAND_OUTFIT_IDS[old_band][g]
    out["outfit"] = outfits[(old_outfits.index(spec["outfit"]) if spec["outfit"] in old_outfits else 0) % len(outfits)]
    if spec["hairColor"] in GREY_COLORS and group not in GREY_BANDS:
        out["hairColor"] = 0
    elif group == "senior" and spec["hairColor"] not in GREY_COLORS:
        out["hairColor"] = sorted(GREY_COLORS)[spec["hairColor"] % len(GREY_COLORS)]  # 60 代以上は白髪（乱数と同じ。梯子で 74 歳の髪が若いままにならない）
    if stage in HAIR_WEIGHTS_STAGE:  # 60 代以上の梯子は、段ごとの白髪の明るさ（60 代 c12・70 代 c13・80 歳以上 c14）
        out["hairColor"] = max(range(len(HAIR)), key=HAIR_WEIGHTS_STAGE[stage].__getitem__)
    if "mouthKind" in spec and spec["mouthKind"] not in MOUTH_KINDS_BAND.get(group, list(range(len(mouths.KINDS)))):
        out["mouthKind"] = 0
    return validate(out)


def main(argv):
    size = 256
    if "--size" in argv:
        i = argv.index("--size")
        size = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    spec = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    Path(argv[1]).write_bytes(render_png(spec, size))


if __name__ == "__main__":
    main(sys.argv[1:])
