"""age_score: 年齢の見え方の点数（推定年齢の表から、段ごとの範囲・隣の段との順序を数える）。"""
import age_score as sc
import ages


def key_of(per_stage=3):
    return [{"stage": st, "age": ages.AGE_RANGE[st][0], "est": None} for st in ages.STAGES for _ in range(per_stage)]


def fill(key, f):
    for k in key:
        k["est"] = f(k)
    return key


def test_perfect_estimates_give_full_order_accuracy_and_range():
    r = sc.score(fill(key_of(), lambda k: k["age"]))
    assert all(v == 1.0 for v in r["order"].values())
    assert r["in_range"]["20s"] == 1.0 and r["outliers"] == []


def test_two_stages_read_as_the_same_age_give_half_order_accuracy():
    r = sc.score(fill(key_of(), lambda k: 45 if k["stage"] in ("40s", "50s") else k["age"]))
    assert r["order"][("40s", "50s")] == 0.5   # 同じ推定は 0.5 点
    assert r["order"][("30s", "40s")] == 1.0


def test_reversed_pair_gives_zero_and_is_listed_as_the_weakest():
    r = sc.score(fill(key_of(), lambda k: {"70s": 82, "80+": 72}.get(k["stage"], k["age"])))
    assert r["order"][("70s", "80+")] == 0.0
    assert r["weakest"][0][0] == ("70s", "80+")


def test_estimate_outside_the_stage_range_is_an_outlier():
    key = fill(key_of(1), lambda k: k["age"])
    key[5]["est"] = 70
    assert [o["stage"] for o in sc.score(key)["outliers"]] == [key[5]["stage"]]


import json
from pathlib import Path

import pytest

ORDER_MEAN_MIN = 0.8   # 隣の段との順序の平均の受け入れ線
ORDER_PAIR_MIN = 0.5   # どの隣の段の組も、年下の段が年上と同じ以上に読まれる（0.5 未満は順序が逆）


@pytest.mark.xfail(reason="受け入れ線は未達（この版の測定は 0.66。描き直しの結果で測り直す）。達したら xfail を外す", strict=False)
def test_latest_age_score_meets_the_acceptance_line():
    r = json.loads((Path(__file__).resolve().parent.parent / "age_score_latest.json").read_text(encoding="utf-8"))
    assert r["order_mean"] >= ORDER_MEAN_MIN, r["order_mean"]
    assert min(r["order"].values()) >= ORDER_PAIR_MIN, sorted(r["order"].items(), key=lambda kv: kv[1])[:3]
