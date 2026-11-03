"""年齢の見え方の点数。絵を段の名前を伏せて見せ、推定した年齢（採点役の出力）から、
  - 段の範囲に入るか（段ごとの中央値も）
  - 隣の段との順序（隣り合う 2 段の全ての組で、年下の段の推定が年上の段より小さい割合。同じ推定は 0.5）
  - 外れた体の一覧
を出す。
python3 age_score.py sheets <1 段の体数>=10 <seed>=1 <出力先>   段を混ぜた一覧画像と答え合わせの表（key.json）を作る
python3 age_score.py score <key.json> <推定の json>          推定（{"<sheet>:<slot>": 年齢}）を点数にする"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

import ages

PAIRS = list(zip(ages.STAGES, ages.STAGES[1:]))


def tolerance(stage: str) -> float:
    lo, hi = ages.AGE_RANGE[stage]
    return 1 + 0.25 * (hi - lo + 1)


def score(key: list[dict]) -> dict:
    """key は [{"stage", "age", "est"}]（est は推定年齢）。"""
    by = {st: [k["est"] for k in key if k["stage"] == st and k["est"] is not None] for st in ages.STAGES}
    in_range = {st: float(np.mean([ages.AGE_RANGE[st][0] - tolerance(st) <= e <= ages.AGE_RANGE[st][1] + tolerance(st) for e in v])) if v else None
                for st, v in by.items()}
    order = {}
    for a, b in PAIRS:
        if by[a] and by[b]:
            cmp = [(eb > ea) + 0.5 * (eb == ea) for ea in by[a] for eb in by[b]]
            order[(a, b)] = float(np.mean(cmp))
    outliers = [dict(k) for k in key if k["est"] is not None and not
                (ages.AGE_RANGE[k["stage"]][0] - tolerance(k["stage"]) <= k["est"] <= ages.AGE_RANGE[k["stage"]][1] + tolerance(k["stage"]))]
    return {"median": {st: float(np.median(v)) if v else None for st, v in by.items()}, "in_range": in_range, "order": order,
            "weakest": sorted(order.items(), key=lambda kv: kv[1]), "outliers": outliers,
            "order_min": min(order.values()) if order else None, "order_mean": float(np.mean(list(order.values()))) if order else None}


def make_sheets(n: int, seed: int, out: Path, cols: int = 5, tile: int = 192) -> list[dict]:
    """全段の n 体ずつを混ぜて 10 体ずつの一覧にする（段は伏せる）。答え合わせの表を返す（key.json にも書く）。"""
    from PIL import Image, ImageDraw

    import render_whole as rw
    out.mkdir(parents=True, exist_ok=True)
    items = []
    for st in rw.STAGES_AVAILABLE:
        for s in rw.random_specs(seed, 20, age_band=st)[:n]:
            items.append({"stage": st, "age": s["age"], "spec": s})
    random.Random(seed).shuffle(items)
    key = []
    for si in range(0, len(items), 10):
        chunk = items[si:si + 10]
        sheet = Image.new("RGB", (cols * tile, 2 * (tile + 18)), (235, 230, 215))
        d = ImageDraw.Draw(sheet)
        for j, it in enumerate(chunk):
            x, y = (j % cols) * tile, (j // cols) * (tile + 18)
            img = rw.render(it["spec"], tile)
            sheet.paste(img, (x, y + 18), img)
            d.text((x + 6, y + 3), f"#{j + 1}", fill=(0, 0, 0))
            key.append({"id": f"{si // 10 + 1}:{j + 1}", "stage": it["stage"], "age": it["age"], "est": None})
        sheet.save(out / f"sheet_{si // 10 + 1:02d}.png")
    (out / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    return key


if __name__ == "__main__":
    if sys.argv[1] == "sheets":
        make_sheets(int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4]))
    else:
        key = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
        est = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
        for k in key:
            k["est"] = est.get(k["id"])
        r = score(key)
        print("段  中央値  範囲内")
        for st in ages.STAGES:
            print(f"{st:6} {r['median'][st]}  {r['in_range'][st]}")
        print("隣の段の順序（弱い順）")
        for p, v in r["weakest"]:
            print(f"{p[0]}→{p[1]}: {v:.2f}")
        print("外れた体:", [(o["id"], o["stage"], o["est"]) for o in r["outliers"]])
