"""20〜30 代（群 adult）の出力の固定（golden）を作り直す。python3 make_golden.py [--write]
seed 40/41/42 の random_specs(age_band="adult") の仕様の SHA-256 と、60 体の 256px の PNG の SHA-256 を tests/golden_adult.json の形で出す。
--write なしは、今の golden と比べて、変わった seed と体の番号を出すだけ。"""
import hashlib
import json
import sys
from pathlib import Path

import render_whole as rw

ROOT = Path(__file__).resolve().parent
GOLD = ROOT / "tests" / "golden_adult.json"


def build():
    out = {}
    for seed in (40, 41, 42):
        specs = rw.random_specs(seed, 20, age_band="adult")
        bare = [{k: v for k, v in s.items() if k != "age_band"} for s in specs]
        out[str(seed)] = {"specs_sha": hashlib.sha256(json.dumps(bare, sort_keys=True).encode()).hexdigest(),
                          "rgba256": [hashlib.sha256(rw.render(s, 256).tobytes()).hexdigest() for s in specs]}
    return out


if __name__ == "__main__":
    new = build()
    old = json.loads(GOLD.read_text(encoding="utf-8"))
    for seed in new:
        same_specs = new[seed]["specs_sha"] == old[seed]["specs_sha"]
        changed = [i for i, (a, b) in enumerate(zip(new[seed]["rgba256"], old[seed]["rgba256"])) if a != b]
        print(f"seed {seed}: 仕様 {'同じ' if same_specs else '変わった'} {old[seed]['specs_sha'][:12]} → {new[seed]['specs_sha'][:12]}、絵が変わった体 {len(changed)}/20 {changed}")
    if "--write" in sys.argv:
        GOLD.write_text(json.dumps(new, indent=1) + "\n", encoding="utf-8")
        print("書いた")
