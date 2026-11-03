"""新しい元の絵を、乱数の対象に入れるまでの手順。
python3 onboard.py measure <id>...          目印を自動で測り、重ねた画像 scratch/onboard/onboard_<番号>.png（6 体ずつ）と scratch/onboard/onboard_marks.json を出す
python3 onboard.py fix <id> <キー> x y       測った目印を直す（キー: eyeL eyeR browL browR nose mouth faceR など）
python3 onboard.py apply <id>...             scratch/onboard/onboard_marks.json の目印を assets/parts_whole/landmarks.manual.json に足す
python3 onboard.py register <群> <段> <id>...  ages_registry.json に足す（群: infant child teen adult mid senior）"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

import auto_marks as am

ROOT = Path(__file__).resolve().parent
SCR = ROOT / "scratch" / "onboard"
MARKS = SCR / "onboard_marks.json"
MANUAL = ROOT / "assets" / "parts_whole" / "landmarks.manual.json"
REG = ROOT / "ages_registry.json"
COL = {"eyeL": (255, 0, 0), "eyeR": (255, 0, 0), "browL": (0, 120, 255), "browR": (0, 120, 255), "nose": (0, 170, 0), "mouth": (255, 0, 255)}


def group_of(pid):
    reg = {i: g for i, g, _ in json.loads(REG.read_text(encoding="utf-8"))["ids"]}
    if pid in reg:
        return reg[pid]
    return PREFIX_GROUP.get(next((p for p in sorted(PREFIX_GROUP, key=len, reverse=True) if pid.startswith(p)), ""), "adult")


PREFIX_GROUP = {"y": "infant", "b": "infant", "k": "child", "n": "teen", "a": "adult", "d": "mid", "o": "senior", "cm": "child", "cf": "child", "q": "child", "w": "child", "u": "infant", "v": "teen"}


def measure(ids):
    marks = json.loads(MARKS.read_text(encoding="utf-8")) if MARKS.exists() else {}
    for pid in ids:
        marks[pid] = {k: [round(v[0]), round(v[1])] for k, v in am.measure(pid, group_of(pid)).items()}
    MARKS.write_text(json.dumps(marks, indent=0) + "\n", encoding="utf-8")
    sheets = []
    for n in range(0, len(ids), 6):
        tiles = []
        for pid in ids[n:n + 6]:
            img = Image.open(ROOT / "whole" / "raw" / f"{pid}_bald.png").convert("RGBA")
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255)); bg.alpha_composite(img)
            d = ImageDraw.Draw(bg)
            for k, (x, y) in marks[pid].items():
                d.ellipse([x - 9, y - 9, x + 9, y + 9], outline=COL[k], width=4)
            e = marks[pid]["eyeL"]; cx = (marks[pid]["eyeL"][0] + marks[pid]["eyeR"][0]) / 2
            tiles.append(bg.convert("RGB").crop((int(cx - 330), int(e[1] - 300), int(cx + 330), int(e[1] + 420))).resize((330, 460)))
            ImageDraw.Draw(tiles[-1]).text((6, 4), pid, fill=(0, 0, 0))
        W = Image.new("RGB", (330 * 3, 460 * 2), (255, 255, 255))
        for i, t in enumerate(tiles):
            W.paste(t, (330 * (i % 3), 460 * (i // 3)))
        out = SCR / f"onboard_{n // 6}.png"
        W.save(out); sheets.append(str(out))
    print("\n".join(sheets))


def fix(pid, key, x, y):
    marks = json.loads(MARKS.read_text(encoding="utf-8"))
    marks[pid][key] = [int(x), int(y)]
    MARKS.write_text(json.dumps(marks, indent=0) + "\n", encoding="utf-8")


def apply(ids):
    marks = json.loads(MARKS.read_text(encoding="utf-8"))
    man = json.loads(MANUAL.read_text(encoding="utf-8"))
    for pid in ids:
        assert pid not in man, f"{pid} はもう目印がある"
        man[pid] = marks[pid]
    MANUAL.write_text(json.dumps(man, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(len(man))


def register(group, stage, ids):
    reg = json.loads(REG.read_text(encoding="utf-8"))
    have = {i for i, _, _ in reg["ids"]}
    for pid in ids:
        assert pid not in have, pid
        reg["ids"].append([pid, group, stage])
    REG.write_text(json.dumps(reg, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
    print(len(reg["ids"]))


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "measure":
        measure(args)
    elif cmd == "fix":
        fix(*args)
    elif cmd == "apply":
        apply(args)
    elif cmd == "register":
        register(args[0], args[1] if args[1] != "-" else None, args[2:])
