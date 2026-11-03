"""Render a contact sheet: one row per age stage, a few random characters in each row.

    python3 sample_sheet.py --out sheet.png                       # all 19 stages
    python3 sample_sheet.py --stages 4-5 30s 80+ --columns 6 --size 128 --seed 3 --out sheet.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

import ages
import lint_faces
import render_whole as rw

LABEL_W = 48
LABEL_H = 16


def sheet(stages, columns: int, size: int, seed: int) -> Image.Image:
    img = Image.new("RGBA", (LABEL_W + columns * size, len(stages) * (size + LABEL_H)), (245, 243, 238, 255))
    draw = ImageDraw.Draw(img)
    for row, stage in enumerate(stages):
        y = row * (size + LABEL_H)
        draw.text((4, y + 2), stage, fill=(40, 40, 40, 255))
        for col, spec in enumerate(lint_faces.clean_specs(seed, columns, age_band=stage)):
            img.alpha_composite(rw.render(spec, size), (LABEL_W + col * size, y + LABEL_H))
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", default=list(ages.STAGES), choices=ages.STAGES)
    ap.add_argument("--columns", type=int, default=6)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sheet(a.stages, a.columns, a.size, a.seed).save(a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
