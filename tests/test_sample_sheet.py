import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def test_sample_sheet_has_a_row_per_age_stage(tmp_path):
    out = tmp_path / "sheet.png"
    subprocess.run([sys.executable, str(ROOT / "sample_sheet.py"), "--seed", "7", "--columns", "2", "--size", "64",
                    "--stages", "0y", "30s", "80+", "--out", str(out)], check=True, cwd=tmp_path)
    w, h = Image.open(out).size
    assert (w, h) == (2 * 64 + 48, 3 * (64 + 16))


def test_committed_overview_is_what_the_code_draws_today():
    # samples/overview.png is `python3 sample_sheet.py --out samples/overview.png` (defaults: seed 1, 6 columns, 128 px).
    import numpy as np

    sys.path.insert(0, str(ROOT))
    import ages
    import sample_sheet

    committed = np.asarray(Image.open(ROOT / "samples" / "overview.png").convert("RGBA"))
    drawn = np.asarray(sample_sheet.sheet(list(ages.STAGES), columns=6, size=128, seed=1))
    assert committed.shape == drawn.shape
    assert np.array_equal(committed, drawn)
