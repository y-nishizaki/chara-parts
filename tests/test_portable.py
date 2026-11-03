"""A fresh clone must work from any directory: parts live under assets/, and no file points at a path outside the repository."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import auto_marks  # noqa: E402
import extract  # noqa: E402
import onboard  # noqa: E402
import render_whole as rw  # noqa: E402

PARTS = ROOT / "assets" / "parts_whole"
# A path into someone's home directory (split in two so that this file does not contain one itself).
HOME_PATHS = "/Us" "ers/|/ho" "me/[a-z]|Path\\.ho" "me\\(\\)|expand" "user"


@pytest.mark.parametrize("module, attr", [(rw, "PARTS"), (extract, "OUT"), (onboard, "MANUAL"), (auto_marks, "PARTS")])
def test_parts_are_read_from_assets(module, attr):
    assert PARTS in (getattr(module, attr), *getattr(module, attr).parents)


def test_every_registered_body_has_its_parts():
    ids = [i for i, _, _ in json.loads((ROOT / "ages_registry.json").read_text(encoding="utf-8"))["ids"]]
    missing = [i for i in ids if not (PARTS / i / "base.png").exists()]
    assert not missing, missing[:5]


def test_no_source_file_points_outside_the_repository():
    bad = []
    for path in [*ROOT.glob("*.py"), *ROOT.glob("tests/*.py"), *ROOT.glob("*.json")]:
        if path.name == Path(__file__).name:
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(HOME_PATHS, text):
            bad.append(path.name)
    assert not bad, bad


def test_render_works_from_another_working_directory(tmp_path):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(rw.random_specs(40, 1)[0]), encoding="utf-8")
    out = tmp_path / "out.png"
    subprocess.run([sys.executable, str(ROOT / "render_whole.py"), str(spec), str(out), "--size", "128"], check=True, cwd=tmp_path)
    assert out.stat().st_size > 1000


def test_silhouettes_stand_in_for_the_original_drawings():
    # The original full-size drawings are not shipped; render_whole needs only their outline and size.
    meta = json.loads((PARTS / "meta.json").read_text(encoding="utf-8"))
    for i in range(0, len(rw.IDS), 20):
        mask = rw.silhouette(i)
        assert mask.dtype == bool and mask.shape[::-1] == tuple(meta[rw.IDS[i]]["size"])
        assert 0.05 < mask.mean() < 0.9


def test_no_code_path_reads_the_original_drawings():
    assert not hasattr(rw, "original")
    assert not (ROOT / "whole").exists()


def _calls_without_encoding(path):
    import ast

    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        if name not in ("read_text", "write_text", "open"):
            continue
        binary = any("b" in str(getattr(a, "value", "")) for a in node.args[1:]) or any(
            k.arg == "mode" and "b" in str(getattr(k.value, "value", "")) for k in node.keywords
        )
        if name == "open" and (binary or not isinstance(node.func, ast.Name)):
            continue  # Image.open(...) and binary opens
        if not any(k.arg == "encoding" for k in node.keywords):
            found.append(node.lineno)
    return found


def test_text_files_are_always_read_and_written_as_utf8():
    # Windows defaults to a legacy code page; the comments, JSON and golden files here contain non-ASCII text.
    bad = {p.name: _calls_without_encoding(p) for p in [*ROOT.glob("*.py"), *ROOT.glob("tests/*.py")]}
    assert {k: v for k, v in bad.items() if v} == {}
