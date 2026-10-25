"""Verification suite for PureMacro v6.0.0 pedagogical showcase notebooks 73, 74, and 75."""
from __future__ import annotations

import ast
import base64
import io
import json
from pathlib import Path
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[2]
STEM_PAIRS = [
    ("73_sovereign_debt_crises_and_restructuring", "73_sovereign_debt_crises_and_restructuring_es"),
    ("74_inequality_distributional_lp_and_hank", "74_inequality_distributional_lp_and_hank_es"),
    ("75_green_transition_networks_and_spatial_iam", "75_green_transition_networks_and_spatial_iam_es"),
]
ALL_STEMS = [stem for pair in STEM_PAIRS for stem in pair]


def _source_cells(path: Path):
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"^# %%(.*)\n", text, flags=re.MULTILINE)
    cells = []
    for i in range(1, len(parts), 2):
        body = parts[i + 1].strip()
        kind = "markdown" if "[markdown]" in parts[i] else "code"
        if kind == "markdown":
            body = "\n".join(
                line[2:] if line.startswith("# ") else "" if line == "#" else line
                for line in body.splitlines()
            ).strip()
        if body:
            cells.append((kind, body))
    return cells


def _source_code(path: Path):
    return [source for kind, source in _source_cells(path) if kind == "code"]


def _text(source):
    return "".join(source) if isinstance(source, list) else source


@pytest.mark.parametrize("en_stem, es_stem", STEM_PAIRS)
def test_bilingual_notebooks_source_parity_and_syntax(en_stem, es_stem):
    en_py = ROOT / "notebooks" / f"{en_stem}.py"
    es_py = ROOT / "notebooks" / f"{es_stem}.py"
    assert en_py.exists(), f"Missing {en_py}"
    assert es_py.exists(), f"Missing {es_py}"

    en_code = _source_code(en_py)
    es_code = _source_code(es_py)
    assert len(en_code) >= 6, f"{en_stem} has fewer than 6 code blocks"
    assert en_code == es_code, f"Code parity mismatch between {en_stem} and {es_stem}"

    for stem in (en_stem, es_stem):
        nb_path = ROOT / "notebooks" / f"{stem}.ipynb"
        assert nb_path.exists(), f"Missing compiled notebook {nb_path}"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        actual = [
            _text(c["source"]).strip()
            for c in nb["cells"]
            if c["cell_type"] == "code" and _text(c["source"]).strip()
        ]
        assert actual == en_code
        for block in actual:
            ast.parse(block)


@pytest.mark.parametrize("stem", ALL_STEMS)
def test_saved_notebooks_outputs_and_opaque_canvas(stem):
    nb_path = ROOT / "notebooks" / f"{stem}.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    images, counts, text = [], [], []

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        if _text(cell["source"]).strip():
            counts.append(cell["execution_count"])
        for out in cell["outputs"]:
            assert out["output_type"] != "error", f"Error output in {stem}: {out}"
            if "text" in out:
                text.append(_text(out["text"]))
            if "image/png" in out.get("data", {}):
                raw = base64.b64decode(_text(out["data"]["image/png"]))
                with Image.open(io.BytesIO(raw)) as image:
                    rgba = np.array(image.convert("RGBA"))
                assert rgba.shape[0] >= 200 and rgba.shape[1] >= 600
                assert np.all(rgba[:, :, 3] == 255), f"Found translucent canvas in {stem}"
                images.append(raw)

    assert len(counts) >= 6 and None not in counts
    assert len(images) >= 1, f"{stem} contains no saved figure images"


@pytest.mark.parametrize("stem", ALL_STEMS)
def test_pedagogical_notebooks_no_prohibited_calls(stem):
    py_path = ROOT / "notebooks" / f"{stem}.py"
    lines = py_path.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines, 1):
        assert not re.search(r"\bplt\.show\s*\(", line), f"{stem}:{idx} contains plt.show()"
        assert not re.search(r"\bplt\.tight_layout\s*\(", line), f"{stem}:{idx} contains plt.tight_layout()"


def test_notebook_73_offline_execution_and_sovereign_mechanisms(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(ROOT / "notebooks"))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setattr(plt, "show", lambda: None)
    namespace = {"__name__": "__notebook_test__"}
    with matplotlib.rc_context():
        try:
            exec(
                compile(
                    (ROOT / "notebooks" / "73_sovereign_debt_crises_and_restructuring.py").read_text(encoding="utf-8"),
                    "73_sovereign_debt_crises_and_restructuring.py",
                    "exec",
                ),
                namespace,
            )
        finally:
            plt.close("all")

    scorecard = namespace.get("scorecard")
    assert scorecard is not None, "Notebook 73 must publish a scorecard"
    assert (scorecard["measured"] <= scorecard["bound"]).all()


def test_notebook_74_offline_execution_and_distributional_mechanisms(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(ROOT / "notebooks"))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setattr(plt, "show", lambda: None)
    namespace = {"__name__": "__notebook_test__"}
    with matplotlib.rc_context():
        try:
            exec(
                compile(
                    (ROOT / "notebooks" / "74_inequality_distributional_lp_and_hank.py").read_text(encoding="utf-8"),
                    "74_inequality_distributional_lp_and_hank.py",
                    "exec",
                ),
                namespace,
            )
        finally:
            plt.close("all")

    scorecard = namespace.get("scorecard")
    assert scorecard is not None, "Notebook 74 must publish a scorecard"
    assert (scorecard["measured"] <= scorecard["bound"]).all()


def test_notebook_75_offline_execution_and_spatial_iam_mechanisms(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(ROOT / "notebooks"))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setattr(plt, "show", lambda: None)
    namespace = {"__name__": "__notebook_test__"}
    with matplotlib.rc_context():
        try:
            exec(
                compile(
                    (ROOT / "notebooks" / "75_green_transition_networks_and_spatial_iam.py").read_text(encoding="utf-8"),
                    "75_green_transition_networks_and_spatial_iam.py",
                    "exec",
                ),
                namespace,
            )
        finally:
            plt.close("all")

    scorecard = namespace.get("scorecard")
    assert scorecard is not None, "Notebook 75 must publish a scorecard"
    assert (scorecard["measured"] <= scorecard["bound"]).all()
