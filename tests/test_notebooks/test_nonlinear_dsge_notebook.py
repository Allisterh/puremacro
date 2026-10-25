"""The bilingual nonlinear DSGE workflows showcase executes its assertions and ships current outputs."""
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
STEMS = ("70_nonlinear_dsge_workflows_and_accuracy", "70_nonlinear_dsge_workflows_and_accuracy_es")


def _source_cells(path):
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"^# %%(.*)\n", text, flags=re.MULTILINE)
    cells = []
    for i in range(1, len(parts), 2):
        body = parts[i + 1].strip()
        kind = "markdown" if "[markdown]" in parts[i] else "code"
        if kind == "markdown":
            body = "\n".join(line[2:] if line.startswith("# ") else "" if line == "#" else line
                             for line in body.splitlines()).strip()
        if body:
            cells.append((kind, body))
    return cells


def _source_code(path):
    return [source for kind, source in _source_cells(path) if kind == "code"]


def _text(source):
    return "".join(source) if isinstance(source, list) else source


def test_bilingual_nonlinear_dsge_computations_and_saved_sources_match():
    sources = [_source_code(ROOT / "notebooks" / (stem + ".py")) for stem in STEMS]
    assert len(sources[0]) == 7 and sources[0] == sources[1]
    for stem, expected in zip(STEMS, sources):
        notebook = json.loads((ROOT / "notebooks" / (stem + ".ipynb")).read_text(encoding="utf-8"))
        actual = [_text(cell["source"]).strip() for cell in notebook["cells"]
                  if cell["cell_type"] == "code" and _text(cell["source"]).strip()]
        assert actual == expected
        actual_cells = [(cell["cell_type"], _text(cell["source"]).strip())
                        for cell in notebook["cells"] if _text(cell["source"]).strip()]
        assert actual_cells == _source_cells(ROOT / "notebooks" / (stem + ".py"))
        for block in actual:
            ast.parse(block)


@pytest.mark.parametrize("stem", STEMS)
def test_saved_nonlinear_dsge_outputs_are_executed_and_figures_opaque(stem):
    notebook = json.loads((ROOT / "notebooks" / (stem + ".ipynb")).read_text(encoding="utf-8"))
    images, counts, text = [], [], []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        if _text(cell["source"]).strip():
            counts.append(cell["execution_count"])
        for output in cell["outputs"]:
            assert output["output_type"] != "error"
            if "text" in output:
                text.append(_text(output["text"]))
            if "image/png" in output.get("data", {}):
                raw = base64.b64decode(_text(output["data"]["image/png"]))
                with Image.open(io.BytesIO(raw)) as image:
                    rgba = np.array(image.convert("RGBA"))
                assert rgba.shape[0] >= 300 and rgba.shape[1] >= 600
                assert np.all(rgba[:, :, 3] == 255)
                images.append(raw)
    assert counts == list(range(1, 8)) and len(images) == 1
    transcript = "\n".join(text)
    assert "collocation_matrix_allocated=False" in transcript
    assert "dense_jvp_blocks = 0" in transcript
    assert "All 11 nonlinear DSGE workflow and solver checks pass." in transcript


def test_nonlinear_dsge_showcase_runs_offline_and_exercise_corners_hold(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(ROOT / "notebooks"))
    namespace = {"__name__": "__notebook_test__"}
    with matplotlib.rc_context():
        try:
            exec(compile((ROOT / "notebooks" / (STEMS[0] + ".py")).read_text(encoding="utf-8"),
                         STEMS[0] + ".py", "exec"), namespace)
        finally:
            plt.close("all")
    scorecard = namespace["scorecard"]
    assert len(scorecard) == 11
    assert (scorecard["measured"] <= scorecard["bound"]).all()
    assert not namespace["ti_analytic"].policy.collocation_matrix_allocated
    assert namespace["nk_dense_blocks"] == 0

    exercise = _source_code(ROOT / "notebooks" / (STEMS[0] + ".py"))[-1]
    for deg, shock in ((4, -0.015), (8, -0.035)):
        mod_code = (
            exercise
            .replace("cheb_order = 6", f"cheb_order = {deg}")
            .replace("elb_shock_size = -0.025", f"elb_shock_size = {shock}")
        )
        exec(compile(mod_code, "nonlinear_dsge_exercise", "exec"), namespace)
        assert namespace["ti_user"].converged
        assert not namespace["ti_user"].policy.collocation_matrix_allocated
        assert namespace["user_gap"] < 1e-8


def test_bilingual_notebook_68_sylvester_diagnostics_upgrade():
    stems_68 = ("68_third_order_perturbation_trust", "68_third_order_perturbation_trust_es")
    sources = [_source_code(ROOT / "notebooks" / (s + ".py")) for s in stems_68]
    assert sources[0] == sources[1]
    for s, expected in zip(stems_68, sources):
        nb = json.loads((ROOT / "notebooks" / (s + ".ipynb")).read_text(encoding="utf-8"))
        actual = [_text(c["source"]).strip() for c in nb["cells"]
                  if c["cell_type"] == "code" and _text(c["source"]).strip()]
        assert actual == expected
        transcript = "\n".join(
            _text(out["text"])
            for c in nb["cells"] if c["cell_type"] == "code"
            for out in c["outputs"] if "text" in out
        )
        assert "lu_cache_hits" in "\n".join(actual)
        assert "ghxxx" in transcript and "Sylvester algebraic residuals" in transcript
        assert "Internal algebra: RBC Sylvester residuals (ghxx, ghs2, ghxxx, ghxss, ghuss)" in transcript
