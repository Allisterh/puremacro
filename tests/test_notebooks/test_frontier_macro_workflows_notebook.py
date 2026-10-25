"""The bilingual frontier macroeconomic workflows showcase executes its assertions and ships current outputs."""
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

from puremacro.trade import baqaee_farhi_decomposition
from puremacro.dsge import solve_nonlinear_occbin

ROOT = Path(__file__).resolve().parents[2]
STEMS = ("71_frontier_macro_workflows", "71_frontier_macro_workflows_es")


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


def test_bilingual_frontier_macro_workflows_computations_and_saved_sources_match():
    sources = [_source_code(ROOT / "notebooks" / (stem + ".py")) for stem in STEMS]
    assert len(sources[0]) >= 8
    # Exact 1-to-1 code block parity between English and Spanish editions
    assert sources[0] == sources[1]

    for stem, expected in zip(STEMS, sources):
        nb_path = ROOT / "notebooks" / (stem + ".ipynb")
        assert nb_path.exists(), f"Missing compiled notebook: {nb_path}"
        notebook = json.loads(nb_path.read_text(encoding="utf-8"))
        actual = [
            _text(cell["source"]).strip()
            for cell in notebook["cells"]
            if cell["cell_type"] == "code" and _text(cell["source"]).strip()
        ]
        assert actual == expected
        for block in actual:
            ast.parse(block)


@pytest.mark.parametrize("stem", STEMS)
def test_saved_frontier_macro_workflows_outputs_are_executed_and_figures_opaque(stem):
    nb_path = ROOT / "notebooks" / (stem + ".ipynb")
    notebook = json.loads(nb_path.read_text(encoding="utf-8"))
    images, counts, text = [], [], []

    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        if _text(cell["source"]).strip():
            counts.append(cell["execution_count"])
        for output in cell["outputs"]:
            assert output["output_type"] != "error", f"Error output in {stem}: {output}"
            if "text" in output:
                text.append(_text(output["text"]))
            if "image/png" in output.get("data", {}):
                raw = base64.b64decode(_text(output["data"]["image/png"]))
                with Image.open(io.BytesIO(raw)) as image:
                    rgba = np.array(image.convert("RGBA"))
                assert rgba.shape[0] >= 200 and rgba.shape[1] >= 600
                assert np.all(rgba[:, :, 3] == 255), f"Found transparent canvas in {stem}"
                images.append(raw)

    assert len(counts) >= 8 and None not in counts
    assert len(images) == 1
    transcript = "\n".join(text)
    assert "PASS (6/6 verified)" in transcript
    assert "All 16 frontier macroeconomic workflow checks pass." in transcript


def test_frontier_macro_workflows_runs_offline_and_exercise_corners_hold(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(ROOT / "notebooks"))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setattr(plt, "show", lambda: None)
    namespace = {"__name__": "__notebook_test__"}
    with matplotlib.rc_context():
        try:
            exec(
                compile(
                    (ROOT / "notebooks" / (STEMS[0] + ".py")).read_text(encoding="utf-8"),
                    STEMS[0] + ".py",
                    "exec",
                ),
                namespace,
            )
        finally:
            plt.close("all")

    scorecard = namespace["scorecard"]
    assert len(scorecard) == 16
    assert (scorecard["measured"] <= scorecard["bound"]).all()
    assert namespace["res_studio"].pass_count == 6
    assert namespace["res_studio"].fail_count == 0

    # Exercise corner validation: test boundaries of advertised range
    Omega_bf = namespace["Omega_bf"]
    b_bf = namespace["b_bf"]
    z_user = np.array([-0.08, 0.0, 0.0], dtype=np.float64)
    m_ref = namespace["m_ref"]
    m_zlb = namespace["m_zlb"]
    constraint_zlb = namespace["constraint_zlb"]

    for test_elasticity in (0.20, 0.50, 1.00):
        res_bf = baqaee_farhi_decomposition(
            Omega_bf,
            b_bf,
            elasticities=test_elasticity,
            tfp_shocks=z_user,
            final_elasticity=test_elasticity,
        )
        assert res_bf.second_order_dlog_y <= 1e-14
        assert -0.01 <= res_bf.second_order_dlog_y <= 1e-14

    for test_shock in (-0.02, -0.05, -0.08):
        res_occ = solve_nonlinear_occbin(
            m_ref,
            m_zlb,
            constraint_zlb,
            np.array([test_shock]),
            order=2,
            sigma=1.0,
            horizon=25,
        )
        assert res_occ.converged
        assert 1 <= res_occ.binding_periods <= 15
        assert res_occ.max_bound_violation < 1e-10
