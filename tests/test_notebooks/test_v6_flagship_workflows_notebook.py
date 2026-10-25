"""The bilingual v6.0.0 flagship macroeconomic workflows showcase executes its assertions and ships current outputs."""
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

from puremacro.sovereign import evaluate_debt_restructuring
from puremacro.climate.spatial_iam import solve_spatial_climate_iam

ROOT = Path(__file__).resolve().parents[2]
STEMS = ("72_macro_v6_flagship_workflows", "72_macro_v6_flagship_workflows_es")


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


def test_bilingual_v6_flagship_workflows_computations_and_saved_sources_match():
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
def test_saved_v6_flagship_workflows_outputs_are_executed_and_figures_opaque(stem):
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
    assert "All 14 v6.0.0 flagship macroeconomic workflow checks pass." in transcript


def test_v6_flagship_workflows_runs_offline_and_exercise_corners_hold(tmp_path, monkeypatch):
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
    assert len(scorecard) == 14
    assert (scorecard["measured"] <= scorecard["bound"]).all()

    # Exercise corner validation: test boundaries of advertised range
    model_iam = namespace["model_iam"]
    res_sov = namespace["res_sov"]

    for test_tax in (10.0, 45.0, 100.0):
        res_iam = solve_spatial_climate_iam(
            model_iam,
            horizon=12,
            policy_scenario="cbam",
            carbon_tax_path=np.full(12, test_tax),
            coalition_regions=[0],
        )
        assert 0.0 <= res_iam.carbon_leakage_rate <= 1.0
        assert res_iam.population_conservation_error < 1e-12
        assert res_iam.carbon_mass_balance_error < 1e-12

    for test_haircut in (0.05, 0.20, 0.40):
        restruct = evaluate_debt_restructuring(
            res_sov,
            haircut=test_haircut,
            maturity_extension=0.0,
        )
        assert restruct.spread_reduction_bps <= 0.0
        assert restruct.post_default_prob <= restruct.pre_default_prob + 1e-6
