"""The bilingual study tutorial executes its assertions and ships current outputs."""
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
import pandas as pd
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[2]
STEMS = ("69_reproducible_studies", "69_reproducible_studies_es")


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


def test_bilingual_study_computations_and_saved_sources_match():
    sources = [_source_code(ROOT / "notebooks" / (stem + ".py")) for stem in STEMS]
    assert len(sources[0]) == 11 and sources[0] == sources[1]
    for stem, expected in zip(STEMS, sources):
        notebook = json.loads((ROOT / "notebooks" / (stem + ".ipynb")).read_text(encoding="utf-8"))
        actual = [_text(cell["source"]).strip() for cell in notebook["cells"]
                  if cell["cell_type"] == "code" and _text(cell["source"]).strip()]
        assert actual == expected
        # Require saved prose to match as well; outputs alone do not certify current claims.
        actual_cells = [(cell["cell_type"], _text(cell["source"]).strip())
                        for cell in notebook["cells"] if _text(cell["source"]).strip()]
        assert actual_cells == _source_cells(ROOT / "notebooks" / (stem + ".py"))
        source = (ROOT / "notebooks" / (stem + ".py")).read_text(encoding="utf-8")
        assert 'puremacro[notebooks]==5.6.0' in source
        for block in actual:
            ast.parse(block)


@pytest.mark.parametrize("stem", STEMS)
def test_saved_study_outputs_are_executed_and_figures_opaque(stem):
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
    assert counts == list(range(1, 12)) and len(images) == 3
    transcript = "\n".join(text)
    assert "all archive round-trips passed" in transcript
    assert "Incompatible comparison rejected:" in transcript
    assert "Response scaling verified for a 50 basis-point" in transcript


def test_study_tutorial_runs_offline_and_exercise_corners_hold(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(ROOT / "notebooks"))
    namespace = {"__name__": "__notebook_test__"}
    # Execute the actual source, including its independent resource identity,
    # complete archive checks, seeded refit and incompatible-comparison control.
    with matplotlib.rc_context():
        try:
            exec(compile((ROOT / "notebooks" / (STEMS[0] + ".py")).read_text(encoding="utf-8"),
                         STEMS[0] + ".py", "exec"), namespace)
        finally:
            plt.close("all")
    assert not list(tmp_path.iterdir()), "The default tutorial must leave no persistent exports"
    mexico = namespace["mexico"]
    assert mexico.audits.set_index("check").loc["identification", "status"] == "not_checked"
    # The provenance assertion must inspect attrs: frame equality alone omits them.
    recovered = namespace["load_study"](namespace["archives"]["mexico"])
    assert recovered.inputs.attrs["source"] == mexico.inputs.attrs["source"]
    assert recovered.inputs.attrs["source"]["data_rights"]
    exercise = _source_code(ROOT / "notebooks" / (STEMS[0] + ".py"))[-1]
    for shock in (0.10, 0.50):
        exec(compile(exercise.replace("shock_pp = 0.50", f"shock_pp = {shock}"),
                     "shock_exercise", "exec"), namespace)
        point = namespace["scaled"].to_frame().set_index(["response", "horizon"])
        assert point.loc[("rate_3m", 0), "estimate"] == pytest.approx(shock)
    for original in namespace["records"].values():
        assert np.isfinite(original.to_frame().estimate).all()
        assert not original.audits.empty
    pd.testing.assert_frame_equal(namespace["repeat"].to_frame(), mexico.to_frame(), check_exact=True)
