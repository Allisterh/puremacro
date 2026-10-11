"""Course lesson T02_F (Gertler-Karadi 2011) reports percent deviations.

``GertlerKaradiResult.to_frame()`` holds level deviations x_t - x_ss (the model
is linearised in levels); ``to_frame(units="pct")`` holds 100 (x_t - x_ss)/x_ss.
The lesson multiplied the level frame by 100 and labelled it "% dev."
(docs/ADVISORY.md, 2026-09-30), so it printed capital at -28.66% on impact
where the true figure is -5.06%.

Every expected percent below is a level deviation divided by its steady state,
times 100, at the lesson's calibration: capital -0.28663 / 5.66010 = -5.06%,
net worth -0.67884 / 1.38109 = -49.15%, investment -0.01005 / 0.14148 = -7.10%.
Q (steady state 1) and the credit spread, a rate shown in annualised basis
points (level x 40000), were already right and must stay so. The lesson's own
peak rule (min if |min| > |max|, else max) and quarter rule (argmax |x| + 1)
apply to the percent series.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CURSO = REPO / "curso" / "notebooks"
NOTEBOOKS = {
    "en": CURSO / "T02_F_gertler_karadi_macrofinance_local.ipynb",
    "es": CURSO / "T02_F_gertler_karadi_macrofinance_local_es.ipynb",
}

#: Default solve (OccBin, -5% capital-quality shock, nu_g = 10):
#: variable -> (impact %, trough/peak %, quarter of trough/peak, final % at t=40).
EXPECTED_SCORECARD = {
    "Y": (-1.60, -2.82, 4, -0.32),
    "I": (-7.10, -9.89, 3, 3.80),
    "C": (-0.65, -3.89, 14, -1.51),
    "Q": (-7.68, -7.68, 1, -0.02),
    "N": (-49.15, -49.15, 1, -7.44),
    "K": (-5.06, -13.49, 7, -3.52),
}
EXPECTED_SPREAD_BPS = (255.8, 255.8, 1, 20.2)

#: Source text that identifies the lesson's closing self-check cell.
SELF_CHECK_MARKER = {"en": "VERIFICATION ASSERTIONS PASSED", "es": "ASERCIONES DE VERIFICACI"}

#: Credit-policy regimes: (impact Y %, trough Y %, impact N %, spread bps,
#: cumulative Y loss as the sum of the 40 quarterly percent deviations, active quarters).
EXPECTED_POLICY = {
    0: (-2.15, -3.63, -61.72, 603.7, -44.06, 11),
    10: (-1.60, -2.82, -49.15, 255.8, -43.99, 14),
    20: (-1.49, -2.65, -46.87, 192.9, -43.95, 15),
}

_NUM = r"([+-]?\d+\.\d+)"
_SCORE_ROW = re.compile(
    rf"^\s*(Y|I|C|Q|N|K)\s+% (?:dev|desv)\.\s+{_NUM}%\s+{_NUM}%\s+[QT](\d+)\s+{_NUM}%\s*$"
)
_SPREAD_ROW = re.compile(
    rf"^\s*(?:Credit Spread|Spread de Crédito) \(prem\)\s+bps (?:ann|anual)\.\s+{_NUM}\s+{_NUM}\s+[QT](\d+)\s+{_NUM}\s*$"
)
_POLICY_ROW = re.compile(
    rf"^\s*\S+ \(nu_g = (\d+)\)\s+{_NUM}%\s+{_NUM}%\s+{_NUM}%\s+\+{_NUM}\s+{_NUM}%?\s+(\d+)\s*$"
)


def _stdout_of_cell(nb: dict, marker: str) -> str:
    """Stream output of the one code cell whose source contains ``marker``."""
    cells = [c for c in nb["cells"] if c["cell_type"] == "code" and marker in "".join(c["source"])]
    assert len(cells) == 1, f"expected one code cell containing {marker!r}, found {len(cells)}"
    return "".join(
        "".join(o.get("text", "")) for o in cells[0].get("outputs", []) if o.get("output_type") == "stream"
    )


def _assert_scorecard(nb: dict) -> None:
    text = _stdout_of_cell(nb, "scorecard_vars")
    rows = {m.group(1): m.groups()[1:] for m in map(_SCORE_ROW.match, text.splitlines()) if m}
    assert set(rows) == set(EXPECTED_SCORECARD), f"scorecard rows found: {sorted(rows)}\n{text}"
    for var, (impact, peak, quarter, final) in EXPECTED_SCORECARD.items():
        got = rows[var]
        assert float(got[0]) == pytest.approx(impact, abs=0.006), (var, "impact", got)
        assert float(got[1]) == pytest.approx(peak, abs=0.006), (var, "trough/peak", got)
        assert int(got[2]) == quarter, (var, "quarter", got)
        assert float(got[3]) == pytest.approx(final, abs=0.006), (var, "final", got)
    spread = [m.groups() for m in map(_SPREAD_ROW.match, text.splitlines()) if m]
    assert len(spread) == 1, text
    impact, peak, quarter, final = spread[0]
    assert (float(impact), float(peak), int(quarter), float(final)) == pytest.approx(
        EXPECTED_SPREAD_BPS, abs=0.06
    )


def _assert_policy_table(nb: dict) -> None:
    text = _stdout_of_cell(nb, "loss_table = pd.DataFrame")
    rows = {int(m.group(1)): m.groups()[1:] for m in map(_POLICY_ROW.match, text.splitlines()) if m}
    assert set(rows) == set(EXPECTED_POLICY), f"policy rows found: {sorted(rows)}\n{text}"
    for nu_g, expected in EXPECTED_POLICY.items():
        got = rows[nu_g]
        assert [float(x) for x in got[:5]] == pytest.approx(list(expected[:5]), abs=0.06), (nu_g, got)
        assert int(got[5]) == expected[5], (nu_g, "active quarters", got)


@pytest.mark.parametrize("lang", sorted(NOTEBOOKS))
def test_saved_scorecard_reports_percent_deviations(lang):
    nb = json.loads(NOTEBOOKS[lang].read_text(encoding="utf-8"))
    _assert_scorecard(nb)


@pytest.mark.parametrize("lang", sorted(NOTEBOOKS))
def test_saved_policy_table_reports_percent_deviations(lang):
    nb = json.loads(NOTEBOOKS[lang].read_text(encoding="utf-8"))
    _assert_policy_table(nb)


@pytest.mark.slow
@pytest.mark.parametrize("lang", sorted(NOTEBOOKS))
def test_lesson_executes_and_reports_percent_deviations(lang, tmp_path, monkeypatch):
    """Run the lesson in a fresh kernel against this checkout and read what it prints."""
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")

    # A throwaway kernelspec for this interpreter, so the user's own `python3`
    # kernelspec is neither used nor rewritten.
    spec = tmp_path / "kernels" / "puremacro-t02f"
    spec.mkdir(parents=True)
    spec.joinpath("kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "puremacro-t02f",
        "language": "python",
    }))
    monkeypatch.setenv("JUPYTER_PATH", str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(REPO), os.environ.get("PYTHONPATH", "")]))
    monkeypatch.setenv("MPLBACKEND", "Agg")

    nb = nbformat.read(NOTEBOOKS[lang], as_version=4)
    nbclient.NotebookClient(
        nb, kernel_name="puremacro-t02f", timeout=900, resources={"metadata": {"path": str(CURSO)}}
    ).execute()

    _assert_scorecard(nb)
    _assert_policy_table(nb)
    checks = _stdout_of_cell(nb, SELF_CHECK_MARKER[lang])
    assert re.search(r"(PASSED|PASARON) \(\d+/\d+\)", checks), checks
