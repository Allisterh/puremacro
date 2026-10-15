"""Staleness guard for the two-asset HANK section of showcase notebook 46.

The committed notebook 46 (EN and ES) stores executed outputs.  These tests
check that

* the portfolio adjustment-cost formula stated in the prose is the one the
  two-asset household block implements (checked numerically against a closed
  form written out here, independently of the library);
* the stored impact response of aggregate consumption to the expansionary
  monetary shock has the economically expected sign (C_0 > 0) and agrees with
  a fresh solve of the same call;
* the stored steady-state table agrees with a fresh solve.

A calibration change in ``hank_two_asset.mod`` or in the two-asset solver makes
these tests fail on purpose: re-execute the notebook with
``tools/build_notebooks.py`` and commit the new outputs.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

PROJ_ROOT = Path(__file__).resolve().parents[1]
NB_DIR = PROJ_ROOT / "notebooks"
STEM = "46_dsge_particle_filtering_and_markov_switching"
EN_IPYNB = NB_DIR / f"{STEM}.ipynb"
ES_IPYNB = NB_DIR / f"{STEM}_es.ipynb"
SOURCES = [
    NB_DIR / f"{STEM}.py",
    NB_DIR / f"{STEM}_es.py",
    EN_IPYNB,
    ES_IPYNB,
]

# The formula every notebook file must state (LaTeX, verbatim).
COST_LATEX = r"\frac{\chi_0}{1+\chi_1} \, \frac{|d|^{1+\chi_1}}{(a + \bar{a})^{\chi_1}}"
# Fragments of the Kaplan-Moll-Violante style formula the solver does NOT implement.
STALE_FRAGMENTS = [r"\chi_0 |d| +", r"\frac{\chi_1}{2}", r"\left(\frac{d}{a}\right)^2"]

C0_PATTERN = re.compile(r"\(C_0\)\s*:\s*([+-]?\d+\.\d+)")


def _closed_form_cost(d, a, chi_0, chi_1, a_bar):
    """chi(d, a) = chi_0/(1+chi_1) |d|^(1+chi_1) / (a + a_bar)^chi_1."""
    return chi_0 / (1.0 + chi_1) * np.abs(d) ** (1.0 + chi_1) / (a + a_bar) ** chi_1


def _read_text(path: Path) -> str:
    if path.suffix == ".ipynb":
        nb = json.loads(path.read_text(encoding="utf-8"))
        return "\n".join("".join(c["source"]) for c in nb["cells"])
    return path.read_text(encoding="utf-8")


def _two_asset_output(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] != "code" or "res_two_asset" not in "".join(cell["source"]):
            continue
        text = "".join(
            "".join(o.get("text", "")) for o in cell.get("outputs", []) if o.get("output_type") == "stream"
        )
        if C0_PATTERN.search(text):
            return text
    raise AssertionError(f"{path.name}: no stored two-asset output with a C_0 line")


@pytest.fixture(scope="module")
def fresh_solve():
    import puremacro.dsge as dsge
    from puremacro.dsge.hank import solve_hank_bridge

    mod = Path(dsge.__file__).parent / "_references" / "hank_two_asset.mod"
    return solve_hank_bridge(mod, shock="eps_m", magnitude=-0.0025, horizon=16)


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_cost_formula_text_matches_solver(path):
    text = _read_text(path)
    assert COST_LATEX in text, f"{path.name}: adjustment-cost formula missing or changed"
    for frag in STALE_FRAGMENTS:
        assert frag not in text, f"{path.name}: stale cost-formula fragment {frag!r}"


@pytest.mark.parametrize("chi_1", [1.0, 1.5, 2.0])
def test_solver_cost_equals_stated_formula(chi_1):
    from puremacro.models.hank_sequence_space import _transaction_cost, _two_asset_cost

    rng = np.random.default_rng(0)
    a = rng.uniform(0.0, 40.0, size=200)
    a_next = rng.uniform(0.0, 40.0, size=200)
    r_a, chi_0, a_bar = 0.0125, 1.0, 0.25
    d = a_next - (1.0 + r_a) * a
    expected = _closed_form_cost(d, a, chi_0, chi_1, a_bar)
    chi = _two_asset_cost(a_next, a, r_a, chi_0, chi_1, a_bar)[0]
    np.testing.assert_allclose(chi, expected, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(
        _transaction_cost(d, a, chi_0=chi_0, chi_1=chi_1, a_bar=a_bar), expected, rtol=1e-12, atol=1e-15
    )
    if chi_1 == 1.0:  # the quadratic special case the prose also states
        np.testing.assert_allclose(expected, 0.5 * chi_0 * d**2 / (a + a_bar), rtol=1e-12)


def test_reference_mod_is_quadratic_case():
    import puremacro.dsge as dsge

    mod = (Path(dsge.__file__).parent / "_references" / "hank_two_asset.mod").read_text()
    assert re.search(r"^\s*chi_1\s*=\s*1(\.0*)?\s*;", mod, flags=re.M), "prose states chi_1 = 1"


def test_en_es_cell_structure_identical():
    en = json.loads(EN_IPYNB.read_text(encoding="utf-8"))["cells"]
    es = json.loads(ES_IPYNB.read_text(encoding="utf-8"))["cells"]
    assert [c["cell_type"] for c in en] == [c["cell_type"] for c in es]


@pytest.mark.parametrize("path", [EN_IPYNB, ES_IPYNB], ids=lambda p: p.name)
def test_stored_impact_consumption_is_current(path, fresh_solve):
    stored = float(C0_PATTERN.search(_two_asset_output(path)).group(1))
    # Economic oracle: a -25 bp monetary shock is expansionary on impact.
    assert stored > 0.0, f"{path.name}: stored C_0 = {stored:+.4f} is not positive"
    fresh = float(fresh_solve.transition_paths["C"].to_numpy()[0])
    assert fresh > 0.0
    assert abs(stored - fresh) <= 5e-4, f"{path.name}: stored C_0 {stored:+.4f} vs fresh {fresh:+.4f}"


@pytest.mark.parametrize("path", [EN_IPYNB, ES_IPYNB], ids=lambda p: p.name)
def test_stored_steady_state_table_is_current(path, fresh_solve):
    text = _two_asset_output(path)
    table = fresh_solve.summary()
    for var in ("C", "D", "r_b", "r_a"):
        m = re.search(rf"^{re.escape(var)}\s+(-?\d+\.\d+)", text, flags=re.M)
        assert m, f"{path.name}: no steady-state row for {var}"
        assert abs(float(m.group(1)) - float(table.loc[var, "steady_state"])) <= 1e-3, (
            f"{path.name}: stored steady state of {var} is stale"
        )
