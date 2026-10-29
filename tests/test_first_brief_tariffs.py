"""The first policy brief: a 10% US tariff on Chinese goods, national and household welfare."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from puremacro.examples import first_brief_tariffs as fb


@pytest.fixture(scope="module")
def solved():
    """One default solve (about 6 s on a quiet machine) shared by the module."""
    return fb.solve_brief()


# --- Task 1: solve stage and national record ---------------------------------

def test_parse_experiment():
    assert fb.parse_experiment("none") is None
    assert fb.parse_experiment("sigma=2") == 2.0
    assert fb.parse_experiment("sigma=0.5") == 0.5
    for bad in ("theta=4", "sigma", "sigma=-1", "sigma=nan", "sigma=0"):
        with pytest.raises(ValueError):
            fb.parse_experiment(bad)


@pytest.mark.parametrize(
    ("importer", "exporter", "sector", "tariff", "offending"),
    [
        ("USA", "USA", "all", 0.1, "USA"),
        ("XXX", "CHN", "all", 0.1, "XXX"),
        ("USA", "XXX", "all", 0.1, "XXX"),
        ("USA", "CHN", "Z", 0.1, "Z"),
        ("USA", "CHN", "all", -0.1, "-0.1"),
        ("USA", "CHN", "all", math.nan, "nan"),
    ],
)
def test_validate_scenario_rejects(importer, exporter, sector, tariff, offending):
    icio, _ = fb.load_bundled()
    with pytest.raises(ValueError, match=offending):
        fb.validate_scenario(icio, importer, exporter, sector, tariff)


def test_default_solution_converged_and_split_adds_up(solved):
    assert solved.base.converged and solved.counterfactual.converged
    rec = fb.national_record(solved)
    total = (rec["price_effect_musd"] + rec["factor_income_effect_musd"]
             + rec["fiscal_transfer_effect_musd"] + rec["decomposition_residual_musd"])
    assert total == pytest.approx(rec["ev_musd"], rel=1e-9)
    assert abs(rec["decomposition_residual_musd"]) < 1e-4
    assert rec["ev_pct_consumption"] > 0
    assert solved.settings["sigma"] == 0.0
    for key, value in fb.SOLVER.items():
        assert solved.settings[key] == value
    assert set(solved.timings) == {"load", "calibration", "baseline", "counterfactual", "welfare"}


def test_nonconvergence_raises(monkeypatch):
    monkeypatch.setattr(
        fb, "run_scenario",
        lambda *a, **k: SimpleNamespace(converged=False, metadata={"solver_termination": "max_iter"}))
    with pytest.raises(RuntimeError, match="counterfactual did not converge.*max_iter"):
        fb.solve_brief()
