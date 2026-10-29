"""Paired-seed grid ladder for the illustrative fiscal calibration and its frozen record."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GRID_RECORD = ROOT / "tests" / "fixtures" / "sovereign_fiscal" / "fiscal_grid_convergence.json"


def test_fiscal_grid_check_uses_the_public_certified_solver(monkeypatch):
    from tools.reference_validation import check_fiscal_grids as grids

    calls = []

    def recorder(**kwargs):
        calls.append(kwargs)
        return "solution"

    monkeypatch.setattr(grids, "solve_fiscal_sovereign_default", recorder)
    assert grids.solve_case(1601, 321) == "solution"
    assert calls == [dict(grids.ILLUSTRATIVE, n_b=1601, n_y=321, policy_search="monotone")]
    assert grids.ILLUSTRATIVE == dict(tau=.2, psi=1., gamma_c=2., gamma_g=2.)


def test_fiscal_tolerances_cover_the_registered_moments():
    from puremacro.sovereign._fiscal_paths import MOMENTS
    from tools.reference_validation.check_fiscal_grids import TOLERANCES

    assert list(TOLERANCES) == [m for m in MOMENTS if m != "boundary_choice_share_pct"]
    assert TOLERANCES["corr_pb_y"] == .02
    assert all(v == .10 for k, v in TOLERANCES.items() if not k.startswith("corr_"))


def test_paired_comparison_accepts_custom_tolerances():
    from tools.reference_validation.check_arellano_grids import paired_comparison

    def runs(shift):
        return dict(runs=[dict(seed=s, moments=dict(debt_pct=1. + shift + .001 * s, corr_pb_y=.3 + shift))
                          for s in (1, 2, 3)])

    result = paired_comparison(runs(0.), runs(.05), tolerances=dict(debt_pct=.10, corr_pb_y=.02))
    assert set(result["moments"]) == {"debt_pct", "corr_pb_y"}
    assert result["moments"]["debt_pct"]["status"] == "passed"
    assert result["moments"]["corr_pb_y"]["status"] == "grid_shift_exceeds_tolerance"
    assert not result["passed"]
