"""The distributional trade example runs end to end on synthetic household groups."""
import json

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from puremacro.examples.distributional_trade_ge import run_application, synthetic_household_groups


def test_synthetic_groups_are_consistent():
    data = synthetic_household_groups()
    categories = data.attrs["categories"]
    assert data.attrs["is_synthetic"] is True
    assert (data[categories].to_numpy() > 0).all()
    budget = data[categories].sum(axis=1)
    assert ((data.cash_wages > 0) & (data.cash_wages < budget)).all()
    shares = data["food"] / budget
    assert np.all(np.diff(shares) < 0)  # food's budget share falls across the groups


def test_example_reconciles_fiscal_and_income_ledgers(tmp_path):
    study = run_application(tmp_path, tariff_rate=0.10)
    assert study["baseline"].converged and study["counterfactual"].converged
    fiscal = study["fiscal_allocation"]
    assert len(fiscal) == 6
    np.testing.assert_allclose(fiscal["fiscal_residual"], 0.0, atol=1e-8)
    np.testing.assert_allclose(fiscal["income_residual"], 0.0, atol=1e-8)
    assert study["fiscal_allocation"]["incremental_tariff_revenue"].iloc[0] > 0
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["is_synthetic"] is True
    assert set(manifest["artifacts"]) >= {"report.md", "conditional_incidence.png", "equilibrium_evidence.npz"}
