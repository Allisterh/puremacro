"""Adversarial numerical-precision tests of the trade calibration.

1. Macro balance / market-clearing supply-demand identity across all 840
   sector-country pairs of a synthetic 70-country 12-sector table.
2. Cost and value-added column share balance across the same pairs.
3. Robustness of masked division against inactive sectors, zeros and extreme
   factor ratios.

Runs offline on synthetic tables only.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from puremacro.trade.calibration import calibrate_trade_model
from puremacro.trade.data import synthetic_icio_table


@pytest.fixture(scope="module")
def reference_data():
    """A synthetic 70-country, 12-sector table and its calibration."""
    raw_data = synthetic_icio_table(70, 12, seed=2)
    calib = calibrate_trade_model(raw_data, nc=70, ns=12, nfd=3)
    return raw_data, calib


# ---------------------------------------------------------------------------
# 1. Macro balance across all sector-country pairs
# ---------------------------------------------------------------------------

class TestMacroBalanceAcrossAllPairs:
    """Leontief market-clearing and cost accounting identities across all 840 sector-country pairs."""

    def test_market_clearing_supply_equals_demand_all_pairs(self, reference_data):
        """Assert supply equals intermediate plus final demand within 1e-12 relative discrepancy."""
        raw_data, calib = reference_data
        nc, ns, nfd = 70, 12, 3
        n_ind = ns * nc

        # 1. Gross output / supply for all 840 sector-country pairs
        y_supply = calib.ytot.flatten(order="F")  # shape (840,)

        # 2. Intermediate demand across all 840 purchasing industries
        interm_demand = np.sum(calib.a * calib.ytot, axis=(1, 2))  # shape (840,)

        # 3. Final demand across all categories and destination countries
        # In baseline calibration:
        xfd_4d = raw_data[:n_ind, n_ind:].reshape(nc, ns, nc, nfd).transpose(1, 0, 3, 2)
        sum_xc = np.sum(xfd_4d.transpose(1, 0, 2, 3).reshape(n_ind, nfd, nc), axis=0, keepdims=True)
        final_demand = np.sum(calib.afd * sum_xc, axis=(1, 2))  # shape (840,)

        total_demand = interm_demand + final_demand
        residual = y_supply - total_demand

        # Relative discrepancy per sector-country pair
        rel_residual = np.abs(residual) / y_supply

        max_rel = float(np.max(rel_residual))
        mean_rel = float(np.mean(rel_residual))

        assert max_rel < 1e-12, (
            f"Market-clearing discrepancy across 840 pairs: max relative error {max_rel:.4e} exceeds 1e-12."
        )
        assert mean_rel < 1e-14, (
            f"Mean market-clearing relative discrepancy {mean_rel:.4e} exceeds 1e-14."
        )

        # Zero violations of 1e-12 across all 840 pairs
        violators = np.where(rel_residual > 1e-12)[0]
        assert len(violators) == 0, f"Found {len(violators)} pairs with relative residual > 1e-12: {violators}"

    def test_cost_value_added_column_share_balance_all_pairs(self, reference_data):
        """Assert total outlay shares (intermediates + taxes + labor + capital) sum to 1.0."""
        raw_data, calib = reference_data
        nc, ns = 70, 12
        n_ind = ns * nc

        # Sum of technical intermediate coefficients for each column (j, k)
        interm_shares = np.sum(calib.a, axis=0)  # shape (12, 70)
        tax_shares = calib.tax[0]                # shape (12, 70)

        l_data = raw_data[n_ind + 1, :n_ind].reshape(nc, ns).T
        k_data = raw_data[n_ind + 2, :n_ind].reshape(nc, ns).T
        y_mat = calib.ytot[0]

        active = y_mat > 0
        l_shares = np.zeros_like(y_mat)
        k_shares = np.zeros_like(y_mat)
        l_shares[active] = l_data[active] / y_mat[active]
        k_shares[active] = k_data[active] / y_mat[active]

        total_cost_shares = interm_shares + tax_shares + l_shares + k_shares
        cost_diff = np.abs(total_cost_shares[active] - 1.0)

        max_cost_diff = float(np.max(cost_diff))
        assert max_cost_diff < 1e-14, (
            f"Cost share adding-up discrepancy {max_cost_diff:.4e} exceeds 1e-14 across 840 pairs."
        )

    def test_world_current_account_zero_balance(self, reference_data):
        """Assert world current account (sum of invforT) equals zero to machine precision."""
        _, calib = reference_data
        world_ca = float(np.sum(calib.invforT))
        world_gdp = float(np.sum(calib.ytot))
        rel_ca = abs(world_ca) / world_gdp

        assert rel_ca < 1e-12, (
            f"Global current account balance {world_ca:.4e} (relative {rel_ca:.4e}) exceeds 1e-12."
        )


# ---------------------------------------------------------------------------
# 2. Masked Division & Adversarial Stress Tests
# ---------------------------------------------------------------------------

class TestMaskedDivisionAdversarialStress:
    """Adversarial stress-testing of masked division, zeros, and boundary conditions."""

    def test_inactive_sectors_warning_free_under_werror(self):
        """Verify multiple inactive sectors across multiple countries emit zero RuntimeWarnings."""
        nc, ns, nfd = 4, 3, 2
        n_ind = ns * nc
        toy_data = np.ones((n_ind + 3, n_ind + nfd * nc), dtype=np.float64) * 5.0

        # Entire country 1 inactive
        c1_slice = slice(ns * 1, ns * 2)
        toy_data[:, c1_slice] = 0.0
        toy_data[c1_slice, :] = 0.0
        toy_data[n_ind:, c1_slice] = 0.0
        toy_data[:, n_ind + nfd * 1 : n_ind + nfd * 2] = 0.0
        toy_data[n_ind, n_ind + nfd * 1 : n_ind + nfd * 2] = 0.0

        # Sector 0 in country 3 inactive
        c3_s0 = 0 + ns * 3
        toy_data[:, c3_s0] = 0.0
        toy_data[c3_s0, :] = 0.0
        toy_data[n_ind:, c3_s0] = 0.0

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            calib = calibrate_trade_model(toy_data, ns=ns, nc=nc, nfd=nfd, validate=False)

        runtime_warnings = [w for w in record if issubclass(w.category, RuntimeWarning)]
        assert len(runtime_warnings) == 0, f"Expected 0 RuntimeWarnings, got: {runtime_warnings}"

        # Inactive elements must be zero, not NaN or Inf
        assert not np.isnan(calib.ytot).any()
        assert not np.isnan(calib.alpha).any()
        assert not np.isnan(calib.beta).any()
        assert not np.isnan(calib.a).any()
        assert not np.isnan(calib.afd).any()
        assert not np.isnan(calib.tax).any()
        assert not np.isinf(calib.beta).any()

        # Inactive country has zero output and parameters
        assert np.all(calib.ytot[0, :, 1] == 0.0)
        assert np.all(calib.alpha[0, :, 1] == 0.0)
        assert np.all(calib.beta[0, :, 1] == 0.0)

    def test_extreme_capital_labor_ratios(self):
        """Verify calibration handles extreme capital/labor ratios without NaN or overflow."""
        nc, ns, nfd = 2, 2, 1
        n_ind = ns * nc
        toy_data = np.ones((n_ind + 3, n_ind + nfd * nc), dtype=np.float64) * 10.0

        # High capital intensity at (s=0, c=0): K = 1e10, L = 1e-5
        toy_data[n_ind + 1, 0] = 1e-5
        toy_data[n_ind + 2, 0] = 1e10

        # High labor intensity at (s=1, c=0): L = 1e10, K = 1e-5
        toy_data[n_ind + 1, 1] = 1e10
        toy_data[n_ind + 2, 1] = 1e-5

        calib = calibrate_trade_model(toy_data, ns=ns, nc=nc, nfd=nfd, validate=False)
        assert not np.isnan(calib.alpha).any()
        assert not np.isnan(calib.beta).any()
        assert 0.999 < calib.alpha[0, 0, 0] <= 1.0
        assert 0.0 <= calib.alpha[0, 1, 0] < 0.001
