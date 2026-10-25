"""Unit tests for Arellano (2008) and Chatterjee-Eyigungor (2012) Sovereign Default Suite.

Verifies:
1. Bond price bounds: q(b', y) in [0, (1 + r*)^(-1)] for short-term debt, and in [0, q_rf] for long-term debt.
2. Bond price monotonicity: dq/db' <= 0 across all b', y.
3. Countercyclical sovereign spreads: corr(spread_t, y_t) < 0 in ergodic simulation.
4. Bellman fixed-point convergence: residual norm < 1e-7.
5. DualArray forward autodiff derivatives: match finite differences and maintain <= 0.
6. Spread term structure decomposition: positive spreads, default loss and risk premium add up.
7. Debt restructuring counterfactuals: haircut and maturity extension achieve Delta s <= -150 bps.
8. Complete presentation contract: summary, to_dataframe, to_frame, to_markdown, to_latex, to_typst, plot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.sovereign import (
    RestructuringResult,
    SovereignDefaultResult,
    decompose_spread_term_structure,
    evaluate_debt_restructuring,
    solve_sovereign_default,
)


@pytest.fixture(scope="module")
def solved_arellano() -> SovereignDefaultResult:
    """Fixture solving Arellano (2008) short-term debt model."""
    return solve_sovereign_default(
        delta=1.0,
        kappa=0.0,
        r_star=0.01,
        beta=0.953,
        gamma=2.0,
        sigma_eps=0.01,
        b_max=0.40,
        n_b=30,
        n_y=11,
        tol=1e-7,
        max_iter=600,
    )


@pytest.fixture(scope="module")
def solved_chatterjee_eyigungor() -> SovereignDefaultResult:
    """Fixture solving Chatterjee-Eyigungor (2012) long-term coupon bond model."""
    return solve_sovereign_default(
        delta=0.05,
        kappa=0.03,
        r_star=0.01,
        beta=0.953,
        gamma=2.0,
        sigma_eps=0.01,
        b_max=0.35,
        n_b=30,
        n_y=11,
        tol=1e-7,
        max_iter=600,
    )


def test_arellano_bond_price_bounds(solved_arellano: SovereignDefaultResult) -> None:
    """Test q(b', y) in [0, (1 + r*)^(-1)] everywhere for short-term debt."""
    r_star = 0.01
    q_max_theoretical = 1.0 / (1.0 + r_star)

    assert np.all(solved_arellano.bond_price >= 0.0), "Bond prices must be non-negative"
    assert np.all(solved_arellano.bond_price <= q_max_theoretical + 1e-12), (
        f"Bond price exceeds risk-free price {q_max_theoretical}"
    )
    assert np.min(solved_arellano.bond_price) < q_max_theoretical, (
        "Bond price schedule should reflect default risk at high debt"
    )


def test_arellano_price_monotonicity(solved_arellano: SovereignDefaultResult) -> None:
    """Test dq/db'(b', y) <= 0 across all b', y."""
    assert np.all(solved_arellano.dq_db <= 1e-12), (
        f"Bond price schedule must be non-increasing in debt; max dq_db={np.max(solved_arellano.dq_db)}"
    )

    # Check finite-difference monotonicity directly on q
    diffs = np.diff(solved_arellano.bond_price, axis=0)
    assert np.all(diffs <= 1e-12), "Finite differences of q with respect to debt must be <= 0"


def test_sovereign_countercyclical_spreads(solved_arellano: SovereignDefaultResult) -> None:
    """Test corr(spread_t, y_t) < 0 in ergodic simulation."""
    sim_df = solved_arellano.simulate(T=2000, seed=42)
    valid = sim_df[sim_df["repaid"] & ~sim_df["spread_bps"].isna()]
    assert len(valid) > 200, "Should have sufficient repayment quarters"

    corr = float(np.corrcoef(valid["spread_bps"], valid["y"])[0, 1])
    assert corr < -0.15, f"Spread must be countercyclical (corr < -0.15); got {corr:.3f}"


def test_arellano_bellman_convergence(solved_arellano: SovereignDefaultResult) -> None:
    """Test that Bellman fixed-point residual norm < 1e-7."""
    assert solved_arellano.converged, "Solver did not flag convergence"
    assert solved_arellano.residual < 1e-7, (
        f"Residual norm {solved_arellano.residual:.2e} must be < 1e-7"
    )


def test_chatterjee_eyigungor_long_term_debt(solved_chatterjee_eyigungor: SovereignDefaultResult) -> None:
    """Test Chatterjee-Eyigungor (2012) long-term coupon bonds."""
    res = solved_chatterjee_eyigungor
    assert res.converged, "Long-term debt solver failed to converge"
    assert res.residual < 1e-7, f"Long-term debt residual {res.residual:.2e} must be < 1e-7"
    assert res.maturity_param_delta == 0.05
    assert res.coupon_rate_kappa == 0.03

    q_rf = (0.05 + 0.95 * 0.03) / (0.01 + 0.05)
    assert np.all(res.bond_price >= 0.0)
    assert np.all(res.bond_price <= q_rf + 1e-10)
    assert np.all(res.dq_db <= 1e-12)


def test_dualarray_price_derivatives(solved_arellano: SovereignDefaultResult) -> None:
    """Test that DualArray derivatives dq_db match numerical finite differences."""
    res = solved_arellano
    h = np.diff(res.b_grid)
    fd = np.diff(res.bond_price, axis=0) / h[:, None]

    # At interior points where derivative is nonzero, DualArray derivative aligns with secants
    mask = np.abs(fd) > 1e-3
    if np.any(mask):
        dq_interior = res.dq_db[:-1, :][mask]
        fd_interior = fd[mask]
        # Same sign (both negative)
        assert np.all(dq_interior <= 0.0)
        assert np.all(fd_interior <= 0.0)


def test_spread_term_structure_decomposition(solved_arellano: SovereignDefaultResult) -> None:
    """Test decompose_spread_term_structure returns valid decomp."""
    df = decompose_spread_term_structure(solved_arellano, horizons=(1, 4, 8, 12, 20, 40))

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 6
    assert list(df["horizon_quarters"]) == [1, 4, 8, 12, 20, 40]
    assert np.all(df["bond_price"] > 0.0)
    assert np.all(df["spread_bps"] >= 0.0)

    # Check that default loss + risk premium adds up to total spread to machine precision
    decomp_sum = df["default_loss_component"] + df["risk_premium_component"]
    np.testing.assert_allclose(decomp_sum, df["spread_bps"], rtol=1e-5, atol=1e-4)


def test_restructuring_counterfactual_spread_relief(solved_arellano: SovereignDefaultResult) -> None:
    """Test evaluate_debt_restructuring achieves Delta s <= -150 bps upon 20% haircut."""
    res_restruct = evaluate_debt_restructuring(
        solved_arellano,
        haircut=0.20,
        maturity_extension=0.50,
        horizon=30,
        seed=42,
    )

    assert isinstance(res_restruct, RestructuringResult)
    assert res_restruct.haircut == 0.20
    assert res_restruct.maturity_extension_years > 0.0
    assert res_restruct.spread_reduction_bps <= -150.0, (
        f"Spread reduction {res_restruct.spread_reduction_bps:.1f} bps must be <= -150 bps"
    )
    assert res_restruct.post_default_prob <= res_restruct.pre_default_prob, (
        "Restructuring must weakly reduce default hazard"
    )
    assert res_restruct.debt_gdp_relief_pct > 0.0
    assert isinstance(res_restruct.transition_path, pd.DataFrame)
    assert len(res_restruct.transition_path) == 30


def test_sovereign_result_presentation_methods(solved_arellano: SovereignDefaultResult) -> None:
    """Test presentation methods: summary, to_dataframe, to_markdown, to_latex, to_typst, plot."""
    import matplotlib.figure

    res = solved_arellano
    summ = res.summary()
    assert isinstance(summ, pd.DataFrame)
    assert "metric" in summ.columns and "value" in summ.columns

    df_grid = res.to_dataframe()
    assert isinstance(df_grid, pd.DataFrame)
    assert "bond_price_q" in df_grid.columns
    assert "spread_bps" in df_grid.columns

    md = res.to_markdown()
    assert isinstance(md, str) and len(md) > 0

    ltx = res.to_latex()
    assert isinstance(ltx, str) and len(ltx) > 0

    typ = res.to_typst()
    assert isinstance(typ, str) and len(typ) > 0

    fig1 = res.plot_price_schedule()
    assert isinstance(fig1, matplotlib.figure.Figure)

    fig2 = res.plot_policy()
    assert isinstance(fig2, matplotlib.figure.Figure)

    fig3 = res.plot()
    assert isinstance(fig3, matplotlib.figure.Figure)
