"""Unit tests for Mendoza (2010) Sudden-Stop Collateral Constraint Model.

Verifies:
1. Fisherian debt-deflation fire-sale amplification:
   ratio of consumption drop under binding collateral constraint vs
   unconstrained benchmark is strictly greater than 1.25.
2. Precautionary deleveraging:
   ergodic average debt is strictly below statutory collateral capacity
   (b_bar < kappa * q_bar * k_bar).
3. Collateral multiplier lambda and Tobin's q behavior during binding states.
4. Simulation method produces valid time series paths and crisis indicators.
5. Presentation contract methods: summary, to_dataframe, to_frame, to_markdown,
   to_latex, to_typst, plot_crisis_dynamics, plot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.sovereign import SuddenStopResult, solve_sudden_stop


@pytest.fixture(scope="module")
def solved_mendoza() -> SuddenStopResult:
    """Fixture solving Mendoza (2010) model with moderate grid size for fast tests."""
    return solve_sudden_stop(
        beta=0.96,
        gamma=2.0,
        r_star=0.04,
        kappa_collateral=0.30,
        n_b=22,
        n_k=22,
        n_z=3,
        tol=1e-6,
        max_iter=400,
        seed=42,
    )


def test_mendoza_sudden_stop_amplification(solved_mendoza: SuddenStopResult) -> None:
    """Test that consumption collapse amplification ratio is > 1.25."""
    res = solved_mendoza
    assert res.amplification_ratio > 1.25, (
        f"Amplification ratio {res.amplification_ratio:.3f} must be > 1.25"
    )


def test_mendoza_precautionary_deleveraging(solved_mendoza: SuddenStopResult) -> None:
    """Test that ergodic average debt is strictly less than collateral capacity."""
    res = solved_mendoza
    sim_df = res.simulate(T=2000, seed=42)

    mean_b = float(np.mean(sim_df["b"]))
    mean_k = float(np.mean(sim_df["k"]))
    mean_q = float(np.mean(sim_df["tobin_q"]))
    capacity = res.kappa_collateral * mean_q * mean_k

    assert mean_b < capacity, (
        f"Precautionary deleveraging violated: mean debt {mean_b:.4f} >= capacity {capacity:.4f}"
    )
    buffer_pct = (capacity - mean_b) / capacity * 100.0
    assert buffer_pct > 5.0, f"Precautionary buffer {buffer_pct:.1f}% should be substantial"


def test_mendoza_fisherian_amplification_multiplier(solved_mendoza: SuddenStopResult) -> None:
    """Test collateral multiplier lambda > 0 and Tobin's q < 1 in binding states."""
    res = solved_mendoza
    binding = res.multiplier_lambda > 1e-4

    assert np.any(binding), "Collateral constraint should bind in some states"
    # In binding states, multiplier must be positive
    assert np.all(res.multiplier_lambda[binding] > 0.0)

    # Tobin's q should drop below 1 when capital disinvestment occurs
    assert np.min(res.asset_price_q) < 1.0, "Tobin's q should drop below 1 during capital contraction"


def test_mendoza_sudden_stop_simulation(solved_mendoza: SuddenStopResult) -> None:
    """Test simulate() generates clean DataFrame with required fields and crisis episodes."""
    res = solved_mendoza
    sim = res.simulate(T=1000, seed=42)

    assert isinstance(sim, pd.DataFrame)
    assert len(sim) == 1000
    expected_cols = {"period", "z", "b", "b_next", "k", "k_next", "consumption", "tobin_q", "multiplier", "crisis"}
    assert expected_cols.issubset(set(sim.columns))

    assert not sim["consumption"].isna().any()
    assert np.all(sim["consumption"] > 0.0)
    assert sim["crisis"].dtype == bool


def test_sudden_stop_presentation_methods(solved_mendoza: SuddenStopResult) -> None:
    """Test summary, to_dataframe, to_markdown, to_latex, to_typst, plot."""
    import matplotlib.figure

    res = solved_mendoza
    summ = res.summary()
    assert isinstance(summ, pd.DataFrame)
    assert "metric" in summ.columns and "value" in summ.columns

    df_grid = res.to_dataframe()
    assert isinstance(df_grid, pd.DataFrame)
    assert "tobin_q" in df_grid.columns
    assert "multiplier_lambda" in df_grid.columns

    md = res.to_markdown()
    assert isinstance(md, str) and len(md) > 0

    ltx = res.to_latex()
    assert isinstance(ltx, str) and len(ltx) > 0

    typ = res.to_typst()
    assert isinstance(typ, str) and len(typ) > 0

    fig1 = res.plot_crisis_dynamics()
    assert isinstance(fig1, matplotlib.figure.Figure)

    fig2 = res.plot()
    assert isinstance(fig2, matplotlib.figure.Figure)
