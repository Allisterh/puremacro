"""Comprehensive Unit Tests for Milestone 1: Result-Object Parity, Presentations, and Elevations.

Tests:
2. BVAR_SVForecast export methods (.to_markdown(), .to_latex(), .to_typst()).
3. ScoreDiagnosticsResult .to_typst() export.
4. Elevated VFISolution (.summary(), .to_frame(), export quintet, headless .plot()).
5. Elevated HJBSolution (.summary(), .to_frame(), export quintet, headless .plot(), and dict mapping protocol).
5b. Report quintet (.to_dataframe(), .to_markdown(), .to_latex(), .to_typst(), .summary()) of the result
    classes of dsge.stacked_newton, built from the smallest fixture the module's own tests use.
6. Deprecation warning update verification (retiring stale 2.0.0 references).
"""
from __future__ import annotations

import warnings
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro.dsge._gradients import ScoreDiagnosticsResult
from puremacro.var.bvar_sv import BVAR_SVForecast
from puremacro.vfi import HJBSolution, VFIProblem, VFISolution, solve_hjb_achdou


@pytest.fixture(autouse=True)
def close_figures():
    """Ensure all figures are closed after each test."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# 2. VAR and DSGE Export Parity
# ---------------------------------------------------------------------------

def test_bvar_sv_forecast_exports():
    forecast = BVAR_SVForecast(
        paths=np.ones((20, 4, 2)),
        h_paths=np.zeros((20, 4, 2)),
        index=pd.RangeIndex(4),
        names=["inflation", "rate"],
        history=pd.DataFrame(np.ones((10, 2)), columns=["inflation", "rate"]),
        ci=0.9,
    )
    md = forecast.to_markdown()
    assert isinstance(md, str)
    assert "inflation" in md
    assert "|" in md

    tex = forecast.to_latex()
    assert isinstance(tex, str)
    assert "\\begin{tabular}" in tex

    typ = forecast.to_typst()
    assert isinstance(typ, str)
    assert "#table(" in typ


def test_score_diagnostics_result_to_typst():
    res = ScoreDiagnosticsResult(
        loglik=-120.5,
        gradient=np.array([0.05, -0.12]),
        param_names=("alpha", "beta"),
        elapsed_sec=0.015,
    )
    typ = res.to_typst()
    assert isinstance(typ, str)
    assert "#table(" in typ
    assert "alpha" in typ
    assert "beta" in typ


# ---------------------------------------------------------------------------
# 3. Elevated VFISolution
# ---------------------------------------------------------------------------

def test_vfi_solution_elevation():
    sol = VFISolution(
        V=np.ones((15, 3)) * 4.2,
        policy_aprime=np.zeros((15, 3), dtype=int),
        policy_d=None,
        n_iter=12,
        sup_norm=5e-8,
        backend="numpy",
        endo_shape=(15,),
        a_grid=np.linspace(0.0, 10.0, 15),
        z_grid=np.array([0.5, 1.0, 1.5]),
    )

    # .summary()
    sm = sol.summary()
    assert isinstance(sm, pd.DataFrame)
    assert "Endogenous Grid Dimension (n_a)" in sm.index
    assert sm.loc["Total State Space", "Value"] == "45"
    assert sm.loc["Iterations", "Value"] == "12"

    # .to_frame()
    df = sol.to_frame()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 45
    assert "V" in df.columns
    assert "policy_aprime" in df.columns
    assert "a" in df.columns
    assert "z" in df.columns
    assert "aprime_val" in df.columns

    # formatters
    assert isinstance(sol.to_markdown(), str)
    assert "\\begin{tabular}" in sol.to_latex()
    assert "#table(" in sol.to_typst()

    # .plot()
    fig = sol.plot(show=False)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 2

    # plot with pre-allocated axes
    fig2, (ax1, ax2) = plt.subplots(1, 2)
    sol.plot(ax=(ax1, ax2), show=False)
    title_loc = plt.rcParams["axes.titlelocation"]
    assert ax1.get_title(loc=title_loc) == "Value Function V(a, z)"
    assert ax2.get_title(loc=title_loc) == "Policy Function a'(a, z)"


def test_vfi_problem_solve_returns_elevated_solution():
    """Verify that VFIProblem.solve populates a_grid and z_grid on VFISolution."""
    a_grid = np.linspace(0.1, 5.0, 10)
    z_grid = np.array([0.8, 1.2])
    P_z = np.array([[0.9, 0.1], [0.1, 0.9]])

    def ret_fn(ap, a, z, xp=np):
        c = z * (a ** 0.3) - ap
        return xp.where(c > 0, xp.log(xp.maximum(c, 1e-10)), -1e10)

    prob = VFIProblem(
        a_grid=a_grid,
        z_grid=z_grid,
        P_z=P_z,
        return_fn=ret_fn,
        beta=0.95,
        options={"max_iter": 50, "tol": 1e-5},
    )
    sol = prob.solve(backend="numpy")
    assert isinstance(sol, VFISolution)
    assert sol.a_grid is not None
    assert sol.z_grid is not None
    assert len(sol.a_grid) == 10
    assert len(sol.z_grid) == 2
    assert isinstance(sol.summary(), pd.DataFrame)
    assert len(sol.to_frame()) == 20


def test_vfi_solution_multi_asset_ragged_grids_to_frame_and_plot():
    """Multi-asset (and multi-shock) VFISolution with unequal grid lengths: to_frame() emits
    per-component state/policy columns and plot() falls back to the flat index axis,
    instead of raising on np.asarray of a ragged list."""
    a1 = np.linspace(0.0, 1.0, 4)
    a2 = np.linspace(0.0, 2.0, 3)
    z_grid = np.array([0.9, 1.1])
    P_z = np.array([[0.9, 0.1], [0.1, 0.9]])

    def ret_fn(ap1, ap2, x1, x2, z, xp=np):
        c = z + 1.02 * (x1 + x2) - ap1 - ap2
        return xp.where(c > 0, xp.log(xp.maximum(c, 1e-10)), -1e10)

    sol = VFIProblem(a_grid=[a1, a2], z_grid=z_grid, P_z=P_z, return_fn=ret_fn,
                     beta=0.9, options={"tol": 1e-6}).solve()
    assert sol.endo_shape == (4, 3)

    df = sol.to_frame()
    assert len(df) == 12 * 2
    assert {"a_idx", "z_idx", "a_0", "a_1", "z", "V", "policy_aprime", "aprime_0", "aprime_1"} <= set(df.columns)
    assert "a" not in df.columns and "aprime_val" not in df.columns
    # State columns unravel the flat endogenous index in C order (component 1 fastest)
    a_flat = df["a_idx"].to_numpy()
    np.testing.assert_array_equal(df["a_0"].to_numpy(), a1[a_flat // 3])
    np.testing.assert_array_equal(df["a_1"].to_numpy(), a2[a_flat % 3])
    # Policy columns agree with policy_components()
    c0, c1 = sol.policy_components()
    np.testing.assert_array_equal(df["aprime_0"].to_numpy(), a1[c0.ravel()])
    np.testing.assert_array_equal(df["aprime_1"].to_numpy(), a2[c1.ravel()])

    fig = sol.plot(show=False)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert fig.axes[0].get_xlabel() == "Asset Index (a_idx)"
    assert isinstance(sol.summary(), pd.DataFrame)
    assert "#table(" in sol.to_typst()

    # Multi-shock exogenous grids get z_<m> columns
    def ret_fn_z(ap, x, z1, z2, xp=np):
        c = z1 * z2 + x - ap
        return xp.where(c > 0, xp.log(xp.maximum(c, 1e-10)), -1e10)

    sol_z = VFIProblem(a_grid=np.linspace(0.0, 1.0, 5), z_grid=[np.array([0.9, 1.1]), np.array([0.5, 1.0, 1.5])],
                       P_z=np.full((6, 6), 1.0 / 6.0), return_fn=ret_fn_z, beta=0.9,
                       options={"tol": 1e-6}).solve()
    df_z = sol_z.to_frame()
    assert {"a", "z_0", "z_1", "aprime_val"} <= set(df_z.columns)
    assert len(df_z) == 5 * 6
    assert sol_z.plot(show=False) is not None


# ---------------------------------------------------------------------------
# 4. Elevated HJBSolution & solve_hjb_achdou
# ---------------------------------------------------------------------------

def test_hjb_solution_elevation_and_dict_compatibility():
    sol = solve_hjb_achdou(Na=30, max_iter=25)
    assert isinstance(sol, HJBSolution)

    # Dict mapping backward-compatibility checks
    assert sol["V"].shape == (30, 2)
    assert sol["c_policy"].shape == (30, 2)
    assert sol["s_drift"].shape == (30, 2)
    assert len(sol["a_grid"]) == 30
    assert len(sol["e_grid"]) == 2
    assert "V" in sol
    assert "c_policy" in sol
    assert "nonexistent_key" not in sol
    assert sol.get("n_iter") > 0
    assert sol.get("missing", 999) == 999
    assert len(sol.keys()) >= 7
    assert len(sol.values()) >= 7
    assert len(sol.items()) >= 7

    # Summary
    sm = sol.summary()
    assert isinstance(sm, pd.DataFrame)
    assert "Asset Grid Points (Na)" in sm.index
    assert sm.loc["Asset Grid Points (Na)", "Value"] == "30"
    assert sm.loc["Income Shock States (Ne)", "Value"] == "2"

    # Tabular DataFrame
    df = sol.to_frame()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 60
    assert set(df.columns) >= {"asset_a", "prod_e", "value_V", "consumption_c", "savings_drift_s"}

    # Formatters
    assert isinstance(sol.to_markdown(), str)
    assert "\\begin{tabular}" in sol.to_latex()
    assert "#table(" in sol.to_typst()

    # Plotting
    fig = sol.plot(show=False)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 3


# ---------------------------------------------------------------------------
# 4b. The stacked Newton-Krylov: the report quintet on real results
# ---------------------------------------------------------------------------
# Built from the smallest fixture the module's own tests use (an analytic growth model).
# Each result must render the quintet.


def _assert_quintet(obj, **frame_kwargs):
    frame = obj.to_dataframe(**frame_kwargs)
    assert isinstance(frame, pd.DataFrame) and not frame.empty
    for method in ("to_markdown", "to_latex", "to_typst"):
        text = getattr(obj, method)()
        assert isinstance(text, str) and text.strip(), method
    if hasattr(obj, "summary"):
        assert isinstance(obj.summary(), str) and obj.summary().strip()


def test_stacked_newton_results_quintet():
    from puremacro.dsge.stacked_newton import (HorizonComparison, StackedNewtonResult, StackedProblem,
                                               compare_horizons, solve_stacked_newton_krylov)

    alpha, beta, delta = 0.33, 0.96, 0.10
    k_ss = (alpha / (1 / beta - (1 - delta))) ** (1 / (1 - alpha))
    c_ss = k_ss ** alpha - delta * k_ss

    def equations_fn(y_plus, y_curr, y_lag, eps):
        euler = 1 / y_curr[0] - beta / y_plus[0] * (alpha * float(eps) * y_curr[1] ** (alpha - 1) + 1 - delta)
        resource = y_curr[1] - (float(eps) * y_lag[1] ** alpha + (1 - delta) * y_lag[1] - y_curr[0])
        return [euler, resource]

    y_ss, y_init = np.array([c_ss, k_ss]), np.array([c_ss, 0.5 * k_ss])
    short = solve_stacked_newton_krylov(StackedProblem(equations_fn, y_init, y_ss, np.ones(40),
                                                       variable_names=["c", "k"]), tol=1e-8)
    long = solve_stacked_newton_krylov(StackedProblem(equations_fn, y_init, y_ss, np.ones(80),
                                                      variable_names=["c", "k"]), tol=1e-8)
    assert isinstance(short, StackedNewtonResult)
    _assert_quintet(short)
    comparison = compare_horizons(short, long, periods=20, tolerance=1e-5)
    assert isinstance(comparison, HorizonComparison)
    _assert_quintet(comparison)


# ---------------------------------------------------------------------------
# 5. Stale Deprecation Warnings Retired
# ---------------------------------------------------------------------------

def test_garch_utils_deprecation_warning():
    import sys
    sys.modules.pop("puremacro.lp.garch_utils", None)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        import puremacro.lp.garch_utils  # noqa: F401
    matches = [w for w in record if "puremacro.lp.garch_utils" in str(w.message)]
    assert len(matches) > 0
    msg = str(matches[0].message)
    # 4.0.0 shipped (2026-09-16) keeping the shim; the target was retargeted to 5.0.0 on
    # 2026-09-22 and tests/test_docs_hygiene_audit_fixes.py guards it against drifting again.
    assert "a future major release (5.0.0)" in msg
    assert "2.0.0" not in msg and "4.0.0" not in msg


def test_sigma_numpy_deprecation_warning():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        from puremacro.sigma.sigma_numpy import SigmaObject
        SigmaObject(sigma=np.array([0.1, 0.2]), R=np.eye(2), labels=["A", "B"])
    matches = [w for w in record if "SigmaObject is deprecated" in str(w.message)]
    assert len(matches) > 0
    msg = str(matches[0].message)
    # 4.0.0 shipped keeping this shim too; retargeted to 5.0.0 with garch_utils (2026-09-23).
    assert "a future major release (5.0.0)" in msg
    assert "2.0.0" not in msg and "4.0.0" not in msg
