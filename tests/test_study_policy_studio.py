"""Tests for puremacro.study.studio (PolicyStudio) and puremacro.reports.result_to_interactive_html."""
from __future__ import annotations

import numpy as np
import pandas as pd

from puremacro.reports import result_to_interactive_html
from puremacro.study import CounterfactualGrid, run_policy_studio
from puremacro.trade import baqaee_farhi_decomposition


def test_policy_studio_2d_baqaee_farhi_grid_and_interactive_html():
    Omega = np.array(
        [
            [0.10, 0.05, 0.05],
            [0.30, 0.10, 0.05],
            [0.10, 0.25, 0.10],
        ],
        dtype=np.float64,
    )
    b = np.array([0.15, 0.35, 0.50], dtype=np.float64)

    grid = CounterfactualGrid(
        {
            "elasticity": [0.25, 0.60, 1.00],
            "energy_shock": [-0.10, -0.05],
        }
    )
    assert len(grid) == 6

    def _eval(params: dict):
        z = np.array([float(params["energy_shock"]), 0.0, 0.0], dtype=np.float64)
        return baqaee_farhi_decomposition(
            Omega,
            b,
            elasticities=float(params["elasticity"]),
            tfp_shocks=z,
            final_elasticity=float(params["elasticity"]),
        )

    studio = run_policy_studio(_eval, grid)
    assert studio.pass_count == 6
    assert studio.warn_count == 0
    assert studio.fail_count == 0
    assert "PASS (6/6 verified)" in studio.overall_audit_status

    piv = studio.pivot("second_order_dlog_y")
    assert piv.shape == (3, 2)
    # At elasticity = 1.0 (Cobb-Douglas), 2nd-order curvature is 0
    assert np.allclose(piv.loc[1.00].to_numpy(), 0.0, atol=1e-14)
    # At elasticity = 0.25 (strong complements), 2nd-order curvature is strictly negative
    assert np.all(piv.loc[0.25].to_numpy() < -1e-4)

    md = studio.to_markdown()
    ltx = studio.to_latex()
    typ = studio.to_typst()
    html_doc = studio.to_interactive_html()
    assert "second_order_dlog_y" in md
    assert "\\begin{tabular}" in ltx
    assert "#table(" in typ
    assert "<!DOCTYPE html>" in html_doc
    assert "<svg" in html_doc
    assert "pm-horizon-slider" in html_doc
    assert "http://" not in html_doc and "https://" not in html_doc


def test_policy_studio_self_auditing_pass_warn_fail_statuses():
    def _mock_solver(params: dict):
        mode = params["mode"]
        if mode == "ok":
            return {"welfare": 1.05, "converged": True, "max_residual": 1e-12}
        if mode == "borderline":
            return {"welfare": 1.01, "converged": True, "max_residual": 5e-5}
        return {"welfare": np.nan, "converged": False, "max_residual": 0.25}

    studio = run_policy_studio(
        _mock_solver,
        {"mode": ["ok", "borderline", "broken"]},
        warn_tol=1e-6,
        fail_tol=1e-3,
    )
    assert studio.pass_count == 1
    assert studio.warn_count == 1
    assert studio.fail_count == 1
    assert studio.overall_audit_status.startswith("FAIL (1/3")


def test_result_to_interactive_html_standalone_dataframe():
    df = pd.DataFrame(
        {
            "output": [0.0, -0.02, -0.015, -0.008, -0.002],
            "inflation": [0.0, -0.01, -0.007, -0.003, -0.001],
        },
        index=pd.Index([1, 2, 3, 4, 5], name="horizon"),
    )
    html_str = result_to_interactive_html(
        df,
        title="IRF Explorer",
        subtitle="Unit Test",
        audit_badge="PASS",
    )
    assert "IRF Explorer" in html_str
    assert 'data-series="output"' in html_str
    assert 'data-series="inflation"' in html_str
    assert "pmToggleSeries" in html_str
    assert "pmFilterRows" in html_str


def test_policy_studio_plot_1d_and_2d():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1D Grid
    res_1d = run_policy_studio(
        lambda p: {"multiplier": float(p["gamma"]) * 2.5, "max_residual": 1e-10},
        {"gamma": [0.5, 1.0, 1.5, 2.0]},
    )
    assert res_1d.overall_audit_status.startswith("PASS")
    fig_1d = res_1d.plot("multiplier")
    assert isinstance(fig_1d, plt.Figure)
    ax_1d = fig_1d.axes[0]
    assert ax_1d.get_xlabel() == "gamma"
    assert ax_1d.get_ylabel() == "multiplier"
    assert "multiplier" in ax_1d.get_title()
    assert "PASS" in ax_1d.get_title()

    # 2D Grid
    res_2d = run_policy_studio(
        lambda p: {"welfare": float(p["alpha"]) - float(p["beta"]), "max_residual": 1e-9},
        {"alpha": [0.3, 0.4], "beta": [0.95, 0.99]},
    )
    fig_2d = res_2d.plot("welfare")
    assert isinstance(fig_2d, plt.Figure)
    ax_2d = fig_2d.axes[0]
    assert ax_2d.get_xlabel() == "beta"
    assert ax_2d.get_ylabel() == "alpha"
    assert len(ax_2d.images) == 1  # imshow heatmap exists
    assert "welfare" in ax_2d.get_title()
    assert "PASS" in ax_2d.get_title()
    plt.close("all")


def test_policy_studio_custom_metric_and_audit_callbacks():
    def custom_eval(p):
        theta = float(p["theta"])
        return {
            "loss": (theta - 1.0) ** 2,
            "residual_norm": abs(theta - 1.0) * 1e-4,
            "extra_info": "custom",
        }

    def custom_metric(raw):
        return {
            "transformed_loss": raw["loss"] * 100.0,
            "sqrt_loss": float(np.sqrt(raw["loss"])),
        }

    def custom_audit(raw):
        # Custom rule: if theta == 2.0, flag as WARN
        if raw["loss"] > 0.5:
            return "WARN", raw["residual_norm"]
        return "PASS", raw["residual_norm"]

    res = run_policy_studio(
        custom_eval,
        {"theta": [1.0, 2.0]},
        metric_fn=custom_metric,
        audit_fn=custom_audit,
    )
    assert res.pass_count == 1
    assert res.warn_count == 1
    assert res.fail_count == 0
    df = res.to_dataframe()
    assert "transformed_loss" in df.columns
    assert "sqrt_loss" in df.columns
    assert df.loc[df["theta"] == 1.0, "transformed_loss"].iloc[0] == 0.0
    assert df.loc[df["theta"] == 2.0, "transformed_loss"].iloc[0] == 100.0
    assert df.loc[df["theta"] == 2.0, "audit_status"].iloc[0] == "WARN"
    assert "WARN" in res.overall_audit_status
