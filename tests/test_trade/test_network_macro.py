"""Tests for puremacro.trade.network_macro (Baqaee-Farhi 2019, 2020 2nd-order network aggregation)."""
from __future__ import annotations

import numpy as np

from puremacro.trade import (
    baqaee_farhi_decomposition,
    solve_exact_ces_network_prices,
)


def _sample_io_network():
    # 3-sector upstream-downstream production chain:
    # Sector 1 (Energy/Upstream), Sector 2 (Intermediate Mfg), Sector 3 (Services/Final)
    Omega = np.array(
        [
            [0.10, 0.05, 0.05],
            [0.35, 0.15, 0.05],
            [0.10, 0.30, 0.10],
        ],
        dtype=np.float64,
    )
    b = np.array([0.10, 0.30, 0.60], dtype=np.float64)
    return Omega, b


def test_cobb_douglas_reduces_to_hulten_exact():
    Omega, b = _sample_io_network()
    z = np.array([-0.10, 0.02, -0.03], dtype=np.float64)

    res = baqaee_farhi_decomposition(
        Omega,
        b,
        elasticities=1.0,
        tfp_shocks=z,
        final_elasticity=1.0,
        sector_names=["Energy", "Manufacturing", "Services"],
    )
    # In a Cobb-Douglas network (theta_i = 1, theta_0 = 1), 2nd-order curvature is identically zero
    assert abs(res.second_order_dlog_y) < 1e-15
    assert np.max(np.abs(res.hessian_matrix)) < 1e-15
    # And Hulten's 1st-order formula equals the exact nonlinear fixed point to machine precision
    assert abs(res.first_order_dlog_y - res.exact_ces_dlog_y) < 1e-13
    # Primary factor is 100% labor in competitive equilibrium, so sum(alpha_i * lambda_i) == 1
    alpha = 1.0 - np.sum(Omega, axis=1)
    assert abs(float(alpha @ res.domar_weights) - 1.0) < 1e-13


def test_baqaee_farhi_hessian_matches_exact_ces_finite_differences():
    Omega, b = _sample_io_network()
    theta = np.array([0.30, 0.25, 0.40], dtype=np.float64)
    theta_0 = 0.50
    z = np.array([-0.08, 0.0, 0.0], dtype=np.float64)  # Negative upstream energy shock

    res = baqaee_farhi_decomposition(
        Omega,
        b,
        elasticities=theta,
        tfp_shocks=z,
        final_elasticity=theta_0,
    )
    # Under gross complements (theta < 1), a shock has strictly negative 2nd-order output effect
    assert res.second_order_dlog_y < -1e-4
    # Second-order Taylor approximation is much closer to exact CES than first-order Hulten
    err_1st = abs(res.exact_ces_dlog_y - res.first_order_dlog_y)
    err_2nd = abs(res.exact_ces_dlog_y - res.total_second_order_dlog_y)
    assert err_2nd < 0.1 * err_1st

    # Verify analytical Hessian H against centered finite differences of solve_exact_ces_network_prices at z=0
    n = Omega.shape[0]
    h_fd = np.zeros((n, n), dtype=np.float64)
    eps = 1e-4
    for k in range(n):
        for l in range(n):
            ek = np.zeros(n)
            el = np.zeros(n)
            ek[k] = eps
            el[l] = eps
            _, y_pp = solve_exact_ces_network_prices(Omega, b, theta, ek + el, final_elasticity=theta_0)
            _, y_pm = solve_exact_ces_network_prices(Omega, b, theta, ek - el, final_elasticity=theta_0)
            _, y_mp = solve_exact_ces_network_prices(Omega, b, theta, -ek + el, final_elasticity=theta_0)
            _, y_mm = solve_exact_ces_network_prices(Omega, b, theta, -ek - el, final_elasticity=theta_0)
            h_fd[k, l] = (y_pp - y_pm - y_mp + y_mm) / (4.0 * eps * eps)

    assert np.max(np.abs(res.hessian_matrix - h_fd)) < 1e-8


def test_baqaee_farhi_markup_wedges_and_exports():
    Omega, b = _sample_io_network()
    z = np.array([-0.05, -0.02, 0.01], dtype=np.float64)
    markups = np.array([1.25, 1.15, 1.05], dtype=np.float64)

    res = baqaee_farhi_decomposition(
        Omega,
        b,
        elasticities=0.4,
        tfp_shocks=z,
        final_elasticity=0.6,
        markups=markups,
        sector_names=["Energy", "Manufacturing", "Services"],
    )
    assert abs(res.allocative_efficiency_dlog_y) > 1e-5
    # With positive markups > 1, revenue Domar weights are strictly smaller than cost Domar weights
    assert np.all(res.domar_weights < res.cost_domar_weights)
    df = res.to_frame()
    assert list(df.index) == ["Energy", "Manufacturing", "Services"]
    assert "Hulten" in res.to_markdown()
    assert "\\begin{tabular}" in res.to_latex()
    assert "#table(" in res.to_typst()
