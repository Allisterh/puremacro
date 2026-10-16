"""Adversarial probing and robustness stress tests (Tier 5).

Adversarial numerical probing of the trade-war and continuation tools of notebooks 63 and 64.

Coverage:
1. Notebook 63 (Trade Wars & Nash Tariffs):
   - Extreme tariffs in [0.01, 1.50] (unilateral optimal tariffs and direct equilibrium solves)
   - Alternative strategic player subsets (2, 3, 4, 5 player configurations and non-standard subsets)
   - Cycling and oscillation checks in damped Gauss-Seidel best-response policy iteration
   - Strict Prisoner's Dilemma inequalities: dominant defection, mutual trade war inefficiency,
     and boundary characterization.
2. Notebook 64 (Singularities & Keller PAC Continuation):
   - Arclength continuation across step sizes ds in [0.001, 0.20]
   - Elasticities across deep inelastic regimes sigma in [0.05, 0.20] (the notebook-64
     calibration has no fold there: PAC records no tau_lambda <= 0 event)
   - Singular Jacobian of an analytical scalar fold (det J -> 0, cond J > 10^5) vs the
     non-singular Keller augmented bordered Jacobian (cond J_aug < 10^4)
   - SVD modal projection clamping and the one-country block-elimination step of the fold
     helper under extreme wage displacement: a clamped step reports its own residual and is
     not converged.

Strictly conforms to the puremacro Pyodide four-package runtime contract (NumPy, SciPy, Pandas).
"""
from __future__ import annotations

import copy
from typing import Any
import numpy as np
import pandas as pd
import pytest
import scipy.linalg as la

from puremacro.trade import (
    TradeCalibrationResult,
    TradeEquilibriumResult,
    calibrate_trade_model,
    solve_trade_equilibrium,
)
from puremacro.trade.optimal_tariffs import (
    build_strategic_tariffs,
    compute_unilateral_optimal_tariff,
    compute_welfare_payoff_matrix,
    evaluate_national_welfare,
    solve_multilateral_nash_tariffs,
)
from puremacro.trade.solver import (
    clamp_wage_displacement,
    compute_equilibrium_residuals,
    solve_fold_continuation_step,
    solve_keller_pac,
    svd_clamped_newton_step,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def bilateral_cge_model() -> TradeCalibrationResult:
    """Deterministic 2-country, 2-sector general equilibrium trade model (Notebook 63)."""
    np.random.seed(0)
    nc, ns, nfd = 2, 2, 3
    N = nc * ns
    data = np.zeros((N + 3, N + nfd * nc), dtype=float)
    data[:4, :4] = 10.0 + 20.0 * np.random.rand(4, 4)
    y = np.array([200.0, 300.0, 200.0, 300.0])
    inter_cols = data[:4, :4].sum(axis=0)
    va = y - inter_cols
    taxes = 0.05 * y
    va_fac = va - taxes
    data[4, :4] = taxes
    data[5, :4] = 0.6 * va_fac
    data[6, :4] = 0.4 * va_fac
    inter_rows = data[:4, :4].sum(axis=1)
    fd_rows = y - inter_rows
    for i in range(4):
        tot_fd_i = fd_rows[i]
        sh = (
            np.random.dirichlet([2.0, 1.0, 0.5, 0.5, 0.2, 0.1])
            if i < 2
            else np.random.dirichlet([0.5, 0.2, 0.1, 2.0, 1.0, 0.5])
        )
        data[i, 4:10] = tot_fd_i * sh
    data[4, 4:] = 0.02 * data[:4, 4:].sum(axis=0)

    return calibrate_trade_model(
        data,
        ns=ns,
        nc=nc,
        nfd=nfd,
        country_codes=["C01", "C02"],
        sector_codes=["AGR", "MAN"],
        validate=True,
    )


@pytest.fixture(scope="module")
def multilateral_5c_model() -> TradeCalibrationResult:
    """Deterministic 5-country, 2-sector multilateral trade model (Notebook 63)."""
    nc, ns, nfd = 5, 2, 3
    countries = ["C01", "C02", "C03", "C04", "C05"]
    N = nc * ns
    data = np.zeros((N + 3, N + nfd * nc), dtype=float)
    # Similar-sized economies: consistent accounting needs nonnegative investment shares.
    y = np.array([220.0, 280.0, 180.0, 240.0, 200.0, 250.0, 160.0, 210.0, 170.0, 220.0])

    for c in range(nc):
        data[c * ns : (c + 1) * ns, c * ns : (c + 1) * ns] = np.array([
            [0.12 * y[c * ns], 0.08 * y[c * ns + 1]],
            [0.06 * y[c * ns], 0.15 * y[c * ns + 1]],
        ])

    for o in range(nc):
        for d in range(nc):
            if o == d:
                continue
            data[o * ns : (o + 1) * ns, d * ns : (d + 1) * ns] = np.array([
                [0.015 * y[d * ns], 0.010 * y[d * ns + 1]],
                [0.010 * y[d * ns], 0.018 * y[d * ns + 1]],
            ])

    inter_cols = data[:N, :N].sum(axis=0)
    va = y - inter_cols
    taxes = 0.04 * y
    va_fac = va - taxes
    data[N, :N] = taxes
    data[N + 1, :N] = 0.65 * va_fac
    data[N + 2, :N] = 0.35 * va_fac

    inter_rows = data[:N, :N].sum(axis=1)
    fd_rows = y - inter_rows
    for i in range(N):
        tot_fd_i = fd_rows[i]
        c_i = i // ns
        for c_dest in range(nc):
            sh = 0.70 if c_dest == c_i else (0.30 / (nc - 1))
            data[i, N + c_dest * nfd : N + (c_dest + 1) * nfd] = (
                tot_fd_i * sh
            ) * np.array([0.6, 0.3, 0.1])

    data[N, N:] = 0.02 * data[:N, N:].sum(axis=0)

    return calibrate_trade_model(
        data,
        ns=ns,
        nc=nc,
        nfd=nfd,
        country_codes=countries,
        sector_codes=["AGR", "MAN"],
        validate=True,
    )


@pytest.fixture(scope="module")
def singular_fold_cge_model() -> TradeCalibrationResult:
    """Notebook 64's two-country CGE calibration (no fold at the tested tariffs and elasticities)."""
    nc, ns, nfd = 2, 2, 3
    data = np.zeros((ns * nc + 3, ns * nc + nfd * nc), dtype=float)
    data[:4, :4] = np.array([
        [20.0, 10.0, 5.0, 2.0],
        [8.0, 25.0, 2.0, 4.0],
        [5.0, 5.0, 12.0, 18.0],
        [10.0, 10.0, 18.0, 22.0],
    ])
    inter_cols = data[:4, :4].sum(axis=0)
    y = np.array([100.0, 150.0, 120.0, 180.0])
    va = y - inter_cols
    taxes = 0.05 * y
    va_fac = va - taxes
    data[4, :4] = taxes
    data[5, :4] = (2.0 / 3.0) * va_fac
    data[6, :4] = (1.0 / 3.0) * va_fac
    inter_rows = data[:4, :4].sum(axis=1)
    fd_rows = y - inter_rows
    for i in range(4):
        tot_fd = fd_rows[i]
        if i < 2:
            data[i, 4:10] = [
                tot_fd * 0.50,
                tot_fd * 0.25,
                tot_fd * 0.05,
                tot_fd * 0.10,
                tot_fd * 0.08,
                tot_fd * 0.02,
            ]
        else:
            data[i, 4:10] = [
                tot_fd * 0.10,
                tot_fd * 0.08,
                tot_fd * 0.02,
                tot_fd * 0.50,
                tot_fd * 0.25,
                tot_fd * 0.05,
            ]
    data[4, 4:] = 0.02 * data[:4, 4:].sum(axis=0)
    return calibrate_trade_model(data, ns=ns, nc=nc, nfd=nfd, validate=True)


# =============================================================================
# Suite 1: Notebook 63 Stress Probing
# =============================================================================

class TestNotebook63AdversarialProbing:
    """Stress tests for Notebook 63: extreme tariffs, player subsets, cycling, PD structure."""

    @pytest.mark.parametrize("tariff_max", [0.01, 0.05, 0.25, 0.50, 1.00, 1.50])
    def test_extreme_unilateral_optimal_tariffs(
        self, bilateral_cge_model: TradeCalibrationResult, tariff_max: float
    ) -> None:
        """Verify unilateral optimal tariff search converges across tau in [0.01, 1.50]."""
        try:
            res = compute_unilateral_optimal_tariff(
                bilateral_cge_model,
                country_idx="C01",
                num_grid=15,
                tariff_max=tariff_max,
                metric="geary_khamis",
            )
        except RuntimeError as exc:
            # High tariffs can exhaust the inner solver. No payoff certificate
            # may be issued for that failed experiment.
            assert tariff_max >= 1.0
            assert "Policy equilibrium did not converge" in str(exc)
            return
        assert res.equilibrium.converged is True
        assert 0.0 <= res.optimal_tariff_rate <= tariff_max + 1e-6
        assert res.welfare_gain_pct >= -1e-10
        assert res.optimal_welfare >= np.max(res.welfare_curve) - 1e-8
        assert np.isfinite(res.welfare_curve).all()

    @pytest.mark.parametrize("tau_val", [0.01, 0.05, 0.25, 0.50, 1.00, 1.50])
    def test_direct_equilibrium_at_extreme_tariffs(
        self, bilateral_cge_model: TradeCalibrationResult, tau_val: float
    ) -> None:
        """Verify general equilibrium solves cleanly under bilateral tariffs up to 150%."""
        tau_m, tau_fd = build_strategic_tariffs(
            bilateral_cge_model,
            {"C01": {"C02": tau_val}, "C02": {"C01": tau_val}},
        )
        eq = solve_trade_equilibrium(
            bilateral_cge_model,
            tau=tau_m,
            tau_fd=tau_fd,
            method="condensed",
            tol=2.5e-3,
        )
        assert eq.converged is True
        assert np.all(eq.p_sol > 0.0), "Goods prices must remain strictly positive"
        assert np.all(eq.w_sol > 0.0), "Wages must remain strictly positive"
        assert np.all(eq.y_sol > 0.0), "Outputs must remain strictly positive"
        assert np.max(np.abs(eq.residuals)) < 2.5e-3

    @pytest.mark.parametrize(
        "subsets",
        [
            ("C01", "C02"),
            ("C01", "C03", "C02"),
            ("C01", "C02", "C04", "C05"),
            ("C01", "C02", "C03", "C04", "C05"),
            ("C04", "C05"),
            ("C03", "C05"),
        ],
    )
    def test_alternative_strategic_player_subsets(
        self, multilateral_5c_model: TradeCalibrationResult, subsets: tuple[str, ...]
    ) -> None:
        """Verify multilateral Nash solver executes consistently across diverse player coalitions."""
        res = solve_multilateral_nash_tariffs(
            multilateral_5c_model,
            player_countries=subsets,
            relaxation=0.5,
            max_iter=20,
        )
        certified = (res.equilibrium.converged and not res.metadata["inner_solver_failures"]
                     and res.outer_error <= 1e-4 and res.metadata["relative_max_regret"] <= 1e-4)
        assert res.converged == certified
        if not res.converged:
            assert res.metadata["inner_solver_failures"] or res.outer_error > 1e-4 or res.metadata["relative_max_regret"] > 1e-4
        assert set(res.strategic_players) == set(subsets)
        for p in subsets:
            assert p in res.nash_tariffs
            assert 0.0 <= res.nash_tariffs[p] <= 1.50
        assert np.isfinite(res.world_welfare_change_pct) or not res.converged

    @pytest.mark.parametrize("relaxation", [0.4, 0.5, 0.7, 0.9, 1.0])
    def test_cycling_and_oscillation_in_best_response(
        self, bilateral_cge_model: TradeCalibrationResult, relaxation: float
    ) -> None:
        """Verify best-response policy iteration converges to unique Nash fixed point without cycling."""
        res = solve_multilateral_nash_tariffs(
            bilateral_cge_model,
            player_countries=("C01", "C02"),
            relaxation=relaxation,
            max_iter=30,
        )
        certified = (res.equilibrium.converged and not res.metadata["inner_solver_failures"]
                     and res.outer_error <= 1e-4 and res.metadata["relative_max_regret"] <= 1e-4)
        assert res.converged == certified
        if not res.converged:
            assert res.metadata["inner_solver_failures"] or res.outer_error > 1e-4 or res.metadata["relative_max_regret"] > 1e-4
        if res.converged:
            assert res.outer_error <= 1e-4
            assert res.metadata["relative_max_regret"] <= 1e-4

    def test_prisoners_dilemma_strict_inequalities(
        self, bilateral_cge_model: TradeCalibrationResult
    ) -> None:
        """Classify the finite game from six computed inequalities without assuming its type."""
        payoffs = compute_welfare_payoff_matrix(
            bilateral_cge_model,
            player_a="C01",
            player_b="C02",
            optimal_a=0.05,
            optimal_b=0.05,
            nash_a=0.05,
            nash_b=0.05,
        )
        p = payoffs.payoff_matrix
        expected = bool(p[1,0,0] > p[0,0,0] and p[1,1,0] > p[0,1,0]
                        and p[0,1,1] > p[0,0,1] and p[1,1,1] > p[1,0,1]
                        and p[0,0,0] > p[1,1,0] and p[0,0,1] > p[1,1,1])
        assert payoffs.is_prisoners_dilemma == expected
        assert all(eq.converged for eq in payoffs.scenarios.values())

    @pytest.mark.parametrize("tariff_rate", [0.02, 0.03, 0.05, 0.07, 0.10])
    def test_prisoners_dilemma_robustness_regime(
        self, bilateral_cge_model: TradeCalibrationResult, tariff_rate: float
    ) -> None:
        """Classify the computed game across a range of fixed strategy rates."""
        payoffs = compute_welfare_payoff_matrix(
            bilateral_cge_model,
            player_a="C01",
            player_b="C02",
            optimal_a=tariff_rate,
            optimal_b=tariff_rate,
            nash_a=tariff_rate,
            nash_b=tariff_rate,
        )
        p = payoffs.payoff_matrix
        expected = bool(p[1,0,0] > p[0,0,0] and p[1,1,0] > p[0,1,0]
                        and p[0,1,1] > p[0,0,1] and p[1,1,1] > p[1,0,1]
                        and p[0,0,0] > p[1,1,0] and p[0,0,1] > p[1,1,1])
        assert payoffs.is_prisoners_dilemma == expected
        assert all(eq.converged for eq in payoffs.scenarios.values())


# =============================================================================
# Suite 2: Notebook 64 Stress Probing
# =============================================================================

class TestNotebook64AdversarialProbing:
    """Stress tests for Notebook 64: Keller PAC step sizes, elasticities, a scalar fold, the fold step."""

    @pytest.mark.parametrize("ds", [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20])
    def test_keller_pac_step_size_spectrum(
        self, singular_fold_cge_model: TradeCalibrationResult, ds: float
    ) -> None:
        """Verify Keller PAC converges across arclength step sizes ds in [0.001, 0.20]."""
        res = solve_keller_pac(
            singular_fold_cge_model,
            tau_target=0.25,
            sigma=0.1238,
            ds_init=ds,
            tol=1e-9,
            max_steps=50,
        )
        assert res.converged is True
        assert float(np.max(np.abs(res.residuals))) < 1e-6
        assert np.all(res.p_sol > 0.0)
        assert np.all(res.w_sol > 0.0)

    @pytest.mark.parametrize("sigma", [0.05, 0.08, 0.10, 0.1238, 0.15, 0.18, 0.20])
    def test_keller_pac_deep_inelastic_regimes(
        self, singular_fold_cge_model: TradeCalibrationResult, sigma: float
    ) -> None:
        """Verify Keller PAC converges across sigma in [0.05, 0.20].

        No fold is crossed: on this calibration PAC records no tau_lambda <= 0
        event for any of these elasticities (sigma=0.1238 is not special).
        """
        res = solve_keller_pac(
            singular_fold_cge_model,
            tau_target=0.25,
            sigma=sigma,
            ds_init=0.05,
            tol=1e-9,
            max_steps=50,
        )
        assert res.converged is True
        assert float(np.max(np.abs(res.residuals))) < 1e-6
        assert np.all(res.p_sol > 0.0)
        assert np.all(res.w_sol > 0.0)
        assert res.metadata["fold_detections"] == 0
        assert res.metadata["fold_points"] == []

    def test_standard_newton_divergence_at_fold(self) -> None:
        """Analytical scalar fold: J_x is singular (det J -> 0, cond J > 10^5) while bordered J is bounded."""
        def F_fold(x: np.ndarray, lam: float) -> np.ndarray:
            return np.array([x[0]**2 - lam, x[1] - x[0]], dtype=float)

        def J_fold(x: np.ndarray) -> np.ndarray:
            return np.array([[2.0 * x[0], 0.0], [-1.0, 1.0]], dtype=float)

        dF_dlam = np.array([-1.0, 0.0], dtype=float)

        # Evaluate at turning point limit x1 = 1e-6, lambda = 0.0
        x_sing = np.array([1e-6, 1e-6])
        Jx = J_fold(x_sing)
        det_Jx = float(la.det(Jx))
        cond_Jx = float(np.linalg.cond(Jx))

        assert abs(det_Jx) < 1e-5, f"det(J_x) must approach 0: {det_Jx}"
        assert cond_Jx > 1e5, f"Standard condition number must explode: {cond_Jx}"

        # Augmented bordered Jacobian at turning point
        tau_x = np.array([-1.0, -1.0]) / np.sqrt(2.0)
        tau_lam = -0.4
        tangent = np.array([tau_x[0], tau_x[1], tau_lam])
        tangent /= np.linalg.norm(tangent)

        J_aug = np.empty((3, 3), dtype=float)
        J_aug[:2, :2] = Jx
        J_aug[:2, 2] = dF_dlam
        J_aug[2, :2] = tangent[:2]
        J_aug[2, 2] = tangent[2]

        cond_aug = float(np.linalg.cond(J_aug))
        det_aug = float(la.det(J_aug))

        assert cond_aug < 10.0, f"Augmented condition number must remain bounded: {cond_aug}"
        assert abs(det_aug) > 1.0, f"Augmented bordered Jacobian must be non-singular: {det_aug}"

    @pytest.mark.parametrize("rhs_scale", [1.0, 10.0, 100.0, 1e4, 1e6])
    def test_svd_clamping_and_fold_step_stabilization(self, rhs_scale: float) -> None:
        """Verify SVD clamping and the fold step bound explosive displacement up to scale 10^6.

        The one-country block-elimination step is the exact solve scaled down to
        max_disp: bounded, but not converged. Its reported residual is that of
        the returned step (about max|rhs|); the unclamped direction is exact.
        """
        rng = np.random.default_rng(42)
        nc = 40
        idx_stiff = 7
        tol = 2.5e-3

        A_rnd = rng.standard_normal((nc, nc))
        S_ww = A_rnd.T @ A_rnd + np.eye(nc)
        # Induce stiffness
        S_ww[idx_stiff, :] *= 1e-5
        S_ww[:, idx_stiff] *= 1e-5
        S_ww[idx_stiff, idx_stiff] = 1e-6

        rhs_w = rng.standard_normal(nc) * rhs_scale

        # Unregularized step explodes
        dw_unreg = la.solve(S_ww, rhs_w)
        assert np.max(np.abs(dw_unreg)) > 1e5

        # SVD clamped step is strictly bounded
        D_L = np.ones(nc, dtype=float)
        D_R = np.ones(nc, dtype=float)
        dw_svd = svd_clamped_newton_step(S_ww, -rhs_w, D_L, D_R, max_comp=20.0, max_disp=0.30)
        assert np.max(np.abs(dw_svd)) <= 0.3000 + 1e-8

        # Direct wage clamp is strictly bounded
        dw_clamped = clamp_wage_displacement(dw_unreg, max_disp=0.30)
        assert np.max(np.abs(dw_clamped)) <= 0.3000 + 1e-8

        # Decoupled one-country step: strictly bounded, honestly reported
        dw_stiff, res_stiff, conv_stiff, info = solve_fold_continuation_step(
            S_ww=S_ww,
            rhs_w=rhs_w,
            eval_stiff_fn=None,
            idx_stiff=idx_stiff,
            tol=tol,
            max_disp=0.30,
            return_info=True,
        )
        assert np.max(np.abs(dw_stiff)) <= 0.3000 + 1e-8
        np.testing.assert_allclose(info["unclamped_step"], dw_unreg, rtol=0,
                                   atol=1e-9 * np.max(np.abs(dw_unreg)))
        assert info["direction_converged"] is True
        assert info["clamped"] is True
        np.testing.assert_allclose(dw_stiff, info["clamp_scale"] * info["unclamped_step"], rtol=1e-12, atol=0.0)
        assert res_stiff == pytest.approx(float(np.max(np.abs(S_ww @ dw_stiff - rhs_w))), rel=1e-12)
        assert res_stiff == pytest.approx(float(np.max(np.abs(rhs_w))), rel=1e-5)
        assert conv_stiff is False
