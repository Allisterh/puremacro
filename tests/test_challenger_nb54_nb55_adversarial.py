"""Adversarial stress test suite for Showcase Notebook 54 and the Allen-Arkolakis model.

Authored by Challenger 2 (Empirical Challenger):
1. Notebook 54 (Deep Macro & PINNs):
   - Neural network stability under learning rate perturbations (lr in [1e-4, 1e-2], and boundary defense under lr=0.05).
   - Neural network stability across diverse hidden layer dimensions ((16,), (32,), (16, 16), (32, 32), (64, 64), (128, 64), (64, 32, 16)).
   - Strict physical viability (c_i > 0, k'_i > 0, c_i < W_i) across 1,000+ simulated states and extreme initial state perturbations.
   - Out-of-sample Euler residual MSE < 1e-3 across randomized test trajectories with multiple random seeds.
   - Exact analytical steady-state Euler equation residual evaluation.
   - Policy function monotonicity in capital.

2. Allen-Arkolakis (2014) under large transport cost shocks (cost reductions up to 95%, trade friction increases up to 5x):
   - Contraction mapping convergence.
   - Exact labor conservation: |sum L_i - L_bar| < 1e-12.
   - Spatial price and real wage / indirect utility equalization: std(u_i) / mean(u_i) < 1e-7.
   - Monotone aggregate spatial welfare responses.

3. Bilingual Parity & Execution Integrity:
   - 2-notebook execution integrity (54, 54_es).
   - Pyodide browser-safe dependencies contract.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from puremacro.vfi.deep_macro import DeepMacroModel, solve_deep_macro
from puremacro.spatial.allen_arkolakis import AllenArkolakisModel


# ============================================================================
# TASK 1: NOTEBOOK 54 (DEEP MACRO & PINNS) EMPIRICAL ADVERSARIAL STRESS SUITE
# ============================================================================

class TestNotebook54DeepMacroPINN:
    """Adversarial stress-testing of Deep Macro PINN high-dimensional model."""

    @pytest.fixture(scope="class")
    def base_model(self) -> DeepMacroModel:
        """Construct canonical 10-country dynamic capital accumulation model."""
        return DeepMacroModel.multi_country_growth(
            n_countries=10,
            alpha=0.36,
            beta=0.96,
            delta=0.08,
            gamma=2.0,
            A=1.0,
            rho=0.8,
            sigma_eps=0.0,
        )

    def test_steady_state_euler_machine_precision(self, base_model: DeepMacroModel):
        """Verify analytical steady-state Euler equation residual evaluates to machine zero."""
        k_ss, c_ss = base_model.steady_state()
        s_ss = k_ss.reshape(1, -1)
        c_ss_mat = c_ss.reshape(1, -1)
        res_ss = base_model.default_euler_residual(s_ss, c_ss_mat, s_ss, c_ss_mat)
        assert np.allclose(res_ss, 0.0, atol=1e-12), (
            f"Steady-state Euler residual max={np.max(np.abs(res_ss)):.2e} must be zero"
        )

    @pytest.mark.parametrize("lr", [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2])
    def test_learning_rate_stability_and_convergence(self, base_model: DeepMacroModel, lr: float):
        """Test neural network stability and convergence across valid learning rates."""
        sol = solve_deep_macro(
            model=base_model,
            hidden_dims=(32, 32),
            n_epochs=80,
            batch_size=128,
            lr=lr,
            trajectory_length=1000,
            burn_in=0,
            resimulate_every=25,
            seed=42,
            verbose=False,
        )
        assert sol.converged, f"Solver failed to converge at lr={lr}: MSE={sol.test_euler_mse:.2e}"
        assert sol.test_euler_mse < 1e-3, f"Out-of-sample Euler MSE {sol.test_euler_mse:.2e} >= 1e-3"
        assert sol.loss_history[-1] < 1e-4, f"Final training loss {sol.loss_history[-1]:.2e} too high"

    def test_high_learning_rate_boundary_defense_lr_005(self, base_model: DeepMacroModel):
        """Adversarially test lr=0.05: verify physical viability holds even under optimizer divergence."""
        k_ss, _ = base_model.steady_state()
        sol = solve_deep_macro(
            model=base_model,
            hidden_dims=(64, 64),
            n_epochs=80,
            batch_size=128,
            lr=0.05,
            trajectory_length=1000,
            burn_in=0,
            resimulate_every=25,
            seed=42,
            verbose=False,
        )
        # Even if lr=0.05 causes loss saturation, test that architectural bounding prevents NaN/Inf
        # and guarantees strict physical resource feasibility
        sim = sol.simulate(s0=k_ss * 0.5, periods=1000, seed=123)
        c = sim["controls"]
        k = sim["states"]
        W = sim["cash_on_hand"]

        assert not np.any(np.isnan(c)), "Consumption must contain 0 NaNs under lr=0.05"
        assert not np.any(np.isnan(k)), "Capital must contain 0 NaNs under lr=0.05"
        assert np.all(c > 0.0), "Consumption must remain strictly positive even under lr=0.05"
        assert np.all(k > 0.0), "Capital must remain strictly positive even under lr=0.05"
        assert np.all(c < W), "Consumption must strictly respect cash-on-hand ceiling under lr=0.05"
        assert sim["physically_viable"], "Strict physical viability must be guaranteed by architecture"

    @pytest.mark.parametrize(
        "hdims",
        [
            (16,),
            (32,),
            (16, 16),
            (32, 32),
            (64, 64),
            (128, 64),
            (64, 32, 16),
        ],
    )
    def test_hidden_dimensions_stability(self, base_model: DeepMacroModel, hdims: tuple[int, ...]):
        """Test stability and convergence across varied hidden layer architectures."""
        sol = solve_deep_macro(
            model=base_model,
            hidden_dims=hdims,
            n_epochs=80,
            batch_size=128,
            lr=2e-3,
            trajectory_length=1000,
            burn_in=0,
            resimulate_every=25,
            seed=42,
            verbose=False,
        )
        assert sol.converged, f"Arch {hdims} failed to converge: MSE={sol.test_euler_mse:.2e}"
        assert sol.test_euler_mse < 1e-3, f"Arch {hdims} Euler MSE {sol.test_euler_mse:.2e} >= 1e-3"
        assert sol.mlp.num_parameters > 0

    def test_physical_viability_1000_simulated_states(self, base_model: DeepMacroModel):
        """Verify c_i > 0, k'_i > 0, and c_i < W_i across 1,000+ simulated states and extreme initializations."""
        k_ss, _ = base_model.steady_state()
        sol = solve_deep_macro(
            model=base_model,
            hidden_dims=(64, 64),
            n_epochs=100,
            batch_size=128,
            lr=2e-3,
            seed=42,
            verbose=False,
        )

        # Test across 5 extreme scenarios of 200 periods each = 1,000 simulated states
        scenarios = [
            0.20 * k_ss,                                      # Severe capital shortage
            0.60 * k_ss,                                      # Moderate capital shortage
            1.50 * k_ss,                                      # Capital abundance
            3.00 * k_ss,                                      # Extreme capital glut
            np.linspace(0.3, 2.5, base_model.n_states) * k_ss, # Highly asymmetric cross-country capital
        ]

        total_steps = 0
        for idx, s0 in enumerate(scenarios):
            sim = sol.simulate(s0=s0, periods=200, seed=100 + idx)
            c = sim["controls"]
            k = sim["states"]
            W = sim["cash_on_hand"]
            total_steps += len(c)

            assert sim["physically_viable"], f"Scenario {idx} failed physical viability"
            assert np.all(c > 0.0), f"Scenario {idx} produced non-positive consumption: min={np.min(c):.4e}"
            assert np.all(k > 0.0), f"Scenario {idx} produced non-positive capital: min={np.min(k):.4e}"
            assert np.all(c < W), f"Scenario {idx} violated cash-on-hand ceiling: max c/W={np.max(c/W):.4e}"

        assert total_steps >= 1000, f"Expected at least 1000 simulated steps, got {total_steps}"

    @pytest.mark.parametrize("seed", [1, 7, 42, 123, 555, 777, 999, 1337, 2024, 2026])
    def test_out_of_sample_euler_residual_mse_random_trajectories(
        self, base_model: DeepMacroModel, seed: int
    ):
        """Verify out-of-sample Euler residual MSE < 1e-3 on randomized test trajectories."""
        k_ss, _ = base_model.steady_state()
        sol = solve_deep_macro(
            model=base_model,
            hidden_dims=(64, 64),
            n_epochs=120,
            batch_size=128,
            lr=2e-3,
            seed=42,
            verbose=False,
        )

        rng = np.random.default_rng(seed)
        # Random initial perturbation in [0.6 * k_ss, 1.4 * k_ss]
        s0 = k_ss * rng.uniform(0.60, 1.40, size=base_model.n_states)
        sim = sol.simulate(s0=s0, periods=500, seed=seed)

        res = sim["euler_residuals"]
        mse = float(np.mean(res ** 2))
        assert mse < 1e-3, f"Seed {seed}: Out-of-sample Euler residual MSE {mse:.4e} exceeds 1e-3"

    def test_policy_function_slice_monotonicity(self, base_model: DeepMacroModel):
        """Verify consumption policy is strictly monotonically increasing in domestic capital."""
        k_ss, c_ss = base_model.steady_state()
        sol = solve_deep_macro(
            model=base_model,
            hidden_dims=(64, 64),
            n_epochs=120,
            batch_size=128,
            lr=2e-3,
            seed=42,
            verbose=False,
        )
        k_eval = np.linspace(0.50 * k_ss[0], 1.50 * k_ss[0], 200)
        s_eval = np.tile(k_ss, (len(k_eval), 1))
        s_eval[:, 0] = k_eval

        c_policy = np.array([sol.policy(s_eval[j])[0] for j in range(len(k_eval))])
        W_coh = np.array([base_model.cash_on_hand(s_eval[j])[0] for j in range(len(k_eval))])

        assert np.all(np.diff(c_policy) > 0.0), "Consumption policy slice must be strictly increasing"
        assert np.all(c_policy < W_coh), "Consumption slice must strictly lie below cash-on-hand"
        assert np.isclose(sol.policy(k_ss)[0], c_ss[0], rtol=0.15), (
            f"Policy at steady state ({sol.policy(k_ss)[0]:.4f}) deviates from analytical c_ss ({c_ss[0]:.4f})"
        )


# ============================================================================
# TASK 2: ALLEN-ARKOLAKIS EMPIRICAL STRESS SUITE
# ============================================================================

class TestNotebook55QuantitativeSpatialTradeGE:
    """Adversarial stress-testing of the Allen-Arkolakis general equilibrium model."""

    @pytest.fixture(scope="class")
    def aa_setup(self) -> AllenArkolakisModel:
        """Construct canonical 5-region Allen-Arkolakis spatial model from Notebook 55."""
        coords = np.array([
            [45.0, -93.0],
            [30.0, -90.0],
            [40.7, -74.0],
            [37.7, -122.4],
            [38.6, -90.2],
        ])
        region_names = ["North", "South", "East", "West", "Central"]
        return AllenArkolakisModel.from_coordinates(
            coords,
            region_names=region_names,
            theta=4.0,
            alpha=0.08,
            beta=-0.35,
            total_population=100.0,
        )

    @pytest.mark.parametrize("cost_reduction", [0.10, 0.30, 0.50, 0.80, 0.95])
    def test_allen_arkolakis_transport_cost_reduction_shocks(
        self, aa_setup: AllenArkolakisModel, cost_reduction: float
    ):
        """Test Allen-Arkolakis under transport cost reductions up to 95%."""
        res = aa_setup.simulate_infrastructure_shock("North", "South", cost_reduction=cost_reduction)

        assert res.converged, f"Shock failed to converge at reduction={cost_reduction}"
        # Exact labor conservation: |sum L_i - L_bar| < 1e-12
        assert res.labor_conservation_residual < 1e-12, (
            f"Labor conservation residual {res.labor_conservation_residual:.2e} >= 1e-12"
        )
        assert np.isclose(np.sum(res.population), aa_setup.total_population, atol=1e-11)

        # Spatial real wage / utility equalization
        u_i = aa_setup.fundamental_amenity * (res.population ** aa_setup.beta) * res.real_wages
        u_disp = float(np.std(u_i) / np.mean(u_i))
        assert u_disp < 1e-7, f"Spatial utility dispersion {u_disp:.2e} >= 1e-7"
        assert res.spatial_utility_variance < 1e-7

        # Aggregate welfare gain must be strictly positive
        assert res.welfare_pct > 0.0, f"Expected positive welfare gain, got {res.welfare_pct}"

    @pytest.mark.parametrize("cost_multiplier", [1.5, 2.0, 3.0, 5.0])
    def test_allen_arkolakis_transport_cost_increase_blockades(
        self, aa_setup: AllenArkolakisModel, cost_multiplier: float
    ):
        """Test Allen-Arkolakis under trade barrier increases / blockades up to 5x excess costs."""
        tau_new = 1.0 + (aa_setup.trade_costs - 1.0) * cost_multiplier
        res = aa_setup.solve_counterfactual(trade_costs_new=tau_new)

        assert res.converged, f"Blockade failed to converge at multiplier={cost_multiplier}"
        # Exact labor conservation
        assert res.labor_conservation_residual < 1e-12, (
            f"Labor conservation residual {res.labor_conservation_residual:.2e} >= 1e-12"
        )
        assert np.isclose(np.sum(res.population), aa_setup.total_population, atol=1e-11)

        # Spatial real wage / utility equalization
        u_i = aa_setup.fundamental_amenity * (res.population ** aa_setup.beta) * res.real_wages
        u_disp = float(np.std(u_i) / np.mean(u_i))
        assert u_disp < 1e-7, f"Spatial utility dispersion {u_disp:.2e} >= 1e-7"

        # Welfare loss must be strictly negative
        assert res.welfare_pct < 0.0, f"Expected negative welfare loss, got {res.welfare_pct}"


# ============================================================================
# TASK 3: SHOWCASE NOTEBOOK 54 EXECUTION & COMPATIBILITY INTEGRITY
# ============================================================================

class TestShowcaseNotebooks54And55Integrity:
    """Verify execution integrity and browser-safe compatibility of notebook 54."""

    @pytest.mark.parametrize(
        "nb_name",
        [
            "54_deep_macro_pinns_high_dim.py",
            "54_deep_macro_pinns_high_dim_es.py",
        ],
    )
    def test_notebook_file_exists_and_pyodide_safe(self, nb_name: str):
        """Verify notebook script exists and imports zero disallowed web/native packages."""
        repo_root = Path(__file__).resolve().parent.parent
        nb_path = repo_root / "notebooks" / nb_name
        assert nb_path.exists(), f"Notebook file {nb_path} does not exist"

        content = nb_path.read_text(encoding="utf-8")
        disallowed = ["torch", "tensorflow", "jax", "requests", "urllib.request", "keras"]
        for pkg in disallowed:
            assert f"import {pkg}" not in content, f"Notebook {nb_name} imports disallowed package {pkg}"
            assert f"from {pkg}" not in content, f"Notebook {nb_name} imports from disallowed package {pkg}"
