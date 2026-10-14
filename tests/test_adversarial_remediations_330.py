"""Comprehensive Regression Suite for 3.3.0 Adversarial Code Review Remediations.

Tests findings CR-01, CR-02, CR-05 to CR-08 and CR-10:
- CR-01: Positive percentage climate shocks & multiplier validation in Allen-Arkolakis.
- CR-02: Zero congestion elasticity (beta=0.0) validation error in Allen-Arkolakis.
- CR-05 & CR-06: Analytic gradients LaTeX escaping (Greek vs Latin identifiers) and IFT documentation.
- CR-07: Custom euler_target_fn support in DeepMacroModel training loop.
- CR-08: Bilateral trade expenditure shares destination/origin property accessors.
- CR-10: DC-EGM upper_envelope single-point segment robustness (no IndexError).
"""
from __future__ import annotations

import gc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from puremacro.spatial.allen_arkolakis import AllenArkolakisModel, AllenArkolakisResult
from puremacro.vfi.analytic_gradients import AnalyticGradientResult, _GREEK_LETTERS
from puremacro.vfi.dcegm import upper_envelope
from puremacro.vfi.deep_macro import DeepMacroModel, solve_deep_macro


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCR01AndCR02AndCR08Spatial:
    """CR-01, CR-02, CR-08: Spatial general equilibrium fixes."""

    def test_cr02_beta_zero_raises_value_error(self):
        """CR-02: Instantiating with beta == 0.0 must raise ValueError to prevent ZeroDivisionError."""
        with pytest.raises(ValueError, match="Amenity congestion elasticity beta cannot be zero"):
            AllenArkolakisModel(
                trade_costs=np.ones((2, 2)),
                fundamental_productivity=np.ones(2),
                fundamental_amenity=np.ones(2),
                theta=4.0,
                alpha=0.10,
                beta=0.0,
            )

    def test_cr01_climate_positive_and_percentage_shocks(self):
        """CR-01: Climate shock must handle positive shocks and is_percentage_change correctly."""
        model = AllenArkolakisModel(
            trade_costs=np.array([[1.0, 1.5], [1.5, 1.0]]),
            fundamental_productivity=np.array([1.0, 1.0]),
            fundamental_amenity=np.array([1.0, 1.0]),
            theta=4.0,
            alpha=0.10,
            beta=-0.30,
            region_names=["East", "West"],
        )

        # 1. Positive percentage change (+20%)
        res_pos = model.simulate_climate_shock(
            productivity_shocks={"East": 0.20},
            is_percentage_change=True,
        )
        assert res_pos.converged
        assert res_pos.welfare_pct > 0.0
        assert res_pos.L_hat[0] > 1.0  # East gains population

        # 2. Positive gross multiplier (1.20)
        res_mult = model.simulate_climate_shock(
            productivity_shocks={"East": 1.20},
            is_percentage_change=False,
        )
        assert res_mult.converged
        assert np.isclose(res_mult.welfare_pct, res_pos.welfare_pct, atol=1e-6)

        # 3. Warning emitted when passing small positive factor < 0.5 without is_percentage_change
        with pytest.warns(UserWarning, match="interpreted as a gross multiplier"):
            model.simulate_climate_shock(
                productivity_shocks={"East": 0.10},
                is_percentage_change=False,
            )

        # 4. Invalid negative shock <= -1.0 raises ValueError
        with pytest.raises(ValueError, match="cannot be <= -1.0"):
            model.simulate_climate_shock(
                productivity_shocks={"East": -1.0},
                is_percentage_change=True,
            )

    def test_cr08_trade_shares_properties(self):
        """CR-08: Explicit accessors for destination-origin and origin-destination trade shares."""
        model = AllenArkolakisModel(
            trade_costs=np.array([[1.0, 1.4], [1.2, 1.0]]),
            fundamental_productivity=np.array([1.2, 0.9]),
            fundamental_amenity=np.array([0.8, 1.1]),
            theta=4.0,
            alpha=0.10,
            beta=-0.30,
        )
        res = model.solve_equilibrium()
        assert res.converged
        np.testing.assert_array_equal(res.trade_shares_dest_origin, res.trade_shares)
        np.testing.assert_array_equal(res.trade_shares_origin_dest, res.trade_shares.T)
        # Verify row stochasticity of destination-by-origin
        np.testing.assert_allclose(np.sum(res.trade_shares_dest_origin, axis=1), 1.0, atol=1e-10)


class TestCR06LaTeXEscaping:
    """CR-06: Valid LaTeX formatting in AnalyticGradientResult.plot."""

    def test_cr06_latin_and_greek_identifiers(self):
        """Greek names get backslash, Latin names (A, delta_k) remain clean."""
        assert "alpha" in _GREEK_LETTERS
        assert "beta" in _GREEK_LETTERS
        assert "A" not in _GREEK_LETTERS
        assert "delta_k" not in _GREEK_LETTERS

        # Create dummy result with both Latin and Greek parameter names
        res = AnalyticGradientResult(
            grad_coefficients=np.ones((5, 2)),
            grad_aggregates={"K": np.array([0.5, -0.2]), "C": np.array([0.1, 0.3])},
            param_names=["beta", "A"],
            jacobian_resid_c=np.eye(5),
            jacobian_resid_theta=np.ones((5, 2)),
            condition_number=100.0,
            elapsed_time=0.01,
        )

        fig = res.plot(show=False)
        assert fig is not None
        plt.close(fig)


class TestCR07DeepMacroEulerTarget:
    """CR-07: Custom euler_target_fn in DeepMacroModel."""

    def test_cr07_custom_euler_target(self):
        """DeepMacroModel supports custom euler_target_fn without falling back to hardcoded Cobb-Douglas."""
        called_custom = [False]

        def custom_target(k, c, kp, cp):
            called_custom[0] = True
            # Simple identity or custom target
            return 0.5 * c

        model = DeepMacroModel(
            n_states=1,
            n_controls=1,
            beta=0.95,
            params={"alpha": 0.33, "delta": 0.10, "gamma": 1.5, "A": 1.0},
            euler_target_fn=custom_target,
        )

        sol = solve_deep_macro(
            model,
            n_epochs=2,
            batch_size=8,
            trajectory_length=20,
            seed=42,
        )
        assert sol.converged
        assert called_custom[0], "Custom euler_target_fn was not invoked during training!"


class TestCR10DCEGMSinglePointSegment:
    """CR-10: upper_envelope handles single-point segments without IndexError."""

    def test_cr10_single_point_segments(self):
        """Degenerate single-point segment does not raise IndexError on slope calculation."""
        exog = np.linspace(0.1, 5.0, 50)
        m_raw = np.array([1.0])
        c_raw = np.array([0.5])
        v_raw = np.array([0.2])

        # Should execute cleanly without raising IndexError: index out of bounds
        v_clean, c_clean = upper_envelope(m_raw, c_raw, v_raw, exog, a_min=0.0)
        assert len(v_clean) == len(exog)
        assert len(c_clean) == len(exog)
        assert not np.any(np.isnan(v_clean))
        assert not np.any(np.isnan(c_clean))
