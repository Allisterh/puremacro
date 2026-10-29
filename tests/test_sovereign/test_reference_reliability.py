"""Independent Bellman and pricing checks for the short-term debt limit."""
import numpy as np
import pytest

from puremacro.sovereign import solve_sovereign_default, evaluate_debt_restructuring


def test_default_cost_uses_stationary_mean_and_reentry_at_zero_debt():
    result = solve_sovereign_default(n_b=41, n_y=7, b_min=-0.3, b_max=0.3,
                                     y_penalty_type="arellano", sigma_eps=0.01,
                                     howard_steps=0, tol=1e-9)
    mean_y = result.metadata["stationary_dist"] @ result.y_grid
    y_default = np.minimum(result.y_grid, 0.969 * mean_y)
    np.testing.assert_allclose(result.metadata["y_default_vec"], y_default, atol=1e-12)
    zero = np.argmin(abs(result.b_grid))
    expected = 1-1/y_default + 0.953*(0.282*result.v_expected[zero] + (1-0.282)*result.v_default) @ result.p_trans.T
    np.testing.assert_allclose(result.v_default, expected, atol=2e-8)


def test_deterministic_default_choice_limit_satisfies_bellman_and_pricing():
    result = solve_sovereign_default(n_b=41, n_y=7, b_min=-0.3, b_max=0.3,
                                     y_penalty_type="arellano", sigma_eps=0.0,
                                     howard_steps=0, enforce_monotonicity=False, tol=1e-9)
    assert result.converged
    b, y, P, q = result.b_grid, result.y_grid, result.p_trans, result.q_bond
    # Recompute one Bellman update directly from the equations, independently
    # of the solver's policy and convergence diagnostics.
    total = np.maximum(result.v_repay, result.v_default[None, :])
    expected_repay = np.empty_like(q)
    for i, debt in enumerate(b):
        for j, income in enumerate(y):
            consumption = income - debt + q[:, j]*b
            utility = np.full(len(b), -np.inf)
            feasible = consumption > 1e-6
            utility[feasible] = 1-1/consumption[feasible]
            expected_repay[i, j] = np.max(utility + 0.953*(total @ P[j]))
    np.testing.assert_allclose(result.v_repay, expected_repay, atol=2e-8)
    default = (result.v_default[None, :] > result.v_repay).astype(float)
    np.testing.assert_array_equal(result.default_prob, default)
    np.testing.assert_allclose(q, (1-default) @ P.T / 1.01, atol=2e-8)


def test_constant_spreads_do_not_report_invented_negative_correlation():
    result = solve_sovereign_default(n_b=5, n_y=3, b_max=1e-6, sigma_eps=1e-5)
    assert np.std(result.spread) < 1e-7
    assert np.isnan(result.spread_y_corr)
    assert result.metadata["spread_y_corr_status"] == "undefined_zero_variation"


@pytest.mark.parametrize("penalty", ["arellano", "chatterjee_eyigungor"])
def test_zero_restructuring_preserves_the_original_model_calibration(penalty):
    original = solve_sovereign_default(n_b=41, n_y=7, b_min=-0.3, b_max=0.3,
                                       sigma_eps=0, y_penalty_type=penalty, m_tauchen=2.0,
                                       d0=-0.10, d1=0.15,
                                       enforce_monotonicity=False, tol=1e-9)
    scenario = evaluate_debt_restructuring(original, haircut=0, maturity_extension=0, horizon=5)
    unchanged = scenario.metadata["res_restructured"]
    np.testing.assert_allclose(unchanged.y_grid, original.y_grid, atol=1e-12)
    np.testing.assert_allclose(unchanged.q_bond, original.q_bond, atol=1e-8)
    np.testing.assert_allclose(unchanged.v_repay, original.v_repay, atol=1e-8)
    np.testing.assert_allclose(unchanged.metadata["y_default_vec"], original.metadata["y_default_vec"], atol=1e-12)


def test_restructuring_recovery_uses_model_default_output_and_zero_debt():
    original = solve_sovereign_default(n_b=41, n_y=7, b_min=-0.4, b_max=0.4,
                                       sigma_eps=0, y_penalty_type="arellano", tol=1e-9)
    scenario = evaluate_debt_restructuring(original, haircut=0, maturity_extension=0,
                                          b_distress=0.4, y_distress=original.y_grid[0], horizon=100)
    path = scenario.transition_path
    excluded = path.spread_bps.isna().to_numpy()
    assert excluded.any()
    default_outputs = scenario.metadata["res_restructured"].metadata["y_default_vec"]
    for output in path.loc[excluded, "output"]:
        assert np.min(abs(default_outputs-output)) < 1e-12
    reentry = np.flatnonzero(excluded[:-1] & ~excluded[1:])+1
    assert len(reentry) > 0
    np.testing.assert_allclose(path.iloc[reentry].debt, 0, atol=1e-12)
