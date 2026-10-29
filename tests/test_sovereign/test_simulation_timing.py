"""The simulation must implement the exclusion timing in the Bellman equation."""
import numpy as np
import pytest

from puremacro.sovereign import solve_sovereign_default


class DefaultReentryRepayTape:
    """Uniforms force default, successful reentry, repayment, then repeat."""

    def __init__(self):
        self.draw = 0

    def random(self):
        value = (0.0, 0.0, np.nextafter(1.0, 0.0))[self.draw % 3]
        self.draw += 1
        return value

    def choice(self, size, *, p):
        return size // 2


def test_reentry_permits_repayment_in_the_same_period(monkeypatch):
    # Patch only the exogenous random stream. Values, prices, default decisions
    # and policies are solved by the real public function.
    monkeypatch.setattr(np.random, "default_rng", lambda seed: DefaultReentryRepayTape())
    result = solve_sovereign_default(beta=.9, theta=1., sigma_eps=.05,
                                     n_b=25, n_y=3, b_min=-.3, b_max=.3,
                                     howard_steps=0, tol=1e-9, max_iter=5000)
    assert result.converged
    zero, middle = 12, 1
    assert result.b_grid[zero] == 0
    assert np.all(result.default_prob > 0)
    assert result.default_prob[zero, middle] < np.nextafter(1., 0.)
    # At theta=1, Bellman says default today and market access tomorrow.
    # Thus 1,500 periods contain 750 default/repayment pairs. The old loop
    # inserted an extra excluded period and produced only 500 observations.
    assert result.metadata["spread_observations"] == 750
    chosen = np.argmin(abs(result.b_grid-result.policy_b_prime[zero, middle]))
    assert result.spread_mean == pytest.approx(result.spread[chosen, middle], abs=1e-12)


def test_zero_reentry_keeps_the_economy_excluded_after_default(monkeypatch):
    monkeypatch.setattr(np.random, "default_rng", lambda seed: DefaultReentryRepayTape())
    result = solve_sovereign_default(beta=.9, theta=0., sigma_eps=.05,
                                     n_b=9, n_y=3, howard_steps=0, tol=1e-9)
    assert result.converged
    assert result.default_prob[0, 1] > 0
    assert result.metadata["spread_observations"] == 0
    assert np.isnan(result.spread_mean)
    assert np.isnan(result.spread_y_corr)
    assert result.metadata["spread_y_corr_status"] == "insufficient_observations"


def test_public_path_starts_and_reenters_at_zero_when_saving_is_allowed(monkeypatch):
    monkeypatch.setattr(np.random, "default_rng", lambda seed: DefaultReentryRepayTape())
    result = solve_sovereign_default(beta=.9, theta=1., sigma_eps=.05,
                                     n_b=25, n_y=3, b_min=-.3, b_max=.3,
                                     howard_steps=0, tol=1e-9)
    path = result.simulate(T=6)
    np.testing.assert_array_equal(path.default, [True, False, True, False, True, False])
    assert path.iloc[0].debt == 0
    np.testing.assert_array_equal(path.loc[path.repaid, "debt"], np.zeros(3))
    # The repayment budget also uses the reset debt, not the saving boundary.
    paid = path[path.repaid]
    np.testing.assert_allclose(paid.c, paid.y + paid.q*paid.b_next, atol=1e-14)


def test_builtin_diagnostics_match_the_public_path_with_the_same_seed():
    result = solve_sovereign_default(n_b=41, n_y=7, b_min=-.3, b_max=.3,
                                     sigma_eps=.02, tol=1e-9)
    path = result.simulate(T=1500, seed=42)
    paid = path[path.repaid]
    assert path.default.any()
    assert result.metadata["spread_observations"] == len(paid)
    assert result.spread_mean == pytest.approx(paid.spread.mean(), abs=1e-13)
    assert result.spread_y_corr == pytest.approx(paid.spread.corr(paid.y), abs=1e-13)


@pytest.mark.parametrize("bounds,n_b", [((-.3, .3), 24), ((-.001, .3), 3)])
def test_saving_grid_contains_exact_zero_without_changing_size_or_bounds(bounds, n_b):
    result = solve_sovereign_default(n_b=n_b, n_y=3, b_min=bounds[0], b_max=bounds[1],
                                     beta=.9, sigma_eps=.05, tol=1e-9)
    assert len(result.b_grid) == n_b
    assert (result.b_grid[0], result.b_grid[-1]) == bounds
    assert np.all(np.diff(result.b_grid) > 0)
    assert np.count_nonzero(result.b_grid == 0) == 1
    assert result.simulate(T=1).iloc[0].debt == 0


def test_two_point_saving_grid_cannot_represent_zero_and_both_bounds():
    with pytest.raises(ValueError, match="n_b.*3"):
        solve_sovereign_default(n_b=2, n_y=3, b_min=-.1, b_max=.1)


@pytest.mark.parametrize("theta", [-.01, 1.01, np.nan, np.inf])
def test_reentry_probability_is_finite_and_between_zero_and_one(theta):
    with pytest.raises(ValueError, match="theta"):
        solve_sovereign_default(theta=theta, n_b=5, n_y=3, max_iter=2)
