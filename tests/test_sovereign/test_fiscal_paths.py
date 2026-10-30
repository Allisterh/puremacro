"""Vectorized fiscal path engine with common random numbers."""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.sovereign import FiscalSovereignResult, solve_fiscal_sovereign_default
from puremacro.sovereign._fiscal_paths import (
    MOMENTS,
    FiscalPaths,
    draw_uniforms,
    ergodic_moments,
    simulate_paths,
)

ZERO = 1  # b_grid = [-0.01, 0, 0.02]


def _toy(default_prob: np.ndarray | None = None, policy_index: np.ndarray | None = None) -> FiscalSovereignResult:
    """Hand-built 3 x 2 fiscal result; only the fields the engine reads matter."""
    b = np.array([-0.01, 0.0, 0.02])
    y = np.array([0.9, 1.1])
    tau = 0.2
    q = np.array([[0.98, 0.98], [0.98, 0.98], [0.9, 0.95]])
    m = np.array([[1, 2], [1, 2], [0, 1]]) if policy_index is None else policy_index
    pdef = np.zeros((3, 2)) if default_prob is None else default_prob
    q_chosen = np.take_along_axis(q, m, axis=0)
    y_def = np.minimum(y, 0.969)
    return FiscalSovereignResult(
        b_grid=b, y_grid=y, bond_price=q, q_bond=q, default_prob=pdef,
        policy_b=b[m], policy_b_prime=b[m], spread=4.0 * (1.0 / q - 1.0 - 0.01),
        p_trans=np.array([[0.5, 0.5], [0.5, 0.5]]),
        metadata={"theta": 0.5, "zero_index": ZERO, "y_default_vec": y_def},
        tau=tau, psi=1.0, policy_index=m,
        g_policy=tau * y[None, :] - b[:, None] + q_chosen * b[m], g_default=tau * y_def,
        primary_balance=b[:, None] - q_chosen * b[m], debt_to_gdp=b[m] / y[None, :],
        private_consumption_repay=(1 - tau) * y, private_consumption_default=(1 - tau) * y_def,
    )


def _uniforms(income, default, reentry) -> np.ndarray:
    return np.array([[income], [default], [reentry]], dtype=float)


def test_draw_uniforms_rows_are_seed_streams() -> None:
    u = draw_uniforms([3, 7], 5)
    assert u.shape == (3, 2, 5)
    assert np.array_equal(u[:, 1, :], np.random.default_rng(7).random((3, 5)))


def test_first_period_decides_at_the_given_state() -> None:
    res = _toy()
    # Income plane at t=0 is ignored; later draws follow cumulative rows [.5, 1].
    u = _uniforms([0.99, 0.3, 0.7, 0.2], [0.5] * 4, [0.5] * 4)
    paths = simulate_paths(res, 2, 0, u)
    assert paths.y_index[0].tolist() == [0, 0, 1, 0]
    assert paths.b_index[0, 0] == 2
    assert paths.b_next_index[0, 0] == res.policy_index[2, 0]
    assert np.array_equal(paths.b_index[0, 1:], paths.b_next_index[0, :-1])
    assert paths.repaid.all() and paths.access.all() and not paths.default_entry.any()
    assert paths.spread[0, 0] == res.spread[res.policy_index[2, 0], 0]
    assert paths.g[0, 0] == res.g_policy[2, 0]
    assert paths.gdp[0, 0] == res.y_grid[0]


def test_reentry_draw_precedes_the_decision() -> None:
    pdef = np.zeros((3, 2))
    pdef[2, 0] = 1.0
    res = _toy(default_prob=pdef)
    u = _uniforms([0.0, 0.7], [0.5, 0.5], [0.9, 0.2])
    paths = simulate_paths(res, 2, 0, u)
    assert paths.default_entry[0, 0] and paths.access[0, 0] and not paths.repaid[0, 0]
    # Reentry uniform 0.2 < theta = 0.5: access and repayment in the same period from zero debt.
    assert paths.access[0, 1] and paths.repaid[0, 1]
    assert paths.b_index[0, 1] == ZERO
    assert paths.b_next_index[0, 1] == res.policy_index[ZERO, 1]


def test_default_entry_then_exclusion() -> None:
    pdef = np.zeros((3, 2))
    pdef[2, 0] = 1.0
    res = _toy(default_prob=pdef)
    u = _uniforms([0.0, 0.1, 0.1], [0.5] * 3, [0.0, 0.9, 0.1])
    paths = simulate_paths(res, 2, 0, u)
    assert paths.access[0].tolist() == [True, False, True]
    assert paths.default_entry[0].tolist() == [True, False, False]
    assert paths.repaid[0].tolist() == [False, False, True]
    assert np.isnan(paths.spread[0, :2]).all() and np.isnan(paths.bond_price[0, :2]).all()
    y_def = res.metadata["y_default_vec"]
    assert paths.gdp[0, 0] == y_def[0] and paths.gdp[0, 1] == y_def[0]
    assert paths.g[0, 1] == res.g_default[0]
    assert paths.primary_balance[0, 1] == 0.0 and paths.debt_to_gdp[0, 1] == 0.0
    assert paths.b_index[0, 1] == ZERO and paths.b_index[0, 2] == ZERO


def test_common_random_numbers() -> None:
    a = _toy()
    b = _toy(policy_index=np.array([[0, 0], [0, 1], [1, 1]]),
             default_prob=np.array([[0.0, 0.0], [0.3, 0.0], [1.0, 0.6]]))
    u = draw_uniforms([1, 2, 3], 50)
    pa, pb = simulate_paths(a, ZERO, 0, u), simulate_paths(b, ZERO, 0, u)
    assert np.array_equal(pa.y_index, pb.y_index)
    assert not np.array_equal(pa.b_next_index, pb.b_next_index)


def _known_path() -> FiscalPaths:
    def row(values, dtype=float):
        return np.array([values], dtype=dtype)

    repaid = row([True, True, False, False, True, True], bool)
    return FiscalPaths(
        y_index=row([0, 1, 0, 0, 1, 0], int), b_index=row([1, 1, 2, 1, 1, 2], int),
        b_next_index=row([1, 2, 1, 1, 2, 1], int),
        access=row([True, True, True, False, True, True], bool),
        default_entry=row([False, False, True, False, False, False], bool), repaid=repaid,
        y=row([0.9, 1.1, 0.9, 0.9, 1.1, 0.9]), gdp=row([0.9, 1.1, 0.8, 0.8, 1.1, 0.9]),
        g=row([0.18, 0.20, 0.16, 0.16, 0.25, 0.10]),
        primary_balance=row([0.0, 0.02, 0.0, 0.0, -0.03, 0.08]),
        debt_to_gdp=row([0.0, 0.02, 0.0, 0.0, 0.03, 0.01]),
        spread=row([0.01, 0.03, np.nan, np.nan, 0.05, 0.02]),
        bond_price=row([0.98, 0.95, np.nan, np.nan, 0.94, 0.97]),
        last_debt_index=2,
    )


def test_ergodic_moments_on_a_known_path() -> None:
    (m,) = ergodic_moments(_known_path(), burn=1)
    assert set(m) == set(MOMENTS)
    # Periods 1..5; repayment periods 1, 4, 5.
    assert m["debt_pct"] == pytest.approx(100 * np.mean([0.02, 0.03, 0.01]), abs=1e-12)
    assert m["spread_mean_pct"] == pytest.approx(100 * np.mean([0.03, 0.05, 0.02]), abs=1e-12)
    assert m["spread_std_pp"] == pytest.approx(100 * np.std([0.03, 0.05, 0.02], ddof=1), abs=1e-12)
    assert m["annual_default_pct"] == pytest.approx(400 * 1 / 4, abs=1e-12)
    spending = [0.20 / 1.1, 0.16 / 0.8, 0.16 / 0.8, 0.25 / 1.1, 0.10 / 0.9]
    assert m["spending_pct"] == pytest.approx(100 * np.mean(spending), abs=1e-12)
    pb = np.array([0.02 / 1.1, -0.03 / 1.1, 0.08 / 0.9])
    logy = np.log([1.1, 1.1, 0.9])
    assert m["corr_pb_y"] == pytest.approx(np.corrcoef(pb, logy)[0, 1], abs=1e-12)
    assert m["boundary_choice_share_pct"] == pytest.approx(100 * 2 / 3, abs=1e-12)


def test_simulate_columns() -> None:
    res = solve_fiscal_sovereign_default(tau=0.2, psi=1.0, n_b=41, n_y=7, b_max=0.06)
    df = res.simulate(periods=200, seed=3)
    expected = ["period", "y", "b", "b_next", "default", "repaid", "excluded", "g",
                "private_consumption", "primary_balance", "debt_to_gdp", "spread", "bond_price"]
    assert set(expected) <= set(df.columns)
    assert len(df) == 200
    assert df.equals(res.simulate(periods=200, seed=3))
    assert not df.equals(res.simulate(periods=200, seed=4))
    # Output = public + private absorption + primary balance (zero outside repayment).
    total = df["g"] + df["private_consumption"] + df["primary_balance"]
    assert np.max(np.abs(total - df["gdp"])) <= 1e-12
    outside = ~df["repaid"]
    gap = (df.loc[outside, "g"] + df.loc[outside, "private_consumption"] - df.loc[outside, "gdp"]).to_numpy()
    assert np.max(np.abs(gap), initial=0.0) <= 1e-12
