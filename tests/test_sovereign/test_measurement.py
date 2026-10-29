"""Public ergodic measurement and welfare for the Arellano sovereign model."""
from __future__ import annotations

import time

import numpy as np
import pytest
from scipy.stats import t as student_t

from _timing import budget
from puremacro.sovereign import solve_sovereign_default
from puremacro.sovereign.measurement import MOMENTS, _simulate_arrays, ergodic_moments

# The public calibration of tests/test_arellano_2008_study.py::_public_monotone.
PUBLIC = dict(beta=.95282, r_star=.017, theta=.282, rho_y=.945, sigma_y=.025,
              b_min=-.15, b_max=.33, sigma_eps=0, y_penalty_type="arellano",
              enforce_monotonicity=False, tol=1e-10, max_iter=10000, policy_search="monotone")


@pytest.fixture(scope="module")
def public_small():
    return solve_sovereign_default(**PUBLIC, n_b=201, n_y=41)


@pytest.fixture(scope="module")
def nosave_smooth():
    return solve_sovereign_default(n_b=40, n_y=7, b_min=0., b_max=.4, sigma_eps=.01)


@pytest.mark.parametrize("fixture", ["public_small", "nosave_smooth"])
@pytest.mark.parametrize("seed", [0, 1979])
def test_path_equals_simulate(fixture, seed, request):
    result = request.getfixturevalue(fixture)
    T = 3000
    fast = _simulate_arrays(result, T, seed)
    slow = result.simulate(T=T, seed=seed)
    assert set(fast) >= {"y", "default", "entry", "repaid", "b", "b_next", "spread"}
    for key in ("y", "default", "repaid", "b", "b_next"):
        assert np.array_equal(fast[key], slow[key].to_numpy()), key
    np.testing.assert_array_equal(fast["spread"], slow["spread"].to_numpy())
    assert all(len(v) == T for v in fast.values())


def test_entries_exclude_continuing_exclusion(public_small):
    path = _simulate_arrays(public_small, 100_000, 7)
    default = path["default"]
    previous = np.roll(default, 1)
    previous[0] = False
    assert path["entry"].sum() == (default & ~previous).sum()
    assert path["entry"].sum() < default.sum()


def _by_hand(path: dict, burn: int) -> dict:
    p = {k: v[burn:] for k, v in path.items()}
    rep = p["repaid"]
    entries = int(p["entry"].sum())
    n_rep = int(rep.sum())
    spread = p["spread"][rep]
    return {
        "debt_output_pct": 100 * p["b_next"][rep].mean() / p["y"][rep].mean(),
        "spread_mean_pct": 100 * spread.mean(),
        "default_annual_pct": 400 * entries / (n_rep + entries),
        "spread_sd_pp": 100 * spread.std(ddof=1),
    }


def test_moments_follow_their_definitions(public_small):
    seeds, periods, burn = [1, 2, 3], 20_000, 100
    table = ergodic_moments(public_small, periods=periods, seeds=seeds, burn=burn)
    assert tuple(table.index) == MOMENTS and table.index.name == "moment"
    assert list(table.columns) == ["mean", "se", "ci95_low", "ci95_high"] + [f"seed_{s}" for s in seeds]
    for s in seeds:
        expected = _by_hand(_simulate_arrays(public_small, periods, s), burn)
        for m in MOMENTS:
            assert table.loc[m, f"seed_{s}"] == pytest.approx(expected[m], abs=1e-12)
    for m in MOMENTS:
        values = table.loc[m, [f"seed_{s}" for s in seeds]].to_numpy(dtype=float)
        se = values.std(ddof=1) / np.sqrt(len(values))
        half = student_t.ppf(.975, len(values) - 1) * se
        assert table.loc[m, "mean"] == pytest.approx(values.mean(), abs=1e-12)
        assert table.loc[m, "se"] == pytest.approx(se, abs=1e-12)
        assert table.loc[m, "ci95_low"] == pytest.approx(values.mean() - half, abs=1e-12)
        assert table.loc[m, "ci95_high"] == pytest.approx(values.mean() + half, abs=1e-12)


@pytest.mark.parametrize("kwargs", [
    dict(periods=100, burn=100),
    dict(periods=100, burn=-1),
    dict(periods=1000, seeds=[1]),
    dict(periods=1000, seeds=[1, 1]),
])
def test_validation(public_small, kwargs):
    kwargs = {"seeds": [1, 2], "burn": 10, **kwargs}
    with pytest.raises(ValueError):
        ergodic_moments(public_small, **kwargs)


def test_too_few_repayment_periods_raises():
    # No reentry and no output penalty: default is absorbing and, with taste
    # shocks, happens within a few quarters. (With the default penalty the
    # default probability is ~1e-26 and no seed ever defaults.)
    result = solve_sovereign_default(theta=0., d0=0., d1=0., n_b=40, n_y=7, sigma_eps=.01)
    path = _simulate_arrays(result, 20_000, 1)
    assert np.flatnonzero(path["entry"])[0] < 19_000
    with pytest.raises(ValueError, match="seed"):
        ergodic_moments(result, periods=20_000, seeds=[1, 2], burn=19_000)


def test_default_design_hang_guard(public_small):
    start = time.perf_counter()
    table = ergodic_moments(public_small)
    elapsed = time.perf_counter() - start
    assert table.shape == (4, 4 + 16)
    assert elapsed <= budget(120), f"{elapsed:.1f} s > budget {budget(120):.0f} s"
