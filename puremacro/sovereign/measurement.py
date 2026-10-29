"""Ergodic measurement of a solved sovereign default model.

:func:`ergodic_moments` measures long simulated paths with exactly the timing
of :meth:`SovereignDefaultResult.simulate`, draw for draw, but without building
a DataFrame per quarter, so 16 paths of 500,000 quarters take seconds.
"""
from __future__ import annotations

from array import array
from bisect import bisect_right
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from puremacro.sovereign._results import SovereignDefaultResult

__all__ = ["MOMENTS", "ergodic_moments"]

MOMENTS: tuple[str, ...] = ("debt_output_pct", "spread_mean_pct", "default_annual_pct", "spread_sd_pp")

_REPAID, _ENTRY, _EXCLUDED = 0, 1, 2


def _next_debt_index(result: SovereignDefaultResult) -> np.ndarray:
    """``argmin(|b_grid - policy_b|)`` per state, ties to the lower index, without an n_b^2 n_y tensor."""
    b_grid = np.asarray(result.b_grid, dtype=float)
    policy = np.asarray(result.policy_b, dtype=float)
    upper = np.clip(np.searchsorted(b_grid, policy, side="left"), 1, len(b_grid) - 1)
    lower = upper - 1
    # Same float expression as simulate's argmin; argmin keeps the first minimum.
    take_upper = np.abs(b_grid[upper] - policy) < np.abs(b_grid[lower] - policy)
    return np.where(take_upper, upper, lower)


def _simulate_arrays(result: SovereignDefaultResult, periods: int, seed: int) -> dict[str, np.ndarray]:
    """The path of ``result.simulate(T=periods, seed=seed)`` as arrays, draw for draw.

    Keys: ``y``, ``default`` (simulate's flag, true in every excluded quarter),
    ``entry`` (an explicit default decision by an economy with access),
    ``repaid``, ``b``, ``b_next`` (0.0 when not repaid) and ``spread`` (NaN when
    not repaid). No burn is removed.
    """
    periods = int(periods)
    n_y = len(result.y_grid)
    p_trans = np.asarray(result.p_trans, dtype=float)
    # Generator.choice(n, p=row) draws one double u and returns
    # searchsorted(cumsum(row) / cumsum(row)[-1], u, side="right").
    cdf = np.cumsum(p_trans, axis=1)
    cdf /= cdf[:, -1:]
    cdf_rows = [row.tolist() for row in cdf]
    default_prob = array("d", np.asarray(result.default_prob, dtype=float).ravel().tolist())
    next_idx = _next_debt_index(result)
    next_flat = array("q", next_idx.ravel().tolist())
    theta = float(result.metadata.get("theta", 0.282))
    zero_idx = int(np.argmin(np.abs(result.b_grid)))

    rng = np.random.default_rng(seed)
    # At most three draws per quarter (income, reentry, default), consumed in order.
    u = rng.random(3 * periods).tolist()

    y_path = array("q", bytes(8 * periods))
    b_path = array("q", bytes(8 * periods))
    state = array("b", bytes(periods))
    y_idx, b_idx, excluded, k = n_y // 2, zero_idx, False, 0
    for t in range(periods):
        y_idx = bisect_right(cdf_rows[y_idx], u[k])
        k += 1
        y_path[t] = y_idx
        if excluded:
            reentered = u[k] < theta
            k += 1
            if not reentered:
                state[t] = _EXCLUDED
                continue
            excluded = False
            b_idx = zero_idx
        b_path[t] = b_idx
        flat = b_idx * n_y + y_idx
        if u[k] < default_prob[flat]:
            k += 1
            excluded = True
            state[t] = _ENTRY
            continue
        k += 1
        state[t] = _REPAID
        b_idx = next_flat[flat]

    yi = np.frombuffer(y_path, dtype=np.int64)
    bi = np.frombuffer(b_path, dtype=np.int64)
    code = np.frombuffer(state, dtype=np.int8)
    repaid = code == _REPAID
    entry = code == _ENTRY
    b_grid = np.asarray(result.b_grid, dtype=float)
    b = np.where(code == _EXCLUDED, 0.0, b_grid[bi])
    b_next = np.where(repaid, np.asarray(result.policy_b, dtype=float)[bi, yi], 0.0)
    spread = np.where(repaid, np.asarray(result.spread, dtype=float)[next_idx[bi, yi], yi], np.nan)
    return {
        "y": np.asarray(result.y_grid, dtype=float)[yi],
        "default": ~repaid,
        "entry": entry,
        "repaid": repaid,
        "b": b,
        "b_next": b_next,
        "spread": spread,
    }


def ergodic_moments(
    result: SovereignDefaultResult,
    *,
    periods: int = 500_000,
    seeds: Iterable[int] = range(1979, 1995),
    burn: int = 1_000,
) -> pd.DataFrame:
    """Raw ergodic moments of a solved model, with 95% intervals across seeds.

    Each seed is one path of ``periods`` quarters with exactly the timing of
    :meth:`SovereignDefaultResult.simulate` (same random stream, start at the
    middle income node and zero debt, reentry before the current period's
    decision). The first ``burn`` quarters are dropped. Over the remaining
    quarters, with *rep* the repayment quarters and an *entry* an explicit
    default decision by an economy with market access (continuing exclusion is
    not counted):

    * ``debt_output_pct = 100 * mean(b_next[rep]) / mean(y[rep])``, debt chosen
      as a percentage of mean income;
    * ``spread_mean_pct = 100 * mean(spread[rep])``, using the solver's linear
      annualization ``4 * max(0, yield - r_star)``;
    * ``default_annual_pct = 400 * entries / (n_rep + entries)``, the quarterly
      default hazard among quarters with access, times 4 (no compounding);
    * ``spread_sd_pp = 100 * std(spread[rep], ddof=1)``.

    This is a raw ergodic measurement. It is **not** the 74-quarter pre-default
    windows of the reference study, and its spread convention is not the
    study's ``q**-4 - R**4``, so its numbers differ from
    ``docs/arellano_replication.md``.

    Returns
    -------
    pandas.DataFrame
        Index ``MOMENTS`` (named ``"moment"``). Columns ``mean``, ``se``
        (standard deviation across seeds over sqrt(n)), ``ci95_low`` and
        ``ci95_high`` (Student-t, n-1 degrees of freedom), then one column
        ``seed_<s>`` per seed in input order, so two results pair by seed.

    Raises
    ------
    ValueError
        If ``burn < 0``, ``periods <= burn``, fewer than two or duplicate seeds
        are given, or a seed has fewer than two repayment quarters after the
        burn (the message names the seed).
    """
    periods, burn = int(periods), int(burn)
    seeds = [int(s) for s in seeds]
    if burn < 0:
        raise ValueError(f"burn must be >= 0; got {burn}")
    if periods <= burn:
        raise ValueError(f"periods ({periods}) must exceed burn ({burn})")
    if len(seeds) < 2:
        raise ValueError("ergodic_moments needs at least two seeds for an interval")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds must be distinct; got {seeds}")

    per_seed: dict[str, list[float]] = {m: [] for m in MOMENTS}
    for seed in seeds:
        path = {k: v[burn:] for k, v in _simulate_arrays(result, periods, seed).items()}
        rep = path["repaid"]
        n_rep = int(rep.sum())
        if n_rep < 2:
            raise ValueError(f"seed {seed}: only {n_rep} repayment quarters after a burn of {burn}; "
                             "moments are undefined (lengthen periods or shorten burn)")
        entries = int(path["entry"].sum())
        spread = path["spread"][rep]
        per_seed["debt_output_pct"].append(100.0 * float(path["b_next"][rep].mean()) / float(path["y"][rep].mean()))
        per_seed["spread_mean_pct"].append(100.0 * float(spread.mean()))
        per_seed["default_annual_pct"].append(400.0 * entries / (n_rep + entries))
        per_seed["spread_sd_pp"].append(100.0 * float(spread.std(ddof=1)))

    n = len(seeds)
    t_crit = float(student_t.ppf(0.975, n - 1))
    rows = {}
    for m in MOMENTS:
        values = np.asarray(per_seed[m], dtype=float)
        mean = float(values.mean())
        se = float(values.std(ddof=1) / np.sqrt(n))
        rows[m] = [mean, se, mean - t_crit * se, mean + t_crit * se, *values.tolist()]
    columns = ["mean", "se", "ci95_low", "ci95_high"] + [f"seed_{s}" for s in seeds]
    table = pd.DataFrame.from_dict(rows, orient="index", columns=columns)
    table.index.name = "moment"
    return table
