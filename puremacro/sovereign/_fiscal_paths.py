"""Vectorized simulation of solved fiscal sovereign-default models.

Paths share pre-drawn uniforms (common random numbers), so different
policies or rules face identical income histories and default/reentry draws.

Timing follows the solver: the first period's decision happens at the given
``(b, y)`` state and income is drawn from the second period on; an excluded
path draws reentry before that period's decision and, on reentry, decides
from the zero-debt node in the same period.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Sequence

import numpy as np

if TYPE_CHECKING:
    from puremacro.sovereign._results import FiscalSovereignResult

__all__ = ["MOMENTS", "FiscalPaths", "draw_uniforms", "ergodic_moments", "simulate_paths"]

MOMENTS = ("debt_pct", "spread_mean_pct", "annual_default_pct", "spread_std_pp", "spending_pct",
           "corr_pb_y", "boundary_choice_share_pct")


class FiscalPaths(NamedTuple):
    """Simulated paths; every array has shape ``(n_paths, periods)``.

    ``spread`` and ``bond_price`` are NaN outside repayment periods; ``gdp`` is
    ``y`` when repaying and the default-cost output ``y_def`` otherwise.
    ``last_debt_index`` is the index of the top debt node (for boundary shares).
    """

    y_index: np.ndarray
    b_index: np.ndarray
    b_next_index: np.ndarray
    access: np.ndarray
    default_entry: np.ndarray
    repaid: np.ndarray
    y: np.ndarray
    gdp: np.ndarray
    g: np.ndarray
    primary_balance: np.ndarray
    debt_to_gdp: np.ndarray
    spread: np.ndarray
    bond_price: np.ndarray
    last_debt_index: int


def draw_uniforms(seeds: Sequence[int], periods: int) -> np.ndarray:
    """Uniforms of shape ``(3, len(seeds), periods)``: income, default, reentry planes.

    Path ``k`` is ``np.random.default_rng(seeds[k]).random((3, periods))``.
    """
    if periods < 1:
        raise ValueError(f"periods must be >= 1; got {periods}")
    out = np.empty((3, len(seeds), periods))
    for k, seed in enumerate(seeds):
        out[:, k, :] = np.random.default_rng(seed).random((3, periods))
    return out


def _start_indices(index: int | np.ndarray, n_paths: int, size: int, name: str) -> np.ndarray:
    arr = np.broadcast_to(np.asarray(index), (n_paths,)).astype(int)
    if arr.min() < 0 or arr.max() >= size:
        raise ValueError(f"{name} must be in [0, {size - 1}]")
    return arr.copy()


def simulate_paths(result: FiscalSovereignResult, b_index: int | np.ndarray, y_index: int | np.ndarray,
                   uniforms: np.ndarray) -> FiscalPaths:
    """Simulate every path of ``uniforms`` from ``(b_index, y_index)`` in good standing."""
    u = np.asarray(uniforms, dtype=float)
    if u.ndim != 3 or u.shape[0] != 3:
        raise ValueError("uniforms must have shape (3, n_paths, periods)")
    _, n_paths, periods = u.shape
    n_b, n_y = result.default_prob.shape
    b = _start_indices(b_index, n_paths, n_b, "b_index")
    y = _start_indices(y_index, n_paths, n_y, "y_index")
    cum = np.cumsum(result.p_trans, axis=1)
    theta = float(result.metadata["theta"])
    zero = int(result.metadata["zero_index"])
    y_def = np.asarray(result.metadata["y_default_vec"])
    access = np.ones(n_paths, dtype=bool)

    int_shape, shape = (n_paths, periods), (n_paths, periods)
    out_y_index = np.empty(int_shape, dtype=int)
    out_b_index = np.empty(int_shape, dtype=int)
    out_b_next = np.empty(int_shape, dtype=int)
    out_access = np.empty(shape, dtype=bool)
    out_entry = np.empty(shape, dtype=bool)
    out_repaid = np.empty(shape, dtype=bool)
    floats = {name: np.empty(shape) for name in
              ("y", "gdp", "g", "primary_balance", "debt_to_gdp", "spread", "bond_price")}

    for t in range(periods):
        if t > 0:
            y = np.minimum((u[0, :, t][:, None] >= cum[y]).sum(axis=1), n_y - 1)
            reentry = ~access & (u[2, :, t] < theta)
            b = np.where(reentry, zero, b)
            access = access | reentry
        entry = access & (u[1, :, t] < result.default_prob[b, y])
        repaid = access & ~entry
        b_next = np.where(repaid, result.policy_index[b, y], zero)

        out_y_index[:, t], out_b_index[:, t], out_b_next[:, t] = y, b, b_next
        out_access[:, t], out_entry[:, t], out_repaid[:, t] = access, entry, repaid
        floats["y"][:, t] = result.y_grid[y]
        floats["gdp"][:, t] = np.where(repaid, result.y_grid[y], y_def[y])
        floats["g"][:, t] = np.where(repaid, result.g_policy[b, y], result.g_default[y])
        floats["primary_balance"][:, t] = np.where(repaid, result.primary_balance[b, y], 0.0)
        floats["debt_to_gdp"][:, t] = np.where(repaid, result.debt_to_gdp[b, y], 0.0)
        floats["spread"][:, t] = np.where(repaid, result.spread[b_next, y], np.nan)
        floats["bond_price"][:, t] = np.where(repaid, result.bond_price[b_next, y], np.nan)

        b, access = b_next, repaid

    return FiscalPaths(y_index=out_y_index, b_index=out_b_index, b_next_index=out_b_next,
                       access=out_access, default_entry=out_entry, repaid=out_repaid,
                       last_debt_index=n_b - 1, **floats)


def _mean(x: np.ndarray) -> float:
    return float(x.mean()) if x.size else float("nan")


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) <= 1e-14 or np.std(b) <= 1e-14:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def ergodic_moments(paths: FiscalPaths, burn: int) -> list[dict[str, float]]:
    """Per-path moments over periods ``t >= burn`` (one dict per path).

    Repayment-period means of ``100*debt_to_gdp`` and ``100*spread``, the
    sample std (ddof=1) of ``100*spread``, ``400 * entries / access periods``,
    the all-period mean of ``100*g/gdp``, the repayment-period correlation of
    ``primary_balance/gdp`` with ``log y``, and the percent of repayment
    periods choosing the top debt node.
    """
    if burn < 0 or burn >= paths.repaid.shape[1]:
        raise ValueError("burn must leave at least one period")
    out = []
    for k in range(paths.repaid.shape[0]):
        sl = slice(burn, None)
        rep = paths.repaid[k, sl]
        spread = paths.spread[k, sl][rep]
        access = int(paths.access[k, sl].sum())
        entries = int(paths.default_entry[k, sl].sum())
        gdp = paths.gdp[k, sl]
        out.append({
            "debt_pct": 100.0 * _mean(paths.debt_to_gdp[k, sl][rep]),
            "spread_mean_pct": 100.0 * _mean(spread),
            "annual_default_pct": 400.0 * entries / access if access else float("nan"),
            "spread_std_pp": 100.0 * float(spread.std(ddof=1)) if spread.size > 1 else float("nan"),
            "spending_pct": 100.0 * _mean(paths.g[k, sl] / gdp),
            "corr_pb_y": _correlation((paths.primary_balance[k, sl] / gdp)[rep], np.log(paths.y[k, sl][rep])),
            "boundary_choice_share_pct": 100.0 * _mean(paths.b_next_index[k, sl][rep] == paths.last_debt_index),
        })
    return out
