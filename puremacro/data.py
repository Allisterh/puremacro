"""Panel-shaping helpers and trend/cycle filters for puremacro.

No I/O assumed: callers pass in DataFrames they already have. The
``hp_filter``, ``hamilton_filter``, and ``bk_filter`` helpers return
trend/cycle decompositions usable in teaching and applied work.

See also
--------
For data **fetchers** (not transforms), see :mod:`puremacro.fetch`
which exposes ``fetch_fred``, ``fetch_fred_alfred``, and the new
``sdmx_get`` (0.6.0+) for OECD/Eurostat/ECB/IMF SDMX-CSV endpoints.
This module (``puremacro.data``) only contains panel-level
transforms (long↔wide, country slicing, HP / Hamilton / BK filters).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Panel reshaping
# ---------------------------------------------------------------------------
def long_to_wide(panel_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot a long-form panel to a (code, date) x variable wide frame.

    ``panel_long`` must have columns ``code, date, variable, value``.
    """
    wide = (
        panel_long.pivot_table(
            index=["code", "date"], columns="variable", values="value"
        ).sort_index()
    )
    wide.columns.name = None
    return wide


def slice_country(wide: pd.DataFrame, code: str) -> pd.DataFrame:
    """Return the per-country wide slice with a sorted DatetimeIndex."""
    if code not in wide.index.get_level_values("code"):
        raise KeyError(f"country {code!r} not in panel")
    sub = wide.xs(code, level="code").copy()
    sub.index = pd.DatetimeIndex(sub.index)
    return sub.sort_index()


def align_yx(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
    start: str | None = None,
    end: str | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Return ``(y, x)`` both non-null, inner-joined in time, within ``[start, end]``."""
    sub = df[[y_col, x_col]].dropna()
    if start is not None:
        sub = sub[sub.index >= pd.Period(start, freq="Q").to_timestamp()]
    if end is not None:
        sub = sub[sub.index <= pd.Period(end, freq="Q").to_timestamp(how="end")]
    return sub[y_col], sub[x_col]


# ---------------------------------------------------------------------------
# Trend / cycle filters
# ---------------------------------------------------------------------------
def hp_filter(y, lamb: float = 1600.0) -> tuple[pd.Series, pd.Series]:
    """Hodrick-Prescott filter via closed-form ``(I + lambda K' K)^{-1} y``.

    Default ``lamb=1600`` is the Ravn-Uhlig-style quarterly value.
    Returns ``(cycle, trend)`` as Series with the same index as ``y``.
    """
    y_arr = np.asarray(y, dtype=float).ravel()
    n = y_arr.size
    if n < 4:
        raise ValueError("hp_filter requires at least 4 observations")
    # Second-difference matrix K of shape (n-2, n).
    K = np.zeros((n - 2, n))
    for i in range(n - 2):
        K[i, i] = 1.0
        K[i, i + 1] = -2.0
        K[i, i + 2] = 1.0
    A = np.eye(n) + lamb * (K.T @ K)
    trend_arr = np.linalg.solve(A, y_arr)
    cycle_arr = y_arr - trend_arr
    if isinstance(y, pd.Series):
        idx = y.index
    else:
        idx = pd.RangeIndex(n)
    cycle = pd.Series(cycle_arr, index=idx, name="cycle")
    trend = pd.Series(trend_arr, index=idx, name="trend")
    return cycle, trend


def hamilton_filter(y, *, h: int = 8, p: int = 4) -> tuple[pd.Series, pd.Series]:
    """Hamilton (2018) regression filter.

    Regress ``y_{t+h}`` on a constant and ``y_t, y_{t-1}, ..., y_{t-(p-1)}``;
    the residual is the cycle, the fitted value is the trend. The first
    ``h + p - 1`` entries of the output are NaN by construction.

    Thin wrapper around :func:`puremacro.cycles.hamilton_filter`, which is
    the canonical implementation; this version always returns a pair of
    ``pandas.Series`` named ``"cycle"`` and ``"trend"`` carrying ``y``'s index
    (or a ``RangeIndex`` for array input). It raises ``ValueError`` when
    ``y`` is not finite or ``T < h + 2p + 1``.
    """
    from .cycles import hamilton_filter as _hamilton

    if isinstance(y, pd.Series):
        idx = y.index
        y_arr = y.values.astype(float)
    else:
        y_arr = np.asarray(y, dtype=float).ravel()
        idx = pd.RangeIndex(y_arr.size)
    res = _hamilton(y_arr, h=h, p=p)
    cycle = pd.Series(np.asarray(res.cycle, dtype=float), index=idx, name="cycle")
    trend = pd.Series(np.asarray(res.trend, dtype=float), index=idx, name="trend")
    return cycle, trend


def bk_filter(y, low: int = 6, high: int = 32, K: int = 12) -> pd.Series:
    """Baxter-King symmetric band-pass filter.

    Pass the cyclical band ``[low, high]`` (in periods) using truncation
    ``K`` on each side. Returns a Series with ``K`` NaN at each end.

    Weights:
        omega_l = 2 pi / high
        omega_h = 2 pi / low
        b_0 = (omega_h - omega_l) / pi
        b_h = (sin(h omega_h) - sin(h omega_l)) / (h pi),  h = 1..K
    Weights are normalised by subtracting their mean so they sum to zero.
    """
    if low <= 0 or high <= low:
        raise ValueError("bk_filter requires 0 < low < high")
    if isinstance(y, pd.Series):
        idx = y.index
        y_arr = y.values.astype(float)
    else:
        y_arr = np.asarray(y, dtype=float).ravel()
        idx = pd.RangeIndex(y_arr.size)
    n = y_arr.size
    if n <= 2 * K:
        raise ValueError("bk_filter: series shorter than 2K+1")
    omega_l = 2.0 * np.pi / high
    omega_h = 2.0 * np.pi / low
    b = np.zeros(K + 1)
    b[0] = (omega_h - omega_l) / np.pi
    for hk in range(1, K + 1):
        b[hk] = (np.sin(hk * omega_h) - np.sin(hk * omega_l)) / (hk * np.pi)
    weights = np.concatenate([b[:0:-1], b])  # length 2K+1
    weights = weights - weights.mean()
    out = np.full(n, np.nan)
    for t in range(K, n - K):
        out[t] = float(np.dot(weights, y_arr[t - K: t + K + 1]))
    return pd.Series(out, index=idx, name="cycle")


__all__ = [
    "long_to_wide",
    "slice_country",
    "align_yx",
    "hp_filter",
    "hamilton_filter",
    "bk_filter",
]
