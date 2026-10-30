"""Return construction, excess returns, compounding and annualization."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ._validate import _align, _as_frame, _check_finite, _validate_returns

__all__ = [
    "to_returns",
    "excess_returns",
    "compound",
    "annualize_return",
    "annualize_volatility",
]


def _restore(df: pd.DataFrame, like: Any):
    """Return ``df`` in the container type of ``like``."""
    if isinstance(like, pd.DataFrame):
        return df
    if isinstance(like, pd.Series):
        out = df.iloc[:, 0]
        out.name = like.name
        return out
    arr = np.asarray(like)
    vals = df.to_numpy()
    return vals[:, 0] if arr.ndim == 1 else vals


def _to_pandas(x: Any):
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x
    arr = np.asarray(x, dtype=float)
    return pd.Series(arr) if arr.ndim == 1 else pd.DataFrame(arr)


def to_returns(prices: Any, kind: str = "simple"):
    """Returns from a price (or total-return index) series.

    ``kind="simple"``: ``p_t / p_{t-1} - 1``; ``kind="log"``:
    ``log(p_t / p_{t-1})``. The first row is dropped. A Series gives a
    Series and a DataFrame a DataFrame (ndarray input is converted to the
    pandas equivalent). Non-positive or non-finite prices raise
    ``ValueError``.
    """
    if kind not in ("simple", "log"):
        raise ValueError(f"to_returns: kind must be 'simple' or 'log'; got {kind!r}")
    p = _to_pandas(prices)
    df = _check_finite(_as_frame(p, "prices", prefix="p"), "prices")
    bad = [str(c) for c in df.columns if np.any(df[c].to_numpy() <= 0)]
    if bad:
        raise ValueError(f"prices: non-positive values in column(s) {bad}; prices must be positive")
    ratio = df / df.shift(1)
    out = ratio - 1.0 if kind == "simple" else np.log(ratio)
    return _restore(out.iloc[1:], p)


def excess_returns(returns: Any, rf: Any, *, check_scale: bool = True):
    """``returns - rf``, pairing rows by date.

    ``rf`` is a scalar or a single series in the same per-period decimal
    units. With DatetimeIndex inputs the rows are inner-joined on dates (an
    ``rf`` with extra dates or another order is fine; mixed frequencies
    raise). Otherwise the inputs are paired by position and must have equal
    length. The output has the container type of ``returns``.
    ``check_scale=False`` skips the percent-scale check on both inputs.
    """
    r = _validate_returns(returns, "returns", check_scale)
    if np.ndim(rf) == 0 and not isinstance(rf, (pd.Series, pd.DataFrame)):
        rf_df = _validate_returns(np.full(len(r), float(rf)), "rf", check_scale)
        rf_df.index = r.index
    else:
        rf_df = _validate_returns(rf, "rf", check_scale)
    if rf_df.shape[1] != 1:
        raise ValueError(f"rf must be a single series; got {rf_df.shape[1]} columns")
    r, rf_df = _align(r, rf_df, names=("returns", "rf"))
    out = r.sub(rf_df.iloc[:, 0], axis=0)
    return _restore(out, returns)


_PERIODS = {"M": "M", "Q": "Q", "A": "Y"}


def compound(returns: Any, to: str = "M", kind: str = "simple", *,
             check_scale: bool = True):
    """Aggregate returns to a lower frequency.

    ``kind="simple"``: ``prod(1 + r) - 1`` within each period; ``"log"``:
    the sum. ``to`` is ``"M"``, ``"Q"`` or ``"A"``; the input needs a
    DatetimeIndex and the output is indexed by period start.
    """
    if to not in _PERIODS:
        raise ValueError(f"compound: to must be 'M', 'Q' or 'A'; got {to!r}")
    if kind not in ("simple", "log"):
        raise ValueError(f"compound: kind must be 'simple' or 'log'; got {kind!r}")
    r = _validate_returns(returns, "returns", check_scale)
    if not isinstance(r.index, pd.DatetimeIndex):
        raise ValueError("compound: returns need a DatetimeIndex")
    key = r.index.to_period(_PERIODS[to]).start_time
    g = r.groupby(key)
    out = (1.0 + r).groupby(key).prod() - 1.0 if kind == "simple" else g.sum()
    out.index = pd.DatetimeIndex(out.index, name=r.index.name)
    return _restore(out, returns)


def _check_ppy(periods_per_year: Any) -> float:
    try:
        ppy = float(periods_per_year)
    except (TypeError, ValueError):
        raise ValueError(f"periods_per_year must be a positive number; got {periods_per_year!r}") from None
    if not np.isfinite(ppy) or ppy <= 0:
        raise ValueError(f"periods_per_year must be a positive number; got {periods_per_year!r}")
    return ppy


def _scalar_or_series(values: np.ndarray, df: pd.DataFrame, like: Any):
    if isinstance(like, pd.DataFrame):
        return pd.Series(values, index=df.columns)
    if np.ndim(like) == 2 and not isinstance(like, pd.Series):
        return np.asarray(values)
    return float(values[0])


def annualize_return(returns: Any, periods_per_year: float, *, check_scale: bool = True):
    """Geometric annualized return ``prod(1 + r)^(ppy / T) - 1``."""
    ppy = _check_ppy(periods_per_year)
    r = _validate_returns(returns, "returns", check_scale)
    T = len(r)
    vals = np.prod(1.0 + r.to_numpy(), axis=0) ** (ppy / T) - 1.0
    return _scalar_or_series(vals, r, returns)


def annualize_volatility(returns: Any, periods_per_year: float, *, check_scale: bool = True):
    """Annualized volatility ``std(r, ddof=1) * sqrt(ppy)``."""
    ppy = _check_ppy(periods_per_year)
    r = _validate_returns(returns, "returns", check_scale)
    vals = r.to_numpy().std(axis=0, ddof=1) * np.sqrt(ppy)
    return _scalar_or_series(vals, r, returns)
