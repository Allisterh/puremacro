"""Input validation shared by the :mod:`puremacro.asset_pricing` estimators.

Every estimator refuses NaN/inf input (naming the offending columns) instead
of dropping rows, refuses percent-scaled net returns, and pairs separately
supplied series by date rather than by position.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

__all__: list[str] = []


def _as_frame(x: Any, name: str, prefix: str = "r") -> pd.DataFrame:
    """Series / ndarray / DataFrame -> float DataFrame (no value checks)."""
    if isinstance(x, pd.DataFrame):
        df = x.astype(float).copy()
    elif isinstance(x, pd.Series):
        col = x.name if x.name is not None else name
        df = x.astype(float).to_frame(name=col)
    else:
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 0:
            raise ValueError(f"{name}: expected a series or array, got a scalar")
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:
            raise ValueError(f"{name}: expected a 1-D or 2-D array; got {arr.ndim}-D")
        df = pd.DataFrame(arr, columns=[f"{prefix}{j}" for j in range(arr.shape[1])])
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError(f"{name}: empty input")
    return df


def _check_finite(df: pd.DataFrame, name: str) -> pd.DataFrame:
    bad = [str(c) for c in df.columns if not np.all(np.isfinite(df[c].to_numpy()))]
    if bad:
        raise ValueError(
            f"{name}: NaN or inf values in column(s) {bad}; drop or fill them "
            "explicitly (no rows are dropped silently)")
    return df


def _validate_finite(x: Any, name: str, prefix: str = "r") -> pd.DataFrame:
    """Float DataFrame with every value finite (no percent check)."""
    return _check_finite(_as_frame(x, name, prefix), name)


def _validate_returns(x: Any, name: str) -> pd.DataFrame:
    """Validate net returns in decimals.

    Returns a float DataFrame (a Series keeps its name, ndarray columns are
    ``"r0" .. "r{N-1}"``). Raises ``ValueError`` on NaN/inf (naming the
    columns) and when ``abs().max() > 1``, the signature of returns quoted in
    percent (Ken French files, FRED ``TB3MS``): ``1.2`` meaning 1.2% would
    otherwise inflate every statistic by a factor of 100.
    """
    df = _validate_finite(x, name)
    big = [str(c) for c in df.columns if float(np.abs(df[c].to_numpy()).max()) > 1.0]
    if big:
        raise ValueError(
            f"{name}: column(s) {big} have |value| > 1, which looks like returns "
            "in percent; pass decimals (divide by 100)")
    return df


def _coarse_freq(index: pd.DatetimeIndex) -> str | None:
    """Coarse inferred frequency code (M, Q, A, W, D, ...) or ``None``."""
    idx = pd.DatetimeIndex(index).sort_values().unique()
    if len(idx) < 3:
        return None
    try:
        f = pd.infer_freq(idx)
    except (TypeError, ValueError):
        return None
    if f is None:
        return None
    code = f.split("-")[0].lstrip("0123456789").upper()
    if code in ("MS", "ME", "M", "BMS", "BME", "BM", "SMS", "SME"):
        return "M"
    if code in ("QS", "QE", "Q", "BQS", "BQE", "BQ"):
        return "Q"
    if code in ("YS", "YE", "Y", "AS", "A", "BYS", "BYE", "BA", "BAS"):
        return "A"
    if code.startswith("W"):
        return "W"
    if code in ("B", "C", "D"):
        return "D"
    return code


def _align(*frames: pd.DataFrame, names: tuple[str, ...] | None = None
           ) -> tuple[pd.DataFrame, ...]:
    """Pair rows of several frames by date (or by position).

    When every frame has a ``DatetimeIndex`` the rows are inner-joined on the
    index (sorted), and two inputs whose frequencies are both inferable and
    differ raise ``ValueError``. Otherwise the frames must have equal length
    and are paired by position (on the first frame's index).
    """
    if names is None:
        names = tuple(f"input {i}" for i in range(len(frames)))
    if all(isinstance(f.index, pd.DatetimeIndex) for f in frames):
        freqs = [(_coarse_freq(f.index), n) for f, n in zip(frames, names)]
        known = [(c, n) for c, n in freqs if c is not None]
        for c, n in known[1:]:
            if c != known[0][0]:
                raise ValueError(
                    f"mixed frequencies: {known[0][1]} is {known[0][0]!r} but {n} is "
                    f"{c!r}; resample (e.g. asset_pricing.compound) before combining")
        for f, n in zip(frames, names):
            if f.index.has_duplicates:
                raise ValueError(f"{n}: duplicate dates in the index")
        common = frames[0].index
        for f in frames[1:]:
            common = common.intersection(f.index)
        common = common.sort_values()
        if len(common) == 0:
            raise ValueError("no common dates between the inputs")
        return tuple(f.loc[common] for f in frames)
    lengths = {len(f) for f in frames}
    if len(lengths) != 1:
        raise ValueError(
            "inputs without a DatetimeIndex are paired by position and must have "
            f"equal length; got lengths {[len(f) for f in frames]}")
    idx = frames[0].index
    return tuple(f.set_axis(idx, axis=0) for f in frames)
