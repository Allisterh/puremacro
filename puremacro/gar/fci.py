"""Financial Conditions Index — NFCI-style PCA aggregation.

Constructs a single-factor index from a wide panel of financial
indicators (rates, spreads, prices, vol, credit aggregates). The
estimator is the first principal component of the standardised panel,
sign-normalised so that **higher values mean tighter financial
conditions**: the convention used by the Chicago Fed NFCI, the
Bloomberg US Financial Conditions Index, and most academic GaR work.

Two routines:

  ``fci`` — fit on a complete-cases prefix; project the full panel.
  ``fci_rolling`` — rolling-window FCI, useful when loadings drift.

References
----------
Brave, S. and Butters, R.A. (2011). Monitoring financial stability:
    a financial conditions index approach. ChiFed Economic Perspectives.
Hatzius, J. et al. (2010). Financial conditions indexes: a fresh look
    after the financial crisis. NBER WP 16150.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def _standardise(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Xs = (X - mu) / sd
    return Xs, mu, sd


def _tightening_idx(columns, tightening_columns) -> list[int]:
    """Positions of ``tightening_columns`` in ``columns``.

    Names that match nothing used to be dropped silently, and when *no*
    name matched the sign normalisation was left unanchored altogether
    (the argmax default sat in an ``else`` arm that never ran), so the
    index carried the raw LAPACK SVD sign: on a 4-column panel that made
    the returned series depend on the *order* of ``df``'s columns, with
    12/24 to 18/24 orderings returning the inverted index. Refuse instead
    — a caller who names columns is asserting sign knowledge, and a typo
    must not degrade quietly to a weaker rule.
    """
    if not tightening_columns:
        return []
    cols = list(columns)
    idx = [cols.index(c) for c in tightening_columns if c in cols]
    if not idx:
        raise ValueError(
            f"none of tightening_columns={list(tightening_columns)!r} is a "
            f"column of df (columns are {cols!r}) — the sign normalisation "
            "would be left unanchored"
        )
    return idx


def fci(
    df: pd.DataFrame,
    *,
    tightening_columns: Sequence[str] | None = None,
    handle_nan: str = "drop",
) -> dict:
    """Single-factor Financial Conditions Index.

    Parameters
    ----------
    df : DataFrame
        Panel of financial indicators, time-indexed. Columns where a
        higher value indicates tighter conditions (spreads, vol) and
        looser conditions (asset prices) are mixed; the function infers
        the sign-normalisation from the loading on the first PC unless
        ``tightening_columns`` is supplied.
    tightening_columns : optional sequence of column names known to
        load positively under "tighter conditions". Used to fix the
        sign of the first PC if its loading-sum on these is negative.
        Raises if none of the names is a column of ``df``.
    handle_nan : ``"drop"`` (default) fits on the complete-cases rows,
        standardising on those same rows, and scores a row with missing
        cells by cross-sectional least squares on its observed subvector
        (NaN if nothing is observed); ``"zero"`` standardises on the full
        sample and replaces NaN with 0 (useful for ragged-edge panels but
        biases toward 0).

    Returns
    -------
    dict with
        ``index``    (T,)  — the FCI series (mean-zero, unit-variance),
        ``loadings`` (n,)  — first-PC loadings on each input,
        ``var_explained`` — share of standardised-data variance captured.
    """
    X = df.values.astype(float)
    idx = _tightening_idx(df.columns, tightening_columns)

    if handle_nan == "drop":
        complete = ~np.isnan(X).any(axis=1)
        if complete.sum() < 5:
            raise ValueError(
                f"too few complete-cases rows ({int(complete.sum())}) "
                "for FCI fitting — drop sparse columns or use "
                'handle_nan="zero"'
            )
        # Standardise on the rows actually fitted. Standardising with
        # np.nanstd over the full sample instead scales each column by an
        # sd computed on a *different* row set from the complete-cases
        # rows the SVD sees, so the fitted matrix is not a correlation
        # matrix at all: on a panel with a common volatility factor where
        # one column is missing in the volatile regime, the columns fed to
        # the SVD had sds (0.378, 0.378, 0.378, 1.0) and the loading vector
        # converged 52.7° away from the population correlation-PC1
        # (var_explained 0.737 vs 0.571), inverting the loading order.
        # Vanishes on a complete panel and asymptotically under MCAR.
        mu = X[complete].mean(axis=0)
        sd = X[complete].std(axis=0, ddof=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        Xs_raw = (X - mu) / sd
        Xs_fit = Xs_raw[complete]
    elif handle_nan == "zero":
        Xs_raw, _, _ = _standardise(X)
        Xs_fit = np.where(np.isnan(Xs_raw), 0.0, Xs_raw)
    else:
        raise ValueError(f"unknown handle_nan {handle_nan!r}")

    # First PC via SVD on the standardised, mean-zero matrix.
    Xs_fit_centred = Xs_fit - Xs_fit.mean(axis=0)
    U, s, Vt = np.linalg.svd(Xs_fit_centred, full_matrices=False)
    loadings = Vt[0]

    # Sign-normalise: tightening loadings should be positive.
    if idx:
        if loadings[idx].sum() < 0:
            loadings = -loadings
    # Default convention: the largest absolute loading carries the
    # sign of "tightening" — flip if negative.
    elif loadings[np.argmax(np.abs(loadings))] < 0:
        loadings = -loadings

    # Project the *full* panel (not just complete cases) onto the loading.
    obs = ~np.isnan(Xs_raw)
    index = np.where(obs, Xs_raw, 0.0) @ loadings
    if handle_nan == "drop":
        # Score a partially observed row by cross-sectional least squares
        # on what it does report: f_t = v_obs'x_obs / ||v_obs||². The bare
        # zero-filled dot product (which is what this branch used to do —
        # both arms of the old if/else were the same zero-fill line, so
        # "drop" silently behaved like "zero" here) drops the denominator
        # and attenuates every ragged row by exactly ||v_obs||², = (n-1)/n
        # when one of n equally-loaded columns is missing: 25% at n=4, 5%
        # at n=20, and the line below then rescaled the untouched history
        # by 1/sd on top. Vanishes on a complete panel, where ||v_obs||²=1.
        w = obs @ (loadings ** 2)
        ok = w > 1e-12
        index = np.where(ok, index / np.where(ok, w, 1.0), np.nan)

    # Re-standardise the index for interpretability.
    finite = np.isfinite(index)
    index = ((index - index[finite].mean())
             / max(index[finite].std(ddof=0), 1e-12))

    var_explained = float((s[0] ** 2) / (s ** 2).sum())
    return {
        "index":         pd.Series(index, index=df.index, name="FCI"),
        "loadings":      pd.Series(loadings, index=df.columns, name="loading"),
        "var_explained": var_explained,
    }


def fci_rolling(
    df: pd.DataFrame,
    window: int,
    *,
    tightening_columns: Sequence[str] | None = None,
) -> pd.Series:
    """Rolling-window Financial Conditions Index.

    Refits the first PC inside a moving window and reads off the FCI
    value at the window's right edge. Useful when the loading structure
    drifts (regime breaks, addition of new indicators); the trade-off
    is increased noise from each short-window PCA.

    Parameters
    ----------
    df : DataFrame
        Panel of financial indicators, time-indexed.
    window : int
        Rolling-window length in periods (must be ``>= 30``).
    tightening_columns : sequence of str, optional
        Columns known to load positively under "tighter conditions";
        forwarded to :func:`fci` for sign normalisation. Without it the
        orientation is pinned to the full-sample first PC (so the series
        agrees with ``fci(df)``) and then carried window to window.

    Returns
    -------
    pandas.Series
        FCI value at each window's right edge, NaN before the first
        full window.
    """
    n = len(df)
    if window < 30:
        raise ValueError("window must be ≥ 30 for stable PCA")
    idx = _tightening_idx(df.columns, tightening_columns)
    # Pin the orientation once. `fci`'s default argmax anchor is a
    # per-fit convention, and re-drawing it in every window let a single
    # returned series mix two opposite orientations: on a constant-loading
    # panel holding one loose-when-high column it flipped 13–68 times and
    # correlated with the true right-edge stress at |ρ| = 0.03–0.55
    # instead of 0.92. `ref` chains to the previous window (adjacent
    # windows overlap in window-1 rows, so their PCs are near-collinear
    # even when the loadings genuinely drift) but starts at the
    # full-sample PC, so the series keeps `fci`'s own orientation.
    # Unnecessary when `tightening_columns` anchors every window alike.
    ref = None
    if not idx:
        ref = fci(df, handle_nan="zero")["loadings"].values
    out = pd.Series(np.nan, index=df.index, name="FCI_rolling")
    for end in range(window, n + 1):
        sub = df.iloc[end - window: end]
        try:
            res = fci(sub, tightening_columns=tightening_columns,
                      handle_nan="zero")
        except (ValueError, ArithmeticError, np.linalg.LinAlgError, Exception):
            continue
        val = float(res["index"].iloc[-1])
        if ref is not None:
            v = res["loadings"].values
            if v @ ref < 0:
                v, val = -v, -val
            ref = v
        out.iloc[end - 1] = val
    return out


__all__ = ["fci", "fci_rolling"]
