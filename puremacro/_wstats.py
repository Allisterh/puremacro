"""Weighted statistics shared by the survey estimators and the dp solutions.

Each function takes values ``x`` (n,) and a weight matrix ``W`` (n, K), one
column per weight vector (a main weight and its replicates, or a single column
of probability mass), and returns one result per column. Every unit is a mass
point. Survey microdata (:mod:`puremacro.fetch.micro`) and model
distributions (:class:`puremacro.dp.DPSolution`) use these same functions, so
the two can never disagree on what a Gini or a top share is.
"""
from __future__ import annotations

import numpy as np

def w_quantile(x: np.ndarray, W: np.ndarray, q: float) -> np.ndarray:
    """Inverse of the weighted empirical CDF: the smallest x whose
    cumulative weight share reaches q (each weight column separately)."""
    order = np.argsort(x, kind="stable")
    xs = x[order]
    cw = np.cumsum(W[order], axis=0)
    tot = cw[-1]
    out = np.full(W.shape[1], np.nan)
    ok = tot > 0
    if not ok.any():
        return out
    target = q * tot[ok]
    idx = (cw[:, ok] >= target - 1e-12 * np.abs(target)).argmax(axis=0)
    out[ok] = xs[idx]
    return out


def lorenz(x: np.ndarray, W: np.ndarray):
    """Sorted values, cumulative weights and cumulative weighted values."""
    order = np.argsort(x, kind="stable")
    xs, Ws = x[order], W[order]
    return xs, Ws, np.cumsum(Ws, axis=0), np.cumsum(Ws * xs[:, None], axis=0)


def w_gini(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Gini coefficient, 1 − Σ_i p_i (L_i + L_{i−1}), from the Lorenz curve of
    the weighted sample (each unit a mass point). Equals the mean absolute
    difference Σ_ij w_i w_j |x_i − x_j| / (2 N Σ w x). Negative values (net
    worth) are allowed and can push it above 1; NaN when the total is <= 0."""
    _, Ws, cw, S = lorenz(x, W)
    N, tot = cw[-1], S[-1]
    S_prev = np.vstack([np.zeros((1, W.shape[1])), S[:-1]])
    with np.errstate(invalid="ignore", divide="ignore"):
        g = 1.0 - (Ws * (S + S_prev)).sum(axis=0) / (N * tot)
    return np.where((N > 0) & (tot > 0), g, np.nan)


def w_top_share(x: np.ndarray, W: np.ndarray, p: float) -> np.ndarray:
    """Share of the total held by the top ``p`` of units by weight,
    1 − L(1 − p), splitting the unit that straddles the cut-off."""
    xs, _, cw, S = lorenz(x, W)
    N, tot = cw[-1], S[-1]
    cut = (1.0 - p) * N
    i = (cw >= cut - 1e-12 * np.abs(cut)).argmax(axis=0)
    cols = np.arange(W.shape[1])
    prev = i > 0
    cw_prev = np.where(prev, cw[np.maximum(i - 1, 0), cols], 0.0)
    S_prev = np.where(prev, S[np.maximum(i - 1, 0), cols], 0.0)
    below = S_prev + (cut - cw_prev) * xs[i]
    with np.errstate(invalid="ignore", divide="ignore"):
        share = 1.0 - below / tot
    return np.where((N > 0) & (tot > 0), share, np.nan)
