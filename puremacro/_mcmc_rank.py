"""Rank-normalized multichain diagnostics (Vehtari et al., 2021).

The normal scores use pooled average ranks and Blom's (r-3/8)/(S+1/4).
ESS combines within-chain autocovariances and between-chain variation before
Geyer's positive/monotone truncation. The truncation convention extends the
existing puremacro single-chain estimator. Independent ArviZ comparisons live
in the test fixture; ArviZ is not a runtime dependency.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata


def _chains(values):
    x = np.asarray(values, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError("chains must have shape (chains, draws) for one parameter")
    return x


def _split(x):
    n = x.shape[1] // 2
    return np.concatenate((x[:, :n], x[:, -n:]), axis=0)


def _rank_scores(x):
    ranks = rankdata(x, method="average").reshape(x.shape)
    return ndtri((ranks - 0.375) / (x.size + 0.25))


def _center_scale(x):
    """Remove a common offset before scaling, with an overflow-safe fallback."""
    with np.errstate(over="ignore", invalid="ignore"):
        centered = x - x.flat[0]
    if np.all(np.isfinite(centered)):
        scale = float(np.max(np.abs(centered)))
        return (centered / scale if scale else centered), scale
    scale = float(np.max(np.abs(x)))
    # Opposite finite extremes may have an unrepresentable difference. Scaling
    # first is necessary only here; ordinary large offsets are removed exactly.
    normalized = x / scale
    return normalized - normalized.flat[0], scale


def _variance_ratio(x):
    count = x.shape[1]
    if np.all(x == x[:, :1]):
        return float("nan") if np.all(x == x.flat[0]) else float("inf")
    within = float(np.var(x, axis=1, ddof=1).mean())
    between = float(np.var(x.mean(axis=1), ddof=1))
    if within == 0:
        return float("inf") if between > 0 else float("nan")
    return float(np.sqrt((count - 1)/count + between/within))


def rank_normalized_rhat(chains: np.ndarray) -> float:
    """Maximum of rank-normalized and folded rank-normalized split R-hat.

    Input is (independent chains, post-warmup draws), for one parameter. At
    least two chains and four draws per chain are required. Odd lengths drop
    the central draw. Ties receive average ranks. Nonfinite/short/identically
    constant input returns NaN, never a reassuring 1. Different constant chains
    return infinity. A finite value is diagnostic evidence, not proof of mixing.

    Vehtari et al. (2021), Bayesian Analysis 16(2), 667–718,
    https://doi.org/10.1214/20-BA1221.
    """
    x = _chains(chains)
    if x.shape[0] < 2 or x.shape[1] < 4 or not np.all(np.isfinite(x)):
        return float("nan")
    halves = _split(x)
    bulk = _variance_ratio(_rank_scores(halves))
    folded = _variance_ratio(_rank_scores(np.abs(halves - np.median(halves))))
    # Folded ranks can be constant when chains occupy different fixed points.
    # Preserve the infinite between-chain warning from the original ranks.
    if np.isnan(folded):
        return bulk
    if np.isnan(bulk):
        return folded
    return max(bulk, folded)


def _pooled_ess(x):
    from .mcmc import _autocovariance

    n = x.shape[1]
    total = x.size
    if np.all(x == x.flat[0]):
        return float("nan")
    # ESS is invariant to a common nonzero scale. Standardize magnitudes before
    # squaring to avoid overflow/underflow for parameters in extreme units.
    x, _ = _center_scale(x)
    covariances = np.stack([_autocovariance(row) for row in x])
    acov = covariances.mean(axis=0)
    within = float(acov[0] * n/(n - 1))
    pooled = float(acov[0])
    if x.shape[0] > 1:
        pooled += float(np.var(x.mean(axis=1), ddof=1))
    if pooled <= 0 or not np.isfinite(pooled):
        return float("nan")
    correlations = 1 - (within - acov)/pooled
    correlations[0] = 1
    # Same finite-sample Geyer convention as effective_sample_size: retain
    # positive adjacent pairs, make their sums monotone, and average the odd
    # truncation with its following positive even lag.
    kept = np.zeros(n)
    kept[:2] = correlations[:2]
    even, odd = 1.0, float(correlations[1])
    t = 1
    while t < n - 3 and even + odd > 0:
        even, odd = float(correlations[t + 1]), float(correlations[t + 2])
        if even + odd >= 0:
            kept[t + 1:t + 3] = (even, odd)
        t += 2
    last = t - 2
    if even > 0:
        kept[last + 1] = even
    for t in range(1, last - 1, 2):
        previous = kept[t - 1] + kept[t]
        if kept[t + 1] + kept[t + 2] > previous:
            kept[t + 1:t + 3] = previous/2
    tau = -1 + 2*kept[:last + 1].sum() + kept[last + 1:last + 2].sum()
    return float(total/max(float(tau), 1/np.log10(total)))


def multichain_ess(chains: np.ndarray, *, method: str = "bulk",
                   tail_probability: float = 0.05) -> float:
    """Split multichain effective sample size for bulk, tails, or a mean.

    ``bulk`` uses pooled rank-normalized draws. ``tail`` is the smaller ESS of
    the indicators below the pooled q and 1-q quantiles (q=0.05 by default).
    ``mean`` uses raw values. Between-chain variation penalizes chains stuck
    in different regions, unlike summing their separate ESS values.

    One chain is accepted but cannot establish agreement between independent
    chains. At least four draws are required; odd lengths omit the middle draw.
    Nonfinite, insufficient, or completely constant input yields NaN. Constant
    tail indicators are uninformative and also yield NaN (an intentional
    difference from software that reports their nominal draw count). Negative
    autocorrelation can produce ESS above the retained count, capped at
    S*log10(S), where S is the number of draws after splitting.
    """
    if method not in {"bulk", "tail", "mean"}:
        raise ValueError("method must be 'bulk', 'tail' or 'mean'")
    if not np.isfinite(tail_probability) or not 0 < tail_probability < .5:
        raise ValueError("tail_probability must lie strictly between zero and 0.5")
    x = _chains(chains)
    if x.shape[0] < 1 or x.shape[1] < 4 or not np.all(np.isfinite(x)):
        return float("nan")
    if method == "bulk":
        return _pooled_ess(_rank_scores(_split(x)))
    if method == "mean":
        return _pooled_ess(_split(x))
    lo, hi = np.quantile(x, [tail_probability, 1-tail_probability])
    lower = _pooled_ess(_split((x <= lo).astype(float)))
    upper = _pooled_ess(_split((x <= hi).astype(float)))
    return float(np.minimum(lower, upper))


def mcse_mean(chains: np.ndarray) -> float:
    """Monte Carlo SE of the pooled mean using raw split multichain ESS.

    This is simulation error, not posterior uncertainty or a confidence bound
    on an economic parameter. Finite posterior variance and a chain CLT are
    needed; rank ESS cannot justify a mean standard error for heavy tails.
    """
    x = _chains(chains)
    ess = multichain_ess(x, method="mean")
    if not np.isfinite(ess) or ess <= 0:
        return float("nan")
    normalized, scale = _center_scale(x)
    return float((np.std(normalized, ddof=1)/np.sqrt(ess))*scale)
