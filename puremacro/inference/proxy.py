"""Inference for human shocks in smoothed, irregularly sampled natural archives.

Annual "proxy" products (ice-core gas splines, tree-ring reconstructions,
filtered temperature ensembles) are interpolants through a limited number of
physical measurements. The smoothing kernel sets the correlation structure of
every regression that uses them, and the effective number of independent
observations is often a small fraction of the nominal length. This module
provides the accounting and the tests that follow from that fact:

effective_sample_size      how many independent observations an annual series
                           carries (AR(1) and integrated-autocorrelation-time
                           versions, for levels and first differences)
smoother_df                effective degrees of freedom tr(S) of the cubic
                           smoothing spline that produced an annual product
rotation_reference         circular-rotation reference distribution for any
                           window statistic
feasible_onset_placebo     the statistic recomputed at every onset away from
                           catalogued events
ou_firn_covariance         closed-form covariance of an Ornstein-Uhlenbeck
                           atmosphere seen through a Gaussian gas-age kernel,
                           at irregular sample ages (and across two records)
OUFirnGP                   restricted marginal-likelihood fit of a two-timescale
                           OU + firn + measurement-error null on the discrete
                           samples (the stochastic null of Mei et al. 2026)
window_template            the atmospheric response to a unit emission pulse
                           confined to a window, integrated by the lifetime and
                           smoothed by the gas-age kernel
matched_filter             GLS / matched-filter estimate of the window
                           amplitude on the discrete samples, with a
                           feasible-onset reference, a parametric bootstrap
                           from the fitted null and the minimum detectable
                           effect
stacked_matched_filter     the same across several records of one atmosphere
                           (shared null and amplitude, record-specific trends
                           and kernels)
calibrate_matched_filter   size and power of the matched filter on a given
                           sampling design, with or without re-fitting the null

The matched filter is the Neyman-Pearson detector for a signal of known shape
in Gaussian noise. On these records it replaces volatility or event-study
regressions on a spline with an estimator that uses the measurements and their
reported errors, carries a fitted stochastic model of the unforced
atmosphere, and returns a detection threshold.

All results are frozen dataclasses with ``summary()``, ``to_frame()``,
``to_markdown()``, ``to_latex()``, ``to_typst()`` and ``plot()``.

References
----------
Mei, W., Hakim, G. J., Proistosescu, C., Bauska, T. K., Buizert, C. and
    Turner, A. J. (2026). Multidecadal preindustrial methane variability can
    be explained by noise in the source-sink imbalance. PNAS 123(25),
    e2601235123.
Rubino, M. et al. (2019). Revised records of atmospheric trace gases CO2,
    CH4, N2O, and d13C-CO2 over the last 2000 years from Law Dome,
    Antarctica. Earth System Science Data 11, 473-492.
Sokal, A. (1997). Monte Carlo methods in statistical mechanics: foundations
    and new algorithms. In Functional Integration, Springer, 131-192.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, special, stats
from scipy.ndimage import gaussian_filter1d

from ._results import _fmt, _render

__all__ = [
    "ESSResult", "effective_sample_size", "smoother_df",
    "ReferenceResult", "rotation_reference", "feasible_onset_placebo",
    "ou_firn_covariance", "OUFirnGP", "GPFit",
    "window_template", "MatchedFilterResult", "matched_filter",
    "ProxyRecord", "StackedMatchedFilterResult", "stacked_matched_filter",
    "CalibrationResult", "calibrate_matched_filter",
]

_Z_MDE = float(stats.norm.ppf(0.975) + stats.norm.ppf(0.80))


def _row_frame(d: dict, digits: int) -> pd.DataFrame:
    return pd.DataFrame([{k: _fmt(v, digits) for k, v in d.items()}])


class _Presentable:
    """Shared rendering for single-row results (``to_frame`` must exist)."""

    def to_markdown(self, digits: int = 4) -> str:
        """Render as GitHub-flavored Markdown."""
        return _render(self.to_frame(digits), "markdown")

    def to_latex(self, digits: int = 4) -> str:
        """Render as a LaTeX ``tabular``."""
        return _render(self.to_frame(digits), "latex")

    def to_typst(self, digits: int = 4) -> str:
        """Render as a Typst ``#table``."""
        return _render(self.to_frame(digits), "typst")


# ============================================================================ effective sample size
@dataclass(frozen=True)
class ESSResult(_Presentable):
    """How many independent observations an annual series carries.

    Attributes
    ----------
    label : str
    n : int
        Length of the series (finite values).
    rho1 : float
        Lag-1 autocorrelation.
    tau_int : float
        Integrated autocorrelation time ``1 + 2 sum_{k>=1} rho_k`` with Sokal's
        automatic window.
    n_eff_ar1 : float
        ``n (1 - rho1) / (1 + rho1)``, the AR(1) approximation (capped at n).
    n_eff_int : float
        ``n / tau_int`` (capped at n).
    rho1_diff, tau_int_diff, n_eff_ar1_diff, n_eff_int_diff : float
        The same on first differences, the growth-rate series that event
        studies and volatility models use.
    acf : numpy.ndarray
        Autocorrelation function of the levels up to lag ``min(n // 2, 200)``.
    """

    label: str
    n: int
    rho1: float
    tau_int: float
    n_eff_ar1: float
    n_eff_int: float
    rho1_diff: float
    tau_int_diff: float
    n_eff_ar1_diff: float
    n_eff_int_diff: float
    acf: np.ndarray = field(repr=False, compare=False, default_factory=lambda: np.empty(0))

    def to_frame(self, digits: int = 4) -> pd.DataFrame:
        """One row with every scalar field."""
        return _row_frame({k: v for k, v in self.__dict__.items() if k != "acf"}, digits)

    def summary(self) -> str:
        return (f"{self.label}: n={self.n}, rho1={self.rho1:.4f}, n_eff(AR1)={self.n_eff_ar1:.1f}, "
                f"n_eff(int)={self.n_eff_int:.1f}; diff: rho1={self.rho1_diff:.4f}, "
                f"n_eff(AR1)={self.n_eff_ar1_diff:.1f}, n_eff(int)={self.n_eff_int_diff:.1f}")

    def plot(self, *, title: str = "", ax=None):
        """Autocorrelation function of the levels. Returns the Figure."""
        from ..plot import _new_ax

        fig, ax = _new_ax(ax)
        lags = np.arange(len(self.acf))
        ax.plot(lags, self.acf, color="0.1", lw=1.0)
        ax.axhline(0.0, color="0.6", lw=0.6)
        ax.set_xlabel("lag")
        ax.set_ylabel("autocorrelation")
        ax.set_title(title or f"{self.label}: n_eff(int) = {self.n_eff_int:.1f} of {self.n}")
        return fig


def _acf(x: np.ndarray, maxlag: int) -> np.ndarray:
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    denom = np.dot(x, x)
    return np.array([np.dot(x[:n - k], x[k:]) / denom for k in range(maxlag + 1)])


def _tau_int(x: np.ndarray, c: float = 5.0) -> float:
    """Integrated autocorrelation time with Sokal's automatic window (stop at M >= c*tau)."""
    n = len(x)
    r = _acf(x, min(n // 2, 2000))
    tau = 1.0
    for M in range(1, len(r)):
        tau = 1.0 + 2.0 * np.sum(r[1:M + 1])
        if M >= c * tau:
            break
    return float(max(tau, 1e-9))


def effective_sample_size(x: Sequence[float], label: str = "") -> ESSResult:
    """Effective number of independent observations in a series and in its first differences.

    Parameters
    ----------
    x : array-like
        The series (non-finite values are dropped).
    label : str
        Name carried into the result.

    Returns
    -------
    ESSResult
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 4:
        raise ValueError("effective_sample_size needs at least 4 finite values")
    rho1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    ti = _tau_int(x)
    d = np.diff(x)
    rho1d = float(np.corrcoef(d[:-1], d[1:])[0, 1])
    tid = _tau_int(d)
    cap = lambda v, m: float(min(v, m))  # noqa: E731  over-differenced series would exceed n
    return ESSResult(label, n, rho1, ti, cap(n * (1 - rho1) / (1 + rho1), n), cap(n / ti, n),
                     rho1d, tid, cap((n - 1) * (1 - rho1d) / (1 + rho1d), n - 1), cap((n - 1) / tid, n - 1),
                     _acf(x, min(n // 2, 200)))


def smoother_df(x: np.ndarray, lam: float, eval_years: np.ndarray | None = None,
                weights: np.ndarray | None = None) -> float:
    """Effective degrees of freedom ``tr(S)`` of the cubic smoothing spline through samples at ``x``.

    The hat matrix is built column by column with :func:`scipy.interpolate.make_smoothing_spline`
    (the Reinsch spline that ice-core providers use) and evaluated at the sample ages.

    Parameters
    ----------
    x : array
        Strictly increasing sample ages.
    lam : float
        Smoothing parameter as in ``make_smoothing_spline``.
    eval_years : array, optional
        Unused; kept for signature compatibility with the paper's replication code.
    weights : array, optional
        Sample weights passed to the spline.
    """
    from scipy.interpolate import make_smoothing_spline

    x = np.asarray(x, float)
    n = len(x)
    S = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        S[:, j] = make_smoothing_spline(x, e, w=weights, lam=lam)(x)
    return float(np.trace(S))


# ============================================================================ reference distributions
@dataclass(frozen=True)
class ReferenceResult(_Presentable):
    """A window statistic against an empirical reference distribution.

    Attributes
    ----------
    label : str
    observed : float
    n_reference : int
    p_one_sided : float
        In the direction of the observed sign, ``(1 + #{ref beyond obs}) / (m + 1)``.
    p_two_sided : float
        ``(1 + #{|ref| >= |obs|}) / (m + 1)``.
    q05, q95 : float
        Quantiles of the reference.
    reference : numpy.ndarray
    """

    label: str
    observed: float
    n_reference: int
    p_one_sided: float
    p_two_sided: float
    q05: float
    q95: float
    reference: np.ndarray = field(repr=False, compare=False)

    def to_frame(self, digits: int = 4) -> pd.DataFrame:
        return _row_frame({k: v for k, v in self.__dict__.items() if k != "reference"}, digits)

    def summary(self) -> str:
        return (f"{self.label}: observed {self.observed:+.4g}; reference n={self.n_reference}, "
                f"5-95% [{self.q05:+.4g}, {self.q95:+.4g}]; p one-sided {self.p_one_sided:.3f}, "
                f"two-sided {self.p_two_sided:.3f}")

    def plot(self, *, title: str = "", ax=None):
        """Histogram of the reference with the observed statistic marked. Returns the Figure."""
        from ..plot import _new_ax

        fig, ax = _new_ax(ax)
        ref = self.reference[np.isfinite(self.reference)]
        ax.hist(ref, bins=min(50, max(10, len(ref) // 10)), color="0.75", edgecolor="0.4", lw=0.4)
        ax.axvline(self.observed, color="0.1", lw=1.4, label=f"observed (p two-sided {self.p_two_sided:.3f})")
        ax.set_xlabel("statistic")
        ax.set_ylabel("count")
        ax.set_title(title or self.label)
        ax.legend(frameon=False, fontsize=8)
        return fig


def _p_from_reference(obs: float, ref: np.ndarray) -> tuple[float, float]:
    ref = np.asarray(ref, float)
    ref = ref[np.isfinite(ref)]
    m = len(ref)
    if obs < 0:
        p1 = (1 + np.sum(ref <= obs)) / (m + 1)
    else:
        p1 = (1 + np.sum(ref >= obs)) / (m + 1)
    p2 = (1 + np.sum(np.abs(ref) >= abs(obs))) / (m + 1)
    return float(p1), float(p2)


def _reference_result(label: str, obs: float, ref: np.ndarray) -> ReferenceResult:
    ref = np.asarray(ref, float)
    if ref.size == 0:
        raise ValueError(f"{label}: the reference distribution is empty")
    p1, p2 = _p_from_reference(obs, ref)
    return ReferenceResult(label, float(obs), len(ref), p1, p2, float(np.quantile(ref, .05)),
                           float(np.quantile(ref, .95)), ref)


def rotation_reference(stat_fn: Callable[[np.ndarray], float], y: np.ndarray,
                       label: str = "rotation") -> ReferenceResult:
    """Circular-rotation reference for a window statistic.

    The outcome is rolled by every offset ``1 .. n-1`` with the regressors, the window and the
    design held fixed, so the reference keeps the outcome's own autocorrelation. The observed
    value is ``stat_fn(y)``.
    """
    y = np.asarray(y, float)
    obs = float(stat_fn(y))
    ref = np.array([stat_fn(np.roll(y, k)) for k in range(1, len(y))])
    return _reference_result(label, obs, ref)


def feasible_onset_placebo(stat_at_onset: Callable[[float], float], onsets: Sequence[float],
                           actual_onset: float, catalog_onsets: Sequence[float] = (),
                           exclusion: float = 50, label: str = "feasible-onset placebo") -> ReferenceResult:
    """The statistic recomputed at every feasible onset.

    An onset is feasible when it lies more than ``exclusion`` (in the units of ``onsets``) from
    ``actual_onset`` and from every catalogued event. The observed value is
    ``stat_at_onset(actual_onset)``.
    """
    obs = float(stat_at_onset(actual_onset))
    cat = np.asarray(list(catalog_onsets) + [actual_onset], float)
    ref = np.array([stat_at_onset(o) for o in onsets if np.all(np.abs(cat - o) > exclusion)])
    return _reference_result(label, obs, ref)


# ============================================================================ OU + firn covariance
def ou_firn_covariance(t: np.ndarray, tau: float, firn_sigma: float, t2: np.ndarray | None = None,
                       firn_sigma2: float | None = None) -> np.ndarray:
    """Unit-variance covariance of an OU process seen through Gaussian gas-age kernels.

    For ages ``t`` (and ``t2``), correlation time ``tau`` and kernel widths ``firn_sigma`` (and
    ``firn_sigma2`` for the second record), with ``D = t_i - t2_j`` and ``s^2 = sigma1^2 + sigma2^2``::

        C(D) = E[exp(-|D + d| / tau)],  d ~ N(0, s^2)
             = 1/2 exp(s^2 / 2 tau^2) [ exp(-D/tau) erfc((s^2/tau - D) / (s sqrt2))
                                       + exp( D/tau) erfc((s^2/tau + D) / (s sqrt2)) ]

    which reduces to ``exp(-|D| / tau)`` as the kernels vanish. Evaluated in log space.
    """
    t = np.asarray(t, float)
    t2 = t if t2 is None else np.asarray(t2, float)
    D = t[:, None] - t2[None, :]
    sig2 = firn_sigma if firn_sigma2 is None else firn_sigma2
    s2 = float(firn_sigma) ** 2 + float(sig2) ** 2
    if s2 <= 0:
        return np.exp(-np.abs(D) / tau)
    s = np.sqrt(s2)
    a = s2 / tau

    def log_half_erfc(x):
        return special.log_ndtr(-x * np.sqrt(2.0))          # log(erfc(x) / 2)

    term1 = -D / tau + log_half_erfc((a - D) / (s * np.sqrt(2.0)))
    term2 = D / tau + log_half_erfc((a + D) / (s * np.sqrt(2.0)))
    m = np.maximum(term1, term2)
    logC = s2 / (2.0 * tau ** 2) + m + np.log(np.exp(term1 - m) + np.exp(term2 - m))
    return np.exp(logC)


@dataclass(frozen=True)
class GPFit(_Presentable):
    """Fitted two-timescale OU + firn + measurement-error null.

    Attributes
    ----------
    var_fast, tau_fast : float
        Variance and correlation time (years) of the fast component.
    var_slow, tau_slow : float
        The same for the slow component.
    firn_sigma : float
        Gas-age kernel width (fixed, not estimated).
    trend_df : int
        Number of polynomial trend terms profiled out.
    loglik : float
        Maximised restricted log-likelihood.
    n : int
        Samples used in the hyperparameter fit (after masking).
    Sigma : numpy.ndarray
        Fitted covariance at ALL sample ages, measurement error included.
    t : numpy.ndarray
        All sample ages.
    """

    var_fast: float
    tau_fast: float
    var_slow: float
    tau_slow: float
    firn_sigma: float
    trend_df: int
    loglik: float
    n: int
    Sigma: np.ndarray = field(repr=False, compare=False)
    t: np.ndarray = field(repr=False, compare=False)

    def to_frame(self, digits: int = 4) -> pd.DataFrame:
        return _row_frame({k: v for k, v in self.__dict__.items() if k not in ("Sigma", "t")}, digits)

    def summary(self) -> str:
        return (f"OU+firn GP: fast sd {np.sqrt(self.var_fast):.3g} (tau {self.tau_fast:.1f} y), "
                f"slow sd {np.sqrt(self.var_slow):.3g} (tau {self.tau_slow:.0f} y), firn sigma {self.firn_sigma} y, "
                f"trend df {self.trend_df}, n {self.n}, loglik {self.loglik:.2f}")

    def plot(self, *, max_lag: float = 300.0, title: str = "", ax=None):
        """Fitted atmospheric covariance (without measurement error) against lag. Returns the Figure."""
        from ..plot import _new_ax

        fig, ax = _new_ax(ax)
        lag = np.linspace(0.0, max_lag, 301)
        c = (self.var_fast * ou_firn_covariance(lag, self.tau_fast, self.firn_sigma, np.zeros(1))[:, 0]
             + self.var_slow * ou_firn_covariance(lag, self.tau_slow, self.firn_sigma, np.zeros(1))[:, 0])
        ax.plot(lag, c, color="0.1", lw=1.0)
        ax.set_xlabel("lag (years)")
        ax.set_ylabel("covariance")
        ax.set_title(title or "OU + firn null")
        return fig


def _trend_basis(t: np.ndarray, df: int) -> np.ndarray:
    tc = (t - t.mean()) / (t.max() - t.min())
    return np.column_stack([tc ** k for k in range(df)])


def _profile_nll(theta, t, y, unc, X, firn_sigma, tau_bounds):
    lv1, lt1, lv2, lt2 = theta
    tau1 = np.exp(lt1)
    tau2 = np.exp(lt2)
    if not (tau_bounds[0] <= tau1 <= tau_bounds[1] and tau_bounds[2] <= tau2 <= tau_bounds[3]):
        return 1e30
    Sigma = (np.exp(lv1) * ou_firn_covariance(t, tau1, firn_sigma)
             + np.exp(lv2) * ou_firn_covariance(t, tau2, firn_sigma)
             + np.diag(unc ** 2))
    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        return 1e30
    Li = np.linalg.solve(L, np.eye(len(t)))
    Sinv = Li.T @ Li
    XtSX = X.T @ Sinv @ X
    beta = np.linalg.solve(XtSX, X.T @ Sinv @ y)
    r = y - X @ beta
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return 0.5 * (logdet + r @ Sinv @ r + np.linalg.slogdet(XtSX)[1])


class OUFirnGP:
    """Restricted maximum-likelihood fit of the stochastic null on discrete samples::

        y_i = trend(t_i) + s_fast(t_i) + s_slow(t_i) + e_i,

    ``s_*`` OU processes smoothed by a Gaussian gas-age kernel, ``e_i ~ N(0, unc_i^2)`` with the
    reported per-sample uncertainties.

    Parameters
    ----------
    firn_sigma : float
        Gas-age kernel width in years.
    trend_df : int
        Number of polynomial trend terms (1 = constant, 2 = linear, 4 = cubic).
    tau_bounds : tuple
        ``(fast lo, fast hi, slow lo, slow hi)`` in years.
    """

    def __init__(self, firn_sigma: float = 5.0, trend_df: int = 2,
                 tau_bounds: tuple = (1.0, 60.0, 60.0, 2000.0)):
        self.firn_sigma = firn_sigma
        self.trend_df = trend_df
        self.tau_bounds = tau_bounds

    def fit(self, t, y, unc, seed: int = 0, mask_window: tuple | None = None, n_starts: int = 6,
            fixed_tau: tuple | None = None) -> GPFit:
        """Fit the four hyperparameters.

        ``mask_window=(lo, hi)`` drops samples with ``lo <= t <= hi`` from the hyperparameter fit
        (leave the tested window out, so that a signal in it cannot shape its own null); the
        returned ``Sigma`` is evaluated at all ages. ``n_starts`` Nelder-Mead restarts (the first
        deterministic). ``fixed_tau=(tau_fast, tau_slow)`` fits the variances only.
        """
        t_all = np.asarray(t, float)
        y_all = np.asarray(y, float)
        unc_all = np.asarray(unc, float)
        keep = np.ones(len(t_all), bool)
        if mask_window is not None:
            keep = ~((t_all >= mask_window[0]) & (t_all <= mask_window[1]))
        t, y, unc = t_all[keep], y_all[keep], unc_all[keep]
        X = _trend_basis(t, self.trend_df)
        sd = np.std(y)
        best = None
        rng = np.random.default_rng(seed)
        tb = tuple(self.tau_bounds)
        if fixed_tau is not None:
            lt1, lt2 = np.log(fixed_tau[0]), np.log(fixed_tau[1])
            tbf = (fixed_tau[0] * 0.999, fixed_tau[0] * 1.001, fixed_tau[1] * 0.999, fixed_tau[1] * 1.001)
            f = lambda v: _profile_nll(np.array([v[0], lt1, v[1], lt2]), t, y, unc, X, self.firn_sigma, tbf)  # noqa: E731
            for s0 in [(np.log(0.25 * sd ** 2), np.log(0.5 * sd ** 2)), (np.log(0.05 * sd ** 2), np.log(sd ** 2))]:
                res = optimize.minimize(f, np.array(s0), method="Nelder-Mead",
                                        options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-6})
                if best is None or res.fun < best.fun:
                    best = res
            best.x = np.array([best.x[0], lt1, best.x[1], lt2])
        else:
            starts = [(np.log(0.25 * sd ** 2), np.log(10.0), np.log(0.5 * sd ** 2), np.log(300.0))]
            for _ in range(max(0, n_starts - 1)):
                starts.append((np.log(sd ** 2 * rng.uniform(.05, 1)), np.log(rng.uniform(2, 40)),
                               np.log(sd ** 2 * rng.uniform(.05, 1)), np.log(rng.uniform(80, 1500))))
            for s0 in starts:
                res = optimize.minimize(_profile_nll, np.array(s0), args=(t, y, unc, X, self.firn_sigma, tb),
                                        method="Nelder-Mead",
                                        options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-6})
                if best is None or res.fun < best.fun:
                    best = res
        lv1, lt1, lv2, lt2 = best.x
        Sigma = (np.exp(lv1) * ou_firn_covariance(t_all, np.exp(lt1), self.firn_sigma)
                 + np.exp(lv2) * ou_firn_covariance(t_all, np.exp(lt2), self.firn_sigma)
                 + np.diag(unc_all ** 2))
        return GPFit(float(np.exp(lv1)), float(np.exp(lt1)), float(np.exp(lv2)), float(np.exp(lt2)),
                     self.firn_sigma, self.trend_df, -float(best.fun), len(t), Sigma, t_all)


# ============================================================================ template + matched filter
def window_template(sample_ages: np.ndarray, start: float, end: float, lifetime: float,
                    firn_sigma: float, grid: tuple = (-300, 2000)) -> np.ndarray:
    """Atmospheric response to a unit emission perturbation on ``[start, end]``.

    ``dC/dt = -C / lifetime + 1{start <= t <= end}`` integrated annually on ``grid``, normalised to
    unit peak before the gas-age kernel, then smoothed by a Gaussian of width ``firn_sigma`` and
    interpolated at the sample ages. A fitted amplitude is therefore the peak atmospheric anomaly
    in the series' own units.
    """
    years = np.arange(grid[0], grid[1] + 1)
    u = ((years >= start) & (years <= end)).astype(float)
    phi = np.exp(-1.0 / lifetime)
    C = np.zeros(len(years))
    for i in range(1, len(years)):
        C[i] = phi * C[i - 1] + u[i]
    if C.max() <= 0:
        raise ValueError("window_template: the window lies outside the integration grid")
    C = C / C.max()
    if firn_sigma > 0:
        C = gaussian_filter1d(C, firn_sigma, mode="nearest")
    return np.interp(np.asarray(sample_ages, float), years, C)


@dataclass(frozen=True)
class MatchedFilterResult(_Presentable):
    """GLS / matched-filter estimate of a window amplitude on discrete samples.

    Attributes
    ----------
    amplitude, se, t : float
        GLS estimate of the peak atmospheric anomaly, its standard error and t.
    p_normal : float
        Two-sided p from N(0, 1).
    t_df : float
        Degrees of freedom of the optional Student-t reference (NaN when not requested).
    p_t : float
        Two-sided p from Student t with ``t_df`` (NaN when not requested).
    p_rotation_one, p_rotation_two : float
        From re-centring the template on every feasible onset.
    p_boot_one, p_boot_two : float
        From parametric draws of the fitted null.
    mde_80 : float
        Minimum detectable amplitude at 80% power, 5% two-sided (normal; Student t when ``t_df``).
    n_samples_in_window : int
    rotation_t, rotation_onsets : numpy.ndarray
        The feasible-onset reference and its onsets.
    """

    label: str
    start: int
    end: int
    amplitude: float
    se: float
    t: float
    p_normal: float
    p_rotation_one: float
    p_rotation_two: float
    n_rotation: int
    p_boot_one: float
    p_boot_two: float
    n_boot: int
    mde_80: float
    n_samples_in_window: int
    t_df: float = float("nan")
    p_t: float = float("nan")
    rotation_t: np.ndarray = field(repr=False, compare=False, default_factory=lambda: np.empty(0))
    rotation_onsets: np.ndarray = field(repr=False, compare=False, default_factory=lambda: np.empty(0))

    def to_frame(self, digits: int = 4) -> pd.DataFrame:
        return _row_frame({k: v for k, v in self.__dict__.items()
                           if k not in ("rotation_t", "rotation_onsets")}, digits)

    def summary(self) -> str:
        extra = f"; Student-t({self.t_df:.0f}) p {self.p_t:.3f}" if np.isfinite(self.t_df) else ""
        return (f"{self.label} [{self.start}-{self.end}]: A = {self.amplitude:+.3g} (SE {self.se:.3g}, t {self.t:+.2f}); "
                f"p_normal {self.p_normal:.3f}{extra}; rotation p {self.p_rotation_one:.3f}/{self.p_rotation_two:.3f} "
                f"(n {self.n_rotation}); bootstrap p {self.p_boot_one:.3f}/{self.p_boot_two:.3f}; "
                f"MDE80 {self.mde_80:.3g}; samples in window {self.n_samples_in_window}")

    def plot(self, *, title: str = "", ax=None):
        """The matched-filter t at every feasible onset, with the tested onset marked. Returns the Figure."""
        from ..plot import _new_ax

        fig, ax = _new_ax(ax)
        if len(self.rotation_onsets):
            ax.plot(self.rotation_onsets, self.rotation_t, color="0.55", lw=0.8, label="feasible onsets")
        ax.axhline(0.0, color="0.8", lw=0.6)
        ax.plot([self.start], [self.t], "o", color="0.1", label=f"{self.start}: t = {self.t:+.2f}")
        ax.set_xlabel("onset")
        ax.set_ylabel("matched-filter t")
        ax.set_title(title or self.label)
        ax.legend(frameon=False, fontsize=8)
        return fig


def _gls(y, X, Sinv):
    XtSX = X.T @ Sinv @ X
    beta = np.linalg.solve(XtSX, X.T @ Sinv @ y)
    V = np.linalg.inv(XtSX)
    return beta, V


def _chol_inv(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    L = np.linalg.cholesky(S)
    Li = np.linalg.solve(L, np.eye(S.shape[0]))
    return L, Li.T @ Li


def _mde(se: float, t_df: float | None) -> float:
    if t_df is None or not np.isfinite(t_df):
        return float(_Z_MDE * se)
    return float((stats.t.ppf(0.975, t_df) + stats.t.ppf(0.80, t_df)) * se)


def matched_filter(t, y, unc, gp: GPFit, start: int, end: int, lifetime: float, label: str = "",
                   trend_df: int | None = None, onset_step: int = 5, onset_range: tuple | None = None,
                   catalog_onsets: Sequence[int] = (), exclusion: int = 50, n_boot: int = 2000,
                   seed: int = 20260908, t_df: float | None = None) -> MatchedFilterResult:
    """GLS estimate of the amplitude of :func:`window_template` under the fitted null ``gp``.

    Parameters
    ----------
    t, y, unc : array
        Sample ages, values and reported 1-sigma uncertainties.
    gp : GPFit
        The fitted null; its ``Sigma`` must be evaluated at ``t``.
    start, end : int
        The tested window.
    lifetime : float
        Atmospheric lifetime (years) used to integrate the pulse.
    trend_df : int, optional
        Trend terms (defaults to ``gp.trend_df``).
    onset_step, onset_range, catalog_onsets, exclusion
        The feasible-onset reference: the template is re-centred every ``onset_step`` years over
        ``onset_range`` (default: 50 years inside the record) at onsets more than ``exclusion``
        years from ``start`` and from every catalogued onset.
    n_boot : int
        Parametric-bootstrap draws from the fitted null (trend + GP + measurement error).
    seed : int
    t_df : float, optional
        When given, also report a two-sided Student-t p with these degrees of freedom and use t
        quantiles for the MDE: a cheap guard against hyperparameter uncertainty that the GLS
        standard error does not carry.

    Returns
    -------
    MatchedFilterResult
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    unc = np.asarray(unc, float)
    if gp.Sigma.shape != (len(t), len(t)):
        raise ValueError("gp.Sigma must be evaluated at the sample ages t")
    df = gp.trend_df if trend_df is None else trend_df
    X0 = _trend_basis(t, df)
    L, Sinv = _chol_inv(gp.Sigma)
    W = end - start

    def fit_at(s: int):
        h = window_template(t, s, s + W, lifetime, gp.firn_sigma)
        X = np.column_stack([X0, h])
        beta, V = _gls(y, X, Sinv)
        return beta[-1], np.sqrt(V[-1, -1])

    A, se = fit_at(start)
    tstat = A / se
    p_norm = 2 * stats.norm.sf(abs(tstat))
    lo, hi = onset_range if onset_range else (int(t.min()) + 50, int(t.max()) - W - 50)
    cat = np.asarray(list(catalog_onsets) + [start], float)
    onsets, ref = [], []
    for s in range(lo, hi + 1, onset_step):
        if np.all(np.abs(cat - s) > exclusion):
            a, e = fit_at(s)
            onsets.append(s)
            ref.append(a / e)
    ref = np.asarray(ref)
    pr1, pr2 = _p_from_reference(tstat, ref) if len(ref) else (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    beta0, _ = _gls(y, X0, Sinv)
    mu = X0 @ beta0
    h = window_template(t, start, end, lifetime, gp.firn_sigma)
    X = np.column_stack([X0, h])
    XtSX_inv = np.linalg.inv(X.T @ Sinv @ X)
    proj = XtSX_inv @ X.T @ Sinv
    se_A = np.sqrt(XtSX_inv[-1, -1])
    Z = rng.standard_normal((n_boot, len(t)))
    Ystar = mu[None, :] + Z @ L.T
    tboot = (proj[-1] @ Ystar.T) / se_A
    pb1, pb2 = _p_from_reference(tstat, tboot)
    p_t = float(2 * stats.t.sf(abs(tstat), t_df)) if t_df is not None else float("nan")
    nin = int(np.sum((t >= start) & (t <= end)))
    return MatchedFilterResult(label, int(start), int(end), float(A), float(se), float(tstat), float(p_norm),
                               pr1, pr2, len(ref), pb1, pb2, int(n_boot), _mde(se, t_df), nin,
                               float(t_df) if t_df is not None else float("nan"), p_t,
                               ref, np.asarray(onsets, float))


# ============================================================================ several records
@dataclass(frozen=True)
class ProxyRecord:
    """One archive's discrete samples of a shared atmosphere.

    Attributes
    ----------
    name : str
    t, y, unc : numpy.ndarray
        Sample ages, values and reported 1-sigma uncertainties.
    firn_sigma : float
        The record's gas-age kernel width (years).
    trend_df : int
        Record-specific polynomial trend terms (inter-core offsets and gradients live here).
    """

    name: str
    t: np.ndarray
    y: np.ndarray
    unc: np.ndarray
    firn_sigma: float
    trend_df: int = 3

    def __post_init__(self):
        n = len(np.asarray(self.t))
        if len(np.asarray(self.y)) != n or len(np.asarray(self.unc)) != n:
            raise ValueError(f"ProxyRecord {self.name!r}: t, y and unc must have the same length")


def _trend_block(records: Sequence[ProxyRecord], keep_masks=None) -> np.ndarray:
    blocks = []
    for i, r in enumerate(records):
        t = r.t if keep_masks is None else r.t[keep_masks[i]]
        tc = (t - r.t.mean()) / (r.t.max() - r.t.min())
        blocks.append(np.column_stack([tc ** k for k in range(r.trend_df)]))
    n = sum(b.shape[0] for b in blocks)
    p = sum(b.shape[1] for b in blocks)
    X = np.zeros((n, p))
    i0 = j0 = 0
    for b in blocks:
        X[i0:i0 + b.shape[0], j0:j0 + b.shape[1]] = b
        i0 += b.shape[0]
        j0 += b.shape[1]
    return X


def _stacked_cov(records, keep_masks, v1, tau1, v2, tau2):
    ts = [r.t if keep_masks is None else r.t[keep_masks[i]] for i, r in enumerate(records)]
    us = [r.unc if keep_masks is None else r.unc[keep_masks[i]] for i, r in enumerate(records)]
    n = sum(len(t) for t in ts)
    S = np.zeros((n, n))
    idx = np.cumsum([0] + [len(t) for t in ts])
    for i, ri in enumerate(records):
        for j, rj in enumerate(records):
            if j < i:
                continue
            K = (v1 * ou_firn_covariance(ts[i], tau1, ri.firn_sigma, ts[j], rj.firn_sigma)
                 + v2 * ou_firn_covariance(ts[i], tau2, ri.firn_sigma, ts[j], rj.firn_sigma))
            S[idx[i]:idx[i + 1], idx[j]:idx[j + 1]] = K
            if j != i:
                S[idx[j]:idx[j + 1], idx[i]:idx[i + 1]] = K.T
    S += np.diag(np.concatenate(us) ** 2)
    return S


def _stacked_nll(theta, records, keep, y, X, tb):
    lv1, lt1, lv2, lt2 = theta
    tau1, tau2 = np.exp(lt1), np.exp(lt2)
    if not (tb[0] <= tau1 <= tb[1] and tb[2] <= tau2 <= tb[3]):
        return 1e30
    S = _stacked_cov(records, keep, np.exp(lv1), tau1, np.exp(lv2), tau2)
    try:
        L, Sinv = _chol_inv(S)
    except np.linalg.LinAlgError:
        return 1e30
    XtSX = X.T @ Sinv @ X
    beta = np.linalg.solve(XtSX, X.T @ Sinv @ y)
    r = y - X @ beta
    return 0.5 * (2 * np.sum(np.log(np.diag(L))) + r @ Sinv @ r + np.linalg.slogdet(XtSX)[1])


@dataclass(frozen=True)
class StackedMatchedFilterResult(_Presentable):
    """Matched filter on several records of one atmosphere.

    Attributes
    ----------
    records : tuple of str
    n_total : int
    n_in_window : tuple of int
    amplitude, se, t, p_normal, p_rotation_two, p_boot_two, mde_80 : float
        As in :class:`MatchedFilterResult`, for the shared amplitude.
    n_rotation : int
    var_fast, tau_fast, var_slow, tau_slow : float
        The shared null, fitted on the stacked vector with the window masked in every record.
    per_record : pandas.DataFrame
        Each record alone under the shared null (amplitude, se, t, MDE).
    """

    label: str
    records: tuple
    n_total: int
    n_in_window: tuple
    amplitude: float
    se: float
    t: float
    p_normal: float
    p_rotation_two: float
    n_rotation: int
    p_boot_two: float
    mde_80: float
    var_fast: float
    tau_fast: float
    var_slow: float
    tau_slow: float
    per_record: pd.DataFrame = field(repr=False, compare=False)

    def to_frame(self, digits: int = 4) -> pd.DataFrame:
        d = {k: v for k, v in self.__dict__.items() if k != "per_record"}
        d["records"] = "+".join(self.records)
        d["n_in_window"] = ",".join(str(v) for v in self.n_in_window)
        return _row_frame(d, digits)

    def summary(self) -> str:
        return (f"{self.label} [{'+'.join(self.records)}; n={self.n_total}; in-window {self.n_in_window}]: "
                f"A = {self.amplitude:+.3g} (SE {self.se:.3g}, t {self.t:+.2f}); p_normal {self.p_normal:.3f}; "
                f"rotation p {self.p_rotation_two:.3f} (n {self.n_rotation}); boot p {self.p_boot_two:.3f}; "
                f"MDE80 {self.mde_80:.3g}; null fast sd {np.sqrt(self.var_fast):.3g}/tau {self.tau_fast:.1f}, "
                f"slow sd {np.sqrt(self.var_slow):.3g}/tau {self.tau_slow:.0f}")

    def plot(self, *, title: str = "", ax=None):
        """Per-record and stacked amplitudes with 95% intervals. Returns the Figure."""
        from ..plot import _new_ax

        fig, ax = _new_ax(ax)
        pr = self.per_record
        names = list(pr["record"]) + ["stacked"]
        est = np.r_[pr["amplitude"].to_numpy(float), self.amplitude]
        se = np.r_[pr["se"].to_numpy(float), self.se]
        y = np.arange(len(names))[::-1]
        ax.errorbar(est, y, xerr=1.96 * se, fmt="o", color="0.1", ecolor="0.5", capsize=2)
        ax.axvline(0.0, color="0.7", lw=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.set_xlabel("window amplitude")
        ax.set_title(title or self.label)
        return fig


def stacked_matched_filter(records: Sequence[ProxyRecord], start: int, end: int, lifetime: float,
                           label: str = "", catalog_onsets: Sequence[int] = (), exclusion: int = 50,
                           onset_step: int = 5, tau_bounds=(1.0, 60.0, 60.0, 2000.0), n_boot: int = 2000,
                           seed: int = 20260908, n_starts: int = 4) -> StackedMatchedFilterResult:
    """Matched filter across several records sharing one atmosphere and one window amplitude.

    Each record keeps its own ages, errors, gas-age kernel and polynomial trend; the two OU
    components and the amplitude are shared, and the cross-record covariance uses the kernel-pair
    form of :func:`ou_firn_covariance`. Hyperparameters by restricted maximum likelihood on the
    stacked vector with ``[start - 20, end + 3 * lifetime]`` masked in every record.
    """
    if len(records) < 1:
        raise ValueError("stacked_matched_filter needs at least one record")
    rng = np.random.default_rng(seed)
    W = end - start
    keep = [~((r.t >= start - 20) & (r.t <= end + 3 * lifetime)) for r in records]
    y_fit = np.concatenate([r.y[k] for r, k in zip(records, keep)])
    X_fit = _trend_block(records, keep)
    sd = float(np.std(y_fit))
    starts = [(np.log(0.25 * sd ** 2), np.log(8.0), np.log(0.5 * sd ** 2), np.log(300.0))]
    for _ in range(n_starts - 1):
        starts.append((np.log(sd ** 2 * rng.uniform(.05, 1)), np.log(rng.uniform(2, 40)),
                       np.log(sd ** 2 * rng.uniform(.05, 1)), np.log(rng.uniform(80, 1500))))
    best = None
    for s0 in starts:
        res = optimize.minimize(_stacked_nll, np.array(s0), args=(records, keep, y_fit, X_fit, tau_bounds),
                                method="Nelder-Mead", options={"maxiter": 3000, "xatol": 1e-4, "fatol": 1e-6})
        if best is None or res.fun < best.fun:
            best = res
    v1, tau1, v2, tau2 = (float(np.exp(v)) for v in best.x)
    y = np.concatenate([r.y for r in records])
    X0 = _trend_block(records)
    S = _stacked_cov(records, None, v1, tau1, v2, tau2)
    L, Sinv = _chol_inv(S)

    def fit_at(s):
        h = np.concatenate([window_template(r.t, s, s + W, lifetime, r.firn_sigma) for r in records])
        X = np.column_stack([X0, h])
        XtSX_inv = np.linalg.inv(X.T @ Sinv @ X)
        proj = XtSX_inv @ X.T @ Sinv
        return float(proj[-1] @ y), float(np.sqrt(XtSX_inv[-1, -1])), proj[-1]

    A, se, proj_act = fit_at(start)
    tstat = A / se
    tmin = max(int(r.t.min()) for r in records) + 50
    tmax = min(int(r.t.max()) for r in records) - W - 50
    cat = np.asarray(list(catalog_onsets) + [start], float)
    ref = []
    for s in range(tmin, tmax + 1, onset_step):
        if np.all(np.abs(cat - s) > exclusion):
            a, e, _ = fit_at(s)
            ref.append(a / e)
    ref = np.asarray(ref)
    pr2 = _p_from_reference(tstat, ref)[1] if len(ref) else float("nan")
    beta0 = np.linalg.solve(X0.T @ Sinv @ X0, X0.T @ Sinv @ y)
    mu = X0 @ beta0
    Z = rng.standard_normal((n_boot, len(y)))
    tboot = ((mu[None, :] + Z @ L.T) @ proj_act) / se
    pb2 = _p_from_reference(tstat, tboot)[1]
    per = []
    for r in records:
        Sr = (v1 * ou_firn_covariance(r.t, tau1, r.firn_sigma) + v2 * ou_firn_covariance(r.t, tau2, r.firn_sigma)
              + np.diag(r.unc ** 2))
        _, Sri = _chol_inv(Sr)
        tc = (r.t - r.t.mean()) / (r.t.max() - r.t.min())
        Xr = np.column_stack([tc ** k for k in range(r.trend_df)]
                             + [window_template(r.t, start, end, lifetime, r.firn_sigma)])
        V = np.linalg.inv(Xr.T @ Sri @ Xr)
        b = V @ Xr.T @ Sri @ r.y
        se_r = float(np.sqrt(V[-1, -1]))
        per.append(dict(record=r.name, n=len(r.t), n_in_window=int(np.sum((r.t >= start) & (r.t <= end))),
                        amplitude=float(b[-1]), se=se_r, t=float(b[-1]) / se_r, mde80=_Z_MDE * se_r))
    return StackedMatchedFilterResult(label, tuple(r.name for r in records), len(y),
                                      tuple(int(np.sum((r.t >= start) & (r.t <= end))) for r in records),
                                      float(A), float(se), float(tstat), float(2 * stats.norm.sf(abs(tstat))),
                                      pr2, len(ref), pb2, _Z_MDE * float(se), v1, tau1, v2, tau2,
                                      pd.DataFrame(per))


# ============================================================================ calibration harness
@dataclass(frozen=True)
class CalibrationResult(_Presentable):
    """Size and power of the matched filter on one sampling design.

    Attributes
    ----------
    label : str
    refit : bool
        Whether each replicate re-fitted the null hyperparameters.
    table : pandas.DataFrame
        One row per amplitude (in units of the design's MDE80): replicates, rejection rates at
        ``alpha`` for the normal, Student-t (when requested) and feasible-onset references,
        ``sd(A_hat) / mean(se)``, and the share of replicates with a correlation time at a bound.
    """

    label: str
    refit: bool
    alpha: float
    mde_80: float
    table: pd.DataFrame = field(repr=False, compare=False)

    def to_frame(self, digits: int = 4) -> pd.DataFrame:
        return self.table.map(lambda v: _fmt(v, digits)) if hasattr(self.table, "map") \
            else self.table.applymap(lambda v: _fmt(v, digits))

    def summary(self) -> str:
        lines = [f"{self.label}: calibration ({'re-fitted' if self.refit else 'fixed'} null), "
                 f"alpha {self.alpha}, MDE80 {self.mde_80:.4g}"]
        for _, r in self.table.iterrows():
            lines.append(f"  A = {r['amplitude_mde']:.2f} x MDE80: reject normal {r['reject_normal']:.3f}, "
                         f"rotation {r['reject_rotation']:.3f}; sd(A)/se {r['sd_over_se']:.3f} (R {int(r['R'])})")
        return "\n".join(lines)

    def plot(self, *, title: str = "", ax=None):
        """Rejection rate against amplitude, with alpha and 0.8 marked. Returns the Figure."""
        from ..plot import _new_ax

        fig, ax = _new_ax(ax)
        tb = self.table
        ax.plot(tb["amplitude_mde"], tb["reject_normal"], "o-", color="0.1", label="normal")
        ax.plot(tb["amplitude_mde"], tb["reject_rotation"], "s--", color="0.5", label="feasible onset")
        ax.axhline(self.alpha, color="0.7", lw=0.6)
        ax.axhline(0.8, color="0.7", lw=0.6, ls=":")
        ax.set_xlabel("amplitude / MDE80")
        ax.set_ylabel("rejection rate")
        ax.set_title(title or self.label)
        ax.legend(frameon=False, fontsize=8)
        return fig


def calibrate_matched_filter(t, unc, gp: GPFit, start: int, end: int, lifetime: float,
                             amplitudes: Sequence[float] = (0.0, 1.0), R: int = 500,
                             refit: OUFirnGP | None = None, alpha: float = 0.05,
                             onset_step: int = 10, catalog_onsets: Sequence[int] = (),
                             exclusion: int = 50, t_df: float | None = None, seed: int = 20260908,
                             label: str = "") -> CalibrationResult:
    """Monte Carlo size and power of :func:`matched_filter` on the sampling design ``(t, unc)``.

    Replicates are drawn from the fitted null ``gp`` (its GLS trend on zero data is zero, so the
    draws are pure GP + measurement noise) plus ``amplitude x MDE80 x template``. With ``refit``
    (an :class:`OUFirnGP`), each replicate re-fits the hyperparameters with the window masked, as
    an applied user would; otherwise the covariance is held at ``gp.Sigma`` and the study isolates
    the estimator and its references.

    Returns
    -------
    CalibrationResult
    """
    t = np.asarray(t, float)
    unc = np.asarray(unc, float)
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(gp.Sigma)
    h = window_template(t, start, end, lifetime, gp.firn_sigma)
    base = matched_filter(t, np.zeros(len(t)) + 1e-12 * rng.standard_normal(len(t)), unc, gp, start, end,
                          lifetime, onset_step=onset_step, catalog_onsets=catalog_onsets,
                          exclusion=exclusion, n_boot=10, seed=seed)
    mde = _mde(base.se, t_df)
    rows = []
    for amp in amplitudes:
        rej_n = rej_r = rej_t = 0
        A_hat, se_hat, at_bound = [], [], 0
        for _ in range(R):
            y = L @ rng.standard_normal(len(t)) + amp * mde * h
            g = gp
            if refit is not None:
                g = refit.fit(t, y, unc, seed=int(rng.integers(1 << 31)),
                              mask_window=(start - 20, end + 3 * lifetime), n_starts=2)
                tb = refit.tau_bounds
                at_bound += int(np.isclose(g.tau_fast, tb[0], rtol=1e-3) or np.isclose(g.tau_fast, tb[1], rtol=1e-3)
                                or np.isclose(g.tau_slow, tb[2], rtol=1e-3) or np.isclose(g.tau_slow, tb[3], rtol=1e-3))
            r = matched_filter(t, y, unc, g, start, end, lifetime, onset_step=onset_step,
                               catalog_onsets=catalog_onsets, exclusion=exclusion, n_boot=10,
                               seed=int(rng.integers(1 << 31)), t_df=t_df)
            rej_n += r.p_normal < alpha
            rej_r += r.p_rotation_two < alpha
            rej_t += (r.p_t < alpha) if t_df is not None else 0
            A_hat.append(r.amplitude)
            se_hat.append(r.se)
        A_hat, se_hat = np.asarray(A_hat), np.asarray(se_hat)
        rows.append(dict(amplitude_mde=float(amp), R=int(R), reject_normal=rej_n / R, reject_rotation=rej_r / R,
                         reject_t=(rej_t / R) if t_df is not None else float("nan"),
                         mean_A=float(A_hat.mean()), sd_over_se=float(A_hat.std(ddof=1) / se_hat.mean()),
                         mc_se=float(np.sqrt(max(rej_n / R * (1 - rej_n / R), 1e-12) / R)),
                         share_tau_at_bound=(at_bound / R) if refit is not None else float("nan")))
    return CalibrationResult(label or f"matched filter [{start}-{end}]", refit is not None, float(alpha),
                             float(mde), pd.DataFrame(rows))
