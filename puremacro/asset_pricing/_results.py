"""Frozen result objects for :mod:`puremacro.asset_pricing`.

Every result is a ``@dataclass(frozen=True)`` built by a validating function
(never by a raising ``__post_init__``) and shares one report mixin:
``to_dataframe()`` / ``to_frame()`` and ``to_markdown()`` / ``to_latex()`` /
``to_typst()`` rendered through :mod:`puremacro.reports`, plus ``summary()``
and (where a figure is meaningful) ``plot(ax=None, show=False)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..inference._results import _coef_frame, _ReportMixin

__all__: list[str] = []


def _new_ax(ax: Any):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 3.4))
        return fig, ax, False
    return ax.figure, ax, True


def _finish(fig: Any, ax: Any, given: bool, show: bool):
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    return ax if given else fig


@dataclass(frozen=True)
class RiskSummaryResult(_ReportMixin):
    """Result of :func:`puremacro.asset_pricing.risk_summary`.

    Every statistic is an ``(N,)`` array aligned with ``names``; ``mean``,
    ``volatility``, ``sharpe``, ``sharpe_se`` and ``sortino`` are annualized
    with ``periods_per_year``.
    """

    names: tuple[str, ...]
    n_obs: int
    periods_per_year: float
    hac_lags: int
    mean: np.ndarray
    volatility: np.ndarray
    sharpe: np.ndarray
    sharpe_se: np.ndarray
    sortino: np.ndarray
    skewness: np.ndarray
    excess_kurtosis: np.ndarray
    max_drawdown: np.ndarray
    max_drawdown_duration: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        """One row per series."""
        return pd.DataFrame({
            "name": list(self.names),
            "mean": self.mean,
            "volatility": self.volatility,
            "sharpe": self.sharpe,
            "sharpe_se": self.sharpe_se,
            "sortino": self.sortino,
            "skewness": self.skewness,
            "excess_kurtosis": self.excess_kurtosis,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
        })

    def summary(self) -> str:
        lines = [
            f"Risk summary (T = {self.n_obs}, {self.periods_per_year:g} periods/year, "
            f"Sharpe se: Lo (2002) HAC, Bartlett lags = {self.hac_lags})",
            f"  {'name':<12}{'mean':>9}{'vol':>9}{'Sharpe':>9}{'(se)':>9}{'Sortino':>9}"
            f"{'maxDD':>8}{'dur':>6}",
        ]
        for j, n in enumerate(self.names):
            lines.append(
                f"  {n:<12}{self.mean[j]:>9.4f}{self.volatility[j]:>9.4f}"
                f"{self.sharpe[j]:>9.3f}{self.sharpe_se[j]:>9.3f}{self.sortino[j]:>9.3f}"
                f"{self.max_drawdown[j]:>8.3f}{int(self.max_drawdown_duration[j]):>6d}")
        lines.append("  Annualized by sqrt(periods/year); Lo's serial-correlation "
                     "factor eta(q) is not applied.")
        return "\n".join(lines)

    def plot(self, ax: Any = None, show: bool = False):
        """Annualized Sharpe ratios with +/- 1.96 se bars."""
        fig, ax, given = _new_ax(ax)
        pos = np.arange(len(self.names))
        ax.bar(pos, self.sharpe, color="0.6", yerr=1.96 * self.sharpe_se,
               ecolor="0.1", capsize=3)
        ax.axhline(0.0, color="0.1", lw=0.7)
        ax.set_xticks(pos)
        ax.set_xticklabels(list(self.names))
        ax.set_ylabel("annualized Sharpe ratio")
        ax.set_title("Sharpe ratios (95% HAC intervals)")
        return _finish(fig, ax, given, show)


@dataclass(frozen=True)
class VaRBacktestResult(_ReportMixin):
    """Result of :func:`puremacro.asset_pricing.var_backtest`.

    ``lr_uc`` / ``p_uc``: Kupiec (1995) unconditional coverage, ``chi2(1)``;
    ``lr_ind`` / ``p_ind``: Christoffersen (1998) first-order independence,
    ``chi2(1)``; ``lr_cc = lr_uc + lr_ind`` / ``p_cc``: conditional
    coverage, ``chi2(2)``. No ``plot`` (scalar diagnostics).
    """

    n_obs: int
    alpha: float
    n_violations: int
    violation_rate: float
    lr_uc: float
    p_uc: float
    lr_ind: float
    p_ind: float
    lr_cc: float
    p_cc: float

    def to_dataframe(self) -> pd.DataFrame:
        """Two-column ``statistic`` / ``value`` table."""
        rows = [("n_obs", self.n_obs), ("alpha", self.alpha),
                ("n_violations", self.n_violations),
                ("violation_rate", self.violation_rate),
                ("lr_uc", self.lr_uc), ("p_uc", self.p_uc),
                ("lr_ind", self.lr_ind), ("p_ind", self.p_ind),
                ("lr_cc", self.lr_cc), ("p_cc", self.p_cc)]
        return pd.DataFrame({"statistic": [r[0] for r in rows],
                             "value": [float(r[1]) for r in rows]})

    def summary(self) -> str:
        return "\n".join([
            f"VaR backtest (T = {self.n_obs}, alpha = {self.alpha:g})",
            f"  violations: {self.n_violations} (rate {self.violation_rate:.4f}, "
            f"expected {self.alpha:.4f})",
            f"  Kupiec unconditional coverage   LR = {self.lr_uc:8.4f}  p = {self.p_uc:.4f}",
            f"  Christoffersen independence     LR = {self.lr_ind:8.4f}  p = {self.p_ind:.4f}",
            f"  Christoffersen cond. coverage   LR = {self.lr_cc:8.4f}  p = {self.p_cc:.4f}",
        ])
