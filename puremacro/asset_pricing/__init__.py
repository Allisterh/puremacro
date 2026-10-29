"""Macro-finance: returns, risk, VaR and asset-pricing tests.

Returns and risk (:mod:`.returns`), value at risk and its backtests, the
time-series (GRS) and Fama-MacBeth factor tests, linear stochastic discount
factor GMM, the Hansen-Jagannathan distance and the Hansen-Singleton
consumption CAPM. The moment-based estimators are built on
:func:`puremacro.inference.gmm`. No asset data are bundled; see
:mod:`puremacro.fetch.kenfrench_factors` and :mod:`.macro_data`.
"""
from .returns import (
    annualize_return,
    annualize_volatility,
    compound,
    excess_returns,
    to_returns,
)
from .risk import expected_shortfall, risk_summary, value_at_risk, var_backtest
from .factor_tests import fama_macbeth, time_series_test
from ._results import (
    FamaMacBethResult,
    RiskSummaryResult,
    TimeSeriesTestResult,
    VaRBacktestResult,
)

__all__ = [
    "to_returns",
    "excess_returns",
    "compound",
    "annualize_return",
    "annualize_volatility",
    "risk_summary",
    "RiskSummaryResult",
    "value_at_risk",
    "expected_shortfall",
    "var_backtest",
    "VaRBacktestResult",
    "time_series_test",
    "TimeSeriesTestResult",
    "fama_macbeth",
    "FamaMacBethResult",
]
