"""FRED inputs for consumption-based asset pricing (no key; fetched, not bundled).

``fred_consumption_growth`` builds gross per-capita real nondurables-plus-
services consumption growth; ``fred_risk_free`` converts the 3-month
Treasury bill rate to a per-period decimal return.
"""
from __future__ import annotations

import pandas as pd

__all__ = ["fred_consumption_growth", "fred_risk_free"]

#: Monthly nominal PCE (nondurables, services), their chain-type price
#: indexes (2017 = 100) and total population (thousands).
CONSUMPTION_IDS = ("PCEND", "PCES", "DNDGRG3M086SBEA", "DSERRG3M086SBEA", "POPTHM")


def _fetch(ids, start, refresh) -> pd.DataFrame:
    from ..fetch.rates import fetch_fred_many

    df = fetch_fred_many(list(ids), start=start, refresh=refresh)
    missing = [m.get("id", m) if isinstance(m, dict) else m
               for m in (df.attrs.get("missing") or [])]
    absent = [i for i in ids if i not in df.columns]
    bad = list(dict.fromkeys([str(m) for m in missing if m in ids] + absent))
    if bad:
        raise RuntimeError(
            f"FRED series {bad} could not be fetched; refusing to build the series "
            "from a partial set")
    return df


def _check_frequency(frequency: str) -> str:
    if frequency not in ("M", "Q"):
        raise ValueError(f"frequency must be 'M' or 'Q'; got {frequency!r}")
    return frequency


def _quarterly_mean(s: pd.Series) -> pd.Series:
    """Average of monthly values within complete calendar quarters."""
    key = s.index.to_period("Q")
    grp = s.groupby(key)
    out = grp.mean()[grp.count() == 3]
    out.index = out.index.start_time
    return out


def fred_consumption_growth(frequency: str = "Q", start: str | None = None,
                            refresh: bool = False) -> pd.Series:
    """Gross growth of per-capita real nondurables-plus-services consumption.

    Real consumption is ``PCEND / P_nd + PCES / P_sv``, each nominal series
    (monthly, SAAR) deflated by its own chain-type price index
    (``DNDGRG3M086SBEA``, ``DSERRG3M086SBEA``) and divided by population
    (``POPTHM``). Summing two chained-dollar series approximates, but is not
    identical to, a chained aggregate. ``"Q"`` averages the monthly levels
    within complete quarters; ``"M"`` uses months. Growth is the ratio of
    consecutive levels, indexed by the later period's start. The sample is
    the span on which all five series are published. Any series that FRED
    does not return raises ``RuntimeError`` naming it.
    """
    _check_frequency(frequency)
    df = _fetch(CONSUMPTION_IDS, start, refresh)
    df = df[list(CONSUMPTION_IDS)].dropna()
    real = (df["PCEND"] / (df["DNDGRG3M086SBEA"] / 100.0)
            + df["PCES"] / (df["DSERRG3M086SBEA"] / 100.0))
    level = real / df["POPTHM"]
    level.index = pd.DatetimeIndex(level.index)
    if frequency == "Q":
        level = _quarterly_mean(level)
    growth = (level / level.shift(1)).iloc[1:]
    growth.name = "consumption_growth"
    growth.index.name = "date"
    growth.attrs = {"source": "FRED", "ids": list(CONSUMPTION_IDS), "frequency": frequency}
    return growth


def fred_risk_free(frequency: str = "M", start: str | None = None,
                   refresh: bool = False) -> pd.Series:
    """Per-period risk-free rate in decimals from FRED ``TB3MS``.

    ``TB3MS`` (3-month Treasury bill, secondary market, annualized percent on
    a discount basis) becomes ``TB3MS / 100 / 12`` per month, or the quarter
    average ``/ 100 / 4`` per quarter. Simple division ignores the discount
    basis and compounding: an approximation adequate for excess returns.
    """
    _check_frequency(frequency)
    df = _fetch(("TB3MS",), start, refresh)
    s = df["TB3MS"].dropna()
    s.index = pd.DatetimeIndex(s.index)
    if frequency == "Q":
        s = _quarterly_mean(s) / 100.0 / 4.0
    else:
        s = s / 100.0 / 12.0
    s.name = "rf"
    s.index.name = "date"
    s.attrs = {"source": "FRED", "ids": ["TB3MS"], "frequency": frequency}
    return s
