"""Live checks of the microdata providers. Opt-in: ``pytest -m network``.

The SCF check is a real oracle: the Board's report on the 2025 survey
(*Changes in U.S. Family Finances from 2022 to 2025*, Table 2) gives
median family net worth of $215,900 and mean of $1,241,500 (2025
dollars). Reproducing both from the summary extract confirms the
download, the implicate handling and the weights in one go. The Board
re-bases every extract to the latest survey's dollars when a new survey
comes out, so these levels move with the next release; the 2022 check
uses the median-to-mean ratio from the 2022 Bulletin ($192,900 /
$1,063,700, 2022 dollars), which re-basing does not change.

The ENIGH check uses INEGI's 2022 headline: average quarterly current
income per household of 63,695 pesos (ENIGH 2022 press release).
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.network


def _scf_networth(year):
    from puremacro.fetch.micro import fetch_scf
    try:
        scf = fetch_scf(year, ["networth"])
    except OSError as exc:                       # unreachable host: not a code failure
        pytest.skip(f"SCF download failed: {exc}")
    return scf.quantile("networth", 0.5).iloc[0], scf.mean("networth").iloc[0]


def test_scf_2025_reproduces_published_net_worth():
    med, mean = _scf_networth(2025)
    assert med["estimate"] == pytest.approx(215_900, rel=0.01)
    assert mean["estimate"] == pytest.approx(1_241_500, rel=0.01)
    assert 0 < med["se"] < 0.1 * med["estimate"]


def test_scf_2022_reproduces_published_median_to_mean_ratio():
    med, mean = _scf_networth(2022)
    assert med["estimate"] / mean["estimate"] == pytest.approx(192_900 / 1_063_700, rel=0.01)
    assert 0 < med["se"] < 0.1 * med["estimate"]


def test_acs_pums_small_state_pull():
    from puremacro import credentials
    if not credentials.get("census"):
        pytest.skip("no Census API key configured")
    from puremacro.fetch.micro import fetch_acs_pums
    acs = fetch_acs_pums(2023, ["AGEP", "WAGP", "ADJINC"], geography="state:11")
    assert len(acs) > 1000
    assert len(acs.design.replicate_weights) == 80
    res = acs.mean("AGEP")
    assert 30 < res.loc[0, "estimate"] < 45 and np.isfinite(res.loc[0, "se"])
    # WAGP == -1 means "under 15": it must be missing, not a negative wage.
    assert (acs.data["WAGP"] < 0).sum() == 0
    young = acs.data["AGEP"] < 15
    assert young.any() and acs.data.loc[young, "WAGP"].isna().all()
    assert acs.data.loc[~young, "WAGP"].notna().all()


def test_enigh_2022_reproduces_published_mean_income():
    from puremacro.fetch.micro import fetch_enigh
    try:
        enigh = fetch_enigh(2022, ["ing_cor"])
    except OSError as exc:
        pytest.skip(f"ENIGH download failed: {exc}")
    res = enigh.mean("ing_cor").iloc[0]
    assert res["estimate"] == pytest.approx(63_695, rel=0.01)
    assert 0 < res["se"] < 0.05 * res["estimate"]
    assert enigh.data["est_dis"].nunique() > 100
