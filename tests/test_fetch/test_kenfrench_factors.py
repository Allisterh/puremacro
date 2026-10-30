"""Offline tests for puremacro.fetch.kenfrench_factors (synthetic zips).

The in-memory CSVs mimic the layout observed on 2026-10-28: a preamble, a
``,col,...`` header, ``YYYYMM`` rows, a blank line, an annual block with
``YYYY`` rows, and for the 25 portfolios further (equal-weighted, ...) blocks.
"""
from __future__ import annotations

import inspect
import io
import re
import zipfile

import numpy as np
import pandas as pd
import pytest

import puremacro.fetch.kenfrench_factors as kf

FF3 = """This file was created using the 202609 CRSP database.
The 1-month TBill rate data until 202405 are from Ibbotson Associates.

,Mkt-RF,SMB,HML,RF
192607,   2.89,  -2.42,  -2.75,   0.22
192608,   2.64,  -1.17,   3.82,   0.25
192609,   0.38,  -1.20, -99.99,   0.23

 Annual Factors: January-December 
,Mkt-RF,SMB,HML,RF
  1927,  29.44,  -3.05,  -3.36,   3.12
  1928,  35.56,   3.73,  -5.26,   3.56

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""

MOM = """This file was created using the 202609 CRSP database.  It
contains a momentum factor.
Missing data are indicated by -99.99 or -999.

,Mom
192608,   0.57
192609,  -999

Annual Factors:
January-December 
,Mom
  1927,  24.52

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""

_P25 = ["SMALL LoBM", "ME1 BM2", "BIG HiBM"]
P25 = f"""This file was created using the 202609 CRSP database.
It contains value- and equal-weighted returns for portfolios formed on ME and BEME.

  Average Value Weighted Returns -- Monthly
,{",".join(_P25)}
192607,   3.7806,  -0.4119,  -1.9223
192608,   1.0000,   2.0000,   3.0000


  Average Equal Weighted Returns -- Monthly
,{",".join(_P25)}
192607,   9.9999,   9.9999,   9.9999
192608,   9.9999,   9.9999,   9.9999


  Average Value Weighted Returns -- Annual
,{",".join(_P25)}
  1927,  46.8119,  35.8799,  19.6342

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""


def _zip(name, text):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(name, text)
    return buf.getvalue()


_FILES = {
    "F-F_Research_Data_Factors_CSV.zip": _zip("F-F_Research_Data_Factors.csv", FF3),
    "F-F_Research_Data_5_Factors_2x3_CSV.zip": _zip(
        "F-F_Research_Data_5_Factors_2x3.csv",
        FF3.replace(",Mkt-RF,SMB,HML,RF", ",Mkt-RF,SMB,HML,RMW,CMA,RF")
           .replace("   0.22", "   0.10,   0.20,   0.22").replace("   0.25", "   0.10,   0.20,   0.25")
           .replace("   0.23", "   0.10,   0.20,   0.23").replace("   3.12", "   1.0,   2.0,   3.12")
           .replace("   3.56", "   1.0,   2.0,   3.56")),
    "F-F_Momentum_Factor_CSV.zip": _zip("F-F_Momentum_Factor.csv", MOM),
    "25_Portfolios_5x5_CSV.zip": _zip("25_Portfolios_5x5.csv", P25),
}


@pytest.fixture
def fake_http(monkeypatch):
    urls = []

    def fake(url, refresh=False, **kw):
        urls.append(url)
        return _FILES[url.rsplit("/", 1)[1]]

    monkeypatch.setattr(kf, "safe_get_bytes_cached", fake)
    return urls


def test_parses_monthly_block_in_decimals(fake_http):
    df = kf.fetch("ff3")
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert df.index.name == "date"
    assert list(df.index) == [pd.Timestamp("1926-07-01"), pd.Timestamp("1926-08-01"),
                              pd.Timestamp("1926-09-01")]
    np.testing.assert_allclose(df["Mkt-RF"].to_numpy(), [0.0289, 0.0264, 0.0038])
    assert df.attrs["copyright"] == "Eugene F. Fama and Kenneth R. French"
    assert df.attrs["url"].endswith("F-F_Research_Data_Factors_CSV.zip")
    assert "fetched_at" in df.attrs and "source" in df.attrs
    ff5 = kf.fetch("ff5")
    assert list(ff5.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    np.testing.assert_allclose(ff5["RMW"].to_numpy(), [0.001] * 3)


def test_parses_annual_block(fake_http):
    df = kf.fetch("ff3", frequency="A")
    assert list(df.index) == [pd.Timestamp("1927-01-01"), pd.Timestamp("1928-01-01")]
    np.testing.assert_allclose(df["RF"].to_numpy(), [0.0312, 0.0356])
    mom = kf.fetch("mom", frequency="A")
    np.testing.assert_allclose(mom.loc["1927-01-01", "Mom"], 0.2452)
    np.testing.assert_allclose(mom.loc["1927-01-01", "RF"], 0.0312)


def test_25_portfolios_reads_value_weighted_block_only_and_adds_rf(fake_http):
    df = kf.fetch("25_size_bm")
    assert list(df.columns) == _P25 + ["RF"]
    assert len(df) == 2
    np.testing.assert_allclose(df.iloc[1, :3].to_numpy(), [0.01, 0.02, 0.03])
    assert not np.any(np.isclose(df.iloc[:, :3].to_numpy(), 0.099999))
    np.testing.assert_allclose(df["RF"].to_numpy(), [0.0022, 0.0025])
    ann = kf.fetch("25_size_bm", frequency="A")
    np.testing.assert_allclose(ann.iloc[0, 0], 0.468119)


def test_sentinels_become_nan(fake_http):
    df = kf.fetch("ff3")
    assert np.isnan(df.loc["1926-09-01", "HML"])
    mom = kf.fetch("mom")
    assert np.isnan(mom.loc["1926-09-01", "Mom"])
    np.testing.assert_allclose(mom.loc["1926-08-01", "RF"], 0.0025)


def test_unknown_dataset_or_frequency_raises(fake_http):
    with pytest.raises(ValueError, match="dataset"):
        kf.fetch("ff6")
    with pytest.raises(ValueError, match="frequency"):
        kf.fetch("ff3", frequency="D")
    assert fake_http == []


def test_writes_nothing_under_data_raw(fake_http, tmp_path, monkeypatch):
    src = inspect.getsource(kf)
    assert "data/raw" not in src and '"raw"' not in src
    assert re.search(r"^_CACHE\s*=", src, flags=re.M) is None  # no parsed-file cache
    from puremacro.fetch._http import CACHE_ROOT as root  # the repo's data/raw

    before = set(root.rglob("*")) if root.exists() else set()
    kf.fetch("ff3")
    kf.fetch("25_size_bm")
    after = set(root.rglob("*")) if root.exists() else set()
    assert after == before


def test_registry_entry_not_redistributable():
    from puremacro.fetch import registry

    info = registry.info("kenfrench.factors")
    assert info.redistributable is False
    assert info.provider == "kenfrench" and info.freq == "M"
    assert "ken.french/data_library" in info.terms_url
    assert registry.load("kenfrench.factors") is kf.fetch
