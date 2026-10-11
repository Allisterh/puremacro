"""Census Microdata API provider against a fake endpoint (offline)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from puremacro.fetch.micro import census

KEY = "SECRETKEY0123456789"

# A slice of the real 2023 ACS PUMS variables.json, labels as published.
CODEBOOK = {"variables": {
    "WAGP": {"values": {"range": [{"min": "4", "max": "999999"}],
                        "item": {"0": "None",
                                 "-1": "N/A (less than 15 years old). Unselect this value "
                                       "to get correct calculation of average for this variable"}}},
    "ESR": {"values": {"item": {"0": "N/A (less than 16 years old)", "1": "Civilian employed, at work",
                                "6": "Not in Labor Force"}}},
    "AGEP": {"values": {"range": [{"min": "1", "max": "99"}], "item": {"0": "Under 1 year"}}},
    "PEMLR": {"values": {"item": {"-1": "Not in Universe", "1": "Employed-At Work"}}},
    "PEHRUSL1": {"values": {"item": {"-1": "Not in Universe", "-4": "Hours Vary"}}},
}}


def _serve_codebook(url, secret_params):
    assert url.endswith("/variables.json")
    assert not secret_params, "the codebook needs no key and must not get one"
    return json.dumps(CODEBOOK).encode()


class FakeCensus:
    """Serves a synthetic person file, honouring get=, for= and the key."""

    def __init__(self, n=30, seed=0):
        rng = np.random.default_rng(seed)
        serial = [f"2023HU{1000 + i // 3:07d}" for i in range(n)]
        cols = {"SERIALNO": serial,
                "SPORDER": [str(i % 3 + 1) for i in range(n)],
                "AGEP": [str(a) for a in rng.integers(18, 90, n)],
                "WAGP": [str(a) for a in rng.integers(0, 200000, n)],
                "ESR": [str(a) for a in rng.choice([1, 6], n)],
                "ADJINC": ["1019518"] * n}
        for stem in ("PWGTP", "WGTP"):
            base = rng.integers(5, 60, n)
            cols[stem] = [str(b) for b in base]
            for r in range(1, 81):
                cols[f"{stem}{r}"] = [str(max(0, b + d)) for b, d in
                                      zip(base, rng.integers(-5, 6, n))]
        self.table = pd.DataFrame(cols)
        self.calls = []

    def __call__(self, url, *, params=None, secret_params=None, refresh=False,
                 timeout=60, headers=None):
        if url.endswith("/variables.json"):
            self.codebook_calls = getattr(self, "codebook_calls", 0) + 1
            return _serve_codebook(url, secret_params)
        self.calls.append({"url": url, "params": dict(params or {}),
                           "secret": dict(secret_params or {})})
        assert secret_params == {"key": KEY}
        assert "key" not in (params or {})
        get = params["get"].split(",")
        assert len(get) <= census.MAX_VARIABLES
        # shuffle row order per call, as a real server may
        t = self.table.sample(frac=1, random_state=len(self.calls))
        rows = [get + (["state"] if "for" in params else [])]
        for _, r in t.iterrows():
            rows.append([r[c] for c in get] + (["11"] if "for" in params else []))
        return json.dumps(rows).encode()


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setenv("CENSUS_API_KEY", KEY)
    f = FakeCensus()
    monkeypatch.setattr(census._http, "cached_get", f)
    return f


def test_acs_pums_chunks_past_the_variable_cap_and_rejoins(fake):
    mf = census.fetch_acs_pums(2023, ["AGEP", "WAGP", "ADJINC"], geography="state:11")
    # 3 analysis + 81 weight columns, 48 per call after the 2 ids
    assert len(fake.calls) == 2
    assert all(c["params"]["for"] == "state:11" for c in fake.calls)
    assert len(mf.data) == 30
    t = fake.table.set_index(["SERIALNO", "SPORDER"])
    got = mf.data.assign(SPORDER=mf.data["SPORDER"].astype(str)).set_index(["SERIALNO", "SPORDER"])
    for col in ("AGEP", "WAGP", "PWGTP", "PWGTP80"):
        assert (got[col].astype(int) == t.loc[got.index, col].astype(int)).all()
    assert not pd.api.types.is_numeric_dtype(mf.data["SERIALNO"])
    assert mf.data["WAGP"].dtype.kind in "if"
    assert mf.design.method == "sdr" and mf.source == "census.acs_pums"
    assert KEY not in mf.query
    assert "state" in mf.data.columns


def test_acs_pums_household_unit_keeps_one_record_per_unit(fake):
    mf = census.fetch_acs_pums(2023, ["WAGP"], unit="household")
    assert len(mf.data) == 10
    assert (mf.data["SPORDER"] == 1).all()
    assert mf.design.weight == "WGTP"
    res = mf.mean("WAGP")
    assert np.isfinite(res.loc[0, "se"])


def test_predicates_are_passed_through(fake):
    census.fetch_acs_pums(2023, ["WAGP"], where={"AGEP": (25, 64), "SEX": 2,
                                                  "ST": ["06", "36"]},
                          replicate_weights=False)
    p = fake.calls[0]["params"]
    assert p["AGEP"] == "25:64" and p["SEX"] == "2" and p["ST"] == "06,36"
    assert len(fake.calls) == 1


def test_missing_key_raises_before_any_request(monkeypatch, tmp_path):
    from puremacro import credentials
    monkeypatch.delenv("CENSUS_API_KEY", raising=False)
    monkeypatch.delenv("PUREMACRO_CENSUS_API_KEY", raising=False)
    monkeypatch.setattr(credentials, "default_config_path", lambda: tmp_path / "none.toml")
    calls = []
    monkeypatch.setattr(census._http, "cached_get", lambda *a, **k: calls.append(1))
    with pytest.raises(credentials.MissingCredentialError):
        census.fetch_acs_pums(2023, ["WAGP"])
    assert not calls


def test_empty_204_response_gives_an_empty_frame(monkeypatch):
    monkeypatch.setenv("CENSUS_API_KEY", KEY)
    monkeypatch.setattr(census._http, "cached_get", lambda *a, **k: b"")
    df = census.fetch_records("u", ["A"], ["SERIALNO", "SPORDER"])
    assert df.empty and list(df.columns) == ["SERIALNO", "SPORDER", "A"]


def test_chunks_with_different_record_sets_are_refused(monkeypatch):
    monkeypatch.setenv("CENSUS_API_KEY", KEY)
    replies = iter([
        json.dumps([["ID", "A"], ["1", "x"], ["2", "y"]]).encode(),
        json.dumps([["ID", "B"], ["1", "x"]]).encode(),
    ])
    monkeypatch.setattr(census._http, "cached_get", lambda *a, **k: next(replies))
    monkeypatch.setattr(census, "MAX_VARIABLES", 2)
    with pytest.raises(ValueError, match="different record sets"):
        census.fetch_records("u", ["A", "B"], ["ID"], )


def test_cps_basic_is_weights_only(monkeypatch):
    monkeypatch.setenv("CENSUS_API_KEY", KEY)
    seen = {}

    def fake(url, *, params=None, secret_params=None, **k):
        if url.endswith("/variables.json"):
            return _serve_codebook(url, secret_params)
        seen["url"] = url
        get = params["get"].split(",")
        rows = [get, ["0001", "11", "1", "1", "40", "2500.5"],
                ["0001", "11", "2", "2", "-4", "2400.0"],
                ["0002", "11", "1", "-1", "-1", "1000.0"]]
        return json.dumps(rows).encode()

    monkeypatch.setattr(census._http, "cached_get", fake)
    mf = census.fetch_cps_basic(2024, 1, ["PEMLR", "PEHRUSL1"])
    assert seen["url"].endswith("/2024/cps/basic/jan")
    assert mf.design.weight == "PWCMPWGT" and not mf.design.has_variance
    assert mf.data["HRHHID"].tolist() == ["0001", "0001", "0002"]
    assert np.isnan(mf.mean("PEMLR").loc[0, "se"])
    # Not in Universe is missing; "Hours Vary" (-4) is a code, not a missing value.
    assert mf.data["PEMLR"].isna().tolist() == [False, False, True]
    assert mf.data["PEHRUSL1"].tolist()[:2] == [40, -4] and np.isnan(mf.data["PEHRUSL1"][2])


def test_not_applicable_codes_become_nan_and_leave_the_mean(fake):
    fake.table.loc[[0, 1, 2], "WAGP"] = "-1"
    fake.table.loc[[0, 1], "ESR"] = "0"
    fake.table.loc[[3], "WAGP"] = "0"          # "None" is a real zero
    mf = census.fetch_acs_pums(2023, ["AGEP", "WAGP", "ESR"], replicate_weights=False)
    by_id = mf.data.set_index(["SERIALNO", "SPORDER"])
    assert by_id["WAGP"].isna().sum() == 3 and by_id["ESR"].isna().sum() == 2
    assert (mf.data["WAGP"] == 0).sum() == (fake.table["WAGP"] == "0").sum()
    assert mf.data["AGEP"].notna().all()
    w = fake.table["PWGTP"].astype(float)
    x = fake.table["WAGP"].astype(float)
    ok = x != -1
    assert mf.mean("WAGP").loc[0, "estimate"] == pytest.approx((w[ok] * x[ok]).sum() / w[ok].sum())
    assert mf.mean("WAGP").loc[0, "n"] == 27
    book = mf.codebook.set_index(["variable", "code"])
    assert book.loc[("WAGP", "-1"), "not_applicable"]
    assert not book.loc[("WAGP", "0"), "not_applicable"]
    assert not book.loc[("AGEP", "0"), "not_applicable"]
    assert fake.codebook_calls == 1

    raw = census.fetch_acs_pums(2023, ["WAGP"], replicate_weights=False, na_to_nan=False)
    assert (raw.data["WAGP"] == -1).sum() == 3 and raw.codebook.empty


def test_codes_match_numerically():
    book = pd.DataFrame({"variable": ["X", "X"], "code": ["01", "N"], "label": ["N/A", "N/A (GQ)"],
                         "not_applicable": [True, True]})
    df = pd.DataFrame({"X": ["1", "N", "2", "10"]})
    out = census._blank_not_applicable(df, book)
    assert out["X"].isna().tolist() == [True, True, False, False]


def test_chunk_layout():
    ch = census._chunks(["A", "B"], [f"v{i}" for i in range(100)], limit=50)
    assert [len(c) for c in ch] == [48, 48, 4]
    assert census._chunks(["A"], []) == [[]]


def test_a_later_empty_chunk_is_refused_not_an_empty_result(monkeypatch):
    monkeypatch.setenv("CENSUS_API_KEY", KEY)
    replies = iter([json.dumps([["ID", "A"], ["1", "x"], ["2", "y"]]).encode(), b""])
    monkeypatch.setattr(census._http, "cached_get", lambda *a, **k: next(replies))
    monkeypatch.setattr(census, "MAX_VARIABLES", 2)
    with pytest.raises(ValueError, match="different record sets"):
        census.fetch_records("u", ["A", "B"], ["ID"])


def test_duplicate_identifiers_are_refused(monkeypatch):
    monkeypatch.setenv("CENSUS_API_KEY", KEY)
    monkeypatch.setattr(census._http, "cached_get", lambda *a, **k: json.dumps(
        [["ID", "A"], ["1", "x"], ["1", "y"]]).encode())
    with pytest.raises(ValueError, match="do not uniquely"):
        census.fetch_records("u", ["A"], ["ID"])
