"""The bundled OECD ICIO table (ISIC Rev.4 sections, G20 countries) and its builder.

The builder ``tools/build_icio_isic.py`` is exercised offline on a synthetic
table in the native OECD ICIO 2023 layout (77 economies, 45 activities, six
final uses). The checks on the bundled file itself run when the file is in the
tree; without it the default loader must say how to get one (checked by
pointing the loader at an empty directory).
"""
from __future__ import annotations

import importlib.util
import re
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from puremacro.trade import data as trade_data
from puremacro.trade.data import (
    RAW_45_SECTOR_CODES,
    OECD_77_COUNTRIES,
    ICIOData,
    generate_synthetic_mrio,
    load_icio_data,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER = REPO_ROOT / "tools" / "build_icio_isic.py"
DATAFILES = REPO_ROOT / "puremacro" / "trade" / "_datafiles"
SOURCES = DATAFILES / "SOURCES.md"
BUNDLED = DATAFILES / trade_data._BUNDLED_ICIO_FILES["oecd_isic"]

needs_bundled = pytest.mark.skipif(not BUNDLED.is_file(), reason=f"{BUNDLED.name} is not built in this tree")


@pytest.fixture(scope="module")
def builder():
    spec = importlib.util.spec_from_file_location("build_icio_isic", BUILDER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_icio_isic"] = module
    spec.loader.exec_module(module)
    return module


def _sources_codes(label: str) -> list[str]:
    text = SOURCES.read_text(encoding="utf-8")
    match = re.search(rf"^{label} \((\d+)\): `([^`]*)`", text, flags=re.M)
    assert match, f"SOURCES.md has no '{label} (n): `...`' line"
    codes = match.group(2).split()
    assert len(codes) == int(match.group(1))
    return codes


def _native_raw(seed: int = 3):
    """A synthetic table in the native OECD layout, with some negative inventory changes."""
    raw = generate_synthetic_mrio("oecd", seed=seed)
    F = np.array(raw.final_demand_matrix, dtype=float).reshape(raw.M, raw.C, raw.K_F)
    invnt = list(raw.fd_categories).index("INVNT")
    gfcf = list(raw.fd_categories).index("GFCF")
    # Inventory draw-downs larger than fixed capital formation: US agriculture (both
    # activities of section A) sold to US investment, so the section's I cell stays negative.
    usa = list(raw.countries).index("USA")
    for activity in (0, 1):
        F[usa * raw.S + activity, usa, invnt] = -(F[usa * raw.S + activity, usa, gfcf] + 3.0)
    return replace(raw, final_demand_matrix=F.reshape(raw.M, -1),
                   taxes_less_subsidies_fd=np.full(raw.C * raw.K_F, 0.01))


# --- the builder --------------------------------------------------------------------


def test_builder_rosters_match_sources(builder):
    assert list(builder.COUNTRY_CODES) == _sources_codes("Country codes")
    assert list(builder.SECTOR_CODES) == _sources_codes("Sector codes")
    assert builder.OUTPUT_NAME == BUNDLED.name
    assert f"{builder.SOURCE_MD5}" in SOURCES.read_text(encoding="utf-8")


def test_rosters_follow_the_stated_rules(builder):
    assert builder.SECTOR_CODES == tuple("ABCDEFGHIJKLMNOPQRST")
    assert set(builder.SECTION_OF) == set(RAW_45_SECTOR_CODES)
    for fine, section in builder.SECTION_OF.items():
        assert fine[0] == section             # an OECD activity code starts with its ISIC section
    assert sum(s == "C" for s in builder.SECTION_OF.values()) == 17
    assert builder.COUNTRY_CODES[-1] == "ROW"
    assert len(builder.COUNTRY_CODES) == 20
    assert set(builder.COUNTRY_CODES[:-1]) <= set(OECD_77_COUNTRIES)


def test_aggregation_conserves_every_total(builder):
    from puremacro.trade._oecd_icio import condense_final_demand

    raw = condense_final_demand(_native_raw())
    coarse = builder.aggregate(raw)
    assert (coarse.C, coarse.S, coarse.K_F) == (20, 20, 3)
    assert list(coarse.countries) == list(builder.COUNTRY_CODES)
    assert list(coarse.sectors) == list(builder.SECTOR_CODES)
    for name in ("Z", "F", "VA", "TLS", "Y"):
        np.testing.assert_allclose(getattr(coarse, name).sum(), getattr(raw, name).sum(), rtol=1e-12)
    np.testing.assert_allclose(coarse.taxes_less_subsidies_fd.sum(), raw.taxes_less_subsidies_fd.sum(), rtol=1e-12)
    # Domestic flows of the United States (a G20 member) stay domestic.
    usa_in = list(raw.countries).index("USA")
    usa_out = list(coarse.countries).index("USA")
    fine = raw.Z[usa_in * 45:(usa_in + 1) * 45, usa_in * 45:(usa_in + 1) * 45].sum()
    np.testing.assert_allclose(coarse.Z[usa_out * 20:(usa_out + 1) * 20, usa_out * 20:(usa_out + 1) * 20].sum(),
                               fine, rtol=1e-12)


def test_negative_final_uses_are_netted_within_the_row(builder):
    F = np.array([[5.0, -2.0, 1.0, 4.0], [1.0, 2.0, 3.0, 0.0]])
    netted, report = builder.net_negative_final_uses(F)
    assert np.all(netted >= 0)
    np.testing.assert_allclose(netted.sum(axis=1), F.sum(axis=1), rtol=1e-15)
    np.testing.assert_array_equal(netted[1], F[1])
    assert report["cells"] == 1 and report["rows"] == 1
    np.testing.assert_allclose(report["amount"], 2.0)
    with pytest.raises(ValueError, match="negative total final use"):
        builder.net_negative_final_uses(np.array([[1.0, -2.0]]))


def test_build_table_is_balanced_nonnegative_and_calibrates(builder):
    from puremacro.trade.calibration import calibrate_trade_model

    matrix, info = builder.build_table(_native_raw())
    n = 20 * 20
    assert matrix.shape == (n + 3, n + 3 * 20)
    assert np.all(matrix[:n, :] >= 0)
    rows, cols = matrix[:n].sum(axis=1), matrix[:, :n].sum(axis=0)
    assert np.max(np.abs(rows - cols) / np.maximum(np.abs(rows), 1.0)) < 1e-9
    assert info["negative_final_uses_netted"]["cells"] >= 1
    calibrate_trade_model(matrix, nc=20, ns=20, nfd=3, validate=True)


def test_builder_refuses_an_unregistered_file(builder, tmp_path):
    source = tmp_path / "2019_SML.csv"
    source.write_text("V1,ARG_A01_02\nARG_A01_02,1\n", encoding="utf-8")
    with pytest.warns(RuntimeWarning, match="not a whitelisted"):
        with pytest.raises(SystemExit, match="not the registered OECD"):
            builder.main(["--source", str(source), "--output", str(tmp_path / "out.npz")])
    assert not (tmp_path / "out.npz").exists()


def test_write_and_load_round_trip(builder, tmp_path, monkeypatch):
    matrix, info = builder.build_table(_native_raw(seed=4))
    path = tmp_path / BUNDLED.name
    builder.write_table(path, matrix, info)
    monkeypatch.setattr(trade_data, "_BUNDLED_ICIO_DIR", tmp_path)
    np.testing.assert_array_equal(load_icio_data(), matrix)
    icio = load_icio_data(return_structured=True)
    assert isinstance(icio, ICIOData)
    assert icio.country_codes == builder.COUNTRY_CODES
    assert icio.sector_codes == builder.SECTOR_CODES
    assert icio.fd_codes == ("C", "I", "Cx")
    assert icio.metadata["source"] == "oecd_isic"
    assert "OECD" in icio.metadata["citation"]


# --- the default loader ----------------------------------------------------------------


def test_removed_and_unknown_sources():
    with pytest.raises(ValueError, match="removed"):
        load_icio_data(source="legacy")
    with pytest.raises(ValueError, match="unknown ICIO source 'oecd'.*oecd_isic"):
        load_icio_data(source="oecd")


def test_missing_bundled_table_names_the_builder(tmp_path, monkeypatch):
    monkeypatch.setattr(trade_data, "_BUNDLED_ICIO_DIR", tmp_path)       # a tree without the file
    with pytest.raises(FileNotFoundError, match="tools/build_icio_isic.py") as excinfo:
        load_icio_data()
    assert "synthetic_icio_table" in str(excinfo.value)


# --- the bundled file ----------------------------------------------------------------------


@needs_bundled
def test_bundled_table_balanced():
    T = load_icio_data()
    n = 20 * 20
    assert T.shape == (n + 3, n + 3 * 20)
    rows, cols = T[:n].sum(axis=1), T[:, :n].sum(axis=0)
    assert np.max(np.abs(rows - cols) / np.abs(rows)) < 1e-6
    assert np.all(T[:n, :] >= 0)


@needs_bundled
def test_bundled_codes_match_sources():
    icio = load_icio_data(return_structured=True)
    assert list(icio.country_codes) == _sources_codes("Country codes")
    assert list(icio.sector_codes) == _sources_codes("Sector codes")


@needs_bundled
def test_world_va_plausible():
    icio = load_icio_data(return_structured=True)
    world_va = (icio.labor_va.sum() + icio.capital_va.sum()) * 1e6       # million USD -> USD
    assert 50e12 < world_va < 150e12


@needs_bundled
def test_bundled_table_calibrates():
    from puremacro.trade.calibration import calibrate_trade_model

    calibrate_trade_model(load_icio_data(), nc=20, ns=20, nfd=3, validate=True)
