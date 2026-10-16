"""The seeded synthetic ICIO table and the table loader on files of your own."""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.trade.data import (
    ICIOData,
    generate_synthetic_mrio,
    load_icio_data,
    synthetic_codes,
    synthetic_icio_table,
)


def test_shape_and_layout():
    T = synthetic_icio_table(5, 4)
    assert T.shape == (23, 35)
    assert np.all(T[:20, :20] >= 0)
    labor, capital = T[21, :20], T[22, :20]
    np.testing.assert_allclose(labor / (labor + capital), 2.0 / 3.0, rtol=0, atol=1e-12)
    # Final-use columns hold no factor payments; every investment cell (category 1) is positive.
    assert np.all(T[21:, 20:] == 0.0)
    assert np.all(T[:20, 20:].reshape(20, 5, 3)[:, :, 1] > 0)


def test_rows_and_columns_balance():
    T = synthetic_icio_table(4, 3, seed=7)
    n = 12
    np.testing.assert_allclose(T[:n].sum(axis=1), T[:, :n].sum(axis=0), rtol=1e-12)


def test_deterministic():
    assert np.array_equal(synthetic_icio_table(5, 4, seed=3), synthetic_icio_table(5, 4, seed=3))
    assert not np.array_equal(synthetic_icio_table(5, 4, seed=3), synthetic_icio_table(5, 4, seed=4))


@pytest.mark.parametrize("nfd", [1, 2, 3, 4])
def test_final_use_count(nfd):
    T = synthetic_icio_table(3, 2, nfd=nfd)
    assert T.shape == (6 + 3, 6 + 3 * nfd)
    # Every grouping conserves the generator's total final use.
    np.testing.assert_allclose(T[:6, 6:].sum(), synthetic_icio_table(3, 2, nfd=3)[:6, 6:].sum(), rtol=1e-12)


def test_any_size_beyond_the_provider_roster():
    """The OECD layout lists 77 economies and 45 activities; larger tables get generated labels."""
    nc, ns = 80, 46
    assert synthetic_icio_table(nc, ns).shape == (nc * ns + 3, nc * ns + 3 * nc)
    raw = generate_synthetic_mrio("oecd", custom_c=nc, custom_s=ns)
    assert (raw.C, raw.S) == (nc, ns)
    assert raw.countries[-1] == "R003" and raw.sectors[-1] == "X001"


@pytest.mark.parametrize("bad", [0, -1, 2.5, True])
def test_bad_dimensions_raise(bad):
    with pytest.raises(ValueError, match="positive integer"):
        synthetic_icio_table(bad, 3)
    with pytest.raises(ValueError, match="positive integer"):
        synthetic_icio_table(3, 3, nfd=bad)


def test_synthetic_codes():
    countries, sectors = synthetic_codes(3, 2)
    assert countries == ["C01", "C02", "C03"] and sectors == ["S01", "S02"]
    assert synthetic_codes(100, 1)[0][0] == "C001"


def test_calibrates_at_70x12():
    from puremacro.trade.calibration import calibrate_trade_model

    calibrate_trade_model(synthetic_icio_table(70, 12, seed=1), ns=12, nc=70, nfd=3, validate=True)


@pytest.mark.parametrize("nc, ns", [(3, 2), (5, 4), (8, 5)])
def test_every_seed_calibrates(nc, ns):
    """Every seed gives a table consistent accounting accepts: investment net of the
    foreign balance (the investment expenditure share) is positive in every country."""
    from puremacro.trade._accounting import _parameters
    from puremacro.trade.calibration import calibrate_trade_model

    for seed in range(20):
        calib = calibrate_trade_model(synthetic_icio_table(nc, ns, seed=seed), ns=ns, nc=nc, nfd=3)
        investment_share = calib.theta[0, 1, :]
        assert np.all(investment_share >= 0.05 - 1e-12), (seed, investment_share)
        _parameters(calib)                      # raises ValueError on a negative share


def test_removed_and_unknown_bundled_sources():
    # The default (bundled) table is covered in test_trade_bundled_table.py.
    with pytest.raises(ValueError, match="removed"):
        load_icio_data(source="legacy")
    with pytest.raises(ValueError, match="unknown ICIO source"):
        load_icio_data(source="oecd")


def test_load_icio_data_reads_a_table_of_your_own(tmp_path):
    T = synthetic_icio_table(5, 4, seed=2)
    path = tmp_path / "table.npz"
    np.savez_compressed(path, data=T)
    assert np.array_equal(load_icio_data(path), T)
    icio = load_icio_data(path, return_structured=True)
    assert isinstance(icio, ICIOData)
    assert icio.country_codes == tuple(synthetic_codes(5, 4)[0])
    assert icio.sector_codes == tuple(synthetic_codes(5, 4)[1])
    assert icio.fd_codes == ("C", "I", "Cx")
    assert icio.intermediate_matrix.shape == (20, 20)
    np.testing.assert_allclose(icio.labor_va / (icio.labor_va + icio.capital_va), 2.0 / 3.0, atol=1e-12)
    np.testing.assert_allclose(icio.gross_output, T[:20].sum(axis=1), rtol=1e-12)
    with pytest.raises(ValueError, match="pass either"):
        load_icio_data(path, source="legacy")
    with pytest.raises(ValueError, match="24, 35"):
        np.savez_compressed(tmp_path / "bad.npz", data=np.zeros((24, 35)))
        load_icio_data(tmp_path / "bad.npz", return_structured=True)
    with pytest.raises(ValueError, match="no longer supported"):
        load_icio_data(tmp_path / "table.mat")


def test_registries_are_provider_rosters_only():
    # The 77-economy roster is exposed only under its provider name; there is no
    # fixed sector registry and no lookup keyed by a sector count.
    import puremacro.trade.data as data

    assert len(data.OECD_77_COUNTRIES) == 77 and data.OECD_77_COUNTRIES[-1] == "ROW"
    assert len(data.OECD_45_SECTORS) == 45
    country_or_sector = [n for n in data.__all__ if n.startswith("CANONICAL_")]
    assert country_or_sector == ["CANONICAL_FINAL_DEMAND_CODES"]
    for name in ("get_sector_codes", "get_sector_names", "get_country_codes"):
        assert not hasattr(data, name), name


def test_oecd_loader_accepts_only_the_release_file_name(tmp_path):
    from puremacro.trade._oecd_icio import read_native
    from puremacro.trade.data import load_oecd_icio_granular

    renamed = tmp_path / "copy_2020_SML.csv"
    renamed.write_text("V1\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_oecd_icio_granular(2020, data_dir=tmp_path)
    with pytest.raises(ValueError, match="filename must identify"):
        read_native(renamed, 2020)
