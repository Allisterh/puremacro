"""Regression tests for the two index bugs fixed when the trade-policy modules returned.

1. ``CaliendoParroModel`` resolved a few country names through a fixed alias map
   (the first three positions of one particular country order), so a code that
   is not in the model silently became a wrong row. Codes now resolve only
   against the model's own ``country_codes``; an unknown code raises
   ``KeyError`` naming it.
2. ``TradePolicySimulator.from_icio(year)`` ignored ``year`` and returned a
   hand-made preset. It now builds the model from the ICIO table of that year,
   or raises ``ValueError`` listing the years it has.
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.trade.caliendo_parro import CaliendoParroModel
from puremacro.trade.data import synthetic_codes, synthetic_icio_table


def _three_country_model(codes):
    n, j = 3, 2
    shares = np.zeros((j, n, n))
    shares[0] = np.array([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
    shares[1] = np.eye(n)
    return CaliendoParroModel(
        trade_shares=shares,
        gamma_va=np.full((n, j), 0.5),
        gamma_io=np.full((n, j, j), 0.25),
        alpha=np.full((n, j), 0.5),
        theta=np.array([5.0, 4.0]),
        labor_income=np.full(n, 100.0),
        nontradables=[1],
        country_codes=codes,
        sector_codes=["Goods", "Services"],
    )


def test_country_index_by_code():
    model = _three_country_model(["MEX", "USA", "CAN"])
    res = model.simulate_tariff_shock("USA", "CAN", tariff_rate=0.2)
    # Only the importer collects tariff revenue: row 1 (USA), not row 0.
    assert res.tariff_revenue_prime[1] > 0.0
    assert res.tariff_revenue_prime[0] == pytest.approx(0.0, abs=1e-12)
    assert res.tariff_revenue_prime[2] == pytest.approx(0.0, abs=1e-12)
    assert model._resolve_country_index("usa") == 1
    assert model._resolve_country_index(2) == 2


@pytest.mark.parametrize("code", ["XYZ", "CHN", "DEU", "ROW"])
def test_unknown_country_code(code):
    # "CHN", "DEU" and "ROW" used to resolve through the fixed alias map.
    model = _three_country_model(["MEX", "USA", "CAN"])
    with pytest.raises(KeyError, match=code):
        model.simulate_tariff_shock(code, "USA", tariff_rate=0.1)


@pytest.mark.parametrize("name", ["manuf", "serv", "XYZ"])
def test_unknown_sector_name(name):
    # "manuf" and "serv" used to map to positions 0 and 1 whatever the sectors were.
    model = _three_country_model(["MEX", "USA", "CAN"])
    with pytest.raises(KeyError, match=name):
        model.simulate_tariff_shock("USA", "CAN", sector=name, tariff_rate=0.1)


def test_from_icio_uses_year():
    from puremacro.models.trade_policy import TradePolicySimulator

    tables = {2019: synthetic_icio_table(4, 3, seed=0), 2020: synthetic_icio_table(4, 3, seed=7)}
    sim_a = TradePolicySimulator.from_icio(2019, tables=tables)
    sim_b = TradePolicySimulator.from_icio(2020, tables=tables)
    assert sim_a.N == 4 and sim_a.J == 3
    assert sim_a.country_codes == tuple(synthetic_codes(4, 3)[0])
    assert not np.allclose(sim_a.model.trade_shares, sim_b.model.trade_shares)
    assert not np.allclose(sim_a.model.labor_income, sim_b.model.labor_income)


def test_from_icio_unavailable_year_lists_years():
    from puremacro.models.trade_policy import TradePolicySimulator

    tables = {2019: synthetic_icio_table(4, 3, seed=0), 2020: synthetic_icio_table(4, 3, seed=7)}
    with pytest.raises(ValueError, match=r"2021.*2019, 2020"):
        TradePolicySimulator.from_icio(2021, tables=tables)
    with pytest.raises(ValueError, match=r"available years: 2019\."):
        TradePolicySimulator.from_icio(2021)


def test_from_icio_defaults_to_the_bundled_2019_table():
    from puremacro.models.trade_policy import TradePolicySimulator
    from puremacro.trade.data import load_icio_data

    bundled = load_icio_data(return_structured=True)
    sim = TradePolicySimulator.from_icio(2019)
    assert sim.N == len(bundled.country_codes) and sim.J == len(bundled.sector_codes)
    assert sim.country_codes == tuple(bundled.country_codes)
    res = sim.simulate_bilateral_tariff(bundled.country_codes[0], bundled.country_codes[1], tariff_rate=0.1)
    assert res.converged


def test_from_icio_without_the_bundled_file_lists_no_years(monkeypatch, tmp_path):
    from puremacro.models.trade_policy import TradePolicySimulator
    from puremacro.trade import data as trade_data

    monkeypatch.setattr(trade_data, "_BUNDLED_ICIO_DIR", tmp_path)
    with pytest.raises(ValueError, match="available years: none"):
        TradePolicySimulator.from_icio(2019)
