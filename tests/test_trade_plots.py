"""Figures of puremacro.trade.plot on a small hand-built scenario batch.

Covers plot_scenario_distributions and plot_terms_of_trade_vs_welfare: Figure
creation, axes and artists, an explicit Axes, the save_path option, the
required scenario name and the optional point labels.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from puremacro.trade._results import (
    GearyKhamisResult,
    ScenarioBatchResult,
    TradeEquilibriumResult,
)
from puremacro.trade import plot as trade_plot
from puremacro.trade.plot import plot_scenario_distributions, plot_terms_of_trade_vs_welfare


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def batch() -> ScenarioBatchResult:
    """A 2-country, 2-sector batch with a baseline and two tariff scenarios."""
    nc, ns = 2, 2
    codes = ("C01", "C02")
    s_codes = ("S01", "S02")

    def _make_eq():
        return TradeEquilibriumResult(
            x_sol=np.zeros(2 * ns * nc + 2 * nc + nc + nc - 1),
            p_sol=np.ones((1, ns, nc)),
            y_sol=np.ones((1, ns, nc)) * 50.0,
            r_sol=np.ones((1, 1, nc)),
            w_sol=np.ones((1, 1, nc)),
            T_sol=np.ones((1, 1, nc)) * 5.0,
            XN_sol=np.array([1.0]),
            terms_of_trade=np.array([1.02, 0.98]),
            country_codes=codes,
            sector_codes=s_codes,
        )

    names = ["base", "tariff_10", "tariff_25"]
    gk = {}
    for name in names:
        growth = np.array([-0.5, -0.8]) if name != "base" else np.array([0.0, 0.0])
        gk[name] = GearyKhamisResult(
            pi=np.ones(3),
            ppp=np.ones(2),
            real_gdp=np.array([100.0, 200.0]),
            nominal_gdp=np.array([100.0, 200.0]),
            gdp_growth=growth,
            country_codes=codes,
        )
    return ScenarioBatchResult(
        scenarios={name: _make_eq() for name in names},
        geary_khamis=gk,
        baseline_scenario="base",
        country_codes=codes,
        sector_codes=s_codes,
    )


def test_public_plot_functions():
    assert sorted(trade_plot.__all__) == ["plot_scenario_distributions", "plot_terms_of_trade_vs_welfare"]


def test_scenario_distributions(batch):
    fig = plot_scenario_distributions(batch, metric="real_gdp_growth")
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    # One box per non-baseline scenario.
    assert [t.get_text() for t in ax.get_xticklabels()] == [r"tariff\_10", r"tariff\_25"]


def test_scenario_distributions_on_given_axes(batch):
    fig, ax = plt.subplots()
    assert plot_scenario_distributions(batch, ax=ax) is fig


def test_terms_of_trade_vs_welfare(batch):
    fig = plot_terms_of_trade_vs_welfare(batch, scenario="tariff_10")
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    offsets = ax.collections[0].get_offsets()
    np.testing.assert_allclose(np.asarray(offsets)[:, 0], [2.0, -2.0])
    np.testing.assert_allclose(np.asarray(offsets)[:, 1], [-0.5, -0.8])
    assert len(ax.texts) == 0  # no labels unless asked


def test_terms_of_trade_vs_welfare_labels(batch):
    fig = plot_terms_of_trade_vs_welfare(batch, scenario="tariff_10", annotate=["C02", "C99"])
    assert [t.get_text() for t in fig.axes[0].texts] == ["C02"]


def test_terms_of_trade_vs_welfare_needs_scenario(batch):
    with pytest.raises(TypeError):
        plot_terms_of_trade_vs_welfare(batch)  # type: ignore[call-arg]
    with pytest.raises(KeyError, match="not found"):
        plot_terms_of_trade_vs_welfare(batch, scenario="tariff_99")


def test_save_path(batch, tmp_path):
    target = tmp_path / "figures" / "distributions.png"
    plot_scenario_distributions(batch, save_path=target)
    assert target.exists() and target.stat().st_size > 0
