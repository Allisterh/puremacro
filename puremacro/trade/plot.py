"""Pyodide-compliant Matplotlib Visualization Engine for puremacro.trade.

Provides standalone figure generation routines for cross-country scenario
distributions and terms-of-trade vs. welfare scatter plots of a
:class:`~puremacro.trade.ScenarioBatchResult`.

Conforms strictly to the puremacro Pyodide runtime contract:
- Exclusively pure NumPy, Pandas, and Matplotlib.
- Zero non-standard imports (no seaborn, no plotly, no bokeh).
- Estimator and result dataclasses remain strictly immutable without attached .plot() methods.
- All functions accept an optional `ax` for grid composition and return `matplotlib.figure.Figure`.
- 100% compliant with modern matplotlib (3.9+ and 3.11).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

import numpy as np
import pandas as pd

from puremacro.trade._results import ScenarioBatchResult


_GRAYS = ["0.15", "0.40", "0.60", "0.25", "0.75", "0.50", "0.30", "0.70"]
_MARKERS = ["o", "s", "^", "v", "D", "P", "X", "*"]


def _palette(n: int) -> list[str]:
    """Return an n-element grayscale / high-contrast palette."""
    if n <= len(_GRAYS):
        return _GRAYS[:n]
    return [f"{g:.3f}" for g in np.linspace(0.1, 0.8, n)]


def _resolve_ax(
    ax: plt.Axes | None, figsize: tuple[float, float] = (7.0, 4.5)
) -> tuple[Figure, plt.Axes]:
    """Helper to resolve or create a Figure and Axes."""
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    if ax is None:
        fig, new_ax = plt.subplots(figsize=figsize)
        return fig, new_ax
    f = ax.figure
    if not isinstance(f, Figure):
        raise TypeError(f"Expected matplotlib.figure.Figure, got {type(f)}")
    return f, ax


# ---------------------------------------------------------------------------
# 1. Cross-Country Scenario Distributions Plot
# ---------------------------------------------------------------------------

def plot_scenario_distributions(
    batch_res: ScenarioBatchResult,
    metric: str = "real_gdp_growth",
    save_path: str | Path | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (7.5, 4.5),
) -> Figure:
    """Plot cross-country distribution box plots across tariff counterfactual scenarios.

    Parameters
    ----------
    batch_res : ScenarioBatchResult
        Container with solved trade equilibrium results.
    metric : str, default 'real_gdp_growth'
        Metric to distribute across countries ('real_gdp_growth', 'inflation',
        'cpi', 'xn_over_gdp', 'trade_balance', 'terms_of_trade').
    save_path : str | Path | None, default None
        Optional file path to save figure.
    ax : plt.Axes | None, default None
        Optional matplotlib Axes.
    figsize : tuple[float, float], default (7.5, 4.5)
        Figure dimensions in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The created Figure object.
    """
    fig, ax = _resolve_ax(ax, figsize=figsize)

    df = batch_res.to_frame(metric)
    scenarios = [s for s in df.columns if s != batch_res.baseline_scenario]
    if not scenarios:
        scenarios = list(df.columns)

    data_list = [df[s].dropna().values for s in scenarios]
    clean_labels = [s.replace("_", r"\_") for s in scenarios]

    ax.boxplot(
        data_list,
        patch_artist=True,
        boxprops=dict(facecolor="0.85", color="black"),
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(color="black", linestyle="--"),
        capprops=dict(color="black"),
        flierprops=dict(marker="o", markerfacecolor="0.5", markersize=3, alpha=0.6),
    )

    # Universally safe tick setting across all matplotlib versions
    ax.set_xticks(list(range(1, len(clean_labels) + 1)))
    ax.set_xticklabels(clean_labels)

    # Overlay sample means with diamond markers
    means = [float(np.mean(d)) for d in data_list]
    ax.plot(
        list(range(1, len(clean_labels) + 1)),
        means,
        marker="D",
        color="black",
        linestyle="none",
        label="Cross-Country Mean",
    )

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Tariff Scenario", fontweight="bold")
    ax.set_ylabel(metric.replace("_", " ").title() + " (%)", fontweight="bold")
    ax.set_title(
        f"Cross-Country Distribution: {metric.replace('_', ' ').title()}",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax.legend(frameon=True, fontsize=8)

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight", dpi=300)

    return fig


# ---------------------------------------------------------------------------
# 2. Terms of Trade vs. Welfare Scatter Plot
# ---------------------------------------------------------------------------

def plot_terms_of_trade_vs_welfare(
    batch_res: ScenarioBatchResult,
    scenario: str,
    save_path: str | Path | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (8.0, 5.0),
    annotate: Sequence[str] | None = None,
) -> Figure:
    """Plot Terms of Trade change vs. Real GDP growth scatter plot for a given scenario.

    Parameters
    ----------
    batch_res : ScenarioBatchResult
        Container with solved trade equilibrium results.
    scenario : str
        Name of the scenario to plot (a key of ``batch_res.scenarios``).
    save_path : str | Path | None, default None
        Optional file path to save figure.
    ax : plt.Axes | None, default None
        Optional matplotlib Axes.
    figsize : tuple[float, float], default (8.0, 5.0)
        Figure dimensions in inches.
    annotate : Sequence[str] | None, default None
        Country codes to label next to their points; codes not in the batch
        are ignored. None labels no point.

    Returns
    -------
    matplotlib.figure.Figure
        The created Figure object.
    """
    fig, ax = _resolve_ax(ax, figsize=figsize)

    if scenario not in batch_res.scenarios:
        raise KeyError(
            f"Scenario '{scenario}' not found in batch results. "
            f"Available: {list(batch_res.scenarios.keys())}"
        )

    eq_res = batch_res[scenario]
    gdp_growth = np.asarray(batch_res.real_gdp_table[scenario], dtype=float)
    all_codes = list(batch_res.country_codes) if batch_res.country_codes else list(batch_res.real_gdp_table.index)

    # Extract or approximate terms of trade change
    if eq_res.terms_of_trade is not None:
        tot_pct = (np.asarray(eq_res.terms_of_trade, dtype=float) - 1.0) * 100.0
    elif eq_res.w_sol is not None:
        # GE wage relative to world numeraire proxy for terms of trade
        w_curr = eq_res.w_sol.ravel()
        base_res = batch_res.scenarios.get(batch_res.baseline_scenario)
        w_base = base_res.w_sol.ravel() if base_res and base_res.w_sol is not None else np.ones_like(w_curr)
        tot_pct = (w_curr / w_base - 1.0) * 100.0
    else:
        tot_pct = np.zeros_like(gdp_growth)

    # Scatter points
    ax.scatter(
        tot_pct,
        gdp_growth,
        color="0.3",
        alpha=0.7,
        edgecolors="black",
        s=40,
        label="Economies",
    )

    # OLS trend line
    if len(tot_pct) > 2 and not np.allclose(tot_pct, tot_pct[0]):
        poly = np.polyfit(tot_pct, gdp_growth, 1)
        x_line = np.linspace(float(tot_pct.min()), float(tot_pct.max()), 50)
        ax.plot(
            x_line,
            poly[0] * x_line + poly[1],
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=f"Linear Fit (slope={poly[0]:.2f})",
        )

    # Annotate the requested economies
    for code in (annotate or ()):
        if code in all_codes:
            idx = all_codes.index(code)
            ax.annotate(
                code,
                (tot_pct[idx], gdp_growth[idx]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
            )

    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Terms of Trade Change (%)", fontweight="bold")
    ax.set_ylabel("Real GDP Growth (%) [Geary-Khamis]", fontweight="bold")
    # Matplotlib renders titles as plain text (no TeX), so the scenario name is
    # used verbatim. Kept outside the f-string: a backslash inside a replacement
    # field is a SyntaxError before Python 3.12 (PEP 701).
    scenario_label = str(scenario)
    ax.set_title(
        f"Terms of Trade vs. Welfare Impact ({scenario_label})",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(frameon=True, fontsize=8)

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight", dpi=300)

    return fig


__all__ = [
    "plot_scenario_distributions",
    "plot_terms_of_trade_vs_welfare",
]
