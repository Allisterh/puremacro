"""Your first policy brief: a 10% US tariff on Chinese goods on the bundled OECD ICIO 2019 table, national EV with its price / factor-income / fiscal split, and synthetic household groups.

Run it with::

    python -m puremacro.examples.first_brief_tariffs --output first_brief

The brief solves the exact trade engine (``calibrate_trade_model`` +
``solve_trade_equilibrium``) twice on the bundled OECD Inter-Country
Input-Output table (2023 edition, reference year 2019, 19 G20 economies plus
ROW, 20 ISIC Rev.4 sections): once without tariffs and once with the
scenario. It reports the importer's Hicksian equivalent variation (EV) with
its endpoint split into purchaser prices, factor income and fiscal transfers
(``compute_hicksian_welfare``), maps the same GE prices and incomes to
household groups (``distributional_welfare_from_results``) and, on request,
repeats the solve under one alternative assumption (``--experiment
sigma=2``: substitution between intermediate input origins).

The default household groups are SYNTHETIC, method illustration only: three
groups built from the importer's national consumption basket by documented
tilts. Supply your own group spending shares with ``--groups CSV``.

The positive US number is a unilateral terms-of-trade gain with no
retaliation: China does not respond in this experiment.
"""
from __future__ import annotations

import math
import time
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from puremacro.trade import (
    TariffScenario,
    calibrate_trade_model,
    compute_hicksian_welfare,
    run_scenario,
    solve_trade_equilibrium,
)
from puremacro.trade.data import ICIOData, load_icio_data
from puremacro.trade.distributional import (
    distributional_welfare_from_results,
    prepare_household_groups,
)

__all__ = [
    "SOLVER", "SYNTHETIC_LABEL", "SYNTHETIC_LABEL_ES", "DEFAULT_EXPERIMENT_SIGMA",
    "BriefSolution", "load_bundled", "validate_scenario", "solve_brief",
    "national_record", "parse_experiment", "national_basket", "supported_sectors",
    "synthetic_groups", "read_groups_csv", "household_incidence",
]

#: Solver settings of every state (baseline, counterfactual, experiment).
#: ``world_income`` makes EV invariant to the order of countries in the table.
SOLVER: dict[str, Any] = {"method": "equilibrated_newton", "tol": 1e-5,
                          "foreign_saving_units": "world_income"}
SYNTHETIC_LABEL = "SYNTHETIC, method illustration only"
SYNTHETIC_LABEL_ES = "SINTÉTICO, solo ilustración del método (SYNTHETIC, method illustration only)"
DEFAULT_EXPERIMENT_SIGMA = 2.0

#: Synthetic groups: mean spending (USD per household per year) and the tilt
#: ``d`` toward agriculture (A) and manufactures incl. food (C). The high
#: group's tilt is set so the spending-weighted mean basket is the national one.
SYNTHETIC_SPENDING = {"low": 30_000.0, "middle": 60_000.0, "high": 120_000.0}
SYNTHETIC_TILT = {"low": 0.30, "middle": 0.0, "high": -0.30 * 30_000.0 / 120_000.0}
GOODS_SECTORS = ("A", "C")
_GROUP_COLUMNS = ("population_weight", "annual_spending")


@dataclass
class BriefSolution:
    """Everything one brief solve produced; reused by the household and output stages."""

    icio: ICIOData
    calib: Any
    base: Any
    counterfactual: Any
    welfare: Any
    scenario: TariffScenario
    settings: dict[str, Any]
    timings: dict[str, float] = field(default_factory=dict)


def load_bundled() -> tuple[ICIOData, Any]:
    """The bundled OECD ICIO 2019 table and its calibration."""
    icio = load_icio_data(return_structured=True)
    calib = calibrate_trade_model(icio.matrix, country_codes=icio.country_codes,
                                  sector_codes=icio.sector_codes)
    return icio, calib


def validate_scenario(icio: ICIOData, importer: str, exporter: str, sector: str,
                      tariff: float) -> None:
    """Raise ``ValueError`` naming the offending value of a scenario, before any solve."""
    countries = list(icio.country_codes)
    if importer not in countries:
        raise ValueError(f"unknown importer {importer!r}; known codes: {', '.join(countries)}")
    if exporter not in countries:
        raise ValueError(f"unknown exporter {exporter!r}; known codes: {', '.join(countries)}")
    if exporter == importer:
        raise ValueError(f"exporter {exporter!r} equals the importer; domestic flows are never tariffed")
    if sector != "all" and sector not in icio.sector_codes:
        raise ValueError(f"unknown sector {sector!r}; use 'all' or one of {', '.join(icio.sector_codes)}")
    try:
        value = float(tariff)
    except (TypeError, ValueError):
        raise ValueError(f"tariff {tariff!r} is not a number") from None
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"tariff {value!r} must be a finite non-negative rate (0.10 = 10%)")


def parse_experiment(text: str) -> float | None:
    """``"none"`` -> None; ``"sigma=<float>"`` -> that positive finite sigma."""
    text = str(text).strip()
    if text.lower() == "none":
        return None
    name, sep, value = text.partition("=")
    if name.strip() != "sigma" or not sep:
        raise ValueError(f"experiment {text!r} must be 'none' or 'sigma=<value>' "
                         "(the exact engine has no trade elasticity theta)")
    try:
        sigma = float(value)
    except ValueError:
        raise ValueError(f"experiment {text!r}: sigma {value!r} is not a number") from None
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"experiment {text!r}: sigma must be finite and positive "
                         "(sigma=0 is the headline, Leontief sourcing)")
    return sigma


def _require_converged(state: Any, name: str) -> None:
    if not getattr(state, "converged", False):
        termination = getattr(state, "metadata", {}).get("solver_termination")
        raise RuntimeError(f"{name} did not converge (solver_termination={termination!r})")


def solve_brief(importer: str = "USA", exporter: str = "CHN", sector: str = "all",
                tariff: float = 0.10, *, sigma: float = 0.0,
                bundled: tuple[ICIOData, Any] | None = None) -> BriefSolution:
    """Solve baseline and counterfactual on the bundled table and the importer's EV."""
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    if bundled is None:
        icio = load_icio_data(return_structured=True)
        timings["load"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        calib = calibrate_trade_model(icio.matrix, country_codes=icio.country_codes,
                                      sector_codes=icio.sector_codes)
        timings["calibration"] = time.perf_counter() - t0
    else:
        icio, calib = bundled
        timings["load"] = 0.0
        timings["calibration"] = 0.0
    validate_scenario(icio, importer, exporter, sector, tariff)
    scenario = TariffScenario(importer, {(exporter, "*" if sector == "all" else sector): float(tariff)},
                              name=f"{importer}_tariff_on_{exporter}_{sector}")
    t0 = time.perf_counter()
    base = solve_trade_equilibrium(calib, sigma=sigma, **SOLVER)
    timings["baseline"] = time.perf_counter() - t0
    _require_converged(base, "baseline")
    t0 = time.perf_counter()
    counterfactual = run_scenario(calib, scenario, base_result=base, sigma=sigma, **SOLVER)
    timings["counterfactual"] = time.perf_counter() - t0
    _require_converged(counterfactual, "counterfactual")
    t0 = time.perf_counter()
    welfare = compute_hicksian_welfare(calib, counterfactual, base_result=base,
                                       target_country=importer)
    timings["welfare"] = time.perf_counter() - t0
    settings = {"importer": importer, "exporter": exporter, "sector": sector,
                "tariff": float(tariff), "sigma": float(sigma), **SOLVER}
    return BriefSolution(icio, calib, base, counterfactual, welfare, scenario, settings, timings)


def national_record(sol: BriefSolution) -> dict[str, float]:
    """The importer's EV and its split, in million USD (the table's unit) and percent."""
    w = sol.welfare
    return {
        "ev_musd": float(w.ev),
        "cv_musd": float(w.cv),
        "ev_pct_consumption": float(w.ev_pct_consumption),
        "ev_pct_gdp": float(w.ev_pct_gdp),
        "consumption_base_musd": float(w.consumption_base),
        "price_effect_musd": float(w.price_effect),
        "factor_income_effect_musd": float(w.factor_income_effect),
        "fiscal_transfer_effect_musd": float(w.fiscal_transfer_effect),
        "tariff_rebate_effect_musd": float(w.tariff_rebate_effect),
        "decomposition_residual_musd": float(w.decomposition_residual),
    }


# --- household groups -------------------------------------------------------

def _consumption_by_sector(icio: ICIOData, importer: str) -> np.ndarray:
    countries, sectors, fds = icio.country_codes, icio.sector_codes, icio.fd_codes
    if importer not in countries:
        raise ValueError(f"unknown importer {importer!r}")
    nc, ns, nfd = len(countries), len(sectors), len(fds)
    n = nc * ns
    column = n + countries.index(importer) * nfd + fds.index("C")
    return np.asarray(icio.matrix[:n, column], dtype=float).reshape(nc, ns).sum(axis=0)


def national_basket(icio: ICIOData, importer: str) -> pd.Series:
    """The importer's final consumption (category C) by sector over all origins, as shares."""
    spend = _consumption_by_sector(icio, importer)
    return pd.Series(spend / spend.sum(), index=list(icio.sector_codes), name="share")


def supported_sectors(icio: ICIOData, importer: str) -> list[str]:
    """Sectors the importer's consumption basket buys (positive C purchases)."""
    spend = _consumption_by_sector(icio, importer)
    return [code for code, value in zip(icio.sector_codes, spend) if value > 0]


def synthetic_groups(icio: ICIOData, importer: str) -> pd.DataFrame:
    """Three SYNTHETIC household groups built from the national basket by documented tilts.

    ``s_g = s * (1 + d_g z)`` with ``s`` the national basket and
    ``z = 1[sector in {A, C}] - sum_{A,C} s``, so every row sums to one and the
    spending-weighted mean basket is ``s``. Low spenders tilt toward
    agriculture and manufactures (including food), high spenders toward
    services. Population weights are 1/3 each (a population of size one).
    """
    s = national_basket(icio, importer)
    goods = s.index.isin(GOODS_SECTORS).astype(float)
    z = goods - float(s[s.index.isin(GOODS_SECTORS)].sum())
    rows = {}
    for group, spending in SYNTHETIC_SPENDING.items():
        shares = s.to_numpy() * (1.0 + SYNTHETIC_TILT[group] * z)
        rows[group] = {"population_weight": 1.0 / 3.0, "annual_spending": spending,
                       **dict(zip(s.index, shares))}
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "group"
    frame.attrs = {"source": "synthetic groups from the bundled national basket "
                             "(puremacro.examples.first_brief_tariffs)",
                   "is_synthetic": True, "label": SYNTHETIC_LABEL}
    return frame


def read_groups_csv(path: str | Path, sector_codes: Sequence[str],
                    supported: Collection[str]) -> pd.DataFrame:
    """Read user household groups: ``group, population_weight, annual_spending`` and one
    budget-share column per sector code (each >= 0, row sum 1 within 1e-6), nothing else."""
    path = Path(path)
    raw = pd.read_csv(path, dtype={"group": str})
    sectors = [str(c) for c in sector_codes]
    expected = ["group", *_GROUP_COLUMNS, *sectors]
    missing = [c for c in expected if c not in raw.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    extra = [str(c) for c in raw.columns if c not in expected]
    if extra:
        raise ValueError(f"{path.name}: unknown columns {extra}; allowed are "
                         "group, population_weight, annual_spending and the sector codes")
    if raw.empty:
        raise ValueError(f"{path.name}: no groups")
    groups = raw["group"]
    if groups.isna().any() or (groups.str.strip() == "").any():
        raise ValueError(f"{path.name}: every row needs a nonempty 'group' label")
    duplicated = sorted(set(groups[groups.duplicated()]))
    if duplicated:
        raise ValueError(f"{path.name}: duplicate group {duplicated[0]!r}")
    frame = raw.set_index("group")[[*_GROUP_COLUMNS, *sectors]]
    try:
        frame = frame.astype(float)
    except (TypeError, ValueError):
        raise ValueError(f"{path.name}: every value except 'group' must be numeric") from None
    allowed = set(supported)
    for group, row in frame.iterrows():
        if not np.isfinite(row.to_numpy()).all():
            bad = [c for c in frame.columns if not np.isfinite(row[c])]
            raise ValueError(f"{path.name}: group {group!r} has non-finite {bad}")
        for column in _GROUP_COLUMNS:
            if row[column] <= 0:
                raise ValueError(f"{path.name}: group {group!r} needs positive {column}, got {row[column]!r}")
        negative = [s for s in sectors if row[s] < 0]
        if negative:
            raise ValueError(f"{path.name}: group {group!r} has negative shares in sectors {negative}")
        total = float(row[sectors].sum())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"{path.name}: group {group!r} shares sum to {total:.6g}, not 1")
        unsupported = [s for s in sectors if row[s] > 0 and s not in allowed]
        if unsupported:
            raise ValueError(f"{path.name}: group {group!r} spends on sectors {unsupported} that the "
                             "importer's consumption basket does not buy in the table")
    frame.index.name = "group"
    frame.attrs = {"source": path.name, "is_synthetic": False, "label": "user-supplied groups"}
    return frame


def _importer_incomes(state: Any, calib: Any, index: int) -> tuple[float, float]:
    factor = (float(np.ravel(state.w_sol)[index]) * float(np.ravel(calib.l_endow)[index])
              + float(np.ravel(state.r_sol)[index]) * float(np.ravel(calib.k_endow)[index]))
    return factor, float(np.ravel(state.T_sol)[index])


def household_incidence(sol: BriefSolution, groups: pd.DataFrame) -> pd.DataFrame:
    """Group EV (% of group spending) from the brief's GE prices and incomes.

    Each group's budget is split into factor income and transfers in the
    importer's national proportions ``(wL+rK)/(wL+rK+T)``; factor income moves
    with the national ``wL+rK`` (w and r of a country move together in this
    engine) and the change in national transfers ``T`` is rebated either in
    proportion to baseline spending (reproduces the national EV) or equally
    per household.
    """
    importer = sol.settings["importer"]
    sectors = list(sol.icio.sector_codes)
    index = list(sol.calib.country_codes).index(importer)
    f0, t0 = _importer_incomes(sol.base, sol.calib, index)
    f1, t1 = _importer_incomes(sol.counterfactual, sol.calib, index)
    weights = groups["population_weight"].astype(float)
    spending = groups["annual_spending"].astype(float)
    expenditure = groups[sectors].astype(float).mul(spending, axis=0)
    synthetic = bool(groups.attrs.get("is_synthetic", False))
    exposure = pd.DataFrame({"factor_income": spending * f0 / (f0 + t0)})
    prepared = prepare_household_groups(
        expenditure, weights, monetary_unit="USD" if synthetic else "currency units",
        period="year", income_exposure=exposure, provenance=dict(groups.attrs))
    pool = (t1 - t0) / (f0 + t0) * float((weights * spending).sum())
    changes = pd.Series({"factor_income": f1 / f0 - 1.0})
    results = {}
    for rule, scores in (("proportional", spending), ("equal", None)):
        results[rule] = distributional_welfare_from_results(
            prepared, sol.calib, sol.base, sol.counterfactual, country=importer,
            factor_income_changes=changes, transfer_total=pool, transfer_weights=scores,
            rule="fixed_baskets")
    prop, equal = results["proportional"].groups, results["equal"].groups
    budget = prop["baseline_budget"]
    out = pd.DataFrame({
        "population_weight": weights.to_numpy(),
        "annual_spending": spending.to_numpy(),
        "ev_pct_proportional": prop["ev_pct"].to_numpy(),
        "ev_pct_equal": equal["ev_pct"].to_numpy(),
        "price_ev_pct": (100.0 * prop["price_ev"] / budget).to_numpy(),
        "income_ev_pct": (100.0 * prop["income_ev"] / budget).to_numpy(),
    }, index=groups.index.copy())
    out.index.name = "group"
    out.attrs = {
        "aggregate_ev_pct_proportional": float(results["proportional"].aggregate["ev_pct"]),
        "aggregate_ev_pct_equal": float(results["equal"].aggregate["ev_pct"]),
        "label": groups.attrs.get("label", "user-supplied groups"),
        "is_synthetic": synthetic,
        "source": groups.attrs.get("source", ""),
    }
    return out
