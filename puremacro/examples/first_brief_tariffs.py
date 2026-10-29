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
from dataclasses import dataclass, field
from typing import Any

from puremacro.trade import (
    TariffScenario,
    calibrate_trade_model,
    compute_hicksian_welfare,
    run_scenario,
    solve_trade_equilibrium,
)
from puremacro.trade.data import ICIOData, load_icio_data

__all__ = [
    "SOLVER", "SYNTHETIC_LABEL", "SYNTHETIC_LABEL_ES", "DEFAULT_EXPERIMENT_SIGMA",
    "BriefSolution", "load_bundled", "validate_scenario", "solve_brief",
    "national_record", "parse_experiment",
]

#: Solver settings of every state (baseline, counterfactual, experiment).
#: ``world_income`` makes EV invariant to the order of countries in the table.
SOLVER: dict[str, Any] = {"method": "equilibrated_newton", "tol": 1e-5,
                          "foreign_saving_units": "world_income"}
SYNTHETIC_LABEL = "SYNTHETIC, method illustration only"
SYNTHETIC_LABEL_ES = "SINTÉTICO, solo ilustración del método (SYNTHETIC, method illustration only)"
DEFAULT_EXPERIMENT_SIGMA = 2.0


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
