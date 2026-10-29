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

import argparse
import hashlib
import json
import math
import os
import platform
import textwrap
import time
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

import puremacro
from puremacro.reports import df_to_latex, df_to_markdown
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
    "synthetic_groups", "read_groups_csv", "household_incidence", "headline_record",
    "page_block", "summary_text", "write_outputs", "run_brief", "main",
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
    started = time.perf_counter()
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
        "elapsed_seconds": time.perf_counter() - started,
        "aggregate_ev_pct_proportional": float(results["proportional"].aggregate["ev_pct"]),
        "aggregate_ev_pct_equal": float(results["equal"].aggregate["ev_pct"]),
        "label": groups.attrs.get("label", "user-supplied groups"),
        "is_synthetic": synthetic,
        "source": groups.attrs.get("source", ""),
    }
    return out


# --- record, outputs and CLI ------------------------------------------------

DEFAULT_OUTPUT = Path("output") / "first_brief_tariffs"
EXPERIMENT_DESCRIPTION = "intermediate sourcing substitution elasticity (0 = Leontief)"

_TEXT = {
    "en": {
        "component": "Component", "musd": "million USD", "pct": "% of baseline consumption",
        "rows": ("Purchaser prices", "Factor income", "Fiscal transfers", "Equivalent variation"),
        "headline_note": ("Positive values are gains. A positive EV here is a unilateral terms-of-trade "
                          "gain with no retaliation: {exporter} does not respond."),
        "groups_caption": "Household groups: {label}.",
        "group": "Group", "prop": "EV, proportional rebate (% of spending)",
        "equal": "EV, equal rebate (% of spending)", "all": "All groups (spending-weighted)",
        "groups_note": ("Proportional: the change in transfers (tariff revenue) is rebated in proportion "
                        "to baseline spending, which adds up to the national EV; equal: the same amount "
                        "per household."),
        "user_label": "user-supplied groups",
        "no_experiment": "No experiment was run (use --experiment sigma=2).",
        "exp_rows": ("Purchaser prices (million USD)", "Factor income (million USD)",
                     "Fiscal transfers (million USD)", "Equivalent variation (million USD)",
                     "Equivalent variation (% of baseline consumption)"),
        "measure": "Measure", "headline_col": "sigma = 0 (headline)", "exp_col": "sigma = {value:g}",
        "exp_note": ("sigma is the substitution elasticity between origins of intermediate inputs "
                     "(0 = Leontief, fixed sourcing); final-use origin baskets stay fixed."),
        "title": "First policy brief: {tariff:g}% tariff by {importer} on {what} from {exporter}",
        "all_goods": "all goods", "sector_goods": "sector {sector} goods",
        "table": "Table: {citation}; {unit}, {year}.",
        "panel_national": "National EV split ({importer})", "panel_groups": "Group EV",
        "axis_pct_cons": "% of baseline consumption", "axis_pct_spend": "% of group spending",
        "legend_prop": "proportional rebate", "legend_equal": "equal rebate",
        "written": "Outputs written to {path}",
    },
    "es": {
        "component": "Componente", "musd": "millones de USD", "pct": "% del consumo base",
        "rows": ("Precios de comprador", "Ingreso de los factores", "Transferencias fiscales",
                 "Variación equivalente"),
        "headline_note": ("Los valores positivos son ganancias. Una VE positiva aquí es una ganancia "
                          "unilateral de términos de intercambio, sin represalias: {exporter} no responde."),
        "groups_caption": "Grupos de hogares: {label}.",
        "group": "Grupo", "prop": "VE, reembolso proporcional (% del gasto)",
        "equal": "VE, reembolso igualitario (% del gasto)", "all": "Todos los grupos (ponderado por gasto)",
        "groups_note": ("Proporcional: el cambio en las transferencias (ingreso arancelario) se reembolsa en "
                        "proporción al gasto base, lo que suma la VE nacional; igualitario: el mismo monto "
                        "por hogar."),
        "user_label": "grupos proporcionados por el usuario",
        "no_experiment": "No se ejecutó ningún experimento (use --experiment sigma=2).",
        "exp_rows": ("Precios de comprador (millones de USD)", "Ingreso de los factores (millones de USD)",
                     "Transferencias fiscales (millones de USD)", "Variación equivalente (millones de USD)",
                     "Variación equivalente (% del consumo base)"),
        "measure": "Medida", "headline_col": "sigma = 0 (principal)", "exp_col": "sigma = {value:g}",
        "exp_note": ("sigma es la elasticidad de sustitución entre orígenes de los insumos intermedios "
                     "(0 = Leontief, abastecimiento fijo); las canastas de origen del uso final no cambian."),
        "title": "Primer informe de política: arancel de {tariff:g}% de {importer} a {what} de {exporter}",
        "all_goods": "todos los bienes", "sector_goods": "los bienes del sector {sector}",
        "table": "Tabla: {citation}; {unit}, {year}.",
        "panel_national": "Descomposición de la VE nacional ({importer})", "panel_groups": "VE por grupo",
        "axis_pct_cons": "% del consumo base", "axis_pct_spend": "% del gasto del grupo",
        "legend_prop": "reembolso proporcional", "legend_equal": "reembolso igualitario",
        "written": "Resultados escritos en {path}",
    },
}


def _lang(lang: str) -> dict[str, Any]:
    if lang not in _TEXT:
        raise ValueError(f"lang {lang!r} must be 'en' or 'es'")
    return _TEXT[lang]


def _fmt_musd(value: float, lang: str) -> str:
    text = f"{value:,.0f}"
    return text.replace(",", " ") if lang == "es" else text


def _fmt_pct(value: float, lang: str) -> str:
    text = f"{value:.4f}"
    return text.replace(".", ",") if lang == "es" else text


def _label(households: dict[str, Any], lang: str) -> str:
    if households["is_synthetic"]:
        return SYNTHETIC_LABEL_ES if lang == "es" else SYNTHETIC_LABEL
    return _lang(lang)["user_label"]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def headline_record(sol: BriefSolution, households: pd.DataFrame,
                    experiment: BriefSolution | None = None) -> dict[str, Any]:
    """The reproducible record behind every table: no timings, no versions."""
    meta = sol.icio.metadata
    record = {
        "schema": 1,
        "scenario": {k: sol.settings[k] for k in ("importer", "exporter", "sector", "tariff")},
        "solver": {**SOLVER, "sigma": float(sol.settings["sigma"])},
        "table": {"citation": meta["citation"], "unit": meta["unit"], "year": meta["year"]},
        "national": national_record(sol),
        "households": {
            "label": households.attrs["label"],
            "is_synthetic": bool(households.attrs["is_synthetic"]),
            "rows": [{"group": str(g), "ev_pct_proportional": float(row["ev_pct_proportional"]),
                      "ev_pct_equal": float(row["ev_pct_equal"])} for g, row in households.iterrows()],
            "aggregate_ev_pct_proportional": float(households.attrs["aggregate_ev_pct_proportional"]),
            "aggregate_ev_pct_equal": float(households.attrs["aggregate_ev_pct_equal"]),
        },
        "experiment": None,
    }
    if experiment is not None:
        record["experiment"] = {"name": "sigma", "value": float(experiment.settings["sigma"]),
                                "description": EXPERIMENT_DESCRIPTION,
                                "national": national_record(experiment)}
    return _jsonable(record)


def _national_frame(national: dict[str, float], lang: str) -> pd.DataFrame:
    t = _lang(lang)
    keys = ("price_effect_musd", "factor_income_effect_musd", "fiscal_transfer_effect_musd", "ev_musd")
    base = national["consumption_base_musd"]
    frame = pd.DataFrame({t["musd"]: [national[k] for k in keys],
                          t["pct"]: [100.0 * national[k] / base for k in keys]},
                         index=pd.Index(t["rows"], name=t["component"]))
    return frame


def _households_frame(households: dict[str, Any], lang: str) -> pd.DataFrame:
    t = _lang(lang)
    rows = [(r["group"], r["ev_pct_proportional"], r["ev_pct_equal"]) for r in households["rows"]]
    rows.append((t["all"], households["aggregate_ev_pct_proportional"],
                 households["aggregate_ev_pct_equal"]))
    frame = pd.DataFrame(rows, columns=[t["group"], t["prop"], t["equal"]]).set_index(t["group"])
    return frame


def _experiment_frame(record: dict[str, Any], lang: str) -> pd.DataFrame:
    t = _lang(lang)
    exp = record["experiment"]
    keys = ("price_effect_musd", "factor_income_effect_musd", "fiscal_transfer_effect_musd",
            "ev_musd", "ev_pct_consumption")
    return pd.DataFrame({t["headline_col"]: [record["national"][k] for k in keys],
                         t["exp_col"].format(value=exp["value"]): [exp["national"][k] for k in keys]},
                        index=pd.Index(t["exp_rows"], name=t["measure"]))


def _markdown_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join([":---"] + ["---:"] * (len(header) - 1)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def page_block(record: dict[str, Any], block: Literal["headline", "households", "experiment"],
               lang: Literal["en", "es"]) -> str:
    """A markdown table plus a one-sentence note, as the docs pages show it."""
    t = _lang(lang)
    if block == "headline":
        frame = _national_frame(record["national"], lang)
        rows = [(name, _fmt_musd(row.iloc[0], lang), _fmt_pct(row.iloc[1], lang))
                for name, row in frame.iterrows()]
        rows[-1] = tuple(f"**{cell}**" for cell in rows[-1])
        note = t["headline_note"].format(exporter=record["scenario"]["exporter"])
        return _markdown_table([t["component"], t["musd"], t["pct"]], rows) + "\n\n" + note
    if block == "households":
        hh = record["households"]
        frame = _households_frame(hh, lang)
        rows = [(name, _fmt_pct(row.iloc[0], lang), _fmt_pct(row.iloc[1], lang))
                for name, row in frame.iterrows()]
        caption = t["groups_caption"].format(label=_label(hh, lang))
        return (caption + "\n\n" + _markdown_table([t["group"], t["prop"], t["equal"]], rows)
                + "\n\n" + t["groups_note"])
    if block == "experiment":
        if record["experiment"] is None:
            return t["no_experiment"]
        frame = _experiment_frame(record, lang)
        fmt = [_fmt_musd] * 4 + [_fmt_pct]
        rows = [(name, f(row.iloc[0], lang), f(row.iloc[1], lang))
                for f, (name, row) in zip(fmt, frame.iterrows())]
        return _markdown_table([t["measure"], *frame.columns], rows) + "\n\n" + t["exp_note"]
    raise ValueError(f"block {block!r} must be 'headline', 'households' or 'experiment'")


def _title(record: dict[str, Any], lang: str) -> str:
    t = _lang(lang)
    sc = record["scenario"]
    what = t["all_goods"] if sc["sector"] == "all" else t["sector_goods"].format(sector=sc["sector"])
    return t["title"].format(tariff=100 * sc["tariff"], importer=sc["importer"], what=what,
                             exporter=sc["exporter"])


def summary_text(record: dict[str, Any], lang: str = "en") -> str:
    """The plain-text summary ``main`` prints."""
    t = _lang(lang)
    table = record["table"]
    parts = [_title(record, lang),
             t["table"].format(citation=table["citation"], unit=table["unit"], year=table["year"]),
             page_block(record, "headline", lang),
             page_block(record, "households", lang)]
    if record["experiment"] is not None:
        parts.append(page_block(record, "experiment", lang))
    return "\n\n".join(parts) + "\n"


def _figure(record: dict[str, Any], lang: str, path: Path) -> None:
    from matplotlib.figure import Figure

    t = _lang(lang)
    fig = Figure(figsize=(10, 4.2), dpi=100)
    left, right = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.1, 1.0]})
    national = _national_frame(record["national"], lang).iloc[:, 1]
    colors = ["#4C72B0", "#DD8452", "#55A868", "0.2"]
    left.barh(list(national.index)[::-1], national.to_numpy()[::-1], color=colors[::-1])
    left.axvline(0, color="0.1", linewidth=0.7)
    left.set_xlabel(t["axis_pct_cons"])
    left.set_title(t["panel_national"].format(importer=record["scenario"]["importer"]), fontsize=10)
    hh = record["households"]
    groups = [r["group"] for r in hh["rows"]]
    y = np.arange(len(groups))
    right.barh(y - 0.2, [r["ev_pct_proportional"] for r in hh["rows"]], height=0.38,
               color="#4C72B0", label=t["legend_prop"])
    right.barh(y + 0.2, [r["ev_pct_equal"] for r in hh["rows"]], height=0.38,
               color="#C44E52", label=t["legend_equal"])
    right.set_yticks(y, groups)
    right.invert_yaxis()
    right.axvline(0, color="0.1", linewidth=0.7)
    right.set_xlabel(t["axis_pct_spend"])
    right.set_title(f"{t['panel_groups']}\n" + textwrap.fill(_label(hh, lang), 48), fontsize=9)
    right.legend(fontsize=8, loc="best")
    fig.suptitle(_title(record, lang), fontsize=11)
    fig.tight_layout()
    fig.savefig(path, metadata={"Software": None})


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_average() -> list[float] | None:
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except (AttributeError, OSError):  # not available on Windows
        return None


def _convergence(state: Any) -> dict[str, Any]:
    meta = state.metadata
    return {"converged": bool(state.converged), "solver_termination": meta.get("solver_termination"),
            "account_residuals": meta.get("account_residuals")}


def write_outputs(sol: BriefSolution, households: pd.DataFrame, output: str | Path, *,
                  lang: str = "en", experiment: BriefSolution | None = None) -> dict[str, Any]:
    """Write tables (md/csv/tex), the figure, ``headline.json`` and ``manifest.json``."""
    t = _lang(lang)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    record = headline_record(sol, households, experiment)
    hh = record["households"]
    label = _label(hh, lang)
    citation_line = t["table"].format(**record["table"])
    files: list[str] = []

    def write(name: str, text: str) -> None:
        (output / name).write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
        files.append(name)

    national = _national_frame(record["national"], lang)
    write("national.md", f"# {_title(record, lang)}\n\n{page_block(record, 'headline', lang)}\n\n{citation_line}")
    write("national.csv", national.to_csv())
    write("national.tex", f"% {citation_line}\n" + df_to_latex(national, digits=4))

    detail = households.copy()
    detail.insert(0, "label", label)
    write("households.md", page_block(record, "households", lang))
    write("households.csv", detail.to_csv())
    write("households.tex", f"% {t['groups_caption'].format(label=label)}\n"
                            + df_to_latex(_households_frame(hh, lang), digits=4))
    if experiment is not None:
        exp = _experiment_frame(record, lang)
        write("experiment.md", page_block(record, "experiment", lang))
        write("experiment.csv", exp.to_csv())
        write("experiment.tex", f"% {t['exp_note']}\n" + df_to_latex(exp, digits=4))
    _figure(record, lang, output / "first_brief.png")
    files.append("first_brief.png")
    write("headline.json", json.dumps(record, indent=2, ensure_ascii=False))

    meta = sol.icio.metadata
    table_path = Path(meta["file_path"])
    timings = {**sol.timings, "households": float(households.attrs.get("elapsed_seconds", 0.0))}
    convergence = {"baseline": _convergence(sol.base), "counterfactual": _convergence(sol.counterfactual)}
    if experiment is not None:
        timings["experiment"] = float(sum(experiment.timings.values()))
        convergence["experiment_baseline"] = _convergence(experiment.base)
        convergence["experiment_counterfactual"] = _convergence(experiment.counterfactual)
    manifest = _jsonable({
        "package_version": puremacro.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "table": {"file": table_path.name, "citation": meta["citation"], "unit": meta["unit"],
                  "year": meta["year"], "source_md5": meta.get("source_md5"),
                  "sha256": _sha256(table_path)},
        "scenario": record["scenario"],
        "solver": record["solver"],
        "convergence": convergence,
        "timings": timings,
        "experiment": None if experiment is None else {
            "name": "sigma", "value": float(experiment.settings["sigma"]),
            "description": EXPERIMENT_DESCRIPTION, "solver": {**SOLVER, "sigma": float(experiment.settings["sigma"])}},
        "groups": {"source": households.attrs.get("source", ""), "label": households.attrs["label"],
                   "is_synthetic": bool(households.attrs["is_synthetic"])},
        "lang": lang,
        "load_average": _load_average(),
        "artifacts": {name: {"sha256": _sha256(output / name), "bytes": (output / name).stat().st_size}
                      for name in files},
    })
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
    return manifest


def run_brief(output: str | Path, *, importer: str = "USA", exporter: str = "CHN", sector: str = "all",
              tariff: float = 0.10, experiment: str | None = None, groups: str | Path | None = None,
              lang: str = "en") -> dict[str, Any]:
    """Solve everything, then write the outputs; nothing is written unless every state converged."""
    _lang(lang)
    sigma = None if experiment is None else parse_experiment(experiment)
    t0 = time.perf_counter()
    icio = load_icio_data(return_structured=True)
    t_load = time.perf_counter() - t0
    t0 = time.perf_counter()
    calib = calibrate_trade_model(icio.matrix, country_codes=icio.country_codes,
                                  sector_codes=icio.sector_codes)
    t_calib = time.perf_counter() - t0
    bundled = (icio, calib)
    validate_scenario(icio, importer, exporter, sector, tariff)
    frame = (synthetic_groups(icio, importer) if groups is None else
             read_groups_csv(groups, icio.sector_codes, supported_sectors(icio, importer)))
    sol = solve_brief(importer, exporter, sector, tariff, bundled=bundled)
    sol.timings.update(load=t_load, calibration=t_calib)
    households = household_incidence(sol, frame)
    alt = (None if sigma is None else
           solve_brief(importer, exporter, sector, tariff, sigma=sigma, bundled=bundled))
    manifest = write_outputs(sol, households, output, lang=lang, experiment=alt)
    return {"record": headline_record(sol, households, alt), "solution": sol,
            "households": households, "experiment": alt, "manifest": manifest}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m puremacro.examples.first_brief_tariffs",
        description="Your first policy brief: a tariff experiment on the bundled OECD ICIO 2019 table.")
    parser.add_argument("--importer", default="USA", help="country levying the tariff (default USA)")
    parser.add_argument("--exporter", default="CHN", help="country whose goods are tariffed (default CHN)")
    parser.add_argument("--sector", default="all", help="'all' or one ISIC section code A..T (default all)")
    parser.add_argument("--tariff", type=float, default=0.10, help="ad valorem rate, 0.10 = 10%% (default)")
    parser.add_argument("--experiment", default="none",
                        help="'none' (default) or 'sigma=<value>', e.g. sigma=2")
    parser.add_argument("--groups", type=Path, default=None,
                        help="CSV of household groups (group, population_weight, annual_spending, A..T)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"output directory (default {DEFAULT_OUTPUT})")
    parser.add_argument("--lang", choices=("en", "es"), default="en", help="language of the outputs")
    args = parser.parse_args(argv)
    try:
        sigma = parse_experiment(args.experiment)
        icio = load_icio_data(return_structured=True)
        validate_scenario(icio, args.importer, args.exporter, args.sector, args.tariff)
        if args.groups is not None:
            if not args.groups.is_file():
                raise ValueError(f"groups file {str(args.groups)!r} does not exist")
            read_groups_csv(args.groups, icio.sector_codes, supported_sectors(icio, args.importer))
    except ValueError as exc:
        parser.error(str(exc))
    study = run_brief(args.output, importer=args.importer, exporter=args.exporter, sector=args.sector,
                      tariff=args.tariff, experiment=None if sigma is None else args.experiment,
                      groups=args.groups, lang=args.lang)
    print(summary_text(study["record"], args.lang))
    print(_lang(args.lang)["written"].format(path=args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
