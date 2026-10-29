"""Sovereign-risk policy brief: reentry, income volatility and persistence.

EN: An illustrative policy brief on PureMacro's public calibration of the
short-term Arellano (2008) model. It varies one parameter at a time (the
market-reentry probability, income volatility and income persistence) and
reports sustainable debt, spreads, default exposure and the value of market
access, each with its 95% interval and its numerical grid change. It is not a
claim about any country and fits no parameter to targets.

ES: Un informe de política ilustrativo sobre la calibración pública de
PureMacro del modelo de Arellano (2008) con deuda de corto plazo. Varía un
parámetro cada vez (la probabilidad de reingreso al mercado, la volatilidad y
la persistencia del ingreso) e informa la deuda sostenible, los diferenciales,
la exposición al impago y el valor del acceso al mercado, cada uno con su
intervalo del 95% y su cambio numérico de malla. No es una afirmación sobre
ningún país y no ajusta ningún parámetro a objetivos.

The page is docs/brief_sovereign_risk.md (ES: docs/es/brief_sovereign_risk.md).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

from puremacro.sovereign import SovereignDefaultResult, solve_sovereign_default
from puremacro.sovereign.measurement import MOMENTS, ergodic_moments, value_of_market_access

# PureMacro's public calibration (tests/test_arellano_2008_study.py::_public_monotone).
BASELINE: dict[str, Any] = dict(
    beta=.95282, r_star=.017, theta=.282, rho_y=.945, sigma_y=.025,
    b_min=-.15, b_max=.33, sigma_eps=0, y_penalty_type="arellano",
    enforce_monotonicity=False, tol=1e-10, max_iter=10000, policy_search="monotone",
)
GRIDS: dict[str, tuple[int, int]] = {"converged": (6401, 641), "error": (3201, 641), "coarse": (801, 161)}
DESIGN: dict[str, Any] = {"periods": 500_000, "seeds": list(range(1979, 1995)), "burn": 1_000}
# The reference study's registered grid tolerances for these moments (pct / pp).
TOLERANCES: dict[str, float] = {m: 0.10 for m in MOMENTS}

_SCENARIO_KEYS = ("theta", "sigma_y", "rho_y")
_SCENARIO_VALUES = {"theta": (.15, .50), "sigma_y": (.020, .030), "rho_y": (.90, .97)}
SCENARIOS: dict[str, dict[str, float]] = {"baseline": {}}
for _key in _SCENARIO_KEYS:
    for _value in _SCENARIO_VALUES[_key]:
        SCENARIOS[f"{_key}={_value:g}"] = {_key: _value}

_SOURCES = ("sovereign/arellano.py", "sovereign/_monotone.py", "sovereign/_results.py",
            "sovereign/measurement.py", "examples/first_brief_sovereign.py")


def parse_scenario(text: str) -> tuple[str, dict[str, float]]:
    """``"baseline"`` or ``key=value`` with key in theta, sigma_y, rho_y -> (canonical name, overrides)."""
    text = text.strip()
    if text == "baseline":
        return "baseline", {}
    key, sep, raw = text.partition("=")
    key = key.strip()
    if not sep or key not in _SCENARIO_KEYS:
        raise ValueError(f"scenario must be 'baseline' or key=value with key one of "
                         f"{', '.join(_SCENARIO_KEYS)}; got {text!r}")
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"scenario value must be a number; got {raw!r}") from None
    if not math.isfinite(value):
        raise ValueError(f"scenario value must be finite; got {raw!r}")
    return f"{key}={value:g}", {key: value}


def solve_scenario(overrides: Mapping[str, float], n_b: int, n_y: int) -> tuple[SovereignDefaultResult, float]:
    """Solve the baseline with ``overrides`` on an ``n_b x n_y`` grid; returns (result, seconds)."""
    start = time.perf_counter()
    result = solve_sovereign_default(**{**BASELINE, **overrides}, n_b=int(n_b), n_y=int(n_y))
    return result, time.perf_counter() - start


def _solver_parameters(result: SovereignDefaultResult) -> dict[str, Any]:
    """Every keyword of ``solve_sovereign_default`` as used for ``result``."""
    md = result.metadata
    params = {
        "beta": md["beta"], "gamma": md["gamma"], "r_star": md["r_star"],
        "delta": result.maturity_param_delta, "kappa": result.coupon_rate_kappa,
        "sigma_eps": result.taste_shock_scale, "rho_y": md["rho_y"], "sigma_y": md["sigma_y"],
        "theta": md["theta"], "d0": md["d0"], "d1": md["d1"],
        "n_b": len(result.b_grid), "b_max": result.b_grid[-1], "b_min": result.b_grid[0],
        "n_y": len(result.y_grid), "m_tauchen": md["m_tauchen"], "tol": md["tol"],
        "max_iter": md["max_iter"], "damping": md["damping"], "recovery_rate": md["recovery_rate"],
        "howard_steps": md["howard_steps"], "enforce_monotonicity": md["enforce_monotonicity"],
        "y_penalty_type": md["y_penalty_type"], "policy_search": md["policy_search"],
    }
    return {k: _plain(v) for k, v in params.items()}


def _plain(value: Any) -> Any:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return value


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def measure_case(result: SovereignDefaultResult, elapsed: float, *, periods: int,
                 seeds: Sequence[int], burn: int) -> dict:
    """Solver diagnostics, ergodic moments (per seed and summary) and welfare, JSON-ready."""
    seeds = [int(s) for s in seeds]
    table = ergodic_moments(result, periods=periods, seeds=seeds, burn=burn)
    md = result.metadata
    return {
        "n_b": len(result.b_grid),
        "n_y": len(result.y_grid),
        "parameters": _solver_parameters(result),
        "solver": {
            "converged": bool(result.converged),
            "iterations": int(result.iterations),
            "residual": float(result.residual),
            "monotone_certification_gap": _optional_float(md.get("monotone_certification_gap")),
            "exhaustive_bellman_residual": _optional_float(md.get("exhaustive_bellman_residual")),
            "elapsed_s": float(elapsed),
        },
        "simulation": {
            "seeds": seeds,
            "per_seed": {m: [float(table.loc[m, f"seed_{s}"]) for s in seeds] for m in MOMENTS},
            "summary": {m: {"mean": float(table.loc[m, "mean"]), "se": float(table.loc[m, "se"]),
                            "ci95": [float(table.loc[m, "ci95_low"]), float(table.loc[m, "ci95_high"])]}
                        for m in MOMENTS},
        },
        "welfare_pct": float(value_of_market_access(result)),
    }


def paired_change(coarse: dict, fine: dict) -> dict:
    """The reference study's paired grid-change rule, fine minus coarse, per seed."""
    seeds_c, seeds_f = coarse["simulation"]["seeds"], fine["simulation"]["seeds"]
    if list(seeds_c) != list(seeds_f) or len(seeds_c) < 2 or len(set(seeds_c)) != len(seeds_c):
        raise ValueError(f"paired changes need the same distinct seeds (>= 2); got {seeds_c} and {seeds_f}")
    n = len(seeds_c)
    t_crit = float(student_t.ppf(.975, n - 1))
    moments = {}
    for m in MOMENTS:
        d = np.asarray(fine["simulation"]["per_seed"][m], float) - np.asarray(coarse["simulation"]["per_seed"][m], float)
        if not np.all(np.isfinite(d)):
            raise ValueError(f"nonfinite moment: {m}")
        mean = float(d.mean())
        half = float(t_crit * d.std(ddof=1) / np.sqrt(n))
        bound = abs(mean) + half
        tol = TOLERANCES[m]
        status = ("passed" if bound <= tol else "grid_shift_exceeds_tolerance"
                  if abs(mean) > tol else "sampling_precision_insufficient")
        moments[m] = {"change": mean, "ci95": [mean - half, mean + half],
                      "absolute_change_bound": bound, "tolerance": tol, "status": status}
    return {"passed": all(v["status"] == "passed" for v in moments.values()), "moments": moments,
            "welfare_change_pct": float(fine["welfare_pct"]) - float(coarse["welfare_pct"])}


def source_hashes() -> dict[str, str]:
    """sha256 of the solver, measurement and this example, keyed by package-relative path."""
    root = Path(__file__).resolve().parents[1]
    return {f"puremacro/{rel}": hashlib.sha256((root / rel).read_bytes()).hexdigest() for rel in _SOURCES}


def _peak_rss_gb() -> float | None:
    try:
        import resource
    except ImportError:  # Windows
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1.0 if sys.platform == "darwin" else 1024.0  # bytes on macOS, KiB on Linux
    return round(peak * scale / 1e9, 3)


def _versions() -> dict[str, str]:
    import pandas
    import scipy

    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "pandas": pandas.__version__}


def _package_version() -> str:
    import puremacro

    return str(getattr(puremacro, "__version__", "unknown"))


def _write_atomic(path: Path, record: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=1, allow_nan=False) + "\n")
    os.replace(tmp, path)


def _case_key(n_b: int, n_y: int) -> str:
    return f"{n_b}x{n_y}"


def build_record(grids: Sequence[tuple[int, int]], *, design: dict, partial: Path | None = None,
                 log: Callable[[str], Any] = print, scenarios: Mapping[str, Mapping[str, float]] | None = None) -> dict:
    """Solve and measure every scenario on every grid (``grids[0]`` is the fine grid).

    After each case the record so far is written to ``partial`` (atomic replace).
    A run restarted with the same ``partial`` reuses its finished cases only if
    their ``source_sha256`` and ``design`` match the current ones; otherwise the
    file is discarded and every case is solved again.
    """
    grids = [(int(n_b), int(n_y)) for n_b, n_y in grids]
    scenarios = SCENARIOS if scenarios is None else scenarios
    hashes = source_hashes()
    design = json.loads(json.dumps(design))
    finished: dict[str, dict] = {}
    old_loads: dict[str, list[float]] = {}
    if partial is not None and Path(partial).exists():
        partial = Path(partial)
        try:
            old = json.loads(partial.read_text())
        except (OSError, ValueError):
            old = {}
        if old.get("source_sha256") == hashes and old.get("design") == design:
            finished = {name: s.get("cases", {}) for name, s in old.get("scenarios", {}).items()}
            old_loads = old.get("load_averages", {})
            log(f"Resuming {sum(map(len, finished.values()))} finished cases from {partial.name}")
        else:
            log(f"Discarding {partial.name}: its source hashes or design differ from the current run")
    record: dict[str, Any] = {
        "schema": 1,
        "package_version": _package_version(),
        "versions": _versions(),
        "source_sha256": hashes,
        "design": design,
        "baseline": dict(BASELINE),
        "tolerances": dict(TOLERANCES),
        "grids": [list(g) for g in grids],
        "peak_rss_gb": _peak_rss_gb(),
        "load_averages": dict(old_loads),
        "scenarios": {},
    }
    for name, overrides in scenarios.items():
        entry: dict[str, Any] = {"overrides": dict(overrides), "cases": {}}
        record["scenarios"][name] = entry
        for n_b, n_y in grids:
            key = _case_key(n_b, n_y)
            done = finished.get(name, {}).get(key)
            if done is not None:
                entry["cases"][key] = done
                log(f"{name} {key}: reused from partial")
                continue
            result, elapsed = solve_scenario(overrides, n_b, n_y)
            start = time.perf_counter()
            case = measure_case(result, elapsed, periods=design["periods"], seeds=design["seeds"], burn=design["burn"])
            del result
            summary = case["simulation"]["summary"]
            log(f"{name} {key}: solved in {elapsed:.1f} s, measured in {time.perf_counter() - start:.1f} s, "
                f"gap {case['solver']['monotone_certification_gap']}, "
                + ", ".join(f"{m} {summary[m]['mean']:.4f}" for m in MOMENTS)
                + f", welfare {case['welfare_pct']:.4f}%")
            entry["cases"][key] = case
            try:
                record["load_averages"][f"{name} {key}"] = [round(v, 2) for v in os.getloadavg()]
            except (AttributeError, OSError):  # Windows
                pass
            record["peak_rss_gb"] = _peak_rss_gb()
            if partial is not None:
                _write_atomic(Path(partial), record)
        if len(grids) >= 2:
            entry["grid_change"] = paired_change(entry["cases"][_case_key(*grids[1])],
                                                 entry["cases"][_case_key(*grids[0])])
    record["peak_rss_gb"] = _peak_rss_gb()
    return record


# --- Renderers ---------------------------------------------------------------

_TEXT: dict[str, dict[str, Any]] = {
    "en": {
        "columns": ["Scenario", "Debt/output (%)", "Mean spread (%)", "Default (% per year)",
                    "Spread SD (pp)", "Value of market access (%)"],
        "baseline": "Baseline",
        "param": {"theta": "θ", "sigma_y": "σ_y", "rho_y": "ρ_y"},
        "param_long": {"theta": "reentry probability", "sigma_y": "income volatility",
                       "rho_y": "income persistence"},
        "outcome": {"debt_output_pct": "debt/output", "spread_mean_pct": "the mean spread",
                    "default_annual_pct": "default frequency", "spread_sd_pp": "spread volatility",
                    "welfare_pct": "the value of market access"},
        "unit": {"debt_output_pct": "%", "spread_mean_pct": "%", "default_annual_pct": "% a year",
                 "spread_sd_pp": " pp", "welfare_pct": "% of consumption"},
        "up": "raises", "down": "lowers", "flat": "leaves unchanged",
        "from_to": "from {a} to {b}",
        "lead": ("In this illustrative model, starting from the public calibration and moving one "
                 "parameter at a time from its low to its high value: "),
        "clause": "a higher {name} ({sym} {lo} → {hi}) {items}",
        "tail": ("Each number is a mean over {n} simulated paths of {periods} quarters; the table "
                 "gives its 95% interval and its grid change."),
        "footnote": ("† At least one moment's paired {coarse} → {fine} grid change does not fit the "
                     "registered 0.10 tolerance (its status is in the record). The row is reported, "
                     "not dropped."),
        "coarse_caveat": ("Coarse grid (801 x 161): for orientation only; numbers differ from the "
                          "converged brief (6,401 x 641) in docs/brief_sovereign_risk.md."),
        "panel": ["Debt/output (%)", "Mean spread (%)", "Default (% per year)", "Spread SD (pp)",
                  "Value of market access (%)"],
        "groups": ["Reentry θ", "Volatility σ_y", "Persistence ρ_y"],
        "suptitle": "Illustrative comparative statics, public Arellano calibration ({grid}); bars: 95% interval",
        "and": "and",
    },
    "es": {
        "columns": ["Escenario", "Deuda/producto (%)", "Diferencial medio (%)", "Impago (% anual)",
                    "DE del diferencial (pp)", "Valor del acceso al mercado (%)"],
        "baseline": "Base",
        "param": {"theta": "θ", "sigma_y": "σ_y", "rho_y": "ρ_y"},
        "param_long": {"theta": "probabilidad de reingreso", "sigma_y": "volatilidad del ingreso",
                       "rho_y": "persistencia del ingreso"},
        "outcome": {"debt_output_pct": "la deuda/producto", "spread_mean_pct": "el diferencial medio",
                    "default_annual_pct": "la frecuencia de impago",
                    "spread_sd_pp": "la volatilidad del diferencial",
                    "welfare_pct": "el valor del acceso al mercado"},
        "unit": {"debt_output_pct": " %", "spread_mean_pct": " %", "default_annual_pct": " % anual",
                 "spread_sd_pp": " pp", "welfare_pct": " % del consumo"},
        "up": "sube", "down": "baja", "flat": "deja sin cambio",
        "from_to": "de {a} a {b}",
        "lead": ("En este modelo ilustrativo, partiendo de la calibración pública y moviendo un "
                 "parámetro cada vez de su valor bajo a su valor alto: "),
        "clause": "una mayor {name} ({sym} {lo} → {hi}) {items}",
        "tail": ("Cada número es una media sobre {n} trayectorias simuladas de {periods} trimestres; "
                 "la tabla da su intervalo del 95 % y su cambio de malla."),
        "footnote": ("† El cambio de malla emparejado {coarse} → {fine} de al menos un momento no cabe "
                     "en la tolerancia registrada de 0,10 (su estado está en el registro). La fila se "
                     "informa, no se descarta."),
        "coarse_caveat": ("Malla gruesa (801 x 161): solo para orientarse; las cifras difieren del "
                          "informe convergido (6.401 x 641) en docs/es/brief_sovereign_risk.md."),
        "panel": ["Deuda/producto (%)", "Diferencial medio (%)", "Impago (% anual)",
                  "DE del diferencial (pp)", "Valor del acceso al mercado (%)"],
        "groups": ["Reingreso θ", "Volatilidad σ_y", "Persistencia ρ_y"],
        "suptitle": "Estática comparativa ilustrativa, calibración pública de Arellano ({grid}); barras: intervalo del 95 %",
        "and": "y",
    },
}


def _check_lang(lang: str) -> dict[str, Any]:
    if lang not in _TEXT:
        raise ValueError(f"lang must be 'en' or 'es'; got {lang!r}")
    return _TEXT[lang]


def _num(value: float, lang: str, fmt: str = ".3f") -> str:
    if fmt.endswith(".3f") and round(value, 3) == 0.0:
        value = 0.0  # no "-0.000"
    text = format(value, fmt)
    return text.replace(".", ",") if lang == "es" else text


def _grid_label(grid: Sequence[int], lang: str) -> str:
    n_b, n_y = grid
    sep = "." if lang == "es" else ","
    return f"{n_b:,}".replace(",", sep) + " x " + f"{n_y:,}".replace(",", sep)


def _fine_key(record: dict, grid: str | None) -> str:
    keys = [_case_key(*g) for g in record["grids"]]
    if grid is None or grid not in keys:
        return keys[0]
    return grid


def _scenario_label(name: str, lang: str) -> str:
    text = _TEXT[lang]
    if name == "baseline":
        return f"**{text['baseline']}**"
    key, _, value = name.partition("=")
    return f"{text['param'][key]} = {_num(float(value), lang, 'g')}"


def _ordered(record: dict) -> list[str]:
    names = [n for n in SCENARIOS if n in record["scenarios"]]
    return names + [n for n in record["scenarios"] if n not in names]


def _paired(record: dict, grid: str) -> bool:
    return len(record["grids"]) >= 2 and grid == _case_key(*record["grids"][0])


def render_table(record: dict, lang: str, grid: str | None = None) -> str:
    """Scenario x moments Markdown table: mean ± 95% half-width (Δ paired grid change)."""
    text = _check_lang(lang)
    grid = _fine_key(record, grid)
    paired = _paired(record, grid)
    lines = ["| " + " | ".join(text["columns"]) + " |", "|---|" + "---:|" * 5]
    flagged = False
    for name in _ordered(record):
        scenario = record["scenarios"][name]
        case = scenario["cases"][grid]
        summary = case["simulation"]["summary"]
        change = scenario.get("grid_change") if paired else None
        label = _scenario_label(name, lang)
        if change is not None and not change["passed"]:
            label += " †"
            flagged = True
        cells = [label]
        for m in MOMENTS:
            mean = summary[m]["mean"]
            half = (summary[m]["ci95"][1] - summary[m]["ci95"][0]) / 2.0
            cell = f"{_num(mean, lang)} ± {_num(half, lang)}"
            if change is not None:
                cell += f" (Δ {_num(change['moments'][m]['change'], lang, '+.3f')})"
            cells.append(cell)
        welfare = _num(case["welfare_pct"], lang)
        if change is not None:
            welfare += f" (Δ {_num(change['welfare_change_pct'], lang, '+.3f')})"
        cells.append(welfare)
        lines.append("| " + " | ".join(cells) + " |")
    if flagged:
        lines += ["", text["footnote"].format(coarse=_grid_label(record["grids"][1], lang),
                                              fine=_grid_label(record["grids"][0], lang))]
    return "\n".join(lines)


def _mean(record: dict, name: str, grid: str, outcome: str) -> float:
    case = record["scenarios"][name]["cases"][grid]
    if outcome == "welfare_pct":
        return float(case["welfare_pct"])
    return float(case["simulation"]["summary"][outcome]["mean"])


def render_answer(record: dict, lang: str, grid: str | None = None) -> str:
    """One paragraph: the direction and size of each effect, low to high value of each parameter."""
    text = _check_lang(lang)
    grid = _fine_key(record, grid)
    outcomes = ("debt_output_pct", "spread_mean_pct", "default_annual_pct", "welfare_pct")
    clauses = []
    for key in _SCENARIO_KEYS:
        lo_v, hi_v = _SCENARIO_VALUES[key]
        lo_name, hi_name = f"{key}={lo_v:g}", f"{key}={hi_v:g}"
        if lo_name not in record["scenarios"] or hi_name not in record["scenarios"]:
            continue
        items = []
        for outcome in outcomes:
            a, b = _mean(record, lo_name, grid, outcome), _mean(record, hi_name, grid, outcome)
            sa, sb = _num(a, lang), _num(b, lang)
            verb = text["up"] if sb > sa and b > a else text["down"] if sb < sa and b < a else text["flat"]
            unit = text["unit"][outcome]
            items.append(f"{verb} {text['outcome'][outcome]} "
                         + text["from_to"].format(a=sa + unit, b=sb + unit))
        joined = "; ".join(items[:-1]) + f"; {text['and']} " + items[-1]
        clauses.append(text["clause"].format(name=text["param_long"][key], sym=text["param"][key],
                                             lo=_num(lo_v, lang, "g"), hi=_num(hi_v, lang, "g"), items=joined))
    design = record["design"]
    periods = f"{design['periods']:,}".replace(",", "." if lang == "es" else ",")
    body = text["lead"] + ". ".join(c[0].upper() + c[1:] if i else c for i, c in enumerate(clauses)) + ". "
    return body + text["tail"].format(n=len(design["seeds"]), periods=periods)


_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")  # dataviz reference palette, slots 1-3 (validated)


def render_figure(record: dict, lang: str, grid: str | None = None):
    """Five panels (four moments and welfare): low / baseline / high per parameter with 95% whiskers."""
    from matplotlib.figure import Figure

    text = _check_lang(lang)
    grid = _fine_key(record, grid)
    fig = Figure(figsize=(12.0, 6.6), layout="constrained", facecolor="#fcfcfb")
    axes = fig.subplots(2, 3).ravel()
    axes[-1].remove()
    outcomes = MOMENTS + ("welfare_pct",)
    for ax, outcome, title in zip(axes[:5], outcomes, text["panel"]):
        ax.set_facecolor("#fcfcfb")
        ticks, labels = [], []
        for g, key in enumerate(_SCENARIO_KEYS):
            lo_v, hi_v = _SCENARIO_VALUES[key]
            names = [f"{key}={lo_v:g}", "baseline", f"{key}={hi_v:g}"]
            values = [lo_v, BASELINE[key], hi_v]
            for j, (name, value) in enumerate(zip(names, values)):
                if name not in record["scenarios"]:
                    continue
                case = record["scenarios"][name]["cases"][grid]
                x = 4 * g + j
                if outcome == "welfare_pct":
                    y, err = case["welfare_pct"], 0.0
                else:
                    s = case["simulation"]["summary"][outcome]
                    y, err = s["mean"], (s["ci95"][1] - s["ci95"][0]) / 2.0
                base = name == "baseline"
                ax.errorbar([x], [y], yerr=[err], fmt="o", ms=7 if base else 8, color=_COLORS[g],
                            mfc="#fcfcfb" if base else _COLORS[g], mew=2, elinewidth=2, capsize=3)
                ticks.append(x)
                label = _num(value, lang, "g")
                labels.append(label + "\n" + text["groups"][g] if j == 1 else label)
        ax.set_xticks(ticks, labels, fontsize=8, color="#52514e")
        ax.set_title(title, fontsize=10, color="#0b0b0b", loc="left")
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#b9b8b2")
        ax.tick_params(axis="y", labelsize=8, colors="#52514e")
        if lang == "es":
            from matplotlib.ticker import FuncFormatter

            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: format(v, "g").replace(".", ",")))
    fig.suptitle(text["suptitle"].format(grid=_grid_label(tuple(int(v) for v in grid.split("x")), lang)),
                 fontsize=11, color="#0b0b0b")
    return fig


def save_svg(fig, path: Path) -> None:
    """Write ``fig`` as an SVG with stable bytes (no date, fixed hash salt)."""
    import matplotlib

    with matplotlib.rc_context({"svg.hashsalt": "puremacro-brief", "svg.fonttype": "path"}):
        fig.savefig(path, format="svg", metadata={"Date": None})


def _csv_rows(record: dict) -> list[list[Any]]:
    header = ["scenario", "grid"]
    for m in MOMENTS:
        header += [f"{m}_mean", f"{m}_ci95_low", f"{m}_ci95_high", f"{m}_grid_change"]
    header += ["welfare_pct", "welfare_grid_change_pct"]
    rows = [header]
    for name in _ordered(record):
        scenario = record["scenarios"][name]
        for grid in record["grids"]:
            key = _case_key(*grid)
            case = scenario["cases"][key]
            change = scenario.get("grid_change") if _paired(record, key) else None
            row: list[Any] = [name, key]
            for m in MOMENTS:
                s = case["simulation"]["summary"][m]
                row += [s["mean"], s["ci95"][0], s["ci95"][1],
                        change["moments"][m]["change"] if change else ""]
            row += [case["welfare_pct"], change["welfare_change_pct"] if change else ""]
            rows.append(row)
    return rows


def write_outputs(record: dict, output: Path, lang: str) -> None:
    """Write table.md, table.csv, figure.png, record.json and manifest.json under ``output``."""
    import csv

    _check_lang(lang)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "table.md").write_text(render_answer(record, lang) + "\n\n" + render_table(record, lang) + "\n")
    with (output / "table.csv").open("w", newline="") as f:
        csv.writer(f).writerows(_csv_rows(record))
    fig = render_figure(record, lang)
    fig.savefig(output / "figure.png", dpi=150, metadata={"Software": None})
    (output / "record.json").write_text(json.dumps(record, indent=1, allow_nan=False) + "\n")
    single = len(record["grids"]) == 1
    manifest = {
        "package_version": record["package_version"],
        "grids": record["grids"],
        "seeds": record["design"]["seeds"],
        "periods": record["design"]["periods"],
        "burn": record["design"]["burn"],
        "measurement": "raw ergodic (ergodic_moments)",
        "peak_rss_gb": record.get("peak_rss_gb"),
        "load_averages": record.get("load_averages", {}),
        "cases": {name: {key: {"monotone_certification_gap": case["solver"]["monotone_certification_gap"],
                               "elapsed_s": case["solver"]["elapsed_s"]}
                         for key, case in s["cases"].items()}
                  for name, s in record["scenarios"].items()},
    }
    if single and tuple(record["grids"][0]) != GRIDS["converged"]:
        manifest["coarse_grid_caveat"] = True
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")


# --- Command line ------------------------------------------------------------

def _parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m puremacro.examples.first_brief_sovereign",
        description=("Illustrative sovereign-risk policy brief on PureMacro's public Arellano "
                     "calibration (not a claim about any country)."))
    p.add_argument("--scenario", help="baseline or key=value with key theta, sigma_y or rho_y "
                                      "(default grid: converged 6401x641, about half an hour)")
    p.add_argument("--sweep", action="store_true", help="all seven scenarios; requires --coarse")
    p.add_argument("--coarse", action="store_true", help="801x161 grid, minutes, orientation only")
    p.add_argument("--grid", help="NBxNY grid override for quick runs, e.g. 201x41")
    p.add_argument("--periods", type=int, help=f"quarters per path (default {DESIGN['periods']})")
    p.add_argument("--n-seeds", type=int, help=f"use the first K of the {len(DESIGN['seeds'])} design seeds")
    p.add_argument("--output", default="output/first_brief_sovereign", help="output directory")
    p.add_argument("--lang", default="en", choices=("en", "es"))
    p.add_argument("--freeze", type=Path, help="maintainer: converged + error-grid sweep, write the record")
    p.add_argument("--from-record", type=Path, help="render outputs from an existing record without solving")
    return p


def _parse_grid(text: str) -> tuple[int, int]:
    n_b, sep, n_y = text.lower().partition("x")
    if not sep:
        raise ValueError(f"--grid must look like 201x41; got {text!r}")
    return int(n_b), int(n_y)


def _load_log(message: str) -> None:
    try:
        load = " (load %.1f %.1f %.1f)" % os.getloadavg()
    except (AttributeError, OSError):
        load = ""
    print(message + load, flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)
    text = _TEXT[args.lang]

    if args.freeze is not None:
        path = Path(args.freeze)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial.json")
        record = build_record([GRIDS["converged"], GRIDS["error"]], design=DESIGN, partial=partial, log=_load_log)
        _write_atomic(path, record)
        print(f"Wrote {path}", flush=True)
        return 0

    if args.from_record is not None:
        record = json.loads(Path(args.from_record).read_text())
        write_outputs(record, Path(args.output), args.lang)
        print(render_table(record, args.lang))
        return 0

    if args.sweep and not args.coarse:
        print("--sweep runs only with --coarse (801 x 161, minutes). The converged sweep takes "
              "about five hours: maintainers use --freeze PATH.", file=sys.stderr)
        return 2
    if not args.sweep and args.scenario is None:
        print("Give --scenario NAME or --sweep --coarse (see --help).", file=sys.stderr)
        return 2

    try:
        if args.sweep:
            scenarios = dict(SCENARIOS)
        else:
            name, overrides = parse_scenario(args.scenario)
            scenarios = {name: overrides}
        grid = (_parse_grid(args.grid) if args.grid else GRIDS["coarse"] if args.coarse else GRIDS["converged"])
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    seeds = list(DESIGN["seeds"])
    if args.n_seeds is not None:
        if not 2 <= args.n_seeds <= len(seeds):
            print(f"--n-seeds must be between 2 and {len(seeds)}", file=sys.stderr)
            return 2
        seeds = seeds[:args.n_seeds]
    design = {"periods": args.periods or DESIGN["periods"], "seeds": seeds, "burn": DESIGN["burn"]}
    if args.coarse:
        print(text["coarse_caveat"], flush=True)
    record = build_record([grid], design=design, scenarios=scenarios, log=_load_log)
    write_outputs(record, Path(args.output), args.lang)
    print(render_answer(record, args.lang))
    print()
    print(render_table(record, args.lang))
    if args.coarse:
        print(text["coarse_caveat"])
    print(f"Outputs in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
