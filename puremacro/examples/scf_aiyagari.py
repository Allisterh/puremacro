"""Aiyagari (1994) calibrated to the Survey of Consumer Finances.

Run ``python -m puremacro.examples.scf_aiyagari --output DIR``.

The script downloads one SCF wave (summary extract plus the 999 replicate
weights, through the shared HTTP cache), estimates wealth-to-income ratios
and wealth-concentration moments with their replicate-weight standard
errors, calibrates the discount factor of a ``puremacro.dp`` Aiyagari
economy to one wealth-to-income ratio, and sets the model's untargeted
concentration moments against the data.

The expected result is a gap. A one-asset economy with idiosyncratic
earnings risk and a borrowing limit produces too little wealth at the top
(Aiyagari 1994; Castañeda, Díaz-Giménez and Ríos-Rull 2003; Krueger,
Mitman and Perri 2016). Nothing here is tuned to close that gap: one
parameter is calibrated to one target, and everything else is reported
as it comes out.

Two targets are run side by side because they pull the discount factor
in opposite directions. Mean wealth over mean income is dominated by the
top of the distribution, which this model lacks, so matching it forces a
very patient household (and an interest rate near or below zero) and may
not be attainable with ``beta < 1`` at all. Median wealth over median
income disciplines the middle of the distribution, and the mean ratio
then becomes an untargeted check.

Comparability. SCF ``networth`` is household net worth at market value
and ``income`` is the previous year's total pre-tax household income,
transfers included. The model's wealth is capital ``a`` and its income
``w*exp(z) + r*a``. Shares, the Gini and the ratios are scale-free, so the
units do not need to match. The SCF has negative net worth; the model's
borrowing limit is zero, so "wealth <= 0" in the model is the mass at the
limit.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from puremacro import dp
from puremacro.fetch.micro import MicroFrame, SurveyDesign

#: Fixed parameters. Capital share and depreciation as in Aiyagari (1994);
#: log-earnings AR(1) from Floden and Lindé (2001), US PSID estimates;
#: log utility.
DEFAULTS = dict(gamma=1.0, alpha=0.36, delta=0.08, rho=0.9136, sigma=0.2064,
                n_z=7, n_a=400, a_max=400.0)

MOMENTS = (
    "mean wealth / mean income",
    "median wealth / median income",
    "Gini",
    "top 1% share",
    "top 10% share",
    "bottom 50% share",
    "share with wealth <= 0",
)
TARGETS = {"mean": MOMENTS[0], "median": MOMENTS[1]}


# ----------------------------------------------------------------------------- model
def aiyagari(beta: float, *, gamma=DEFAULTS["gamma"], alpha=DEFAULTS["alpha"],
             delta=DEFAULTS["delta"], rho=DEFAULTS["rho"], sigma=DEFAULTS["sigma"],
             n_z=DEFAULTS["n_z"], n_a=DEFAULTS["n_a"], a_max=DEFAULTS["a_max"]) -> dp.Model:
    """The Aiyagari economy as a ``dp.Model`` with the interest rate as its price.

    The asset grid is cubic in the index, dense near the borrowing limit
    (where the constrained households are) and sparse at the top. The
    upper bracket for ``r`` stays below ``1/beta - 1``, where savings
    diverge.
    """
    grid = a_max * np.linspace(0.0, 1.0, int(n_a)) ** 3
    m = dp.Model("aiyagari")
    m.parameters(beta=beta, gamma=gamma, alpha=alpha, delta=delta)
    m.prices(r=(-0.75 * delta, 1.0 / beta - 1.0 - 0.003))
    m.exogenous("z", dp.AR1(rho=rho, sigma=sigma, n=int(n_z), method="rouwenhorst"))
    m.state("a", grid)
    m.local("w = (1 - alpha)*(alpha/(r + delta))^(alpha/(1 - alpha))")
    m.local("c = w*exp(z) + (1 + r)*a - a(+1)")
    m.reward("crra(c, gamma)")
    m.subject_to("c > 0")
    m.aggregate(K="a", L="exp(z)")
    m.clear("K - L*(alpha/(r + delta))^(1/(1 - alpha))")
    return m


def solve(beta: float, **params) -> dp.DPSolution:
    """Stationary equilibrium by EGM, the equilibrium rate to 1e-10."""
    return aiyagari(beta, **params).solve("egm", tol=1e-7, xtol=1e-10)


def model_frame(sol: dp.DPSolution) -> MicroFrame:
    """The stationary distribution as a weights-only microdata frame, one row
    per (asset, earnings) node with its mass as the weight, so model and
    data moments come from the same estimators."""
    a = np.broadcast_to(sol.grids["a"][:, None], sol.distribution.shape)
    data = pd.DataFrame({"networth": a.ravel(),
                         "income": sol.policy("w*exp(z) + r*a").ravel(),
                         "mass": sol.distribution.ravel()})
    return MicroFrame(data[data["mass"] > 0].reset_index(drop=True),
                      SurveyDesign.weights_only("mass", note="model stationary distribution"),
                      unit="household", source="model", vintage="stationary")


# ----------------------------------------------------------------------------- moments
def _interpolated_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    """``q``-quantile of the piecewise-linear CDF through the mass points
    (mass at equal values pooled first, so the CDF has one knot per value)."""
    xs, inv = np.unique(x, return_inverse=True)
    cw = np.cumsum(np.bincount(inv, weights=w))
    return float(np.interp(q * cw[-1], cw, xs))


def moments(mf: MicroFrame, *, interpolate: bool = False) -> pd.DataFrame:
    """The comparison moments of net worth, with standard errors where the
    design has them (``NaN`` for the model).

    ``interpolate=True`` (used for the model) takes the medians from the
    piecewise-linear CDF through the mass points. A distribution on an
    asset grid puts its median on a grid node, so the plain inverse-CDF
    median jumps from node to node as ``beta`` moves; a calibration to it
    would be a root search on a step function. Every other moment is
    already continuous in the masses.
    """
    nonpos = mf.data["networth"].le(0).astype(float).where(mf.data["networth"].notna())
    work = MicroFrame(mf.data.assign(_nonpos=nonpos), mf.design, mf.unit, mf.source, mf.vintage)
    bottom = work.top_share("networth", 0.5)
    bottom.loc[:, "estimate"] = 1.0 - bottom["estimate"]
    rows = [
        work.ratio("networth", "income"),
        work.ratio("networth", "income", q=0.5),
        work.gini("networth"),
        work.top_share("networth", 0.01),
        work.top_share("networth", 0.10),
        bottom,
        work.mean("_nonpos"),
    ]
    out = pd.concat(rows, ignore_index=True)[["estimate", "se"]]
    out.index = pd.Index(MOMENTS, name="moment")
    if interpolate:
        d = mf.data.dropna(subset=["networth", "income"])
        w = d[mf.design.weight].to_numpy(dtype=float)
        out.loc[MOMENTS[1]] = (
            _interpolated_quantile(d["networth"].to_numpy(dtype=float), w, 0.5)
            / _interpolated_quantile(d["income"].to_numpy(dtype=float), w, 0.5), np.nan)
    return out


def model_moments(sol: dp.DPSolution) -> pd.DataFrame:
    """:func:`moments` of a solved economy's stationary distribution."""
    return moments(model_frame(sol), interpolate=True)


# ----------------------------------------------------------------------------- calibration
@dataclass
class Calibration:
    """``beta`` set so that the model matches one SCF ratio.

    ``attained`` is False when no ``beta`` in the bracket reaches the
    target; ``beta`` and ``solution`` are then those at the bracket end
    closest to it and ``moments`` show how far off that is. ``se_beta``
    carries the target's sampling error through the model by the delta
    method, and ``moments["se"]`` carries it on to every model moment.
    """
    target: str
    target_value: float
    target_se: float
    attained: bool
    beta: float
    se_beta: float
    solution: dp.DPSolution
    moments: pd.DataFrame
    bracket: tuple[float, float]
    evaluations: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def r(self) -> float:
        return float(self.solution.prices["r"])

    @property
    def capital_output(self) -> float:
        """K/Y from the equilibrium aggregates, (K/L)^(1 - alpha)."""
        agg, alpha = self.solution.aggregates, self.solution.params["p_alpha"]
        return float((agg["K"] / agg["L"]) ** (1.0 - alpha))


def calibrate(target: str, value: float, se: float = np.nan, *,
              bracket=(0.90, 0.999), h: float = 2e-3, **params) -> Calibration:
    """Find ``beta`` in ``bracket`` whose stationary equilibrium matches
    ``value`` for the ratio named by ``target`` (``"mean"`` or ``"median"``)."""
    from scipy.optimize import brentq

    name = TARGETS[target]
    cache: dict[float, tuple[dp.DPSolution, pd.DataFrame]] = {}

    def at(beta):
        if beta not in cache:
            sol = solve(beta, **params)
            cache[beta] = (sol, model_moments(sol))
        return cache[beta]

    def gap(beta):
        return float(at(beta)[1].loc[name, "estimate"]) - value

    lo, hi = bracket
    g_lo, g_hi = gap(lo), gap(hi)
    notes = []
    if g_lo * g_hi > 0:
        beta = lo if abs(g_lo) < abs(g_hi) else hi
        sol, mom = at(beta)
        notes.append(f"{name} = {value:.4g} is outside what beta in [{lo}, {hi}] attains "
                     f"([{gap(lo) + value:.4g}, {gap(hi) + value:.4g}]); reported at beta = {beta}")
        return Calibration(target, value, se, False, beta, np.nan, sol,
                           mom.assign(se=np.nan), (lo, hi), len(cache), notes)
    beta = brentq(gap, lo, hi, xtol=1e-6)
    sol, mom = at(beta)
    # Delta method: d(moment)/d(beta) by central differences, then
    # se(beta) = se(target) / |d target / d beta|.
    up, dn = at(min(beta + h, hi))[1], at(max(beta - h, lo))[1]
    step = min(beta + h, hi) - max(beta - h, lo)
    slope = (up["estimate"] - dn["estimate"]) / step
    se_beta = float(se / abs(slope[name])) if np.isfinite(se) else np.nan
    mom = mom.assign(se=(slope.abs() * se_beta).to_numpy())
    return Calibration(target, value, se, True, float(beta), se_beta, sol, mom, (lo, hi),
                       len(cache), notes)


# ----------------------------------------------------------------------------- application
def lorenz(mf: MicroFrame, points=np.linspace(0.0, 1.0, 101)) -> pd.Series:
    """Lorenz curve of net worth at population shares ``points`` (point estimates)."""
    frame = MicroFrame(mf.data, SurveyDesign(weight=mf.design.weight, implicate=mf.design.implicate,
                                             note="point estimates only"),
                       mf.unit, mf.source, mf.vintage)
    vals = [0.0 if p == 0 else 1.0 if p == 1 else
            1.0 - float(frame.top_share("networth", 1.0 - p).loc[0, "estimate"]) for p in points]
    return pd.Series(vals, index=pd.Index(points, name="population share"), name="wealth share")


def comparison_table(data: pd.DataFrame, calibrations: dict[str, Calibration]) -> pd.DataFrame:
    """SCF estimates next to each calibrated model, with standard errors."""
    out = data.rename(columns={"estimate": "SCF", "se": "SCF se"})
    for key, cal in calibrations.items():
        label = f"model ({key} target)"
        out[label] = cal.moments["estimate"]
        out[f"{label} se"] = cal.moments["se"]
    return out


def run_application(scf: MicroFrame | None = None, *, year: int = 2025,
                    output: Path | None = None, targets=("mean", "median"),
                    bracket=(0.90, 0.999), **params) -> dict:
    """Estimate the SCF moments, calibrate one model per target, compare.

    ``scf`` is a :func:`puremacro.fetch.micro.fetch_scf` frame with
    ``networth`` and ``income``; it is downloaded when ``None``.
    """
    if scf is None:
        from puremacro.fetch.micro import fetch_scf
        scf = fetch_scf(year, ["networth", "income"])
    params = {**DEFAULTS, **params}
    data = moments(scf)
    calibrations = {
        t: calibrate(t, float(data.loc[TARGETS[t], "estimate"]), float(data.loc[TARGETS[t], "se"]),
                     bracket=bracket, **params)
        for t in targets}
    table = comparison_table(data, calibrations)
    summary = pd.DataFrame({
        key: {"target": TARGETS[key], "attained": cal.attained, "beta": cal.beta,
              "se(beta)": cal.se_beta, "r": cal.r, "K/Y": cal.capital_output,
              "equilibria solved": cal.evaluations}
        for key, cal in calibrations.items()}).T
    result = {"scf": scf, "data": data, "calibrations": calibrations, "table": table,
              "summary": summary, "params": params}
    if output is not None:
        _write(Path(output), result)
    return result


def _write(output: Path, result: dict) -> None:
    from puremacro import __version__
    from puremacro.reports import df_to_markdown

    output.mkdir(parents=True, exist_ok=True)
    scf, table, summary = result["scf"], result["table"], result["summary"]
    table.to_csv(output / "moments.csv")
    summary.to_csv(output / "calibration.csv")
    curves = {"SCF": lorenz(scf)}
    for key, cal in result["calibrations"].items():
        curves[f"model ({key} target)"] = lorenz(model_frame(cal.solution))
    curves = pd.DataFrame(curves)
    curves.to_csv(output / "lorenz.csv")
    notes = [n for cal in result["calibrations"].values() for n in cal.notes]
    manifest = {
        "package_version": __version__,
        "application": "Aiyagari (1994) calibrated to the Survey of Consumer Finances",
        "data": {"source": scf.source, "vintage": scf.vintage, "query": scf.query,
                 "design": scf.design.note, "rows": len(scf)},
        "fixed_parameters": result["params"],
        "calibration": json.loads(summary.to_json(orient="index")),
        "notes": notes,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(
        f"# Aiyagari economy vs SCF {scf.vintage}\n\n"
        "Discount factor calibrated to one wealth-to-income ratio per column; every other row is "
        "untargeted. SCF standard errors combine the 999 replicate weights and the five "
        "implicates; model standard errors carry the target's sampling error through the "
        "calibrated discount factor.\n\n"
        + df_to_markdown(table.round(4)) + "\n\n" + df_to_markdown(summary) + "\n\n"
        + "".join(f"- {n}\n" for n in notes), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color="0.75", linewidth=0.8)
    for (label, s), style in zip(curves.items(), ("-", "--", ":")):
        ax.plot(s.index, s.to_numpy(), style, label=label)
    ax.set(xlabel="Share of households, poorest first", ylabel="Share of net worth",
           title=f"Wealth Lorenz curves, SCF {scf.vintage}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "lorenz.png", dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--year", type=int, default=2025, help="SCF wave (default 2025)")
    parser.add_argument("--output", type=Path, default=Path("research_output/scf_aiyagari"))
    args = parser.parse_args(argv)
    result = run_application(year=args.year, output=args.output)
    with pd.option_context("display.width", 140, "display.float_format", "{:.4f}".format):
        print(result["table"])
        print()
        print(result["summary"])
    for cal in result["calibrations"].values():
        for n in cal.notes:
            print("note:", n)
    print(f"Saved moments, calibration, Lorenz curves and provenance to {args.output}")
    return result


if __name__ == "__main__":
    main()
