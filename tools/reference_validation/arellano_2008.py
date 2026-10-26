"""Offline Arellano (2008) archive study, separate from PureMacro's public solver.

The author-code timing, grid and literal precision are deliberate. A canonical
timing counterfactual changes just the value/timing of exclusion. No calibration
is fitted to the published moments. See tests/fixtures/arellano_2008/README.md.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import ndtr
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/arellano_2008"
# Fortran real(8) parameters are initialized from unsuffixed default-real literals.
BETA, THETA, GROSS_R, CAP = map(lambda x: float(np.float32(x)), (.95282, .282, 1.017, .969))
MOMENTS = {
    "debt_pct": 5.95, "spread_mean_pct": 3.58, "annual_default_pct": 3.00,
    "spread_std_pp": 6.36, "tb_std_pp": 1.50, "consumption_std_pct": 6.38,
    "output_std_pct": 5.81, "corr_spread_y": -.29, "corr_tb_y": -.25,
    "corr_c_y": .97, "corr_tb_spread": .43, "corr_c_spread": -.36,
}


@dataclass
class Solution:
    assets: np.ndarray
    income: np.ndarray
    transition: np.ndarray
    default_income: np.ndarray
    value: np.ndarray
    price: np.ndarray
    policy: np.ndarray
    default: np.ndarray
    timing: str
    theta: float = THETA
    gross_r: float = GROSS_R
    diagnostics: dict | None = None


def verified_inputs() -> tuple[np.ndarray, np.ndarray]:
    manifest = json.loads((FIXTURE / "source_manifest.json").read_text())
    for name, digest in manifest["files_sha256"].items():
        if hashlib.sha256((FIXTURE / "original" / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Archived source changed: {name}")
    return (np.loadtxt(FIXTURE / "original/YMAT.dat"),
            np.loadtxt(FIXTURE / "original/PIMAT.dat"))


def author_grid(refinement: int = 0) -> np.ndarray:
    """Reproduce process.f90, then refine by inserting midpoints (fixed support)."""
    if refinement < 0:
        raise ValueError("refinement must be nonnegative")
    assets = np.empty(200)
    assets[0] = float(np.float32(-3.3))
    step = (1.5 - assets[0]) / 197
    for i in range(1, 200):
        assets[i] = assets[i-1] + step
        if assets[i] > 0 and assets[i-1] < 0:
            assets[i] = 0
    for _ in range(refinement):
        fine = np.empty(2*len(assets)-1)
        fine[::2] = assets
        fine[1::2] = (assets[:-1]+assets[1:])/2
        assets = fine
    return assets


def income_process(n_y: int = 21) -> tuple[np.ndarray, np.ndarray]:
    log_y, p = verified_inputs()
    if n_y == 21:
        return 10*np.exp(log_y), p
    if n_y < 3 or n_y % 2 != 1:
        raise ValueError("n_y must be odd and at least three")
    # The archived inputs match Tauchen (1986), rho=.945, sigma=.025, m=3,
    # to their 8-digit printing precision; retain identical support on refinement.
    z = np.linspace(log_y[0], log_y[-1], n_y)
    edges = np.r_[-np.inf, (z[:-1]+z[1:])/2, np.inf]
    p = np.diff(ndtr((edges[None, :]-.945*z[:, None])/.025), axis=1)
    return 10*np.exp(z), p


def load_original(path: Path = FIXTURE / "reference.npz") -> Solution:
    income, p = income_process()
    with np.load(path) as a:
        return Solution(a["b"], income, p, a["conaut"], a["v"], a["p"],
                        a["i_s_star"].astype(int)-1, a["def"].astype(bool), "archive")


def solve(refinement: int = 0, n_y: int = 21, timing: str = "archive") -> Solution:
    """Independent NumPy solver; verify Bellman and pricing residuals on return.

    Howard evaluations accelerate each fixed-price problem. Price iteration and
    default comparisons follow the archived code, including its extra period
    excluded on entry and its rounded (not renormalized) transition matrix.
    """
    if timing not in ("archive", "canonical"):
        raise ValueError("timing must be archive or canonical")
    started = time.monotonic()
    assets = author_grid(refinement)
    income, p = income_process(n_y)
    yd = np.minimum(income, 10*CAP)
    ud = -1/yd
    n_b = len(assets)
    zero = int(np.flatnonzero(assets == 0)[0])
    inverse = np.linalg.inv(np.eye(n_y)-BETA*(1-THETA)*p)
    v = np.broadcast_to(np.linalg.solve(np.eye(n_y)-BETA*p, ud), (n_b, n_y)).copy()
    price = np.full_like(v, 1/GROSS_R)
    rows, cols = np.arange(n_b), np.arange(n_y)

    def combine(repay):
        excluded = inverse @ (ud+BETA*THETA*(p @ repay[zero]))
        vd = ud+BETA*(p @ excluded) if timing == "archive" else excluded
        default = (assets[:, None] < 0) & (repay < vd)
        return np.where(default, vd, repay), default

    def improve(value, utilities):
        ev = BETA*(value @ p.T)
        repay = np.empty_like(value)
        policy = np.empty(value.shape, dtype=int)
        for j in range(n_y):
            objective = utilities[j]+ev[:, j][None, :]
            # The archived >= tie rule chooses the last maximizing index.
            policy[:, j] = n_b-1-np.argmax(objective[:, ::-1], axis=1)
            repay[:, j] = objective[rows, policy[:, j]]
        value_new, default = combine(repay)
        return value_new, default, policy

    inner_total = 0
    for outer in range(1, 501):
        utilities = []
        for j in range(n_y):
            consumption = income[j]+assets[:, None]-price[:, j]*assets
            utility = np.full_like(consumption, -1e19)
            np.divide(-1., consumption, out=utility, where=consumption > 0)
            utilities.append(utility)
        for inner in range(1000):
            new, default, policy = improve(v, utilities)
            error = float(np.max(abs(new-v)))
            v = new
            inner_total += 1
            if error < 1e-11:
                break
            selected = np.column_stack([utilities[j][rows, policy[:, j]] for j in range(n_y)])
            for _ in range(20):
                ev = BETA*(v @ p.T)
                v, _ = combine(selected+ev[policy, cols])
        else:
            raise RuntimeError(f"Value iteration did not converge: {error}")
        target = (1-default.astype(float) @ p.T)/GROSS_R
        price_error = float(np.max(abs(target-price)))
        if price_error < 1e-8:
            break
        price = .5*(price+target)
    else:
        raise RuntimeError(f"Price iteration did not converge: {price_error}")
    checked, checked_default, checked_policy = improve(v, utilities)
    residual = float(np.max(abs(checked-v)))
    if residual > 2e-11 or not np.array_equal(default, checked_default):
        raise RuntimeError(f"Final Bellman check failed: {residual}")
    return Solution(assets, income, p, yd, v, price, checked_policy, checked_default, timing,
                    diagnostics=dict(outer_iterations=outer, policy_improvements=inner_total,
                                     bellman_residual=residual, price_residual=price_error,
                                     elapsed_seconds=time.monotonic()-started))


def compare_original(solution: Solution, original: Solution) -> dict:
    errors = dict(assets=float(np.max(abs(solution.assets-original.assets))),
                  value=float(np.max(abs(solution.value-original.value))),
                  price=float(np.max(abs(solution.price-original.price))),
                  default_mismatches=int(np.sum(solution.default != original.default)),
                  policy_mismatches=int(np.sum(solution.policy != original.policy)))
    return dict(errors=errors, passed=errors["assets"] < 1e-14 and errors["value"] < 1e-8
                and errors["price"] < 1e-8 and errors["default_mismatches"] == 0
                and errors["policy_mismatches"] == 0)


def simulate(solution: Solution, seed: int, periods: int = 100000, burn: int = 1000) -> dict:
    """Simulate a path with model-consistent exclusion timing and explicit flows.

    Uniforms are pre-drawn separately for income and reentry, giving common
    random numbers across cases. The archived matrix's tiny row-mass error is
    normalized for sampling only, never for solving the archived equations.
    """
    if periods <= burn or burn < 0:
        raise ValueError("periods must exceed nonnegative burn")
    rng = np.random.default_rng(seed)
    uniforms = rng.random((periods, 2))
    p = solution.transition / solution.transition.sum(axis=1, keepdims=True)
    cdf = p.cumsum(axis=1)
    cdf[:, -1] = 1
    zero = int(np.argmin(abs(solution.assets)))
    b, y, excluded = len(solution.assets)-1, 0, False
    consumption, output, prices, next_assets = np.zeros((4, periods))
    entry, unavailable, boundary = np.zeros((3, periods), dtype=bool)
    for t in range(periods):
        y = int(np.searchsorted(cdf[y], uniforms[t, 0], side="right"))
        # Canonical D allows reentry after the entry period itself; archive D
        # guarantees an additional excluded period before the first reentry draw.
        enters = not excluded and bool(solution.default[b, y])
        autarky = excluded or enters
        entry[t], unavailable[t] = enters, autarky
        if autarky:
            consumption[t] = output[t] = solution.default_income[y]
            b = zero
            can_draw = excluded or solution.timing == "canonical"
            excluded = not (can_draw and uniforms[t, 1] < solution.theta)
        else:
            chosen = solution.policy[b, y]
            q = solution.price[chosen, y]
            output[t] = solution.income[y]
            consumption[t] = output[t]+solution.assets[b]-q*solution.assets[chosen]
            prices[t], next_assets[t] = q, solution.assets[chosen]
            boundary[t] = chosen == 0 or chosen == len(solution.assets)-1
            b = chosen
    return {name: values[burn:] for name, values in dict(consumption=consumption, output=output,
            price=prices, assets=next_assets, entry=entry, excluded=unavailable, boundary=boundary).items()}


def annual_spread(price: np.ndarray, gross_r: float) -> np.ndarray:
    if np.any(price <= 0):
        raise ValueError("Spreads require positive repayment-period prices")
    return price**-4-gross_r**4


def eligible_windows(path: dict, length: int = 74) -> np.ndarray:
    """Calendar windows ending just before entry, containing no excluded period."""
    if length < 2:
        raise ValueError("window length must be at least two")
    indices = np.flatnonzero(path["entry"])
    return np.array([i for i in indices if i >= length and
                     np.all(path["price"][i-length:i] > 0)], dtype=int)


def correlation(a, b) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-14 and np.std(b) > 1e-14 else float("nan")


def summarize_path(path: dict, gross_r: float = GROSS_R, windows: int = 100) -> dict:
    ends = eligible_windows(path)
    if len(ends) < windows:
        raise ValueError(f"Only {len(ends)} eligible windows; {windows} required")
    rows = []
    for end in ends[:windows]:
        sl = slice(end-74, end)
        c, y = path["consumption"][sl], path["output"][sl]
        spread = annual_spread(path["price"][sl], gross_r)
        tb, logc, logy = (y-c)/y, np.log(c), np.log(y)
        rows.append(dict(debt_pct=-100*path["assets"][sl].mean()/y.mean(),
                         spread_mean_pct=100*spread.mean(), spread_std_pp=100*spread.std(ddof=1),
                         tb_std_pp=100*tb.std(ddof=1), consumption_std_pct=100*logc.std(ddof=1),
                         output_std_pct=100*logy.std(ddof=1), corr_spread_y=correlation(spread, logy),
                         corr_tb_y=correlation(tb, logy), corr_c_y=correlation(logc, logy),
                         corr_tb_spread=correlation(tb, spread), corr_c_spread=correlation(logc, spread)))
    moments = {name: float(np.mean([row[name] for row in rows])) for name in rows[0]}
    entries = int(path["entry"].sum())
    repayment_periods = int(np.sum(~path["excluded"]))
    # Reproduce the archive's denominator (repayment periods), while separately
    # reporting the hazard among all periods with access at the start.
    archive_frequency = entries/repayment_periods
    hazard = entries/(repayment_periods+entries)
    moments["annual_default_pct"] = 100*(1-(1-archive_frequency)**4)
    diagnostics = dict(default_entries=entries, eligible_windows=len(ends), selected_windows=windows,
                       repayment_periods=repayment_periods, simulated_periods=len(path["price"]),
                       annual_access_hazard_pct=100*(1-(1-hazard)**4),
                       excluded_share_pct=100*float(path["excluded"].mean()),
                       boundary_choice_share_pct=100*float(path["boundary"].sum()/repayment_periods))
    return dict(moments=moments, diagnostics=diagnostics)


def monte_carlo(solution: Solution, seeds: list[int], periods: int, burn: int, windows: int = 100) -> dict:
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("At least two distinct seeds are required for Monte Carlo uncertainty")
    runs = [dict(seed=seed, **summarize_path(simulate(solution, seed, periods, burn),
                                          solution.gross_r, windows)) for seed in seeds]
    moments = {}
    for name, target in MOMENTS.items():
        values = np.array([run["moments"][name] for run in runs])
        mean, sd = float(values.mean()), float(values.std(ddof=1))
        half = float(student_t.ppf(.975, len(seeds)-1)*sd/np.sqrt(len(seeds)))
        moments[name] = dict(published=target, mean=mean, difference=mean-target,
                             mc_se=sd/np.sqrt(len(seeds)), mc_ci95=[mean-half, mean+half],
                             replication_sd=sd, replication_range=[float(values.min()), float(values.max())])
    return dict(moments=moments, runs=runs)
