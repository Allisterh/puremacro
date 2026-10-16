"""Explicit tariff scenarios and a warm-started batch runner.

A :class:`TariffScenario` names one importing country and the ad valorem rates it
levies by exporter and sector. :func:`build_tariff_matrices` turns it into the
bilateral multiplier tensors of a calibration, :func:`run_scenario` solves one
scenario and :func:`run_scenario_batch` solves several from a common baseline
(warm-started) and adds Geary-Khamis real GDP.

There is no registry of named scenarios: every scenario states its tariffs.

Pure NumPy, vectorised.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np

from puremacro.trade._results import (
    GearyKhamisResult,
    ScenarioBatchResult,
    TradeCalibrationResult,
    TradeEquilibriumResult,
)
from puremacro.trade.geary_khamis import compute_geary_khamis
from puremacro.trade.solver import build_initial_guess, solve_trade_equilibrium

NAMED_SCENARIOS_REMOVED = "named scenarios were removed; build a TariffScenario"

ALL_SECTORS = "*"
"""Sector wildcard in :attr:`TariffScenario.tariffs` keys: every sector of the exporter."""


def _check_rate(rate: Any, where: str) -> float:
    if isinstance(rate, bool) or not isinstance(rate, Real):
        raise TypeError(f"Tariff rate for {where} must be a real number, got {rate!r}")
    value = float(rate)
    if math.isnan(value) or math.isinf(value) or value < 0.0:
        raise ValueError(f"Tariff rate for {where} must be a finite non-negative number, got {rate!r}")
    return value


@dataclass(frozen=True)
class TariffScenario:
    """One importing country's ad valorem import tariffs, by exporter and sector.

    Attributes
    ----------
    importer : str
        Code of the country that levies the tariffs.
    tariffs : Mapping[tuple[str, str], float] or np.ndarray
        Rates (0.10 = 10%). A mapping is keyed by ``(exporter, sector)`` codes;
        the sector ``"*"`` stands for every sector of that exporter, and an
        explicit sector entry overrides it. An array has shape ``(nc, ns)``
        (exporter country by sector, in the calibration's order) and must be
        zero on the importer's own row. Flows not listed carry no tariff, and
        domestic flows are never tariffed.
    name : str, default "scenario"
        Label used in results and batch tables.
    description : str, default ""
        Free text.
    metadata : dict, default empty
        Free metadata, copied to the result's metadata.

    Examples
    --------
    >>> scen = TariffScenario("C01", {("C02", "*"): 0.10}, name="ten_percent_on_C02")
    """

    importer: str
    tariffs: Mapping[tuple[str, str], float] | np.ndarray
    name: str = "scenario"
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.importer, str) or not self.importer:
            raise TypeError("importer must be a non-empty country code (str)")
        if not isinstance(self.name, str):
            raise TypeError("name must be a str")
        if isinstance(self.tariffs, np.ndarray):
            arr = np.array(self.tariffs, dtype=float)
            if arr.ndim != 2:
                raise ValueError(f"A tariff array must have shape (nc, ns); got {arr.shape}")
            if not np.isfinite(arr).all() or np.any(arr < 0):
                raise ValueError("Tariff rates in the array must be finite and non-negative")
            arr.setflags(write=False)
            object.__setattr__(self, "tariffs", arr)
        elif isinstance(self.tariffs, Mapping):
            rates: dict[tuple[str, str], float] = {}
            for key, rate in self.tariffs.items():
                if (not isinstance(key, tuple) or len(key) != 2
                        or not all(isinstance(k, str) for k in key)):
                    raise TypeError(f"Tariff keys must be (exporter, sector) code pairs; got {key!r}")
                rates[key] = _check_rate(rate, f"{key}")
            object.__setattr__(self, "tariffs", rates)
        else:
            raise TypeError(
                "tariffs must be a mapping {(exporter, sector): rate} or an (nc, ns) array of rates"
            )

    def rate_matrix(self, country_codes: Sequence[str], sector_codes: Sequence[str]) -> np.ndarray:
        """Rates as an ``(nc, ns)`` array (exporter by sector) for the given code lists.

        Raises ``ValueError`` for an unknown importer, exporter or sector code, a
        tariff on the importer's own goods, or an array of the wrong shape.
        """
        countries = [str(c) for c in country_codes]
        sectors = [str(s) for s in sector_codes]
        nc, ns = len(countries), len(sectors)
        if self.importer not in countries:
            raise ValueError(f"Unknown country code {self.importer!r} (importer); known codes: {countries}")
        d_idx = countries.index(self.importer)
        if isinstance(self.tariffs, np.ndarray):
            if self.tariffs.shape != (nc, ns):
                raise ValueError(
                    f"Tariff array has shape {self.tariffs.shape}; this calibration needs (nc, ns) = ({nc}, {ns})"
                )
            rates = np.array(self.tariffs, dtype=float)
        else:
            rates = np.zeros((nc, ns), dtype=float)
            explicit: list[tuple[int, int, float]] = []
            for (exporter, sector), rate in self.tariffs.items():
                if exporter not in countries:
                    raise ValueError(f"Unknown country code {exporter!r} (exporter); known codes: {countries}")
                o_idx = countries.index(exporter)
                if sector == ALL_SECTORS:
                    rates[o_idx, :] = rate
                elif sector in sectors:
                    explicit.append((o_idx, sectors.index(sector), rate))
                else:
                    raise ValueError(f"Unknown sector code {sector!r}; known codes: {sectors}")
            for o_idx, s_idx, rate in explicit:
                rates[o_idx, s_idx] = rate
        if np.any(rates[d_idx] != 0.0):
            raise ValueError(f"Domestic flows are never tariffed: {self.importer!r} lists a rate on its own goods")
        return rates


def _require_scenario(scenario: Any) -> TariffScenario:
    if isinstance(scenario, str):
        raise TypeError(NAMED_SCENARIOS_REMOVED)
    if not isinstance(scenario, TariffScenario):
        raise TypeError(f"scenario must be a TariffScenario, got {type(scenario).__name__}")
    return scenario


def build_tariff_matrices(
    calib: TradeCalibrationResult,
    scenario: TariffScenario,
) -> tuple[np.ndarray, np.ndarray]:
    """Bilateral tariff multipliers ``1 + rate`` of one scenario.

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibration whose dimensions and codes index the tensors.
    scenario : TariffScenario
        The scenario. A string raises ``TypeError`` (named scenarios were removed).

    Returns
    -------
    tau : np.ndarray, shape (ns, nc, ns, nc)
        Intermediate multipliers ``tau[s_orig, c_orig, s_dest, c_dest]``.
    tau_fd : np.ndarray, shape (ns, nc, nfd, nc)
        Final-demand multipliers ``tau_fd[s_orig, c_orig, fd_dest, c_dest]``.

    Both apply the scenario's rate on goods of ``(c_orig, s_orig)`` bought by
    the importer, for every using sector and final-use category; every other
    entry, including all domestic flows, is one. ``solve_trade_equilibrium``
    accepts both arrays as given.
    """
    scenario = _require_scenario(scenario)
    if not isinstance(calib, TradeCalibrationResult):
        raise TypeError("calib must be a TradeCalibrationResult")
    nc, ns, nfd = calib.n_countries, calib.n_sectors, calib.n_final_demand
    rates = scenario.rate_matrix(calib.country_codes, calib.sector_codes)
    d_idx = list(calib.country_codes).index(scenario.importer)

    tau = np.ones((ns, nc, ns, nc), dtype=float)
    tau_fd = np.ones((ns, nc, nfd, nc), dtype=float)
    mult = 1.0 + rates.T  # (ns, nc): origin sector by origin country
    tau[:, :, :, d_idx] = mult[:, :, None]
    tau_fd[:, :, :, d_idx] = mult[:, :, None]
    tau[:, d_idx, :, d_idx] = 1.0
    tau_fd[:, d_idx, :, d_idx] = 1.0
    return tau, tau_fd


def run_scenario(
    calib: TradeCalibrationResult,
    scenario: TariffScenario,
    **solver_opts: Any,
) -> TradeEquilibriumResult:
    """Solve the equilibrium of one tariff scenario.

    ``solver_opts`` go to :func:`puremacro.trade.solver.solve_trade_equilibrium`
    (``method``, ``tol``, ``x0``, ``base_result``, ...). The result's metadata
    records ``scenario_name`` and ``scenario_importer``.
    """
    scenario = _require_scenario(scenario)
    for key in ("tau", "tau_fd", "tauf", "tauf_fd"):
        if key in solver_opts:
            raise TypeError(f"run_scenario takes its tariffs from the scenario; do not pass {key}=")
    tau, tau_fd = build_tariff_matrices(calib, scenario)
    res = solve_trade_equilibrium(calib, tau=tau, tau_fd=tau_fd, **solver_opts)
    res.metadata["scenario_name"] = scenario.name
    res.metadata["scenario_importer"] = scenario.importer
    if scenario.metadata:
        res.metadata["scenario_metadata"] = dict(scenario.metadata)
    return res


def run_scenario_batch(
    calib: TradeCalibrationResult,
    scenarios: Sequence[TariffScenario],
    *,
    baseline_name: str = "baseline",
    base_result: TradeEquilibriumResult | None = None,
    warm_start: bool = True,
    compute_gk: bool = True,
    numeraire: str | None = None,
    **solver_opts: Any,
) -> ScenarioBatchResult:
    """Solve a baseline and several tariff scenarios, warm-starting each solve.

    The baseline (no tariffs) is solved first under ``baseline_name`` unless
    ``base_result`` supplies it. Each scenario then starts from the previous
    solution when ``warm_start`` is true (from the calibrated state otherwise).
    With ``compute_gk`` the Geary-Khamis PPP and real GDP of every state are
    computed against the baseline (``numeraire`` is passed to
    :func:`~puremacro.trade.geary_khamis.compute_geary_khamis`).

    ``solver_opts`` go to every solve. Scenario names must be distinct and
    differ from ``baseline_name``.
    """
    if isinstance(scenarios, (str, TariffScenario)):
        raise TypeError("scenarios must be a sequence of TariffScenario objects")
    scen_list = [_require_scenario(s) for s in scenarios]
    names = [s.name for s in scen_list]
    if len(set(names)) != len(names) or baseline_name in names:
        raise ValueError(f"Scenario names must be distinct and differ from {baseline_name!r}; got {names}")
    if "x0" in solver_opts:
        raise TypeError("run_scenario_batch chooses the starting points; do not pass x0=")

    results: dict[str, TradeEquilibriumResult] = {}
    if base_result is None:
        base_result = solve_trade_equilibrium(calib, **solver_opts)
    results[baseline_name] = base_result
    prev_x = base_result.x_sol.copy()

    for scen in scen_list:
        x_init = prev_x.copy() if warm_start else build_initial_guess(calib)
        res = run_scenario(calib, scen, x0=x_init, base_result=base_result, **solver_opts)
        results[scen.name] = res
        if warm_start:
            prev_x = res.x_sol.copy()

    gk_results: dict[str, GearyKhamisResult] = {}
    if compute_gk:
        for name, res in results.items():
            gk_results[name] = compute_geary_khamis(res, base_result, calib, method="iterative",
                                                    numeraire=numeraire, tol=1e-10, max_iter=1000)

    return ScenarioBatchResult(
        scenarios=results,
        geary_khamis=gk_results,
        baseline_scenario=baseline_name,
        country_codes=tuple(calib.country_codes),
        sector_codes=tuple(calib.sector_codes),
        metadata={
            "warm_start": warm_start,
            "numeraire": numeraire,
            "solver_options": {k: v for k, v in solver_opts.items() if k != "base_result"},
            "converged": {name: bool(res.converged) for name, res in results.items()},
        },
    )


__all__ = [
    "TariffScenario",
    "build_tariff_matrices",
    "run_scenario",
    "run_scenario_batch",
]
