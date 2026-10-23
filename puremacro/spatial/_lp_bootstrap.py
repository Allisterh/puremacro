"""Opt-in circular block-pairs inference for stationary panel LP rows.

Blocks contain complete cross-sections and all horizon outcomes. Leads/lags
are constructed before resampling, so block seams do not invent observations.
The reference is a stationary, weakly dependent regression-row process with
fixed N and increasing T, not a unit-root or endogenous-shock theorem.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from ..lp._panel_helpers import two_way_fe_within


@dataclass(frozen=True)
class SpatialLPBootstrap:
    """Explicit work budget and assumptions for spatial LP block inference.

    ``assume_stationary=True`` asserts stationarity of the *regression rows*,
    including controls. Integrated outcomes can use stationary cumulative
    changes and ``lagged_outcome='difference'``; integrated level controls do
    not become stationary merely by setting this flag. Block length is in
    retained periods and must be selected before inspecting rejections.
    """

    n_boot: int
    block_length: int
    assume_stationary: bool = False
    seed: int | None = None

    def validate(self) -> None:
        for name, minimum in (("n_boot", 19), ("block_length", 1)):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"SpatialLPBootstrap: {name} must be an integer >= {minimum}")
        if self.assume_stationary is not True:
            raise ValueError("SpatialLPBootstrap requires assume_stationary=True for the regression rows")
        if self.seed is not None and (
            isinstance(self.seed, (bool, np.bool_)) or not isinstance(self.seed, (int, np.integer)) or self.seed < 0
        ):
            raise ValueError("SpatialLPBootstrap: seed must be a nonnegative integer or None")


@dataclass(frozen=True, eq=False)
class SpatialLPBootstrapResult:
    """Common-sample bootstrap-t bands and a joint spillover max-t reference.

    ``table`` has pointwise and simultaneous intervals for all displayed
    contrasts. ``joint_p_value`` tests zero spillovers over all requested
    horizons/orders using centered, re-studentized draws, not a chi-square
    approximation. It is asymptotic under the documented stationary-row
    assumptions, not a finite-sample exact test. ``coefficient_draws`` keeps
    all draws (none retried or discarded). Original LP estimates are unchanged.
    """

    table: pd.DataFrame
    coefficient_draws: np.ndarray
    statistic_draws: np.ndarray
    joint_statistic: float | None
    joint_p_value: float | None
    joint_mc_se: float | None
    metadata: dict[str, Any]

    def to_frame(self) -> pd.DataFrame:
        return self.table.copy()

    def summary(self) -> str:
        return (
            "Spatial LP circular block-pairs bootstrap-t (stationary regression rows)\n"
            f"{self.metadata['n_boot']} draws; block length {self.metadata['block_length']}; "
            f"common periods {self.metadata['n_periods']}; all cross-sections resampled together\n"
            f"Joint zero-spillover max-t p-value: {self.joint_p_value}\n"
            "Intervals use common-support estimates; stationarity and exogeneity are assumptions."
        )


def _circular_indices(rng: np.random.Generator, n: int, length: int) -> np.ndarray:
    starts = rng.integers(0, n, size=(n + length - 1) // length)
    return ((starts[:, None] + np.arange(length)) % n).ravel()[:n]


def _balanced_fit(values: np.ndarray, entities: np.ndarray) -> dict[str, Any]:
    """Refit with the shared two-way FE engine, including each draw's means."""
    T, N, _ = values.shape
    columns = ["outcome"] + [f"x{j}" for j in range(values.shape[2] - 1)]
    index = pd.MultiIndex.from_arrays(
        [np.tile(entities, T), np.repeat(np.arange(T), N)], names=["unit", "period"])
    frame = pd.DataFrame(values.reshape(T * N, -1), index=index, columns=columns)
    return two_way_fe_within(frame, y_col="outcome", x_cols=columns[1:],
                             entity_level="unit", time_level="period")


def _bootstrap_lp(
    frames: Sequence[pd.DataFrame], config: SpatialLPBootstrap,
    horizons: Sequence[int], selectors: Sequence[tuple[str, np.ndarray]], n_focal: int,
    covariance: Callable[[dict], np.ndarray], cov_kind: str, alpha: float,
    entity_level: str, time_level: str, *, lagged_outcome: str,
) -> SpatialLPBootstrapResult:
    entities = pd.Index(frames[0].index.get_level_values(entity_level).unique()).sort_values()
    if len(entities) < 2:
        raise ValueError("spatial LP bootstrap needs at least two entities")
    periods = pd.Index(frames[0].index.get_level_values(time_level).unique()).sort_values()
    for frame in frames:
        times = pd.Index(frame.index.get_level_values(time_level).unique()).sort_values()
        units = pd.Index(frame.index.get_level_values(entity_level).unique()).sort_values()
        if not units.equals(entities) or len(frame) != len(times) * len(entities):
            raise ValueError("spatial LP bootstrap requires a balanced complete panel at every horizon")
        periods = periods.intersection(times, sort=False)
    T, N = len(periods), len(entities)
    if T < 2 * config.block_length:
        raise ValueError("spatial LP bootstrap needs at least two block lengths on common support")
    index = pd.MultiIndex.from_product([periods, entities], names=[time_level, entity_level])
    values = []
    for frame in frames:
        grid = frame.reorder_levels([time_level, entity_level]).reindex(index).to_numpy(dtype=float)
        if not np.isfinite(grid).all():
            raise ValueError("spatial LP bootstrap needs finite complete common-support regression rows")
        values.append(grid.reshape(T, N, -1))
    S = np.stack([selector for _, selector in selectors])

    def fit_all(sample: np.ndarray):
        coefs, points, errors = [], [], []
        for grid in values:
            fit = _balanced_fit(grid[sample], entities.to_numpy())
            beta = fit["beta"][:n_focal]
            V = covariance(fit)[:n_focal, :n_focal]
            variances = np.einsum("ij,jk,ik->i", S, V, S)
            # The identically-zero indirect contrast without spillover orders
            # is excluded from studentized inference, rather than dividing by 0.
            active = np.any(S != 0, axis=1)
            if not np.isfinite(variances).all() or np.any(variances[active] <= 0):
                raise ValueError("nonpositive or nonfinite bootstrap studentizing variance")
            coefs.append(beta)
            points.append(S @ beta)
            errors.append(np.sqrt(variances))
        return np.asarray(coefs), np.asarray(points), np.asarray(errors)

    _, point, se = fit_all(np.arange(T))
    active = np.any(S != 0, axis=1)
    rng = np.random.default_rng(config.seed)
    coefficients = np.empty((config.n_boot, len(horizons), n_focal))
    statistics = np.zeros((config.n_boot, len(horizons), len(S)))
    for b in range(config.n_boot):
        sample = _circular_indices(rng, T, config.block_length)
        try:
            coefficients[b], pb, sb = fit_all(sample)
            statistics[b][:, active] = (pb - point)[:, active] / sb[:, active]
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise ValueError(f"spatial LP bootstrap draw {b} failed; no draws discarded: {exc}") from exc
    abs_t = np.abs(statistics)
    critical = np.quantile(abs_t, 1 - alpha, axis=0, method="higher")
    simultaneous = float(np.quantile(abs_t[:, :, active].max(axis=(1, 2)), 1 - alpha, method="higher"))
    rows = []
    for h, horizon in enumerate(horizons):
        for j, (label, _) in enumerate(selectors):
            observed = abs(point[h, j] / se[h, j]) if active[j] else 0.0
            p = (1 + int(np.count_nonzero(abs_t[:, h, j] >= observed))) / (config.n_boot + 1)
            rows.append({"h": horizon, "label": label, "point": point[h, j], "se": se[h, j],
                         "lo": point[h, j] - critical[h, j] * se[h, j],
                         "hi": point[h, j] + critical[h, j] * se[h, j],
                         "simultaneous_lo": point[h, j] - simultaneous * se[h, j],
                         "simultaneous_hi": point[h, j] + simultaneous * se[h, j],
                         "critical_value": critical[h, j], "p_value": p,
                         "mc_se": np.sqrt(p * (1 - p) / (config.n_boot + 1))})
    # Each raw spillover has its own per-order displayed contrast. Scaling
    # cancels in |t|, except deliberately zero-scaled contrasts, which cannot
    # test their underlying gamma and are explicitly excluded.
    spill = [j for j, (name, _) in enumerate(selectors)
             if name.startswith("indirect") and name != "indirect_all" and active[j]]
    joint_statistic = joint_p = joint_mc = None
    if spill:
        joint_statistic = float(np.max(np.abs(point[:, spill] / se[:, spill])))
        maxima = abs_t[:, :, spill].max(axis=(1, 2))
        joint_p = float((1 + np.count_nonzero(maxima >= joint_statistic)) / (config.n_boot + 1))
        joint_mc = float(np.sqrt(joint_p * (1 - joint_p) / (config.n_boot + 1)))
    return SpatialLPBootstrapResult(
        table=pd.DataFrame(rows).set_index(["h", "label"]), coefficient_draws=coefficients,
        statistic_draws=statistics, joint_statistic=joint_statistic,
        joint_p_value=joint_p, joint_mc_se=joint_mc,
        metadata={"method": "circular_block_pairs_studentized", "n_boot": config.n_boot,
                  "block_length": config.block_length, "seed": config.seed, "alpha": alpha,
                  "n_periods": T, "n_entities": N, "common_periods": tuple(periods),
                  "cov_type": cov_kind, "lagged_outcome": lagged_outcome,
                  "stationarity_assumed": True, "draws_discarded": 0,
                  "p_value_resolution": 1 / (config.n_boot + 1),
                  "simultaneous_critical_value": simultaneous,
                  "joint_labels": tuple(selectors[j][0] for j in spill),
                  "scope": "fixed-N stationary weakly dependent regression rows; exogenous shocks",
                  "point_sample": "intersection of horizon samples; original LP rows unchanged"},
    )
