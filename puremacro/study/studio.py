"""Self-Auditing Counterfactual Policy Studio for PureMacro.

Provides a unified declarative grid runner (:func:`run_policy_studio`) that
evaluates DSGE, HANK, Trade, Spatial, or econometric counterfactuals across a
multi-dimensional parameter/policy space, automatically auditing every cell for
equilibrium convergence, conservation laws, and numerical residuals, and
exporting publication-ready Markdown, LaTeX, Typst, and zero-server interactive
HTML5/SVG reports.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping, Sequence

if TYPE_CHECKING:
    import matplotlib.figure

import numpy as np
import pandas as pd

from puremacro.reports import (
    _df_to_latex,
    _df_to_markdown,
    _df_to_typst,
    result_to_interactive_html,
)

__all__ = [
    "CounterfactualGrid",
    "PolicyStudioCell",
    "PolicyStudioResult",
    "run_policy_studio",
]


@dataclass(frozen=True)
class CounterfactualGrid:
    """Cartesian grid of policy / structural parameter counterfactuals."""

    axes: dict[str, tuple[Any, ...]]
    fixed_params: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        axes: Mapping[str, Sequence[Any]],
        fixed_params: Mapping[str, Any] | None = None,
    ) -> None:
        if not axes:
            raise ValueError("CounterfactualGrid requires at least one axis.")
        clean_axes: dict[str, tuple[Any, ...]] = {}
        for k, vals in axes.items():
            t_vals = tuple(vals)
            if len(t_vals) == 0:
                raise ValueError(f"Axis {k!r} cannot be empty.")
            clean_axes[str(k)] = t_vals
        object.__setattr__(self, "axes", clean_axes)
        object.__setattr__(self, "fixed_params", dict(fixed_params or {}))

    @property
    def axis_names(self) -> tuple[str, ...]:
        return tuple(self.axes.keys())

    def __len__(self) -> int:
        n = 1
        for vals in self.axes.values():
            n *= len(vals)
        return n

    def iter_points(self) -> Iterator[dict[str, Any]]:
        keys = list(self.axes.keys())
        value_lists = [self.axes[k] for k in keys]
        for combo in itertools.product(*value_lists):
            pt = dict(self.fixed_params)
            for k, v in zip(keys, combo):
                pt[k] = v
            yield pt


@dataclass(frozen=True)
class PolicyStudioCell:
    """Single evaluated cell in a PolicyStudio counterfactual grid."""

    coordinates: dict[str, Any]
    metrics: dict[str, float]
    audit_status: str
    audit_residual: float
    converged: bool
    notes: str = ""
    raw_result: Any = None


@dataclass(frozen=True)
class PolicyStudioResult:
    """Aggregated self-auditing counterfactual studio result."""

    grid: CounterfactualGrid
    cells: tuple[PolicyStudioCell, ...]
    pass_count: int
    warn_count: int
    fail_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def overall_audit_status(self) -> str:
        total = len(self.cells)
        if self.fail_count > 0:
            return f"FAIL ({self.fail_count}/{total} failed)"
        if self.warn_count > 0:
            return f"WARN ({self.warn_count}/{total} warnings)"
        return f"PASS ({self.pass_count}/{total} verified)"

    def to_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        axis_names = self.grid.axis_names
        for cell in self.cells:
            row: dict[str, Any] = {k: cell.coordinates[k] for k in axis_names}
            for m_k, m_v in cell.metrics.items():
                row[m_k] = float(m_v)
            row["audit_status"] = cell.audit_status
            row["audit_residual"] = float(cell.audit_residual)
            row["converged"] = bool(cell.converged)
            rows.append(row)
        return pd.DataFrame(rows)

    def to_frame(self) -> pd.DataFrame:
        return self.to_dataframe()

    def summary(self) -> pd.DataFrame:
        return self.to_dataframe()

    def pivot(
        self,
        metric: str,
        row_axis: str | None = None,
        col_axis: str | None = None,
    ) -> pd.DataFrame:
        df = self.to_dataframe()
        axes = self.grid.axis_names
        if len(axes) < 2 and (row_axis is None or col_axis is None):
            return df.set_index(axes[0])[[metric]]
        r_ax = row_axis or axes[0]
        c_ax = col_axis or axes[1]
        return df.pivot(index=r_ax, columns=c_ax, values=metric)

    def to_markdown(self, **kwargs: Any) -> str:
        return _df_to_markdown(self.to_dataframe(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        return _df_to_latex(self.to_dataframe(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        return _df_to_typst(self.to_dataframe(), **kwargs)

    def to_interactive_html(
        self,
        *,
        title: str = "PureMacro PolicyStudio Counterfactual Explorer",
        subtitle: str | None = None,
    ) -> str:
        sub = subtitle or f"Grid axes: {', '.join(self.grid.axis_names)} | Audit: {self.overall_audit_status}"
        return result_to_interactive_html(
            self.to_dataframe(),
            title=title,
            subtitle=sub,
            audit_badge=self.overall_audit_status,
        )

    def plot(
        self,
        metric: str | None = None,
        figsize: tuple[float, float] = (7.5, 4.5),
    ) -> matplotlib.figure.Figure:
        import matplotlib.pyplot as plt

        df = self.to_dataframe()
        axes = self.grid.axis_names
        metric_cols = [
            c
            for c in df.columns
            if c not in axes and c not in ("audit_status", "audit_residual", "converged")
        ]
        m_col = metric or (metric_cols[0] if metric_cols else "audit_residual")
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        if len(axes) >= 2:
            piv = self.pivot(m_col, axes[0], axes[1])
            im = ax.imshow(piv.to_numpy(dtype=float), aspect="auto", cmap="viridis")
            ax.set_xticks(range(len(piv.columns)))
            ax.set_xticklabels([str(x) for x in piv.columns])
            ax.set_yticks(range(len(piv.index)))
            ax.set_yticklabels([str(y) for y in piv.index])
            ax.set_xlabel(axes[1])
            ax.set_ylabel(axes[0])
            ax.set_title(f"{m_col} ({self.overall_audit_status})", loc="center")
            fig.colorbar(im, ax=ax)
        else:
            ax.plot(df[axes[0]], df[m_col], marker="o", lw=2.0)
            ax.set_xlabel(axes[0])
            ax.set_ylabel(m_col)
            ax.set_title(f"{m_col} ({self.overall_audit_status})", loc="center")
        return fig


def _default_extract_metrics(res: Any) -> dict[str, float]:
    if isinstance(res, Mapping):
        return {
            str(k): float(v)
            for k, v in res.items()
            if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool)
        }
    out: dict[str, float] = {}
    for attr in (
        "first_order_dlog_y",
        "second_order_dlog_y",
        "total_second_order_dlog_y",
        "exact_ces_dlog_y",
        "binding_periods",
        "precautionary_shift_norm",
        "welfare",
        "welfare_hat",
        "log_likelihood",
        "objective",
    ):
        val = getattr(res, attr, None)
        if isinstance(val, (int, float, np.floating, np.integer)) and not isinstance(val, bool):
            out[attr] = float(val)
        elif isinstance(val, np.ndarray) and val.size > 0:
            out[f"{attr}_mean"] = float(np.mean(val))
    if not out and hasattr(res, "simulated_path"):
        sim = getattr(res, "simulated_path")
        if isinstance(sim, pd.DataFrame) and len(sim) > 0:
            for col in sim.columns[:3]:
                out[f"{col}_impact"] = float(sim[col].iloc[0])
                out[f"{col}_min"] = float(sim[col].min())
    return out


def _default_audit_result(
    res: Any,
    *,
    warn_tol: float = 1e-6,
    fail_tol: float = 1e-3,
) -> tuple[str, float, bool, str]:
    if isinstance(res, Mapping):
        conv = bool(res.get("converged", True))
        resid = float(res.get("residual", res.get("max_residual", 0.0)))
    else:
        conv = bool(getattr(res, "converged", True))
        resid_candidates: list[float] = []
        for attr in (
            "max_bound_violation",
            "max_residual",
            "max_bellman_residual",
            "population_conservation_error",
            "labor_conservation_residual",
            "migration_row_stochastic_error",
        ):
            val = getattr(res, attr, None)
            if isinstance(val, (int, float, np.floating)):
                resid_candidates.append(abs(float(val)))
        resid = max(resid_candidates) if resid_candidates else 0.0

    if not conv or not np.isfinite(resid) or resid > fail_tol:
        return "FAIL", resid, False, "Non-convergent or residual exceeded fail_tol"
    if resid > warn_tol:
        return "WARN", resid, True, "Converged with residual above warn_tol"
    return "PASS", resid, True, "Verified"


def run_policy_studio(
    evaluator: Callable[[dict[str, Any]], Any],
    grid: CounterfactualGrid | Mapping[str, Sequence[Any]],
    *,
    fixed_params: Mapping[str, Any] | None = None,
    metric_fn: Callable[[Any], Mapping[str, float]] | None = None,
    audit_fn: Callable[[Any], tuple[str, float]] | None = None,
    warn_tol: float = 1e-6,
    fail_tol: float = 1e-3,
) -> PolicyStudioResult:
    """Evaluate and self-audit a counterfactual model across a parameter grid."""
    cf_grid = (
        grid
        if isinstance(grid, CounterfactualGrid)
        else CounterfactualGrid(grid, fixed_params=fixed_params)
    )

    cells: list[PolicyStudioCell] = []
    n_pass = 0
    n_warn = 0
    n_fail = 0

    for pt in cf_grid.iter_points():
        raw = evaluator(dict(pt))
        metrics = (
            {str(k): float(v) for k, v in metric_fn(raw).items()}
            if metric_fn is not None
            else _default_extract_metrics(raw)
        )
        if audit_fn is not None:
            status, resid = audit_fn(raw)
            status = status.upper()
            conv = status != "FAIL"
            notes = ""
        else:
            status, resid, conv, notes = _default_audit_result(
                raw, warn_tol=warn_tol, fail_tol=fail_tol
            )

        if status == "PASS":
            n_pass += 1
        elif status == "WARN":
            n_warn += 1
        else:
            n_fail += 1

        cells.append(
            PolicyStudioCell(
                coordinates=dict(pt),
                metrics=metrics,
                audit_status=status,
                audit_residual=float(resid),
                converged=conv,
                notes=notes,
                raw_result=raw,
            )
        )

    return PolicyStudioResult(
        grid=cf_grid,
        cells=tuple(cells),
        pass_count=n_pass,
        warn_count=n_warn,
        fail_count=n_fail,
    )
