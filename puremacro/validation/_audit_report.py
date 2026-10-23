"""Result audit records: deterministic evidence, without a universal validity badge."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AuditCheck:
    """One scoped numerical check. Missing evidence is never a pass."""

    category: str
    check: str
    value: float | bool | None
    tolerance: float | None
    units: str
    status: str
    evidence_source: str
    location: tuple[int, ...] | str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        # Preserve the kind of failure in strict JSON; never emit nonstandard NaN.
        return "NaN" if np.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported audit evidence type: {type(value).__name__}")


@dataclass(frozen=True)
class AuditReport:
    """Snapshot of evidence for one result, including residual maps.

    ``solver_converged`` describes the solver's own declaration only. There is
    deliberately no overall ``passed`` property: approximation accuracy,
    feasibility, and external validation are different claims.
    """

    records: tuple[AuditCheck, ...]
    maps: Mapping[str, np.ndarray] = field(default_factory=dict, repr=False)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"

    @property
    def solver_converged(self) -> bool | None:
        for row in self.records:
            if row.check == "solver_converged" and row.value is not None:
                return bool(row.value)
        return None

    def to_dataframe(self) -> pd.DataFrame:
        """One row per check; metadata is retained in ``DataFrame.attrs``."""
        columns = ["category", "check", "value", "tolerance", "units", "status",
                   "evidence_source", "location", "details"]
        frame = pd.DataFrame([{
            name: _jsonable(getattr(row, name)) for name in columns
        } for row in self.records], columns=columns)
        frame.attrs = _jsonable({"schema_version": self.schema_version, **self.metadata})
        return frame

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return _jsonable({"schema_version": self.schema_version,
                          "solver_converged": self.solver_converged,
                          "records": [asdict(row) for row in self.records],
                          "maps": self.maps, "metadata": self.metadata})

    def to_json(self, *, indent: int | None = 2) -> str:
        """Strict, deterministic JSON; nonfinite failed values are named strings."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent, allow_nan=False)

    def to_markdown(self) -> str:
        from puremacro.reports import df_to_markdown
        table = self.to_dataframe().drop(columns=["details"])
        return ("Result audit (each check has its own scope)\n\n" + df_to_markdown(table)
                + "\n\nEvidence metadata: `" + json.dumps(_jsonable(self.metadata), sort_keys=True) + "`")


@dataclass(frozen=True)
class ResidualCheck:
    """Explicit equation/constraint callback ``evaluate(result, model)``.

    Return signed residuals (equality) or nonnegative violations (inequality).
    The audit takes their absolute maximum. Only executed when ``equations``
    is selected. Callbacks must not mutate inputs or access the network.
    """

    name: str
    evaluate: Callable[[Any, Any], Any]
    tolerance: float = 1e-8
    units: str = "equation units"
    evidence_source: str = "caller_recomputed_equations"


@dataclass(frozen=True)
class DerivativeCheck:
    """Compare a supplied derivative to an independent five-point stencil.

    ``function(x)`` returns a scalar or vector; ``jacobian`` must have shape
    ``function(x).shape + x.shape`` (a gradient for a scalar). ``step`` is
    relative to ``max(1, abs(x))``. Only runs when ``derivatives`` is selected.
    """

    name: str
    function: Callable[[np.ndarray], Any]
    point: np.ndarray
    jacobian: Any
    step: float = 1e-4
    tolerance: float = 1e-5
    units: str = "relative derivative error"
    evidence_source: str = "independent_five_point_difference"


@dataclass(frozen=True)
class RefinementCheck:
    """Explicit grid/horizon experiment, run only with ``refinement`` selected.

    ``solve()`` returns a refined result. ``metrics`` extracts the same named
    scalars from the baseline and refined results (assets, consumption, Euler
    errors or selected Jacobian entries). With no extractor, stationary DP
    consumption/assets and audit KKT/boundary metrics are used. Acceptance is
    ``abs(new-old) <= atol + rtol*abs(new)``; near-zero denominators use the
    absolute tolerance. Supply distinct experiments to refine both grids and
    expand each bound; the audit never chooses a grid or horizon silently.
    """

    name: str
    solve: Callable[[], Any]
    metrics: Callable[[Any], Mapping[str, float]] | None = None
    rtol: float = 0.005
    atol: float = 1e-8
    units: Mapping[str, str] = field(default_factory=dict)
    evidence_source: str = "caller_refinement_experiment"


@dataclass(frozen=True)
class PathAuditContext:
    """Explicit boundary and equation evidence for a perfect-foresight path.

    ``equations(lead, current, lag, shock)`` returns one residual per variable.
    An explicitly declared ``steady_state`` argument receives ``y_end`` as a
    fixed reference. Shocks must contain exactly the reported dates; no
    padding, boundary solve or model compilation is performed by the audit.
    Pass this as ``model=`` to audit a legacy or manually constructed result.
    """

    equations: Callable[..., Any]
    y_init: Any
    y_end: Any
    exogenous_path: Any
    variable_names: Sequence[str] = ()
    mode: str = "perfect_foresight"


@dataclass(frozen=True)
class RefinementLevel:
    """One explicitly configured solve in a refinement ladder.

    ``cost`` is a caller-declared work unit, not a measured time or memory
    bound. ``settings`` records the intended mesh, domain or horizon. The
    callback must return a new result without mutating earlier results.
    """

    name: str
    solve: Callable[[], Any]
    settings: Mapping[str, Any] = field(default_factory=dict)
    cost: float = 1.0


@dataclass(frozen=True)
class RefinementLadder:
    """Bounded adjacent-level comparisons, only run with ``refinement`` selected.

    Metrics and a finite ordered set of solve callbacks are mandatory. Stop
    after ``consecutive_passes`` consecutive comparisons meet every metric's
    ``atol + rtol * abs(new)`` target, or at the first failed callback, invalid
    result, exhausted level list or explicit solve/work budget. Stabilization
    is evidence for the chosen metrics only, not proof of global accuracy.
    ``validate_pair`` may raise if a callback changed the economic experiment
    instead of its numerical resolution. No such equivalence is inferred.
    """

    name: str
    levels: Sequence[RefinementLevel]
    metrics: Callable[[Any], Mapping[str, float]]
    max_solves: int = 3
    max_cost: float | None = None
    consecutive_passes: int = 2
    rtol: float = 0.005
    atol: float = 1e-8
    units: Mapping[str, str] = field(default_factory=dict)
    validate_pair: Callable[[Any, Any], None] | None = None
    evidence_source: str = "caller_bounded_refinement_ladder"
