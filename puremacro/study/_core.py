"""Study contracts and thin adapters over the existing LP and VAR engines."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import scipy

from puremacro import __version__
from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst
from puremacro.runtime import capabilities, store
from puremacro.structural.lp import _expected_grid

if TYPE_CHECKING:
    from ._scenario import ScenarioResult


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _integer(value: Any, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _label(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be an explicit nonempty string")


@dataclass(frozen=True)
class VariableSpec:
    """A column's transformation and final units; no units are inferred from its name.

    ``scale`` multiplies the transformed series. ``unit`` describes the series
    *after* transformation and scaling (for example percent log growth).
    """
    name: str
    unit: str
    transform: str = "identity"
    scale: float = 1.0

    def __post_init__(self) -> None:
        _label(self.name, "variable name")
        _label(self.unit, "variable unit")
        if self.transform not in {"identity", "log", "diff", "logdiff"}:
            raise ValueError("transform must be identity, log, diff or logdiff")
        if isinstance(self.scale, bool) or not isinstance(self.scale, (int, float)) or not math.isfinite(self.scale) or self.scale == 0:
            raise ValueError("scale must be finite and nonzero")


@dataclass(frozen=True)
class StudySpec:
    """One fixed-lag study with explicit identification and response conventions.

    LP requires ``identification='observed_exogenous'`` and estimates
    ``y[t+h]-y[t-1]`` conditional on the declared regressors. VAR requires
    ``identification='recursive'`` and an ordering of every variable, and
    estimates a level response to an impact of ``shock_size`` in the shock
    variable's transformed units. These are different comparison contracts.
    """
    name: str
    method: str
    variables: tuple[VariableSpec, ...]
    responses: tuple[str, ...]
    shock: str
    frequency: str
    identification: str
    lags: int
    horizon: int
    shock_size: float
    controls: tuple[str, ...] = ()
    ordering: tuple[str, ...] = ()
    sample_start: str | None = None
    sample_end: str | None = None
    ci: float = 0.90
    seed: int = 0
    n_boot: int = 0

    def __post_init__(self) -> None:
        for name in ("name", "shock", "frequency"):
            _label(getattr(self, name), name)
        for name in ("variables", "responses", "controls", "ordering"):
            value = getattr(self, name)
            if isinstance(value, str):
                raise ValueError(f"{name} must be a sequence, not a string")
            object.__setattr__(self, name, tuple(value))
        if not self.variables or not all(isinstance(v, VariableSpec) for v in self.variables):
            raise ValueError("variables must contain VariableSpec instances")
        names = tuple(v.name for v in self.variables)
        if len(set(names)) != len(names):
            raise ValueError("variable names must be unique")
        for key in ("responses", "controls", "ordering"):
            seq = getattr(self, key)
            if any(not isinstance(s, str) for s in seq) or len(set(seq)) != len(seq) or not set(seq) <= set(names):
                raise ValueError(f"{key} must contain unique declared variable names")
        if not self.responses or self.shock not in names:
            raise ValueError("responses and shock must be declared")
        if self.method == "lp":
            if self.identification != "observed_exogenous" or self.ordering:
                raise ValueError("LP requires observed_exogenous identification and no recursive ordering")
            if self.shock in self.responses or set(self.controls) & (set(self.responses) | {self.shock}):
                raise ValueError("LP responses, shock and controls must be distinct")
            if set(names) != set(self.responses) | {self.shock} | set(self.controls):
                raise ValueError("every LP variable must be a response, shock or control")
            if self.n_boot != 0:
                raise ValueError("LP uses HAC intervals; n_boot must be zero")
        elif self.method == "var_cholesky":
            if self.identification != "recursive" or set(self.ordering) != set(names) or self.controls:
                raise ValueError("VAR requires recursive identification, ordering of every variable, and no separate controls")
        else:
            raise ValueError("method must be lp or var_cholesky")
        _integer(self.lags, "lags", 1 if self.method == "var_cholesky" else 0)
        _integer(self.horizon, "horizon")
        _integer(self.seed, "seed")
        _integer(self.n_boot, "n_boot")
        if self.n_boot == 1:
            raise ValueError("bootstrap requires at least two draws, or zero for no intervals")
        if isinstance(self.shock_size, bool) or not isinstance(self.shock_size, (int, float)) or not math.isfinite(self.shock_size) or self.shock_size == 0:
            raise ValueError("shock_size must be finite and nonzero")
        if isinstance(self.ci, bool) or not isinstance(self.ci, (int, float)) or not 0 < self.ci < 1:
            raise ValueError("ci must be between zero and one")
        for bound in (self.sample_start, self.sample_end):
            if bound is not None:
                _label(bound, "sample bound")

    @property
    def response_convention(self) -> str:
        return "y[t+h]-y[t-1]" if self.method == "lp" else "level_irf"


def _finite(frame: pd.DataFrame, name: str) -> None:
    """Never archive NaN/Inf numeric cells, complex values, or arbitrary objects."""
    for column in frame.columns:
        values = frame[column]
        if pd.api.types.is_numeric_dtype(values):
            if np.iscomplexobj(values.to_numpy()) or not np.isfinite(values.to_numpy(dtype=float)).all():
                raise ValueError(f"{name}.{column} contains nonfinite or complex values")


def _encode(frame: pd.DataFrame, name: str) -> bytes:
    _finite(frame, name)
    payload = store.dumps_frame(frame)
    # The runtime codec supports many pandas types, but does not preserve every
    # subclass-specific attribute or named MultiIndex column layout. Refuse
    # those structures instead of silently labelling a lossy record "portable".
    try:
        pd.testing.assert_frame_equal(frame, store.loads_frame(payload))
    except AssertionError as exc:
        raise ValueError(f"{name} cannot be recorded losslessly by the runtime frame codec: {exc}") from exc
    return payload


@dataclass(frozen=True)
class StudyResult:
    """Immutable snapshot. Frame and manifest accessors return independent copies.

    Numerical output is tidy: response, horizon, estimate, n_obs, unit,
    shock_unit, plus lower/upper when intervals were actually computed.
    ``audits`` are input/estimation diagnostics, not economic-validity tests.
    """
    spec: StudySpec
    _frames: tuple[tuple[str, bytes], ...]
    _manifest_json: str

    def __post_init__(self) -> None:
        if not isinstance(self.spec, StudySpec) or not isinstance(self._manifest_json, str):
            raise TypeError("study snapshots require a StudySpec and JSON manifest")
        frames = tuple(tuple(item) for item in self._frames)
        if any(len(item) != 2 or not isinstance(item[0], str) or not isinstance(item[1], bytes) for item in frames):
            raise TypeError("study frame snapshots must contain names and immutable bytes")
        object.__setattr__(self, "_frames", frames)

    def _frame(self, name: str) -> pd.DataFrame:
        return store.loads_frame(dict(self._frames)[name])

    @property
    def inputs(self) -> pd.DataFrame:
        """Selected raw input observations, before declared transformations."""
        return self._frame("inputs")

    @property
    def data(self) -> pd.DataFrame:
        return self._frame("data")

    @property
    def audits(self) -> pd.DataFrame:
        return self._frame("audits")

    @property
    def manifest(self) -> dict[str, Any]:
        return json.loads(self._manifest_json)

    def to_frame(self) -> pd.DataFrame:
        return self._frame("results")

    def to_markdown(self) -> str:
        meta = self.manifest
        assumptions = pd.DataFrame({"assumption": list(meta["assumptions"]), "value": list(meta["assumptions"].values())})
        return (f"Study: {self.spec.name}\n\n" + df_to_markdown(assumptions, index=False)
                + "\n\n" + df_to_markdown(self.to_frame(), index=False)
                + "\n\nDiagnostics (not economic validity):\n\n" + df_to_markdown(self.audits, index=False)
                + f"\n\nRecord SHA-256: {meta['record_sha256']}\n")

    def to_latex(self) -> str:
        return df_to_latex(self.to_frame(), index=False)

    def to_typst(self) -> str:
        return df_to_typst(self.to_frame(), index=False)

    def to_json(self) -> str:
        """Manifest and tables; the portable archive additionally retains input frames."""
        return _json({"manifest": self.manifest, "results": self.to_frame().to_dict("records"),
                      "audits": self.audits.to_dict("records")})

    def write_report(self, directory: str | Path) -> Path:
        """Create a new directory containing tables, assumptions and the full archive.

        The directory must not exist. Formatting reuses ``puremacro.reports``;
        the archive is the lossless record, whereas CSV is for presentation.
        """
        from ._archive import save_study
        archive = save_study(self)
        assert isinstance(archive, bytes)
        contents = {"report.md": self.to_markdown(), "results.tex": self.to_latex(),
                    "results.typ": self.to_typst(), "results.csv": self.to_frame().to_csv(index=False),
                    "audits.csv": self.audits.to_csv(index=False), "manifest.json": self._manifest_json + "\n"}
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=False)
        for name, text in contents.items():
            (target / name).write_text(text, encoding="utf-8")
        (target / "study.pmstudy").write_bytes(archive)
        return target


@dataclass(frozen=True)
class StudyComparison:
    """Compatible studies; lag and interval choices remain explicit per study."""
    studies: tuple[StudyResult | ScenarioResult, ...]
    comparison_note: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "studies", tuple(self.studies))
        _check_comparable(self.studies, self.comparison_note)

    def to_frame(self) -> pd.DataFrame:
        result = pd.concat([s.to_frame().assign(study=s.spec.name, lags=getattr(s.spec, "lags", None),
                         ci=getattr(s.spec, "ci", None), n_boot=getattr(s.spec, "n_boot", None)) for s in self.studies], ignore_index=True)
        result.attrs = {"comparison_note": self.comparison_note,
                        "scope": "Declared compatible quantities; economic equivalence is not certified."}
        return result

    def to_markdown(self) -> str:
        note = f"Comparison rationale: {self.comparison_note}\n\n" if self.comparison_note else ""
        return note + "\n\n".join(s.to_markdown() for s in self.studies)

    def to_latex(self) -> str:
        return df_to_latex(self.to_frame(), index=False)

    def to_typst(self) -> str:
        return df_to_typst(self.to_frame(), index=False)


def _prepare(data: pd.DataFrame, spec: StudySpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not isinstance(data, pd.DataFrame) or data.empty or not data.columns.is_unique:
        raise ValueError("data must be a nonempty DataFrame with unique columns")
    names = [v.name for v in spec.variables]
    if set(data.columns) != set(names):
        raise ValueError("variables must declare every input column exactly once")
    if not isinstance(data.index, (pd.DatetimeIndex, pd.PeriodIndex)) or data.index.hasnans:
        raise ValueError("data require a DatetimeIndex or PeriodIndex without missing dates")
    if not data.index.is_unique or not data.index.is_monotonic_increasing:
        raise ValueError("data dates must be unique and increasing")
    if not data.index.equals(_expected_grid(data.index, spec.frequency)):
        raise ValueError("data must be consecutive at the declared frequency; missing periods cannot be compressed")
    raw = data.loc[:, names].copy(deep=True)
    if spec.sample_start is not None or spec.sample_end is not None:
        # Exact dates avoid pandas' partial-string selection silently expanding a bound.
        def bound(value: str | None, default: Any) -> Any:
            if value is None:
                return default
            parsed = pd.Period(value, freq=raw.index.freq) if isinstance(raw.index, pd.PeriodIndex) else pd.Timestamp(value)
            if parsed not in raw.index:
                raise ValueError(f"sample bound {value!r} is not an observation")
            return parsed
        start = bound(spec.sample_start, raw.index[0])
        stop = bound(spec.sample_end, raw.index[-1])
        raw = raw.loc[start:stop].copy()
    if raw.empty:
        raise ValueError("selected sample is empty")
    if any(not pd.api.types.is_numeric_dtype(raw[c]) or pd.api.types.is_bool_dtype(raw[c]) for c in raw):
        raise ValueError("study inputs must be real numeric columns")
    _encode(raw, "inputs")  # Validate metadata serialization before invoking any estimator.
    transformed = raw.astype(float)
    for v in spec.variables:
        series = transformed[v.name]
        if v.transform in {"log", "logdiff"}:
            if (series <= 0).any():
                raise ValueError(f"{v.name}: log transformation requires strictly positive values")
            series = np.log(series)
        if v.transform in {"diff", "logdiff"}:
            series = series.diff()
        transformed[v.name] = series * v.scale
    if any(v.transform in {"diff", "logdiff"} for v in spec.variables):
        transformed = transformed.iloc[1:].copy()
    if transformed.empty:
        raise ValueError("no observations remain after transformation")
    _finite(transformed, "transformed data")
    return raw, transformed


def _validate_design(data: pd.DataFrame, spec: StudySpec) -> None:
    """Check every design before any numerical fit, including every LP response."""
    n, k = data.shape
    if spec.method == "var_cholesky":
        if n - spec.lags <= 1 + k * spec.lags:
            raise ValueError("not enough observations for VAR lags and covariance degrees of freedom")
        y = data.loc[:, list(spec.ordering)].to_numpy()
        design = np.column_stack([np.ones(n - spec.lags)] + [y[spec.lags - lag:n - lag] for lag in range(1, spec.lags + 1)])
        if np.linalg.matrix_rank(design) < design.shape[1]:
            raise ValueError("VAR lag design is rank deficient")
        return
    first = max(1, spec.lags)
    for response in spec.responses:
        for h in range(spec.horizon + 1):
            t = np.arange(first, n - h)
            columns = [np.ones(len(t)), data[spec.shock].to_numpy()[t]]
            for lag in range(1, spec.lags + 1):
                columns.extend(data[c].to_numpy()[t - lag] for c in (spec.shock, response, *spec.controls))
            columns.extend(data[c].to_numpy()[t] for c in spec.controls)
            design = np.column_stack(columns)
            if len(t) <= design.shape[1] or h + 1 >= len(t):
                raise ValueError("not enough observations for every LP horizon/design/HAC bandwidth")
            if np.linalg.matrix_rank(design) < design.shape[1]:
                raise ValueError(f"LP design is rank deficient for {response}, horizon {h}")


def run_study(data: pd.DataFrame, spec: StudySpec) -> StudyResult:
    """Run one explicit offline study without changing inputs or global random state.

    Transform the selected raw sample, removing only its known leading
    difference observation. Missing values are errors. LP effective samples
    shrink by horizon; VAR uses one fitted sample for every horizon. Both are
    reported rather than represented as identical estimation samples.
    """
    if not isinstance(spec, StudySpec):
        raise TypeError("spec must be a StudySpec")
    raw, frame = _prepare(data, spec)
    _validate_design(frame, spec)
    units = {v.name: v.unit for v in spec.variables}
    rows: list[dict[str, Any]] = []
    audit = [{"check": "input_and_design", "status": "passed", "detail": "Regular dates, finite inputs, explicit transformations and full-rank designs checked."},
             {"check": "identification", "status": "not_checked", "detail": "Identification is a declared economic assumption, not established by this workflow."}]
    samples = []
    numerical: dict[str, Any] = {}
    if spec.method == "lp":
        from puremacro.lp import lp_hac
        for response in spec.responses:
            # The estimator creates dy_h / *_Lk columns. A caller may use those
            # legitimate names, so use a collision-free adapter frame instead.
            lp_frame = pd.DataFrame({"response": frame[response], "shock": frame[spec.shock]}, index=frame.index)
            controls = [f"control{i}" for i in range(len(spec.controls))]
            for key, name in zip(controls, spec.controls):
                lp_frame[key] = frame[name]
            fit = lp_hac(lp_frame, y="response", x="shock", horizons=range(spec.horizon + 1),
                         n_lags=spec.lags, controls=controls, alpha=1 - spec.ci)
            for h in range(spec.horizon + 1):
                entry = fit.loc[h]
                lo, hi = sorted((float(entry["lo"]) * spec.shock_size, float(entry["hi"]) * spec.shock_size))
                rows.append(dict(response=response, horizon=h, estimate=float(entry["beta"]) * spec.shock_size,
                                 lower=lo, upper=hi, se=float(entry["se"]) * abs(spec.shock_size),
                                 n_obs=len(frame) - max(1, spec.lags) - h,
                                 unit=units[response], shock_unit=units[spec.shock]))
        for h in range(spec.horizon + 1):
            samples.append({"horizon": h, "start": str(frame.index[max(1, spec.lags)]),
                            "end": str(frame.index[len(frame) - h - 1]),
                            "n_obs": len(frame) - max(1, spec.lags) - h})
        inference = "Pointwise normal HAC intervals, Bartlett bandwidth h+1; no simultaneous coverage claim."
    else:
        from puremacro.var.estimate import companion, estimate_var
        from puremacro.var.identify.cholesky import cholesky_factor, cholesky_svar
        y = frame.loc[:, list(spec.ordering)].to_numpy()
        fit_var = estimate_var(y, spec.lags)
        impact = cholesky_factor(fit_var.Sigma)
        k = spec.ordering.index(spec.shock)
        factor = float(spec.shock_size / impact[k, k])
        fit_irf = cholesky_svar(y, p=spec.lags, horizon=spec.horizon, n_boot=spec.n_boot, ci=spec.ci, seed=spec.seed)
        radius = float(np.max(np.abs(np.linalg.eigvals(companion(fit_var.A_list)))))
        numerical.update(spectral_radius=radius, impact_scale=factor, bootstrap_failures=fit_irf.n_fail)
        audit.append({"check": "var_stability", "status": "passed" if radius < 1 else "failed",
                      "detail": f"Companion spectral radius {radius:.12g}; threshold < 1."})
        audit.append({"check": "bootstrap", "status": "not_checked" if not spec.n_boot else ("failed" if fit_irf.n_fail else "passed"),
                      "detail": f"Requested {spec.n_boot}, rejected {fit_irf.n_fail} draws; existing engine drop/warn rules apply."})
        for response in spec.responses:
            j = spec.ordering.index(response)
            for h in range(spec.horizon + 1):
                row = dict(response=response, horizon=h, estimate=float(fit_irf.irf_point[h, j, k]) * factor,
                           n_obs=len(frame) - spec.lags, unit=units[response], shock_unit=units[spec.shock])
                if spec.n_boot:
                    lo, hi = sorted((float(fit_irf.irf_lower[h, j, k]) * factor, float(fit_irf.irf_upper[h, j, k]) * factor))
                    row.update(lower=lo, upper=hi)
                rows.append(row)
        samples = [{"horizon": h, "start": str(frame.index[spec.lags]), "end": str(frame.index[-1]),
                    "n_obs": len(frame) - spec.lags} for h in range(spec.horizon + 1)]
        inference = ("Residual-bootstrap pointwise percentile intervals; normalization fixed at the point-estimate impact, not renormalized per draw."
                     if spec.n_boot else "No uncertainty intervals requested (n_boot=0).")
    results = pd.DataFrame(rows)
    _finite(results, "results")
    audit.append({"check": "finite_output", "status": "passed", "detail": "Every exported numerical estimate and computed interval is finite."})
    frames = tuple((name, _encode(value, name)) for name, value in
                   (("inputs", raw), ("data", frame), ("results", results), ("audits", pd.DataFrame(audit))))
    caps = capabilities()
    specification = asdict(spec)
    manifest: dict[str, Any] = {
        "format": "puremacro-study", "schema_version": 1, "spec": specification,
        "spec_sha256": _hash(_json(specification).encode()),
        "frames": {name: {"sha256": _hash(payload), "bytes": len(payload)} for name, payload in frames},
        "versions": {"puremacro": __version__, "python": sys.version.split()[0],
                     "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
        "runtime": {"host": caps.host, "device": caps.device},
        "sample": {"raw_start": str(raw.index[0]), "raw_end": str(raw.index[-1]), "raw_n_obs": len(raw),
                   "transformed_start": str(frame.index[0]), "transformed_end": str(frame.index[-1]), "transformed_n_obs": len(frame),
                   "effective_by_horizon": samples},
        "assumptions": {"identification": spec.identification, "shock": spec.shock,
                        "shock_size": spec.shock_size, "shock_unit": units[spec.shock],
                        "response_convention": spec.response_convention, "frequency": spec.frequency,
                        "lags": f"Fixed {spec.lags}; no automatic lag selection.",
                        "regressors": ("constant, shock[t], lags of response/shock/controls, contemporaneous controls" if spec.method == "lp" else "constant and lags of all variables in the recursive ordering"),
                        "inference": inference, "sample_policy": "Common transformed input window; actual estimator windows recorded per horizon."},
        "numerical_diagnostics": numerical,
        "scope": "Conditional statistical responses; no economic-validity, causal-identification or global-accuracy certificate.",
    }
    manifest["record_sha256"] = _hash(_json(manifest).encode())
    return StudyResult(spec, frames, _json(manifest))


def compare_studies(*studies: StudyResult | ScenarioResult, comparison_note: str | None = None) -> StudyComparison:
    """Compare like estimands; reject mixed LP/VAR conventions or unlike inputs.

    Lags, confidence level, seed and bootstrap count may differ and remain
    visible. These choices can change effective sample sizes, which are shown
    for every result; this function does not claim equal fitted observations.
    """
    return StudyComparison(tuple(studies), comparison_note)


def _check_comparable(studies: tuple[StudyResult | ScenarioResult, ...], comparison_note: str | None) -> None:
    from ._scenario import ScenarioResult
    if len(studies) < 2 or not all(isinstance(s, (StudyResult, ScenarioResult)) for s in studies):
        raise ValueError("comparison requires at least two study/scenario result objects")
    has_scenario = any(isinstance(s, ScenarioResult) for s in studies)
    if has_scenario:
        _label(comparison_note, "scenario comparison_note")
    if len({s.spec.name for s in studies}) != len(studies):
        raise ValueError("comparison study names must be unique")
    first = studies[0]
    def contract(s: StudyResult | ScenarioResult) -> dict[str, Any]:
        d = asdict(s.spec)
        if has_scenario:
            d = {key: d[key] for key in ("variables", "responses", "shock", "frequency", "identification", "horizon", "shock_size")}
        else:
            for name in ("name", "lags", "seed", "ci", "n_boot", "sample_start", "sample_end"):
                d.pop(name)
        d["response_convention"] = s.spec.response_convention
        return d
    reference = contract(first)
    for study in studies[1:]:
        candidate = contract(study)
        different = [k for k in reference if reference[k] != candidate[k]]
        if different:
            raise ValueError(f"unlike comparison contracts: {', '.join(different)}; observed LP shocks and recursive VAR innovations are not interchangeable")
        for key in ("inputs", "data"):
            if dict(first._frames)[key] != dict(study._frames)[key]:
                raise ValueError("comparison requires identical raw inputs, transformations, metadata and sample dates")
