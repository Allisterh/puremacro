"""Record evaluated model scenarios without executing or serializing code."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import io
import math
from pathlib import Path
import re
import sys
from typing import Any
import zipfile

import numpy as np
import pandas as pd
import scipy

from puremacro import __version__
from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst
from puremacro.runtime import capabilities, store
from ._core import VariableSpec, _encode, _finite, _hash, _integer, _json, _label


@dataclass(frozen=True)
class ScenarioSpec:
    """Declared model estimand; labels are explicit, never inferred from output.

    ``variables`` describe final response/shock units and transformations as in
    StudySpec. Convention and identification strings remain open for calibrated
    static/transition scenarios. Matching an empirical study requires exact
    contract equality plus an explicit comparison note, not a model name alone.
    """
    name: str
    model: str
    variables: tuple[VariableSpec, ...]
    responses: tuple[str, ...]
    shock: str
    frequency: str
    identification: str
    response_convention: str
    horizon: int
    shock_size: float

    def __post_init__(self) -> None:
        for key in ("name", "model", "shock", "frequency", "identification", "response_convention"):
            _label(getattr(self, key), key)
        if isinstance(self.variables, str) or isinstance(self.responses, str):
            raise ValueError("variables/responses must be sequences")
        object.__setattr__(self, "variables", tuple(self.variables))
        object.__setattr__(self, "responses", tuple(self.responses))
        if not self.variables or not all(isinstance(v, VariableSpec) for v in self.variables):
            raise ValueError("variables must contain VariableSpec instances")
        names = [v.name for v in self.variables]
        if len(set(names)) != len(names) or self.shock not in names:
            raise ValueError("scenario variable names must be unique and include the shock")
        if not self.responses or any(not isinstance(v, str) for v in self.responses) or len(set(self.responses)) != len(self.responses) or not set(self.responses) <= set(names):
            raise ValueError("responses must be unique declared names")
        _integer(self.horizon, "horizon")
        if isinstance(self.shock_size, bool) or not isinstance(self.shock_size, (int, float)) or not math.isfinite(self.shock_size) or self.shock_size == 0:
            raise ValueError("shock_size must be finite and nonzero")


@dataclass(frozen=True)
class ScenarioResult:
    """Immutable evaluated scenario, its source, inputs, evidence and audit limits."""
    spec: ScenarioSpec
    _frames: tuple[tuple[str, bytes], ...]
    _source: str
    _manifest_json: str

    def __post_init__(self) -> None:
        if not isinstance(self.spec, ScenarioSpec) or not isinstance(self._source, str) or not isinstance(self._manifest_json, str):
            raise TypeError("invalid scenario snapshot")
        frames = tuple(tuple(item) for item in self._frames)
        if any(len(item) != 2 or not isinstance(item[0], str) or not isinstance(item[1], bytes) for item in frames):
            raise TypeError("scenario frames must be immutable byte snapshots")
        object.__setattr__(self, "_frames", frames)

    def _frame(self, name: str) -> pd.DataFrame:
        return store.loads_frame(dict(self._frames)[name])

    @property
    def inputs(self) -> pd.DataFrame:
        return self._frame("inputs")

    @property
    def data(self) -> pd.DataFrame:
        return self._frame("data")

    @property
    def audits(self) -> pd.DataFrame:
        return self._frame("audits")

    @property
    def evidence(self) -> dict[str, pd.DataFrame]:
        return {name.removeprefix("evidence."): self._frame(name) for name, _ in self._frames if name.startswith("evidence.")}

    @property
    def provenance(self) -> dict[str, Any]:
        return self._frame("provenance").attrs

    @property
    def model_source(self) -> str:
        return self._source

    @property
    def manifest(self) -> dict[str, Any]:
        import json
        return json.loads(self._manifest_json)

    def to_frame(self) -> pd.DataFrame:
        return self._frame("results")

    def to_markdown(self) -> str:
        meta = self.manifest
        assumptions = pd.DataFrame({"assumption": list(meta["assumptions"]), "value": list(meta["assumptions"].values())})
        return (f"Scenario: {self.spec.name}\n\n" + df_to_markdown(assumptions, index=False)
                + "\n\n" + df_to_markdown(self.to_frame(), index=False)
                + "\n\nRecorded diagnostics (scope supplied by the caller):\n\n" + df_to_markdown(self.audits, index=False)
                + f"\n\n{meta['scope']}\n\nRecord SHA-256: {meta['record_sha256']}\n")

    def to_latex(self) -> str:
        return df_to_latex(self.to_frame(), index=False)

    def to_typst(self) -> str:
        return df_to_typst(self.to_frame(), index=False)

    def to_json(self) -> str:
        return _json({"manifest": self.manifest, "results": self.to_frame().to_dict("records"), "audits": self.audits.to_dict("records")})

    def write_report(self, directory: str | Path) -> Path:
        from ._archive import save_study
        archive = save_study(self)
        assert isinstance(archive, bytes)
        contents = {"report.md": self.to_markdown(), "results.tex": self.to_latex(),
                    "results.typ": self.to_typst(), "results.csv": self.to_frame().to_csv(index=False),
                    "audits.csv": self.audits.to_csv(index=False), "manifest.json": self._manifest_json + "\n",
                    "model_source.txt": self._source}
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=False)
        for name, text in contents.items():
            (target / name).write_text(text, encoding="utf-8")
        (target / "study.pmstudy").write_bytes(archive)
        return target


def _audit_frame(audits: pd.DataFrame) -> None:
    if not isinstance(audits, pd.DataFrame) or audits.empty or set(audits) != {"check", "status", "detail"}:
        raise ValueError("audits require nonempty check/status/detail columns")
    if audits["check"].duplicated().any() or any(not isinstance(v, str) or not v.strip() for key in audits for v in audits[key]):
        raise ValueError("audit labels/details must be nonempty strings with unique checks")
    if not set(audits["status"]) <= {"passed", "failed", "not_checked", "unsupported"}:
        raise ValueError("invalid audit status")


def record_scenario(
    outputs: pd.DataFrame, spec: ScenarioSpec, *, inputs: pd.DataFrame,
    model_source: str, parameters: Mapping[str, float], audits: pd.DataFrame,
    provenance: Mapping[str, Any], transformed_inputs: pd.DataFrame | None = None,
    evidence: Mapping[str, pd.DataFrame] | None = None,
) -> ScenarioResult:
    """Snapshot already evaluated model outputs; never invoke a solver or callback.

    Outputs contain exactly response/horizon/estimate, in response-major order
    with horizons 0..H. Source is retained as inert text. Parameters are finite
    scalars; general calibration matrices, full joint covariance, fitted/held-out
    targets or paths belong in named evidence frames. This function validates
    recording integrity, not the truth of supplied audit or identification claims.
    """
    if not isinstance(spec, ScenarioSpec):
        raise TypeError("spec must be a ScenarioSpec")
    _label(model_source, "model_source")
    if not isinstance(outputs, pd.DataFrame) or set(outputs) != {"response", "horizon", "estimate"}:
        raise ValueError("scenario outputs require exactly response/horizon/estimate columns")
    if len(outputs) != len(spec.responses) * (spec.horizon + 1):
        raise ValueError("scenario outputs do not cover the declared horizons/responses")
    coordinates = [(r, h) for r in spec.responses for h in range(spec.horizon + 1)]
    if list(zip(outputs.response, outputs.horizon)) != coordinates:
        raise ValueError("scenario coordinates are missing, reordered or duplicated")
    for key in ("estimate", "horizon"):
        if not pd.api.types.is_numeric_dtype(outputs[key]) or pd.api.types.is_bool_dtype(outputs[key]):
            raise ValueError("scenario estimates/horizons must be real numeric values")
    _audit_frame(audits)
    if not isinstance(parameters, Mapping) or not isinstance(provenance, Mapping):
        raise ValueError("parameters and provenance must be mappings")
    parameter_values = {}
    for key, value in parameters.items():
        _label(key, "parameter name")
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not math.isfinite(value):
            raise ValueError("parameters require finite real scalar values; use evidence frames for arrays")
        parameter_values[key] = float(value)
    if transformed_inputs is None and any(v.transform != "identity" or v.scale != 1 for v in spec.variables):
        raise ValueError("nonidentity variable definitions require explicit transformed_inputs")
    data = inputs if transformed_inputs is None else transformed_inputs
    units = {v.name: v.unit for v in spec.variables}
    output = outputs.copy(deep=True)
    output["unit"] = [units[r] for r, _ in coordinates]
    output["shock_unit"] = units[spec.shock]
    provenance_frame = pd.DataFrame({"record": ["scenario provenance"]})
    provenance_frame.attrs = dict(provenance)
    frames = {"inputs": inputs, "data": data, "results": output, "audits": audits, "provenance": provenance_frame}
    for name, frame in (evidence or {}).items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("evidence names may contain only letters, numbers, underscores and hyphens")
        frames["evidence." + name] = frame
    for name, frame in frames.items():
        if not isinstance(frame, pd.DataFrame) or frame.empty or not frame.columns.is_unique:
            raise ValueError(f"{name} must be a nonempty DataFrame with unique columns")
    encoded = tuple((name, _encode(frame, name)) for name, frame in sorted(frames.items()))
    caps = capabilities()
    specification = asdict(spec)
    manifest = {"format": "puremacro-scenario", "schema_version": 1, "spec": specification,
        "spec_sha256": _hash(_json(specification).encode()), "model_source_sha256": _hash(model_source.encode()),
        "parameters": parameter_values,
        "frames": {name: {"sha256": _hash(payload), "bytes": len(payload)} for name, payload in encoded},
        "versions": {"puremacro": __version__, "python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__},
        "runtime": {"host": caps.host, "device": caps.device},
        "assumptions": {"model": spec.model, "identification": spec.identification,
                        "frequency": spec.frequency, "response_convention": spec.response_convention,
                        "shock": spec.shock, "shock_size": spec.shock_size, "shock_unit": units[spec.shock],
                        "inference": "Evaluated model/scenario outputs; no sampling intervals inferred from these paths."},
        "scope": "Model outputs and supplied audits are recorded, not independently certified. Explicit matching assumptions are required for empirical comparison."}
    manifest["record_sha256"] = _hash(_json(manifest).encode())
    return ScenarioResult(spec, encoded, model_source, _json(manifest))


def _validate_scenario(result: ScenarioResult) -> None:
    manifest = result.manifest
    expected = {"format", "schema_version", "spec", "spec_sha256", "model_source_sha256", "parameters", "frames", "versions", "runtime", "assumptions", "scope", "record_sha256"}
    if set(manifest) != expected or manifest["format"] != "puremacro-scenario" or type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError("unsupported or incomplete scenario manifest")
    digest = manifest.pop("record_sha256")
    if digest != _hash(_json(manifest).encode()) or manifest["model_source_sha256"] != _hash(result._source.encode()):
        raise ValueError("scenario source/manifest hash mismatch")
    if _json(asdict(result.spec)) != _json(manifest["spec"]) or manifest["spec_sha256"] != _hash(_json(manifest["spec"]).encode()):
        raise ValueError("scenario specification hash mismatch")
    frames = dict(result._frames)
    required = {"inputs", "data", "results", "audits", "provenance"}
    if len(frames) != len(result._frames) or not required <= set(frames) or set(frames) != set(manifest["frames"]):
        raise ValueError("scenario frame inventory is incomplete or duplicated")
    evidence = {}
    decoded = {}
    for name, payload in frames.items():
        if manifest["frames"][name] != {"sha256": _hash(payload), "bytes": len(payload)}:
            raise ValueError("scenario frame hash mismatch")
        frame = store.loads_frame(payload)
        _finite(frame, name)
        decoded[name] = frame
        if name not in required:
            if not name.startswith("evidence."):
                raise ValueError("unknown scenario frame")
            evidence[name.removeprefix("evidence.")] = frame
    results = decoded["results"]
    if list(decoded["provenance"].columns) != ["record"] or decoded["provenance"]["record"].tolist() != ["scenario provenance"]:
        raise ValueError("invalid scenario provenance frame")
    if set(results) != {"response", "horizon", "estimate", "unit", "shock_unit"}:
        raise ValueError("invalid scenario result schema")
    # Reuse boundary validation without executing model code, and verify that
    # supplied units/assumptions agree with the declared contract.
    rebuilt = record_scenario(results.drop(columns=["unit", "shock_unit"]), result.spec,
        inputs=decoded["inputs"], transformed_inputs=decoded["data"], model_source=result._source,
        parameters=manifest["parameters"], audits=decoded["audits"], provenance=decoded["provenance"].attrs, evidence=evidence)
    if rebuilt.to_frame()[["unit", "shock_unit"]].to_dict("list") != results[["unit", "shock_unit"]].to_dict("list"):
        raise ValueError("scenario output units do not match specification")
    if manifest["assumptions"] != rebuilt.manifest["assumptions"]:
        raise ValueError("scenario assumptions do not match specification")


def _save_scenario(result: ScenarioResult, path: str | Path | None) -> bytes | Path:
    _validate_scenario(result)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        contents = {"manifest.json": result._manifest_json.encode(), "model_source.txt": result._source.encode(),
                    **{name + ".npz": payload for name, payload in result._frames}}
        for name, payload in sorted(contents.items()):
            archive.writestr(zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0)), payload)
    payload = output.getvalue()
    if path is None:
        return payload
    target = Path(path)
    with target.open("xb") as stream:
        stream.write(payload)
    return target


def _load_scenario(payload: bytes, max_bytes: int, manifest: dict[str, Any]) -> ScenarioResult:
    from ._archive import _members, _check_frame_payload
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = _members(archive, max_bytes)
        frame_names = manifest["frames"]
        if names != {"manifest.json", "model_source.txt", *(name + ".npz" for name in frame_names)}:
            raise ValueError("scenario member inventory does not match manifest")
        frames = tuple((name, archive.read(name + ".npz")) for name in sorted(frame_names))
        source = archive.read("model_source.txt").decode("utf-8")
    remaining = max_bytes
    for _, frame in frames:
        remaining -= _check_frame_payload(frame, remaining)
    raw_spec = manifest["spec"]
    spec = ScenarioSpec(**{**raw_spec, "variables": tuple(VariableSpec(**v) for v in raw_spec["variables"])})
    result = ScenarioResult(spec, frames, source, _json(manifest))
    _validate_scenario(result)
    return result
