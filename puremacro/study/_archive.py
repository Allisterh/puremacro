"""Versioned, bounded study archives. No pickle, code imports or model execution."""
from __future__ import annotations

from dataclasses import asdict
import io
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any
import zipfile

import numpy as np
import pandas as pd

from puremacro.runtime import store
from ._core import StudyResult, StudySpec, VariableSpec, _finite, _hash, _json, _expected_grid

if TYPE_CHECKING:
    from ._scenario import ScenarioResult

_FRAMES = ("inputs", "data", "results", "audits")
_LIMIT = 64 * 1024 * 1024


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def _members(archive: zipfile.ZipFile, max_bytes: int) -> set[str]:
    entries = archive.infolist()
    names = [e.filename for e in entries]
    if len(names) != len(set(names)) or any(e.is_dir() or e.flag_bits & 1 for e in entries):
        raise ValueError("duplicate, directory or encrypted archive member")
    if any(n.startswith(("/", "\\")) or "\\" in n or any(p in {"", ".", ".."} for p in n.split("/")) for n in names):
        raise ValueError("noncanonical archive member name")
    if sum(e.file_size for e in entries) > max_bytes:
        raise ValueError("archive exceeds uncompressed size limit")
    return set(names)


def _check_frame_payload(payload: bytes, max_bytes: int) -> int:
    """Bound NPY allocations as well as ZIP expansion before NumPy decodes.

    A tiny NPY member can advertise a huge shape; ZIP size limits alone do
    not prevent NumPy allocating that shape before discovering truncated data.
    The store writes one-dimensional, non-object arrays in NPY v1/v2 format.
    """
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = _members(archive, max_bytes)
        if "__puremacro_schema__.npy" not in names or any(not n.endswith(".npy") or "/" in n for n in names):
            raise ValueError("invalid frame member inventory")
        for entry in archive.infolist():
            with archive.open(entry) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("unsupported frame array format")
                if len(shape) != 1 or dtype.hasobject or dtype.fields is not None or dtype.itemsize < 1:
                    raise ValueError("frame arrays require one-dimensional fixed-width non-object values")
                size = math.prod(shape) * dtype.itemsize
                if size > max_bytes or size != entry.file_size - stream.tell():
                    raise ValueError("frame array shape exceeds its payload or size limit")
        total = sum(item.file_size for item in archive.infolist())
    # The runtime store also reads v1 frames, which have no attrs. A study
    # archive must use v2 so provenance can never disappear silently.
    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
        schema = json.loads(bytes(arrays["__puremacro_schema__"]).decode("utf-8"),
                            object_pairs_hook=_pairs, parse_constant=_reject_constant)
        if schema.get("version") != 2 or "attrs" not in schema:
            raise ValueError("study frame requires version-2 metadata with attrs")
    return total


def _validate(result: StudyResult) -> None:
    manifest = result.manifest
    if manifest.get("format") != "puremacro-study" or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("unsupported study archive format/version")
    expected_keys = {"format", "schema_version", "spec", "spec_sha256", "frames", "versions", "runtime", "sample",
                     "assumptions", "numerical_diagnostics", "scope", "record_sha256"}
    if set(manifest) != expected_keys:
        raise ValueError("study manifest has missing or unknown fields")
    record = manifest.pop("record_sha256")
    if record != _hash(_json(manifest).encode()):
        raise ValueError("study manifest hash mismatch")
    if _json(asdict(result.spec)) != _json(manifest["spec"]) or manifest["spec_sha256"] != _hash(_json(manifest["spec"]).encode()):
        raise ValueError("study specification hash mismatch")
    if len(result._frames) != 4 or set(dict(result._frames)) != set(_FRAMES) or set(manifest["frames"]) != set(_FRAMES):
        raise ValueError("study requires exactly inputs, data, results and audits")
    decoded = {}
    for name, payload in result._frames:
        if manifest["frames"][name] != {"sha256": _hash(payload), "bytes": len(payload)}:
            raise ValueError(f"study frame hash mismatch: {name}")
        frame = store.loads_frame(payload)
        if frame.empty or not frame.columns.is_unique:
            raise ValueError(f"empty or duplicate-column study frame: {name}")
        _finite(frame, name)
        decoded[name] = frame
    spec = result.spec
    for name in ("inputs", "data"):
        frame = decoded[name]
        if list(frame.columns) != [v.name for v in spec.variables] or not isinstance(frame.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            raise ValueError(f"invalid {name} schema")
        if any(not pd.api.types.is_numeric_dtype(frame[c]) or pd.api.types.is_bool_dtype(frame[c]) for c in frame):
            raise ValueError(f"{name} must contain real numeric values")
        if frame.index.hasnans or not frame.index.is_unique or not frame.index.is_monotonic_increasing or not frame.index.equals(_expected_grid(frame.index, spec.frequency)):
            raise ValueError(f"invalid {name} sample dates/frequency")
    raw, data = decoded["inputs"], decoded["data"]
    first = int(any(v.transform in {"diff", "logdiff"} for v in spec.variables))
    if not data.index.equals(raw.index[first:]):
        raise ValueError("transformed sample does not match the declared differencing policy")
    results = decoded["results"]
    cols = {"response", "horizon", "estimate", "n_obs", "unit", "shock_unit"}
    if spec.method == "lp":
        cols |= {"lower", "upper", "se"}
    elif spec.n_boot:
        cols |= {"lower", "upper"}
    if set(results) != cols:
        raise ValueError("invalid numerical result schema")
    if len(results) != len(spec.responses) * (spec.horizon + 1) or max(1, spec.lags) >= len(data) or (spec.method == "lp" and max(1, spec.lags) + spec.horizon >= len(data)):
        raise ValueError("result dimensions exceed the input sample")
    expected = [(response, h) for response in spec.responses for h in range(spec.horizon + 1)]
    if list(zip(results["response"], results["horizon"])) != expected:
        raise ValueError("missing, duplicated or reordered result coordinates")
    numeric = cols - {"response", "unit", "shock_unit"}
    if any(not pd.api.types.is_numeric_dtype(results[c]) for c in numeric):
        raise ValueError("numerical result cells must be numeric")
    if (results["n_obs"] <= 0).any() or not np.equal(results["n_obs"], np.floor(results["n_obs"])).all():
        raise ValueError("invalid result sample counts")
    units = {v.name: v.unit for v in spec.variables}
    if results["unit"].tolist() != [units[r] for r, _ in expected] or set(results["shock_unit"]) != {units[spec.shock]}:
        raise ValueError("result units do not match the specification")
    offset = max(1, spec.lags) if spec.method == "lp" else spec.lags
    samples = [{"horizon": h, "start": str(data.index[offset]),
                "end": str(data.index[-h - 1] if spec.method == "lp" else data.index[-1]),
                "n_obs": len(data) - offset - (h if spec.method == "lp" else 0)}
               for h in range(spec.horizon + 1)]
    sample = {"raw_start": str(raw.index[0]), "raw_end": str(raw.index[-1]), "raw_n_obs": len(raw),
              "transformed_start": str(data.index[0]), "transformed_end": str(data.index[-1]),
              "transformed_n_obs": len(data), "effective_by_horizon": samples}
    if manifest["sample"] != sample or results["n_obs"].tolist() != [samples[h]["n_obs"] for _, h in expected]:
        raise ValueError("sample manifest/counts do not match input dates and estimator conventions")
    assumptions = manifest["assumptions"]
    for key, value in {"identification": spec.identification, "shock": spec.shock, "shock_size": spec.shock_size,
                       "shock_unit": units[spec.shock], "response_convention": spec.response_convention,
                       "frequency": spec.frequency}.items():
        if assumptions.get(key) != value:
            raise ValueError(f"assumption {key} does not match the specification")
    if "lower" in results and (results["lower"] > results["upper"]).any():
        raise ValueError("interval bounds are reversed")
    audits = decoded["audits"]
    if set(audits) != {"check", "status", "detail"} or not set(audits["status"]) <= {"passed", "failed", "not_checked", "unsupported"}:
        raise ValueError("invalid audit schema/status")
    if audits["check"].duplicated().any() or not {"identification", "input_and_design", "finite_output"} <= set(audits["check"]):
        raise ValueError("required audit evidence is missing or duplicated")


def save_study(result: StudyResult | ScenarioResult, path: str | Path | None = None) -> bytes | Path:
    """Write a deterministic archive, or return its bytes for an in-memory runtime.

    Existing files are never overwritten. Checksums detect changed content;
    they do not authenticate the author or establish scientific validity.
    """
    from ._scenario import ScenarioResult, _save_scenario
    if isinstance(result, ScenarioResult):
        return _save_scenario(result, path)
    if not isinstance(result, StudyResult):
        raise TypeError("result must be a StudyResult")
    _validate(result)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        contents = {"manifest.json": result._manifest_json.encode("utf-8"),
                    **{name + ".npz": payload for name, payload in result._frames}}
        for name, payload in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            archive.writestr(info, payload)
    payload = output.getvalue()
    if path is None:
        return payload
    target = Path(path)
    with target.open("xb") as stream:
        stream.write(payload)
    return target


def load_study(source: str | Path | bytes, *, max_bytes: int = _LIMIT) -> StudyResult | ScenarioResult:
    """Verify and load a study without executing estimators or user code.

    The compressed input, outer archive and cumulative inner frame payloads
    are bounded by ``max_bytes``. Members are read in memory, never extracted.
    Frame codecs retain their metadata using the runtime store's allowlist.
    """
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    if isinstance(source, bytes):
        payload = source
    else:
        path = Path(source)
        if path.stat().st_size > max_bytes:
            raise ValueError("archive exceeds compressed size limit")
        payload = path.read_bytes()
    if len(payload) > max_bytes:
        raise ValueError("archive exceeds compressed size limit")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = _members(archive, max_bytes)
            if "manifest.json" not in names:
                raise ValueError("archive is missing its manifest")
            manifest = json.loads(archive.read("manifest.json"), object_pairs_hook=_pairs,
                                  parse_constant=_reject_constant)
            if manifest.get("format") == "puremacro-scenario":
                from ._scenario import _load_scenario
                return _load_scenario(payload, max_bytes, manifest)
            if names != {"manifest.json", *(n + ".npz" for n in _FRAMES)}:
                raise ValueError("study archive member set does not match its schema")
            frames = tuple((name, archive.read(name + ".npz")) for name in _FRAMES)
        remaining = max_bytes
        for _, frame_payload in frames:
            remaining -= _check_frame_payload(frame_payload, remaining)
        raw_spec = manifest["spec"]
        spec = StudySpec(**{**raw_spec, "variables": tuple(VariableSpec(**v) for v in raw_spec["variables"])})
        result = StudyResult(spec, frames, _json(manifest))
        _validate(result)
    except (KeyError, IndexError, AttributeError, TypeError, UnicodeDecodeError, zipfile.BadZipFile, OSError, OverflowError, RecursionError) as exc:
        raise ValueError(f"invalid study archive: {exc}") from exc
    return result
