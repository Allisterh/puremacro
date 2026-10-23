"""Refit complete paths, preserving joint draws and failed attempts."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import copy
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from puremacro.runtime import store
from ..types import RiskIndex
from ._draws import _label, _number, _series


def _json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, ensure_ascii=False)


def _outcome(value: Any, names: pd.Index | None = None) -> pd.Series:
    if not isinstance(value, pd.Series) or value.empty or not value.index.is_unique:
        raise ValueError("refit must return a nonempty Series with unique outcome labels")
    if not all(isinstance(v, str) and v.strip() for v in value.index):
        raise ValueError("outcome labels must be nonempty strings")
    if names is not None and not value.index.equals(names):
        raise ValueError("refit changed outcome names or their order")
    _number(value.to_numpy(), "refit outcome")
    store.dumps_frame(value.to_frame("outcome"))  # No silent loss of callback provenance.
    return copy.deepcopy(value)


@dataclass(frozen=True)
class SignalPropagation:
    """Immutable joint refit record, not a total-uncertainty certificate.

    Failed attempts occupy all-NaN columns in ``draws`` and have explicit
    exception/status rows in ``attempts``. Summaries reject any failed attempt.
    Accessors return fresh copies; callbacks themselves are never retained.
    """
    _baseline: bytes
    _draws: bytes
    _attempts: bytes
    _metadata: str

    @property
    def baseline(self) -> pd.Series:
        return store.loads_frame(self._baseline).iloc[:, 0]

    @property
    def draws(self) -> pd.DataFrame:
        result = store.loads_frame(self._draws)
        result.columns = pd.MultiIndex.from_tuples([tuple(v) for v in self.metadata["draw_keys"]], names=["source", "draw_id"])
        return result

    @property
    def attempts(self) -> pd.DataFrame:
        return store.loads_frame(self._attempts)

    @property
    def metadata(self) -> dict[str, Any]:
        return json.loads(self._metadata)

    @property
    def passed(self) -> bool:
        """Every requested refit returned the same finite outcome coordinates."""
        return bool((self.attempts.status == "passed").all())

    def summary(self, source: str, *, ci: float = 0.90) -> pd.DataFrame:
        """Measurement percentiles for doc draws; envelopes for alternatives.

        Pointwise percentiles describe conditional document-measurement
        variation only. Kernel/lexicon/corpus values are labelled scenario
        envelopes and never treated as exchangeable probabilistic draws.
        No estimator standard errors or confidence intervals are combined.
        """
        if not self.passed:
            raise ValueError("cannot summarize incomplete refits; inspect attempts and retain failed draws")
        if isinstance(ci, bool) or not isinstance(ci, (int, float)) or not 0 < ci < 1:
            raise ValueError("ci must be between zero and one")
        draws = self.draws
        if source not in draws.columns.get_level_values("source"):
            raise ValueError("requested source is absent")
        selected = draws[source]
        if source == "doc":
            if self.metadata["source_semantics"].get("doc") != "conditional_document_measurement_bootstrap" or selected.shape[1] < 2:
                raise ValueError("document percentiles require declared measurement-bootstrap semantics and at least two draws")
            low, high = np.quantile(selected.to_numpy(), [(1 - ci) / 2, (1 + ci) / 2], axis=1)
            scope = "pointwise conditional document-measurement percentiles"
        else:
            low, high = selected.min(axis=1).to_numpy(), selected.max(axis=1).to_numpy()
            scope = "sensitivity envelope (not a probability interval)"
        result = pd.DataFrame({"baseline": self.baseline, "lower": low, "upper": high,
                               "n_draws": selected.shape[1], "unit": pd.Series(self.metadata["outcome_units"])})
        result.attrs = {"source": source, "scope": scope, "ci": ci if source == "doc" else None,
                        "outcome_convention": self.metadata["outcome_convention"],
                        "sampling_uncertainty_combined": False}
        return result


def propagate_signal(
    index: RiskIndex, refit: Callable[[pd.Series], pd.Series], *,
    model_label: str, outcome_units: Mapping[str, str], outcome_convention: str,
) -> SignalPropagation:
    """Apply a trusted refit callback once to baseline and every full signal path.

    The callback owns sample alignment, identification and any random seed. It
    must return the same ordered Series of finite point estimates for every
    path. Return point estimates, not estimator confidence-band endpoints.
    Every path is passed intact, keeping dependence across dates and outcomes.
    Exceptions/nonfinite outcomes are recorded, never silently dropped or
    replaced by baseline. Baseline failure raises before any draw is attempted.
    """
    if not isinstance(index, RiskIndex) or not callable(refit):
        raise TypeError("propagation requires a RiskIndex and callable refit")
    index = copy.deepcopy(index)
    _label(model_label, "model_label")
    _label(outcome_convention, "outcome_convention")
    _series(index.series, "baseline signal")
    index.__post_init__()  # RiskIndex is mutable: recheck alignment at this boundary.
    if index.draws is None or index.draws.empty or not index.draws.columns.is_unique:
        raise ValueError("propagation requires nonempty, uniquely labelled signal draws")
    keys = list(index.draws.columns)
    if any(not isinstance(k[1], (str, int)) or isinstance(k[1], bool) for k in keys):
        raise ValueError("draw IDs must be strings or integers")
    if not isinstance(outcome_units, Mapping) or not outcome_units:
        raise ValueError("outcome_units must declare each outcome")
    for key, unit in outcome_units.items():
        _label(key, "outcome name")
        _label(unit, "outcome unit")
    baseline_input = copy.deepcopy(index.series)
    baseline_input_frame = baseline_input.to_frame("signal")
    baseline_input_frame.attrs = copy.deepcopy(index.metadata)
    baseline_input_payload = store.dumps_frame(baseline_input_frame)
    flat_inputs = copy.deepcopy(index.draws)
    flat_inputs.columns = [f"draw_{i}" for i in range(len(keys))]
    input_payload = store.dumps_frame(flat_inputs)
    baseline = _outcome(refit(copy.deepcopy(baseline_input)))
    if set(outcome_units) != set(baseline.index):
        raise ValueError("outcome_units must match every baseline outcome exactly")
    attempts, values = [], []
    output_metadata = {}
    for i, (source, draw_id) in enumerate(keys):
        try:
            signal = index.draws.iloc[:, i].copy(deep=True)
            signal.name = index.series.name
            signal.attrs = copy.deepcopy(index.series.attrs)
            _series(signal, "signal draw")
            result = _outcome(refit(signal), baseline.index)
            values.append(result.to_numpy(dtype=float))
            output_metadata[f"draw_{i}"] = copy.deepcopy(result.attrs)
            attempts.append({"source": source, "draw_id": str(draw_id), "position": i,
                             "status": "passed", "error_type": "", "error": ""})
        except Exception as exc:
            values.append(np.full(len(baseline), np.nan))
            attempts.append({"source": source, "draw_id": str(draw_id), "position": i,
                             "status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
    output = pd.DataFrame(np.column_stack(values), index=baseline.index,
                          columns=[f"draw_{i}" for i in range(len(keys))])
    output.attrs = {"refit_metadata": output_metadata}
    provenance = index.metadata.get("signal_draws", {})
    info = {"schema_version": 1, "model_label": model_label,
        "outcome_units": dict(outcome_units), "outcome_convention": outcome_convention,
        "draw_keys": keys, "n_requested": len(keys), "n_failed": sum(r["status"] == "failed" for r in attempts),
        "source_semantics": provenance.get("source_semantics", {}),
        "signal_metadata": copy.deepcopy(index.metadata),
        "signal_sha256": hashlib.sha256(baseline_input_payload).hexdigest(),
        "input_draws_sha256": hashlib.sha256(input_payload).hexdigest(),
        "scope": "Joint conditional measurement/sensitivity refits. Sampling uncertainty, calibration uncertainty and causal validity are not combined or established."}
    # Provenance may contain safe timestamps and other runtime metadata. Keep
    # it losslessly in the baseline frame; the JSON summary names its hash.
    output_baseline = baseline.to_frame("baseline")
    output_baseline.attrs = {"signal_metadata": info.pop("signal_metadata"),
                            "signal_series_metadata": copy.deepcopy(index.series.attrs),
                            "refit_metadata": copy.deepcopy(baseline.attrs)}
    return SignalPropagation(store.dumps_frame(output_baseline), store.dumps_frame(output),
                             store.dumps_frame(pd.DataFrame(attempts)), _json(info))
