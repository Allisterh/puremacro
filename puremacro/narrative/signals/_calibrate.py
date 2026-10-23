"""Calibration trained on declared dates and evaluated on separate dates."""
from __future__ import annotations

from dataclasses import asdict
import copy
import hashlib
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from puremacro.runtime import store
from ..types import RiskIndex, SignalQualityReport, BenchmarkScore, SurveyScore, EventPanelScore
from ._draws import _label, _series, _window


def _aligned(index: RiskIndex, target: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(index, RiskIndex):
        raise TypeError("index must be a RiskIndex")
    x, y = _series(index.series, "signal"), _series(target, "target")
    if not target.index.equals(index.series.index):
        raise ValueError("target must have exactly the signal dates; no silent inner join or missing-value drop")
    return x, y


def _quality(index: RiskIndex) -> SignalQualityReport:
    if index.quality is not None:
        return copy.deepcopy(index.quality)
    return SignalQualityReport(pd.Series(dtype="int64"), pd.Series(dtype=float), [])


def _correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2:
        return None
    def centered(values: np.ndarray) -> np.ndarray:
        # Subtract a representable observed anchor before scaling, retaining
        # small representable variation around a large level. Opposite extreme
        # signs can overflow subtraction, so only that case scales first.
        with np.errstate(over="ignore", invalid="ignore"):
            delta = values - values[0]
        if not np.isfinite(delta).all():
            scaled = values / max(float(np.max(np.abs(values))), 1.0)
            delta = scaled - scaled[0]
        scale = float(np.max(np.abs(delta)))
        return delta / scale if scale else delta
    x, y = centered(x), centered(y)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def calibrate_signal(
    index: RiskIndex, target: pd.Series, *, name: str, target_unit: str,
    train_window: tuple[str, str], evaluation_window: tuple[str, str],
    kind: str = "benchmark", mapping: str = "affine",
) -> RiskIndex:
    """Return a copied index calibrated to a benchmark or survey.

    Affine coefficients use train dates only; evaluation must follow training
    and have no overlap. The target must already be aligned exactly. ``none``
    evaluates original units without fitting. Normalized inputs require known
    normalization provenance; only an explicit full-sample normalization may
    use evaluation observations, and that choice stays labelled in metadata.
    Draws receive the same fixed affine map; parameter-estimation uncertainty
    is not added. Unrelated source metadata and quality diagnostics survive.
    """
    _label(name, "calibration name")
    _label(target_unit, "target_unit")
    if kind not in {"benchmark", "survey"} or mapping not in {"affine", "none"}:
        raise ValueError("kind must be benchmark/survey and mapping must be affine/none")
    x, y = _aligned(index, target)
    train = _window(index.series.index, train_window, "train_window")
    evaluation = _window(index.series.index, evaluation_window, "evaluation_window")
    if train.sum() < 3 or evaluation.sum() < 3:
        raise ValueError("training and evaluation each require at least three observations")
    if (train & evaluation).any() or np.flatnonzero(train)[-1] >= np.flatnonzero(evaluation)[0]:
        raise ValueError("evaluation must strictly follow training without overlap")
    provenance = copy.deepcopy(index.metadata.get("signal_draws", {}))
    reference = provenance.get("normalization_reference", {})
    if index.normalization != "raw":
        if reference.get("policy") not in {"declared_window", "explicit_full_sample"}:
            raise ValueError("normalized signal needs explicit normalization provenance before calibration")
        if reference["policy"] != "explicit_full_sample":
            end = pd.Timestamp(reference["end"])
            first_eval = index.series.index[np.flatnonzero(evaluation)[0]]
            if isinstance(first_eval, pd.Period):
                first_eval = first_eval.start_time
            if end >= first_eval:
                raise ValueError("normalization window overlaps evaluation; use training-only normalization")
    intercept, slope = 0.0, 1.0
    if mapping == "affine":
        design = np.column_stack([np.ones(int(train.sum())), x[train]])
        if np.linalg.matrix_rank(design) < 2:
            raise ValueError("training signal has no variation for affine calibration")
        intercept, slope = map(float, np.linalg.lstsq(design, y[train], rcond=None)[0])
    calibrated = intercept + slope * x
    if not np.isfinite(calibrated).all():
        raise ValueError("calibration produced nonfinite values")
    error = calibrated[evaluation] - y[evaluation]
    if not np.isfinite(error).all():
        raise ValueError("calibration evaluation produced nonfinite errors")
    error_scale = float(np.max(np.abs(error))) or 1.0
    relative_error = error / error_scale
    cls = BenchmarkScore if kind == "benchmark" else SurveyScore
    score = cls(int(evaluation.sum()), _correlation(calibrated[evaluation], y[evaluation]),
        _correlation(rankdata(calibrated[evaluation]), rankdata(y[evaluation])),
        float(error_scale * np.sqrt(np.mean(relative_error ** 2))),
        float(error_scale * np.mean(np.abs(relative_error))), float(error_scale * np.mean(relative_error)), target_unit)
    result = copy.deepcopy(index)
    result.series = pd.Series(calibrated, index=index.series.index, name=index.series.name)
    result.series.attrs = copy.deepcopy(index.series.attrs)
    if index.draws is not None:
        index.__post_init__()
        if not np.isfinite(index.draws.to_numpy(dtype=float)).all():
            raise ValueError("calibration cannot discard nonfinite measurement draws")
        with np.errstate(over="ignore", invalid="ignore"):
            result.draws = intercept + slope * index.draws.copy(deep=True)
        if not np.isfinite(result.draws.to_numpy(dtype=float)).all():
            raise ValueError("calibration produced nonfinite transformed draws")
    # Once mapped to target units, the series is no longer a z-score or BBD-100
    # index. Retain the complete original normalization in calibration metadata.
    result.normalization = "raw"
    result.quality = _quality(index)
    if result.quality.doc_bootstrap_sd is not None:
        with np.errstate(over="ignore", invalid="ignore"):
            result.quality.doc_bootstrap_sd *= abs(slope)
        if not np.isfinite(result.quality.doc_bootstrap_sd.to_numpy(dtype=float)).all():
            raise ValueError("calibration produced nonfinite document bootstrap SD")
    if result.quality.corpus_loo_max_swing is not None:
        with np.errstate(over="ignore", invalid="ignore"):
            result.quality.corpus_loo_max_swing *= abs(slope)
        if not np.isfinite(result.quality.corpus_loo_max_swing.to_numpy(dtype=float)).all():
            raise ValueError("calibration produced nonfinite corpus sensitivity")
    if kind == "benchmark":
        result.quality.benchmark_scores[name] = score
    else:
        result.quality.survey_scores[name] = score
    paired = pd.DataFrame({"signal": x, "target": y}, index=index.series.index)
    details: dict[str, Any] = {"kind": kind, "mapping": mapping, "intercept": intercept, "slope": slope,
        "target_unit": target_unit, "train_window": list(train_window), "evaluation_window": list(evaluation_window),
        "n_train": int(train.sum()), "score": asdict(score), "input_normalization": index.normalization,
        "normalization_reference": reference, "paired_input_sha256": hashlib.sha256(store.dumps_frame(paired)).hexdigest(),
        "scope": "Held-out agreement; fixed fitted affine map applied to all measurement draws. Calibration-parameter uncertainty is not propagated."}
    history = copy.deepcopy(result.metadata.get("calibrations", {}))
    if not isinstance(history, dict) or name in history:
        raise ValueError("calibration names must be unique; existing provenance is not overwritten")
    history[name] = details
    result.metadata["calibrations"] = history
    result.quality.metadata.setdefault("calibrations", {})[name] = details
    return result


def evaluate_events(
    index: RiskIndex, events: pd.Series, *, evaluation_window: tuple[str, str],
    polarity: int = 1,
) -> RiskIndex:
    """Attach binary-event ranking on declared evaluation dates to a copied index.

    ``polarity`` explicitly states whether high (+1) or low (-1) scores predict
    an event. AUC handles ties by average ranks; top-decile membership includes
    every tie at the cutoff, so ``top_n`` may exceed ceil(n/10). This evaluates
    a supplied event panel, not an automatically selected or fitted classifier.
    """
    if type(polarity) is not int or polarity not in {-1, 1}:
        raise ValueError("polarity must be +1 or -1")
    x, y = _aligned(index, events)
    if not np.isin(y, [0, 1]).all():
        raise ValueError("events must be explicitly numeric binary values (0 or 1)")
    mask = _window(index.series.index, evaluation_window, "evaluation_window")
    first, last = index.series.index[mask][0], index.series.index[mask][-1]
    history = index.metadata.get("calibrations", {})
    if not isinstance(history, dict):
        raise ValueError("invalid calibration history")
    for details in history.values():
        training = details["train_window"]
        if isinstance(index.series.index, pd.PeriodIndex):
            start, end = (pd.Period(v, freq=index.series.index.freq) for v in training)
        else:
            start, end = (pd.Timestamp(v) for v in training)
        if start <= last and end >= first:
            raise ValueError("event evaluation overlaps a known calibration training window")
    x, y = polarity * x[mask], y[mask]
    if len(x) < 3:
        raise ValueError("event evaluation requires at least three observations")
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    auc = None if not n1 or not n0 else float((rankdata(x)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    cutoff = np.sort(x)[-max(1, int(np.ceil(len(x) / 10)))]
    top = x >= cutoff
    score = EventPanelScore(len(x), n1, _correlation(rankdata(x), rankdata(y)), auc,
                            float(y[top].mean()), int(top.sum()), polarity)
    result = copy.deepcopy(index)
    result.quality = _quality(index)
    if result.quality.event_panel is not None:
        raise ValueError("existing event-panel evidence must not be overwritten")
    result.quality.event_panel = score
    result.quality.metadata["event_evaluation"] = {"window": list(evaluation_window), "polarity": polarity,
        "event_sha256": hashlib.sha256(store.dumps_frame(events.to_frame("event"))).hexdigest(),
        "evaluation_independence": "known_calibration_training_excluded" if history else "unverified_no_training_provenance",
        "scope": "Descriptive ranking; known calibration training excluded, other model/score selection independence unverified. No causal, probability-calibration or significance claim."}
    return result
