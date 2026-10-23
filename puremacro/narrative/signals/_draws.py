"""Mean-score indices with explicit resampling units and fixed normalization."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import copy
import hashlib
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from puremacro.runtime import store
from ..types import RiskIndex, SignalQualityReport


def _number(values: Any, label: str) -> np.ndarray:
    original = np.asarray(values)
    if original.dtype.kind not in "fiu" or original.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional real numeric values")
    values = original.astype(float)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains nonfinite values")
    return values


def _integer(value: Any, label: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")


def _label(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be an explicit nonempty string")


def _window(index: pd.Index, window: tuple[str, str], label: str) -> np.ndarray:
    if not isinstance(window, tuple) or len(window) != 2 or any(not isinstance(v, str) for v in window):
        raise ValueError(f"{label} must be a (start, end) tuple of exact date strings")
    dates = [pd.Period(v, freq=index.freq) if isinstance(index, pd.PeriodIndex) else pd.Timestamp(v) for v in window]
    if any(v not in index for v in dates) or dates[0] > dates[1]:
        raise ValueError(f"{label} bounds must identify ordered observations")
    return np.asarray((index >= dates[0]) & (index <= dates[1]))


def _series(series: pd.Series, label: str) -> np.ndarray:
    if not isinstance(series, pd.Series) or series.empty or not isinstance(series.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        raise ValueError(f"{label} requires a nonempty dated Series")
    if series.index.hasnans or not series.index.is_unique or not series.index.is_monotonic_increasing:
        raise ValueError(f"{label} dates must be unique and increasing")
    return _number(series.to_numpy(), label)


def score_documents(records: Iterable[tuple], kernel: Callable, *, score_name: str = "score") -> pd.DataFrame:
    """Adapt an existing per-document kernel without changing its score semantics.

    Require one ``(date, score)`` per four-field input record, in input order.
    Preserve corpus/cluster/weight metadata when supplied for every record.
    The returned table can be joined with alternative kernels by its record ID.
    Callbacks are trusted code executed now, never serialized or auto-imported.
    """
    _label(score_name, "score_name")
    if score_name in {"date", "n_tokens", "corpus", "cluster", "weight"}:
        raise ValueError("score_name collides with a document metadata column")
    materialized = copy.deepcopy(list(records))
    if not materialized or any(len(r) != 4 or not isinstance(r[1], str) or not isinstance(r[3], Mapping) for r in materialized):
        raise ValueError("records require (date, text, source_url, metadata) four-tuples")
    dates = pd.DatetimeIndex([pd.Timestamp(r[0]) for r in materialized])
    if dates.hasnans or dates.tz is not None:
        raise ValueError("document dates must be finite timezone-naive timestamps")
    points = list(kernel(copy.deepcopy(materialized)))
    if len(points) != len(materialized) or any(len(p) != 2 for p in points):
        raise ValueError("kernel must return exactly one (date, score) per document")
    if not pd.DatetimeIndex([p[0] for p in points]).equals(dates):
        raise ValueError("kernel reordered or changed document dates")
    frame = pd.DataFrame({"date": dates, score_name: _number([p[1] for p in points], "scores"),
                          "n_tokens": [len(r[1].split()) for r in materialized]})
    frame.index.name = "document_id"
    for key in ("corpus", "cluster", "weight"):
        present = [key in r[3] for r in materialized]
        if any(present) and not all(present):
            raise ValueError(f"document metadata {key} must be present for every record")
        if all(present):
            frame[key] = [r[3][key] for r in materialized]
    return frame


def build_signal_draws(
    scores: pd.DataFrame, *, name: str, country: str, language: str,
    method: str, corpus: str, resampling: str,
    n_draws: int = 199, seed: int = 0, frequency: str = "Q",
    score_column: str = "score", weight_column: str | None = None,
    cluster_column: str | None = None, corpus_column: str | None = None,
    alternatives: Mapping[str, Mapping[str, str]] | None = None,
    normalization: str = "raw", normalization_window: tuple[str, str] | str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RiskIndex:
    """Build a mean-score signal and labelled measurement/sensitivity paths.

    ``resampling`` is explicit: ``within_period`` independently resamples the
    same number of documents in each period, or ``cluster`` resamples persistent
    clusters once per draw and applies the same weights to every date. Cluster
    mode requires each cluster in every period. Positive document weights, if
    declared, stay attached to each sampled document and enter both numerator
    and denominator. Dates are never shuffled or replaced.

    ``alternatives={'kernel': {'label': 'column'}, 'lexicon': {...}}`` uses
    precomputed scores on the same documents. ``corpus_column`` adds leave-one-
    corpus-out paths. These are sensitivity scenarios, not random draws.
    Nonraw normalization requires a declared window or ``'full_sample'``;
    baseline mean/std are then fixed for all draws. No silent regularization.
    """
    for key, value in {"name": name, "country": country, "language": language, "corpus": corpus}.items():
        _label(value, key)
    _integer(n_draws, "n_draws")
    _integer(seed, "seed")
    if n_draws == 1:
        raise ValueError("n_draws must be zero or at least two")
    if resampling not in {"within_period", "cluster"}:
        raise ValueError("resampling must be within_period or cluster")
    if not isinstance(scores, pd.DataFrame) or scores.empty or not scores.columns.is_unique or not scores.index.is_unique:
        raise ValueError("scores require a nonempty DataFrame with unique document IDs and columns")
    if "date" not in scores or score_column not in scores:
        raise ValueError("scores require date and the declared score column")
    if "weight" in scores and weight_column is None:
        raise ValueError("declare weight_column explicitly; document weights cannot be silently ignored")
    if resampling == "cluster" and cluster_column is None:
        raise ValueError("cluster resampling requires cluster_column")
    if resampling != "cluster" and cluster_column is not None:
        raise ValueError("cluster_column is only used by cluster resampling")
    frame = copy.deepcopy(scores)
    encoded = store.dumps_frame(frame)  # Reject provenance objects that cannot be retained.
    dates = pd.DatetimeIndex(frame["date"])
    if dates.hasnans or dates.tz is not None:
        raise ValueError("document dates must be finite timezone-naive timestamps")
    periods = dates.to_period(frequency)
    grid = pd.period_range(periods.min(), periods.max(), freq=frequency)
    code = grid.get_indexer(periods)
    counts = np.bincount(code, minlength=len(grid))
    if (counts == 0).any():
        raise ValueError("signal draws require observations in every calendar period")
    values = _number(frame[score_column].to_numpy(), "document scores")
    weights = np.ones(len(frame)) if weight_column is None else _number(frame[weight_column].to_numpy(), "document weights")
    if (weights <= 0).any():
        raise ValueError("document weights must be strictly positive")
    def aggregate(v: np.ndarray, w: np.ndarray) -> np.ndarray:
        denominator = np.bincount(code, weights=w, minlength=len(grid))
        if (denominator <= 0).any():
            raise ValueError("a draw has zero denominator in a calendar period")
        return np.bincount(code, weights=v * w, minlength=len(grid)) / denominator
    baseline = aggregate(values, weights)
    index = grid.to_timestamp(how="start")
    index.name = "date"
    if normalization not in {"raw", "zscore", "bbd_100"}:
        raise ValueError("normalization must be raw, zscore or bbd_100")
    offset, scale = 0.0, 1.0
    reference: dict[str, Any] = {"policy": "none"}
    if normalization == "raw":
        if normalization_window is not None:
            raise ValueError("raw normalization does not use normalization_window")
    else:
        if normalization_window == "full_sample":
            mask = np.ones(len(grid), dtype=bool)
            policy = "explicit_full_sample"
        else:
            mask = _window(grid, normalization_window, "normalization_window")  # type: ignore[arg-type]
            policy = "declared_window"
        mu, sd = float(baseline[mask].mean()), float(baseline[mask].std(ddof=0))
        if sd <= 0 or not np.isfinite(sd):
            raise ValueError("normalization reference must have positive finite variation")
        scale = (50.0 if normalization == "bbd_100" else 1.0) / sd
        offset = (100.0 if normalization == "bbd_100" else 0.0) - mu * scale
        reference = {"policy": policy, "start": str(index[mask][0]), "end": str(index[mask][-1]),
                     "baseline_mean": mu, "baseline_sd": sd, "scale": scale, "offset": offset}
    paths: dict[tuple[str, str], np.ndarray] = {}
    rng = np.random.default_rng(seed)
    clusters = None
    if resampling == "cluster":
        assert cluster_column is not None
        labels = frame[cluster_column]
        if labels.isna().any() or not all(isinstance(x, str) and x.strip() for x in labels):
            raise ValueError("persistent cluster labels must be nonempty strings")
        unique = sorted(labels.unique())
        if len(unique) < 2:
            raise ValueError("cluster bootstrap requires at least two persistent clusters")
        clusters = np.array([unique.index(x) for x in labels])
        if any(set(clusters[code == t]) != set(range(len(unique))) for t in range(len(grid))):
            raise ValueError("cluster bootstrap requires balanced cluster coverage in every period")
    for draw in range(n_draws):
        if clusters is None:
            sampled = np.zeros(len(grid))
            for t in range(len(grid)):
                positions = np.flatnonzero(code == t)
                chosen = rng.choice(positions, size=len(positions), replace=True)
                sampled[t] = np.dot(values[chosen], weights[chosen]) / weights[chosen].sum()
        else:
            n_clusters = int(clusters.max()) + 1
            multipliers = np.bincount(rng.integers(n_clusters, size=n_clusters), minlength=n_clusters)
            sampled = aggregate(values, weights * multipliers[clusters])
        paths[("doc", str(draw))] = sampled
    for source, mapping in (alternatives or {}).items():
        if source not in {"kernel", "lexicon"} or not isinstance(mapping, Mapping) or not mapping:
            raise ValueError("alternatives must map kernel/lexicon to labelled score columns")
        for label, column in mapping.items():
            _label(label, "alternative label")
            if column not in frame:
                raise ValueError(f"alternative score column {column!r} is missing")
            paths[(source, label)] = aggregate(_number(frame[column].to_numpy(), f"alternative {label}"), weights)
    if corpus_column is not None:
        labels = frame[corpus_column]
        if labels.isna().any() or not all(isinstance(x, str) and x.strip() for x in labels) or labels.nunique() < 2:
            raise ValueError("corpus scenarios require at least two nonempty corpus labels")
        for label in sorted(labels.unique()):
            paths[("corpus", "omit:" + label)] = aggregate(values, weights * (labels.to_numpy() != label))
    if not paths:
        raise ValueError("request bootstrap draws or at least one alternative/corpus scenario")
    draws = pd.DataFrame({key: offset + scale * value for key, value in paths.items()}, index=index)
    draws.columns = pd.MultiIndex.from_tuples(paths, names=["source", "draw_id"])
    baseline_series = pd.Series(offset + scale * baseline, index=index, name=name)
    if not np.isfinite(draws.to_numpy()).all() or not np.isfinite(baseline_series).all():
        raise ValueError("nonfinite signal after aggregation/normalization")
    quality = SignalQualityReport(pd.Series(counts, index=grid, dtype="int64"),
        pd.Series(dtype=float), [], metadata={"aggregation": "document-weighted mean" if weight_column else "document mean"})
    if "n_tokens" in frame:
        lengths = _number(frame["n_tokens"].to_numpy(), "document lengths")
        if (lengths < 0).any():
            raise ValueError("document lengths must be nonnegative")
        quality.avg_doc_length = pd.Series(np.bincount(code, weights=lengths) / counts, index=grid)
    if n_draws:
        with np.errstate(over="ignore", invalid="ignore"):
            quality.doc_bootstrap_sd = draws["doc"].std(axis=1, ddof=1)
        if not np.isfinite(quality.doc_bootstrap_sd.to_numpy()).all():
            raise ValueError("nonfinite document bootstrap SD")
    if "corpus" in draws.columns.get_level_values("source"):
        with np.errstate(over="ignore", invalid="ignore"):
            quality.corpus_loo_max_swing = draws["corpus"].sub(baseline_series, axis=0).abs().max(axis=1)
        if not np.isfinite(quality.corpus_loo_max_swing.to_numpy()).all():
            raise ValueError("nonfinite corpus sensitivity")
    if "kernel" in draws.columns.get_level_values("source"):
        from ._calibrate import _correlation
        kernels = [baseline_series.to_numpy(), *[draws[c].to_numpy() for c in draws if c[0] == "kernel"]]
        correlations = [correlation_value for a, b in combinations(kernels, 2)
                        if (correlation_value := _correlation(a, b)) is not None]
        if correlations:
            quality.kernel_agreement = pd.Series({"mean_pairwise": float(np.mean(correlations))})
        quality.metadata["kernel_pairs_defined"] = len(correlations)
    info = copy.deepcopy(dict(metadata or {}))
    if "signal_draws" in info:
        raise ValueError("metadata.signal_draws is reserved for generated provenance")
    info["signal_draws"] = {"schema_version": 1, "frequency": frequency, "aggregation": "mean",
        "weight_column": weight_column, "resampling": resampling, "cluster_column": cluster_column,
        "n_draws": n_draws, "seed": seed, "input_sha256": hashlib.sha256(encoded).hexdigest(),
        "normalization_reference": reference,
        "source_semantics": {"doc": "conditional_document_measurement_bootstrap", "kernel": "sensitivity_scenarios",
                             "lexicon": "sensitivity_scenarios", "corpus": "sensitivity_scenarios"},
        "dependence": ("Persistent-cluster weights shared across all dates; requires balanced coverage." if resampling == "cluster"
                       else "Independent document resampling within each period; no cross-period measurement dependence is estimated."),
        "scope": "Conditional measurement/sensitivity paths; not time-series sampling draws or total estimator uncertainty."}
    # Validate metadata without discarding unsupported nested provenance.
    check = baseline_series.to_frame()
    check.attrs = info
    store.dumps_frame(check)
    return RiskIndex(name, country, baseline_series, method, corpus, language, normalization,
                     metadata=info, quality=quality, draws=draws)
