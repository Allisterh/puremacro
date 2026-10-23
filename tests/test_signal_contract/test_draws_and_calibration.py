"""Scientific contracts for document uncertainty, refits and held-out scores."""
import copy
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from puremacro.narrative.signals import (
    build_signal_draws, score_documents, propagate_signal,
    calibrate_signal, evaluate_events,
)
from puremacro.narrative.types import RiskIndex, BenchmarkScore, SurveyScore, EventPanelScore


@pytest.fixture
def scores():
    rows = []
    for t, date in enumerate(pd.date_range("2000-01-01", periods=12, freq="QS")):
        for j, label in enumerate(("a", "b", "c")):
            rows.append({"date": date, "score": float(t + 2 * j), "alternative": float(t + j),
                         "cluster": label, "corpus": label, "weight": float(j + 1), "n_tokens": 10 + j})
    return pd.DataFrame(rows)


def build(scores, **kwargs):
    options = dict(name="test", country="USA", language="en", method="keyword_count", corpus="synthetic",
                   resampling="within_period", n_draws=30, seed=83, weight_column="weight")
    options.update(kwargs)
    return build_signal_draws(scores, **options)


def test_weighted_mean_and_document_bootstrap_manual_rng_parity(scores):
    original = scores.copy(deep=True)
    global_rng = np.random.get_state()
    ri = build(scores)
    np.testing.assert_allclose(ri.series, np.arange(12) + 8 / 3)
    rng = np.random.default_rng(83)
    first = []
    for t in range(12):
        chosen = rng.choice(np.arange(3 * t, 3 * t + 3), size=3, replace=True)
        first.append(np.dot(scores.score.iloc[chosen], scores.weight.iloc[chosen]) / scores.weight.iloc[chosen].sum())
    np.testing.assert_allclose(ri.draws[("doc", "0")], first)
    np.testing.assert_allclose(ri.quality.doc_bootstrap_sd, ri.draws["doc"].std(axis=1, ddof=1))
    assert list(ri.draws.columns.names) == ["source", "draw_id"]
    assert ri.quality.n_docs_per_period.tolist() == [3] * 12
    pd.testing.assert_frame_equal(scores, original)
    after = np.random.get_state()
    np.testing.assert_array_equal(global_rng[1], after[1])
    assert global_rng[2:] == after[2:]
    pd.testing.assert_frame_equal(build(scores).draws, ri.draws)


def test_persistent_clusters_keep_joint_path_and_correct_weight_denominator(scores):
    ri = build(scores, resampling="cluster", cluster_column="cluster")
    # Every cluster has path t + fixed offset. Shared cluster multipliers imply
    # each complete bootstrap path differs only by a constant, not date noise.
    np.testing.assert_allclose(np.diff(ri.draws.to_numpy(), axis=0), 1)
    rng = np.random.default_rng(83)
    multiplicities = np.bincount(rng.integers(3, size=3), minlength=3)
    w = np.array([1, 2, 3]) * multiplicities
    offset = np.dot([0, 2, 4], w) / w.sum()
    np.testing.assert_allclose(ri.draws[("doc", "0")], np.arange(12) + offset)
    incomplete = scores.drop(index=0)
    with pytest.raises(ValueError, match="balanced"):
        build(incomplete, resampling="cluster", cluster_column="cluster")


def test_declared_monthly_frequency_survives_instrument_adapter(scores):
    monthly = scores.copy()
    monthly["date"] = np.repeat(pd.date_range("2000-01-01", periods=12, freq="MS"), 3)
    ri = build(monthly, frequency="M")
    assert ri.as_instrument().frequency == "M"
    assert build(scores).as_instrument().frequency == "Q"


def test_fixed_normalization_and_scenario_labels(scores):
    ri = build(scores, normalization="zscore", normalization_window=("2000Q1", "2001Q2"),
               alternatives={"kernel": {"alternative score": "alternative"}}, corpus_column="corpus")
    baseline = np.arange(12) + 8 / 3
    mu, sd = baseline[:6].mean(), baseline[:6].std(ddof=0)
    np.testing.assert_allclose(ri.series, (baseline - mu) / sd)
    # Alternative scores use baseline scale, not their own recentered normalizer.
    np.testing.assert_allclose(ri.draws[("kernel", "alternative score")], (np.arange(12) + 4 / 3 - mu) / sd)
    np.testing.assert_allclose(ri.draws[("corpus", "omit:a")], (np.arange(12) + 16 / 5 - mu) / sd)
    np.testing.assert_allclose(ri.quality.corpus_loo_max_swing,
                               ri.draws["corpus"].sub(ri.series, axis=0).abs().max(axis=1))
    assert ri.quality.kernel_agreement["mean_pairwise"] == pytest.approx(1)
    assert ri.metadata["signal_draws"]["source_semantics"]["kernel"] == "sensitivity_scenarios"


@pytest.mark.parametrize("bad", ["implicit_weights", "zero_weight", "nan_score", "complex_score", "gap",
    "bad_normalization", "constant_reference", "duplicate_id", "zero_corpus_denominator", "one_draw"])
def test_draw_boundary_rejections(scores, bad):
    changed = scores.copy()
    options = {}
    if bad == "implicit_weights": options["weight_column"] = None
    elif bad == "zero_weight": changed.loc[0, "weight"] = 0
    elif bad == "nan_score": changed.loc[0, "score"] = np.nan
    elif bad == "complex_score": changed["score"] = changed.score.astype(complex) + 1j
    elif bad == "gap": changed = changed.drop(index=[3, 4, 5])
    elif bad == "bad_normalization": options["normalization"] = "zscore"
    elif bad == "constant_reference":
        changed["score"] = 1.0
        options.update(normalization="zscore", normalization_window="full_sample")
    elif bad == "duplicate_id": changed.index = [0] * len(changed)
    elif bad == "zero_corpus_denominator":
        changed.loc[:2, "corpus"] = "only"
        options["corpus_column"] = "corpus"
    else: options["n_draws"] = 1
    with pytest.raises(ValueError): build(changed, **options)


def test_existing_kernel_adapter_checks_output_alignment_and_retains_weights():
    from puremacro.narrative.indices._kernels import keyword_count_kernel
    records = [("2020-01-01", "risk risk", "url", {"corpus": "a", "weight": 2}),
               ("2020-04-01", "no match", "url2", {"corpus": "b", "weight": 1})]
    table = score_documents(records, lambda rows: keyword_count_kernel(rows, terms=frozenset({"risk"})))
    assert table.score.tolist() == [2., 0.]
    assert table.weight.tolist() == [2, 1] and table.n_tokens.tolist() == [2, 2]
    with pytest.raises(ValueError, match="dates"):
        score_documents(records, lambda rows: [(rows[1][0], 0.), (rows[0][0], 2.)])
    with pytest.raises(ValueError, match="exactly one"):
        score_documents(records, lambda rows: [(rows[0][0], 2.)])


def test_joint_refits_and_sensitivity_not_pooled_with_bootstrap(scores):
    ri = build(scores, resampling="cluster", cluster_column="cluster", corpus_column="corpus", n_draws=20)
    ri.metadata["nested"] = {"date": pd.Timestamp("2020-01-01")}
    calls = []
    def refit(signal):
        calls.append(signal.copy())
        out = pd.Series({"mean": signal.mean(), "double_mean": 2 * signal.mean()})
        out.attrs["solver"] = "analytical"
        signal.iloc[0] = -999  # callback mutation must not affect another path.
        return out
    result = propagate_signal(ri, refit, model_label="linear example", outcome_units={"mean": "points", "double_mean": "points"},
                              outcome_convention="means of complete signal paths")
    assert len(calls) == 24 and result.passed
    np.testing.assert_allclose(result.draws.loc["double_mean"], 2 * result.draws.loc["mean"])
    assert result.baseline["mean"] == pytest.approx(ri.series.mean())
    assert result.baseline.attrs["signal_metadata"] == ri.metadata
    assert result.draws.attrs["refit_metadata"]["draw_0"] == {"solver": "analytical"}
    assert result.summary("doc").attrs["sampling_uncertainty_combined"] is False
    scenario = result.summary("corpus")
    assert "not a probability" in scenario.attrs["scope"] and scenario.attrs["ci"] is None
    np.testing.assert_allclose(scenario.lower, result.draws["corpus"].min(axis=1))
    changed = result.draws
    changed.iloc[0, 0] = 12345
    assert result.draws.iloc[0, 0] != 12345
    with pytest.raises(FrozenInstanceError): result._metadata = "changed"
    assert ri.series.iloc[0] != -999


@pytest.mark.parametrize("failure", ["exception", "nonfinite", "coordinates", "metadata", "input"])
def test_refit_failures_retained_and_block_summaries(scores, failure):
    ri = build(scores, n_draws=4)
    if failure == "input": ri.draws.iloc[0, 1] = np.nan
    n = 0
    def refit(signal):
        nonlocal n
        n += 1
        if n == 3 and failure != "input":
            if failure == "exception": raise RuntimeError("deliberate refit failure")
            if failure == "nonfinite": return pd.Series({"mean": np.inf})
            if failure == "coordinates": return pd.Series({"other": 0.})
            out = pd.Series({"mean": 0.})
            out.attrs["opaque"] = object()
            return out
        return pd.Series({"mean": signal.mean()})
    result = propagate_signal(ri, refit, model_label="probe", outcome_units={"mean": "points"}, outcome_convention="sample mean")
    assert not result.passed and result.metadata["n_failed"] == 1
    assert result.attempts.status.tolist() == ["passed", "failed", "passed", "passed"]
    assert result.attempts.loc[1, "position"] == 1
    assert result.draws.iloc[:, 1].isna().all()
    with pytest.raises(ValueError, match="incomplete"):
        result.summary("doc")


def test_calibration_train_only_analytical_mapping_and_metadata(scores):
    ri = build(scores, resampling="cluster", cluster_column="cluster")
    ri.metadata["vintage"] = {"source": "original"}
    target = 3 + 2 * ri.series
    target.iloc[6:] += 100
    result = calibrate_signal(ri, target, name="benchmark", target_unit="survey points",
                              train_window=("2000-01-01", "2001-04-01"),
                              evaluation_window=("2001-07-01", "2002-10-01"))
    details = result.metadata["calibrations"]["benchmark"]
    assert details["intercept"] == pytest.approx(3) and details["slope"] == pytest.approx(2)
    score = result.quality.benchmark_scores["benchmark"]
    assert isinstance(score, BenchmarkScore)
    assert score.rmse == pytest.approx(100) and score.mean_bias == pytest.approx(-100)
    assert score.n_obs == 6 and score.pearson == pytest.approx(1)
    np.testing.assert_allclose(result.draws, 3 + 2 * ri.draws)
    np.testing.assert_allclose(result.quality.doc_bootstrap_sd, 2 * ri.quality.doc_bootstrap_sd)
    assert result.metadata["vintage"] == ri.metadata["vintage"]
    assert "calibrations" not in ri.metadata and ri.quality.benchmark_scores == {}
    survey = calibrate_signal(ri, 3 + 2 * ri.series, name="survey", target_unit="points", kind="survey",
                             train_window=("2000-01-01", "2001-04-01"), evaluation_window=("2001-07-01", "2002-10-01"))
    assert isinstance(survey.quality.survey_scores["survey"], SurveyScore)
    with pytest.raises(FrozenInstanceError): score.rmse = 0


@pytest.mark.parametrize("bad", ["overlap", "reversed", "misaligned", "missing", "constant_train", "normalization_leak"])
def test_calibration_leakage_alignment_and_invalid_training_rejected(scores, bad):
    ri = build(scores)
    target = 3 + 2 * ri.series
    train = ("2000-01-01", "2001-04-01")
    evaluation = ("2001-07-01", "2002-10-01")
    if bad == "overlap": evaluation = ("2001-01-01", "2002-10-01")
    elif bad == "reversed": train, evaluation = evaluation, train
    elif bad == "misaligned": target = target.iloc[1:]
    elif bad == "missing": target.iloc[3] = np.nan
    elif bad == "constant_train": ri.series.iloc[:6] = 1.
    else: ri = build(scores, normalization="zscore", normalization_window=("2000Q1", "2002Q4"))
    with pytest.raises(ValueError):
        calibrate_signal(ri, target, name="test", target_unit="points", train_window=train, evaluation_window=evaluation)


def test_explicit_full_sample_normalization_remains_labelled(scores):
    ri = build(scores, normalization="zscore", normalization_window="full_sample")
    result = calibrate_signal(ri, ri.series * 3, name="b", target_unit="points",
                             train_window=("2000-01-01", "2001-04-01"), evaluation_window=("2001-07-01", "2002-10-01"))
    assert result.metadata["calibrations"]["b"]["normalization_reference"]["policy"] == "explicit_full_sample"
    assert result.normalization == "raw"


@pytest.mark.parametrize("part", ["draws", "sd", "corpus"])
def test_calibration_rejects_overflow_in_all_transformed_uncertainty(scores, part):
    ri = build(scores, corpus_column="corpus")
    if part == "draws": ri.draws.iloc[0, 0] = 1e308
    elif part == "sd": ri.quality.doc_bootstrap_sd.iloc[0] = 1e308
    else: ri.quality.corpus_loo_max_swing.iloc[0] = 1e308
    with pytest.raises(ValueError, match="nonfinite"):
        calibrate_signal(ri, ri.series * 4 + 1, name="overflow", target_unit="points",
            train_window=("2000-01-01", "2001-04-01"), evaluation_window=("2001-07-01", "2002-10-01"))


def test_event_evaluation_rejects_every_known_training_window(scores):
    ri = build(scores)
    calibrated = calibrate_signal(ri, ri.series * 2 + 1, name="first", target_unit="points",
        train_window=("2000-01-01", "2001-04-01"), evaluation_window=("2001-07-01", "2002-10-01"))
    events = pd.Series(np.arange(12) % 2, index=ri.series.index, dtype=float)
    with pytest.raises(ValueError, match="training"):
        evaluate_events(calibrated, events, evaluation_window=("2000-01-01", "2000-10-01"))
    result = evaluate_events(calibrated, events, evaluation_window=("2001-07-01", "2002-10-01"))
    assert result.quality.metadata["event_evaluation"]["evaluation_independence"] == "known_calibration_training_excluded"
    unverified = evaluate_events(ri, events, evaluation_window=("2001-07-01", "2002-10-01"))
    assert unverified.quality.metadata["event_evaluation"]["evaluation_independence"] == "unverified_no_training_provenance"
    calibrated.metadata["calibrations"]["earlier_source_fit"] = {"train_window": ["2002-01-01", "2002-10-01"]}
    with pytest.raises(ValueError, match="training"):
        evaluate_events(calibrated, events, evaluation_window=("2001-07-01", "2002-10-01"))


def test_correlation_retains_representable_variation_around_large_offset():
    from puremacro.narrative.signals._calibrate import _correlation
    delta = np.array([0., 2., 4., 8., 6., 12., 10.])
    y = np.array([0., 4., 2., 6., 10., 8., 14.])
    assert _correlation(1e16 + delta, 1e16 + y) == pytest.approx(np.corrcoef(delta, y)[0, 1], abs=1e-15)
    assert _correlation(1e-200 * delta, 1e-200 * y) == pytest.approx(np.corrcoef(delta, y)[0, 1], abs=1e-15)
    assert _correlation(np.array([-1e308, 0., 1e308]), np.array([-1e308, 0., 1e308])) == pytest.approx(1)


def test_events_tie_aware_auc_ranking_and_undefined_metrics(scores):
    ri = build(scores)
    ri.series = pd.Series([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], index=ri.series.index, dtype=float)
    events = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], index=ri.series.index, dtype=float)
    result = evaluate_events(ri, events, evaluation_window=("2000-01-01", "2002-10-01"))
    score = result.quality.event_panel
    assert isinstance(score, EventPanelScore)
    positive, negative = ri.series[events == 1], ri.series[events == 0]
    oracle = np.mean([float(p > n) + .5 * float(p == n) for p in positive for n in negative])
    assert score.auc == pytest.approx(oracle)
    assert score.top_n == 2 and score.top_decile_hit_rate == 1
    inverted = evaluate_events(ri, events, evaluation_window=("2000-01-01", "2002-10-01"), polarity=-1)
    assert inverted.quality.event_panel.auc == pytest.approx(1 - oracle)
    all_events = evaluate_events(ri, events * 0 + 1, evaluation_window=("2000-01-01", "2002-10-01"))
    assert all_events.quality.event_panel.auc is None and all_events.quality.event_panel.spearman is None


@pytest.mark.pyodide_smoke
def test_signal_workflow_offline_smoke(scores):
    ri = build(scores, n_draws=4)
    result = propagate_signal(ri, lambda s: pd.Series({"mean": s.mean()}), model_label="mean",
                              outcome_units={"mean": "points"}, outcome_convention="sample mean")
    assert result.passed and len(result.summary("doc")) == 1
    calibrated = calibrate_signal(ri, ri.series * 2 + 1, name="benchmark", target_unit="points",
        train_window=("2000-01-01", "2001-04-01"), evaluation_window=("2001-07-01", "2002-10-01"))
    assert calibrated.quality.benchmark_scores["benchmark"].rmse < 1e-10
