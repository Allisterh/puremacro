> 🇬🇧 English · 🇪🇸 [Español](es/SIGNAL_CONTRACT.md)

# Signal contract

> Sparsity diagnostics remain opt-in on existing index constructors.
> `puremacro.narrative.signals` adds explicit document draws, joint refits,
> sensitivity scenarios and held-out calibration. These do not establish
> causal identification or total estimator uncertainty.

The signal contract is the per-`RiskIndex` data shape that lets a
downstream LP / SVAR estimator know how reliable an index reading is
and propagate explicitly defined measurement paths into downstream refits.

## Schema (Slice 1)

`puremacro.narrative.types.RiskIndex` carries two optional fields:

```python
@dataclass
class RiskIndex:
    name: str
    country: str
    series: pd.Series
    method: str
    corpus: str
    language: str
    normalization: str
    metadata: dict
    quality: SignalQualityReport | None = None   # 0.65.0+
    draws:   pd.DataFrame | None          = None
```

Both default to `None` — every pre-0.65.0 caller behaves identically.

## Opt-in (Slice 1)

```python
from puremacro.narrative.indices import lui

ri = lui(records, country="USA", language="en", with_quality=True)
ri.quality.summary()      # one-row DataFrame: mean_n_docs, mean_doc_length, n_coverage_gaps, ...
```

The same `with_quality=False` kwarg sits on every canonical index in
`puremacro.narrative.indices.__all__`: `epu`, `mpu`, `gpr`, `tone`,
`wui`, `lui`, `ltui`, `ltui_up`, `ltui_down`, `lwui`, `lwui_wage`,
`bbui`, `cboui`, `ep_ui`, `erpui`, `eurlex_ui`, `sotuui`, `bluesky_ui`.

## `SignalQualityReport`

| Field                 | Populated in | Description                                          |
|-----------------------|--------------|------------------------------------------------------|
| `n_docs_per_period`   | 0.65.0       | docs per quarter that fed the kernel                 |
| `avg_doc_length`      | 0.65.0       | mean tokens per doc per quarter                      |
| `coverage_gaps`       | 0.65.0       | quarters with zero docs inside the date range        |
| `kernel_agreement`    | Explicit signal workflow | mean pairwise time-series correlation of baseline and supplied kernel alternatives; undefined constant pairs omitted and counted |
| `multilingual_parity` | Not computed by this workflow | caller-defined language-comparability analysis remains separate |
| `doc_bootstrap_sd`    | Explicit signal workflow | per-period sample SD across document bootstrap paths |
| `corpus_loo_max_swing`| Explicit signal workflow | largest absolute change from omitting one corpus |
| `benchmark_scores`    | `calibrate_signal` | held-out Pearson/Spearman/RMSE/MAE/bias in target units |
| `event_panel`         | `evaluate_events` | declared evaluation-date rank correlation, top-decile hit rate and tie-aware AUC |
| `survey_scores`       | `calibrate_signal(kind="survey")` | same held-out agreement metrics against a survey |

## Validation

`RiskIndex.__post_init__` validates `draws` (when set):
- `draws.index` must equal `series.index`.
- `draws.columns` must be a 2-level `pd.MultiIndex` named `['source', 'draw_id']`.
- Every value of the `source` level must be in `{'kernel', 'lexicon', 'doc', 'corpus'}`.

The new workflow additionally validates finite scores, unique draw coordinates,
calendar coverage, positive declared weights and callback output coordinates.
Existing index constructor defaults are unchanged.

## Document draws with an explicit estimand

```python
from puremacro.narrative.signals import score_documents, build_signal_draws
from puremacro.narrative.indices._kernels import keyword_count_kernel

# records: (date, text, source_url, metadata), one row per document.
scores = score_documents(
    records,
    lambda rows: keyword_count_kernel(rows, terms=frozenset({"uncertainty"})),
)
ri = build_signal_draws(
    scores, name="uncertainty", country="USA", language="en",
    method="keyword_count", corpus="declared local corpus",
    resampling="within_period", n_draws=199, seed=71,
    normalization="raw",
)
```

The adapter requires exactly one dated score per record in input order. It
retains document lengths and complete corpus/cluster/weight metadata. You may
instead supply a DataFrame with `date` and `score` directly. Precomputed scores
keep remote scoring, human coding and model selection outside the resampling
loop. The callback is trusted code; it is never stored or imported by an archive.

Aggregation is the document mean. With `weight_column="weight"`, the mean
uses positive weights in both numerator and denominator. A column named
`weight` cannot be silently ignored. Draws resample documents uniformly and
retain their attached weights; this is a bootstrap of the weighted ratio.
Other aggregation rules are outside this workflow.

Choose the resampling unit explicitly:

- `within_period` samples the original number of documents independently in
  each calendar period. Dates and period meaning stay fixed. Cross-period
  dependence of measurement error is **not** estimated by this choice.
- `cluster`, with `cluster_column`, samples persistent cluster labels once
  per draw and applies those multiplicities at every date. Every cluster must
  appear in every period. Per-date weighted denominators account for different
  numbers or weights of documents. Incomplete cluster panels are rejected;
  periods are never shuffled or moved to another date.

The data must cover every period at the declared period `frequency` (default
`"Q"`). Output dates are period starts. `n_draws=0` allows sensitivity scenarios
alone; one bootstrap draw is rejected. Seeds use a local NumPy generator.

Nonraw normalization requires `normalization_window=("2000Q1", "2005Q4")`
or the explicit choice `"full_sample"`. The baseline reference mean and
population SD are fixed across all paths; each draw is not recentered or
rescaled. Constant reference windows fail. `bbd_100` follows the existing
package convention of mean 100 and SD 50. For forward evaluation, use a
training-only reference window; deliberate full-sample normalization stays
labelled as such in provenance.

## Scoring and corpus alternatives are sensitivity scenarios

Supply additional per-document score columns with
`alternatives={"kernel": {"alternative label": "column"}, "lexicon": {...}}`.
Use `corpus_column="corpus"` for leave-one-corpus-out paths. Every retained
period needs a positive denominator after omission. All paths use the same
baseline normalization. Kernel, lexicon and corpus alternatives are labelled
scenarios, not exchangeable probabilistic draws; they are never pooled into
bootstrap percentiles. Multilingual measurement equivalence is not inferred.

## Refit a downstream model on each complete path

```python
import pandas as pd
from puremacro.narrative.signals import propagate_signal

def refit(signal):
    # Replace this illustration with your estimator using the same dates,
    # identification, units and outcome labels on every call.
    return pd.Series({"mean": signal.mean(), "lag1_cov": signal.cov(signal.shift(1))})

propagation = propagate_signal(
    ri, refit, model_label="declared downstream estimator",
    outcome_units={"mean": "score units", "lag1_cov": "squared score units"},
    outcome_convention="sample mean and lag-one covariance",
)
print(propagation.attempts)
measurement_percentiles = propagation.summary("doc", ci=0.90)
```

Each callback receives the entire baseline/path and must return the same
ordered Series of finite point estimates. The result preserves joint outcome
draws, their source IDs, metadata and input hashes. The callback owns its
sample selection, identification and random seed. Return point estimates;
do not mix the estimator's confidence endpoints into this vector.

Failed draws retain their original positions, error type and message; their
outcome columns are explicitly missing. **Any failure blocks summary output.**
No baseline substitution or automatic dropping occurs. `summary("doc")`
reports pointwise conditional measurement percentiles only. Other sources
produce labelled min/max sensitivity envelopes. Sampling uncertainty,
calibration-parameter uncertainty and measurement uncertainty are not added
together. A full joint time-series/measurement resampling design remains a
separate methodological choice.

## Train calibration and evaluate on later dates

```python
from puremacro.narrative.signals import calibrate_signal, evaluate_events

calibrated = calibrate_signal(
    ri, aligned_benchmark, name="external benchmark", target_unit="benchmark points",
    train_window=("2000-01-01", "2005-10-01"),
    evaluation_window=("2006-01-01", "2010-10-01"),
)
evaluated = evaluate_events(
    calibrated, aligned_binary_events,
    evaluation_window=("2006-01-01", "2010-10-01"), polarity=1,
)
```

Targets/events must match every signal date exactly; there is no inner join,
interpolation or missing-value deletion. Training and evaluation each need
at least three observations; evaluation must follow training without overlap.
The default affine map is fitted only on training dates and applied unchanged
to all draws. `mapping="none"` evaluates pre-aligned units without fitting.
`kind="survey"` attaches a `SurveyScore`. The mapped series is labelled `raw`
in target units, with original normalization and calibration details retained.
Undefined correlations/AUC are `None`, never a fabricated zero or perfect score.
Known calibration training windows, including earlier calibration history, may
not overlap event evaluation. Without training provenance, independence is
explicitly marked unverified; the function cannot rule out earlier score/model
selection on those dates. Event polarity is explicit; AUC uses average ranks for ties, and top-decile
membership includes all ties at its cutoff, with `top_n` reported.

Calibration returns a copy, preserves existing metadata, and refuses to overwrite
an existing calibration name. Event evaluation refuses to overwrite an existing
event-panel result. These are agreement/ranking diagnostics, not a test
of causal identification or an automatically calibrated event probability.
