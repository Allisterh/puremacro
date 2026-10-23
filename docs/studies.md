> 🇬🇧 English · 🇪🇸 [Español](es/studies.md)

# Reproducible studies and comparisons

`puremacro.study` records one explicit statistical study: selected inputs,
transformations, units, identification assumptions, estimator settings, results,
diagnostics, dependency versions and checksums. It reuses the existing LP and
recursive VAR estimators, report formatters and portable runtime frame codec.
It does not infer a causal interpretation or certify an economic model.

## Run and compare local projections

This complete offline example uses synthetic observations. Its exogeneity
assumption is declared for illustration; the workflow does not establish it.

```python
from dataclasses import replace
import numpy as np
import pandas as pd
from puremacro.study import (
    VariableSpec, StudySpec, run_study, compare_studies,
    save_study, load_study,
)

rng = np.random.default_rng(173)
observations = rng.normal(size=(240, 3))
for t in range(1, len(observations)):
    observations[t] += 0.4 * observations[t - 1]
data = pd.DataFrame(
    observations,
    columns=["shock", "output", "price"],
    index=pd.period_range("1960Q1", periods=240, freq="Q"),
)
data.attrs = {"source": "synthetic demonstration", "data_seed": 173}

spec = StudySpec(
    name="LP, two lags", method="lp",
    variables=(
        VariableSpec("shock", "percentage points"),
        VariableSpec("output", "index points"),
        VariableSpec("price", "index points"),
    ),
    responses=("output",), shock="shock", controls=("price",),
    frequency="Q", identification="observed_exogenous",
    lags=2, horizon=8, shock_size=0.25, ci=0.90,
)
baseline = run_study(data, spec)
alternative = run_study(data, replace(spec, name="LP, four lags", lags=4))
comparison = compare_studies(baseline, alternative)
print(comparison.to_frame())
print(baseline.audits)

# A new directory only: existing results are not overwritten.
baseline.write_report("study_output/lp_two_lags")
# Or retain the complete record in memory, including in a browser kernel.
payload = save_study(baseline)
restored = load_study(payload)
assert restored.manifest == baseline.manifest
```

`write_report` writes Markdown, LaTeX, Typst and CSV tables, `manifest.json`,
and the lossless `study.pmstudy` archive. Markdown and the manifest include
methods, assumptions, effective samples and audit limits. CSV is a presentation
format; use the archive to preserve index types and supported `DataFrame.attrs`.
`to_json()` contains the manifest and output tables, not the input frames.
Both `StudyResult` and `StudyComparison` are frozen records; returned frames
and manifest dictionaries are independent copies.

## State the transformations, units and time grid

Every input column needs a `VariableSpec`. Available transformations are
`identity`, `log`, `diff`, and `logdiff`; `scale` multiplies the transformed
series. The declared `unit` describes that final quantity. For example,
`VariableSpec("output", "percent log growth", "logdiff", 100)` computes
`100 * diff(log(output))`. The caller supplies positive output levels.
There is no variable-name heuristic for percent conversion or annualization.

The input index must be a unique, increasing, consecutive `PeriodIndex` or
`DatetimeIndex`. `frequency="Q"` describes quarterly periods; for timestamp
indices use the corresponding calendar offset, such as `"QS"` or `"MS"`.
Quarter/month-end aliases accept the supported pandas spelling. A missing
period or nonfinite value is an error; time is never compressed by dropping
missing rows. Extra undeclared columns are errors too.

`sample_start` and `sample_end`, if supplied, must identify observations on
that grid. Selection precedes transformation. Differencing removes exactly
the first selected observation from every series; it does not borrow a value
from outside the selected sample. The archive retains the selected raw inputs,
not rows outside the requested sample. Source, vintage and retrieval notes can
be stored in `data.attrs`; unsupported metadata objects raise an error instead
of being silently discarded.

## Understand the estimator contracts

| Setting | `method="lp"` | `method="var_cholesky"` |
|---|---|---|
| Identification | `observed_exogenous`: the observed shock is exogenous conditional on regressors, as an assumption | `recursive`: declared contemporaneous recursive ordering |
| Response at horizon `h` | `y[t+h] - y[t-1]` | Level impulse response at `h` |
| Shock size | Multiple of the observed shock's transformed unit | Named shock variable's impact in its transformed unit |
| `lags=p` | Lags 1–p of the individual response, shock and controls; contemporaneous controls and a constant | Lags 1–p of every ordered variable and a constant; p ≥ 1 |
| Effective sample | First shock date is `max(1,p)`; last is `T-h-1` | Dates p through T−1 for the fitted VAR |
| Uncertainty | Pointwise normal HAC intervals; Bartlett bandwidth h+1 | Optional residual bootstrap pointwise percentile intervals |

Horizons are inclusive: `horizon=8` returns 0 through 8. LP responses have
different estimation windows by horizon. Changing lags can also change fitted
observations. `n_obs` is reported per result, and `effective_by_horizon` in the
manifest records the shock/regressor dates. Identical raw and transformed input
windows do not imply identical fitted observations.

LP response, shock and control names must be distinct. With multiple responses,
each LP uses its own response lags; other responses are not silently inserted
as controls. `n_boot` must be zero for LP. For VAR, ordering contains every
variable exactly once and `controls` is empty:

```python
recursive = replace(
    spec, name="Recursive VAR", method="var_cholesky",
    identification="recursive", controls=(),
    ordering=("price", "shock", "output"), n_boot=199, seed=71,
)
var_result = run_study(data, recursive)
```

The VAR normalizes the entire response by `shock_size / P[k,k]`, where `P` is
the point estimate's Cholesky factor and `k` is the shock's ordered position.
Bootstrap intervals use this **fixed** normalization: they do not renormalize
every draw to the same impact. Negative shocks reverse interval endpoints
correctly. `n_boot=0` omits interval columns; it does not fabricate zero-width
bands. Failed bootstrap draws and estimated VAR instability remain visible
in the audit table. Nonfinite exported estimates or intervals raise an error.

`compare_studies` accepts at least two uniquely named studies with matching
method, identification, ordering, horizon, variable definitions, response
convention, shock size and units, and identical raw/transformed data and metadata.
Lags, confidence level, seed and bootstrap count may differ and are recorded.
Mixed LP/VAR comparisons are rejected: an observed exogenous shock with a
cumulative response is not automatically the same estimand as a recursively
identified innovation with a level response. The API does not add a cosmetic
overlay that conceals this difference.

## Preserve and reload evidence safely

`save_study(result, "study.pmstudy")` writes a new file; `save_study(result)`
returns bytes. `load_study(path_or_bytes, max_bytes=64 * 1024 * 1024)` validates
the exact archive inventory, schema, frame and manifest SHA-256 checksums,
units, sample counts and finite numerical values. ZIP expansion and NPY array
shapes are bounded before decoding. It uses JSON and the version-2 runtime
NPZ codec, including supported frame metadata. It neither extracts archive
paths nor unpickles objects, imports callbacks or executes an estimator.
No Parquet engine, cloud account or network connection is required.

Checksums detect changed content; they do not authenticate authors or prove
the correctness of a deliberately rehashed result. Seeds and numerical package
versions are retained. Bitwise rerun equality is expected only within the same
numerical environment; different BLAS libraries or package versions can change
floating-point results. The diagnostic statuses are `passed`, `failed`,
`not_checked` and `unsupported`; identification remains `not_checked`.

## Record an evaluated model or scenario

`ScenarioSpec`, `ScenarioResult` and `record_scenario` record existing DSGE,
dynamic-programming, HANK, structural-fit or static scenario outputs. They do
not run a second solver. Source is inert text; parameters, evaluated outputs,
input tables, audit statuses and optional evidence frames are retained.

The following illustrates an explicitly calibrated response path, without
claiming that its parameters were estimated from the example observations:

```python
from puremacro.study import ScenarioSpec, record_scenario

scenario_spec = ScenarioSpec(
    name="Calibrated response illustration", model="declared geometric response",
    variables=spec.variables, responses=spec.responses, shock=spec.shock,
    frequency=spec.frequency, identification=spec.identification,
    response_convention="y[t+h]-y[t-1]",
    horizon=spec.horizon, shock_size=spec.shock_size,
)
outputs = pd.DataFrame({
    "response": ["output"] * (spec.horizon + 1),
    "horizon": range(spec.horizon + 1),
    "estimate": [0.1 * 0.7**h for h in range(spec.horizon + 1)],
})
scenario = record_scenario(
    outputs, scenario_spec,
    inputs=baseline.inputs, transformed_inputs=baseline.data,
    model_source="response[h] = 0.1 * persistence**h; pre-shock baseline = 0",
    parameters={"persistence": 0.7, "impact": 0.1},
    audits=pd.DataFrame([{
        "check": "economic_identification", "status": "not_checked",
        "detail": "External shock equivalence is assumed for this illustration.",
    }]),
    provenance={"purpose": "calibrated illustration; parameters not estimated"},
)
overlay = compare_studies(
    baseline, scenario,
    comparison_note="Assume the model innovation maps to the declared observed "
                    "exogenous shock and the same pre-shock baseline; illustrative only.",
)
scenario.write_report("study_output/calibrated_response")
```

Outputs contain exactly `response`, `horizon`, `estimate`, in response-major
order and horizons 0..H. Units are supplied by `variables`. Audit tables have
`check`, `status`, `detail`; failed, unsupported or not-checked rows are retained.
`parameters` contains finite real scalars; `evidence={"name": dataframe}` can
store general calibration matrices, actual full joint moment covariance,
fit/held-out targets, distribution/state paths or separate draw records. The
recorder never constructs a covariance from marginal LP bands.

`inputs` may be a static model/calibration table. `transformed_inputs` is
mandatory when variable definitions declare a nonidentity transform or scale;
the recorder does not infer or recompute model data transformations. For a
static input-output scenario, `horizon=0`, `frequency="annual_static"`,
`identification="calibrated_input_output_scenario"` and
`response_convention="static_level_change"` are honest explicit labels.
No empirical date index is required for a standalone scenario.

Any comparison containing scenarios requires `comparison_note`. It also checks
exact shock/response units and transformations, shock size, identification,
frequency, horizon and response convention, and identical raw/transformed input
snapshots. A note cannot override an inconsistent contract. These checks verify
declared comparability; they do not prove economic equivalence. Scenario tables
do not fabricate empirical observation counts or confidence intervals.

`save_study`/`load_study` distinguish empirical and scenario archive tags.
Scenario reports additionally include `model_source.txt`; lossless evidence and
provenance remain inside the archive. Unsupported frame structures, such as
named MultiIndex column levels or subclass-specific result attributes, are
rejected if the runtime codec cannot preserve them. Convert deliberately to a
plain table and retain coordinate definitions in provenance. All numeric
evidence must be finite; retain failed-draw status separately instead of hiding
or replacing failed numerical output.

This workflow does not automate IV identification, structural estimation or
model/scenario selection. To construct correlated empirical targets and fit a
model callback, use the existing [structural bridge](structural_bridge.md) and
[research workflows](research_workflows.md), then record their actual outputs.
