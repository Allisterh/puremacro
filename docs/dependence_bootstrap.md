> 🇬🇧 English · 🇪🇸 [Español](es/dependence_bootstrap.md)

# Dependence-aware bootstrap references

Two optional interfaces supplement the existing spatial LP and GVAR diagnostics.
They make different assumptions and do not replace the default asymptotic results.
Neither is an exact finite-sample test or a claim that all documented size distortions
have been eliminated.

## Spatial LP: whole-cross-section time blocks

```python
from puremacro.spatial import SpatialLPBootstrap, spatial_lp

result = spatial_lp(
    panel, "output", "shock", weights, horizons=range(6), n_lags=2,
    lagged_outcome="difference", alpha=0.05,
    bootstrap=SpatialLPBootstrap(
        n_boot=399, block_length=6, assume_stationary=True, seed=814,
    ),
)
inference = result.bootstrap_result
print(inference.summary())
print(inference.table)
```

`result` still contains the original horizon-specific fits. **The bootstrap table
contains a separate fit on the intersection of all horizon samples.** Its `point`
and `se` can differ from the original table. `metadata['common_periods']` identifies
the observations actually used. Every horizon must have a balanced, complete panel
with the same entities, and the common sample must contain at least two block lengths.

The algorithm constructs the outcomes and controls before resampling, samples
circular time blocks of complete cross-sections, and uses the same sampled periods
at every horizon. Thus it preserves contemporaneous spatial dependence and the
within-block cross-horizon dependence, without constructing fictitious leads or
lags across block seams. Every draw re-estimates both fixed effects, coefficients,
and the selected Driscoll–Kraay or Conley studentizing covariance. Entity-cluster
studentization is rejected only for this optional bootstrap: with a fixed number
of regions its scale has no supported fixed-N limit here. Legacy cluster fits remain
available without the bootstrap.

`table` reports symmetric bootstrap-t pointwise intervals, simultaneous max-|t|
intervals, centered two-sided p-values, and their conditional Monte Carlo standard
errors. The simultaneous family contains **all displayed nonzero contrasts at all
requested horizons**, including direct, indirect, and total responses.
`joint_p_value` uses the maximum absolute t over the per-order indirect contrasts
and horizons; `metadata['joint_labels']` specifies that family. An intentionally
zero-scaled indirect contrast is excluded. Without any nonzero indirect contrasts,
the joint statistic and p-value are `None`. No chi-square Wald reference is used.

The stationarity assertion concerns the **entire regression-row process**, including
controls. The dependent variable is still `y[t+h] - y[t-1]`. The new optional
`lagged_outcome='difference'` uses lagged changes of `y` and, when requested, lagged
changes of `Wy`; the default remains `'levels'`. This permits an integrated outcome
with stationary changes and controls. It does not make arbitrary integrated controls,
structural breaks, endogenous shocks, or misspecified dynamics valid. The underlying
argument is fixed N, increasing T, stationary weak dependence and an identified
exogenous-shock regression. The `total` contrast remains relative to the period mean,
as explained in the [spatial guide](spatial.md).

Choose the block length before inspecting significance, report sensitivity to more
than one defensible length, and remember that a long block leaves fewer effective
blocks in a short panel. All requested draws are retained. A singular regression or
invalid studentizing variance in any draw raises an error that names the draw;
there is no selective retry or deletion.
Empirical-quantile intervals and plus-one p-values may disagree exactly at a
discrete Monte Carlo boundary by roughly one draw; neither is claimed to be an
exact finite-sample inversion of the other.
The stationary block-bootstrap asymptotic justification requires block length to
grow with T while its ratio to T tends to zero; a fixed short block does not
universally accommodate arbitrary serial dependence. The same condition applies
to dependent-innovation blocks in the GVAR reference below; length one is its
iid-innovation special case.

The circular moving-block construction follows the stationary resampling framework
of [Künsch (1989)](https://doi.org/10.1214/aos/1176347265) and
[Politis and Romano (1992)](https://profiles.stanford.edu/joseph-romano?tab=publications).
This implementation is a panel regression-row application of that framework, not
an implementation or proof of uniform unit-root LP inference. The latter requires
the different assumptions and procedures studied by
[Montiel Olea and Plagborg-Møller (2021)](https://joseluismontielolea.com/lp_inference_ecta.pdf).

## GVAR: an explicitly stronger independent-country null

```python
# fitted_gvar was estimated with weak_exogeneity=True.
reference = fitted_gvar.independent_null_bootstrap(
    null="independent_countries", integration="i1",
    n_boot=399, block_length=1, null_lags=1, seed=815,
)
print(reference.summary())
print(reference.table[["country", "star_variable", "F",
                       "p_independent_null", "mc_se"]])
```

This reference assumes **independent country innovation processes and no
cross-country lag dynamics**. It is stronger than general weak exogeneity. The
reported p-values concern that specified reference distribution for the existing
residual-augmented auxiliary statistic; they do not become the error-correction
test of Dees et al. Country independence is not established by failing to reject,
and the statistic can have low power against some departures from independence.

For each country, a VAR of order `null_lags` is estimated in levels under
`integration='stationary'`, or in first differences under `integration='i1'`.
The latter simulates stationary increments and integrates them from the observed
initial level. Initial VAR lags are conditioned on. No unit-root pretest silently
chooses a mode. Mixed integration orders and cointegrated within-country blocks
are outside the I(1) reference. A nonstationary fitted null VAR raises; roots are
never clipped. Exogenous regressors and deterministic trends beyond `'n'` or `'c'`
are not currently supported by the null simulator.

Centered innovation vectors are resampled in circular blocks **independently across
countries**, retaining contemporaneous dependence among variables within each country.
Block length one is the iid-innovation special case. Each synthetic global panel
then re-estimates every original country VARX* and every auxiliary regression.
The generated first-stage residuals are never frozen across draws. Original country
and auxiliary lag orders are held fixed; prior information-criterion selection is
not repeated. This does not simulate an unrestricted fitted GVAR under an alternative.

The original `weak_exogeneity.table` is unchanged and remains diagnostic-only.
The new table uses the explicit names `p_independent_null`,
`p_holm_independent_null`, and `reject_independent_null`. Holm adjustment covers
all table rows. The p-value is `(1 + number of bootstrap F >= observed F)/(B+1)`;
`mc_se` describes simulation noise conditional on the fitted null, not model
uncertainty. The attainable resolution is `1/(B+1)`: for example, 99 draws cannot
produce a p-value below 0.01 and therefore cannot reject a six-test Holm family
at 5% at its first step. All draws and fixed reference settings are retained;
any failed draw makes the inference fail explicitly.
Restricted and unrestricted auxiliary designs must both have full column rank
in the observed data and every draw. For example, extra lags in a levels auxiliary
regression can span the generated residual exactly, making its tested coefficient
non-estimable; the bootstrap rejects that specification even though the legacy
diagnostic-only table remains available.

For the distinction between VARX* and cointegrating VECX* weak-exogeneity tests,
see [Chudik and Pesaran, Theory and Practice of GVAR Modelling](https://doi.org/10.1111/joes.12095)
and the [GVAR guide](gvar.md). This restricted simulator does not add a VECX* layer.

## Reproducible size, coverage, and power evidence

`tools/reference_validation/validate_dependence_bootstrap.py` runs bounded, seeded
experiments and writes JSON after every design. Its spatial designs use N=20/40,
T=50/100, six horizons, spatially correlated AR(0.4) errors, and both zero and
nonzero spillovers. They use differenced outcome controls: they are not a relabeling
of the older level-control experiment. GVAR designs include independent I(1)
countries at T=200/300, stationary AR(0.5), block-length sensitivity, and two
innovation-leakage alternatives, including the existing persistent-foreign-country
power design.

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 python tools/reference_validation/validate_dependence_bootstrap.py \
  --domain spatial --replications 200 --draws 199 --output /tmp/spatial-mc.json
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 python tools/reference_validation/validate_dependence_bootstrap.py \
  --domain gvar --replications 200 --draws 399 --output /tmp/gvar-mc.json
```

The output reports rejection, coverage and power with replication-level Monte Carlo
uncertainty, plus failed replications. Correlated GVAR tests within a replication
are averaged before computing its Monte Carlo standard error. Binary rates use
Wilson intervals, so zero observed failures do not imply zero uncertainty.
Inference failures are recorded separately; successful-replication rates must be
read alongside their denominator and failures. These are design-specific
experiments, not an automatic calibration gate or a universal nominal-size claim.

### Observed limitations in the preset matrix

The October 2026 run used **200 independent DGP replications per design**, seed
812000, 199 spatial bootstrap draws and 399 GVAR draws. All 2,200 DGP replications
completed without inference failures. Settings were fixed before the results;
none were tuned to obtain nominal rejection rates. Percentages below include
95% Monte Carlo intervals in brackets, using the uncertainty conventions above.

| Spatial N / T / block / true spillover | Pointwise coverage range | Simultaneous coverage (target 95%) | Joint rejection at 5% |
|---|---:|---:|---:|
| 20 / 50 / 4 / 0 | 94.5–98.0 | 93.5 [89.2, 96.2] | 3.0 [1.4, 6.4] |
| 20 / 50 / 8 / 0 | 92.0–95.5 | 90.5 [85.6, 93.8] | 2.5 [1.1, 5.7] |
| 40 / 50 / 4 / 0 | 90.0–94.5 | 88.5 [83.3, 92.2] | 9.0 [5.8, 13.8] |
| 20 / 100 / 6 / 0 | 92.5–97.5 | 91.0 [86.2, 94.2] | 5.5 [3.1, 9.6] |
| 20 / 50 / 4 / 0.25 | 92.5–97.5 | 92.0 [87.4, 95.0] | 100.0 [98.1, 100.0] (power) |

Spatial pointwise coverage here concerns the indirect response over six horizons;
the simultaneous family also contains direct and total responses. **The bootstrap
does not solve the short-panel coverage problem:** N=40 has 88.5% simultaneous
coverage and 9% null rejection, and even T=100 under-covers. A longer block did not
uniformly improve results. Treat this as an experimental, design-dependent
reference and report the sensitivity, not as a calibrated replacement for all
default bands. These runs use Driscoll–Kraay studentization; they do not establish
corresponding finite-sample performance for Conley.

| GVAR reference/design | Legacy nominal rejection | Independent-null bootstrap rejection |
|---|---:|---:|
| Independent I(1), T=200, block 1 | 10.25 | 3.25 [2.20, 4.30] |
| Independent I(1), T=300, block 1 | 10.92 | 4.08 [2.96, 5.21] |
| Independent AR(0.5), T=200, block 1 | 0.08 | 4.58 [3.40, 5.76] |
| Independent I(1), T=200, block 4 | 10.25 | 3.33 [2.30, 4.37] |
| I(1) innovation leakage 1, T=200 (power) | 9.00 | 3.00 [1.35, 4.65] |
| Stationary leakage 2, T=250 (power) | 100.00 | 100.00 [98.12, 100.00] |

Null rows average six auxiliary tests per replication; power rows average the
two tests for the leaking country. The last design has own persistence 0.5 and
foreign persistence 0.9. The restricted reference reduces the documented null
overrejection, sometimes conservatively, but its **3% power against the I(1)
leakage alternative is a material blind spot**. It is not an omnibus independence
test, and nonrejection cannot validate the original GVAR weak-exogeneity assumption.
