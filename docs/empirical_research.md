> 🇬🇧 English · 🇪🇸 [Español](es/empirical_research.md)

# Real-data published replication

This study extends the [research workflows](research_workflows.md) with an
original-paper specification and its original data. It asks whether
puremacro reproduces a specified published result.

## Romer–Romer (2010): original baseline tax response

```bash
python -m puremacro.examples.romer_romer_2010_replication --output research_output/rr2010
```

This study checks one baseline result in
[Romer and Romer (2010), *The Macroeconomic Effects of Tax Changes*](https://www.aeaweb.org/articles?id=10.1257/aer.100.3.763).
The original workbook and `EXOGERNR.RAT` specification come from the
[authors' data archive](https://eml.berkeley.edu/~dromer/papers/DataSet.zip).
Their hashes and the derived data/reference hashes are retained.

```python
from puremacro.replication import estimate_rr2010_baseline

replication = estimate_rr2010_baseline()
print(replication.to_frame().loc[10])
# Preserve the complete response covariance when using these empirical targets.
targets = replication.to_moment_targets()
print(targets.labels)
```

For 1950Q1–2007Q4, the baseline regression explains
`100 * delta log(real GDP)` with a constant and contemporaneous through twelve
lags of `100 * (DEFIC + LONGR) / NOMGDP`. Cumulative sums of the thirteen tax
coefficients give the response of the GDP level to a tax increase equal to
one percent of GDP. Positive tax shocks mean tax increases.

The original specification uses conventional OLS covariance with the residual
degrees-of-freedom adjustment. Cumulative-response covariance applies the same
linear cumulative-sum transformation to the full coefficient covariance.
HAC and local projections can be useful extensions, but they change this
published specification.

Two different comparisons are reported:

1. **Numerical parity:** all coefficients, covariance entries and cumulative
   responses against a frozen independent-software run that does not import
   puremacro. Tolerances allow numerical roundoff.
2. **Published evidence:** the horizon-ten trough, GDP response `-3.08` and
   t statistic `-3.53`, reported on printed page 781. Only published rounding
   justifies the looser tolerance here; it is not used for software parity.

The original data give a horizon-ten response of **-3.0808127631** and
t statistic **-3.5249960750**. Software parity passes, and the response and trough
horizon reproduce the printed results. The t statistic misses the -3.53
rounding interval by **0.000003925**. An independent read of the authors' RATS
databanks corroborates this value. Its source is unresolved; neither intermediate
rounding nor a wider tolerance is used to force agreement. The published case
and application therefore report **FAIL**, with a nonzero command exit status.
This is a verified discrepancy, not a claim that all published targets agree.

The scope is this original-data baseline. It does not reproduce every
robustness exercise or independently establish the narrative exclusion
assumption. The earlier `regression.romer_romer_tax_multiplier_ols` gallery
case is a modified horizon-eight LP with a different sample and revised data;
its coarse headline comparison is explicitly labeled as such.

## Reproducibility

The application runs offline with packaged numerical inputs and reference
evidence. The exported report includes data/model definitions, sample and units,
estimates, uncertainty diagnostics and figures. Source conversion and independent
reference regeneration are development steps, distinct from the offline
runtime. Reference software is not a runtime dependency.

Use the [structural bridge guide](structural_bridge.md) for covariance conventions
and the [benchmark guide](research_benchmarks.md) for provenance, strict comparisons
and negative controls.
