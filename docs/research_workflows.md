> 🇬🇧 English · 🇪🇸 [Español](es/research_workflows.md)

# Independent evidence and structural estimation

The [cross-country forecasting workflow](cross_country_forecasting.md) adds a
frozen three-country GDP panel, an expanding-window comparison, and portable
data provenance. It is latest-vintage forecast evidence, with no real-time or
causal claim; it is included in puremacro 4.8.1.

These research workflows connect independent validation and structural
estimation. Each can run offline from an installed package with
the standard numerical dependencies.

The [real-data studies](empirical_research.md) add observed-moment SW07
estimation with explicit conditional-calibration diagnostics and an original-data
Romer–Romer (2010) baseline replication against independent software and published
rounding targets. The synthetic NK example below remains a parameter-recovery
test; the new empirical studies have their own data, assumptions and conclusions.

| Task | Entry point | Guide |
|---|---|---|
| Inspect independent numerical/data evidence | `puremacro.validation.run_research_benchmarks` | [Research benchmarks](research_benchmarks.md) |
| Fit structural models to correlated empirical moments | `puremacro.structural.fit_structural` and `lp_moment_targets` | [Empirical-to-structural bridge](structural_bridge.md) |
| Compare SW07 moment corrections and weights | `puremacro.structural.run_sw07_estimator_experiment` | [Paired estimator controls](sw07_estimator_experiment.md) |

## 1. Run the independent benchmarks

```python
from puremacro.validation import run_research_benchmarks

report = run_research_benchmarks()
print(report.passed)  # False: the documented RR2010 t-rounding comparison fails.
report.write("research_output/benchmarks")
```

The dossier distinguishes analytical solutions, authenticated external Dynare
output and the original Romer–Romer baseline with independent software and
published targets. It records every observed/reference value, source, units,
tolerance, limitation and negative-control result. Four of five cases pass;
the original-data RR2010 t statistic misses the paper's rounding interval by
about 0.000003925. That failed comparison is retained, as explained in the
[real-data guide](empirical_research.md). A pass establishes only its stated scope.

## 2. Run a complete empirical-to-structural study

```bash
python -m puremacro.examples.structural_irf_matching --output research_output/structural
```

The application independently simulates a linear three-equation New Keynesian
economy with a persistent policy disturbance and measurement errors. It
estimates output/inflation LP responses, constructs their **full joint HAC
covariance**, and fits risk aversion, Phillips-curve slope and shock persistence
by solving the model's equations through `load_mod`. Model responses and targets
use a 0.01 quarterly-rate innovation (one percentage point).

Horizons 0–4 are fitted; horizons 5–8 are descriptive out-of-fit checks using the
same sample. The application exports the simulated observations, moment vector,
joint covariance, fitted parameters, held-out comparisons, a figure and a
manifest containing the model, data hash and random seed. This is a transparent
synthetic recovery exercise, not a historical monetary-policy replication.

For an empirical study replace the simulator with your transformed data and
identified shocks, or provide your own `MomentTargets`. The same model callback
can wrap a DSGE, VFI or HANK solve. Model equations, moment definitions, shock
units, frequency and cumulative/level conventions must agree. The bridge does
not supply missing identification assumptions or HA equilibrium derivatives.
