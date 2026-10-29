# Arellano: original-code replication and grid sensitivity

**The archived solver is reproduced; convergence of the economic moments is
not established.** This benchmark compares the original Arellano code, an
independent implementation, and PureMacro's current public solver. It keeps
their model conventions explicit and makes no new fit to the published targets.

## Run it locally

Use the development checkout with Python 3.11+ and the normal development
dependencies. The full study takes a few minutes on the tested workstation.

```sh
python -m pytest tests/test_arellano_2008_study.py -q
MPLBACKEND=Agg python -m tools.reference_validation.run_arellano_study --output /tmp/arellano-study
```

Open `grid_sensitivity.png` or `grid_sensitivity.pdf`. `moments.csv` contains the
target/result table; `results.json` includes every seed, sampling interval,
episode count, solver diagnostic and code hash. The ordinary run needs neither
network access, gfortran nor Octave; frozen external references are included.

## Independent checks

The author's Fortran code converged in 49 price iterations after a documented
portability patch. The independent study solver reproduces all 4,200 debt choices
and default decisions. Prices match exactly, and the largest value difference is
5.3e-11. Separately, the author's MATLAB moment-aggregation block was executed in
Octave on identical path data and windows: all 12 moments agree within 1.1e-14.
These are checks of code and measurement, not a declaration that the published
quantitative model has been fully replicated.

Sources are the [AER paper](https://doi.org/10.1257/aer.98.3.690), the author's
[July 2007 public draft](https://mpra.ub.uni-muenchen.de/7867/1/MPRA_paper_7867.pdf),
and the [archived replication package](https://doi.org/10.3886/E113248V1).
The draft also contains the Table 4 model targets below. Source files, licenses,
hashes and regeneration details are in the local fixture guide,
`tests/fixtures/arellano_2008/README.md`.

## Published targets and the archived grid

The study uses 16 independent seeds, 100,000 quarters per seed and a 1,000-quarter
burn. Business-cycle moments average 100 windows of 74 quarters immediately
before default, each with uninterrupted market access. The intervals below
measure Monte Carlo uncertainty in the **mean across replications**; they are
not parameter confidence intervals or the dispersion of a single 100-window
experiment. The JSON reports that dispersion separately.

| Moment | Published model | Original 200 × 21 grid | 95% MC interval |
|---|---:|---:|---:|
| Debt / output (%) | 5.95 | 6.283 | [6.083, 6.482] |
| Mean annual spread (%) | 3.58 | 3.405 | [3.362, 3.448] |
| Annual default frequency (%) | 3.00 | 2.822 | [2.756, 2.887] |
| Spread SD (percentage points) | 6.36 | 6.009 | [5.891, 6.126] |
| Trade balance / output SD (percentage points) | 1.50 | 1.485 | [1.463, 1.508] |
| Log consumption SD (%) | 6.38 | 6.391 | [6.323, 6.459] |
| Log output SD (%) | 5.81 | 5.826 | [5.754, 5.898] |
| Correlation: spread, output | -0.29 | -0.251 | [-0.265, -0.236] |
| Correlation: trade balance, output | -0.25 | -0.245 | [-0.252, -0.239] |
| Correlation: consumption, output | 0.97 | 0.971 | [0.970, 0.971] |
| Correlation: trade balance, spread | 0.43 | 0.432 | [0.423, 0.441] |
| Correlation: consumption, spread | -0.36 | -0.331 | [-0.347, -0.316] |

Several gaps persist relative to the estimated simulation mean. The paper's
original random draws are unavailable, and its stated 100-window experiment
differs from the archived MATLAB script, which scans candidate defaults 2:300.
An exact replay of the published table is therefore not claimed.

## Grid convergence remains unresolved

Debt grids are nested with fixed support and an exact zero. Income grids retain
the same log-income endpoints. Every independently solved case has a Bellman
residual below 4e-12 and a price residual below 1e-8. No simulated repayment
choice hits either debt-grid boundary. Despite those checks, moments move:

| Assets × income states | Debt / output (%) | Mean spread (%) | Annual default (%) | Spread SD (pp) |
|---|---:|---:|---:|---:|
| 200 × 21, original Fortran | 6.283 | 3.405 | 2.822 | 6.009 |
| 399 × 21 | 6.322 | 3.248 | 2.663 | 5.402 |
| 797 × 21 | 6.347 | 3.320 | 2.740 | 5.348 |
| 797 × 41 | 6.010 | 4.043 | 3.558 | 4.442 |
| 797 × 81 | 5.491 | 3.504 | 3.080 | 3.477 |

Small equation residuals and absence of boundary mass do not establish economic
grid convergence. The 41-to-81-state movements exceed Monte Carlo uncertainty;
spread volatility is particularly sensitive. Being closer to one published
target on a finer grid is not evidence that all targets have converged.

## Specification and measurement differences

The archive sets default output to `min(y, 10*.969)` while mean income is about
10.0307. PureMacro normalizes stationary mean income and applies the cap to that
mean. The archived inputs match a 21-state Tauchen grid to printing precision,
although the draft names Tauchen–Hussey. The archive's actual debt grid also ends
at 1.53502537 rather than its nominal 1.5.

Exclusion timing is material: the archive guarantees an additional excluded
period after entry. With only that timing changed to the canonical Bellman
equation, mean debt/output falls from 6.283% to 5.440% and mean spread rises from
3.405% to 4.184%. The public solver's separate 321 × 21 calibration yields
5.055% debt/output and a 4.329% spread. That case also changes normalization,
grid and literal precision, so it is **not** a timing-only comparison.

The study compounds annual spreads as `q**-4 - Rf**4`, uses within-window sample
standard deviations, and retains raw correlations. The target-comparison
default frequency uses the archived repayment-period denominator; the JSON
also reports the hazard with default-entry periods included in exposure. All
counts use the same post-burn sample. The public solver's short built-in
simulation diagnostics use different measurement conventions and are not used
here. The earlier built-in exclusion loop kept an extra excluded period relative
to its canonical Bellman equation. That bug is now corrected; successful
reentry permits a decision in the same quarter. Both public simulation
interfaces reset debt to the zero node and agree with the same seed. The shared
study simulator continues to follow each case's Bellman timing.

## Next decision

The public timing and measurement contract is now documented in the
[sovereign guide](sovereign_default.md). Resolve the grid
sensitivity with a documented convergence criterion, before recalibrating or
interpreting policy effects. Reentry, income volatility and persistence are
the next controlled experiments once those checks pass. Maturity and
restructuring extensions follow their own external validation.

## Check the public calibration's grids

```sh
MPLBACKEND=Agg python -m tools.reference_validation.check_arellano_grids --output /tmp/arellano-grids
```

This uses independently solved public-model equations, checked against
`solve_sovereign_default` on a matched 401 × 9 grid. Calibration, support,
canonical timing and measurement remain fixed. Default runs compare 401/801
asset nodes and 321/641 income states, with 16 paired seeds per case.

Every percentage moment must change by at most 0.10 percentage points, and
every correlation by at most 0.02. The bound includes the absolute paired mean
change **plus** its 95% Student-t half-width. Both the final debt-grid refinement
and the final income-grid refinement must pass, with at least 797 debt points
and 321 income states on the finest grid. These are numerical tolerances;
published targets do not determine acceptance. A nonzero exit status means
convergence has not been established, distinguishing a large grid shift from
insufficient simulation precision. A passed baseline does not validate new
parameter values, policy effects or the long-term extension.

The report records every seed, source hash, residual, moment and comparison.
Verified solution caches can be reused with `--periods 500000 --windows 1000`
to reduce sampling uncertainty while keeping the seeds fixed. Earlier archived
results above remain historical evidence for a distinct calibration/timing.
