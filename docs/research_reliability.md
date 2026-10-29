# Research reliability and the next PureMacro milestones

The agreed order is **research reliability and replications**, then **economic
models and policy applications**, then **teaching and usability**.

## Current correction branch

The v6.0.0 source tag did not reach PyPI. As checked on 2026-10-26, the published
package is 5.7.1. Use the development checkout for these corrections; a new
release still needs the supported-platform CI matrix and publishing gates.

Native model adapters now preserve the supplied equations and loadings.
First-order DSGE scores use Kalman sensitivities; DAG scores centrally difference
the native Whittle likelihood while holding the supplied steady state fixed.
Unsupported native operations raise errors. Explicit `model=None` demonstrations
warn about their scope. See [adapter details](autodiff_and_protocol.md).

The default nonlinear HANK DAG uses prescribed household responses. Its
aggregate-risk example uses prescribed moments and matrices. They are explicitly
demonstrations; a supplied model cannot be silently ignored. The legacy
`asymmetry_ratio` reports nonlinear/linear peak amplification, not two solved
opposite-shock paths. See [HANK scope](hank_ssj_and_aggregate_risk.md).

Sovereign corrections include deterministic default, stationary-mean default
costs, zero-debt reentry and unadjusted simulation statistics. A constant spread
has an undefined correlation, reported as `NaN` with a status field.

## Tests to run first

From the repository root, in a Python 3.11+ environment with `.[dev,notebooks]`:

```sh
python -m pytest tests/test_protocol_model_fidelity.py tests/test_research_reliability_references.py tests/test_sovereign/test_reference_reliability.py -q
python tools/release_check.py --no-tests
```

For an interactive tour, run notebook **72** in English or Spanish. Notebook
**73** explores sovereign debt; **74** covers synthetic distributional LPs,
Gaussian inference and reduced transitions. Their numerical scorecards have
the specific scope described in each notebook, not blanket economic validation.

## Independent evidence so far

| Comparison | Result on the tested grids | What remains open |
|---|---|---|
| Arellano short-term deterministic equations | Matched-grid equations pass; archived Fortran and MATLAB moments reproduce; public simulation timing and zero-debt reentry are corrected; public-calibration grids converge at 6,401 × 641, and the public solver reaches that grid with a certified monotone search | Published-moment discrepancies and long-term extension |
| One-asset household block | SSJ authors' code matches policies and consumption within 5e-11; return Jacobians within 4e-4 | Finer-grid economic convergence, full published equilibrium and nonlinear transition references |

Both references are frozen with hashes and provenance in
`tests/fixtures/research_reliability/`. Regeneration tools live in
`tools/reference_validation/`; the external SSJ package is not a runtime
dependency. Matched-grid agreement does not prove discretization accuracy.

## Sovereign study: baseline completed, public grids converged

The [original-code and published-moment comparison](arellano_replication.md)
now includes 16 independent simulations per case, nested debt grids and 21 to
321 income states. At the archived calibration, refinement moves the moments
far more than simulation uncertainty, and spread volatility halves. For the
public calibration, a pre-registered paired-seed criterion establishes grid
convergence at 6,401 debt nodes and 641 income states. A small Bellman residual
alone would not have justified that conclusion.

1. Public reentry timing, zero-debt initialization and measurement are now
   documented and tested. Keep the archive and public-calibration results
   distinct; they use different economic conventions.
2. Done for the public calibration: grids converge at 6,401 × 641 under the
   registered criterion, with sampling uncertainty kept separate from
   discretization error and every raw moment retained.
   `solve_sovereign_default(policy_search="monotone")` solves that grid,
   certified by an exhaustive check, and matches the reference arrays.
3. Study how income volatility, persistence and market-reentry probability
   change sustainable debt and default exposure in the validated short-term
   model. Report welfare and numerical uncertainty before adding maturity or
   restructuring mechanisms that require additional validation.

This creates a research result that can be reviewed from equations through
policy tables. It also supplies the model and output contracts for future work.

The collateral/sudden-stop diagnostics need a separate correction and reference:
the current amplification includes a legacy fallback and the loose-constraint
branch prescribes diagnostic values. The term-structure residual component is
accounting, not an independently identified risk premium. Notebook 73 and the
[sovereign guide](sovereign_default.md) state these remaining limits explicitly.

## Then: household incidence and teaching

Complete a published one-asset HANK benchmark with matched fiscal rules,
return timing and equilibrium closure. Independently validate nonlinear
transitions before linking household deciles to distributional LPs on observed
data. A general higher-order aggregate-risk solver comes after those checks.

Use the validated study to build a short bilingual path: install a verified
wheel, reproduce one baseline, change one policy assumption, and inspect a
diagnostic table. Keep demonstrations visibly separate from empirical results.
