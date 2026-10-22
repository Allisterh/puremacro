# Auditing a solved result

`puremacro.validation.audit_result` inspects a calculation you have already
solved. A solver's convergence flag is reported separately from budget,
Euler/KKT, distribution and approximation evidence. There is no universal
“economically correct” flag.

```python
from puremacro.validation import audit_result

report = audit_result(sol)          # stationary dp EGM solution
print(report.to_markdown())
frame = report.to_dataframe()
encoded = report.to_json()
budget_errors = report.maps["budget_residual"]
```

Every row contains `category`, `check`, `value`, `tolerance`, `units`, `status`,
`evidence_source`, `location`, and `details`. The statuses are `passed`, `failed`,
`not_checked`, and `unsupported`. Missing evidence never becomes a pass. The
report retains model/result metadata and residual maps; JSON and Markdown are
deterministic and require no optional reporting dependency. JSON represents
nonfinite failed values as named strings (`"NaN"`, `"Infinity"`).

The default categories are `solver`, `equations`, and `distribution`. The default
call evaluates equations and pushes a supplied distribution one period; it does
not re-solve, refine, differentiate, normalize mass, mutate the input or use the
network. `derivatives` and `refinement` require explicit selection. User callbacks
must respect the same no-mutation/no-network contract.

## Supported results and evidence

| Result | Checks and limits |
|---|---|
| Stationary `dp.DPSolution`, `method="egm"`, one asset | Declared budget and feasibility, continuous-policy Euler/KKT residual, mass, stationarity and asset-boundary mass. No life cycle, discrete choice, HJB or aggregate-risk adapter. The one-asset compiler's budget probes are identified as probes, without a symbolic identity claim for arbitrary functions. |
| Stationary two-asset `dp.DPSolution` | Both asset constraints, declared budget, marginal values, adjustment cost, deposit timing, Euler/KKT maps, weighted quantiles and separate distribution diagnostics. Model context comes from the result; `model=another_dp_model` checks the result against that declaration. |
| First-order `dsge.LinearModel` or `KleinSolution` | Rebuilt decision-rule and shock-loading equations, steady-state residuals through existing diagnostics. A bare Klein solution requires `model=linear_model`. Empty state blocks are marked `not_checked`. Optional five-point derivative checks keep steady-state operators fixed. This is not a nonlinear global-accuracy certificate. |
| Dynamic trade steady state or transition | Original 17-check certificate, values, original tolerance and provenance. With `model=dynamic_economy`, the steady-state certificate is independently rebuilt at the supplied `z`. Recorded transition certificates describe the solve-time result; they are not silently represented as new checks of altered paths. |
| Dynamic trade `HorizonComparisonResult` | With `checks=("refinement",)`, preserves window, welfare and selected terminal-test scope, including the weaker discounted-wealth mode. |

Dynare state declarations are also checked against every original lag column.
`unrepresented_lag_terms` fails if a control has any nonzero lag coefficient,
even when the truncated decision rule has zero residual. The map retains those
coefficients and names the offending controls; this structural check uses zero
tolerance independently of the numerical `equations` tolerance.

Trade certificate rows retain the certificate's `max(1e-8, result.tol)` tolerance;
the generic `equations` override does not change an inherited certificate's claim.
Unsupported solvers are reported explicitly. Supply a `ResidualCheck` for an
application-specific nonlinear residual or portfolio constraint:

```python
from puremacro.validation import ResidualCheck

report = audit_result(
    nonlinear_result,
    model=my_model,
    residuals=[ResidualCheck("nonlinear_equilibrium", recompute_residuals,
                             tolerance=1e-8, units="goods")],
)
```

`recompute_residuals(result, model)` returns equality residuals, or nonnegative
constraint violations. Failed callbacks become named failed checks with their
error; they never fall back to solver convergence.

## Household accuracy

For two assets, the audit recomputes envelope marginal values from supplied
consumption and the recognized convex adjustment cost, then interpolates them at
the **actual continuous policies** with the solver's bilinear lottery and shock
probabilities. The choice derivatives are

\[
g_b=\beta E[V_b(a',b',s')]-u'(c),\qquad
 g_a=\beta E[V_a(a',b',s')]-(1+\chi_d)u'(c).
\]

Each is divided by current marginal utility. An interior choice requires zero;
a lower-bound choice requires `g <= 0`; an upper-bound choice requires `g >= 0`.
The map reports the violated part at bounds. Upper asset bounds are explicitly
artificial truncations. Boundary classification uses local grid spacing so narrow
cells are not accidentally treated as bound states.

Maps are indexed `(illiquid, liquid, productivity)` regardless of declaration
order. Each worst residual includes its indices and economic coordinates.
Stationary-weighted 50th/95th/99th percentiles use the nearest-rank convention;
invalid or missing distributions make these summaries `not_checked`. Total mass,
minimum probability, stationarity and mass at both upper boundaries remain
separate checks. The audit never repairs or normalizes the supplied distribution.

Default screening tolerances are Euler/KKT `0.01`, equation residual `1e-8`,
stationarity `1e-10`, unit-mass error `1e-12`, negative probability `1e-14`, and
upper-bound mass `0.001`. They are application choices, independent of an EGM
stopping tolerance:

```python
report = audit_result(sol, tolerances={"euler": 0.002, "boundary_mass": 1e-4})
```

A fixed synthetic 30×20-grid reference in `tests/test_result_audit.py` has maximum
normalized errors about `0.001965` (illiquid) and `0.003831` (liquid), with weighted
99th percentiles `0.001641` and `0.002088`. An EGM tolerance of `1e-9` does not make
these errors `1e-9`. Tests also compare continuous-policy expectations against
SciPy interpolation, choice/envelope derivatives against an independent
five-point stencil, and two policies against independent Bellman optimization.
The Bellman objective gaps are about `0.001745` and `0.000322`; interpolation of
value and of marginal value are different approximations. These are synthetic
numerical checks, not external or empirical validation.

## Explicit derivative and refinement experiments

```python
from puremacro.validation import DerivativeCheck, RefinementCheck

report = audit_result(
    sol,
    checks=("solver", "equations", "distribution", "derivatives", "refinement"),
    derivatives=[DerivativeCheck("posterior_score", log_posterior, theta, score)],
    refinements=[
        RefinementCheck("refine_both_grids", solve_finer_household),
        RefinementCheck("expand_illiquid_bound", solve_larger_illiquid_grid),
        RefinementCheck("expand_liquid_bound", solve_larger_liquid_grid),
    ],
)
```

Derivative callbacks receive a copied vector; the audit compares the supplied
Jacobian to a central five-point stencil. Refinement callbacks take no arguments
and return a new result. Default stationary-DP metrics include mean consumption,
current asset holdings, Euler/KKT maxima and percentiles, and upper-bound masses.
The acceptance rule is `abs(new-old) <= atol + rtol*abs(new)` with defaults
`atol=1e-8`, `rtol=0.005`. Near-zero comparisons explicitly use an absolute floor;
they do not report a misleading percentage change. Coarse-grid reference
experiments fail this 0.5% target for some moments: a successful solve is not a
refinement certificate.

For a horizon experiment or selected Jacobian entries, supply
`RefinementCheck(..., metrics=extract_metrics, units={...})`. The extractor returns
the same nonempty mapping of scalar names for both results; e.g. consumption in
the economic window, terminal gaps and `J_C_w[0,0]`. Every re-solve and Jacobian
calculation belongs in an explicitly supplied callback. No experiment is inferred
from a result type or silently executed.
