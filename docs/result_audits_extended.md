> 🇬🇧 English · 🇪🇸 [Español](es/result_audits_extended.md)

# Transition audits and bounded refinement

The [result audit](result_audits.md) can inspect perfect-foresight and built-in
one-asset nonlinear HANK paths. It evaluates supplied paths without choosing a
replacement solution. A separate opt-in refinement ladder runs explicitly
supplied callbacks within a declared work budget.

Editable `HANKResult` closures (`closure='model'`) are also supported. The audit
re-evaluates the original aggregate equations and linear household Jacobian response
on the supplied paths. For nonlinear household closures it compares outputs with
the retained final household transition only after verifying that the supplied
household input paths still match that experiment. A changed input fails its context
check and leaves household consistency `not_checked`; no household solve runs.
Retained stationary mass is checked separately. Missing micro Euler, budget, and
distribution-transport evidence stays explicitly `not_checked`.

## Perfect-foresight paths

```python
from puremacro.validation import audit_result

result = model.solve_perfect_foresight(periods=80)
report = audit_result(result)
print(report.to_markdown())
errors_by_date_and_equation = report.maps["path_equations"]
```

New `PerfectForesightResult` objects retain their resolved initial and terminal
boundaries, exogenous path, variable order and prepared equation callable.
The audit evaluates each original equation at the supplied path, including its
two boundary dates. `STEADY_STATE()` uses the solved fixed reference. It never
re-solves an `endval` block or reconstructs shocks from a current model.

Residual maps have shape `(period, equation)`, with periods numbered from one.
An altered trajectory fails even if its `converged` flag and recorded residual
still claim success. The reported distance between the final simulated date
and the terminal boundary is an ungraded horizon diagnostic: the boundary is
imposed at `T+1`, not at `T`.

For legacy or manually constructed results, supply the missing context:

```python
from puremacro.validation import PathAuditContext

context = PathAuditContext(
    equations=equations,  # (lead, current, lag, shock) -> residual vector
    y_init=initial_state,
    y_end=terminal_state,
    exogenous_path=shocks,  # exactly one row per reported date
    variable_names=result.variable_names,
)
report = audit_result(result, model=context)
```

If the callable explicitly declares a `steady_state` argument, it receives the
fixed terminal state. The caller supplies any parameter binding needed by a
raw callable. Without sufficient context the equation check is `not_checked`.
Rolling surprise paths and MCP paths have different expectation/complementarity
contracts and are not covered by this adapter. No probability-distribution
claim is made for a representative-agent perfect-foresight path.

## Nonlinear HANK paths

```python
from puremacro.models.hank_sequence_space import solve_nonlinear_transition

transition = solve_nonlinear_transition(
    steady_state, shock_seq=monetary_shocks, horizon=80,
    retain_audit_paths=True,
)
report = audit_result(transition)
budget_map = report.maps["household_budget"]
transport_map = report.maps["distribution_transport"]
```

Effective `phi_pi`, `kappa` and household parameters are retained, including
overrides that differ from the supplied steady-state object's parameters.
Even without household path retention, the audit recomputes the goods-market,
Phillips-curve and Taylor/Fisher identities from the reported paths. The
Phillips curve uses the model's zero terminal inflation deviation. These are
checks of the built-in closure, not an arbitrary `.mod` aggregate block.

`retain_audit_paths=True` stores consumption and savings policies, entering-date
and terminal distributions, realized returns, wages and the terminal consumption
policy from the already computed final household pass. It does not add an EGM
solve. Main retained arrays require approximately
`8 * (3*T + 2) * n_assets * n_productivity` bytes, plus small input arrays.
The default is `False`; unavailable household evidence is `not_checked`.
Audit maps themselves also consume memory proportional to the path size.

With retained paths, checks include:

- Resource budgets at every `(date, assets, productivity)` point, positive
  consumption, asset-domain feasibility and the realized-return timing.
- Independent aggregation of consumption and re-evaluation of goods-market
  clearing, including fiscal spending and the balanced interest-payment tax.
- Euler/KKT residuals using marginal utility interpolated at actual continuous
  savings policies. The final Euler equation uses the retained continuation
  consumption. Weighted percentiles weight dates equally and use each date's
  supplied probability mass; they are not stationary-distribution percentiles.
- Each date's total and negative mass, the initial distribution, policy-lottery
  transport, mean-asset preservation and mass at the artificial upper boundary.

Nothing normalizes an invalid distribution, clips a supplied policy or solves
for a better household. A 25-point, eight-date synthetic test clears equations
near `2e-12` while its Euler error is about `0.0023` and upper-bound mass about
`0.0056`; the latter fails the default `0.001` screen. Such observations describe
that calibration and grid, not a universal numerical error bound.

## Explicit refinement ladders

Existing `RefinementCheck` callbacks compare one candidate with a baseline.
`RefinementLadder` instead compares adjacent resolutions, records every attempt,
and can stop before calling the remaining solvers:

```python
from puremacro.validation import RefinementLadder, RefinementLevel

ladder = RefinementLadder(
    "horizon",
    levels=[
        RefinementLevel("T160", solve_T160, settings={"horizon": 160}, cost=2),
        RefinementLevel("T320", solve_T320, settings={"horizon": 320}, cost=4),
        RefinementLevel("T640", solve_T640, settings={"horizon": 640}, cost=8),
    ],
    metrics=extract_economic_metrics,
    max_solves=3, max_cost=14, consecutive_passes=2,
    rtol=0.005, atol=1e-8,
    units={"consumption_window": "goods", "terminal_asset_gap": "assets"},
    validate_pair=check_same_economic_experiment,
)
report = audit_result(
    baseline,
    checks=("solver", "equations", "distribution", "refinement"),
    refinement_ladders=[ladder],
)
print(report.metadata["refinement_ladders"]["horizon"]["stop_reason"])
```

`metrics(result)` is mandatory and returns the same nonempty mapping of finite
scalars at every level. Choose quantities that answer the research question:
consumption in a fixed economic window, wealth or welfare, a terminal gap,
Euler error quantiles, or selected Jacobian entries. A callback may do expensive
work only because this experiment is explicitly selected. For incomparable
units, scale metrics deliberately or use separate ladders with different
tolerances. No metric, horizon or grid is inferred.

Every metric must meet `abs(new-old) <= atol + rtol*abs(new)`. Success requires
the chosen number of consecutive passing comparisons; a failed comparison
resets the streak. A solver that declares non-convergence, a failed callback,
nonfinite metric, changed metric keys or rejected pair ends the ladder without
a stability claim. Previous failed comparisons remain visible after eventual
stabilization. Stability concerns only the selected metrics and resolutions.

`validate_pair(previous, new)` may raise if the economic specification,
boundaries or shock experiment changed. Without it, experiment equivalence
is explicitly `not_checked`; settings are caller-supplied provenance, not
an inspection of what the solve callback actually did.

`max_solves` limits attempted callbacks, including failures. Each attempted
callback consumes its declared positive `cost`; a callback is not started if
its cost would exceed `max_cost`. These work units are **not** elapsed seconds,
memory, or a hard execution timeout. The ladder cannot interrupt a running
callback. Stops distinguish `stable_metrics`, `solve_budget_exhausted`,
`cost_budget_exhausted`, `levels_exhausted`, and named computation failures.
The report retains budgets, attempts, metric values and stopping evidence in
deterministic JSON, Markdown and DataFrame metadata. Merely supplying a ladder
without selecting `refinement` executes no metric or solve callback.
