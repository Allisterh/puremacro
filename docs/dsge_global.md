> 🇬🇧 English · 🇪🇸 [Español](es/dsge_global.md)

# Local and global equilibrium policies

Use one nonlinear model specification for several calculations, while keeping
their economic questions and accuracy checks distinct:

| Calculation | Object being computed | Treatment of the future | Evidence to examine |
| --- | --- | --- | --- |
| `model.solve(order=1/2/3)` | A local Taylor approximation to decision rules | Shock covariance enters the perturbation calculation; realized innovations enter policy evaluation | Determinacy, conditioning, and nonlinear residuals near and away from the expansion point |
| `solve_perfect_foresight` / `solve_transition` | A finite trajectory for specified initial conditions and a terminal closure | The announced shock path is known within that experiment | Original path equations, feasibility, information timing, and sensitivity to the terminal closure/horizon |
| `solve_time_iteration` | An interpolated recursive policy over a declared state domain | A conditional integration rule evaluates future policies at possible successor states | Node convergence, off-grid Euler/KKT residuals, domain coverage, and sensitivity to approximation and integration |

A perfect-foresight path is not a stochastic policy: replacing unknown future
shocks by zero generally changes a nonlinear expectation. Likewise, a global
policy does not mean a solution on every possible state or a proof of global
uniqueness. Here, “global” describes approximation across the chosen domain.
See [nonlinear model loading](dsge_nonlinear.md) and
[nonlinear transitions](dsge_transitions.md) for the other two workflows, and
the [DSGE workflow guide](dsge_workflows.md) for method selection and teaching.

The global interface does not infer a planner's objective from arbitrary
equilibrium equations. It requires explicit states, controls, transitions,
conditional probabilities, domains and admissibility. There is no general
existence, uniqueness or convergence guarantee.

All Python blocks below run **in order in one session**, using the public API.
The examples are identical in both language editions.

## The recursive equilibrium contract

Write the problem in terms of states known when decisions are made, controls,
and future innovations:

\[
x_t=\pi(s_t),\qquad s_{t+1}=g(s_t,x_t,\varepsilon_{t+1}),\qquad
E_t\!\left[F(s_t,x_t,s_{t+1},\pi(s_{t+1}),\varepsilon_{t+1})\right]=0.
\]

`EquilibriumProblem` describes this system. It is not a Bellman objective and
does not infer a welfare-maximizing planner. States must contain enough
information to make the supplied transition and conditional shock rule valid.
An omitted lag, regime, or persistent shock cannot be recovered by increasing
the polynomial degree.

Let `ns` be the number of states, `nx` the number of controls, `ne` the number of
innovation coordinates, and `q` the number of integration nodes.

| Argument or callback | Required shape and meaning |
| --- | --- |
| `state_names`, `control_names` | Ordered names, of lengths `ns` and `nx`; names are unique within each group |
| `domain` | `(ns, 2)` finite, increasing bounds; the approximation rectangle |
| `transition(state, controls, innovation)` | Receives `(ns,)`, `(nx,)`, `(ne,)`; returns a finite `(ns,)` successor state |
| `residual(state, controls, next_state, next_controls, innovation)` | Returns a finite `(nx,)` vector before conditional expectation; one equation per control |
| `shock_nodes`, `shock_weights` | `(q, ne)` future innovations and `(q,)` nonnegative probabilities summing to one |
| `shock_rule(state)` | Optional conditional `(nodes, weights)` pair; overrides the fixed rule and is checked at each evaluation |
| `control_bounds` | `(nx, 2)` ordered bounds, or a callback returning them from `(ns,)`; infinite endpoints are allowed |
| `admissible(state, controls)` | Optional Boolean economic-feasibility check, such as positive consumption |
| `residual_scale` | `(nx,)` positive, finite equation scales; defaults to ones |

Callbacks receive **one point at a time**. Returned policies also support
batches: `result(state)` returns `(nx,)`, while `result(states)` for
`states.shape == (n, ns)` returns `(n, nx)`. A deterministic problem can omit the
shock rule: the default has one innovation of shape `(0,)` with weight one.

With `EquilibriumProblem.from_model`, the additional mappings are:

| Mapping | Contract |
| --- | --- |
| `current(state, controls)` | Full current endogenous vector in `model.variables` order; the same mapping at successor inputs supplies the lead vector |
| `lagged(state)` | Full lagged endogenous vector in `model.variables` order |
| `exogenous(state)` | **Current** innovations in `model.shocks` order; defaults to zeros |
| `equations` | One distinct source equation per control, selected by equation name or zero-based integer index |
| `steady_state` | Optional full endogenous vector or mapping used by source `steady_state(...)` expressions |

Use `model.equation_names` to inspect available names. The bridge copies the
model and its calibration. Callback closures remain the caller's responsibility:
changing a parameter referenced by `transition` or `current` changes that
callback, even though the source residual calibration was copied. The bridge
does not verify that every unselected source equation is identically satisfied
by the mappings; establish those identities when constructing the problem.

## One model, explicit recursive timing

Consider log utility, full depreciation, Cobb–Douglas production, and independent
Gaussian log-productivity innovations:

\[
\max E_0\sum_{t\ge0}\beta^t\log c_t,\quad
y_t=e^{a_t}k_{t-1}^{\alpha},\quad c_t+k_t=y_t,\quad a_t=\varepsilon_t.
\]

This tractable specialization of stochastic growth has the exact policy
\(k_t=\alpha\beta y_t\), with
\(c_t=(1-\alpha\beta)y_t\). Substitution into
\(1/c_t=\beta E_t[\alpha y_{t+1}/(k_t c_{t+1})]\) verifies the Euler equation:
future output cancels, leaving the same current marginal utility. This provides
an economic oracle separate from the numerical solver. The stochastic growth
framework originates in [Brock and Mirman (1972)](https://www.sciencedirect.com/science/article/abs/pii/0022053172901354).

The recursive state is `(inherited_capital, observed_log_productivity)` and the
control is savings, which is also the end-of-current-period capital `k` in the
`.mod` file. `read_mod` compiles the model without requiring a local solution.

```python
from dataclasses import replace
import numpy as np
from puremacro.dsge import (
    read_mod, EquilibriumProblem, gaussian_quadrature,
    solve_time_iteration, audit_policy, compare_policies,
    solve_adaptive_time_iteration, refine_time_iteration,
    evaluate_perturbation, diagnose_perturbation,
)

model = read_mod("""
var y c k a; varexo e;
parameters alpha beta;
alpha=0.33; beta=0.95;
model;
    [name='resources'] c+k=y;
    [name='production'] y=exp(a)*k(-1)^alpha;
    [name='Euler'] 1/c=beta*alpha*y(+1)/(k*c(+1));
    [name='productivity'] a=e;
end;
steady_state_model;
    k=(alpha*beta)^(1/(1-alpha));
    y=k^alpha; c=(1-alpha*beta)*y; a=0;
end;
shocks; var e; stderr 0.02; end;
""")
alpha, beta = (model.parameter_values[key] for key in ("alpha", "beta"))
names = tuple(model.variables)

def output(state):
    return np.exp(state[1]) * state[0] ** alpha

def current(state, control):
    y = output(state)
    values = dict(y=y, c=y-control[0], k=control[0], a=state[1])
    return np.array([values[name] for name in names])

def lagged(state):
    return np.array([state[0] if name == "k" else 0.0 for name in names])

def transition(state, control, innovation):
    return np.array([control[0], innovation[0]])

nodes, weights = gaussian_quadrature([[0.02**2]], order=3)
problem = EquilibriumProblem.from_model(
    model, state_names=("inherited_capital", "observed_log_productivity"),
    control_names=("savings",), domain=((0.1, 0.5), (-0.05, 0.05)),
    current=current, lagged=lagged, transition=transition,
    equations=("Euler",), exogenous=lambda state: state[1:2],
    shock_nodes=nodes, shock_weights=weights,
    control_bounds=lambda state: [(1e-6, output(state)-1e-6)],
    admissible=lambda state, control: 0 < control[0] < output(state),
)
initial = lambda state: np.array([0.26 * output(state)])
result = solve_time_iteration(problem, initial, orders=(7, 2), tol=1e-8)
assert result.converged
print(result.summary())
grid_values = result.to_frame()
assert len(grid_values) == (7 + 1) * (2 + 1)
```

The state contains inherited capital and **observed current** productivity.
The transition consumes a **future** innovation. `current` constructs current
and next endogenous vectors and imposes the resource/production identities;
`lagged` supplies the historical capital needed by the model. Only the remaining
Euler equation is selected. Conditional expectation is applied to the selected
residual after evaluating the future policy. Future marginal utility must stay
inside this expectation; multiplying the equation by random future consumption
would change the stochastic model.

Multiplication by a factor known at time `t` preserves an equality conditional
on that information when the factor is nonzero. Multiplication by a random
future variable does not generally do so. Keep expressions such as
`return_next / consumption_next` together inside the residual being integrated.

For a growth model with endogenous labor, a useful alternative is
`controls=(consumption, labor)`: impose production and resource accounting in
`current`, let `transition` carry the resulting savings, and select the
intratemporal and Euler equations. The MAV T02_A/T02_B notebooks demonstrate
that construction. A different choice of controls requires a corresponding
change in the residuals, bounds, and transition; adding controls without
additional independent equations does not define a solvable system.

## What time iteration solves

Each iteration holds the **previous interpolated future policy** fixed. At each
state node it solves the conditional equilibrium equations for current
controls using bounded nonlinear least squares. It then optionally dampens the
nodal update, interpolates the new controls, and recomputes the equations
using the updated policy. This is an Euler-equation policy iteration, not
Howard improvement for a supplied Bellman value function. Projection methods
represent functions in a finite basis and assess residual errors; see
[Judd (1992)](https://kenjudd.org/wp-content/uploads/2022/05/1992-Projection-Methods-For-Solving-Aggregate-Growth-Models.pdf)
for the broader numerical framework.

| Option | Meaning |
| --- | --- |
| `tol=1e-7` | Maximum accepted scaled KKT/equation residual and absolute nodal control update |
| `max_iter=200` | Outer policy-update budget |
| `inner_max_iter=100` | Per-node nonlinear function-evaluation budget |
| `damping=1.0` | Fraction of the proposed nodal update, in `(0, 1]` |
| `max_nodes=10000` | State-grid construction guard; not an accuracy target |
| `strict=False` | Keep a failed result with a warning; `True` raises on numerical nonconvergence |

`inner_max_iter` bounds the damped Newton iterations per collocation node. The
reported evaluation count tracks residual evaluations; numerical-Jacobian probes
can add residual calls when `transition_jacobian` or `residual_jacobian`
callbacks are omitted (with `EquilibriumProblem.from_model`, compiled equation
derivatives are reused automatically via `TrajectoryJacobian`).
`result.callback_evaluations` separately tracks `"transition_evaluations"`,
`"residual_evaluations"`, `"transition_jacobian_evaluations"`,
`"residual_jacobian_evaluations"`, and `"trial_rejections"`. Neither the
iteration budget nor `max_nodes` is a wall-clock or total-memory guarantee.

The initial policy can be a callable, one constant `(nx,)` vector, or a nodal
array matching the constructed grid. It must be finite and satisfy control
bounds and `admissible` at the nodes. The inner solve enforces box bounds and
uses domain-safe feasible backtracking (`trial_rejections` in `history` and
`callback_evaluations`) if a full nodal or outer policy update would violate
bounds or `admissible` at collocation or quadrature successor states;
`admissible` is a feasibility check, **not** a general nonlinear-constraint
optimizer. Tensor Chebyshev fitting uses separable 1D LU factorizations along
each state axis rather than assembling a dense N-dimensional Vandermonde
inverse (`policy.collocation_matrix_allocated` is `False`). Keep the residual
finite on relevant trial controls through the economic parameterization and
bounds.

`result.converged` requires the updated node audit, a small control update, and
no failed node solve. A small least-squares gradient or a small update alone is
insufficient. The history exposes `policy_change`, `max_kkt_residual`,
`inner_failures`, evaluation counts, and domain-exit mass. These are numerical
conditions on this discretization; inspect the economic accuracy separately.

## Check points that were not used to solve

```python
rng = np.random.default_rng(824)
points = rng.uniform([0.12, -0.04], [0.47, 0.04], size=(64, 2))
exact = lambda state: np.array([alpha * beta * output(state)])
comparison = compare_policies(
    problem, {"time iteration": result, "analytic": exact}, points, tol=1e-3,
)
print(comparison)
audit = audit_policy(problem, result, points, tol=1e-3)
assert audit.passed
print(audit.diagnostics)

reference = np.array([exact(state) for state in points])
policy_error = float(np.max(np.abs(result(points) - reference)))
assert policy_error < 3e-5
assert audit_policy(problem, exact, points, tol=1e-12).passed

def direct_euler(state, policy):
    savings = policy(state)[0]
    consumption = output(state) - savings
    expected_return = 0.0
    for innovation, weight in zip(nodes, weights):
        next_state = np.array([savings, innovation[0]])
        next_output = output(next_state)
        next_consumption = next_output - policy(next_state)[0]
        expected_return += weight * alpha * next_output / (savings * next_consumption)
    return 1 / consumption - beta * expected_return

np.testing.assert_allclose(
    [direct_euler(state, result) for state in points],
    audit.residuals["Euler"].to_numpy(), atol=1e-12,
)
print("maximum savings error against exact policy:", policy_error)
```

`audit_policy` evaluates the supplied policy; it never solves for a replacement.
The random points above are independent of the collocation nodes, and the
direct Euler calculation independently checks the `.mod` bridge. Random points
are useful coverage, but a finite sample is not a supremum-error bound. Add
points near economically relevant boundaries and constraint switches.

| Audit field | Interpretation |
| --- | --- |
| `residuals` | Conditional expected residuals in original equation units |
| `kkt_residuals` | Residuals divided by `residual_scale`, with natural-map residuals replacing declared complementarity rows |
| `diagnostics.feasible`, `future_feasible` | Current and quadrature-successor controls satisfy their bounds and economic predicate |
| `state_inside`, `next_state_outside_mass` | Domain membership and summed integration weights of successor states outside the declared rectangle |
| `failure` | Numerical/domain failure recorded at that audit point |
| `passed` | Every supplied point passes the requested tolerance, feasibility, and default domain checks |

Failed evaluations can leave infinite sentinels in residuals or domain-exit
mass. Inspect `failure` before treating an exit-mass entry as a probability.

`max_residual` and `max_kkt_residual` summarize the first two tables. A binding
constraint can have a nonzero raw residual and a valid KKT residual. Across
models, residuals in consumption, marginal utility, or resource units cannot
be compared as if they were the same error. `residual_scale` is explicit, and
the package does not label a generic residual as a consumption-equivalent or
welfare error. `compare_policies` applies the same problem and points to every
method and uses the strict domain audit by default.

## Integrate the conditional equation

For each current state, the solver computes
\(\sum_{j=1}^{q}w_j F(s,x,g(s,x,\varepsilon_j),\pi(g(s,x,\varepsilon_j)),\varepsilon_j)\).
It does not plug mean shocks or mean future controls into a nonlinear equation.
`gaussian_quadrature(covariance, order=3)` supplies a zero-mean Gauss–Hermite
rule with covariance already incorporated in its nodes. Do not multiply those
nodes by the innovation standard deviation a second time.

Correlated and singular positive-semidefinite covariance matrices are supported.
Integration is performed in the covariance's positive-eigenvalue subspace;
the work grows as `order ** rank`, and `max_nodes` can reject an oversized rule.
A `shock_rule(state)` can instead supply state-dependent innovation nodes and
weights, including conditional Markov transition probabilities. Those weights
must be conditional on the current state, not unconditional regime frequencies.
Invalid probabilities are rejected, never silently normalized.

Gaussian integration is a finite quadrature rule, not a claim that the shock
distribution has bounded support. Even zero reported exit mass concerns only
the supplied nodes. Increase quadrature order and examine the economic tails
separately.

The following checks Gaussian moments, then solves on a wider productivity
domain so a seven-node rule can be audited without extrapolation. Both audits
evaluate **the same final policy** with different integration rules.

```python
covariance = np.array([[0.0004, 0.00015], [0.00015, 0.0009]])
joint_nodes, joint_weights = gaussian_quadrature(covariance, order=3)
np.testing.assert_allclose(joint_weights @ joint_nodes, 0, atol=1e-15)
np.testing.assert_allclose(
    joint_nodes.T @ (joint_nodes * joint_weights[:, None]), covariance, atol=1e-14,
)

fine_nodes, fine_weights = gaussian_quadrature([[0.02**2]], order=7)
wide_problem = replace(problem, domain=((0.1, 0.5), (-0.09, 0.09)),
                       shock_nodes=fine_nodes, shock_weights=fine_weights)
warm_start = replace(result.policy, extrapolation="allow")
wide_result = solve_time_iteration(wide_problem, warm_start, orders=(11, 4), tol=1e-8)
assert wide_result.converged
same_domain_rule3 = replace(wide_problem, shock_nodes=nodes, shock_weights=weights)
audit3 = audit_policy(same_domain_rule3, wide_result, points, tol=1e-4)
audit7 = audit_policy(wide_problem, wide_result, points, tol=1e-4)
assert audit3.passed and audit7.passed
print("three-node / seven-node residuals:", audit3.max_residual, audit7.max_residual)
```

Explicit extrapolation is used only to form the wider-domain initial guess;
`wide_result` itself uses default strict domain convergence and evaluation.
The Brock–Mirman exact Euler equation cancels future output, making this
benchmark unusually insensitive to the integration rule. It checks the wiring
and interpolation, not the adequacy of three nodes for other nonlinear models.

## Choose the basis and state domain

| `basis` | Interpretation of `orders` | Useful properties and limitations |
| --- | --- | --- |
| `"chebyshev"` | Scalar degree in every state, or one degree per state | Tensor Chebyshev–Lobatto nodes; `(degree + 1)` nodes per dimension. Smooth approximation; node count is the product across dimensions |
| `"smolyak"` | A single positive sparse-grid level | Uses the existing Smolyak polynomial basis; reduces tensor-grid growth but still has construction, interpolation, and node-solve costs |
| `"linear"` | One degree per state, defining the same tensor node counts | Piecewise multilinear interpolation on those nodes; useful around kinks, without polynomial oscillations between feasible nodal values for constant box bounds |

Smolyak's level is not a per-dimension polynomial degree, so changing `basis`
while keeping the same integer `orders` does not hold cost or accuracy fixed.
The current wrapper guards sparse construction at 12 state dimensions and
level 8, in addition to `max_nodes`; these are work limits, not a claim of
economic accuracy at those sizes. Existing VFI/EGM and interpolation tools
remain useful when their objective and model assumptions match the problem;
this interface is specifically for explicitly supplied equilibrium residuals.

Domain selection and anisotropic approximation matter as well as the total
number of nodes; [Judd, Maliar, Maliar and Valero (2014)](https://kenjudd.org/wp-content/uploads/2021/11/2014-Smolyak-Method-For-Solving-Dynamic-Economic-Models.pdf)
study these choices for dynamic models. The implementation here uses the
existing sparse basis and a separate Chebyshev refinement heuristic; it does
not claim to reproduce every algorithm in that paper.

Choose a domain that includes relevant states and inspect transitions from
both its interior and edges. A wider rectangle with unchanged degree can
reduce resolution where the policy is most curved. Higher degree can still
produce infeasible controls between nodes. Evaluate bounds and `admissible`
off-grid; plotting a smooth line is not a feasibility certificate.

Two options govern different decisions:

- `extrapolation="raise"` controls evaluation of the returned policy and is the
  default. `"allow"` explicitly extends its polynomial or linear interpolant;
  it does not clip a state to the nearest boundary.
- `require_domain=True` controls node-convergence acceptance. Setting it to
  `False` allows convergence conditional on that extension, while retaining
  domain-exit diagnostics. It does not disable economic feasibility checks.

Trial controls internally use polynomial/linear extension so a nonlinear trial
is not silently changed by clipping. Default final acceptance still requires
domain closure under the supplied rule. For a Gaussian process, closure under
finite quadrature is not closure under the complete shock distribution.

## Refine the approximation within a work budget

```python
adaptive = solve_adaptive_time_iteration(
    problem, initial, points, orders=(3, 2), tol=1e-4, max_solves=6,
)
print(adaptive.history)
assert adaptive.accepted

ladder = refine_time_iteration(
    problem, initial,
    levels=[{"orders": (7, 2)}, {"orders": (11, 4)}],
    points=points, tol=1e-4, max_solves=2,
)
print(ladder.history)

limited = refine_time_iteration(
    problem, initial, levels=[{"orders": (7, 2)}],
    points=points, tol=1e-4, max_solves=1,
)
assert not limited.accepted
```

Adaptive refinement increases the Chebyshev degree in the state direction with
the largest high-degree coefficient tail. The tail is a heuristic, not an error
bound. Acceptance also requires independent Euler/KKT checks and agreement
between adjacent policies on the same supplied points. Exhausting `max_solves`
does not create a pass.

The explicit refinement ladder can additionally change `domain`, `shock_nodes`
and `shock_weights`. This permits separate domain and integration checks, with
the same audit points. `basis="smolyak"` uses the existing sparse-grid basis,
with scalar `orders` interpreted as its level; the automatic anisotropic driver
currently uses Chebyshev only. `basis="linear"` uses piecewise-linear
interpolation on the tensor nodes and is useful near constraint kinks. All
methods still need independent off-grid checks and adequate state coverage.

`PolicyRefinementResult.accepted` is stronger than its final inner
`result.converged`: the final off-grid audit must pass, the inner solve must
converge, and the largest absolute control change from the previous level at
the same points must be at most the refinement tolerance. The first level
therefore cannot establish acceptance. `max_solves` truncates the ladder;
`max_nodes` guards construction and raises if a requested grid is oversized.
Neither setting is an automatic certificate at the budget boundary.

For interpretation, vary one numerical choice at a time: grid resolution,
then quadrature on a fixed policy/domain, then domain size with comparable
resolution. A ladder may change all three, but a changed answer then has
several possible causes. Store the history, final policy, and audit points
together. Fixed evaluation points facilitate reproducibility; a fresh held-out
set is useful after selecting the refinement level.

## Compare local approximations using the same expectation

```python
local_policies, local_solutions = {}, {}
for order in (1, 2, 3):
    solution = model.solve(order=order)
    local_solutions[order] = solution
    rules = solution.decision_rules()
    assert tuple(rules.state_variables) == ("k",)
    assert tuple(rules.shock_names) == ("e",)
    capital_index = list(rules.variable_names).index("k")

    def local_policy(state, solution=solution, capital_index=capital_index):
        values = evaluate_perturbation(solution, [state[0]], [state[1]])
        return values[[capital_index]]

    local_policies[f"order {order}"] = local_policy
local_policies["global"] = adaptive.result
local_comparison = compare_policies(problem, local_policies, points, tol=1e-4)
print(local_comparison)
local_diagnostics = diagnose_perturbation(local_solutions[3])
print(local_diagnostics.summary())

local_seeded = solve_time_iteration(problem, local_policies["order 2"], orders=(7, 2), tol=1e-8)
assert local_seeded.converged
np.testing.assert_allclose(local_seeded(points), result(points), atol=1e-7)

batch = evaluate_perturbation(local_solutions[3], points[:, :1], points[:, 1:2])
assert batch.shape == (len(points), len(model.variables))
```

The global state includes current observed productivity, whereas the local
decision rule's predetermined state in this i.i.d. benchmark is only inherited
capital. The code explicitly supplies current productivity as the current
innovation. In an AR(1) model, supply the lagged productivity and current
innovation consistently with its law of motion; do not confuse observed
current productivity with its lag.

All policies in the comparison use the same exact static accounting identities
and integration rule. They are compared over the declared points, not over
different simulation draws or different shock variances. An increased local
order may improve a neighborhood and still fail farther away or violate a
bound. Do not require a preferred ordering of errors before observing them.

The second-order rule also provides an alternative feasible initialization of
time iteration. Agreement from two initializations is useful numerical
evidence, not a proof of equilibrium uniqueness. In other models or wider
domains, a Taylor initial policy may violate bounds and must be reconsidered.

### Levels, coefficient order, and risk scaling

`evaluate_perturbation` evaluates coefficients already present in the solution.
Its optional `order` can truncate the available approximation; it cannot create
third-order coefficients from a first-order result. The inputs and output are:

| Quantity | Convention |
| --- | --- |
| `lagged_states` | Previous-period state levels in `decision_rules().state_variables` order; one vector or a row batch |
| `shocks` | Current realized innovations in `shock_names` order; omitted means zero, and one vector broadcasts across rows |
| `deviations=True` | Changes only the state input to deviations; output remains endogenous levels |
| Returned array | Current levels in `variable_names` order; `(n_variables,)` or `(n_points, n_variables)` |
| `sigma` | Nonnegative multiplier for anticipated risk terms; it does not rescale the supplied innovations |

For example, hold the realized innovation fixed when isolating the retained
risk terms:

```python
third_order = local_solutions[3]
ss = third_order.decision_rules().ys
lagged_capital = np.array([float(ss["k"])])
realized_innovation = np.array([0.01])
with_risk = evaluate_perturbation(third_order, lagged_capital, realized_innovation, sigma=1)
without_risk = evaluate_perturbation(third_order, lagged_capital, realized_innovation, sigma=0)
print("retained risk-term contribution:", with_risk - without_risk)
```

In coefficient notation, the evaluator includes the second-order constant
`ghs2 * sigma**2 / 2` and the third-order mixed terms
`(ghxss @ state_deviation + ghuss @ innovation) * sigma**2 / 2`, together with
the ordinary linear, quadratic, and cubic terms and their Taylor factorials.
The covariance used to solve those coefficients is already embedded in their
construction. `sigma=0` is an evaluation of the retained expansion with those
risk terms removed, not a fresh solution of a recalibrated stochastic model.
Relative covariance changes require re-solving the approximation.
This log/full-depreciation benchmark has no precautionary savings response;
small or zero risk contributions here should not be generalized to RBC models.

Orders two and three use direct tensor contractions without building Kronecker
vectors. These are raw Taylor policy evaluations for initialization and residual
checks. A pruned simulation retains separate order components of its state
history and is a different calculation. `diagnose_perturbation` reports the
selected transition roots near the unit circle and original dynamic Jacobian
rank/conditioning when retained. It does not replace a Blanchard–Kahn
classification or certify nonlinear uniqueness. The original Jacobian rank
uses the concatenated lag/current/lead blocks when retained, and its condition
number depends on equation and variable units. Missing Jacobians are reported
as not checked. A near-unit transition root is evidence of persistence and
sensitivity, not by itself a proof of an invalid equilibrium.

**Higher-order Sylvester conditioning and exact factorization reuse.** The
recursive Real Schur (Bartels–Stewart) Sylvester solver keys its LU cache by
exact IEEE-754 floating-point values (`float` for $1\times 1$ blocks and four
floats for $2\times 2$ blocks), eliminating the 12-decimal rounding collision
reported in 5.7.1 where distinct near-unit Schur products could reuse a
factorization. `diagnose_perturbation(solution, max_curvature_order=3)` reports
`schur_block_condition` (the fast diagonal Schur-block lower bound on the
higher-order operator condition number), `curvature_operator_condition` and
`curvature_coupling_amplification` (the full Kronecker operator condition number
and off-diagonal Schur coupling ratio, summarized in `curvature_conditioning`),
`sylvester_residuals` (`ghxx`, `ghs2`, `ghxxx`, `ghxss`, `ghuss`), and
`sylvester_work` (`SylvesterWorkCounts`) so higher-order operator
ill-conditioning is distinguished from the first-order dynamic Jacobian
condition number.

Pruning avoids recursively generating terms beyond the chosen approximation
order and carries separate state components. The distinction between a raw
Taylor rule and a pruned state-space recursion is developed by
[Andreasen, Fernández-Villaverde and Rubio-Ramírez](https://public.econ.duke.edu/~jfr23/Current/Pruning_20140120.pdf).
An audit of the raw rule is therefore not an audit of an arbitrary pruned
simulation history. Use the solution's pruned simulation methods for those
paths; do not reconstruct pruning by repeatedly feeding total simulated states
into `evaluate_perturbation`.

## Complementarity and a binding-bound example

Bounds alone restrict admissible controls; they do not turn equations into
inequalities. With `complementarity={equation_index: control_index}`, let `x`
denote that control and `F` its conditional expected residual. The sign
convention is:

\[
x=\ell\Rightarrow F\ge0,\qquad
\ell<x<u\Rightarrow F=0,\qquad
x=u\Rightarrow F\le0.
\]

The reported natural-map residual is
\(x-\operatorname{clip}(x-F/d,\ell,u)\), where `d` is the positive equation
scale. Its zero set expresses those KKT conditions. This projection is part of
the complementarity equation; it does not clip the returned policy or the
state transition. Feasibility is checked separately. The mapping is explicit
because the economic control need not have the same index as its equation.

Here a nonnegative adjustment `x` minimizes `(x-s)**2 / 2`. Future profitability
is `0.5*s` and is unaffected by the decision, so the exact policy is
`max(s, 0)`. This small constrained economic problem isolates the sign and
binding-bound contract without a hidden reference solve.

```python
constrained_problem = EquilibriumProblem(
    state_names=("profitability",), control_names=("adjustment",),
    domain=((-1, 1),), transition=lambda s, x, e: 0.5 * s,
    residual=lambda s, x, sp, xp, e: x - s,
    control_bounds=[(0, 2)], complementarity={0: 0},
)
constrained = solve_time_iteration(
    constrained_problem, [0.5], basis="linear", orders=2, tol=1e-7,
)
constraint_points = np.linspace(-0.98, 0.98, 41)[:, None]
constraint_audit = audit_policy(constrained_problem, constrained, constraint_points, tol=2e-6)
assert constrained.converged and constraint_audit.passed
np.testing.assert_allclose(
    constrained(constraint_points)[:, 0], np.maximum(constraint_points[:, 0], 0), atol=2e-6,
)
assert constraint_audit.max_residual > 0.9
assert constraint_audit.max_kkt_residual < 2e-6
```

The nonzero raw residual at negative profitability is the correct lower-bound
inequality. Forcing it to zero would reject the economically valid corner.
The kink is represented exactly by the linear basis in this special example;
arbitrary nonlinear constraint boundaries still require refinement and
off-grid checks. General equilibrium KKT conditions are necessary conditions
supplied by the modeler, not a guarantee of a unique global welfare optimum.

## Export results and plot an explicit policy slice

Policy audits, time-iteration results, refinement results and local diagnostics
expose `to_frame()` / `to_dataframe()`, `to_markdown()`, `to_latex()` and
`to_typst()` through the package's shared table formatters. Refinement tables
contain their solve history; local diagnostic tables contain transition roots.

```python
print(audit.to_markdown(index=False))
slice_points = np.column_stack((np.linspace(0.12, 0.47, 40), np.zeros(40)))
figure = result.plot(slice_points, state="inherited_capital", controls=["savings"], show=False)
axis = figure.axes[0]
assert adaptive.plot(slice_points, state=0, ax=axis) is axis
latex_table = local_diagnostics.to_latex(index=False)
typst_table = adaptive.to_typst(index=False)

import matplotlib.pyplot as plt
plt.close(figure)
```

Plots require full state coordinates. Here productivity is explicitly held at
zero. `state` selects the horizontal coordinate and `controls` selects plotted
controls; point order is preserved. Passing `ax` returns that axis; otherwise
the method returns a new figure. Matplotlib is loaded only when plotting, and
no display is requested unless `show=True`. Refinement results delegate to the
final policy with the same explicit points.

`result.to_frame()` contains state nodes and nodal controls;
`audit.to_frame()` contains per-point diagnostics. Detailed residual tables
remain available separately. Exporters return strings and do not write files
or publish output.

## Domain exits, failures, and limits

Default convergence requires node residuals, policy-change tolerance, feasibility
and closure under the supplied integration rule. Policies raise when evaluated
outside their domain. For a deliberately truncated persistent-shock domain,
`extrapolation="allow", require_domain=False` permits a solve **conditional on
the interpolation extension**, retaining its exit-mass diagnostics. This is
not validation of the omitted region. In particular, the MAV AR(1) example's
endpoint quadrature exits remain visible even when a strict interior audit
passes. No shocks or policies are silently clipped.

For an AR(1), even a stationary process can exit a finite domain. Increasing
the persistence or quadrature order can increase reported edge exits without
indicating an algebraic error. Expand the domain, examine a justified interior
region, or explicitly accept conditional extrapolation and report that choice.
Simply clipping shocks changes the integration law and can manufacture a
misleading residual pass. The default `audit_policy` remains strict even if
the solver was run with `require_domain=False`; conditional auditing requires
its own explicit `require_domain=False`.

On numerical failure or an iteration limit, the result remains
`converged=False` and a warning is emitted; `strict=True` raises instead.
The last iterate and iteration history remain available in the ordinary
non-strict result. Invalid shapes, probabilities, calibration mappings, and
oversized grids are input errors and may raise immediately instead.

When a solve fails, use the history to distinguish a node-solve failure from
stagnation, domain exits, and infeasible future controls. Verify timing and
equation signs first; then examine scales, a feasible initialization, damping,
and a small well-resolved domain. More iterations do not repair a missing
state, wrong conditional expectation, or an incompatible constraint mapping.

The method is intended for manageable continuous-state equilibria. Multiple
equilibria, discontinuous regimes, highly nonlinear constraints, unbounded
tails, and large DSGE/HANK state spaces need model-specific treatment. A finite
set of residual and refinement checks supplies evidence about the reported
approximation; it does not establish existence, uniqueness, transversality,
or accuracy everywhere.
