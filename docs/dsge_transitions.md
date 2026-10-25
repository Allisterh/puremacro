> 🇬🇧 English · 🇪🇸 [Español](es/dsge_transitions.md)

# Nonlinear transitions, domains and announcement dates

Puremacro can solve nonlinear deterministic boundary problems supplied as a
supported `.mod` model or as Python equilibrium residuals. The equations are
solved jointly across time; a prior perturbation solution is not required.
This does **not** imply that every model has a solution, that the solution is
unique, or that Newton iteration will find it from arbitrary starting values.
The model must fit the supported timing and equation interfaces, and its
boundary conditions, domains and information assumptions remain the caller's
economic choices.

| Entry point | What it computes |
| --- | --- |
| `solve_perfect_foresight` | One finite deterministic boundary problem. |
| `solve_transition` | The same problem with explicit continuation, domain transforms, equation scales and optional horizon acceptance. |
| `solve_announced_path` | Realized segments of finite plans revised on specified information dates. |
| `extended_path` | Rolling plans with the current innovation realized and future innovations set to zero; this is not nonlinear integration over future risk. |

These routines compute trajectories, not a global stochastic policy function.
For that separate problem, see [global equilibrium methods](dsge_global.md).
The [DSGE workflow guide](dsge_workflows.md) compares the available methods and
connects them to the teaching notebooks.
For model loading and command support, see the [nonlinear model guide](dsge_nonlinear.md)
and [`.mod` inspection guide](dsge_mod_inspection.md).

## The finite-horizon mathematical problem

Write the model with `n` endogenous variables and `m` exogenous inputs as

\[
f(y_{t+1},y_t,y_{t-1},\varepsilon_t;\theta,\bar y)=0,
\qquad t=1,\ldots,T.
\]

Each call returns `n` residuals. The unknown vector is
\(Y=(y_1^\top,\ldots,y_T^\top)^\top\in\mathbb R^{nT}\).
The initial lag and final lead are fixed:

\[
y_0=y_{\mathrm{init}},\qquad y_{T+1}=y_{\mathrm{end}}.
\]

The stack contains **T interior dates**, with `n*T` equations and unknowns.
The solver does not impose an extra equation saying `y_T == y_end`. A model
without leads does not use the final boundary in its dynamic residuals; a model
without lags does not use the initial boundary there. Supply a complete vector
in declared variable order nevertheless. The equations determine which entries
of that vector are economically relevant; the solver does not infer a new
state/jump-variable boundary specification for the caller.

The residual callback order is exactly
`f(lead, current, lag, shock)`. In a `.mod` equation, the residual orientation is
left-hand side minus right-hand side. The interface shown here has one lead,
one current value and one lag; a model needing longer histories must be reduced
to this form with explicit auxiliary states or supported parser transformations.
This guide does not promise automatic support for every timing construct.

`y_end` is also the fixed reference \(\bar y\) for `STEADY_STATE()` within one
solve. It is not replaced by the current iterate. A callback that explicitly
declares a `steady_state` argument receives the same reference. An explicit
terminal vector is a boundary assignment; it need not be a steady state merely
because the argument is named `y_end`. The low-level alias `y_ss` has the same
role. Bare `.mod` `endval` assignments and explicit terminal steady-state solves
are distinct operations; see the model guide for command ordering.

### Array and date conventions

| Object | Convention |
| --- | --- |
| `y_init`, `y_end` | One physical value per endogenous variable. `solve_transition` also accepts mappings keyed by variable name. |
| `shocks` | Prefer shape `(T, m)`; row `0` is date `1`. For one shock, a length-`T` vector is also accepted. Zero shocks use shape `(T, 0)`. Values are actual exogenous inputs in the equations, with no automatic logarithm, percentage conversion or standard-deviation scaling. |
| `initial_path` | Physical levels, shape `(T, n)`, with no boundary rows. In continuation it initializes the baseline solve; accepted paths warm-start subsequent stages. |
| Returned `.path` | A DataFrame indexed `1,...,T`, containing interior dates only. Horizon refinement can return more rows than the original shock path. |
| `terminal_error` | `max(abs(y_T - y_end))`; a descriptive distance, not an extra equilibrium residual or automatic proof of truncation accuracy. |

`solve_transition` derives `T` from the supplied shock path. Do not pass
`periods`, `n_periods`, `exogenous_path`, `y_ss` or `surprise` to this wrapper;
they conflict with its explicit experiment. For the lower-level
`solve_perfect_foresight`, pass `periods=T` explicitly: omitting it does not infer
the horizon from `shocks`. That lower-level API also accepts a legacy `T+2`-row
shock array and drops its first/last rows, or truncates other longer arrays.
Exactly `T` rows avoids that ambiguity.

## Start from a nonlinear model

`read_mod` constructs a nonlinear specification without first solving a local
approximation. The following small model keeps the timing visible. Run this
guide's Python blocks **in order in one session**; later blocks reuse these
imports and the linear example. English and Spanish use identical executable
blocks. Plotting in the final block requires Matplotlib.

```python
import numpy as np
from puremacro.dsge import read_mod, solve_perfect_foresight, solve_transition

model = read_mod("""
var x; varexo e;
model; x = 0.25*x(+1) + 0.4*x(-1) + e; end;
initval; x=0; e=0; end;
""")
shocks = np.r_[0.1, np.zeros(39)][:, None]
transition = solve_transition(
    model, y_init=[0.0], y_end=[0.0], shocks=shocks,
    horizon_tolerance=1e-8, compare_periods=20,
    max_periods=160, shock_tail=[0.0], linear_solver="krylov",
)
assert transition.converged, transition.summary()
assert transition.reason == "horizon_accepted"
path = transition.path
print(transition.summary())
```

The target is a date-1 innovation followed by zeros. The workflow solves that
experiment and then checks that doubling the horizon has a sufficiently small
effect on the first 20 dates. `transition.path` returns a copy; always inspect
`transition.converged` before treating it as an accepted target.

## Direct sparse Newton or matrix-free Newton–Krylov

At a current trajectory define

\[
A_t=\frac{\partial f_t}{\partial y_{t+1}},\quad
B_t=\frac{\partial f_t}{\partial y_t},\quad
C_t=\frac{\partial f_t}{\partial y_{t-1}}.
\]

For fixed boundaries, the stacked Jacobian and its action are

\[
J=\begin{bmatrix}
B_1&A_1&&\\
C_2&B_2&A_2&\\
&\ddots&\ddots&\ddots\\
&&C_T&B_T
\end{bmatrix},\qquad
(Jv)_t=A_tv_{t+1}+B_tv_t+C_tv_{t-1},
\quad v_0=v_{T+1}=0.
\]

Both backends seek a Newton direction from \(Jd=-F(Y)\) and use a line search.
`linear_solver="direct"` assembles the sparse stacked matrix and solves the
linear system directly. For a parsed model passed directly to
`solve_perfect_foresight`, symbolic incidence limits stored entries to the
structural pattern. Callback blocks, including transformed wrapper callbacks,
may store a fuller per-date pattern.

`linear_solver="krylov"` applies the stacked Jacobian through products and uses
LGMRES by default; restarted GMRES is an explicit option. It is useful when
forming/factoring the full stack is costly, provided the linear iteration has
an effective preconditioner. Matrix-free products do not make every storage
cost small: per-date blocks and their factors can still be dense. Neither
backend is guaranteed to be faster on every model.

`method="auto"` uses symbolic first derivatives for parsed models, preserving
the fixed `STEADY_STATE` reference. For callbacks it probes complex-step
compatibility and otherwise uses central differences. Supply
`jacobian_fn(lead,current,lag,shock) -> (A,B,C)` to control the derivatives, as
below. `method="analytic"` requires a parsed model or that callback. Explicit
`method="central"` and `"complex"` remain available; complex-step requires the
residual implementation to preserve complex perturbations and be suitable for
that differentiation. An automatic probe is not a proof of that property.

```python
def equilibrium(lead, current, lag, shock):
    return current - 0.25*lead - 0.4*lag - shock

def date_jacobian(lead, current, lag, shock):
    return np.array([[-0.25]]), np.array([[1.0]]), np.array([[-0.4]])

fixed_problem = dict(
    y_init=[0.0], y_end=[0.0], periods=len(shocks), shocks=shocks,
    variable_names=["x"], tol=1e-10,
)
direct = solve_perfect_foresight(model, **fixed_problem)
krylov = solve_perfect_foresight(
    equilibrium, jacobian_fn=date_jacobian,
    linear_solver="krylov", **fixed_problem,
)
assert direct.converged and krylov.converged
np.testing.assert_allclose(direct.path, krylov.path, atol=1e-9, rtol=0)
assert krylov.solver_diagnostics["jvp_method"] == "analytic"
```

### Preconditioners and supported combinations

The default Krylov preconditioner factors a block-tridiagonal system formed
from fixed terminal-reference Jacobian blocks. Named preconditioners include
`steady_block_tridiagonal`, `steady_splu` and `current_block_tridiagonal`; the
last rebuilds from the current trajectory. See the
[stacked Newton guide](dsge_stacked_newton.md) for the full operator protocol,
block-Jacobi options and linear residual diagnostics.

`solve_transition` and Krylov `extended_path` reuse the default steady
factorization only if the horizon, block shapes, dtypes and numerical entries
match **exactly**. Changing parameters, reference states or horizon can require
a rebuild. `reuse_preconditioner=False` disables the cache. An explicit
`krylov_options["preconditioner"]`, including `None`, or an explicit `time_block`
setting takes precedence over automatic reuse. The cache count reports its own
builds; it does not count every factorization inside a user-supplied operator.

| Requested combination | Current behavior |
| --- | --- |
| Direct + ordinary equations, domains/scales or MCP | Supported. |
| Direct + terminal policy, including explicit MCP mapping | Supported; the terminal chain rule is included. |
| Krylov + ordinary equations and domain/scale transforms | Supported. |
| Krylov + MCP bounds or `mcp=True` | Supported; active-row preconditioned semismooth Newton–Krylov. |
| Krylov + `terminal_policy` | Supported; terminal chain rule enters the date-`T` JVP and preconditioner block. |
| Direct + any non-`None` `krylov_options` | Rejected explicitly. |
| `terminal_jacobian` without `terminal_policy` | Rejected explicitly. |

The high-level `krylov_options` keys are `preconditioner`, `forcing`,
`krylov_method`, `inner_m`, `outer_k`, `krylov_maxiter`, `side`, `max_log_move`,
`time_block`, `progress` and `checkpoint`. Other lower-level engine controls
are not automatically accepted here. The chosen Krylov solver never silently
switches the nonlinear problem to the direct backend. A block-Thomas
preconditioner may use its documented SuperLU factorization fallback when a
pivot is unsuitable; that is a preconditioner construction choice, not a switch
of nonlinear solver.

## Continuation, baseline and solve budget

Continuation solves a family of explicitly stated experiments. For baseline
inputs \((E^0,b^0,\theta^0)\) and target inputs \((E^1,b^1,\theta^1)\),

\[
E^\lambda=(1-\lambda)E^0+\lambda E^1,\qquad
b^\lambda=(1-\lambda)b^0+\lambda b^1,\qquad
\theta^\lambda=(1-\lambda)\theta^0+\lambda\theta^1.
\]

The implementation copies endpoint values exactly at fractions zero and one.
The initial lag `y_init` stays fixed. The terminal boundary and its
`STEADY_STATE` reference follow `b` at each stage. A supplied terminal policy is
kept as supplied; it is not silently recalibrated when parameters change.

The default baseline is zero shocks and `baseline_end=y_init`. This is a
computational baseline, not an assertion that the initial state is a steady
state. Supply `baseline_shocks` with the target path's shape and `baseline_end`
when a different baseline is economically intended. The baseline itself must
solve successfully. `continuation=False` requests one target solve directly.

For a `NonlinearModel`, `baseline_params` varies only the named, already
calibrated parameters; others retain their target values. The source model's
parameter dictionary is not mutated. No intermediate steady states are solved
automatically. Supported parameter-dependent MCP tag bounds are resolved at
each stage's calibration. User-supplied residual/derivative/terminal callbacks
must remain consistent with the experiment they represent.

The starting increment is `initial_step=0.25`. A failed stage halves the
increment; an accepted stage multiplies it by 1.5, capped at the remaining
distance to one. `min_step=1/1024` bounds reductions. `max_iter` applies to each
numerical solve, whereas `max_solves=40` counts the baseline, accepted and failed
continuation attempts, and horizon refinements together. These are iteration
and solve-count budgets, not wall-clock deadlines.

```python
calibrated = read_mod("""
var x; parameters theta; theta=999;
model; x^3 = 1 + theta; end;
initval; x=1; end;
""")
parameter_path = solve_transition(
    calibrated, y_init=[1.0], y_end=[1.0], shocks=np.zeros((3, 0)),
    baseline_params={"theta": 0.0}, max_iter=4, max_solves=40, tol=1e-9,
)
assert parameter_path.converged, parameter_path.summary()
np.testing.assert_allclose(parameter_path.path["x"], 10.0, atol=1e-9)
assert parameter_path.continuation_history[-1]["fraction"] == 1.0
assert calibrated.parameter_values["theta"] == 999.0

limited = solve_transition(
    model, y_init=[0.0], y_end=[0.0], shocks=shocks, max_solves=1,
)
assert not limited.converged
assert limited.reason == "solve_budget_exhausted"
```

The cubic example has no exogenous variables, hence the `(3, 0)` shock array.
Its exact target is `x=10`. The last check demonstrates that a solved baseline
alone is never reported as target success. Continuation can help reach a root;
it does not remove singularities, guarantee a branch is unique, or ensure that
all possible baselines lead to the same target solution.

## Economic domains and residual scales

`variable_bounds` specifies **open domains**. With positive
`variable_scale[j]=s_j`, the solver uses monotone coordinates:

| Domain for `y_j` | Physical value decoded from `z_j` |
| --- | --- |
| Unbounded | \(y_j=s_jz_j\) |
| \(y_j>\ell_j\) | \(y_j=\ell_j+\exp(s_jz_j)\) |
| \(y_j<u_j\) | \(y_j=u_j-\exp(-s_jz_j)\) |
| \(\ell_j<y_j<u_j\) | \(y_j=\ell_j+(u_j-\ell_j)/(1+\exp(-s_jz_j))\) |

Boundary vectors, initial guesses and terminal-policy outputs must be strictly
inside these domains. During transformed residual evaluations, overflowing or
numerically saturated candidates are rejected before the equilibrium callback;
no economic policy is clipped to a feasible value. This protects logarithms and
powers defined only in the declared domain, but it cannot cure every
ill-conditioned root.

`equation_scale[i]=q_i>0` replaces row `f_i` by `f_i/q_i`. For an ordinary
transition, acceptance requires both backend convergence and

\[
\max_{t,i}\left|f_i(y_{t+1},y_t,y_{t-1},\varepsilon_t)/q_i\right|
\leq\texttt{tol}.
\]

That explicit row-scaled check is stricter than merely trusting a backend flag
whose residual test may also use a Jacobian magnitude scale. It does not use
`variable_scale` as a substitute for economically meaningful equation units.
Scales can be positive scalars, vectors or full mappings. A mapping follows
residual row order through declared variable names; use a vector when variable
names would obscure a different equation order.

```python
def log_equation(lead, current, lag, shock):
    return 1000.0 * (np.log(current) - shock)

positive = solve_transition(
    log_equation, y_init=[1.0], y_end=[1.0],
    shocks=np.full(10, -3.0), variable_names=["c"],
    variable_bounds={"c": (0.0, None)},
    equation_scale=[1000.0], variable_scale=[2.0], tol=1e-9,
)
assert positive.converged, positive.summary()
np.testing.assert_allclose(positive.path["c"], np.exp(-3.0), rtol=1e-8)
assert positive.scaled_residual_norm <= 1e-9
physical_error = np.max(np.abs(1000.0*(np.log(positive.path["c"]) + 3.0)))
np.testing.assert_allclose(positive.solution.residual_norm, physical_error, atol=1e-12)
```

For an ordinary `PerfectForesightResult` inside `transition.solution`,
`residual_norm` and `initial_residual_norm` retain original equation units;
`transition.scaled_residual_norm` reports the row-scaled acceptance measure.
For `MCPResult`, residuals and complementarity diagnostics remain in the
transformed/scaled solver units, even though the returned path is decoded to
physical levels. Check physical feasibility and complementarity independently
when their economic tolerances matter.

An open domain cannot represent a constraint that should bind. Use MCP for
that condition. If a variable has both a domain and a complementarity bound,
the latter must lie strictly within its domain to have a finite transformed
coordinate; a zero lower domain and a binding zero lower bound are incompatible.

## Horizon acceptance and permanent shock tails

A small stacked residual verifies a finite boundary problem. It does not
establish that the chosen boundary adequately approximates an infinite horizon.
With `horizon_tolerance`, the workflow first solves the exact target and then
sets \(T'=\min(2T,\texttt{max_periods})\). It preserves the initial condition,
shock prefix, target calibration and terminal policy/reference, extends the
shock path explicitly, and warm-starts the longer solve.

For `P=compare_periods`, the acceptance statistic is

\[
\Delta_{T,T'}=\max_{1\leq t\leq P}\max_j
\left|y^{T'}_{t,j}-y^T_{t,j}\right|.
\]

This is an **absolute difference in physical variable units**. It is not
divided by `equation_scale` or `variable_scale`. Choose its tolerance with the
reported variables' units in mind. `compare_periods` cannot exceed the initial
horizon; `max_periods` must exceed it. Refinement stops after the first accepted
comparison, at the horizon cap, after a failed extended solve, or at the shared
solve budget. The comparison establishes only the stated common-window check.

An explicit `shock_tail` is mandatory. It may be a scalar/row broadcastable to
the extra dates, or a callable `shock_tail(old_T, new_T)` returning the extension.
The arguments are horizon lengths, not one-based shock dates. No silent
permanent-versus-transitory interpretation is imposed.

For example, if log productivity satisfies
\(a_t=\rho a_{t-1}+u_t\), changing `y_end` alone does not hold productivity at
`a_star`. From `a_0=0`, a permanent level `a_star` beginning at date 1 requires
`u_1=a_star`, then `u_t=(1-rho)*a_star`. The same recurring innovation must be
used in a refined tail:

```python
productivity = read_mod("""
var a; varexo innovation; parameters rho; rho=0.9;
model; a = rho*a(-1) + innovation; end;
initval; a=0; innovation=0; end;
""")
rho, target_a, initial_horizon = 0.9, 0.2, 12
permanent_innovations = np.full((initial_horizon, 1), (1.0-rho)*target_a)
permanent_innovations[0, 0] = target_a
permanent = solve_transition(
    productivity, y_init=[0.0], y_end=[target_a],
    shocks=permanent_innovations, continuation=False,
    horizon_tolerance=1e-10, compare_periods=12, max_periods=48,
    shock_tail=[(1.0-rho)*target_a],
)
assert permanent.converged, permanent.summary()
np.testing.assert_allclose(permanent.path["a"], target_a, atol=1e-10)
assert permanent.horizon_history[-1]["accepted"]
```

`horizon_history` records old/new horizons, the comparison window, the maximum
difference, tolerance and acceptance. `terminal_error` can remain positive in a
valid finite solution and is not used as a replacement for this comparison.

## Announcements, surprises and belief revisions

In one perfect-foresight solve, the entire supplied shock path is known at the
start. A nonzero shock at date 8 can therefore alter dates 1–7 through forward
equations. A surprise revealed only at date 8 is a different information
experiment, even if the realized exogenous array eventually looks the same.

`ShockAnnouncement(date, shocks, y_end=None)` records a new full-calendar belief
path. The first announcement must occur at date 1; later dates must strictly
increase within the same calendar. Every array has the same shape. Revisions
may change current/future shocks and optionally the terminal boundary/reference,
but all rows before the announcement date must equal previously realized
shocks. If `y_end` is omitted, the previously announced boundary persists.

```python
from puremacro.dsge import ShockAnnouncement, solve_announced_path

news = np.zeros((40, 1))
news[7, 0] = 0.1
anticipated = solve_announced_path(
    model, [ShockAnnouncement(1, news)], y_init=[0.0], y_end=[0.0],
)
surprise = solve_announced_path(
    model, [ShockAnnouncement(1, np.zeros_like(news)), ShockAnnouncement(8, news)],
    y_init=[0.0], y_end=[0.0],
)
assert anticipated.converged and surprise.converged
assert np.max(np.abs(anticipated.path.iloc[:7].to_numpy())) > 1e-4
np.testing.assert_allclose(surprise.path.iloc[:7], 0.0, atol=1e-12)
assert surprise.solver_diagnostics["completed_periods"] == 40
```

At announcement date `d`, the planner solves dates `d,...,T`, starting with the
actual state from `d-1`, and implements only the segment before the next
announcement. Future revisions are never revealed early. The remaining
calendar shrinks; it is not silently replaced by a constant rolling horizon.
A failed plan stops implementation and returns `converged=False`, the completed
prefix, and per-announcement diagnostics. There is no single ex ante
perfect-foresight audit context for the spliced realization.

`solve_announced_path` accepts the low-level perfect-foresight solver options,
not the continuation/domain/horizon wrapper options. `surprise`, `periods`,
`n_periods`, `exogenous_path`, `y_ss` and a single `initial_path` conflict with
its dated plans and are rejected explicitly. Automatic horizon acceptance for
this whole information calendar is not currently provided; compare explicitly
constructed longer calendars if needed.

For repeated one-period surprises, `simulate_surprise_shocks` instead solves a
fixed rolling horizon, realizes its first date, and repeats. Its default future
shock baseline is zero. `solve_perfect_foresight(..., surprise=True)` dispatches
to that rolling experiment. `extended_path` can supply or draw the current
innovation and likewise sets future innovations to zero. These future paths
are a certainty-equivalent approximation; nonlinear expectations are not
obtained merely by substituting a mean innovation into nonlinear equations.

## Complementarity: variables, equation rows and signs

For a lower bound on variable `j`, explicit mapping `(i,s)` means

\[
y_{t,j}-\ell_j\geq0,\qquad s f_{t,i}\geq0,\qquad
(y_{t,j}-\ell_j)s f_{t,i}=0,\qquad s\in\{-1,+1\}.
\]

For an upper bound, the conditions use `u_j-y_j >= 0` and `s*f_i <= 0`.
With two finite bounds, `s*f_i` is nonnegative at the lower bound, zero in the
interior and nonpositive at the upper bound. Other equation rows remain
equalities. This is why an active constraint's original equality residual need
not be zero.

`mcp_bounds` maps variable names or zero-based variable indices to `(lower, upper)`;
use `None` for an absent bound. `mcp_equations` maps those same keys to a
zero-based residual row or `(row, sign)`. An integer row alone means sign `+1`.
An explicit mapping must cover exactly the bounded variables, with distinct
equation rows. Equation order need not match variable order.

The supported source tag `[mcp='r > lower']` means a lower bound on `r`, paired
with the equation carrying the tag and its existing left-minus-right sign.
Tags admit `<`, `<=`, `>` and `>=` with one variable and a finite bound expression
depending only on calibrated parameters. The notation represents a closed MCP
bound, not an open domain. Dynamic bounds, duplicate variable tags and arbitrary
inequalities are not a general supported constraint language; two-sided bounds
use the Python API.

```python
bounded_model = read_mod("""
var x r; varexo e; parameters lower; lower=0;
model;
[mcp='r > lower'] r=e;
x=1-3*r;
end;
initval; x=1; r=0; e=0; end;
""")
bounded = solve_transition(
    bounded_model, y_init=[1.0, 0.0], y_end=[1.0, 0.0],
    shocks=-np.ones((5, 1)), mcp=True,
)
assert bounded.converged, bounded.summary()
np.testing.assert_allclose(bounded.path["r"], 0.0, atol=1e-8)

def reversed_residual(lead, current, lag, shock):
    x, r = current
    return np.array([shock[0]-r, x+3.0*r-1.0])

explicit = solve_perfect_foresight(
    reversed_residual, y_init=[1.0, 0.0], y_end=[1.0, 0.0],
    periods=5, shocks=-np.ones((5, 1)), variable_names=["x", "r"],
    mcp_bounds={"r": (0.0, None)}, mcp_equations={"r": (0, -1)},
)
assert explicit.converged
np.testing.assert_allclose(explicit.path, bounded.path, atol=1e-8)
gap = bounded.path["r"].to_numpy()
margin = gap + 1.0
assert np.min(gap) >= -1e-8 and np.min(margin) >= -1e-8
assert np.max(np.abs(np.hypot(gap, margin)-gap-margin)) < 1e-8
```

Here variable `r` is second, but its complementary equation is first. The tag
implies `mcp_equations={"r": (0, 1)}`; the callback deliberately reverses that
equation and therefore uses `(0, -1)`. The final assertions check physical
nonnegative gaps and a Fischer–Burmeister residual independently of the
solver's flag.

Use `mcp=True` to activate supported model tags. Explicit nonempty bounds also
select the native MCP solver. Without explicit mappings/tags, the older Python
bound interface retains its loading-based row/sign heuristic; supplying the
mapping is clearer when equations have economic sign restrictions. This is the
native semismooth solver and does not claim equivalence to every Dynare
complementarity algorithm or option. Both `linear_solver="direct"` and
`linear_solver="krylov"` support MCP bounds and tags; Krylov updates its active
row projection and preconditioner when the active constraint set changes, and
records `physical_mcp_feasibility`, `physical_complementarity_residual`, and
`physical_residual_norm` on both `MCPResult` and `TransitionResult`.

## Terminal policies and the final Jacobian block

If a reliable terminal map `g` is available, `terminal_policy(last_state)` sets
\(y_{T+1}=g(y_T)\). It must return a finite vector in the same variable order.
`y_end` remains the fixed `STEADY_STATE` reference. The last diagonal block
becomes

\[
\frac{d f_T}{d y_T}=B_T+A_T Dg(y_T).
\]

Provide `terminal_jacobian(last_state)` as a square matrix, or let the solver
differentiate the map centrally. With domains/scales, `solve_transition`
accepts these callbacks in physical coordinates and composes their chain rule
with the coordinate transformation (`physical_scale` and
`normalized_horizon_tolerance` also provide unit-invariant horizon acceptance
when variables span disparate scales).

For the linear example, a stable tail `x_t=r^t*x_0` satisfies
`0.25*r**2-r+0.4=0`. Using its smaller root gives an exact tail for this specific
shock-free linear model:

```python
stable_root = (1.0 - np.sqrt(1.0 - 4.0*0.25*0.4)) / (2.0*0.25)
terminal = solve_perfect_foresight(
    model, y_init=[1.0], y_end=[0.0], periods=5, shocks=np.zeros((5, 1)),
    terminal_policy=lambda last: stable_root * last,
    terminal_jacobian=lambda last: np.array([[stable_root]]), tol=1e-10,
)
assert terminal.converged
np.testing.assert_allclose(terminal.path["x"], stable_root**np.arange(1, 6), atol=1e-10)

from puremacro.validation import audit_result
terminal_audit = audit_result(terminal, checks=["equations"])
assert np.max(np.abs(terminal_audit.maps["path_equations"])) < 1e-9
```

A local linear terminal map in a nonlinear model is an approximation whose
adequacy still needs a horizon or economic check. Both `"direct"` and `"krylov"`
support terminal policies and retain the same audit replay metadata. Ordinary
retained path audits replay the policy and preserve the fixed
reference; they also compare its original terminal input/output to detect
changed callable state. Keep callbacks unchanged after solving. That replay is
not a mathematical identity check over the whole policy domain. MCP paths need
their own physical complementarity checks, as above.

## Diagnostics, failed runs and reporting

`TransitionResult.converged` requires the exact target and every requested
horizon check. A retained `solution` is the last available numerical result; on
failure it may be intermediate or failed. `path` is empty if no numerical result
was retained. A converged inner solution alone does not establish wrapper success.

| `reason` | Interpretation |
| --- | --- |
| `target_solved` | Exact target accepted; no horizon check was requested. |
| `horizon_accepted` | Exact target and a requested common-window horizon comparison passed. |
| `baseline_failed` | The specified baseline did not pass numerical acceptance. |
| `target_failed` | A direct target attempt with `continuation=False` failed. |
| `continuation_step_exhausted` | Further reductions would violate `min_step`. |
| `solve_budget_exhausted` | The shared solve-count budget ended before completing all requested work. |
| `horizon_solve_failed` | An extended target problem failed numerically. |
| `horizon_budget_exhausted` | The maximum horizon was reached without the requested agreement. |

`continuation_history` records each attempt's fraction, periods, parameters and
acceptance. Completed numerical attempts include residuals, iterations and
warnings; numerical exceptions retain an error record instead. Input contract
errors are raised, rather than relabeled as successful partial results.
`solves`, `scaled_residual_norm` and `preconditioner_builds` summarize the retained
workflow; inspect the histories to distinguish a difficult equation solve from
an inadequate horizon.

For ordinary solutions, `solution.solver_diagnostics` includes derivative/backend
details and the wrapper's scales and original residual norm. Krylov's
`linear_history` reports the returned direction's `true_linear_residual`,
requested linear tolerance, iteration status, step and backtracking information.
A positive inner iteration status is not itself nonlinear convergence; inspect
the final result and residuals. `extended_path.solver_diagnostics` additionally
records rolling solves and preconditioner builds/reuses.

Tables and plots expose the retained candidate; they do not override its status.
`to_frame()` and `to_dataframe()` return copies. Markdown, LaTeX and Typst use
the shared `puremacro.reports` formatting/escaping layer. `plot(ax=..., show=False)`
respects caller axes and does not display a window by default.
If no numerical solution was retained, plotting raises a clear error.

```python
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

assert transition.converged, transition.summary()
continuation_table = pd.DataFrame(transition.continuation_history)
horizon_table = pd.DataFrame(transition.horizon_history)
assert continuation_table.iloc[-1]["fraction"] == 1.0
assert bool(horizon_table.iloc[-1]["accepted"])
exported = transition.to_frame()
pd.testing.assert_frame_equal(exported, transition.to_dataframe())
print(transition.to_markdown(head=3, digits=6))
assert r"\begin{tabular}" in transition.to_latex(head=3)
assert "#table(" in transition.to_typst(head=3)
figure, axes = plt.subplots()
assert transition.plot(ax=axes, show=False) is figure
plt.close(figure)

equation_audit = audit_result(transition.solution, checks=["equations"])
assert np.max(np.abs(equation_audit.maps["path_equations"])) < 1e-8
```

The ordinary equation audit checks the saved finite experiment independently
of the convergence flag. Its small residual does not certify uniqueness,
infinite-horizon accuracy or stochastic optimality. The numerical tests in
`tests/test_dsge_transition_workflows.py` cover exact linear/stable-root paths,
derivative agreement, continuation budgets and exact endpoints, domains/scales,
information dates, MCP signs, terminal audits, preconditioner reuse and result
exports. They establish those contracts on the tested cases, not universal
convergence for arbitrary models.
