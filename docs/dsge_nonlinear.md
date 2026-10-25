> 🇬🇧 English · 🇪🇸 [Español](es/dsge_nonlinear.md)

# One nonlinear specification, explicit solution methods

`read_mod` compiles a `.mod` file into a `NonlinearModel` without computing a
steady state or solving a local rational-expectations approximation. You choose
the numerical problem afterward. The same specification can supply nonlinear
residuals to perfect foresight, perturbation and an explicitly constructed
global equilibrium problem.

For method selection and the course sequence, start with the
[DSGE workflow guide](dsge_workflows.md).

| Task | Entry point | What is computed |
|---|---|---|
| Inspect syntax and command compatibility | `inspect_mod(source, ordered=True)` | A report; no numerical solution |
| Retain equations and calibration | `read_mod(source, params=...)` | A compiled `NonlinearModel`; declarations may be evaluated |
| Find a static equilibrium | `model.steady(...)` | A checked dictionary of endogenous levels |
| Obtain a local policy rule | `model.solve(order=1)`, `2` or `3` | The existing perturbation solution |
| Solve a specified deterministic path | `model.solve_perfect_foresight(...)` | Endogenous values at dates 1 through T |
| Run supported commands in file order | `model.execute_mod_commands()` | A command history and retained result snapshots |
| Keep the established load-and-solve workflow | `load_mod(source, order=...)` | A local solution immediately |

A finite boundary problem can be solvable even when the local policy-rule
solver rejects the model. Conversely, successful compilation does not establish
existence, uniqueness, determinacy or convergence. Perfect foresight still
requires a well-posed system, appropriate boundaries and a feasible numerical
path.

## Start with an exact deterministic path

```python
import numpy as np
from puremacro.dsge import read_mod, NonlinearModel

model = read_mod("""
var x;
model; [name='accumulation'] x=0.5*x(-1); end;
initval; x=1; end;
""")
assert model.shocks == ()
assert model.equation_names == ("accumulation",)
steady = model.steady()
path = model.solve_perfect_foresight(y_init=[1], y_end=[0], periods=8)
assert path.converged
np.testing.assert_allclose(path.path.x, 0.5 ** np.arange(1, 9), atol=1e-10)
```

This is depreciation without investment: the date-zero stock is 1 and each
period retains half of the previous stock. No `varexo` declaration is needed.
`path.path` contains eight rows, excluding the supplied date-zero and date-nine
boundaries. In this backward-looking example the terminal value does not enter
any equation; a forward-looking equation would use it.

`steady()` returns the verified static equilibrium, here `{"x": 0.0}`. It does
not change `model.guess` or `model.steady_state`. The latter, when present, is a
supplied analytical declaration, not a numerical convergence certificate.
Pass the returned values explicitly when they are needed as boundaries or as
the reference for a subsequent calculation.

A purely forward-looking finite boundary problem also needs no prior local
policy rule. Here the specified terminal price is 1 at date 5, and backward
substitution determines the four reported prices. The initial value does not
enter this equation:

```python
forward_only = read_mod("var p; model; p=2*p(+1); end;")
finite_path = forward_only.solve_perfect_foresight(
    y_init=[0], y_end=[1], periods=4,
)
assert finite_path.converged
np.testing.assert_allclose(finite_path.path.p, [16, 8, 4, 2], atol=1e-10)
```

This example establishes a finite path under the supplied terminal value; it
does not select an infinite-horizon economic solution.

## Migrate from `load_mod` deliberately

Existing `load_mod` calls keep their behavior. Use `read_mod` when the next task
needs the original equations without requiring a local solution first. Obtain
a local solution explicitly when you need decision rules, IRFs or moments:

```python
from puremacro.dsge import load_mod

local_source = """
var x; varexo e; parameters rho; rho=0.7;
model; x=rho*x(-1)+e; end;
initval; x=0; end;
shocks; var e; stderr 0.1; end;
"""
legacy = load_mod(local_source, order=1)
specification = read_mod(local_source)
local = specification.solve(order=1)
np.testing.assert_allclose(local.decision_rules().ghx,
                           legacy.decision_rules().ghx, atol=1e-12)
assert specification.steady_state is None
```

There are three migration details to preserve:

- `read_mod(...).solve()` defaults to **order 1**. It does not select its order
  from a `stoch_simul` command. Set `order=2` or `3` explicitly when that is the
  intended approximation. `load_mod(order=None)` retains its existing
  command-derived order behavior.
- Ordered execution is the default on `NonlinearModel`. On a model returned by
  `load_mod`, use `execute_mod_commands(ordered=True)` explicitly; loading with
  `command_mode="ordered"` performs compatibility preflight before its local
  solve. For a deterministic-only task, `read_mod` avoids that extra solve.
- `read_mod(params=...)` re-evaluates analytical steady-state, initial-value and
  shock-covariance expressions at the merged calibration. Ordered dated-shock
  expressions are also evaluated at that calibration. Legacy `load_mod` keeps
  its existing shock-covariance behavior; use its `shock_cov=` argument when
  changing that covariance explicitly.

Calibrate every parameter and override only declared names. An override does
not re-run dependent top-level assignments: if the source contains
`r=0.04; beta=1/(1+r);`, overriding `r` alone leaves the calibrated `beta`
unchanged. Override both, or use a parameter-dependent `#` model-local
expression where an identity should remain symbolic. For another calibration,
create another specification with `read_mod(source, params=...)`; this also
keeps the experiment's provenance clear.

A `Path` or a string without semicolons/newlines denotes a file. Raw source
containing relative macro includes can use `base_dir=`. Reading retains the
expanded source snapshot: changing an included file afterward does not change
an already read command program. Read it again to use the new contents.

## Array timing and the callback contract

For n compiled endogenous variables and m shocks, the residual interface is

$$F(y_{t+1},y_t,y_{t-1},e_t;\theta,\bar y)=0.$$

| Argument or result | Shape | Meaning |
|---|---|---|
| `lead` | `(n,)` | All endogenous values at t+1 |
| `current` | `(n,)` | All endogenous values at t |
| `lag` | `(n,)` | All endogenous values at t−1 |
| `shocks` | `(m,)` | Current exogenous values/innovations, in declared order |
| `steady_state` | `(n,)` when supplied | A fixed endogenous reference for `STEADY_STATE()` |
| Returned residual | `(n,)` | One residual per compiled equation, in equation order |
| PF `shocks` path | `(T, m)` | Exogenous values at dates 1 through T |
| PF `initial_path` | `(T, n)` | Newton's starting guess, separate from the boundaries |

`model.residuals(...)` accepts these one-dimensional arrays or mappings with
all required names. It is a **single-date** evaluator, not a batch evaluator.
When there are no shocks, pass `[]` or an array of shape `(0,)`. Residual arrays
preserve complex inputs, which allows suitable smooth callbacks to participate
in complex-step differentiation. Avoid casting callback inputs to `float` or
using absolute-value/branching operations when requesting that derivative
method.

The model uses the timing written in the equations. For example,
`y=exp(a)*k(-1)^alpha; k=(1-delta)*k(-1)+i;` makes `k` the end-of-period stock;
production at t uses the inherited stock. Pass the complete endogenous vector
at each date, not just the state variables selected by a perturbation solver.

`from_equations` wraps the same interface. Its callback receives four positional
arrays and may explicitly declare `params` and/or `steady_state`; those optional
arguments are passed only when named in the callback signature. `params` is a
mapping. `steady_state` is an array or `None`. The callback supplies equations
already arranged as residuals equal to zero:

```python
def accumulation(lead, current, lag, shocks, *, params):
    return current - params["rho"] * lag

callback = NonlinearModel.from_equations(
    accumulation, variables=["x"], params={"rho": 0.5},
    equation_names=["accumulation"], guess=[0],
)
other_path = callback.solve_perfect_foresight(
    y_init=[1], y_end=[0], periods=8, linear_solver="krylov",
)
np.testing.assert_allclose(other_path.path, path.path, atol=1e-10)
```

`variables` must be nonempty, and endogenous and shock names must be distinct.
Optional `equation_names` has one distinct name per variable. Optional `timing`
records your declared timing; it does not shift or rewrite a Python callback.
`guess`, `steady_state` and `shock_cov` provide explicit initial-value,
reference and covariance metadata. A callback model has no `.mod` source to
execute; use its numerical methods directly.

A mixed backward/forward system shows why all three endogenous vectors matter.
Let `z` follow an AR(1) and let `q` be a deterministic discounted value of `z`,
with a zero terminal value. The following callback and `.mod` specification
produce the same finite-horizon path:

```python
def valuation_equations(lead, current, lag, shocks, *, params):
    z, q = current
    return np.array([
        z - params["rho"] * lag[0] - shocks[0],
        q - z - params["beta"] * lead[1],
    ])

valuation = NonlinearModel.from_equations(
    valuation_equations, variables=["z", "q"], shocks=["e"],
    params={"rho": 0.5, "beta": 0.8},
    equation_names=["state", "valuation"],
    timing={"z": [-1, 0], "q": [0, 1]}, guess=[0, 0],
)
valuation_mod = read_mod("""
var z q; varexo e; parameters rho beta; rho=0.5; beta=0.8;
model;
[name='state'] z=rho*z(-1)+e;
[name='valuation'] q=z+beta*q(+1);
end;
""")
innovations = np.zeros((8, 1))
innovations[0, 0] = 1
arguments = dict(y_init=[0, 0], y_end=[0, 0], periods=8, shocks=innovations)
callback_path = valuation.solve_perfect_foresight(**arguments)
compiled_path = valuation_mod.solve_perfect_foresight(**arguments)
assert callback_path.converged and compiled_path.converged
np.testing.assert_allclose(callback_path.path, compiled_path.path, atol=1e-10)
expected_z = 0.5 ** np.arange(8)
expected_q = np.zeros(9)
for t in range(7, -1, -1):
    expected_q[t] = expected_z[t] + 0.8 * expected_q[t + 1]
np.testing.assert_allclose(compiled_path.path.z, expected_z, atol=1e-10)
np.testing.assert_allclose(compiled_path.path.q, expected_q[:-1], atol=1e-10)
```

### Names, references and augmented equations

| Attribute | Interpretation |
|---|---|
| `variables`, `shocks` | The authoritative array order used by the compiled specification |
| `parameter_values` | The current fixed calibration |
| `equation_names` | Equation tags such as `[name='Euler']`, or generated names |
| `timing` | Endogenous dates appearing in the compiled equations |
| `equation_sources` | Locations of original source equations |
| `source_hashes` | Hashes identifying the source snapshot and included files |

The parser can augment higher leads/lags and dated exogenous variables with
auxiliary equations. Consequently, `model.variables` can be longer than the
source `var` list. Inspect it before constructing arrays; auxiliary rows may
not have their own original source-equation span. For nonconstant historical
paths, provide a consistently augmented state through Python. The ordered
boundary subset accepts only unindexed values or date zero, not a complete
multi-date `histval` calendar.

For a dynamic residual audit, supply the fixed `steady_state` reference
explicitly when the equations contain `STEADY_STATE()`. Calling `residuals`
without it uses the current vector for that operator, which is appropriate for
a static residual calculation. The attribute `model.steady_state` is not
implicitly substituted by `residuals`. In a perfect-foresight solve, `y_end`
provides both the terminal boundary and the fixed reference:

```python
reference_model = read_mod("""
var p;
model; p=0.5*p(+1)+0.5*STEADY_STATE(p); end;
""")
fixed = reference_model.residuals([3], [2], [0], [], steady_state=[1])
static_reference = reference_model.residuals([3], [2], [0], [])
np.testing.assert_allclose(fixed, [0])
np.testing.assert_allclose(static_reference, [-0.5])
```

## Steady states and positive domains

`model.steady(exo_values=..., guess=..., tol=...)` solves the static original
equations and checks the resulting residual against `tol`. `exo_values` is a
mapping by shock name or a shock-order vector; it specifies the constant
exogenous levels at this equilibrium. A `guess` is a starting point, not a
restriction on the solution. Without an explicit guess the method uses an
analytical declaration, then `initval`, when available.

For log, reciprocal or fractional-power equations, provide a feasible guess.
In an ordered program a `steady_state_model` declaration supplies the
analytical seed even when `initval` is absent. The equations are still solved
and verified at the active exogenous values. A changed terminal exogenous
level therefore cannot be replaced silently by the initial equilibrium.

This positive level model has the exact static relation
`x=center*exp(e/0.8)`. The overridden center is 3, and the permanent terminal
shock raises the steady state to 12:

```python
positive_program = read_mod("""
var x; varexo e; parameters center; center=2;
model; log(x)=0.2*log(x(-1))+0.8*log(center)+e; end;
steady_state_model; x=center; end;
steady;
endval; e=0.8*log(4); end;
steady;
perfect_foresight_setup(periods=3);
perfect_foresight_solver(tolf=1e-10);
""", params={"center": 3})
positive_results = positive_program.execute_mod_commands()
static_results = [entry.result for entry in positive_results.invocations
                  if entry.command == "steady"]
np.testing.assert_allclose([value.x for value in static_results], [3, 12])
expected = 3 * 4 ** (1 - 0.2 ** np.arange(1, 4))
np.testing.assert_allclose(positive_results.perfect_foresight.path.x,
                           expected, atol=1e-8)
```

`perfect_foresight_setup(endval_steady, periods=...)` can request the terminal
static solve as part of setup. This native option prepares a checked endpoint;
it does not promise Dynare's full homotopy or terminal-steady-state algorithm.
For nonlinear continuation and independent horizon checks, use the
[transition workflow](dsge_transitions.md).

## Execute deterministic commands in source order

The commands use concepts described in the Dynare manual's
[initial and terminal conditions](https://www.dynare.org/manual/the-model-file.html#initial-and-terminal-conditions)
and [deterministic simulation](https://www.dynare.org/manual/the-model-file.html#deterministic-simulation)
sections. The tables below define the native puremacro subset. Dynare provides
additional command forms and algorithms beyond this subset.

The first experiment below is transitory. The second changes the constant
exogenous level and computes the associated terminal steady state. A shock
redeclared after the first setup deliberately demonstrates its snapshot
semantics:

```python
program = read_mod("""
var x; varexo e; parameters rho; rho=0.5;
model; x=rho*x(-1)+e; end;
initval; x=0; e=0; end;
steady;
shocks; var e; periods 1; values 1; end;
perfect_foresight_setup(periods=4);
shocks; var e; periods 1; values 9; end;
perfect_foresight_solver(tolf=1e-10);
endval; x=8; e=1; end;
steady;
shocks(overwrite); end;
perfect_foresight_setup(periods=8);
perfect_foresight_solver;
""")
results = program.execute_mod_commands()
solves = [entry for entry in results.invocations
          if entry.command == "perfect_foresight_solver"]
np.testing.assert_allclose(solves[0].result.path.x, [1, .5, .25, .125], atol=1e-10)
np.testing.assert_allclose(solves[1].result.path.x,
                           2 * (1 - .5 ** np.arange(1, 9)), atol=1e-10)
print(results.to_frame())
```

The first solved path still begins at 1: its setup was prepared before the
value 9 was declared. The second setup uses the updated terminal equilibrium
and the cleared dated-shock declarations. `endval; x=8; e=1; end; steady;`
produces terminal `x=2`, because `steady` solves `x=0.5*x+1`. Omitting that
`steady` would retain the literal terminal value 8.

### Boundary and shock state

| Statement | Effect when reached |
|---|---|
| `initval` | Starts a new initial/terminal experiment; omitted endogenous and exogenous values become zero; clears historical and terminal assignments |
| `steady` before `endval` | Replaces initial endogenous values with their static equilibrium at the current initial exogenous levels |
| `histval` | Supplies the historical endogenous boundary (`x`, `x(0)`, or `x(-k)` mapped onto auxiliary lags), overriding initial values there |
| `endval` | Updates the terminal values (`x`, `x(0)`, or `x(+k)` mapped onto auxiliary leads); omissions inherit the existing terminal context or initial values |
| `steady` after `endval` | Replaces terminal endogenous values with the equilibrium at terminal exogenous levels |
| `shocks` | Accumulates dated assignments and covariance declarations; a redeclared date replaces only that date |
| `shocks(overwrite)` | Clears previous dated shocks and covariance declarations before reading the new block |
| `perfect_foresight_setup` | Copies boundaries, exogenous path and initial iterate for the selected horizon |
| `perfect_foresight_solver` | Solves that prepared experiment; a repeated successful solve starts from its previous path |
| `simul` | Prepares and solves the current experiment in one command |

The runner begins with zero initial values. An `endval` by itself does not set
the historical boundary in this subset: use an explicit `initval`, `histval`, or initial
`steady`. `initval` does not clear shock declarations; use `shocks(overwrite)`
when starting an experiment that should discard them. `histval` and `endval`
can be combined within one ordered experiment (`histval` sets `y_init` while
`endval` sets `y_end`).

A setup contains `y_init` at date 0 and `y_end` at T+1. Its initial Newton path
repeats the terminal endogenous vector at dates 1 through T. Its exogenous path
uses the terminal exogenous levels, or initial levels when there is no
`endval`, with dated assignments replacing those levels at the specified
dates. These are exogenous **levels/innovations as written in the equations**;
there is no automatic conversion to percentage changes.

Dated values can use finite expressions of the preceding calibrated
parameters. Dates are positive integers within the setup horizon. A single
value may fill all dates in a declared range; otherwise supply one value for
every expanded date. For portability, avoid combining multiple ranges with
range-specific values; write separate shock stanzas instead.

A `stderr` or covariance declaration controls stochastic variance; it does not
create a deterministic realization. Conversely, a block containing only dated
shocks leaves unspecified stochastic variances at zero and may produce a
parser warning to that effect. Keep both kinds of declarations explicit in
programs that also use `stoch_simul`. The mixed ordered runner requires zero
exogenous reference values for stochastic commands; recenter the model before
running a stochastic experiment around a nonzero exogenous baseline.

## Supported options and compatibility boundaries

The following defaults belong to **puremacro's ordered runner**. Options must
be supported literals; arbitrary expressions in command options are not
evaluated by compatibility inspection.

| Command | Accepted options | Initial defaults |
|---|---|---|
| `steady` in a deterministic program | `tol` or `tolf` | Residual tolerance `1e-10` |
| `perfect_foresight_setup` | `periods`, `endval_steady` | `100`, false |
| `perfect_foresight_solver` | `tol` or `tolf`; `maxit` or `max_iter`; matching `periods`; `stack_solve_algo`; `linear_solver`; `mcp` | `1e-8`, `50`, `100`, `0`, `'direct'`, false |
| `simul` | Setup and solver options together | Same PF defaults |

PF option values persist when omitted later in the same program. A later
`tol=` replaces an earlier `tolf=` value and vice versa; likewise for the
iteration-limit aliases. Supplying both aliases in one command is rejected.
The initial `steady` tolerance is separate from the PF solver tolerance.
Changing a solver's `periods` without preparing that horizon first is rejected.
A new setup replaces the previous prepared path; a repeated solver on the same
setup reuses the last successful path.

| Feature | Native ordered support or required alternative |
|---|---|
| Sparse direct stacked Newton | `stack_solve_algo=0` or `linear_solver='direct'` |
| Newton–Krylov | `linear_solver='krylov'`; omit explicit `stack_solve_algo` |
| Complementarity tags | Native `mcp` with `'direct'` or `'krylov'`; simple variable/parameter bound expressions |
| Full Dynare algorithm IDs | Values other than `stack_solve_algo=0` are rejected; no numerical-equivalence claim |
| `tolx`, `lmmcp`, solver/homotopy option families | Outside the ordered subset; use supported native Python controls |
| Parameter changes between experiments | Top-level reassignments between commands update the active calibration, or create separate specifications with `params=` |
| Historical date calendars | Unindexed, `x(0)`, and `x(-k)` lag boundary assignments mapped to declared/auxiliary lags |
| `varexo_det`, trend/regime/estimation declarations | May be parsed or reported, but are not executed by this ordered route |
| Expectation-error, learned-shock and calendar commands | Not implemented by this ordered route |
| MATLAB/Octave scripts and external helper execution | Not executed; move the study logic to Python |
| Ramsey/OSR or heterogeneous-agent programs | Use the corresponding explicit APIs/entry points; they are outside this ordered subset |

MCP tags such as `[mcp='r > floor']` keep the original LHS-minus-RHS equation
orientation. At a lower bound the associated residual must have the
nonnegative sign required by the complementarity condition. The tag does not
infer or correct your economic inequality. Repeated bounds for one variable,
dynamic-bound expressions and unsupported tag syntax are rejected. Native
MCP is not a `lmmcp` alias, and supports both `'direct'` and `'krylov'` when
`[mcp=...]` tags are present in `model;`. You can also inspect or run ordered
`.mod` programs from the command line via `puremacro-dynare inspect study.mod --ordered`
and `puremacro-dynare execute study.mod [--outdir DIR] [--data CSV]`.

The compiled specification uses the existing parser and supported bounded
macro preprocessing. It is not a general MATLAB interpreter. Boundary
assignment expressions must be evaluable in the available calibration/value
context; expressions that depend on a previous numerical solution may fail
already during reading. For such studies, solve in Python and pass the
resulting arrays explicitly. Treat the compatibility report for the exact
entry point as authoritative: recognition of a keyword alone does not imply
its execution is supported.

## Inspect, diagnose and recover

Compatibility inspection is separate from solving. `coverage="complete"`
means the syntax was collected completely; `exit_code==0` means the selected
execution route has no compatibility findings. Neither certifies a numerical
solution. The report identifies source locations, unsupported option names
and suggested remedies. `read_mod` can retain a specification whose commands
are unsupported; execution performs the compatibility check before starting
its numerical commands.

```python
from puremacro.dsge import inspect_mod, ModCompatibilityError, ModExecutionError

unsupported_source = """
var x; model; x=0.5*x(-1); end;
initval; x=1; end;
simul(periods=4, tolx=1e-8);
"""
report = inspect_mod(unsupported_source, ordered=True)
assert report.coverage == "complete" and report.exit_code != 0
assert any(issue.code == "UNSUPPORTED_OPTION" for issue in report.issues)
try:
    read_mod(unsupported_source).execute_mod_commands()
except ModCompatibilityError as error:
    assert any(issue.code == "UNSUPPORTED_OPTION" for issue in error.report.issues)
else:
    raise AssertionError("An unsupported option must not be ignored")
```

Once a numerical command starts, a failure raises `ModExecutionError` with
`.report` and `.results`. Completed entries and the failed entry remain
available. A failed PF result is retained when the solver returned one; a
failed static solve need not have a numerical result. This deliberately
impossible real-valued static equation demonstrates the distinction:

```python
impossible = read_mod("""
var x; model; x^2+1=0; end;
initval; x=0; end;
steady;
simul(periods=3);
""")
try:
    impossible.execute_mod_commands()
except ModExecutionError as error:
    partial = error.results
    assert partial.invocations[-1].status == "failed"
    assert partial.invocations[-1].command == "steady"
    assert "SteadyStateError" in partial.invocations[-1].error
else:
    raise AssertionError("An unverified steady state must not be accepted")
```

| Symptom | Check and recovery |
|---|---|
| A local QZ/determinacy failure blocks a deterministic study | Load with `read_mod`; independently check that the finite boundary system is well posed |
| `ModelError` during reading | Check declarations, parameter expressions, equation count, includes and parser coverage; unsupported source constructs cannot be fixed by increasing iterations |
| `ModCompatibilityError` | Read `.report`; replace the unsupported option/command or move that operation to its explicit Python API |
| A log/power/division produces nonfinite residuals | Check positivity and denominators in guesses, boundary values and all trial paths; use a feasible analytical seed |
| `SteadyStateError` | Inspect the static original equations and calibration; a supplied analytical declaration must satisfy them too |
| PF does not converge | Inspect the failed entry's result and `solver_diagnostics`; check timing, independent equations, boundary consistency and scaling before increasing `maxit` |
| A changed shock appears to have no effect | Check the setup snapshot; declarations after setup require another setup |
| Tiny residual but a questionable transition | Recompute original residuals, compare longer horizons and inspect economic feasibility; convergence alone does not verify the terminal approximation |

The direct Python PF method may return `converged=False`; the ordered runner
turns that outcome into `ModExecutionError` and retains the failed result.
Always check convergence when calling numerical methods directly. For domain
checks, continuation, terminal-policy boundaries and horizon studies, see
[transition controls](dsge_transitions.md) and
[trajectory audits](result_audits_extended.md).

## Retain and export the experiment

`DynareCommandResults.invocations` contains numerical command snapshots,
including `steady` and setup. Declaration blocks update the input state and do
not each create a numerical result. Each invocation records its command,
source location, explicit/effective options, input fingerprint, status,
complete `result`, and `selected_result`. A variable selection narrows
presentation; it does not solve a smaller equilibrium system.

`effective_options["input_state"]` retains calibration, initial/terminal
values, historical values, dated shocks and covariance. Solver entries also
retain the prepared inputs. Each `PerfectForesightSetup` exposes `periods`,
`variable_names`, `shock_names`, `y_init`, `y_end`, `exogenous_path`,
`initial_path` and `parameters`. Its arrays have the dimensions listed above.

Snapshots are copied between invocations; later source declarations and later
solves do not overwrite earlier results. The convenience alias
`results.perfect_foresight` is a separate copy of the last successful PF result.
Arrays and DataFrames inside a snapshot remain ordinary mutable objects: make
your own copy before editing a retained experiment.

```python
setups = [entry.result for entry in results.invocations
          if entry.command == "perfect_foresight_setup"]
first_setup = setups[0]
np.testing.assert_allclose(first_setup.to_frame().e, [1, 0, 0, 0])
assert first_setup.to_dataframe().equals(first_setup.to_frame())
print(first_setup.summary())
print(results.to_frame())
markdown_shocks = first_setup.to_markdown()
latex_shocks = first_setup.to_latex()
typst_shocks = first_setup.to_typst()
assert "e" in markdown_shocks and "tabular" in latex_shocks and "#table" in typst_shocks
```

| Result | Tables and reports |
|---|---|
| `PerfectForesightSetup` | `summary()` describes dimensions; `to_frame()`/`to_dataframe()` return the dated shock path; Markdown/LaTeX/Typst exporters render that table |
| `DynareCommandResults` | `summary()` and `to_frame()` describe execution history; `to_markdown()`, `to_latex()`, `to_typst()` export it |
| `PerfectForesightResult` | `path`/`to_frame()` contain endogenous trajectories; `summary()` includes convergence evidence; table exporters and `plot()` support presentation |
| `MCPResult` | Adds binding periods and complementarity diagnostics to its path and convergence information |

`terminal_error` measures the distance between the last simulated endogenous
row and the supplied terminal vector. It is not a transversality test or an
estimate of every truncation error. Compare paths at economically relevant
early dates across horizons, alongside original-equation residuals.

## MAV T02_A: one specification for a permanent productivity change

The course file uses log productivity `a_t=rho_a*a_(t-1)+e_a`. Keeping
productivity at 1.02 requires a first innovation `log(1.02)` and subsequent
innovations `(1-rho_a)*log(1.02)`. A single innovation to an AR(1) with
`rho_a<1` instead decays. The same source equations can be retained for local
and deterministic studies.

Run the following block from the matching repository root; the course model
file is a source-checkout dependency rather than an installed-package data
file:

```python
# requires: matching source checkout with curso/notebooks/
from pathlib import Path

source = Path("curso/notebooks/modelos/neoclasico_prevision_perfecta.mod").read_text()
experiment = read_mod(source + """
endval; e_a=(1-rho_a)*log(1.02); end;
steady;
shocks; var e_a; periods 1; values log(1.02); end;
perfect_foresight_setup(periods=80);
perfect_foresight_solver(tolf=1e-10);
perfect_foresight_setup(periods=160);
perfect_foresight_solver;
""")
study = experiment.execute_mod_commands()
transitions = [entry.result for entry in study.invocations
               if entry.command == "perfect_foresight_solver"]
assert all(result.converged for result in transitions)
np.testing.assert_allclose(transitions[0].path.a, np.log(1.02), atol=1e-10)
early_difference = np.max(np.abs(
    transitions[0].path.iloc[:20].to_numpy() -
    transitions[1].path.iloc[:20].to_numpy()
))
print(early_difference)
```

The original course file's first-order `stoch_simul` executes before the
appended transition commands. The two setup/solver pairs retain their separate
horizons and endpoints. Inspect `early_difference` and the residual evidence;
a small stacked residual alone does not certify an infinite-horizon solution.

For a local/global comparison, build an `EquilibriumProblem.from_model` with
explicit state, control, transition and expectation mappings; see
[global equilibrium methods](dsge_global.md). A `.mod` file does not by itself
identify that recursive formulation or a planner's objective. For better
finite-path initialization, continuation and terminal audits, continue with
[nonlinear transitions](dsge_transitions.md). These routes reuse the equations
while making their distinct information sets and approximation assumptions
explicit.
