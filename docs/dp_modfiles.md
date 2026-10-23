> 🇬🇧 English · 🇪🇸 [Español](es/dp_modfiles.md)

# Dynamic programs in a `.mod` file

Write the household or firm's objective and constraints as equations, save a
text file, and solve it with `puremacro.dp.solve_mod`. The reader compiles the
declarations to the existing [`dp.Model`](dp.md), so file and Python models use
the same VFI, EGM and HJB solvers and return the same `DPSolution` objects.

This is **Puremacro's DP language with Dynare-style syntax**. These files are
not DSGE equilibrium systems and are not executable by Dynare itself. Use
`puremacro.dsge.load_mod` for ordinary Dynare DSGE files.

## A complete VFI model

Save this as `growth.mod`:

```text
parameters beta alpha;
beta = 0.95;
alpha = 0.36;
state k = geomspace(0.01, 2.0, 300);

model(name='Growth by value function iteration');
  c = k^alpha - k(+1);
  reward = log(c);
end;
constraints;
  c > 0;
end;
solve(method=vfi, tol=1e-9, distribution=false);
```

```python
from puremacro import dp

sol = dp.solve_mod("growth.mod")
print(sol.summary())
consumption = sol.policy("c")
next_capital = sol.policy("k(+1)")
```

Here `k` is current capital and `k(+1)` is next period's capital, chosen on the
declared grid by VFI. The compiler supplies the Bellman problem; there is no
need to write an Euler equation or a Python return-function closure. In this
log-utility/full-depreciation model, the continuous exact policy is
`k_next = alpha*beta*k**alpha`, which provides an independent accuracy check.
The finite grid introduces approximation error.

The equivalent command-line calls are:

```bash
puremacro-dp growth.mod --check
puremacro-dp growth.mod
python -m puremacro.dp growth.mod --param beta=0.97
```

`--check` validates and compiles declarations but performs no numerical solve.
It does not certify convergence, grid accuracy or economic assumptions.

## Income risk and saving

```text
parameters beta gamma r w;
beta = 0.96; gamma = 2; r = 0.02; w = 1;
state a = 40*linspace(0, 1, 120)^2;
exogenous z = rouwenhorst(rho=0.9, sigma=0.15, n=5);

model;
  c = (1+r)*a + w*exp(z) - a(+1);
  reward = crra(c, gamma);
end;
constraints; c > 0; end;
aggregates; A = a; C = c; end;
solve(method=egm, tol=1e-9);
```

EGM uses continuous saving policies between grid nodes. Its supported budget
and preference restrictions are unchanged; choosing another method does not
change the declared economic problem. Use `method=vfi` for the discrete
grid-search solution. Multiple `state` declarations and one `discrete` choice
are supported by the underlying VFI compiler.

For one-asset EGM, the solver verifies the affine budget from the equations,
including local aliases, and checks the next-asset coefficient separately from
income scale. Nonlinear terms or extra constraints cannot be accepted just
because they vanish at grid nodes. Custom functions of exogenous income remain
supported; use VFI for general asset-dependent schedules or constraints. See
the [one-asset EGM restrictions](dp.md).

```python
model = dp.load_mod("household.mod")  # validates, never solves
egm = dp.solve_mod("household.mod")   # honors the file's solve options
vfi = dp.solve_mod("household.mod", method="vfi", tol=1e-10)
alternative = dp.solve_mod("household.mod", params={"r": 0.01})
```

Python solve options override file options. Calibration overrides apply before
dependent calibrations and grids are evaluated. Names must be declared in the
file. `model.solve(...)` uses the ordinary Python API defaults, not the optional
file command; `solve_mod` applies that command explicitly.

## Declaration reference

| Declaration | Meaning |
|---|---|
| `parameters beta gamma; beta=.96; gamma=2;` | Declare and calibrate once, in dependency order. Arrays are allowed for age-varying parameters. |
| `state a = linspace(0, 40, 120);` | Endogenous state and increasing finite grid. Its next-period value is a choice. |
| `exogenous z = ar1(rho=.9, sigma=.2, n=5);` | Gaussian AR(1), Tauchen by default; `method=rouwenhorst` selects that discretization. |
| `exogenous z = markov([.5, 1.5], [.9,.1; .2,.8]);` | Explicit row-stochastic Markov chain. Nested lists also work. |
| `discrete h = [0, 1];` | One contemporaneous discrete choice; include its reward/cost in the model. |
| `taste_shocks = sigma_h;` | EV1 taste shocks for the existing discrete-choice EGM solver. |
| `model; c = ...; reward = ...; end;` | Named local expressions and exactly one objective. `# c = ...` is also accepted. |
| `constraints; c > 0; end;` | Feasibility conditions, using the DP expression language. |
| `aggregates; A = a; C = c; end;` | Integrals under the solved distribution; timing follows the declared expression. |
| `price r = [-.01, .04]; clear = A - K_demand;` | One endogenous price bracket and scalar stationary market-clearing residual. Define demand as an expression using declared parameters/locals. |
| `discount patience;` | Use a calibrated parameter named `patience` instead of `beta`. |
| `horizon = 30; survival = full(30,.99);` | Finite horizon and optional survival vector. `age` starts at zero. |
| `terminal = ...; newborns = ...;` | Optional arrays with the shapes required by `Model.horizon`. |
| `control c; drift a = r*a + z - c; discount_rate rho;` | Continuous-time model, solved by HJB. Use `jump(values, rates)` for its shocks. |
| `solve(method=egm, tol=1e-9);` | Optional final command. Only `solve_mod` executes it. |

Calibration expressions allow arithmetic, `^`, arrays, previously calibrated
parameters, `linspace`, `geomspace`, `logspace`, `full`, `exp`, `log`, `sqrt`,
`abs`, and binary `min`/`max`. Matrix columns require commas; semicolons separate
rows. Grids and calibration arrays are limited to one million elements by this
reader, and AR(1) transition matrices are bounded by the same limit. The Python
API remains available for larger inputs. Multiple independent shock chains are
combined by the existing compiler; their product can still be large.

The final `solve` accepts `method`, `backend`, `tol`, `howard`, `max_iter`,
`distribution`, `xtol`, and `max_evals`, subject to the selected engine.
Explicit options that the engine would ignore are rejected: `backend` and
`howard` for EGM/HJB; `tol`, `max_iter` and `howard` for finite horizons; and
`xtol`/`max_evals` without equilibrium prices. Finite-horizon and equilibrium
solves require a distribution, so `distribution=false` is rejected there.
Python overrides are checked after merging with file options; omitted options
retain the ordinary solver defaults. See [DP solvers](dp.md).
Aggregate-risk/Krusell–Smith declarations
remain available through the Python API and are not part of this file grammar.

Comments use `//`, `%`, or `/* ... */`. Declarations end in semicolons. The
reader reports file names and statement line numbers and rejects unknown
commands, duplicate declarations, incomplete blocks and unsupported options.
It does not execute Python/MATLAB scripts, macros or includes. Custom economic
functions can be registered explicitly with
`dp.load_mod(source, functions={"tax": my_tax})`, under the existing
`Model.functions` contract; a file cannot define or import such functions.

## Examples and evidence

Three examples ship in the installed package:

```python
from importlib.resources import files
from puremacro import dp

source = files("puremacro.dp").joinpath("_examples/household.mod").read_text()
sol = dp.solve_mod(source)
```

The other names are `growth.mod` and `lifecycle.mod`. Tests compare file and
Python declarations under VFI, EGM and HJB, independently enumerate a two-state
Bellman maximum, check a final-age choice analytically, and compare growth
policies to the closed form. These are numerical/compiler checks; they do not
establish empirical validity. Use [result audits](result_audits.md) and explicit
grid refinement for the supported solved-result types.
