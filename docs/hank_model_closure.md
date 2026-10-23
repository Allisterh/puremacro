> 🇬🇧 English · 🇪🇸 [Español](es/hank_model_closure.md)

<!-- Language: en | Counterpart: es/hank_model_closure.md -->

# Editable HANK aggregate equations

The `.mod` HANK bridge keeps its existing built-in New Keynesian closure by
default. Set `closure="model"` to use the parsed aggregate equations instead.
This opt-in mode solves homogeneous affine equations in **level deviations**,
together with explicitly mapped household responses. `model(linear)` makes
that intent clear; the parser also recognizes affine `model` blocks.

```python
import numpy as np
from puremacro.dsge.hank import solve_hank_bridge

source = """
var Y C r pi i;
varexo monetary government;
parameters beta gamma r_ss phi_pi kappa;
beta=.985; gamma=1; r_ss=.01; phi_pi=1.5; kappa=.1;
hetagent_block; model=one_asset_hank; n_a=30; a_max=30; end;
model(linear);
Y=C+government;
pi=beta*pi(+1)+kappa*Y;
i=.3*i(-1)+phi_pi*pi+monetary;
r=i-pi(+1);
end;
"""
T = 12
result = solve_hank_bridge(
    source, horizon=T, closure="model",
    household_inputs={"r": "r", "Y": "Y"},
    household_outputs={"C": "C"},
    shock_paths={"monetary": -.002*.6**np.arange(T),
                 "government": .001*.4**np.arange(T)},
)
assert result.converged
assert result.diagnostics["aggregate_residual_max_abs"] < 1e-9
print(result.transition_paths.head())
```

Each mapping goes from a **household name** to a declared aggregate name.
Inputs may map to endogenous variables or exogenous shocks. Outputs must map
to distinct endogenous variables. There must be exactly one equation per
endogenous variable after adding one household-response equation per mapped
output. Unsupported names, duplicate targets, nonhomogeneous constants and
nonlinear expressions fail explicitly. Unresolved variables are not filled
with zero. For example, write `i=phi_pi*pi+monetary` for deviations, not
`i=r_ss+phi_pi*pi+monetary`.

`shock_paths` contains finite arrays of exactly `horizon` observations; omitted
declared shocks are zero. Without it, the selected declared `shock` receives
`magnitude*rho**arange(horizon)`. In this mode the shock name determines which
equation is affected. All endogenous and exogenous deviations before date 0
and after date `T-1` are zero. This is a finite-horizon boundary condition;
extend the horizon and compare paths when terminal effects matter.

The solver checks full matrix rank and independently evaluates the original
aggregate equations and household identities after solving. Diagnostics report
rank, dimension, condition number and both maximum residuals. Full rank means
**finite-horizon solvability**, not Blanchard–Kahn determinacy, uniqueness of an
infinite-horizon equilibrium, or household grid accuracy. `closure_tol` controls
the absolute residual acceptance threshold (default `1e-9`). No pseudoinverse
or regularization is silently substituted. This implementation assembles a
dense matrix of size `(variables*horizon)**2`; start with moderate horizons.

## Household units and timing

All inputs and outputs use level deviations, including rates expressed in the
same period units as their calibration. They are not percentages or log
deviations. Mappings rename coordinates without scaling them.

| Household | Available inputs | Available outputs | Convention |
|---|---|---|---|
| Built-in one asset | `r`, `Y` | `C` | `r` is the ex-ante rate realized next period; `Y` changes gross wages by `w_ss/C_ss`, retaining the household's balanced-budget interest tax. |
| Built-in two assets | `r_b`, `r_a`, `Y` | `C`, `D`, `A`, `B`, `CHI`, `z` | Ex-ante rates; nonfinancial income is closed using output less returns on inherited assets. `A` and `B` are saving choices. |
| Declarative `dp.Model` or `DPSolution` | Declared supported household inputs | Declared aggregate expressions and supported aliases | Raw household price timing and units; no additional fiscal equation or wage scaling. |

For a declarative household, `A="a"` means assets carried into the date, while
`S="a(+1)"` means saving chosen that date. Those meanings survive mapping to
aggregate variables. Specify wage, fiscal, asset-market and price equations
explicitly in the aggregate block. Supplying an already solved `DPSolution`
also fixes which household calibration generates the Jacobians.

The built-in defaults remain useful for comparison. Options that only make
sense with editable equations are rejected under `closure="builtin"`.

## Nonlinear household equilibrium

With the same mappings, `nonlinear=True` replaces the linear household identities
with exact backward-policy and forward-distribution transitions. The aggregate
equations still must be affine and homogeneous in deviations. This supports
built-in and declarative one- and two-asset households; it is not an arbitrary
nonlinear aggregate equation language.

```python
nonlinear = solve_hank_bridge(
    source, horizon=T, closure="model", nonlinear=True,
    household_inputs={"r": "r", "Y": "Y"},
    household_outputs={"C": "C"},
    shock_paths={"monetary": -.002*.6**np.arange(T)},
    closure_tol=1e-9, closure_max_iter=100,
)
assert nonlinear.converged
assert nonlinear.diagnostics["household_budget_max_abs"] < 1e-8
assert nonlinear.diagnostics["distribution_mass_max_abs"] < 1e-10
```

The linear solution initializes damped Broyden iterations. An infeasible initial
guess is retried at zero endogenous deviations; trial paths must reduce the
residual norm and preserve feasible household budgets and distribution mass.
Consumption-floor paths are rejected. Initial distributions and terminal
household continuation values are their calibrated steady-state values; the
ex-ante return convention includes the return realized at the terminal date.
Declarative household inputs retain their original realized-date convention.

Diagnostics report iteration count, residual history, final budget and mass
errors, and termination reason. Failed iterations warn and return the last
path with `converged=False`; the result is never replaced by a linear answer.
An exogenous input that makes even the initial household path infeasible raises
`ValueError`. Successful numerical convergence does not certify equilibrium
uniqueness, grid convergence, or an independent Euler audit. The private context
retains final aggregate household responses without triggering another solve.

## Two-asset household panels

A solved two-asset EGM household supports the regular panel interface:

```python
# sol is a two-asset DPSolution, with declared assets a and b.
panel = sol.simulate(
    n_agents=1000, periods=40, seed=2026, burn_in=10,
    record=["a", "b", "a(+1)", "b(+1)", "c"],
)
microdata = panel.to_microframe(t=20)
```

Both asset choices are interpolated bilinearly and remain continuous. The
recorded consumption and adjustment costs follow the model's budget at those
actual choices, rather than a separately interpolated consumption table.
Productivity follows the solved Markov chain. The same seed reproduces the
panel exactly, including burn-in; reversed asset declaration order is handled
explicitly. Fixed initial values may be off-grid, and out-of-grid values are
clipped with a warning.

Stationary initialization samples the solved joint grid distribution. Later
continuous policies need not preserve the discretized lottery distribution
exactly: compare simulated moments with `sol.mean(...)` and refine grids when
that approximation matters. These are stationary-policy panels, not panels
along a general-equilibrium transition.
