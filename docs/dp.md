> 🇬🇧 English · 🇪🇸 [Español](es/dp.md)

# Declarative dynamic programming (`puremacro.dp`)

`puremacro.dp` does for dynamic programs what a Dynare `.mod` file does for
DSGE models: you declare the model as equations and the library writes the
solver call. States, shocks, choices, the reward, constraints and an optional
market-clearing condition are strings in Dynare expression syntax, parsed by
the same parser as `.mod` files. `Model.solve()` compiles them to the
`puremacro.vfi` engines, so you never write a return-function closure, broadcast
grids by hand or pass positional parameter lists.

| `solve(method=...)` | engine | what it handles |
|---|---|---|
| `"vfi"` (default without a drift) | `VFIProblem`, `FiniteHorizonProblem` | any discrete-time model on grids; several states |
| `"egm"` | `vfi.solve_egm`, one EGM step per age | one asset with budget `c + a(+1) = R*a + y(z)`; life cycles; a discrete choice with or without taste shocks; Krusell-Smith |
| `"hjb"` (default with a drift) | `vfi.solve_hjb_achdou` | continuous time: one state whose drift is linear in one control |

With `prices(...)` and `clear(...)` the household solve sits inside a root-find
for the market-clearing price. With `aggregate_state(...)` it sits inside the
Krusell and Smith (1998) forecast-rule loop.

## An Aiyagari economy

```python
import numpy as np
from puremacro import dp

m = dp.Model("aiyagari")
m.parameters(beta=0.96, gamma=1.0, alpha=0.36, delta=0.08)
m.prices(r=(0.005, 0.0396))                       # bracket for the root-find
m.exogenous("z", dp.AR1(rho=0.9, sigma=0.2, n=5)) # Tauchen; method="rouwenhorst"
m.state("a", np.linspace(1e-4, 80.0, 150))
m.local("w = (1 - alpha)*(alpha/(r + delta))^(alpha/(1 - alpha))")
m.local("c = w*exp(z) + (1 + r)*a - a(+1)")
m.reward("crra(c, gamma)")
m.subject_to("c > 0")
m.aggregate(K="a", L="exp(z)")
m.clear("K - L*(alpha/(r + delta))^(1/(1 - alpha))")

sol = m.solve("egm")
print(sol.prices["r"], sol.aggregates["K"])       # about 0.0305 and 7.49
print(sol.summary())
```

`a` is today's value of a state and `a(+1)` its value next period. An
identifier that is not a state, a shock or the discrete choice resolves, in
order, to a local, a parameter, a price, an aggregate (only inside `clear`) and
`age` (only in finite-horizon models). `crra(c, gamma)` is
`(c^(1-gamma) - 1)/(1 - gamma)`, and `log(c)` at `gamma = 1`. Any other
felicity can be written out as an expression: EGM takes its marginal utility
from the symbolic derivative and inverts it numerically.

The same model solves by discrete VFI with `m.solve()` (r about 0.0303 on this
grid, in about 8 s against about 1 s for EGM). The two differ because VFI puts
`a(+1)` on grid nodes and EGM does not.

### Reading a solution

`DPSolution` holds the solution whatever the engine:

- `sol.V`, `sol.distribution`: arrays of shape (n_states, n_shocks), with a
  leading age axis in a life cycle;
- `sol.policy(expr)`: any model expression at the optimal choice, e.g.
  `sol.policy("c")` or `sol.policy("a(+1)")`;
- `sol.mean(expr)`: its population mean under the stationary distribution, one
  value per age in a life cycle;
- `sol.prices`, `sol.aggregates`, `sol.equilibrium`, `sol.grids`, `sol.P`;
- `sol.elapsed`: wall-clock solve time in seconds (`float`);
- `sol.to_microframe(columns=None, *, age=None)`: converts the stationary or
  life-cycle grid-node distribution (`sol.dist > 0`) into a `MicroFrame` with
  `SurveyDesign.weights_only("weight")` (`unit="node"`), attaching state, shock,
  policy, value, and `local` columns;
- `sol.raw`: the underlying engine's result.

```python
c = sol.policy("c")                 # (150, 5)
print(sol.mean("c"), sol.mean("a")) # mean consumption and assets
```

**When `max_iter` runs out.** The outcome depends on the engine. The one-asset
`"vfi"` and `"egm"` engines (discrete-choice EGM included) return no
`converged` field, so they raise `RuntimeError("... did not converge ...")` and
hand back no partial solution. The two-asset EGM solver and the `"hjb"` solver
emit a `RuntimeWarning` and return the last iterate with
`sol.raw.converged = False`; the Krusell-Smith loop does the same at
`max_outer` (`sol.equilibrium.converged`). Check the flag before using the
policy: `dp_sequence_jacobian` and `dp_nonlinear_transition` refuse an
unconverged two-asset household with `RuntimeError`.

## Life cycles

`horizon(T, survival=..., terminal=..., newborns=...)` makes the problem finite.
`age` (0 to T-1) can appear in any equation, so income profiles, retirement and
age-varying returns are written in place. A relational expression such as
`(age < T_ret)` is 1 when it holds and 0 otherwise.

```python
T = 60
lc = dp.Model("life cycle")
lc.parameters(beta=0.96, gamma=2.0, r=0.03, T_ret=45)
lc.exogenous("z", dp.AR1(rho=0.95, sigma=0.1, n=5))
lc.state("a", np.linspace(0.0, 30.0, 120))
lc.local("y = (age < T_ret)*exp(z + 0.04*age - 0.0008*age^2) + (age >= T_ret)*0.6")
lc.local("c = (1 + r)*a + y - a(+1)")
lc.reward("crra(c, gamma)")
lc.subject_to("c > 0")
lc.horizon(T, survival=np.full(T, 0.99))

life = lc.solve("egm")
assets = life.mean("a")             # mean assets by age; they peak at retirement
print(int(assets.argmax()))         # 45
```

Each cohort has mass one in `life.distribution`, and survival is not applied
there: it only discounts the future. `life.population_shares()` gives each age's
share of the population with survival applied, and the statistics that pool
ages use it (see [Panels and inequality](#panels-and-inequality)). Newborns
start at the lowest grid point with shocks drawn from their stationary
distribution unless `newborns=` says otherwise.

## Discrete choices and taste shocks

`discrete("h", values)` adds a discrete choice made together with `a(+1)`.
Under `method="egm"` each option is inverted by the Euler equation and an upper
envelope keeps the optimal segments (Iskhakov, Jørgensen, Rust and Schjerning
2017). The borrowing-limit and top-of-grid corners are admitted only where
their Kuhn-Tucker condition holds. `taste_shocks(scale)` adds type-I extreme
value shocks: values become log-sums, choices logit probabilities, and
`mean("h")` is a participation rate.

```python
w = dp.Model("work or not")
w.parameters(beta=0.95, gamma=2.0, r=0.03, chi=0.5, b=0.3, sig=0.05)
w.exogenous("z", dp.AR1(rho=0.9, sigma=0.2, n=5))
w.state("a", np.linspace(0.0, 60.0, 200))
w.discrete("h", [0.0, 1.0])
w.local("c = (1 + r)*a + h*exp(z) + (1 - h)*b - a(+1)")
w.reward("crra(c, gamma) - chi*h")
w.subject_to("c > 0")
w.taste_shocks("sig")
w.aggregate(H="h", A="a")

work = w.solve("egm")
print(work.aggregates["H"])         # participation rate, about 0.63
```

The reward and income may depend on the option; the return on assets may not.
Aggregates over a discrete choice converge slowly in the number of grid points
under either method, because the distribution's mass sits on grid nodes: check
them on a finer grid before reporting them.

## Continuous time

Declare a `control`, a `drift` for the state and a `discount_rate`, with
`dp.Jump` (a Poisson generator) for the shocks. The compiler checks that the
drift is linear in the control with slope -1 and derives u' and its inverse
from the reward. Prices and `clear` work as in discrete time, with aggregates
integrated against the Kolmogorov forward density.

```python
ct = dp.Model("continuous-time Aiyagari")
ct.parameters(rho=0.05, gamma=2.0, alpha=0.33, delta=0.05)
ct.prices(r=(0.005, 0.048))
ct.exogenous("e", dp.Jump([0.2, 1.0], [[-0.1, 0.1], [0.1, -0.1]]))
ct.state("a", np.linspace(0.0, 30.0, 100))
ct.local("w = (1 - alpha)*(alpha/(r + delta))^(alpha/(1 - alpha))")
ct.control("c")
ct.drift("a", "r*a + w*e - c")
ct.discount_rate("rho")
ct.reward("crra(c, gamma)")
ct.aggregate(K="a", L="e")
ct.clear("K - L*(alpha/(r + delta))^(1/(1 - alpha))")

hjb = ct.solve()                    # method="hjb" because the model has a drift
print(hjb.prices["r"])              # about 0.0200
```

Several `Jump` shocks combine as their product chain. `sol.distribution` is
the mass at each node (the density times the quadrature weights).

## Aggregate risk: Krusell and Smith

`aggregate_shock("Z", process)` and `aggregate_state("K", grid, mean="a")` turn
the model into the Krusell and Smith (1998) fixed point: households forecast `K`
with a log-linear rule, the economy is simulated, and the rule is re-estimated
until it stops moving. Prices and technology are model equations, so any firm
and any household reward work.

```python
ks = dp.Model("krusell-smith")
ks.parameters(beta=0.99, gamma=1.0, alpha=0.36, delta=0.025, L=0.9)
ks.exogenous("e", dp.Markov([0.5, 1.1], [[0.6, 0.4], [0.2, 0.8]]))
ks.aggregate_shock("Z", dp.Markov([0.99, 1.01], [[0.875, 0.125], [0.125, 0.875]]))
ks.state("a", np.linspace(0.0, 150.0, 100))
ks.aggregate_state("K", np.linspace(25.0, 45.0, 5), mean="a")
ks.local("r = alpha*Z*(K/L)^(alpha - 1) - delta", "w = (1 - alpha)*Z*(K/L)^alpha")
ks.local("c = (1 + r)*a + w*e - a(+1)")
ks.reward("crra(c, gamma)")
ks.subject_to("c > 0")

agg = ks.solve("egm", ks=dict(T=1000, burn_in=200, max_outer=60))
rule = agg.equilibrium                 # log K' = b0 + b1 log K, one rule per Z
print(rule.b1, rule.r_squared, rule.converged)   # b1 about 0.96; converges in 8 rounds
```

`L` is the mean of the idiosyncratic chain `e` (its stationary distribution is
(1/3, 2/3)). A long simulation (`T` in the thousands) gives tighter forecast
coefficients than this quick run.

**Use `method="egm"` here.** Discrete VFI puts next-period assets on grid
nodes, and on a coarse grid simulated capital then never moves: the forecast
regression returns `b1 = 0` with R² = 1 and passes the tolerance. Both this
loop and `vfi.krusell_smith` raise a `RuntimeWarning` in that case and when
they stop at `max_outer` without converging. Read `rule.converged` and
`rule.r_squared` before using the forecast rule.

## Panels and inequality

Inequality statistics are computed exactly from `sol.distribution`, with no
simulation: each grid node is a mass point with its stationary mass.
`sol.gini(expr)` is the standard weighted Gini, `1 - sum_i w_i (L_i + L_{i-1})`
on the Lorenz curve, the same function as `MicroFrame.gini`. Negative values
are allowed (the Gini can then exceed 1) and it is NaN when the total is not
positive. `sol.top_share(expr, p)`, `sol.quantile(expr, q)` and
`sol.lorenz(expr)` complete the picture.

```python
print(sol.gini("a"), sol.top_share("a", 0.10))   # about 0.62 and 0.41
print(sol.quantile("a", 0.5))                    # median wealth, about 3.76
curve = sol.lorenz("a")                          # columns pop_share, value_share
```

`sol.simulate(n_agents=10_000, periods=None, *, seed=0, init=None, burn_in=0,
record=None)` draws individual histories under the solved policy and returns a
`dp.Panel`. `periods=None` and `init=None` take per-horizon defaults: 200
periods and `"stationary"` in an infinite horizon, the `T` ages of the horizon and
`"newborn"` in a life cycle. In an infinite horizon each agent
starts in a (state, shock) cell drawn from `sol.distribution`, so the panel is
in the stationary distribution from the first period; `init={"a": 0.1}` starts
everyone at a given state instead, and `burn_in` drops the first periods. VFI
states stay on grid nodes, EGM interpolates the policy between them. A discrete
choice is simulated too: the argmax option without taste shocks, a logit draw
with them, recorded as the option's value. The panel records the states, the
shocks and the controls (`record=` takes any model expression, such as
`"a(+1)"` or `"w*exp(z)"`). The same seed gives a bit-identical panel.

```python
panel = sol.simulate(n_agents=10_000, periods=200, seed=1)
print(panel)                        # Panel(n_agents=10000, periods=200, columns=['a', 'z', 'c'])
gini_t = panel.gini("a", by="t")    # one cross-sectional Gini per period
print(gini_t.iloc[-1])              # about 0.62, the exact value up to sampling noise
df = panel.to_frame()               # long format: id, t, a, z, c
```

`panel.mean`, `panel.quantile`, `panel.gini` and `panel.top_share` pool every
period, or group with `by="t"` (or `by="age"` in a life cycle).
`panel.cross_section(t)` and `panel.path(i)` return one period or one agent,
and `panel.to_microframe()` hands the panel to the survey-data estimators.
`n_agents * periods` above 5e8 raises `ValueError` with the memory it would
need.

In a life cycle the panel runs over the ages of the horizon. Agents start as
newborns and, with declared survival, an agent alive at age j lives to j+1
with probability `survival[j]`. `panel.alive` records this. The dead keep their
rows but are left out of every statistic. The exact statistics pool ages with
`population_shares()` as weights, or take one cohort with `age=j`.

```python
shares = life.population_shares()   # survival 0.99: 0.022 at age 0, 0.012 at age 59
print(life.gini("a"))               # all ages pooled, about 0.73
print(life.gini("a", age=30))       # the 30-year-old cohort, about 0.50
lp = life.simulate(n_agents=5000, seed=0)
print(lp.alive[:, -1].mean())       # about 0.55 reach age 59
print(lp.mean("a", by="age")[45])   # about 7.7; life.mean("a")[45] is 7.68
```

Continuous-time (HJB) and Krusell–Smith solutions are simulated with the same
`sol.simulate(...)` call:

- For continuous-time (`method="hjb"`), `sol.simulate(..., dt=1.0)` advances the
  endogenous state by $\dot{a}\,\Delta t$ interpolated on the state grid, draws
  Markov jump transitions from the matrix exponential $\exp(Q\,\Delta t)$
  (`process_type="poisson"`), or steps continuous diffusions by Euler–Maruyama
  with boundary reflection (`process_type="diffusion"`).
- For Krusell–Smith (`"vfi+ks"`, `"egm+ks"`), `sol.simulate(...)` evolves the
  aggregate state path $(Z_t, K_t)$ under the converged forecast rule
  (`sol.forecast`) and interpolates the 4D policy `pol[i_Z, i_e, i_a, i_K]`
  across agents in every period. Exact stationary-distribution methods
  (`sol.mean`, `sol.gini`, `sol.quantile`, `sol.top_share`, `sol.to_microframe`)
  raise `ValueError` on Krusell–Smith solutions because there is no single
  stationary distribution; compute cross-sectional moments on the simulated
  `Panel` instead.

## User-registered functions

`m.functions(**callables)` registers vectorised Python/NumPy callables so model
expressions in `local`, `subject_to`, `aggregate`, `clear`, and `Panel`
evaluation can call custom schedules (such as piecewise tax-and-transfer rules)
across VFI, EGM, and HJB (`reward` also accepts registered functions under
`method="vfi"`). If a callable accepts an `xp` keyword argument, `dp` passes the
active array backend (`numpy`, `jax.numpy`, or `cupy`):

```text
m.functions(tax=lambda y, xp=np: 0.15 * y + 0.10 * xp.maximum(y - 1.0, 0.0))
m.local("y = w * exp(z)", "c = (1 + r)*a + y - tax(y) - a(+1)")
```

## Sequence-space HANK bridge

An infinite-horizon single-state `dp.Model` supported by EGM, without a discrete
choice (or its solved `DPSolution`), connects
directly to the Auclert, Bardóczy, Rognlie & Straub (2021) sequence-space engine
via `dp_sequence_jacobian(model_or_sol, inputs, outputs=None, T=300, dx=1e-4)`
and `dp_nonlinear_transition(model_or_sol, input_paths, outputs=None)` in
`puremacro.models.hank_sequence_space`, or by passing `household_model=m` to
`HANKModel`, `load_hank_mod`, and `solve_hank_bridge` in `puremacro.dsge.hank`.
See [Non-Linear HANK Transitions & Effective Lower Bound](hank_nonlinear.md) for
details.

### Two-asset households (5.4)

Two-state `method="egm"` supports a stationary CRRA household with liquid bonds
`b`, illiquid assets `a`, and convex portfolio adjustment costs. The compiler
checks the declared budget, utility and feasible set against this specialized
problem; it rejects unsupported equations instead of substituting a different
household problem. For example:

```python
import numpy as np
from puremacro import dp

m = dp.Model("two_asset")
m.state("a", 40 * np.linspace(0, 1, 20)**2)
m.state("b", 10 * np.linspace(0, 1, 20)**1.5)
m.exogenous("z", dp.Markov(np.array([0.5, 1.5]),
                           np.array([[0.95, 0.05], [0.10, 0.90]])))
m.parameters(beta=0.98, gamma=1.0, r_b=0.005, r_a=0.0125,
             w=1.0, chi_0=1.0, chi_1=1.0, a_bar=0.25)
m.local("d = a(+1) - (1+r_a)*a")
m.local("chi = chi_0/(1+chi_1)*abs(d)*(abs(d)/(a+a_bar))^chi_1")
m.local("c = (1+r_b)*b + (1+r_a)*a + w*z - a(+1) - b(+1) - chi")
m.reward("crra(c, gamma)")
m.subject_to("c > 0")
m.aggregate(C="c", A="a", B="b", COST="chi")
sol = m.solve(method="egm", tol=1e-8)
```

The supported contract has an infinite horizon, no discrete choice or
endogenous equilibrium price, at most one Markov productivity shock, at least
two nodes per asset, nonnegative illiquid holdings, `0 < beta < 1`, positive
`gamma`, `w`, `chi_0`, `a_bar`, `chi_1 >= 1`, and positive gross returns.
Consumption must be a local named `c`, `cons`, or `consumption`; utility is
`crra(consumption, gamma)` (including a named curvature parameter or constant)
or `log(consumption)`. Log utility always has curvature one, irrespective of an
unrelated parameter named `gamma`. The budget is recognized structurally after
inlining locals; reordered sums/products, expanded products, renamed assets and
the equivalent cost forms `chi_0*abs(d)^(1+chi_1)/((1+chi_1)*(a+a_bar)^chi_1)` and
`chi_0/(1+chi_1)*(abs(d)/(a+a_bar))^(1+chi_1)*(a+a_bar)` are supported.
Asset roles follow the recognized cost, including reversed declarations.

Price bindings accept `r_b`/`rb`/`r`/`r_b_ss`/`r_ss`/`r_liquid`,
`r_a`/`ra`/`r_a_ss`/`r_k`/`r_illiquid`, and `w`/`w_ss`/`z`/`y` (one used parameter
per role); `a_bar` also accepts `b_bar`. These same bindings apply to stationary
solutions, price paths and Jacobians. Discounting may use `discount("patience")`.
Cost parameters remain explicitly named `chi_0`, `chi_1`, and `a_bar`/`b_bar`.
Arbitrary registered functions in the budget, unsupported denominators or real
powers, extra budget terms, constraints, life cycles and discrete choices are
rejected before solving. Numerical probes are a second check, not the proof of
the budget identity. Registered functions in aggregates remain supported.

Arrays use `(illiquid asset, liquid asset, productivity)` order. `policy("a")`
returns current assets; `policy("a(+1)")` returns the saving policy. Inspect
`sol.raw.converged` and `sol.raw.sup_norm`: exhaustion of the iteration budget
warns, and an infeasible consumption floor raises an error. Refine and enlarge
asset grids to check numerical accuracy and top-grid mass.
`converged` also requires the distribution's stationarity residual to be below
`1e-10`; finite, normalized mass alone is insufficient.

[`audit_result(sol)`](result_audits.md) exposes budget and Euler/KKT maps, locations,
stationary-weighted percentiles, separate mass/stationarity checks and upper-bound
mass. Grid or horizon refinement requires explicit callbacks. The first-order
bridge permits price changes to the recognized liquid return, illiquid return and
income parameters; it does not authorize arbitrary cost or preference shocks.

`dp_sequence_jacobian(sol, inputs=["r_b", "r_a", "w"], outputs=["C", "COST"], T=24)`
and `dp_nonlinear_transition` evaluate declared aggregates, including registered
functions and their direct price dependence. Declared aggregate expressions
preserve their timing: `A="a"` is current wealth, while `A="a(+1)"` is saving.
Undeclared uppercase aliases `A` and `B` denote end-of-period holdings.
**Migration from 5.3:** the one-asset bridge previously interpreted a declared
`A="a"` as saving. Use `A="a(+1)"` explicitly to retain that timing; current
wealth has zero impact response under a fixed initial distribution.
`HANKModel(..., household_model=m)` supports linear general-equilibrium dynamics.
Two-asset panel simulation and nonlinear general-equilibrium simulation of a
declarative household are not implemented and raise `NotImplementedError`.

## Checking a specification

`m.check()` compiles the model without solving it and raises `ModelSpecError`
with the reason, for instance an unknown identifier or a name declared twice.
`solve("egm")` and `solve("hjb")` add their own checks and raise the same error
for a budget that is not of the form `c + a(+1) = R*a + y(z)` or a drift that is
not linear in the control.

## What `dp` does not do yet

- EGM with multiple assets outside the specialized two-asset contract above
  (use `method="vfi"`, which takes several states);
- more than one price in continuous time;
- Krusell-Smith forecast rules with more than one moment.

## Porting from vfi.Model

`vfi.Model` is deprecated and will be removed in 6.0. A `vfi.Model` took Python
functions for the utility and the budget; `dp` takes expressions. Each lambda
becomes its expression string, the budget becomes a local for `c`, and
`c > 0` replaces the `-1e10` penalty. This is the precautionary-savings model of
notebook 21, first as it was written for `vfi.Model`:

```text
m = vfi.Model("Precautionary Savings Model")
m.set_param("beta", 0.96).set_param("gamma", 2.0).set_param("alpha", 0.36).set_param("delta", 0.10)
m.add_state("a", 0.1, 15.0, 60).add_shock("z", 0.90, 0.15, 5)
m.set_utility(lambda c, p: c**(1 - p["gamma"]) / (1 - p["gamma"]))
m.set_budget(lambda a, ap, z, p: np.exp(z) * a**p["alpha"] + (1 - p["delta"]) * a - ap)
res = m.solve(howard_steps=15, tol=1e-6)
```

and as a `dp` model:

```python
pm = dp.Model("Precautionary Savings Model")
pm.parameters(beta=0.96, gamma=2.0, alpha=0.36, delta=0.10)
pm.exogenous("z", dp.AR1(rho=0.90, sigma=0.15, n=5))   # Tauchen, as add_shock
pm.state("a", np.linspace(0.1, 15.0, 60))
pm.local("c = exp(z)*a^alpha + (1 - delta)*a - a(+1)")
pm.reward("c^(1 - gamma)/(1 - gamma)")
pm.subject_to("c > 0")

ps = pm.solve(method="vfi", tol=1e-6)
print(ps.mean("a"), ps.gini("a"))   # 6.5733 and 0.3494
```

The budget is not of the form `R*a + y(z)`, so this model uses
`method="vfi"`. The `dp` solution chooses the same next-period grid node as
`vfi.Model` in every state, and the value functions agree to 7e-7.
`res["V"]`, `res["policy_a"]` and `res["a_grid"]` become `ps.V`,
`ps.policy("a(+1)")` and `ps.grids["a"]`. `res["mean_assets"]` and
`res["gini"]` (6.6767 and 0.3924) came from a 2000-household, 100-period
simulation, and that Gini subtracted the minimum asset holding first.
`ps.mean("a")` and `ps.gini("a")` are the exact values of the stationary
distribution, and `ps.simulate(...)` gives the panel.

## Validation

The Aiyagari, Huggett, life-cycle and two-asset models written as `dp` models
reproduce `vfi.examples` exactly (same prices, policies and value functions).
Independent checks do not go through the compiler: two-period log-utility
models match their closed form to 1e-12, the logit last period matches its
closed form to 1e-12, a two-period discrete-choice model matches a
200,001-point brute-force maximisation, and the continuous-time Aiyagari
equilibrium matches `vfi.solve_aiyagari_continuous_hjb` (r* to 1e-9). The tests
are in `tests/test_dp/`.

## Worked example

`python -m puremacro.examples.scf_aiyagari` calibrates the discount factor of a
`dp` Aiyagari model (EGM) to a wealth-to-income ratio in the Survey of Consumer
Finances and compares untargeted moments with the data, with standard errors.
See [Survey microdata](micro_data.md).

## References

- Aiyagari, S. R. (1994). Uninsured idiosyncratic risk and aggregate saving. *Quarterly Journal of Economics*, 109(3), 659–684.
- Achdou, Y., Han, J., Lasry, J.-M., Lions, P.-L. and Moll, B. (2022). Income and wealth distribution in macroeconomics: a continuous-time approach. *Review of Economic Studies*, 89(1), 45–86.
- Carroll, C. D. (2006). The method of endogenous gridpoints for solving dynamic stochastic optimization problems. *Economics Letters*, 91(3), 312–320.
- Iskhakov, F., Jørgensen, T. H., Rust, J. and Schjerning, B. (2017). The endogenous grid method for discrete-continuous dynamic choice models with (or without) taste shocks. *Quantitative Economics*, 8(2), 317–365.
- Krusell, P. and Smith, A. A. (1998). Income and wealth heterogeneity in the macroeconomy. *Journal of Political Economy*, 106(5), 867–896.
