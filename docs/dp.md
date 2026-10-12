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
- `sol.raw`: the underlying engine's result.

```python
c = sol.policy("c")                 # (150, 5)
print(sol.mean("c"), sol.mean("a")) # mean consumption and assets
```

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

Each cohort has mass one, and newborns start at the lowest grid point with
shocks drawn from their stationary distribution unless `newborns=` says
otherwise.

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

## Checking a specification

`m.check()` compiles the model without solving it and raises `ModelSpecError`
with the reason, for instance an unknown identifier or a name declared twice.
`solve("egm")` and `solve("hjb")` add their own checks and raise the same error
for a budget that is not of the form `c + a(+1) = R*a + y(z)` or a drift that is
not linear in the control.

## What `dp` does not do yet

- EGM with more than one asset (use `method="vfi"`, which takes several states);
- more than one price in continuous time;
- Krusell-Smith forecast rules with more than one moment;
- panel simulation of individual histories and Python-callable rewards; for
  those, `vfi.Model` and the `vfi` solvers remain supported.

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
