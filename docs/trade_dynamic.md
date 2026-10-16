> 🇬🇧 English · 🇪🇸 [Español](es/trade_dynamic.md)

# Dynamic MRIO with sector-specific capital

`puremacro.trade.dynamic` is a perfect-foresight, deterministic dynamic
extension of the fixed-coefficient multi-country input-output model in which
every source country-industry cell keeps its own installed capital stock with
Tobin-q adjustment costs. Results are frozen objects with `to_dataframe()`,
`to_markdown()`, `to_latex()` and `to_typst()`, constructors validate their
input, errors are structured, and no solve returns `converged=True` without
passing an independent accounting certificate at every date.

It is **not** the static CGE of `solve_trade_equilibrium`: there is no
Armington substitution, labour is inelastic and nationally mobile, countries
are financially autarkic, and government consumption sits inside the
household basket. Its welfare measure is a CRRA consumption equivalent, not
the Hicksian EV of [`compute_hicksian_welfare`](trade_welfare.md). Read the
limitations at the end before interpreting any number.

## Quick start on a synthetic table

```python
from puremacro.trade import calibrate_trade_model
from puremacro.trade.data import synthetic_codes, synthetic_icio_table
from puremacro.trade.dynamic import (
    DynamicAccounts, DynamicTariff, DynamicEconomy, calibrate_dynamic,
    solve_dynamic_steady_state, solve_dynamic_transition, tariff_path,
    stability_report,
)

# A seeded synthetic table: twelve countries, five sectors (S01 and S02 are merchandise).
countries, sectors = synthetic_codes(12, 5)
calib = calibrate_trade_model(synthetic_icio_table(12, 5, seed=2), country_codes=countries, sector_codes=sectors)
accounts = DynamicAccounts.from_trade_calibration(calib, merchandise_sectors=("S01", "S02"))
calibration = calibrate_dynamic(accounts, beta=.96, delta=.08)
print(calibration.summary())

economy = DynamicEconomy(calibration, adjustment_cost=2., risk_aversion=2.)
baseline = solve_dynamic_steady_state(economy)                     # the benchmark
tariff = DynamicTariff.uniform(calibration, "C01", .05, sectors="merchandise")
terminal = solve_dynamic_steady_state(economy, tariff)             # exact 2C condensation
print(terminal.to_dataframe().loc[["C01", "C02"]])

report = stability_report(economy, baseline)                       # Blanchard-Kahn count
print(report.summary())

run = solve_dynamic_transition(economy, tariff_path(economy.zero_policy(), tariff, horizon=40),
                               terminal=terminal)
print(run.summary())
print(run.welfare.to_dataframe().loc[["C01", "C02"]])
```

The synthetic flows are generic random draws, so the numbers are software
outputs, not estimates. The calibration needs no adjustment (60 active cells,
no factor or investment repair), the benchmark is determinate (60 stable roots
for 60 predetermined stocks), and a permanent 5% duty by `C01` on all its
merchandise imports solves at 40 dates in three Newton iterations with a
certificate near `2e-11`. The 40-date path is a solved horizon, not an
accepted one: use `run_horizon_ladder` (next section) before reading its
welfare numbers.

## A small example reproduced by the test suite

The two-country, two-sector analytic accounts are the fixture used by the
tests in this repository. They are a software fixture, never empirical
evidence.

```python
import numpy as np
from puremacro.trade.dynamic import (
    analytic_two_country_accounts, calibrate_dynamic, DynamicEconomy,
    DynamicTariff, tariff_path, solve_dynamic_steady_state, run_horizon_ladder,
)

accounts = analytic_two_country_accounts()
calibration = calibrate_dynamic(accounts)
economy = DynamicEconomy(calibration)
zero = economy.zero_policy()
rates = np.zeros((calibration.n_cells, calibration.n_countries))
rates[calibration.country == 0, 1] = .06     # BBB taxes AAA's goods at 6%
rates[calibration.country == 1, 0] = .08     # AAA taxes BBB's goods at 8%
shock = DynamicTariff.build(rates, consumption_rates=rates * .7,
                            investment_rates=rates * 1.3, label="analytic tariff")
terminal = solve_dynamic_steady_state(economy, shock, tol=1e-11)

# Announced at date 0, implemented at date 2, permanent.
ladder = run_horizon_ladder(
    economy, lambda T: tariff_path(zero, shock, horizon=T, announcement=2),
    horizons=(80, 160, 320), terminal=terminal, tol=1e-10,
)
print(ladder.to_dataframe())
accepted = ladder.accepted            # the 320-date solution (status "verified")
print(accepted.welfare.summary())      # AAA -0.24%, BBB +0.11% of benchmark consumption
```

On this fixture the 80 to 160 comparison agrees on the first 20 dates
(3.6e-7 in logs) but welfare still moves by 0.0055 percentage points, so it
is rejected; 160 to 320 passes the window (2.7e-12), welfare (4.3e-5 pp) and
the terminal state gap (4.3e-5). The slowest stable root is about 0.98, which
is why hundreds of dates are needed.

## The model

Cell `j` in country `c(j)`, date `t`, benchmark prices equal to one. Value
added is benchmark-normalized Cobb-Douglas in sector-specific capital and
national labour; intermediate inputs and final baskets are fixed
coefficients:

```
v[j,t] = w[c(j),t]^(1-alpha[j]) (R[j,t]/R0[j])^alpha[j]
(1-tax[j]) p[j,t] = sum_i A[i,j] (1+tau[i,c(j),t]) p[i,t] + b[j] v[j,t]
L[j,t] = (1-alpha[j]) b[j] v[j,t] y[j,t] / w[c(j),t]
K[j,t] = alpha[j] b[j] v[j,t] y[j,t] / R[j,t]
y = A y + omegaC C + omegaI I_national
PC[c] = sum_i (1+tauC[i,c]) omegaCtax[i,c] p[i] / (1-tC[c])   (exempt X part added)
PI[c] = sum_i (1+tauI[i,c]) omegaI[i,c] p[i] / (1-tI[c])
```

Capital, installation and the Euler equation:

```
K[j,t+1] = (1-delta) K[j,t] + K[j,t] Phi(x[j,t]),   x = I/K
Phi(x) = x - phi/2 (x-delta)^2
q[j,t] = PI[c(j),t] / Phi'(x[j,t])
q[j,t] = m[c,t+1] (R[j,t+1] + q[j,t+1] G[j,t+1]),  G = 1-delta+Phi(x)-x Phi'(x)
m[c,t+1] = beta (C[c,t+1]/C[c,t])^(-sigma) PC[c,t]/PC[c,t+1]
```

The household budget pays for exactly the consumption and investment
delivered by goods clearing:

```
PC C + PI sum_{j in c} I[j] + XN0[c] w_last
    = w L0[c] + sum_{j in c} R K + product and production tax receipts + tariff receipts
```

There is no traded bond. Benchmark net exports `XN0` are balanced world
transfers held fixed in units of the last country's wage (the numeraire).
One redundant budget (the country with the largest budget scale) is replaced
by the numeraire equation and checked afterwards with every other account.
Because the last country of the registry is both the wage numeraire and the
transfer anchor, the order of the country registry is economically
meaningful: reordering countries changes the closure and therefore real
allocations whenever relative wages move (by about the change in the
anchor's relative wage times `XN0`). Reordering sectors is a pure relabelling.

**Elimination.** The retained unknowns per date are
`z_t = [log w (C), log C/C0 (C), log K_{t+1}/K0 (N)]`, so `N + 2C` per date.
Installation is inverted on its increasing branch,
`x = delta + 2g/(1 + sqrt(1 - 2 phi g))` with `g = K'/K - 1`; goods clearing
uses cached Leontief responses `YC = (I-A)^-1 omegaC`, `YI = (I-A)^-1 omegaI`;
factor prices follow from the Cobb-Douglas conditions; producer prices solve
one tariff-adjusted linear system per distinct policy (sparse LU up to 600
cells; above 600 cells, dense LU when the coefficient matrix has density
above .25 and otherwise preconditioned BiCGSTAB with a GMRES fallback). The
retained residual is `C` labour-clearing equations, `C` budgets with one
numeraire, and `N` Euler equations. `DynamicEconomy.state_jvp`/`states_jvp`
give exact analytic Jacobian-vector products, batched per policy.

**Certificate.** Every accepted date passes 17 independently rebuilt checks
(goods, zero profit, labour, capital accumulation, factor payments, capital
and labour demand, technology, investment FOC, Euler, tariff revenue,
government budget, all budgets including the omitted one, current account,
world transfer balance, investment goods value) at `max(1e-8, tol)`.

**Steady state.** `solve_dynamic_steady_state(method="condensed")` solves an
exact `2C` condensation in national wages and investment prices with a dense
analytic Jacobian (default above 240 unknowns); `"full"` solves the `N + 2C`
stationary equations with matrix-free Newton. Both re-evaluate the full
residual and the certificate. Both are local Newton methods without a
continuation fallback, and `method="auto"` does not retry with the other
method on failure: a large duty may need `start` from the steady state of a
smaller duty, and a prohibitive duty may have no interior steady state at
all (on the analytic fixture a 100% uniform duty raises
`DynamicSolveError`). An infeasible `start` or a policy whose tariff-adjusted
price system has no positive solution raises `EconomicDomainError` (a
`ValueError`) before any Newton step.

**Transition.** `solve_dynamic_transition` stacks `T (N + 2C)` unknowns with
fixed initial capital and a separately solved terminal steady state under the
last policy (held beyond the horizon), and runs a matrix-free inexact Newton:
`lgmres` (with a `gmres` fallback) preconditioned by an approximate
capital-time tridiagonal plus national blocks (`preconditioner="capital"`),
forcing term `max(1e-5, min(.1, sqrt(||F||)))`, a `.75` cap on log moves and
Armijo halving. A line-search stall raises `DynamicSolveError` naming the
equation, date and cell, the smallest installation rate on the last accepted
iterate (including the terminal boundary date) and what the full Newton step
would do to it. The default initial guess is an exponential blend from the
initial to the terminal state; when a large shock would make that guess
disinvest (installation rate `x <= 0`) the solver uses a linear blend
instead, and when even the linear blend is infeasible no path of that horizon
can keep investment interior, which is reported as a `DynamicSolveError`
naming the cell (`metadata["start"]` records the guess used). Steady-state
line-search failures name the equation with the largest residual in
`location`. By default the transition first runs `stability_report` on the
terminal state and refuses an indeterminate closure
(`DynamicDeterminacyError`); `require_determinacy=False` solves the finite
stack anyway.

## Stationary calibration ledger

One input-output year is not a steady state. `calibrate_dynamic` derives
`R0 = PI0 (1/beta - 1 + delta)`, `K0 = alpha VA0 / R0`, replacement investment
`delta K0`, reassigns the stationary investment flow inside observed
absorption (rebasketing only where the observed composition cannot finance
replacement, with `investment_policy="reallocate"`), pools inventory changes,
valuables and exempt purchases abroad into the consumption basket, and
reconciles final-use taxes. `factor_policy="reclassify_losses"` turns
negative factor components into production subsidies and clips shares to
`capital_share_bounds`. Everything is recorded in `calibration.report` and
`calibration.to_dataframe("countries" | "factors" | "investment")`.

Bridges: `DynamicAccounts.from_icio` accepts `ICIOData` (processed C/I/Cx
layout: an I column that contains inventory changes can have negative cells,
and then `negative_investment` must be `"raise"` or `"to_inventory"`; capital
shares come from the table's labour and capital rows), `RawIOData` (loader
categories such as HFCE/NPISH/GGFC/GFCF/INVNT/DPABR are mapped onto the
contract `("C","G","X","V","VAL")`; without factor detail the 2/3-1/3 split is
assumed and recorded) and any input-output table object with `Z`, `F`,
`fd_codes`, `VA`, `TLS`, `TFD`, `output`, `country_codes` and `sector_codes`.
The EXIOBASE loader's `EXPORT` column (exports to regions outside the table)
has no default mapping because it is not residents' purchases abroad; pass
`fd_map={"EXPORT": "X"}` or another target explicitly. `from_arrays` takes
`F` as `(M, N, K)` or `(M, N*K)` and `TFD` as `(N, K)` or flat `(N*K,)`; a
transposed `(K, N)` tax table is rejected when `N != K` (with as many
countries as final-use codes the orientation cannot be detected, so pass
destination rows and category columns). The default
`merchandise_sectors="isic_a_c"` marks as goods the 22 OECD 45-industry goods
codes and codes that are the single letter A, B or C or start with one of them
followed by a digit, except FIGARO's C33 (repair and installation); a table
object that carries its own `merchandise_mask` keeps it, and other registries,
such as EXIOBASE's `i01...` codes or synthetic `S01...` codes, need an
explicit sector list. The rule and the resulting goods list are recorded in
`metadata["merchandise_sectors"]`. `from_trade_calibration` reads
`calib.data_calibra`; its Cx column becomes X (residents' purchases abroad)
only for a table without a recorded final-use mapping and for the OECD
`DPABR` mapping, the FIGARO, WIOD and Eora loaders' Cx (signed inventories and
valuables) becomes V, and EXIOBASE's Cx, which includes its export column,
needs an explicit `cx_category`.

## Policies

`DynamicTariff` stores three `N x C` ad-valorem rate tables (intermediate,
consumption, investment deliveries) by source cell and destination country;
domestic deliveries and residents' purchases abroad never pay. Destination
sector specific duties are not representable. Rates must exceed `-1`: zero
is free trade and negative rates are import subsidies.
`DynamicTariff.uniform(cal, "C01", .1, sectors="merchandise", exporters=("C02",), uses=("intermediate",))`,
`from_rates` and `build` validate the tables. The fingerprint (sha256 of the
three tables) is always derived from the stored tables, never passed in, so
`dataclasses.replace(policy, rates=...)` produces a new fingerprint;
identical policies share one price-network factorisation, and a policy whose
tables no longer match its fingerprint is refused. A selection with no
foreign source cell (for example `exporters=("C01",)` for importer `"C01"`)
raises `ValueError` instead of returning a zero policy under a tariff label.
`tariff_path(baseline, shock, horizon=T, announcement=2, duration=5)` builds
anticipated, permanent or temporary schedules.

## Horizon acceptance

A finite stack is an approximation to an infinite-horizon path.
`compare_horizons(short, long)` requires: (1) the first `periods=20` retained
states to agree within `1e-5` in logs; (2) consumption-equivalent welfare to
agree within `1e-4` percentage points; (3) a terminal check on the long
solution. The default `terminal_check="state_gap"` requires
`max |z_T - z*| <= 1e-4` and yields status `verified`. The opt-in
`"discounted_wealth"` mode requires the discounted endpoint capital value

```
D[c] = beta^(T-1) (C[c,T-1]/C0[c])^(-sigma) / PC[c,T-1] * sum_{j in c} q[j,T-1] K[j,T] / C0[c]
```

to be at most `1e-6` and yields the explicitly weaker status
`verified_window_and_welfare`: it certifies neither full terminal-state
convergence nor a rigorous bound on the utility tail. None of these
tolerances is an error bound against the unknown infinite-horizon solution.
Both solutions must come from the same economy (the transition result records
a fingerprint of the calibration arrays, adjustment cost, risk aversion and
wage numeraire in `metadata["economy_fingerprint"]`), start from the same
initial state, end at the same terminal state and follow the same announced
policy. `run_horizon_ladder` solves increasing horizons, warm-starting each
from the previous path, and stops at the first accepted comparison.

## Welfare

`consumption_equivalent_welfare` returns, per country, the permanent
proportional change in benchmark consumption `e` with
`sum_t beta^t u(e) = sum_t beta^t u(C_t/C0)` for CRRA `u` (the same
`risk_aversion` as the Euler equation; log at one), the stationary tail
weighted `beta^T/(1-beta)`, the annual expenditure equivalent at benchmark
prices and its percent of benchmark GDP. There is no cross-country
aggregation and no decomposition into channels.

## Determinacy diagnostic

`stability_report(economy, steady)` linearises the date equations with the
model's own JVPs, forms the pencil
`[[A_plus,0],[0,I]] s_{t+1} = [[-A_0,-A_minus_K],[E_K,0]] s_t` with
`s_t = (dz_t, dK_t)`, and counts generalized eigenvalues: determinacy needs
exactly `N` stable roots for the `N` predetermined stocks. The result reports
the slowest stable root, the smallest unstable root and the country loading
shares of its eigenvector. The point must be stationary under the policy
passed (or carried by a steady-state result): the report refuses, with a
`ValueError`, a state whose residual `F(z, z, z)` exceeds `stationarity_tol`
(default `1e-8`). The dense problem has order `n_vars + N` and its cost grows
with the cube of that order (seconds at a few hundred cells, tens of seconds
near a thousand). `solve_dynamic_transition` caches the report on the economy
by terminal-policy fingerprint, so a horizon ladder pays it once.

## What is validated here and what is not

Verified by tests in this repository (`tests/test_trade_dynamic.py`), on the
analytic fixture, the frozen OECD three-region fixture and seeded synthetic
tables:

* Benchmark stationarity, factor and tax identities, independent scalar-loop
  accounts at steady states and along transitions, analytic JVPs against
  central differences (2e-6), exact condensation against the full solve, the
  condensed Jacobian against finite differences, batched evaluation,
  anticipation timing, temporary policies, horizon-comparison guards, the
  discounted endpoint diagnostic, and the consumption-equivalent welfare
  identities.
* `puremacro.dsge.solve_perfect_foresight` on the same equations at 40 dates
  as an independent stacked-Newton oracle; the iterative price and goods
  backends agree with sparse LU to 1e-10 on a seeded sparse table.
* Guards: a policy rebuilt with `dataclasses.replace` is solved as the new
  policy, horizon comparisons refuse solutions of different economies,
  `stability_report` refuses nonstationary points, and the determinacy gate
  raises a structured error on an indeterminate fixture.
* Synthetic tables: steady states with and without a 10% duty on a 5 by 4
  table, and (slow) a determinate 12 by 5 table with a certified 40-date
  transition and a horizon ladder.

Not validated here:

* Parameters. A single input-output table does not identify historical
  capital stocks, depreciation, adjustment costs or intertemporal
  preferences. The examples use `beta=.96`, `delta=.08`, adjustment cost 2
  and CRRA risk aversion 2; numerical acceptance does not validate those
  economic parameter choices.
* Investment. This is an interior investment model; it does not solve an
  investment-irreversibility complementarity problem. The admissible shock
  can depend on the horizon: when a long horizon makes installation in some
  cell approach zero at the terminal boundary date, the line search stalls
  and the error names the cell. A line-search failure is evidence, not a
  proof, that no interior path exists.
* Scope. Fixed sourcing, Cobb-Douglas value added, fixed national labour and
  financial autarky with balanced transfers; there are no Armington or
  labour-supply elasticities. Recorded inventory changes and valuables do not
  identify inventory stocks, and consumption includes government and
  nonprofit final demand in the representative household basket.
* Horizons. Long horizons are intrinsic (slowest stable roots near 0.98 on
  the fixtures), and an accepted ladder on one table says nothing about
  another.
* Determinacy. Local indeterminacy is a property of the linearised closure on
  a given table, not a solver defect, and large tables can be indeterminate.
  Run `stability_report` before relying on `require_determinacy=False`. The
  `state_gap` criterion can reject a ladder that `discounted_wealth` would
  accept; treat `discounted_wealth` results as early-window statements only.

## Related

* [Consistent trade accounting](trade_accounting.md), [Hicksian consumption
  welfare](trade_welfare.md) and [tariff policy](trade_policy.md) for the
  static model this extension does not replace.
* `puremacro.dsge.solve_perfect_foresight` for the generic stacked Newton
  used as the independent oracle.
* [Structural validation status](STRUCTURAL_VALIDATION_STATUS.md).
