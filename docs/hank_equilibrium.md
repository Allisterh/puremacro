> 🇬🇧 English · 🇪🇸 [Español](es/hank_equilibrium.md)

# Published one-asset HANK equilibrium

`calibrate_one_asset_hank` and `solve_one_asset_hank` implement the one-asset
model in the authors' [HANK notebook](https://github.com/shade-econ/sequence-jacobian/blob/eb4ac4c3157357a4cacbea4b5a7a281f5b48cab9/notebooks/hank.ipynb),
associated with [Auclert, Bardóczy, Rognlie and Straub (2021)](https://doi.org/10.3982/ECTA17434).
The implementation is NumPy/SciPy. The authors' package is used only in a
separate environment to generate independent comparison arrays.

```python
from puremacro.models import calibrate_one_asset_hank, solve_one_asset_hank

steady = calibrate_one_asset_hank()  # 500 asset nodes, 7 income states
linear = solve_one_asset_hank(steady, horizon=300)
nonlinear = solve_one_asset_hank(steady, horizon=300, nonlinear=True)
assert steady.converged and linear.converged and nonlinear.converged
print(steady.summary())
print(nonlinear.diagnostics)
print(nonlinear.to_dataframe().head())
```

The shock reduces the Taylor-rule intercept by `0.0025 * 0.61**t`. Set
`magnitude=0.0025` for the opposite shock. Paths are **level deviations**, with
rates and inflation in decimal units. A consumption response of `0.001` is
0.1% of steady consumption, which is normalized to one. Results also provide
`summary`, `to_markdown`, `to_latex` and `to_typst`.

## Equations and timing

Households choose consumption, hours and saving:

$$c_t+a_{t+1}=(1+r_t)a_t+w_t e_t n_t+(Div_t-Tax_t)e_t,\quad a_{t+1}\ge0.$$

Marginal utility is $c^{-1/\mathrm{EIS}}$; the labor first-order condition is
$v_\phi n^{1/\mathrm{Frisch}}=wec^{-1/\mathrm{EIS}}$. EGM solves the interior
problem and a constrained budget/FOC solve handles the borrowing limit.
Transfers and dividends are proportional to skill, whose stationary mean is one.

Output and steady productivity are one. Firms hire $L_t=Y_t$, pay dividends
$Div_t=Y_t-w_tL_t-Adj_t$, and incur

$$Adj_t=\frac{\mu}{\mu-1}\frac{\log(1+\pi_t)^2}{2\kappa}Y_t.$$

Government debt is fixed at $B=5.6$ and $Tax_t=r_tB$. Realized returns include
the date-zero revaluation of previously issued nominal debt:

$$1+r_t=\frac{1+r^\star_{t-1}+\phi\pi_{t-1}}{1+\pi_t}.$$

The solver determines $(\pi_t,w_t,Y_t)$ from $A_t=B$, $NE_t=L_t$ and

$$0=\kappa(w_t-1/\mu)+\frac{Y_{t+1}}{Y_t}
\frac{\log(1+\pi_{t+1})}{1+r_{t+1}}-\log(1+\pi_t).$$

Goods clearing $Y_t=C_t+Adj_t$ and household budgets are separate checks.
Initial nominal policy is predetermined. Terminal household values and future
aggregate quantities equal their steady-state values; horizon refinement
measures the effect of that terminal assumption.

## Calibration and independent checks

Defaults are EIS = Frisch = 0.5, log-skill persistence 0.966, unconditional
log-skill standard deviation 0.5, markup 1.2, Phillips coefficient 0.1, Taylor
coefficient 1.5 and quarterly steady interest rate 0.005. Beta and labor
disutility are calibrated to assets 5.6 and effective labor one. Matching
those targets alone does not demonstrate numerical convergence.

The pinned author-code fixture compares saving, consumption, hours, the
distribution, calibrated parameters, and nine aggregate paths. Linear paths
must agree within `1e-6` per unit of the monetary shock; nonlinear paths for
both signs must agree within `1e-7` in levels. Market residuals must be below
`1e-8`. The reference generator verifies the installed author sources against
the pinned checkout and records array hashes and dependency versions.

```sh
python -m pytest tests/test_models/test_one_asset_hank.py -q
python -m tools.reference_validation.check_hank_equilibrium \
  --output /tmp/hank-equilibrium --convergence
```

The refinement protocol compares the first 40 quarters. Its criterion is 1%
of the finer response's peak, with a level floor of `1e-6`; changes in beta and
labor disutility must be below `1e-4`. Asset grids, income-state counts, the
asset bound and transition horizons are checked separately. Richer income
grids require a wider asset bound than the notebook's 150.

The frozen run in `tests/fixtures/hank_equilibrium/convergence.json` records:

| Check | Result |
|---|---|
| Published 500 × 7 calibration | beta 0.9822435541; labor disutility 0.7864334303 |
| Largest author-code linear path difference | 9.23e-14 |
| Largest nonlinear path difference, either sign | 9.67e-10 |
| Assets, 500 → 1,000 nodes | Passed |
| Income states, 11 → 15 and 15 → 21 | Failed; retained in the record |
| Income states, 21 → 31 | Passed; worst response change 0.614% of its peak |
| Assets, 500 → 1,000, at 31 income states | Passed |
| Horizon, 300 → 600, at 31 income states | Passed |

The extended income study uses assets up to 600. The asset-bound check from
300 to 600 at 15 states passes; the largest terminal-grid mass across all
reported transitions is 1.09e-29. The notebook calibration reproduces the
authors' discretization; the extended grid establishes the stated refinement
criterion. It does not certify every intermediate income grid.

This benchmark validates a deterministic perfect-foresight monetary transition
around one published steady state. It does not establish accuracy for arbitrary
large shocks, a binding lower bound on nominal rates, two assets, or aggregate
stochastic welfare. The existing `solve_hank_sequence_space` and DAG examples
have different closures; their results are not this benchmark.
