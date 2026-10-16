> 🇬🇧 English · 🇪🇸 [Español](es/trade_condensed.md)

# Condensed one-factor Leontief tariff model

`puremacro.trade.condensed` solves a static multi-country, multi-sector tariff
model with fixed input coefficients, one composite factor per country and
lump-sum rebated import duties, eliminates prices and outputs exactly, certifies
every reported equilibrium from the raw table flows, and reports only
numeraire-free measures. It is a different model from
`solve_trade_equilibrium` (consistent accounting); the comparison table below
says where they differ.

## Runnable example on a synthetic table

```python
import numpy as np
from puremacro.trade import calibrate_trade_model
from puremacro.trade.data import synthetic_codes, synthetic_icio_table
from puremacro.trade.condensed import (
    BalancedIOTable, build_tariff_wedges, calibrate_condensed, compute_measures, solve_condensed,
)

# A seeded synthetic table: five countries, four sectors (the first two are merchandise).
countries, sectors = synthetic_codes(5, 4)
trade_calib = calibrate_trade_model(synthetic_icio_table(5, 4, seed=0),
                                    country_codes=countries, sector_codes=sectors)
table = BalancedIOTable.from_trade_calibration(trade_calib, merchandise_mask=[True, True, False, False])
calib = calibrate_condensed(table)

# C01 raises a 15% duty on merchandise from C02 and C03, and 5% on everyone else.
wedges = build_tariff_wedges(calib, {"C01": {"*": 0.05, "C02": 0.15, "C03": 0.15}})
print(wedges.to_dataframe().head())
result = solve_condensed(calib, wedges)                      # gate, Newton, certificate
print(result.summary())
measures = compute_measures(calib, wedges, result.state, country="C01", partners=("C02", "C03"))
print(measures.to_markdown())
```

The synthetic sector codes `S01`..`S04` belong to no known registry, so the
bridge takes an explicit `merchandise_mask`; the duties then fall on the two
merchandise sectors only. The run takes six Newton iterations and returns a
certificate maximum block near `3e-12` (below the `1e-10` gate). For `C01`
the effective rate on dutiable imports lies between the 5% and 15% rates
(about 11.7%), tariff revenue is about 1.8% of GDP, and the equivalent
variation splits exactly into real factor income, real net revenue and the
deficit-and-inventory transfer. The numbers come from generic random flows,
not from any economy. Three-region and other synthetic examples are in
`tests/test_trade_condensed.py`.

## What it computes

Indices: countries `k`, cells `i = country * S + sector` (M = N S), final
categories `f` in {C, G, X} with Cobb-Douglas shares and exogenous inventories V.
The benchmark has all prices, wages and multipliers equal to one.

Calibration from a balanced table (`table_from_arrays`, then `calibrate_condensed`):

```
a_ij = Z_ij / y0_j     b_j = VA_j / y0_j     t_j = TLS_j / y0_j
omega^f_ik = F^f_ik / B^f_k        A0^f_k = B^f_k + TFD^f_k       t^f_k = TFD^f_k / A0^f_k
theta^f_k = A0^f_k / sum_f A0^f_k  qV_ik = F^V_ik                 TV_k = TFD^V_k
Fbar_k = sum_{j in k} VA_j         Y0_k = Fbar_k + sum_{j in k} TLS_j + sum_f TFD^f_k + TV_k
XN0_k = fob exports - imports      s_k = XN0_k / sum_l Fbar_l
```

Equilibrium in levels (2M + 3N unknowns, one equation redundant by Walras' law):

```
(E1) (1 - t_j) p_j = b_j w_c(j) + sum_i a_ij tau_ij p_i                       zero profit
(E2) y_i = sum_j a_ij y_j + sum_k (sum_f omega^f_ik q^f_k + qV_ik)             goods clearing
(E3) sum_{j in k} b_j y_j = Fbar_k                                              factor market
(E4) Y_k = w_k Fbar_k + PT_k + TR_k                                             income
(E5) XN_k = s_k sum_l w_l Fbar_l                                                closure
(E6) sum_l w_l Fbar_l = sum_l Fbar_l                                            numeraire
P^f_k = sum_i omega^f_ik tau^f_ik p_i     E^f_k = theta^f_k (Y_k - XN_k - E^V_k)
q^f_k = (1 - t^f_k) E^f_k / P^f_k         E^V_k = sum_i tau^V_ik p_i qV_ik + TV_k P^G_k
```

`PT_k` collects product taxes on output and on final purchases; `TR_k` collects
every duty (`(tau - 1) x producer value`) on intermediate, final and inventory
purchases and is rebated through (E4). The trade-balance closure (E5) keeps
each country's balance a fixed share of world factor income (Dekle, Eaton and
Kortum, 2008).

Condensation: given `z = (log w, Y / Y0)`, (E1) is solved exactly by one LU of
`I - B_tau^T` with `B_tau,ij = a_ij tau_ij / (1 - t_j)`, and (E2) by one LU of
`I - a`. The residuals are (E3) and (E4) scaled by base quantities, with the
reference country's income equation replaced by the numeraire; that dropped
equation is the Walras check reported with every state. The Jacobian uses
forward differences (`h = 1e-7`) with prices cached across the income columns.

Solvers (`solve_condensed`): damped Newton with Armijo backtracking, a
`0.3` clamp on log-wage displacements and the stopping rule
`max |r| <= 1e-12`, `|Walras| <= 1e-10`; natural-parameter continuation in the
tariff scale (halve on failure down to `1/256`, secant predictor); a nested
fallback that solves the affine income system given wages; optional
pseudo-arclength continuation with fold monitors (`dlambda/ds`, `sign det J`,
`sigma_min / sigma_max`) and a landing check at `1e-10`; optional near-start
multistart raising `MultipleEquilibria` when a converged start lands farther
than `1e-9`. A Collatz-Wielandt existence gate `rho(B_tau) < 1 - 1e-3`
(`puremacro.trade.regularize.compute_spectral_radius`) runs before every
solve and fails closed: bounds that certify `rho < 1` but not the margin raise
`ProductivityUncertified` ("margin not met", with the bounds attached), bounds
that certify nothing raise it as "inconclusive", and neither claims
nonexistence; wedges built for another table are refused with
`CalibrationError`. When continuation stalls, the nested fallback's diagnostic
tag and its stage record (`continuation_lam_reached`) name the stall.

Certificate (`certify_raw_flows`): ten blocks recomputed from `Z, F, VA, TLS,
TFD` and the wedges, never from `a, b, omega, theta` or the factorizations:
zero profit on active cells, goods clearing, factor markets, household budget,
national income, tariff-revenue identity, closure, current-account identity
(`NX` from bilateral producer values), world adding-up and the numeraire. Every
block must be at most `1e-10`; `solve_condensed` raises `CertificationFailure`
otherwise and never returns `converged=True` after a failure. The certificate
uses the state's basket prices and quantities, so it is independent of the
calibrated coefficients but not of the demand system that produced them. The
two per-cell blocks (zero profit, goods clearing) are scaled by
`max(p_j y_j, cell_floor)` and `max(y_i, cell_floor)` with `cell_floor = 1` in
the table's own units, so cells below one unit are judged at absolute
`1e-10`; pass a smaller `cell_floor` to `certify_raw_flows` for tables in
large units. Phantom cells (empty cells the build fills with `1e-6` flows) are
left out of the zero-profit block only; every other block uses their true
output ratio `y_j / y0_j`, because the model lets phantom output move and
phantom value added is part of `Fbar`. `solve_condensed(..., certify=False)`
skips the certificate for diagnostics only: such a result is `converged` but
neither `passed` nor `certified`.

Measures (`compute_measures`, all numeraire-free, `P^U_k = prod_f (P^f_k /
P0^f_k)^theta^f_k`):

| Measure | Definition |
|---|---|
| consumer prices relative to wages | `P^C_k / (P0^C_k w_k) - 1`, the inverse real consumption wage (not a CPI) |
| equivalent variation | `Atil^1_k / P^U_k - Atil^0_k`, percent of base GDP, with the exact split into real factor income, real net revenue (tariff and other taxes) and the deficit-and-inventory transfer |
| real national income | `(Y_k / P^U_k) / Y0_k - 1` |
| Fisher terms of trade | fob, ex-duty, `sqrt(L^X P^X / (L^M P^M)) - 1` at base quantities |
| tariff revenue | `TR_k / Y_k` and the effective rate on dutiable imports (inventories count as dutiable only when the state was solved with `inventories="tariffed"`) |
| constant-price real GDP | `sum (VA_j + TLS_j) yhat_j + sum_f TFD^f qhat^f + TFD^V`, a composition effect only |
| PE unit-cost bound | `p^PE = (I - B_tau^T)^{-1} b / (1 - t)`, `ln(1 + Pi^C) = ln(1 + Pi^{C,PE}) + Delta^rel`, and the first-order direct/cascade split |
| sector incidence | `sector_incidence(..., groups=...)` with output-weighted within-group dispersion |

`separate_baseline_tariffs(calib, wedges)` re-expresses observed benchmark
duties as explicit gross wedges with residual product taxes, so counterfactual
increments stack on nonzero initial duties without counting benchmark customs
revenue twice (gates at `1e-11`). It is an accounting separation, not evidence
that an unobserved tariff is zero.

## Data contract

`table_from_arrays` (or `BalancedIOTable.from_arrays`) accepts the layout
`Z (M, M)`, `F (M, N, K)`, `VA (M,)`, `TLS (M,)`, `TFD (N, K)`,
`country_codes`, `sector_codes` and `fd_codes` in any of the six OECD codes,
`("C", "G", "X", "V")`, `("C", "G", "X", "V", "VAL")` (VAL merged into V and
recorded), `("C", "G", "X")`, the processed-table `("C", "I", "Cx")`, or the
final-demand vocabularies of the `puremacro.trade.data` loaders: FIGARO
(`P3_S14`, `P3_S15`, `P3_S13` summed into C, `P51G` into G, `P5M` into V),
EXIOBASE (`HFCE`, `NPISH`, `GGFC`, `GFCF`, `INVNT`, `VALUABLES` merged into V,
and an `EXPORT` column that must be identically zero in a closed table), WIOD
(`CONS_h`, `CONS_np`, `CONS_g`, `GFCF`, `INVT`) and Eora (`HFCE`, `NPISH`,
`GGFC`, `GFCF`, `INVNT`, `ACQ_VAL` merged into V). Codes that map to one
category are summed; C and G are required, X and V are zero when absent.
Layouts without a purchases-abroad column (all but the OECD one) give an empty
X basket, so `calibrate_condensed` needs `allow_empty_purchases_abroad=True`
for them (X copies the C basket as an auxiliary price; its expenditure is
identically zero). Three-category layouts and any table with negative G cells
need `negative_investment="raise"` (default) or `"to_inventory"`.
`BalancedIOTable.from_io_table` takes any object with those attributes (sparse
`Z` included), `from_raw` takes a `puremacro.trade.data.RawIOData` from any of
the five loaders, and `from_trade_calibration` bridges a
`TradeCalibrationResult` (labour and capital are summed into the single
factor; G contains inventories; Cx becomes X for a table without a recorded
final-use mapping and for the OECD `DPABR` mapping, while the FIGARO, WIOD and
Eora loaders' Cx becomes the exogenous V, so those tables calibrate with
`allow_empty_purchases_abroad=True`, and EXIOBASE's Cx needs an explicit
`cx_category`). The build adds `1e-6` phantoms to empty cells, floors
non-positive value added at `1e-3` of output, recomputes TLS as the column
residual and gates the change (`1e-4` of world |TLS|, `1e-2` of a cell's
output), and assigns world round-off in trade balances to the reference
country (the last code by default). The merchandise mask (ISIC Rev.4 sections
A-C; utilities, construction and services are not merchandise) is inferred
for the OECD 45-industry registry, the ISIC Rev.4 section letters and the
frozen three-sector fixture; other
registries (the FIGARO, EXIOBASE, WIOD and Eora sector codes, synthetic codes)
pass `merchandise_mask`, for example from `puremacro.trade.mrio.goods_mask`.

Tariffs are ad-valorem rates `r >= 0`; domestic transactions are never
tariffed and residents' purchases abroad (X) are exempt. Multipliers below one
(import subsidies) are refused by `TariffWedges.from_arrays`, which keeps the
cost matrix nondecreasing in the tariff scale so that one existence
certificate at the full duty covers every continuation stage.
`build_tariff_wedges` takes a scalar, an `(N, S)` origin-by-sector table for
one importer, an `(N, N, S)` array, a mapping `{importer: {origin | "*": rate}}`
(retaliation; a named origin always overrides `"*"`, whatever the key order) or
a callable such as `TariffScenario.get_rate`. `wedges.to_dataframe()` lists
the nonzero rates by origin, importer and sector. Coverage is merchandise by
default (`coverage="all"` includes services); `fd_tariffed` selects which of
C, G, V carry the duty. Under `coverage="goods"` a sector-explicit
specification (an array with a sector axis, an `(S,)` array in a mapping, or a
callable) that places duties on service sectors is trimmed to merchandise with
a `RuntimeWarning`, and the trimmed sectors are recorded in
`wedges.metadata["dropped_service_rates"]`; scalar rates mean "every covered
sector" and are silent. Concordances and aggregated tables belong to
`puremacro.trade.mrio` (its tables enter through `from_io_table`).

## Two models, not one

| | `solve_trade_equilibrium` (consistent accounting) | `puremacro.trade.condensed` |
|---|---|---|
| Factors | labour and capital, Cobb-Douglas value added | one composite factor per country |
| Final demand | C, I, Cx shares of income; fixed foreign saving deducted from I | C, G, X Cobb-Douglas over absorption net of exogenous inventories V |
| Trade balance | fixed baseline foreign saving (numeraire units or a share of world factor income) | shares of world factor income (also world GDP or own GDP) |
| Numeraire | first producer price = 1 | world factor income (or any country's wage) |
| Unknowns | full system (Newton, sparse LU, Krylov, Keller PAC) | exact 2N condensation `(log w, Y / Y0)` |
| Tolerance and acceptance | residual and account ledger at the solver tolerance | `1e-12` residual, `1e-10` Walras, ten raw-flow blocks at `1e-10` |
| Welfare | Hicksian EV/CV with endpoint Shapley attribution | EV with an exact additive split, real national income, Fisher ToT |
| Baseline duties | observed flows are basic-price flows | `separate_baseline_tariffs` |

Results are not interchangeable across the two columns. Use the condensed
model when a certified one-factor Leontief benchmark with numeraire-free
measures is wanted; use consistent accounting when the two-factor structure or
the Hicksian policy tools are needed.

## What is and is not validated

Verified in this repository (`tests/test_trade_condensed.py`): benchmark
reproduction at `1e-14` for all closures and numeraires; certificates at or
below `1e-10` on a synthetic 3x4 table, the frozen OECD 3x3 fixture, a
synthetic 20x6 table and a synthetic 5x4 table; the certificate reacts to
`1e-5` perturbations of `w, p, y, Y, XN, TR` in the named block and to a
`1e-9` wage perturbation; certificates of natural-unit tables with one to six
empty (phantom) cells; EV-split additivity at `1e-12`; numeraire invariance of
eighteen measures and split components at `1e-9`; a duty on fixed inventory
quantities is neutral under the factor closure and not under the world-GDP
closure; homogeneity under scaling the table by 1000; an exact `10%` effective
rate for a uniform duty, also when inventories are untariffed; forward and
central Jacobians agree to `1e-6` relative; the PE bound lies below the GE
consumer price; continuation stages, stall detection, the nested fallback,
arclength monitors and landing, near-start multistart and the far-start
recovery on the 20x6 table; a fail-closed existence boundary on the synthetic
3x4 economy (a 500% goods duty by its first country passes the existence
gate, but the equilibrium with a positive wage in that country ceases to exist
near a 259% duty, `lambda* = 0.51773` of 500%: along the branch `lambda`
rises monotonically while the wage tends to zero; natural-parameter
continuation stalls at `lambda = 0.515625`, pseudo-arclength raises
`FoldDetected` where the Jacobian is numerically singular, and
`solve_condensed` ends in `EquilibriumNotFound` naming the stall); and the
five loader layouts of `puremacro.trade.data` on synthetic 3x3 tables.

Not validated: no global uniqueness, universal fold threshold or global Newton
guarantee is asserted. Numerical residual checks do not establish global
existence, uniqueness or optimality of an equilibrium, and the certificate
thresholds are floating-point verification criteria, not interval root
enclosures. Arclength and multistart are evidence, not proofs. No turning point
has been exhibited on an economic example: the fold monitor is exercised by a
synthetic determinant sign change and by the existence boundary above, so
`FoldDetected` means that the path cannot be continued regularly (a turning
point or an existence boundary), and the boundary is not a universal fold
threshold. Static EV includes investment in the absorption composite. Full
45-sector runs on the OECD tables use dense LU factorizations of
`M x M` matrices with `M` in the thousands and are not a browser target.

## References

- Dekle, R., Eaton, J., & Kortum, S. (2008). "Global Rebalancing with Gravity: Measuring the Burden of Adjustment." *IMF Staff Papers*, 55(3), 511–540.

Related: [consistent trade accounting](trade_accounting.md), [Hicksian consumption
welfare](trade_welfare.md), [Hicksian tariff policy](trade_policy.md),
[MRIO tables](trade_mrio.md),
[structural validation status](STRUCTURAL_VALIDATION_STATUS.md).
