> 🇬🇧 English · 🇪🇸 [Español](es/tariff_validation.md)

# Why the tariff brief has positive consumption EV

The [default brief](first_brief_tariffs.md) gives US consumption EV of
**+6,427.19 million USD (+0.0383%)**. A separate Leontief equilibrium solver
reproduces it within 0.000001 million USD. It is a consumption result in a
static model with investment as another final use.

| US final use | Quantity change valued at baseline purchaser prices, million USD |
|---|---:|
| Consumption, C | +6,427.19 |
| Investment, I | −7,095.64 |
| Residents' purchases abroad, Cx | −466.70 |

Only the C row is consumption EV. Adding these rows does not produce an
intertemporal or social welfare measure: future returns to investment are not
modeled, and the table's C aggregate includes government and NPISH consumption.

## Budget and resource explanation

The model allocates fixed expenditure shares to final uses. In the default
experiment US nominal disposable income, including fiscal rebates, increases
0.0660%, while its consumption price increases 0.0277%. Consumption therefore
increases 0.0383%. Investment and purchases abroad face larger price increases
(0.1972% and 0.3304%); their quantities fall 0.1417% and 0.2635%. Investment
spending also includes the prescribed foreign-balance adjustment. These are
equilibrium budget identities, not separately identified causal contributions.

For fixed sourcing, let A be intermediate input coefficients, F final basket
coefficients and v unit factor-resource requirements at baseline factor prices.
Then factor-resource accounting prices satisfy

$$p^R=(I-A^T)^{-1}v,\qquad P^R=F^T p^R.$$

Fixed factor endowments imply $\sum_j P_j^R\Delta c_j=0$. The independent
calculation gives world changes of **+6,761.06** for C, **−6,493.69** for I and
**−267.37** for Cx, in million baseline factor-value units; their sum is less
than 0.000001 in magnitude. These tax-free resource values differ from consumer
prices and are not welfare weights. The world consumption increase uses
resources released by the other final categories.

The positive US C result persists when both equilibria are recomputed without
domestic taxes (+6,500.05 million), with zero foreign balances (+5,892.06), or
with both changes (+5,923.46). Neither domestic-tax distortions nor nonzero
foreign balances alone explains the sign. These are separate model closures,
not an additive decomposition of the headline.

## Independent checks and scope

`tools/reference_validation/tariff_reference.py` shares calibration arrays
only. Conditional on country wages, it solves separate price, quantity and
rebated-income linear systems, then finds relative wages from factor-market
clearing. It does not call the production equilibrium, residual, accounting,
Jacobian or welfare helpers. State comparisons pass the registered scaled
`1e-8` tolerance, and US EV agrees within `1e-5` million USD.

A separate evaluator reconstructs goods flows, zero profit, labor and capital
use, purchaser prices, household and government budgets, external balances,
and expenditure-function EV for both `sigma=0` and `sigma=2`. Maximum normalized
equation errors are below `7e-11`; EV errors are below `1.5e-8` million USD.
This independently checks the `sigma=2` solution's equations, without claiming
a second nonlinear equilibrium solver for that variant.

The default tariff applies to **all 20 modeled sectors, including services**.
The alternative `sigma=2` composite substitutes across **all country-sector
intermediate inputs**, including different sectors; it is not restricted to
origins within the same input sector. Final-use baskets remain fixed. Its US C
gain is +23,497.93 million, alongside I of −2,840.78 and Cx of +1,470.69 at
baseline purchaser prices. The fixed-A resource identity above is specific to
`sigma=0` and is not applied to the changing-input technology.

Frozen inputs, source hashes, tolerances, state errors and closure comparisons
are in `tests/fixtures/first_brief_tariffs/independent_validation.json`.
Regenerate from the development checkout:

```bash
python -m tools.reference_validation.check_tariff_welfare --output /tmp/tariff-validation.json
```

No retaliation, transition dynamics or estimated substitution elasticities are
included. The numerical sign is validated for this stated model and closure;
it does not establish that the tariff benefits real-world households.
