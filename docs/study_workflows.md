> 🇬🇧 English · 🇪🇸 [Español](es/study_workflows.md)

# Runnable study and scenario templates

These development-source examples run offline with the standard dependencies.
They connect data or model declarations, estimation or solution, diagnostics,
and a portable report archive. They demonstrate methods and their assumptions;
they do not reproduce a published causal policy finding.

```bash
python -m puremacro.examples.study_workflows --output study_output
# Or select --workflow mexico, household or tariff.
```

The output directory must be new. Each subdirectory contains Markdown, LaTeX,
Typst, CSV, a manifest and `study.pmstudy`. Model scenarios also retain their
source as inert text. The archive includes the original inputs and named
evidence frames; [study recording](studies.md) explains the integrity checks.

## Mexican interest rates and inflation

`run_mexico` uses the already frozen course snapshots of OECD observations
distributed by FRED: [three-month interbank interest rates](https://fred.stlouisfed.org/series/IR3TIB01MXM156N)
and [year-on-year CPI inflation](https://fred.stlouisfed.org/series/CPALTT01MXM659N).
The packaged manifest records the source URLs, units and exact input hashes.
The original retrieval date was not recorded; these are not real-time vintages.
The common monthly window is explicitly January 2000–December 2023. No interior
missing month is deleted or interpolated.

The recursive VAR uses six lags, orders inflation before the rate, and reports
horizons 0–18 after a 0.25 percentage-point increase in the **annualized** rate.
Inflation is already year-on-year growth; no further log/difference/annualization
is applied. The ordering assumes inflation does not respond within the month to
the identified rate innovation. This is an identifying restriction; the result
is not an external-instrument or narrative-surprise estimate. The small two-variable
system omits other potentially relevant information and allows a price puzzle.
The 199 seeded residual-bootstrap draws use the fixed normalization described
in the study guide. No structural HANK parameters are silently fitted to this VAR.

```python
from puremacro.examples.study_workflows import run_mexico

result = run_mexico(n_boot=199)
print(result.to_frame().head())
print(result.audits)
```

## A household transfer written as a dynamic program

`run_household` loads the saving model from DP `.mod` declarations, solves EGM,
then evaluates a nonlinear 40-quarter transition from the stationary distribution.
A date-0 transfer of 0.05 consumption units per household is externally funded.
Prices and earnings risk stay fixed; there is no government financing or general
equilibrium response. This is an illustrative calibration, **not a survey fit**.

The report gives changes in consumption and entering assets, subtracting the
zero-shock path under identical terminal values. It retains both paths, the
transfer, initial distribution, policies, shock transition matrix, parameters
and stationary audit. Assets cannot jump on impact; subsequently, consumption
plus next-period assets exhaust current resources. Tests check this identity.
The audit preserves any grid/boundary failures; a small solve tolerance is not a
grid-accuracy certificate. Use the explicit [refinement API](result_audits_extended.md)
when changing grids or horizons. For survey calibration see the existing
[SCF Aiyagari example](micro_data.md#worked-example-an-aiyagari-economy-against-the-scf).

```python
from puremacro.examples.study_workflows import run_household

scenario = run_household(amount=0.05, horizon=40, n_a=100)
print(scenario.to_frame().head())
print(scenario.audits)
```

## Tariffs through a production network

`run_tariff` aggregates the bundled **OECD ICIO 2023 edition, reference year
2019** to USA/Mexico/rest and resources/manufacturing/other industries (including
utilities, construction and services). Every original
economy and activity belongs to a group; all intermediate, final-use and
value-added totals are conserved. Source hashes, grouping maps and the
calibration table are archived. Cite OECD (2023), OECD Inter-Country Input-Output
Tables, [OECD ICIO](https://www.oecd.org/en/data/datasets/inter-country-input-output-tables.html).
The bundled-data attribution and adaptation terms in
`puremacro/trade/_datafiles/SOURCES.md` also apply to redistributed outputs.

The illustrative scenario is a 5% US tariff on Mexican manufactures, with
lump-sum duty rebates and fixed foreign saving in numeraire units. The USA
resources producer price is fixed at one. Value added is split two-thirds labor,
one-third capital by assumption. The existing [accounting guide](trade_accounting.md)
states the technology, fiscal and closure conventions. Results are changes in
gross output at baseline prices, **not GDP, welfare or a policy forecast**.
Both equilibria must converge; their named accounting checks, full physical-equation
residuals, prices and outputs are retained. The equation maximum mixes native
units (quantities, relative prices, factors and nominal budgets); it is not a
single monetary amount. Parameter uncertainty and alternative closures are not estimated.

```python
from puremacro.examples.study_workflows import run_tariff

tariff = run_tariff(rate=0.05)
print(tariff.to_frame())
```

The empirical study and model scenarios stay separate. Comparing model and
empirical responses requires matching frequency, units, shock interpretation,
response convention and input sample, plus an explicit economic rationale.
