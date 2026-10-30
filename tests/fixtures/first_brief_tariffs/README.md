# Tariff brief evidence

`headline.json` and `manifest.json` are generated from actual native solves by
`python -m tools.reference_validation.refresh_tariff_brief --output /tmp/tariff-brief`.
This also updates both documentation pages and figures. The manifest records
the environment, original bundled OECD file hash, solver and output hashes.

`independent_validation.json` comes from
`python -m tools.reference_validation.check_tariff_welfare --output tests/fixtures/first_brief_tariffs/independent_validation.json`.
It records source hashes, the float64 table-array hash, equation errors,
independent Leontief state comparisons, all final-use changes and four closure
comparisons. Its evaluator never calls production solver/accounting/welfare
helpers; the runner calls the production solver only to obtain states to check.

Registered tolerances: EV 1e-5 million USD, normalized equilibrium 1e-9,
scaled independent-state errors 1e-8, global factor-resource balance 1e-5 million.
The resource identity is limited to fixed sourcing. I/Cx changes are quantity
values, not components of consumption or intertemporal welfare. The bundled
2019 data are an OECD adaptation, not contemporary tariff-outcome observations.
