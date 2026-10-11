puremacro 4.8.0 connects cross-country macro data to a reproducible offline forecasting study and preserves panel provenance when saving portable files. It includes the panel builders recorded under 4.7.0 in the changelog; 4.7.0 was not published to PyPI, so these are also new for users upgrading from 4.6.0.

**Publication status:** this candidate was not published. The Windows release gates found a cartridge verification bug; the upload was blocked. The unchanged `v4.8.0` tag is retained as evidence. Release preparation continues as 4.8.1.

- **Cross-country data:** annual, quarterly and monthly panel builders for WDI, OECD, Eurostat, IMF, BIS, ILOSTAT, Penn World Table, Maddison, and rates/yields sources. Panels share a `(code, date)` index and record sources, units and missing-data information. `splice_sources` combines histories by explicit precedence and records accepted and rejected seams.
- **A complete forecasting application:** frozen WDI GDP levels for the United States, Mexico and Brazil (1960–2024), with fixed expanding-window zero-growth, historical-mean and AR(1) forecasts for 2000–2024. One command exports all 225 forecasts, seven CSV tables, a figure, a portable `.pmz` cartridge and a checksummed evidence manifest. The package includes the frozen data; running the study requires no provider access.
- **Portable provenance:** `runtime.store` preserves supported nested `DataFrame.attrs` without pickle. New cartridge checksums cover metadata as well as observations. Unsupported metadata types and cycles fail explicitly instead of silently losing provenance.
- **Numerical and workflow fixes:** a more stable Kalman smoother for augmented DSGE states, validation of annual depreciation rates in `qna_capital`, faster storage of repeated vintage dates, corrected provider readers, and safer, more accessible Colab offload cards. The release also includes the equilibrated Newton trade solver and the documented limits of clean-table tariff scenarios.

After PyPI publication completes, install and run the study:

```bash
python -m pip install "puremacro==4.8.0"
python -m puremacro.examples.cross_country_forecasting --output research_output/cross_country_forecasting
```

Read `report.md` and `accuracy.csv` in the output directory. This is a **latest-vintage historical evaluation**, using revised data captured in October 2026. It does not reconstruct the information available at each forecast origin or establish real-time performance, causal effects or general model superiority.

**Storage compatibility:** newly written portable frames use schema v2 and require puremacro 4.8.0 or later to read. Existing v1 files need no conversion and old cartridges remain verifiable, although v1 did not retain frame attributes. Upgrade every reader before exchanging new v2 files.

Guides: [English](https://jalonso1979.github.io/puremacro/docs/cross_country_forecasting/) · [Español](https://jalonso1979.github.io/puremacro/docs/es/cross_country_forecasting/). See the [complete changelog](https://github.com/jalonso1979/puremacro/blob/v4.8.0/CHANGELOG.md) for provider-specific changes and numerical validation limits.

Validation: all five local release gates passed, including 19,010 passing tests with no unexpected failures or setup errors (140 skipped, 216 deselected and 276 expected failures). The final wheel reproduced all seven study tables outside the checkout. In Node-hosted Pyodide 0.28.3, it passed 83 provenance tests and completed the full application. The [tagged release workflow](https://github.com/jalonso1979/puremacro/actions/runs/38111212258) runs the platform matrix and opt-in suites before PyPI publication.
