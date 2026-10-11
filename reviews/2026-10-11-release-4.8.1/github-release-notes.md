puremacro 4.8.1 connects cross-country macro data to a reproducible offline forecasting study and preserves panel provenance when saving portable files. It includes the features recorded under 4.7.0 and 4.8.0 in the changelog; neither version was published to PyPI, so these are new for users upgrading from 4.6.0.

**Publication status:** preparation and validation of 4.8.1 are in progress. The 4.8.0 upload was blocked by legacy v1 cartridge verification failures on Windows. Its tag is unchanged; the correction is being released as 4.8.1. PyPI publication and the subsequent installation check remain pending.

- **Cross-country data:** annual, quarterly and monthly panel builders for WDI, OECD, Eurostat, IMF, BIS, ILOSTAT, Penn World Table, Maddison, and rates/yields sources. Panels share a `(code, date)` index and record sources, units and missing-data information. `splice_sources` combines histories by explicit precedence and records accepted and rejected seams.
- **A complete forecasting application:** frozen WDI GDP levels for the United States, Mexico and Brazil (1960–2024), with fixed expanding-window zero-growth, historical-mean and AR(1) forecasts for 2000–2024. One command exports all 225 forecasts, seven CSV tables, a figure, a portable `.pmz` cartridge and a checksummed evidence manifest. The package includes the frozen data; running the study requires no provider access.
- **Portable provenance:** `runtime.store` preserves supported nested `DataFrame.attrs` without pickle. New cartridge checksums cover metadata as well as observations. Unsupported metadata types and cycles fail explicitly instead of silently losing provenance. The 4.8.1 patch fixes verification of v1 and v2 cartridges created on another operating system: ZIP creator-OS metadata could cause a false checksum mismatch. Corruption and in-memory mutation detection remain enabled.
- **Numerical and workflow fixes:** a more stable Kalman smoother for augmented DSGE states, validation of annual depreciation rates in `qna_capital`, faster storage of repeated vintage dates, corrected provider readers, and safer, more accessible Colab offload cards. The release also includes the equilibrated Newton trade solver and the documented limits of clean-table tariff scenarios.

After PyPI publication completes, install and run the study:

```bash
python -m pip install "puremacro==4.8.1"
python -m puremacro.examples.cross_country_forecasting --output research_output/cross_country_forecasting
```

Read `report.md` and `accuracy.csv` in the output directory. This is a **latest-vintage historical evaluation**, using revised data captured in October 2026. It does not reconstruct the information available at each forecast origin or establish real-time performance, causal effects or general model superiority.

**Storage compatibility:** newly written portable frames use schema v2. Use puremacro 4.8.1 or later for reading and verification across supported platforms. Existing v1 files need no conversion and old cartridges remain verifiable, although v1 did not retain frame attributes. Upgrade every reader before exchanging new v2 files.

Guides: [English](https://jalonso1979.github.io/puremacro/docs/cross_country_forecasting/) · [Español](https://jalonso1979.github.io/puremacro/docs/es/cross_country_forecasting/). See the [complete changelog](https://github.com/jalonso1979/puremacro/blob/v4.8.1/CHANGELOG.md) for provider-specific changes and numerical validation limits.

Validation results for the final 4.8.1 commit and public package will be recorded after the checks complete. The 4.8.0 local results are retained separately and are not presented as validation of this patch.
