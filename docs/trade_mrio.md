> 🇬🇧 English · 🇪🇸 [Español](es/trade_mrio.md)

# Native MRIO tables: provenance, readers and aggregation

`puremacro.trade.mrio` reads published multi-regional input-output databases
at their native resolution, records where every number came from, balances
them under a stated contract and aggregates them exactly. It is the ingestion
layer that the trade solvers sit on: a table becomes a calibration matrix
through `to_calibration_matrix`, and a calibration comes back as a table
through `MRIOTable.from_trade_calibration`.

```python
from puremacro.trade import calibrate_trade_model
from puremacro.trade.data import synthetic_codes, synthetic_icio_table
from puremacro.trade.mrio import Concordance, MRIOTable, aggregate_mrio, to_calibration_matrix

# A seeded synthetic table (six countries, four sectors) and its table view.
countries, sectors = synthetic_codes(6, 4)
calib = calibrate_trade_model(synthetic_icio_table(6, 4, seed=2), country_codes=countries, sector_codes=sectors)
table = MRIOTable.from_trade_calibration(calib)
print(table.summary())
print(table.accounting_report().to_markdown())

# Keep C01 and C02, merge the rest; two sector groups.
blocs = Concordance.from_mapping(
    "blocs", {c: c if c in ("C01", "C02") else "REST" for c in table.country_codes})
two = Concordance.from_mapping(
    "two", {s: "GOODS" if k < 2 else "SERVICES" for k, s in enumerate(table.sector_codes)})
coarse = aggregate_mrio(table, two, region_concordance=blocs)
print(coarse.to_markdown("countries"))
print(coarse.metadata["aggregation"]["aggregated_identity_rel"])   # row-column identity after aggregation

matrix = to_calibration_matrix(coarse)                             # (9, 15): 3 countries x 2 sectors
coarse_calib = calibrate_trade_model(matrix, country_codes=coarse.country_codes,
                                     sector_codes=coarse.sector_codes)
```

The synthetic table is balanced, so the accounting report shows row, column
and output identities at rounding level, and aggregation conserves every total
(the aggregated identity residual is about `3e-16` relative). Native files are
read the same way:

```python
# requires: the native OECD ICIO 2023, Eurostat FIGARO 2026 and EXIOBASE 3.8.2 files (not shipped)
from puremacro.trade.mrio import aggregate_mrio, read_exiobase_native, read_figaro_native, read_oecd_native, regularize_table

oecd = read_oecd_native("2019_SML.csv", 2019)          # refuses the known-corrupted digest
figaro = read_figaro_native("matrix_eu-ic-io_ind-by-ind_26ed_2019.csv")
exio = read_exiobase_native("IOT_2019_ixi.zip")         # sparse Z, members checked against the archive CRCs
core = read_exiobase_native("exiobase_core", archive="IOT_2019_ixi.zip")  # extracted members checked too
balanced, report = regularize_table(oecd)
sections = aggregate_mrio(balanced, "isic_section")
print(report.to_markdown("gates"))
```

A two-country FIGARO file and a two-region EXIOBASE directory in the official
layouts ship under `tests/fixtures/trade/mrio/`, so
`read_figaro_native("tests/fixtures/trade/mrio/figaro_2c2s_26ed.csv")` runs
from a checkout (with a `RuntimeWarning` that the fixture is not the
registered release).

## The table contract

An `MRIOTable` is a frozen dataclass whose arrays are read-only copies. Cells
are country-major, `i = country_index * S + sector_index`.

| attribute | shape | meaning |
|---|---|---|
| `Z` | `(M, M)` dense or `scipy.sparse.csr_matrix` | basic-price intermediate deliveries; rows sell, columns buy |
| `F` | `(M, N, K)` | final deliveries by seller cell, destination country, category |
| `fd_codes` | `K` strings | subset of `("C", "G", "X", "V", "VAL")` in that order |
| `VA`, `TLS`, `output` | `(M,)` | value added as recorded, net taxes less subsidies on the buyer's intermediate purchases, gross output |
| `TFD` | `(N, K)` | net taxes less subsidies on final purchases |
| `production_taxes`, `labor_compensation`, `operating_surplus` | `(M,)` or `None` | factor detail where the source has it |
| `merchandise_mask` | `(S,)` bool or `None` | goods sectors (ISIC A-C) |
| `metadata` | dict | `sources` (tuple of `SourceRecord`), `transformations` (tuple of strings), conventions |

C is household, NPISH and government consumption; G is gross fixed capital
formation; X is residents' purchases abroad (the OECD DPABR basket); V is the
signed change in inventories; VAL is valuables. The validating constructor is
`MRIOTable.from_arrays`: `Z`, G and X must be nonnegative, everything must be
finite, and labour and operating surplus are supplied together. Negative
consumption, inventories, valuables, value added and taxes are real source
data and are kept. `accounting_report()` recomputes the row and column
identities independently and counts the negative entries; nothing is
rebalanced there. `net_exports()` measures fob balances over all uses and
assigns the floating-point round-off of their world sum to
`reference_country`. `to_raw("cix")` gives a `RawIOData` with the processed
C, I = G + V + VAL, Cx = X columns.

## Provenance

`identify_source(path, expected_sha256=..., expected_md5=...)` streams the
file in 4 MiB blocks and refuses an absent or changed file.
`verify_zip_member(archive, member, extracted_path=None)` checks a member's
size and CRC-32 against the archive directory. Every reader stores the
resulting `SourceRecord`s in `metadata["sources"]`, with a `status` of
`"verified"`, `"whitelisted"`, `"hashed"` or `"unknown_edition"`.

`check_oecd_source(path, year=...)` compares an OECD CSV (or the year member
of the 2016-2020 ZIP) with two registries:

| registry | digest | meaning |
|---|---|---|
| `OECD_ICIO_MD5[2019]` | `28cba31491177955445051d459053744` | accepted `2019_SML.csv`, OECD ICIO 2023 edition |
| `OECD_ICIO_MD5[2020]` | `d3e0f4979d85d6c0bb7cf4c43e324287` | accepted `2020_SML.csv` |
| `OECD_KNOWN_CORRUPTED_MD5` | `d1b887aaafa54ab3f28fde78fcd21cdf` | a copy of the 2020 SML file with lost decimal points: tokens with 3-4 decimals lost their decimal point and tokens below 0.001 became 0; refused |

A corrupted digest raises `MRIOIntegrityError`; a whitelisted digest is
recorded; anything else is read with a `RuntimeWarning` naming it an
unauthenticated edition. The FIGARO reader compares the CSV with
`FIGARO_2026ED_2019_SHA256` and the EXIOBASE reader compares the archive with
`EXIOBASE_382_2019_IXI_SHA256` in the same way; both enforce an
`expected_sha256` when one is passed, and a registered file read for a year
other than 2019 raises. EXIOBASE members read from the archive are checked
against its CRC-32 entries. An extracted EXIOBASE directory is authenticated
only when its archive is passed (`archive=`): the archive is hashed and every
extracted member is checked against the archive entry's size and CRC-32, so
one changed byte raises. Read alone, a directory's members are hashed,
recorded as `"unknown_edition"` and flagged with a `RuntimeWarning`;
`metadata["source_authenticated"]` says which case applied. The readers never
download data and never generate a substitute fixture (the OECD archive
download helper of [the bundled-table tools](spatial_and_trade_ge.md) is a
separate, explicit step).

## Readers and source conventions

| | OECD ICIO 2023 edition | Eurostat FIGARO 2026 edition | EXIOBASE 3.8.2 ixi |
|---|---|---|---|
| reader | `read_oecd_native(path, year)` | `read_figaro_native(path, year=2019)` | `read_exiobase_native(path_or_zip, year=2019, archive=None)` |
| layout | 77 economies x 45 industries SML CSV with TLS/VA/OUT rows | `rowLabels` header, `{region}_{industry}` cells, five final-use columns per region, six trailing `W2_*` rows; dimensions derived from the labels | `Z.txt`, `Y.txt`, `x.txt`, `industries.txt`, `satellite/F.txt`, `satellite/F_Y.txt` with two-level region/sector headers |
| units | USD million, current prices | EUR million, current basic prices | EUR million, current basic prices |
| final uses | C = HFCE + NPISH + GGFC, G = GFCF, X = DPABR, V = INVNT | C = P3_S13 + P3_S14 + P3_S15, G = P51G, X = 0, V = P5M | C = y01 + y02.a + y02.b, G = y04, X = 0, V = y05.a, VAL = y05.b |
| consumption | resident consumption with a separate DPABR basket; includes government | territorial consumption; includes government; `W2_OP_RES` and `W2_OP_NRES` totals kept in metadata | published consumption; includes government; no DPABR basket |
| inventories | INVNT is the signed change; no stock is observed | P5M merges inventory changes and valuables; the source does not separate them | inventories and valuables are separate flows |
| factor detail | none: the VA row has no observed labour/capital split | `W2_D1` labour, `W2_D29X39` other net production taxes, `W2_B2A3G` gross operating surplus and mixed income | three skill rows for labour, four operating-surplus rows (including land and resource rents), other net taxes on production |
| output | provider OUT column | row total of the published transactions | `x.txt` |
| year | 1995-2020 | 2019 authenticated | 2019, a nowcast |
| storage | dense | dense (`sparse_matrix=True` optional) | CSR streamed in 128-row blocks |

Region codes are mapped to ISO-3 through `FIGARO_ISO3` (`FIGW1` becomes
`ROW`) and `EXIOBASE_ISO3` (the five rest-of-world regions keep distinct
names). Merchandise masks follow `goods_mask`: ISIC sections A-C for the OECD
industries and the ISIC section groups, NACE A-C except C33 (repair and
installation, which is not a merchandise shipment) for FIGARO, industries
i01-i37 except i01.w for EXIOBASE.

The FIGARO reader applies, and records under
`metadata["rounding_adjustments"]`, two bounded rounding repairs: an empty
cell whose output is one rounding unit negative (`-0.001 <= y < 0` with
absolute input mass at most `0.002`) is zeroed, and a consumption entry in
`[-0.001, 0)` is moved to V so that row totals and country expenditure are
unchanged. Anything larger raises. `rounding_repair=False` keeps the published
rounding. The EXIOBASE reader refuses a nonzero `Exports: Total (fob)` column
and nonzero final-demand factor rows, because those would need an explicit
destination model.

## Regularization contract

`regularize_table(table)` balances a raw table with the build contract of
`puremacro.trade.condensed.table_from_arrays`, for any table with named
categories. With `out` the output column, `row` the row totals and
`active = out > 0`:

1. Identity guard: `median_{active} |row_i - out_i| / out_i <= 1e-5`, no
   negative output, and every zero-output cell is completely empty.
2. Phantoms: each zero-output cell receives `F[i, own country, C] = VA_i = 1e-6`
   so its row and column agree exactly (`inactive="mask"` keeps them empty).
3. Value-added floor: `VA_i = 1e-3 * out_i` only where active and `VA_i <= 0`.
4. Net product taxes are recomputed as the residual that closes every column,
   `tls_j = row_j - sum_i Z_ij - VA_j`. At cells that were not floored the
   change must satisfy `sum_j |tls_j - TLS_j| <= 1e-4 * sum_j |TLS_j|` and
   `max_j |tls_j - TLS_j| / max(out_j, 1) <= 1e-2` (with a `1e-9` relative
   slack on the second gate).
5. Trade balances are measured fob over all uses; `|sum_o xn_o| <= 1e-9 * world GDP`
   with `world GDP = sum VA + sum tls + sum TFD`.

The returned table has `output` equal to the row totals after phantoms, the
residual `TLS`, and the step appended to `metadata["transformations"]`; the
`MRIOBuildReport` carries the phantoms, floors, both residual-tax totals, the
world trade-balance round-off, world GDP and every gate with its value,
tolerance and pass flag, and names the offending cells where a gate is about
cells (`"cells"`, `"n_cells"`, and `"worst_cell"` for the per-cell gate).
Every tolerance is a keyword; `strict=False` returns the report with
`passed=False` instead of raising (the per-cell gate is tight, and a published
release can sit at or just above it). Such a table is marked
`metadata["regularized"] = False` with the failed gates in
`metadata["failed_gates"]` and in its transformation string, and
`aggregate_mrio` and `to_calibration_matrix` warn when they receive it. Under
`strict=False` only empty cells with output exactly zero receive a phantom;
cells with negative output or stray entries keep their data and are listed in
`report.skipped_phantoms`. The phantom `1e-6` and the `max(out, 1)` scale of
the per-cell gate are absolute amounts that assume a table in millions of
currency units, so rescale `phantom` for a table in other units.
`spectral=True` adds the Collatz-Wielandt bounds of `Z / output` from
`puremacro.trade.regularize.compute_spectral_radius`. This is a different
contract from `regularize_mrio_table`, which floors every active cell's value
added at `max(1e-3 Y, min(1, 0.5 Y))` (4.3.0: `max(1e-3 Y, 1)`, which lifted
value added above output on cells below 1 M USD) and re-derives output from row
sales before the residual unless `Y` is supplied; the two are not
interchangeable.

## Aggregation

`aggregate_mrio(table, concordance)` computes the exact aggregation with
`P = I_N kron C^T`, where `C` is the `(S, G)` 0/1 matrix of a `Concordance`:

```
Z_G = P Z P^T,   F_G[:, n, k] = P F[:, n, k],   VA_G = P VA,   TLS_G = P TLS,   output_G = P output
```

`TFD` is unchanged, factor detail is summed the same way, and a
`region_concordance` merges origins and destinations as well. Both accounting
identities are linear, so aggregation commutes with the residual-tax closure
whenever no floor applies, and every total is conserved to floating-point
rounding. When the fine table is balanced the coarse table must be too
(`max_j |row_j - col_j| / max(|col_j|, 1) <= 1e-8`); the residual is recorded
in `metadata["aggregation"]` either way. A coarse group is merchandise only
when every member is; mixed groups are listed.

One named map of the 45 OECD industries is registered in `CONCORDANCES`:

| name | groups | goods groups |
|---|---|---|
| `"isic_section"` | `ISIC_SECTION_CODES`: the 20 ISIC Rev.4 sections A-T | A, B, C |

Any other grouping is a `Concordance` built with `Concordance.from_mapping`,
`Concordance.build` or `Concordance.identity`, or a plain mapping passed to
`aggregate_mrio`; `compose` chains two maps. `goods_mask` decides the
merchandise mask label by label, so any order or subset of a registry gives
the same answer per code, and a code outside the named registry raises
`ValueError`.

## Bridges to the calibration

`to_calibration_matrix(table)` returns the `(M + 3, M + 3N)` matrix for
`calibrate_trade_model`: rows `0..M-1` are `[Z | F]` with C, I = G + V + VAL
and Cx = X per destination, row `M` is TLS with the final taxes condensed the
same way, rows `M + 1` and `M + 2` are labour and capital.
`calibrate_trade_model` reconstructs each column from a Cobb-Douglas split
and accepts it only when both factors are strictly positive (a capital share
strictly between 0 and 1) or both are zero, so the factor rows are built to
that standard:

* Other net production taxes move to the TLS row by default
  (`production_taxes="to_tls"`), which the calibration models as an output
  tax, except at cells where they are at least as large as value added. There
  they stay in value added and a `RuntimeWarning` names the cells (published
  FIGARO and EXIOBASE tables have such cells). `"to_factors"` keeps every
  production tax inside value added.
* Labour and capital split the remaining factor value added in the observed
  proportions where both are strictly positive, and `labor_share`
  (2/3-1/3 by default) elsewhere: phantom cells, cells with a negative factor,
  and cells with one factor exactly zero. Without detail every cell uses the
  default split.
* A cell with negative factor value added raises `MRIOIntegrityError`;
  `regularize_table` floors value added where it is not positive.

`return_details=True` also returns the cells on each fallback. A raw table
triggers a `RuntimeWarning` recommending `regularize_table` first, and so does
a table whose regularization gates failed.
`MRIOTable.from_trade_calibration(calib)` is the inverse bridge from
`calib.data_calibra`. It reads the calibration's Cx column as X
(tariff-exempt residents' purchases abroad) only for a table without a
recorded final-use mapping and for the OECD `DPABR` mapping
(`calib.metadata["final_use_mapping"]`); the FIGARO, WIOD and Eora loaders'
Cx (signed inventories and valuables) becomes V, and EXIOBASE's Cx, which
includes its export column, raises `ValueError` unless `cx_category` names
the category. A negative investment cell moved by
`negative_investment="to_inventory"` is added to V. A table bridged from a
FIGARO, WIOD or Eora calibration therefore has no X, and
`to_calibration_matrix` returns it with an empty Cx column, which the
consistent accounting treats as an inactive category.

## What is and is not validated

`tests/test_trade_mrio.py` checks, on hand-designed fixtures with
hand-derived expected arrays: the provenance helpers against `hashlib` and
corrupted or mismatched members; concordance algebra (projection equals the
explicit Kronecker product, composition, coarse masks); the table contract;
the FIGARO and EXIOBASE readers, including the rounding repairs, the zip,
archive-checked directory and chunked paths, a changed byte in an extracted
member, and every rejection; the OECD reader on a tiny SML-layout file; every
regularization gate in strict and reporting mode, and that a failed gate never
yields a table marked regularized; conservation, composition, region
aggregation and the 45-industry to ISIC-section goods consistency;
calibration of the aggregated OECD fixture with a baseline solve at prices of
one; calibration of tables with zero-factor cells and with production taxes
above value added; and the round trip through a synthetic calibration.

Not validated, and not claimed:

* Full-size general equilibrium on FIGARO or EXIOBASE. This module promises
  native-resolution tables, not native-resolution equilibria; EXIOBASE is not
  calibrated here at all, because `to_calibration_matrix` densifies `Z`.
* Comparability of welfare across databases. FIGARO keeps territorial
  consumption, and a residence bridge has not been validated; the EXIOBASE
  2019 release is a nowcast, and no stronger measurement claim is implied;
  `active_mask` means positive reported output, not an economic feasibility
  certificate.
* Other editions and years. The digests identify specific 2019 files; a
  different edition is read with a warning and no claim.
* A GTAP reader. The GTAP data are licensed and cannot ship even as fixtures,
  and the one complete open-source HAR parser is GPL-3 licensed, so it cannot
  be redistributed under puremacro's MIT license.
* The per-cell residual-tax gate is brittle by construction. Its tolerance is
  a keyword and the report is returned in either case.

## Related documents

* [Consistent trade accounting](trade_accounting.md) for the solver the
  calibration matrix feeds.
* [Hicksian consumption welfare](trade_welfare.md): native OECD C includes
  government consumption.
* [Condensed one-factor Leontief model](trade_condensed.md), which shares the
  regularization contract.
* [Quantitative spatial and trade general equilibrium](spatial_and_trade_ge.md)
  for the bundled table and the harmonized adapters.
* [Structural validation status](STRUCTURAL_VALIDATION_STATUS.md).
