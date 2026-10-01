# Third-party data bundled with `puremacro.trade`

puremacro's own code is MIT-licensed (see `LICENSE` at the repository root).
**That licence does not cover the data files in this directory.** They are
third-party material redistributed with puremacro so that the trade model and
its parity suites run without asking you for a file. Their terms are below.

If you publish results computed from these files, cite the original source,
not puremacro.

---

## `icio_77c_11s.npz` — OECD Inter-Country Input-Output tables (regression fixture)

| | |
|---|---|
| Array | `data`, shape `(850, 1078)`, float64 |
| Origin | OECD Inter-Country Input-Output (ICIO) tables |
| Publisher | Organisation for Economic Co-operation and Development (OECD) |
| Source page | <https://www.oecd.org/sti/ind/inter-country-input-output-tables.htm> |
| Coverage | 77 countries × 11 composite sectors, 3 final-demand categories |
| Terms | Redistributed under the OECD terms of use, which permit reuse and redistribution with attribution to the OECD as the source |

**Attribution.** Work using this matrix should credit the OECD ICIO tables.
The OECD is not affiliated with puremacro and does not endorse it.

**Provenance: a regression fixture, not OECD estimates.** The array is a
bit-exact copy of MATLAB `data_77c_11s.mat`, which the original
sectoral-misallocation pipeline built from a `data_2020_SML.csv` export of the
OECD ICIO tables. That export is corrupted (MD5
`d1b887aaafa54ab3f28fde78fcd21cdf`): tokens with three or four decimals lost
their decimal point and tokens below 0.001 became zero. World value added in
the 77x11 aggregation is 7.05e11 USD million, against 7.97e7 in the clean OECD
2020 release. The file is kept unchanged as the fixture of the MATLAB parity
suites, which compare puremacro with MATLAB solutions of this same table. Do
not report outputs computed from it as OECD-based estimates. For empirical
work, read a clean OECD release with
`puremacro.trade.data.load_oecd_icio_granular` or
`puremacro.trade.mrio.read_oecd_native`. The same record is available in code
as `puremacro.trade.data.BUNDLED_ICIO_PROVENANCE`, and
`load_icio_data(return_structured=True).metadata` carries it. See the
2026-09-22 entry of `docs/ADVISORY.md`. `MANIFEST.json` predates this note: it
still names only the OECD origin, and its SHA-256 is unchanged.

**What was changed.** That export of the OECD tables was aggregated to the
77-country, 11-sector layout used by the sectoral-misallocation computation,
then converted from MATLAB `data_77c_11s.mat` to a compressed `.npz`. The
conversion is bit-exact: `numpy.array_equal` against the MATLAB array was
asserted when the file was written, and `tests/test_no_matlab_dependency.py`
checks the file still loads from the installed package.

---

## `trade_reference_solutions.npz` and `trade_results_workbook.npz` — reference outputs

| | |
|---|---|
| Origin | MATLAB `results_77c_11s_*.mat` and `results.xls` of the sectoral-misallocation computation |
| Author | Jorge Alonso-Ortiz |
| Terms | Same MIT licence as puremacro's own code |

These are the equilibrium arrays, calibration arrays and workbook sheets that
puremacro's trade parity suites compare against. They are verbatim copies,
dtype included, so the comparison remains an **external** check rather than
puremacro grading its own output. Regenerating them with puremacro would turn
every one of those tests into a tautology.

`results_77c_11s_t10_54.mat` could not be included: the source file is a
dataless placeholder that reads zero bytes. Tests needing that scenario skip
by name rather than substitute a value.

---

## Machine-readable metadata

`MANIFEST.json`, `REFERENCE_MANIFEST.json` and `WORKBOOK_MANIFEST.json` in this
directory record each file's arrays, shapes, provenance and SHA-256.
