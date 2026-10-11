# Financial panel baseline failure

The failure was reproduced independently from the clean export of commit
`4e4b80fb` at `/tmp/puremacro-release-4.8.0-source`:

```bash
env MPLCONFIGDIR=/tmp/puremacro-release-mpl XDG_CACHE_HOME=/tmp/puremacro-release-cache PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/puremacro-release-4.8.0-venv/bin/python -m pytest tests/test_fetch/test_modular_panels.py::TestFinancialPanel::test_temporal_boundary_filters -q --tb=short -o cache_dir=/tmp/financial-triage-cache
```

Result: **1 failed, 4 warnings in 1.89 s**. At test line 365,
`assert not panel.empty` failed. Warnings showed attempts to read the HTTP
cache and download the World Bank Pink Sheet. The same failure occurred with
only this test selected, so suite ordering was not required to trigger it.

The test requested 2016–2018, although the frozen financial CSVs and commodity
workbook begin in 2020. Its nonempty expectation was incorrect. The fixture
patched `puremacro.fetch._http.cached_get`, but Pink Sheet had moved to its own
`cached_get` seam backed by the shared urllib transport. That missed patch
allowed nominally offline tests to read live or cached commodity observations,
which could conceal the incorrect date expectation.

Commit `29d187e9` fixes the test, without changing financial panel production
code. The boundary check now requires exactly 2020Q2 and 2020Q3, and separately
requires the unsupported 2016–2018 window to be empty. The fixture patches the
current Pink Sheet seam, resolves current module objects with `importlib`, and
blocks both urllib and requests transports. Teardown checks detect attempts
even when a fetcher catches the exception; a mechanism-control test verifies
that both blockers activate.

Validation ran from the original repository with the release interpreter:

```bash
env MPLCONFIGDIR=/tmp/puremacro-release-mpl XDG_CACHE_HOME=/tmp/puremacro-release-cache PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/puremacro-release-4.8.0-venv/bin/python -m pytest tests/test_fetch/test_financial.py tests/test_wb_pink_sheet.py tests/test_fetch/test_modular_panels.py tests/test_test_quality.py -q --tb=short -o cache_dir=/tmp/financial-triage-cache
```

Result: **91 passed, 19 warnings in 63.84 s**. Eighteen warnings correctly
identified the intentionally frozen commodity workbook as stale; one came
from the explicit Pink Sheet download-failure test. There were no unmocked
transport attempts. The clean export was not edited, and its baseline pytest
cache was not used. This targeted result is separate from the subsequent full
release-baseline rerun.
