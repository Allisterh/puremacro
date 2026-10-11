# puremacro 4.8.1 release evidence

Candidate commit: `580d2e3d94a3d5ae2a464138943a720ca7598f09`.

The unchanged public `v4.8.0` tag was blocked by Windows CI and never uploaded
to PyPI. See the [prior release evidence](../2026-10-10-release-4.8.0/README.md).
This candidate separates original archive integrity checks from checks for
changes to loaded frames, avoiding platform-specific ZIP container bytes.
[Diagnosis and regression coverage](cartridge-portability-triage.md).

## Validation

- 192 targeted tests passed; the fix also passed independent review.
- Strict documentation build passed.
- Clean wheel and sdist built; metadata checks passed. All 901 wheel package
  entries match the clean export, with required frozen data and no PNGs.
  [Artifact checks](artifact-check.json).
- Installed wheel reproduced all seven CSVs byte for byte outside the checkout,
  with verified cartridge and provenance. The validation patched
  `socket.create_connection` with a positive control; it was not an OS-level
  network sandbox. [Wheel proof](wheel-validation.json).
- The regenerated study retains all seven 4.8.0 numerical tables exactly;
  an independent raw-JSON/scipy oracle confirms 195 levels and 225 forecasts.
  [Study comparison](study-comparison.json), [independent validation](validation.json).
- Actual Node-hosted Pyodide 0.28.3 passed all 93 provenance tests and the
  complete study, including its plot and cartridge. Maximum accuracy difference
  from CPython: 4.44e-16. [Pyodide proof](pyodide-result.json).
- **All five release gates passed on the exact tagged commit:** 19,020 passed,
  140 skipped, 216 deselected, 276 expected failures, no failures or errors,
  in 1,905.80 seconds. The known-failures allowlist is empty.
  [Full gate log](release-gate.log).

The clean source export excludes unrelated notebook and raw-data work.
See [environment.json](environment.json).

## Publication

Published to PyPI on 2026-10-11 at 20:27 UTC: [puremacro 4.8.1](https://pypi.org/project/puremacro/4.8.1/).
The [release workflow](https://github.com/jalonso1979/puremacro/actions/runs/38157855111) (attempt 2) passed
the nine-target platform matrix and the opt-in suite on the tagged commit before trusted publishing uploaded
the package; `twine check` passed on both files. [Run record](github-publication-run.json).

- `puremacro-4.8.1-py3-none-any.whl`, 21,153,373 bytes,
  SHA256 `0b785039a585151c3bd23f55b8153ae9aff37cc1e30d32c97d10e130dad2bc3d`
- `puremacro-4.8.1.tar.gz`, 22,522,850 bytes,
  SHA256 `b8a917f832e0c30645fa73e8c05576620f4f302a6738cc356d252a805a08b2c1`

The public wheel differs from the locally built one in [artifact-check.json](artifact-check.json) in archive
bytes only: the sizes match and all 901 package members hash identically; ZIP member metadata differs between
the two builds.

### Verification from a fresh PyPI installation

In an empty Python 3.12 virtual environment, `pip install puremacro==4.8.1` from pypi.org resolved numpy 2.5.4,
pandas 3.0.6, scipy 1.18.1 and matplotlib 3.11.2 ([install log](pypi-install.log)). The downloaded wheel
([download log](pypi-download.log)) was compared with the tagged commit: every one of the 901 package members is
byte-identical to the tagged source and to the installed files, all 821 Python sources are present, and pip's
installation report carries the same wheel digest.
[Public artifact proof](public-artifact-validation.json), [log](public-artifact-validation.log).

The installed package then re-ran the study from a separate working directory with `socket.create_connection`
blocked (positive control included): 225 forecasts, verified cartridge, and all seven CSV tables byte-identical
to the local-wheel proof and therefore to the 4.8.0 study.
[Fresh-install study proof](pypi-wheel-validation.json), [log](pypi-wheel-validation.log).

GitHub release [v4.8.1](https://github.com/jalonso1979/puremacro/releases/tag/v4.8.1) was published at
20:31 UTC with [these notes](github-release-notes.md). The bilingual guides deploy from `main`, which carries
the 4.8.1 line merged with the upstream fixes (85f90b87), through the Pages workflow.

### Workflow interruption

Release attempt 1 was cancelled at 2026-10-11T18:02:32Z by the maintainer, from a parallel session, to hold
publication pending a scoping decision; no test had failed. Four complete platform jobs and all nine early
cartridge checks had passed. Attempt 2 re-ran the unfinished jobs on the same unchanged tag, and the maintainer
then confirmed publication of 4.8.1 as it is. [Attempt 1 record](github-attempt-1.json).
