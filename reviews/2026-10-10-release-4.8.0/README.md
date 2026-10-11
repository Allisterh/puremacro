# puremacro 4.8.0 release evidence

Tagged commit: `29d187e95b0e56e5ed4f7a838f4d5f2b832a7d1d` (`v4.8.0`), including
local cross-country panel work and remote main through `de972072`.
The full release gate, archive builds and installed-wheel replays use a clean
`git archive` export; pre-existing notebook and raw-data work is excluded from
the release. See [environment.json](environment.json).

## Completed local checks

**All five release gates passed on the release commit: 19,010 passed,
140 skipped, 216 deselected and 276 expected failures, with zero unexpected
failures or setup errors, in 1,978.99 seconds. The known-failures allowlist remains
empty.**

- 185 provenance, cartridge, runtime and forecasting tests passed.
- 106 upstream smoother, vintage-cache, capital and Colab tests passed; one
  opt-in test was deselected.
- All four fast release gates passed: static Pyodide contract, public API
  snapshot, five-file version agreement, and Python 3.11 syntax in 1,705 files.
- Strict MkDocs build passed.
- Clean sdist and wheel built successfully; `twine check` passed for both.
  Archive inspection confirmed the frozen study data and required source-replay
  fixtures, with no PNG files in the wheel. [Artifact checks](artifact-check.json).
- Installed wheel, separate working directory and HTTP connections blocked:
  all seven CSV tables match the canonical study byte for byte, with cartridge
  verification and metadata preserved. [Wheel replay](wheel-validation.log).
- Actual Node-hosted Pyodide 0.28.3: 83 provenance tests passed; complete study
  produced 225 forecasts, a plot and verified cartridge. Maximum accuracy
  difference from CPython was 4.44e-16. [Pyodide log](pyodide.log).
- Independent numerical validation checked 195 data levels and 225 forecasts;
  see the [research evidence](../2026-10-10-cross-country-forecasting/validation.json).

The full baseline is recorded in [release-gate.log](release-gate.log).
The dated research directory also retains initial development evidence; the
release evidence here distinguishes final 4.8.0 checks from those earlier runs.

## Initial baseline diagnosis

The first full run completed 18,974 passing tests but could not pass the release
gate: the sandbox blocked local HTTP-server and Jupyter-kernel ports, and one
financial-panel test requested dates preceding every frozen fixture. The
financial test also bypassed the current Pink Sheet mock and could read live
data. The test now uses the frozen 2020 window, checks an earlier empty window,
and rejects accidental network access. There were no production-code changes
after this diagnosis. Thirteen local HTTP tests and two representative notebook
tests passed with socket access; 91 related financial/fixture tests passed.
The complete baseline was then restarted from a new clean export with the
required socket access and passed all five gates. Original evidence: [sandbox run](release-gate-sandbox.log),
[HTTP diagnosis](local-http-triage.log), [notebook diagnosis](notebook-triage-sandbox.log).

## Publication

**4.8.0 was not published to PyPI or as a public GitHub Release.** The public
`v4.8.0` tag remains unchanged. The [tagged release workflow](https://github.com/jalonso1979/puremacro/actions/runs/38111212258)
passed all six Linux/macOS jobs and the slow/reference/replication suite, but all
three Windows jobs failed the same two legacy-cartridge verification tests.
Trusted publishing correctly skipped the upload. See [workflow results](github-workflow.json)
and [Windows failure summaries](windows-failures.txt).

Diagnosis: re-encoding an NPZ payload changes ZIP creator-OS headers between
Unix and Windows, changing the archive SHA despite identical stored arrays.
The regression exposed a cross-platform verification bug affecting v1 and v2
cartridges. A new 4.8.1 tag will carry the fix; this failed tag is not rewritten.

These are developer checks, not outside-user adoption evidence. Full notebook
execution and a deployed browser UI test are not claimed by this local check.
