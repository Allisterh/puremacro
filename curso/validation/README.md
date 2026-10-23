# Course execution evidence

[`course-2026A-5.5.0.json`](course-2026A-5.5.0.json) records the successful
execution of all 72 Python notebooks against the exact 5.5.0 release-candidate
wheel identified by its SHA-256. It records the actual archive executed;
public-wheel content-equivalence verification belongs to the release record.

The run used Python 3.12.13, frozen course inputs and cached reference-model
responses. All 1,020 code cells executed, including every required DP/text
section. All 544 input files remained unchanged. All 2,184 economic assertions
were preserved, and the 348 saved PNG figures have opaque backgrounds.

The manifest includes dependency versions, relative input-file hashes,
per-notebook output hashes, cell-source hashes in notebook order, executed code
cell indices, required-capability cell indices and optional-path outcomes.
Notebook input hashes include outputs present before execution; the runner
clears them before running. The final output hashes identify the committed
executed notebooks. Cell indices start at zero.

Both historical T01_B notebooks used the explicitly labelled representative
EU-KLEMS branch. Live data refreshes and local model downloads remain optional
and were not exercised. The JavaScript labs are separate teaching implementations.

To repeat the audit, follow the [course instructions](../README.md#reproduce-the-execution-audit)
and use [`scripts/ejecutar_edicion.py`](../scripts/ejecutar_edicion.py). The runner
writes detailed per-cell timing and execution evidence into the chosen audit
directory. Generated QNA exports stay in each notebook's `generated/qna/`
directory, separate from frozen inputs.

## RBC uncertainty update — 22 October 2026

The original 72-notebook record above applies to the immutable
`course-2026A-5.5.0` edition. The two T02_D notebooks amended on 22 October added
irreversible RBC investment and stochastic volatility; their revised source
and output hashes are recorded separately in
[`rbc-uncertainty-2026-10-22.json`](rbc-uncertainty-2026-10-22.json).

Both updated notebooks were executed in fresh, offline IPython processes with
the published puremacro 5.5.0 runtime, Python 3.12 and the edition's frozen
helpers and model files. Each executed 20 code cells and produced eight figures.
The checks cover Bellman residuals, feasible choices, complementarity, Euler
residuals at states between grid nodes, grid refinement and interpolation bounds.
The slide figure retains the independently checked 5.3.0 reference calculation;
the new 5.5.0 run checks compatibility and numerical agreement. Local Python is
the verified route; this update does not claim a new Colab, Juno or browser run.
The original tag and its execution manifest were not changed.

## DSGE solution methods — 23 October 2026

[`dsge-methods-2026-10-23.json`](dsge-methods-2026-10-23.json) records the
English and Spanish T02_A–D revisions against the 5.7.0 release source,
with setup cells, package requirements and Colab links pinned to `v5.7.0`.
It identifies the package files, notebook sources and saved outputs, helper
and model files, frozen data, dependency versions, cell timings and warnings.
Source-checkout paths in text diagnostics are made relative for portability.
All eight notebooks completed: 120 code cells and 72 saved figures. T02_A
retains a Klein conditioning warning in the capital-share sweep; both language
versions record it. All eight completed in one fresh execution batch after the release pins were
finalized. The earlier development batch is preserved separately in
[`dsge-development-2026-10-23.json`](dsge-development-2026-10-23.json), including
the corrected Spanish T02_B assertion and its complete rerun. The development
and release batches use identical runtime package files and frozen inputs.
The runner rejects write-back if package code, notebook sources or inputs
change during execution. This is a source-checkout validation, not a released
wheel or a new Colab, Juno or browser certification.

T02_A checks callback/`.mod` path agreement, direct/Krylov agreement, continuation,
an explicit permanent-shock tail, horizon doubling, news versus surprise timing,
repeated ordered `.mod` commands and local/global policy errors at independent
points with a richer quadrature rule. The construction domain reports its
quadrature exits explicitly; the audit covers the declared interior points.
T02_B compares orders 1–3 using expected marginal-utility equations; T02_C
adds local conditioning diagnostics. T02_D checks the library's nonlinear
irreversible-investment transition against an independent residual function,
retaining its horizon and stochastic-volatility exercises.

Install the matching `v5.7.0` checkout and execute from the repository root:

```bash
python -m pip install -e ".[notebooks]"
python curso/scripts/ejecutar_desarrollo_dsge.py \
  --work-dir /tmp/puremacro-dsge-validation --jobs 2 --write-back
```

Use a new work directory for each run. Every notebook gets a fresh IPython
process and offline input directory; no code cells or economic assertions are
skipped. The earlier manifests remain historical evidence for their original
revisions. The current notebook hashes belong to this new record.
