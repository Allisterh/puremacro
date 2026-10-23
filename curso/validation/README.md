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
`course-2026A-5.5.0` edition. The two T02_D notebooks on `main` now add
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
