# Inspecting and executing `.mod` command programs

`inspect_mod` inventories the syntax and execution support of a Dynare-style
file without evaluating calibration or steady-state expressions, compiling
residuals/derivatives, solving, estimating, simulating, importing plotting tools,
or changing the RNG. It reads the source and its explicitly requested local
macro includes. This is a compatibility report, not a determinacy, numerical
accuracy or economic-validity check.

```python
from puremacro.dsge import inspect_mod

report = inspect_mod("study.mod", ordered=True)
print(report.summary())
print(report.to_json())
```

```bash
puremacro-dynare inspect study.mod --ordered
puremacro-dynare inspect study.mod --ordered --format json
puremacro-dynare inspect study.mod --entry-point load_mod --output report.txt
```

The CLI writes only to stdout unless `--output` is supplied. Exit codes are
`0` for a complete report without findings, `1` for compatibility findings or
required caller input, and `2` for incomplete inspection or unreadable input.
JSON and text expose the same statements and findings; JSON also provides the
complete source hashes, option statuses and provenance records.

The default entry point is `execute_mod_commands`; `load_mod` and
`load_hank_mod` can be selected explicitly. Loading a model does not itself run
its `stoch_simul`, smoothing or decomposition commands. An option accepted by
the parser is reported as supported only if the selected runner consumes it.
Unknown scripts, MATLAB control flow, unsupported options, `EXPECTATION`, and
`STEADY_STATE` applied to an exogenous variable receive named findings.
`hetagent_block` is identified as a puremacro extension requiring the HANK
entry point, not as native Dynare syntax. That loader uses its configured
built-in household unless `household_model` is supplied. Fields retained but
not consumed by HANKModel receive separate findings.

Each statement includes its original source location, expanded location,
excerpt, status, explicit options and requested variables. Equation and
model-local entries are children of the enclosing model block. Comments are
inventoried too. Macro directives, including inactive branches, follow the
expanded statement inventory and carry original positions rather than an
invented expanded position. Included lines retain their include chain. Loop
expansions have distinct instance identifiers even when they originate from
the same source line. Columns are `null` when interpolation prevents an exact
mapping; a statement spanning multiple origins also carries the individual
origin spans. Input hashes use SHA-256 of the original UTF-8 bytes, including
line endings; parsing normalizes LF, CRLF and CR separately. Reports contain no timestamps.

Macro preprocessing is the only expression-evaluation step in inspection;
it uses the existing macro engine with a work budget (`macro_limit=100_000`)
and 2 MB source/expansion limits. Oversized expansions, failed includes,
malformed syntax, and incomplete blocks produce incomplete coverage. Reports
never substitute a successful compatibility verdict for those failures.
Parameter expressions remain unevaluated, including invalid numerical
expressions such as `log(-1)`; a clear report therefore does not establish that
calibration or a solve will succeed.

For smoothing and decomposition, `data=` optionally checks the DataFrame's
columns, numeric conversion, duplicates and infinities. NaNs are allowed.
The likelihood and identification are not evaluated. Without data, the report
names the missing input. A `datafile` option does not implicitly read a file.

## Opt-in ordered execution

Use ordered loading to reject unsupported programs **before any solve**:

```python
from puremacro.dsge import load_mod

model = load_mod("study.mod", command_mode="ordered")
results = model.execute_mod_commands(ordered=True, data=data)
for invocation in results.invocations:
    print(invocation.command, invocation.explicit_options,
          invocation.effective_options, invocation.requested_variables)
    # Full numerical result and selected presentation are kept separately.
    full = invocation.result
    selected = invocation.selected_result
```

For a simulation-only program, omit `data`. Ordered loading returns a
first-order `LinearModel`; each `stoch_simul` invocation requests its own order
through the existing solver. Do not pass `order=2` or `order=3` to ordered
loading itself. The loader consumes the expanded source snapshot that was
inspected, and later execution uses that report even if an included file
changes. Macro programs must be loaded with `command_mode="ordered"` to retain
this provenance. For plain text, `execute_mod_commands(ordered=True)` can also
check an already loaded model, but loading has already performed its solve.

The stochastic subset supports repeated `stoch_simul`, `calib_smoother` and
`shock_decomposition` at a fixed calibration. The 5.7.0 development extension
also replays deterministic boundary blocks, shock declarations and repeated
perfect-foresight setup/solver commands, including mixed deterministic and
stochastic programs. Use `read_mod(source).execute_mod_commands()` to start
without an implicit steady-state or perturbation solve; see the
[nonlinear model guide](dsge_nonlinear.md) for the supported options and snapshots.
Interleaved parameter assignments, policy commands and HANK commands remain
unsupported in ordered execution. Known incompatibilities, supplied data schema
and requested samples are checked before the first numerical command.

| Command | Supported explicit options | Effective behavior |
|---|---|---|
| `stoch_simul` | `order`, `irf`, `periods`, `ar`, `drop`, `hp_filter`, `pruning`, `nograph` | Initial defaults: order 2, irf 40, periods 0, ar 5, drop 100, no filter; `nograph` must be enabled. |
| `calib_smoother` | `prefilter`, `first_obs`, `nobs` | Default prefilter off, first observation 1, all remaining observations. |
| `shock_decomposition` | `first_obs`, `nobs` | Uses the supplied sample and full underlying state system. |

Options use literal values in this subset. A symbolic value is unresolved in
inspection, rather than silently evaluated or substituted. `first_obs` is
one-based and `nobs` is a count. Shock standard deviations can be overridden
with `execute_mod_commands(sigma=...)`; the override reaches all three handlers.

Supported settings persist within one program, with two explicit exceptions:
**omitting `order` on each `stoch_simul` resets it to 2**, and **omitting an
output-variable list selects all available outputs for that invocation**.
`ar`, `irf`, filtering and simulation settings otherwise persist; sample
settings `first_obs`/`nobs` are shared by smoothing and decomposition.
`prefilter` persists too. Because the native decomposition method does not
implement prefiltering, a decomposition following `calib_smoother(prefilter=1)`
is rejected unless a preceding smoother explicitly resets `prefilter=0`.
Independent executions start from fresh option defaults.

For example:

```text
stoch_simul(order=1, irf=3, ar=1) y;
stoch_simul(order=1, irf=7) x;
stoch_simul(irf=9);
```

This produces three entries, with `(order, irf, ar)` equal to `(1,3,1)`,
`(1,7,1)`, and `(2,9,1)`. Their selected outputs are `y`, `x`, and all variables.
A `stoch_simul` selected result narrows displayed moments/IRFs but retains the
complete decision rules. Smoother selections narrow available state/observable
presentation frames; a requested variable absent from both is rejected before
execution. Decomposition selections narrow component tables.

Each `DynareCommandInvocation` retains source location, explicit/effective
options, requested selection, input/calibration fingerprint, status, and result
snapshots. `results.stoch_simul`, `.smoother`, and `.shock_decomposition` remain
conveniences for the **last successful full result** of that kind. Results from
earlier calls are not overwritten. `ModExecutionError.results` retains all
completed entries and the failed invocation if a numerical method fails;
`ModCompatibilityError.report` contains the preflight findings.

The legacy defaults remain available: `load_mod(...)` and
`execute_mod_commands()` still aggregate by command category. Inspection names
that mode's repeated-command collapse and ignored output-variable selections.

## External evidence and limits

The small offline corpus in `tests/fixtures/dynare_commands/` stores a Dynare
7.0 generated MATLAB driver excerpt and its source hash. It verifies per-command
order reset, persistent `ar`/`irf`, fresh output selections, and shared sample
settings. It is **preprocessor evidence**, not a numerical Dynare comparison.
Tests additionally compare each supported puremacro invocation with its direct
Python numerical method and exercise partial failures and unsupported sequences.

The [Dynare manual](https://www.dynare.org/manual/the-model-file.html#stoch_simul)
describes output selections and defaults. The fixed Dynare 7.2
[statement writer](https://git.dynare.org/api/v4/projects/25/repository/blobs/3e5412a2ec69318ae70da8c3b46f3133333c122e/raw)
and [runtime](https://git.dynare.org/api/v4/projects/26/repository/blobs/557e89c0fcf2b5fe8ca7cf8b8afd4f7408c37bd6/raw)
also establish why defaults must not be interpreted as resetting every option.
This subset does not claim general Dynare command-program parity, global
MATLAB `options_` compatibility, or support for scripts and mutable model
programs. Use `check`, `resid`, and `model_diagnostics` after solving for
numerical/model diagnostics.
