> 🇬🇧 English · 🇪🇸 [Español](es/performance_scaling.md)

# Solver choice and reproducible scaling measurements

`solve_perfect_foresight(..., linear_solver="krylov")` opts into the existing
matrix-free stacked Newton-Krylov engine. The default `"direct"` retains the
sparse direct solver. Both use the same model/callable preparation, shock paths,
initial history, literal or solved terminal conditions, and fixed
`STEADY_STATE()` reference. `extended_path` forwards the choice at every date;
surprise-shock simulation also forwards it. MCP with Krylov raises an explicit
error before numerical evaluation.

```python
import numpy as np
from puremacro.dsge import solve_perfect_foresight

def equations(lead, current, lag, shock):
    return current - 0.2 * lead - 0.5 * lag - shock

result = solve_perfect_foresight(
    equations, y_init=[0.0], y_ss=[0.0], periods=40,
    shocks=0.01 * 0.7 ** np.arange(40), linear_solver="krylov",
    krylov_options={"preconditioner": "steady_block_tridiagonal"},
)
assert result.converged
print(result.residual_norm)
print(result.solver_diagnostics["converged_by"])
```

The Krylov route requires a small residual relative to the terminal Jacobian
scale **and** a small full Newton direction. The tolerance is `tol`, the step
tolerance is `1e-10 * max(1, |path|_inf)`, and `dampening` caps the initial line
search step. A numerical failure warns and returns `converged=False` with its
last path and diagnostics. It never silently changes to the direct backend.
Invalid inputs and errors from residual callbacks still raise. Inspect
`solver_diagnostics["linear_history"]` for Krylov status, recomputed linear
residuals, requested tolerances, and accepted steps. The direct backend retains
its existing convergence rules; matching flags alone are not a parity check.

`krylov_options` accepts `preconditioner`, `forcing`, `krylov_method`, `inner_m`,
`outer_k`, `krylov_maxiter`, `side`, `max_log_move`, `time_block`, `progress`, and
`checkpoint`. The default terminal-block preconditioner is factored once per
solve. The current-block choice reuses its most recent factorization only when
all prepared block shapes, dtypes and entries are exactly equal. Changed blocks
rebuild it; no model-identity or global cache is used. Rolling extended-path
problems create separate solver instances. The dense blocks still need
`O(T n²)` storage, so this does not establish suitability for thousands of
variables per date. See [the stacked solver's contract](dsge_stacked_newton.md).

## Run the scaling suite

From a source checkout with its development dependencies:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python benchmarks/benchmark_scaling.py --profile smoke --repetitions 3 --output scaling-smoke.json
python benchmarks/benchmark_scaling.py --profile standard --repetitions 5 --output scaling-standard.json
```

Set the thread variables for both commands in your shell when comparing runs.
On Windows, set them through PowerShell's `$env:` syntax before launching Python.
`--families pf hank var mrio` selects workloads; `--seed` fixes synthetic data and
bootstrap draws. The smoke profile is a mechanism check, not a performance claim.
The standard profile varies PF and HANK horizons, bootstrap draw counts, and
MRIO sector counts. Setup and reference construction are outside the timed region.

The JSON records Python/package/NumPy/SciPy versions, platform, CPU information,
NumPy build configuration, thread environment, source revision and dirty-tree
status. Runtime uses the continuous-VFI benchmark's `perf_counter` engine with
one untimed warmup and the requested repetitions (minimum, mean, standard
deviation). Memory uses a separate warmup and run with `tracemalloc`; it is peak
**traced allocation**, not process RSS or a bound on all native memory. Accuracy
is evaluated separately. Compare runs on the same environment and retain the
full JSON; there is no universal speedup threshold or automatic backend choice.

| Workload | Checked evidence | Scope limit |
|---|---|---|
| PF direct and Krylov | Manufactured exact path and recomputed nonlinear equations | Three variables; synthetic equations |
| HANK horizons | Household-response, NKPC, Taylor/Fisher, goods-market equations and distribution mass | Built-in one-asset linear GE; no grid-refinement certificate |
| VAR bootstrap | First seeded draw against scalar recursive simulation and IRF reconstruction; finite ordered bands | Reproducibility/implementation check, not confidence-band coverage |
| MRIO sectors | Sparse aggregation against direct dense sums and accounting conservation | Synthetic balanced tables; no native archive or trade-equilibrium solve |

Failed numerical checks produce exit status 1 and remain in the JSON. Workload
exceptions are recorded as errors; invalid CLI options return status 2. Larger
profiles should be run deliberately, rather than added as timing-sensitive CI gates.
