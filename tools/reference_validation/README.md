# Independent structural references

These tools compare actual external inputs with puremacro. Ordinary tests use
small frozen fixtures and do not require MATLAB or network access.

## Dynare 7.0 / MATLAB

Five source models in `tests/fixtures/dynare_live/` cover quadratic forward
feedback, a cubic model with correlated innovations, nonlinear state dynamics,
nonlinear dynamics with multiple correlated shocks, and a positive-steady-state
RBC model with a static output variable. Both orders two and three are run.

From MATLAB, using **absolute paths** appropriate to your installation:

```matlab
addpath('/path/to/puremacro/tools/reference_validation');
export_dynare('/path/to/Dynare/7.0', ...
              '/path/to/puremacro/tests/fixtures/dynare_live', ...
              '/tmp/puremacro-dynare-live');
```

The exporter copies each model into its own working directory, appends
`stoch_simul(order=2/3,pruning,irf=0,nograph,noprint,nomoments,nocorr)`, and
runs Dynare. It saves the actual decision rules, covariance, labels, versions,
innovations and `simult_` paths. Original model files are not modified.

The checked-in innovations use NumPy `default_rng(20260920)`, 2,500 standard
normal draws transformed by each model's Cholesky covariance. MATLAB reads the
same decimal CSV values; it does not draw another random sample. The first
500 observations are omitted only from the moment comparisons.

From the repository root:

```sh
PYTHONPATH=. MPLBACKEND=Agg python tools/reference_validation/validate_dynare.py \
  --freeze /tmp/puremacro-dynare-live --output /tmp/dynare-comparison.json
```

**`--freeze` replaces reference fixtures.** Use it only on exports from an actual
Dynare run. Omit it to verify existing references. The exporter records a common run ID, source text and a completion marker.
The freezer rejects stale exports and sources/innovations that differ from the
executed inputs. The manifest records hashes of model sources, innovation CSVs,
raw MATLAB exports and normalized NPZ files.
Normalization only unpermutes DR rows to declaration order; state and shock
columns are aligned by labels. The `gh*` arrays contain unfolded derivatives,
so no Taylor-factor or folded-column conversion is applied. `ys` is already in
declaration order. Simulations are compared in levels after dropping Dynare's
initial-condition column and adding puremacro's deterministic steady state.

Every requested tensor, including `ghxxu`, `ghxuu`, `ghxss` and `ghuss`, is tested.
The benchmark also compares innovation covariance, every simulated observation,
and sample means, covariance and lag-one covariance. It does **not** compare
analytical higher-order moments, prove approximation accuracy far from steady
state, or extend the public parity dashboard beyond its supported orders.

Runtime caveat for the recorded run: MATLAB R2026a Update 5 executed all ten
cases and saved complete finite exports, then suffered a segmentation fault
during process shutdown on this machine. The benchmark success refers to the
exported numerical comparisons, not a clean MATLAB exit. Regeneration requires
checking that every model/order export exists and its recorded runtime version
is appropriate. The processing log and the caveat are retained in the review.
