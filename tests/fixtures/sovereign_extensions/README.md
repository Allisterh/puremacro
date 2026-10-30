# Independent sovereign extension equations

Regenerate from the repository root:

```sh
python -m tools.reference_validation.check_sovereign_extensions --output tests/fixtures/sovereign_extensions/independent_validation.json
```

The NumPy/SciPy reference in `tools/reference_validation/sovereign_extensions.py`
uses no production solver helpers. The runner invokes production solvers only
to compare their outputs. Recorded source hashes pin both implementations.
Registered thresholds are 1e-7 for value/price agreement and 1e-8 for accounting
and Bellman residual checks. Tests separately check simulation draws, survival
pricing, crisis measurement, missingness and capacity averages.

Long-term cases validate the existing smoothed quantal-policy operator, not a
deterministic optimum or an entropy-inclusive-value model. Its deterministic
maximization gap is retained. The collateral reference solves the illustrative
adjustment-cost planner, including a binding-constraint case; it does not
reproduce Mendoza's full competitive model. These small-grid checks establish
neither extension grid convergence nor published empirical moments.
