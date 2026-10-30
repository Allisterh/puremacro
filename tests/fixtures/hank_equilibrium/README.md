# Full one-asset HANK equilibrium

Independent reference: Auclert, Bardóczy, Rognlie, Straub (2021), Econometrica
89(5), 2375–2408, DOI10.3982/ECTA17434; author repository
https://github.com/shade-econ/sequence-jacobian at commit
`eb4ac4c3157357a4cacbea4b5a7a281f5b48cab9`, `notebooks/hank.ipynb` and
`src/sequence_jacobian/examples/hank.py`.

The two NPZ files are generated only by author code. They include household
policies, distribution, grids, and complete linear/nonlinear/opposite-shock
equilibrium paths. JSON companions record the calibration, sources, versions,
array SHA-256 and solve times. `100_3_40` supports fast CI; `500_7_300` is the
published notebook's numerical calibration. This is a matched-equilibrium
comparison, not a comparison of estimated empirical moments.

In an isolated reference environment with the pinned checkout installed:

```sh
python tools/reference_validation/hank_equilibrium_reference.py \
  --reference-repo /path/to/pinned/sequence-jacobian --output tests/fixtures/hank_equilibrium
python tools/reference_validation/hank_equilibrium_reference.py \
  --reference-repo /path/to/pinned/sequence-jacobian --output tests/fixtures/hank_equilibrium \
  --n-a 100 --n-e 3 --horizon 40
```

The pinned reference's `solve_steady_state` discards nested block options. The
generator therefore calibrates with SciPy root calling the authors' own
steady-state evaluator directly, with household backward tolerance1e-12 and
forward tolerance1e-13. Final asset/labor residuals are below1e-9. No PureMacro
household code participates in reference generation.

`convergence.json` is the native study from Python3.11.17, NumPy2.4.6,
SciPy1.17.1. Its native source hash is checked against the committed model.
The protocol was fixed before evaluation: author-policy error1e-7 (saving1e-6),
parameter error1e-7, linear paths1e-6 per unit shock, nonlinear paths1e-7,
markets1e-8. Refinement compares the first40 quarters: 1% of each finer peak,
with a1e-6 level floor, parameter changes1e-4 and upper-grid mass1e-8.
The failed11→15 and15→21 income refinements remain in the record. Expanding
to21→31 states passes; asset and horizon refinement at31 states also pass.
Full runtime was413 seconds on the shared workstation; this is not a timing
guarantee. Ordinary tests use small fixtures, not the expensive study.

Regenerate the native study using normal PureMacro dependencies:

```sh
python -m tools.reference_validation.check_hank_equilibrium --output /tmp/hank-equilibrium --convergence
```

This validates the stated perfect-foresight monetary-shock model and grids.
It does not validate the older reduced HANK closure, arbitrary large shocks,
two-asset models, lower-bound episodes or aggregate stochastic welfare.
