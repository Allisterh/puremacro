# Independent references, generated 2026-10-26

The manifests record calibration, scope, observed comparison errors and hashes.
These are references from separate calculations, not snapshots of PureMacro
outputs. Regenerate them using `tools/reference_validation/validate_*_reference.py`.

- **HANK:** the authors' `sequence-jacobian` household block at commit
  `eb4ac4c3157357a4cacbea4b5a7a281f5b48cab9`, with matched income states,
  asset grids, balanced-budget income taxes and realized-return timing.
  Policies, distribution and consumption use absolute tolerance 1e-8;
  wage Jacobians 1e-7; return Jacobians 5e-4. The finite step is 1e-4.
  Asset maximum 100 avoids the upper-boundary mass observed at default maximum
  30. Agreement on a grid does not prove grid convergence. This does not
  replicate a full published HANK calibration or validate aggregate risk.
- **Arellano:** a separate asset-sign Bellman implementation of the equations
  documented by Sargent and Stachurski in the QuantEcon lecture. No production
  solver or discretization helper generates this reference. Utility is
  normalized to `1 - 1/c`, output to stationary mean one, default is
  deterministic and debt is short term. Absolute tolerance is 1e-8.
  This checks equations, prices and policies on matched grids, not the paper's
  empirical moments or long-term debt extension.

Sources: [SSJ code](https://github.com/shade-econ/sequence-jacobian/tree/eb4ac4c3157357a4cacbea4b5a7a281f5b48cab9),
[QuantEcon equations](https://eqm.quantecon.org/arellano.html),
[Arellano (2008)](https://doi.org/10.1257/aer.98.3.690).
