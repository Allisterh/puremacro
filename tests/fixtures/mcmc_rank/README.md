# Independent MCMC diagnostics

`draws.npz` contains the exact arrays used by the actual ArviZ 0.22.0 execution
recorded in `reference.json`. Its hash is checked by the offline tests. The
reference exporter imports ArviZ, not Puremacro, and records the installed
diagnostic source hash and numerical-library versions:

```bash
python tools/reference_validation/export_mcmc_rank.py --output /tmp/mcmc-reference
```

Cases include independent draws, chains with different locations or scales,
Cauchy tails, discrete ties, odd-length chains, a single chain, and positive/
negative serial correlation. These test diagnostic calculations, not the
correctness or convergence of a particular economic posterior sampler.

One intentional difference is explicit in the tests: in `ties` the 95th
percentile equals the maximum category. ArviZ assigns that constant indicator
nominal ESS, while Puremacro returns NaN for the unavailable tail evidence.
Globally constant inputs likewise return NaN in Puremacro. Independent
reference values are retained unchanged; their tolerances are not relaxed.
