# Sovereign-risk brief: frozen converged sweep

`sweep.json` is the frozen record behind the
[sovereign-risk policy brief](../../../docs/brief_sovereign_risk.md). It holds
illustrative comparative statics of PureMacro's public Arellano calibration,
not estimates for any country. Tests and docs read this file; nothing re-runs
the sweep.

## Contents

Seven scenarios (the baseline, then `theta`, `sigma_y` and `rho_y` moved one at
a time to a low and a high value), each solved on the converged 6,401 x 641 grid
and the 3,201 x 641 error grid with `policy_search="monotone"`. For each of the
14 cases the record stores the full solver keywords, the solver diagnostics
(convergence, iterations, Bellman residual, monotone certification gap,
exhaustive Bellman residual, wall time), the per-seed raw ergodic moments from
`puremacro.sovereign.ergodic_moments` (16 seeds, 1979-1994, 500,000 quarters,
1,000-quarter burn) with their means and 95% t-intervals, and the value of market
access from `puremacro.sovereign.value_of_market_access`. Each scenario also has
its paired 3,201 -> 6,401 grid change under the reference study's rule and
tolerances (0.10 per moment). The record also stores the package, Python, NumPy,
SciPy and pandas versions, SHA-256 hashes of the solver, measurement and example
sources, peak memory and the load average after each case. It contains no local
paths.

Two scenarios do not pass the grid criterion and are marked † on the page:
`theta=0.5` (mean spread, default frequency and spread SD shift by 0.17-0.18)
and `rho_y=0.97` (spread SD shifts by -0.13). They are reported, not re-run with
different tolerances.

## Provenance

```sh
MPLBACKEND=Agg python -u -m puremacro.examples.first_brief_sovereign \
    --freeze tests/fixtures/sovereign_brief/sweep.json
```

Run on 2026-10-29 from commit `1752c83` (PureMacro 6.0.1 sources plus this
branch) on an Apple M3 Pro laptop (12 cores, 36 GB), shared with other jobs
(load averages 4-54 during the run). Wall time 6,851 s (1.9 h); per-case solve
times were 540-880 s at 6,401 x 641 and 250-390 s at 3,201 x 641, and each
measurement took under 8 s. Peak resident memory was 3.4 GB. The run resumes
from `sweep.partial.json` after an interruption, reusing only cases whose source
hashes and design match.

After the freeze, one type annotation was added to
`puremacro/examples/first_brief_sovereign.py` (`entry: dict[str, Any]` in
`build_record`, for mypy). It changes no behaviour, so the record was not
re-run; its `source_sha256` entry for the example is that of commit `1752c83`,
and the other four source hashes still match.
