> 🇬🇧 English · 🇪🇸 [Español](es/mcmc_diagnostics.md)

# Multichain MCMC diagnostics

Puremacro provides rank-normalized split R-hat, multichain bulk/tail effective
sample size (ESS), and a Monte Carlo standard error for the posterior mean.
They require only NumPy and SciPy. `NUTSResult.summary()` uses them automatically;
the stored posterior draws and sampler algorithm are unchanged.

```python
import numpy as np
from puremacro.mcmc import rank_normalized_rhat, multichain_ess, mcse_mean

chains = np.random.default_rng(12).normal(size=(4, 1000))
print(rank_normalized_rhat(chains))
print(multichain_ess(chains, method="bulk"))
print(multichain_ess(chains, method="tail"))
print(mcse_mean(chains))
```

Supply **independently initialized post-warmup chains** for one parameter, with
shape `(chains, draws)`. Do not stack unrelated parameters as if they were
chains, or concatenate chains before computing ESS. The input is not modified.

## What each statistic measures

| Statistic | Calculation and interpretation |
|---|---|
| `rank_normalized_rhat` | Split each chain, pool average ranks, transform with normal quantiles, and take the larger R-hat of those scores and the folded scores. Folding helps reveal scale differences between chains. |
| `multichain_ess(..., method="bulk")` | Rank-normalized split-chain ESS, combining within-chain autocovariances with between-chain variation before Geyer's truncation. |
| `multichain_ess(..., method="tail")` | The smaller ESS of indicators below the pooled 5th and 95th percentiles. `tail_probability` changes these symmetric probabilities. |
| `multichain_ess(..., method="mean")` | Split-chain ESS of the untransformed draws, for inference about a finite-variance posterior mean. |
| `mcse_mean` | Pooled sample standard deviation divided by the square root of mean ESS. This is simulation error, not posterior parameter uncertainty. |

Rank scores use Blom's `(rank - 3/8)/(S + 1/4)`. Odd-length chains drop their
central draw when split. Tail thresholds use all supplied draws before the
indicators are split. ESS can exceed the retained count for negatively
autocorrelated chains; the finite-sample cap is `S*log10(S)`.

R-hat requires at least two original chains and four draws per chain. ESS
accepts one chain, with the same minimum length, but cannot then test agreement
between independent chains. Short, nonfinite or globally constant input yields
NaN; different constant chains give infinite R-hat. If a tail indicator is
constant, its evidence is unavailable and the reported tail ESS is NaN. This
last policy is deliberately more explicit than assigning nominal ESS to a
constant indicator.

Values near one for R-hat and large ESS are useful screening evidence. They do
not prove global exploration, identify an economic model, or compensate for a
wrong likelihood. Examine divergences, energy diagnostics, traces, parameter
regions and the log posterior as well. Mean MCSE requires finite posterior
variance and appropriate Markov-chain limit conditions; rank normalization
does not make the mean of a Cauchy posterior well defined.

## Migration from 5.5.0

Previously, the NUTS helpers called “bulk” and “tail” ESS summed separate-chain
ESS estimates and did not penalize chains trapped in different regions. They
now use the multichain definitions above. NUTS summary `r_hat` now uses the
rank-normalized/folded statistic, and `mcse_mean` is an additional column.
One-chain R-hat is NaN. Summary attributes identify the diagnostic methods.
The classic `gelman_rubin` and single-chain `effective_sample_size` APIs remain
available. `compute_split_rhat` retains its classical split definition, but
constant/invalid draws no longer produce an artificial success value of one;
the energy helper likewise returns NaN for a constant trace.

## Independent verification

The offline regression fixture records an actual **ArviZ 0.22.0** run on eight
stored input arrays, with input hashes, package versions and a source hash.
Finite informative comparisons use relative tolerance `2e-11` for ESS and
MCSE, and `2e-12` for R-hat. The discrete-tie case's constant upper-tail
indicator is an explicitly tested policy difference, with the original ArviZ
reference retained unchanged. ArviZ is used only to regenerate development
fixtures and is not a dependency of Puremacro.

These methods follow [Vehtari et al. (2021)](https://doi.org/10.1214/20-BA1221).
The independent software implementation is
[ArviZ 0.22.0 diagnostics](https://github.com/arviz-devs/arviz/blob/v0.22.0/arviz/stats/diagnostics.py).
