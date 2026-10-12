> 🇬🇧 English · 🇪🇸 [Español](es/micro_data.md)

# Data sources registry and survey microdata

Two pieces sit on top of the existing fetchers:

- `puremacro.fetch.registry`: one table of every source puremacro can reach
  (series, panels, real-time vintages, microdata and the narrative text
  connectors), with what each one needs before you call it.
- `puremacro.fetch.micro`: household and person surveys fetched through public
  APIs or download files, returned with their survey design so that every
  estimate comes with a design-based standard error.

## The registry

The registry describes; it does not wrap. Existing fetchers keep their own
signatures, and an entry records where the callable lives, whether it needs a
key, the terms, and whether results may be redistributed. Importing it pulls in
no network code, so it also works in the browser.

```python
from puremacro.fetch import registry

micro = registry.sources("micro")
print(micro[["id", "auth", "credential", "redistributable"]])
print(registry.sources()["kind"].value_counts())   # series, panel, vintage, micro, text
print(registry.info("fed.scf").terms_url)
```

`auth` is one of `none`, `key_optional`, `key`, `registration` (a free account
with terms accepted per collection), `approval` (a reviewed project proposal)
or `unaudited`. The narrative text connectors are listed as `unaudited` and
not redistributable: their terms have not been reviewed one by one, so check a
source's own terms before republishing its text. `registry.load(id)` imports
and returns the fetcher; `registry.register(SourceInfo(...))` adds your own.

## Microdata providers

| function | survey | variance design |
|---|---|---|
| `fetch_acs_pums(year, variables, ...)` | Census American Community Survey PUMS (1- or 5-year), persons or households | 80 successive-difference replicate weights |
| `fetch_cps_basic(year, month, variables, ...)` | Census Current Population Survey, basic monthly | weights only (no published replicates) |
| `fetch_scf(year, variables, ...)` | Federal Reserve Survey of Consumer Finances, summary extract | five implicates, 999 bootstrap replicate weights, Rubin's rules |
| `fetch_enigh(year, variables, ...)` | INEGI ENIGH (Mexico), 2016 onwards | strata and PSUs, Taylor linearisation |

The Census API needs a free key: store it as described in
[Credentials](CREDENTIALS.md) (service `census`) rather than passing it in code.
Keys never reach the cache file names, the manifest or error messages. The SCF
and ENIGH need no key.

```python
# requires: network (Census API key)
from puremacro.fetch.micro import fetch_acs_pums

acs = fetch_acs_pums(2023, ["AGEP", "WAGP", "SEX"], geography="state:11",
                     where={"AGEP": (25, 64)})
acs.mean("WAGP", by="SEX")          # estimate, se, n per group
```

```python
# requires: network (federalreserve.gov)
from puremacro.fetch.micro import fetch_scf

scf = fetch_scf(2025, ["networth", "income", "age"])
scf.quantile("networth", 0.5)                  # published: $215,900
scf.ratio("networth", "income", q=0.5)         # median wealth / median income
scf.gini("networth"), scf.top_share("networth", 0.01)
```

Downloads go through the shared HTTP cache, so a second call is offline.

### Facts worth knowing before you estimate

- **Not-applicable codes.** Census codes such as `WAGP == -1` (persons under 15)
  or `ESR == 0` are set to `NaN` by default, read from the endpoint's own
  codebook, so they do not enter means and totals as values. `na_to_nan=False`
  keeps the raw codes and `frame.codebook` lists every labelled code. CPS sends
  some earnings codes with implied decimals (`-1` arrives as `-0.01`); those are
  handled too.
- **SCF dollars.** The Fed publishes every wave of the summary extract in the
  dollars of the latest survey. With the 2025 release (October 2026) the 2022
  levels rose by 9.0%; ratios, shares and the Gini are unaffected.
- **SCF implicates.** The SCF imputes missing answers five times. Each household
  has five rows; estimates average over implicates and the standard error adds
  the between-implicate variance (Rubin's rules) to the sampling variance from
  the replicates on implicate 1, as the Fed's documentation prescribes.
- **ENIGH** has no replicate weights. Means, totals and quantiles use Taylor
  linearisation with the ultimate-cluster variance; `ratio`, `gini` and
  `top_share` raise `NotImplementedError` there rather than return an estimate
  without its standard error.

## Estimating from a `MicroFrame`

Every provider returns a `MicroFrame`: the records, a `SurveyDesign` that says
which columns are weights and how to turn replicates into variances, and the
query that produced them. The estimators `mean`, `total`, `quantile`, `share`,
`ratio`, `gini` and `top_share` all return the estimate, its standard error, the
unweighted `n` and the number of implicates, optionally `by` groups.

The block below builds a frame by hand so it runs offline; a fetched frame
behaves the same way.

```python
import numpy as np
import pandas as pd
from puremacro.fetch.micro import MicroFrame, SurveyDesign

rng = np.random.default_rng(0)
n = 2000
w = rng.uniform(50, 150, n)
records = pd.DataFrame({"PWGTP": w, "WAGP": rng.lognormal(10.5, 0.8, n),
                        "SEX": rng.integers(1, 3, n)})
replicates = pd.DataFrame(w[:, None] * rng.uniform(0.7, 1.3, (n, 80)),
                          columns=[f"PWGTP{r}" for r in range(1, 81)])
frame = MicroFrame(pd.concat([records, replicates], axis=1),
                   SurveyDesign.acs_pums(), unit="person",
                   source="synthetic", vintage="demo")

print(frame.mean("WAGP", by="SEX"))
print(frame.quantile("WAGP", 0.5))
print(frame.gini("WAGP"))
print(frame.top_share("WAGP", 0.10))
```

`SurveyDesign(...)` describes any other survey: a main weight, replicate
weights with their method (`"sdr"`, `"bootstrap"`, `"jk1"`, ...), scale and
centring, or strata and PSUs for `method="taylor"`.

## Worked example: an Aiyagari economy against the SCF

```text
python -m puremacro.examples.scf_aiyagari --output results/
```

The script fetches one SCF wave, estimates wealth-to-income ratios and
concentration moments with replicate standard errors, calibrates the discount
factor of a [`dp`](dp.md) Aiyagari model (solved by EGM) to one ratio, carries
the target's sampling error into `beta` by the delta method, and sets the
untargeted moments against the data. On SCF 2025, with `beta` calibrated to
median wealth over median income:

| moment | SCF (se) | model (se) |
|---|---|---|
| median wealth / median income | 2.628 (0.077) | targeted |
| mean wealth / mean income | 8.518 (0.175) | 3.912 (0.101) |
| Gini | 0.824 (0.004) | 0.591 (0.001) |
| top 1% share | 0.313 (0.009) | 0.063 (0.000) |
| top 10% share | 0.721 (0.007) | 0.379 (0.001) |
| bottom 50% share | 0.021 (0.001) | 0.088 (0.001) |

`beta` = 0.952 (se 0.002), r = 4.08%, K/Y = 2.98. The gap at the top is the
known failure of the one-asset model with earnings risk and a borrowing limit;
nothing in the example is tuned to close it. Targeting the mean ratio instead
is not possible with `beta < 1`: the model reaches at most 7.55, with r < 0.
