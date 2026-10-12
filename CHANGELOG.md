# Changelog

This file records user-visible changes per release. Internal refactors that don't change behaviour are listed under "Internal" so a returning user can see what shifted under the hood without surprise.

## 5.0.0a1 (unreleased)

This pre-release removes material that is not part of the library and prepares the 5.0 line. 4.8.1 remains on PyPI for anyone who needs what was removed.

### Removed

- The trade/MRIO modules (`puremacro.trade`, `puremacro.models.trade_policy`), their data, notebooks and docs are no longer part of puremacro. The validation gallery therefore has 110 cases in 15 subsystems (the four trade cases are gone), the `gpu` extra is gone with the accelerated trade solvers, and the wheel no longer bundles the OECD input-output table. Users who rely on the trade modules should stay on 4.8.1, the last release that contains them.
- The research studies and applications that had been added to the package since 4.4.0 are no longer part of puremacro: the SW07 estimator studies and their examples, the household incidence application and its dataset, and the cross-country forecasting application and its page. The DSGE fertility model, the identification spec-curve notebook and its panel, the state-dependent local-projection section of notebook 18 (Beveridge curve) that read that panel, and the internal project records (reviews, plans, specs, technical reports) were also removed. 4.8.1 is the last release that contains them. `puremacro.structural` (`MomentTargets`, `fit_structural`, the canonical LP designs), `validation.run_research_benchmarks` and the Romer–Romer (2010) replication stay.

**Declarative dynamic programming front end (`puremacro.dp`).** Declare states,
shocks, choices, a reward, constraints and an optional market-clearing condition
as equations in Dynare expression syntax (parsed by the `.mod` parser), and
`Model.solve()` compiles them to `VFIProblem`, `FiniteHorizonProblem` or
`stationary_equilibrium`. No return-function closures, grid broadcasting or
positional parameter lists. Aiyagari, Huggett, the life-cycle and two-asset
examples written as `dp` models reproduce `vfi.examples` exactly (same prices,
policies and value functions). `vfi.Model` stays supported: `dp` has no panel simulation or Python-callable utilities yet.

**Endogenous grid method for `dp` models.** `Model.solve(method="egm")` compiles
a one-asset model whose budget is `c + a(+1) = R*a + y(z)` to `vfi.solve_egm`.
The compiler checks that form (symbolically for the reward, numerically on the
grid for the budget and constraints) and raises `ModelSpecError` otherwise.
Marginal utility is the symbolic derivative of the reward, so felicities other
than CRRA work (closed-form inverse for `crra` and `log`, bisection otherwise).
Stationary equilibria warm-start each price evaluation. On the Aiyagari example
the equilibrium takes 0.8 s against 4.6 s for discrete VFI, and the EGM value
function is closer to a 1,500-point VFI solution. `vfi.solve_egm` takes
`u_prime`/`u_prime_inv` (any concave felicity) and a `c0` warm start; CRRA calls
are unchanged, and its inversion step is public as `vfi.egm.egm_step`.

Life-cycle models (`horizon(T)`) also solve with `method="egm"`: one EGM step per
age, with age-varying returns, income and parameters, survival, and an optional
terminal value (its slope taken by finite differences). Two-period log-utility
models match their closed form to 1e-12. On the 40-age example the EGM mean
assets by age are within 0.02 of a 1,191-point VFI solution, against 0.22 for
VFI on the same 120-point grid. Value functions interpolate linearly between
nodes, which understates them where they are most curved (near the borrowing
limit with gamma = 2).

A discrete choice (`discrete("h", ...)`, deterministic, no taste shocks) also
compiles to EGM, in both life cycles and infinite horizons: the Euler inversion
for each option is followed by an upper envelope over all segments between
adjacent endogenous points, with the borrowing-limit and top-of-grid corners
admitted only where their Kuhn-Tucker condition holds. The reward may depend on
the option (`crra(c, gamma) - chi*h`) and so may income; the return on assets
may not. A dominated option reproduces plain EGM to 1e-8. On a 30-age
labour-supply model the EGM value function on 100 points is less than half as
far from a 793-point VFI solution as VFI on 100 points, and the two methods agree
on a fine grid (work share by age within 0.02). Aggregates over the discrete
choice converge slowly in the grid for either method because the
distribution's mass sits on grid nodes. `vfi.DCEGMProblem` is unchanged: its
horizon repeats one stationary step and returns only the first period, so it
cannot carry age-varying income or survival.

`Model.taste_shocks(scale)` adds type-I extreme value shocks to the discrete
choice (solved by `method="egm"` only): values are log-sums, choices logit, the
Euler right side averages marginal utility over options and the distribution
splits mass by choice probability, so `mean("h")` is the participation rate.
The last period matches the closed-form logit to 1e-12 and a two-period model
matches a 200,001-point brute-force maximisation to 1e-5.

**Continuous-time `dp` models (`method="hjb"`).** Declare `control("c")`,
`drift("a", "r*a + w*e - c")`, `discount_rate("rho")` and `dp.Jump` shocks
(Poisson generators; several are combined as their product chain), and
`solve()` compiles to `vfi.solve_hjb_achdou`. The compiler checks that the
drift is linear in the control with slope -1 and derives u' and its inverse from
the reward. Prices and `clear()` work as in discrete time, with aggregates
integrated against the KFE density; `distribution` is the mass at each node.
The household matches `solve_hjb_achdou` to 1e-12 and the continuous-time
Aiyagari equilibrium matches `solve_aiyagari_continuous_hjb` (r* to 1e-9).
`solve_hjb_achdou` takes `income` (any (Na, Ne) drift before consumption) and
`utility`/`u_prime`/`u_prime_inv`; default calls are bit-identical.
`Model.solve(method=None)` picks `"hjb"` for a model with a drift and `"vfi"`
otherwise.

**Aggregate risk in `dp` models (Krusell-Smith).** `aggregate_shock("Z", ...)`
and `aggregate_state("K", grid, mean="a")` add the Krusell and Smith (1998)
forecast-rule fixed point, with prices and technology written as model
equations (any firm, any household reward). With the same grids and starting
distribution, the Cobb-Douglas model reproduces `vfi.krusell_smith` bit for bit
(forecast coefficients and capital path). `solve("egm")` solves the household
by EGM and simulates with continuous policies: on a 100-point asset grid it
converges in 36 outer iterations (24 s) to `b1` = 0.965 and 0.969 with R^2 above
0.9999. Discrete VFI policies sit on grid nodes, so on a 60-point grid
simulated capital never moves and the regression is degenerate; at its
defaults (200 points) `vfi.krusell_smith` had not converged after its 60 outer
iterations (470 s, `b1` = 0.897 and 0.954). Loop options go in
`solve(ks=dict(T=..., burn_in=..., seed=..., damping=..., tol=..., max_outer=..., mu0=...))`.
Both `vfi.krusell_smith` and dp's loop now raise a `RuntimeWarning` when they stop
at `max_outer` without converging (they used to return `converged=False`
silently), and when simulated capital never moves under aggregate risk, the
degenerate regression above, which would otherwise pass `tol` and look converged.

`vfi.solve_egm` and dp's EGM accept a return on assets that varies with the
exogenous state (`r` of shape (n_z,); Euler u'(c) = beta E[(1 + r') u'(c')]).
Scalar-`r` calls are unchanged.

### Added

- `puremacro.fetch.registry` lists every data source (series, panels, vintage providers, microdata and the narrative text connectors) with its auth level, credential, terms and whether results may be redistributed; `registry.load(id)` returns the fetcher. `puremacro.fetch.micro` adds `fetch_acs_pums` (Census ACS PUMS, 80 replicate weights, chunked past the API's 50-variable cap), `fetch_cps_basic` (weights only) and `fetch_scf` (Fed SCF summary extract, five implicates, 999 replicate weights) and `fetch_enigh` (INEGI ENIGH, Mexico, 2016+; strata and PSUs, so standard errors come from Taylor linearisation, `SurveyDesign(method="taylor")`). Each returns a `MicroFrame` whose `mean`/`total`/`quantile`/`share` give replicate standard errors, combined across implicates by Rubin's rules. The Census fetchers set the codebook's not-applicable codes (N/A, Not in Universe, Blank, Don't Know, Refused, CPS "Met No Conditions To Assign", including CPS earnings codes served with implied decimals as `-0.01`; e.g. `WAGP == -1` for persons under 15) to `NaN` by default, read from the endpoint's `variables.json`, so they no longer enter means and totals as values; `na_to_nan=False` keeps the raw codes and `MicroFrame.codebook` lists every labelled code.

- `MicroFrame.ratio`, `gini` and `top_share`: ratio of means (or of `q`-quantiles) of two columns, the Gini coefficient and the share held by the top `p`, with the same replicate/implicate standard errors as `mean`. Taylor designs (ENIGH) raise `NotImplementedError` for these rather than return an estimate without its standard error.

- **Worked example: an Aiyagari economy calibrated to the SCF** (`python -m puremacro.examples.scf_aiyagari`). It fetches one SCF wave, estimates wealth-to-income ratios and concentration moments with replicate standard errors, calibrates `beta` of a `dp` Aiyagari model (EGM) to one ratio and reports the untargeted moments next to the data, with the target's sampling error carried through `beta` by the delta method. On SCF 2025, with `beta` calibrated to median wealth over median income (2.63), the model gives a wealth Gini of 0.59 against 0.82, a top 1% share of 0.06 against 0.31 and a top 10% share of 0.38 against 0.72. The mean wealth-to-income ratio (8.52) is out of reach for `beta < 1` (at most 7.55, with r < 0).

- **Narrative corpus is now stored and exportable.** `harvest_narrative_corpus`
  used to keep nothing: its SQLite table was never written (and had no text
  column), and `use_cache`/`db_path` were ignored. A harvest given `db_path=`,
  `store=` or `use_cache=True` (default path: `narrative_corpus.db` beside the
  HTTP cache) now upserts into a `CorpusStore`; without them it stays in memory
  and writes nothing to disk. The store keeps full
  text (latest version, with a flag and the previous hash when a document
  was edited; see `store.revised()`) and a log of harvest runs; `store.load()`
  returns the accumulated corpus. `NarrativeCorpus` gains `to_frame(metadata=
  "promote"|"json"|"drop")`, `to_jsonl`/`from_jsonl` (no dependency),
  `to_parquet`/`from_parquet` (`io` extra), `from_frame`, `from_records` (any
  connector's output, including query-driven ones), and a provenance
  `manifest()` written next to every export, with `verify_manifest` to check a
  re-harvest against it.
- **New `puremacro.text`: tokenizer, segmentation and a sparse document-term
  matrix.** `build_dtm(texts, languages=...)` returns a `DocumentTermMatrix`
  whose `X` is a `scipy.sparse.csr_matrix` with a row-aligned `docs` frame;
  `.filter(min_df, max_df, max_features, terms)`, `.tfidf()` (scikit-learn's
  definition), `.aggregate(by="country", freq="MS")` (term-count panels),
  `.transform(new_texts)`, `.to_tidy()`, and `.save(dir)` writing `X.mtx` +
  `vocab.txt` + `docs.csv` for R (`Matrix::readMM`, quanteda, stm). The default
  `RegexTokenizer` keeps numbers and percentages ("25", "5.25%"; or masks them
  with `numbers="mask"`) and picks stop words from each document's language
  (en, es, pt, de, fr, it). Any `callable(text, language) -> list[str]` (spaCy,
  NLTK) can replace it. `NarrativeCorpus.dtm(...)` and
  `NarrativeCorpus.segment("paragraph"|"sentence"|callable)` build on it.
- `puremacro.llm`: one entry point for local language models (MLX, llama.cpp, Ollama and
  OpenAI-compatible servers) with a device-independent completion cache and memory-based model
  selection. MLX is selected on the iPad under Juno, where `mlx` and `mlx-lm` import; a
  completion there is not yet verified on a device.
  - `complete(prompt, *, model="default", engine="auto", mode=None, system=None,
    max_tokens=256, temperature=0.0, json_mode=False, timeout=120.0, base_url=None,
    cache=None)` and `chat(messages, **kw)`. `engine="auto"` takes the first available of MLX,
    llama.cpp, Ollama on port 11434 and OpenAI-compatible servers on ports 8080 (llamafile,
    llama-server) and 1234 (LM Studio); `base_url=` or `PUREMACRO_LLM_BASE_URL` names one
    server instead.
  - Modes `auto` (cache, then model, then store), `cache` (cache only; a miss raises `CacheMiss`,
    an unreadable cache `CacheUnavailable`; no engine, socket or model) and `off` (model only),
    from `mode=`, else `PUREMACRO_LLM_MODE`.
  - `CompletionCache` (SQLite, `~/.cache/puremacro/llm_cache.sqlite` or `PUREMACRO_LLM_CACHE`)
    keyed on the canonical model alias and the prompt before any chat template, so a row written
    by one engine answers the same call on any other; `export`, `merge` and `stats` to ship one.
    `export(since=...)` takes any ISO 8601 form (`Z`, offsets, fractional seconds) and compares
    in UTC; an answer that is not valid UTF-8 is returned but not stored; a cache path that cannot
    be read raises `CacheUnavailable` in cache mode rather than counting as a miss.
  - `complete()` refuses a `temperature` that is negative, not finite or not a number, and an
    `engine` name it does not know (`validate_engine_name` is public), before the cache is read.
  - A model ladder of three Qwen3 models (Apache-2.0): `qwen3-1.7b` (`small`), `qwen3-4b`
    (`default`, the reference model) and `qwen3-8b` (`large`). `pick_model()` sizes the model to
    the device's memory (`device_budget_gb()`), `ModelTooLarge` refuses a model that does not
    fit, and `PUREMACRO_LLM_MODEL` changes what `default` means. The four models of 4.8.1 keep
    their names.
  - The Qwen3 thinking is switched off on every engine and any `<think>` block left is removed
    (`strip_thinking`).
  - `report()` lists the host, memory, budget, engines, chosen model, cache and mode.
  - `pip install "puremacro[local-llm]"` now also installs `huggingface_hub`, through which
    llama.cpp fetches the registry's GGUF files, and installs `mlx-lm` only on Apple silicon
    (`platform_machine == 'arm64'`), so the extra no longer fails on an Intel Mac, where `mlx`
    has no wheel.
  - The `local_llm_uncertainty` notebooks and the `_tutor.py` copies still show the 4.8.1 model
    names (`qwen2.5-3b-instruct`, `qwen2.5:3b` and the tutor's `qwen2.5:3b-instruct`), which
    still resolve; they are synced in a separate change.

### Changed

- `narrative.topics.TfidfVectorizer` now builds its matrix sparse, so memory
  scales with non-zero cells rather than documents x terms. Its default
  tokenizer and its dense return value are unchanged (results match the
  previous version to 1e-14); pass `sparse_output=True` or `tokenizer=` to opt
  in. `NMF` accepts sparse input, and `DynamicTopicModel` uses it.
- `fetch.cdc_births_state` and `fetch.cdc_births_county` no longer default to a data directory outside the repository. `PUREMACRO_FERTILITY_ROOT` must name the directory that holds the natality extracts; when it is unset, `fetch_monthly`, `fetch_quarterly`, `fetch_yearly` and `fetch` raise `NatalityRootError`, which names the variable, and nothing is written. `build_subnational_panel` then builds without births, as it already did when the files were missing. The county loader's artifact paths are `nchs_artifact_path()` and `combined_artifact_path()` instead of module constants.
- The `fetch` HTTP helpers (`fetch._http.cached_get`, the FRED CSV reader, the CDC natality readers) send a `puremacro/<version>` User-Agent instead of the name of the monorepo the package came from.

- The score cache of `narrative.indices.llm_prob_kernel` now takes its path from
  `PUREMACRO_LLM_SCORES_CACHE` (the default, `~/.cache/puremacro/llm_scores.sqlite`,
  is unchanged). It used to read `PUREMACRO_LLM_CACHE`, which now names the
  completion cache of `puremacro.llm`. They are different caches with different
  tables, and one file for both would let each get in the other's way. If you had
  pointed `PUREMACRO_LLM_CACHE` at a score cache, set the new variable instead.

- The local narrative backends go through `puremacro.llm.complete`, so they read and fill the
  completion cache and follow the mode:
  - `LocalBackend`, `LocalProvider`, `get_default_backend` and `get_default_provider` default to
    `model="default"` (`qwen3-4b`; it was `qwen2.5-3b-instruct`), take a new `mode=` argument,
    and resolve no engine when they are built, so a cache-backed `score_llm` or
    `llm_prob_kernel` runs with no engine installed. `base_url` defaults to None:
    `PUREMACRO_LLM_BASE_URL`, then the default local servers.
  - They check `engine` and `mode` when they are built (`PUREMACRO_LLM_MODE` included): a
    misspelt name raises `ValueError` there, as the engine resolution of 4.8.1 did, and not at
    the first record.
  - `LocalBackend`'s `json_mode` now defaults to False: the prompts ask for a JSON array, which
    llama.cpp's object-only JSON mode cannot produce.
  - `LocalProvider.name` is `"local"` and its `model` the canonical alias, so the score cache of
    `llm_prob_kernel` is shared by every engine. Score-cache rows written by 4.8.1, keyed on the
    engine name, no longer hit.
  - `OllamaBackend` and `OllamaProvider` default to the registry alias `qwen3-4b`: the server
    still gets `qwen3:4b`, and the cache is shared with `LocalBackend("default")`. An explicit
    Ollama tag passes through unchanged and keys the cache on the tag.
  - `OllamaBackend` and `OllamaProvider` now default `base_url` to None, so
    `PUREMACRO_LLM_BASE_URL` applies to them as it does to `LocalBackend(engine='ollama')`;
    without the variable they still use the local Ollama server.
  - In `cache` mode `score_llm` raises `CacheMiss` / `CacheUnavailable` for a record the cache
    does not hold, instead of dropping it. `score_llm` also raises `ModelTooLarge` instead of
    dropping every record as malformed.
  - `get_default_backend` / `get_default_provider` look for engines with
    `puremacro.llm.engines.available_engines(base_url=…)` (in `cache` mode, nothing is probed).
    The print lines `[puremacro] local LLM backend: engine=…`,
    `[puremacro] local LLM provider: engine=…`, `[puremacro] using MockBackend (zero events).`
    and `[puremacro] using MockProvider.` are replaced by the install hint (when no engine is
    found) plus one status line,
    `[puremacro] local LLM: mode=<mode> engine=<name|cache|mock> model=<alias>`.
  - The example `puremacro.examples.narrative_local_llm` uses `"default"` and
    `ollama pull qwen3:4b`.
- `puremacro.narrative._local_engines` is now a compatibility re-export of `puremacro.llm`. Its
  `chat` reads the completion cache, and its `base_url` defaults to None
  (`PUREMACRO_LLM_BASE_URL`, then the default local servers) instead of Ollama's URL.
  `MODEL_ALIASES` is a read-only view of `puremacro.llm.MODELS` built from the registry, with
  exact GGUF file names instead of globs: adding an entry to it changes nothing, and a custom
  model id passes through to the engine unchanged. `HTTPEngine()` keeps its 4.8.1 defaults
  (Ollama API at http://localhost:11434).
- Local engines: MLX is chosen whenever `mlx` and `mlx_lm` can be imported, on any platform;
  llama.cpp offloads every layer to the GPU (`n_gpu_layers=-1`; it ran on the CPU) and loads
  registry files through `huggingface_hub`, which finds a downloaded file offline; `engine="auto"`
  also tries OpenAI-compatible servers on ports 8080 and 1234; and a server URL ending in `/v1`
  (LM Studio's documented form) is accepted.

### Fixed

- The MLX engine was never selected on iPadOS: it required `platform.system() == "Darwin"` and
  `platform.machine() == "arm64"`. Engines are now chosen by what can be imported.
- `score_llm`:
  - It aborted the whole corpus when an event's implementation date fell in the announcement
    quarter, or `horizon_quarters` was 0. Such events now get an empty implementation profile
    (implemented at the announcement), and any event `NarrativeEvent` refuses is dropped as
    malformed instead of raising.
  - Answers with `"sign": +1` (not valid JSON, but what the prompt's own schema line suggests)
    or wrapped in `{"events": [...]}` are now parsed, instead of yielding zero events.
- The llama.cpp ids of 4.8.1 were filename globs (`*Q4_K_M.gguf`), and the one for
  `qwen2.5-3b-instruct` matched no file on macOS and Linux, where matching is case-sensitive.
  Registry entries now name the exact file.

- `fetch.registry` listed the 67 narrative text connectors as `redistributable=True`, although their terms are marked `unaudited`. They are now `redistributable=False` until someone reviews them.

- `test_t2_f22_sw07_order2_repeated_solve_stability` compared its wall time with a fixed 0.20 s instead of going through `tests/_timing.budget`. CI's `PUREMACRO_TIMING_SCALE` therefore did not apply, and the test failed on Ubuntu with Python 3.11 in two PR runs on 2026-10-12 (0.48 s and 0.52 s; the other samples took 0.11 s).

- `fetch._http.cached_get` gains `params=` and `secret_params=`; an API key passed as a secret parameter is no longer written into the cache file name, the manifest, or HTTP error text.

- `harvest_narrative_corpus` fixes that changed what the narrative corpus contains:
  - In a fresh session `harvest_narrative_corpus` returned an empty corpus: it
    looked connectors up with `getattr(sources, module_name)`, which is `None`
    until the lazily loaded submodule has been imported, and skipped them
    silently. Connectors are now imported on demand.
  - Three registry entries named functions that do not exist (`us_federal_register`,
    `us_dod_contracts`, `uk_obr`) and harvested nothing; fixed.
  - Connectors that yield legacy `(date, text, url)` 3-tuples (the RSS/Atom
    helpers: `uk_hmt`, `uk_obr`, `de_bmf`, `fr_tresor`, `it_mef`, `eu_ecfin`,
    `ca_dof`, `us_treasury`, `us_dod_contracts`) were rejected by the schema
    check, so these sources were always empty. They are now accepted.
  - The per-record `doctype` that 33 connectors emit was ignored (harvest read
    `doc_type`); both are now read.
  - A failing connector was swallowed by a bare `except`. Failures are now in
    `corpus.harvest_report` and raise a warning (`on_error="warn"|"raise"|"ignore"`);
    the documents a connector yielded before failing are kept, as before.
  - Newly registered: `us_treasury`, `mas`, `macro_blogs`, `bluesky`; and, via
    `QUERY_SOURCE_REGISTRY` + `source_kwargs`, `gdelt`, `google_news`, `reddit`,
    `hackernews`, `local_csv`. Unknown source names now raise instead of being
    dropped silently.
  - Harvested `doc_id`s now also hash the record's explicit title (or `id`), and
    records sharing a key but differing in text all get text-qualified ids, so
    the ids do not depend on feed order (a reordered `us_warn` feed no longer
    shows up as revisions). A key with a single text keeps its plain id, which is
    what lets the store flag revisions; if a second text later appears under that
    key, both records get new ids and the old row stays in the store.

- `kalman_filter` / `kalman_smoother` with `diffuse_states=[...]` (exact-diffuse initialisation) returned wrong filtered and smoothed states, covariances and log-likelihood whenever a diffuse state's loading was not 1. Four defects, each now checked against an exact batch posterior and the closed-form diffuse log-likelihood (2,400 random 3-state / 2-observable models agree to ≤4e-7 relative error, previously wrong in most of them):
  - The diffuse-step update of the finite covariance scaled `K0 K0'` by `F_* / F_inf` instead of `F_*` (Koopman and Durbin 2000; Durbin and Koopman 2012, §5.2). The two agree only when `F_inf = 1`, as in the local-level `Z = 1` examples and tests. A local level with `Z = 2` gave a first-period variance of 0.0625 instead of `H / Z² = 0.25`.
  - Correlated measurement errors (non-diagonal `H`) were ignored during the diffuse phase, which processes a period's observations one at a time using only `H[k, k]`. Each period is now rotated onto the eigenvectors of `H` first.
  - A fully missing period during the diffuse phase did not carry the diffuse part through the transition (`P_inf ← T P_inf T'`). This changed the log-likelihood for a stationary diffuse state and the filtered states when `T` moves the diffuse direction.
  - The test for a diffuse direction was an absolute `F_inf > 1e-12`. With small loadings, a genuine diffuse direction fell below it; with large ones, the rounding residual of a resolved `P_inf` passed it and triggered spurious diffuse updates, sometimes with non-finite output. `P_inf` is now carried in factored form `A A'`, with one column per diffuse direction not yet pinned down by data. A diffuse update removes one column exactly, so no residual is left in a resolved direction. The cut-off compares `A'z` with its own rounding bound `|A|'|z|`, so it follows the loadings' units and keeps genuine diffuse directions however small their scale (nearly collinear loadings, or a diffuse component shrunk by `T`). Directions that `T` annihilates are dropped by a numerical-rank test on `T A`, with each state scaled to its rounding bound `|T| |A|`. A direction that `T` only shrinks therefore stays diffuse however far below the others it falls; before, `T = diag(1, 1e-16)`, or `diag(1, 1e-4)` over four missing periods, dropped it, which moved `a_filt` by up to 0.46 and `loglik` by 37. Once no column is left, the standard multivariate recursion (which also stores `F` and `K`) takes over. From there `kalman_smoother` uses its Durbin–Koopman recursion. Before, a positive rounding residual in `P_inf` could keep every period on the exact-diffuse path, and with it the smoother on its RTS step: this happened for 43 of 200 random 3-state models over 20 periods.

  The diffuse-period log-likelihood term `-0.5 (log 2π + log F_inf)` is unchanged, but later periods' `F_*` depend on the corrected `P_*`. **`loglik` therefore changes whenever `F_inf ≠ 1`**. Results are unchanged (to rounding) when every diffuse step has `F_inf = 1` (e.g. a local level with `Z = 1`), `H` is diagonal and no period in the diffuse phase is fully missing. No other library code passes `diffuse_states`. Tests in `tests/test_state_space_coverage.py` (`TestExactDiffuseNonUnitLoadings`, `TestExactDiffuseEdgeCases`).
- `kalman_smoother` with `diffuse_states=[...]` returned wrong `a_smooth` and `P_smooth` in every period that ended with part of the diffuse prior still unresolved. It ran the Rauch–Tung–Striebel recursion there on `P_pred` / `P_filt`, which hold only the finite part `P_*` of the covariance while the diffuse phase lasts (the diffuse part `P_inf` was never returned). Those periods now follow the exact initial smoother of Durbin and Koopman (2012, §5.3), the `r⁽⁰⁾, r⁽¹⁾, N⁽⁰⁾, N⁽¹⁾, N⁽²⁾` recursions, run backwards over the filter's observation-by-observation steps, including its rotation onto the eigenvectors of a non-diagonal `H`. They start from the Durbin–Koopman recursion's `r` and `N` at the end of the diffuse phase, with the `1/κ` terms at zero. The periods after the diffuse phase were already exact and are unchanged.
  - Checked against the batch GLS posterior of `α_1..α_T`, with zero prior precision on the diffuse indices of `α_1`. Before the fix:
    - A local linear trend with level and slope both diffuse was off by 0.18 in `a_smooth` and 0.17 in `P_smooth` at t = 0.
    - An AR(0.5) diffuse state loaded with `Z = 1.5` and `y[0]` missing was off by 0.18 and 2.6 at t = 0.
    - With `T` swapping two states and `y[0]` missing, it was off by 0.041 and 1.09 over t = 0, 1.
  - All three now agree to ≤2e-15, as does a 3-state model with correlated `H` and a value missing inside the diffuse phase (≤2e-14 in `P_smooth`).
  - Over 2,898 random models (1–4 states, 1–3 observables, random diffuse subsets, a quarter of the values missing, diagonal or correlated `H`), the median relative error is 5e-14 and the 99th percentile 5e-8.
  - Ten models exceed 1e-6 against a 60-digit evaluation of the same posterior (worst 3e-2 in `P_smooth`, at most 1e-5 in `a_smooth`). In nine of them some observation resolves a diffuse direction while barely loading on it: `F_inf` is at most 2e-4 of `(|z|' sd)²`, where `sd` holds the standard deviations of the `P_inf` entering the period, against a median of 1 over all the models (the tenth, at 2e-6, has 1.5e-2). The backward pass then cancels terms of order `1/F_inf²`.

  Results are unchanged when the diffuse prior is fully resolved within the first period, as in a local level with `Z = 1` (`python -m puremacro.examples.exact_diffuse_kalman`), and without `diffuse_states` (bit-identical). Tests in `tests/test_state_space_coverage.py` (`TestExactDiffuseSmoother`).
- The exact-diffuse filter took a period's observations in the order given, so a diffuse step that barely loads the direction it resolves could come before an observation that loads it strongly. That step leaves a large `P_*` for the next observation to cancel, and the smoother's expansion in `1/κ` then cancels terms of order `1/F_inf²`. Results depended on the order of the observables. In the test model a row loads the last diffuse state with weight 1e-5 and comes first; filtering with the rows swapped moved `P_filt` by 8e-8, and `a_smooth` was 6e-7 off the GLS posterior.
  - Within each diffuse period the filter now takes first, among the observations that still carry diffuse information, the one with the largest `F_inf / F_*`. That ratio is free of the units of `y`, of the states and of the scale of the diffuse factor. The order does not change the exact result.
  - Over the 2,898 random models of the previous entry, models above 1e-6 fall from ten to five and the worst error from 3e-2 to 2e-3. Each of the five left has a weak diffuse step in every order of its observables (smallest `F_inf / F_*` between 2e-8 and 1e-4, the same in every order): its diffuse loadings are nearly collinear within a period.
  - Reversing the observables moves the log-likelihood by at most 5e-8 relative (2e-7 before).
  - Filter output is bit-identical for 2,238 of 3,000 random models and within 1.4e-12 for 99%.
  - Tests: `test_invariant_to_the_order_of_the_observables`, `test_weakly_loaded_observation_before_a_strong_one`.

### Documentation

- New guides, each with a Spanish twin and with code blocks that CI runs: `docs/dp.md` (declarative dynamic programming: VFI, EGM and DC-EGM, continuous time, Krusell-Smith, and what `dp` does not cover yet), `docs/micro_data.md` (the source registry, the ACS, CPS, SCF and ENIGH providers, design-based standard errors and the SCF-calibrated Aiyagari example) and `docs/text_analysis.md` (corpus storage and export, the tokenizer and the sparse document-term matrix). README, README.es and the site navigation link them. `docs/CREDENTIALS.md` now names the microdata providers as the `census` key's consumers.
- The JOSS paper (`paper/paper.md`) describes `puremacro.dp`, the survey microdata and the text tools, uses the SCF example as its worked case, and adds dolo to the state of the field. It stays within the 1,750-word limit (1,730 by `tests/test_paper.py`'s count).
- New guide `docs/local_llm.md` (Spanish twin `docs/es/local_llm.md`): what runs where, installation per platform (iPad, Mac, Windows, Linux, Colab), the model ladder with licences, the modes, shipping a completion cache, the environment variables, the iPad notes and `report()`. Its model-free code blocks run in CI. `docs/tablet.md` gains a section "Local models on the iPad"; README, README.es and the site navigation link the guide, and the README's local-model section now describes `puremacro.llm`.

- Course lesson T02_F (Gertler-Karadi 2011, English and Spanish) reports percent deviations from `to_frame(units="pct")`. It multiplied level deviations by 100 and labelled them %, so it printed capital at -28.66% and bank net worth at -67.88% on impact where the true figures are -5.06% and -49.15% (`docs/ADVISORY.md`, 2026-09-30). The scorecard, the policy table, the three figures, the figures quoted in the text and the lesson's own checks are corrected; the spread in annualised basis points and Q, whose steady state is 1, were already right. The lesson now installs `puremacro>=4.4,<5`, the first versions with `units="pct"`. `tests/test_notebooks/test_curso_t02f_units.py` pins the printed values and, under `-m slow`, re-runs the lesson.
- `docs/monetary_transmission.md` (English and Spanish) documents `MonetaryTransmissionSimulator` and the HANK vs. RANK browser lab, which used to share a page with the removed trade simulator. The page prints the simulator's own values, attributes the 60–80% indirect share to Kaplan, Moll and Violante (2018) and describes the lab as plain JavaScript (d3 and KaTeX from a CDN); recipe 1's shock is a 25 bp quarterly (100 bp annualized) hike.
- README, README.es, ARCHITECTURE.md, `docs/index.md`, `docs/VALIDATION.md` (110 cases in 15 subsystems) and the JOSS paper describe the library without the removed material. `docs/STRUCTURAL_VALIDATION_STATUS.md` points its MATLAB caveat at `tools/reference_validation/README.md`.

### Internal

- `tools/research_guard.py` and `tests/test_no_research_content.py`: a guard that scans the tracked tree and built sdists or wheels for material that does not belong in the public repository (generic forbidden directory prefixes, the SHA-256 digests of moved paths and of a private name list, and machine-specific paths such as a user's home directory), with the tests that pin its behaviour. The guard names no project; `python tools/research_guard.py --git` runs it locally.
- `playground/dist/`, a stale JupyterLite build, is no longer tracked; GitHub Pages rebuilds it.
- `pyproject.toml` and `MANIFEST.in` drop the removed packages, their data files and the `gpu` extra (its torch and `mlx` pins served only the removed solvers; the MLX engine of `puremacro.llm` comes with the `local-llm` extra, and the `accel`/`backend` and `cuda` extras are unchanged).
- The notebook integrity test restores matplotlib's `rcParams` after exercising `_nbstyle.apply_style()`, so `tests/test_dsge/test_diagnostics.py::test_check_unit_circle_plot` passes in any test order.
- `.gitignore` no longer lists LaTeX intermediates of a directory that is not part of the repository.
- `tools/release_check.py` reads a PEP 440 pre-release heading (`## 5.0.0a1`) from this file, so the version-sync gate compares `5.0.0a1` across the five version-bearing files instead of reporting `5.0.0` against `5.0.0a1`; `tests/test_release_check.py` pins every reader on a pre-release value. The deprecation-target tests for the `puremacro.lp.garch_utils` and `SigmaObject` shims compare `packaging` versions, so their 5.0.0 removal stays in the future of 5.0.0a1.

## Releases before 5.0

The complete changelog of earlier releases is not part of the public repository. Development before 0.92.0 (2026-07-20), the first public release, was private.

- **4.8.1** (2026-10-11) — Cross-platform cartridge verification (NPZ creator-OS metadata no longer causes false checksum mismatches), shipping the cross-country panel builders, portable `DataFrame.attrs` provenance and v2 cartridges prepared for 4.7.0 and 4.8.0.
- **4.8.0** (2026-10-10; not on PyPI) — Portable panel provenance: `DataFrame.attrs` survive storage and cartridge verification. Legacy v1 cartridge verification failed on the Windows CI targets and the release workflow blocked the upload; 4.8.1 carries the correction.
- **4.7.0** (2026-10-07; not on PyPI) — Cross-country macro panels: ten one-call panel builders for the World Bank (WDI), the OECD (annual and quarterly national accounts, short-term indicators, labour force and vacancies), Eurostat, the IMF data portal, the BIS, ECB/FRED rates and yields, ILOSTAT, the Penn World Table and the Maddison Project, through urllib with an on-disk cache and one `(code, date)` contract; `splice_sources`; and repairs to the BIS, Eurostat, ILOSTAT and OECD-LFS readers.
- **4.6.0** (2026-10-04) — The validation gallery runs inside Pyodide, the Smets–Wouters Table 2 (training-sample) marginal-likelihood computation is implemented and checked against Dynare 8, and the opt-in slow, reference and replication tests gate every release.
- **4.5.0** (2026-10-03) — Honest convergence and corrected numbers: the wrong-number and false-success known issues of 4.4.0, each checked against an independent reference.
- **4.4.0** (2026-10-02) — Empirical-to-structural estimation (`puremacro.structural`: `MomentTargets`, `fit_structural`, canonical LP designs with joint HAC covariance), independent research benchmarks, a matrix-free stacked Newton-Krylov solver for perfect-foresight transitions (`dsge.stacked_newton`), and the verified fixes of the 30 September library review.
- **4.3.0** (2026-09-20) — Audited structural solvers: `docs/STRUCTURAL_VALIDATION_STATUS.md` states each surface's acceptance contract, evidence and limits, with five models checked against live Dynare 7.0 at orders two and three.
- **4.2.0** (2026-09-18) — Dual-theme grayscale card styling for every notebook (`_nbstyle.py`, synchronized between the showcase and course trees) and a polish pass over the notebook corpus.
- **4.1.1** (2026-09-17) — Restores the fallback exception handling that 4.1.0's explicit exception tuples had dropped across 142 files.
- **4.1.0** (2026-09-17; not on PyPI) — Latin America real-time nowcasting with the Bańbura–Modugno news decomposition, interactive Double ML (IRM and DML-IV), the HANK vs. RANK monetary transmission simulator with its browser lab, showcase notebooks 59 and 60, and three wrong numbers fixed (HP-filtered theoretical moments, LP-IV standard errors, stationary distributions of nearly decomposable chains).
- **4.0.1** (2026-09-16) — An attribution notice for the third-party data that 4.0.0 had started redistributing; no code change.
- **4.0.0** (2026-09-16) — MATLAB removed from the reproduction path: every documented result reproduces with Python and the installed package alone, and the `matlab/` companion toolbox is gone (a major release because public API was removed).
- **3.4.0** (2026-09-15; not on PyPI) — `.to_typst()` export parity across result objects, implicit continuous-time HJB and adjoint KFE solvers (`vfi.hjb_achdou`), hardened multi-constraint OccBin, Double/Debiased Machine Learning (`causal.dml`), and Latin American real-time connectors.
- **3.3.0** (2026-09-13) — Continuous dynamic programming (Chebyshev collocation, FEM Galerkin, shape-preserving splines, Smolyak grids, DC-EGM), Deep Macro PINNs, quantitative spatial general equilibrium (Allen–Arkolakis 2014), and `pyarrow`/`openpyxl` moved to the `io` extra so the base install resolves inside Pyodide.
- **3.2.1** (2026-09-11) — Documentation overhaul and the live MkDocs site on GitHub Pages; PyPI metadata aligned.
- **3.2.0** (2026-09-11) — Applied macroeconomist showcase suite (notebooks 47–50) and the global climate, energy, commodity and financial data ecosystem with modular panel builders.
- **3.1.0** (2026-09-10) — Optimal policy under discretion and commitment, DSGE-VAR, news shocks, nonlinear particle filtering and Markov-switching DSGE; the Dynare macro processor, live IRF parameter sliders and rank identification criteria.
- **3.0.0** (2026-09-09; not on PyPI) — Exact analytic likelihood gradients, pure-Python HMC/NUTS Bayesian estimation, and the heterogeneous-agent (HANK) sequence-space bridge.
- **2.9.0** (2026-09-09) — Dynare parity, Tier 3: the `stoch_simul` filtering and simulation surface, extended path (Fair–Taylor 1983), forecasting and shock decompositions, the parity dashboard and CLI.
- **2.8.0** (2026-09-08) — Dynare parity, Tier 2: order-3 perturbation and pruning, deterministic transitions and a semismooth Newton MCP solver, multi-constraint OccBin with a piecewise Kalman filter, SMC and particle filtering, nonlinear Ramsey policy and BGP detrending.
- **2.7.0** (2026-09-08) — Dynare parity, Tier 0: a native macro preprocessor, recursive-descent AST parser, symbolic differentiation with common-subexpression elimination and a fast Sylvester solver replace the regular-expression front end.
- **2.6.0** (2026-09-07) — Two tracks: a Dynare `.mod` file becomes estimable (posterior, diagnostics, robust steady state and policy regimes), and native regression and inference replace `statsmodels` for the calls an applied notebook makes.
- **2.5.0** (2026-09-06) — Spatial econometrics phases 2–4 (spatial models, panels and local projections, spillover-robust DiD), the wavelet package split, `simex_ols` and runtime helpers; `wavelet_coherence` now discards boundary-contaminated coefficients by default.
- **2.4.0** (2026-09-05) — `puremacro.spatial`, phase 1: spatial weights, Moran's I and Geary's C, Conley and space-time HAC, Conley errors for panel local projections, and shift-share IV with Adão–Kolesár–Morales errors.
- **2.3.1** (2026-09-05) — Correctness release after a full audit of 2.3.0: the DSGE engine solves lead/lag systems, decision-rule control rows are aligned, Blanchard–Kahn violations raise, and the Gertler–Karadi responses are the documented ones.
- **2.3.0** (2026-09-05; not on PyPI) — Narrative sign restrictions, honest DiD sensitivity bounds, smooth local projections, nonlinear sequence-space HANK transitions, the Gertler–Karadi (2011) banking-frictions DSGE and BVAR with stochastic volatility, each with a bilingual guide.
- **2.2.0** (2026-09-05) — The `puremacro-dynare` command-line tool, the sequence-space Fake News algorithm and targeted transfer simulations, and the frontier Dynare showcase notebooks (41, 42).
- **2.1.0** (2026-09-04) — FEVD and historical shock decomposition, OccBin occasionally binding constraints, deterministic perfect-foresight simulation and the Bayesian DSGE estimation pipeline.
- **2.0.1** (2026-09-04) — Packaging and PyPI release update for 2.0.
- **2.0.0** (2026-09-04) — A unified result architecture (`LPResult` and `.plot()`/`.to_latex()`/`.to_typst()`/`.to_markdown()` on every result object), standardized estimator signatures (`lags`, `horizon`, `ci`), the Dynare-compatible solver with second-order pruning, and full type completeness.
- **1.10.0** (2026-09-03) — Three estimator fixes affecting every earlier release: `dols` omitted the contemporaneous term, `fm_ols` dropped the lag-0 block of the long-run covariance, and `kleibergen_paap_f` was `n²` too large.
- **1.9.0** (2026-09-02) — `ana_by_activity` (annual value added and labour input by ISIC activity, with `chain_volume` for additive volumes), the ISIC breakdown of the quarterly labour block, and the seven-estimator fix that reached every release from 0.92.0 to 1.8.0.
- **1.8.0** (2026-08-23) — The longest quarterly national accounts panel the sources support (Spain to 1970, Japan to 1955), spliced onto the OECD spine with the evidence for every join reported.
- **1.7.0** (2026-08-23) — Real-time data: an accessor that returns the archive, six vintage providers behind one panel, the news-vs-noise test as a first-class estimator; cross-country revision work grows from 2 to 34 usable countries.
- **1.6.0** (2026-08-23) — The capital stock the production functions assumed, a panel that imports without a network stack, and two silent data defects fixed (one variable carrying two scales, five countries publishing the wrong kind of volume).
- **1.5.0** (2026-08-22) — Two hours series wrong by an order of magnitude, a labour fetch that no longer drags the national accounts behind it, and the retirement of a route that reached only a third of the block.
- **1.4.0** (2026-08-21) — The labour input the quarterly national accounts measure, which `qna_panel` could not reach.
- **1.3.2** (2026-08-20) — A DSGE solver that no longer depends on the BLAS, an Aiyagari routine that solves, a suite that passes on Windows, and the worked national-accounts example.
- **1.3.1** (2026-08-20) — The `requests`-free fetch path that 1.3.0 was tagged one commit too early to include, and a cartridge path that survives pandas 3.
- **1.3.0** (2026-08-20) — All three measurements of GDP in one panel, availability-driven country discovery, one price reference year across countries, and the expenditure, output and income identities scored inside their own flows.
- **1.2.0** (2026-08-19) — Capability detection, browser networking, portable data cartridges, resumable estimation and a DSGE model DSL, plus a corrected Klein policy function.
- **1.1.0** (2026-08-16) — Quarterly national accounts historical vintages, Mankiw–Shapiro news-vs-noise testing, and multilingual narrative harvesting across 50+ institutional sources.
- **1.0.0** (2026-08-15) — GARCH-MIDAS volatility, business-cycle bandpass filters and Beveridge–Nelson decomposition, weak-IV robust Anderson–Rubin sets for LP-IV, panel LP block bootstrap with sup-t bands, a 73-case validation gallery across 13 subsystems, and the Rosetta Stone syntax cheatsheet.
- **0.96.0** (2026-08-08; not on PyPI) — Six-package base install (`requests` and `pyarrow` promoted out of the extras), a local-install, Spanish-only course companion, sup-t simultaneous IRF bands, the Galí (1999) and Kilian (2009) replications, and the district-level Beige Book uncertainty index.
- **0.95.0** (2026-07-24) — Published to PyPI without a changelog entry of its own; the work since 0.92.0 is recorded under 0.96.0.
- **0.94.0** (2026-07-24) — Published to PyPI without a changelog entry of its own; see 0.96.0.
- **0.93.0** (2026-07-24) — Published to PyPI without a changelog entry of its own; see 0.96.0.
- **0.92.0** (2026-07-20) — First public release (the version had been cut privately on 2026-05-30): free local-LLM backends for narrative scoring (`score_llm` and `llm_prob_kernel` run at $0 locally through MLX, llama.cpp or an Ollama server), on top of the estimators, data layer and `puremacro.vfi` built during the private period.
