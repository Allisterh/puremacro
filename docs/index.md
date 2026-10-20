> 🇬🇧 English · 🇪🇸 [Español](es/index.md)

# puremacro

**Macroeconometrics and structural modeling on the core scientific-Python stack.**

[![PyPI Version](https://img.shields.io/pypi/v/puremacro.svg)](https://pypi.org/project/puremacro/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![JupyterLite Playground](https://img.shields.io/badge/JupyterLite-Live%20IDE-orange.svg)](https://jalonso1979.github.io/puremacro/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Estimate policy effects, compare economic models, and export results for a
research note or policy brief. Start with Python 3.11 or later:

```bash
python -m pip install puremacro
```

| Start with your task | First steps |
|---|---|
| **Analyze data** | Estimate a shock response with the [quickstart](quickstart.md), then explore [data sources](micro_data.md) and [research workflows](research_workflows.md). |
| **Simulate policy** | Declare a household model with [dynamic programming](dp.md), study [monetary transmission](monetary_transmission.md), or build a [trade counterfactual](trade_accounting.md). |
| **Learn or teach** | Choose a [notebook learning path](notebooks.md) or use the [bilingual course and interactive labs](https://github.com/jalonso1979/puremacro/tree/main/curso). The course has its own pinned package version. |

**Release status, checked 20 October 2026:** the latest published
[PyPI release](https://pypi.org/project/puremacro/) is **5.2.0**. This source tree
and its documentation include **5.4.0 development work**. Additions recorded
under 5.3.0 and 5.4.0 in the
[changelog](https://github.com/jalonso1979/puremacro/blob/main/CHANGELOG.md)
require the matching source checkout; they are not included in the published
5.2.0 package.

## What is puremacro?

`puremacro` brings econometric estimators and quantitative macroeconomic models
together using NumPy, SciPy, pandas and Matplotlib, with no compiled extension of
its own. A local workstation installation is the supported target. The numerical
core also runs under Pyodide / WebAssembly; browser and tablet workflows remain
best effort, with limits on file formats, network access and computation. See the
[runtime guide](tablet.md) and the [structural validation status](STRUCTURAL_VALIDATION_STATUS.md)
for the scope of the evidence.

### Key Subsystems

1. **Structural DSGE & Dynare Parity**:
   - **Native Dynare `.mod` Parser & `puremacro-dynare` CLI**: Parse and solve standard `.mod` files directly in pure Python.
   - **Likelihood Scores**: An analytic Kalman recursion for structural parameters in the supported first-order stationary likelihood; structural derivative inputs may use finite differences. NUTS also offers numerical and user-supplied gradients; the estimation interface determines gradient selection and fallback ([Guide](dsge_estimation.md)).
   - **Pure-Python HMC & NUTS**: Hamiltonian Monte Carlo with dual-averaging step-size adaptation, diagonal mass adaptation, and split-$\hat{R}$ diagnostics without rank normalization.
   - **Optimal Discretionary Policy vs Commitment**: Dennis (2007) Riccati matrix iterations for Markov-perfect discretionary policy vs timeless-perspective commitment, decomposing inflation and stabilization biases.
   - **DSGE-VAR Hybrid Modeling**: Del Negro & Schorfheide (2004) DSGE-VAR($\lambda$) prior optimization and structural identification.
   - **Anticipated News Shocks**: State-space companion augmentation with nilpotent shift operators preserving Blanchard-Kahn determinacy.
   - **Nonlinear Particle Filtering & MS-DSGE**: Bootstrap and auxiliary particle filters with stochastic volatility, and Foerster et al. (2016) Markov-switching rational expectations solutions with closed-form analytical GIRFs.
   - **Perturbation and Occasionally Binding Constraints**: Pruned perturbation for unconstrained models and piecewise-linear OccBin for binding regimes ([Guide](dsge_build.md)). The development `solve_pruned_occbin` supports linear OccBin, including anticipated shocks, and slack second-order paths with one initial innovation; nonlinear binding episodes and nonlinear anticipated innovations raise explicit errors ([Supported limits](dsge_higher_order.md)).

2. **Heterogeneous-Agent Models (Declarative DP & HANK)**:
   - **Household Models as Equations**: Declare states, choices, rewards and constraints with `dp.Model`, then select VFI, EGM, continuous-time HJB or a supported Krusell–Smith solve ([Guide](dp.md)).
   - **Sequence-Space HANK** (Auclert, Bardóczy, Rognlie & Straub 2021, *Econometrica*): General equilibrium incomplete-markets models solved in $\mathcal{O}(T^3)$ sequence space.
   - **Fake News Algorithm**: Fast $\mathcal{O}(T^2)$ computation of sequence-space consumption Jacobians via expectation vectors and cumulation identities.
   - **HANK Sequence-Space Bridge**: Declare `hetagent_block` directly inside standard Dynare `.mod` files.
   - **Targeted Fiscal Transfers**: Dynamic consumption IRFs and cumulative fiscal multipliers across wealth deciles.
   - **Continuous-Time HJB**: Achdou et al. (2022) **implicit** upwind finite-difference M-matrix schemes, the adjoint Kolmogorov forward equation for the stationary density, and continuous-time Aiyagari general equilibrium.

3. **Econometric Engines & Local Projections**:
   - **Unified `LPResult`**: Standardized Local Projections (`lp_hac`, `lp_iv`, `lp_state_dep`, `panel_lp`) with Newey-West HAC, fixed-$b$, and Driscoll-Kraay standard errors.
   - **Structural VARs & FAVAR**: Cholesky, Blanchard-Quah, Sign Restrictions, Narrative Sign Restrictions, Proxy/External Instruments, Max-Share/News, and Factor-Augmented VAR.
   - **Modern Difference-in-Differences**: Staggered adoption estimators robust to treatment heterogeneity (Callaway & Sant'Anna, Sun & Abraham, Borusyak-Jaravel-Spiess, Synthetic DiD, Honest DiD).
   - **Spatial Econometrics & Global VAR**: Conley spatial HAC, spatial autoregressive models (`sar`, `sem`, `sdm`), spatial local projections, and the Pesaran Global VAR (`var.gvar`).

4. **Applied Macroeconomic Policy Suites**:
   - **Central Bank Policy Stance & Fan Charts**: Real-time monetary policy stance evaluation against counterfactual Taylor rules, multi-period fan chart projections, and statement sentiment extraction ([Guide](notebooks.md)).
   - **Real-Time Nowcasting & News Decomposition**: Giannone-Reichlin-Small mixed-frequency dynamic factor models (DFM), handling asynchronous ragged-edge data vintages and release-day news attribution ([Guide](nowcast.md)).
   - **Macroprudential Risk & Financial Stability**: Adrian-Boyarchenko-Giannone Growth-at-Risk (GaR) conditional skew-$t$ densities, and Diebold-Yilmaz systemic connectedness networks ([Guide](notebooks.md)).
   - **Fiscal Policy & Sovereign DSA**: Multi-method fiscal multipliers (SVAR vs LP vs Narrative LP-IV) and stochastic sovereign Debt Sustainability Analysis fan charts under macro-climate stress scenarios.

5. **Global Macro Data Ecosystem & Modular Panels**:
   - **Survey Microdata**: ACS, CPS, SCF and ENIGH readers retain each source's available survey design; a shared `MicroFrame` interface provides supported statistics and uncertainty estimates ([Guide](micro_data.md)).
   - **Emissions & Climate Data**: World Bank WDI and OECD SDMX greenhouse gas accounts across countries and economic sectors ([Guide](data_ecosystem.md)).
   - **Energy Balances & Commodities**: Primary energy consumption, clean energy transition shares, and expanded World Bank Pink Sheet commodity benchmark suites ([Guide](data_ecosystem.md)).
   - **International Financial Stability**: Sovereign yield curves (10Y/2Y), central bank policy rates, BIS credit-to-GDP gaps, and real property prices ([Guide](data_ecosystem.md)).
   - **Modular Panel Builders**: One-call constructors `build_climate_panel` and `build_financial_panel` with automated frequency harmonization (M$\to$Q, A$\to$Q).

6. **Causal ML, Continuous-Time Heterogeneous Agents & Regional Real Time**:
   - **Double / Debiased Machine Learning**: Chernozhukov et al. (2018) partially linear regression with Neyman-orthogonal Robinson scores, $K$-fold cross-fitting, and pure-NumPy penalized learners — lasso and ridge only, no elastic net.
   - **Implicit HJB, Adjoint KFE & Continuous Aiyagari GE**: Sparse M-matrix policy iteration, a stationary density normalized with trapezoid quadrature weights so it integrates to one on non-uniform grids, and a general-equilibrium loop whose `converged` flag means $|K^s - K^d| \le$ `tol_ge`.
   - **Multi-Constraint OccBin**: $M \ge 2$ simultaneous occasionally binding constraints, with a regime bitmask (bit $k$ set when constraint $k$ binds) and non-convergence reported as `converged=False` plus a warning naming each reason ([Guide](dsge_build.md)).
   - **Montiel Olea-Pflueger Weak-IV Inference**: Effective $F$-statistic critical values from the closed form $\chi^2_{K,\,1-\alpha}(K/\tau)/K$, and multi-instrument Anderson-Rubin sets inverted exactly rather than on a grid ([Guide](lp.md)).
   - **Latin American Real-Time Connectors**: Banco de México, INEGI, Banco Central do Brasil and Banco Central de Chile. These four sources publish only the current edition, so their vintage date is the local *snapshot* date and revision history accumulates only across repeated captures on different days.

7. **International Trade and MRIO** — restored in the published 5.2.0 package:
   - **Policy Counterfactuals**: Multi-country, multi-sector trade models with producer/purchaser accounting, explicit fiscal receipts and an explicit equilibrium closure ([Guide](trade_accounting.md)).
   - **Reproducible OECD Data**: The bundled OECD ICIO table (2023 edition, reference year 2019) covers the 19 G20 member countries plus the rest of the world and 20 ISIC sections, with a rebuild script and source attribution ([Data and models](spatial_and_trade_ge.md)).
   - **Household Incidence**: Static EV/CV welfare comparisons conditional on supplied purchaser prices, income changes and fiscal transfers; household demand does not feed back into the trade equilibrium ([Guide](trade_distributional.md)).

8. **Publication-Grade Reporting**:
   - Zero-dependency, camera-ready table export directly to **LaTeX** (`.to_latex()`), **Typst** (`.to_typst()`), and **Markdown** (`.to_markdown()`), complete with standard errors and significance stars ([Guide](reporting.md)).

9. **Running Anywhere**:
   - A Pyodide-compatible numerical core for browser notebooks, with best-effort tablet and browser support ([Guide](tablet.md)), Google Colab offloading helpers (`runtime.colab`), chunked long-run execution (`longrun`), and portable `.pmz` data cartridges (`pocket`).

---

## Installation

The command at the top installs the core package. Optional tools add file-format
support and notebook utilities:

The parquet and Excel readers (the course's ENOE lessons, `build_all`, the shock atlas) need the optional file-format engines:

```bash
pip install "puremacro[io]"
```

Or install with full notebook tools:

```bash
pip install "puremacro[notebooks]"
```
