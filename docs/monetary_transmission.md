> 🇬🇧 English · 🇪🇸 [Español](es/monetary_transmission.md)

# Monetary Transmission Simulator: HANK vs. RANK

`puremacro.models.MonetaryTransmissionSimulator` runs a side-by-side simulation of **Heterogeneous-Agent New Keynesian (HANK)** and **Representative-Agent New Keynesian (RANK)** economies, featuring the **Kaplan, Moll, and Violante (2018, *American Economic Review*)** direct versus indirect transmission decomposition across Marginal Propensity to Consume (MPC) wealth deciles. It is built on `solve_hank_sequence_space` ([Sequence-Space HANK](models.md)) and returns frozen dataclass containers (`MonetaryTransmissionResult`) with a multi-format presentation suite (`.summary()`, `.plot()`, `.to_markdown()`, `.to_latex()`, `.to_typst()`). The course site ships a browser lab for the same comparison (section 3).

---

## 1. Monetary & Macroprudential Transmission Simulator (`MonetaryTransmissionSimulator`)

### 1.1 Theoretical Framework: HANK vs. RANK

Textbook Representative-Agent New Keynesian (RANK) models dictate that monetary policy transmits almost exclusively via the **direct intertemporal substitution channel**: identical unconstrained households adjust savings along a single Euler equation ($c_t = \mathbb{E}_t c_{t+1} - \frac{1}{\gamma} (i_t - \mathbb{E}_t \pi_{t+1})$).

In contrast, Heterogeneous-Agent New Keynesian (HANK) models (Kaplan, Moll, and Violante 2018; Auclert et al. 2021) incorporate uninsurable idiosyncratic productivity risk and borrowing constraints. A large share of households are **hand-to-mouth** (either poor hand-to-mouth with zero net worth, or wealthy hand-to-mouth with illiquid assets), exhibiting high marginal propensities to consume (MPCs) out of current income.

Consequently, in the quantitative HANK model of Kaplan, Moll, and Violante (2018), the **indirect general equilibrium channel**—transmitted through labor demand, aggregate output, and disposable income—drives the dominant share of consumption movements.

### 1.2 Kaplan-Moll-Violante (2018) Sequence-Space Decomposition

Using the sequence-space Jacobian methodology (Auclert, Bardóczy, Rognlie, and Straub 2021), the aggregate consumption response vector $d\mathbf{C} \in \mathbb{R}^T$ over horizon $T$ is decomposed into:

$$d\mathbf{C} = \underbrace{\mathbf{J}^{C, r} \, d\mathbf{r}}_{\text{Direct Channel (Substitution)}} + \underbrace{\mathbf{J}^{C, Y} \, d\mathbf{Y}}_{\text{Indirect Channel (GE Income)}}$$

where $\mathbf{J}^{C, r} \equiv \frac{\partial \mathbf{C}}{\partial \mathbf{r}}$ and $\mathbf{J}^{C, Y} \equiv \frac{\partial \mathbf{C}}{\partial \mathbf{Y}}$ are $T \times T$ sequence-space Jacobian matrices.

In general equilibrium, the New Keynesian Phillips Curve and Taylor rule define:
$$d\boldsymbol{\pi} = \mathbf{K}_\pi d\mathbf{Y}, \quad d\mathbf{r} = \mathbf{M}_{r, Y} d\mathbf{Y} + d\boldsymbol{\epsilon}$$

Substituting into the household block yields the sequence-space general equilibrium solve:
$$\left( \mathbf{I} - \mathbf{J}^{C, Y} - \mathbf{J}^{C, r} \mathbf{M}_{r, Y} \right) d\mathbf{Y} = \mathbf{J}^{C, r} d\boldsymbol{\epsilon}$$

| Dimension | RANK Economy | HANK Economy |
|---|---|---|
| **Market Structure** | Complete asset markets | Incomplete markets, borrowing constraint $a' \ge 0$ |
| **Household MPCs** | Uniform across deciles: $1 - \beta \approx 0.015$ | Steep gradient: Decile 1 ($>0.40$) to Decile 10 ($<0.06$) |
| **Direct Channel Share** | Identically $100\%$ | $20\% - 40\%$ in Kaplan, Moll, and Violante (2018); $85.0\%$ in this simulator (Recipe 1) |
| **Indirect Channel Share** | Identically $0\%$ | $60\% - 80\%$ in Kaplan, Moll, and Violante (2018); $15.0\%$ in this simulator (Recipe 1) |
| **Propagation Core** | Intertemporal Euler equation | General equilibrium labor income multiplier |

The literature shares are those of Kaplan, Moll, and Violante (2018): indirect effects account for 80 percent of the consumption response over the first year in their baseline and for 60 percent when government debt absorbs most of the fiscal imbalance in the short run (their Tables 7 and 8). They are not this simulator's output. In Recipe 1 below, `MonetaryTransmissionSimulator` prints `indirect_share_hank=15.0%`, the indirect channel's share of the date-0 consumption response, so the direct channel carries the other 85.0 percent: the simulator's stylised calibration gives a smaller indirect share than the quantitative HANK model in that paper.

### 1.3 Python Usage & Transmission Recipes

#### Recipe 1: Monetary Policy Tightening (25 bps Hike)
```python
from puremacro.models import MonetaryTransmissionSimulator

# Initialize simulator with continuous asset grid parameters
sim_mon = MonetaryTransmissionSimulator(
    beta=0.985,
    gamma=1.0,
    r_ss=0.01,
    phi_pi=1.5,
    kappa=0.1,
    n_a=50,
    a_max=30.0,
)

# Steady-state aggregate MPC
print(f"HANK Steady-State MPC: {sim_mon.steady_state_mpc:.4f}")

# Simulate a 25 bps rate hike with persistence rho=0.7 over 40 quarters
res_mon = sim_mon.simulate_rate_shock(magnitude=0.0025, rho=0.7, T=40)

print(repr(res_mon))
# Output: MonetaryTransmissionResult(shock_type='rate', magnitude=+0.0025, T=40, mpc_hank=0.1027, mpc_rank=0.0150, indirect_share_hank=15.0%)

# Display comprehensive side-by-side summary
print(res_mon.summary())

# Extract peak responses
peaks = res_mon.peak_responses()
print(f"HANK Peak Output Contraction: {peaks['output_hank']*100:.3f}%")
print(f"RANK Peak Output Contraction: {peaks['output_rank']*100:.3f}%")

# 4-panel publication visualization (IRFs, MPC ladder, KMV channels)
fig = res_mon.plot(kind="all", figsize=(12.0, 8.5))
fig.savefig("monetary_transmission_hank_rank.png", dpi=300)
```

#### Recipe 2: Targeted Balance-Sheet & Fiscal Transfers
```python
# Simulate a targeted cash transfer to liquidity-constrained hand-to-mouth households
res_tf = sim_mon.simulate_balance_sheet_intervention(
    amount=1.0,
    target="borrowers",   # Poorest 25% of households
    T=40,
)

print(f"Initial Consumption Multiplier in HANK: {res_tf.irf_consumption_hank[0]:.4f}")
print(f"Initial Consumption Multiplier in RANK: {res_tf.irf_consumption_rank[0]:.4f}")
# Output: Initial Consumption Multiplier in HANK: 0.3163
#         Initial Consumption Multiplier in RANK: 0.0150
# In HANK, high-MPC borrowers spend more of it on impact (about 21x the response in RANK)
```

---

## 2. Comprehensive API Reference

### 2.1 `MonetaryTransmissionSimulator`

| Method / Property | Signature | Description |
|---|---|---|
| `steady_state_mpc` | `float` *(property)* | Steady-state aggregate quarterly MPC in HANK economy. |
| `simulate_rate_shock()` | `magnitude=0.0025, rho=0.7, T=40` | Solve HANK vs. RANK sequence-space equilibrium for an interest rate shock. |
| `simulate_balance_sheet_intervention()` | `amount=1.0, target='borrowers', T=40` | Simulate targeted fiscal transfer or balance-sheet injection. |
| `simulate_transmission()` | `shock_type='rate', shock_path=None, ...` | General simulation entry point supporting custom exogenous shock paths. |

### 2.2 `MonetaryTransmissionResult`

| Attribute / Method | Type / Signature | Description |
|---|---|---|
| `irf_output_hank / rank` | `np.ndarray (T,)` | Output impulse response paths $d\mathbf{Y}$. |
| `irf_consumption_hank / rank` | `np.ndarray (T,)` | Consumption impulse response paths $d\mathbf{C}$. |
| `irf_inflation_hank / rank` | `np.ndarray (T,)` | Quarterly inflation impulse response paths $d\boldsymbol{\pi}$ (`plot()` shows them annualized, $\times 400$). |
| `irf_rate_hank / rank` | `np.ndarray (T,)` | Ex-ante real interest rate paths $d\mathbf{r}$. |
| `mpc_deciles_hank / rank` | `pd.Series (10,)` | Marginal propensity to consume across 10 wealth deciles. |
| `aggregate_mpc_hank / rank` | `float` | Aggregate quarterly economy-wide MPC. |
| `direct_channel_hank` | `np.ndarray (T,)` | KMV direct intertemporal substitution channel $\mathbf{J}^{C, r} d\mathbf{r}$. |
| `indirect_channel_hank` | `np.ndarray (T,)` | KMV indirect general equilibrium labor income channel $\mathbf{J}^{C, Y} d\mathbf{Y}$. |
| `indirect_share_hank` | `float` | Percentage share of date-0 consumption driven by indirect GE channel. |
| `peak_responses()` | `-> dict[str, float]` | Peak responses for output, inflation, and consumption. |
| `summary()` | `-> str` | Formatted ASCII report comparing HANK and RANK mechanisms. |
| `to_frame()` | `-> pd.DataFrame` | Tidy quarterly path DataFrame indexed by quarter $t \in [0, T-1]$. |
| `plot(kind)` | `kind='all' \| 'irf' \| 'mpc' \| 'kmv'` | Comparative HANK vs. RANK transmission figures. |

---

## 3. Interactive Browser Lab in JavaScript (`curso/site/labs/`)

The course site in `curso/site/labs/` includes a client-side browser lab for this simulator: static HTML and JavaScript, with no build step.

### 3.1 `politica-monetaria-hank.html` & `.js` (HANK vs. RANK Transmission Lab)

- **Path**: `curso/site/labs/politica-monetaria-hank.html` and `curso/site/labs/politica-monetaria-hank.js`.
- **Core Features**:
  * A policy-scenario preset (hike of 100 bp, cut of 100 bp, persistent shock with $\rho = 0.85$, liquidity transfer) and six sliders: the interest rate shock (−200 to +200 bp, annualized), its persistence $\rho$, the transfer to constrained households (% of GDP), the Taylor rule response $\phi_\pi$, the Phillips curve slope $\kappa$ and risk aversion $\gamma$.
  * Line charts of the output and inflation impulse responses, HANK against RANK.
  * A line chart of the quarterly MPC by wealth decile: HANK's declining ladder against RANK's flat line at $1 - \beta$.
  * The Kaplan-Moll-Violante decomposition of HANK consumption into its direct and indirect channels, and tiles for the aggregate MPC, the impact output contraction, the indirect channel share and the transfer multiplier.
  * The HANK household Jacobians and MPC deciles ($T = 40$) are fixed arrays in the script; moving a control re-solves the linear general-equilibrium system in the browser.

### 3.2 Running Locally

To serve the labs locally (each page loads d3 7.9.0 and KaTeX 0.16.11, with its auto-render extension, from the jsDelivr CDN, and the Manrope font from Google Fonts, so the charts need network access):

```bash
# From the puremacro repository root
cd curso/site
python3 -m http.server 8000
```
Open your browser to `http://localhost:8000/labs/comercio-aranceles.html` or `http://localhost:8000/labs/politica-monetaria-hank.html`.

---

## References

1. **Auclert, A., Bardóczy, B., Rognlie, M., and Straub, L. (2021).** "Using the Sequence-Space Jacobian to Solve and Estimate Heterogeneous-Agent Models." *Econometrica*, 89(6), 3115–3148.
2. **Kaplan, G., Moll, B., and Violante, G. L. (2018).** "Monetary Policy According to HANK." *American Economic Review*, 108(3), 697–743.
