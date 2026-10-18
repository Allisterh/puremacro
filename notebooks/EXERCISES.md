# Graduate Macroeconomics Exercises Catalog: Advanced Computational Methods

This catalog provides rigorous, graduate-level exercises for Advanced Macroeconomics showcase replications. Each notebook unit features four progressive problems: one Basic, two Intermediate, and one Stretch. Every problem is structured into:
1. **Economic Intuition**: Fundamental macro theory, mechanisms, and policy implications.
2. **Analytical Derivation**: Step-by-step mathematical proofs, undetermined coefficients, and Lagrangian optimality conditions.
3. **Computational Task & Assertions**: Reproducible execution code using `puremacro`'s solvers and numerical assertions.

**Prerequisite:** run the corresponding notebook (66, 67 or 68) from top to bottom with its default calibration, then add these snippets as new cells in the same kernel. Helpers such as `variances`, `level_moments`, `huggett_price`, `aiyagari_rate`, and `claim_to_cube` are defined in those notebooks; they are not all public package functions. Extensions and the Monte Carlo experiment are assignments for the student to implement. Numerical inequalities describe specific calibrations.

---

## Notebook 66: Clarida–Galí–Gertler (1999) Optimal Monetary Policy

### Exercise 1: Why Commitment Raises the Nominal Rate Less: Real Rates and the IS Curve
- **Level**: Basic
- **Economic Intuition**:
  In the canonical Clarida, Galí and Gertler (1999) New Keynesian framework, a central bank acting under discretion cannot credibly bind future actions. Confronted with a cost-push shock, it can only trade current inflation off against the current output gap by aggressively hiking the nominal and real policy rates on impact. In contrast, a committed central bank exploits the forward-looking nature of inflation expectations: by promising to keep the output gap negative in future periods after the shock has subsided, it dampens inflation expectations today. Because expected inflation drops substantially, the contemporaneous nominal rate rises much less, and the impact real rate can even fall. Nevertheless, output contracts because the forward-looking IS curve is driven by the entire infinite sum of expected future real rates (without discounting in this IS equation).
- **Analytical Derivation**:
  1. The dynamic forward-looking IS curve in the absence of demand shocks ($g_t = 0$) is:
     $$x_t = E_t x_{t+1} - \varphi (i_t - E_t \pi_{t+1})$$
     Define the ex-ante real interest rate as $r_t \equiv i_t - E_t \pi_{t+1}$. Rewriting gives:
     $$\varphi r_t = E_t x_{t+1} - x_t$$
  2. Iterate the difference equation forward from $t=0$ to $T$:
     $$\varphi \sum_{j=0}^T r_j = \sum_{j=0}^T (x_{j+1} - x_j) = x_{T+1} - x_0$$
     Assuming transversality and stability such that $\lim_{T \to \infty} x_{T+1} = 0$, we obtain the exact intertemporal relation:
     $$x_0 = -\varphi \sum_{j=0}^\infty r_j$$
  3. For some calibrations, commitment can produce $r_0 < 0$ followed by positive real rates; these signs are not a general implication of commitment. The cumulative sum $\sum_{j=0}^\infty r_j > 0$ is positive, accounting for the initial output contraction $x_0 < 0$ despite $i_0^c \ll i_0^d$.
- **Computational Task & Assertions**:
  ```python
  import numpy as np

  # Use the notebook's baseline calibration and solved policies.
  r_d = i_d[:-1] - pi_d[1:]
  r_c = i_c[:-1] - pi_c[1:]
  assert np.max(np.abs(CALIB["phi"] * r_d - np.diff(x_d))) < 1e-9
  assert np.max(np.abs(CALIB["phi"] * r_c - np.diff(x_c))) < 1e-9
  long_c = comm.linear_model.irf("eps_u", horizon=600)
  r_long = long_c["i"].to_numpy()[:-1] - long_c["pi"].to_numpy()[1:]
  assert abs(long_c["x"].iloc[0] + CALIB["phi"] * r_long.sum()) < 1e-9
  print("Impact real rates (discretion, commitment):", r_d[0], r_c[0])
  # The impact signs depend on the calibration; the IS identities do not.
  ```

---

### Exercise 2: How Much of the Commitment Gain Can a Conservative Central Banker Deliver?
- **Level**: Intermediate
- **Economic Intuition**:
  Rogoff (1985) demonstrated that delegating monetary policy to an agent who is more inflation-averse ("conservative") than society can alleviate the classical inflation bias. Clarida, Galí and Gertler (1999) extend this insight to stabilization bias: commitment within a simple state-contingent rule $x_t = -\omega u_t$ yields an allocation isomorphic to discretion under an adjusted relative weight on output stabilization, $w = \alpha (1 - \beta\rho) < \alpha$. When cost-push shocks are serially uncorrelated ($\rho = 0$), $w = \alpha$, so conservative delegation achieves zero gain over discretion. The share of the timeless commitment gain depends on persistence and the other parameters. Compare both loss criteria: timeless commitment need not minimize unconditional loss, especially under heavy discounting.
- **Analytical Derivation**:
  1. Under discretion with delegate weight $w$, the central banker solves:
     $$\min_{\pi_t, x_t} \frac{1}{2} (\pi_t^2 + w x_t^2) \quad \text{s.t.} \quad \pi_t = \lambda x_t + f_t$$
     where $f_t \equiv \beta E_t \pi_{t+1} + u_t$ is taken as given. The first-order condition is $x_t = -\frac{\lambda}{w}\pi_t$.
  2. Substituting into the Phillips curve under $u_t = \rho u_{t-1} + \epsilon_t$ yields:
     $$\pi_t = \frac{w}{\lambda^2 + w(1 - \beta\rho)} u_t, \qquad x_t = -\frac{\lambda}{\lambda^2 + w(1 - \beta\rho)} u_t$$
  3. Under commitment within the class of simple feedback rules $x_t = -\omega u_t$, the planner minimizes the true social loss $\text{Var}(\pi_t) + \alpha \text{Var}(x_t)$ subject to the rational-expectations Phillips curve. The optimal coefficient is:
     $$\omega^c = \frac{\lambda}{\lambda^2 + \alpha(1 - \beta\rho)^2}$$
     Equating the discretionary response to $\omega^c$ requires setting $w = \alpha(1 - \beta\rho)$.
  4. The share of commitment gain attained by the conservative delegate is defined as:
     $$\text{share} = \frac{L_{\text{disc}} - L_{\text{simple}}}{L_{\text{disc}} - L_{\text{comm}}}$$
- **Computational Task & Assertions**:
  ```python
  from puremacro.dsge import discretionary_policy

  # variances and cgg_model are helpers defined by notebook 66.
  p_ex = {**CALIB, "beta": 0.99, "rho": 0.9}
  a_ex, b_ex, rho_ex = (p_ex[k] for k in ("alpha", "beta", "rho"))
  w_c = a_ex * (1.0 - b_ex * rho_ex)
  society = discretionary_policy(cgg_model(p_ex), ["pi", "x"],
      {"pi": 1.0, "x": a_ex}, "i", beta=b_ex, tol=1e-12, max_iter=50000)
  cons = discretionary_policy(cgg_model(p_ex), ["pi", "x"],
      {"pi": 1.0, "x": w_c}, "i", beta=b_ex, tol=1e-12,
      max_iter=50000, compare_commitment=False)
  vp, vx = variances(cons.linear_model)
  loss_simple = vp + a_ex * vx
  cf_ex = cgg_closed_forms(**p_ex)
  assert abs(loss_simple / cf_ex["loss_simple"] - 1.0) < 1e-9
  share = (society.loss - loss_simple) / (society.loss - society.commitment_result.loss)
  print("Share at this calibration:", share)
  # Change rho, then beta; distinguish conditional and unconditional losses.
  ```

---

### Exercise 3: A Price-Level Target Implements Timeless Commitment
- **Level**: Intermediate
- **Economic Intuition**:
  Timeless-perspective commitment requires policy to be history-dependent, adhering to the first-order condition $\pi_t = -\frac{\alpha}{\lambda}(x_t - x_{t-1})$. Because inflation is the first difference of the log price level, $\pi_t = p_t - p_{t-1}$, cumulating this optimality condition over time from initial conditions $x_{-1} = 0, p_{-1} = 0$ collapses the history of inflation rates into a stationary level relationship: $p_t + \frac{\alpha}{\lambda}x_t = 0$. Consequently, a central bank that targets the price level rather than the inflation rate naturally implements the timeless commitment allocation without having to maintain complex Lagrangian state histories.
- **Analytical Derivation**:
  1. The timeless first-order condition is:
     $$\pi_t = -\frac{\alpha}{\lambda}(x_t - x_{t-1})$$
  2. Summing from period $0$ to $t$:
     $$\sum_{s=0}^t \pi_s = p_t - p_{-1} = -\frac{\alpha}{\lambda}\sum_{s=0}^t (x_s - x_{s-1}) = -\frac{\alpha}{\lambda}(x_t - x_{-1})$$
     With initial price level $p_{-1} = 0$ and lagged output gap $x_{-1} = 0$, this yields:
     $$p_t + \frac{\alpha}{\lambda}x_t = 0 \quad \forall t \ge 0$$
  3. Consider closing the model with an interest-rate rule targeting the price-level gap with gain $\phi_p$:
     $$i_t = \frac{g_t}{\varphi} + E_t \pi_{t+1} + \phi_p \left(p_t + \frac{\alpha}{\lambda} x_t\right)$$
     As $\phi_p \to \infty$, the rule forces $p_t + \frac{\alpha}{\lambda}x_t \to 0$ in equilibrium, and the policy deviation shrinks at rate $\mathcal{O}(1/\phi_p)$.
- **Computational Task & Assertions**:
  ```python
  # The notebook already solves timeless commitment. Check its exact price target.
  price_level = np.cumsum(pi_c)
  target_gap = price_level + CALIB["alpha"] / CALIB["lam"] * x_c
  assert np.max(np.abs(target_gap)) < 1e-9
  # Extension: add p = p(-1) + pi to the model and implement the finite-gain
  # interest rule. Solve and check determinacy before comparing loss and paths.
  # Estimate the convergence rate as the gain increases; do not assume it.
  ```

---

### Exercise 4: Discretion, Timeless Commitment, and the Unconditional Optimum as One Family of Targeting Rules (Jensen–McCallum 2002)
- **Level**: Stretch
- **Economic Intuition**:
  Jensen and McCallum (2002) show that optimal discretion, timeless commitment, and unconditional loss minimization are members of a unified one-parameter family of targeting rules: $\pi_t = -\frac{\alpha}{\lambda}(x_t - c x_{t-1})$. Setting $c = 0$ corresponds to discretion, $c = 1$ gives timeless commitment, while $c = \beta$ minimizes the unconditional expectation of social loss $E[\pi_t^2 + \alpha x_t^2]$. Under the unconditional criterion, period $t$ is not discounted relative to period $t-1$, which modifies the intertemporal first-order condition so that the optimal coefficient on lagged output is exactly the discount factor $\beta$.
- **Analytical Derivation**:
  1. Consider the unconditional Lagrangian:
     $$\min_{\{\pi_t, x_t\}} E[\pi_t^2 + \alpha x_t^2] \quad \text{s.t.} \quad \pi_t = \lambda x_t + \beta \pi_{t+1} + u_t \quad \forall t$$
  2. Because the unconditional expectation does not discount future dates, the Lagrangian is:
     $$\mathcal{L} = E\left[ \pi_t^2 + \alpha x_t^2 - \zeta_t (\pi_t - \lambda x_t - \beta \pi_{t+1} - u_t) \right]$$
     Notice that $\pi_t$ appears in constraint $t$ with multiplier $\zeta_t$ and in constraint $t-1$ as $\pi_{(t-1)+1}$ with multiplier $\beta \zeta_{t-1}$.
  3. Differentiating with respect to $\pi_t$ and $x_t$:
     $$\frac{\partial \mathcal{L}}{\partial \pi_t} = 2\pi_t - \zeta_t + \beta \zeta_{t-1} = 0$$
     $$\frac{\partial \mathcal{L}}{\partial x_t} = 2\alpha x_t + \lambda \zeta_t = 0 \implies \zeta_t = -\frac{2\alpha}{\lambda}x_t$$
  4. Substituting $\zeta_t$ into the first equation:
     $$2\pi_t + \frac{2\alpha}{\lambda}x_t - \beta \left(\frac{2\alpha}{\lambda}x_{t-1}\right) = 0 \implies \pi_t = -\frac{\alpha}{\lambda}(x_t - \beta x_{t-1})$$
     Thus, $c^* = \beta$ minimizes the unconditional loss.
- **Computational Task & Assertions**:
  ```python
  from scipy.optimize import minimize_scalar

  p_ex = {**CALIB, "beta": 0.99, "rho": 0.5}
  a_ex, l_ex, b_ex = (p_ex[k] for k in ("alpha", "lam", "beta"))
  def targeting_loss(c):
      rule = f"pi = -({a_ex!r}/{l_ex!r})*(x - {float(c)!r}*x(-1));"
      model_ex = cgg_model(p_ex, rule=rule)
      vp, vx = variances(model_ex)
      return vp + a_ex * vx
  cf_ex = cgg_closed_forms(**p_ex)
  assert abs(targeting_loss(0) / cf_ex["loss_disc"] - 1) < 1e-9
  assert abs(targeting_loss(1) / cf_ex["loss_comm"] - 1) < 1e-9
  opt = minimize_scalar(targeting_loss, bounds=(0, 1), method="bounded",
                        options={"xatol": 1e-10})
  assert abs(opt.x - b_ex) < 1e-4
  assert targeting_loss(b_ex) <= min(targeting_loss(0), targeting_loss(1))
  ```

---

## Notebook 67: Bewley–Huggett–Aiyagari Classics Replication

### Exercise 1: Table I Against the Process It Approximates: Closed-Form Lognormal Moments
- **Level**: Basic
- **Economic Intuition**:
  In Aiyagari (1994), Table I reports summary statistics of the discretized earnings chain, which approximates a continuous lognormal earnings level $Z_t = \exp(x_t)$, where $x_t$ follows an AR(1) process: $x_t = \rho x_{t-1} + \epsilon_t$ with $\epsilon_t \sim \mathcal{N}(0, \sigma_\epsilon^2)$ and unconditional variance $\sigma^2 = \sigma_\epsilon^2 / (1-\rho^2)$. When discretizing this continuous process into a finite-state Markov chain, the choice of discretization algorithm (Tauchen vs. Rouwenhorst) significantly affects the resulting moments. Rouwenhorst preserves the underlying log moments $(\sigma, \rho)$ exactly for any grid size $N$, but its level coefficient of variation converges only asymptotically as $N \to \infty$ because the binomial grid is non-Gaussian. With fixed span $m=3$, Tauchen combines tail truncation with discretization error. At the exercise calibration a seven-state chain overstates the coefficient of variation; this error does not have a universal sign.
- **Analytical Derivation**:
  1. For a Gaussian variable $x \sim \mathcal{N}(0, \sigma^2)$, the moment generating function is $M_x(t) = E[e^{tx}] = \exp\left(\frac{t^2\sigma^2}{2}\right)$.
  2. The mean and variance of $Z = e^x$ are:
     $$E[Z] = \exp\left(\frac{\sigma^2}{2}\right), \qquad E[Z^2] = \exp(2\sigma^2)$$
     $$\text{Var}(Z) = E[Z^2] - (E[Z])^2 = \exp(\sigma^2)(\exp(\sigma^2) - 1)$$
     The true coefficient of variation is:
     $$\text{c.v.} = \frac{\sqrt{\text{Var}(Z)}}{E[Z]} = \sqrt{\exp(\sigma^2) - 1}$$
  3. The first-order autocovariance of $Z_t = e^{x_t}$ with $\text{Cov}(x_t, x_{t-1}) = \rho \sigma^2$ is:
     $$\text{Cov}(e^{x_t}, e^{x_{t-1}}) = E[e^{x_t + x_{t-1}}] - (E[e^{x_t}])^2 = \exp(\sigma^2(1+\rho)) - \exp(\sigma^2)$$
     Dividing by $\text{Var}(Z)$ yields the exact first-order autocorrelation:
     $$\text{Corr}(Z_t, Z_{t-1}) = \frac{\exp(\rho\sigma^2) - 1}{\exp(\sigma^2) - 1}$$
- **Computational Task & Assertions**:
  ```python
  import numpy as np
  from puremacro.vfi import markov_stationary, rouwenhorst, tauchen

  # log_moments and level_moments are helpers defined by notebook 67.
  sigma_ex, rho_ex = 0.4, 0.9
  sigma_eps = sigma_ex * np.sqrt(1 - rho_ex**2)
  cv_true = np.sqrt(np.expm1(sigma_ex**2))
  ac_true = np.expm1(rho_ex * sigma_ex**2) / np.expm1(sigma_ex**2)
  errors = []
  for n in [3, 7, 25, 101]:
      grid_ex, P_ex = rouwenhorst(n, rho_ex, sigma_eps)
      sd_log, ac_log = log_moments(grid_ex, P_ex)
      np.testing.assert_allclose([sd_log, ac_log], [sigma_ex, rho_ex], atol=1e-12)
      cv_ex, ac_ex = level_moments(grid_ex, P_ex)
      errors.append(abs(cv_ex - cv_true))
  assert all(a > b for a, b in zip(errors, errors[1:]))
  assert errors[-1] < 1e-3
  cv_t7, ac_t7 = level_moments(*tauchen(7, rho_ex, sigma_eps, m=3.0))
  assert cv_t7 - cv_true > 0.05
  print("Continuous process (cv, ac):", cv_true, ac_true)
  print("Tauchen-7 chain (cv, ac):", cv_t7, ac_t7)
  ```

---

### Exercise 2: Huggett: How Loose Must Credit Be for a Positive Risk-Free Rate?
- **Level**: Intermediate
- **Economic Intuition**:
  In the Huggett (1993) pure-exchange economy, households trade a risk-free bond in zero net supply to smooth idiosyncratic endowment shocks against a borrowing limit $\underline{a} = -B$. Because agents face uninsurable idiosyncratic risk and incomplete markets, they accumulate precautionary balances. When the credit limit is tight (e.g., $B = 2$), households are exposed to severe risk of binding at the constraint, driving precautionary bond demand high and pushing the equilibrium interest rate into negative territory ($r < 0$). Relaxing the borrowing limit allows households to use borrowing rather than asset holding as a buffer, expanding bond issuance and raising the equilibrium interest rate toward the rate of time preference $1/\beta - 1$. More risk-averse households ($\mu = 3$ vs. $\mu = 1.5$) exhibit stronger precautionary motives, requiring substantially more credit before $r$ becomes positive.
- **Analytical Derivation**:
  1. Household Euler equation for next-period goods-denominated assets $a_{t+1} \ge -B$:
     $$u'(c_t) \ge \beta (1+r) E_t [u'(c_{t+1})], \quad \text{with equality if } a_{t+1} > -B$$
  2. In pure exchange, aggregate asset supply is fixed at zero:
     $$\int a^*(a, z; r, B) \, d\lambda(a, z) = 0$$
  3. For a fixed positive net rate and income bounded below by $y_{\min}>0$, the natural borrowing bound in goods units is $-y_{\min}/r$. Do not extrapolate this formula to zero or negative rates, or infer an equilibrium limit as $B$ increases.
  4. Tighter credit limits constrain smoothing. Compare the equilibrium rates at finite limits; their quantitative response is a property of the calibration and stationary equilibrium.
- **Computational Task & Assertions**:
  ```python
  # huggett_price returns a bond price; annual_rate converts it to annual percent.
  limits_ex = [-2, -3, -4, -6, -8, -10]
  rates_ex = {mu_ex: [annual_rate(huggett_price(mu_ex, limit_ex))
                        for limit_ex in limits_ex] for mu_ex in [1.5, 3.0]}
  for mu_ex, values in rates_ex.items():
      assert all(np.isfinite(values))
      assert all(r < annual_rate(BETA_H) for r in values)
      print(mu_ex, dict(zip(limits_ex, values)))
  # Locate zero crossings and compare risk aversion. Refine both asset grids
  # before treating an ordering or threshold as robust numerical evidence.
  ```

---

### Exercise 3: What Does Aiyagari Hold Fixed When He Raises Persistence?
- **Level**: Intermediate
- **Economic Intuition**:
  When evaluating the impact of earnings persistence $\rho$ on aggregate capital and the equilibrium interest rate in Aiyagari (1994), one must distinguish between holding unconditional variance $\sigma^2$ constant versus holding innovation variance $\sigma_\epsilon^2$ constant. Under fixed unconditional variance, as $\rho \to 1$, innovation variance $\sigma_\epsilon^2 = \sigma^2(1-\rho^2) \to 0$. Innovation variance then vanishes, but the stationary limit as persistence approaches one is singular. A finite-grid comparison alone does not establish a limiting saving rate or interest rate. In contrast, when innovation variance $\sigma_\epsilon^2$ is held constant, higher persistence magnifies the unconditional variance $\sigma^2 \to \infty$, changing both dispersion and persistence. The resulting equilibrium rate must be solved rather than inferred from persistence alone.
- **Analytical Derivation**:
  1. The continuous earnings process is $\log z_t = \rho \log z_{t-1} + \epsilon_t$, $\text{Var}(\epsilon_t) = \sigma_\epsilon^2$. The unconditional variance is:
     $$\sigma_z^2 = \frac{\sigma_\epsilon^2}{1 - \rho^2}$$
  2. **Design (uncond)**: Hold $\sigma_z = 0.4$ constant. Then $\sigma_\epsilon(\rho) = 0.4\sqrt{1-\rho^2}$. As $\rho \to 1$, the variance of new news per period $\sigma_\epsilon^2 \to 0$. Household income becomes more persistent while one-period innovation risk falls. The notebook calibration exhibits a non-monotonic rate response; test whether it survives refinement.
  3. **Design (innov)**: Hold $\sigma_\epsilon = 0.4\sqrt{1-0.81}$ fixed. As $\rho \to 1$, $\sigma_z^2(\rho) \to \infty$. The unconditional variance diverges. This statistical statement does not by itself establish unbounded equilibrium capital or a limiting rate of $-\delta$.
- **Computational Task & Assertions**:
  ```python
  def persistence_rate(rho_ex, sigma_eps_ex):
      grid_ex, P_ex = rouwenhorst(7, rho_ex, sigma_eps_ex)
      return aiyagari_rate(grid_ex, P_ex, mu=3.0)

  unconditional = {r: persistence_rate(r, 0.4 * np.sqrt(1-r*r))
                   for r in [0.6, 0.9, 0.99]}
  innovation = {r: persistence_rate(r, 0.4 * np.sqrt(1-0.9**2))
                for r in [0.6, 0.9, 0.97]}
  assert np.isclose(unconditional[0.9], innovation[0.9])
  print("Fixed unconditional s.d.:", unconditional)
  print("Fixed innovation s.d.:", innovation)
  # Compare the shapes, then repeat with larger income and asset grids.
  # These finite-grid experiments do not establish the limit as rho approaches 1.
  ```

---

### Exercise 4: An Error Budget for the Corner Gap: Chain Identification vs. 10,000 Simulated Draws
- **Level**: Stretch
- **Economic Intuition**:
  Modern exact policy function iteration and distribution tracking on fine asset grids reveal discrepancies ("gaps") of up to $+0.256$ percentage points relative to Aiyagari's (1994) published Table II at the high-risk corner ($\sigma = 0.4, \rho = 0.9, \mu = 5$). Aiyagari obtained aggregate asset demand by simulating a single 10,000-period time series. Because idiosyncratic income and asset accumulation are highly autocorrelated, 10,000 simulated model periods (years in this Aiyagari calibration) provide relatively few independent observations, creating substantial Monte Carlo sampling noise whose standard error scales as $\mathcal{O}(1/\sqrt{T})$.
- **Analytical Derivation**:
  1. For a stationary process with geometrically decaying autocorrelations $\rho_a^j$, the large-sample variance of the sample mean $\bar{a}_T = \frac{1}{T}\sum_{t=1}^T a_t$ is:
     $$\text{Var}(\bar{a}_T) \approx \frac{\sigma_a^2}{T} \left( \frac{1 + \rho_a}{1 - \rho_a} \right)$$
     When $\rho_a \approx 0.98$, $\frac{1+\rho_a}{1-\rho_a} \approx 99$, reducing a nominal sample of 10,000 draws to an effective sample size of only $\approx 100$ independent draws.
  2. Using the implicit function theorem on the asset-market clearing condition $E[a](r) - K(r) = 0$, the sampling variance in mean assets translates into an interest-rate sampling error:
     $$\Delta r \approx -\frac{\bar{a}_T - E[a]}{\frac{\partial (E[a] - K)}{\partial r}}$$
  3. Consequently, the Monte Carlo standard error of the estimated rate satisfies:
     $$\frac{\text{s.d.}_{20,000}}{\text{s.d.}_{10,000}} \approx \sqrt{\frac{10,000}{20,000}} = \frac{1}{\sqrt{2}} \approx 0.707$$
- **Computational Task & Assertions**:
  **Student implementation:** construct the width sweep and simulations described above using notebook 67's `tauchen`, `level_moments`, `aiyagari_market`, and `solve_egm`. State the seed, burn-in, units (percentage points), and method for translating asset errors into rate errors. Compare the standard-deviation ratio with $1/\sqrt{2}$ and report its Monte Carlo uncertainty; do not require a random draw to always lie within two standard deviations.

---

## Notebook 68: Third-Order Perturbation & Skewness Channels

### Exercise 1: Which Risk Term Flips Sign?
- **Level**: Basic
- **Economic Intuition**:
  In third-order perturbation of dynamic models with asymmetric payoffs (e.g., cubic utility or production terms), risk impacts policy rules through both state-risk interactions (`ghxss`) and shock-risk interactions (`ghuss`). When shock persistence $\rho$ flips from positive to negative, the transmission of innovation risk reverses: an unexpected positive innovation $u_t$ implies an expectation of a negative state next period ($\rho u_t < 0$). Under convex marginal payoffs, this reverses the sign of the shock-risk cross-derivative `ghuss`. In contrast, the state-risk interaction `ghxss` is proportional to $\rho^2$, remaining strictly positive.
- **Analytical Derivation**:
  1. Consider the asset-pricing model:
     $$y_t = \beta E_t y_{t+1} + x_t^3, \qquad x_t = \rho x_{t-1} + s u_t$$
  2. Posit the third-order expansion for $y_t$:
     $$y_t = A x_t^3 + B x_t s^2$$
  3. Substitute into the model equation and take expectations:
     $$E_t y_{t+1} = A E_t [(\rho x_t + s u_{t+1})^3] + B E_t [(\rho x_t + s u_{t+1}) s^2]$$
     Using $E_t[u_{t+1}] = E_t[u_{t+1}^3] = 0$ and $E_t[u_{t+1}^2] = 1$:
     $$E_t y_{t+1} = A (\rho^3 x_t^3 + 3\rho x_t s^2) + B \rho x_t s^2$$
  4. Matching coefficients on $x_t^3$ and $x_t s^2$:
     $$A = \beta A \rho^3 + 1 \implies A = \frac{1}{1 - \beta \rho^3}$$
     $$B = 3\beta A \rho + \beta B \rho \implies B = \frac{3\beta A \rho}{1 - \beta \rho}$$
  5. For the notebook convention $x_t=\rho x_{t-1}+e_t$ with $\operatorname{Var}(e_t)=s^2$, the decision-rule derivatives are with respect to the unstandardized innovation $e_t$. Since $B$ above excludes $s^2$, they are:
     $$ghuss = \frac{\partial^3 y}{\partial e \partial \sigma^2} = 2B s^2, \qquad ghxss = \frac{\partial^3 y}{\partial x_{t-1} \partial \sigma^2} = 2B \rho s^2$$
     The ratio is exactly:
     $$\frac{ghxss}{ghuss} = \rho$$
     When $\rho < 0$, $B$ and $ghuss$ flip sign, but $ghxss = \rho ghuss > 0$ remains positive.
- **Computational Task & Assertions**:
  ```python
  rho_ex = -0.8
  solution_ex = claim_to_cube(beta=0.95, rho=rho_ex, s=0.1)
  _, tensors_ex = labelled_tensors(solution_ex)
  g_uss = tensors_ex["ghuss"].loc["y"].iloc[0]
  g_xss = tensors_ex["ghxss"].loc["y"].iloc[0]
  assert np.sign(g_uss) == np.sign(rho_ex)
  assert g_xss > 0
  assert abs(g_xss / g_uss - rho_ex) < 1e-12
  # Innovations e have variance s**2; derivative columns are with respect to e,
  # not to a standardized innovation e/s. Compare cube_closed_form(0.95, rho_ex, 0.1).
  ```

---

### Exercise 2: Quadratic Payoff: The Risk Constant Versus the Ergodic Mean
- **Level**: Intermediate
- **Economic Intuition**:
  In nonlinear stochastic models, agents adjust their choices not only to state variables but also to background uncertainty. In a second-order or third-order expansion, the risk constant $\frac{1}{2} g_{\sigma\sigma}$ contributes to the risky steady state, the fixed point under zero realized shocks while future uncertainty remains in expectations. In a model with endogenous states, finding that fixed point also requires propagating the risk correction through the state transition. This differs fundamentally from the unconditional ergodic mean $E[y_t]$. For the scalar claim below, the gap is the curvature term $C\operatorname{Var}(x_t)$. General DSGE models also require shock-curvature terms and propagation through the state dynamics.
- **Analytical Derivation**:
  1. Let $y_t = \beta E_t y_{t+1} + x_t^3 + c_2 x_t^2$ with $x_t = \rho x_{t-1} + s u_t$.
  2. Posit the solution with undetermined coefficients:
     $$y_t = A x_t^3 + C x_t^2 + B x_t s^2 + D s^2$$
  3. Evaluating $E_t y_{t+1}$:
     $$E_t y_{t+1} = A (\rho^3 x_t^3 + 3\rho x_t s^2) + C (\rho^2 x_t^2 + s^2) + B \rho x_t s^2 + D s^2$$
  4. Matching coefficients on $x_t^2$ and $s^2$:
     $$C = \beta C \rho^2 + c_2 \implies C = \frac{c_2}{1 - \beta \rho^2}$$
     $$D = \beta C + \beta D \implies D = \frac{\beta C}{1 - \beta} = \frac{\beta c_2}{(1-\beta)(1-\beta\rho^2)}$$
  5. The risky steady state corresponds to $x = 0$, so $y_{\text{rss}} = D s^2$.
     The unconditional ergodic mean is:
     $$E[y_t] = C E[x_t^2] + D s^2 = C \frac{s^2}{1 - \rho^2} + D s^2$$
     As $\beta \to 1$, $D \to \infty$ because un-discounted constant variance compounds indefinitely.
- **Computational Task & Assertions**:
  ```python
  def claim_quad(beta, rho, s, c2):
      return load_mod(
          f"var y x; varexo e; parameters beta rho c2; "
          f"beta={beta!r}; rho={rho!r}; c2={c2!r}; "
          "model; x=rho*x(-1)+e; y=beta*y(+1)+x^3+c2*x^2; end; "
          f"initval; x=0; y=0; end; shocks; var e; stderr {s!r}; end;",
          states=["x"], order=3)

  b_ex, r_ex, s_ex, c2_ex = 0.95, 0.8, 0.1, 0.5
  quad_ex = claim_quad(b_ex, r_ex, s_ex, c2_ex)
  dr_ex, tensors_ex = labelled_tensors(quad_ex)
  g_ex = {f: tensors_ex[f].loc["y"].iloc[0] for f in AXES}
  A_ex = 1 / (1-b_ex*r_ex**3)
  B_ex = 3*b_ex*A_ex*r_ex*s_ex**2 / (1-b_ex*r_ex)
  C_ex = c2_ex / (1-b_ex*r_ex**2)
  D_ex = b_ex*C_ex*s_ex**2 / (1-b_ex)
  v_ex = s_ex**2 / (1-r_ex**2)
  np.testing.assert_allclose([g_ex[f] for f in ["ghs2", "ghxx", "ghuu", "ghxss", "ghuss"]],
      [2*D_ex, 2*C_ex*r_ex**2, 2*C_ex, 2*B_ex*r_ex, 2*B_ex], atol=1e-12)
  mean_ex = quad_ex.theoretical_moments().moments.loc["y", "Mean"]
  assert abs(mean_ex - (C_ex*v_ex + D_ex)) < 1e-12
  assert abs(dr_ex.ghs2.loc["y"] / 2 - D_ex) < 1e-12
  ```

---

### Exercise 3: A Symmetry Test Sees What a Path Check Cannot
- **Level**: Intermediate
- **Economic Intuition**:
  Higher-order perturbation solvers produce multilinear derivative tensors (such as $g_{xxx}, g_{xxu}$). In dynamic simulations, these tensors are contracted with symmetric Kronecker products of state and shock vectors: $x_t \otimes x_t \otimes x_t$. Because permutation of indices in $x_{1,t} x_{2,t} x_{3,t}$ produces identical scalar products, any antisymmetric component of the tensor is annihilated upon contraction: $(x \otimes x \otimes x)^T \Delta_{\text{antisym}} = 0$. Consequently, standard path simulation tests or oracle trajectory comparisons against Dynare are completely blind to antisymmetric coding errors in higher-order solvers. A direct tensor symmetry defect audit is essential.
- **Analytical Derivation**:
  1. Let $T \in \mathbb{R}^{n \times n \times n}$ be an order-3 tensor. Its fully symmetrized counterpart is:
     $$T^{\text{sym}}_{i,j,k} = \frac{1}{3!} \sum_{\pi \in S_3} T_{\pi(i), \pi(j), \pi(k)}$$
  2. For dimension $n=2$, the total number of tensor elements is $2^3 = 8$. The number of unique symmetric combinations is $\binom{n+k-1}{k} = \binom{2+3-1}{3} = 4$.
  3. The orthogonal complement consists of $8 - 4 = 4$ independent antisymmetric directions. For any error $\Delta$ in this subspace:
     $$(x \otimes x \otimes x)^T \text{vec}(\Delta) = 0 \quad \forall x \in \mathbb{R}^2$$
  4. Define the tensor symmetry defect norm:
     $$\text{defect}(T) = \max_{\pi \in S_k} \|T - P_\pi(T)\|_\infty$$
- **Computational Task & Assertions**:
  ```python
  # Use the planted errors and contractions computed in notebook 68, section 3.
  assert schwarz_err < 1e-12
  assert blind.loc["largest effect on a simulated observation", "antisymmetric"] < 1e-15
  assert blind.loc["Schwarz asymmetry |D(k,a,e) - D(a,k,e)|", "antisymmetric"] > 2e-3
  assert blind.loc["Schwarz asymmetry |D(k,a,e) - D(a,k,e)|", "symmetric"] == 0
  assert blind.loc["largest effect on a simulated observation", "symmetric"] > 1e-10
  # Extension: symmetrize a third-order tensor over all state permutations,
  # and construct an error that preserves symmetry but changes its contraction.
  ```

---

### Exercise 4: Precaution or Jensen? Decompose the Rise in Mean Capital
- **Level**: Stretch
- **Economic Intuition**:
  In some stochastic growth calibrations, the unconditional ergodic mean of capital exceeds its deterministic steady state. Neither this sign nor the sign of a risk correction is universal. Macroeconomists frequently attribute this entire wedge to "precautionary savings" stemming from prudence ($u''' > 0$). However, Jensen's inequality (the curvature of policy functions over state variance and shock variance) also generates an upward shift in mean capital even in models where risk corrections are identically zero (such as the log-utility Brock–Mirman model). Using higher-order perturbation, the shift in ergodic capital can be cleanly decomposed into: (1) explicit policy risk corrections, (2) endogenous state dispersion curvature, and (3) exogenous shock variance curvature.
- **Analytical Derivation**:
  1. The second-order state transition is:
     $$x_{t+1} = h_x x_t + \frac{1}{2} H_{xx} (x_t \otimes x_t) + \frac{1}{2} H_{uu} (u_{t+1} \otimes u_{t+1}) + \frac{1}{2} h_{\sigma\sigma}$$
  2. Taking unconditional expectations ($E[x_{t+1}] = E[x_t]$):
     $$(I - h_x) E[x] = \frac{1}{2} H_{xx} \text{vec}(\Omega) + \frac{1}{2} H_{uu} \text{vec}(\Sigma) + \frac{1}{2} h_{\sigma\sigma}$$
     where $\Omega \equiv \text{Var}(x_t)$ solves the discrete Lyapunov equation $\Omega = h_x \Omega h_x^T + h_u \Sigma h_u^T$.
  3. Precautionary component: $x_{\text{prec}} = \frac{1}{2}(I - h_x)^{-1} h_{\sigma\sigma}$, which equals the risky steady state shift.
  4. Jensen curvature component: $x_{\text{jensen}} = \frac{1}{2}(I - h_x)^{-1} [H_{xx} \text{vec}(\Omega) + H_{uu} \text{vec}(\Sigma)]$.
  5. In the Brock–Mirman model ($\gamma = 1$), $h_{\sigma\sigma} \equiv 0$, so the entire shift is Jensen's inequality:
     $$E[k_t] - k_{ss} \approx \frac{1}{2} k_{ss} \text{Var}(\log k_t)$$
  6. For a chosen CRRA calibration, test whether $h_{\sigma\sigma}^k$ changes sign as $\gamma$ varies. Any estimated cutoff belongs to that calibration and coordinate system; it is not a general CRRA threshold.
- **Computational Task & Assertions**:
  ```python
  from scipy.linalg import solve_discrete_lyapunov

  rbc_ex = load_mod(RBC, order=2)
  dr_ex = rbc_ex.decision_rules()
  states_ex = list(dr_ex.state_variables)
  H_ex = dr_ex.ghx.loc[states_ex].to_numpy()
  U_ex = dr_ex.ghu.loc[states_ex].to_numpy()
  Sigma_ex = np.array([[0.01**2]])  # shock variance declared in RBC
  Omega_ex = solve_discrete_lyapunov(H_ex, U_ex @ Sigma_ex @ U_ex.T)
  risk_ex = 0.5 * dr_ex.ghs2.loc[states_ex].to_numpy()
  curvature_ex = 0.5 * (dr_ex.ghxx.loc[states_ex].to_numpy() @ Omega_ex.ravel()
                         + dr_ex.ghuu.loc[states_ex].to_numpy() @ Sigma_ex.ravel())
  parts_ex = np.linalg.solve(np.eye(len(states_ex)) - H_ex,
                             np.column_stack([risk_ex, curvature_ex]))
  mean_shift_ex = (rbc_ex.theoretical_moments().moments.loc[states_ex, "Mean"]
                   - dr_ex.ys.loc[states_ex]).to_numpy()
  np.testing.assert_allclose(parts_ex.sum(axis=1), mean_shift_ex, rtol=1e-9, atol=1e-12)
  print(pd.DataFrame(parts_ex, index=states_ex, columns=["risk correction", "curvature"]))
  assert brock_mirman.decision_rules().ghs2.abs().max() < 1e-14
  # The signs depend on calibration and coordinates. Repeat with CRRA utility;
  # locate any sign change numerically rather than assuming a universal cutoff.
  ```
