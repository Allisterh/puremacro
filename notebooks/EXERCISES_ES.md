# Catálogo de ejercicios de posgrado en macroeconomía: métodos computacionales avanzados

Este catálogo proporciona ejercicios rigurosos a nivel de posgrado para las réplicas de los cuadernos de demostración de Macroeconomía Avanzada. Cada unidad de cuaderno presenta cuatro problemas progresivos: uno Básico, dos Intermedios y uno Avanzado (Stretch). Cada problema se estructura en:
1. **Intuición económica**: Teoría macroeconómica fundamental, mecanismos de transmisión e implicaciones de política.
2. **Derivación analítica**: Demostraciones matemáticas paso a paso, coeficientes indeterminados y condiciones de optimalidad lagrangianas.
3. **Tarea computacional y aserciones**: Código de ejecución reproducible utilizando los solucionadores de `puremacro` y aserciones numéricas formales.

**Requisito:** ejecuta primero el cuaderno correspondiente (66, 67 o 68), de arriba abajo, con su calibración predeterminada, y añade los fragmentos en celdas nuevas del mismo kernel. Funciones como `variances`, `level_moments`, `huggett_price`, `aiyagari_rate` y `claim_to_cube` se definen en esos cuadernos; no son todas funciones públicas del paquete. Las extensiones y el experimento Monte Carlo son tareas que el estudiante debe implementar. Las desigualdades numéricas describen calibraciones específicas.

---

## Notebook 66: Clarida–Galí–Gertler (1999) Política monetaria óptima

### Ejercicio 1: ¿Por qué el compromiso eleva menos la tasa nominal? Tasas reales y la curva IS
- **Nivel**: Básico
- **Intuición económica**:
  En el marco neokeynesiano canónico de Clarida, Galí y Gertler (1999), un banco central que actúa bajo discreción no puede comprometer de manera creíble sus acciones futuras. Frente a un choque de costos (cost-push), solo puede arbitrar la inflación presente contra la brecha de producto contemporánea subiendo agresivamente la tasa de política nominal y real en el impacto. Por el contrario, un banco central con credibilidad y compromiso aprovecha la naturaleza forward-looking de las expectativas de inflación: al prometer mantener la brecha de producto negativa en periodos posteriores (una vez que el choque se haya disipado), deprime las expectativas de inflación hoy. Como la inflación esperada cae significativamente, la tasa nominal contemporánea requiere un incremento mucho menor e incluso la tasa real de impacto puede reducirse. No obstante, el producto se contrae porque la curva IS depende de la suma infinita de tasas reales futuras esperadas, sin descuento en esta ecuación IS.
- **Derivación analítica**:
  1. La curva IS dinámica forward-looking en ausencia de choques de demanda ($g_t = 0$) es:
     $$x_t = E_t x_{t+1} - \varphi (i_t - E_t \pi_{t+1})$$
     Definiendo la tasa de interés real ex-ante como $r_t \equiv i_t - E_t \pi_{t+1}$, se obtiene:
     $$\varphi r_t = E_t x_{t+1} - x_t$$
  2. Iterando esta ecuación de diferencias hacia adelante desde $t=0$ hasta $T$:
     $$\varphi \sum_{j=0}^T r_j = \sum_{j=0}^T (x_{j+1} - x_j) = x_{T+1} - x_0$$
     Bajo las condiciones de transversalidad y estabilidad donde $\lim_{T \to \infty} x_{T+1} = 0$, se deduce la relación intertemporal exacta:
     $$x_0 = -\varphi \sum_{j=0}^\infty r_j$$
  3. En algunas calibraciones, el compromiso puede producir $r_0 < 0$ seguido de tasas positivas; esos signos no son una propiedad general del compromiso. La suma acumulada $\sum_{j=0}^\infty r_j > 0$ es estrictamente positiva, explicando la contracción inicial del producto $x_0 < 0$ a pesar de que $i_0^c \ll i_0^d$.
- **Tarea computacional y aserciones**:
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

### Ejercicio 2: ¿Cuánto de la ganancia del compromiso puede lograr un banquero central conservador?
- **Nivel**: Intermedio
- **Intuición económica**:
  Rogoff (1985) demostró que delegar la política monetaria en un agente con mayor aversión a la inflación ("conservador") que la sociedad mitiga el sesgo de inflación clásico. Clarida, Galí y Gertler (1999) extienden este resultado al sesgo de estabilización: el compromiso dentro de una regla simple contingente al estado $x_t = -\omega u_t$ genera una asignación isomorfa a la discreción bajo una ponderación ajustada sobre la brecha de producto, $w = \alpha (1 - \beta\rho) < \alpha$. Cuando los choques de costos no tienen correlación serial ($\rho = 0$), $w = \alpha$, por lo que la delegación conservadora no aporta ganancia alguna sobre la discreción. La fracción de la ganancia depende de la persistencia y los demás parámetros. Compara pérdidas condicionales e incondicionales: el compromiso atemporal no siempre minimiza estas últimas, especialmente con descuento intenso.
- **Derivación analítica**:
  1. Bajo discreción con ponderación del delegado $w$, el banco central resuelve:
     $$\min_{\pi_t, x_t} \frac{1}{2} (\pi_t^2 + w x_t^2) \quad \text{s.a.} \quad \pi_t = \lambda x_t + f_t$$
     donde $f_t \equiv \beta E_t \pi_{t+1} + u_t$ se toma como dado. La condición de primer orden es $x_t = -\frac{\lambda}{w}\pi_t$.
  2. Sustituyendo en la curva de Phillips con $u_t = \rho u_{t-1} + \epsilon_t$:
     $$\pi_t = \frac{w}{\lambda^2 + w(1 - \beta\rho)} u_t, \qquad x_t = -\frac{\lambda}{\lambda^2 + w(1 - \beta\rho)} u_t$$
  3. Bajo compromiso en la clase de reglas simples $x_t = -\omega u_t$, el planificador minimiza la pérdida social verdadera $\text{Var}(\pi_t) + \alpha \text{Var}(x_t)$ sujeta a la curva de Phillips con expectativas racionales. El coeficiente óptimo es:
     $$\omega^c = \frac{\lambda}{\lambda^2 + \alpha(1 - \beta\rho)^2}$$
     Igualar la respuesta discrecional a $\omega^c$ exige exactamente fijar $w = \alpha(1 - \beta\rho)$.
  4. La fracción de la ganancia del compromiso alcanzada por el delegado conservador se define como:
     $$\text{share} = \frac{L_{\text{disc}} - L_{\text{simple}}}{L_{\text{disc}} - L_{\text{comm}}}$$
- **Tarea computacional y aserciones**:
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

### Ejercicio 3: Una meta sobre el nivel de precios implementa el compromiso atemporal
- **Nivel**: Intermedio
- **Intuición económica**:
  El compromiso desde la perspectiva atemporal exige que la política monetaria exhiba dependencia histórica, cumpliendo la condición de primer orden $\pi_t = -\frac{\alpha}{\lambda}(x_t - x_{t-1})$. Dado que la inflación es la primera diferencia del logaritmo del nivel de precios, $\pi_t = p_t - p_{t-1}$, sumar esta condición de optimalidad en el tiempo a partir de condiciones iniciales $x_{-1} = 0, p_{-1} = 0$ reduce la historia de tasas de inflación a una relación de niveles estacionaria: $p_t + \frac{\alpha}{\lambda}x_t = 0$. Por consiguiente, un banco central que persigue una meta sobre el nivel de precios en lugar de una meta sobre la tasa de inflación implementa de forma natural la asignación del compromiso atemporal, sin necesidad de arrastrar multiplicadores de Lagrange históricos.
- **Derivación analítica**:
  1. La condición de optimalidad atemporal es:
     $$\pi_t = -\frac{\alpha}{\lambda}(x_t - x_{t-1})$$
  2. Sumando desde el periodo $0$ hasta $t$:
     $$\sum_{s=0}^t \pi_s = p_t - p_{-1} = -\frac{\alpha}{\lambda}\sum_{s=0}^t (x_s - x_{s-1}) = -\frac{\alpha}{\lambda}(x_t - x_{-1})$$
     Con nivel de precios inicial $p_{-1} = 0$ y brecha de producto rezagada $x_{-1} = 0$:
     $$p_t + \frac{\alpha}{\lambda}x_t = 0 \quad \forall t \ge 0$$
  3. Al cerrar el modelo con una regla de tasa de interés que reacciona a la brecha del nivel de precios con ponderación $\phi_p$:
     $$i_t = \frac{g_t}{\varphi} + E_t \pi_{t+1} + \phi_p \left(p_t + \frac{\alpha}{\lambda} x_t\right)$$
     Cuando $\phi_p \to \infty$, la regla impone $p_t + \frac{\alpha}{\lambda}x_t \to 0$ en equilibrio, y la desviación respecto al óptimo decrece a una tasa $\mathcal{O}(1/\phi_p)$.
- **Tarea computacional y aserciones**:
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

### Ejercicio 4: Discreción, compromiso atemporal y el óptimo incondicional como una sola familia de reglas de meta (Jensen–McCallum 2002)
- **Nivel**: Avanzado (Stretch)
- **Intuición económica**:
  Jensen y McCallum (2002) demostraron que la discreción óptima, el compromiso atemporal y la minimización incondicional de pérdidas son miembros de una familia unificada uniparamétrica de reglas de meta: $\pi_t = -\frac{\alpha}{\lambda}(x_t - c x_{t-1})$. Fijar $c = 0$ corresponde a discreción, $c = 1$ genera el compromiso atemporal, mientras que $c = \beta$ minimiza la esperanza incondicional de la función de pérdida social $E[\pi_t^2 + \alpha x_t^2]$. Bajo el criterio incondicional, el periodo $t$ no está descontado respecto al periodo $t-1$, lo que modifica la condición intertemporal de primer orden haciendo que el coeficiente óptimo sobre el producto rezagado sea exactamente el factor de descuento $\beta$.
- **Derivación analítica**:
  1. Considérese el lagrangiano incondicional:
     $$\min_{\{\pi_t, x_t\}} E[\pi_t^2 + \alpha x_t^2] \quad \text{s.a.} \quad \pi_t = \lambda x_t + \beta \pi_{t+1} + u_t \quad \forall t$$
  2. Debido a que la esperanza incondicional no descuenta fechas futuras, el lagrangiano es:
     $$\mathcal{L} = E\left[ \pi_t^2 + \alpha x_t^2 - \zeta_t (\pi_t - \lambda x_t - \beta \pi_{t+1} - u_t) \right]$$
     Nótese que $\pi_t$ entra en la restricción $t$ con multiplicador $\zeta_t$ y en la restricción $t-1$ como $\pi_{(t-1)+1}$ con multiplicador $\beta \zeta_{t-1}$.
  3. Diferenciando respecto a $\pi_t$ y $x_t$:
     $$\frac{\partial \mathcal{L}}{\partial \pi_t} = 2\pi_t - \zeta_t + \beta \zeta_{t-1} = 0$$
     $$\frac{\partial \mathcal{L}}{\partial x_t} = 2\alpha x_t + \lambda \zeta_t = 0 \implies \zeta_t = -\frac{2\alpha}{\lambda}x_t$$
  4. Sustituyendo $\zeta_t$ en la primera ecuación:
     $$2\pi_t + \frac{2\alpha}{\lambda}x_t - \beta \left(\frac{2\alpha}{\lambda}x_{t-1}\right) = 0 \implies \pi_t = -\frac{\alpha}{\lambda}(x_t - \beta x_{t-1})$$
     Por tanto, $c^* = \beta$ minimiza de forma exacta la pérdida incondicional.
- **Tarea computacional y aserciones**:
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

## Notebook 67: Bewley–Huggett–Aiyagari Réplica de clásicos

### Ejercicio 1: El Cuadro I frente al proceso que aproxima: momentos analíticos log-normales
- **Nivel**: Básico
- **Intuición económica**:
  En Aiyagari (1994), el Cuadro I presenta estadísticas de la cadena de ingresos discretizada que aproxima el nivel de ingreso log-normal continuo $Z_t = \exp(x_t)$, donde $x_t$ sigue un proceso AR(1): $x_t = \rho x_{t-1} + \epsilon_t$ con $\epsilon_t \sim \mathcal{N}(0, \sigma_\epsilon^2)$ y varianza incondicional $\sigma^2 = \sigma_\epsilon^2 / (1-\rho^2)$. Al discretizar este proceso continuo en una cadena de Markov finita, la elección del método de discretización (Tauchen frente a Rouwenhorst) influye profundamente en los momentos resultantes. Rouwenhorst reproduce los momentos logarítmicos $(\sigma, \rho)$ con exactitud para cualquier tamaño de rejilla $N$, pero su coeficiente de variación en niveles converge solo asintóticamente cuando $N \to \infty$ debido a que la rejilla binomial no es gaussiana. Tauchen con ancho fijo $m=3$ combina truncamiento de colas y error de discretización. La cadena de siete estados sobreestima el coeficiente de variación en esta calibración; el signo del error no es universal.
- **Derivación analítica**:
  1. Para una variable gaussiana $x \sim \mathcal{N}(0, \sigma^2)$, la función generadora de momentos es $M_x(t) = E[e^{tx}] = \exp\left(\frac{t^2\sigma^2}{2}\right)$.
  2. La media y la varianza de $Z = e^x$ son:
     $$E[Z] = \exp\left(\frac{\sigma^2}{2}\right), \qquad E[Z^2] = \exp(2\sigma^2)$$
     $$\text{Var}(Z) = E[Z^2] - (E[Z])^2 = \exp(\sigma^2)(\exp(\sigma^2) - 1)$$
     El verdadero coeficiente de variación es:
     $$\text{c.v.} = \frac{\sqrt{\text{Var}(Z)}}{E[Z]} = \sqrt{\exp(\sigma^2) - 1}$$
  3. La autocovarianza de primer orden de $Z_t = e^{x_t}$ con $\text{Cov}(x_t, x_{t-1}) = \rho \sigma^2$ es:
     $$\text{Cov}(e^{x_t}, e^{x_{t-1}}) = E[e^{x_t + x_{t-1}}] - (E[e^{x_t}])^2 = \exp(\sigma^2(1+\rho)) - \exp(\sigma^2)$$
     Dividiendo entre $\text{Var}(Z)$ se deduce la autocorrelación analítica de primer orden en niveles:
     $$\text{Corr}(Z_t, Z_{t-1}) = \frac{\exp(\rho\sigma^2) - 1}{\exp(\sigma^2) - 1}$$
- **Tarea computacional y aserciones**:
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

### Ejercicio 2: Huggett: ¿qué tan amplio debe ser el crédito para que la tasa libre de riesgo sea positiva?
- **Nivel**: Intermedio
- **Intuición económica**:
  En la economía de intercambio puro de Huggett (1993), los hogares intercambian un bono libre de riesgo en oferta neta agregada cero para suavizar choques idiosincrásicos de dotación contra un límite de endeudamiento $\underline{a} = -B$. Debido al riesgo no asegurable y a la incompletitud de mercados, los agentes acumulan saldos precautorios. Cuando el límite crediticio es estricto (por ejemplo, $B = 2$), los hogares enfrentan un alto riesgo de quedar restringidos en el límite, lo que dispara la demanda precautoria de bonos y empuja la tasa de interés de equilibrio a terreno negativo ($r < 0$). Relajar el límite de endeudamiento permite utilizar la deuda como amortiguador en lugar de acumular activos, expandiendo la emisión de deuda y elevando la tasa de interés hacia la tasa de preferencia temporal $1/\beta - 1$. Los hogares con mayor aversión al riesgo ($\mu = 3$ frente a $\mu = 1.5$) poseen motivos precautorios más intensos, necesitando un margen de endeudamiento notablemente mayor para que $r > 0$.
- **Derivación analítica**:
  1. Ecuación de Euler para activos del próximo periodo en unidades de bienes con $a_{t+1} \ge -B$:
     $$u'(c_t) \ge \beta (1+r) E_t [u'(c_{t+1})], \quad \text{con igualdad si } a_{t+1} > -B$$
  2. En intercambio puro, la oferta agregada de bonos es cero:
     $$\int a^*(a, z; r, B) \, d\lambda(a, z) = 0$$
  3. Con tasa neta fija positiva e ingreso mínimo positivo, el límite natural en unidades de bienes es $-y_{\min}/r$. No extrapoles esa fórmula a tasas no positivas ni infieras un límite de equilibrio al ampliar el crédito.
  4. Los límites de crédito más estrechos restringen la suavización. Compara tasas en límites finitos; su respuesta cuantitativa depende de la calibración y del equilibrio estacionario.
- **Tarea computacional y aserciones**:
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

### Ejercicio 3: ¿Qué mantiene fijo Aiyagari cuando incrementa la persistencia?
- **Nivel**: Intermedio
- **Intuición económica**:
  Al analizar el efecto de la persistencia del ingreso $\rho$ sobre el capital agregado y la tasa de interés de equilibrio en Aiyagari (1994), es crucial distinguir entre mantener constante la varianza incondicional $\sigma^2$ o mantener constante la varianza de las innovaciones $\sigma_\epsilon^2$. Fijando la varianza incondicional, cuando $\rho \to 1$, la varianza de las innovaciones $\sigma_\epsilon^2 = \sigma^2(1-\rho^2) \to 0$. La varianza de las innovaciones desaparece, pero el límite estacionario cuando la persistencia se acerca a uno es singular. Un experimento con malla finita no demuestra una tasa de ahorro o de interés límite. Por el contrario, si se fija la varianza de las innovaciones $\sigma_\epsilon^2$, un aumento en la persistencia hace explotar la varianza incondicional $\sigma^2 \to \infty$, cambiando tanto la dispersión como la persistencia. La tasa de equilibrio debe resolverse; no se deduce únicamente de la persistencia.
- **Derivación analítica**:
  1. El proceso continuo de ingreso es $\log z_t = \rho \log z_{t-1} + \epsilon_t$, $\text{Var}(\epsilon_t) = \sigma_\epsilon^2$. La varianza incondicional es:
     $$\sigma_z^2 = \frac{\sigma_\epsilon^2}{1 - \rho^2}$$
  2. **Diseño (incondicional)**: Mantener $\sigma_z = 0.4$ constante. Entonces $\sigma_\epsilon(\rho) = 0.4\sqrt{1-\rho^2}$. Cuando $\rho \to 1$, la varianza de las nuevas noticias por periodo $\sigma_\epsilon^2 \to 0$. El ingreso se vuelve más persistente y cae el riesgo de innovación a un periodo. La calibración del cuaderno presenta una respuesta no monótona; comprueba su estabilidad al refinar la malla.
  3. **Diseño (innovaciones)**: Mantener $\sigma_\epsilon = 0.4\sqrt{1-0.81}$ fijo. Cuando $\rho \to 1$, $\sigma_z^2(\rho) \to \infty$. La varianza incondicional diverge. Este resultado estadístico no demuestra por sí solo capital de equilibrio ilimitado ni una tasa límite de $-\delta$.
- **Tarea computacional y aserciones**:
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

### Ejercicio 4: Un presupuesto de error para la brecha de la esquina: identificación de la cadena vs 10,000 simulaciones
- **Nivel**: Avanzado (Stretch)
- **Intuición económica**:
  La solución numérica moderna exacta mediante iteración de políticas y seguimiento de distribuciones en rejillas finas de activos revela discrepancias ("brechas") de hasta $+0.256$ puntos porcentuales respecto al Cuadro II publicado por Aiyagari (1994) en la esquina de alto riesgo ($\sigma = 0.4, \rho = 0.9, \mu = 5$). Aiyagari obtuvo la demanda agregada de activos simulando una única serie temporal de 10,000 periodos. Dado que el ingreso idiosincrásico y la acumulación de riqueza presentan una elevada autocorrelación, 10,000 periodos simulados (años en esta calibración de Aiyagari) proporcionan un número muy reducido de observaciones independientes, introduciendo un error de muestreo de Monte Carlo considerable cuyo error estándar decrece a razón de $\mathcal{O}(1/\sqrt{T})$.
- **Derivación analítica**:
  1. Para un proceso estacionario con autocorrelaciones geométricas $\rho_a^j$, la varianza asintótica de la media muestral $\bar{a}_T = \frac{1}{T}\sum_{t=1}^T a_t$ es:
     $$\text{Var}(\bar{a}_T) \approx \frac{\sigma_a^2}{T} \left( \frac{1 + \rho_a}{1 - \rho_a} \right)$$
     Cuando $\rho_a \approx 0.98$, el factor $\frac{1+\rho_a}{1-\rho_a} \approx 99$, reduciendo una muestra nominal de 10,000 extracciones a un tamaño efectivo de apenas $\approx 100$ observaciones independientes.
  2. Aplicando el teorema de la función implícita sobre la condición de vaciado del mercado de activos $E[a](r) - K(r) = 0$, la dispersión muestral de los activos medios se traduce en un error sobre la tasa de interés:
     $$\Delta r \approx -\frac{\bar{a}_T - E[a]}{\frac{\partial (E[a] - K)}{\partial r}}$$
  3. En consecuencia, el error estándar de Monte Carlo de la tasa estimada cumple:
     $$\frac{\text{desv.est.}_{20,000}}{\text{desv.est.}_{10,000}} \approx \sqrt{\frac{10,000}{20,000}} = \frac{1}{\sqrt{2}} \approx 0.707$$
- **Tarea computacional y aserciones**:
  **Implementación del estudiante:** construye el barrido de anchos y las simulaciones descritas arriba usando `tauchen`, `level_moments`, `aiyagari_market` y `solve_egm` del cuaderno 67. Declara la semilla, el descarte inicial, las unidades (puntos porcentuales) y el método para trasladar el error de activos a tasas. Compara la razón de desviaciones estándar con $1/\sqrt{2}$ y reporta su incertidumbre Monte Carlo; no impongas que una muestra aleatoria caiga siempre dentro de dos desviaciones estándar.

---

## Notebook 68: Perturbación de tercer orden y canales de asimetría

### Ejercicio 1: ¿Cuál término de riesgo invierte su signo?
- **Nivel**: Básico
- **Intuición económica**:
  En la perturbación de tercer orden de modelos dinámicos con pagos asimétricos (por ejemplo, funciones de utilidad o producción cúbicas), el riesgo impacta las reglas de decisión mediante interacciones estado-riesgo (`ghxss`) e innovaciones-riesgo (`ghuss`). Cuando la persistencia del choque $\rho$ invierte su signo de positivo a negativo, la transmisión del riesgo de las innovaciones se revierte: una innovación positiva imprevista $u_t$ induce una expectativa de estado negativo en el periodo siguiente ($\rho u_t < 0$). Con pagos marginales convexos, esto invierte el signo de la derivada cruzada innovación-riesgo `ghuss`. Por el contrario, la interacción estado-riesgo `ghxss` es proporcional a $\rho^2$, manteniéndose estrictamente positiva.
- **Derivación analítica**:
  1. Considérese el modelo de fijación de precios de activos:
     $$y_t = \beta E_t y_{t+1} + x_t^3, \qquad x_t = \rho x_{t-1} + s u_t$$
  2. Propóngase la aproximación de tercer orden para $y_t$:
     $$y_t = A x_t^3 + B x_t s^2$$
  3. Sustituyendo en la ecuación del modelo y tomando esperanzas condicionales:
     $$E_t y_{t+1} = A E_t [(\rho x_t + s u_{t+1})^3] + B E_t [(\rho x_t + s u_{t+1}) s^2]$$
     Dado que $E_t[u_{t+1}] = E_t[u_{t+1}^3] = 0$ y $E_t[u_{t+1}^2] = 1$:
     $$E_t y_{t+1} = A (\rho^3 x_t^3 + 3\rho x_t s^2) + B \rho x_t s^2$$
  4. Igualando coeficientes en $x_t^3$ y en $x_t s^2$:
     $$A = \beta A \rho^3 + 1 \implies A = \frac{1}{1 - \beta \rho^3}$$
     $$B = 3\beta A \rho + \beta B \rho \implies B = \frac{3\beta A \rho}{1 - \beta \rho}$$
  5. Con la convención del cuaderno $x_t=\rho x_{t-1}+e_t$, $\operatorname{Var}(e_t)=s^2$, las derivadas son respecto a la innovación no estandarizada $e_t$. Como $B$ arriba excluye $s^2$:
     $$ghuss = \frac{\partial^3 y}{\partial e \partial \sigma^2} = 2B s^2, \qquad ghxss = \frac{\partial^3 y}{\partial x_{t-1} \partial \sigma^2} = 2B \rho s^2$$
     El cociente exacto es:
     $$\frac{ghxss}{ghuss} = \rho$$
     Cuando $\rho < 0$, $B$ y $ghuss$ invierten su signo, mientras que $ghxss = \rho ghuss > 0$ se mantiene positivo.
- **Tarea computacional y aserciones**:
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

### Ejercicio 2: Pago cuadrático: la constante de riesgo frente a la media ergódica
- **Nivel**: Intermedio
- **Intuición económica**:
  En modelos estocásticos no lineales, los agentes ajustan sus decisiones no solo a las variables de estado sino también a la incertidumbre de fondo. En una aproximación de segundo o tercer orden, la constante $\frac{1}{2}g_{\sigma\sigma}$ contribuye al estado estacionario con riesgo: el punto fijo sin choques realizados, mientras las expectativas conservan incertidumbre futura. Con estados endógenos hay que propagar esa corrección mediante la transición de estados. Este difiere sustancialmente de la media ergódica incondicional $E[y_t]$. En el ejemplo escalar siguiente, la brecha es $C\operatorname{Var}(x_t)$. En modelos DSGE generales también intervienen la curvatura respecto a choques y la propagación mediante los estados.
- **Derivación analítica**:
  1. Sea $y_t = \beta E_t y_{t+1} + x_t^3 + c_2 x_t^2$ con $x_t = \rho x_{t-1} + s u_t$.
  2. Propóngase la solución por coeficientes indeterminados:
     $$y_t = A x_t^3 + C x_t^2 + B x_t s^2 + D s^2$$
  3. Evaluando $E_t y_{t+1}$:
     $$E_t y_{t+1} = A (\rho^3 x_t^3 + 3\rho x_t s^2) + C (\rho^2 x_t^2 + s^2) + B \rho x_t s^2 + D s^2$$
  4. Igualando coeficientes en $x_t^2$ y en $s^2$:
     $$C = \beta C \rho^2 + c_2 \implies C = \frac{c_2}{1 - \beta \rho^2}$$
     $$D = \beta C + \beta D \implies D = \frac{\beta C}{1 - \beta} = \frac{\beta c_2}{(1-\beta)(1-\beta\rho^2)}$$
  5. El estado estacionario con riesgo evalúa en $x = 0$, obteniéndose $y_{\text{rss}} = D s^2$.
     La media ergódica incondicional es:
     $$E[y_t] = C E[x_t^2] + D s^2 = C \frac{s^2}{1 - \rho^2} + D s^2$$
     Cuando $\beta \to 1$, $D \to \infty$ debido a que la varianza no descontada se acumula sin límite temporal.
- **Tarea computacional y aserciones**:
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

### Ejercicio 3: Una prueba de simetría detecta lo que una comparación de trayectorias no puede ver
- **Nivel**: Intermedio
- **Intuición económica**:
  Los solucionadores de perturbación de orden superior generan tensores multilineales de derivadas (como $g_{xxx}, g_{xxu}$). En simulaciones dinámicas, estos tensores se contraen con productos de Kronecker simétricos de vectores de estado e innovaciones: $x_t \otimes x_t \otimes x_t$. Debido a que la permutación de índices en $x_{1,t} x_{2,t} x_{3,t}$ produce productos escalares idénticos, cualquier componente antisimétrica del tensor se anula en la contracción: $(x \otimes x \otimes x)^T \Delta_{\text{antisym}} = 0$. En consecuencia, las pruebas basadas únicamente en comparar trayectorias simuladas frente a oráculos (como Dynare) son completamente ciegas a errores de programación antisimétricos en tensores de derivadas. Se requiere una auditoría explícita de defectos de simetría tensorial.
- **Derivación analítica**:
  1. Sea $T \in \mathbb{R}^{n \times n \times n}$ un tensor de orden 3. Su contraparte completamente simetrizada es:
     $$T^{\text{sym}}_{i,j,k} = \frac{1}{3!} \sum_{\pi \in S_3} T_{\pi(i), \pi(j), \pi(k)}$$
  2. Para dimensión $n=2$, el número total de elementos del tensor es $2^3 = 8$. El número de combinaciones simétricas únicas es $\binom{n+k-1}{k} = \binom{2+3-1}{3} = 4$.
  3. El subespacio ortogonal complementario consta de $8 - 4 = 4$ direcciones antisimétricas independientes. Para cualquier error $\Delta$ en este subespacio:
     $$(x \otimes x \otimes x)^T \text{vec}(\Delta) = 0 \quad \forall x \in \mathbb{R}^2$$
  4. Se define la norma de defecto de simetría tensorial como:
     $$\text{defect}(T) = \max_{\pi \in S_k} \|T - P_\pi(T)\|_\infty$$
- **Tarea computacional y aserciones**:
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

### Ejercicio 4: ¿Precaución o Jensen? Descomposición del aumento en el capital medio
- **Nivel**: Avanzado (Stretch)
- **Intuición económica**:
  En algunas calibraciones de crecimiento estocástico, la media ergódica del capital supera su estado estacionario determinista. Ese signo y el de la corrección por riesgo no son universales. La literatura macroeconómica suele atribuir íntegramente esta diferencia al "ahorro precautorio" derivado de la prudencia ($u''' > 0$). Sin embargo, la desigualdad de Jensen (la curvatura de las funciones de política sobre la varianza de los estados y choques) también genera un desplazamiento positivo del capital medio, incluso en modelos donde las correcciones de riesgo son estrictamente nulas (como el modelo de Brock–Mirman con utilidad logarítmica). Mediante perturbación de orden superior, el desplazamiento del capital ergódico puede descomponerse rigurosamente en: (1) correcciones explícitas de riesgo de la política, (2) curvatura de dispersión de estados endógenos, y (3) curvatura de varianza de choques exógenos.
- **Derivación analítica**:
  1. La transición de estados de segundo orden es:
     $$x_{t+1} = h_x x_t + \frac{1}{2} H_{xx} (x_t \otimes x_t) + \frac{1}{2} H_{uu} (u_{t+1} \otimes u_{t+1}) + \frac{1}{2} h_{\sigma\sigma}$$
  2. Tomando esperanzas incondicionales ($E[x_{t+1}] = E[x_t]$):
     $$(I - h_x) E[x] = \frac{1}{2} H_{xx} \text{vec}(\Omega) + \frac{1}{2} H_{uu} \text{vec}(\Sigma) + \frac{1}{2} h_{\sigma\sigma}$$
     donde $\Omega \equiv \text{Var}(x_t)$ resuelve la ecuación de Lyapunov discreta $\Omega = h_x \Omega h_x^T + h_u \Sigma h_u^T$.
  3. Componente precautorio: $x_{\text{prec}} = \frac{1}{2}(I - h_x)^{-1} h_{\sigma\sigma}$, idéntico al desplazamiento del estado estacionario con riesgo.
  4. Componente de curvatura de Jensen: $x_{\text{jensen}} = \frac{1}{2}(I - h_x)^{-1} [H_{xx} \text{vec}(\Omega) + H_{uu} \text{vec}(\Sigma)]$.
  5. En el modelo de Brock–Mirman ($\gamma = 1$), $h_{\sigma\sigma} \equiv 0$, de modo que todo el desplazamiento es desigualdad de Jensen:
     $$E[k_t] - k_{ss} \approx \frac{1}{2} k_{ss} \text{Var}(\log k_t)$$
  6. En una calibración CRRA elegida, investiga si $h_{\sigma\sigma}^k$ cambia de signo al variar $\gamma$. Cualquier umbral depende de la calibración y del sistema de coordenadas; no es universal.
- **Tarea computacional y aserciones**:
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
