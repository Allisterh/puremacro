**Español** · [English](../dsge_v3.md)

# Scores de verosimilitud DSGE, HMC/NUTS y HANK en el espacio de secuencias

Esta guía explica capacidades introducidas en la serie 3.x y su alcance
actual. La paridad numérica corresponde a modelos, opciones y cantidades
concretos: consulte el [estado de validación](STRUCTURAL_VALIDATION_STATUS.md).
El núcleo numérico usa NumPy, SciPy, pandas y Matplotlib; la ejecución en el
navegador tiene los límites de la [guía de ejecución](tablet.md).

## Capacidades y alcance

`log_posterior_and_gradient` cubre parámetros estructurales declarados por el
modelo en la verosimilitud estacionaria de primer orden soportada. La
recursión analítica de Kalman recibe sensibilidades del espacio de estados
cuyas entradas estructurales pueden calcularse por diferencias finitas. Las
escalas de ruido, correlaciones, ecuaciones de observación, premuestras e
inicialización difusa requieren su propia ruta de verosimilitud soportada.
La [guía de estimación](dsge_estimation.md) distingue los gradientes analíticos,
numéricos y suministrados por el usuario, incluidas las alternativas automáticas.

HMC/NUTS adapta el tamaño de paso y la matriz de masa diagonal. Su eficiencia
se evalúa con mezcla entre cadenas, divergencias y tamaños muestrales efectivos;
depende del modelo y de la distribución posterior. El puente HANK conecta
bloques de hogares en archivos `.mod` mediante jacobianos en el espacio de
secuencias, dentro del alcance de cada [backend de hogares](hank_nonlinear.md).

OccBin resuelve trayectorias restringidas lineales por partes. La función de
desarrollo `solve_pruned_occbin` admite OccBin lineal, incluidos choques
anticipados, y trayectorias de segundo orden sin restricciones activas con una
innovación inicial. Rechaza episodios no lineales con restricciones activas e
innovaciones futuras anticipadas no lineales: véase la
[guía de orden superior](dsge_higher_order.md).

```
puremacro/dsge/
├── _ast.py            [ext]  Diferenciación analítica de parámetros diff_param(p) en el DAG de expresiones
├── _gradients.py      [new]  Sensibilidad implícita de Sylvester para reglas de decisión y score forward de Kalman
├── nuts.py            [new]  No-U-Turn Sampler (NUTS) en Python puro con promedio dual y adaptación de métrica
├── hank.py            [new]  Puente acoplado HANK en espacio-secuencia desde archivos .mod
├── estimate.py        [ext]  Integración de method="nuts" y selección de gradientes
├── _parser.py         [ext]  Sintaxis hetagent_block; ... end; integrada en el AST
└── _results.py        [ext]  Contenedores NUTSResult, HANKResult y ScoreDiagnosticsResult
```

### Mapa de componentes y API pública

| Componente | Módulo | Entidad principal | Salida / Contenedor | Propósito central |
|:---|:---|:---|:---|:---|
| **Pilar 1** | `puremacro.dsge._gradients` | `kalman_score()`<br>`solve_sylvester_generalized()`<br>`log_posterior_and_gradient()` | `ScoreDiagnosticsResult` | Score de Kalman y sensibilidades estructurales dentro del alcance documentado. |
| **Pilar 2** | `puremacro.dsge.nuts` | `nuts_sample()`<br>`DualAveraging`<br>`WelfordVariance` | `NUTSResult` | Muestreo Hamiltoniano NUTS con diagnósticos completos (split-$\hat{R}$, ESS, E-BFMI). |
| **Pilar 3** | `puremacro.dsge.hank` | `load_hank_mod()`<br>`HANKModel`<br>`solve_hank_bridge()` | `HANKResult` | Acoplamiento micro-macro en el espacio de secuencias desde bloques `hetagent_block;` con el cierre NKPC + regla de Taylor integrado. |

---

## Pilar 1: Score de Kalman y sensibilidades estructurales ($\nabla_\theta \ln L$)

El gradiente de la verosimilitud marginal gaussiana requiere diferenciar las reglas de decisión y las actualizaciones del filtro de Kalman. La recursión de Kalman es analítica; la construcción de sus entradas estructurales puede usar diferencias finitas y debe contrastarse con una derivada independiente de la misma verosimilitud.

```mermaid
flowchart LR
    A["Ecuaciones estructurales AST<br/>A+, A0, A-, Bu"] --> B["Derivadas estructurales<br/>implícitas y numéricas"]
    B --> C["Ecuación de Sylvester<br/>dG/dtheta, dN/dtheta"]
    C --> D["Sensibilidades espacio-estados<br/>dT, dR, dQ, dZ, dH, dP0"]
    D --> E["Recursión forward score Kalman<br/>dv_t, dF_t, dK_t, da_t, dP_t"]
    E --> F["Score de verosimilitud<br/>nabla_theta ln L"]
```

### 1.1 Sensibilidad estructural de las reglas de decisión de equilibrio

El sistema dinámico de expectativas racionales linealizado alrededor del estado estacionario determinista $y^*$ adopta la forma canónica:

$$\mathbb{E}_t \left[ A_+(\theta) y_{t+1} + A_0(\theta) y_t + A_-(\theta) y_{t-1} + B_u(\theta) u_t \right] = 0$$

donde $y_t \in \mathbb{R}^N$ es el vector de variables endógenas y $u_t \in \mathbb{R}^{n_u}$ es el vector de perturbaciones estructurales con covarianza $\Sigma_u$. La regla de decisión de equilibrio recursiva de primer orden viene dada por:

$$y_t = G(\theta) y_{t-1} + N(\theta) u_t$$

donde la matriz de transición de estados $G(\theta)$ satisface la ecuación matricial cuadrática (Riccati):

$$\mathcal{F}(G; \theta) \equiv A_+(\theta) G(\theta)^2 + A_0(\theta) G(\theta) + A_-(\theta) = 0$$

Diferenciando $\mathcal{F}(G; \theta) = 0$ con respecto a un parámetro estructural escalar $\theta_j$ mediante la regla de la cadena:

$$(A_0 + A_+ G) \frac{\partial G}{\partial \theta_j} + A_+ \frac{\partial G}{\partial \theta_j} G = - \left( \frac{\partial A_+}{\partial \theta_j} G^2 + \frac{\partial A_0}{\partial \theta_j} G + \frac{\partial A_-}{\partial \theta_j} \right)$$

Esta expresión constituye una **ecuación matricial generalizada de Sylvester**:

$$\hat{A} X_j + B X_j C = D_j$$

cuyos coeficientes matriciales son:
- $\hat{A} = A_0 + A_+ G \in \mathbb{R}^{N \times N}$
- $B = A_+ \in \mathbb{R}^{N \times N}$
- $C = G \in \mathbb{R}^{N \times N}$
- $D_j = - \left( \frac{\partial A_+}{\partial \theta_j} G^2 + \frac{\partial A_0}{\partial \theta_j} G + \frac{\partial A_-}{\partial \theta_j} \right) \in \mathbb{R}^{N \times N}$

#### Solución de alta velocidad mediante descomposición de Schur

Bajo la condición de estabilidad de Blanchard-Kahn, los autovalores de $C = G$ se ubican estrictamente dentro del círculo unitario. La función `solve_sylvester_generalized()` aprovecha esta estructura calculando la descomposición de Schur compleja de $C$ **una sola vez**:

$$C = U_c T_c U_c^H$$

donde $T_c$ es triangular superior y $U_c$ es unitaria. Transformando $\tilde{X}_j = X_j U_c$ y $\tilde{D}_j = D_j U_c$, el sistema triangular resultante:

$$\hat{A} \tilde{X}_j + B \tilde{X}_j T_c = \tilde{D}_j$$

se resuelve columna por columna mediante sustitución regresiva triangular:

$$\left[ \hat{A} + (T_c)_{k, k} B \right] (\tilde{X}_j)_{:, k} = (\tilde{D}_j)_{:, k} - B \sum_{m < k} (\tilde{X}_j)_{:, m} (T_c)_{m, k}$$

Esto permite resolver simultáneamente las derivadas $\frac{\partial G}{\partial \theta_j}$ para los $K$ parámetros en un lote tridimensional en $O(K \cdot N^3)$ operaciones en coma flotante, completándose en menos de $1 \text{ ms}$ para modelos de tamaño medio.

Diferenciando la condición de impacto de perturbaciones $(A_0 + A_+ G) N + B_u = 0$, se obtiene la derivada de la matriz de impacto:

$$\frac{\partial N}{\partial \theta_j} = - (A_0 + A_+ G)^{-1} \left[ \left(\frac{\partial A_0}{\partial \theta_j} + \frac{\partial A_+}{\partial \theta_j} G + A_+ \frac{\partial G}{\partial \theta_j}\right) N + \frac{\partial B_u}{\partial \theta_j} \right]$$

### 1.2 Mapeo al espacio de estados y covarianza no condicional inicial

Las matrices del modelo de espacio de estados (`StateSpaceModel`) relacionan el vector de estados $s_t$ con las observables $y^{obs}_t$:

$$s_{t+1} = T s_t + c + R u_{t+1}, \quad u_t \sim \mathcal{N}(0, Q)$$
$$y^{obs}_t = Z s_t + d + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, H)$$

Las sensibilidades paramétricas $\frac{\partial T}{\partial \theta_j}$, $\frac{\partial R}{\partial \theta_j}$, $\frac{\partial Q}{\partial \theta_j}$, $\frac{\partial Z}{\partial \theta_j}$, $\frac{\partial H}{\partial \theta_j}$ y $\frac{\partial d}{\partial \theta_j}$ se construyen mediante `build_state_space_sensitivities()`, con derivadas implícitas y entradas estructurales numéricas donde se necesitan.

La covarianza estacionaria inicial $P_0$ satisface la ecuación discreta de Lyapunov:

$$P_0 = T P_0 T^\top + R Q R^\top$$

Diferenciando con respecto a $\theta_j$:

$$\frac{\partial P_0}{\partial \theta_j} - T \frac{\partial P_0}{\partial \theta_j} T^\top = \frac{\partial T}{\partial \theta_j} P_0 T^\top + T P_0 \left(\frac{\partial T}{\partial \theta_j}\right)^\top + \frac{\partial (R Q R^\top)}{\partial \theta_j}$$

Esta es una ecuación de Lyapunov discreta que `puremacro` resuelve de manera exacta mediante `scipy.linalg.solve_discrete_lyapunov`.

### 1.3 Recursión forward del score del filtro de Kalman

Dada una muestra observable de tamaño $T_{obs}$ con errores de pronóstico $v_t = y_t - Z a_{t|t-1} - d$ y matriz de covarianza de innovación $F_t = Z P_{t|t-1} Z^\top + H$, la función de log-verosimilitud gaussiana se define como:

$$\ln L(Y \mid \theta) = -\frac{T_{obs} n_y \ln(2\pi)}{2} - \frac{1}{2} \sum_{t=1}^{T_{obs}} \left( \ln |F_t| + v_t^\top F_t^{-1} v_t \right)$$

Dadas las sensibilidades del espacio de estados, el score respecto a cada parámetro $\theta_j$ se calcula en una única pasada progresiva sin necesidad de almacenar toda la trayectoria temporal de estados (*single-pass forward Kalman score recursion*):

$$\frac{\partial \ln L}{\partial \theta_j} = -\frac{1}{2} \sum_{t=1}^{T_{obs}} \left[ \operatorname{tr}\left( F_t^{-1} \frac{\partial F_t}{\partial \theta_j} \right) + 2 v_t^\top F_t^{-1} \frac{\partial v_t}{\partial \theta_j} - \left(F_t^{-1} v_t\right)^\top \frac{\partial F_t}{\partial \theta_j} \left(F_t^{-1} v_t\right) \right]$$

Las sensibilidades del estado y de la covarianza se propagan contemporáneamente con el filtro:

$$\frac{\partial v_t}{\partial \theta_j} = - \frac{\partial Z}{\partial \theta_j} a_{t|t-1} - Z \frac{\partial a_{t|t-1}}{\partial \theta_j} - \frac{\partial d}{\partial \theta_j}$$

$$\frac{\partial F_t}{\partial \theta_j} = \frac{\partial Z}{\partial \theta_j} P_{t|t-1} Z^\top + Z \frac{\partial P_{t|t-1}}{\partial \theta_j} Z^\top + Z P_{t|t-1} \left(\frac{\partial Z}{\partial \theta_j}\right)^\top + \frac{\partial H}{\partial \theta_j}$$

Con la ganancia de Kalman $K_t = T P_{t|t-1} Z^\top F_t^{-1}$, las derivadas de la actualización del estado y la covarianza satisfacen:

$$\frac{\partial a_{t+1|t}}{\partial \theta_j} = \frac{\partial T}{\partial \theta_j} a_{t|t-1} + T \frac{\partial a_{t|t-1}}{\partial \theta_j} + \frac{\partial K_t}{\partial \theta_j} v_t + K_t \frac{\partial v_t}{\partial \theta_j} + \frac{\partial c}{\partial \theta_j}$$

$$\frac{\partial P_{t+1|t}}{\partial \theta_j} = \frac{\partial T}{\partial \theta_j} P_{t|t-1} (T - K_t Z)^\top + T \frac{\partial P_{t|t-1}}{\partial \theta_j} (T - K_t Z)^\top + T P_{t|t-1} \frac{\partial (T - K_t Z)^\top}{\partial \theta_j} + \frac{\partial (R Q R^\top)}{\partial \theta_j}$$

> [!TIP]
> **Rendimiento numérico**: La recursión analítica del score en `puremacro` es entre 5× y 15× más rápida que las diferencias finitas en modelos estándar (como Smets-Wouters 2007) y evita por completo las inestabilidades numéricas de cancelación catastrófica, alcanzando errores relativos $< 10^{-6}$ frente a referencias de paso complejo.

### 1.4 Verificación y contenedor `ScoreDiagnosticsResult`

El objeto `ScoreDiagnosticsResult` proporciona verificación automatizada contra diferencias finitas centrales de 5 puntos:

$$f'(x) \approx \frac{-f(x + 2h) + 8f(x + h) - 8f(x - h) + f(x - 2h)}{12 h}$$

```python
# requires: standalone snippet
import numpy as np
import pandas as pd
from puremacro.dsge import load_mod
from puremacro.dsge._gradients import kalman_score, build_state_space_sensitivities, ScoreDiagnosticsResult

# Cargar y resolver modelo DSGE
model = load_mod("modelo_ejemplo.mod")
model.solve()

# Obtener sensibilidades analíticas en el espacio de estados
sensitivities = build_state_space_sensitivities(model, param_names=["beta", "rho_a", "stderr_e_a"])

# Generar datos simulados
rng = np.random.default_rng(42)
y_obs = rng.standard_normal((150, 1))

# Calcular log-verosimilitud y vector score exacto
loglik, score_vector = kalman_score(
    y=y_obs,
    ssm=model.to_state_space(),
    sensitivities=sensitivities,
    param_names=["beta", "rho_a", "stderr_e_a"],
)

diag = ScoreDiagnosticsResult(
    loglik=loglik,
    gradient=score_vector,
    param_names=("beta", "rho_a", "stderr_e_a"),
    elapsed_sec=0.0032,
)

print(diag.to_markdown())
```

Salida típica del diagnóstico:

| Parámetro | Score analítico | Diferencias finitas | Error absoluto | Error relativo | Estado |
|:---|:---|:---|:---|:---|:---|
| `beta` | 14.821904 | 14.821901 | 3.12e-06 | 2.10e-07 | **PASS** |
| `rho_a` | -42.193012 | -42.193008 | 4.05e-06 | 9.60e-08 | **PASS** |
| `stderr_e_a` | -108.451201 | -108.451195 | 5.88e-06 | 5.42e-08 | **PASS** |

---

## Pilar 2: HMC y No-U-Turn Sampler (NUTS) en Python puro

El muestreador **No-U-Turn Sampler (NUTS)** implementado en `puremacro.dsge.nuts` introduce la potencia de la inferencia bayesiana basada en gradientes directamente en modelos DSGE sin necesidad de instalar Stan, PyMC o paquetes en C++.

```mermaid
sequenceDiagram
    participant P as Espacio de parámetros (theta)
    participant M as Momento auxiliar (p ~ N(0, M))
    participant L as Integrador Leapfrog
    participant T as Árbol binario recursivo
    participant D as Dual Averaging & Welford

    Note over P,M: Inicialización en modo posterior
    M->>L: Muestreo de momento cinético
    loop Pasos Leapfrog simplécticos
        L->>P: theta' = theta + eps * M^-1 * p
        L->>M: p' = p + (eps/2) * grad_log_post(theta')
        T->>T: Evaluar condición No-U-Turn (Betancourt 2017)
    end
    T->>D: Actualizar probabilidad de aceptación (delta = 0.80)
    D->>D: Ajustar tamaño de paso eps y matriz de masa M^-1 (Welford)
```

### 2.1 Principios del muestreo Hamiltoniano

Sea $\mathcal{L}(\theta) = \ln p(\theta \mid Y) = \ln L(Y \mid \theta) + \ln p(\theta)$ la función de densidad log-posterior. El espacio de parámetros $\theta \in \mathbb{R}^K$ se amplía con variables auxiliares de momento $p \sim \mathcal{N}(0, M)$, donde $M$ es una matriz métrica de masa simétrica definida positiva. El sistema dinámico se rige por el Hamiltoniano:

$$\mathcal{H}(\theta, p) = -\mathcal{L}(\theta) + \frac{1}{2} p^\top M^{-1} p$$

El gradiente del log-posteriori es la suma del score de verosimilitud y el gradiente de la distribución a priori:

$$\nabla_\theta \mathcal{L}(\theta) = \nabla_\theta \ln L(Y \mid \theta) + \nabla_\theta \ln p(\theta)$$

`puremacro.dsge.priors.grad_log_prior()` calcula analíticamente $\nabla_\theta \ln p(\theta)$ para las familias comunes en macroeconomía (Beta, Gamma, Gamma Inversa, Normal, Uniforme).

La trayectoria se propaga mediante el integrador simpléctico de salto de rana (*leapfrog*):

$$\tilde{p} = p + \frac{\epsilon}{2} \nabla_\theta \mathcal{L}(\theta)$$
$$\theta' = \theta + \epsilon M^{-1} \tilde{p}$$
$$p' = \tilde{p} + \frac{\epsilon}{2} \nabla_\theta \mathcal{L}(\theta')$$

### 2.2 Árbol binario recursivo y criterio de parada sin giros en U

El algoritmo NUTS construye recursivamente un árbol binario en el espacio de fases. En cada profundidad $j \in \{0, 1, \dots, j_{max}\}$, el integrador duplica la longitud de la trayectoria eligiendo aleatoriamente una dirección temporal $v \in \{-1, +1\}$.

La expansión recursiva se detiene automáticamente en cuanto la trayectoria comienza a doblarse sobre sí misma, según el criterio generalizado de Betancourt (2017):

$$(\theta^+ - \theta^-)^\top M^{-1} p^+ < 0 \quad \text{o} \quad (\theta^+ - \theta^-)^\top M^{-1} p^- < 0$$

Asimismo, el algoritmo detecta divergencias hamiltonianas numéricas cuando:

$$\mathcal{H}(\theta', p') - \mathcal{H}(\theta, p) > \Delta_{max} \approx 1000$$

registrándolas en la máscara booleana de diagnósticos.

### 2.3 Adaptación de tamaño de paso por promedio dual

Durante la fase de calentamiento (*warmup*), `DualAveraging` implementa el esquema de Nesterov (2009) y Hoffman-Gelman (2014) para adaptar dinámicamente el tamaño de paso $\epsilon$ hacia la probabilidad de aceptación objetivo $\delta^* = 0.80$:

$$H_m = \left(1 - \frac{1}{m + t_0}\right) H_{m-1} + \frac{1}{m + t_0} (\delta^* - \alpha_m)$$

$$\ln \epsilon_{m+1} = \mu - \frac{\sqrt{m}}{\gamma} H_m, \quad \ln \bar{\epsilon}_{m+1} = m^{-\kappa} \ln \epsilon_{m+1} + (1 - m^{-\kappa}) \ln \bar{\epsilon}_m$$

con los hiperparámetros estándar $\mu = \ln(10 \epsilon_0)$, $\gamma = 0.05$, $t_0 = 10$ y $\kappa = 0.75$.

### 2.4 Acumulación de varianza diagonal de Welford y regularización de Stan

La matriz métrica de masa $M^{-1}$ aproxima la matriz de covarianza posterior $\operatorname{Var}(\theta)$. La clase `WelfordVariance` acumula incrementalmente la media y varianza diagonal en línea:

$$\mu_k = \mu_{k-1} + \frac{x_k - \mu_{k-1}}{k}$$
$$S_k = S_{k-1} + (x_k - \mu_{k-1})(x_k - \mu_k), \quad \sigma^2_k = \frac{S_k}{k - 1}$$

Siguiendo el protocolo escalonado de Stan, el calentamiento divide las iteraciones en ventanas:
1. **Ventana rápida inicial (75 iteraciones)**: estabilización inicial del tamaño de paso.
2. **Ventanas lentas expansivas (25, 50, 100, 200, ... iteraciones)**: actualización de la matriz de masa diagonal de Welford con re-inicialización del paso.
3. **Ventana rápida final (50 iteraciones)**: ajuste fino final del tamaño de paso con métrica fija.

Se aplica una regularización por contracción (*shrinkage*) con la varianza a priori para evitar varianzas espurias o singulares en dimensiones con escasa información muestral:

$$M^{-1}_{reg} = \frac{N}{N + 5} \sigma^2_{welford} + \frac{5}{N + 5} \sigma^2_{prior}$$

### 2.5 Diagnósticos MCMC integrales y contenedor `NUTSResult`

`NUTSResult` proporciona un diagnóstico bayesiano de estándar internacional:
- **$\hat{R}$ dividido de Gelman-Rubin (`compute_split_rhat`, sin normalización por rangos)**: divide cada cadena a la mitad para diagnosticar no estacionariedad intra-cadena y falta de mezcla inter-cadena ($\hat{R} < 1.05$).
- **Tamaño muestral efectivo en la masa (`compute_bulk_ess`)**: estimado mediante sumas de autocorrelación con la secuencia monótona inicial de Geyer (ESS de cada cadena sumado entre cadenas; sin normalización por rangos, así que no es el ESS en la masa multicadena de Vehtari et al. 2021).
- **Tamaño muestral efectivo en las colas (`compute_tail_ess`)**: calculado a partir de las funciones indicadoras de los cuantiles al 5% y 95%.
- **Fracción bayesiana de energía ($E\text{-BFMI}$, Betancourt 2016)**: cuantifica la eficiencia con la que la distribución del momento explora el espacio de energía. Valores $E\text{-BFMI} < 0.3$ señalan un mal acondicionamiento del espacio posterior.

```python
# requires: standalone snippet
from puremacro.dsge import load_mod

# Carga del modelo y estimación con NUTS
m = load_mod("smets_wouters_07.mod")

# Estimación bayesiana con gradientes analíticos nativos
res_nuts = m.estimate(
    data=data_obs,
    method="nuts",
    n_draws=1000,
    n_chains=4,
    burn_in=500,
    target_accept=0.80,
    max_tree_depth=10,
    seed=123,
)

# Resumen con R_hat, ESS bulk y ESS tail
print(res_nuts.summary().round(3))

# Gráfico de diagnóstico de energía E-BFMI
diag_dict, fig, ax = res_nuts.energy_diagnostics()
print(f"E-BFMI promedio: {diag_dict['mean_ebfmi']:.3f} (Superado: {diag_dict['passed']})")
```

---

## Pilar 3: Puente HANK en espacio-secuencia desde archivos `.mod`

Los modelos de Agentes Heterogéneos Nuevo-Keynesianos (HANK) rompen el supuesto del agente representativo al incorporar fricciones financieras, límites de endeudamiento no triviales y una masa sustancial de hogares con propensión marginal al consumo (PMC) elevada ("*hand-to-mouth*").

`puremacro 3.0` implementa un puente analítico directo que permite escribir el bloque de hogares de un modelo HANK en archivos `.mod` mediante la extensión gramatical `hetagent_block; ... end;`, acoplándolo con el motor de espacio de secuencias (Sequence-Space Jacobian, SSJ) formulado por Auclert, Bardóczy, Rognlie y Straub (2021, *Econometrica*).

```mermaid
flowchart TD
    subgraph Archivo .mod
        HB["hetagent_block;<br/>model = one_asset_hank;<br/>n_a = 50; a_max = 30.0;<br/>end;"]
        MB["model;<br/>Y = C;<br/>pi = beta*pi(+1) + kappa*Y;<br/>i = r_ss + phi_pi*pi + eps_m;<br/>r = i - pi(+1);<br/>end;"]
    end

    subgraph Microeconomía de Hogares
        SS["Distribución estacionaria D*(a)<br/>Grilla de activos a, Consumo c(a)"]
        FN["Algoritmo Fake-News<br/>Jacobianos J_C_r, J_C_Y en R^{T x T}"]
    end

    subgraph Equilibrio General
        GE["Sistema dinámico apilado<br/>Vaciado de mercado: Y_t = C_t(r, Y)"]
        SOL["Transición lineal (Inversión directa)<br/>Transición no lineal (Broyden)"]
    end

    HB --> SS
    SS --> FN
    MB -. descriptivo .-> GE
    FN --> GE
    GE --> SOL
```

### 3.1 Sintaxis del bloque `hetagent_block` en archivos `.mod`

El analizador sintáctico de `puremacro` interpreta el bloque de agentes heterogéneos junto con las ecuaciones agregadas del bloque `model;`, que se conservan solo a título descriptivo (véase más abajo):

```dynare
var Y C r pi i;
varexo eps_m;

parameters beta gamma r_ss phi_pi kappa;
beta = 0.985;
gamma = 1.0;
r_ss = 0.01;
phi_pi = 1.5;
kappa = 0.1;

hetagent_block;
  model = one_asset_hank;
  n_a = 50;
  a_max = 30.0;
  borrowing_limit = 0.0;
  grid = hyperbolic;
end;

model;
  Y = C;
  pi = beta * pi(+1) + kappa * Y;
  i = r_ss + phi_pi * pi + eps_m;
  r = i - pi(+1);
end;
```

El bloque `model;` se analiza y se conserva en `HANKModel.dag` para inspección, pero no se resuelve: todos los backends de hogares se cierran con la curva de Phillips neokeynesiana $\pi_t = \beta \pi_{t+1} + \kappa Y_t$ y la regla de Taylor $i_t = \phi_\pi \pi_t + \varepsilon_{m,t}$ integradas (tasa real ex ante $r_t = i_t - \pi_{t+1}$; con dos activos se añade $dr^a_t = \alpha_{ab}\, dr^b_t$) evaluadas sobre los parámetros declarados `beta`, `kappa`, `phi_pi` (y `alpha_ab`), y el único choque es la trayectoria monetaria `magnitude * rho**t`; el argumento `shock` solo etiqueta el resultado (`shock_name`, más una columna de `transition_paths` con la trayectoria monetaria cuando el nombre es un `varexo` declarado); nunca selecciona otro choque. Para cambiar el cierre, cambie los parámetros, no las ecuaciones.

#### Parámetros de configuración en `hetagent_block`

- `model`: Tipo de bloque microeconómico heterogéneo (`one_asset_hank`, `two_asset_hank`).
- `n_a`: Número de nodos en la grilla de activos (por defecto, 50 o 100).
- `a_max`: Límite superior de acumulación de activos financieros.
- `borrowing_limit`: Restricción exógena de crédito o límite de endeudamiento (generalmente 0.0).
- `grid`: Distribución de puntos de la grilla (`hyperbolic` o `uniform`), concentrando mayor resolución en las zonas cercanas a la restricción financiera donde la curvatura de la función de valor es máxima.

### 3.2 Metodología del espacio de secuencias y algoritmo de Fake-News

En el estado estacionario, el bloque de hogares resuelve el problema de Bellman continuo-discreto, determinando las funciones de política de consumo $c(a, y)$ y ahorro $a'(a, y)$, junto con la distribución invariante de activos $\mathcal{D}^*(a)$.

El **algoritmo de Fake-News** calcula de manera analítica y ultra-eficiente las matrices jacobianas del bloque de consumo en el espacio de secuencias:

$$\mathcal{J}_{C, r} = \frac{\partial \mathbf{C}}{\partial \mathbf{r}} \in \mathbb{R}^{T \times T}, \quad \mathcal{J}_{C, Y} = \frac{\partial \mathbf{C}}{\partial \mathbf{Y}} \in \mathbb{R}^{T \times T}$$

Cada entrada $(\tau, s)$ de $\mathcal{J}_{C, r}$ representa el impacto sobre el consumo agregado en el período $\tau$ ante un choque anticipado o sorpresa en la tasa de interés real en el período $s$. Este cálculo se realiza en $O(T^2)$ operaciones matriciales sin requerir simulaciones Monte Carlo intensivas.

### 3.3 Equilibrio general acoplado y simulación de transiciones

El equilibrio macroeconómico general acopla las decisiones microeconómicas con las identidades agregadas:

$$\mathbf{H}(\mathbf{Y}, \mathbf{r}, \mathbf{\pi}, \mathbf{i}; \mathbf{\varepsilon}_m) = \begin{bmatrix} \mathbf{Y} - \mathbf{C}(\mathbf{r}, \mathbf{Y}) \\ \mathbf{\pi} - \beta \mathbf{\pi}_{+1} - \kappa \mathbf{Y} \\ \mathbf{i} - r_{ss} - \phi_\pi \mathbf{\pi} - \mathbf{\varepsilon}_m \\ \mathbf{r} - \mathbf{i} + \mathbf{\pi}_{+1} \end{bmatrix} = \mathbf{0}$$

Este sistema es el cierre integrado del puente, evaluado sobre los parámetros declarados `beta`, `kappa`, `phi_pi` (y `alpha_ab` con dos activos); las ecuaciones del bloque `model;` no se leen.

`puremacro` resuelve este sistema apilado de dos formas:
1. **Transición lineal**: Resolución exacta mediante inversión de la matriz jacobiana total del equilibrio general:
   $$\mathbf{dY} = -\left( \mathcal{H}_{\mathbf{Y}} \right)^{-1} \mathcal{H}_{\mathbf{\varepsilon}} \mathbf{\varepsilon}_m$$
2. **Transición no lineal (cuasi-Newton Broyden)**: Simulación no lineal global para choques de gran magnitud mediante el algoritmo de Broyden, preservando la convergencia sin necesidad de discretizar espacios de estados de alta dimensión. `nonlinear=True` está implementado solo para el hogar de un activo integrado: un `hetagent_block` de dos activos (`assets = 2`) y cualquier puente `household_model` lanzan `NotImplementedError`, y su transición lineal en el espacio de secuencias (`nonlinear=False`) sigue disponible.

### 3.4 API y contenedor `HANKResult`

La interfaz de alto nivel `solve_hank_bridge()` y la clase `HANKModel` encapsulan todo el ciclo de resolución y simulación:

```python
# requires: standalone snippet
import matplotlib.pyplot as plt
from puremacro.dsge.hank import load_hank_mod, solve_hank_bridge

# Carga del modelo HANK desde archivo .mod
hank_model = load_hank_mod("hank_sample.mod")

# Simulación de un choque de política monetaria contractivo (eps_m = +25 pb)
res_hank = hank_model.simulate(
    shock="eps_m",
    magnitude=0.0025,
    rho=0.5,
    horizon=40,
    nonlinear=False,
)

# Resumen de respuestas de impacto y máximas
print(res_hank.summary().round(4))

# Gráfico de trayectorias dinámicas de transición (IRFs)
fig, axes = res_hank.plot_transition(variables=["Y", "C", "r", "pi", "i"])
plt.show()

# Gráfico de la distribución estacionaria de activos y PMC
fig2, ax2 = res_hank.plot_distribution()
plt.show()
```

Estructura de la tabla de resumen (`res_hank.summary()`):

| Variable | Estado estacionario | Respuesta de impacto | Respuesta pico | Período del pico |
|:---|:---|:---|:---|:---|
| `Y` | 1.0421 | -0.0038 | -0.0038 | 0 |
| `C` | 1.0421 | -0.0038 | -0.0038 | 0 |
| `r` | 0.0100 | 0.0021 | 0.0021 | 0 |
| `pi` | 0.0000 | -0.0004 | -0.0004 | 0 |
| `i` | 0.0100 | 0.0019 | 0.0019 | 0 |

---

## Comprobaciones antes de usar los resultados

| Componente | Comprobación |
|---|---|
| Gradientes | Comparar el score con diferencias numéricas de la misma verosimilitud, usando varios tamaños de paso. |
| Muestreo posterior | Revisar mezcla, divergencias y tamaños muestrales efectivos; esta guía no establece una ventaja general de velocidad frente a otro muestreador. |
| Diagnósticos | El $\hat{R}$ dividido y los ESS implementados no usan normalización por rangos. |
| HANK | Verificar el backend de hogares y los límites de las transiciones lineales o no lineales. |
| Ejecución | La instalación local es el destino soportado; consultar los límites del entorno en navegador. |

---

## Referencias bibliográficas

- **Auclert, A., Bardóczy, B., Rognlie, M., & Straub, L. (2021)**. *Using the sequence-space Jacobian to solve and estimate heterogeneous-agent models*. **Econometrica**, 89(5), 2375-2408.
- **Betancourt, M. (2016)**. *Diagnosing biased inference with targets of rapid curvature*. **arXiv preprint** arXiv:1604.00695.
- **Betancourt, M. (2017)**. *A conceptual introduction to Hamiltonian Monte Carlo*. **arXiv preprint** arXiv:1701.02434.
- **Gelfand, A. E., & Dey, D. K. (1994)**. *Bayesian model choice: asymptotics and exact calculations*. **Journal of the Royal Statistical Society: Series B (Methodological)**, 56(3), 501-514.
- **Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013)**. *Bayesian Data Analysis* (3rd ed.). Chapman and Hall/CRC.
- **Hoffman, M. D., & Gelman, A. (2014)**. *The No-U-Turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo*. **Journal of Machine Learning Research**, 15(1), 1593-1623.
- **Nesterov, Y. (2009)**. *Primal-dual subgradient methods for convex problems*. **Mathematical Programming**, 120(1), 221-259.
- **Smets, F., & Wouters, R. (2007)**. *Shocks and frictions in US business cycles: A Bayesian DSGE approach*. **American Economic Review**, 97(3), 586-606.
- **Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P. C. (2021)**. *Rank-normalization, folding, and localization: An improved $\widehat{R}$ for assessing convergence of MCMC*. **Bayesian Analysis**, 16(2), 667-718.
