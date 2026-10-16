> 🇬🇧 [English](../trade_gpu.md) · 🇪🇸 Español

# Backends de hardware: GPU, Apple MLX y jacobianos por lotes para comercio y GE espacial

`puremacro._backend` y `puremacro.trade.gpu` añaden aceleración por hardware opcional a los solucionadores de equilibrio general documentados en [Economía espacial cuantitativa y GE de comercio con gravedad](spatial_and_trade_ge.md). Intervienen dos capas:

1. **La capa de espacio de nombres de arreglos** (`puremacro._backend`): `numpy`, `mlx` (Apple Silicon) y `cupy` (NVIDIA) comparten una API de arreglos al estilo de NumPy. `AllenArkolakisModel.solve_equilibrium(..., backend=)` / `solve_counterfactual(..., backend=)` ejecutan su aplicación de contracción a través de ella, con un `RuntimeWarning` y un respaldo en NumPy cuando el backend pedido falta o falla. El solucionador de comercio `solve_trade_equilibrium` es solo NumPy: `backend="numpy"` es el único valor aceptado y cualquier otro lanza `NotImplementedError`.
2. **La capa de dispositivo** (`puremacro.trade.gpu`): un motor en PyTorch (CUDA / Apple MPS / CPU) y en Apple MLX nativo para el modelo de comercio multipaís y multisectorial de contabilidad consistente de `puremacro.trade` (insumos intermedios Cobb-Douglas, devolución de suma fija de la recaudación arancelaria), construido sobre operadores de Leontief preinvertidos, un jacobiano por lotes evaluado como un único GEMM con varios lados derechos, un paso de Levenberg-Marquardt (1944, 1963) con búsqueda lineal hacia atrás y continuación homotópica adaptativa (Allgower y Georg, 1990) en el esquema arancelario.

Ambas capas son opcionales. Nada de esta página es necesario para ejecutar los modelos: la ruta NumPy sigue siendo la implementación de referencia y el oráculo de corrección, y es la única disponible en Pyodide (iPad, navegador). Todo resultado acelerado de esta página se compara con ella.

---

## 1. Marco algorítmico

### 1.1 Qué backend ejecuta qué

`puremacro._backend.SUPPORTED` es `("numpy", "numba", "mlx", "cupy")`. `backend_available(name)` consulta `importlib.util.find_spec`, `available_backends()` enumera los instalados (NumPy primero) y `get_array_namespace(name)` devuelve `numpy`, `mlx.core` o `cupy` (`"numba"` es un backend de núcleos compilados, no un espacio de nombres de arreglos, y aquí lanza `ValueError`). `to_numpy(x)` trae cualquier arreglo al anfitrión, pasando CuPy por `cp.asnumpy` porque CuPy bloquea la conversión implícita.

- **`solve_trade_equilibrium(calib, ..., backend=)`** solo acepta `"numpy"`. Las soluciones aceleradas de comercio pasan por la capa de dispositivo de abajo, que resuelve el mismo modelo y devuelve el mismo tipo de resultado.
- **`AllenArkolakisModel.solve_equilibrium(backend=)`**: toda la aplicación de contracción amortiguada sobre $(w_i, L_i)$ (matriz núcleo $\tau_{ji}^{-\theta}$, acceso al mercado, el softmax de igualación de utilidades) corre en el dispositivo en `float32` con MLX y en `float64` con CuPy. Un dispositivo float32 solo puede certificar el punto fijo hasta su propio piso, así que cuando ese piso es más grueso que la `tol` pedida (siempre en MLX, donde los pisos del dispositivo son $\max(\text{tol}, 10^{-6})$ en el residuo y $10^{-5}$ en la dispersión de utilidades) el bucle del dispositivo se trata como arranque en caliente y la iteración **se termina en el anfitrión en NumPy float64** con la `tol` pedida. La tolerancia pedida se respeta y `converged` es veraz sea cual sea el backend; el ejemplo 4.4 muestra a MLX y NumPy coincidiendo a $10^{-9}$ tras las mismas 27 iteraciones. Tras la convergencia la población se renormaliza a $\bar L$, los salarios a media unitaria, y el índice de precios, los términos de acceso al mercado y los salarios reales se recalculan en NumPy float64, por lo que el residuo de conservación del trabajo es cero (al redondeo) en ambos backends. `solve_counterfactual` pasa `backend` tanto a la solución base como a la contrafactual.

### 1.2 Selección de dispositivo y `DeviceInfo`

`puremacro.trade.gpu.select_compute_device(preferred)` asigna a un nombre un par `(backend, device)`, donde `backend` es `"torch"`, `"mlx"` o `"numpy"` y `device` es `"cuda"`, `"mps"`, `"gpu"` o `"cpu"`:

| `preferred` | Resultado | Si no está disponible |
|---|---|---|
| `None` / `"auto"` | el primero de: torch CUDA, torch MPS, MLX GPU, torch CPU, NumPy CPU | nunca falla |
| `"numpy"` | `("numpy", "cpu")`, el evaluador serial de NumPy | nunca falla |
| `"cpu"` | `("torch", "cpu")` si PyTorch está instalado, `("numpy", "cpu")` si no | nunca falla |
| `"cuda"`, `"cuda:N"` | ese dispositivo CUDA de PyTorch | `RuntimeError` (falta PyTorch, `torch.cuda.is_available()` es `False` o el índice `N` está fuera de rango) |
| `"mps"` | `("torch", "mps")` | `RuntimeError` (falta PyTorch o `torch.backends.mps.is_available()` es `False`) |
| `"torch"` / `"pytorch"` | el mejor dispositivo PyTorch: CUDA, luego MPS, luego CPU | `RuntimeError` si falta PyTorch |
| `"mlx"` / `"apple_mlx"` | `("mlx", "gpu")` | `RuntimeError` si falta `mlx` |
| `"gpu"` | la mejor GPU de cualquier backend: CUDA, luego MPS, luego MLX | `RuntimeError` si no hay ninguna de las tres |

Todas las cadenas aceptadas están en la tabla; cualquier otra lanza `ValueError: Unknown device/backend ...; expected one of (...)` en lugar de resolverse en silencio como `"auto"`. `"numpy"` es un nombre real, así que se puede pedir explícitamente el evaluador serial, y `"cpu"` no exige PyTorch: se degrada a `("numpy", "cpu")`, que es lo que obtiene todo kernel de Pyodide.

`detect_device(preferred)` envuelve la misma elección en una dataclass congelada:

| Campo de `DeviceInfo` | Significado |
|---|---|
| `backend` | `"torch"`, `"mlx"` o `"numpy"` |
| `device` | `"cuda"`, `"mps"`, `"gpu"` (MLX) o `"cpu"` |
| `device_name` | nombre del adaptador CUDA, `"Apple Metal (<cpu>)"`, `"Apple MLX (<cpu>)"` o el procesador anfitrión |
| `total_memory_gb` | CUDA: `total_memory` de las propiedades del dispositivo; Apple Silicon: `sysctl hw.memsize` (la reserva unificada; `None` si no se puede ejecutar `sysctl`); `None` en CPU simple |
| `supports_float64` | `True` para CUDA y CPU, `False` para MPS, `False` para el flujo GPU de MLX |
| `is_uma` | `True` en Apple Silicon (memoria unificada: anfitrión y GPU comparten una reserva, sin copia al dispositivo) |

El `repr` resume la situación de precisión, p. ej. `f32-only` para MPS y `f32-gpu/f64-cpu-stream` para MLX.

### 1.3 Precisión: float64 frente a float32

Una diferencia finita hacia adelante divide el error de redondeo del residuo por el paso, de modo que un residuo en float32 produce un error relativo grande en el jacobiano y un paso de Newton divergente. **La regla de la capa de dispositivo es: ensamblar el jacobiano en float64 dondequiera que el dispositivo lo permita, y usar un dispositivo float32 solo cuando se pide explícitamente, y solo para la fase gruesa.**

- **CUDA y la CPU de PyTorch** ejecutan todo en float64.
- **MPS** (Apple Metal vía PyTorch) no tiene float64. Con `device="auto"` el solucionador sustituye en silencio el evaluador float64 de PyTorch en **CPU**. Con `device="mps"` pedido explícitamente emite un `RuntimeWarning` ("... only supports float32; finite-difference Jacobians are unusable in float32, so it is used for the coarse phase only and the Jacobian is polished in float64 on torch:cpu"), usa MPS mientras $\max_i |F_i| \ge 5 \times 10^6$ y solo en las cinco primeras iteraciones, y pule en float64 después.
- **MLX** guarda constantes float64 creadas bajo `mx.stream(mx.cpu)`, donde MLX ejecuta Accelerate/LAPACK (y donde `mx.exp`, que solo es exacto en float32, se sustituye por un `e ** x` exacto), y copias float32 para el flujo GPU. `BatchedJacobianEvaluator` **usa por defecto el flujo CPU**; el flujo GPU float32 solo se alcanza con `stream="gpu"` o en la fase gruesa de un dispositivo explícito. `solve_trade_equilibrium_mlx` es `solve_trade_equilibrium_gpu(device="mlx")` y sigue exactamente la misma política.

Sea cual sea el dispositivo, `result.metadata["jacobian_evaluations"]` cuenta los jacobianos por `backend:device:dtype`, de modo que la ruta de precisión que siguió una ejecución queda en el resultado. El residuo puntual `eval_macro_single`, los puntos de prueba de la búsqueda lineal y la comprobación final del residuo son NumPy float64 en el anfitrión, así que `max_residual` y `converged` son siempre magnitudes float64; solo las columnas del jacobiano de la fase gruesa pueden llevar redondeo float32. El ejemplo 4.3 lo mide: el jacobiano float32 de MPS es inservible (su error supera la mayor entrada), mientras que los jacobianos de la CPU de PyTorch y del flujo CPU de MLX reproducen el de NumPy serial al redondeo.

### 1.4 El jacobiano por lotes

El sistema de contabilidad consistente de `puremacro.trade` (véase [Contabilidad de comercio consistente](trade_accounting.md)) se condensa en un vector macro. Con $N$ países, $S$ sectores y $M = SN$ pares país-sector, `BatchedJacobianEvaluator` trabaja con

$$x_m = \begin{bmatrix} \log w \\ T \end{bmatrix} \in \mathbb{R}^{2N} \quad\text{o}\quad x_m = \begin{bmatrix} \log r \\ \log w \\ T \end{bmatrix} \in \mathbb{R}^{3N},$$

donde $w$ son los salarios, $r$ las rentas del capital y $T$ las transferencias de suma fija (presupuestos públicos). El ahorro externo es exógeno, como en el modelo completo, en unidades del numerario o como proporción fija del ingreso factorial mundial (`foreign_saving_units`), así que no es una incógnita. Los residuos macro son el primer precio de productor menos uno (el numerario), el vaciado del mercado de trabajo en todos los países salvo el primero (implicado por la ley de Walras una vez que vacían todos los mercados de bienes), el vaciado del mercado de capital (solo en el esquema completo) y los presupuestos públicos. Una solución del sistema macro resuelve todas las ecuaciones del sistema completo de contabilidad consistente.

El primer esquema impone $r_c = w_c$, que solo es exacto cuando la participación del capital es la misma en todos los sectores de un país: el vaciado del mercado de capital se sigue entonces del de trabajo. El esquema se elige con el argumento solo por nombre `factor_equivalence`, y `batch_size` es la grafía antigua, *validada* contra la calibración. `factor_equivalence=None` (por defecto) elige el esquema reducido exactamente cuando es exacto para la calibración y el completo en otro caso; `factor_equivalence=True` en una calibración donde no es exacto lanza `ValueError` en lugar de eliminar en silencio la condición de vaciado del capital. Si se pasa `batch_size`, debe valer $2N$ o $3N$ para *su* $N$: en la calibración de juguete de abajo, 4 o 6, y `batch_size=5` lanza `ValueError: batch_size=5 is inconsistent with a 2-country calibration: expected 4 (reduced layout, r == w) or 6 (full layout).` Pasar ambos argumentos con implicaciones contradictorias también lanza un error.

Dado $x_m$, los precios y las producciones brutas salen de dos sistemas de Leontief,

$$p = (I - B^\top)^{-1}\, v(x_m), \qquad y = (I - A)^{-1}\, d(x_m, p),$$

con $B^\top = (A \circ \tau)^\top / (1 - \text{tax})$ el operador de participaciones de costo con aranceles, $v$ el valor agregado unitario Cobb-Douglas y $A$ los coeficientes de insumo-producto. Ambas inversas se forman **una vez** (`(I - A)^{-1}` en la construcción, `(I - B^\top)^{-1}` en `update_tariffs`, es decir, una vez por esquema arancelario). Un jacobiano por diferencias hacia adelante necesita $F$ en los $B$ puntos perturbados $x_m + h_j e_j$, con $h_j = \varepsilon$ para los precios factoriales en logaritmos y $h_j = \varepsilon \max(|T_j|, s)$ para las transferencias, donde $s$ es la mediana del ingreso factorial base ($\varepsilon = 10^{-6}$ por defecto, `eps_fd`). Apilando los puntos perturbados como filas de $X \in \mathbb{R}^{B \times B}$, las soluciones de precios y producción de todo el lote se convierten en dos productos matriciales,

$$P = \left[(I - B^\top)^{-1} V^\top\right]^\top, \qquad Y = \left[(I - A)^{-1} D^\top\right]^\top, \qquad V, D \in \mathbb{R}^{B \times M},$$

y cada cuenta (demandas de factores, impuestos a la producción y al uso final, aranceles) se convierte en un producto matricial por lotes. Un único lanzamiento de núcleo sustituye a $B$ evaluaciones seriales del residuo, y

$$J_{:,j} = \frac{F(x_m + h_j e_j) - F(x_m)}{h_j}.$$

`ad_mode="forward"` sustituye las diferencias por diferenciación automática en modo directo (`torch.func.jacfwd` en PyTorch, `mx.jvp` columna a columna en MLX) y `ad_mode="vjp"` por el modo inverso (`torch.func.jacrev`, `mx.vjp`); ambos eliminan la sensibilidad al tamaño del paso. Ante cualquier fallo el evaluador emite un `RuntimeWarning` y vuelve a las diferencias finitas (nunca en silencio), y en el backend `numpy` cualquier modo AD avisa y usa diferencias finitas. Un vector macro cuya longitud no coincide con el esquema lanza `ValueError`, y también un `ad_mode` desconocido. El método `jvp(xm, v)` devuelve la derivada direccional $J v$ del mismo modo, con una diferencia centrada en float64 como último recurso.

### 1.5 Paso de Levenberg-Marquardt, equilibrado y búsqueda lineal

Cada iteración de `solve_trade_equilibrium_gpu` / `solve_trade_equilibrium_mlx` escala el jacobiano por ambos lados (equilibrado de Jacobi), $\tilde J = D_L J D_R$ con $D_R = \operatorname{diag}(1/\lVert J_{:,j}\rVert_2)$ y $D_L = \operatorname{diag}(1/\lVert (J D_R)_{i,:}\rVert_2)$, y resuelve las ecuaciones normales regularizadas

$$\left(\tilde J^\top \tilde J + \mu I\right) u = -\tilde J^\top D_L F(x_m), \qquad \Delta x_m = D_R\, u,$$

con `torch.linalg.solve` en float64 cuando PyTorch es el backend evaluador (en el dispositivo CUDA cuando ese es el de pulido, en la CPU en otro caso), luego `scipy.linalg.solve` y luego `scipy.linalg.lstsq`; gana el primero que funciona. El paso se reescala para que su bloque de precios factoriales quede dentro de $0.3$ en valor absoluto, y los precios factoriales en logaritmos se recortan a $[-5, 5]$. Una búsqueda lineal hacia atrás sobre la función de mérito $\Phi(x) = \lVert D_L F(x) \rVert_2$ prueba $\alpha = 1, \tfrac12, \tfrac14, \dots$ (como mucho 12 puntos) y acepta el primero con $\max_i|F_i| \le \text{tol}$, $\Phi(x + \alpha \Delta) < \Phi(x)$ o $\max_i |F_i(x + \alpha\Delta)| < \max_i |F_i(x)|$: una regla de descenso simple al estilo del retroceso de Armijo (1966), sin constante de descenso suficiente. El amortiguamiento sigue la actualización clásica de Marquardt: $\mu \leftarrow \max(0.3\mu, 10^{-8})$ tras un paso aceptado, $\mu \leftarrow \max(0.5\mu, 10^{-8})$ cuando solo mejoró el mérito el mejor punto de prueba, y $\mu \leftarrow \min(5\mu, 10^{2})$ en otro caso. La iteración solo se detiene cuando el residuo macro está dentro de `tol` **y** todas las ecuaciones del sistema completo en el estado reconstruido están dentro de `tol`; el resultado devuelto se posprocesa y audita exactamente como una solución NumPy.

### 1.6 Continuación homotópica en el esquema arancelario

Los choques arancelarios grandes y asimétricos pueden derrotar a un método tipo Newton iniciado en la base. `solve_homotopy_continuation` inserta el esquema objetivo en la trayectoria

$$\tau(\lambda) = (1-\lambda)\,\tau_0 + \lambda\,\tau_1, \qquad \tau^{fd}(\lambda) = (1-\lambda)\,\tau^{fd}_0 + \lambda\,\tau^{fd}_1, \qquad \lambda \in [0, 1],$$

y la recorre con la continuación sin predictor y con arranque en caliente de Allgower y Georg (1990, cap. 1): resuelve en $\lambda_0 = 0$ (o acepta `x0`) y luego resuelve repetidamente en $\min(\lambda + \Delta\lambda, 1)$ partiendo del último estado convergido. El control del paso es adaptativo: un paso que convergió en 4 iteraciones o menos hace crecer el siguiente $1.5\times$ (con tope `max_step`), uno que necesitó al menos `max_iter_per_step - 5` lo reduce a $0.75$ (con piso `min_step`), y un paso fallido se descarta y se reintenta con la mitad de tamaño. Si el paso cae por debajo de `min_step`, si la propia base no converge o si ningún paso tiene éxito, la función **lanza `RuntimeError`** (no devuelve un resultado parcial); el gancho `callback(lambda, result)`, llamado tras cada paso con éxito, es la forma de conservar los equilibrios intermedios. Los esquemas objetivo y base aceptan todas las formas que acepta `solve_trade_equilibrium` (multiplicadores o tasas por país importador).

---

## 2. Opciones metodológicas

| Opción | Capa de espacio de nombres (`backend=`, solucionador espacial) | `solve_trade_equilibrium_gpu` | `solve_trade_equilibrium_mlx` |
|---|---|---|---|
| **Nombres aceptados** | `"numpy"`, `"mlx"`, `"cupy"` (`"numba"` avisa y recurre a NumPy; cualquier otra cadena lanza `ValueError`) | `device` en `"auto"`, `"numpy"`, `"cpu"`, `"cuda"`, `"cuda:N"`, `"mps"`, `"mlx"`, `"gpu"`; `backend` en `"torch"`, `"mlx"`, `"numpy"` | ninguno: solo MLX, lanza `RuntimeError` sin él |
| **Precisión** | contracción float32 en MLX, terminada en el anfitrión en float64 | jacobiano float64 dondequiera que el dispositivo lo permita; un dispositivo float32 solo si se pide explícitamente, y solo para la fase gruesa | la misma política, en el flujo CPU de MLX |
| **Jacobiano** | no aplica (iteración de punto fijo) | diferencias finitas por lotes, `ad_mode="forward"` (jacfwd) o `"vjp"` (jacrev) | diferencias finitas por lotes, `ad_mode="forward"` (jvp) o `"vjp"` (mx.vjp) |
| **Regla de parada** | la `tol` pedida, terminada en el anfitrión cuando el piso del dispositivo es más grueso | `tol=2.5e-3` en todas las ecuaciones del sistema completo, absoluta en las unidades de valor de la calibración | igual |
| **Política ante fallos** | `RuntimeWarning` y luego NumPy, para un backend ausente, una excepción *y* una solución acelerada no convergida | las excepciones se propagan; los fallos de AD avisan y recurren a diferencias finitas | igual |
| **Dónde se prueba** | un modelo espacial de 5 regiones (`tests/test_trade_solver_accelerated.py`) | una calibración sintética de 8 países y 5 sectores contra el solucionador NumPy (`tests/test_gpu_acceleration.py`), omitida cuando no hay acelerador importable | igual |

La tolerancia de comercio por defecto es $2.5 \times 10^{-3}$ tanto para el solucionador NumPy como para los acelerados. Es absoluta en las unidades de valor de la calibración y holgada; ajústela para trabajos de paridad, como hacen los ejemplos de abajo.

---

## 3. Instalación de los aceleradores

Ninguno de los aceleradores forma parte de la instalación base, y la instalación base ejecuta todo lo de esta página por la ruta NumPy.

| Acelerador | Instalación | Plataformas | Aporta |
|---|---|---|---|
| Apple MLX | `pip install "puremacro[backend]"` (el extra `[accel]` tiene el mismo contenido: `numba` más `mlx` en macOS) | solo macOS en Apple Silicon (marcador `sys_platform == 'darwin'`) | `backend="mlx"` en el solucionador espacial, `solve_trade_equilibrium_mlx`, `device="mlx"` en la capa de dispositivo |
| NVIDIA CuPy | `pip install "puremacro[cuda]"` (`cupy-cuda12x`; elija la rueda acorde con su CUDA) | GPU NVIDIA con CUDA | `backend="cupy"` en el solucionador espacial |
| PyTorch | `pip install torch` (**no** hay extra de puremacro; use el selector de pytorch.org para una compilación CUDA) | CUDA, Apple MPS o CPU | `solve_trade_equilibrium_gpu`, `solve_homotopy_continuation`, `device="cuda"/"mps"/"cpu"` |

`puremacro.trade.gpu` importa `torch` y `mlx` de forma **perezosa**: `import puremacro.trade`, e incluso `from puremacro.trade.gpu import solve_trade_equilibrium_gpu`, los dejan fuera de `sys.modules` en una máquina donde están instalados y funcionan en una donde no lo están. Una llamada que de verdad necesita un acelerador ausente lanza `RuntimeError` cuando lo pide por nombre, mientras que `device=None` / `"auto"` se resuelve como `("numpy", "cpu")` y ejecuta el evaluador serial. `puremacro.runtime.capabilities().backends` enumera los backends de *espacio de nombres* instalados (`numpy`, `numba`, `mlx`, `cupy`); PyTorch no forma parte de esa tupla aunque esté instalado.

**Pyodide, iPad y el navegador.** Ni PyTorch, ni MLX, ni CuPy pueden instalarse en un kernel de Pyodide, así que allí `capabilities().backends == ("numpy",)` (véase [Ejecutar en cualquier sitio](tablet.md)), `detect_device()` devuelve `numpy:cpu` por construcción y los solucionadores de dispositivo ejecutan el evaluador NumPy. Un script escrito contra la ruta NumPy corre sin cambios en una tableta.

---

## 4. Ejemplos ejecutables

Todos los ejemplos corren sin conexión. Se ejecutaron en una estación Apple Silicon con PyTorch (MPS) y MLX instalados y sin GPU NVIDIA; las líneas impresas reproducen esa ejecución. Los ejemplos se continúan unos a otros, como una sola sesión.

### 4.1 ¿Qué hardware tengo?

```python
from puremacro._backend import available_backends
from puremacro.trade.gpu import detect_device, get_memory_usage, has_mlx, has_torch, select_compute_device

print("namespace backends:", available_backends())
print("torch installed:", has_torch(), "| mlx installed:", has_mlx())
print("auto selection   :", select_compute_device())
print(detect_device())
if has_mlx():
    print(detect_device("mlx"))
if has_torch():
    print(detect_device("cpu"))
try:
    select_compute_device("cuda")
except RuntimeError as exc:
    print("cuda:", exc)
print(sorted(get_memory_usage()))
```

```text
namespace backends: ('numpy', 'numba', 'mlx')
torch installed: True | mlx installed: True
auto selection   : ('torch', 'mps')
<DeviceInfo: torch:mps (Apple Metal (arm), 36.0 GB, f32-only, UMA)>
<DeviceInfo: mlx:gpu (Apple MLX (arm), 36.0 GB, f32-gpu/f64-cpu-stream, UMA)>
<DeviceInfo: torch:cpu (arm, f64)>
cuda: CUDA requested but torch.cuda.is_available() is False.
['allocated_mb', 'backend', 'device', 'driver_mb', 'peak_mb']
```

Con PyTorch instalado, `"auto"` prefiere MPS a MLX en un Mac, y el `DeviceInfo` de MPS indica `f32-only`. El diccionario de memoria siempre lleva `backend`, `device`, `allocated_mb` y `peak_mb`; MPS añade `driver_mb`, CUDA y MLX añaden `cache_mb`, y el respaldo NumPy informa del tamaño máximo del conjunto residente del proceso según `resource.getrusage`.

### 4.2 La referencia NumPy en una calibración de juguete

La calibración es la tabla de 2 países, 2 sectores y 3 usos finales de `tests/test_trade_solver_accelerated.py`. En su base la tabla ya es un equilibrio, así que el ejemplo impone un arancel del 25% a todo lo que el país 1 importa del país 0 para que los solucionadores trabajen. `solve_trade_equilibrium` da la referencia y rechaza un backend distinto de NumPy.

```python
import numpy as np

from puremacro.trade import calibrate_trade_model, solve_trade_equilibrium


def toy_calibration():
    """2-country, 2-sector, 3-final-use table (tests/test_trade_solver_accelerated.py)."""
    nc, ns, nfd = 2, 2, 3
    data = np.zeros((7, 10), dtype=float)
    data[:4, :4] = np.array([
        [20.0, 15.0, 5.0, 2.0],
        [10.0, 25.0, 2.0, 8.0],
        [5.0, 5.0, 12.0, 18.0],
        [10.0, 10.0, 18.0, 22.0],
    ])
    inter_col_sums = data[:4, :4].sum(axis=0)
    y = np.array([100.0, 150.0, 120.0, 180.0])
    va = y - inter_col_sums
    taxes = 0.05 * y
    va_fac = va - taxes
    data[4, :4] = taxes
    data[5, :4] = (2.0 / 3.0) * va_fac
    data[6, :4] = (1.0 / 3.0) * va_fac
    fd_row_sums = y - data[:4, :4].sum(axis=1)
    for i in range(4):
        shares = [0.50, 0.25, 0.05, 0.10, 0.08, 0.02] if i < 2 else [0.10, 0.08, 0.02, 0.50, 0.25, 0.05]
        data[i, 4:] = fd_row_sums[i] * np.array(shares)
    data[4, 4:] = 0.02 * data[:4, 4:].sum(axis=0)
    return calibrate_trade_model(data, ns=ns, nc=nc, nfd=nfd, validate=True)


calib = toy_calibration()
ns, nc, nfd = calib.n_sectors, calib.n_countries, calib.n_final_demand

# 25% tariff on everything country 1 imports from country 0 (intermediates and final demand)
tau = np.ones((ns, nc, ns, nc))
tau[:, 0, :, 1] = 1.25
tau_fd = np.ones((ns, nc, nfd, nc))
tau_fd[:, 0, :, 1] = 1.25

ref = solve_trade_equilibrium(calib, tau=tau, tau_fd=tau_fd, tol=1e-9)
print(f"numpy reference: converged={ref.converged} iterations={ref.iterations} w={np.round(ref.w_sol.ravel(), 6)}")
try:
    solve_trade_equilibrium(calib, tau=tau, tau_fd=tau_fd, backend="mlx")
except NotImplementedError as exc:
    print("NotImplementedError:", exc)
```

```text
numpy reference: converged=True iterations=6 w=[1.000255 0.977642]
NotImplementedError: solve_trade_equilibrium supports the NumPy backend only
```

### 4.3 El jacobiano por lotes en la misma calibración

El evaluador acepta directamente los tensores arancelarios 4-D y, con `batch_size` en `None`, elige el esquema reducido $[\log w; T]$ (exacto aquí: la participación del capital es uniforme) con $B = 2 \cdot 2 = 4$ entradas. El punto macro es el equilibrio de referencia, donde el residuo macro se anula. La entrada MLX se pide como `device="mlx"`, así que el evaluador usa el flujo CPU float64; `stream="gpu"` sería la forma de pedir float32.

```python
from puremacro.trade.gpu import BatchedJacobianEvaluator

x_m = np.concatenate([np.log(ref.w_sol.ravel()), ref.T_sol.ravel()])   # [log w; T] at the reference
ev_np = BatchedJacobianEvaluator(calib, tau, tau_fd, device="cpu", backend="numpy")
J_np = ev_np.evaluate_batched_jacobian(x_m)
print("layout: B =", ev_np.batch_size, "| factor_equivalence =", ev_np.factor_equivalence,
      "| max|J| =", f"{np.max(np.abs(J_np)):.2f}")
print("macro residual at the reference:", f"{np.max(np.abs(ev_np.eval_macro_single(x_m)[0])):.1e}")

for backend, device in [("torch", "cpu"), ("torch", "mps"), ("mlx", "mlx")]:
    try:
        ev = BatchedJacobianEvaluator(calib, tau, tau_fd, device=device, backend=backend)
    except RuntimeError as exc:            # accelerator absent on this machine
        print(f"{backend}:{device} unavailable ({exc})")
        continue
    J = ev.evaluate_batched_jacobian(x_m)
    print(f"{backend}:{device:4s} dtype={ev.dtype}  max|J - J_numpy| = {np.max(np.abs(J - J_np)):.1e}")

ev_t = BatchedJacobianEvaluator(calib, tau, tau_fd, device="cpu", backend="torch")
J_ad = ev_t.evaluate_batched_jacobian(x_m, ad_mode="forward")   # torch.func.jacfwd, no step size
print("forward-AD vs finite differences:", f"{np.max(np.abs(J_ad - J_np)):.1e}")
```

```text
layout: B = 4 | factor_equivalence = True | max|J| = 21.95
macro residual at the reference: 5.7e-14
torch:cpu  dtype=torch.float64  max|J - J_numpy| = 0.0e+00
torch:mps  dtype=torch.float32  max|J - J_numpy| = 3.5e+01
mlx:mlx  dtype=None  max|J - J_numpy| = 2.8e-08
forward-AD vs finite differences: 9.3e-06
```

El jacobiano float64 de la CPU de PyTorch reproduce exactamente el de NumPy serial, y MLX en su flujo CPU float64 a $3 \times 10^{-8}$ (los núcleos float64 propios de MLX). El único dispositivo float32 de la tabla, MPS, se desvía más que la mayor entrada: el redondeo float32 dividido por el paso relativo $10^{-6}$ anula la derivada, que es justamente por lo que los solucionadores nunca pulen con él (sección 1.3). La diferencia de $9 \times 10^{-6}$ entre la AD en modo directo y las diferencias finitas es el error de truncamiento de las diferencias; `ad_mode="vjp"` (modo inverso) reproduce el jacobiano AD. (`ev.dtype` solo se fija en la ruta PyTorch; MLX elige float32 o float64 según el flujo.)

### 4.4 Equilibrio espacial en NumPy y MLX

```python
import numpy as np

from puremacro.spatial import AllenArkolakisModel

coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])
model = AllenArkolakisModel.from_coordinates(coords)

res_np = model.solve_equilibrium(backend="numpy")
res_mlx = model.solve_equilibrium(backend="mlx")
print("iterations: numpy", res_np.iterations, "| mlx", res_mlx.iterations)
print("population (numpy):", np.round(res_np.population, 6))
print("max |dw| =", f"{np.max(np.abs(res_np.wages - res_mlx.wages)):.1e}",
      "| max |dL| =", f"{np.max(np.abs(res_np.population - res_mlx.population)):.1e}",
      "| max |dP| =", f"{np.max(np.abs(res_np.price_index - res_mlx.price_index)):.1e}")
print("labor conservation:", res_np.labor_conservation_residual, res_mlx.labor_conservation_residual)

cf_np = model.solve_counterfactual(productivity_new=np.array([1.2, 1.0, 1.0, 1.0, 1.0]), backend="numpy")
cf_mlx = model.solve_counterfactual(productivity_new=np.array([1.2, 1.0, 1.0, 1.0, 1.0]), backend="mlx")
print(f"welfare change: numpy {cf_np.welfare_pct:+.5f}% | mlx {cf_mlx.welfare_pct:+.5f}%")
```

```text
iterations: numpy 27 | mlx 27
population (numpy): [0.199763 0.199763 0.199763 0.199763 0.200947]
max |dw| = 5.4e-10 | max |dL| = 2.7e-10 | max |dP| = 2.7e-12
labor conservation: 0.0 0.0
welfare change: numpy +4.39312% | mlx +4.39312%
```

Ambos backends informan de las mismas 27 iteraciones y coinciden a $10^{-9}$, porque el bucle float32 de MLX cede el paso al bucle float64 de NumPy en cuanto alcanza su propio piso (sección 1.1): la ejecución se certifica contra la `tol=1e-8` pedida, no contra el $10^{-6}$ del dispositivo. El efecto sobre el bienestar de una ganancia de productividad del 20% en la primera región coincide a cinco decimales. La población suma exactamente $\bar L$ en ambos backends porque la renormalización ocurre en NumPy.

### 4.5 El solucionador de dispositivo y la homotopía en la calibración de juguete

Continuando desde 4.2:

```python
from puremacro.trade.gpu import solve_homotopy_continuation, solve_trade_equilibrium_gpu

res_gpu = solve_trade_equilibrium_gpu(calib, tau=tau, tau_fd=tau_fd, tol=1e-9)
print(f"gpu       converged={res_gpu.converged} iterations={res_gpu.iterations} "
      f"max_residual={res_gpu.max_residual:.1e} w={np.round(res_gpu.w_sol.ravel(), 6)}")
print("max |w_gpu - w_numpy| =", f"{np.max(np.abs(res_gpu.w_sol - ref.w_sol)):.1e}")

meta = res_gpu.metadata
print("device used   :", f"{meta['backend']}:{meta['device']}",
      "| requested:", f"{meta['requested_backend']}:{meta['requested_device']}")
print("layout        :", "B =", meta["batch_size"], "factor_equivalence =", meta["factor_equivalence"])
print("jacobians     :", dict(meta["jacobian_evaluations"]))

steps = []
hom = solve_homotopy_continuation(calib, target_tau=tau, target_tau_fd=tau_fd, tol=1e-9,
                                  initial_step=0.25, max_step=0.5,
                                  callback=lambda lam, r: steps.append((round(lam, 3), r.iterations)))
print("homotopy steps:", steps)
print(f"homotopy  converged={hom.converged} max_residual={hom.max_residual:.1e} "
      f"w={np.round(hom.w_sol.ravel(), 6)}")
```

```text
gpu       converged=True iterations=5 max_residual=8.5e-14 w=[1.000255 0.977642]
max |w_gpu - w_numpy| = 1.2e-14
device used   : torch:cpu | requested: torch:mps
layout        : B = 4 factor_equivalence = True
jacobians     : {'torch:cpu:float64': 4}
homotopy steps: [(0.25, 4), (0.625, 4), (1.0, 4)]
homotopy  converged=True max_residual=1.8e-10 w=[1.000255 0.977642]
```

El solucionador de dispositivo alcanza los salarios de NumPy a $10^{-14}$. Los metadatos dicen qué ocurrió: `"auto"` eligió MPS, la política float64 de la sección 1.3 movió en silencio el jacobiano a `torch:cpu`, se eligió el esquema reducido porque aquí es exacto ($B = 4$) y todos los jacobianos fueron float64. La homotopía dio tres pasos: $0.25$, luego $0.625$ tras el crecimiento $1.5\times$ de un paso que convergió en 4 iteraciones o menos, y luego $1.0$.

### 4.6 Una tabla sintética mayor

El GEMM por lotes compensa a medida que crece $M = SN$. Las mismas llamadas corren sobre cualquier calibración; aquí, una tabla sintética de 8 países y 5 sectores (`synthetic_icio_table`, flujos aleatorios genéricos) con un arancel del 20% de su primer país sobre todas sus importaciones:

```python
from puremacro.trade.data import synthetic_icio_table

nc, ns = 8, 5
big = calibrate_trade_model(synthetic_icio_table(nc, ns, seed=3), ns=ns, nc=nc, nfd=3)
rates = np.zeros(nc)
rates[0] = 0.20                                   # country C01 taxes all its imports at 20 percent
ref_big = solve_trade_equilibrium(big, tau=rates, tau_fd=rates, tol=1e-6)
gpu_big = solve_trade_equilibrium_gpu(big, tau=rates, tau_fd=rates, tol=1e-6)
print("B =", gpu_big.metadata["batch_size"], "| converged:", ref_big.converged, gpu_big.converged,
      "| iterations:", ref_big.iterations, gpu_big.iterations)
print("max |w_gpu - w_numpy| =", f"{np.max(np.abs(gpu_big.w_sol - ref_big.w_sol)):.1e}")
```

```text
B = 16 | converged: True True | iterations: 5 4
max |w_gpu - w_numpy| = 1.0e-10
```

Las tasas arancelarias por país importador se aceptan exactamente como en `solve_trade_equilibrium`; `solve_trade_equilibrium_mlx(big, tau=rates, tau_fd=rates)` y `solve_homotopy_continuation(big, target_tau=rates, target_tau_fd=rates)` toman los mismos argumentos.

---

## 5. Especificación completa de la API

```text
# puremacro._backend
SUPPORTED = ("numpy", "numba", "mlx", "cupy")
backend_available(name: str) -> bool                     # ValueError for unknown names
available_backends() -> tuple[str, ...]                  # installed, numpy first
get_array_namespace(name: str)                           # numpy | mlx.core | cupy; ImportError with the pip hint
to_numpy(x) -> np.ndarray

# backend= on the model solvers
solve_trade_equilibrium(calib, ..., *, backend="numpy", ...)   # NumPy only; other values raise NotImplementedError
AllenArkolakisModel.solve_equilibrium(tol=1e-8, max_iter=2500, damping=0.35,
                                      backend="numpy") -> AllenArkolakisResult
AllenArkolakisModel.solve_counterfactual(trade_costs_new=None, productivity_new=None, amenity_new=None,
                                         tol=1e-8, max_iter=2500, damping=0.35,
                                         backend="numpy") -> AllenArkolakisResult

# puremacro.trade.gpu.backend
DeviceInfo(backend: Literal["torch", "mlx", "numpy"], device: str, device_name: str,
           total_memory_gb: float | None = None, supports_float64: bool = True, is_uma: bool = False)
has_torch() -> bool
has_mlx() -> bool
select_compute_device(preferred: str | None = None) -> tuple[str, str]
detect_device(preferred: str | None = None) -> DeviceInfo
device_context(device: str | None = None, backend: str | None = None)   # context manager -> (backend, device)
get_memory_usage(device: str | None = None, backend: str | None = None) -> dict[str, float | str]
reset_peak_memory(device: str | None = None, backend: str | None = None) -> None
to_tensor(array, device: str | None = None, dtype=None, backend: str = "torch")
to_numpy(tensor) -> np.ndarray

# puremacro.trade.gpu.batched_jacobian
BatchedJacobianEvaluator(calib: TradeCalibrationResult, tau_a, taufd_a,
                         device: str | None = None, backend: str | None = None,
                         batch_size: int | None = None, factor_equivalence: bool | None = None,
                         *, foreign_saving_units: str = "numeraire")
    .update_tariffs(tau_a, taufd_a) -> None
    .eval_macro_single(xm, compute_full=False)
        -> (f_macro, f_full | None, p_vec, p, y_vec)
    .full_residuals(x_full) -> np.ndarray
    .evaluate_batched_jacobian(xm, f_base=None, eps_fd=1e-6,
                               ad_mode="finite_diff" | "forward" | "vjp", stream=None) -> np.ndarray (B, B)
    .jvp(xm, v) -> np.ndarray
    .eval_macro_batched_torch(X_batch) / .eval_macro_batched_mlx(X_batch, stream=None)

# puremacro.trade.gpu.solver_gpu
solve_trade_equilibrium_gpu(calib, tau=None, tau_fd=None, tauf=None, tauf_fd=None, x0=None,
                            device=None, backend=None, batch_size=None, max_iter=50, tol=2.5e-3,
                            damping=1e-4, ad_mode="finite_diff", base_result=None, verbose=False, *,
                            factor_equivalence=None, foreign_saving_units="numeraire") -> TradeEquilibriumResult

# puremacro.trade.gpu.mlx_solver
solve_trade_equilibrium_mlx(calib, tau=None, tau_fd=None, tauf=None, tauf_fd=None, x0=None,
                            batch_size=None, max_iter=50, tol=2.5e-3, damping=1e-4,
                            ad_mode="finite_diff", base_result=None, verbose=False, *,
                            factor_equivalence=None, foreign_saving_units="numeraire") -> TradeEquilibriumResult

# puremacro.trade.gpu.homotopy
solve_homotopy_continuation(calib, target_tau, target_tau_fd, base_tau=None, base_tau_fd=None, x0=None,
                            device=None, backend=None, batch_size=None,
                            initial_step=0.1, min_step=1e-4, max_step=0.5, tol=2.5e-3, damping=1e-4,
                            max_iter_per_step=30, base_result=None, verbose=False, callback=None, *,
                            factor_equivalence=None, foreign_saving_units="numeraire") -> TradeEquilibriumResult
```

Todos los nombres de arriba se importan desde `puremacro.trade.gpu`.

### Parámetros comunes de los solucionadores acelerados

- `calib`: un `TradeCalibrationResult` de `calibrate_trade_model`.
- `tau`, `tau_fd`: aranceles en cualquier forma que acepte `solve_trade_equilibrium`: multiplicadores ($1 + $ tasa) de forma `(ns, nc, ns, nc)` / `(ns, nc, nfd, nc)` o en el esquema interno `(M, ns, nc)` / `(M, nfd, nc)`, o tasas por país importador (un escalar o un vector de longitud `nc`); `None` significa sin aranceles.
- `tauf`, `tauf_fd` (solo solucionadores directos): tasas arancelarias por país importador, usadas cuando `tau` / `tau_fd` es `None`, exactamente como en `solve_trade_equilibrium`.
- `x0`: `None` (rendimientos factoriales unitarios y las transferencias calibradas), el vector de estado completo de longitud $2M + 4N - 1$, o un vector macro de cualquiera de los dos esquemas, $2N$ o $3N$; un vector reducido que inicia el esquema completo fija $r = w$. Cualquier otra longitud lanza `ValueError` nombrando las tres esperadas.
- `device` / `backend`: los valores explícitos se mantienen; `select_compute_device` completa el que falte. El par que acaba evaluando el jacobiano puede diferir del pedido (la política float64 de la sección 1.3); `metadata["device"]` / `["backend"]` registran el par efectivo y `metadata["requested_device"]` / `["requested_backend"]` el pedido. `solve_trade_equilibrium_mlx` no tiene ninguno de los dos.
- `batch_size` / `factor_equivalence`: el selector de esquema de la sección 1.4. Prefiera `factor_equivalence` (`None` = elegir por exactitud, `True` = reducido $[\log w; T]$, `False` = completo $[\log r; \log w; T]$); `batch_size` debe valer $2N$ o $3N$ para la calibración.
- `foreign_saving_units`: `"numeraire"` (por defecto) o `"world_income"`, el cierre de ahorro externo de `solve_trade_equilibrium`.
- `tol`: tolerancia absoluta en norma infinito, en las unidades de valor de la calibración, sobre todas las ecuaciones del sistema completo para `converged`; `max_iter`: iteraciones de Levenberg-Marquardt (`max_iter_per_step` por paso de continuación).
- `damping`: $\mu$ inicial.
- `ad_mode`: `"finite_diff"` (diferencias por lotes, por defecto), `"forward"` (AD en modo directo: `torch.func.jacfwd` / `mx.jvp`) o `"vjp"` (modo inverso: `torch.func.jacrev` / `mx.vjp`). Un modo AD que falla en el dispositivo avisa y recurre a diferencias finitas; un valor desconocido lanza `ValueError`.
- `base_result`: equilibrio base que usa el posprocesamiento para el IPC y los índices de términos de intercambio.
- `verbose`: imprime por iteración `max_res`, el paso y $\mu$ (y, en la homotopía, el registro de pasos).
- `callback` (solo homotopía): `callback(lambda_value, result)` tras cada paso con éxito.

---

## 6. Interfaz de resultados

### `TradeEquilibriumResult` (solucionadores acelerados)

Los solucionadores acelerados devuelven la misma dataclass congelada que `solve_trade_equilibrium`, posprocesada y auditada por el mismo código, así que `x_sol`, `p_sol`, `y_sol`, `r_sol`, `w_sol`, `T_sol`, `XN_sol`, los tensores de flujos bilaterales, `gdp`, `imports`, `exports`, `tariffs`, `cpi` y `terms_of_trade` se documentan una sola vez en la [página de contabilidad del comercio](trade_accounting.md). Los campos que fijan los solucionadores de esta página son:

- `converged`, `iterations`, `max_residual`, `diff` / `residual_norm` (norma L1) y `residuals`: evaluados sobre el residuo del sistema completo en float64.
- `metadata`: `method` (`"gpu_accelerated"` o `"mlx_accelerated"`), el `device` y el `backend` efectivos junto con `requested_device` y `requested_backend`, `factor_equivalence`, `batch_size`, `jacobian_evaluations` (un recuento de jacobianos por `backend:device:dtype`), `ad_mode`, `tol`, `max_iter`, `macro_residual_max`, `solve_duration_seconds` y `memory_usage` (el diccionario de `get_memory_usage`).
- `summary(detailed=False) -> pd.DataFrame`: la tabla de diagnóstico del solucionador; `detailed=True` da la tabla por país (PIB, comercio, aranceles, IPC, términos de intercambio).
- `to_frame(detailed=False)`, `to_markdown(detailed=False)`, `to_latex(detailed=False)`, `to_typst(detailed=False)`: la misma tabla en cada formato.
- `TradeEquilibriumResult` no tiene método `plot`; las figuras salen de `puremacro.trade.plot`.

### `AllenArkolakisResult` (solucionador espacial)

No cambia con el backend: `wages`, `population`, `price_index`, `real_wages`, `welfare`, `consumer_market_access`, `firm_market_access`, `trade_shares` y, en contrafactuales, `w_hat`, `L_hat`, `welfare_hat`, `welfare_pct`; diagnósticos `converged`, `iterations`, `max_residual`, `labor_conservation_residual`, `spatial_utility_variance`; presentación `summary()`, `to_frame()`, `to_markdown()`, `to_latex()`, `to_typst()` y `plot(kind="spatial")`. Nada en el resultado registra qué backend lo produjo.

### `DeviceInfo` y el diccionario de memoria

`DeviceInfo` es la dataclass congelada de la sección 1.2. `get_memory_usage()` devuelve `{"backend", "device", "allocated_mb", "peak_mb"}` más `cache_mb` (CUDA, MLX) o `driver_mb` (MPS); `reset_peak_memory()` reinicia los contadores de pico (CUDA, MLX) y vacía las cachés del dispositivo (CUDA, MPS, MLX), y `device_context(...)` lo llama al salir y entrega el par `(backend, device)` sin contrastar ambos, repitiendo las cadenas pasadas.

---

## 7. Advertencias y limitaciones

- **Alcance del modelo.** La capa de dispositivo resuelve el modelo de contabilidad consistente con insumos intermedios Cobb-Douglas (`sigma = 0`) y devolución de suma fija de la recaudación arancelaria. Para insumos CES, otros cierres fiscales o la extensión de capacidad, use `solve_trade_equilibrium`.
- **Dos capas, dos vocabularios.** La capa de espacio de nombres acepta `"numpy"`, `"mlx"`, `"cupy"` (`"numba"` avisa y recurre a NumPy, y cualquier otra cosa, `"torch"` incluido, lanza `ValueError: Unknown backend`). La capa de dispositivo acepta `"auto"`, `"numpy"`, `"cpu"`, `"cuda"`, `"cuda:N"`, `"mps"`, `"mlx"`, `"gpu"` y lanza `ValueError` con cualquier otra cosa.
- **Precisión.** MPS no tiene float64 y el flujo Metal de MLX es float32, así que un dispositivo float32 pedido explícitamente solo se usa en la fase gruesa y avisa cuando se usa. Espere coincidencia al redondeo en las rutas float64 y un jacobiano inservible en una float32 (ejemplo 4.3).
- **Tolerancias.** `tol=2.5e-3` en los solucionadores de comercio NumPy y acelerados es absoluta y holgada; ajústela explícitamente para trabajos de paridad.
- **Costo de importación.** `import puremacro.trade` no carga PyTorch ni MLX; ambos se importan de forma perezosa en la primera llamada que los necesita, que paga la importación del marco.
- **Rutas no verificadas.** La rama CuPy del solucionador espacial y la rama CUDA de la capa de dispositivo no se ejecutaron al escribir esta página (no hay hardware NVIDIA); las cubren la lógica de respaldo y pruebas que se omiten cuando falta el acelerador.
- **Sin aceleración en problemas pequeños.** Todas las rutas de dispositivo añaden latencia de transferencia y lanzamiento; la calibración de juguete y el modelo de 5 regiones corren más lento en un acelerador que en NumPy. [Rendimiento y benchmarks](benchmarks.md) publica por ahora solo tiempos en CPU.
- **Sin conexión y determinista.** Nada aquí toca la red; las rutas aceleradas son tan deterministas como las reducciones del dispositivo, y la ruta NumPy es la referencia para las afirmaciones de reproducibilidad.

---

## Referencias

- Allen, T., & Arkolakis, C. (2014). "Trade and the Topography of the Spatial Economy." *The Quarterly Journal of Economics*, 129(3), 1085–1140.
- Allgower, E. L., & Georg, K. (1990). *Numerical Continuation Methods: An Introduction*. Springer Series in Computational Mathematics 13. Berlin: Springer.
- Armijo, L. (1966). "Minimization of functions having Lipschitz continuous first partial derivatives." *Pacific Journal of Mathematics*, 16(1), 1–3.
- Caliendo, L., & Parro, F. (2015). "Estimates of the Trade and Welfare Effects of NAFTA." *The Review of Economic Studies*, 82(1), 1–44.
- Levenberg, K. (1944). "A method for the solution of certain non-linear problems in least squares." *Quarterly of Applied Mathematics*, 2(2), 164–168.
- Marquardt, D. W. (1963). "An algorithm for least-squares estimation of nonlinear parameters." *Journal of the Society for Industrial and Applied Mathematics*, 11(2), 431–441.
