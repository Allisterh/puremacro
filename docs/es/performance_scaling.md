> 🇬🇧 [English](../performance_scaling.md) · 🇪🇸 Español

# Elección del solucionador y mediciones reproducibles de escala

`solve_perfect_foresight(..., linear_solver="krylov")` activa el motor existente
de Newton-Krylov apilado sin matriz explícita. El valor predeterminado `"direct"`
conserva el solucionador disperso directo. Ambos comparten la preparación del
modelo, los choques, la historia inicial, las condiciones terminales literales
o resueltas y la referencia fija de `STEADY_STATE()`. `extended_path` transmite
la selección en cada fecha; la simulación de choques sorpresa también. MCP con
Krylov genera un error explícito antes de evaluar el modelo numéricamente.

```python
import numpy as np
from puremacro.dsge import solve_perfect_foresight

def equations(lead, current, lag, shock):
    return current - 0.2 * lead - 0.5 * lag - shock

result = solve_perfect_foresight(
    equations, y_init=[0.0], y_ss=[0.0], periods=40,
    shocks=0.01 * 0.7 ** np.arange(40), linear_solver="krylov",
    krylov_options={"preconditioner": "steady_block_tridiagonal"},
)
assert result.converged
print(result.residual_norm)
print(result.solver_diagnostics["converged_by"])
```

Krylov exige un residuo pequeño respecto a la escala del Jacobiano terminal
**y** una dirección completa de Newton pequeña. La tolerancia es `tol`; la del
paso es `1e-10 * max(1, |path|_inf)`. `dampening` limita el paso inicial de la
búsqueda. Un fallo numérico emite una advertencia y devuelve `converged=False`
con la última trayectoria y los diagnósticos. Nunca cambia silenciosamente al
motor directo. Las entradas inválidas y los errores de las funciones de residuos
siguen generando excepciones. `solver_diagnostics["linear_history"]` muestra el
estado de Krylov, los residuos lineales recalculados, las tolerancias y los pasos
aceptados. El motor directo conserva sus criterios de convergencia; comparar
solo las banderas no demuestra paridad numérica.

`krylov_options` admite `preconditioner`, `forcing`, `krylov_method`, `inner_m`,
`outer_k`, `krylov_maxiter`, `side`, `max_log_move`, `time_block`, `progress` y
`checkpoint`. El precondicionador terminal se factoriza una vez por resolución.
La opción de bloques corrientes reutiliza su última factorización únicamente
cuando las formas, los tipos y todas las entradas de los bloques preparados son
exactamente iguales. Cualquier cambio exige reconstruirla; no hay caché global
ni basada en la identidad del modelo. Los problemas sucesivos de extended path
usan instancias independientes. Los bloques densos aún requieren `O(T n²)`;
esto no demuestra capacidad para miles de variables por fecha. Véase el
[contrato del solucionador apilado](dsge_stacked_newton.md).

## Ejecutar la batería de escala

Desde una copia del código con las dependencias de desarrollo:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python benchmarks/benchmark_scaling.py --profile smoke --repetitions 3 --output scaling-smoke.json
python benchmarks/benchmark_scaling.py --profile standard --repetitions 5 --output scaling-standard.json
```

Configure las variables de hilos para ambos comandos al comparar ejecuciones.
En Windows, use `$env:` de PowerShell antes de ejecutar Python.
`--families pf hank var mrio` selecciona los ejercicios; `--seed` fija los datos
sintéticos y las extracciones bootstrap. El perfil smoke comprueba el mecanismo,
sin establecer rendimiento. El perfil standard varía horizontes PF/HANK,
extracciones bootstrap y sectores MRIO. La preparación y las referencias no
forman parte del tiempo medido.

El JSON registra las versiones de Python, paquete, NumPy y SciPy, plataforma,
CPU, configuración de NumPy, entorno de hilos, revisión y cambios sin guardar
del repositorio. El tiempo usa `perf_counter` del benchmark de VFI continua:
un calentamiento sin medir y las repeticiones solicitadas, con mínimo, media y
desviación estándar. La memoria usa otra ejecución y otro calentamiento con
`tracemalloc`: es el máximo de **asignaciones rastreadas**, no RSS ni una cota
de toda la memoria nativa. La exactitud se evalúa por separado. Compare el mismo
entorno y conserve el JSON completo; no hay umbral universal de aceleración ni
selección automática del motor.

| Ejercicio | Evidencia comprobada | Límite |
|---|---|---|
| PF directo y Krylov | Trayectoria exacta construida y ecuaciones no lineales recalculadas | Tres variables; ecuaciones sintéticas |
| Horizontes HANK | Respuesta del hogar, NKPC, Taylor/Fisher, mercado de bienes y masa de distribución | GE lineal de un activo; no certifica refinamiento de la malla |
| Bootstrap VAR | Primera extracción con simulación recursiva escalar y reconstrucción de IRF; bandas finitas y ordenadas | Reproducibilidad e implementación, no cobertura estadística |
| Sectores MRIO | Agregación dispersa frente a sumas densas y conservación contable | Tablas sintéticas balanceadas; sin archivos nativos ni equilibrio comercial |

Una comprobación numérica fallida devuelve código 1 y permanece en el JSON.
Las excepciones se registran como errores; opciones inválidas devuelven 2.
Los perfiles mayores deben ejecutarse deliberadamente, sin convertir sus
tiempos en condiciones frágiles de integración continua.
