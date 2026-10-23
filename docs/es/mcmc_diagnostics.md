> 🇬🇧 [English](../mcmc_diagnostics.md) · 🇪🇸 Español

# Diagnósticos MCMC entre cadenas

Puremacro ofrece R-hat dividido con normalización por rangos y plegado, tamaño
efectivo de muestra (ESS) entre cadenas para el centro y las colas, y error
estándar Monte Carlo de la media posterior. Solo requieren NumPy y SciPy.
`NUTSResult.summary()` los utiliza automáticamente; no cambian las muestras
guardadas ni el algoritmo de muestreo.

```python
import numpy as np
from puremacro.mcmc import rank_normalized_rhat, multichain_ess, mcse_mean

cadenas = np.random.default_rng(12).normal(size=(4, 1000))
print(rank_normalized_rhat(cadenas))
print(multichain_ess(cadenas, method="bulk"))
print(multichain_ess(cadenas, method="tail"))
print(mcse_mean(cadenas))
```

La entrada tiene dimensiones `(cadenas, extracciones)` para un parámetro y
contiene cadenas inicializadas independientemente, después del calentamiento.
No se deben tratar distintos parámetros como cadenas ni concatenarlas antes de
calcular ESS. Las funciones no modifican la entrada.

| Estadístico | Cálculo e interpretación |
|---|---|
| `rank_normalized_rhat` | Divide las cadenas, normaliza los rangos conjuntos y toma el máximo R-hat entre esos valores y los rangos de las desviaciones absolutas respecto a la mediana. Detecta diferencias de localización y escala. |
| `multichain_ess(..., method="bulk")` | ESS de los rangos normalizados; combina autocovarianzas internas y variación entre cadenas antes del truncamiento de Geyer. |
| `multichain_ess(..., method="tail")` | Menor ESS de los indicadores bajo los percentiles conjuntos 5 y 95. `tail_probability` permite cambiar estas probabilidades simétricas. |
| `multichain_ess(..., method="mean")` | ESS de las extracciones sin transformar, para una media posterior de varianza finita. |
| `mcse_mean` | Desviación estándar conjunta dividida por la raíz del ESS de la media. Mide error de simulación, no incertidumbre del parámetro. |

Los rangos usan `(rango - 3/8)/(S + 1/4)`. Al dividir una cadena impar se omite
la extracción central. Los umbrales de cola usan todas las extracciones antes
de dividir los indicadores. La autocorrelación negativa puede producir ESS
mayor que el número de extracciones; el límite es `S*log10(S)`.

R-hat necesita dos cadenas originales y cuatro extracciones por cadena. ESS
admite una cadena, pero entonces no puede evaluar concordancia entre cadenas
independientes. Entradas cortas, no finitas o completamente constantes devuelven
NaN. Cadenas constantes en valores diferentes producen R-hat infinito. Un
indicador de cola constante produce ESS de cola NaN, en lugar de un conteo
nominal que podría parecer evidencia informativa.

R-hat próximo a uno y ESS grande son diagnósticos, no una demostración de
exploración global. Revise también divergencias, energía, trazas y log posterior.
El error Monte Carlo de la media requiere varianza posterior finita y condiciones
de límite apropiadas; normalizar rangos no hace existir la media de una Cauchy.

## Migración desde 5.5.0 y verificación

Los antiguos ESS “bulk” y “tail” de NUTS sumaban ESS individuales sin penalizar
cadenas atrapadas en regiones diferentes. Ahora usan las definiciones entre
cadenas. `r_hat` en el resumen usa normalización por rangos y plegado;
`mcse_mean` es una columna adicional y los atributos identifican los métodos.
R-hat con una sola cadena es NaN. Las funciones clásicas `gelman_rubin` y
`effective_sample_size` permanecen disponibles. `compute_split_rhat` conserva
la definición clásica, pero los casos degenerados ya no devuelven artificialmente
uno. La energía constante también produce un diagnóstico no disponible.

Las pruebas independientes congelan una ejecución real de **ArviZ 0.22.0**
con ocho conjuntos de cadenas, hashes de entradas y versiones registradas.
Las comparaciones informativas usan tolerancia relativa `2e-11` para ESS/MCSE
y `2e-12` para R-hat. El indicador constante de la cola superior en el caso
discreto es una diferencia explícita de política; se conserva el valor original
de ArviZ sin modificarlo. ArviZ no es una dependencia de ejecución.

Métodos: [Vehtari et al. (2021)](https://doi.org/10.1214/20-BA1221).
Referencia independiente:
[ArviZ 0.22.0](https://github.com/arviz-devs/arviz/blob/v0.22.0/arviz/stats/diagnostics.py).
