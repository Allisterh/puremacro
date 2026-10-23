> 🇬🇧 [English](../dependence_bootstrap.md) · 🇪🇸 Español

# Bootstrap con dependencia explícita

Las nuevas opciones complementan los diagnósticos espaciales y GVAR existentes.
No sustituyen la inferencia asintótica por defecto ni garantizan tamaño nominal
exacto en muestras finitas.

## Proyecciones locales espaciales

```python
from puremacro.spatial import SpatialLPBootstrap, spatial_lp

result = spatial_lp(
    panel, "output", "shock", weights, horizons=range(6), n_lags=2,
    lagged_outcome="difference", alpha=0.05,
    bootstrap=SpatialLPBootstrap(
        n_boot=399, block_length=6, assume_stationary=True, seed=814,
    ),
)
print(result.bootstrap_result.summary())
print(result.bootstrap_result.table)
```

**La tabla bootstrap estima sobre la intersección de las muestras de todos los
horizontes.** Sus estimaciones y errores estándar pueden diferir de los del
resultado original, que conserva las muestras específicas de cada horizonte.
`metadata['common_periods']` identifica la muestra común. Cada horizonte exige un
panel completo y balanceado con las mismas unidades; la muestra común debe contener
al menos dos longitudes de bloque.

Los resultados y controles de cada horizonte se construyen antes del remuestreo.
Se toman bloques temporales circulares que incluyen todas las regiones, con los
mismos índices para todos los horizontes. Se preserva así la dependencia espacial
contemporánea y temporal dentro de cada bloque, sin inventar adelantos o rezagos
en las uniones. Cada réplica vuelve a estimar los efectos fijos, los coeficientes y
la covarianza elegida para la studentización: Driscoll–Kraay o Conley. La opción
cluster por entidad se rechaza únicamente en este bootstrap: con N fijo su escala
no tiene aquí una justificación asintótica admitida. Las estimaciones cluster
anteriores siguen disponibles sin el bootstrap.

La tabla contiene intervalos bootstrap-t simétricos puntuales y simultáneos,
p-valores centrados y errores Monte Carlo condicionales. La familia simultánea
incluye **todos los contrastes mostrados no nulos y todos los horizontes**.
`joint_p_value` usa el máximo |t| de los contrastes indirectos por orden y horizonte;
`metadata['joint_labels']` precisa la familia. Los contrastes escalados deliberadamente
a cero se excluyen. Sin contrastes indirectos no nulos, el estadístico conjunto y
su p-valor son `None`. No se usa una referencia Wald chi-cuadrado.

`assume_stationary=True` afirma estacionariedad de **todas las filas de regresión**,
incluidos los controles; no es una prueba de ese supuesto. El resultado dependiente
sigue siendo `y[t+h]-y[t-1]`. La opción `lagged_outcome='difference'` incorpora rezagos
de cambios de `y` y, cuando corresponda, de `Wy`; el valor por defecto sigue siendo
`'levels'`. Una variable integrada puede generar cambios y controles estacionarios,
pero los controles integrados arbitrarios, quiebres y shocks endógenos quedan fuera
de esta justificación. La aproximación considera N fijo y T creciente, dependencia
débil y regresores identificados. `total` sigue siendo un contraste respecto a la
media temporal, no un multiplicador agregado.

Elija la longitud del bloque antes de mirar significancia y reporte sensibilidad.
Un bloque largo reduce el número efectivo de bloques. Una regresión singular o
varianza inválida en cualquier réplica produce un error identificado; ninguna
réplica se elimina o repite selectivamente.
Los intervalos por cuantiles empíricos y los p-valores con corrección más uno
pueden diferir en el límite discreto Monte Carlo en aproximadamente una réplica;
no se presentan como inversiones exactas entre sí en muestras finitas.
La justificación asintótica exige que la longitud del bloque crezca con T mientras
su cociente con T tienda a cero; un bloque corto fijo no corrige universalmente
una dependencia serial arbitraria. Esta condición también se aplica a los bloques
de innovaciones dependientes de la referencia GVAR; longitud uno es su caso iid.

El método aplica el marco de remuestreo estacionario de
[Künsch (1989)](https://doi.org/10.1214/aos/1176347265) y
[Politis y Romano (1992)](https://profiles.stanford.edu/joseph-romano?tab=publications).
No implementa ni demuestra la teoría uniforme con raíces unitarias de
[Montiel Olea y Plagborg-Møller (2021)](https://joseluismontielolea.com/lp_inference_ecta.pdf).

## GVAR: nula restringida de países independientes

```python
reference = fitted_gvar.independent_null_bootstrap(
    null="independent_countries", integration="i1",
    n_boot=399, block_length=1, null_lags=1, seed=815,
)
print(reference.summary())
print(reference.table)
```

La referencia supone innovaciones independientes entre países y ausencia de
dinámica rezagada entre países. **Es una nula más fuerte que la exogeneidad débil
general**, y no es el contraste de corrección de error de Dees et al. No rechazar
no demuestra independencia: el estadístico auxiliar puede tener poca potencia
contra algunas alternativas.

Se estima un VAR propio por país en niveles (`integration='stationary'`) o en
primeras diferencias (`'i1'`), con `null_lags` rezagos. En el segundo caso se simulan
incrementos estacionarios y se integran desde el nivel inicial observado. Se
condiciona en los rezagos iniciales. Ningún precontraste elige automáticamente
el orden de integración. La referencia I(1) no cubre bloques con órdenes mixtos
o cointegración interna. Un VAR nulo estimado no estacionario se rechaza sin
recortar raíces. No se admiten exógenas ni tendencias distintas de `'n'` o `'c'`.

Se centran las innovaciones y se remuestrean vectores en bloques circulares de
forma independiente entre países, preservando la dependencia contemporánea dentro
de cada país. Longitud uno corresponde a innovaciones iid. Cada panel simulado
vuelve a estimar todos los VARX* originales y sus regresiones auxiliares; nunca
congela los residuos generados en la primera etapa. Se mantienen los órdenes de
rezago originales y auxiliares; no se repite una selección previa por criterios
de información.

El diagnóstico original `weak_exogeneity.table` no cambia. La nueva tabla distingue
`p_independent_null`, `p_holm_independent_null` y `reject_independent_null`.
Holm abarca todas las filas. El p-valor es `(1 + réplicas con F >= F observado)/(B+1)`;
`mc_se` mide ruido de simulación condicional, no incertidumbre sobre el modelo nulo.
La resolución es `1/(B+1)`: con 99 réplicas no se puede obtener menos de 0.01 ni
superar el primer umbral Holm al 5% para seis pruebas. Se conservan todas las
réplicas y configuraciones; cualquier fallo invalida explícitamente la inferencia.
Los diseños auxiliares restringido y no restringido deben tener rango completo,
tanto en la muestra observada como en cada réplica. Los rezagos adicionales en
niveles pueden generar exactamente el residuo estimado y hacer que su restricción
no sea identificable. El bootstrap rechaza esa especificación; el diagnóstico
anterior continúa disponible con su interpretación limitada.

La distinción entre VARX* y los contrastes de exogeneidad débil con cointegración
se explica en [Chudik y Pesaran](https://doi.org/10.1111/joes.12095) y en la
[guía GVAR](gvar.md). Esta opción no añade una capa VECX*.

## Evidencia reproducible

`tools/reference_validation/validate_dependence_bootstrap.py` produce JSON con
simulaciones acotadas y semillas explícitas. Los diseños espaciales incluyen
N=20/40, T=50/100, seis horizontes, errores AR(0.4) correlacionados espacialmente,
sensibilidad a bloques y spillovers nulos/no nulos. Usan controles de cambios,
por lo que no equivalen al experimento antiguo con controles en niveles.
Los diseños GVAR incluyen países I(1) independientes con T=200/300, AR(0.5),
sensibilidad a bloques y dos alternativas de transmisión de innovaciones.

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 python tools/reference_validation/validate_dependence_bootstrap.py \
  --domain spatial --replications 200 --draws 199 --output /tmp/spatial-mc.json
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 python tools/reference_validation/validate_dependence_bootstrap.py \
  --domain gvar --replications 200 --draws 399 --output /tmp/gvar-mc.json
```

Se reportan rechazo, cobertura y potencia con incertidumbre Monte Carlo entre
réplicas independientes, además de réplicas fallidas. Las pruebas GVAR correlacionadas
dentro de una réplica se promedian antes de calcular su error Monte Carlo. Las tasas
binarias usan intervalos Wilson: cero fallos observados no significa incertidumbre
cero. Las tasas sobre réplicas exitosas deben leerse junto con su denominador y
los fallos. Esta evidencia específica de cada diseño no es una calibración universal.

### Limitaciones observadas en la matriz predefinida

La ejecución de octubre de 2026 usó **200 réplicas independientes del DGP por
diseño**, semilla 812000, 199 remuestreos espaciales y 399 remuestreos GVAR.
Las 2,200 réplicas terminaron sin fallos de inferencia. Las configuraciones se
fijaron previamente, sin ajustarlas para alcanzar tasas nominales. Las cifras
son porcentajes; los corchetes contienen intervalos Monte Carlo al 95% según las
convenciones anteriores.

| Espacial N / T / bloque / spillover verdadero | Rango de cobertura puntual | Cobertura simultánea (objetivo 95%) | Rechazo conjunto al 5% |
|---|---:|---:|---:|
| 20 / 50 / 4 / 0 | 94.5–98.0 | 93.5 [89.2, 96.2] | 3.0 [1.4, 6.4] |
| 20 / 50 / 8 / 0 | 92.0–95.5 | 90.5 [85.6, 93.8] | 2.5 [1.1, 5.7] |
| 40 / 50 / 4 / 0 | 90.0–94.5 | 88.5 [83.3, 92.2] | 9.0 [5.8, 13.8] |
| 20 / 100 / 6 / 0 | 92.5–97.5 | 91.0 [86.2, 94.2] | 5.5 [3.1, 9.6] |
| 20 / 50 / 4 / 0.25 | 92.5–97.5 | 92.0 [87.4, 95.0] | 100.0 [98.1, 100.0] (potencia) |

La cobertura puntual corresponde a la respuesta indirecta en seis horizontes;
la familia simultánea incluye también respuestas directas y totales. **El bootstrap
no resuelve la cobertura insuficiente en paneles cortos:** con N=40, la cobertura
simultánea es 88.5% y el rechazo bajo la nula es 9%; incluso T=100 presenta
subcobertura. Un bloque mayor no mejoró uniformemente los resultados. Es una
referencia experimental dependiente del diseño, no un reemplazo calibrado para
todas las bandas. Estas simulaciones usan studentización Driscoll–Kraay y no
establecen el desempeño de Conley en muestras finitas.

| Referencia/diseño GVAR | Rechazo nominal anterior | Rechazo bootstrap de nula independiente |
|---|---:|---:|
| I(1) independientes, T=200, bloque 1 | 10.25 | 3.25 [2.20, 4.30] |
| I(1) independientes, T=300, bloque 1 | 10.92 | 4.08 [2.96, 5.21] |
| AR(0.5) independientes, T=200, bloque 1 | 0.08 | 4.58 [3.40, 5.76] |
| I(1) independientes, T=200, bloque 4 | 10.25 | 3.33 [2.30, 4.37] |
| Transmisión de innovaciones I(1), intensidad 1, T=200 (potencia) | 9.00 | 3.00 [1.35, 4.65] |
| Transmisión estacionaria, intensidad 2, T=250 (potencia) | 100.00 | 100.00 [98.12, 100.00] |

Las filas bajo la nula promedian seis pruebas auxiliares por réplica; las filas
de potencia promedian las dos del país transmisor. El último diseño tiene
persistencia propia 0.5 y extranjera 0.9. La referencia restringida reduce el
sobrerrechazo documentado, a veces conservadoramente, pero su **potencia de 3%
ante la transmisión I(1) revela una limitación importante**. No es una prueba
general de independencia; no rechazar tampoco valida la exogeneidad débil del GVAR.
