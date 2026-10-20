> 🇪🇸 Español · 🇬🇧 [English](../index.md)

# puremacro

**Macroeconometría y modelos estructurales sobre el núcleo científico de Python.**

[![Versión en PyPI](https://img.shields.io/pypi/v/puremacro.svg)](https://pypi.org/project/puremacro/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Entorno JupyterLite](https://img.shields.io/badge/JupyterLite-Live%20IDE-orange.svg)](https://jalonso1979.github.io/puremacro/)
[![Licencia: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Estime efectos de política, compare modelos económicos y exporte resultados
para una nota de investigación o de política. Comience con Python 3.11 o posterior:

```bash
python -m pip install puremacro
```

| Comience por su tarea | Primeros pasos |
|---|---|
| **Analizar datos** | Estime una respuesta a un choque con el [inicio rápido](quickstart.md), luego explore las [fuentes de datos](micro_data.md) y los [flujos de investigación](research_workflows.md). |
| **Simular políticas** | Declare un modelo de hogares con [programación dinámica](dp.md), estudie la [transmisión monetaria](monetary_transmission.md) o construya un [contrafactual comercial](trade_accounting.md). |
| **Aprender o enseñar** | Elija una [ruta de cuadernos](notebooks.md) o use el [curso bilingüe y sus laboratorios interactivos](https://github.com/jalonso1979/puremacro/tree/main/curso). El curso fija su propia versión del paquete. |

**Estado de publicación, verificado el 20 de octubre de 2026:** la última
versión publicada en [PyPI](https://pypi.org/project/puremacro/) es **5.2.0**.
Esta copia contiene el código de 5.3.0. La documentación también describe
**desarrollo de 5.3/5.4**, con notas de versión cuando cambia el comportamiento.
Las modificaciones de código de 5.4.0 no están incluidas aquí; esas notas no
establecen su disponibilidad en esta copia ni en el paquete publicado. Véase el
[historial de cambios](https://github.com/jalonso1979/puremacro/blob/main/CHANGELOG.md).

## ¿Qué es puremacro?

`puremacro` reúne estimadores econométricos y modelos macroeconómicos cuantitativos
sobre NumPy, SciPy, pandas y Matplotlib, sin extensiones compiladas propias.
El destino soportado es una instalación local en una estación de trabajo.
El núcleo numérico también funciona bajo Pyodide / WebAssembly; los flujos en
navegadores y tabletas tienen soporte limitado por los formatos de archivo,
el acceso a la red y la capacidad de cómputo. Consulte la
[guía de ejecución](tablet.md) y el [estado de validación estructural](STRUCTURAL_VALIDATION_STATUS.md)
para conocer el alcance de la evidencia.

### Subsistemas Clave

1. **Modelos DSGE Estructurales y Paridad con Dynare**:
   - **Analizador Nativo de Archivos `.mod` de Dynare y CLI `puremacro-dynare`**: Carga y resolución de archivos `.mod` estándar directamente en Python puro.
   - **Scores de Verosimilitud**: Recursión analítica de Kalman para parámetros estructurales en la verosimilitud estacionaria de primer orden soportada; las derivadas estructurales de entrada pueden usar diferencias finitas. NUTS también ofrece gradientes numéricos y suministrados por el usuario; la interfaz de estimación determina su selección y las alternativas automáticas ([Guía](dsge_estimation.md)).
   - **HMC y NUTS en Python Puro**: Monte Carlo Hamiltoniano con adaptación dual averaging de tamaño de paso, adaptación de masa diagonal y diagnósticos de convergencia $\hat{R}$ divididos, sin normalización por rangos.
   - **Política Discrecional Óptima vs Compromiso**: Iteraciones matriciales de Riccati de Dennis (2007) para política discrecional perfecta en el sentido de Markov frente a compromiso en perspectiva atemporal, descomponiendo los sesgos de inflación y estabilización.
   - **Modelado Híbrido DSGE-VAR**: Optimización de previas e identificación estructural DSGE-VAR($\lambda$) de Del Negro y Schorfheide (2004).
   - **Choques Anticipados de Noticias**: Aumento del espacio de estados con operadores nilpotentes que preservan la determinancia de Blanchard-Kahn.
   - **Filtrado No Lineal de Partículas y MS-DSGE**: Filtros de partículas bootstrap y auxiliares con volatilidad estocástica y soluciones racionales con cambio de régimen de Markov de Foerster et al. (2016) con GIRFs analíticas cerradas.
   - **Perturbación y Restricciones Ocasionalmente Activas**: Perturbación podada para modelos sin restricciones y OccBin lineal por partes para regímenes con restricciones activas ([Guía](dsge_build.md)). La función de desarrollo de 5.4 `solve_pruned_occbin` admite OccBin lineal, incluidos choques anticipados, y trayectorias de segundo orden sin restricciones activas con una sola innovación inicial; los episodios no lineales con restricciones activas y las innovaciones futuras anticipadas no lineales generan errores explícitos. Esta función no está presente en el código de 5.3 ([Alcance de desarrollo](dsge_v3.md)).

2. **Modelos de Agentes Heterogéneos (DP Declarativa y HANK)**:
   - **Modelos de Hogares como Ecuaciones**: Declare estados, decisiones, recompensas y restricciones con `dp.Model`, y elija VFI, EGM, HJB en tiempo continuo o una resolución de Krusell–Smith admitida ([Guía](dp.md)).
   - **HANK en el Espacio de Secuencias** (Auclert, Bardóczy, Rognlie y Straub 2021, *Econometrica*): Modelos de equilibrio general con mercados incompletos resueltos en $\mathcal{O}(T^3)$.
   - **Algoritmo de Noticias Falsas (Fake News)**: Cálculo acelerado $\mathcal{O}(T^2)$ de jacobianos de consumo en el espacio de secuencias mediante vectores de expectativa e identidades de acumulación.
   - **Puente HANK en Espacio de Secuencias**: Declaración directa del bloque `hetagent_block` en archivos `.mod` estándar de Dynare.
   - **Transferencias Fiscales Focalizadas**: Respuestas dinámicas de consumo y multiplicadores acumulados por deciles de riqueza.
   - **HJB en Tiempo Continuo**: Esquemas **implícitos** de M-matriz upwind en diferencias finitas de Achdou et al. (2022), la ecuación de Kolmogorov hacia adelante adjunta para la densidad estacionaria y el equilibrio general de Aiyagari en tiempo continuo.

3. **Motores Econométricos y Proyecciones Locales**:
   - **`LPResult` Unificado**: Proyecciones locales estandarizadas (`lp_hac`, `lp_iv`, `lp_state_dep`, `panel_lp`) con errores estándar Newey-West HAC, fixed-$b$ y Driscoll-Kraay.
   - **SVAR y FAVAR**: Cholesky, Blanchard-Quah, restricciones de signo, restricciones narrativas de signo, instrumentos externos/proxy, max-share/noticias y VAR aumentado por factores.
   - **Diferencias en Diferencias Modernas**: Estimadores con adopción escalonada robustos a heterogeneidad en el tratamiento (Callaway y Sant'Anna, Sun y Abraham, Borusyak-Jaravel-Spiess, SDID, Honest DiD).
   - **Econometría Espacial y VAR Global**: Errores estándar Conley HAC, modelos autorregresivos espaciales (`sar`, `sem`, `sdm`), proyecciones locales espaciales y VAR Global de Pesaran (`var.gvar`).

4. **Suites de Política Macroeconómica Aplicada**:
   - **Postura Monetaria y Gráficos de Abanico**: Evaluación de la postura monetaria en tiempo real frente a reglas de Taylor contrafactuales, proyecciones en abanico y sentimiento textual en comunicados de bancos centrales ([Guía](notebooks.md)).
   - **Nowcasting en Tiempo Real y Descomposición de Noticias**: Modelos de factores dinámicos (DFM) de frecuencia mixta (Giannone, Reichlin y Small), tratamiento de extremos irregulares (ragged edges) y atribución de sorpresas de datos ([Guía](nowcast.md)).
   - **Riesgo Macroprudencial y Estabilidad Financiera**: Densidades condicionales skew-$t$ de Growth-at-Risk (GaR) y redes de desbordamiento de riesgo sistémico de Diebold-Yilmaz ([Guía](notebooks.md)).
   - **Política Fiscal y DSA Soberano**: Multiplicadores fiscales trimétodo y análisis estocástico de sostenibilidad de la deuda pública ante escenarios de estrés macroeconómico y climático.

5. **Ecosistema de Datos Globales y Paneles Modulares**:
   - **Microdatos de Encuestas**: Los lectores de ACS, CPS, SCF y ENIGH conservan el diseño muestral disponible en cada fuente; la interfaz común `MicroFrame` ofrece las estadísticas y estimaciones de incertidumbre admitidas ([Guía](micro_data.md)).
   - **Emisiones y Datos Climáticos**: Cuentas de emisiones y gases de efecto invernadero del Banco Mundial WDI y la OCDE SDMX a nivel internacional y sectorial ([Guía](data_ecosystem.md)).
   - **Balances Energéticos y Materias Primas**: Consumo de energía primaria, cuotas de generación limpia y suites ampliadas de precios de materias primas del Pink Sheet del Banco Mundial ([Guía](data_ecosystem.md)).
   - **Estabilidad Financiera Internacional**: Curvas de rendimiento soberano (10Y/2Y), tasas de política monetaria, brechas de crédito del BPI (BIS) y precios reales de vivienda ([Guía](data_ecosystem.md)).
   - **Constructores Modulares de Paneles**: Funciones de alto nivel `build_climate_panel` y `build_financial_panel` con armonización temporal automatizada (M$\to$Q, A$\to$Q).

6. **ML Causal, Agentes Heterogéneos en Tiempo Continuo y Tiempo Real Regional**:
   - **Aprendizaje Automático Doble / Desesgado**: Regresión parcialmente lineal de Chernozhukov et al. (2018) con puntajes ortogonales de Neyman-Robinson, validación cruzada por $K$ bloques y aprendices penalizados en NumPy puro: únicamente lasso y ridge, sin red elástica.
   - **HJB Implícito, KFE Adjunta y EG de Aiyagari Continuo**: Iteración de política sobre una M-matriz dispersa, una densidad estacionaria normalizada con pesos de cuadratura trapezoidal que integra a uno también en mallas no uniformes, y un bucle de equilibrio general cuyo indicador `converged` significa $|K^s - K^d| \le$ `tol_ge`.
   - **OccBin Multirrestricción**: $M \ge 2$ restricciones ocasionalmente activas simultáneas, con máscara de bits de régimen (el bit $k$ se activa cuando la restricción $k$ está activa) y no convergencia reportada como `converged=False` más una advertencia que nombra cada motivo ([Guía](dsge_build.md)).
   - **Inferencia con Instrumentos Débiles de Montiel Olea-Pflueger**: Valores críticos del estadístico $F$ efectivo a partir de la forma cerrada $\chi^2_{K,\,1-\alpha}(K/\tau)/K$, y conjuntos de Anderson-Rubin multiinstrumentales invertidos de forma exacta y no sobre una malla ([Guía](lp.md)).
   - **Conectores en Tiempo Real de América Latina**: Banco de México, INEGI, Banco Central do Brasil y Banco Central de Chile. Estas cuatro fuentes publican únicamente la edición vigente, así que su fecha de añada es la fecha de *captura* local y el historial de revisiones sólo se acumula con capturas repetidas en días distintos.

7. **Comercio Internacional y MRIO** — reincorporados en el paquete 5.2.0 publicado:
   - **Contrafactuales de Política**: Modelos comerciales de varios países y sectores con contabilidad de productor/comprador, recaudación fiscal explícita y un cierre de equilibrio explícito ([Guía](trade_accounting.md)).
   - **Datos OCDE Reproducibles**: La tabla OCDE ICIO incluida (edición 2023, año de referencia 2019) abarca los 19 países miembros del G20 más el resto del mundo y 20 secciones CIIU, con un script de reconstrucción y atribución a la fuente ([Datos y modelos](spatial_and_trade_ge.md)).
   - **Incidencia en los Hogares**: Comparaciones estáticas de bienestar EV/CV condicionadas a los precios de comprador, cambios de ingreso y transferencias fiscales suministrados; la demanda de los hogares no retroalimenta el equilibrio comercial ([Guía](trade_distributional.md)).

8. **Generación de Informes y Publicación**:
   - Exportación de tablas con calidad de imprenta directamente a **LaTeX** (`.to_latex()`), **Typst** (`.to_typst()`) y **Markdown** (`.to_markdown()`), con errores estándar y estrellas de significancia ([Guía](reporting.md)).

9. **Ejecución en Cualquier Dispositivo**:
   - Núcleo numérico compatible con Pyodide para cuadernos en el navegador, con soporte limitado en tabletas y navegadores ([Guía](tablet.md)), herramientas para descargar cómputo a Google Colab (`runtime.colab`), ejecución segmentada (`longrun`) y cartuchos portátiles `.pmz` (`pocket`).

---

## Instalación

La orden al inicio instala el paquete básico. Las herramientas opcionales añaden
soporte para formatos de archivo y utilidades para cuadernos:

Los lectores de parquet y Excel (las lecciones ENOE del curso, `build_all`, el atlas de choques) necesitan los motores opcionales de formato de archivo:

```bash
pip install "puremacro[io]"
```

O con herramientas completas para cuadernos interactivos:

```bash
pip install "puremacro[notebooks]"
```
