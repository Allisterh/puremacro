> 🇬🇧 [English](../study_workflows.md) · 🇪🇸 Español

# Plantillas ejecutables de estudios y escenarios

Estos ejemplos requieren Puremacro 5.6.0 y funcionan sin conexión con sus
dependencias estándar. Conectan datos o declaraciones del modelo, estimación o
solución, diagnósticos e informes archivables. Demuestran métodos y supuestos;
no replican un resultado causal publicado de política económica.

El [cuaderno ejecutado 69 en español](https://github.com/jalonso1979/puremacro/blob/main/notebooks/69_reproducible_studies_es.ipynb)
y su [fuente editable](https://github.com/jalonso1979/puremacro/blob/main/notebooks/69_reproducible_studies_es.py) recorren los tres
experimentos, con figuras, auditorías, archivos verificables y ejercicios.
Requieren Python 3.11+, Puremacro 5.6.0 y `_nbstyle.py` junto al cuaderno.

```bash
python -m puremacro.examples.study_workflows --output study_output
# También puede seleccionar --workflow mexico, household o tariff.
```

El directorio debe ser nuevo. Cada subdirectorio contiene Markdown, LaTeX,
Typst, CSV, un manifiesto y `study.pmstudy`. Los escenarios guardan además el
modelo como texto inerte. El archivo retiene los insumos y tablas de evidencia;
véanse las verificaciones de integridad en [estudios](studies.md).

## Tasas de interés e inflación mexicanas

`run_mexico` usa los archivos congelados del curso de observaciones OCDE
distribuidas por FRED: [tasa interbancaria a tres meses](https://fred.stlouisfed.org/series/IR3TIB01MXM156N)
e [inflación interanual del IPC](https://fred.stlouisfed.org/series/CPALTT01MXM659N).
El manifiesto guarda fuentes, unidades y hashes. No se registró la fecha de
descarga original; no son vintages en tiempo real. La ventana común es enero de
2000–diciembre de 2023, mensual, sin eliminar ni interpolar meses interiores.

Los datos OCDE distribuidos por FRED son de terceros y no están cubiertos por
la licencia MIT del código. Cite a la OCDE y conserve la atribución al compartir.
El manifiesto incluido conserva fuentes, hashes, citas y
[condiciones OCDE](https://www.oecd.org/en/about/terms-conditions.html).

El VAR recursivo tiene seis rezagos, ordena inflación antes de la tasa y muestra
horizontes 0–18 ante un aumento de 0.25 puntos porcentuales de la tasa
**anualizada**. La inflación ya es crecimiento interanual; no se transforma
otra vez. El orden supone que no responde dentro del mes a la innovación de
tasas identificada. No es una sorpresa identificada con instrumento externo ni
un choque narrativo. El sistema de dos variables omite información relevante
y puede producir una respuesta positiva de precios. Los 199 remuestreos con
semilla usan la normalización fija documentada en la guía de estudios. No se
calibra automáticamente un HANK contra este VAR.

```python
from puremacro.examples.study_workflows import run_mexico

result = run_mexico(n_boot=199)
print(result.to_frame().head())
print(result.audits)
```

## Transferencia a hogares escrita como programa dinámico

`run_household` carga declaraciones DP `.mod`, resuelve EGM y calcula una
transición no lineal de 40 trimestres desde la distribución estacionaria. Una
transferencia inicial de 0.05 unidades de consumo por hogar proviene del
exterior del modelo. Los precios y el riesgo de ingresos permanecen fijos; no
hay financiamiento fiscal ni respuesta de equilibrio general. Es una calibración
ilustrativa, **no un ajuste a una encuesta**.

Se reportan cambios en consumo y activos al entrar en cada fecha, restando la
trayectoria sin choque bajo idénticos valores terminales. Se conservan ambas
trayectorias, transferencia, distribución inicial, políticas, matriz de transición,
parámetros y auditoría estacionaria. Los activos no saltan al impacto; después,
consumo más ahorro agotan los recursos disponibles. Las pruebas verifican esa
identidad. La auditoría conserva fallos de malla o frontera; la tolerancia del
solver no certifica precisión de discretización. Use [refinamiento explícito](result_audits_extended.md)
al cambiar mallas u horizontes. Para calibración con encuesta consulte el
[ejemplo Aiyagari con SCF](micro_data.md).

```python
from puremacro.examples.study_workflows import run_household

scenario = run_household(amount=0.05, horizon=40, n_a=100)
print(scenario.to_frame().head())
print(scenario.audits)
```

## Aranceles en una red de producción

`run_tariff` agrega la tabla **OCDE ICIO, edición 2023, año de referencia 2019**
a Estados Unidos/México/resto y recursos/manufacturas/otras industrias (incluidos
servicios públicos, construcción y servicios). Cada país y
actividad pertenece a un grupo; se conservan todos los totales de transacciones
intermedias, usos finales y valor agregado. Se archivan hashes, mapas y tabla de
calibración. Cite OECD (2023), OECD Inter-Country Input-Output Tables,
[OCDE ICIO](https://www.oecd.org/en/data/datasets/inter-country-input-output-tables.html).
Los términos de atribución y adaptación de
`puremacro/trade/_datafiles/SOURCES.md` también se aplican a resultados redistribuidos.

El escenario ilustrativo es un arancel estadounidense de 5% sobre manufacturas
mexicanas, con devolución de recaudación a suma fija y ahorro externo fijo en
unidades del numerario. El precio de recursos estadounidenses vale uno. Por
supuesto de calibración, dos tercios del valor agregado remuneran trabajo y un
tercio capital. Véanse las convenciones en la [guía contable](trade_accounting.md).
Los cambios son de producción bruta a precios base, **no PIB, bienestar ni
pronóstico de política**. Ambos equilibrios deben converger y se conservan
las verificaciones contables identificadas, todos los residuos de las ecuaciones
físicas, precios y cantidades. El máximo mezcla unidades propias de las ecuaciones
(cantidades, precios relativos, factores y presupuestos nominales); no representa
un único importe monetario. No se estima incertidumbre paramétrica
ni se seleccionan cierres alternativos.

```python
from puremacro.examples.study_workflows import run_tariff

tariff = run_tariff(rate=0.05)
print(tariff.to_frame())
```

Estudios empíricos y escenarios permanecen separados. Una comparación requiere
frecuencia, unidades, interpretación del choque, convención de respuesta e
insumos compatibles, además de una justificación económica explícita.
