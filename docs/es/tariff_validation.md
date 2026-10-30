> 🇬🇧 [English](../tariff_validation.md) · 🇪🇸 Español

# Por qué el informe arancelario tiene una VE del consumo positiva

El [informe principal](first_brief_tariffs.md) obtiene una VE del consumo de
Estados Unidos de **+6.427,19 millones de USD (+0,0383%)**. Un solucionador
Leontief independiente reproduce la cifra dentro de 0,000001 millones de USD.
Es un resultado del consumo en un modelo estático con inversión como otro uso
final.

| Uso final de Estados Unidos | Cambio de cantidad a precios de comprador base, millones de USD |
|---|---:|
| Consumo, C | +6.427,19 |
| Inversión, I | −7.095,64 |
| Compras de residentes en el exterior, Cx | −466,70 |

Solo C es VE del consumo. Sumar las filas no produce bienestar intertemporal
ni social: no se modelan los rendimientos futuros de la inversión, y C incluye
consumo público y de instituciones sin fines de lucro al servicio de hogares.

## Explicación presupuestaria y de recursos

El modelo asigna proporciones fijas del gasto a los usos finales. En el
experimento principal el ingreso disponible nominal de Estados Unidos,
incluidos reembolsos fiscales, aumenta 0,0660%, y el precio de su consumo
aumenta 0,0277%. El consumo sube así 0,0383%. La inversión y las compras en el
exterior tienen mayores aumentos de precios (0,1972% y 0,3304%); sus cantidades
caen 0,1417% y 0,2635%. El gasto de inversión también incorpora el ajuste
prescrito del saldo externo. Son identidades presupuestarias de equilibrio,
no contribuciones causales identificadas por separado.

Con abastecimiento fijo, A contiene los coeficientes de insumos intermedios,
F las canastas finales y v los requerimientos unitarios de factores valorados
a precios base. Los precios contables de recursos satisfacen

$$p^R=(I-A^T)^{-1}v,\qquad P^R=F^T p^R.$$

Las dotaciones fijas implican $\sum_j P_j^R\Delta c_j=0$. La comprobación
independiente obtiene cambios mundiales de **+6.761,06** para C, **−6.493,69**
para I y **−267,37** para Cx, en millones de unidades de valor factorial base;
la suma absoluta es menor que 0,000001. Estos valores de recursos sin impuestos
difieren de los precios de consumo y no son ponderaciones de bienestar. El
consumo mundial adicional usa recursos liberados por las otras categorías.

El resultado positivo de C persiste al recalcular ambos equilibrios sin
impuestos internos (+6.500,05 millones), con saldos externos cero (+5.892,06)
o con ambos cambios (+5.923,46). Ni las distorsiones tributarias internas ni
los saldos externos distintos de cero explican por sí solos el signo. Son
cierres alternativos del modelo, no una descomposición aditiva del resultado.

## Comprobación independiente y alcance

`tools/reference_validation/tariff_reference.py` comparte solo arreglos de
calibración. Dados salarios por país, resuelve sistemas lineales separados
para precios, cantidades e ingresos con reembolsos, y encuentra salarios
relativos que equilibran mercados de factores. No llama al equilibrio,
residuos, contabilidad, jacobianos ni bienestar del código de producción.
Los estados cumplen la tolerancia escalada registrada de `1e-8`; la VE de
Estados Unidos coincide dentro de `1e-5` millones de USD.

Otro evaluador reconstruye flujos físicos, beneficios cero, trabajo y capital,
precios de comprador, presupuestos privados y públicos, saldos externos y VE
para `sigma=0` y `sigma=2`. Los errores normalizados máximos son menores que
`7e-11`, y los errores de VE menores que `1.5e-8` millones de USD. Esto verifica
independientemente las ecuaciones de `sigma=2`, sin afirmar un segundo
solucionador no lineal para esa variante.

El arancel principal cubre **los 20 sectores modelados, incluidos servicios**.
El agregado `sigma=2` sustituye entre **todos los insumos intermedios país-sector**,
incluidos sectores distintos; no se limita a orígenes dentro de un mismo
sector. Las canastas finales siguen fijas. Su ganancia de C es +23.497,93
millones, junto a I de −2.840,78 y Cx de +1.470,69 a precios base. La identidad
de recursos con A fijo corresponde a `sigma=0`; no se aplica a la tecnología
con insumos cambiantes.

Los hashes, tolerancias, errores y cierres están congelados en
`tests/fixtures/first_brief_tariffs/independent_validation.json`.
Desde el checkout de desarrollo:

```bash
python -m tools.reference_validation.check_tariff_welfare --output /tmp/tariff-validation.json
```

No hay represalias, transiciones dinámicas ni elasticidades estimadas. El signo
numérico está validado para este modelo y cierre; no demuestra beneficios del
arancel para hogares reales.
