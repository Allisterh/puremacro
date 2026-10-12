> 🇬🇧 [English](../text_analysis.md) · 🇪🇸 Español

# Texto como datos: corpus, tokenizador y matrices documento-término

Dos capas convierten documentos de política en números:

- `puremacro.narrative`: recolecta comunicados, minutas, discursos,
  presupuestos y noticias de bancos centrales, ministerios de hacienda y otras
  agencias en un `NarrativeCorpus`, que se puede almacenar, recargar y exportar.
- `puremacro.text`: un tokenizador, segmentación en oraciones y párrafos, y una
  matriz documento-término (DTM) dispersa. Funciona con cualquier lista de
  cadenas, no sólo con un corpus recolectado, y no importa nada además de NumPy,
  SciPy y pandas.

## Recolectar y almacenar un corpus

```python
# requires: network
from puremacro.narrative.harvest import harvest_narrative_corpus

corpus = harvest_narrative_corpus(countries=["USA", "MEX"], policy_domains="monetary",
                                  start_date="2020-01-01", db_path="corpus.db")
print(corpus.harvest_report)          # documentos y fallas por conector
```

Sin `db_path=`, `store=` ni `use_cache=True`, el corpus queda en memoria y no se
escribe nada en disco. Con alguno de ellos, cada recolección inserta o actualiza
en un `CorpusStore` de SQLite que guarda el texto completo más reciente de cada
documento, marca los documentos cuyo texto cambió desde que se guardaron por
primera vez (`store.revised()`) y registra cada recolección. `store.load(...)`
devuelve el corpus acumulado.

Un conector que falla emite una advertencia y aparece en
`corpus.harvest_report` (`on_error="warn"`, `"raise"` o `"ignore"`). Los
conectores de texto figuran en el [registro de fuentes](micro_data.md#el-registro)
como `unaudited`: sus términos no se han revisado uno por uno, así que conviene
consultar los términos de cada fuente antes de republicar su texto y compartir
el manifiesto en su lugar.

### Exportar

- `corpus.to_frame(metadata="promote" | "json" | "drop")`;
- `corpus.to_jsonl(path)` / `NarrativeCorpus.from_jsonl(path)` (sin dependencias
  adicionales) y `to_parquet` / `from_parquet` (el extra `io`);
- `NarrativeCorpus.from_records(records, source=...)` construye un corpus a
  partir de los registros `(date, text, url, metadata)` de cualquier conector,
  incluidos los basados en consultas y sus propios archivos;
- cada exportación escribe a su lado un `manifest()` de procedencia (versión de
  puremacro, conteos por fuente y un hash del contenido de cada documento). Un
  paquete de replicación puede incluir el manifiesto en lugar del texto;
  `verify_manifest` compara una nueva recolección con él.

## Tokenizador

`RegexTokenizer` pasa a minúsculas, conserva números y porcentajes (los textos
de bancos centrales están llenos de ellos) y elimina las palabras vacías del
idioma de cada documento (inglés, español, portugués, alemán, francés,
italiano).

```python
from puremacro.text import RegexTokenizer

tok = RegexTokenizer()
print(tok("The Committee decided to raise the target range by 25 basis points to 5.25%.", "en"))
# ['committee', 'decided', 'raise', 'target', 'range', '25', 'basis', 'points', '5.25%']
print(tok("subió al 4,2 % en marzo", "es"))       # ['subió', '4,2%', 'marzo']
print(RegexTokenizer(numbers="mask")("cut by 50 basis points to 4.5%"))
# ['cut', '<num>', 'basis', 'points', '<pct>']
```

Cualquier `callable(text, language) -> list[str]` puede sustituirlo, así que un
lematizador de spaCy o un stemmer de NLTK se conectan como funciones simples.

## Matriz documento-término

```python
import pandas as pd
from puremacro.text import build_dtm

texts = [
    "The Committee decided to raise the target range by 25 basis points to 5.25%.",
    "Inflation remains elevated; the Committee will keep policy restrictive.",
    "La Junta de Gobierno decidió mantener la tasa de interés en 11.25%.",
    "La inflación general anual disminuyó y la Junta prevé recortes graduales.",
]
docs = pd.DataFrame({"doc_id": ["fomc-1", "fomc-2", "bx-1", "bx-2"],
                     "country": ["USA", "USA", "MEX", "MEX"],
                     "date": ["2023-07-26", "2023-09-20", "2023-08-10", "2023-09-28"]})

dtm = build_dtm(texts, languages=["en", "en", "es", "es"], docs=docs)
print(dtm)                               # 4 documentos x 29 términos, conteos dispersos
print(dtm.filter(min_df=2).vocab)        # ['committee', 'junta']

weights = dtm.tfidf()                    # la definición de TfidfTransformer de scikit-learn
panel = dtm.aggregate(by="country", freq="QS")   # una fila por país y trimestre
print(panel.docs)
tidy = dtm.to_tidy()                     # doc_id, term, value
```

- `dtm.X` es una `scipy.sparse.csr_matrix` y `dtm.docs` un marco alineado por
  filas, así que la memoria crece con las celdas no nulas.
- `filter(min_df, max_df, max_features, terms)` poda el vocabulario, y
  `ngram_range=(1, 2)` añade bigramas (formados tras quitar las palabras
  vacías).
- `aggregate(by=..., freq=...)` suma conteos en un panel por país y mes (o
  trimestre, año, semana) listo para una regresión o un VAR; agregue primero los
  conteos y aplique `tfidf()` al resultado.
- `transform(new_texts)` cuenta documentos nuevos con el mismo vocabulario.
- `save(directory)` escribe `X.mtx`, `vocab.txt` y `docs.csv`, que R lee con
  `Matrix::readMM` (y de ahí quanteda o stm).

Desde un corpus recolectado, `corpus.dtm(...)` y
`corpus.segment("paragraph" | "sentence" | callable)` hacen lo mismo con los
documentos del corpus, con las palabras vacías del idioma de cada documento.

## Modelos de tópicos

`narrative.topics.TfidfVectorizer` construye su matriz dispersa, `NMF` acepta
entradas dispersas y `DynamicTopicModel` usa ambos (`puremacro.narrative.topics`).
Los índices de incertidumbre basados en texto al estilo de Baker, Bloom y Davis
(2016) están en `narrative.indices`.
