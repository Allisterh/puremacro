> 🇬🇧 English · 🇪🇸 [Español](es/text_analysis.md)

# Text as data: corpus, tokenizer and document-term matrices

Two layers turn policy documents into numbers:

- `puremacro.narrative`: harvests statements, minutes, speeches, budgets and
  news from central banks, treasuries and other agencies into a
  `NarrativeCorpus`, which can be stored, reloaded and exported.
- `puremacro.text`: a tokenizer, sentence and paragraph segmentation, and a
  sparse document-term matrix (DTM). It works on any list of strings, not only
  on a harvested corpus, and imports nothing beyond NumPy, SciPy and pandas.

## Harvesting and storing a corpus

```python
# requires: network
from puremacro.narrative.harvest import harvest_narrative_corpus

corpus = harvest_narrative_corpus(countries=["USA", "MEX"], policy_domains="monetary",
                                  start_date="2020-01-01", db_path="corpus.db")
print(corpus.harvest_report)          # documents and failures per connector
```

Without `db_path=`, `store=` or `use_cache=True` the corpus stays in memory and
nothing is written to disk. With one of them, each harvest upserts into a SQLite
`CorpusStore` that keeps the latest full text of every document, flags documents
whose text changed since they were first stored (`store.revised()`), and logs
each harvest run. `store.load(...)` returns the accumulated corpus.

A connector that fails raises a warning and is listed in
`corpus.harvest_report` (`on_error="warn"`, `"raise"` or `"ignore"`). The text
connectors are listed in the [source registry](micro_data.md#the-registry) as
`unaudited`: their terms have not been reviewed one by one, so check a
source's own terms before republishing its text, and share the manifest instead.

### Exporting

- `corpus.to_frame(metadata="promote" | "json" | "drop")`;
- `corpus.to_jsonl(path)` / `NarrativeCorpus.from_jsonl(path)` (no extra
  dependency) and `to_parquet` / `from_parquet` (the `io` extra);
- `NarrativeCorpus.from_records(records, source=...)` builds a corpus from any
  connector's `(date, text, url, metadata)` records, including query-driven
  ones and your own files;
- every export writes a provenance `manifest()` next to it (puremacro version,
  per-source counts, and a content hash per document). A replication package can
  ship the manifest instead of the text; `verify_manifest` checks a re-harvest
  against it.

## Tokenizer

`RegexTokenizer` lowercases, keeps numbers and percentages (central-bank text
is full of them), and removes the stop words of each document's language
(English, Spanish, Portuguese, German, French, Italian).

```python
from puremacro.text import RegexTokenizer

tok = RegexTokenizer()
print(tok("The Committee decided to raise the target range by 25 basis points to 5.25%.", "en"))
# ['committee', 'decided', 'raise', 'target', 'range', '25', 'basis', 'points', '5.25%']
print(tok("subió al 4,2 % en marzo", "es"))       # ['subió', '4,2%', 'marzo']
print(RegexTokenizer(numbers="mask")("cut by 50 basis points to 4.5%"))
# ['cut', '<num>', 'basis', 'points', '<pct>']
```

Any `callable(text, language) -> list[str]` can replace it, so a spaCy
lemmatizer or an NLTK stemmer plugs in as a plain function.

## Document-term matrix

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
print(dtm)                               # 4 docs x 29 terms, sparse counts
print(dtm.filter(min_df=2).vocab)        # ['committee', 'junta']

weights = dtm.tfidf()                    # scikit-learn's TfidfTransformer definition
panel = dtm.aggregate(by="country", freq="QS")   # one row per country and quarter
print(panel.docs)
tidy = dtm.to_tidy()                     # doc_id, term, value
```

- `dtm.X` is a `scipy.sparse.csr_matrix` and `dtm.docs` a row-aligned frame, so
  memory scales with non-zero cells.
- `filter(min_df, max_df, max_features, terms)` prunes the vocabulary,
  `ngram_range=(1, 2)` adds bigrams (formed after stop-word removal).
- `aggregate(by=..., freq=...)` sums counts into a country-by-month (or
  quarter, year, week) panel ready for a regression or a VAR; aggregate the
  counts first, then call `tfidf()` on the result.
- `transform(new_texts)` counts new documents on the same vocabulary.
- `save(directory)` writes `X.mtx`, `vocab.txt` and `docs.csv`, which R reads
  with `Matrix::readMM` (and from there quanteda or stm).

From a harvested corpus, `corpus.dtm(...)` and
`corpus.segment("paragraph" | "sentence" | callable)` do the same on the
corpus documents, with stop words taken from each document's language.

## Topic models

`narrative.topics.TfidfVectorizer` builds its matrix sparse, `NMF` accepts
sparse input, and `DynamicTopicModel` uses both (`puremacro.narrative.topics`).
Text-based uncertainty indices in the style of Baker, Bloom and
Davis (2016) live in `narrative.indices`.
