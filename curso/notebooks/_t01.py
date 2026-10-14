"""Data contracts for the T01 A -> B -> C teaching sequence.

No network access: this module only reads the frozen files in ``data_curso/``.
The downloads live in puremacro and are called by T01_A under ``MAV_T01_VIVO``;
the notebooks remain the pedagogical source, and this module shares the reading
path and the calendar checks so that independent kernels use the same
observations.
"""
from __future__ import annotations

import gzip
import hashlib
import inspect
import json
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from _datos import ruta_dato


def _sort_codes(s):
    """Integer keys that sort like ``sort_values`` sorts ``s`` (a categorical by its category order)."""
    if isinstance(s.dtype, pd.CategoricalDtype):
        return s.cat.codes.to_numpy()
    return pd.factorize(s, sort=True)[0]


def validate_long(frame, freq):
    """Unique (code, date, variable), finite values, dates at period start.

    Returns the rows sorted by code, date and variable. Sorted through integer keys
    and one ``take``: no copy of the frame before it is needed (a panel has millions of rows).
    A categorical label sorts by its category order, and a missing label sorts first
    (``sort_values`` put it last); a validated panel has none.
    """
    date = pd.to_datetime(frame["date"])
    keys = [_sort_codes(frame["code"]), date.to_numpy().astype("int64"), _sort_codes(frame["variable"])]
    if _strictly_increasing(keys):           # already sorted and unique (a spliced panel): nothing to move
        order = None
    else:
        order = np.lexsort(keys[::-1])
        if _has_ties([k[order] for k in keys]):
            raise ValueError("Duplicate country / period / variable")
    if not np.isfinite(frame["value"].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite observations; represent missingness by an absent row")
    if not _period_starts(date.to_numpy(), freq):
        raise ValueError(f"Dates must denote the beginning of a {freq} period")
    out = frame.reset_index(drop=True) if order is None else frame.take(order).reset_index(drop=True)
    if order is not None or out["date"].dtype != date.dtype:
        out["date"] = date.to_numpy() if order is None else date.to_numpy()[order]
    return out


def _period_starts(v, freq):
    """Every date is the first instant of its year, quarter or month (numpy only: no Period objects)."""
    if freq not in ("Y", "Q", "M"):
        d = pd.Series(v)
        return bool((d == d.dt.to_period(freq).dt.to_timestamp()).all())
    start = v.astype("datetime64[Y]" if freq == "Y" else "datetime64[M]")
    ok = start.astype(v.dtype) == v
    if freq == "Q":
        ok &= start.astype("int64") % 3 == 0        # months since 1970-01: a quarter starts at 0, 3, 6, 9
    return bool(ok.all())


def _strictly_increasing(keys):
    """Rows already in strict lexicographic order of ``keys`` (so sorted, and no two alike)."""
    if len(keys[0]) < 2:
        return True
    greater, equal = np.zeros(len(keys[0]) - 1, bool), np.ones(len(keys[0]) - 1, bool)
    for k in keys:
        greater |= equal & (k[1:] > k[:-1])
        equal &= k[1:] == k[:-1]
    return bool(greater.all())


def _has_ties(sorted_keys):
    """Two consecutive rows equal on every key."""
    if len(sorted_keys[0]) < 2:
        return False
    equal = np.ones(len(sorted_keys[0]) - 1, bool)
    for k in sorted_keys:
        equal &= k[1:] == k[:-1]
    return bool(equal.any())


_NAMES = {"Y": "annual", "Q": "quarterly", "M": "monthly"}

# What B and C read from each panel: T01_A checks every name has a record and
# survives the stacking; B and C load exactly these (``load_panel(f, variables=CONTRACT[f])``).
CONTRACT = {
    "Y": ["gdp_real", "population", "gdp_pc_ppp"],
    "Q": [v for c in ["gdp", "cons_hh", "cons_gov", "inv", "capform", "exports", "imports"]
          for v in (c, c + "_defl")]
         + ["hours", "hours_factor", "urate", "comp_emp", "va_total", "emp", "emp_employees",
            "inv_equip_defl", "inv_struct_defl"],
    "M": ["ip", "urate"],
}
_CAT = {"code": "category", "variable": "category", "source": "category"}


def load_panel(freq, variables=None, codes=None, meta=False, folder=None):
    """The frozen long panel, filtered before validation.

    ``variables`` and ``codes`` keep only those rows (B reads two monthly
    series out of dozens, and the full files are large). ``meta=True`` joins
    the series dictionary ``t01_series.csv`` (units, kind, aggregation, sa)
    when it exists. Text columns are read as categoricals to keep memory
    within the tablet's budget; Namibia's ISO2 code ``NA`` is not a missing value.
    ``folder`` reads another delivery (T01_A rehearsing a refresh) instead of data_curso/.
    """
    name = f"t01_{_NAMES[freq]}.csv.gz"
    path = ruta_dato(name) if folder is None else Path(folder) / name
    d = pd.read_csv(path, dtype=_CAT, keep_default_na=False, na_values=[""])
    # A name that is not in the file is an error, not an empty frame (a renamed series would pass silently).
    for col, asked in (("variable", variables), ("code", codes)):
        unknown = sorted(set(asked if asked is not None else ()) - set(d[col].cat.categories))
        if unknown:
            raise ValueError(f"Not in the {_NAMES[freq]} panel ({col}): {', '.join(unknown)} (see t01_series.csv)")
    if variables is not None:
        d = d[d["variable"].isin(list(variables))]
    if codes is not None:
        d = d[d["code"].isin(list(codes))]
    for c in d.columns:
        if isinstance(d[c].dtype, pd.CategoricalDtype):
            d[c] = d[c].cat.remove_unused_categories()
    d = validate_long(d, freq)
    if meta:
        series = ruta_dato("t01_series.csv")
        if series.is_file():
            s = pd.read_csv(series, keep_default_na=False)
            cols = [c for c in ("variable", "units", "kind", "aggregation", "sa") if c in s]
            d = d.merge(s.loc[s["freq"] == freq, cols], on="variable", how="left")
    return d


def wide_panel(d):
    """One column per variable, rows (code, date): the long panel turned wide.

    ``unstack`` of the unique keys, not ``pivot_table``: nothing to aggregate, and a
    tenth of the memory. Labels that a filtered categorical no longer uses give no column.
    A duplicate (code, date, variable) raises, where ``pivot_table(aggfunc="first")``
    kept the first silently: pass a validated panel (``validate_long``).
    """
    w = d.set_index(["code", "date", "variable"])["value"].unstack("variable").sort_index()
    v = d["variable"]
    order = v.cat.categories if isinstance(v.dtype, pd.CategoricalDtype) else sorted(v.unique())
    w = w[[c for c in order if c in w.columns and w[c].notna().any()]]     # the order a groupby gives
    w.columns = w.columns.astype(str)
    w.columns.name = "variable"
    if isinstance(w.index.levels[0].dtype, pd.CategoricalDtype):
        w.index = w.index.set_levels(w.index.levels[0].astype(str), level=0)
    return w


def coverage(d, freq):
    """One receipt per country-variable, including missing interior calendar periods.

    Grouped on the labels as they come (categoricals stay codes, not a million
    strings), then sorted by their text: the same table whatever the dtype.
    """
    o = pd.Series(pd.PeriodIndex(pd.to_datetime(d["date"]), freq=freq).asi8, index=d.index)
    out = (o.groupby([d["code"], d["variable"]], observed=True).agg(n="count", lo="min", hi="max")
           .reset_index().astype({"code": str, "variable": str})
           .sort_values(["code", "variable"]).reset_index(drop=True))
    first = pd.PeriodIndex.from_ordinals(out["lo"].to_numpy(), freq=freq).astype(str)
    last = pd.PeriodIndex.from_ordinals(out["hi"].to_numpy(), freq=freq).astype(str)
    return pd.DataFrame({"code": out["code"], "variable": out["variable"],
                         "n": out["n"].astype("int64"),
                         "first": list(first), "last": list(last),
                         "internal_gaps": (out["hi"] - out["lo"] + 1 - out["n"]).astype("int64")})


def regular_block(frame, columns, freq="Q", start=None, end=None, min_obs=40, which="longest"):
    """Longest complete contiguous block; no differencing over missing periods.

    Equal-length ties select the most recent block. Returns None when no block
    is long enough. Callers report the retained start/end/n for every statistic.
    """
    d = frame.loc[start:end, list(columns)].sort_index().copy()
    if d.empty:
        return None
    periods = pd.PeriodIndex(d.index, freq=freq)
    if periods.has_duplicates:
        raise ValueError("More than one row per calendar period")
    d.index = periods
    d = d.reindex(pd.period_range(periods.min(), periods.max(), freq=freq))
    good = d.notna().all(axis=1) & np.isfinite(d).all(axis=1)
    groups = (~good).cumsum()
    blocks = [b for _, b in d.loc[good].groupby(groups[good]) if len(b) >= min_obs]
    if not blocks:
        return None
    if which not in ("longest", "latest"):
        raise ValueError(which)
    out = max(blocks, key=lambda b: (len(b), b.index[-1]) if which == "longest" else b.index[-1])
    out.index = out.index.to_timestamp()
    return out


def sahm(u):
    """3-month mean minus the minimum mean in the preceding 12 calendar months.

    Percentage points; no rounding before thresholding. The current month is
    excluded from the baseline. Requires a regular monthly index.
    """
    periods = pd.PeriodIndex(u.index, freq="M")
    if periods.has_duplicates:
        raise ValueError("Duplicate month")
    s = pd.Series(u.to_numpy(dtype=float), index=periods).sort_index()
    s = s.reindex(pd.period_range(s.index.min(), s.index.max(), freq="M"))
    ma = s.rolling(3, min_periods=3).mean()
    result = ma - ma.shift(1).rolling(12, min_periods=12).min()
    result.index = result.index.to_timestamp()
    return result


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------------------------
# Plumbing for T01_A. The notebook keeps every puremacro call, every rule and every receipt;
# what lives here is bookkeeping that would only lengthen its cells: block files and their
# sidecars, stacking long frames without losing their categories, the rounding and byte-level
# writing of the delivery, and the manifest.
# ----------------------------------------------------------------------------------------------

LABELS = {"code": "category", "variable": "category", "source": "category"}


def call_text(download):
    """The puremacro call inside ``lambda: f(...)``, exactly as written, for the manifest."""
    try:
        src = inspect.getsource(download)
    except (OSError, TypeError):
        return None
    if "lambda:" not in src:
        return f"{download.__name__}()"
    body, depth = src.split("lambda:", 1)[1].lstrip(), 0
    for i, ch in enumerate(body):
        depth += (ch == "(") - (ch == ")")
        if ch == ")" and depth == 0:
            return " ".join(body[: i + 1].split())
    return None


def release_of(attrs):
    """The provider's own vintage stamp, never our clock (the OECD and the BIS publish none)."""
    stamp = attrs.get("release") or attrs.get("lastupdated")
    if not stamp and isinstance(attrs.get("last_update"), dict) and attrs["last_update"]:
        stamp = max(map(str, attrs["last_update"].values()))
    return stamp


def download_quietly(download):
    """Run a download; return its frame and the provider's UserWarnings with folders cut.

    Other warnings are dropped: a warning's file name would print a local path.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        d = download()
    said = [re.sub(r"(/[^/\s'\"]+)+/", "", str(w.message))[:200]
            for w in caught if issubclass(w.category, UserWarning)]
    return d, said


def read_block(path):
    """A frozen block as the live call returned it: labels as categories, attrs from its sidecar.

    ``NA`` is Namibia, never a missing value; floats are parsed exactly as written
    (pandas' fast parser can miss by one bit, enough to flip a later rounding).
    """
    path = Path(path)
    d = pd.read_csv(path, keep_default_na=False, na_values=[""], float_precision="round_trip",
                    dtype={"code": "category", "variable": "category"})
    side = path.with_name(path.name.split(".")[0] + ".meta.json")
    try:
        d.attrs.update(json.loads(side.read_text()))
    except (OSError, ValueError):
        pass
    return d


def stack(frames, labels=LABELS):
    """``pd.concat`` of long frames that keeps the label columns categorical.

    Concatenating categoricals with different categories yields plain strings,
    a few million of them; here every frame is first recoded to the union of
    categories (sorted, as ``astype("category")`` would sort them).
    """
    frames = list(frames)
    dtypes = {}
    for col in labels:
        seen = set()
        for f in frames:
            s = f[col]
            seen.update(s.cat.categories if isinstance(s.dtype, pd.CategoricalDtype) else s.unique())
        dtypes[col] = sorted(map(str, seen))
    return pd.concat([f.assign(**{c: _recode(f[c], cats) for c, cats in dtypes.items() if c in f})
                      for f in frames], ignore_index=True)


def _recode(s, categories):
    """``s`` as a categorical with exactly these categories, in this order.

    ``astype`` would not do: two unordered categorical dtypes with the same
    categories in another order compare equal, and the cast is skipped.
    """
    if isinstance(s.dtype, pd.CategoricalDtype):
        return s.cat.set_categories(categories)
    return s.astype(pd.CategoricalDtype(categories))


def relabel(s, rename, where=None):
    """A categorical Series with its labels passed through ``rename`` (a function).

    Works on the categories, not on the rows: a million rows with forty labels cost
    forty calls, and no million strings are built. Two labels may become one (KOS
    and RKS are both XKX). With ``where`` (a boolean array) only those rows are renamed.
    """
    s = s if isinstance(s.dtype, pd.CategoricalDtype) else s.astype("category")
    s = s.cat.remove_unused_categories()
    old = [str(c) for c in s.cat.categories]
    new = [rename(c) for c in old]
    cats = sorted(set(new) | (set(old) if where is not None else set()))
    pos = {c: i for i, c in enumerate(cats)}
    codes = s.cat.codes.to_numpy()
    out = np.array([pos[n] for n in new] + [-1], dtype=np.int64)[codes]        # code -1 (missing) stays -1
    if where is not None:
        out = np.where(where, out, np.array([pos[o] for o in old] + [-1], dtype=np.int64)[codes])
    return pd.Series(pd.Categorical.from_codes(out, categories=cats), index=s.index,
                     name=s.name).cat.remove_unused_categories()


def variable_slices(variables, n):
    """``variables`` in ``n`` sorted groups of about equal size (a splice per group)."""
    names = sorted(variables)
    return [list(g) for g in np.array_split(np.array(names, dtype=object), max(1, min(n, len(names))))]


def stack_splices(parts):
    """Join the (panel, provenance, seams) of several splices into one triple, sorted.

    ``parts`` may be a generator: each panel's labels become categorical as soon as
    it arrives, so the strings of one slice are gone before the next is spliced.
    A slice without seams returns an empty, untyped table: it is left out so the
    columns keep their types.
    """
    panels, provs, seams = [], [], []
    for panel, prov, seam in parts:
        panels.append(panel.astype({c: "category" for c in LABELS}))
        provs.append(prov)
        seams.append(seam)
        del panel
        release_memory()            # the slice's strings, back to the system before the next slice
    panel = stack(panels).sort_values(["code", "date", "variable"], ignore_index=True)
    prov = pd.concat([p for p in provs if len(p)] or provs[:1], ignore_index=True)
    seam = pd.concat([p for p in seams if len(p)] or seams[:1], ignore_index=True)
    return panel, prov.sort_values(["code", "variable"], ignore_index=True), seam


def drop_keys(raw, keys):
    """``raw`` without the rows whose (source, code, variable) is a row of ``keys``."""
    keys = pd.MultiIndex.from_frame(keys[["source", "code", "variable"]].astype(str))
    near = raw[raw.variable.isin(set(keys.get_level_values(2))) & raw.source.isin(set(keys.get_level_values(0)))]
    hit = pd.MultiIndex.from_arrays([near.source.astype(str), near.code.astype(str),
                                     near.variable.astype(str)]).isin(keys)
    return raw.drop(near.index[hit])


def replace_variables(old, new, variables):
    """Swap the rows of ``variables`` in a (panel, provenance, seams) triple for those of ``new`` (unsorted)."""
    keep = [o[~o.variable.isin(variables)] for o in old]
    return stack([keep[0], new[0]]), *(pd.concat([k, n], ignore_index=True) for k, n in zip(keep[1:], new[1:]))


def frozen(long):
    """The frame as it goes to disk: the five columns, sorted by code, date and variable.

    Labels stay categorical with sorted categories, so the sort is the text order
    and ``to_csv`` writes the same bytes a frame of strings would.
    """
    d = long[["code", "date", "variable", "value", "source"]]
    labels = {c: sorted(map(str, d[c].unique())) for c in ("code", "variable", "source")}
    if not all(isinstance(d[c].dtype, pd.CategoricalDtype) and list(d[c].cat.categories) == cats
               for c, cats in labels.items()):
        d = d.assign(**{c: _recode(d[c], cats) for c, cats in labels.items()})
    keys = [d["code"].cat.codes.to_numpy(), d["date"].to_numpy().astype("int64"), d["variable"].cat.codes.to_numpy()]
    if _strictly_increasing(keys):           # a panel straight from validate_long: already in file order
        return d.reset_index(drop=True)
    return d.sort_values(["code", "date", "variable"], ignore_index=True)


class _Sink:
    """A text file that hashes what is written to it and, optionally, gzips it to disk."""

    def __init__(self, gz=None):
        self.h, self.gz = hashlib.sha256(), gz

    def write(self, text):
        b = text.encode()
        self.h.update(b)
        if self.gz is not None:
            self.gz.write(b)
        return len(text)


def csv_digest(frame):
    """sha256 of the CSV text ``frame`` makes (what 'bit for bit' compares), without holding it."""
    sink = _Sink()
    frame.to_csv(sink, index=False, lineterminator="\n", chunksize=50_000)
    return sink.h.hexdigest()


def content_digest(path):
    """sha256 of a delivered file's content: gzip files are read decompressed, in pieces."""
    h = hashlib.sha256()
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path, frame):
    """Write ``frame`` as CSV —gzip with its clock at zero when ``.gz``: same numbers, same hash.

    Returns the sha256 of the content (before compression), like ``csv_digest``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as raw:
        if path.name.endswith(".gz"):
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz:
                sink = _Sink(gz)
                frame.to_csv(sink, index=False, lineterminator="\n", chunksize=50_000)
        else:
            sink = _Sink(raw)
            frame.to_csv(sink, index=False, lineterminator="\n", chunksize=50_000)
    return sink.h.hexdigest()


def write_block(out, entry, frame, attrs):
    """One refreshed block and its ``.meta.json`` sidecar; fills the entry's hash and rows."""
    write_csv(Path(out) / entry["file"], frame)
    (Path(out) / entry["file"]).with_name(Path(entry["file"]).name.split(".")[0] + ".meta.json").write_text(
        json.dumps(attrs, indent=1, ensure_ascii=False, default=str) + "\n")
    entry.update(sha256=sha256(Path(out) / entry["file"]), rows=len(frame))


def listed(m):
    """Every manifest entry that names a file and its hash, at any depth."""
    for v in m.values():
        if isinstance(v, dict):
            yield from ([v] if {"file", "sha256"} <= set(v) else listed(v))


def new_manifest(old, where, blocks, panels, pwt, extra, version):
    """The delivery's manifest: one entry per file with its hash, and who refreshed what.

    ``where(file)`` is the path of the file to hash; ``blocks`` is the notebook's
    registry (an entry with a ``frame`` was refreshed in this run); ``panels``
    maps a name to (file, frequency, long frame); ``pwt`` maps the two PWT
    blocks to their wide frames; ``extra`` holds ranks, prefer and thresholds.
    Entries of blocks not refreshed are kept from ``old``.
    """
    def entry(file, **more):
        return {"file": file, "sha256": sha256(where(file)), **more}

    fresh = {b: e for b, e in blocks.items() if e["frame"] is not None}
    kept = {k: old.get(k, {}) for k in ("blocks", "pwt", "pwt_labor", "sahm_sources")}
    names = {"Y": "annual", "Q": "quarterly", "M": "monthly"}

    def pwt_entry(b, key):
        if b not in fresh:
            return kept[key]
        e = blocks[b]
        return {**entry(e["file"], countries=pwt[b].code.nunique()), "release": e["release"],
                "doi": e["frame"].attrs.get("doi"), "call": e["call"], "fetched_utc": e["fetched_utc"]}

    sahm = {sid: entry(f"t01_{sid}.csv", url=f"https://fred.stlouisfed.org/series/{sid}",
                       fetched_utc=blocks["sahm"]["fetched_utc"])
            for sid in ("UNRATE", "SAHMCURRENT", "SAHMREALTIME")} if "sahm" in fresh else kept["sahm_sources"]
    return {
        "assembled_utc": datetime.now(timezone.utc).isoformat(),
        "description": old["description"],
        "puremacro_version": version,
        "live_blocks": sorted(fresh),
        "panels": {name: entry(file, countries=p.code.nunique(), variables=p.variable.nunique(),
                               observations=len(p), first=f"{p.date.min():%Y-%m-%d}",
                               last=f"{p.date.max():%Y-%m-%d}", frequency=f)
                   for name, (file, f, p) in panels.items()},
        "blocks": {**kept["blocks"], **{b: {"file": e["file"], "sha256": e["sha256"], "rows": e["rows"],
                                            "call": e["call"], "provider": e["frame"].attrs.get("source"),
                                            "release": e["release"], "fetched_utc": e["fetched_utc"],
                                            "origin": "live"} for b, e in fresh.items()}},
        "dictionary": entry("t01_series.csv"),
        "countries": entry("t01_countries.csv"),
        "provenance": entry("t01_provenance.csv"),
        "seams": entry("t01_seams.csv"),
        "coverage": {names[f]: entry(f"t01_coverage_{names[f]}.csv") for f in "YQM"},
        **extra,
        "pwt": pwt_entry("t01_pwt110", "pwt"),
        "pwt_labor": pwt_entry("t01_pwt110_labor", "pwt_labor"),
        "pwt_dictionary": entry("t01_pwt110_dictionary.csv",
                                note="PWT's variable list, kept from the previous pipeline; not rebuilt here"),
        "sahm_sources": sahm,
    }


def peak_rss_mb():
    """Peak resident memory of this process in MB (NaN where ``resource`` is missing, as in Pyodide)."""
    try:
        import resource
        import sys
    except ImportError:
        return float("nan")
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e6 if sys.platform == "darwin" else 1e3)


def assert_same_rows(a, b, rtol=1e-12):
    """Two ``frozen`` frames hold the same rows: labels and dates exactly, values to ``rtol``.

    ``assert_frame_equal`` takes seconds per million categorical rows; this compares
    the category lists and their codes instead.
    """
    assert list(a.columns) == list(b.columns) and len(a) == len(b), (list(a.columns), len(a), len(b))
    for c in a.columns:
        x, y = a[c], b[c]
        if isinstance(x.dtype, pd.CategoricalDtype):
            assert list(x.cat.categories) == list(y.cat.categories), c
            assert np.array_equal(x.cat.codes.to_numpy(), y.cat.codes.to_numpy()), c
        elif c == "value":
            assert np.allclose(x.to_numpy(float), y.to_numpy(float), rtol=rtol, atol=0), c
        else:
            assert np.array_equal(x.to_numpy(), y.to_numpy()), c


def release_memory():
    """Collect garbage and hand the freed heap pages back to the system.

    The allocators of macOS and glibc keep the pages of freed mid-sized arrays inside
    the process, where they still count as its resident memory (and against the
    iPad's memory limit) until something asks for them back; this asks. A no-op
    where neither call exists (Pyodide, Windows).
    """
    import gc
    import sys
    gc.collect()
    if "pyarrow" in sys.modules:                    # pandas' strings live in Arrow's own pool
        try:
            sys.modules["pyarrow"].default_memory_pool().release_unused()
        except Exception:
            pass
    try:
        import ctypes
        if sys.platform == "darwin":
            ctypes.CDLL("/usr/lib/libSystem.B.dylib").malloc_zone_pressure_relief(None, 0)
        elif sys.platform.startswith("linux"):
            ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (ImportError, OSError, AttributeError):
        pass
