"""Fama-French factors and the 25 size/book-to-market portfolios.

Downloads the CSV zips of Kenneth French's data library on demand. The data
are "Copyright Eugene F. Fama and Kenneth R. French" with no redistribution
licence, so nothing is bundled with puremacro: the raw bytes are cached only
in the user's HTTP cache (``~/.cache/puremacro/http`` or
``$PUREMACRO_HTTP_CACHE_DIR``) and nothing is written inside the package.

Data library: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
"""
from __future__ import annotations

import io
import re
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .._http import safe_get_bytes_cached

__all__ = ["fetch", "DATASETS"]

_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

#: dataset -> zip file name
DATASETS = {
    "ff3": "F-F_Research_Data_Factors_CSV.zip",
    "ff5": "F-F_Research_Data_5_Factors_2x3_CSV.zip",
    "mom": "F-F_Momentum_Factor_CSV.zip",
    "25_size_bm": "25_Portfolios_5x5_CSV.zip",
}
_FREQS = ("M", "A")
_COPYRIGHT = "Eugene F. Fama and Kenneth R. French"


def _blocks(text: str) -> list[tuple[str, list[str], list[list[str]]]]:
    """Split a data-library CSV into (title, columns, rows) blocks."""
    out = []
    title_lines: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith(",") and i + 1 < len(lines) and re.match(r"^\s*\d{4}", lines[i + 1]):
            cols = [c.strip() for c in ln.split(",")[1:]]
            rows = []
            i += 1
            while i < len(lines) and re.match(r"^\s*\d{4}(\d{2})?\s*,", lines[i]):
                rows.append([f.strip() for f in lines[i].split(",")])
                i += 1
            out.append((" ".join(t.strip() for t in title_lines), cols, rows))
            title_lines = []
            continue
        # the title is the run of non-blank lines just above the header
        title_lines = title_lines + [ln] if ln.strip() else []
        i += 1
    return out


def _parse(raw: bytes, frequency: str, value_weighted: bool) -> pd.DataFrame:
    zf = zipfile.ZipFile(io.BytesIO(raw))
    name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
    text = zf.read(name).decode("latin-1")
    width = 6 if frequency == "M" else 4
    for title, cols, rows in _blocks(text):
        if not rows or len(rows[0][0]) != width:
            continue
        if value_weighted and "value weighted" not in title.lower():
            raise RuntimeError(
                "kenfrench_factors: the first block of the 25-portfolio file is not "
                f"'Value Weighted' (found {title!r}); the file layout changed")
        stamps = [r[0] for r in rows]
        vals = np.array([[float(v) for v in r[1:len(cols) + 1]] for r in rows])
        vals[np.isclose(vals, -99.99) | np.isclose(vals, -999.0)] = np.nan
        if frequency == "M":
            idx = pd.to_datetime(stamps, format="%Y%m")
        else:
            idx = pd.to_datetime(stamps, format="%Y")
        return pd.DataFrame(vals / 100.0, index=pd.DatetimeIndex(idx, name="date"),
                            columns=cols)
    raise RuntimeError(f"kenfrench_factors: no {'monthly' if frequency == 'M' else 'annual'} "
                       "block found; the file layout changed")


def fetch(dataset: str = "ff3", frequency: str = "M", refresh: bool = False) -> pd.DataFrame:
    """Fama-French factors or the 25 size/book-to-market portfolios.

    Parameters
    ----------
    dataset : {"ff3", "ff5", "mom", "25_size_bm"}
        Three factors, five factors (2x3), momentum, or the 25 portfolios
        formed on size and book-to-market (value-weighted returns only).
    frequency : {"M", "A"}
        Monthly or annual (January-December) block.
    refresh : bool
        Re-download instead of using the user HTTP cache.

    Returns
    -------
    pandas.DataFrame
        Wide frame indexed by ``date`` (period start), returns in decimals
        (the files are in percent), missing-data sentinels ``-99.99`` /
        ``-999`` as ``NaN``, columns as published plus ``"RF"`` (taken from
        the three-factor file for ``mom`` and ``25_size_bm``). ``attrs``:
        ``source``, ``url``, ``copyright``, ``fetched_at``.
    """
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {tuple(DATASETS)}; got {dataset!r}")
    if frequency not in _FREQS:
        raise ValueError(f"frequency must be one of {_FREQS}; got {frequency!r}")
    url = _BASE + DATASETS[dataset]
    df = _parse(safe_get_bytes_cached(url, refresh=refresh), frequency,
                value_weighted=dataset == "25_size_bm")
    if "RF" not in df.columns:
        ff3 = _parse(safe_get_bytes_cached(_BASE + DATASETS["ff3"], refresh=refresh),
                     frequency, value_weighted=False)
        df["RF"] = ff3["RF"].reindex(df.index)
    df.attrs = {
        "source": "Kenneth R. French data library",
        "url": url,
        "copyright": _COPYRIGHT,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return df
