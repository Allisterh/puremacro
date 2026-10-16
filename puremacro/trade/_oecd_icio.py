"""Strict native reader for the OECD 2023 regular ICIO CSV schema.

This is deliberately edition-specific: 77 economies, 45 activities, six final
uses and the provider's TLS/VA/OUT rows. No positional harmonized fallback.
"""
from __future__ import annotations

import csv
import hashlib
import io
import os
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

SOURCE_URL = "https://www.oecd.org/en/data/datasets/inter-country-input-output-tables.html"
README_URL = "https://webfs-sti.oecd.org/files/STI-PIE/ICIO/2023/ReadMe_ICIO_small.xlsx"
ARCHIVE_URL = "https://webfs-sti.oecd.org/files/STI-PIE/ICIO/2023/2016-2020_SML.zip"
ARCHIVE_NAME = "2016-2020_SML.zip"
FD_GROUPS = {"C": ("HFCE", "NPISH", "GGFC"), "I": ("GFCF", "INVNT"), "Cx": ("DPABR",)}
FD_SEMANTICS = {
    "C": "final consumption expenditure (households, NPISH, government)",
    "I": ("gross capital formation (gross fixed capital formation plus signed changes in inventories); "
          "receives the foreign-balance closure"),
    "Cx": "residents' direct purchases abroad (DPABR)",
}


def _parse_frame(frame, *, countries, sectors, year, metadata=None):
    from .data import RawIOData, RAW_6_FINAL_DEMAND_CODES

    nodes = [f"{c}_{s}" for c in countries for s in sectors]
    fd = [f"{c}_{f}" for c in countries for f in RAW_6_FINAL_DEMAND_CODES]
    if not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError("Native OECD table contains duplicate row or column labels")
    for actual, expected, axis in ((frame.index, nodes + ["TLS", "VA", "OUT"], "rows"),
                                   (frame.columns, nodes + fd + ["OUT"], "columns")):
        if set(actual) != set(expected):
            missing = sorted(set(expected) - set(actual))[:5]
            extra = sorted(set(actual) - set(expected))[:5]
            raise ValueError(f"Native OECD 2023 {axis} mismatch: missing={missing}, extra={extra}")
    frame = frame.loc[nodes + ["TLS", "VA", "OUT"], nodes + fd + ["OUT"]]
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Native OECD table contains nonfinite values")
    n = len(nodes)
    Z = values[:n, :n].copy()
    F = values[:n, n:-1].copy()
    TLS, VA = values[n, :n].copy(), values[n + 1, :n].copy()
    Y = values[:n, -1].copy()
    if np.any(Z < 0) or np.any(Y < 0):
        raise ValueError("Native OECD intermediate flows and output must be nonnegative")
    if not np.allclose(Y, values[n + 2, :n], atol=1e-4, rtol=1e-12):
        raise ValueError("Native OECD OUT row and column disagree")
    row_error = Z.sum(axis=1) + F.sum(axis=1) - Y
    col_error = Z.sum(axis=0) + TLS + VA - Y
    accounting = {
        "max_abs_sales_minus_reported_output": float(np.max(np.abs(row_error))),
        "max_abs_outlays_minus_reported_output": float(np.max(np.abs(col_error))),
        "sales_minus_reported_output_total": float(row_error.sum()),
        "outlays_minus_reported_output_total": float(col_error.sum()),
        "max_abs_sales_minus_outlays": float(np.max(np.abs(row_error-col_error))),
        "max_relative_sales_outlays_to_world_output": float(np.max(np.abs(row_error-col_error)) / max(1., Y.sum())),
        "negative_final_demand_entries": int(np.sum(F < 0)),
        "negative_value_added_entries": int(np.sum(VA < 0)),
        "zero_output_nodes": int(np.sum(Y == 0)),
        "world_reported_output": float(Y.sum()),
        "world_value_added": float(VA.sum()),
        "world_final_demand_basic_prices": float(F.sum()),
        "unit": "current million USD",
        "note": "Measured before repairs; a valid schema does not imply exact accounting balance.",
    }
    return RawIOData(
        countries=list(countries), sectors=list(sectors), intermediate_matrix=Z,
        final_demand_matrix=F, value_added=VA, taxes_less_subsidies=TLS,
        gross_output=Y, fd_categories=list(RAW_6_FINAL_DEMAND_CODES), year=year,
        source="OECD_ICIO_2023_regular", unit="M_USD",
        taxes_less_subsidies_fd=values[n, n:-1].copy(),
        metadata={**(metadata or {}), "is_synthetic": False, "edition": "2023 regular",
                  "schema": "OECD 2023 regular native labeled CSV",
                  "schema_validation": "exact label sets, uniqueness, finite values, matching OUT margins",
                  "provider_url": SOURCE_URL, "schema_documentation_url": README_URL,
                  "price_basis": "current basic prices; TLS supplied separately",
                  "provider_accounting": accounting},
    )


def read_native(path: str | Path, year: int):
    """Read a provider CSV or select exactly one year member from its ZIP."""
    from .data import OECD_77_COUNTRIES, OECD_45_SECTORS

    if not 1995 <= year <= 2020:
        raise ValueError("OECD 2023 regular coverage is 1995–2020")
    path = Path(path)
    payload = path.read_bytes()
    metadata = {"file_path": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest()}
    if path.suffix.lower() == ".zip":
        with ZipFile(io.BytesIO(payload)) as archive:
            candidates = [n for n in archive.namelist() if Path(n).name == f"{year}_SML.csv"]
            if len(candidates) != 1:
                raise ValueError(f"Expected exactly one {year}_SML.csv member in OECD ZIP")
            metadata.update(archive_sha256=metadata["sha256"], archive_member=candidates[0])
            payload = archive.read(candidates[0])
            metadata["sha256"] = hashlib.sha256(payload).hexdigest()
    else:
        # A native CSV carries its reference year in the provider filename.
        if path.name != f"{year}_SML.csv":
            raise ValueError(f"OECD CSV filename must identify the requested year {year}")
    header = next(csv.reader(io.StringIO(payload.decode("utf-8-sig"))))
    if len(header) != len(set(header)):
        raise ValueError("Native OECD table contains duplicate column labels")
    frame = pd.read_csv(io.BytesIO(payload), index_col=0)
    return _parse_frame(frame, countries=OECD_77_COUNTRIES, sectors=OECD_45_SECTORS,
                        year=year, metadata=metadata)


# The download identifies itself honestly as puremacro; it never imitates a
# browser. The OECD file server sometimes refuses a plain scripted GET, so a
# second attempt asks for the whole file as an open-ended range request. When
# both fail, the error tells the user to download the archive in a browser.
_ARCHIVE_SOURCE_URL = "https://github.com/jalonso1979/puremacro"


def _archive_request_forms() -> tuple[dict, ...]:
    from puremacro import __version__

    user_agent = f"puremacro/{__version__} (+{_ARCHIVE_SOURCE_URL})"
    return (
        {"user_agent": user_agent},
        {"user_agent": user_agent, "headers": {"Range": "bytes=0-"}},
    )


def archive_cache_dir() -> Path:
    """Directory that holds the downloaded OECD archive.

    ``$PUREMACRO_OECD_ICIO_DIR`` when set, else ``~/.cache/puremacro/oecd_icio``.
    """
    env = os.environ.get("PUREMACRO_OECD_ICIO_DIR")
    return Path(env).expanduser() if env else Path.home() / ".cache" / "puremacro" / "oecd_icio"


def download_archive(dest_dir: str | Path | None = None, *, refresh: bool = False,
                     timeout: float = 600.0) -> Path:
    """Download the OECD 2023-edition ``2016-2020_SML.zip`` (about 85 MB) once and return its path.

    The archive is fetched from :data:`ARCHIVE_URL` through
    :func:`puremacro._http.safe_get_bytes` into ``dest_dir`` (default
    :func:`archive_cache_dir`). An existing copy is reused unless
    ``refresh=True``. The bytes are written to a temporary file and renamed, so
    an interrupted download leaves no partial archive. Nothing here checks the
    content: pass the returned path to :func:`puremacro.trade.mrio.check_oecd_source`
    (``year=2019``) to verify the member against the registered MD5.
    """
    from puremacro import _http

    dest = Path(dest_dir) if dest_dir is not None else archive_cache_dir()
    path = dest / ARCHIVE_NAME
    if path.is_file() and not refresh:
        return path
    dest.mkdir(parents=True, exist_ok=True)
    body, errors = None, []
    for request in _archive_request_forms():
        try:
            body = _http.safe_get_bytes(ARCHIVE_URL, timeout=timeout, **request)
            break
        except Exception as exc:  # network failure, HTTP error (a 403 refusal), timeout
            errors.append(f"{type(exc).__name__}: {exc}")
    if body is None:
        raise RuntimeError(
            f"could not download {ARCHIVE_URL} ({'; '.join(errors)}); download it in a browser from "
            f"{SOURCE_URL} and pass the file path instead, or place it at {path}"
        )
    if body[:4] != b"PK\x03\x04":
        raise RuntimeError(f"{ARCHIVE_URL} did not return a ZIP archive (first bytes {body[:16]!r})")
    tmp = path.with_suffix(".zip.part")
    tmp.write_bytes(body)
    tmp.replace(path)
    return path


def condense_final_demand(raw):
    """Map the native final uses to the model's C, I, Cx closure, conserving sums.

    ``C = HFCE + NPISH + GGFC``, ``I = GFCF + INVNT``, ``Cx = DPABR``. The
    solver applies the foreign balance to final-use index 1 (investment).
    The meaning of each slot is recorded in ``metadata["final_use_semantics"]``.
    Passing six OECD categories directly would apply it to NPISH instead.
    Negative inventory changes are preserved in I; nothing is clipped, so a
    cell with ``GFCF + INVNT < 0`` stays negative. Such cells are counted per
    condensed category in ``metadata["negative_final_demand_cells"]``: the
    ``accounting="consistent"`` path rejects any negative final-use cell (the
    native 2019 table has 90 of them in I although every country's I total
    is positive), so a native table needs aggregation or an explicit
    inventory treatment before that path can run.

    A missing ``taxes_less_subsidies_fd`` (synthetic tables) stays ``None``.
    """
    indices = [[list(raw.fd_categories).index(f) for f in group] for group in FD_GROUPS.values()]
    F = raw.F.reshape(raw.M, raw.C, raw.K_F)
    grouped = np.stack([F[:, :, idx].sum(axis=2) for idx in indices], axis=2)
    taxes = None
    if raw.taxes_less_subsidies_fd is not None:
        tfd = np.asarray(raw.taxes_less_subsidies_fd, dtype=float).reshape(raw.C, raw.K_F)
        taxes = np.stack([tfd[:, idx].sum(axis=1) for idx in indices], axis=1).ravel()
    mapping = {k: list(v) for k, v in FD_GROUPS.items()}
    negative = {k: int(np.count_nonzero(grouped[:, :, j] < 0)) for j, k in enumerate(FD_GROUPS)}
    return replace(raw, final_demand_matrix=grouped.reshape(raw.M, -1),
                   fd_categories=list(FD_GROUPS), taxes_less_subsidies_fd=taxes,
                   metadata={**raw.metadata, "source_fd_categories": list(raw.fd_categories),
                             "final_demand_mapping": mapping, "final_use_mapping": mapping,
                             "negative_final_demand_cells": negative,
                             "final_use_semantics": dict(FD_SEMANTICS),
                             "inventory_treatment": "signed INVNT added to GFCF; no clipping"})
