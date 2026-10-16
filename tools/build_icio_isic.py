"""tools/build_icio_isic.py -- the bundled OECD ICIO table: ISIC Rev.4 sections, G20 countries.

Builds ``puremacro/trade/_datafiles/icio_oecd2023_2019_isic_g20.npz``, the table
``puremacro.trade.data.load_icio_data()`` returns by default, from the public
OECD Inter-Country Input-Output tables (2023 edition, reference year 2019,
file ``2019_SML.csv``; https://oe.cd/icio). The recipe, also stated in
``puremacro/trade/_datafiles/SOURCES.md``:

1. Check the source file's MD5 against the registered 2023-edition digest
   (``puremacro.trade.mrio.OECD_ICIO_MD5[2019]``) and read it with the strict
   native reader (77 economies, 45 activities, six final uses).
2. Final uses to three categories: C = HFCE + NPISH + GGFC, I = GFCF + INVNT,
   Cx = DPABR.
3. Countries: the 19 G20 member countries (the European Union and the African
   Union, members as blocs, are not separate economies) keep their own rows
   and columns; the other 57 economies and the OECD's own rest of the world
   are summed into ``ROW``.
4. Sectors: the 45 OECD activities are summed into the 20 ISIC Rev.4 sections
   they belong to (A to T; the OECD tables have no section U).
5. A product row whose final uses contain negative cells (inventory
   draw-downs larger than the other final sales) keeps its total final use:
   the negative cells are set to zero and the positive cells of the same row
   are scaled down to the row's total. A row with a negative total stops the
   build.
6. The library's own table pipeline (``puremacro.trade.data._transaction_table``
   with ``regularize=True``): output set to row sales, the residual net-tax
   closure, value-added floors on near-empty nodes, and value added split 2/3
   labour and 1/3 capital. The split is a modelling convention, not OECD data
   (the OECD tables do not split value added between labour and capital). Layout ``(ns*nc + 3, ns*nc + 3*nc)``: rows are the
   nodes (country-major), net taxes, labour and capital.

Usage (from the repository root)::

    python tools/build_icio_isic.py --download
    python tools/build_icio_isic.py --source /path/to/2019_SML.csv
    python tools/build_icio_isic.py --source /path/to/2016-2020_SML.zip

``--download`` fetches the OECD's ``2016-2020_SML.zip`` (about 85 MB) from
https://webfs-sti.oecd.org/files/STI-PIE/ICIO/2023/ once and keeps it in
``$PUREMACRO_OECD_ICIO_DIR`` (default ``~/.cache/puremacro/oecd_icio``). The
ZIP's ``2019_SML.csv`` member is read. A file whose MD5 is not the registered
one is refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from puremacro.trade.data import (  # noqa: E402
    OECD_77_COUNTRIES,
    RAW_45_SECTOR_CODES,
    RawIOData,
    _transaction_table,
)

RELEASE = "OECD ICIO 2023 edition"
YEAR = 2019
SOURCE_FILE = f"{YEAR}_SML.csv"
OUTPUT_NAME = "icio_oecd2023_2019_isic_g20.npz"
OUTPUT = ROOT / "puremacro" / "trade" / "_datafiles" / OUTPUT_NAME
CITATION = "OECD (2023), OECD Inter-Country Input-Output Tables, http://oe.cd/icio"
UNIT = "current million USD"


def _registered_md5() -> str:
    from puremacro.trade.mrio import OECD_ICIO_MD5

    return OECD_ICIO_MD5[YEAR]


SOURCE_MD5 = _registered_md5()

G20_COUNTRIES: tuple[str, ...] = (
    "ARG", "AUS", "BRA", "CAN", "CHN", "DEU", "FRA", "GBR", "IDN", "IND",
    "ITA", "JPN", "KOR", "MEX", "RUS", "SAU", "TUR", "USA", "ZAF",
)
"""The 19 G20 member countries (ISO 3166-1 alpha-3), alphabetical."""

COUNTRY_CODES: tuple[str, ...] = G20_COUNTRIES + ("ROW",)

SECTOR_CODES: tuple[str, ...] = tuple("ABCDEFGHIJKLMNOPQRST")

SECTOR_NAMES: dict[str, str] = {
    "A": "Agriculture, forestry and fishing",
    "B": "Mining and quarrying",
    "C": "Manufacturing",
    "D": "Electricity, gas, steam and air conditioning supply",
    "E": "Water supply; sewerage, waste management and remediation activities",
    "F": "Construction",
    "G": "Wholesale and retail trade; repair of motor vehicles and motorcycles",
    "H": "Transportation and storage",
    "I": "Accommodation and food service activities",
    "J": "Information and communication",
    "K": "Financial and insurance activities",
    "L": "Real estate activities",
    "M": "Professional, scientific and technical activities",
    "N": "Administrative and support service activities",
    "O": "Public administration and defence; compulsory social security",
    "P": "Education",
    "Q": "Human health and social work activities",
    "R": "Arts, entertainment and recreation",
    "S": "Other service activities",
    "T": ("Activities of households as employers; undifferentiated goods- and "
          "services-producing activities of households for own use"),
}
"""ISIC Rev.4 section titles (United Nations, ISIC Rev.4, 2008)."""

SECTION_OF: dict[str, str] = {code: code[0] for code in RAW_45_SECTOR_CODES}
"""ISIC Rev.4 section of each OECD ICIO activity: an OECD code starts with its section letter."""

FINAL_USE_CODES: tuple[str, ...] = ("C", "I", "Cx")


def _group_index(fine: list[str], coarse: tuple[str, ...], group_of: dict[str, str]) -> np.ndarray:
    position = {code: k for k, code in enumerate(coarse)}
    return np.array([position[group_of[code]] for code in fine], dtype=np.intp)


def aggregate(raw: RawIOData) -> RawIOData:
    """Sum a native OECD table (final uses already condensed) into G20 + ROW and ISIC sections.

    Every total (intermediate flows, final uses, value added, net taxes on
    intermediate and final uses, output) is conserved.
    """
    if list(raw.countries) != list(OECD_77_COUNTRIES) or list(raw.sectors) != list(RAW_45_SECTOR_CODES):
        raise ValueError("aggregate expects the native OECD ICIO 2023 rosters (77 economies, 45 activities)")
    VA = np.asarray(raw.value_added, dtype=float)
    if VA.ndim != 1:
        raise ValueError("aggregate expects one value-added row")
    country_group = {c: (c if c in G20_COUNTRIES else "ROW") for c in raw.countries}
    cidx = _group_index(list(raw.countries), COUNTRY_CODES, country_group)
    sidx = _group_index(list(raw.sectors), SECTOR_CODES, SECTION_OF)
    nc, ns, nfd = len(COUNTRY_CODES), len(SECTOR_CODES), raw.K_F
    node = (cidx[:, None] * ns + sidx[None, :]).ravel()          # fine node -> coarse node
    m = nc * ns

    def rows(a: np.ndarray) -> np.ndarray:
        out = np.zeros((m,) + a.shape[1:])
        np.add.at(out, node, a)
        return out

    Z = rows(np.asarray(raw.Z, dtype=float))
    Z = rows(Z.T).T
    F = rows(np.asarray(raw.F, dtype=float).reshape(raw.M, raw.C * nfd))
    F3 = np.zeros((m, nc, nfd))
    np.add.at(F3, (slice(None), cidx), F.reshape(m, raw.C, nfd))   # destination countries
    tfd = None
    if raw.taxes_less_subsidies_fd is not None:
        tfd = np.zeros((nc, nfd))
        np.add.at(tfd, cidx, np.asarray(raw.taxes_less_subsidies_fd, dtype=float).reshape(raw.C, nfd))
        tfd = tfd.ravel()
    return replace(
        raw, countries=list(COUNTRY_CODES), sectors=list(SECTOR_CODES),
        intermediate_matrix=Z, final_demand_matrix=F3.reshape(m, nc * nfd),
        value_added=rows(VA), taxes_less_subsidies=rows(np.asarray(raw.TLS, dtype=float)),
        gross_output=rows(np.asarray(raw.Y, dtype=float)), taxes_less_subsidies_fd=tfd,
        metadata={**raw.metadata, "country_rule": "G20 member countries + ROW",
                  "sector_grouping": "ISIC Rev.4 sections A-T"},
    )


def net_negative_final_uses(F: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Set negative final-use cells to zero and scale the row's positive cells to its total.

    Each row keeps its total final use exactly, so output (row sales) is
    unchanged. A row whose final uses sum to a negative number is refused.
    """
    F = np.array(F, dtype=float, copy=True)
    negative_rows = np.flatnonzero((F < 0).any(axis=1))
    report = {"rows": int(negative_rows.size), "cells": int(np.count_nonzero(F < 0)),
              "amount": float(-F[F < 0].sum())}
    for r in negative_rows:
        total = F[r].sum()
        if total < 0:
            raise ValueError(f"row {r} has a negative total final use ({total:.6g}); it cannot be netted")
        positive = np.clip(F[r], 0.0, None)
        F[r] = positive * (total / positive.sum()) if positive.sum() > 0 else 0.0
    return F, report


def build_table(native: RawIOData) -> tuple[np.ndarray, dict[str, Any]]:
    """Native OECD table (six final uses) -> the processed bundled table and a build record."""
    from puremacro.trade._oecd_icio import condense_final_demand

    coarse = aggregate(condense_final_demand(native))
    F, netted = net_negative_final_uses(coarse.F)
    coarse = replace(coarse, final_demand_matrix=F)
    built = _transaction_table(coarse, regularize=True)
    matrix = built["matrix"]
    n = coarse.M
    info = {
        "release": RELEASE, "year": YEAR, "citation": CITATION, "unit": UNIT,
        "country_codes": list(COUNTRY_CODES), "sector_codes": list(SECTOR_CODES),
        "fd_codes": list(FINAL_USE_CODES),
        "country_rule": ("the 19 G20 member countries; the other 57 economies and the OECD rest of "
                         "the world summed into ROW"),
        "sector_grouping": "ISIC Rev.4 sections A-T (45 OECD activities summed by section)",
        "final_uses": {"C": "HFCE + NPISH + GGFC", "I": "GFCF + INVNT", "Cx": "DPABR"},
        "value_added_split": "2/3 labour, 1/3 capital",
        "negative_final_uses_netted": netted,
        "regularization": {
            "va_floored_nodes": [d["node"] for d in built["regularization"].get("va_floored_nodes", [])],
            "phantom_nodes": list(built["regularization"].get("phantom_nodes", [])),
        },
        "totals": {
            "world_value_added": float(matrix[n + 1:n + 3, :n].sum()),
            "world_gross_output": float(matrix[:n].sum()),
            "world_net_taxes_intermediate": float(matrix[n, :n].sum()),
        },
        "source_metadata": {k: native.metadata[k] for k in ("sha256", "archive_member") if k in native.metadata},
    }
    return matrix, info


def write_table(path: Path, matrix: np.ndarray, info: dict[str, Any]) -> None:
    """Write the table, its codes and the build record (JSON) to a compressed ``.npz``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, data=np.ascontiguousarray(matrix, dtype=np.float64),
        country_codes=np.array(info["country_codes"]), sector_codes=np.array(info["sector_codes"]),
        fd_codes=np.array(info["fd_codes"]), metadata_json=np.array(json.dumps(info, sort_keys=True)),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", help=f"{SOURCE_FILE} or the OECD 2016-2020_SML.zip already on disk")
    group.add_argument("--download", action="store_true",
                       help="download the OECD 2016-2020_SML.zip from the OECD site (cached in "
                            "$PUREMACRO_OECD_ICIO_DIR or ~/.cache/puremacro/oecd_icio)")
    parser.add_argument("--output", default=str(OUTPUT), help=f"output .npz (default {OUTPUT})")
    args = parser.parse_args(argv)

    from puremacro.trade._oecd_icio import download_archive, read_native
    from puremacro.trade.mrio import check_oecd_source

    source = download_archive() if args.download else Path(args.source)
    record = check_oecd_source(source, year=YEAR)
    if record.status != "whitelisted" or record.md5 != SOURCE_MD5:
        raise SystemExit(f"{source}: md5 {record.md5} is not the registered OECD {RELEASE} {SOURCE_FILE} "
                         f"(md5 {SOURCE_MD5}); refusing to build")
    native = read_native(source, YEAR)
    matrix, info = build_table(native)
    info["source_md5"] = record.md5
    output = Path(args.output)
    write_table(output, matrix, info)
    summary = {"output": str(output), "shape": list(matrix.shape), "file_sha256": _sha256(output),
               "file_bytes": output.stat().st_size, "source_md5": record.md5, **info["totals"],
               "negative_final_uses_netted": info["negative_final_uses_netted"],
               "regularization": info["regularization"]}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
