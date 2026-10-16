"""Loader of the checksum-verified OECD ICIO 3-region x 3-sector test fixture.

The fixture aggregates the public OECD ICIO 2023 small table (``manifest.json``
records the source, edition and checksums) and is used by the trade tests that
need a real, small input-output table.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from puremacro.trade.data import RawIOData

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "trade" / "oecd_2023"


def load_oecd_3x3() -> RawIOData:
    """The fixture as :class:`~puremacro.trade.data.RawIOData` (checksum verified)."""
    manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
    path = FIXTURES / "2019_3regions_3sectors.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["fixture_sha256"]:
        raise ValueError("OECD fixture checksum mismatch")
    with np.load(path, allow_pickle=False) as r:
        return RawIOData(countries=list(r["countries"]), sectors=list(r["sectors"]),
                         fd_categories=list(r["fd_categories"]), intermediate_matrix=r["Z"],
                         final_demand_matrix=r["F"], value_added=r["VA"], taxes_less_subsidies=r["TLS"],
                         gross_output=r["Y"], taxes_less_subsidies_fd=r["TLS_fd"],
                         year=int(manifest["year"]), source="OECD_ICIO_2023_regular_aggregated", unit="M_USD",
                         metadata=manifest["source_metadata"])
