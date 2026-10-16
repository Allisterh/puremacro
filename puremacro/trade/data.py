"""Input-output data for the trade model: registries, containers, loaders and synthetic tables.

This module holds the identifier registries of the supported multi-regional
input-output (MRIO) providers (OECD ICIO, FIGARO, EXIOBASE, WIOD, Eora), the
:class:`RawIOData` and :class:`ICIOData` containers, harmonized loaders that
calibrate a provider table, a loader for processed tables of your own
(:func:`load_icio_data`) and two seeded synthetic generators:
:func:`generate_synthetic_mrio` (a provider's layout) and
:func:`synthetic_icio_table` (the processed layout ``calibrate_trade_model``
reads, at any number of countries and sectors).

The bundled table (``load_icio_data()``) aggregates the OECD Inter-Country
Input-Output tables (2023 edition, 2019) to the 19 G20 member countries plus a
rest of the world and the 20 ISIC Rev.4 sections; its terms, attribution and
recipe are in ``_datafiles/SOURCES.md``. For other years, countries or
sectors read a provider release with :func:`load_oecd_icio_granular` or the
other loaders.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import os
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from puremacro.trade._results import TradeCalibrationResult
from puremacro.trade.regularize import regularize_mrio_table

# ---------------------------------------------------------------------------
# Code registries
# ---------------------------------------------------------------------------

# OECD ICIO 2023 edition: 76 economies (sorted by ISO-3 code) and the rest of the world (ROW, last).
OECD_77_COUNTRIES: tuple[str, ...] = (
    "ARG", "AUS", "AUT", "BEL", "BGD", "BGR", "BLR", "BRA", "BRN", "CAN",
    "CHE", "CHL", "CHN", "CIV", "CMR", "COL", "CRI", "CYP", "CZE", "DEU",
    "DNK", "EGY", "ESP", "EST", "FIN", "FRA", "GBR", "GRC", "HKG", "HRV",
    "HUN", "IDN", "IND", "IRL", "ISL", "ISR", "ITA", "JOR", "JPN", "KAZ",
    "KHM", "KOR", "LAO", "LTU", "LUX", "LVA", "MAR", "MEX", "MLT", "MMR",
    "MYS", "NGA", "NLD", "NOR", "NZL", "PAK", "PER", "PHL", "POL", "PRT",
    "ROU", "RUS", "SAU", "SEN", "SGP", "SVK", "SVN", "SWE", "THA", "TUN",
    "TUR", "TWN", "UKR", "USA", "VNM", "ZAF", "ROW",
)

EU_COUNTRY_CODES: tuple[str, ...] = (
    "AUT", "BEL", "BGR", "CYP", "CZE", "DEU", "DNK", "ESP", "EST", "FIN",
    "FRA", "GRC", "HRV", "HUN", "IRL", "ITA", "LTU", "LUX", "LVA", "MLT",
    "NLD", "POL", "PRT", "ROU", "SVK", "SVN", "SWE",
)

RAW_45_SECTOR_CODES: tuple[str, ...] = (
    "A01_02", "A03", "B05_06", "B07_08", "B09",
    "C10T12", "C13T15", "C16", "C17_18", "C19",
    "C20", "C21", "C22", "C23", "C24",
    "C25", "C26", "C27", "C28", "C29",
    "C30", "C31T33", "D", "E", "F",
    "G", "H49", "H50", "H51", "H52",
    "H53", "I", "J58T60", "J61", "J62_63",
    "K", "L", "M", "N", "O",
    "P", "Q", "R", "S", "T",
)

RAW_45_SECTOR_NAMES: dict[str, str] = {
    "A01_02": "Agriculture, hunting, forestry",
    "A03": "Fishing and aquaculture",
    "B05_06": "Mining and quarrying, energy producing products",
    "B07_08": "Mining and quarrying, non-energy producing products",
    "B09": "Mining support service activities",
    "C10T12": "Food products, beverages and tobacco",
    "C13T15": "Textiles, textile products, leather and footwear",
    "C16": "Wood and products of wood and cork",
    "C17_18": "Paper products and printing",
    "C19": "Coke and refined petroleum products",
    "C20": "Chemical and chemical products",
    "C21": "Pharmaceuticals, medicinal chemical and botanical products",
    "C22": "Rubber and plastics products",
    "C23": "Other non-metallic mineral products",
    "C24": "Basic metals",
    "C25": "Fabricated metal products",
    "C26": "Computer, electronic and optical equipment",
    "C27": "Electrical equipment",
    "C28": "Machinery and equipment, nec",
    "C29": "Motor vehicles, trailers and semi-trailers",
    "C30": "Other transport equipment",
    "C31T33": "Manufacturing nec; repair and installation of machinery and equipment",
    "D": "Electricity, gas, steam and air conditioning supply",
    "E": "Water supply; sewerage, waste management and remediation activities",
    "F": "Construction",
    "G": "Wholesale and retail trade; repair of motor vehicles",
    "H49": "Land transport and transport via pipelines",
    "H50": "Water transport",
    "H51": "Air transport",
    "H52": "Warehousing and support activities for transportation",
    "H53": "Postal and courier activities",
    "I": "Accommodation and food service activities",
    "J58T60": "Publishing, audiovisual and broadcasting activities",
    "J61": "Telecommunications",
    "J62_63": "IT and other information services",
    "K": "Financial and insurance activities",
    "L": "Real estate activities",
    "M": "Professional, scientific and technical activities",
    "N": "Administrative and support services",
    "O": "Public administration and defence; compulsory social security",
    "P": "Education",
    "Q": "Human health and social work activities",
    "R": "Arts, entertainment and recreation",
    "S": "Other service activities",
    "T": "Activities of households as employers; undifferentiated goods- and services-producing activities of households for own use",
}

CANONICAL_FINAL_DEMAND_CODES: tuple[str, ...] = (
    "C", "I", "Cx",
)

RAW_6_FINAL_DEMAND_CODES: tuple[str, ...] = (
    "HFCE", "NPISH", "GGFC", "GFCF", "INVNT", "DPABR",
)

# ---------------------------------------------------------------------------
# Authentic Database Rosters (FIGARO, EXIOBASE, WIOD, Eora, OECD)
# ---------------------------------------------------------------------------

FIGARO_COUNTRIES: tuple[str, ...] = (
    "AUT", "BEL", "BGR", "CYP", "CZE", "DEU", "DNK", "ESP", "EST", "FIN",
    "FRA", "GRC", "HRV", "HUN", "IRL", "ITA", "LTU", "LUX", "LVA", "MLT",
    "NLD", "POL", "PRT", "ROU", "SVK", "SVN", "SWE",
    "ARG", "AUS", "BRA", "CAN", "CHE", "CHN", "GBR", "IDN", "IND", "JPN",
    "KOR", "MEX", "NOR", "RUS", "SAU", "TUR", "USA", "ZAF", "W2",
)

FIGARO_64_SECTORS: tuple[str, ...] = (
    "A01", "A02", "A03", "B", "C10-C12", "C13-C15", "C16", "C17", "C18",
    "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26", "C27", "C28",
    "C29", "C30", "C31_C32", "C33", "D35", "E36", "E37-E39", "F", "G45",
    "G46", "G47", "H49", "H50", "H51", "H52", "H53", "I", "J58", "J59_J60",
    "J61", "J62_J63", "K64", "K65", "K66", "L68", "M69_M70", "M71",
    "M72", "M73", "M74_M75", "N77", "N78", "N79", "N80-N82", "O84", "P85",
    "Q86", "Q87_Q88", "R90-R92", "R93", "S94", "S95", "S96", "T", "U",
)

FIGARO_FD_CATEGORIES: tuple[str, ...] = (
    "P3_S14", "P3_S15", "P3_S13", "P51G", "P5M",
)
# Harmonized semantic order (households, NPISH, government, GFCF, inventories and
# valuables), aligned with the other rosters and the synthetic generator's
# positional weights. It is NOT the Eurostat file order (P3_S13, P3_S14, P3_S15,
# P51G, P5M): ``load_figaro`` reads final-use columns by their COUNTRY_CODE
# labels and accepts either order.

# Mapping of each provider roster onto the calibration's (C, I, Cx) closure.
# Index 1 (I) receives the foreign-balance closure in ``calibrate_trade_model``,
# so it holds gross fixed capital formation; final consumption of households,
# NPISH and government goes to C, and the remaining uses (changes in
# inventories and valuables, EXIOBASE's export column) go to Cx.
#
# Cx therefore does NOT mean what it means in the OECD mapping
# (``_oecd_icio.FD_GROUPS``: I = GFCF + INVNT, Cx = DPABR, residents' direct
# purchases abroad), which only OECD publishes. The OECD-like alternative
# (inventories and valuables in I) is not used because it would leave Cx
# identically zero in every non-OECD table. The meaning of each slot is recorded in
# ``metadata["final_use_semantics"]``; bridges that read ``("C", "I", "Cx")``
# as (C, G, X) must consult ``metadata["final_use_mapping"]`` first.
_PROVIDER_FINAL_USE_MAPPINGS: dict[str, dict[str, tuple[str, ...]]] = {
    "figaro": {"C": ("P3_S14", "P3_S15", "P3_S13"), "I": ("P51G",), "Cx": ("P5M",)},
    "exiobase": {"C": ("HFCE", "NPISH", "GGFC"), "I": ("GFCF",), "Cx": ("INVNT", "VALUABLES", "EXPORT")},
    "wiod": {"C": ("CONS_h", "CONS_np", "CONS_g"), "I": ("GFCF",), "Cx": ("INVT",)},
    "eora": {"C": ("HFCE", "NPISH", "GGFC"), "I": ("GFCF",), "Cx": ("INVNT", "ACQ_VAL")},
}

_PROVIDER_FINAL_USE_SEMANTICS: dict[str, dict[str, str]] = {
    provider: {
        "C": "final consumption expenditure (households, NPISH, government)",
        "I": "gross fixed capital formation; receives the foreign-balance closure",
        "Cx": ("signed changes in inventories and valuables"
               + (" plus unallocated exports ('Exports: Total (fob)')" if provider == "exiobase" else "")
               + "; NOT residents' direct purchases abroad (the OECD meaning of Cx)"),
    }
    for provider in _PROVIDER_FINAL_USE_MAPPINGS
}

_INVESTMENT_CODES = frozenset({"I", "G", "GFCF", "P51G", "P5", "GCF"})
"""Final-use codes accepted at index 1, where ``calibrate_trade_model`` adds the foreign balance."""

EXIOBASE_COUNTRIES: tuple[str, ...] = (
    "AUT", "BEL", "BGR", "CYP", "CZE", "DEU", "DNK", "ESP", "EST", "FIN",
    "FRA", "GBR", "GRC", "HRV", "HUN", "IRL", "ITA", "LTU", "LUX", "LVA",
    "MLT", "NLD", "POL", "PRT", "ROU", "SVK", "SVN", "SWE",
    "AUS", "BRA", "CAN", "CHE", "CHN", "IDN", "IND", "JPN", "KOR", "MEX",
    "NOR", "RUS", "TUR", "TWN", "USA", "ZAF",
    "WA", "WE", "WF", "WL", "WM",
)

EXIOBASE_163_SECTORS: tuple[str, ...] = tuple(f"i{i:03d}" for i in range(1, 164))
EXIOBASE_200_SECTORS: tuple[str, ...] = tuple(f"p{i:03d}" for i in range(1, 201))
EXIOBASE_FD_CATEGORIES: tuple[str, ...] = (
    "HFCE", "NPISH", "GGFC", "GFCF", "INVNT", "VALUABLES", "EXPORT",
)

WIOD_44_COUNTRIES: tuple[str, ...] = (
    "AUS", "AUT", "BEL", "BGR", "BRA", "CAN", "CHE", "CHN", "CYP", "CZE",
    "DEU", "DNK", "ESP", "EST", "FIN", "FRA", "GBR", "GRC", "HRV", "HUN",
    "IDN", "IND", "IRL", "ITA", "JPN", "KOR", "LTU", "LUX", "LVA", "MEX",
    "MLT", "NLD", "NOR", "POL", "PRT", "ROU", "RUS", "SVK", "SVN", "SWE",
    "TUR", "TWN", "USA", "ROW",
)

WIOD_56_SECTORS: tuple[str, ...] = tuple(f"S{i:02d}" for i in range(1, 57))
WIOD_FD_CATEGORIES: tuple[str, ...] = (
    "CONS_h", "CONS_np", "CONS_g", "GFCF", "INVT",
)

EORA_189_COUNTRIES: tuple[str, ...] = (
    "AFG", "ALB", "DZA", "AND", "AGO", "ATG", "ARG", "ARM", "ABW", "AUS",
    "AUT", "AZE", "BHS", "BHR", "BGD", "BRB", "BLR", "BEL", "BLZ", "BEN",
    "BMU", "BTN", "BOL", "BIH", "BWA", "BRA", "VGB", "BRN", "BGR", "BFA",
    "BDI", "KHM", "CMR", "CAN", "CPV", "CYM", "CAF", "TCD", "CHL", "CHN",
    "COL", "COG", "CRI", "CIV", "HRV", "CUB", "CYP", "CZE", "DNK", "DJI",
    "DOM", "ECU", "EGY", "SLV", "ERI", "EST", "ETH", "FJI", "FIN", "FRA",
    "PYF", "GAB", "GMB", "GEO", "DEU", "GHA", "GRC", "GRL", "GTM", "GIN",
    "GUY", "HTI", "HND", "HKG", "HUN", "ISL", "IND", "IDN", "IRN", "IRQ",
    "IRL", "ISR", "ITA", "JAM", "JPN", "JOR", "KAZ", "KEN", "KWT", "KGZ",
    "LAO", "LVA", "LBN", "LSO", "LBR", "LBY", "LIE", "LTU", "LUX", "MAC",
    "MDG", "MWI", "MYS", "MDV", "MLI", "MLT", "MRT", "MUS", "MEX", "MDA",
    "MCO", "MNG", "MNE", "MAR", "MOZ", "MMR", "NAM", "NPL", "NLD", "ANT",
    "NCL", "NZL", "NIC", "NER", "NGA", "PRK", "MKD", "NOR", "OMN", "PAK",
    "PAN", "PNG", "PRY", "PER", "PHL", "POL", "PRT", "PSE", "QAT", "KOR",
    "ROU", "RUS", "RWA", "WSM", "SMR", "STP", "SAU", "SEN", "SRB", "SYC",
    "SLE", "SGP", "SVK", "SVN", "SOM", "ZAF", "SSD", "ESP", "LKA", "SDN",
    "SUD", "SUR", "SWZ", "SWE", "CHE", "SYR", "TWN", "TJK", "TZA", "THA",
    "TGO", "TTO", "TUN", "TUR", "TKM", "UGA", "UKR", "ARE", "GBR", "USA",
    "URY", "UZB", "VUT", "VEN", "VNM", "YEM", "ZMB", "ZWE", "ROW",
)

EORA_26_SECTORS: tuple[str, ...] = tuple(f"SEC{i:02d}" for i in range(1, 27))
EORA_FD_CATEGORIES: tuple[str, ...] = (
    "HFCE", "NPISH", "GGFC", "GFCF", "INVNT", "ACQ_VAL",
)

OECD_45_SECTORS: tuple[str, ...] = RAW_45_SECTOR_CODES
OECD_FD_CATEGORIES: tuple[str, ...] = RAW_6_FINAL_DEMAND_CODES


@dataclass
class RawIOData:
    """Container for unaggregated/raw multi-country input-output transaction matrices."""

    countries: Sequence[str]
    sectors: Sequence[str]
    intermediate_matrix: np.ndarray  # (M, M)
    final_demand_matrix: np.ndarray  # (M, C * K_F) or (M, C, K_F)
    value_added: np.ndarray  # (M,) or (K_VA, M)
    taxes_less_subsidies: np.ndarray  # (M,)
    gross_output: np.ndarray  # (M,)
    fd_categories: Sequence[str]
    year: int = 0
    source: str = ""
    unit: str = "M_USD"
    taxes_less_subsidies_fd: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def C(self) -> int:
        return len(self.countries)

    @property
    def S(self) -> int:
        return len(self.sectors)

    @property
    def M(self) -> int:
        return self.C * self.S

    @property
    def K_F(self) -> int:
        return len(self.fd_categories)

    @property
    def Z(self) -> np.ndarray:
        return self.intermediate_matrix

    @property
    def F(self) -> np.ndarray:
        return self.final_demand_matrix

    @property
    def VA(self) -> np.ndarray:
        return self.value_added

    @property
    def TLS(self) -> np.ndarray:
        return self.taxes_less_subsidies

    @property
    def Y(self) -> np.ndarray:
        return self.gross_output



def get_eu_country_codes() -> list[str]:
    """Return the list of 27 European Union (EU-27) member state ISO-3 country codes."""
    return list(EU_COUNTRY_CODES)


def get_raw_45_sector_codes() -> list[str]:
    """Return the canonical list of 45 unaggregated OECD ICIO sector codes."""
    return list(RAW_45_SECTOR_CODES)


def get_raw_45_sector_names() -> dict[str, str]:
    """Return descriptive mapping from 45 unaggregated sector codes to full names."""
    return dict(RAW_45_SECTOR_NAMES)


def get_final_demand_codes() -> list[str]:
    """Return the canonical list of 3 final demand category codes (['C', 'I', 'Cx'])."""
    return list(CANONICAL_FINAL_DEMAND_CODES)


def get_raw_final_demand_codes() -> list[str]:
    """Return the canonical list of 6 raw OECD ICIO final demand category codes."""
    return list(RAW_6_FINAL_DEMAND_CODES)


# ---------------------------------------------------------------------------
# Structured ICIO Container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ICIOData:
    """Structured container for a processed ICIO transaction table.

    The table has shape ``(n_ind + 3, n_ind + n_fd)`` with ``n_ind = nc * ns``
    country-sector nodes (country-major) and ``n_fd = nc * nfd`` final-use
    columns; its rows after the nodes are net taxes, labor and capital.

    Attributes
    ----------
    matrix : np.ndarray, shape (nc*ns + 3, nc*ns + nc*nfd)
        Full float64 transaction table.
    country_codes : tuple[str, ...]
        The ``nc`` country labels, in table order.
    sector_codes : tuple[str, ...]
        The ``ns`` sector labels, in table order.
    fd_codes : tuple[str, ...]
        The ``nfd`` final-use labels, by default ``('C', 'I', 'Cx')``.
    metadata : dict
        Provenance and processing record (source path). Excluded from
        equality comparisons.
    """

    matrix: np.ndarray
    country_codes: tuple[str, ...]
    sector_codes: tuple[str, ...]
    fd_codes: tuple[str, ...] = CANONICAL_FINAL_DEMAND_CODES
    metadata: dict = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.matrix, np.ndarray):
            raise TypeError(f"matrix must be np.ndarray, got {type(self.matrix).__name__}")
        if self.matrix.ndim != 2:
            raise ValueError(f"matrix must be 2-dimensional, got shape {self.matrix.shape}")
        if not isinstance(self.country_codes, tuple):
            object.__setattr__(self, "country_codes", tuple(self.country_codes))
        if not isinstance(self.sector_codes, tuple):
            object.__setattr__(self, "sector_codes", tuple(self.sector_codes))
        if not isinstance(self.fd_codes, tuple):
            object.__setattr__(self, "fd_codes", tuple(self.fd_codes))

    @property
    def intermediate_matrix(self) -> np.ndarray:
        """(n_ind, n_ind) intermediate transactions Z(i, j)."""
        n_ind = len(self.country_codes) * len(self.sector_codes)
        return self.matrix[:n_ind, :n_ind]

    @property
    def final_demand_matrix(self) -> np.ndarray:
        """(n_ind, n_fd) final demand deliveries FD(i, j)."""
        n_ind = len(self.country_codes) * len(self.sector_codes)
        n_fd = len(self.country_codes) * len(self.fd_codes)
        return self.matrix[:n_ind, n_ind:n_ind + n_fd]

    @property
    def net_taxes_intermediate(self) -> np.ndarray:
        """(n_ind,) net production taxes of the country-sector nodes."""
        n_ind = len(self.country_codes) * len(self.sector_codes)
        return self.matrix[n_ind, :n_ind]

    @property
    def net_taxes_final_demand(self) -> np.ndarray:
        """(n_fd,) net taxes on final demand by country and category."""
        n_ind = len(self.country_codes) * len(self.sector_codes)
        n_fd = len(self.country_codes) * len(self.fd_codes)
        return self.matrix[n_ind, n_ind:n_ind + n_fd]

    @property
    def labor_va(self) -> np.ndarray:
        """(n_ind,) labor compensation of the country-sector nodes."""
        n_ind = len(self.country_codes) * len(self.sector_codes)
        return self.matrix[n_ind + 1, :n_ind]

    @property
    def capital_va(self) -> np.ndarray:
        """(n_ind,) gross capital return of the country-sector nodes."""
        n_ind = len(self.country_codes) * len(self.sector_codes)
        return self.matrix[n_ind + 2, :n_ind]

    @property
    def gross_output(self) -> np.ndarray:
        """(n_ind,) gross output y computed as column outlay sums."""
        n_ind = len(self.country_codes) * len(self.sector_codes)
        n_rows = n_ind + 3
        return np.sum(self.matrix[:n_rows, :n_ind], axis=0)


# ---------------------------------------------------------------------------
# Data Loading Entrypoints
# ---------------------------------------------------------------------------

def _load_raw_table(file_path: Path) -> np.ndarray:
    """Load raw matrix from TSV or CSV with optional header detection."""
    with open(file_path, "r", encoding="utf-8") as f:
        sample = ""
        for line in f:
            stripped = line.strip()
            if stripped:
                sample = stripped
                break
    delim = "\t" if "\t" in sample else ","
    if "V1" in sample or sample.startswith(","):
        import pandas as pd
        df = pd.read_csv(file_path, index_col=0)
        return df.to_numpy(dtype=np.float64)
    return np.loadtxt(file_path, delimiter=delim, dtype=np.float64)


_BUNDLED_ICIO_DIR = Path(__file__).resolve().parent / "_datafiles"
_BUNDLED_ICIO_FILES: dict[str, str] = {"oecd_isic": "icio_oecd2023_2019_isic_g20.npz"}
_DEFAULT_ICIO_SOURCE = "oecd_isic"


def _load_bundled(source: str) -> tuple[np.ndarray, dict[str, Any]]:
    """The bundled table ``source`` and its metadata (codes, citation, build record)."""
    if source == "legacy":
        raise ValueError(
            "load_icio_data(source='legacy') was removed; the bundled table is "
            f"source={_DEFAULT_ICIO_SOURCE!r} (the default), or pass path=... or use synthetic_icio_table(nc, ns)"
        )
    if source not in _BUNDLED_ICIO_FILES:
        raise ValueError(f"unknown ICIO source {source!r}; the bundled table is source={_DEFAULT_ICIO_SOURCE!r}")
    target = _BUNDLED_ICIO_DIR / _BUNDLED_ICIO_FILES[source]
    if not target.is_file():
        raise FileNotFoundError(
            f"the bundled ICIO table {target.name} is not in this installation ({target.parent}). "
            "Build it from the OECD release with tools/build_icio_isic.py, pass path=... "
            "to a table of your own, or use synthetic_icio_table(nc, ns)."
        )
    import json

    with np.load(target, allow_pickle=False) as bundle:
        matrix = np.asarray(bundle["data"], dtype=np.float64)
        record = json.loads(str(bundle["metadata_json"]))
        codes = {key: tuple(str(c) for c in bundle[key]) for key in ("country_codes", "sector_codes", "fd_codes")}
    metadata = {"source": source, "file_path": str(target), **record, **{k: list(v) for k, v in codes.items()}}
    return matrix, metadata


def _read_table_file(path: str | Path) -> np.ndarray:
    """Return a processed transaction table stored in ``path`` as float64.

    ``path`` is a ``.npz`` written by numpy (key ``data``, or the single
    stored array) or a delimited text table; MATLAB ``.mat`` input was removed
    in 4.0.0.
    """
    target = Path(path)
    # Check the suffix before existence: someone migrating from 3.x passes a
    # .mat path, and "file not found" would send them hunting for the file
    # rather than telling them the format is gone.
    if target.suffix == ".mat":
        raise ValueError(
            "MATLAB .mat input is no longer supported (removed in 4.0.0). "
            "Convert it once with "
            "`numpy.savez_compressed(out, data=scipy.io.loadmat(src)['data'])`."
        )
    if target.is_dir():
        raise ValueError(f"{target} is a directory; pass the table file itself")
    if not target.is_file():
        raise FileNotFoundError(f"Specified ICIO data file not found: {target}")
    if target.suffix == ".npz":
        with np.load(target) as bundle:
            key = "data" if "data" in bundle.files else bundle.files[0]
            return np.asarray(bundle[key], dtype=np.float64)
    return _load_raw_table(target)


def _table_dimensions(shape: tuple[int, ...], nfd: int) -> tuple[int, int]:
    """``(nc, ns)`` of a processed table of ``shape`` with ``nfd`` final-use categories per country.

    The layout is ``(ns*nc + 3, ns*nc + nfd*nc)``, so ``nc = (cols - (rows - 3)) / nfd`` and
    ``ns = (rows - 3) / nc``; a shape that gives no positive integer pair raises ``ValueError``.
    """
    if isinstance(nfd, bool) or not isinstance(nfd, (int, np.integer)) or nfd < 1:
        raise ValueError(f"nfd must be a positive integer, got {nfd!r}")
    if len(shape) != 2:
        raise ValueError(f"an ICIO table is two-dimensional, got shape {shape}")
    rows, cols = shape
    n_ind = rows - 3
    fd_cols = cols - n_ind
    if n_ind < 1 or fd_cols < 1 or fd_cols % nfd:
        raise ValueError(
            f"table shape ({rows}, {cols}) is not (ns*nc + 3, ns*nc + nfd*nc) for nfd={nfd}"
        )
    nc = fd_cols // nfd
    if n_ind % nc:
        raise ValueError(
            f"table shape ({rows}, {cols}) is not (ns*nc + 3, ns*nc + nfd*nc) for nfd={nfd}: "
            f"{n_ind} industry rows do not divide among nc={nc} countries"
        )
    return int(nc), int(n_ind // nc)


def load_icio_data(
    path: str | Path | None = None,
    *,
    source: str | None = _DEFAULT_ICIO_SOURCE,
    return_structured: bool = False,
    nfd: int = 3,
    country_codes: Sequence[str] | None = None,
    sector_codes: Sequence[str] | None = None,
    fd_codes: Sequence[str] | None = None,
) -> np.ndarray | ICIOData:
    """Load the bundled ICIO table, or a processed transaction table of your own.

    The table has the layout :func:`synthetic_icio_table` produces and
    ``calibrate_trade_model`` reads: shape ``(ns*nc + 3, ns*nc + nfd*nc)``,
    rows ``ns*nc`` intermediate goods then net taxes, labor and capital,
    columns ``ns*nc`` intermediate uses then ``nfd`` final uses per country.
    Raw provider releases go through the provider loaders instead
    (:func:`load_oecd_icio_granular`, :func:`load_figaro`, ...).

    Without ``path`` the bundled table is returned: the OECD Inter-Country
    Input-Output tables, 2023 edition, reference year 2019, in current million
    USD, aggregated to the 19 G20 member countries plus ``ROW`` and to the 20
    ISIC Rev.4 sections ``A`` to ``T``, with final uses ``C``, ``I`` and
    ``Cx`` (shape ``(403, 460)``). Credit it as "OECD (2023), OECD
    Inter-Country Input-Output Tables, http://oe.cd/icio"; the terms and the
    recipe are in ``puremacro/trade/_datafiles/SOURCES.md``.

    Parameters
    ----------
    path : str or Path, optional
        A ``.npz`` written by numpy (key ``data``, or its single array) or a
        delimited text table. MATLAB ``.mat`` input was removed in 4.0.0.
    source : str, default "oecd_isic"
        Name of the bundled table, read when ``path`` is None. ``"oecd_isic"``
        is the only one (``source="legacy"`` was removed).
    return_structured : bool, default False
        If True, return an :class:`ICIOData` container; the dimensions are
        read from the table's shape and ``nfd``.
    nfd : int, default 3
        Final-use categories per country, used only with ``return_structured``.
    country_codes, sector_codes, fd_codes : sequence of str, optional
        Labels for the structured container; by default the bundled table's
        own codes, and for a file of your own the neutral
        :func:`synthetic_codes` and ``("C", "I", "Cx")`` (``FD01``, ... when
        ``nfd != 3``).

    Returns
    -------
    np.ndarray or ICIOData
        The table as float64, or the structured container whose ``metadata``
        records the file path (and, for the bundled table, its citation and
        build record).

    Raises
    ------
    FileNotFoundError
        If the bundled table is not in the installation; the message names
        the builder ``tools/build_icio_isic.py``.
    """
    if path is not None and source not in (None, _DEFAULT_ICIO_SOURCE):
        raise ValueError("pass either `path` (a file of your own) or `source` (a bundled table), not both")
    if path is None:
        matrix, metadata = _load_bundled(_DEFAULT_ICIO_SOURCE if source is None else source)
        if not return_structured:
            return matrix
        return ICIOData(
            matrix=matrix,
            country_codes=tuple(country_codes) if country_codes is not None else tuple(metadata["country_codes"]),
            sector_codes=tuple(sector_codes) if sector_codes is not None else tuple(metadata["sector_codes"]),
            fd_codes=tuple(fd_codes) if fd_codes is not None else tuple(metadata["fd_codes"]),
            metadata=metadata,
        )

    raw_data = _read_table_file(path)
    if not return_structured:
        return raw_data

    nc, ns = _table_dimensions(raw_data.shape, nfd)
    default_countries, default_sectors = synthetic_codes(nc, ns)
    countries = tuple(country_codes) if country_codes is not None else tuple(default_countries)
    sectors = tuple(sector_codes) if sector_codes is not None else tuple(default_sectors)
    if fd_codes is not None:
        finals = tuple(fd_codes)
    elif nfd == len(CANONICAL_FINAL_DEMAND_CODES):
        finals = CANONICAL_FINAL_DEMAND_CODES
    else:
        finals = tuple(f"FD{k:02d}" for k in range(1, nfd + 1))
    for name, codes, expected in (("country_codes", countries, nc), ("sector_codes", sectors, ns),
                                  ("fd_codes", finals, nfd)):
        if len(codes) != expected:
            raise ValueError(f"{name} has {len(codes)} labels; the table needs {expected}")
    return ICIOData(
        matrix=raw_data,
        country_codes=countries,
        sector_codes=sectors,
        fd_codes=finals,
        metadata={"file_path": str(Path(path).resolve())},
    )


# ---------------------------------------------------------------------------
# Deterministic Synthetic MRIO Generator (generic numbers, provider layouts)
# ---------------------------------------------------------------------------


def _sized_roster(roster: list[str], size: int | None, prefix: str) -> list[str]:
    """The first ``size`` labels of ``roster``, padded with ``<prefix>001``, ... past its end."""
    if size is None or size <= 0:
        return list(roster)
    extra = [f"{prefix}{k:03d}" for k in range(1, size - len(roster) + 1)]
    return list(roster[:size]) + extra


def generate_synthetic_mrio(
    dataset: str,
    year: int = 2019,
    seed: int = 42,
    n_inactive: int = 0,
    n_neg_va: int = 0,
    custom_c: int | None = None,
    custom_s: int | None = None,
    model: str = "ixi",
    unit: str = "M_USD",
) -> RawIOData:
    """Generate a deterministic synthetic Multi-Regional Input-Output table in a provider's layout.

    The numbers are generic random draws, not a model of any database:
    lognormal country sizes (``lognormal(8, 1)``), uniform sector weights,
    gravity trade shares with home bias, uniform intermediate and value-added
    cost shares, and taxes less subsidies as the residual ``Y - column
    purchases - VA`` (roughly 7-22% of gross output at ``custom_c=custom_s=3``).
    Every distribution parameter and the seed are the same for all providers.

    ``dataset`` sets only the layout: the country, sector and final-use label
    lists (hence the dimensions when ``custom_c``/``custom_s`` are not given),
    the number of final-use categories, and x8/x4/x2.5 size multipliers for
    named large economies (USA, CHN; DEU, JPN, GBR, FRA; ITA, CAN, KOR, IND,
    BRA, RUS) when they are in the roster. It does not change the coefficient
    matrix: for a given ``seed`` and dimensions, two providers whose truncated
    rosters contain none of those named economies give a bit-identical ``Z``
    and the same ``Y``, ``VA/Y`` and ``TLS/Y`` up to rounding (for example the
    first three countries and sectors of 'oecd', 'figaro', 'exiobase', 'wiod'
    and 'eora').

    Parameters
    ----------
    dataset : str
        Provider layout: 'figaro', 'exiobase', 'wiod', 'eora', or 'oecd' / 'oecd_icio'.
    year : int, default=2019
        Reference year.
    seed : int, default=42
        Deterministic random seed.
    n_inactive : int, default=0
        Number of inactive sector nodes (zero sales and outlays) to inject for testing.
    n_neg_va : int, default=0
        Number of negative value-added sector nodes to inject for testing.
    custom_c : int, optional
        Override number of economies: the first ``custom_c`` of the roster, then
        generated labels ``R001``, ``R002``, ... beyond its length.
    custom_s : int, optional
        Override number of sectors: the first ``custom_s`` of the roster, then
        generated labels ``X001``, ``X002``, ... beyond its length.
    model : str, default='ixi'
        EXIOBASE model formulation ('ixi' or 'pxp').
    unit : str, default='M_USD'
        Unit label of the accounts ('M_USD', 'M_EUR', or '000_USD'); '000_USD'
        also multiplies every flow by 1000.

    Returns
    -------
    RawIOData
        Container encapsulating synthetic MRIO transaction matrices
        (``metadata["is_synthetic"] = True``).
    """
    ds = dataset.lower().strip()
    if ds in ("figaro", "eurostat_figaro"):
        full_countries = list(FIGARO_COUNTRIES)
        full_sectors = list(FIGARO_64_SECTORS)
        fd_categories = list(FIGARO_FD_CATEGORIES)
    elif ds in ("exiobase", "exiobase3"):
        full_countries = list(EXIOBASE_COUNTRIES)
        full_sectors = list(EXIOBASE_200_SECTORS) if model == "pxp" else list(EXIOBASE_163_SECTORS)
        fd_categories = list(EXIOBASE_FD_CATEGORIES)
    elif ds in ("wiod", "wiod2016"):
        full_countries = list(WIOD_44_COUNTRIES)
        full_sectors = list(WIOD_56_SECTORS)
        fd_categories = list(WIOD_FD_CATEGORIES)
    elif ds in ("eora", "eora26"):
        full_countries = list(EORA_189_COUNTRIES)
        full_sectors = list(EORA_26_SECTORS)
        fd_categories = list(EORA_FD_CATEGORIES)
    elif ds in ("oecd", "oecd_icio", "icio"):
        full_countries = list(OECD_77_COUNTRIES)
        full_sectors = list(OECD_45_SECTORS)
        fd_categories = list(OECD_FD_CATEGORIES)
    else:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Choose from: 'figaro', 'exiobase', 'wiod', 'eora', 'oecd'."
        )

    countries = _sized_roster(full_countries, custom_c, "R")
    sectors = _sized_roster(full_sectors, custom_s, "X")

    C = len(countries)
    S = len(sectors)
    M = C * S
    K_F = len(fd_categories)

    rng = np.random.default_rng(seed)

    # 1. Base economic size distribution (lognormal GDP)
    base_gdp = rng.lognormal(mean=8.0, sigma=1.0, size=C)
    if unit == "000_USD":
        base_gdp *= 1000.0
    for c_idx, c_code in enumerate(countries):
        if c_code in ("USA", "CHN"):
            base_gdp[c_idx] *= 8.0
        elif c_code in ("DEU", "JPN", "GBR", "FRA"):
            base_gdp[c_idx] *= 4.0
        elif c_code in ("ITA", "CAN", "KOR", "IND", "BRA", "RUS"):
            base_gdp[c_idx] *= 2.5

    # 2. Sectoral weight distribution
    sector_weights = rng.uniform(0.5, 2.0, size=S)
    sector_weights /= np.sum(sector_weights)

    Y_target_2d = np.outer(base_gdp, sector_weights)  # (C, S)
    Y_target = Y_target_2d.reshape(M)

    # 3. Bilateral trade gravity matrix with domestic preference
    dist = rng.uniform(1.0, 3.0, size=(C, C))
    np.fill_diagonal(dist, 0.2)
    friction = np.exp(-0.8 * dist)
    gravity = base_gdp[:, None] * friction

    if C > 1:
        log_gdp = np.log(base_gdp)
        log_range = float(np.max(log_gdp) - np.min(log_gdp))
        gdp_norm = (log_gdp - np.min(log_gdp)) / log_range if log_range > 1e-12 else np.zeros(C)
        omega_dom = 0.70 + 0.18 * gdp_norm

        for d in range(C):
            off_sum = float(np.sum(gravity[:, d]) - gravity[d, d])
            if off_sum > 0:
                gravity[d, d] = off_sum * (omega_dom[d] / (1.0 - omega_dom[d]))

    trade_share = gravity / np.sum(gravity, axis=0, keepdims=True)  # (C_orig, C_dest)

    # 4. Intermediate Technical Coefficients & Flows
    inter_cost_share = rng.uniform(0.42, 0.52, size=M)
    va_cost_share = rng.uniform(0.42, 0.50, size=M)

    io_tech = rng.uniform(0.05, 0.35, size=(S, S))
    np.fill_diagonal(io_tech, np.diag(io_tech) + 0.4)
    io_tech /= np.sum(io_tech, axis=0, keepdims=True)

    total_purchases = (inter_cost_share * Y_target).reshape(C, S)
    input_flows = io_tech[None, :, :] * total_purchases[:, None, :]
    Z_4d = trade_share[:, None, :, None] * np.transpose(input_flows, (1, 0, 2))[None, :, :, :]
    Z = Z_4d.reshape(M, M)

    # 5. Final Demand Flows
    fd_cat_weights = np.array([0.58, 0.04, 0.18, 0.15, 0.03, 0.01, 0.01][:K_F], dtype=np.float64)
    fd_cat_weights /= np.sum(fd_cat_weights)

    good_fd = (base_gdp * 0.75)[:, None] * sector_weights[None, :]
    cat_flow = good_fd[:, :, None] * fd_cat_weights[None, None, :]
    F_4d = trade_share[:, None, :, None] * np.transpose(cat_flow, (1, 0, 2))[None, :, :, :]
    F_2d = F_4d.reshape(M, C * K_F)

    # 6. Gross Output from Row Sales
    Y = np.sum(Z, axis=1) + np.sum(F_2d, axis=1)

    # 7. Value Added & Residual TLS
    VA = va_cost_share * Y
    TLS = Y - np.sum(Z, axis=0) - VA

    # 8. Test Edge Cases Injection
    if n_inactive > 0:
        inactive_indices = [(step * S) % M for step in range(n_inactive)]
        for idx in inactive_indices:
            Z[idx, :] = 0.0
            Z[:, idx] = 0.0
            F_2d[idx, :] = 0.0
            VA[idx] = 0.0
            TLS[idx] = 0.0
            Y[idx] = 0.0
        # Reconcile Y and TLS for remaining sectors so row sales and column outlays match
        Y = np.sum(Z, axis=1) + np.sum(F_2d, axis=1)
        for idx in inactive_indices:
            Y[idx] = 0.0
        TLS = Y - np.sum(Z, axis=0) - VA
        for idx in inactive_indices:
            TLS[idx] = 0.0

    if n_neg_va > 0:
        start_idx = (n_inactive * S + 1) % M
        neg_indices = [(start_idx + step * 2) % M for step in range(n_neg_va)]
        for idx in neg_indices:
            VA[idx] = -50.0

    return RawIOData(
        countries=countries,
        sectors=sectors,
        intermediate_matrix=Z,
        final_demand_matrix=F_2d,
        value_added=VA,
        taxes_less_subsidies=TLS,
        gross_output=Y,
        fd_categories=fd_categories,
        year=year,
        source=f"Synthetic_{dataset.upper()}",
        unit=unit,
        metadata={"is_synthetic": True, "seed": seed, "generator": "generate_synthetic_mrio", "model": model},
    )


# ---------------------------------------------------------------------------
# Calibration Pipeline Packaging
# ---------------------------------------------------------------------------


def package_mrio_to_calibration_result(
    raw: RawIOData,
    regularize: bool = True,
    validate: bool = True,
    *,
    regularize_options: dict[str, Any] | None = None,
) -> TradeCalibrationResult:
    """Package RawIOData flows into a canonical TradeCalibrationResult.

    Optionally regularizes flows via ``regularize_mrio_table``, decomposes value added
    into labor (2/3) and capital (1/3), constructs the calibrated transaction table,
    and runs ``calibrate_trade_model`` with invariant validation.

    The final-use categories are passed through in the order of
    ``raw.fd_categories``; ``calibrate_trade_model`` adds the foreign balance
    to index 1, so that index must be investment (the provider loaders
    condense their rosters to ``("C", "I", "Cx")`` first). When index 1 is not
    an investment code (``I``, ``G``, ``GFCF``, ``P51G``, ``P5``, ``GCF``), for
    example a native six-category OECD roster whose index 1 is NPISH, a
    ``RuntimeWarning`` names the category that receives the closure.

    Metadata recorded on the result: ``adjustments`` (per array: changed
    entries, maximum absolute change, net change), ``regularization`` (the
    value-added floor rule, every floored node with its label, value added
    before and after and gross output, phantom-output node labels, and the
    certified spectral bounds), ``negative_final_demand_cells`` (count per
    final-use category after regularization; the ``accounting="consistent"``
    path rejects any negative cell), ``closure_category`` (the category at
    index 1) and ``fd_categories``.

    Parameters
    ----------
    raw : RawIOData
        Container with unaggregated transaction matrices.
    regularize : bool, default=True
        Whether to apply phantom output injection, VA flooring with dual TLS debit,
        and residual TLS reconciliation.
    validate : bool, default=True
        Whether to enforce budget balance and table reconstruction checks.
    regularize_options : dict, optional
        Keyword-only overrides forwarded to ``regularize_mrio_table``; allowed
        keys ``floor_output``, ``floor_va_ratio``, ``floor_va_abs``,
        ``floor_va_max_share``.

    Returns
    -------
    TradeCalibrationResult
        Calibrated trade model result container.
    """
    table = _transaction_table(raw, regularize=regularize, regularize_options=regularize_options)
    matrix, regularization = table["matrix"], table["regularization"]
    C, S, K_F = raw.C, raw.S, raw.K_F
    M = C * S
    F_2d = table["F"]

    closure_category = str(list(raw.fd_categories)[1 if K_F >= 2 else 0])
    if K_F >= 2 and closure_category not in _INVESTMENT_CODES:
        warnings.warn(
            f"package_mrio_to_calibration_result: calibrate_trade_model adds the foreign balance to "
            f"final-use index 1, which is {closure_category!r} in {list(raw.fd_categories)}, not investment. "
            "Condense the roster first (puremacro.trade._oecd_icio.condense_final_demand for OECD rosters; "
            "the load_* functions condense every provider) so that index 1 is investment.",
            RuntimeWarning, stacklevel=2,
        )

    # calibration imports this module, so the calibrator is imported at call time.
    from puremacro.trade.calibration import calibrate_trade_model

    result = calibrate_trade_model(
        matrix,
        ns=S,
        nc=C,
        nfd=K_F,
        country_codes=raw.countries,
        sector_codes=raw.sectors,
        validate=validate,
    )

    before = table["before"]
    adjustments = {
        name: {"changed_entries": int(np.count_nonzero(table[name] != before[name])),
               "max_abs_change": float(np.max(np.abs(table[name] - before[name]), initial=0.0)),
               "net_change": float(np.sum(table[name] - before[name]))}
        for name in ("Z", "F", "VA", "TLS", "Y")
    }
    F_cat = F_2d.reshape(M, C, K_F)
    negative_cells = {str(cat): int(np.count_nonzero(F_cat[:, :, k] < 0))
                      for k, cat in enumerate(raw.fd_categories)}
    return replace(result, metadata={
        **result.metadata, **raw.metadata,
        "source": raw.source, "year": raw.year, "unit": raw.unit,
        "schema": raw.metadata.get("schema", "RawIOData country-major sector ordering"),
        "is_synthetic": bool(raw.metadata.get("is_synthetic", raw.source.startswith("Synthetic_"))),
        "regularized": regularize, "adjustments": adjustments,
        "regularization": regularization,
        "negative_final_demand_cells": negative_cells,
        "output_reconciliation": "transaction row sales" if regularize else "provider output",
        "labor_share_assumption": 2.0 / 3.0, "capital_share_assumption": 1.0 / 3.0,
        "fd_categories": tuple(raw.fd_categories),
        "closure_category": closure_category,
    })


def _transaction_table(
    raw: RawIOData,
    *,
    regularize: bool,
    regularize_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the processed transaction table of ``raw`` (the input of ``calibrate_trade_model``).

    Optionally regularizes the flows with ``regularize_mrio_table`` (after
    reconciling output to row sales), splits value added into labor (2/3) and
    capital (1/3), and stacks ``[[Z, F], [TLS], [labor], [capital]]``. Returns
    the matrix, the regularization record, the arrays after (``Z``, ``F``,
    ``VA``, ``TLS``, ``Y``) and before (``before``) any change.
    """
    options = dict(regularize_options or {})
    allowed = {"floor_output", "floor_va_ratio", "floor_va_abs", "floor_va_max_share"}
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise ValueError(f"regularize_options accepts {sorted(allowed)}, got unknown keys {unknown}")

    C, S, K_F = raw.C, raw.S, raw.K_F
    M = C * S

    Z = np.array(raw.intermediate_matrix, dtype=np.float64, copy=True)
    F = np.array(raw.final_demand_matrix, dtype=np.float64, copy=True)
    VA = np.array(raw.value_added, dtype=np.float64, copy=True)
    TLS = np.array(raw.taxes_less_subsidies, dtype=np.float64, copy=True)
    Y = np.array(raw.gross_output, dtype=np.float64, copy=True)

    expected_shapes = {"Z": (M, M), "F": (M, C * K_F), "TLS": (M,), "Y": (M,)}
    F = F.reshape(M, C * K_F) if F.ndim == 3 else F
    for name, arr in (("Z", Z), ("F", F), ("VA", VA), ("TLS", TLS), ("Y", Y)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite data")
        if name in expected_shapes and arr.shape != expected_shapes[name]:
            raise ValueError(f"{name} shape {arr.shape} != {expected_shapes[name]}")
    if VA.shape not in ((M,),) and not (VA.ndim == 2 and VA.shape[1] == M):
        raise ValueError("Value added must have one column per country-sector node")
    if len(set(raw.countries)) != C or len(set(raw.sectors)) != S:
        raise ValueError("Country and sector labels must be unique")
    before = {name: arr.copy() for name, arr in (("Z", Z), ("F", F), ("VA", VA), ("TLS", TLS), ("Y", Y))}
    labels = [f"{c}_{s}" for c in raw.countries for s in raw.sectors]
    regularization: dict[str, Any] = {"applied": bool(regularize)}
    if regularize:
        # Provider OUT margins can differ from the released transactions.
        # Calibrate to actual sales and record the change from reported Y;
        # otherwise TLS repair leaves consumer budgets unbalanced.
        Y = Z.sum(axis=1) + F.sum(axis=1)
        Z, F, VA, TLS, Y, report = regularize_mrio_table(
            Z, F, VA, TLS, Y, n_countries=C, n_sectors=S, return_report=True, **options
        )
        floor = report["va_floor"]
        regularization.update({
            "va_floor_rule": floor["rule"],
            "va_floored_nodes": [
                {"node": labels[i], "value_added_before": before_va, "value_added_after": after_va,
                 "gross_output": output, "binding": bind}
                for i, before_va, after_va, output, bind in zip(
                    floor["nodes"], floor["value_added_before"], floor["value_added_after"],
                    floor["gross_output"], floor["binding"])
            ],
            "phantom_nodes": [labels[i] for i in report["phantom_nodes"]],
            "spectral": report["spectral"],
        })
    elif options:
        raise ValueError("regularize_options require regularize=True")

    va_row = np.sum(VA, axis=0) if VA.ndim == 2 else VA
    labor = (2.0 / 3.0) * va_row
    capital = (1.0 / 3.0) * va_row

    if getattr(raw, "taxes_less_subsidies_fd", None) is not None:
        tfd = np.array(raw.taxes_less_subsidies_fd, dtype=np.float64, copy=True).reshape(C * K_F)
        TLS_full = np.hstack([TLS, tfd])
    else:
        TLS_full = np.hstack([TLS, np.zeros(C * K_F)])
    labor_full = np.hstack([labor, np.zeros(C * K_F)])
    capital_full = np.hstack([capital, np.zeros(C * K_F)])

    F_2d = F.reshape(M, C * K_F) if F.ndim == 3 else F
    matrix = np.vstack([
        np.hstack([Z, F_2d]),
        TLS_full,
        labor_full,
        capital_full,
    ])
    return {"matrix": matrix, "regularization": regularization, "before": before,
            "Z": Z, "F": F_2d, "VA": VA, "TLS": TLS, "Y": Y}


def synthetic_codes(nc: int, ns: int) -> tuple[list[str], list[str]]:
    """Neutral country and sector labels ``C01..`` and ``S01..`` (wider when ``nc`` or ``ns`` >= 100)."""
    for name, n in (("nc", nc), ("ns", ns)):
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError(f"{name} must be a positive integer, got {n!r}")
    cw, sw = max(2, len(str(nc))), max(2, len(str(ns)))
    return [f"C{k:0{cw}d}" for k in range(1, nc + 1)], [f"S{k:0{sw}d}" for k in range(1, ns + 1)]


#: Smallest investment expenditure share (investment plus net exports, over
#: income) in a :func:`synthetic_icio_table` country with ``nfd >= 2``.
_MIN_INVESTMENT_SHARE = 0.05


def _floor_national_saving(raw: RawIOData, consumption: np.ndarray,
                           investment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Move spending from consumption to investment where investment plus net
    exports falls below ``_MIN_INVESTMENT_SHARE`` of income.

    ``consumption`` and ``investment`` are ``(M, C)`` final-use columns. The
    move is proportional to each country's consumption column, so row sums and
    bilateral flows (hence net exports) do not change.
    """
    C, S = raw.C, raw.S
    Z = np.asarray(raw.intermediate_matrix, dtype=float)
    F = np.asarray(raw.final_demand_matrix, dtype=float).reshape(raw.M, C, raw.K_F)
    flows = Z.reshape(C, S, C, S).sum(axis=(1, 3)) + F.sum(axis=2).reshape(C, S, C).sum(axis=1)
    np.fill_diagonal(flows, 0.0)
    net_exports = flows.sum(axis=1) - flows.sum(axis=0)
    income = F.sum(axis=(0, 2)) + net_exports
    shortfall = _MIN_INVESTMENT_SHARE * income - (investment.sum(axis=0) + net_exports)
    move = np.maximum(shortfall, 0.0)
    if not np.any(move > 0):
        return consumption, investment
    total = consumption.sum(axis=0)
    if np.any(move >= total):
        raise ValueError("synthetic table: consumption too small to raise the investment share")
    shifted = consumption * (move / total)[None, :]
    return consumption - shifted, investment + shifted


def synthetic_icio_table(nc: int, ns: int, *, nfd: int = 3, seed: int = 0) -> np.ndarray:
    """A seeded synthetic processed ICIO table with ``nc`` countries and ``ns`` sectors.

    The table has the layout ``calibrate_trade_model`` reads, shape
    ``(ns*nc + 3, ns*nc + nfd*nc)``: rows are the ``ns*nc`` country-sector
    nodes (country-major), net taxes, labor and capital; columns are the
    nodes, then ``nfd`` final uses per country. The flows come from
    :func:`generate_synthetic_mrio` in the OECD layout (generic random draws,
    not a model of any economy), regularized like the provider loaders, with
    value added split 2/3 labor and 1/3 capital. Final uses are grouped as
    consumption (households, NPISH, government), investment (fixed capital
    formation and inventories, positive in every cell) and residents'
    purchases abroad; with ``nfd == 2`` the purchases abroad join
    consumption, with ``nfd == 1`` all final uses form one category, and with
    ``nfd > 3`` they are split equally over the categories after investment.
    Label the result with :func:`synthetic_codes`.

    Calibration adds each country's net exports to its investment, so the
    investment expenditure share is (investment + net exports) / income. Where
    the random trade draws would leave that share below 5%, spending moves
    from consumption to investment inside that country, in proportion to its
    consumption basket. Every row sum, bilateral flow and country total is
    unchanged, and every seed gives a table that consistent accounting accepts
    (``nfd >= 2``).

    The same arguments give a bit-identical table.

    Parameters
    ----------
    nc, ns : int
        Number of countries and sectors (any positive size).
    nfd : int, default 3
        Final-use categories per country.
    seed : int, default 0
        Seed of the generator.

    Returns
    -------
    np.ndarray
        Float64 array of shape ``(ns*nc + 3, ns*nc + nfd*nc)``.
    """
    synthetic_codes(nc, ns)                     # validates nc and ns
    if isinstance(nfd, bool) or not isinstance(nfd, (int, np.integer)) or nfd < 1:
        raise ValueError(f"nfd must be a positive integer, got {nfd!r}")
    raw = generate_synthetic_mrio("oecd", custom_c=nc, custom_s=ns, seed=seed)
    F = np.asarray(raw.final_demand_matrix, dtype=float).reshape(raw.M, raw.C, raw.K_F)
    use = {code: F[:, :, k] for k, code in enumerate(raw.fd_categories)}
    consumption = use["HFCE"] + use["NPISH"] + use["GGFC"]
    investment = use["GFCF"] + use["INVNT"]
    abroad = use["DPABR"]
    if nfd >= 2:
        consumption, investment = _floor_national_saving(raw, consumption, investment)
    if nfd == 1:
        groups = [consumption + investment + abroad]
    elif nfd == 2:
        groups = [consumption + abroad, investment]
    else:
        groups = [consumption, investment] + [abroad / (nfd - 2)] * (nfd - 2)
    labels = list(CANONICAL_FINAL_DEMAND_CODES) if nfd == 3 else [f"FD{k:02d}" for k in range(1, nfd + 1)]
    grouped = replace(raw, final_demand_matrix=np.stack(groups, axis=2).reshape(raw.M, -1),
                      fd_categories=labels)
    return _transaction_table(grouped, regularize=True)["matrix"]


def _condense_provider_final_uses(raw: RawIOData, provider: str) -> RawIOData:
    """Condense a provider's native final uses to ``("C", "I", "Cx")``, conserving sums.

    Uses ``_PROVIDER_FINAL_USE_MAPPINGS[provider]`` (see its comment); records
    ``source_fd_categories``, ``final_use_mapping``, ``final_use_semantics``
    and ``negative_final_demand_cells`` in the metadata. Signed entries
    (negative inventory changes) are kept; nothing is clipped.
    """
    mapping = _PROVIDER_FINAL_USE_MAPPINGS[provider]
    cats = [str(c) for c in raw.fd_categories]
    mapped = [code for group in mapping.values() for code in group]
    if sorted(mapped) != sorted(cats):
        raise ValueError(
            f"{provider} final-use roster {cats} does not match the condensation mapping {mapping}"
        )
    K = len(cats)
    F = np.asarray(raw.final_demand_matrix, dtype=float).reshape(raw.M, raw.C, K)
    grouped = np.stack([F[:, :, [cats.index(code) for code in group]].sum(axis=2)
                        for group in mapping.values()], axis=2)
    taxes = None
    if raw.taxes_less_subsidies_fd is not None:
        tfd = np.asarray(raw.taxes_less_subsidies_fd, dtype=float).reshape(raw.C, K)
        taxes = np.stack([tfd[:, [cats.index(code) for code in group]].sum(axis=1)
                          for group in mapping.values()], axis=1).ravel()
    record = {key: list(group) for key, group in mapping.items()}
    negative = {key: int(np.count_nonzero(grouped[:, :, j] < 0)) for j, key in enumerate(mapping)}
    return replace(
        raw, final_demand_matrix=grouped.reshape(raw.M, -1), fd_categories=list(mapping),
        taxes_less_subsidies_fd=taxes,
        metadata={**raw.metadata, "source_fd_categories": cats, "final_use_mapping": record,
                  "final_demand_mapping": record, "negative_final_demand_cells": negative,
                  "final_use_semantics": dict(_PROVIDER_FINAL_USE_SEMANTICS[provider]),
                  "inventory_treatment": "signed inventory changes and valuables kept in Cx; no clipping"},
    )


# ---------------------------------------------------------------------------
# Harmonized Ingestion Adapters for 5 Major Databases
# ---------------------------------------------------------------------------


_COMPRESSION_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x1f\x8b", "gzip"), (b"BZh", "bz2"), (b"\xfd7zXZ\x00", "xz"),
    (b"PK\x03\x04", "zip"), (b"\x28\xb5\x2f\xfd", "zstd"),
)


def _detect_compression(path: Path) -> str | None:
    """Compression method from the file's magic bytes (``None`` for plain files)."""
    with open(path, "rb") as handle:
        head = handle.read(6)
    for magic, method in _COMPRESSION_MAGIC:
        if head.startswith(magic):
            return method
    return None


def _read_first_row(path: Path, sep: str, compression: str | None) -> list[str]:
    """Cells of the first line as raw strings, decompressed like the full read.

    Goes through ``pandas.read_csv`` with ``header=None`` so compressed
    harmonized files (gzip, bz2, xz, single-member zip) are sniffed exactly as
    they are read, and duplicate labels are not renamed.
    """
    try:
        frame = pd.read_csv(path, sep=sep, header=None, nrows=1, dtype=str, keep_default_na=False,
                            compression=compression or "infer", encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} is not a UTF-8 text table (or a gzip/bz2/xz/zip-compressed one)") from exc
    except pd.errors.EmptyDataError:
        return []
    return [str(cell) for cell in frame.iloc[0].tolist()] if len(frame) else []


def _find_unique_match(directories: Sequence[Path], patterns: Sequence[str], what: str) -> Path | None:
    """First directory with a match for ``patterns``; refuses ambiguous directories.

    Matches are de-duplicated and sorted, so the result does not depend on the
    filesystem order; more than one match in the same directory raises
    ``ValueError`` asking for an explicit ``file_path``.
    """
    for d in directories:
        if not d.is_dir():
            continue
        matches = sorted({p for pattern in patterns for p in d.glob(pattern) if p.is_file()})
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous {what} in {d}: {[m.name for m in matches]} all match; pass file_path= explicitly."
            )
        if matches:
            return matches[0]
    return None


def load_figaro(
    year: int = 2019,
    eur_to_usd: float = 1.1195,
    fallback_to_synthetic: bool = False,
    *,
    data_dir: str | Path | None = None,
    file_path: str | Path | None = None,
    custom_c: int | None = None,
    custom_s: int | None = None,
    seed: int = 42,
    regularize: bool = True,
) -> TradeCalibrationResult:
    """Load Eurostat FIGARO table with automatic EUR -> USD currency harmonization.

    Schema: 46 countries x 64 NACE Rev. 2 sectors = 2,944 nodes, 5 final demand categories.

    Accepted file: the puremacro harmonized labelled layout (not the native
    Eurostat archive, which :func:`puremacro.trade.mrio.read_figaro_native`
    reads). Rows are ``COUNTRY_SECTOR`` nodes in country-major order followed by
    a ``TLS`` and a ``VA`` row; columns are the same nodes followed by five
    final-use columns per country labelled ``COUNTRY_CODE`` with ``CODE`` in
    ``P3_S13, P3_S14, P3_S15, P51G, P5M``. Final-use columns are read by label,
    in any order (the Eurostat order P3_S13, P3_S14, P3_S15, P51G, P5M and the
    order of :data:`FIGARO_FD_CATEGORIES` both work); unknown or duplicated
    codes raise ``ValueError``.

    The file may be gzip-, bz2- or xz-compressed or a single-member ZIP
    (detected from its bytes, not its name); ``.tsv`` in the file suffixes
    selects tab separation.

    Final uses are condensed before calibration to ``C = P3_S13 + P3_S14 +
    P3_S15`` (final consumption of government, households and NPISH),
    ``I = P51G`` (gross fixed capital formation, which receives the
    foreign-balance closure) and ``Cx = P5M`` (changes in inventories and
    valuables); the mapping and the meaning of each slot are recorded in
    ``metadata["final_use_mapping"]`` and ``metadata["final_use_semantics"]``.
    Cx here is not the OECD
    Cx (DPABR, residents' direct purchases abroad), so bridges that read
    ``("C", "I", "Cx")`` as (C, G, X) must consult the mapping. Up to 4.3.0 the five native
    categories were passed through and the closure landed on P3_S15 (NPISH).

    Parameters
    ----------
    year : int, default=2019
        Reference year.
    eur_to_usd : float, default=1.1195
        Assumed USD per EUR conversion factor. The default is a 2019 convention;
        supply the appropriate factor explicitly for other years.
    fallback_to_synthetic : bool, default=False
        Explicitly opt into generated teaching data if the archive is missing.
    data_dir : str or Path, optional
        Directory containing FIGARO CSV/TSV files.
    file_path : str or Path, optional
        Exact path to FIGARO table file.
    custom_c : int, optional
        Override country count for testing.
    custom_s : int, optional
        Override sector count for testing.
    seed : int, default=42
        Deterministic random seed for synthetic fallback.
    regularize : bool, default=True
        Whether to apply phantom injection and TLS reconciliation.

    Returns
    -------
    TradeCalibrationResult
        Calibrated general equilibrium model.
    """
    if custom_c is not None or custom_s is not None:
        if not fallback_to_synthetic:
            raise ValueError("Custom dimensions generate synthetic data; set fallback_to_synthetic=True explicitly.")
        if file_path is not None or data_dir is not None:
            raise ValueError("Custom synthetic dimensions cannot be combined with an empirical file or data directory.")
        raw_eur = generate_synthetic_mrio("figaro", year=year, seed=seed, custom_c=custom_c, custom_s=custom_s, unit="M_EUR")
        raw_usd = RawIOData(
            countries=raw_eur.countries,
            sectors=raw_eur.sectors,
            intermediate_matrix=raw_eur.intermediate_matrix * eur_to_usd,
            final_demand_matrix=raw_eur.final_demand_matrix * eur_to_usd,
            value_added=raw_eur.value_added * eur_to_usd,
            taxes_less_subsidies=raw_eur.taxes_less_subsidies * eur_to_usd,
            gross_output=raw_eur.gross_output * eur_to_usd,
            fd_categories=raw_eur.fd_categories,
            year=year,
            source=raw_eur.source,
            metadata={**raw_eur.metadata, "exchange_rate_usd_per_eur": eur_to_usd, "original_unit": "M_EUR"},
            unit="M_USD",
        )
        return package_mrio_to_calibration_result(_condense_provider_final_uses(raw_usd, "figaro"),
                                                  regularize=regularize)

    target: Path | None = None
    if file_path is not None:
        p = Path(file_path)
        if p.is_file():
            target = p
    else:
        candidates = [Path(os.environ["IO_FIGARO_DIR"]) ] if os.environ.get("IO_FIGARO_DIR") else []
        if data_dir:
            candidates.insert(0, Path(data_dir))
        target = _find_unique_match(candidates, (f"*{year}*.csv", f"*{year}*.tsv"), f"FIGARO {year} tables")

    if target is None or not target.is_file():
        if fallback_to_synthetic:
            raw_eur = generate_synthetic_mrio("figaro", year=year, seed=seed, unit="M_EUR")
            raw_usd = RawIOData(
                countries=raw_eur.countries,
                sectors=raw_eur.sectors,
                intermediate_matrix=raw_eur.intermediate_matrix * eur_to_usd,
                final_demand_matrix=raw_eur.final_demand_matrix * eur_to_usd,
                value_added=raw_eur.value_added * eur_to_usd,
                taxes_less_subsidies=raw_eur.taxes_less_subsidies * eur_to_usd,
                gross_output=raw_eur.gross_output * eur_to_usd,
                fd_categories=raw_eur.fd_categories,
                year=year,
                source=raw_eur.source,
            metadata={**raw_eur.metadata, "exchange_rate_usd_per_eur": eur_to_usd, "original_unit": "M_EUR"},
                unit="M_USD",
            )
            return package_mrio_to_calibration_result(_condense_provider_final_uses(raw_usd, "figaro"),
                                                      regularize=regularize)
        raise FileNotFoundError(f"Eurostat FIGARO table for year {year} not found.")

    sep = "\t" if ".tsv" in [suffix.lower() for suffix in target.suffixes] else ","
    compression = _detect_compression(target)
    header = _read_first_row(target, sep, compression)
    from collections import Counter
    duplicated = sorted(label for label, count in Counter(header).items() if count > 1)
    if duplicated:
        raise ValueError(f"FIGARO table has duplicate column labels {duplicated[:10]}")
    df = pd.read_csv(target, sep=sep, index_col=0, low_memory=False, compression=compression or "infer")

    # Supported interchange schema: labeled country-major intermediate rows
    # and columns, five final-demand categories per country, then TLS and VA.
    # Native Eurostat archives require conversion to this explicit layout.
    if len(df) < 3 or [str(x).upper() for x in df.index[-2:]] != ["TLS", "VA"]:
        raise ValueError(
            "FIGARO harmonized table requires final TLS and VA rows; the native Eurostat "
            "file (trailing W2_* rows) is read by puremacro.trade.mrio.read_figaro_native"
        )
    M = len(df) - 2
    int_cols = list(df.columns[:M])
    if any("_" not in label for label in int_cols):
        raise ValueError("Intermediate columns must be labeled COUNTRY_SECTOR")
    countries = list(dict.fromkeys(label.split("_", 1)[0] for label in int_cols))
    sectors = list(dict.fromkeys(label.split("_", 1)[1] for label in int_cols))
    N, S = len(countries), len(sectors)
    expected = [f"{c}_{sector}" for c in countries for sector in sectors]
    if int_cols != expected or list(df.index[:M]) != expected:
        raise ValueError("Intermediate rows and columns must share complete country-major labels")
    n_fd = N * len(FIGARO_FD_CATEGORIES)
    if df.shape[1] != M + n_fd:
        raise ValueError("Final-demand columns must contain five categories per country")
    # Read final uses by label (COUNTRY_CODE), never by position.
    fd_cols = [str(c) for c in df.columns[M:]]
    if len(set(fd_cols)) != len(fd_cols):
        raise ValueError("Final-demand column labels must be unique")
    split = [label.split("_", 1) if "_" in label else [label, ""] for label in fd_cols]
    unknown_countries = sorted({part[0] for part in split} - set(countries))
    if unknown_countries:
        raise ValueError(f"Final-demand columns name countries absent from the rows: {unknown_countries}")
    unknown_codes = sorted({part[1] for part in split} - set(FIGARO_FD_CATEGORIES))
    if unknown_codes:
        raise ValueError(
            f"Unknown FIGARO final-use code(s) {unknown_codes}: label final-demand columns "
            f"COUNTRY_CODE with CODE in {list(FIGARO_FD_CATEGORIES)} (any order)"
        )
    wanted = [f"{c}_{code}" for c in countries for code in FIGARO_FD_CATEGORIES]
    missing = sorted(set(wanted) - set(fd_cols))
    if missing:
        raise ValueError(f"Missing FIGARO final-demand columns: {missing[:10]}")
    order = [M + fd_cols.index(label) for label in wanted]
    Z = df.iloc[:M, :M].to_numpy(dtype=float) * eur_to_usd
    F = df.iloc[:M, order].to_numpy(dtype=float) * eur_to_usd
    TLS = df.iloc[M, :M].to_numpy(dtype=float) * eur_to_usd
    TLS_fd = df.iloc[M, order].to_numpy(dtype=float) * eur_to_usd
    VA = df.iloc[M + 1, :M].to_numpy(dtype=float) * eur_to_usd
    OUT = np.sum(Z, axis=1) + np.sum(F, axis=1)

    raw = RawIOData(
        countries=countries,
        sectors=sectors,
        intermediate_matrix=Z,
        final_demand_matrix=F,
        value_added=VA,
        taxes_less_subsidies=TLS,
        gross_output=OUT,
        fd_categories=list(FIGARO_FD_CATEGORIES),
        year=year,
        source="Eurostat_FIGARO_Harmonized",
        unit="M_USD",
        taxes_less_subsidies_fd=TLS_fd,
    )
    raw.metadata.update({
        "exchange_rate_usd_per_eur": eur_to_usd, "original_unit": "M_EUR",
        "is_synthetic": False, "file_path": str(target.resolve()),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "schema": "puremacro labelled harmonized table (not a native archive reader)",
        "schema_validation": ("country-major node labels, final uses read by COUNTRY_CODE label, "
                              "finite values; native archive parity unverified"),
        "source_fd_column_order": [part[1] for part in split[:len(FIGARO_FD_CATEGORIES)]],
    })
    return package_mrio_to_calibration_result(_condense_provider_final_uses(raw, "figaro"), regularize=regularize)


def load_exiobase(
    year: int = 2019,
    model: str = "ixi",
    eur_to_usd: float = 1.1195,
    fallback_to_synthetic: bool = False,
    *,
    data_dir: str | Path | None = None,
    file_path: str | Path | None = None,
    custom_c: int | None = None,
    custom_s: int | None = None,
    seed: int = 42,
    regularize: bool = True,
) -> TradeCalibrationResult:
    """Load EXIOBASE 3 table with automatic EUR -> USD currency harmonization.

    Schema: 49 regions x 163 industries (ixi) or 200 products (pxp), 7 final demand categories.

    Accepted file: only the puremacro harmonized positional layout, a single
    headerless tab-separated numeric table with ``49 * S`` intermediate rows
    followed by 7 value-added rows (summed), and ``49 * S`` intermediate
    columns followed by ``49 * 7`` final-use columns in
    :data:`EXIOBASE_FD_CATEGORIES` order per region (``S = 163`` for ixi,
    ``200`` for pxp). The native EXIOBASE archive (``IOT_<year>_<model>.zip``
    with ``Z.txt``, ``Y.txt``, ``x.txt`` and ``satellite/F.txt``, labelled with
    two header rows) is not read here: use
    ``puremacro.trade.mrio.read_exiobase_native``. A ZIP archive with more
    than one member or with a native ``Z.txt``/``Y.txt``/``x.txt``/``F.txt``
    member, or a labelled native ``Z.txt``/``Y.txt``, raises ``ValueError``
    saying so. The harmonized table itself may be gzip-, bz2- or
    xz-compressed or a single-member ZIP (detected from its bytes).

    Final uses are condensed before calibration to ``C = HFCE + NPISH + GGFC``,
    ``I = GFCF`` (receives the foreign-balance closure) and
    ``Cx = INVNT + VALUABLES + EXPORT``; the mapping and the meaning of each
    slot are recorded in ``metadata["final_use_mapping"]`` and
    ``metadata["final_use_semantics"]``. Cx here is not the OECD
    Cx (DPABR, residents' direct purchases abroad), so bridges that read
    ``("C", "I", "Cx")`` as (C, G, X) must consult the mapping.
    Up to 4.3.0 the seven native categories were passed through and the
    closure landed on NPISH.

    Parameters
    ----------
    year : int, default=2019
        Reference year.
    model : str, default='ixi'
        Model formulation: 'ixi' (industry-by-industry) or 'pxp' (product-by-product).
    eur_to_usd : float, default=1.1195
        Assumed USD per EUR conversion factor. The default is a 2019 convention;
        supply the appropriate factor explicitly for other years.
    fallback_to_synthetic : bool, default=False
        Whether to generate synthetic benchmark data if file is missing.
    data_dir : str or Path, optional
        Directory containing EXIOBASE files.
    file_path : str or Path, optional
        Exact path to EXIOBASE file.
    custom_c : int, optional
        Override country count for testing.
    custom_s : int, optional
        Override sector count for testing.
    seed : int, default=42
        Deterministic random seed for synthetic fallback.
    regularize : bool, default=True
        Whether to apply phantom injection and TLS reconciliation.

    Returns
    -------
    TradeCalibrationResult
        Calibrated general equilibrium model.
    """
    if custom_c is not None or custom_s is not None:
        if not fallback_to_synthetic:
            raise ValueError("Custom dimensions generate synthetic data; set fallback_to_synthetic=True explicitly.")
        if file_path is not None or data_dir is not None:
            raise ValueError("Custom synthetic dimensions cannot be combined with an empirical file or data directory.")
        raw_eur = generate_synthetic_mrio("exiobase", year=year, model=model, seed=seed, custom_c=custom_c, custom_s=custom_s, unit="M_EUR")
        raw_usd = RawIOData(
            countries=raw_eur.countries,
            sectors=raw_eur.sectors,
            intermediate_matrix=raw_eur.intermediate_matrix * eur_to_usd,
            final_demand_matrix=raw_eur.final_demand_matrix * eur_to_usd,
            value_added=raw_eur.value_added * eur_to_usd,
            taxes_less_subsidies=raw_eur.taxes_less_subsidies * eur_to_usd,
            gross_output=raw_eur.gross_output * eur_to_usd,
            fd_categories=raw_eur.fd_categories,
            year=year,
            source=raw_eur.source,
            metadata={**raw_eur.metadata, "exchange_rate_usd_per_eur": eur_to_usd, "original_unit": "M_EUR"},
            unit="M_USD",
        )
        return package_mrio_to_calibration_result(_condense_provider_final_uses(raw_usd, "exiobase"),
                                                  regularize=regularize)

    target: Path | None = None
    if file_path is not None:
        p = Path(file_path)
        if p.is_file():
            target = p
    else:
        candidates = [Path(os.environ["IO_EXIOBASE_DIR"]) ] if os.environ.get("IO_EXIOBASE_DIR") else []
        if data_dir:
            candidates.insert(0, Path(data_dir))
        target = _find_unique_match(candidates, (f"*{model}*{year}*", f"*{year}*{model}*"),
                                    f"EXIOBASE {model} {year} tables")

    if target is None or not target.is_file():
        if fallback_to_synthetic:
            raw_eur = generate_synthetic_mrio("exiobase", year=year, model=model, seed=seed, unit="M_EUR")
            raw_usd = RawIOData(
                countries=raw_eur.countries,
                sectors=raw_eur.sectors,
                intermediate_matrix=raw_eur.intermediate_matrix * eur_to_usd,
                final_demand_matrix=raw_eur.final_demand_matrix * eur_to_usd,
                value_added=raw_eur.value_added * eur_to_usd,
                taxes_less_subsidies=raw_eur.taxes_less_subsidies * eur_to_usd,
                gross_output=raw_eur.gross_output * eur_to_usd,
                fd_categories=raw_eur.fd_categories,
                year=year,
                source=raw_eur.source,
            metadata={**raw_eur.metadata, "exchange_rate_usd_per_eur": eur_to_usd, "original_unit": "M_EUR"},
                unit="M_USD",
            )
            return package_mrio_to_calibration_result(_condense_provider_final_uses(raw_usd, "exiobase"),
                                                      regularize=regularize)
        raise FileNotFoundError(f"EXIOBASE 3 ({model}) table for year {year} not found.")

    native_hint = ("the native EXIOBASE archive (IOT_<year>_<model>.zip with Z.txt, Y.txt, x.txt and "
                   "satellite/F.txt) is read by puremacro.trade.mrio.read_exiobase_native; load_exiobase "
                   "reads only the harmonized positional layout (a headerless tab-separated numeric table "
                   "of 49*S intermediate rows plus 7 value-added rows by 49*S intermediate columns plus "
                   "49*7 final-use columns)")
    import zipfile
    compression = _detect_compression(target)
    if compression == "zip" or target.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(target) as archive:
                members = [name for name in archive.namelist() if not name.endswith("/")]
        except zipfile.BadZipFile as exc:
            raise ValueError(f"{target} is not a readable ZIP archive: {native_hint}") from exc
        native_members = {"z.txt", "y.txt", "x.txt", "f.txt"}
        if len(members) != 1 or Path(members[0]).name.lower() in native_members:
            raise ValueError(f"{target} is a ZIP archive with members {members[:6]}: {native_hint}")
        compression = "zip"
    first_cells = [cell.strip().lower() for cell in _read_first_row(target, "\t", compression)[:3]]
    try:
        [float(cell) for cell in first_cells if cell]
    except ValueError:
        raise ValueError(f"{target} starts with labels {first_cells}: {native_hint}") from None

    df = pd.read_csv(target, sep="\t", header=None, low_memory=False, compression=compression or "infer")
    exio_sectors = EXIOBASE_200_SECTORS if model == "pxp" else EXIOBASE_163_SECTORS
    N = len(EXIOBASE_COUNTRIES)
    S = len(exio_sectors)
    M = N * S
    n_fd_cols = N * len(EXIOBASE_FD_CATEGORIES)
    if df.shape[0] < M + 7 or df.shape[1] < M + n_fd_cols:
        raise ValueError(
            f"{target} has shape {df.shape}; the harmonized EXIOBASE {model} layout needs at least "
            f"({M + 7}, {M + n_fd_cols}): {native_hint}"
        )

    Z = df.iloc[:M, :M].values.astype(np.float64) * eur_to_usd
    F = df.iloc[:M, M : M + n_fd_cols].values.astype(np.float64) * eur_to_usd

    va_raw = df.iloc[M : M + 7, :M].values.astype(np.float64) * eur_to_usd
    VA = np.sum(va_raw, axis=0)
    TLS = np.zeros(M, dtype=np.float64)
    Y = np.sum(Z, axis=1) + np.sum(F, axis=1)

    raw = RawIOData(
        countries=list(EXIOBASE_COUNTRIES),
        sectors=list(exio_sectors),
        intermediate_matrix=Z,
        final_demand_matrix=F,
        value_added=VA,
        taxes_less_subsidies=TLS,
        gross_output=Y,
        fd_categories=list(EXIOBASE_FD_CATEGORIES),
        year=year,
        source=f"EXIOBASE3_{model.upper()}_Harmonized",
        unit="M_USD",
    )
    raw.metadata.update({
        "exchange_rate_usd_per_eur": eur_to_usd, "original_unit": "M_EUR",
        "is_synthetic": False, "file_path": str(target.resolve()),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "schema": "puremacro positional harmonized table (not a native archive reader)",
        "schema_validation": "shape and finite values; native archive parity unverified",
    })
    return package_mrio_to_calibration_result(_condense_provider_final_uses(raw, "exiobase"), regularize=regularize)


def load_wiod(
    year: int = 2014,
    fallback_to_synthetic: bool = False,
    *,
    data_dir: str | Path | None = None,
    file_path: str | Path | None = None,
    custom_c: int | None = None,
    custom_s: int | None = None,
    seed: int = 42,
    regularize: bool = True,
) -> TradeCalibrationResult:
    """Load World Input-Output Database (WIOD) 2016 release table.

    Schema: 44 countries x 56 sectors = 2,464 nodes, 5 final demand categories.

    Accepted file: a WIOD 2016 wide table (``.dta`` or ``.csv``) with columns
    ``Country``, ``IndustryCode`` and ``v<COUNTRY><r>`` value columns
    (``r = 1..56`` intermediate uses, ``57..61`` final uses CONS_h, CONS_np,
    CONS_g, GFCF, INVT), and ``VA``, ``TXSP`` and ``GO`` rows. Other layouts
    (for example the long ``lr_wiod_wiot`` format) raise ``ValueError``.

    Final uses are condensed before calibration to ``C = CONS_h + CONS_np +
    CONS_g``, ``I = GFCF`` (receives the foreign-balance closure) and
    ``Cx = INVT`` (changes in inventories and valuables); the mapping and the
    meaning of each slot are recorded in ``metadata["final_use_mapping"]``
    and ``metadata["final_use_semantics"]``. Cx here is not the OECD
    Cx (DPABR, residents' direct purchases abroad), so bridges that read
    ``("C", "I", "Cx")`` as (C, G, X) must consult the mapping.
    Up to 4.3.0 the five native categories were passed through and the
    closure landed on CONS_np (NPISH).

    Parameters
    ----------
    year : int, default=2014
        Reference year (2000 to 2014).
    fallback_to_synthetic : bool, default=False
        Whether to generate synthetic benchmark data if file is missing.
    data_dir : str or Path, optional
        Directory containing WIOD Stata/Excel files.
    file_path : str or Path, optional
        Exact path to WIOD table file (.dta or .csv).
    custom_c : int, optional
        Override country count for testing.
    custom_s : int, optional
        Override sector count for testing.
    seed : int, default=42
        Deterministic random seed for synthetic fallback.
    regularize : bool, default=True
        Whether to apply phantom injection and TLS reconciliation.

    Returns
    -------
    TradeCalibrationResult
        Calibrated general equilibrium model.
    """
    if custom_c is not None or custom_s is not None:
        if not fallback_to_synthetic:
            raise ValueError("Custom dimensions generate synthetic data; set fallback_to_synthetic=True explicitly.")
        if file_path is not None or data_dir is not None:
            raise ValueError("Custom synthetic dimensions cannot be combined with an empirical file or data directory.")
        raw = generate_synthetic_mrio("wiod", year=year, seed=seed, custom_c=custom_c, custom_s=custom_s, unit="M_USD")
        return package_mrio_to_calibration_result(_condense_provider_final_uses(raw, "wiod"), regularize=regularize)

    target: Path | None = None
    if file_path is not None:
        p = Path(file_path)
        if p.is_file():
            target = p
    else:
        candidates = [Path(os.environ["IO_WIOD_DIR"]) ] if os.environ.get("IO_WIOD_DIR") else []
        if data_dir:
            candidates.insert(0, Path(data_dir))
        target = _find_unique_match(candidates, (f"*{year}*.dta", f"*{year}*.csv"), f"WIOD {year} tables")

    if target is None or not target.is_file():
        if fallback_to_synthetic:
            raw = generate_synthetic_mrio("wiod", year=year, seed=seed, unit="M_USD")
            return package_mrio_to_calibration_result(_condense_provider_final_uses(raw, "wiod"),
                                                      regularize=regularize)
        raise FileNotFoundError(f"WIOD table for year {year} not found.")

    if str(target).endswith(".dta"):
        df = pd.read_stata(target)
    else:
        df = pd.read_csv(target, low_memory=False, compression=_detect_compression(target) or "infer")
    missing_columns = [c for c in ("Country", "IndustryCode") if c not in df.columns]
    if missing_columns:
        raise ValueError(
            f"{target} lacks the WIOD 2016 wide-table columns {missing_columns}; load_wiod expects "
            "Country, IndustryCode and v<COUNTRY><1..61> value columns with VA, TXSP and GO rows"
        )

    all_countries = df["Country"].dropna().unique()
    countries = [str(c) for c in all_countries if str(c) not in ("TOT", "")]
    N = len(countries)
    S = 56
    M = N * S

    int_cols: list[str] = []
    for c in countries:
        for r in range(1, S + 1):
            int_cols.append(f"v{c}{r}")

    fd_cols: list[str] = []
    for c in countries:
        for r in range(57, 62):
            fd_cols.append(f"v{c}{r}")

    sectors = [f"S{r:02d}" for r in range(1, S + 1)]

    Z = df.iloc[:M][int_cols].values.astype(np.float64)
    F = df.iloc[:M][fd_cols].values.astype(np.float64)

    va_mask = (df["IndustryCode"] == "VA")
    txsp_mask = (df["IndustryCode"] == "TXSP")
    go_mask = (df["IndustryCode"] == "GO")

    if not va_mask.any() or not txsp_mask.any() or not go_mask.any():
        raise ValueError("Accounting rows (VA, TXSP, GO) missing in WIOD table.")

    va_idx = df[va_mask].index[0]
    txsp_idx = df[txsp_mask].index[0]
    go_idx = df[go_mask].index[0]

    VA = df.iloc[va_idx][int_cols].values.astype(np.float64)
    TLS = df.iloc[txsp_idx][int_cols].values.astype(np.float64)
    Y = df.iloc[go_idx][int_cols].values.astype(np.float64)

    raw = RawIOData(
        countries=countries,
        sectors=sectors,
        intermediate_matrix=Z,
        final_demand_matrix=F,
        value_added=VA,
        taxes_less_subsidies=TLS,
        gross_output=Y,
        fd_categories=list(WIOD_FD_CATEGORIES),
        year=year,
        source="WIOD_2016",
        unit="M_USD",
    )
    raw.metadata.update({
        "is_synthetic": False, "file_path": str(target.resolve()),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "schema": "puremacro positional harmonized table (not a native archive reader)",
        "schema_validation": "shape and finite values; native archive parity unverified",
    })
    return package_mrio_to_calibration_result(_condense_provider_final_uses(raw, "wiod"), regularize=regularize)


def load_eora(
    year: int = 2015,
    fallback_to_synthetic: bool = False,
    *,
    data_dir: str | Path | None = None,
    file_path: str | Path | None = None,
    custom_c: int | None = None,
    custom_s: int | None = None,
    seed: int = 42,
    regularize: bool = True,
) -> TradeCalibrationResult:
    """Load Eora26 MRIO table with unit harmonization from Thousand USD to Million USD.

    Schema: 189 countries x 26 sectors = 4,914 nodes, 6 final demand categories.

    Accepted file: the puremacro harmonized positional layout, a single
    headerless numeric table (tab-separated ``.txt`` or ``.csv``) with 4,914
    intermediate rows followed by 6 value-added rows (summed), and 4,914
    intermediate columns followed by ``189 * 6`` final-use columns in
    :data:`EORA_FD_CATEGORIES` order per country, in thousand USD.

    Final uses are condensed before calibration to ``C = HFCE + NPISH + GGFC``,
    ``I = GFCF`` (receives the foreign-balance closure) and
    ``Cx = INVNT + ACQ_VAL``; the mapping and the meaning of each slot are
    recorded in ``metadata["final_use_mapping"]`` and
    ``metadata["final_use_semantics"]``. Cx here is not the OECD
    Cx (DPABR, residents' direct purchases abroad), so bridges that read
    ``("C", "I", "Cx")`` as (C, G, X) must consult the mapping.
    Up to 4.3.0 the six native categories were passed through and the
    closure landed on NPISH.

    Parameters
    ----------
    year : int, default=2015
        Reference year.
    fallback_to_synthetic : bool, default=False
        Whether to generate synthetic benchmark data if file is missing.
    data_dir : str or Path, optional
        Directory containing Eora files.
    file_path : str or Path, optional
        Exact path to Eora table file.
    custom_c : int, optional
        Override country count for testing.
    custom_s : int, optional
        Override sector count for testing.
    seed : int, default=42
        Deterministic random seed for synthetic fallback.
    regularize : bool, default=True
        Whether to apply phantom injection and TLS reconciliation.

    Returns
    -------
    TradeCalibrationResult
        Calibrated general equilibrium model.
    """
    scale = 1.0 / 1000.0
    if custom_c is not None or custom_s is not None:
        if not fallback_to_synthetic:
            raise ValueError("Custom dimensions generate synthetic data; set fallback_to_synthetic=True explicitly.")
        if file_path is not None or data_dir is not None:
            raise ValueError("Custom synthetic dimensions cannot be combined with an empirical file or data directory.")
        raw_th = generate_synthetic_mrio("eora", year=year, seed=seed, custom_c=custom_c, custom_s=custom_s, unit="000_USD")
        raw = RawIOData(
            countries=raw_th.countries,
            sectors=raw_th.sectors,
            intermediate_matrix=raw_th.intermediate_matrix * scale,
            final_demand_matrix=raw_th.final_demand_matrix * scale,
            value_added=raw_th.value_added * scale,
            taxes_less_subsidies=raw_th.taxes_less_subsidies * scale,
            gross_output=raw_th.gross_output * scale,
            fd_categories=raw_th.fd_categories,
            year=year,
            source=raw_th.source,
            metadata={**raw_th.metadata, "unit_scale": scale, "original_unit": "000_USD"},
            unit="M_USD",
        )
        return package_mrio_to_calibration_result(_condense_provider_final_uses(raw, "eora"), regularize=regularize)

    target: Path | None = None
    if file_path is not None:
        p = Path(file_path)
        if p.is_file():
            target = p
    else:
        candidates = [Path(os.environ["IO_EORA_DIR"]) ] if os.environ.get("IO_EORA_DIR") else []
        if data_dir:
            candidates.insert(0, Path(data_dir))
        target = _find_unique_match(candidates, (f"*Eora26*{year}*.txt", f"*Eora26*{year}*.csv"),
                                    f"Eora26 {year} tables")

    if target is None or not target.is_file():
        if fallback_to_synthetic:
            raw_th = generate_synthetic_mrio("eora", year=year, seed=seed, unit="000_USD")
            raw = RawIOData(
                countries=raw_th.countries,
                sectors=raw_th.sectors,
                intermediate_matrix=raw_th.intermediate_matrix * scale,
                final_demand_matrix=raw_th.final_demand_matrix * scale,
                value_added=raw_th.value_added * scale,
                taxes_less_subsidies=raw_th.taxes_less_subsidies * scale,
                gross_output=raw_th.gross_output * scale,
                fd_categories=raw_th.fd_categories,
                year=year,
                source=raw_th.source,
            metadata={**raw_th.metadata, "unit_scale": scale, "original_unit": "000_USD"},
                unit="M_USD",
            )
            return package_mrio_to_calibration_result(_condense_provider_final_uses(raw, "eora"),
                                                      regularize=regularize)
        raise FileNotFoundError(f"Eora26 table for year {year} not found.")

    sep = "\t" if ".txt" in [suffix.lower() for suffix in target.suffixes] else ","
    df = pd.read_csv(target, sep=sep, header=None, low_memory=False,
                     compression=_detect_compression(target) or "infer")

    N = len(EORA_189_COUNTRIES)
    S = len(EORA_26_SECTORS)
    M = N * S
    n_fd_cols = N * len(EORA_FD_CATEGORIES)
    if df.shape[0] < M + 6 or df.shape[1] < M + n_fd_cols:
        raise ValueError(
            f"{target} has shape {df.shape}; the harmonized Eora26 layout needs at least "
            f"({M + 6}, {M + n_fd_cols}) (headerless numeric table, thousand USD)"
        )

    Z = df.iloc[:M, :M].values.astype(np.float64) * scale
    F = df.iloc[:M, M : M + n_fd_cols].values.astype(np.float64) * scale
    va_raw = df.iloc[M : M + 6, :M].values.astype(np.float64) * scale
    VA = np.sum(va_raw, axis=0)
    TLS = np.zeros(M, dtype=np.float64)
    Y = np.sum(Z, axis=1) + np.sum(F, axis=1)

    raw = RawIOData(
        countries=list(EORA_189_COUNTRIES),
        sectors=list(EORA_26_SECTORS),
        intermediate_matrix=Z,
        final_demand_matrix=F,
        value_added=VA,
        taxes_less_subsidies=TLS,
        gross_output=Y,
        fd_categories=list(EORA_FD_CATEGORIES),
        year=year,
        source="Eora26_Harmonized",
        unit="M_USD",
    )
    raw.metadata.update({
        "is_synthetic": False, "file_path": str(target.resolve()),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "schema": "puremacro positional harmonized table (not a native archive reader)",
        "schema_validation": "shape and finite values; native archive parity unverified",
    })
    return package_mrio_to_calibration_result(_condense_provider_final_uses(raw, "eora"), regularize=regularize)


def load_oecd_icio_granular(
    year: int = 2019,
    fallback_to_synthetic: bool = False,
    *,
    data_dir: str | Path | None = None,
    file_path: str | Path | None = None,
    custom_c: int | None = None,
    custom_s: int | None = None,
    seed: int = 42,
    regularize: bool = True,
) -> TradeCalibrationResult:
    """Load granular OECD ICIO 45-sector transaction dataset.

    Reads the OECD 2023 regular native labeled CSV or ZIP: 77 economies,
    45 activities, TLS/VA/OUT margins and six final uses. Final demand is
    mapped to C=(HFCE, NPISH, GGFC), I=(GFCF, INVNT), Cx=DPABR before
    calibration, so the foreign-balance closure applies to investment. The
    synthetic teaching branches (``custom_c``/``custom_s`` and
    ``fallback_to_synthetic``) use the same mapping, so they share the real
    path's closure and three final-use categories (up to 4.3.0 they kept the six
    native categories and put the closure on NPISH). Provider output margins,
    pre-repair accounting discrepancies and the per-category count of negative
    condensed final-use cells (``negative_final_demand_cells``; signed INVNT can
    make I negative, which ``accounting="consistent"`` rejects) are retained in
    metadata. Native 2025 and extended archives are unsupported.

    Parameters
    ----------
    year : int, default=2019
        Reference year (1995 to 2020).
    fallback_to_synthetic : bool, default=False
        Explicitly opt into generated teaching data if the archive is missing.
    data_dir : str or Path, optional
        Directory containing OECD ICIO CSV files.
    file_path : str or Path, optional
        Exact path to an OECD 2023 regular CSV table or ZIP archive.
    custom_c : int, optional
        Override country count for testing.
    custom_s : int, optional
        Override sector count for testing.
    seed : int, default=42
        Deterministic random seed for synthetic fallback.
    regularize : bool, default=True
        Whether to apply phantom injection and TLS reconciliation.

    Returns
    -------
    TradeCalibrationResult
        Calibrated general equilibrium model.
    """
    if custom_c is not None or custom_s is not None:
        if not fallback_to_synthetic:
            raise ValueError("Custom dimensions generate synthetic data; set fallback_to_synthetic=True explicitly.")
        if file_path is not None or data_dir is not None:
            raise ValueError("Custom synthetic dimensions cannot be combined with an empirical file or data directory.")
        from ._oecd_icio import condense_final_demand
        raw = generate_synthetic_mrio("oecd", year=year, seed=seed, custom_c=custom_c, custom_s=custom_s, unit="M_USD")
        return package_mrio_to_calibration_result(condense_final_demand(raw), regularize=regularize)

    target: Path | None = None
    if file_path is not None:
        p = Path(file_path)
        if p.is_file():
            target = p
    else:
        candidates: list[Path] = []
        if data_dir:
            candidates.append(Path(data_dir))
        env_dir = os.environ.get("IO_OECD_DIR")
        if env_dir:
            candidates.append(Path(env_dir))
        for d in candidates:
            if d.is_dir():
                for name in (f"{year}_SML.csv",):
                    candidate_file = d / name
                    if candidate_file.is_file():
                        target = candidate_file
                        break
            if target is not None:
                break

    if target is None or not target.is_file():
        if fallback_to_synthetic:
            from ._oecd_icio import condense_final_demand
            raw = generate_synthetic_mrio("oecd", year=year, seed=seed, unit="M_USD")
            return package_mrio_to_calibration_result(condense_final_demand(raw), regularize=regularize)
        raise FileNotFoundError(f"OECD ICIO granular CSV for year {year} not found.")

    from ._oecd_icio import read_native, condense_final_demand
    raw = condense_final_demand(read_native(target, year))
    return package_mrio_to_calibration_result(raw, regularize=regularize)


# ---------------------------------------------------------------------------
# Accounting & Balance Invariant Helpers
# ---------------------------------------------------------------------------


def compute_trade_balances(
    Z: np.ndarray,
    F: np.ndarray,
    nc: int,
    ns: int,
    nfd: int,
) -> np.ndarray:
    """Compute multilateral net foreign surplus vector XN across economies."""
    M = nc * ns
    F_2d = F.reshape(M, nc * nfd) if F.ndim == 3 else F
    T_inter = np.sum(Z.reshape(nc, ns, nc, ns), axis=(1, 3))
    T_fd = np.sum(F_2d.reshape(nc, ns, nc, nfd), axis=(1, 3))

    np.fill_diagonal(T_inter, 0.0)
    np.fill_diagonal(T_fd, 0.0)

    X0 = np.sum(T_inter, axis=1)
    M0 = np.sum(T_inter, axis=0)
    XFD = np.sum(T_fd, axis=1)
    MFD = np.sum(T_fd, axis=0)

    return (X0 + XFD - M0 - MFD)


def verify_zero_leakage(XN: np.ndarray, tol: float = 1e-10) -> bool:
    """Verify that multilateral trade balances sum to identically zero."""
    leakage = float(np.abs(np.sum(XN)))
    scale = float(np.sum(np.abs(XN)))
    if scale > 0:
        return (leakage / scale) <= tol or leakage <= 1e-8
    return leakage <= tol


def verify_accounting_invariants(calib: TradeCalibrationResult) -> dict[str, Any]:
    """Verify core accounting and balance invariants on a calibrated trade model.

    Checks:
    1. Gross output sales match column outlays: sum_i Z_ij + VA_j + TLS_j == Y_j
    2. Consumer budget balance: sum_f theta_c^f == 1.0 for each country c
    3. Multilateral zero-leakage trade balance: sum_c XN_c == 0.0
    4. Non-negative factor endowments: KT > 0 and LT > 0
    """
    data = calib.data_calibra
    nc = calib.nc
    ns = calib.ns
    n_ind = nc * ns

    # Outlays vs Sales
    col_outlays = np.sum(data[:, :n_ind], axis=0)
    sales = np.sum(data[:n_ind, :], axis=1)
    outlays_sales_err = float(np.max(np.abs(col_outlays - sales)))

    # Zero leakage: sum_c invforT_c == 0
    xn_sum = float(np.sum(calib.invforT))
    total_xn = float(np.sum(np.abs(calib.invforT)))
    zero_leakage = bool((abs(xn_sum) <= 1e-8) or (total_xn > 0 and (abs(xn_sum) / total_xn) <= 1e-10))

    # Budget balance: theta sums to 1.0 for each country
    theta_country_sums = np.sum(calib.theta, axis=1).flatten()
    budget_err = float(np.max(np.abs(theta_country_sums - 1.0)))

    # Factor non-negativity
    factors_positive = bool(np.all(calib.KT > 0) and np.all(calib.LT > 0))

    valid = bool(
        outlays_sales_err <= 1e-9 * float(np.max(sales))
        and zero_leakage
        and budget_err <= 1e-9
        and factors_positive
    )

    return {
        "valid": valid,
        "max_outlays_sales_error": outlays_sales_err,
        "trade_leakage_sum": xn_sum,
        "zero_leakage_certified": zero_leakage,
        "max_budget_error": budget_err,
        "factors_positive": factors_positive,
    }


__all__ = [
    "EU_COUNTRY_CODES",
    "RAW_45_SECTOR_CODES",
    "RAW_45_SECTOR_NAMES",
    "CANONICAL_FINAL_DEMAND_CODES",
    "RAW_6_FINAL_DEMAND_CODES",
    "FIGARO_COUNTRIES",
    "FIGARO_64_SECTORS",
    "FIGARO_FD_CATEGORIES",
    "EXIOBASE_COUNTRIES",
    "EXIOBASE_163_SECTORS",
    "EXIOBASE_200_SECTORS",
    "EXIOBASE_FD_CATEGORIES",
    "WIOD_44_COUNTRIES",
    "WIOD_56_SECTORS",
    "WIOD_FD_CATEGORIES",
    "EORA_189_COUNTRIES",
    "EORA_26_SECTORS",
    "EORA_FD_CATEGORIES",
    "OECD_77_COUNTRIES",
    "OECD_45_SECTORS",
    "OECD_FD_CATEGORIES",
    "get_eu_country_codes",
    "get_raw_45_sector_codes",
    "get_raw_45_sector_names",
    "get_final_demand_codes",
    "get_raw_final_demand_codes",
    "ICIOData",
    "RawIOData",
    "load_icio_data",
    "synthetic_icio_table",
    "synthetic_codes",
    "generate_synthetic_mrio",
    "package_mrio_to_calibration_result",
    "load_figaro",
    "load_exiobase",
    "load_wiod",
    "load_eora",
    "load_oecd_icio_granular",
    "compute_trade_balances",
    "verify_zero_leakage",
    "verify_accounting_invariants",
]
