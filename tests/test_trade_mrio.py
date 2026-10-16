"""Tests for ``puremacro.trade.mrio``: provenance, native readers, regularization and aggregation.

Analytic fixtures are hand-designed (``tests/fixtures/trade/mrio``) and every
expected array below is derived by hand from the fixture text; the other tables
are seeded random or synthetic tables. Everything runs offline.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import warnings
import zipfile
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import puremacro.trade.data as trade_data
from puremacro.trade import mrio as m
from puremacro.trade.calibration import calibrate_trade_model
from puremacro.trade.data import (
    RAW_45_SECTOR_CODES,
    RawIOData,
    package_mrio_to_calibration_result,
    synthetic_icio_table,
    verify_accounting_invariants,
)

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures" / "trade" / "mrio"
FIGARO_FIXTURE = FIXTURES / "figaro_2c2s_26ed.csv"
EXIO_FIXTURE = FIXTURES / "exiobase_2r3s_ixi"


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def toy_table(**overrides):
    """Two countries, two industries, a signed inventory drawdown."""
    z = sparse.csr_matrix([[1, .1, .2, 0], [.3, 1, 0, .1], [.1, 0, 2, .2], [0, .2, .1, 1]])
    C = np.array([[4., .5], [1., 1.], [.4, 5.], [1., 3.]])
    G = np.array([[1., .1], [2., .2], [.1, 1.5], [.2, 2.]])
    X = np.array([[.1, .2], [0, 0], [.3, .2], [0, 0]])
    V = np.array([[-.4, 0], [.2, .1], [.1, -.2], [0, 0]])
    VAL = np.array([[.02, 0], [0, .01], [0, 0], [0, 0]])
    F = np.stack([C, G, X, V, VAL], axis=2)
    output = np.asarray(z.sum(1)).ravel() + F.sum(axis=(1, 2))
    taxes = np.array([.1, -.1, .2, .1])
    va = output - np.asarray(z.sum(0)).ravel() - taxes
    prodtax = np.array([.02, .04, .01, .03])
    kw = dict(
        Z=z, F=F, VA=va, TLS=taxes, TFD=np.array([[.2, .1, 0, -.1, 0], [.4, .3, 0, .1, 0]]),
        country_codes=("A", "B"), sector_codes=("goods", "services"), fd_codes=m.FINAL_USE_CATEGORIES,
        output=output, production_taxes=prodtax, labor_compensation=.6 * (va - prodtax),
        operating_surplus=.4 * (va - prodtax), merchandise_mask=np.array([True, False]),
        metadata={"dataset": "synthetic_unit_fixture", "units": "synthetic monetary units"},
    )
    kw.update(overrides)
    return m.MRIOTable.from_arrays(**kw)


def random_table(seed=0, n=3, s=4, *, sectors=None, countries=None, fd_codes=("C", "G", "X", "V"),
                 z_scale=0.05, negative_inventories=True):
    """Balanced random table with strictly positive value added (no floors, no phantoms)."""
    rng = np.random.default_rng(seed)
    M = n * s
    k = len(fd_codes)
    Z = rng.uniform(0.0, z_scale, (M, M))
    F = rng.uniform(0.2, 1.0, (M, n, k))
    if "X" in fd_codes:
        F[:, :, fd_codes.index("X")] *= 0.1
    if "V" in fd_codes and negative_inventories:
        F[:, :, fd_codes.index("V")] = rng.uniform(-0.05, 0.05, (M, n))
    y = Z.sum(1) + F.sum((1, 2))
    TLS = 0.03 * y
    VA = y - Z.sum(0) - TLS
    assert np.all(VA > 0)
    TFD = 0.04 * F.sum(0)
    ccodes = countries or tuple(["USA", "CHN", "ROW", "DEU", "JPN"][:n])
    scodes = sectors or tuple(f"S{i}" for i in range(s))
    return m.MRIOTable.from_arrays(Z, F, VA, TLS, TFD, country_codes=ccodes, sector_codes=scodes, fd_codes=fd_codes,
                                   reference_country=ccodes[-1], metadata={"dataset": f"random{seed}", "year": 2019})


def random_45(seed=7, n=3):
    """A balanced 45-industry table with the OECD registry and goods mask."""
    t = random_table(seed, n, 45, sectors=RAW_45_SECTOR_CODES, z_scale=0.02)
    return replace(t, merchandise_mask=m.goods_mask(RAW_45_SECTOR_CODES))


def _maxdiff(a, b):
    if sparse.issparse(a) or sparse.issparse(b):
        d = abs(sparse.csr_matrix(a) - sparse.csr_matrix(b))
        return float(d.max()) if d.nnz else 0.0
    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def _read_exio_dir(core=EXIO_FIXTURE, **kw):
    """Read an extracted EXIOBASE directory without its archive (warned as unauthenticated)."""
    with pytest.warns(RuntimeWarning, match="without its archive"):
        return m.read_exiobase_native(core, **kw)


def _write_exio_zip(target, core=EXIO_FIXTURE, prefix="IOT_2019_ixi/"):
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(core.rglob("*")):
            if p.is_file():
                z.write(p, prefix + p.relative_to(core).as_posix())
    return target


def _tiny_oecd_csv(directory, name="2019_SML.csv"):
    """A 2 x 2 table in the OECD SML layout; returns (path, Z, F6, VA, TLS, TFD6, Y)."""
    countries, sectors = ("AAA", "BBB"), ("A01_02", "D")
    nodes = [f"{c}_{s}" for c in countries for s in sectors]
    fd = [f"{c}_{f}" for c in countries for f in m.OECD_FD_CODES]
    Z = np.array([[1, 2, 0.5, 0], [0.5, 3, 0, 1], [0.2, 0, 4, 1], [0, 0.3, 0.6, 2.]])
    F6 = np.zeros((4, 2, 6))
    F6[0] = [[5, 0.2, 1, 0.5, -0.1, 0.3], [0.4, 0, 0, 0.1, 0, 0]]
    F6[1] = [[2, 0.1, 0.5, 3, 0.2, 0], [0.1, 0, 0.2, 0.3, 0, 0]]
    F6[2] = [[0.3, 0, 0, 0.2, 0, 0], [6, 0.1, 1.5, 1, 0.1, 0.4]]
    F6[3] = [[0.1, 0, 0, 0.1, 0, 0], [3, 0.2, 0.9, 2, -0.2, 0]]
    Y = Z.sum(1) + F6.sum((1, 2))
    TLS = 0.05 * Y
    VA = Y - Z.sum(0) - TLS
    TFD6 = 0.02 * F6.sum(0)
    top = pd.DataFrame(np.hstack([Z, F6.reshape(4, 12), Y[:, None]]), index=nodes, columns=nodes + fd + ["OUT"])
    bottom = pd.DataFrame([np.r_[TLS, TFD6.ravel(), 0.0], np.r_[VA, np.zeros(12), 0.0], np.r_[Y, np.zeros(12), 0.0]],
                          index=["TLS", "VA", "OUT"], columns=top.columns)
    frame = pd.concat([top, bottom])
    frame.index.name = "V1"
    path = pathlib.Path(directory) / name
    frame.to_csv(path)
    return path, countries, sectors, Z, F6, VA, TLS, TFD6, Y


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_identify_source_matches_hashlib_and_rejects_changes(self, tmp_path):
        src = tmp_path / "source.csv"
        src.write_bytes(b"not an empirical fixture\n")
        sha = hashlib.sha256(src.read_bytes()).hexdigest()
        md5 = hashlib.md5(src.read_bytes()).hexdigest()
        rec = m.identify_source(src, expected_sha256=sha, expected_md5=md5)
        assert (rec.sha256, rec.md5, rec.bytes, rec.status) == (sha, md5, 25, "verified")
        assert rec.to_dict()["path"] == str(src.resolve())
        assert m.identify_source(src).status == "hashed"
        src.write_bytes(b"changed contents\n")
        with pytest.raises(m.MRIOIntegrityError, match="SHA-256 mismatch"):
            m.identify_source(src, expected_sha256=sha)
        with pytest.raises(m.MRIOIntegrityError, match="MD5 mismatch"):
            m.identify_source(src, expected_md5=md5)
        with pytest.raises(FileNotFoundError, match="Authenticated native source"):
            m.identify_source(tmp_path / "absent.csv")

    def test_source_record_renderers(self, tmp_path):
        src = tmp_path / "s.csv"
        src.write_bytes(b"abc\n")
        rec = m.identify_source(src)
        df = rec.to_dataframe()
        assert df.loc["status", "Value"] == "hashed" and df.loc["bytes", "Value"] == 4
        assert rec.summary().startswith("s.csv: hashed, 4 bytes")
        assert rec.to_markdown() and "tabular" in rec.to_latex() and "#table" in rec.to_typst()

    def test_verify_zip_member_detects_size_and_crc_mismatch(self, tmp_path):
        member = tmp_path / "x.txt"
        member.write_text("region\tsector\tindout\nAT\ta\t20.15\n", encoding="utf-8")
        archive = tmp_path / "a.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.write(member, "IOT/x.txt")
        rec = m.verify_zip_member(archive, "IOT/x.txt")
        assert rec.status == "verified" and rec.archive_member == "IOT/x.txt" and rec.crc32 is not None
        assert rec.sha256 == hashlib.sha256(member.read_bytes()).hexdigest()
        assert m.verify_zip_member(archive, "IOT/x.txt", member).sha256 == rec.sha256
        member.write_text("region\tsector\tindout\nAT\ta\t20.16\n", encoding="utf-8")
        with pytest.raises(m.MRIOIntegrityError, match="CRC-32"):
            m.verify_zip_member(archive, "IOT/x.txt", member)
        member.write_text("short\n", encoding="utf-8")
        with pytest.raises(m.MRIOIntegrityError, match="bytes"):
            m.verify_zip_member(archive, "IOT/x.txt", member)
        with pytest.raises(m.MRIOIntegrityError, match="no member"):
            m.verify_zip_member(archive, "IOT/missing.txt")

    def test_check_oecd_source_registries(self, tmp_path):
        path = tmp_path / "2019_SML.csv"
        path.write_text("V1,AAA_A\nAAA_A,1\n", encoding="utf-8")
        md5 = hashlib.md5(path.read_bytes()).hexdigest()
        # The real registries are facts about other files: this digest is unknown to them.
        assert md5 not in m.OECD_KNOWN_CORRUPTED_MD5 and md5 not in m.OECD_ICIO_MD5.values()
        with pytest.warns(RuntimeWarning, match="not a whitelisted"):
            rec = m.check_oecd_source(path)
        assert rec.status == "unknown_edition"
        # Test-only registries exercise the refusal and whitelist paths.
        with pytest.raises(m.MRIOIntegrityError, match="lost decimal points"):
            m.check_oecd_source(path, corrupted={md5: m.OECD_KNOWN_CORRUPTED_MD5["d1b887aaafa54ab3f28fde78fcd21cdf"]})
        rec = m.check_oecd_source(path, year=2019, whitelist={2019: md5})
        assert rec.status == "whitelisted" and "2019_SML.csv" in rec.note
        with pytest.raises(m.MRIOIntegrityError, match="not the requested year"):
            m.check_oecd_source(path, year=2020, whitelist={2019: md5})
        archive = tmp_path / "2016-2020_SML.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.write(path, "2019_SML.csv")
        rec = m.check_oecd_source(archive, year=2019, whitelist={2019: md5})
        assert rec.status == "whitelisted" and rec.archive_member == "2019_SML.csv"
        with pytest.raises(ValueError, match="year is required"):
            m.check_oecd_source(archive)
        assert m.OECD_ICIO_MD5 == {2019: "28cba31491177955445051d459053744", 2020: "d3e0f4979d85d6c0bb7cf4c43e324287"}
        assert list(m.OECD_KNOWN_CORRUPTED_MD5) == ["d1b887aaafa54ab3f28fde78fcd21cdf"]


# ---------------------------------------------------------------------------
# Concordances and goods masks
# ---------------------------------------------------------------------------


class TestConcordance:
    def test_build_validation(self):
        with pytest.raises(ValueError, match="1-based"):
            m.Concordance.build("x", ("a", "b"), (0, 1))
        with pytest.raises(ValueError, match="entries"):
            m.Concordance.build("x", ("a", "b"), (1,))
        with pytest.raises(ValueError, match="empty"):
            m.Concordance.build("x", ("a", "b"), (1, 3))
        with pytest.raises(ValueError, match="coarse codes"):
            m.Concordance.build("x", ("a", "b"), (1, 2), ("g1",))
        with pytest.raises(ValueError, match="unique"):
            m.Concordance.build("x", ("a", "a"), (1, 1))
        c = m.Concordance.build("x", ("a", "b", "c"), (1, 2, 1))
        assert c.coarse_codes == ("G01", "G02") and c.members("G01") == ("a", "c") and c.group_of("b") == "G02"

    def test_from_mapping_identity_and_compose(self):
        c = m.Concordance.from_mapping("map", {"a": "g", "b": "h", "c": "g"})
        assert c.coarse_codes == ("g", "h") and c.groups == (1, 2, 1)
        with pytest.raises(ValueError, match="lacks"):
            m.Concordance.from_mapping("map", {"a": "g"}, fine_codes=("a", "b"))
        ident = m.Concordance.identity(("a", "b"))
        assert ident.groups == (1, 2) and ident.coarse_codes == ("a", "b")
        outer = m.Concordance.from_mapping("outer", {"g": "all", "h": "all"})
        composed = c.compose(outer)
        assert composed.groups == (1, 1, 1) and composed.coarse_codes == ("all",)
        with pytest.raises(ValueError, match="fine_codes"):
            c.compose(c)

    def test_projection_matches_kron_and_coarse_mask_rules(self):
        c = m.Concordance.build("x", ("a", "b", "c"), (1, 2, 1))
        C = c.matrix
        assert C.shape == (3, 2) and np.array_equal(C, [[1, 0], [0, 1], [1, 0]])
        P = c.projection(2)
        assert np.array_equal(P, np.kron(np.eye(2), C.T))
        assert np.array_equal(c.projection(2, sparse_matrix=True).toarray(), P)
        mask = np.array([True, False, False])
        assert c.coarse_mask(mask, "all").tolist() == [False, False]
        assert c.coarse_mask(mask, "any").tolist() == [True, False]
        assert c.coarse_mask(np.array([True, False, True]), "majority").tolist() == [True, False]
        with pytest.raises(ValueError):
            c.coarse_mask(mask, "none")
        df = c.to_dataframe()
        assert list(df.index) == ["a", "b", "c"] and df.loc["c", "coarse"] == "G01"
        assert "| fine" in c.to_markdown() and "tabular" in c.to_latex() and "#table" in c.to_typst()
        assert c.summary() == "concordance 'x': 3 fine codes -> 2 groups (sizes 2, 1)"

    def test_registered_concordance_isic_sections(self):
        assert set(m.CONCORDANCES) == {"isic_section"}
        isic = m.CONCORDANCES["isic_section"]
        assert isic.fine_codes == RAW_45_SECTOR_CODES and isic.n_coarse == 20
        assert isic.groups == m.ISIC_SECTION
        assert isic.members("A") == ("A01_02", "A03") and isic.members("C") == RAW_45_SECTOR_CODES[5:22]
        assert isic.members("H") == ("H49", "H50", "H51", "H52", "H53")
        assert isic.members("J") == ("J58T60", "J61", "J62_63") and isic.members("D") == ("D",)
        assert m.ISIC_SECTION_NAMES["F"] == "Construction"
        assert m.ISIC_SECTION_CODES == tuple("ABCDEFGHIJKLMNOPQRST")
        assert isic.coarse_codes == m.ISIC_SECTION_CODES

    def test_goods_mask_classifications(self):
        oecd = m.goods_mask(RAW_45_SECTOR_CODES)
        assert oecd.sum() == 22 and oecd[:22].all() and not oecd[22:].any()
        assert m.goods_mask(m.ISIC_SECTION_CODES).tolist() == [True, True, True] + [False] * 17
        fig = m.goods_mask(m.FIGARO_NATIVE_64_SECTORS)
        assert fig.sum() == 22 and not fig[m.FIGARO_NATIVE_64_SECTORS.index("C33")]
        assert m.goods_mask(("i01.a", "i01.w.1", "i37", "i38", "i60.2")).tolist() == [True, False, True, False, False]
        # Utilities (D, E) are not merchandise.
        assert m.goods_mask(("A", "C", "D", "E", "U")).tolist() == [True, True, False, False, False]
        with pytest.raises(ValueError, match="unknown"):
            m.goods_mask(("foo", "bar"))
        # Composite labels without a documented content are in no registry: an explicit mask is needed.
        with pytest.raises(ValueError, match="merchandise_mask"):
            m.goods_mask(("GOODS1", "GOODS2", "MIXED"))
        with pytest.raises(ValueError, match="unknown classification"):
            m.goods_mask(("GOODS1", "GOODS2", "MIXED"), "custom11")
        # A code outside the named registry is refused instead of silently counted as a service.
        for codes, kind in ((("A01_02", "GOODS1"), "oecd45"), (("A01", "C99"), "figaro64"),
                            (("A", "ZZ"), "isic_section")):
            with pytest.raises(ValueError, match=f"outside the {kind}"):
                m.goods_mask(codes, kind)


# ---------------------------------------------------------------------------
# The table contract
# ---------------------------------------------------------------------------


class TestTableContract:
    def test_from_arrays_validation(self):
        t = toy_table()
        with pytest.raises(ValueError, match="Z shape"):
            toy_table(Z=np.eye(3))
        with pytest.raises(ValueError, match="F shape"):
            toy_table(F=np.zeros((4, 3, 5)))
        with pytest.raises(ValueError, match="canonical order"):
            toy_table(fd_codes=("G", "C", "X", "V", "VAL"))
        with pytest.raises(ValueError, match="subset"):
            toy_table(fd_codes=("C", "G", "X", "V", "OTHER"))
        with pytest.raises(m.MRIOIntegrityError, match="nonnegative"):
            toy_table(Z=-np.eye(4))
        with pytest.raises(m.MRIOIntegrityError, match="NaN"):
            toy_table(output=np.array([1., np.nan, 2., 3.]))
        F = np.array(t.F)
        F[0, 0, 1] = -1.0
        with pytest.raises(m.MRIOIntegrityError, match="negative fixed investment"):
            toy_table(F=F)
        assert toy_table(F=F, strict=False).accounting_report().negative_investment_entries == 1
        with pytest.raises(ValueError, match="together"):
            toy_table(operating_surplus=None)
        with pytest.raises(ValueError, match="reference_country"):
            toy_table(reference_country="ZZZ")
        with pytest.raises(ValueError, match="unique"):
            toy_table(country_codes=("A", "A"))
        with pytest.raises(ValueError, match="merchandise_mask"):
            toy_table(merchandise_mask=np.array([1, 2]))
        with pytest.raises(ValueError, match="TFD shape"):
            toy_table(TFD=np.zeros((2, 4)))

    def test_arrays_are_read_only_copies(self):
        F = np.ones((4, 2, 5))
        t = toy_table(F=F, strict=False)
        F[0, 0, 0] = 100.0
        assert t.F[0, 0, 0] == 1.0
        with pytest.raises(ValueError):
            t.F[0, 0, 0] = 5.0
        with pytest.raises(ValueError):
            t.VA[0] = 5.0
        assert t.is_sparse and not t.Z.data.flags.writeable
        dense = toy_table(Z=np.eye(4))
        assert not dense.is_sparse and not dense.Z.flags.writeable and np.array_equal(dense.Z_dense, np.eye(4))

    def test_toy_accounts(self):
        d = toy_table()
        assert d.fd_codes == m.FINAL_USE_CATEGORIES == ("C", "G", "X", "V", "VAL")
        assert d.final_use("G")[1, 0] == 2 and d.final_use("X")[1, 0] == 0
        assert d.final_use("V")[0, 0] == -.4 and d.final_use("VAL")[0, 0] == .02
        np.testing.assert_array_equal(d.country_of_cell, [0, 0, 1, 1])
        assert d.cell_labels == ("A_goods", "A_services", "B_goods", "B_services")
        rep = d.accounting_report()
        assert rep.row_output_max_absolute_gap < 1e-13 and rep.column_output_max_absolute_gap < 1e-13
        assert rep.factor_decomposition_max_absolute_gap < 1e-13
        assert rep.negative_inventory_entries == 2 and rep.zero_output_cells == 0
        assert d.transformations == () and d.is_raw
        np.testing.assert_allclose(d.factor_VA, d.VA - d.production_taxes)
        np.testing.assert_allclose(d.capital_share, .4)
        assert d.capital_share_available.all()
        assert not np.allclose(d.operating_surplus / d.VA, d.capital_share)
        np.testing.assert_allclose(d.final_demand, d.F.sum(axis=2))
        assert rep.final_category_totals["V"] == pytest.approx(-.2)

    def test_missing_and_negative_factors_remain_unavailable_without_imputation(self):
        d = toy_table()
        missing = toy_table(labor_compensation=None, operating_surplus=None, production_taxes=None)
        assert np.isnan(missing.capital_share).all() and not missing.capital_share_available.any()
        np.testing.assert_allclose(missing.factor_VA, missing.VA)
        cap = np.array(d.operating_surplus)
        cap[0] = -1
        negative = toy_table(operating_surplus=cap)
        assert negative.operating_surplus[0] == -1 and np.isnan(negative.capital_share[0])
        assert not negative.capital_share_available[0]
        assert negative.accounting_report().factor_decomposition_max_absolute_gap > 0

    def test_zero_output_cell_is_preserved_without_phantom(self):
        d = toy_table()
        z = d.Z.toarray()
        z[0] = 0
        z[:, 0] = 0
        changes = {"Z": z}
        for key in ("F", "VA", "TLS", "production_taxes", "labor_compensation", "operating_surplus", "output"):
            arr = np.array(getattr(d, key))
            arr[0] = 0
            changes[key] = arr
        empty = toy_table(**changes)
        assert not empty.active_mask[0] and empty.output[0] == empty.VA[0] == 0
        assert empty.accounting_report().zero_output_cells == 1

    def test_source_rounding_and_signed_inventory_are_reported_without_repair(self):
        d = toy_table()
        output = np.array(d.output)
        output[0] = -.001
        F = np.array(d.F)
        F[0, 0, 0] = -.001
        raw = toy_table(output=output, F=F)
        assert raw.output[0] == -.001 and raw.final_use("C")[0, 0] == -.001
        rep = raw.accounting_report()
        assert rep.negative_output_cells == rep.negative_consumption_entries == 1
        assert rep.row_output_max_absolute_gap > 0

    def test_net_exports_sum_to_zero_and_reference_absorbs_roundoff(self):
        t = random_table(3, 4, 3)
        xn = t.net_exports()
        assert abs(xn.sum()) < 1e-12
        B = t.bilateral_flows()
        np.testing.assert_allclose(xn[:-1], (B.sum(1) - B.sum(0))[:-1], atol=1e-12)
        sp = replace(t, Z=sparse.csr_matrix(t.Z))
        np.testing.assert_allclose(sp.net_exports(), xn, atol=1e-12)
        np.testing.assert_allclose(sp.bilateral_flows(), B, atol=1e-12)

    def test_from_raw_layouts(self):
        Z = np.array([[1, 2, 0.5, 0], [0.5, 3, 0, 1], [0.2, 0, 4, 1], [0, 0.3, 0.6, 2.]])
        F6 = np.abs(np.random.default_rng(1).normal(size=(4, 2, 6)))
        F6[:, :, 4] -= 0.3
        Y = Z.sum(1) + F6.sum((1, 2))
        raw = RawIOData(countries=["AAA", "BBB"], sectors=["A01_02", "D"], intermediate_matrix=Z,
                        final_demand_matrix=F6.reshape(4, 12), value_added=Y - Z.sum(0), taxes_less_subsidies=np.zeros(4),
                        gross_output=Y, fd_categories=list(m.OECD_FD_CODES), year=2019, source="test",
                        taxes_less_subsidies_fd=np.zeros(12))
        t = m.MRIOTable.from_raw(raw)
        assert t.fd_codes == ("C", "G", "X", "V")
        np.testing.assert_allclose(t.final_use("C"), F6[:, :, :3].sum(2))
        np.testing.assert_allclose(t.final_use("V"), F6[:, :, 4])
        np.testing.assert_allclose(t.final_use("X"), F6[:, :, 5])
        assert set(t.metadata["final_use_components"]) == set(m.OECD_FD_CODES)
        # Processed C, I, Cx with a negative investment cell.
        F3 = np.stack([F6[:, :, :3].sum(2), F6[:, :, 3] + F6[:, :, 4], F6[:, :, 5]], axis=2)
        assert (F3[:, :, 1] < 0).any()
        processed = replace(raw, final_demand_matrix=F3.reshape(4, 6), fd_categories=["C", "I", "Cx"],
                         taxes_less_subsidies_fd=np.zeros(6))
        with pytest.raises(m.MRIOIntegrityError, match="to_inventory"):
            m.MRIOTable.from_raw(processed)
        moved = m.MRIOTable.from_raw(processed, negative_investment="to_inventory")
        assert moved.fd_codes == ("C", "G", "X", "V") and moved.final_use("G").min() == 0
        np.testing.assert_allclose(moved.final_use("G") + moved.final_use("V"), F3[:, :, 1])
        assert any("negative investment" in x for x in moved.transformations)
        with pytest.raises(ValueError, match="negative_investment"):
            m.MRIOTable.from_raw(processed, negative_investment="clip")
        native = replace(raw, final_demand_matrix=F6[:, :, :2].reshape(4, 4), fd_categories=["C", "V"],
                         taxes_less_subsidies_fd=None)
        assert m.MRIOTable.from_raw(native).fd_codes == ("C", "V")
        with pytest.raises(ValueError, match="fd_mapping"):
            m.MRIOTable.from_raw(replace(raw, fd_categories=["h", "n", "g", "i", "s", "d"]))
        custom = m.MRIOTable.from_raw(replace(raw, fd_categories=["h", "n", "g", "i", "s", "d"]),
                                      fd_mapping={"h": "C", "n": "C", "g": "C", "i": "G", "s": "V", "d": "X"})
        np.testing.assert_allclose(custom.F, t.F)

    def test_renderers_and_summary(self):
        t = toy_table()
        for which in ("summary", "countries", "categories", "sources"):
            df = t.to_dataframe(which)
            assert isinstance(df, pd.DataFrame)
            assert t.to_markdown(which) and "tabular" in t.to_latex(which) and "#table" in t.to_typst(which)
        with pytest.raises(ValueError):
            t.to_dataframe("other")
        assert "synthetic_unit_fixture" in t.summary()
        rep = t.accounting_report()
        assert rep.to_markdown() and rep.to_latex() and rep.to_typst() and "active cells" in rep.summary()
        reg, build = m.regularize_table(t)
        for which in ("summary", "gates", "floored"):
            assert build.to_markdown(which) and build.to_latex(which) and build.to_typst(which)
        assert "phantoms=0" in build.summary()
        assert json.dumps(build.to_dict()) and json.dumps(rep.to_dict())
        countries = t.to_dataframe("countries")
        assert list(countries.columns[:3]) == ["output", "value_added", "TLS"] and "V_expenditure" in countries.columns
        assert t.with_metadata(dataset="renamed").dataset == "renamed" and t.dataset == "synthetic_unit_fixture"


# ---------------------------------------------------------------------------
# Readers on the hand fixtures
# ---------------------------------------------------------------------------

FIGARO_F = np.array([
    [[43., 3., 0., -0.3], [6.5, 1., 0., 0.2]],
    [[66., 8., 0., 1.], [0., 2., 0., 0.4995]],
    [[7.2, 0.5, 0., 0.], [54.5, 6., 0., -1.2]],
    [[0., 0., 0., 0.], [0., 0., 0., 0.]],
])
FIGARO_Z = np.array([[10, 30, 2, 0], [5, 20, 3, 0], [4, 6, 15, 0], [0, 0, 0, 0]], dtype=float)


class TestFigaroReader:
    def _read(self, **kw):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return m.read_figaro_native(FIGARO_FIXTURE, **kw)

    def test_fixture_arrays_with_rounding_repair(self):
        with pytest.warns(RuntimeWarning, match="unauthenticated"):
            t = m.read_figaro_native(FIGARO_FIXTURE)
        assert t.country_codes == ("AUT", "ROW") and t.sector_codes == ("A01", "C10T12")
        assert t.fd_codes == ("C", "G", "X", "V") and t.reference_country == "ROW"
        np.testing.assert_allclose(t.Z, FIGARO_Z)
        np.testing.assert_allclose(t.F, FIGARO_F, atol=1e-15)
        np.testing.assert_allclose(t.VA, [73.4, 43.9995, 70.0, 0.0])
        np.testing.assert_allclose(t.labor_compensation, [40, 30, 45, 0])
        np.testing.assert_allclose(t.production_taxes, [-1.4, 1, 0.5, 0])
        np.testing.assert_allclose(t.operating_surplus, [34.8, 12.9995, 24.5, 0])
        np.testing.assert_allclose(t.TLS, [3, 5.5, 2, 0])
        np.testing.assert_allclose(t.TFD, [[4.3, 0.5, 0, 0.05], [5.4, 0.6, 0, -0.02]])
        np.testing.assert_allclose(t.output, [95.4, 105.4995, 92.0, 0.0])
        assert t.merchandise_mask.tolist() == [True, True]
        assert t.metadata["purchases_abroad_residents"] == pytest.approx(2.3)
        assert t.metadata["purchases_by_nonresidents"] == pytest.approx(2.3)
        adj = t.metadata["rounding_adjustments"]
        assert [a["reason"] for a in adj] == ["negative empty-cell rounding", "rounding reclassified from C to V"]
        assert adj[0]["cell"] == "FIGW1_C10T12" and adj[0]["output"] == -0.001 and adj[0]["absolute_input_mass"] == pytest.approx(0.002)
        assert adj[1]["cell"] == "AT_C10T12" and adj[1]["destination"] == "FIGW1" and adj[1]["amount"] == -0.0005
        assert len(t.transformations) == 1 and t.build_report.rounding_adjustments == adj
        rep = t.accounting_report()
        assert rep.row_column_max_absolute_gap < 1e-12 and rep.zero_output_cells == 1 and rep.negative_inventory_entries == 2
        assert rep.world_production_taxes == pytest.approx(0.1)
        assert t.sources[0].status == "unknown_edition" and t.sources[0].sha256 == hashlib.sha256(FIGARO_FIXTURE.read_bytes()).hexdigest()
        # Repairs preserve every row total and each country's total expenditure.
        raw = self._read(rounding_repair=False)
        np.testing.assert_allclose(t.row_sales()[:3], raw.row_sales()[:3])
        np.testing.assert_allclose(t.F.sum(axis=(0, 2))[:2], raw.F.sum(axis=(0, 2))[:2] + np.array([0, 0.001]))

    def test_fixture_without_repair_keeps_source_rounding(self):
        raw = self._read(rounding_repair=False)
        assert raw.is_raw and raw.output[3] == -0.001 and raw.TLS[3] == 0.001
        assert raw.final_use("C")[1, 1] == -0.0005 and raw.final_use("V")[1, 1] == 0.5
        assert raw.final_use("C")[3, 1] == -0.001
        rep = raw.accounting_report()
        assert rep.negative_output_cells == 1 and rep.negative_consumption_entries == 2

    def test_layout_errors_and_digest_checks(self, tmp_path):
        text = FIGARO_FIXTURE.read_text(encoding="utf-8")
        sha = hashlib.sha256(FIGARO_FIXTURE.read_bytes()).hexdigest()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ok = m.read_figaro_native(FIGARO_FIXTURE, expected_sha256=sha)
        assert ok.sources[0].status == "verified"
        with pytest.raises(m.MRIOIntegrityError, match="SHA-256 mismatch"):
            m.read_figaro_native(FIGARO_FIXTURE, expected_sha256="0" * 64)
        bad = tmp_path / "bad.csv"
        lines = text.splitlines()
        bad.write_text("\n".join(lines[:5] + lines[6:7] + lines[5:6] + lines[7:]) + "\n", encoding="utf-8")  # swap two trailing rows
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with pytest.raises(m.MRIOIntegrityError, match="must end with the rows"):
                m.read_figaro_native(bad)
            bad.write_text(text.replace("AT_P3_S13,AT_P3_S14", "AT_P3_S14,AT_P3_S13"), encoding="utf-8")
            with pytest.raises(m.MRIOIntegrityError, match="final-use columns"):
                m.read_figaro_native(bad)
            bad.write_text(text.replace("AT_A01,AT_C10T12,FIGW1_A01,FIGW1_C10T12,AT_P3", "AT_A01,FIGW1_A01,AT_C10T12,FIGW1_C10T12,AT_P3"), encoding="utf-8")
            with pytest.raises(m.MRIOIntegrityError):
                m.read_figaro_native(bad)
            bad.write_text(text.replace("FIGW1_C10T12,0,0,0,0,0,0,0,0,0,0,-0.001", "FIGW1_C10T12,0,0,0,0,0,0,0,0,0,0,-0.5"), encoding="utf-8")
            with pytest.raises(m.MRIOIntegrityError, match="Non-rounding negative output"):
                m.read_figaro_native(bad)
            sp = m.read_figaro_native(FIGARO_FIXTURE, sparse_matrix=True)
        assert sp.is_sparse and _maxdiff(sp.Z, FIGARO_Z) == 0


class TestFigaroRegisteredDigest:
    def test_registered_digest_is_bound_to_its_year(self, monkeypatch):
        sha = hashlib.sha256(FIGARO_FIXTURE.read_bytes()).hexdigest()
        monkeypatch.setattr(m, "FIGARO_2026ED_2019_SHA256", sha)  # test-only: treat the fixture as the registered file
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            t = m.read_figaro_native(FIGARO_FIXTURE)
        assert t.sources[0].status == "verified" and t.year == 2019
        with pytest.raises(m.MRIOIntegrityError, match="not the requested year 2020"):
            m.read_figaro_native(FIGARO_FIXTURE, year=2020)


EXIO_F = np.array([
    [[6.0, 0.4, 0, -0.3, 0.05], [0.7, 0.2, 0, 0.1, 0]],
    [[0.1, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    [[4.3, 0.5, 0, 0, 0], [0.3, 0.1, 0, 0.02, 0]],
    [[0.8, 0.3, 0, 0.05, 0], [9.8, 1.0, 0, -0.4, 0.1]],
    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    [[0.3, 0.4, 0, 0, 0], [5.1, 1.2, 0, 0.1, 0.02]],
])
EXIO_Z = np.array([
    [2.0, 0.1, 0.5, 0.3, 0, 0.1],
    [0.2, 0.4, 0.0, 0.0, 0, 0.0],
    [0.6, 0.1, 1.5, 0.2, 0, 0.4],
    [0.4, 0.0, 0.1, 3.0, 0, 0.7],
    [0, 0, 0, 0, 0, 0],
    [0.1, 0.0, 0.3, 0.9, 0, 2.5],
])


class TestExiobaseReader:
    def test_fixture_arrays(self):
        t = _read_exio_dir()
        assert t.country_codes == ("AUT", "ROW_ASIA_PACIFIC") and t.sector_codes == ("i01.a", "i01.w.1", "i60.2")
        assert t.fd_codes == ("C", "G", "X", "V", "VAL") and t.is_sparse and t.reference_country == "ROW_ASIA_PACIFIC"
        assert _maxdiff(t.Z, EXIO_Z) == 0 and t.Z.nnz == 20
        np.testing.assert_allclose(t.F, EXIO_F, atol=1e-15)
        np.testing.assert_allclose(t.TFD, [[0.62, 0.05, 0, 0.01, 0], [1.08, 0.1, 0, -0.01, 0.005]], atol=1e-15)
        np.testing.assert_allclose(t.TLS, [0.35, 0.01, 0.2, 0.45, 0, 0.3])
        np.testing.assert_allclose(t.production_taxes, [-0.1, 0.02, 0.12, 0.2, 0, -0.05])
        np.testing.assert_allclose(t.labor_compensation, [3.5, 0.05, 3.0, 6.3, 0, 4.0])
        np.testing.assert_allclose(t.operating_surplus, [3.1, 0.02, 2.3, 4.5, 0, 2.97])
        np.testing.assert_allclose(t.VA, [6.5, 0.09, 5.42, 11.0, 0, 6.92])
        np.testing.assert_allclose(t.output, [10.15, 0.7, 8.02, 15.85, 0, 10.92])
        assert t.merchandise_mask.tolist() == [True, False, False]
        assert t.metadata["nowcast"] is True and t.metadata["sector_names"][2] == "Other land transport"
        assert [s.archive_member for s in t.sources] == ["Z.txt", "Y.txt", "x.txt", "industries.txt", "satellite/F.txt",
                                                          "satellite/F_Y.txt", "finaldemands.txt", "metadata.json"]
        # Without its archive a directory is hashed, not authenticated.
        assert {s.status for s in t.sources} == {"unknown_edition"} and t.metadata["source_authenticated"] is False
        assert set(t.metadata["factor_components"]) == set(m.EXIOBASE_FACTOR_NAMES)
        rep = t.accounting_report()
        assert rep.row_column_max_absolute_gap < 1e-12 and rep.zero_output_cells == 1 and rep.negative_inventory_entries == 2
        assert rep.final_category_totals["VAL"] == pytest.approx(0.17)
        assert t.is_raw

    def test_zip_equals_directory_and_chunking_and_dense(self, tmp_path):
        archive = _write_exio_zip(tmp_path / "IOT_2019_ixi.zip")
        base = _read_exio_dir()
        with pytest.warns(RuntimeWarning, match="unauthenticated"):
            z = m.read_exiobase_native(archive)
        assert _maxdiff(z.Z, base.Z) == 0 and np.array_equal(z.F, base.F) and np.array_equal(z.VA, base.VA)
        assert z.sources[0].status == "unknown_edition" and z.sources[1].archive_member == "IOT_2019_ixi/Z.txt"
        assert all(s.status == "verified" and s.crc32 is not None for s in z.sources[1:])
        sha = hashlib.sha256(archive.read_bytes()).hexdigest()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            v = m.read_exiobase_native(archive, expected_sha256=sha)
        assert v.sources[0].status == "verified"
        with pytest.raises(m.MRIOIntegrityError, match="SHA-256 mismatch"):
            m.read_exiobase_native(archive, expected_sha256="0" * 64)
        for chunk in (1, 2, 5, 100):
            c = _read_exio_dir(chunksize=chunk)
            assert _maxdiff(c.Z, base.Z) == 0
        dense = _read_exio_dir(sparse_matrix=False, chunksize=2)
        assert not dense.is_sparse and np.array_equal(dense.Z, EXIO_Z)
        # A ZIP without the year prefix is accepted when the members sit at the root.
        flat = _write_exio_zip(tmp_path / "flat.zip", prefix="")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            f = m.read_exiobase_native(flat)
        assert _maxdiff(f.Z, base.Z) == 0
        with pytest.raises(FileNotFoundError):
            m.read_exiobase_native(tmp_path / "absent")

    def test_directory_authenticated_against_its_archive(self, tmp_path, monkeypatch):
        archive = _write_exio_zip(tmp_path / "IOT_2019_ixi.zip")
        sha = hashlib.sha256(archive.read_bytes()).hexdigest()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            t = m.read_exiobase_native(EXIO_FIXTURE, archive=archive, expected_sha256=sha)
        assert t.sources[0].status == "verified" and t.sources[0].sha256 == sha and t.build_report.sha256 == sha
        members = t.sources[1:]
        assert [s.archive_member for s in members] == ["IOT_2019_ixi/" + x for x in (
            "Z.txt", "Y.txt", "x.txt", "industries.txt", "satellite/F.txt", "satellite/F_Y.txt", "finaldemands.txt",
            "metadata.json")]
        assert all(s.status == "verified" and s.crc32 is not None for s in members)
        assert members[0].path == str((EXIO_FIXTURE / "Z.txt").resolve()) and t.metadata["source_authenticated"] is True
        base = _read_exio_dir()
        assert _maxdiff(t.Z, base.Z) == 0 and np.array_equal(t.F, base.F) and np.array_equal(t.VA, base.VA)
        # An unregistered archive is read with a warning; its extracted members are still size- and CRC-checked.
        with pytest.warns(RuntimeWarning, match="unauthenticated"):
            u = m.read_exiobase_native(EXIO_FIXTURE, archive=archive)
        assert u.sources[0].status == "unknown_edition" and all(s.status == "verified" for s in u.sources[1:])
        assert u.metadata["source_authenticated"] is False
        with pytest.raises(m.MRIOIntegrityError, match="SHA-256 mismatch"):
            m.read_exiobase_native(EXIO_FIXTURE, archive=archive, expected_sha256="0" * 64)
        # One changed byte, or one added byte, in an extracted member is refused.
        core = tmp_path / "core"
        shutil.copytree(EXIO_FIXTURE, core)
        # Bytes, not text: text mode on Windows writes CRLF, which changes the
        # size and trips the size check before the CRC check under test.
        text = (core / "x.txt").read_bytes()
        (core / "x.txt").write_bytes(text.replace(b"10.15", b"10.16"))
        with pytest.raises(m.MRIOIntegrityError, match="CRC-32"):
            m.read_exiobase_native(core, archive=archive, expected_sha256=sha)
        (core / "x.txt").write_bytes(text.replace(b"10.15", b"10.150"))
        with pytest.raises(m.MRIOIntegrityError, match="bytes"):
            m.read_exiobase_native(core, archive=archive, expected_sha256=sha)
        # The archive digest cannot authenticate a directory on its own; archive= needs a directory.
        with pytest.raises(ValueError, match="pass archive"):
            m.read_exiobase_native(EXIO_FIXTURE, expected_sha256=sha)
        with pytest.raises(ValueError, match="extracted directory"):
            m.read_exiobase_native(archive, archive=archive)
        with pytest.raises(FileNotFoundError, match="archive not found"):
            m.read_exiobase_native(EXIO_FIXTURE, archive=tmp_path / "absent.zip")
        # The registered digest is bound to 2019.
        monkeypatch.setattr(m, "EXIOBASE_382_2019_IXI_SHA256", sha)  # test-only: treat this archive as the registered one
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            r = m.read_exiobase_native(EXIO_FIXTURE, archive=archive)
        assert r.sources[0].status == "verified" and r.metadata["source_authenticated"] is True
        with pytest.raises(m.MRIOIntegrityError, match="not the requested year 2020"):
            m.read_exiobase_native(archive, year=2020)

    @pytest.mark.filterwarnings("ignore:.*without its archive:RuntimeWarning")
    def test_rejections(self, tmp_path):
        def variant(edit):
            core = tmp_path / "core"
            if core.exists():
                shutil.rmtree(core)
            shutil.copytree(EXIO_FIXTURE, core)
            edit(core)
            return core

        def set_exports(core):
            p = core / "Y.txt"
            lines = p.read_text(encoding="utf-8").splitlines()
            parts = lines[3].split("\t")
            parts[2 + 6] = "0.5"  # AT exports column of the first cell
            lines[3] = "\t".join(parts)
            p.write_text("\n".join(lines) + "\n", encoding="utf-8")

        with pytest.raises(m.MRIOIntegrityError, match="unallocated exports"):
            m.read_exiobase_native(variant(set_exports))

        def set_fd_factor(core):
            p = core / "satellite" / "F_Y.txt"
            lines = p.read_text(encoding="utf-8").splitlines()
            parts = lines[4].split("\t")  # "Other net taxes on production" row
            parts[1] = "0.1"
            lines[4] = "\t".join(parts)
            p.write_text("\n".join(lines) + "\n", encoding="utf-8")

        with pytest.raises(m.MRIOIntegrityError, match="primary-factor payments"):
            m.read_exiobase_native(variant(set_fd_factor))

        def rename_industry(core):
            p = core / "industries.txt"
            p.write_text(p.read_text(encoding="utf-8").replace("Other land transport", "Other transport"), encoding="utf-8")

        with pytest.raises(m.MRIOIntegrityError, match="ordering"):
            m.read_exiobase_native(variant(rename_industry))

        def rename_factor(core):
            p = core / "satellite" / "F.txt"
            p.write_text(p.read_text(encoding="utf-8").replace("Other net taxes on production", "Other taxes"), encoding="utf-8")

        with pytest.raises(m.MRIOIntegrityError, match="factor labels"):
            m.read_exiobase_native(variant(rename_factor))

        def drop_member(core):
            (core / "x.txt").unlink()

        with pytest.raises(FileNotFoundError, match="x.txt"):
            m.read_exiobase_native(variant(drop_member))

        def rename_category(core):
            p = core / "finaldemands.txt"
            p.write_text(p.read_text(encoding="utf-8").replace("Changes in valuables", "Valuables"), encoding="utf-8")

        with pytest.raises(m.MRIOIntegrityError, match="finaldemands"):
            m.read_exiobase_native(variant(rename_category))


class TestOecdReader:
    def test_tiny_csv_with_monkeypatched_registries(self, tmp_path, monkeypatch):
        path, countries, sectors, Z, F6, VA, TLS, TFD6, Y = _tiny_oecd_csv(tmp_path)
        monkeypatch.setattr(trade_data, "OECD_77_COUNTRIES", countries)
        monkeypatch.setattr(trade_data, "OECD_45_SECTORS", sectors)
        md5 = hashlib.md5(path.read_bytes()).hexdigest()
        with pytest.warns(RuntimeWarning, match="not a whitelisted"):
            t = m.read_oecd_native(path, 2019)
        assert t.fd_codes == ("C", "G", "X", "V") and t.country_codes == countries and t.sector_codes == sectors
        np.testing.assert_allclose(t.Z, Z)
        np.testing.assert_allclose(t.final_use("C"), F6[:, :, :3].sum(2))
        np.testing.assert_allclose(t.final_use("G"), F6[:, :, 3])
        np.testing.assert_allclose(t.final_use("V"), F6[:, :, 4])
        np.testing.assert_allclose(t.final_use("X"), F6[:, :, 5])
        np.testing.assert_allclose(t.VA, VA)
        np.testing.assert_allclose(t.TLS, TLS)
        np.testing.assert_allclose(t.output, Y)
        np.testing.assert_allclose(t.TFD, np.column_stack([TFD6[:, :3].sum(1), TFD6[:, 3], TFD6[:, 5], TFD6[:, 4]]))
        assert t.merchandise_mask.tolist() == [True, False] and t.labor_compensation is None
        assert t.sources[0].status == "unknown_edition" and t.metadata["whitelist_status"] == "unknown_edition"
        assert t.metadata["production_taxes_observed"] is False and t.reference_country == "BBB"
        assert t.metadata["final_use_components"]["HFCE"].shape == (4, 2)
        assert not t.is_sparse and t.build_report.md5 == md5
        with pytest.raises(m.MRIOIntegrityError, match="refused"):
            m.read_oecd_native(path, 2019, corrupted={md5: "a test-only corrupted registry entry"})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            w = m.read_oecd_native(path, 2019, whitelist={2019: md5})
            u = m.read_oecd_native(path, 2019, check=False)
        assert w.sources[0].status == "whitelisted" and u.sources[0].status == "hashed"
        archive = tmp_path / "2016-2020_SML.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.write(path, "2019_SML.csv")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            zt = m.read_oecd_native(archive, 2019, whitelist={2019: md5})
        assert np.array_equal(zt.F, t.F) and zt.sources[0].archive_member == "2019_SML.csv"
        raw = trade_data.load_oecd_icio_granular(2019, file_path=path)
        assert raw.n_sectors == 2
        # to_raw("cix") reproduces the C, I, Cx condensation used by load_oecd_icio_granular.
        from puremacro.trade._oecd_icio import condense_final_demand, read_native

        cond = condense_final_demand(read_native(path, 2019))
        np.testing.assert_allclose(t.to_raw("cix").final_demand_matrix, cond.final_demand_matrix)
        np.testing.assert_allclose(t.to_raw("cix").taxes_less_subsidies_fd, cond.taxes_less_subsidies_fd)


# ---------------------------------------------------------------------------
# Regularization
# ---------------------------------------------------------------------------


class TestRegularize:
    def test_figaro_fixture_phantom_and_residual(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            t = m.read_figaro_native(FIGARO_FIXTURE)
        reg, rep = m.regularize_table(t)
        assert rep.passed and rep.phantoms == ("ROW_C10T12",) and rep.floored == () and rep.n_zero_output == 1
        assert rep.skipped_phantoms == () and rep.failed_gates == ()
        assert reg.F[3, 1, 0] == 1e-6 and reg.VA[3] == 1e-6 and reg.output[3] == pytest.approx(1e-6)
        np.testing.assert_allclose(reg.TLS, [3, 5.5, 2, 0], atol=1e-15)
        np.testing.assert_allclose(reg.output[:3], [95.4, 105.4995, 92.0])
        assert rep.dtls_accounting_abs < 1e-13 and rep.dtls_floor_abs == 0 and rep.world_abs_tls == pytest.approx(10.5)
        assert rep.world_gdp == pytest.approx(73.4 + 43.9995 + 70 + 1e-6 + 10.5 + (4.3 + 0.5 + 0.05 + 5.4 + 0.6 - 0.02))
        assert rep.negative_fd_cells == {"C": 0, "G": 0, "X": 0, "V": 2}
        assert rep.rounding_adjustments == t.metadata["rounding_adjustments"]
        assert len(reg.transformations) == 2 and reg.metadata["regularized"] is True and reg.build_report is rep
        assert all(g["passed"] for g in rep.gates.values()) and set(rep.gates) == {
            "nonnegative_output", "row_identity_median_rel", "zero_output_cells_empty", "dtls_world_share",
            "dtls_cell_share_of_output", "trade_balance_world_sum"}
        # The regularized accounts are exactly balanced.
        assert reg.accounting_report().row_column_max_absolute_gap < 1e-13
        # Regularizing twice is idempotent.
        again, rep2 = m.regularize_table(reg)
        np.testing.assert_allclose(again.TLS, reg.TLS, atol=1e-15)
        assert rep2.phantoms == ()

    def test_gates_strict_and_report_flags(self):
        t = toy_table()
        Z = t.Z.toarray()
        Z[0] = 0
        Z[:, 0] = 0
        out = np.array(t.output)
        out[0] = 0
        F = np.array(t.F)
        F[0] = 0
        junk = toy_table(Z=Z, output=out, F=F, strict=False)  # VA and TLS of the dead cell still non-zero
        with pytest.raises(m.MRIOIntegrityError, match="zero-output cell has non-zero"):
            m.regularize_table(junk)
        _, rep = m.regularize_table(junk, strict=False)
        assert not rep.passed and not rep.gates["zero_output_cells_empty"]["passed"]
        neg = toy_table(output=np.array([-1., 1, 1, 1]))
        with pytest.raises(m.MRIOIntegrityError, match="negative gross output"):
            m.regularize_table(neg)
        skewed = toy_table(output=np.array(t.output) * 1.5)
        with pytest.raises(m.MRIOIntegrityError, match="median relative row identity"):
            m.regularize_table(skewed)
        _, rep = m.regularize_table(skewed, strict=False)
        assert rep.gates["row_identity_median_rel"]["value"] == pytest.approx(1 / 3)
        # A large residual-tax change trips the per-cell gate.
        big = toy_table(TLS=np.array(t.TLS) + np.array([2., 0, 0, 0]), strict=False)
        with pytest.raises(m.MRIOIntegrityError, match="moves cell A_goods"):
            m.regularize_table(big)
        _, rep = m.regularize_table(big, strict=False)
        assert not rep.gates["dtls_cell_share_of_output"]["passed"]
        assert rep.gates["dtls_cell_share_of_output"]["value"] == pytest.approx(2 / max(t.output[0], 1))
        with pytest.raises(ValueError, match="inactive"):
            m.regularize_table(t, inactive="drop")
        with pytest.raises(TypeError):
            m.regularize_table(np.eye(2))

    def test_failed_gates_are_never_marked_regularized(self):
        t = toy_table()
        big = toy_table(TLS=np.array(t.TLS) + np.array([2., 0, 0, 0]), strict=False)
        reg, rep = m.regularize_table(big, strict=False)
        assert not rep.passed and "dtls_cell_share_of_output" in rep.failed_gates
        g = rep.gates["dtls_cell_share_of_output"]
        assert g["cells"] == ("A_goods",) and g["n_cells"] == 1 and g["worst_cell"] == "A_goods"
        assert reg.metadata["regularized"] is False and reg.metadata["regularization_passed"] is False
        assert reg.metadata["failed_gates"] == rep.failed_gates
        assert "FAILED gates" in reg.transformations[-1] and "failed:" in rep.summary()
        assert rep.to_dataframe().loc["failed_gates", "Value"] == ", ".join(rep.failed_gates)
        assert json.loads(json.dumps(rep.to_dict()))["gates"]["dtls_cell_share_of_output"]["cells"] == ["A_goods"]
        # The failure travels through aggregation and into the calibration bridge, loudly.
        with pytest.warns(RuntimeWarning, match="gates failed"):
            agg = m.aggregate_mrio(reg, m.Concordance.identity(reg.sector_codes))
        assert agg.metadata["regularized"] is False and agg.build_report.passed is False
        assert agg.metadata["aggregation"]["input_gates_passed"] is False
        with pytest.warns(RuntimeWarning, match="failed regularize_table gates"):
            m.to_calibration_matrix(agg)
        # A table that passes is marked and moves on silently.
        ok, ok_rep = m.regularize_table(t)
        assert ok.metadata["regularized"] is True and ok.metadata["failed_gates"] == () and ok_rep.failed_gates == ()
        assert ok_rep.gates["dtls_cell_share_of_output"]["n_cells"] == 0
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            agg_ok = m.aggregate_mrio(ok, m.Concordance.identity(ok.sector_codes))
            m.to_calibration_matrix(agg_ok)
        assert agg_ok.metadata["aggregation"]["input_gates_passed"] is True

    def test_phantoms_skip_dirty_and_negative_cells_under_strict_false(self):
        t = toy_table()
        Z = t.Z.toarray()
        Z[0] = 0
        Z[:, 0] = 0
        out = np.array(t.output)
        out[0] = 0
        F = np.array(t.F)
        F[0] = 0
        junk = toy_table(Z=Z, output=out, F=F, strict=False)  # VA and TLS of the dead cell still non-zero
        with pytest.raises(m.MRIOIntegrityError, match="non-zero entries: A_goods"):
            m.regularize_table(junk)
        reg, rep = m.regularize_table(junk, strict=False)
        assert rep.phantoms == () and rep.skipped_phantoms == ("A_goods",)
        assert reg.VA[0] == junk.VA[0] and reg.F[0].sum() == 0  # the stray value added is kept, not overwritten
        assert rep.gates["zero_output_cells_empty"]["cells"] == ("A_goods",)
        assert "left unchanged" in reg.transformations[-1] and reg.metadata["regularized"] is False
        neg = toy_table(output=np.array([-1., 1, 1, 1]))
        with pytest.raises(m.MRIOIntegrityError, match="negative gross output at A_goods"):
            m.regularize_table(neg)
        reg_n, rep_n = m.regularize_table(neg, strict=False)
        assert rep_n.skipped_phantoms == ("A_goods",) and rep_n.phantoms == () and reg_n.VA[0] == neg.VA[0]
        assert rep_n.gates["nonnegative_output"]["cells"] == ("A_goods",)

    def test_mask_mode_keeps_empty_cells(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            t = m.read_figaro_native(FIGARO_FIXTURE)
        reg, rep = m.regularize_table(t, inactive="mask")
        assert rep.inactive == "mask" and rep.phantoms == ("ROW_C10T12",)
        assert reg.output[3] == 0 and reg.VA[3] == 0 and reg.F[3].sum() == 0 and reg.TLS[3] == 0

    def test_floor_only_where_va_nonpositive_and_dtls_bookkeeping(self):
        t = random_table(11, 2, 3)
        VA = np.array(t.VA)
        VA[2] = -0.5
        VA[4] = 0.0
        reg, rep = m.regularize_table(replace(t, VA=VA))
        labels = t.cell_labels
        assert [f[0] for f in rep.floored] == [labels[2], labels[4]]
        assert rep.floored[0][1] == -0.5 and rep.floored[0][2] == pytest.approx(1e-3 * t.output[2])
        assert reg.VA[2] == pytest.approx(1e-3 * t.output[2]) and reg.VA[4] == pytest.approx(1e-3 * t.output[4])
        untouched = [i for i in range(6) if i not in (2, 4)]
        np.testing.assert_allclose(reg.VA[untouched], t.VA[untouched])
        # Residual TLS absorbs the floor at the floored cells only; the rest is unchanged.
        expected = t.output - t.Z.sum(0) - reg.VA
        np.testing.assert_allclose(reg.TLS, expected, atol=1e-13)
        assert rep.dtls_floor_abs == pytest.approx(abs(expected[2] - t.TLS[2]) + abs(expected[4] - t.TLS[4]))
        assert rep.dtls_accounting_abs < 1e-12
        assert reg.accounting_report().row_column_max_absolute_gap < 1e-12

    def test_custom_gate_values_and_spectral_option(self):
        t = random_table(5, 3, 3)
        reg, rep = m.regularize_table(t, spectral=True, phantom=1e-4, va_floor_rel=0.01)
        assert rep.spectral_radius is not None and 0 < rep.spectral_radius[0] < 1
        lo, hi = rep.spectral_radius[1], rep.spectral_radius[2]
        assert lo <= rep.spectral_radius[0] + 1e-9 and rep.spectral_radius[0] <= hi + 1e-9
        assert "spectral_radius" in rep.to_dataframe().index
        with pytest.raises(ValueError, match="nonnegative"):
            m.regularize_table(t, phantom=-1)
        sp = replace(t, Z=sparse.csr_matrix(t.Z))
        reg_sp, rep_sp = m.regularize_table(sp, spectral=True)
        assert rep_sp.spectral_radius is None and reg_sp.is_sparse
        np.testing.assert_allclose(reg_sp.TLS, reg.TLS, atol=1e-14)

    def test_commutes_with_aggregation_when_no_floors_apply(self):
        t = random_45(3)
        reg, _ = m.regularize_table(t)
        ag_then_reg, _ = m.regularize_table(m.aggregate_mrio(t, "isic_section"))
        reg_then_ag = m.aggregate_mrio(reg, "isic_section")
        assert _maxdiff(ag_then_reg.TLS, reg_then_ag.TLS) < 1e-12
        assert _maxdiff(ag_then_reg.output, reg_then_ag.output) < 1e-12
        assert _maxdiff(ag_then_reg.Z, reg_then_ag.Z) < 1e-12


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _check_conservation(fine, coarse, conc):
    n, k = fine.n_countries, fine.n_categories
    assert np.isclose(np.asarray(coarse.Z.sum()), np.asarray(fine.Z.sum()), rtol=1e-12)
    np.testing.assert_allclose(coarse.F.sum(axis=0), fine.F.sum(axis=0), rtol=1e-12, atol=1e-13)
    assert np.isclose(coarse.VA.sum(), fine.VA.sum(), rtol=1e-12) and np.isclose(coarse.TLS.sum(), fine.TLS.sum(), rtol=1e-12)
    np.testing.assert_array_equal(coarse.TFD, fine.TFD)
    g = conc.n_coarse
    np.testing.assert_allclose(coarse.output.reshape(n, g).sum(1), fine.output.reshape(n, -1).sum(1), rtol=1e-12)
    np.testing.assert_allclose(coarse.VA.reshape(n, g).sum(1), fine.VA.reshape(n, -1).sum(1), rtol=1e-12)
    np.testing.assert_allclose(coarse.bilateral_flows(), fine.bilateral_flows(), rtol=1e-12)


class TestAggregate:
    def test_conservation_on_random_fixture_and_oecd_fixture(self):
        t = random_table(1, 3, 4)
        conc = m.Concordance.build("pairs", t.sector_codes, (1, 1, 2, 2), ("P", "Q"))
        agg = m.aggregate_mrio(t, conc)
        _check_conservation(t, agg, conc)
        assert agg.metadata["aggregation"]["aggregated_identity_rel"] <= 1e-8
        assert agg.sector_codes == ("P", "Q") and agg.transformations[-1].startswith("aggregate_mrio")
        assert agg.metadata["aggregation"]["fine_balanced"] is True
        from _trade_oecd_fixture import load_oecd_3x3 as load_fixture

        oecd = m.MRIOTable.from_raw(load_fixture(), metadata={"dataset": "oecd_icio_3x3"})
        reg, _ = m.regularize_table(oecd)
        conc2 = m.Concordance.from_mapping("two", {"RESOURCES": "GOODS", "MANUFACTURING": "GOODS", "SERVICES": "SERVICES"})
        agg2 = m.aggregate_mrio(reg, conc2)
        _check_conservation(reg, agg2, conc2)
        assert agg2.metadata["aggregation"]["aggregated_identity_rel"] <= 1e-8
        # Grouping ids or a mapping give the same table as the Concordance object.
        by_ids = m.aggregate_mrio(reg, (1, 1, 2), coarse_codes=("GOODS", "SERVICES"))
        by_map = m.aggregate_mrio(reg, {"RESOURCES": "GOODS", "MANUFACTURING": "GOODS", "SERVICES": "SERVICES"})
        for other in (by_ids, by_map):
            assert _maxdiff(other.Z, agg2.Z) == 0 and np.array_equal(other.F, agg2.F)

    def test_identity_concordance_is_a_noop(self):
        t = toy_table()
        ident = m.aggregate_mrio(t, m.Concordance.identity(t.sector_codes))
        assert _maxdiff(ident.Z, t.Z) == 0 and np.array_equal(ident.F, t.F) and np.array_equal(ident.VA, t.VA)
        assert ident.is_sparse and ident.merchandise_mask.tolist() == [True, False]
        np.testing.assert_array_equal(ident.labor_compensation, t.labor_compensation)

    def test_composition_and_explicit_kron(self):
        t = random_45(2)
        isic = m.CONCORDANCES["isic_section"]
        first = m.aggregate_mrio(t, isic)
        outer = m.Concordance.from_mapping("two", {c: ("GOODS" if c in ("A", "B", "C") else "SERV")
                                                  for c in m.ISIC_SECTION_CODES})
        two_step = m.aggregate_mrio(first, outer)
        one_step = m.aggregate_mrio(t, isic.compose(outer))
        assert _maxdiff(two_step.Z, one_step.Z) < 1e-12 and _maxdiff(two_step.F, one_step.F) < 1e-12
        assert _maxdiff(two_step.VA, one_step.VA) < 1e-12 and two_step.sector_codes == one_step.sector_codes
        P = np.kron(np.eye(3), isic.matrix.T)
        assert _maxdiff(P @ t.Z @ P.T, first.Z) < 1e-12
        np.testing.assert_allclose(P @ t.VA, first.VA, rtol=1e-13)
        sp = replace(t, Z=sparse.csr_matrix(t.Z))
        agg_sp = m.aggregate_mrio(sp, isic)
        assert agg_sp.is_sparse and _maxdiff(agg_sp.Z, first.Z) < 1e-12

    def test_region_concordance(self):
        t = random_table(4, 4, 3, countries=("USA", "CHN", "DEU", "ROW"))
        regions = m.Concordance.from_mapping("blocs", {"USA": "USA", "CHN": "REST", "DEU": "REST", "ROW": "REST"})
        agg = m.aggregate_mrio(t, m.Concordance.identity(t.sector_codes), region_concordance=regions)
        assert agg.country_codes == ("USA", "REST") and agg.n_cells == 6 and agg.reference_country == "REST"
        assert np.isclose(np.asarray(agg.Z.sum()), np.asarray(t.Z.sum()), rtol=1e-12)
        np.testing.assert_allclose(agg.TFD, np.vstack([t.TFD[0], t.TFD[1:].sum(0)]), rtol=1e-12)
        np.testing.assert_allclose(agg.F.sum(axis=(0, 2)), np.r_[t.F[:, 0].sum(), t.F[:, 1:].sum()], rtol=1e-12)
        B = t.bilateral_flows()
        np.testing.assert_allclose(agg.bilateral_flows()[0, 1], B[0, 1:].sum(), rtol=1e-12)
        by_map = m.aggregate_mrio(t, m.Concordance.identity(t.sector_codes),
                                  region_concordance={"USA": "USA", "CHN": "REST", "DEU": "REST", "ROW": "REST"})
        assert _maxdiff(by_map.Z, agg.Z) == 0
        with pytest.raises(ValueError, match="other country codes"):
            m.aggregate_mrio(t, m.Concordance.identity(t.sector_codes),
                             region_concordance=m.Concordance.identity(("X", "Y", "Z", "W")))

    def test_goods_consistency_45_to_sections(self):
        t = random_45(5)
        isic = m.aggregate_mrio(t, "isic_section")
        assert isic.merchandise_mask.tolist() == [True, True, True] + [False] * 17
        fine_goods = (t.output.reshape(3, 45) * t.merchandise_mask).sum(1)
        np.testing.assert_allclose((isic.output.reshape(3, 20) * isic.merchandise_mask).sum(1), fine_goods, rtol=1e-12)
        assert isic.metadata["aggregation"]["mixed_goods_groups"] == ()
        with pytest.raises(ValueError, match="other sector codes"):
            m.aggregate_mrio(random_table(1, 3, 4), "isic_section")
        with pytest.raises(ValueError, match="unknown concordance"):
            m.aggregate_mrio(t, "nace")

    def test_mixed_goods_groups_recorded_and_raw_tables_allowed(self):
        t = random_45(6)
        mixed = m.Concordance.build("mixed", RAW_45_SECTOR_CODES, [1] * 23 + [2] * 22, ("GOODS_AND_D", "REST"))
        agg = m.aggregate_mrio(t, mixed)
        assert agg.merchandise_mask.tolist() == [False, False]
        assert agg.metadata["aggregation"]["mixed_goods_groups"] == ("GOODS_AND_D",)
        out = np.array(t.output) * 1.01  # an unbalanced (raw) table is aggregated and recorded, not refused
        raw = replace(t, output=out)
        agg_raw = m.aggregate_mrio(raw, "isic_section")
        assert agg_raw.metadata["aggregation"]["fine_balanced"] is True  # row = column identity does not involve output
        skew = replace(t, TLS=np.array(t.TLS) + 1.0)
        agg_skew = m.aggregate_mrio(skew, "isic_section")
        assert agg_skew.metadata["aggregation"]["fine_balanced"] is False
        assert agg_skew.metadata["aggregation"]["aggregated_identity_rel"] > 1e-8


# ---------------------------------------------------------------------------
# Bridges to the calibration
# ---------------------------------------------------------------------------


class TestCalibrationBridge:
    def test_oecd_fixture_calibrates_and_solves_at_benchmark(self):
        from puremacro.trade import solve_trade_equilibrium
        from _trade_oecd_fixture import load_oecd_3x3 as load_fixture

        oecd = m.MRIOTable.from_raw(load_fixture(), metadata={"dataset": "oecd_icio_3x3"})
        with pytest.warns(RuntimeWarning, match="regularize_table"):
            m.to_calibration_matrix(oecd)
        reg, rep = m.regularize_table(oecd)
        assert rep.passed and rep.phantoms == () and rep.floored == ()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            D = m.to_calibration_matrix(reg)
        assert D.shape == (12, 18)
        np.testing.assert_allclose(D[:9].sum(1), D[:, :9].sum(0), rtol=1e-13)
        calib = calibrate_trade_model(D, ns=3, nc=3, nfd=3, country_codes=reg.country_codes, sector_codes=reg.sector_codes)
        assert verify_accounting_invariants(calib)["valid"]
        np.testing.assert_allclose(D[10, :9] / (D[10, :9] + D[11, :9]), 2 / 3)
        base = solve_trade_equilibrium(calib, tol=1e-5, max_iter=100)
        assert base.converged and np.max(np.abs(base.p_sol - 1)) < 1e-8
        np.testing.assert_allclose(base.y_sol, calib.ytot, rtol=1e-8)
        packaged = package_mrio_to_calibration_result(reg.to_raw(), regularize=False)
        assert verify_accounting_invariants(packaged)["valid"]
        np.testing.assert_allclose(packaged.ytot, calib.ytot, rtol=1e-12)
        raw = reg.to_raw("native")
        assert raw.fd_categories == list(reg.fd_codes) and raw.metadata["transformations"] == list(reg.transformations)
        assert raw.metadata["final_demand_mapping"]["C"] == ["C"]

    def test_figaro_factor_detail_and_production_taxes(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            t = m.read_figaro_native(FIGARO_FIXTURE)
        reg, _ = m.regularize_table(t)
        D = m.to_calibration_matrix(reg)
        M = 4
        np.testing.assert_allclose(D[:M].sum(1), D[:, :M].sum(0), atol=1e-12)
        # Production taxes go to the TLS row; labour and capital split the factor value added by observed shares.
        np.testing.assert_allclose(D[M, :M], reg.TLS + reg.production_taxes)
        factor = reg.VA - reg.production_taxes
        lab, cap = reg.labor_compensation, reg.operating_surplus
        both = (lab > 0) & (cap > 0)
        share = np.where(both, lab / np.where(both, lab + cap, 1), 2 / 3)
        np.testing.assert_allclose(D[M + 1, :M], share * factor)
        np.testing.assert_allclose(D[M + 2, :M], (1 - share) * factor)
        assert D[M + 1, 3] == pytest.approx(2 / 3 * 1e-6)  # the phantom cell has no observed split
        # I merges G and V; Cx is the zero X column; final taxes condense the same way.
        F3 = D[:M, M:].reshape(M, 2, 3)
        np.testing.assert_allclose(F3[:, :, 0], reg.final_use("C"))
        np.testing.assert_allclose(F3[:, :, 1], reg.final_use("G") + reg.final_use("V"))
        assert not F3[:, :, 2].any()
        np.testing.assert_allclose(D[M, M:].reshape(2, 3), np.column_stack([reg.TFD[:, 0], reg.TFD[:, 1] + reg.TFD[:, 3], reg.TFD[:, 2]]))
        D2 = m.to_calibration_matrix(reg, production_taxes="to_factors")
        np.testing.assert_allclose(D2[M, :M], reg.TLS)
        np.testing.assert_allclose(D2[M + 1, :M] + D2[M + 2, :M], reg.VA)
        with pytest.raises(ValueError):
            m.to_calibration_matrix(reg, production_taxes="drop")
        with pytest.raises(ValueError):
            m.to_calibration_matrix(reg, labor_share=1.5)
        calib = calibrate_trade_model(D, ns=2, nc=2, nfd=3, country_codes=reg.country_codes, sector_codes=reg.sector_codes)
        assert verify_accounting_invariants(calib)["valid"]

    def test_zero_factor_and_large_production_tax_cells_still_calibrate(self):
        t = toy_table()
        lab, cap = np.array(t.labor_compensation), np.array(t.operating_surplus)
        prod = np.array(t.production_taxes)
        fva = t.VA - prod
        lab[1], cap[1] = fva[1], 0.0              # zero operating surplus
        prod[2] = t.VA[2] + 0.5                   # production taxes above value added
        lab[2], cap[2] = 0.3, -0.8
        lab[3], cap[3] = 0.0, fva[3]              # zero labour
        odd = toy_table(labor_compensation=lab, operating_surplus=cap, production_taxes=prod)
        assert odd.accounting_report().factor_decomposition_max_absolute_gap < 1e-12
        reg, rep = m.regularize_table(odd)
        assert rep.passed and rep.floored == ()
        with pytest.warns(RuntimeWarning, match=r"production taxes of 1 cell are .*\(B_goods\)"):
            D, det = m.to_calibration_matrix(reg, return_details=True)
        assert det["production_taxes_retained_cells"] == ("B_goods",)
        assert set(det["share_fallback_cells"]) == {"A_services", "B_goods", "B_services"}
        assert det["observed_share_cells"] == 1 and det["zero_factor_cells"] == ()
        M = 4
        assert (D[M + 1, :M] > 0).all() and (D[M + 2, :M] > 0).all()
        np.testing.assert_allclose(D[M, :M], reg.TLS + np.where(np.arange(M) == 2, 0.0, reg.production_taxes))
        np.testing.assert_allclose(D[M + 1, 2] + D[M + 2, 2], reg.VA[2])  # production taxes kept inside value added
        np.testing.assert_allclose(D[M + 1, 1] / (D[M + 1, 1] + D[M + 2, 1]), 2 / 3)
        np.testing.assert_allclose(D[M + 1, 0] / (D[M + 1, 0] + D[M + 2, 0]), 0.6)
        np.testing.assert_allclose(D[:M].sum(1), D[:, :M].sum(0), atol=1e-12)
        calib = calibrate_trade_model(D, ns=2, nc=2, nfd=3, country_codes=reg.country_codes, sector_codes=reg.sector_codes)
        assert verify_accounting_invariants(calib)["valid"] and np.all((calib.alpha > 0) & (calib.alpha < 1))
        D2, det2 = m.to_calibration_matrix(reg, production_taxes="to_factors", return_details=True)
        assert det2["production_taxes_retained_cells"] == ()
        np.testing.assert_allclose(D2[M + 1, :M] + D2[M + 2, :M], reg.VA)
        calibrate_trade_model(D2, ns=2, nc=2, nfd=3)

    def test_negative_factor_value_added_is_refused(self):
        t = toy_table()
        VA = np.array(t.VA)
        VA[1] = -0.5
        raw = toy_table(VA=VA, labor_compensation=None, operating_surplus=None, production_taxes=None)
        with pytest.raises(m.MRIOIntegrityError, match="negative factor value added.*A_services"):
            m.to_calibration_matrix(raw, check_balance=False)
        reg, rep = m.regularize_table(raw)
        assert rep.passed and [f[0] for f in rep.floored] == ["A_services"]
        D = m.to_calibration_matrix(reg)  # the value-added floor makes the cell calibratable
        calibrate_trade_model(D, ns=2, nc=2, nfd=3)

    def test_from_trade_calibration_round_trip_on_a_synthetic_table(self):
        calib = calibrate_trade_model(synthetic_icio_table(5, 4, seed=0))
        t = m.MRIOTable.from_trade_calibration(calib)
        assert "X" in t.fd_codes and (t.n_countries, t.n_sectors) == (5, 4)
        assert t.transformations[0] == "bridged from TradeCalibrationResult.data_calibra"
        assert t.merchandise_mask is None and "outside the known registries" in t.metadata["sector_label_note"]
        assert t.production_taxes is None and t.labor_compensation is not None
        D = m.to_calibration_matrix(t)
        assert D.shape == calib.data_calibra.shape
        np.testing.assert_allclose(D, calib.data_calibra, rtol=1e-12, atol=1e-9)
        again = calibrate_trade_model(D)
        np.testing.assert_allclose(again.ytot, calib.ytot, rtol=1e-12)
        with pytest.raises(TypeError):
            m.MRIOTable.from_trade_calibration(D)
        with pytest.raises(ValueError, match="merchandise_mask"):
            m.goods_mask(t.sector_codes)
        assert t.accounting_report().row_column_max_absolute_gap < 1e-9 * t.output.max()

    def test_from_trade_calibration_moves_negative_investment_only_on_request(self):
        calib = calibrate_trade_model(synthetic_icio_table(5, 4, seed=0))
        M = calib.n_countries * calib.n_sectors
        D = np.array(calib.data_calibra, dtype=float, copy=True)
        for row, dest in ((0, 1), (6, 3)):  # two investment cells turned negative, row totals kept by C
            D[row, M + 3 * dest] += D[row, M + 3 * dest + 1] + 1.0
            D[row, M + 3 * dest + 1] = -1.0
        neg = replace(calib, data_calibra=D)
        with pytest.raises(m.MRIOIntegrityError, match="to_inventory"):
            m.MRIOTable.from_trade_calibration(neg)
        t = m.MRIOTable.from_trade_calibration(neg, negative_investment="to_inventory")
        assert "V" in t.fd_codes and (t.final_use("V") < 0).sum() == 2 and t.final_use("G").min() == 0
        assert "2 negative investment cells moved into V" in t.transformations
        np.testing.assert_allclose(m.to_calibration_matrix(t), D, rtol=1e-12, atol=1e-9)


class TestCalibrationBridgeFinalUseSemantics:
    """The Cx slot of a loader calibration is residents' purchases abroad only for OECD (DPABR)."""

    LOADERS = ("load_figaro", "load_wiod", "load_eora")

    @staticmethod
    def _calib(loader):
        return getattr(trade_data, loader)(custom_c=3, custom_s=2, seed=1, fallback_to_synthetic=True)

    @pytest.mark.parametrize("loader", LOADERS)
    def test_provider_inventories_are_never_bridged_as_purchases_abroad(self, loader):
        calib = self._calib(loader)
        t = m.MRIOTable.from_trade_calibration(calib, negative_investment="to_inventory")
        assert "X" not in t.fd_codes and not t.final_use("X").any()
        M, n = calib.n_countries * calib.n_sectors, calib.n_countries
        F3 = np.asarray(calib.data_calibra)[:M, M:].reshape(M, n, 3)
        np.testing.assert_array_equal(t.final_use("V"), F3[:, :, 2])
        np.testing.assert_array_equal(t.final_use("G"), F3[:, :, 1])
        assert t.metadata["final_use_bridge"]["cx_category"] == "V"
        assert t.metadata["final_use_bridge"]["final_use_mapping"] == calib.metadata["final_use_mapping"]

    def test_exiobase_export_column_is_refused_unless_named(self):
        calib = self._calib("load_exiobase")
        with pytest.raises(ValueError, match=r"INVNT', 'VALUABLES', 'EXPORT'.*cx_category"):
            m.MRIOTable.from_trade_calibration(calib, negative_investment="to_inventory")
        t = m.MRIOTable.from_trade_calibration(calib, negative_investment="to_inventory", cx_category="V")
        assert "X" not in t.fd_codes and t.metadata["final_use_bridge"]["rule"] == "explicit cx_category"
        with pytest.raises(ValueError, match="cx_category must be one of"):
            m.MRIOTable.from_trade_calibration(calib, cx_category="G")

    def test_oecd_mapping_and_the_unmapped_layout_bridge_cx_as_purchases_abroad(self):
        calib = self._calib("load_oecd_icio_granular")
        assert calib.metadata["final_use_mapping"]["Cx"] == ["DPABR"]
        t = m.MRIOTable.from_trade_calibration(calib, negative_investment="to_inventory")
        unmapped = replace(calib, metadata={k: v for k, v in calib.metadata.items() if k != "final_use_mapping"})
        plain = m.MRIOTable.from_trade_calibration(unmapped, negative_investment="to_inventory")
        assert t.fd_codes == plain.fd_codes and "X" in t.fd_codes
        for name in ("F", "VA", "TLS", "TFD"):
            np.testing.assert_array_equal(getattr(t, name), getattr(plain, name))
        np.testing.assert_array_equal(t.Z_dense, plain.Z_dense)
        assert t.final_use("X").sum() > 0

    def test_unrecognised_mapping_is_refused_with_the_mapping_named(self):
        calib = self._calib("load_figaro")
        odd = replace(calib, metadata={**calib.metadata, "final_use_mapping": {"C": ["C"], "I": ["I"], "Cx": ["FOO"]}})
        with pytest.raises(ValueError, match=r"\['FOO'\].*fd_mapping"):
            m.MRIOTable.from_trade_calibration(odd, negative_investment="to_inventory")

    @staticmethod
    def _with_cell(calib, column, value, *, row=0, dest=1):
        """Set one final-use cell of ``data_calibra`` (column 0=C, 1=I, 2=Cx), keeping the row total."""
        n, s = calib.n_countries, calib.n_sectors
        M = n * s
        D = np.array(calib.data_calibra, dtype=float, copy=True)
        col, c_col = M + 3 * dest + column, M + 3 * dest
        D[row, c_col] += D[row, col] - value
        D[row, col] = value
        return replace(calib, data_calibra=D)

    def test_negative_fixed_investment_moves_into_v_when_cx_is_v(self):
        # Provider layouts bridge Cx to V, so the negative-investment policy must
        # add to that V column; the three bridges must agree cell by cell.
        from puremacro.trade.condensed.table import BalancedIOTable
        from puremacro.trade.dynamic.accounts import FINAL_CATEGORIES, DynamicAccounts

        calib = self._with_cell(self._calib("load_figaro"), 1, -1.0)
        with pytest.raises(m.MRIOIntegrityError, match="negative_investment='to_inventory'"):
            m.MRIOTable.from_trade_calibration(calib)
        t = m.MRIOTable.from_trade_calibration(calib, negative_investment="to_inventory")
        assert t.fd_codes == ("C", "G", "V") and t.final_use("G").min() == 0.0
        assert "1 negative investment cells moved into V" in t.transformations
        M, n = calib.n_countries * calib.n_sectors, calib.n_countries
        F3 = np.asarray(calib.data_calibra)[:M, M:].reshape(M, n, 3)
        np.testing.assert_array_equal(t.final_use("G") + t.final_use("V"), F3[:, :, 1] + F3[:, :, 2])
        dyn = DynamicAccounts.from_trade_calibration(calib, negative_investment="to_inventory")
        cond = BalancedIOTable.from_trade_calibration(calib, negative_investment="to_inventory",
                                                      merchandise_mask=np.array([True, False]))
        for j, cat in enumerate(("C", "G", "X", "V")):
            mrio_cat = t.final_use(cat) if cat in t.fd_codes else np.zeros((M, n))
            np.testing.assert_array_equal(mrio_cat, dyn.F[:, :, FINAL_CATEGORIES.index(cat)])
            np.testing.assert_array_equal(mrio_cat, cond.F[:, :, j])

    def test_signed_inventory_cells_stay_in_v(self):
        calib = self._with_cell(self._calib("load_figaro"), 2, -1.0)
        t = m.MRIOTable.from_trade_calibration(calib, negative_investment="to_inventory")
        assert "X" not in t.fd_codes and t.final_use("V")[0, 1] == -1.0
        assert "negative investment" not in " ".join(t.transformations)

