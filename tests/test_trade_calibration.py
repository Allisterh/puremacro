"""Calibration of the trade model and its result containers."""
from __future__ import annotations

import re
import warnings
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import puremacro.trade as trade
from puremacro.trade import _results
from puremacro.trade._results import (
    GearyKhamisResult,
    ScenarioBatchResult,
    TradeCalibrationResult,
    TradeEquilibriumResult,
)
from puremacro.trade.calibration import calibrate_trade_model
from puremacro.trade.data import (
    CANONICAL_FINAL_DEMAND_CODES,
    EU_COUNTRY_CODES,
    ICIOData,
    get_eu_country_codes,
    get_final_demand_codes,
    load_icio_data,
    synthetic_codes,
    synthetic_icio_table,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def toy_icio_matrix() -> tuple[np.ndarray, int, int, int]:
    """A balanced 3-country, 2-sector, 1-final-use table, shape (9, 9)."""
    nc = 3
    ns = 2
    nfd = 1
    n_ind = ns * nc  # 6

    data = np.zeros((n_ind + 3, n_ind + nfd * nc), dtype=np.float64)

    # Intermediate inputs (domestic and cross-border trade)
    for k1 in range(nc):
        for k2 in range(nc):
            base_val = 10.0 if k1 == k2 else 2.0
            r_slice = slice(ns * k1, ns * (k1 + 1))
            c_slice = slice(ns * k2, ns * (k2 + 1))
            data[r_slice, c_slice] = np.array([
                [base_val, base_val * 0.8],
                [base_val * 0.6, base_val * 1.2],
            ])

    # Final-use deliveries (columns 6..8)
    for k1 in range(nc):
        for k2 in range(nc):
            fd_val = 15.0 if k1 == k2 else 3.0
            r_slice = slice(ns * k1, ns * (k1 + 1))
            c_idx = n_ind + k2
            data[r_slice, c_idx] = np.array([fd_val, fd_val * 1.5])

    # Row sum (sales) equals column sum (outlays) for every industry
    row_sales = np.sum(data[:n_ind, :], axis=1)
    col_interm = np.sum(data[:n_ind, :n_ind], axis=0)

    for j in range(n_ind):
        tax_val = 2.0
        va_val = row_sales[j] - col_interm[j] - tax_val
        data[n_ind, j] = tax_val
        data[n_ind + 1, j] = (2.0 / 3.0) * va_val  # labour compensation
        data[n_ind + 2, j] = (1.0 / 3.0) * va_val  # capital compensation

    # Taxes on final use
    for j in range(nc):
        data[n_ind, n_ind + j] = 1.2

    return data, nc, ns, nfd


@pytest.fixture
def toy_calibration_result(toy_icio_matrix) -> TradeCalibrationResult:
    data, nc, ns, nfd = toy_icio_matrix
    return calibrate_trade_model(
        data,
        nc=nc,
        ns=ns,
        nfd=nfd,
        country_codes=("CT1", "CT2", "CT3"),
        sector_codes=("SEC1", "SEC2"),
    )


# ---------------------------------------------------------------------------
# Dimensions read from the table
# ---------------------------------------------------------------------------

def test_dimensions_inferred():
    res = calibrate_trade_model(synthetic_icio_table(5, 4))
    assert res.nc == 5 and res.ns == 4
    assert res.n_final_demand == 3
    assert res.a.shape == (20, 4, 5)
    # Without labels the calibration uses the neutral generated codes.
    countries, sectors = synthetic_codes(5, 4)
    assert res.country_codes == tuple(countries)
    assert res.sector_codes == tuple(sectors)


def test_bad_shape_names_shape():
    with pytest.raises(ValueError, match="24, 35"):
        calibrate_trade_model(np.zeros((24, 35)))


def test_dimensions_inferred_for_other_final_use_counts():
    for nfd in (1, 2, 4):
        res = calibrate_trade_model(synthetic_icio_table(4, 3, nfd=nfd, seed=5), nfd=nfd)
        assert (res.nc, res.ns, res.n_final_demand) == (4, 3, nfd)


def test_explicit_dimensions_must_match_the_table():
    T = synthetic_icio_table(5, 4)
    res = calibrate_trade_model(T, ns=4, nc=5)
    assert (res.nc, res.ns) == (5, 4)
    with pytest.raises(ValueError, match=r"23, 35"):
        calibrate_trade_model(T, ns=5, nc=4)
    with pytest.raises(ValueError, match=r"23, 35"):
        calibrate_trade_model(T, nc=7)


def test_label_counts_must_match_the_table():
    T = synthetic_icio_table(3, 2)
    with pytest.raises(ValueError, match="country_codes"):
        calibrate_trade_model(T, country_codes=["A", "B"])
    with pytest.raises(ValueError, match="sector_codes"):
        calibrate_trade_model(T, sector_codes=["X", "Y", "Z"])


def test_calibration_matches_explicit_dimensions_bit_for_bit():
    T = synthetic_icio_table(6, 5, seed=11)
    inferred = calibrate_trade_model(T)
    explicit = calibrate_trade_model(T, ns=5, nc=6, nfd=3)
    for name in ("a", "afd", "alpha", "beta", "tax", "tax_fd", "theta", "invforT", "data_calibra"):
        assert np.array_equal(getattr(inferred, name), getattr(explicit, name)), name


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

class TestTradeCalibrationResultUnit:
    """TradeCalibrationResult schema, immutability and presentation."""

    def test_required_arrays_presence_validation(self, toy_calibration_result):
        res = toy_calibration_result
        required_names = [
            "a", "afd", "alpha", "beta", "k_endow", "l_endow", "invforT", "tax", "ytot"
        ]

        for name in required_names:
            kwargs = {k: getattr(res, k) for k in required_names}
            kwargs[name] = None
            with pytest.raises(ValueError, match=f"requires array '{name}'"):
                TradeCalibrationResult(**kwargs)

    def test_required_arrays_type_validation(self, toy_calibration_result):
        res = toy_calibration_result
        kwargs = {
            "a": res.a, "afd": res.afd, "alpha": res.alpha, "beta": res.beta,
            "k_endow": res.k_endow, "l_endow": res.l_endow, "invforT": res.invforT,
            "tax": res.tax, "ytot": res.ytot,
        }
        kwargs["alpha"] = [1.0, 2.0, 3.0]  # list instead of np.ndarray
        with pytest.raises(TypeError, match="must be a numpy.ndarray"):
            TradeCalibrationResult(**kwargs)

    def test_required_arrays_empty_validation(self, toy_calibration_result):
        res = toy_calibration_result
        kwargs = {
            "a": res.a, "afd": res.afd, "alpha": res.alpha, "beta": res.beta,
            "k_endow": res.k_endow, "l_endow": res.l_endow, "invforT": res.invforT,
            "tax": res.tax, "ytot": res.ytot,
        }
        kwargs["ytot"] = np.array([])
        with pytest.raises(ValueError, match="cannot be empty"):
            TradeCalibrationResult(**kwargs)

    def test_dimensions_default_to_the_output_array(self, toy_calibration_result):
        """Built by hand without counts, the container reads (ns, nc) from ``ytot``."""
        res = toy_calibration_result
        kwargs = {k: getattr(res, k) for k in
                  ("a", "afd", "alpha", "beta", "k_endow", "l_endow", "invforT", "tax", "ytot")}
        bare = TradeCalibrationResult(**kwargs, n_final_demand=1)
        assert (bare.nc, bare.ns) == (3, 2)
        assert bare.a_4d.shape == (2, 3, 2, 3)
        with pytest.raises(ValueError, match="ytot"):
            TradeCalibrationResult(**{**kwargs, "ytot": np.ones(6)})

    def test_frozen_immutability(self, toy_calibration_result):
        res = toy_calibration_result
        with pytest.raises(FrozenInstanceError):
            res.alpha = np.zeros_like(res.alpha)

        with pytest.raises(FrozenInstanceError):
            res.n_countries = 99

    def test_property_aliases(self, toy_calibration_result):
        """KT, LT and ytot_base are the same arrays as k_endow, l_endow and ytot."""
        res = toy_calibration_result
        assert res.KT is res.k_endow
        assert res.LT is res.l_endow
        assert res.ytot_base is res.ytot

    def test_summary_dataframe_structure(self, toy_calibration_result):
        res = toy_calibration_result
        df = res.summary(detailed=True)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == res.n_countries
        assert list(df.index) == list(res.country_codes)
        expected_cols = {
            "Capital_Endow", "Labor_Endow", "Factor_Income",
            "Net_Transfers", "Gross_Output", "Mean_TFP", "Mean_TaxRate"
        }
        assert expected_cols.issubset(set(df.columns))

    def test_summary_aggregate_mode(self, toy_calibration_result):
        res = toy_calibration_result
        df = res.summary(detailed=False)
        assert isinstance(df, pd.DataFrame)
        assert "Global Gross Output" in df.index

    def test_to_frame_alias(self, toy_calibration_result):
        res = toy_calibration_result
        pd.testing.assert_frame_equal(res.to_frame(), res.summary())

    def test_to_markdown_formatting(self, toy_calibration_result):
        res = toy_calibration_result
        md = res.to_markdown()
        assert isinstance(md, str)
        assert "| Country |" in md or "| Country" in md
        assert "|---" in md
        for code in res.country_codes:
            assert code in md

    def test_to_latex_formatting(self, toy_calibration_result):
        res = toy_calibration_result
        latex = res.to_latex()
        assert isinstance(latex, str)
        assert "\\begin{tabular}" in latex
        assert "\\end{tabular}" in latex
        for code in res.country_codes:
            assert code in latex

    def test_downstream_result_stubs(self):
        """The equilibrium, PPP and batch containers build and render on their own."""
        x = np.ones(10)
        p = np.ones((1, 2, 2))
        eq = TradeEquilibriumResult(
            x_sol=x, p_sol=p, y_sol=p, r_sol=p, w_sol=p, T_sol=p, XN_sol=x
        )
        assert isinstance(eq.summary(), pd.DataFrame)
        assert "CONVERGED" in eq.to_markdown()
        assert "\\begin{tabular}" in eq.to_latex()
        # Without codes the country count comes from the price array.
        f = np.ones((1, 1, 2))
        eq2 = TradeEquilibriumResult(x_sol=x, p_sol=p, y_sol=p, r_sol=f, w_sol=f, T_sol=f, XN_sol=np.ones(1))
        assert list(eq2.summary(detailed=True).index) == ["C00", "C01"]
        assert eq2.bilateral_trade_frame().shape == (2, 2)

        gk = GearyKhamisResult(
            ppp=np.ones(3), pi=np.ones(2), real_gdp=np.ones(3), nominal_gdp=np.ones(3)
        )
        assert isinstance(gk.summary(), pd.DataFrame)
        assert "Country" in gk.to_markdown()
        assert "\\begin{tabular}" in gk.to_latex()
        assert "World_Real_GDP_GK" in gk.summary(detailed=False).index

        batch = ScenarioBatchResult(
            scenarios={"base": eq},
            ppp_results={"base": gk},
            real_gdp_table=pd.DataFrame({"base": [0.0]}, index=["C01"]),
            trade_balance_table=pd.DataFrame({"base": [0.0]}, index=["C01"]),
            cpi_table=pd.DataFrame({"base": [0.0]}, index=["C01"]),
        )
        assert isinstance(batch.summary(), pd.DataFrame)
        assert "C01" in batch.to_markdown()
        assert "\\begin{tabular}" in batch.to_latex()
        assert "C01" in batch.to_typst(metric="cpi")


class TestScenarioBatchResult:
    """Comparative tables across scenarios, built from equilibrium results."""

    @staticmethod
    def _result(xn, nc=3):
        p = np.ones((1, 2, nc))
        return TradeEquilibriumResult(
            x_sol=np.ones(4), p_sol=p, y_sol=p, r_sol=np.ones((1, 1, nc)),
            w_sol=np.ones((1, 1, nc)), T_sol=np.ones((1, 1, nc)), XN_sol=np.asarray(xn, float),
            country_codes=("A", "B", "C"),
        )

    def test_rest_of_world_closes_the_trade_balance(self):
        """With nc-1 net exports the last country is minus their sum (balances add to zero)."""
        batch = ScenarioBatchResult(scenarios={"base": self._result([2.0, -0.5]),
                                               "tariff_10": self._result([1.0, 0.25])})
        tb = batch.trade_balance_table
        assert list(tb.index) == ["A", "B", "C"]
        np.testing.assert_allclose(tb["base"], [2.0, -0.5, -1.5])
        np.testing.assert_allclose(tb["tariff_10"], [1.0, 0.25, -1.25])
        np.testing.assert_allclose(tb.sum(axis=0), 0.0, atol=1e-15)

    def test_country_count_without_codes_comes_from_the_results(self):
        p = np.ones((1, 2, 4))
        res = TradeEquilibriumResult(x_sol=np.ones(4), p_sol=p, y_sol=p, r_sol=p, w_sol=p,
                                     T_sol=p, XN_sol=np.zeros(3))
        batch = ScenarioBatchResult(scenarios={"base": res})
        assert batch.trade_balance_table.shape == (4, 1)

    def test_summary_is_a_metric_table(self):
        batch = ScenarioBatchResult(scenarios={"base": self._result([2.0, -0.5])})
        pd.testing.assert_frame_equal(batch.summary(), batch.to_frame("real_gdp_growth"))
        pd.testing.assert_frame_equal(batch.summary("trade_balance"), batch.trade_balance_table)
        with pytest.raises(ValueError, match="Unknown metric"):
            batch.summary("no_such_metric")

    def test_methods_are_the_general_set(self):
        """Comparative tables only: no fixed-country or fixed-layout report tables."""
        public = {n for n in vars(ScenarioBatchResult) if not n.startswith("_") and callable(getattr(ScenarioBatchResult, n))}
        assert public == {"keys", "values", "items", "with_geary_khamis", "to_frame", "summary",
                          "to_markdown", "to_latex", "to_typst"}


def test_result_containers_are_the_general_set():
    assert sorted(_results.__all__) == sorted([
        "TradeCalibrationResult",
        "TradeEquilibriumResult",
        "GearyKhamisResult",
        "ScenarioBatchResult",
        "OptimalTariffResult",
        "NashTariffResult",
        "WelfarePayoffMatrixResult",
    ])
    for name in _results.__all__:
        assert getattr(trade, name) is getattr(_results, name)
    assert trade.calibrate_trade_model is calibrate_trade_model
    assert set(_results.__all__) | {"calibrate_trade_model"} <= set(trade.__all__)


# ---------------------------------------------------------------------------
# Economics of the calibrated toy model
# ---------------------------------------------------------------------------

class TestToyCalibrationEconomics:

    def test_cobb_douglas_alpha_is_one_third(self, toy_calibration_result):
        """The toy table pays capital one third of value added."""
        res = toy_calibration_result
        np.testing.assert_allclose(res.alpha, 1.0 / 3.0, rtol=1e-12, atol=1e-12)

    def test_validation_checklist_passes(self, toy_calibration_result):
        res = toy_calibration_result
        checks = res.validate()
        for key in ("dim_a", "dim_afd", "dim_alpha", "dim_beta", "dim_tax", "dim_ytot",
                    "dim_k_endow", "dim_l_endow", "dim_invforT", "nonneg_ytot",
                    "nonneg_k_endow", "nonneg_l_endow", "nonneg_a", "nonneg_afd",
                    "alpha_bounds", "global_transfer_balance"):
            assert checks[key], key

    def test_technical_coefficients_bounds(self, toy_calibration_result):
        """0 <= a_ij and final-use sourcing shares sum to one."""
        res = toy_calibration_result
        assert np.all(res.a >= 0.0)
        assert np.all(res.afd >= 0.0)
        afd_sum = res.afd.sum(axis=0)
        np.testing.assert_allclose(afd_sum, 1.0, rtol=1e-10)

    def test_tensors_3d_and_4d(self, toy_calibration_result):
        res = toy_calibration_result
        assert res.a.shape == (6, 2, 3)
        assert res.afd.shape == (6, 1, 3)
        assert res.a_3d.shape == (6, 2, 3)
        assert res.afd_3d.shape == (6, 1, 3)
        assert res.a_4d.shape == (2, 3, 2, 3)
        assert res.afd_4d.shape == (2, 3, 1, 3)

    def test_inactive_sectors_division_warning_free(self):
        """A table with an inactive sector (y=0, l=0) calibrates without RuntimeWarnings."""
        nc, ns, nfd = 3, 2, 1
        n_ind = ns * nc
        toy_data = np.ones((n_ind + 3, n_ind + nfd * nc), dtype=np.float64) * 10.0
        # Sector 1 in country 0 is completely inactive
        toy_data[:, 1] = 0.0
        toy_data[1, :] = 0.0
        toy_data[n_ind:, 1] = 0.0

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            calib = calibrate_trade_model(toy_data, ns=ns, nc=nc, nfd=nfd, validate=False)

        runtime_warnings = [w for w in record if issubclass(w.category, RuntimeWarning)]
        assert len(runtime_warnings) == 0, f"Expected 0 RuntimeWarnings on inactive sectors, got: {runtime_warnings}"
        assert calib.ytot[0, 1, 0] == 0.0
        assert calib.alpha[0, 1, 0] == 0.0
        assert calib.beta[0, 1, 0] == 0.0

    def test_budget_violation_is_rejected(self, toy_icio_matrix):
        data, nc, ns, nfd = toy_icio_matrix
        bad = data.copy()
        bad[ns * nc + 1, 0] += 5.0     # labour income with no matching spending
        with pytest.raises(ValueError, match="budget"):
            calibrate_trade_model(bad)
        calibrate_trade_model(bad, validate=False)


# ---------------------------------------------------------------------------
# Calibration as read by the consistent accounting
# ---------------------------------------------------------------------------

class TestConsistentAccountingParameters:
    """``_accounting._parameters`` recovers final-use tax shares from a calibration."""

    @staticmethod
    def _two_country_table() -> np.ndarray:
        """Two countries, one sector, two final uses; taxes on production and final use."""
        Z = np.array([[10., 12.], [8., 14.]])
        F = np.array([[30., 8., 20., 20.], [15., 15., 50., 18.]])
        production_tax = np.array([4., 6.])
        final_tax = np.array([2., 1., 3., 2.])
        output = Z.sum(1) + F.sum(1)
        va = output - Z.sum(0) - production_tax
        return np.vstack([np.hstack([Z, F]), np.r_[production_tax, final_tax],
                          np.r_[2 * va / 3, np.zeros(4)], np.r_[va / 3, np.zeros(4)]])

    def test_tax_share_on_actual_final_purchases(self):
        from puremacro.trade import _accounting

        data = self._two_country_table()
        calib = calibrate_trade_model(data, nfd=2)
        shares, basic, inv = _accounting._parameters(calib)
        assert inv == 1
        expected = data[2, 2:] / (data[:2, 2:].sum(0) + data[2, 2:])
        np.testing.assert_allclose(shares.ravel(order="F"), expected, rtol=1e-14)
        np.testing.assert_allclose(basic.ravel(order="F"), data[:2, 2:].sum(0), rtol=1e-14)
        # The calibrated tax_fd divides by spending that includes the foreign balance.
        assert not np.allclose(calib.tax_fd.ravel(order="F"), expected)

    @pytest.mark.parametrize("omit_transfer", [False, True])
    def test_shares_without_the_source_table(self, omit_transfer):
        from dataclasses import replace

        from puremacro.trade import _accounting

        calib = calibrate_trade_model(self._two_country_table(), nfd=2)
        reduced = replace(calib, data_calibra=None, T=None if omit_transfer else calib.T)
        full_shares, full_basic, _ = _accounting._parameters(calib)
        shares, basic, _ = _accounting._parameters(reduced)
        np.testing.assert_allclose(shares, full_shares, rtol=0, atol=1e-14)
        np.testing.assert_allclose(basic, full_basic, rtol=1e-12)

    def test_invalid_calibrations_are_rejected(self):
        from dataclasses import replace

        from puremacro.trade import _accounting

        calib = calibrate_trade_model(self._two_country_table(), nfd=2)
        theta = calib.theta.copy()
        theta[0, 0, 0] = -0.01
        with pytest.raises(ValueError, match="nonnegative"):
            _accounting._parameters(replace(calib, theta=theta))
        with pytest.raises(ValueError, match="sum to zero"):
            _accounting._parameters(replace(calib, invforT=calib.invforT + 1.0))
        bad = calib.data_calibra.copy()
        bad[:2, 2] = -1.0
        with pytest.raises(ValueError, match="Signed final-use"):
            _accounting._parameters(replace(calib, data_calibra=bad))


# ---------------------------------------------------------------------------
# Data module registries and the structured table
# ---------------------------------------------------------------------------

class TestTradeDataModule:

    def test_code_registries(self):
        eu = get_eu_country_codes()
        assert len(eu) == 27 and list(EU_COUNTRY_CODES) == eu
        assert "DEU" in eu and "FRA" in eu and "ITA" in eu
        assert get_final_demand_codes() == ["C", "I", "Cx"] == list(CANONICAL_FINAL_DEMAND_CODES)

    def test_icio_data_container_and_properties(self, tmp_path):
        """ICIOData slices a processed table and its accounting identities hold."""
        nc, ns = 6, 5
        n = nc * ns
        path = tmp_path / "table.npz"
        np.savez_compressed(path, data=synthetic_icio_table(nc, ns, seed=4))
        icio = load_icio_data(path, return_structured=True)
        assert isinstance(icio, ICIOData)
        assert icio.matrix.shape == (n + 3, n + 3 * nc)
        assert icio.intermediate_matrix.shape == (n, n)
        assert icio.final_demand_matrix.shape == (n, 3 * nc)
        assert icio.net_taxes_intermediate.shape == (n,)
        assert icio.net_taxes_final_demand.shape == (3 * nc,)
        assert icio.labor_va.shape == (n,)
        assert icio.capital_va.shape == (n,)
        assert icio.gross_output.shape == (n,)
        valid = icio.labor_va > 0
        np.testing.assert_allclose(icio.capital_va[valid] / icio.labor_va[valid], 0.5, rtol=1e-12)
        row_sales = np.sum(icio.matrix[:n, :], axis=1)
        np.testing.assert_allclose(icio.gross_output, row_sales, rtol=1e-12)

    def test_load_icio_data_calibration_validation(self, tmp_path):
        """A table read from disk calibrates and passes every validate() check."""
        path = tmp_path / "table.npz"
        np.savez_compressed(path, data=synthetic_icio_table(6, 5, seed=4))
        checks = calibrate_trade_model(load_icio_data(path)).validate()
        failing = [k for k, v in checks.items() if not v]
        assert not failing, failing


# ---------------------------------------------------------------------------
# Subpackage integrity
# ---------------------------------------------------------------------------

class TestTradeSubpackageIntegrity:

    def test_no_machine_specific_paths_in_trade_modules(self):
        """No home directory, user path, cloud account or email address in puremacro.trade."""
        forbidden = {
            "unix home directory": re.compile(r"[\"']/(?:Users|home)/[A-Za-z0-9._-]+/"),
            "windows user directory": re.compile(r"[A-Za-z]:\\\\Users\\\\"),
            "cloud account in a path": re.compile(r"GoogleDrive-[^\"'\s/]+|OneDrive-[^\"'\s/]+"),
            "bare email address": re.compile(r"[\"'][^\"'\s]*@(?:gmail|hotmail|outlook|yahoo)\.[a-z]+"),
        }

        trade_pkg_dir = Path(trade.__file__).resolve().parent
        sources = sorted(trade_pkg_dir.rglob("*.py"))
        assert len(sources) >= 4, f"Expected at least 4 trade modules, found {len(sources)}"

        hits = []
        for path in sources:
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if line.lstrip().startswith("#"):
                    continue
                for label, pattern in forbidden.items():
                    if pattern.search(line):
                        hits.append(f"{path.name}:{lineno} [{label}]: {line.strip()}")

        assert not hits, "Machine-specific path in puremacro.trade:\n  " + "\n  ".join(hits)
