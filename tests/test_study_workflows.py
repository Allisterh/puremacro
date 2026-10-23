"""Complete offline workflows retain real inputs, explicit units and solver evidence."""
import numpy as np
import pandas as pd
import pytest

from puremacro.examples.study_workflows import (
    mexico_data, run_mexico, run_household, run_tariff, tariff_table,
)
from puremacro.study import load_study, save_study


@pytest.fixture(scope="module")
def workflows():
    return run_mexico(n_boot=19), run_household(), run_tariff()


def test_mexico_observations_are_consecutive_and_use_declared_units(workflows):
    data = mexico_data()
    assert data.index.equals(pd.period_range("2000-01", "2023-12", freq="M"))
    assert data.shape == (288, 2) and np.isfinite(data.to_numpy()).all()
    result = workflows[0]
    table = result.to_frame().set_index(["response", "horizon"])
    assert table.loc[("rate_3m", 0), "estimate"] == pytest.approx(.25)
    assert table.loc[("inflation_yoy", 0), "estimate"] == 0
    assert table.loc[("rate_3m", 0), "shock_unit"] == "percentage points per annum"
    assert "assumption" in result.inputs.attrs["identification_note"]


def test_mexico_data_rights_survive_archive_and_report(workflows, tmp_path):
    result = workflows[0]
    source = result.inputs.attrs["source"]
    assert source["retrieval_date"] == "not recorded in original course snapshot"
    rights = source["data_rights"]
    assert rights["code_license_applies"] is False
    assert rights["terms_url"] == "https://www.oecd.org/en/about/terms-conditions.html"
    assert rights["terms_section"] == "3. Data"
    assert "later recipients" in rights["downstream_requirement"]
    assert "not the unrecorded date" in rights["review_date_note"]
    for details in source["files"].values():
        assert details["dataset"] == "Main Economic Indicators"
        assert "OECD" in details["citation"]
        assert details["series"] in details["citation"]
        assert details["source_url"] in details["citation"]
    restored = load_study(save_study(result))
    assert restored.inputs.attrs["source"] == source
    result.write_report(tmp_path / "mexico")
    exported = load_study(tmp_path / "mexico" / "study.pmstudy")
    assert exported.inputs.attrs["source"] == source


def test_household_transfer_preserves_resource_timing(workflows):
    result = workflows[1]
    frame = result.to_frame().pivot(index="horizon", columns="response", values="estimate")
    # A is assets entering each date. The transfer is consumed or carried to t+1.
    assert frame.A.iloc[0] == pytest.approx(0, abs=1e-12)
    assert 0 < frame.C.iloc[0] < .05
    np.testing.assert_allclose(frame.C.iloc[:-1] + frame.A.iloc[1:].to_numpy(),
                               1.01*frame.A.iloc[:-1] + result.evidence["shock_path"].transfer.iloc[:-1],
                               atol=2e-10)
    assert "external transfer" in result.provenance["financing"]
    assert result.model_source.startswith("\nparameters")
    assert abs(result.inputs.mass.sum() - 1) < 1e-10


def test_oecd_aggregation_keeps_every_economy_and_flow_block():
    from puremacro.trade.data import load_icio_data
    original = load_icio_data(return_structured=True)
    small, countries, sectors, metadata = tariff_table()
    n = len(original.country_codes)*len(original.sector_codes)
    assert set(metadata["country_mapping"]) == set(original.country_codes)
    assert countries == ("USA", "MEX", "REST") and len(sectors) == 3
    assert sectors[-1] == "OTHER_INDUSTRIES"
    for activity in ("D", "E", "F"):
        assert metadata["sector_mapping"][activity] == "OTHER_INDUSTRIES"
    for old, new in ((original.matrix[:n, :n], small[:9, :9]),
                     (original.matrix[:n, n:], small[:9, 9:]),
                     (original.matrix[n:, :n], small[9:, :9]),
                     (original.matrix[n:, n:], small[9:, 9:])):
        assert new.sum() == pytest.approx(old.sum(), rel=1e-13)


def test_tariff_outputs_have_explicit_gross_output_scope(workflows):
    result = workflows[2]
    assert result.to_frame().response.tolist() == ["output_USA", "output_MEX", "output_REST"]
    assert (result.to_frame().unit == "percent of baseline gross output").all()
    assert set(result.audits.status) == {"passed"}
    assert max(result.provenance["max_residual"].values()) < 1e-5
    assert any("not GDP" in s for s in result.provenance["limitations"])
    assert result.audits.detail.str.contains("mixed native equation units").all()
    for name in ("baseline", "tariff"):
        checks = result.evidence[name + "_accounting_checks"].set_index("check")
        physical = result.evidence[name + "_physical_residuals"]
        assert physical.shape == (30, 2)
        assert abs(physical.residual).max() == pytest.approx(
            checks.loc["physical_equations", "max_absolute_residual"])
        assert "gdp_income_expenditure" in checks.index


def test_complete_report_archives_roundtrip(workflows, tmp_path):
    for i, result in enumerate(workflows):
        restored = load_study(save_study(result))
        assert restored.manifest == result.manifest
        pd.testing.assert_frame_equal(restored.inputs, result.inputs)
        if i == 2:
            for name, frame in result.evidence.items():
                pd.testing.assert_frame_equal(restored.evidence[name], frame)
        result.write_report(tmp_path / str(i))
        reread = load_study(tmp_path / str(i) / "study.pmstudy")
        pd.testing.assert_frame_equal(reread.to_frame(), result.to_frame())


def test_model_experiments_reject_undefined_shock_sizes():
    with pytest.raises(ValueError, match="positive"):
        run_household(amount=0)
    with pytest.raises(ValueError, match="positive"):
        run_tariff(rate=0)
