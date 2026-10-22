"""Parity at the quarterly/monthly boundary, including duplicate precedence."""
import numpy as np
import pandas as pd
import pytest

from puremacro.financial_panel import _EMPTY, _project_quarterly_to_monthly


def _scalar_reference(frame):
    records = []
    for row in frame.to_dict("records"):
        dt = pd.Timestamp(row["date"])
        name = str(row["variable"])
        name = name[:-2] + "_m" if name.endswith("_q") else (
            name if name.endswith("_m") else name + "_m"
        )
        for offset in range(min(3, 13 - dt.month)):
            records.append({
                "code": row["code"],
                "date": pd.Timestamp(year=dt.year, month=dt.month + offset, day=1),
                "variable": name, "value": float(row["value"]),
                "sa_source": "quarterly_ffill" if offset else row["sa_source"],
                "source": f"resampled_from_Q:{row['source']}", "is_imputed": offset != 0,
            })
    return pd.DataFrame(records).drop_duplicates(["code", "date", "variable"], keep="first")


@pytest.mark.parametrize("dates", [
    ["2020-01-01", "2020-04-01T00:00:00", "2020-01-01"],
    pd.to_datetime(["2020-01-01", "2020-04-01", "2020-01-01"]),
    ["2020-01-01T00:30:00+09:00", "2020-04-01T00:30:00-03:00", "2020-01-01T00:30:00+09:00"],
])
@pytest.mark.parametrize("values", [[1.0, 2.0, 999.0], [True, False, True], ["1", "2.5", "9"]])
def test_projection_preserves_scalar_parsing_schema_order_and_first_duplicate(dates, values):
    frame = pd.DataFrame({
        "code": ["USA"] * 3, "date": dates,
        "variable": ["credit_q", "credit_q", "credit_m"], "value": values,
        "source": ["bis", "bis", "override"], "sa_source": ["nsa"] * 3,
        "private_column": [10, 20, 30],
    }, index=[7, 7, 1])
    before = frame.copy(deep=True)
    result = _project_quarterly_to_monthly(frame)
    pd.testing.assert_frame_equal(result, _scalar_reference(frame))
    assert list(result.columns) == list(_EMPTY.columns)
    assert result["value"].dtype == np.dtype(float)
    assert result["is_imputed"].dtype == np.dtype(bool)
    assert len(result) == 6
    pd.testing.assert_frame_equal(frame, before)


def test_projection_empty_output_is_independent():
    result = _project_quarterly_to_monthly(pd.DataFrame())
    pd.testing.assert_frame_equal(result, _EMPTY)
    result["extra"] = []
    assert "extra" not in _EMPTY


def test_projection_year_end_and_variable_suffixes():
    frame = pd.DataFrame({
        "code": ["USA", "MEX", "GBR"],
        "date": pd.to_datetime(["2020-10-01", "2020-11-01", "2020-12-01"]),
        "variable": ["credit_q", "credit", "credit_m"], "value": [1, 2, 3],
        "source": ["a", "b", "c"], "sa_source": ["sa", None, "nsa"],
    })
    result = _project_quarterly_to_monthly(frame)
    pd.testing.assert_frame_equal(result, _scalar_reference(frame))
    assert result.groupby("code").size().to_dict() == {"GBR": 1, "MEX": 2, "USA": 3}
    assert (result.date.dt.year == 2020).all()


@pytest.mark.parametrize("dates", [[pd.NaT], pd.to_datetime([pd.NaT]), ["not a date"]])
def test_projection_does_not_silently_drop_invalid_dates(dates):
    frame = pd.DataFrame({
        "code": ["USA"], "date": dates, "variable": ["credit_q"],
        "value": [1], "source": ["bis"], "sa_source": ["nsa"],
    })
    with pytest.raises(ValueError):
        _project_quarterly_to_monthly(frame)
