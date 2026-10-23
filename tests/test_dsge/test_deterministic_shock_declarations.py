"""Dated shock declarations preserve cumulative and overwrite semantics."""
import numpy as np
import pytest

from puremacro.dsge import parse_mod


BASE = "var x; varexo e; model; x=.5*x(-1)+e; end;"


def test_deterministic_model_can_omit_varexo():
    parsed = parse_mod("var x; model; x=.5*x(-1); end;")
    assert parsed["shocks"] == []
    assert parsed["_compiled"].n_e == 0


def test_dated_shocks_accumulate_and_only_replace_the_redeclared_date():
    parsed = parse_mod(BASE + """
        shocks; var e; periods 1,3; values 1,3; end;
        shocks; var e; periods 2,3; values 2,30; end;
    """)
    assert parsed["anticipated_shocks"]["e"] == {"periods": [1, 2, 3], "values": [1, 2, 30]}


def test_shocks_overwrite_clears_prior_dates_and_covariance():
    parsed = parse_mod(BASE + """
        shocks; var e; stderr 2; var e; periods 1; values 4; end;
        shocks(overwrite); var e; periods 2; values 3; end;
    """)
    assert parsed["anticipated_shocks"]["e"] == {"periods": [2], "values": [3]}
    assert parsed["shock_cov"][0, 0] == 0


def test_dated_values_resolve_preceding_calibration_in_structured_parser():
    parsed = parse_mod(BASE + """
        parameters impulse scale; scale=2; impulse=scale/4;
        shocks; var e; periods 1,2; values impulse,2*impulse; end;
    """)
    assert "_dag" in parsed
    assert parsed["anticipated_shocks"]["e"]["values"] == [0.5, 1.0]


@pytest.mark.parametrize("values", ["unresolved", "exp(1000)"])
def test_invalid_dated_value_expressions_do_not_enter_legacy_fallback(values):
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="Shock values"):
        parse_mod(BASE + f"shocks; var e; periods 1; values {values}; end;")


@pytest.mark.parametrize("periods,values", [("0", "1"), ("1,2", "1,2,3"), ("1.5", "1"), ("1:2.5", "1"), ("1:1.5:3", "1")])
def test_invalid_shock_dates_or_value_counts_are_rejected(periods, values):
    with pytest.raises(ValueError, match="Shock periods"):
        parse_mod(BASE + f"shocks; var e; periods {periods}; values {values}; end;")
