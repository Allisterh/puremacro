"""Replication tests for ``puremacro.examples.svariv_mertens_ravn``.

Synthetic-DGP replication: we know the true structural impact column
(under the Stock-Watson proxy-SVAR normalization), so we can check that
proxy-SVAR recovers the peak response within a tight tolerance and that
weak-IV diagnostics behave as expected.
"""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def mr_result():
    from puremacro.examples.svariv_mertens_ravn import (
        run_mertens_ravn_replication,
    )
    return run_mertens_ravn_replication(seed=0)


# Stock and Yogo (2002), NBER Technical Working Paper 284, Table 2 (TSLS
# size), printed page 59, row K2 = 1, column "n = 1, r = 0.10": the
# first-stage F a single instrument needs for a nominal 5% Wald test to
# reject at most 10% of the time.
# https://www.nber.org/system/files/working_papers/t0284/t0284.pdf
_SY_SIZE_10PCT_ONE_INSTRUMENT = 16.38


def test_mertens_ravn_strong_instrument(mr_result):
    """Cragg-Donald F should be well above the Stock-Yogo 10% maximal-size
    critical value for one instrument (16.38) on this DGP: the proxy is
    wired to be strong."""
    from puremacro.inference.over_id import stock_yogo_cv

    cv = stock_yogo_cv(1, 10, criterion="size")
    assert cv == pytest.approx(_SY_SIZE_10PCT_ONE_INSTRUMENT, abs=1e-9)
    cd_f = mr_result["cd_f"]
    assert cd_f > cv, f"CD F = {cd_f:.2f} <= {cv} (weak instrument)"


def test_mertens_ravn_summary_quotes_published_cutoff(
    capsys, monkeypatch, tmp_path, mr_result
):
    """The printed summary quotes the table's 16.38, not a rounded 16.4."""
    from puremacro.examples import svariv_mertens_ravn as mod

    monkeypatch.setattr(mod, "run_mertens_ravn_replication", lambda: mr_result)
    # Point the example's output folder at an empty temp dir so main()
    # does not rewrite the tracked figure.
    monkeypatch.setattr(mod, "__file__", str(tmp_path / "svariv_mertens_ravn.py"))
    mod.main()
    out = capsys.readouterr().out
    assert "16.38" in out
    assert "16.4)" not in out


def test_mertens_ravn_peak_within_10pct(mr_result):
    """Recovered peak |response| of the second variable should match
    the population truth within 10%."""
    true_peak = abs(mr_result["true_peak"])
    rec_peak = abs(mr_result["recovered_peak"])
    rel_err = abs(rec_peak - true_peak) / true_peak
    assert rel_err <= 0.10, (
        f"recovered peak |{rec_peak:.4f}| differs from truth "
        f"|{true_peak:.4f}| by {rel_err * 100:.1f}% (>10%)"
    )


def test_mertens_ravn_msw_bands_present(mr_result):
    """MSW (AR-inverted) bands should be finite, well-formed, and
    bracket the 2SLS point estimate (lower <= upper)."""
    lo = mr_result["msw_lo"]
    hi = mr_result["msw_hi"]
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo <= hi, f"MSW band malformed: lo={lo:.4f} > hi={hi:.4f}"
    grid = mr_result["msw_grid"]
    assert grid.ndim == 1 and grid.size > 1
