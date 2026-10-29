"""Network replication: GRS of the FF three-factor model on the 25 Size-B/M portfolios.

Published value
---------------
Fama, E.F. and French, K.R. (2015). A five-factor asset pricing model.
Journal of Financial Economics 116(1), 1-22. Version read: the Author's
Accepted Manuscript (Elsevier PII S0304-405X(14)00232-3, DOI
10.1016/j.jfineco.2014.10.010), Table 5 "Summary statistics for tests of
three-, four-, and five-factor models; July 1963 to December 2013, 606
months", Panel A "25 Size-B/M portfolios", row "HML" (the factors that
augment R_M - R_F and SMB, i.e. the three-factor model), column "2x3
Factors", GRS = 3.62 (A|a_i| = 0.102). Manuscript page 38 (PDF page 40),
read from the rendered page on 2026-10-28.

The data library is revised after publication (CRSP corrections, breakpoint
updates), so the comparison allows 15%.
"""
from __future__ import annotations

import numpy as np
import pytest

PUBLISHED_GRS_FF3_25_SIZE_BM = 3.62  # FF (2015) Table 5, Panel A, HML row, 2x3 factors


@pytest.mark.network
def test_grs_ff3_25_size_bm_matches_fama_french_2015():
    from puremacro.asset_pricing import time_series_test
    from puremacro.fetch.kenfrench_factors import fetch

    ff3 = fetch("ff3")
    p25 = fetch("25_size_bm")
    s = slice("1963-07-01", "2013-12-01")
    excess = p25.loc[s].drop(columns="RF").sub(p25.loc[s, "RF"], axis=0)
    res = time_series_test(excess, ff3.loc[s, ["Mkt-RF", "SMB", "HML"]], hac_lags=0)
    print(f"GRS ours = {res.grs_stat:.4f} (published {PUBLISHED_GRS_FF3_25_SIZE_BM}), "
          f"p = {res.grs_pvalue:.2e}, T = {res.n_obs}, mean |a| = "
          f"{100 * np.abs(res.alphas).mean():.3f}%")
    assert res.n_obs == 606
    assert abs(res.grs_stat / PUBLISHED_GRS_FF3_25_SIZE_BM - 1) <= 0.15
    # the published statistic and ours both reject at 5%
    from scipy import stats

    assert stats.f.sf(PUBLISHED_GRS_FF3_25_SIZE_BM, 25, 606 - 25 - 3) < 0.05
    assert res.grs_pvalue < 0.05


@pytest.mark.network
def test_fetchers_live_smoke():
    from puremacro.asset_pricing import fred_consumption_growth, fred_risk_free
    from puremacro.fetch.kenfrench_factors import fetch

    for d in ("ff3", "ff5", "mom", "25_size_bm"):
        df = fetch(d)
        assert len(df) > 500 and "RF" in df.columns
        assert df["RF"].dropna().abs().max() < 0.05  # decimals, not percent
    g = fred_consumption_growth()
    assert len(g) > 100 and 0.95 < g.mean() < 1.05
    rf = fred_risk_free()
    assert len(rf) > 500 and rf.abs().max() < 0.05
