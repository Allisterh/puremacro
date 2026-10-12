"""scf_aiyagari: calibration hits its target, model moments come from the
same estimators as the data, and the run writes its outputs. Offline: the
SCF is replaced by a synthetic frame with the SCF's design."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from puremacro.examples import scf_aiyagari as ex
from puremacro.fetch.micro import MicroFrame, SurveyDesign

SMALL = {"n_a": 60, "n_z": 5}


def _synthetic_scf(n=300, seed=0):
    rng = np.random.default_rng(seed)
    design = SurveyDesign.scf()
    w = rng.uniform(1e4, 6e4, n)
    reps = w[:, None] * rng.uniform(0.0, 2.0, (n, len(design.replicate_weights)))
    income = rng.lognormal(11.1, 0.8, n)
    frames = []
    for i in range(1, 6):
        f = pd.DataFrame(reps, columns=list(design.replicate_weights))
        f["wgt"], f["implicate"] = w, i
        f["networth"] = np.where(rng.random(n) < 0.1, -rng.lognormal(9, 1, n),
                                 rng.lognormal(12, 1.9, n))
        f["income"] = income * rng.uniform(0.95, 1.05, n)
        frames.append(f)
    return MicroFrame(pd.concat(frames, ignore_index=True), design, "household",
                      "fed.scf", "synthetic")


@pytest.fixture(scope="module")
def solved():
    return ex.solve(0.95, **SMALL)


def test_model_frame_reproduces_the_solution(solved):
    mf = ex.model_frame(solved)
    assert mf.data["mass"].sum() == pytest.approx(1.0, abs=1e-10)
    assert mf.mean("networth").loc[0, "estimate"] == pytest.approx(solved.aggregates["K"], rel=1e-10)
    inc = solved.mean("w*exp(z) + r*a")
    assert mf.ratio("networth", "income").loc[0, "estimate"] == pytest.approx(
        solved.aggregates["K"] / inc, rel=1e-10)


def test_model_moments_are_sane_and_have_no_sampling_error(solved):
    m = ex.model_moments(solved)
    assert list(m.index) == list(ex.MOMENTS)
    assert m["se"].isna().all()
    est = m["estimate"]
    assert 0 < est["Gini"] < 1
    assert 0 < est["top 1% share"] < est["top 10% share"] < 1
    assert est["bottom 50% share"] + est["top 10% share"] < 1
    # Mass at the borrowing limit equals the solved distribution's first row.
    assert est["share with wealth <= 0"] == pytest.approx(solved.distribution[0].sum(), rel=1e-12)


def test_model_median_ratio_moves_smoothly_with_beta():
    # On a grid the inverse-CDF median sits on a node and moves in steps;
    # the calibration needs the interpolated one to rise with patience.
    vals = [ex.model_moments(ex.solve(b, **SMALL)).loc[ex.TARGETS["median"], "estimate"]
            for b in np.arange(0.90, 0.951, 0.01)]
    assert np.all(np.diff(vals) > 0)


def test_interpolated_quantile_pools_equal_values():
    x = np.array([1.0, 2.0, 2.0, 3.0])
    w = np.array([1.0, 1.0, 1.0, 1.0])
    # Pooled knots (1, .25), (2, .75), (3, 1): the median is 1.5.
    assert ex._interpolated_quantile(x, w, 0.5) == pytest.approx(1.5)


def test_calibration_hits_an_attainable_target():
    target = float(ex.moments(ex.model_frame(ex.solve(0.94, **SMALL))).loc[ex.TARGETS["median"], "estimate"])
    cal = ex.calibrate("median", target, 0.1, bracket=(0.90, 0.99), **SMALL)
    assert cal.attained
    assert cal.beta == pytest.approx(0.94, abs=2e-3)
    assert cal.moments.loc[ex.TARGETS["median"], "estimate"] == pytest.approx(target, rel=1e-3)
    # Delta method: the target's own model se is the data se it was fed.
    assert cal.moments.loc[ex.TARGETS["median"], "se"] == pytest.approx(0.1, rel=1e-6)
    assert cal.se_beta > 0
    assert 0 < cal.r < 1 / cal.beta - 1


def test_unattainable_target_is_reported_not_forced():
    cal = ex.calibrate("mean", 1e3, 1.0, bracket=(0.90, 0.95), **SMALL)
    assert not cal.attained
    assert cal.beta == 0.95
    assert np.isnan(cal.se_beta) and cal.moments["se"].isna().all()
    assert "outside what beta" in cal.notes[0]


def test_run_application_writes_outputs(tmp_path):
    res = ex.run_application(_synthetic_scf(), output=tmp_path, targets=("median",), **SMALL)
    table = res["table"]
    assert list(table.index) == list(ex.MOMENTS)
    assert list(table.columns) == ["SCF", "SCF se", "model (median target)", "model (median target) se"]
    assert table["SCF se"].gt(0).all()
    for name in ("moments.csv", "calibration.csv", "lorenz.csv", "manifest.json", "report.md"):
        assert (tmp_path / name).exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["data"]["vintage"] == "synthetic"
    assert manifest["calibration"]["median"]["attained"] is True
    lorenz = pd.read_csv(tmp_path / "lorenz.csv", index_col=0)
    assert lorenz.iloc[0].abs().max() < 1e-12 and (lorenz.iloc[-1] - 1).abs().max() < 1e-12
