"""The sovereign-risk policy brief example: scenarios, record builder, CLI."""
from __future__ import annotations

import copy
import inspect
import json
import os

import numpy as np
import pytest
from scipy.stats import t as student_t

from puremacro.examples import first_brief_sovereign as brief
from puremacro.sovereign import solve_sovereign_default
from puremacro.sovereign.measurement import MOMENTS

TINY_GRIDS = [(61, 11), (31, 11)]
TINY_DESIGN = {"periods": 5_000, "seeds": [1, 2, 3], "burn": 100}


@pytest.fixture(scope="module")
def tiny_record():
    return brief.build_record(TINY_GRIDS, design=TINY_DESIGN, log=lambda *a: None)


def test_scenarios_are_one_at_a_time():
    assert list(brief.SCENARIOS) == ["baseline", "theta=0.15", "theta=0.5", "sigma_y=0.02",
                                     "sigma_y=0.03", "rho_y=0.9", "rho_y=0.97"]
    assert brief.SCENARIOS["baseline"] == {}
    expected = {"theta": {.15, .50}, "sigma_y": {.020, .030}, "rho_y": {.90, .97}}
    seen = {key: set() for key in expected}
    for name, overrides in brief.SCENARIOS.items():
        if name == "baseline":
            continue
        assert len(overrides) == 1
        (key, value), = overrides.items()
        assert name == f"{key}={value:g}"
        assert value != brief.BASELINE[key]
        seen[key].add(value)
    assert seen == expected


def test_parse_scenario():
    assert brief.parse_scenario("theta=0.15") == ("theta=0.15", {"theta": .15})
    assert brief.parse_scenario("theta=0.50")[0] == "theta=0.5"
    assert brief.parse_scenario("baseline") == ("baseline", {})
    with pytest.raises(ValueError, match="theta"):
        brief.parse_scenario("beta=0.9")
    with pytest.raises(ValueError):
        brief.parse_scenario("theta=nan")
    with pytest.raises(ValueError):
        brief.parse_scenario("theta")


def test_baseline_is_the_public_calibration():
    # Literal copy of the kwargs of _public_monotone in tests/test_arellano_2008_study.py.
    assert brief.BASELINE == dict(beta=.95282, r_star=.017, theta=.282, rho_y=.945, sigma_y=.025,
                                  b_min=-.15, b_max=.33, sigma_eps=0, y_penalty_type="arellano",
                                  enforce_monotonicity=False, tol=1e-10, max_iter=10000,
                                  policy_search="monotone")
    assert brief.GRIDS == {"converged": (6401, 641), "error": (3201, 641), "coarse": (801, 161)}
    assert brief.DESIGN == {"periods": 500_000, "seeds": list(range(1979, 1995)), "burn": 1_000}


def test_tolerances_match_the_registered_study_values():
    from tools.reference_validation import check_arellano_grids

    for m in MOMENTS:
        assert brief.TOLERANCES[m] == check_arellano_grids.TOLERANCES["debt_pct"] == 0.10
    assert set(brief.TOLERANCES) == set(MOMENTS)


def _check_summary(case):
    sim = case["simulation"]
    n = len(sim["seeds"])
    t_crit = student_t.ppf(.975, n - 1)
    for m in MOMENTS:
        values = np.asarray(sim["per_seed"][m])
        assert len(values) == n
        se = values.std(ddof=1) / np.sqrt(n)
        summary = sim["summary"][m]
        assert summary["mean"] == pytest.approx(values.mean(), abs=1e-12)
        assert summary["se"] == pytest.approx(se, abs=1e-12)
        assert summary["ci95"] == pytest.approx([values.mean() - t_crit * se, values.mean() + t_crit * se], abs=1e-12)


def test_build_record_schema_and_recomputation(tiny_record, tmp_path):
    record = tiny_record
    for key in ("schema", "package_version", "versions", "source_sha256", "design", "baseline",
                "tolerances", "grids", "peak_rss_gb", "scenarios"):
        assert key in record, key
    assert record["schema"] == 1
    assert set(record["versions"]) == {"python", "numpy", "scipy", "pandas"}
    assert record["source_sha256"] == brief.source_hashes()
    assert record["grids"] == [[61, 11], [31, 11]]
    assert record["design"] == TINY_DESIGN and record["baseline"] == brief.BASELINE
    assert list(record["scenarios"]) == list(brief.SCENARIOS)
    solver_kwargs = set(inspect.signature(solve_sovereign_default).parameters)
    for name, scenario in record["scenarios"].items():
        assert scenario["overrides"] == brief.SCENARIOS[name]
        assert list(scenario["cases"]) == ["61x11", "31x11"]
        for case in scenario["cases"].values():
            assert set(case) == {"n_b", "n_y", "parameters", "solver", "simulation", "welfare_pct"}
            assert set(case["parameters"]) == solver_kwargs
            assert set(case["solver"]) == {"converged", "iterations", "residual", "monotone_certification_gap",
                                           "exhaustive_bellman_residual", "elapsed_s"}
            assert case["simulation"]["seeds"] == TINY_DESIGN["seeds"]
            _check_summary(case)
        expected = brief.paired_change(scenario["cases"]["31x11"], scenario["cases"]["61x11"])
        assert scenario["grid_change"] == expected
    text = json.dumps(record, allow_nan=False)
    for forbidden in (os.sep + "Users", "/home/", "/private/", str(tmp_path)):
        assert forbidden not in text


def test_paired_change_rule(tiny_record):
    cases = tiny_record["scenarios"]["baseline"]["cases"]
    coarse, fine = cases["31x11"], cases["61x11"]
    change = brief.paired_change(coarse, fine)
    for m in MOMENTS:
        d = np.asarray(fine["simulation"]["per_seed"][m]) - np.asarray(coarse["simulation"]["per_seed"][m])
        half = student_t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        entry = change["moments"][m]
        assert entry["change"] == pytest.approx(d.mean(), abs=1e-12)
        assert entry["absolute_change_bound"] == pytest.approx(abs(d.mean()) + half, abs=1e-12)
        bound, tol = entry["absolute_change_bound"], brief.TOLERANCES[m]
        assert entry["status"] == ("passed" if bound <= tol else "grid_shift_exceeds_tolerance"
                                   if abs(d.mean()) > tol else "sampling_precision_insufficient")
    assert change["passed"] == all(change["moments"][m]["status"] == "passed" for m in MOMENTS)
    assert change["welfare_change_pct"] == pytest.approx(fine["welfare_pct"] - coarse["welfare_pct"], abs=1e-15)
    other = copy.deepcopy(coarse)
    other["simulation"]["seeds"] = [1, 2, 4]
    with pytest.raises(ValueError, match="seed"):
        brief.paired_change(other, fine)


def test_freeze_resume_discards_stale_partial(tmp_path, monkeypatch):
    grids = [(31, 11)]
    calls = []
    real = brief.solve_scenario

    def counting(overrides, n_b, n_y):
        calls.append((tuple(overrides.items()), n_b, n_y))
        return real(overrides, n_b, n_y)

    monkeypatch.setattr(brief, "solve_scenario", counting)
    partial = tmp_path / "sweep.partial.json"

    # A complete run writes the partial after every case.
    first = brief.build_record(grids, design=TINY_DESIGN, partial=partial, log=lambda *a: None)
    assert len(calls) == len(brief.SCENARIOS)
    saved = json.loads(partial.read_text())
    assert saved["source_sha256"] == brief.source_hashes()

    # A partial from an older source is discarded: every case is re-solved.
    stale = copy.deepcopy(saved)
    stale["source_sha256"] = {k: "0" * 64 for k in stale["source_sha256"]}
    partial.write_text(json.dumps(stale))
    calls.clear()
    messages = []
    second = brief.build_record(grids, design=TINY_DESIGN, partial=partial, log=messages.append)
    assert len(calls) == len(brief.SCENARIOS)
    assert any("discard" in str(m).lower() and partial.name in str(m) for m in messages)
    assert second["source_sha256"] == brief.source_hashes()

    # A matching partial with one finished case reuses it.
    matching = copy.deepcopy(saved)
    for name in list(matching["scenarios"])[1:]:
        del matching["scenarios"][name]
    partial.write_text(json.dumps(matching))
    calls.clear()
    third = brief.build_record(grids, design=TINY_DESIGN, partial=partial, log=lambda *a: None)
    assert len(calls) == len(brief.SCENARIOS) - 1
    assert third["scenarios"]["baseline"]["cases"] == first["scenarios"]["baseline"]["cases"]

    # A partial with a different design is discarded too.
    other_design = {**TINY_DESIGN, "periods": 4_000}
    partial.write_text(json.dumps(saved))
    calls.clear()
    brief.build_record(grids, design=other_design, partial=partial, log=lambda *a: None)
    assert len(calls) == len(brief.SCENARIOS)
