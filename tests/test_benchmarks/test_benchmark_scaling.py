"""Numerical checks and failure reporting of the offline scaling runner."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy import sparse

from benchmarks import benchmark_scaling as scaling


def test_smoke_cli_records_all_workloads_and_measurement_scope(tmp_path):
    output = tmp_path / "scaling.json"
    result = subprocess.run(
        [sys.executable, str(Path(scaling.__file__)), "--profile", "smoke",
         "--repetitions", "1", "--output", str(output)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report == json.loads(result.stdout)
    assert report["passed"]
    assert len(report["records"]) == 10
    assert {r["family"] for r in report["records"]} == {"pf", "hank", "var", "mrio"}
    assert report["measurement"]["runtime_repetitions"] == 1
    assert report["measurement"]["runtime_warmups"] == 1
    assert "not process RSS" in report["measurement"]["memory_metric"]
    assert report["environment"]["python"]
    assert "numpy_build" in report["environment"]
    for row in report["records"]:
        assert row["accuracy"]["passed"] and row["status"] == "passed"
        assert row["accuracy"]["error"] <= row["accuracy"]["tolerance"]
        assert row["peak_traced_kib"] > 0
        assert 0 <= row["runtime_ms"]["min"] <= row["runtime_ms"]["mean"]


@pytest.mark.parametrize("family", ["pf", "hank", "var", "mrio"])
def test_workload_accuracy_checks_reject_corrupted_results(family):
    workload = {"pf": lambda: scaling._pf(5, "krylov"), "hank": lambda: scaling._hank(5),
                "var": lambda: scaling._var(8, 2026), "mrio": lambda: scaling._mrio(4)}[family]()
    result = workload.solve()
    assert workload.check(result)["passed"]
    if family == "pf":
        result = replace(result, path=result.path + 0.1)
    elif family == "hank":
        result = replace(result, irf_rate=result.irf_rate + 0.1)
    elif family == "var":
        result["draws"][0, 0, 0, 0] += 0.1
    else:
        result = replace(result, Z=result.Z + sparse.eye(result.Z.shape[0]))
    assert not workload.check(result)["passed"]


def test_cli_failed_accuracy_is_nonzero_and_keeps_evidence(monkeypatch, tmp_path, capsys):
    original = scaling._pf
    def inaccurate(T, backend):
        workload = original(T, backend)
        solve = workload.solve
        def bad_path():
            result = solve()
            return replace(result, path=result.path + 0.1)
        return replace(workload, solve=bad_path)
    monkeypatch.setattr(scaling, "_pf", inaccurate)
    output = tmp_path / "failed.json"
    assert scaling.main(["--families", "pf", "--repetitions", "1", "--output", str(output)]) == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert not report["passed"]
    assert all(r["status"] == "failed" for r in report["records"])
    assert json.loads(capsys.readouterr().out) == report


def test_bootstrap_measurements_restart_the_seed():
    workload = scaling._var(8, 2026)
    first, second = workload.solve(), workload.solve()
    np.testing.assert_array_equal(first["draws"], second["draws"])


def test_invalid_repetition_count_rejected(capsys):
    with pytest.raises(SystemExit) as exc:
        scaling.main(["--repetitions", "0"])
    assert exc.value.code == 2
