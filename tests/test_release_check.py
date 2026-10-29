"""Tests for tools/release_check.py. Imports the module as a script-like module."""
import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "release_check.py"

_spec = importlib.util.spec_from_file_location("release_check", SCRIPT)
release_check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release_check)


def _run(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        cwd=str(REPO_ROOT),
    )


def test_script_exists():
    assert SCRIPT.exists(), f"{SCRIPT} must exist"


def test_help_flag():
    res = _run("--help")
    assert res.returncode == 0
    assert "release-gate" in res.stdout.lower() or "release check" in res.stdout.lower()


def test_default_run_emits_report_header():
    res = _run("--report-only")
    # --report-only short-circuits actual gate execution; still must print the header.
    assert "release_check" in res.stdout.lower() or "gate" in res.stdout.lower()


def test_read_pyproject_version(tmp_path):
    f = tmp_path / "pyproject.toml"
    f.write_text(textwrap.dedent("""
        [build-system]
        requires = ["setuptools"]
        [project]
        name = "x"
        version = "1.2.3"
    """), encoding="utf-8")
    assert release_check.read_pyproject_version(f) == "1.2.3"


def test_read_init_version(tmp_path):
    f = tmp_path / "__init__.py"
    f.write_text('__version__ = "9.8.7"\n', encoding="utf-8")
    assert release_check.read_init_version(f) == "9.8.7"


def test_read_changelog_version(tmp_path):
    f = tmp_path / "CHANGELOG.md"
    f.write_text(textwrap.dedent("""
        # Changelog

        ## 0.46.0 — 2026-05-22
        Some content.

        ## 0.45.0 — 2026-05-21
        Older.
    """), encoding="utf-8")
    assert release_check.read_changelog_version(f) == "0.46.0"


@pytest.mark.parametrize(
    "heading,expected",
    [
        ("## 5.0.0a1 (unreleased)", "5.0.0a1"),
        ("## 5.0.0a1 (2026-10-20)", "5.0.0a1"),
        ("## 4.8.1 (2026-10-11)", "4.8.1"),
    ],
    ids=["alpha-unreleased", "alpha-dated", "final"],
)
def test_read_changelog_version_accepts_a_pre_release(tmp_path, heading, expected):
    """A ``## 5.0.0a1`` heading is the version ``5.0.0a1``, not ``5.0.0``.

    Gate 4 compares the CHANGELOG heading with the four other version-bearing
    files, so a reader that drops the PEP 440 pre-release suffix reports a false
    mismatch (``5.0.0`` against ``5.0.0a1``) on every alpha, beta or rc heading.
    """
    f = tmp_path / "CHANGELOG.md"
    f.write_text(
        f"# Changelog\n\nIntro paragraph.\n\n{heading}\n\nSome content.\n\n"
        "## 4.8.0 (2026-10-10)\nOlder.\n",
        encoding="utf-8",
    )
    assert release_check.read_changelog_version(f) == expected


def test_the_other_version_readers_accept_a_pre_release(tmp_path):
    """The four other readers already keep the suffix; pin that so Gate 4 can compare 5.0.0a1 everywhere."""
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "5.0.0a1"\n', encoding="utf-8")
    (tmp_path / "__init__.py").write_text('__version__ = "5.0.0a1"\n', encoding="utf-8")
    (tmp_path / "CITATION.cff").write_text("cff-version: 1.2.0\nversion: 5.0.0a1\n", encoding="utf-8")
    (tmp_path / "jupyter_lite_config.json").write_text(
        json.dumps({"PipliteAddon": {"piplite_urls": ["./wheels/puremacro-5.0.0a1-py3-none-any.whl"]}}),
        encoding="utf-8",
    )
    assert release_check.read_pyproject_version(tmp_path / "pyproject.toml") == "5.0.0a1"
    assert release_check.read_init_version(tmp_path / "__init__.py") == "5.0.0a1"
    assert release_check.read_citation_version(tmp_path / "CITATION.cff") == "5.0.0a1"
    assert release_check.read_playground_pin_version(tmp_path / "jupyter_lite_config.json") == "5.0.0a1"


def test_gate4_version_sync_pass():
    r = release_check.gate_version_sync(
        pyproject_version="0.46.0",
        init_version="0.46.0",
        changelog_version="0.46.0",
        citation_version="0.46.0",
    )
    assert r["name"] == "version_sync"
    assert r["passed"] is True


def test_gate4_version_sync_fail():
    r = release_check.gate_version_sync(
        pyproject_version="0.12.1",
        init_version="0.46.0",
        changelog_version="0.46.0",
        citation_version="0.46.0",
    )
    assert r["name"] == "version_sync"
    assert r["passed"] is False
    assert "0.12.1" in r["report"]
    assert "0.46.0" in r["report"]


def test_gate4_catches_a_stale_citation_file():
    """The 1.3.0 -> 1.3.1 drift: three files bumped, CITATION.cff forgotten."""
    r = release_check.gate_version_sync(
        pyproject_version="1.3.1",
        init_version="1.3.1",
        changelog_version="1.3.1",
        citation_version="1.3.0",
    )
    assert r["passed"] is False
    assert "CITATION.cff" in r["report"]
    assert "1.3.0" in r["report"]


def test_read_citation_version_parses_the_real_file(tmp_path):
    cff = tmp_path / "CITATION.cff"
    cff.write_text(
        'cff-version: 1.2.0\ntitle: "x"\nversion: 2.5.1\ndate-released: 2026-01-01\n',
        encoding="utf-8",
    )
    assert release_check.read_citation_version(cff) == "2.5.1"

    # quoted form, as some cff files write it
    cff.write_text('version: "2.5.1"\n', encoding="utf-8")
    assert release_check.read_citation_version(cff) == "2.5.1"

    cff.write_text("title: no version here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="version not found"):
        release_check.read_citation_version(cff)


def test_the_repo_s_own_citation_file_is_in_sync():
    assert (release_check.read_citation_version(REPO_ROOT / "CITATION.cff")
            == release_check.read_pyproject_version(REPO_ROOT / "pyproject.toml"))


def test_gate1_compare_exact_match():
    whitelist = {"a::test_x", "b::test_y"}
    failing = {"a::test_x", "b::test_y"}
    r = release_check.compare_failures(failing, whitelist)
    assert r["passed"] is True
    assert r["new"] == set()
    assert r["recovered"] == set()


def test_gate1_compare_new_failure():
    whitelist = {"a::test_x"}
    failing = {"a::test_x", "c::test_z"}
    r = release_check.compare_failures(failing, whitelist)
    assert r["passed"] is False
    assert r["new"] == {"c::test_z"}
    assert r["recovered"] == set()


def test_gate1_compare_recovered_test():
    whitelist = {"a::test_x", "b::test_y"}
    failing = {"a::test_x"}
    r = release_check.compare_failures(failing, whitelist)
    # Recovered is a warning, not a failure — gate still passes.
    assert r["passed"] is True
    assert r["new"] == set()
    assert r["recovered"] == {"b::test_y"}


def test_gate1_load_whitelist(tmp_path):
    f = tmp_path / "kf.json"
    f.write_text('{"schema_version": 1, "entries": [{"nodeid": "a::test_x", "reason": "x", "since_version": "0.1.0", "owner_note": "x"}]}', encoding="utf-8")
    assert release_check.load_whitelist(f) == {"a::test_x"}


def test_gate2_pyodide_passthrough_pass(monkeypatch):
    # Smoke: gate_pyodide returns a dict with 'passed' / 'report'.
    r = release_check.gate_pyodide(REPO_ROOT)
    assert "passed" in r
    assert "report" in r
    assert r["name"] == "pyodide"


def test_gate1_pytest_collect_error_path(monkeypatch, tmp_path):
    """When pytest exits with a non-0/1 code, gate_test_baseline must fail (not silently pass)."""
    def fake_run(*args, **kwargs):
        class P:
            returncode = 4  # pytest usage error
            stdout = ""
            stderr = "ERROR: pytest config invalid"
        return P()
    monkeypatch.setattr(release_check.subprocess, "run", fake_run)
    # Stub the whitelist loader so gate_test_baseline doesn't need a real file.
    monkeypatch.setattr(release_check, "load_whitelist", lambda _p: set())
    r = release_check.gate_test_baseline(tmp_path)
    assert r["passed"] is False
    assert "could not run" in r["report"] or "exited with code" in r["report"]


@pytest.mark.parametrize("output", ["1 failed in 0.01s\n", "", "\x1b[31mFAILED tests/test_example.py::test_x\x1b[0m\n"])
def test_gate1_failure_without_readable_nodeids_cannot_pass(monkeypatch, tmp_path, output):
    """Suppressed/truncated/colored summaries must not turn pytest exit 1 green."""
    def fake_run(cmd, **kwargs):
        assert "--color=no" in cmd
        return subprocess.CompletedProcess(cmd, 1, stdout=output, stderr="")

    monkeypatch.setattr(release_check.subprocess, "run", fake_run)
    monkeypatch.setattr(release_check, "load_whitelist", lambda _p: set())
    result = release_check.gate_test_baseline(tmp_path)
    assert result["passed"] is False
    assert "no FAILED/ERROR node ids" in result["report"]


def test_gate1_parsed_failures_still_use_the_declared_baseline(monkeypatch, tmp_path):
    output = "FAILED tests/test_example.py::test_x - AssertionError\n1 failed in 0.01s\n"
    monkeypatch.setattr(release_check.subprocess, "run", lambda cmd, **kwargs:
                        subprocess.CompletedProcess(cmd, 1, stdout=output, stderr=""))
    monkeypatch.setattr(release_check, "load_whitelist", lambda _p: {"tests/test_example.py::test_x"})
    result = release_check.gate_test_baseline(tmp_path)
    assert result["passed"] is True
    assert result["new"] == set()


def test_gate3_snapshot_equal(tmp_path):
    snap = {"all": {"puremacro._linalg": ["inv_xtx", "safe_cholesky"]}}
    f = tmp_path / "snap.json"
    f.write_text(json.dumps(snap), encoding="utf-8")
    r = release_check.compare_snapshot(snap, f)
    assert r["passed"] is True
    assert r["name"] == "public_api_snapshot"


def test_gate3_snapshot_diff(tmp_path):
    on_disk = {"all": {"puremacro._linalg": ["inv_xtx"]}}
    fresh = {"all": {"puremacro._linalg": ["inv_xtx", "safe_cholesky"]}}
    f = tmp_path / "snap.json"
    f.write_text(json.dumps(on_disk), encoding="utf-8")
    r = release_check.compare_snapshot(fresh, f)
    assert r["passed"] is False
    assert "safe_cholesky" in r["report"]
    assert r["name"] == "public_api_snapshot"


def test_gate3_snapshot_diff_result_classes(tmp_path):
    """A result_classes-only change must produce a FAIL with a non-empty diff body."""
    on_disk = {
        "all": {"puremacro.foo": ["bar"]},
        "result_classes": {"puremacro.foo.BarResult": ["x"]},
    }
    fresh = {
        "all": {"puremacro.foo": ["bar"]},
        "result_classes": {"puremacro.foo.BarResult": ["x", "y"]},  # field added
    }
    f = tmp_path / "snap.json"
    f.write_text(json.dumps(on_disk), encoding="utf-8")
    r = release_check.compare_snapshot(fresh, f)
    assert r["passed"] is False
    assert r["name"] == "public_api_snapshot"
    assert "BarResult.y" in r["report"]



def _summary_count_matches_reported_gates(out: str) -> bool:
    """The "all N gates PASS" line must agree with the number of gates printed.

    Asserting a literal N breaks every time a gate is added (Gate 7 did exactly
    that), so compare the summary against the gate lines the run emitted.
    """
    import re as _re

    summary = _re.search(r"all (\d+) gates pass", out.lower())
    assert summary is not None, f"no summary line in:\n{out}"
    reported = len(_re.findall(r"^\s*gate \d+[^\n]*:\s*pass", out.lower(), _re.M))
    return int(summary.group(1)) == reported

def test_main_emits_summary_pass(capsys, monkeypatch):
    # Force all gates to be no-ops returning pass.
    monkeypatch.setattr(release_check, "gate_test_baseline", lambda _r: {"name": "test_baseline", "passed": True, "report": "  Gate 1: PASS"})
    monkeypatch.setattr(release_check, "gate_pyodide", lambda _r: {"name": "pyodide", "passed": True, "report": "  Gate 2: PASS"})
    monkeypatch.setattr(release_check, "gate_snapshot", lambda _r: {"name": "public_api_snapshot", "passed": True, "report": "  Gate 3: PASS"})
    monkeypatch.setattr(release_check, "gate_version_sync", lambda **kw: {"name": "version_sync", "passed": True, "report": "  Gate 4: PASS"})
    monkeypatch.setattr(release_check, "gate_min_python_syntax", lambda *a, **kw: {"name": "min_python_syntax", "passed": True, "report": "  Gate 7: PASS"})
    monkeypatch.setattr(release_check, "gate_floor_mypy", lambda *a, **kw: {"name": "floor_mypy", "passed": True, "report": "  Gate 8: PASS"})
    rc = release_check.main([])
    captured = capsys.readouterr()
    assert rc == 0
    assert _summary_count_matches_reported_gates(captured.out)


def test_main_emits_summary_fail(capsys, monkeypatch):
    monkeypatch.setattr(release_check, "gate_test_baseline", lambda _r: {"name": "test_baseline", "passed": True, "report": "  Gate 1: PASS"})
    monkeypatch.setattr(release_check, "gate_pyodide", lambda _r: {"name": "pyodide", "passed": True, "report": "  Gate 2: PASS"})
    monkeypatch.setattr(release_check, "gate_snapshot", lambda _r: {"name": "public_api_snapshot", "passed": True, "report": "  Gate 3: PASS"})
    monkeypatch.setattr(release_check, "gate_version_sync", lambda **kw: {"name": "version_sync", "passed": False, "report": "  Gate 4: FAIL"})
    monkeypatch.setattr(release_check, "gate_floor_mypy", lambda *a, **kw: {"name": "floor_mypy", "passed": True, "report": "  Gate 8: PASS"})
    rc = release_check.main([])
    captured = capsys.readouterr()
    assert rc == 1
    assert "fail" in captured.out.lower()
    assert "version_sync" in captured.out.lower()


def test_gate5_all_pass_or_skip(tmp_path):
    data = {
        "schema_version": 1,
        "generated_at": "2026-05-22T14:30:00Z",
        "examples": {
            "a": {"status": "PASS", "reason": None, "runtime_s": 1.0, "figures": [], "last_run": "2026-05-22T14:30:00Z"},
            "b": {"status": "SKIP", "reason": "network unavailable", "runtime_s": 0.5, "figures": [], "last_run": "2026-05-22T14:30:00Z"},
        },
    }
    f = tmp_path / "gallery.json"
    f.write_text(json.dumps(data), encoding="utf-8")
    r = release_check.gate_examples_gallery(f, examples_source_dir=tmp_path / "no_examples")
    assert r["passed"] is True
    assert r["name"] == "examples_gallery"


def test_gate5_one_fail(tmp_path):
    data = {
        "schema_version": 1,
        "generated_at": "2026-05-22T14:30:00Z",
        "examples": {
            "a": {"status": "PASS", "reason": None, "runtime_s": 1.0, "figures": [], "last_run": "2026-05-22T14:30:00Z"},
            "b": {"status": "FAIL", "reason": "ValueError", "runtime_s": 0.1, "figures": [], "last_run": "2026-05-22T14:30:00Z"},
        },
    }
    f = tmp_path / "gallery.json"
    f.write_text(json.dumps(data), encoding="utf-8")
    r = release_check.gate_examples_gallery(f, examples_source_dir=tmp_path / "no_examples")
    assert r["passed"] is False
    assert "b" in r["report"]


def test_gate5_missing_json(tmp_path):
    f = tmp_path / "gallery.json"  # does not exist
    r = release_check.gate_examples_gallery(f, examples_source_dir=tmp_path / "no_examples")
    assert r["passed"] is False
    assert "not rendered" in r["report"].lower() or "missing" in r["report"].lower()


def test_gate5_stale_json_warns(tmp_path):
    import time
    examples_dir = tmp_path / "examples"
    examples_dir.mkdir()
    (examples_dir / "a.py").write_text("# new\n", encoding="utf-8")
    time.sleep(0.05)
    # JSON's generated_at is OLDER than the example file mtime.
    old_iso = "2020-01-01T00:00:00Z"
    data = {
        "schema_version": 1,
        "generated_at": old_iso,
        "examples": {
            "a": {"status": "PASS", "reason": None, "runtime_s": 1.0, "figures": [], "last_run": old_iso},
        },
    }
    f = tmp_path / "gallery.json"
    f.write_text(json.dumps(data), encoding="utf-8")
    r = release_check.gate_examples_gallery(f, examples_source_dir=examples_dir)
    # Stale warns; does NOT fail.
    assert r["passed"] is True
    assert "stale" in r["report"].lower() or "consider re-rendering" in r["report"].lower()


# --- Gate 5 known-example-failures allowlist (tests/known_example_failures.json) ---

_RR_EXAMPLE = "romer_romer_2010_replication"
_RR_REASON = (
    "RR2010 baseline: FAIL; h10=-3.08081276, t=-3.52499608; "
    "output=output/romer_romer_2010_replication"
)
_RR_ENTRY = {
    "example": _RR_EXAMPLE,
    "reason_prefix": "RR2010 baseline: FAIL; h10=-3.08081276, t=-3.52499608",
    "reason": "documented published-rounding miss (test fixture)",
    "documented_in": ["docs/research_benchmarks.md"],
    "since": "5.4.0",
}


def _gallery_entry(status, reason=None):
    return {"status": status, "reason": reason, "runtime_s": 1.0, "figures": [],
            "last_run": "2026-05-22T14:30:00Z"}


def _gallery_json(tmp_path, examples):
    """Write a docs/examples_gallery.json-shaped file and return its path."""
    f = tmp_path / "gallery.json"
    f.write_text(json.dumps({"schema_version": 1, "generated_at": "2026-05-22T14:30:00Z",
                             "examples": examples}), encoding="utf-8")
    return f


def _known_example_failures(tmp_path, entries):
    """Write a tests/known_example_failures.json-shaped allowlist and return its path."""
    f = tmp_path / "known_example_failures.json"
    f.write_text(json.dumps({"schema_version": 1, "entries": entries}), encoding="utf-8")
    return f


def test_gate5_known_failure_with_matching_prefix_passes(tmp_path):
    """A FAIL whose recorded reason starts with the allowlisted prefix is documented, not fatal."""
    gallery = _gallery_json(tmp_path, {"a": _gallery_entry("PASS"),
                                       _RR_EXAMPLE: _gallery_entry("FAIL", _RR_REASON)})
    allow = _known_example_failures(tmp_path, [_RR_ENTRY])
    r = release_check.gate_examples_gallery(
        gallery, examples_source_dir=tmp_path / "no_examples", known_failures_path=allow)
    assert r["passed"] is True, r["report"]
    assert r["name"] == "examples_gallery"
    assert "1 known FAIL (documented)" in r["report"]
    assert "0 new FAIL" in r["report"]
    assert f"known FAIL (documented) {_RR_EXAMPLE}" in r["report"]
    assert "recovered" not in r["report"]


@pytest.mark.parametrize("reason", [
    "RR2010 baseline: FAIL; h10=-3.09000000, t=-3.52499608; output=output/romer_romer_2010_replication",
    "RR2010 baseline: FAIL; h10=-3.08081276, t=-3.53000000; output=output/romer_romer_2010_replication",
    "Traceback (most recent call last): ... ValueError: boom",
    None,
], ids=["h10-drift", "t-drift", "traceback", "no-reason"])
def test_gate5_known_failure_reason_drift_fails(tmp_path, reason):
    """A listed example whose reason no longer starts with the pinned prefix is a hard FAIL again."""
    gallery = _gallery_json(tmp_path, {_RR_EXAMPLE: _gallery_entry("FAIL", reason)})
    allow = _known_example_failures(tmp_path, [_RR_ENTRY])
    r = release_check.gate_examples_gallery(
        gallery, examples_source_dir=tmp_path / "no_examples", known_failures_path=allow)
    assert r["passed"] is False
    assert "1 example(s) failed" in r["report"]
    assert f"    FAIL {_RR_EXAMPLE} — " in r["report"]
    assert "known FAIL" not in r["report"]


def test_gate5_unlisted_fail_still_fails_with_allowlist_present(tmp_path):
    """The allowlist covers only the examples it names; any other FAIL still fails the gate."""
    gallery = _gallery_json(tmp_path, {_RR_EXAMPLE: _gallery_entry("FAIL", _RR_REASON),
                                       "b": _gallery_entry("FAIL", "ValueError")})
    allow = _known_example_failures(tmp_path, [_RR_ENTRY])
    r = release_check.gate_examples_gallery(
        gallery, examples_source_dir=tmp_path / "no_examples", known_failures_path=allow)
    assert r["passed"] is False
    assert "1 example(s) failed" in r["report"]
    assert "    FAIL b — ValueError" in r["report"]
    # The documented one is still shown, but as known rather than as a new failure.
    assert f"known FAIL (documented) {_RR_EXAMPLE}" in r["report"]


def test_gate5_recovered_known_failure_warns(tmp_path):
    """A listed example that now passes is a warning to shrink the allowlist, not a failure."""
    gallery = _gallery_json(tmp_path, {_RR_EXAMPLE: _gallery_entry("PASS")})
    allow = _known_example_failures(tmp_path, [_RR_ENTRY])
    r = release_check.gate_examples_gallery(
        gallery, examples_source_dir=tmp_path / "no_examples", known_failures_path=allow)
    assert r["passed"] is True
    assert "1 PASS" in r["report"]
    assert f"recovered: {_RR_EXAMPLE}" in r["report"]
    assert "known FAIL" not in r["report"]


def test_gate5_empty_prefix_never_matches(tmp_path):
    """An empty reason_prefix would allowlist any reason, so the gate refuses to match it."""
    gallery = _gallery_json(tmp_path, {_RR_EXAMPLE: _gallery_entry("FAIL", _RR_REASON)})
    allow = _known_example_failures(tmp_path, [dict(_RR_ENTRY, reason_prefix="")])
    r = release_check.gate_examples_gallery(
        gallery, examples_source_dir=tmp_path / "no_examples", known_failures_path=allow)
    assert r["passed"] is False
    assert "known FAIL" not in r["report"]


def test_gate5_missing_allowlist_file_behaves_like_none(tmp_path):
    gallery = _gallery_json(tmp_path, {_RR_EXAMPLE: _gallery_entry("FAIL", _RR_REASON)})
    r = release_check.gate_examples_gallery(
        gallery, examples_source_dir=tmp_path / "no_examples",
        known_failures_path=tmp_path / "does_not_exist.json")
    assert r["passed"] is False
    assert f"    FAIL {_RR_EXAMPLE} — " in r["report"]


def test_gate5_live_gallery_passes_with_repo_allowlist():
    """The committed gallery carries the documented RR2010 FAIL; with the repo allowlist Gate 5 passes.

    This is the regression for "Gate 5 cannot pass": the example exits nonzero
    on purpose (published t-statistic rounding miss, see
    docs/research_benchmarks.md), the gallery honestly records it as FAIL, and
    tests/known_example_failures.json is what lets the release gate distinguish
    that from a broken example.
    """
    r = release_check.gate_examples_gallery(
        REPO_ROOT / "docs" / "examples_gallery.json",
        examples_source_dir=REPO_ROOT / "puremacro" / "examples",
        known_failures_path=REPO_ROOT / "tests" / "known_example_failures.json",
    )
    assert r["passed"] is True, r["report"]
    assert r["report"].startswith("  Gate 5 (examples gallery): PASS")


def test_known_example_failures_entries_are_well_formed():
    """tests/known_example_failures.json must describe the committed gallery, entry by entry."""
    path = REPO_ROOT / "tests" / "known_example_failures.json"
    assert path.exists(), f"{path} must exist (tools/release_check.py reads it)"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data.get("schema_version") == 1
    assert isinstance(data.get("entries"), list)
    gallery = json.loads(
        (REPO_ROOT / "docs" / "examples_gallery.json").read_text(encoding="utf-8"))["examples"]
    required = {"example", "reason_prefix", "reason", "documented_in", "since"}
    names = [e["example"] for e in data["entries"]]
    assert len(names) == len(set(names)), "duplicate example in known_example_failures.json"
    for i, entry in enumerate(data["entries"]):
        missing = required - set(entry)
        assert not missing, f"entry {i} missing fields: {missing}"
        for key in ("example", "reason_prefix", "reason", "since"):
            assert isinstance(entry[key], str) and entry[key], f"entry {i}: {key} must be a non-empty string"
        assert isinstance(entry["documented_in"], list) and entry["documented_in"]
        for ref in entry["documented_in"]:
            assert (REPO_ROOT / ref.split(" ")[0]).exists(), f"entry {i}: {ref!r} names a missing file"
        assert (REPO_ROOT / "puremacro" / "examples" / f"{entry['example']}.py").exists()
        live = gallery.get(entry["example"])
        assert live is not None, f"{entry['example']} is not in docs/examples_gallery.json"
        assert live["status"] == "FAIL", (
            f"{entry['example']} is {live['status']} in the gallery; remove its allowlist entry")
        assert (live.get("reason") or "").startswith(entry["reason_prefix"]), (
            f"{entry['example']}: gallery reason {live.get('reason')!r} drifted from "
            f"prefix {entry['reason_prefix']!r}")


def test_main_summary_5_gates_with_examples_flag(capsys, monkeypatch):
    """With --examples, the summary count matches the gates actually run."""
    monkeypatch.setattr(release_check, "gate_test_baseline", lambda _r: {"name": "test_baseline", "passed": True, "report": "  Gate 1: PASS"})
    monkeypatch.setattr(release_check, "gate_pyodide", lambda _r: {"name": "pyodide", "passed": True, "report": "  Gate 2: PASS"})
    monkeypatch.setattr(release_check, "gate_snapshot", lambda _r: {"name": "public_api_snapshot", "passed": True, "report": "  Gate 3: PASS"})
    monkeypatch.setattr(release_check, "gate_version_sync", lambda **kw: {"name": "version_sync", "passed": True, "report": "  Gate 4: PASS"})
    monkeypatch.setattr(release_check, "gate_min_python_syntax", lambda *a, **kw: {"name": "min_python_syntax", "passed": True, "report": "  Gate 7: PASS"})
    monkeypatch.setattr(
        release_check, "gate_examples_gallery",
        lambda _p, *, examples_source_dir, known_failures_path=None: {"name": "examples_gallery", "passed": True, "report": "  Gate 5: PASS"},
    )
    monkeypatch.setattr(release_check, "gate_floor_mypy", lambda *a, **kw: {"name": "floor_mypy", "passed": True, "report": "  Gate 8: PASS"})
    rc = release_check.main(["--examples"])
    captured = capsys.readouterr()
    assert rc == 0
    assert _summary_count_matches_reported_gates(captured.out)


def test_gate6_pass(monkeypatch, tmp_path):
    """gate_pyodide_smoke wraps pyodide_smoke.run; PASS path."""
    import sys
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    import pyodide_smoke
    monkeypatch.setattr(
        pyodide_smoke, "run",
        lambda _r: {"passed": True, "report": "  Gate 6 (pyodide smoke): PASS — 8 passed"},
    )
    r = release_check.gate_pyodide_smoke(REPO_ROOT)
    assert r["passed"] is True
    assert r["name"] == "pyodide_smoke"


def test_gate6_fail(monkeypatch):
    """gate_pyodide_smoke surfaces wrapper FAIL."""
    import sys
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    import pyodide_smoke
    monkeypatch.setattr(
        pyodide_smoke, "run",
        lambda _r: {"passed": False, "report": "  Gate 6 (pyodide smoke): FAIL — 1 failed"},
    )
    r = release_check.gate_pyodide_smoke(REPO_ROOT)
    assert r["passed"] is False
    assert "fail" in r["report"].lower()
    assert r["name"] == "pyodide_smoke"


def test_main_summary_with_pyodide_flag(capsys, monkeypatch):
    """With --examples --pyodide, the summary count matches the gates run."""
    monkeypatch.setattr(release_check, "gate_test_baseline", lambda _r: {"name": "test_baseline", "passed": True, "report": "  Gate 1: PASS"})
    monkeypatch.setattr(release_check, "gate_pyodide", lambda _r: {"name": "pyodide", "passed": True, "report": "  Gate 2: PASS"})
    monkeypatch.setattr(release_check, "gate_snapshot", lambda _r: {"name": "public_api_snapshot", "passed": True, "report": "  Gate 3: PASS"})
    monkeypatch.setattr(release_check, "gate_version_sync", lambda **kw: {"name": "version_sync", "passed": True, "report": "  Gate 4: PASS"})
    monkeypatch.setattr(release_check, "gate_min_python_syntax", lambda *a, **kw: {"name": "min_python_syntax", "passed": True, "report": "  Gate 7: PASS"})
    monkeypatch.setattr(
        release_check, "gate_examples_gallery",
        lambda _p, *, examples_source_dir, known_failures_path=None: {"name": "examples_gallery", "passed": True, "report": "  Gate 5: PASS"},
    )
    monkeypatch.setattr(
        release_check, "gate_pyodide_smoke",
        lambda _r: {"name": "pyodide_smoke", "passed": True, "report": "  Gate 6: PASS"},
    )
    monkeypatch.setattr(release_check, "gate_floor_mypy", lambda *a, **kw: {"name": "floor_mypy", "passed": True, "report": "  Gate 8: PASS"})
    rc = release_check.main(["--examples", "--pyodide"])
    captured = capsys.readouterr()
    assert rc == 0
    assert _summary_count_matches_reported_gates(captured.out)


def test_check_external_oracle_fixtures(tmp_path, monkeypatch):
    """External parity oracle warning helper detects missing and present fixtures."""
    monkeypatch.delenv("PUREMACRO_ABDATA", raising=False)
    monkeypatch.delenv("PUREMACRO_ADH_RDA", raising=False)
    warns = release_check.check_external_oracle_fixtures(tmp_path)
    assert len(warns) == 2
    assert any("abdata.dta" in w for w in warns)
    assert any("ADH.rda" in w for w in warns)

    ab = tmp_path / "abdata.dta"
    ab.write_bytes(b"stata")
    adh = tmp_path / "ADH.rda"
    adh.write_bytes(b"rda")
    monkeypatch.setenv("PUREMACRO_ABDATA", str(ab))
    monkeypatch.setenv("PUREMACRO_ADH_RDA", str(adh))
    assert release_check.check_external_oracle_fixtures(tmp_path) == []


# ---------------------------------------------------------------------------
# Gate 8 — mypy under the requires-python floor interpreter
# ---------------------------------------------------------------------------

def _fake_floor_run(mypy_by_platform, *, version_line="3 11", mypy_installed=True):
    """A subprocess.run stand-in for Gate 8: version probe, mypy probe, mypy runs."""
    def fake_run(cmd, **kwargs):
        cmd = [str(c) for c in cmd]
        joined = " ".join(cmd)
        if "sys.version_info" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout=version_line + "\n", stderr="")
        if "import mypy.version" in joined:
            if mypy_installed:
                return subprocess.CompletedProcess(cmd, 0, stdout="1.19.1 2.4.6\n", stderr="")
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="ModuleNotFoundError: No module named 'mypy'")
        if "-m" in cmd and "mypy" in cmd:
            platform = cmd[cmd.index("--platform") + 1] if "--platform" in cmd else "native"
            rc, out = mypy_by_platform[platform]
            return subprocess.CompletedProcess(cmd, rc, stdout=out, stderr="")
        raise AssertionError(f"unexpected command: {cmd}")
    return fake_run


_CLEAN = (0, "Success: no issues found in 890 source files\n")


def test_gate8_passes_when_the_floor_interpreter_type_checks_natively_and_as_win32(monkeypatch, tmp_path):
    monkeypatch.setattr(release_check.subprocess, "run",
                        _fake_floor_run({"native": _CLEAN, "win32": _CLEAN}))
    g = release_check.gate_floor_mypy(tmp_path, min_version=(3, 11), interpreter="/fake/.venv311/bin/python")
    assert g["name"] == "floor_mypy"
    assert g["passed"] is True
    assert "Gate 8" in g["report"] and "3.11" in g["report"]
    assert "win32" in g["report"] and "mypy 1.19.1" in g["report"] and "numpy 2.4.6" in g["report"]


def test_gate8_reports_the_file_and_line_of_a_floor_only_typing_error(monkeypatch, tmp_path):
    error = ("puremacro/lp/distributional.py:957: error: No overload variant of \"__call__\" "
             "of \"_UFunc_Nin1_Nout1\" matches argument type [call-overload]\n"
             "Found 1 error in 1 file (checked 889 source files)\n")
    monkeypatch.setattr(release_check.subprocess, "run",
                        _fake_floor_run({"native": (1, error), "win32": _CLEAN}))
    g = release_check.gate_floor_mypy(tmp_path, min_version=(3, 11), interpreter="/fake/py311")
    assert g["passed"] is False
    assert "FAIL" in g["report"]
    assert "[native]" in g["report"] and "distributional.py:957" in g["report"]
    assert "Found 1 error" in g["report"]


def test_gate8_fails_when_mypy_is_not_installed_in_the_floor_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(release_check.subprocess, "run",
                        _fake_floor_run({"native": _CLEAN, "win32": _CLEAN}, mypy_installed=False))
    g = release_check.gate_floor_mypy(tmp_path, min_version=(3, 11), interpreter="/fake/py311")
    assert g["passed"] is False
    assert "install" in g["report"].lower() and "/fake/py311" in g["report"]


def test_gate8_fails_with_setup_instructions_when_no_floor_interpreter_exists(monkeypatch, tmp_path):
    monkeypatch.setattr(release_check, "find_floor_interpreter",
                        lambda repo_root, min_version: (None, ["running interpreter: /usr/bin/python3 is Python 3.13"]))
    g = release_check.gate_floor_mypy(tmp_path, min_version=(3, 11))
    assert g["passed"] is False
    assert "no Python 3.11 interpreter" in g["report"]
    assert "uv python install 3.11" in g["report"] and ".venv311" in g["report"]
    assert "PUREMACRO_FLOOR_PYTHON" in g["report"]
    assert "/usr/bin/python3 is Python 3.13" in g["report"]


def test_find_floor_interpreter_takes_the_first_candidate_of_the_floor_version(monkeypatch, tmp_path):
    venv_python = tmp_path / ".venv311" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("")
    versions = {"/fake/py312": (3, 12), sys.executable: (3, 13), str(venv_python): (3, 11)}
    monkeypatch.setenv("PUREMACRO_FLOOR_PYTHON", "/fake/py312")
    monkeypatch.setattr(release_check, "_probe_interpreter_version", lambda exe: versions.get(str(exe)))
    monkeypatch.setattr(release_check.shutil, "which", lambda name: None)
    found, notes = release_check.find_floor_interpreter(tmp_path, (3, 11))
    assert found == str(venv_python)
    # The rejected candidates are named with the version they actually report.
    assert any("PUREMACRO_FLOOR_PYTHON" in n and "3.12" in n for n in notes)
    assert any("running interpreter" in n and "3.13" in n for n in notes)


def test_find_floor_interpreter_accepts_the_running_interpreter_when_it_is_the_floor(monkeypatch, tmp_path):
    monkeypatch.delenv("PUREMACRO_FLOOR_PYTHON", raising=False)
    monkeypatch.setattr(release_check, "_probe_interpreter_version",
                        lambda exe: (3, 11) if str(exe) == sys.executable else None)
    monkeypatch.setattr(release_check.shutil, "which", lambda name: None)
    found, notes = release_check.find_floor_interpreter(tmp_path, (3, 11))
    assert found == sys.executable
    assert notes == []


def test_find_floor_interpreter_reports_every_rejected_candidate(monkeypatch, tmp_path):
    monkeypatch.delenv("PUREMACRO_FLOOR_PYTHON", raising=False)
    monkeypatch.setattr(release_check, "_probe_interpreter_version", lambda exe: None)
    monkeypatch.setattr(release_check.shutil, "which", lambda name: "/opt/bin/python3.11")
    found, notes = release_check.find_floor_interpreter(tmp_path, (3, 11))
    assert found is None
    assert any("unavailable" in n and "/opt/bin/python3.11" in n for n in notes)


def test_main_runs_gate8_in_the_full_gate_and_only_on_request_with_no_tests(capsys, monkeypatch):
    for name, fn in {
        "gate_test_baseline": lambda _r: {"name": "test_baseline", "passed": True, "report": "  Gate 1: PASS"},
        "gate_pyodide": lambda _r: {"name": "pyodide", "passed": True, "report": "  Gate 2: PASS"},
        "gate_snapshot": lambda _r: {"name": "public_api_snapshot", "passed": True, "report": "  Gate 3: PASS"},
        "gate_version_sync": lambda **kw: {"name": "version_sync", "passed": True, "report": "  Gate 4: PASS"},
        "gate_min_python_syntax": lambda *a, **kw: {"name": "min_python_syntax", "passed": True, "report": "  Gate 7: PASS"},
    }.items():
        monkeypatch.setattr(release_check, name, fn)
    calls = []

    def gate8(_r):
        calls.append(_r)
        return {"name": "floor_mypy", "passed": True, "report": "  Gate 8: PASS"}

    monkeypatch.setattr(release_check, "gate_floor_mypy", gate8)

    assert release_check.main(["--no-tests"]) == 0
    assert calls == [], "--no-tests alone must keep the fast path fast; CI relies on it"
    assert _summary_count_matches_reported_gates(capsys.readouterr().out)

    assert release_check.main(["--no-tests", "--mypy-floor"]) == 0
    assert len(calls) == 1
    assert _summary_count_matches_reported_gates(capsys.readouterr().out)

    assert release_check.main([]) == 0
    assert len(calls) == 2, "the full pre-tag gate must run Gate 8 without being asked"
    assert _summary_count_matches_reported_gates(capsys.readouterr().out)
