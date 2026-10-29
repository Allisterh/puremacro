"""The frozen converged sovereign-risk brief sweep and the pages generated from it.

The record is produced once by the maintainer
(``python -m puremacro.examples.first_brief_sovereign --freeze
tests/fixtures/sovereign_brief/sweep.json``, 6,851 s or 1.9 h measured under
load on a shared workstation, including the error grid); these tests
read it and never re-run the sweep.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import t as student_t

from puremacro.examples import first_brief_sovereign as brief
from puremacro.sovereign.measurement import MOMENTS

ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = ROOT / "tests" / "fixtures" / "sovereign_brief" / "sweep.json"


@pytest.fixture(scope="module")
def record():
    return json.loads(RECORD_PATH.read_text())


def _cases(record):
    for name, scenario in record["scenarios"].items():
        for key, case in scenario["cases"].items():
            yield name, key, case


def test_record_design(record):
    assert list(record["scenarios"]) == list(brief.SCENARIOS)
    assert record["grids"] == [[6401, 641], [3201, 641]]
    assert record["design"] == brief.DESIGN
    assert record["tolerances"] == brief.TOLERANCES
    assert record["baseline"] == brief.BASELINE
    for name, scenario in record["scenarios"].items():
        assert scenario["overrides"] == brief.SCENARIOS[name]
        assert list(scenario["cases"]) == ["6401x641", "3201x641"]
    assert sum(1 for _ in _cases(record)) == 14


def test_record_solver_diagnostics(record):
    for name, key, case in _cases(record):
        solver = case["solver"]
        assert solver["converged"], (name, key)
        assert solver["monotone_certification_gap"] <= 1e-11, (name, key)
        assert solver["residual"] < 1e-10, (name, key)
        assert solver["exhaustive_bellman_residual"] < 2e-11, (name, key)


def test_record_recomputes_from_per_seed(record):
    for name, key, case in _cases(record):
        sim = case["simulation"]
        assert sim["seeds"] == brief.DESIGN["seeds"]
        n = len(sim["seeds"])
        t_crit = student_t.ppf(.975, n - 1)
        for m in MOMENTS:
            values = np.asarray(sim["per_seed"][m])
            se = values.std(ddof=1) / np.sqrt(n)
            summary = sim["summary"][m]
            assert summary["mean"] == pytest.approx(values.mean(), abs=1e-12), (name, key, m)
            assert summary["se"] == pytest.approx(se, abs=1e-12), (name, key, m)
            assert summary["ci95"] == pytest.approx([values.mean() - t_crit * se, values.mean() + t_crit * se],
                                                    abs=1e-12), (name, key, m)
    for name, scenario in record["scenarios"].items():
        expected = brief.paired_change(scenario["cases"]["3201x641"], scenario["cases"]["6401x641"])
        assert scenario["grid_change"]["passed"] == expected["passed"], name
        assert scenario["grid_change"]["welfare_change_pct"] == pytest.approx(expected["welfare_change_pct"], abs=1e-12)
        for m in MOMENTS:
            got, want = scenario["grid_change"]["moments"][m], expected["moments"][m]
            assert got["status"] == want["status"], (name, m)
            for field in ("change", "absolute_change_bound", "tolerance"):
                assert got[field] == pytest.approx(want[field], abs=1e-12), (name, m, field)
            assert got["ci95"] == pytest.approx(want["ci95"], abs=1e-12)


def test_record_has_no_local_paths():
    text = RECORD_PATH.read_text()
    for forbidden in ("/Users/", "/home/", "/private/", "C:\\\\", "\\\\Users\\\\"):
        assert forbidden not in text, forbidden


def test_baseline_is_near_the_study_record(record):
    # A sanity link to the reference study at the same grid, not equality: the
    # study measures 74-quarter pre-default windows, this record raw ergodic
    # moments with a linear default annualization.
    summary = record["scenarios"]["baseline"]["cases"]["6401x641"]["simulation"]["summary"]
    assert abs(summary["debt_output_pct"]["mean"] - 4.003) <= 1.5
    assert abs(summary["default_annual_pct"]["mean"] - 2.997) <= 1.0


# --- The bilingual pages ------------------------------------------------------

PAGES = {"en": ROOT / "docs" / "brief_sovereign_risk.md", "es": ROOT / "docs" / "es" / "brief_sovereign_risk.md"}


def _first_paragraph(text: str) -> str:
    body = text.split("\n# ", 1)[1].split("\n", 1)[1].strip()
    return body.split("\n\n", 1)[0]


@pytest.mark.parametrize("lang", ["en", "es"])
def test_pages_carry_generated_table_and_answer(record, lang):
    page = PAGES[lang].read_text(encoding="utf-8")
    assert brief.render_table(record, lang) in page
    assert brief.render_answer(record, lang) in page


@pytest.mark.parametrize("lang,words", [("en", ("illustrative", "not a claim about any country")),
                                        ("es", ("ilustrativ", "ningún país"))])
def test_pages_frame_the_research_rule(lang, words):
    first = _first_paragraph(PAGES[lang].read_text(encoding="utf-8"))
    for word in words:
        assert word in first, word
    assert "arellano_replication.md" in first


@pytest.mark.parametrize("lang,headings", [
    ("en", ["## Question", "## Answer", "## Results", "## Run it yourself", "## Assumptions and limits"]),
    ("es", ["## Pregunta", "## Respuesta", "## Resultados", "## Ejecútelo usted mismo", "## Supuestos y límites"]),
])
def test_pages_have_required_sections(lang, headings):
    page = PAGES[lang].read_text(encoding="utf-8")
    lines = page.splitlines()
    positions = [lines.index(h) for h in headings]
    assert positions == sorted(positions)
    assert "python -m puremacro.examples.first_brief_sovereign --scenario theta=0.15" in page
    assert ("Español" if lang == "en" else "English") in page[:600]
    assert "```python" not in page


@pytest.mark.parametrize("lang,ref", [("en", "assets/brief_sovereign_risk.en.svg"),
                                      ("es", "../assets/brief_sovereign_risk.es.svg")])
def test_figure_assets_exist_and_are_referenced(lang, ref):
    assert (ROOT / "docs" / "assets" / f"brief_sovereign_risk.{lang}.svg").is_file()
    assert f"]({ref})" in PAGES[lang].read_text(encoding="utf-8")


def test_timings_are_quoted_consistently(record, capsys):
    # One timing source: the frozen theta=0.15 converged solve (measured on a
    # shared machine under load) and the full freeze wall time, 6,851 s (1.9 h).
    case = record["scenarios"]["theta=0.15"]["cases"]["6401x641"]
    minutes = round(case["solver"]["elapsed_s"] / 60)
    help_text = brief._parser().format_help()
    assert f"about {minutes} minutes" in " ".join(help_text.split())
    assert "under load" in " ".join(help_text.split())
    assert brief.main(["--sweep"]) == 2
    err = " ".join(capsys.readouterr().err.split())
    assert "1.9 h" in err and "under load" in err
    load = record["load_averages"]["theta=0.15 6401x641"]
    for lang, page_path in PAGES.items():
        page = " ".join(page_path.read_text(encoding="utf-8").split())
        assert (f"{minutes} minutes" if lang == "en" else f"{minutes} minutos") in page
        assert ("under load" if lang == "en" else "bajo carga") in page
        assert "6,851 s (1.9 h)" in page or "6.851 s (1,9 h)" in page
        sep = "/" 
        loads = sep.join(f"{v:.0f}" for v in load)
        assert loads in page, loads
    assert "1.9 h" in " ".join(__doc__.split())


_LIBRARY_SOURCES = ("puremacro/sovereign/arellano.py", "puremacro/sovereign/_monotone.py",
                    "puremacro/sovereign/_results.py", "puremacro/sovereign/measurement.py")
_EXAMPLE = "puremacro/examples/first_brief_sovereign.py"
# The example's bytes changed after the freeze (record hash 4823e9ef…, commit
# 1752c83) without touching solve_scenario, measure_case or paired_change:
# 60a04ce0 added one type annotation, then the final-review fixes (caad961d,
# 38682a89, 072e549f) changed only answer/table/figure wording and CLI text.
# Any further edit must be reviewed the same way and its hash added here.
_EXAMPLE_HASHES = {
    "4823e9efa4b98fd3f1cd309baf1d5f79ebe201b9c918e3f15f0a17b297fd5858",  # frozen (1752c83)
    "59b96c52dcba6bf43184209d47af3901742f975acaef788117344435718a8ec2",  # current (072e549f)
}


def _provenance_problems(recorded: dict, current: dict) -> list[str]:
    problems = [f"{path}: record {recorded.get(path)} != current {current[path]}"
                for path in _LIBRARY_SOURCES if recorded.get(path) != current[path]]
    if recorded.get(_EXAMPLE) not in _EXAMPLE_HASHES:
        problems.append(f"{_EXAMPLE}: record hash {recorded.get(_EXAMPLE)} not in the allow-list")
    if current[_EXAMPLE] not in _EXAMPLE_HASHES:
        problems.append(f"{_EXAMPLE}: current hash {current[_EXAMPLE]} not in the allow-list")
    if set(recorded) != set(_LIBRARY_SOURCES) | {_EXAMPLE}:
        problems.append(f"unexpected source keys {sorted(recorded)}")
    return problems


def test_record_sources_match_the_current_files(record):
    assert _provenance_problems(record["source_sha256"], brief.source_hashes()) == []


@pytest.mark.parametrize("path", [*_LIBRARY_SOURCES, _EXAMPLE])
def test_provenance_check_catches_a_changed_source(record, path):
    perturbed = dict(record["source_sha256"])
    perturbed[path] = "0" * 64
    assert _provenance_problems(perturbed, brief.source_hashes())
    current = brief.source_hashes()
    current[path] = "f" * 64
    assert _provenance_problems(record["source_sha256"], current)


def _paired(record, low: str, high: str, moment: str) -> tuple[float, float]:
    a = np.asarray(record["scenarios"][low]["cases"]["6401x641"]["simulation"]["per_seed"][moment])
    b = np.asarray(record["scenarios"][high]["cases"]["6401x641"]["simulation"]["per_seed"][moment])
    d = b - a
    return float(d.mean()), float(student_t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))


@pytest.mark.parametrize("lang", ["en", "es"])
def test_volatility_sentence_quotes_the_record(record, lang):
    # All of the volatility effect on spread and default comes between
    # sigma_y 0.02 and 0.025; from 0.025 to 0.03 neither rises.
    page = " ".join(PAGES[lang].read_text(encoding="utf-8").split())
    fmt = (lambda v, f=".3f": format(v, f)) if lang == "en" else (lambda v, f=".3f": format(v, f).replace(".", ","))
    for moment in ("spread_mean_pct", "default_annual_pct"):
        means = [record["scenarios"][n]["cases"]["6401x641"]["simulation"]["summary"][moment]["mean"]
                 for n in ("sigma_y=0.02", "baseline", "sigma_y=0.03")]
        assert " / ".join(fmt(v) for v in means) in page, moment
        change, half = _paired(record, "baseline", "sigma_y=0.03", moment)
        digits = ".4f" if moment == "spread_mean_pct" else ".3f"
        assert f"{fmt(change, '+' + digits)} ± {fmt(half, digits)}" in page, (moment, change, half)


@pytest.mark.parametrize("lang,words", [
    ("en", ("1 - (1 - entries / repayment quarters)^4", "compounded", "2.989", "2.997")),
    ("es", ("1 - (1 - entradas / trimestres de pago)^4", "compuesta", "2,989", "2,997")),
])
def test_measurement_bullet_explains_the_default_convention(record, lang, words):
    limits = PAGES[lang].read_text(encoding="utf-8").split("\n## ")[-1]
    text = " ".join(limits.split())
    for word in words:
        assert word in text, word
    baseline = record["scenarios"]["baseline"]["cases"]["6401x641"]["simulation"]["summary"]
    assert f"{baseline['default_annual_pct']['mean']:.3f}" == "2.989"
