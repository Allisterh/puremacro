"""The first policy brief: a 10% US tariff on Chinese goods, national and household welfare."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from _timing import budget

import puremacro
from puremacro.examples import first_brief_tariffs as fb


@pytest.fixture(scope="module")
def solved():
    """One default solve (about 6 s on a quiet machine) shared by the module."""
    return fb.solve_brief()


# --- Task 1: solve stage and national record ---------------------------------

def test_parse_experiment():
    assert fb.parse_experiment("none") is None
    assert fb.parse_experiment("sigma=2") == 2.0
    assert fb.parse_experiment("sigma=0.5") == 0.5
    for bad in ("theta=4", "sigma", "sigma=-1", "sigma=nan", "sigma=0"):
        with pytest.raises(ValueError):
            fb.parse_experiment(bad)


@pytest.mark.parametrize(
    ("importer", "exporter", "sector", "tariff", "offending"),
    [
        ("USA", "USA", "all", 0.1, "USA"),
        ("XXX", "CHN", "all", 0.1, "XXX"),
        ("USA", "XXX", "all", 0.1, "XXX"),
        ("USA", "CHN", "Z", 0.1, "Z"),
        ("USA", "CHN", "all", -0.1, "-0.1"),
        ("USA", "CHN", "all", math.nan, "nan"),
    ],
)
def test_validate_scenario_rejects(importer, exporter, sector, tariff, offending):
    icio, _ = fb.load_bundled()
    with pytest.raises(ValueError, match=offending):
        fb.validate_scenario(icio, importer, exporter, sector, tariff)


def test_default_solution_converged_and_split_adds_up(solved):
    assert solved.base.converged and solved.counterfactual.converged
    rec = fb.national_record(solved)
    total = (rec["price_effect_musd"] + rec["factor_income_effect_musd"]
             + rec["fiscal_transfer_effect_musd"] + rec["decomposition_residual_musd"])
    assert total == pytest.approx(rec["ev_musd"], rel=1e-9)
    assert abs(rec["decomposition_residual_musd"]) < 1e-4
    assert rec["ev_pct_consumption"] > 0
    assert solved.settings["sigma"] == 0.0
    for key, value in fb.SOLVER.items():
        assert solved.settings[key] == value
    assert set(solved.timings) == {"load", "calibration", "baseline", "counterfactual", "welfare"}


def test_nonconvergence_raises(monkeypatch):
    monkeypatch.setattr(
        fb, "run_scenario",
        lambda *a, **k: SimpleNamespace(converged=False, metadata={"solver_termination": "max_iter"}))
    with pytest.raises(RuntimeError, match="counterfactual did not converge.*max_iter"):
        fb.solve_brief()


# --- Task 2: household groups --------------------------------------------------

@pytest.fixture(scope="module")
def icio():
    return fb.load_bundled()[0]


def _shares(frame, sectors):
    return frame[list(sectors)]


def test_synthetic_groups_reproduce_national_basket(icio):
    groups = fb.synthetic_groups(icio, "USA")
    sectors = list(icio.sector_codes)
    assert list(groups.index) == ["low", "middle", "high"]
    assert groups.index.name == "group"
    assert list(groups.columns) == ["population_weight", "annual_spending", *sectors]
    np.testing.assert_allclose(groups["population_weight"], [1 / 3] * 3)
    np.testing.assert_allclose(groups["annual_spending"], [30_000, 60_000, 120_000])
    shares = _shares(groups, sectors)
    np.testing.assert_allclose(shares.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    spend = groups["population_weight"] * groups["annual_spending"]
    mean = (shares.mul(spend, axis=0).sum(axis=0) / spend.sum())
    basket = fb.national_basket(icio, "USA")
    np.testing.assert_allclose(mean.to_numpy(), basket.reindex(sectors).to_numpy(), rtol=0, atol=1e-12)
    goods = shares["A"] + shares["C"]
    assert goods["low"] > goods["middle"] > goods["high"]
    assert groups.attrs["is_synthetic"] is True
    assert groups.attrs["label"] == fb.SYNTHETIC_LABEL


def test_synthetic_groups_other_importer(icio):
    groups = fb.synthetic_groups(icio, "MEX")
    sectors = list(icio.sector_codes)
    supported = set(fb.supported_sectors(icio, "MEX"))
    for s in sectors:
        if s not in supported:
            assert (groups[s] == 0).all(), s
    np.testing.assert_allclose(_shares(groups, sectors).sum(axis=1), 1.0, rtol=0, atol=1e-12)


def test_household_aggregate_matches_national(solved):
    groups = fb.synthetic_groups(solved.icio, "USA")
    inc = fb.household_incidence(solved, groups)
    national = fb.national_record(solved)["ev_pct_consumption"]
    assert list(inc.columns) == ["population_weight", "annual_spending", "ev_pct_proportional",
                                 "ev_pct_equal", "price_ev_pct", "income_ev_pct"]
    assert abs(inc.attrs["aggregate_ev_pct_proportional"] - national) < 1e-4
    assert abs(inc.attrs["aggregate_ev_pct_equal"] - national) < 1e-4
    assert inc.attrs["is_synthetic"] is True
    assert inc.attrs["label"] == fb.SYNTHETIC_LABEL


def test_groups_csv_round_trip(solved, tmp_path):
    icio_ = solved.icio
    sectors = list(icio_.sector_codes)
    supported = fb.supported_sectors(icio_, "USA")
    original = fb.synthetic_groups(icio_, "USA")
    path = tmp_path / "groups.csv"
    original.to_csv(path)
    back = fb.read_groups_csv(path, sectors, supported)
    np.testing.assert_allclose(back.to_numpy(), original.to_numpy(), rtol=1e-12, atol=0)
    assert list(back.index) == list(original.index)
    a = fb.household_incidence(solved, original)
    b = fb.household_incidence(solved, back)
    np.testing.assert_allclose(b.to_numpy(), a.to_numpy(), rtol=1e-10, atol=0)

    basket = fb.national_basket(icio_, "USA")
    row_a = {"group": "a", "population_weight": 0.6, "annual_spending": 40_000,
             **{s: float(basket[s]) for s in sectors}}
    row_b = {"group": "b", "population_weight": 0.4, "annual_spending": 80_000,
             **{s: 0.0 for s in sectors}}
    row_b.update({"C": 0.5, "G": 0.3, "L": 0.2})
    tiny = tmp_path / "tiny.csv"
    pd.DataFrame([row_a, row_b]).to_csv(tiny, index=False)
    user = fb.read_groups_csv(tiny, sectors, supported)
    assert user.attrs["is_synthetic"] is False
    assert user.attrs["source"] == "tiny.csv"
    inc = fb.household_incidence(solved, user)
    assert list(inc.index) == ["a", "b"]
    assert inc.attrs["is_synthetic"] is False
    assert "SYNTHETIC" not in inc.attrs["label"]


def _valid_rows(sectors):
    base = {s: 1.0 / len(sectors) for s in sectors}
    return [{"group": "x", "population_weight": 0.5, "annual_spending": 50_000, **base},
            {"group": "y", "population_weight": 0.5, "annual_spending": 70_000, **base}]


def _mutate(case, rows, sectors):
    if case == "missing_T":
        for r in rows:
            r.pop("T")
    elif case == "extra_Z":
        for r in rows:
            r["Z"] = 0.0
    elif case == "sum_0.9":
        rows[1]["A"] -= 0.1
    elif case == "negative_share":
        rows[1]["B"] += rows[1]["A"] + 0.01  # row still sums to one
        rows[1]["A"] = -0.01
    elif case == "zero_weight":
        rows[1]["population_weight"] = 0.0
    elif case == "negative_spending":
        rows[1]["annual_spending"] = -1.0
    elif case == "duplicate_group":
        rows[1]["group"] = "x"
    elif case == "unsupported":
        rows[1]["T"] += 0.0  # positive already; the supported set excludes T
    return rows


@pytest.mark.parametrize(("case", "pattern"), [
    ("missing_T", r"'T'"),
    ("extra_Z", r"'Z'"),
    ("sum_0.9", r"'y'"),
    ("negative_share", r"'y'.*'A'|'A'.*'y'"),
    ("zero_weight", r"population_weight.*'y'|'y'.*population_weight"),
    ("negative_spending", r"annual_spending.*'y'|'y'.*annual_spending"),
    ("duplicate_group", r"'x'"),
    ("unsupported", r"'T'"),
])
def test_read_groups_csv_rejects(case, pattern, tmp_path):
    sectors = [chr(c) for c in range(ord("A"), ord("T") + 1)]
    supported = [s for s in sectors if s != "T"] if case == "unsupported" else sectors
    rows = _mutate(case, _valid_rows(sectors), sectors)
    path = tmp_path / f"{case}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError, match=pattern):
        fb.read_groups_csv(path, sectors, supported)


# --- Task 3: record, outputs and CLI -----------------------------------------

NO_RETALIATION = {"en": "unilateral terms-of-trade gain with no retaliation",
                  "es": "ganancia unilateral de términos de intercambio, sin represalias"}
CITATION = "OECD (2023), OECD Inter-Country Input-Output Tables, http://oe.cd/icio"
BASE_FILES = {"national.md", "national.csv", "national.tex", "households.md", "households.csv",
              "households.tex", "first_brief.png", "headline.json", "manifest.json"}


@pytest.fixture(scope="module")
def households(solved):
    return fb.household_incidence(solved, fb.synthetic_groups(solved.icio, "USA"))


def test_outputs_written(solved, households, tmp_path):
    fb.write_outputs(solved, households, tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == BASE_FILES
    written = json.loads((tmp_path / "headline.json").read_text(encoding="utf-8"))
    assert written == fb.headline_record(solved, households)


@pytest.mark.parametrize("lang", ["en", "es"])
def test_synthetic_label_everywhere(solved, households, tmp_path, lang):
    fb.write_outputs(solved, households, tmp_path, lang=lang)
    for name in ("households.md", "households.csv", "households.tex", "headline.json", "manifest.json"):
        assert "SYNTHETIC" in (tmp_path / name).read_text(encoding="utf-8"), name
    summary = fb.summary_text(fb.headline_record(solved, households), lang)
    assert "SYNTHETIC" in summary
    assert NO_RETALIATION[lang] in (tmp_path / "national.md").read_text(encoding="utf-8")
    assert NO_RETALIATION[lang] in summary


def test_manifest_provenance(solved, households, tmp_path):
    manifest = fb.write_outputs(solved, households, tmp_path)
    assert manifest == json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["table"]["citation"] == CITATION
    npz = Path(solved.icio.metadata["file_path"])
    assert manifest["table"]["sha256"] == hashlib.sha256(npz.read_bytes()).hexdigest()
    assert manifest["table"]["file"] == npz.name
    assert manifest["solver"] == {**fb.SOLVER, "sigma": 0.0}
    assert manifest["convergence"]["baseline"]["converged"] is True
    assert manifest["convergence"]["counterfactual"]["converged"] is True
    assert {"load", "calibration", "baseline", "counterfactual", "welfare", "households"} <= set(manifest["timings"])
    assert manifest["package_version"] == puremacro.__version__
    assert manifest["experiment"] is None
    for name in BASE_FILES - {"manifest.json"}:
        data = (tmp_path / name).read_bytes()
        assert manifest["artifacts"][name] == {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def test_page_block_formats(solved, households):
    record = fb.headline_record(solved, households)
    nat = record["national"]
    en = fb.page_block(record, "headline", "en")
    assert "Purchaser prices" in en and f"{nat['ev_musd']:,.0f}" in en
    es = fb.page_block(record, "headline", "es")
    assert "Precios de comprador" in es
    assert f"{nat['ev_pct_consumption']:.4f}".replace(".", ",") in es
    assert f"{nat['ev_musd']:,.0f}".replace(",", " ") in es
    assert fb.SYNTHETIC_LABEL in fb.page_block(record, "households", "en")
    assert fb.SYNTHETIC_LABEL_ES in fb.page_block(record, "households", "es")


@pytest.mark.parametrize(("argv", "offending"), [
    (["--exporter", "USA"], "USA"),
    (["--importer", "XXX"], "XXX"),
    (["--sector", "Z"], "Z"),
    (["--tariff", "-0.1"], "-0.1"),
    (["--experiment", "theta=4"], "theta=4"),
    (["--lang", "fr"], "fr"),
])
def test_cli_rejects_bad_scenario(argv, offending, tmp_path, capsys, monkeypatch):
    def no_solve(*a, **k):
        raise AssertionError("main solved before validating its arguments")
    monkeypatch.setattr(fb, "solve_brief", no_solve)
    t0 = time.perf_counter()
    with pytest.raises(SystemExit) as exc:
        fb.main([*argv, "--output", str(tmp_path / "out")])
    elapsed = time.perf_counter() - t0
    assert exc.value.code == 2
    assert offending in capsys.readouterr().err
    assert elapsed < budget(5), f"CLI validation took {elapsed:.1f} s > {budget(5):.1f} s"
    assert not (tmp_path / "out").exists()


def test_no_outputs_on_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        fb, "run_scenario",
        lambda *a, **k: SimpleNamespace(converged=False, metadata={"solver_termination": "max_iter"}))
    with pytest.raises(RuntimeError, match="counterfactual did not converge"):
        fb.run_brief(tmp_path / "out")
    assert not (tmp_path / "out" / "headline.json").exists()
