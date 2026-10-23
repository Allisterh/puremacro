"""Numerical parity, explicit estimands and portable study evidence."""
from dataclasses import FrozenInstanceError, replace
import hashlib
import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from puremacro.study import (
    VariableSpec, StudySpec, StudyComparison, compare_studies, load_study, run_study, save_study,
    ScenarioSpec, ScenarioResult, record_scenario,
)


@pytest.fixture
def data():
    rng = np.random.default_rng(173)
    y = np.zeros((240, 3))
    a = np.array([[.5, .03, 0], [.2, .4, .07], [0, .1, .6]])
    impact = np.array([[1, 0, 0], [.3, .8, 0], [.1, .2, .7]])
    for t in range(1, len(y)):
        y[t] = a @ y[t - 1] + impact @ rng.normal(size=3)
    frame = pd.DataFrame(y, index=pd.period_range("1960Q1", periods=len(y), freq="Q"), columns=["shock", "output", "price"])
    frame.attrs = {"source": "synthetic VAR", "seed": 173, "nested": {"vintage": pd.Timestamp("2026-01-01")}}
    return frame


@pytest.fixture
def lp_spec():
    return StudySpec(name="baseline", method="lp", variables=(VariableSpec("shock", "pp"),
        VariableSpec("output", "log points"), VariableSpec("price", "log points")),
        responses=("output",), shock="shock", frequency="Q", identification="observed_exogenous",
        lags=2, horizon=4, shock_size=.25, controls=("price",))


def var_spec(spec, **kwargs):
    return replace(spec, name="recursive", method="var_cholesky", identification="recursive",
                   controls=(), ordering=("price", "shock", "output"), **kwargs)


def test_lp_matches_direct_estimator_and_independent_design(data, lp_spec):
    from puremacro.lp import lp_hac
    result = run_study(data, lp_spec)
    direct = lp_hac(data, y="output", x="shock", controls=["price"], horizon=4, lags=2, ci=.9)
    got = result.to_frame()
    np.testing.assert_allclose(got.estimate, direct.beta * .25)
    np.testing.assert_allclose(got.lower, direct.lo * .25)
    np.testing.assert_allclose(got.upper, direct.hi * .25)
    # Independent regression at h=3, not a copy of the adapter's code path.
    t = np.arange(2, len(data) - 3)
    x = np.column_stack([np.ones(len(t)), data.shock.iloc[t],
        data.shock.iloc[t - 1], data.output.iloc[t - 1], data.price.iloc[t - 1],
        data.shock.iloc[t - 2], data.output.iloc[t - 2], data.price.iloc[t - 2], data.price.iloc[t]])
    y = data.output.to_numpy()[t + 3] - data.output.to_numpy()[t - 1]
    beta = np.linalg.lstsq(x, y, rcond=None)[0][1]
    assert got.loc[3, "estimate"] == pytest.approx(beta * .25)
    assert got.n_obs.tolist() == [238, 237, 236, 235, 234]
    assert result.manifest["sample"]["effective_by_horizon"][3]["end"] == str(data.index[-4])
    assert result.manifest["assumptions"]["response_convention"] == "y[t+h]-y[t-1]"
    assert result.audits.set_index("check").loc["identification", "status"] == "not_checked"


def test_var_matches_direct_engine_fixed_impact_and_ordering(data, lp_spec):
    from puremacro.var.estimate import estimate_var
    from puremacro.var.identify.cholesky import cholesky_svar
    spec = var_spec(lp_spec, n_boot=20, seed=71, shock_size=-.5)
    result = run_study(data, spec)
    ordered = data.loc[:, list(spec.ordering)].to_numpy()
    direct = cholesky_svar(ordered, p=2, horizon=4, n_boot=20, seed=71, ci=.9)
    impact = np.linalg.cholesky(estimate_var(ordered, 2).Sigma)
    factor = -.5 / impact[1, 1]
    got = result.to_frame()
    np.testing.assert_allclose(got.estimate, direct.irf_point[:, 2, 1] * factor)
    np.testing.assert_allclose(got.lower, direct.irf_upper[:, 2, 1] * factor)
    np.testing.assert_allclose(got.upper, direct.irf_lower[:, 2, 1] * factor)
    assert result.manifest["assumptions"]["response_convention"] == "level_irf"
    assert got.n_obs.tolist() == [238] * 5
    own = run_study(data, replace(spec, responses=("shock",), n_boot=0))
    assert own.to_frame().loc[0, "estimate"] == pytest.approx(-.5)
    assert "lower" not in own.to_frame()
    assert own.audits.set_index("check").loc["bootstrap", "status"] == "not_checked"


def test_lp_reserved_source_names_do_not_collide(data, lp_spec):
    baseline = run_study(data, lp_spec).to_frame()
    renamed = data.rename(columns={"shock": "dy_h", "output": "dy_h_L1", "price": "odd name_%"})
    spec = replace(lp_spec, shock="dy_h", responses=("dy_h_L1",), controls=("odd name_%",),
        variables=tuple(VariableSpec(name, v.unit) for name, v in zip(renamed.columns, lp_spec.variables)))
    got = run_study(renamed, spec).to_frame()
    np.testing.assert_array_equal(got.estimate, baseline.estimate)


def test_transform_sample_units_are_explicit_and_inputs_immutable(data, lp_spec):
    original = data.copy(deep=True)
    positive = data.assign(output=np.exp(data.output / 20))
    spec = replace(lp_spec, variables=(lp_spec.variables[0], VariableSpec("output", "percent log growth", "logdiff", 100), lp_spec.variables[2]),
                   sample_start="1970Q1", sample_end="2000Q4")
    result = run_study(positive, spec)
    expected = np.log(positive.loc["1970Q1":"2000Q4", "output"]).diff().iloc[1:] * 100
    pd.testing.assert_series_equal(result.data.output, expected)
    assert result.manifest["sample"]["raw_n_obs"] == 124
    assert result.manifest["sample"]["transformed_n_obs"] == 123
    assert result.to_frame().unit.unique().tolist() == ["percent log growth"]
    pd.testing.assert_frame_equal(data, original)
    assert result.inputs.attrs == data.attrs
    frame = result.data
    frame.iloc[0, 0] = 999
    frame.attrs["nested"]["vintage"] = "changed"
    manifest = result.manifest
    manifest["spec"]["name"] = "changed"
    assert result.data.iloc[0, 0] != 999
    assert result.data.attrs == data.attrs and result.spec.name == "baseline"
    with pytest.raises(FrozenInstanceError):
        result.spec.name = "changed"


def test_signed_lp_shocks_and_no_name_based_scaling(data, lp_spec):
    positive = run_study(data, lp_spec).to_frame()
    negative = run_study(data, replace(lp_spec, shock_size=-.25)).to_frame()
    np.testing.assert_array_equal(negative.estimate, -positive.estimate)
    np.testing.assert_array_equal(negative.lower, -positive.upper)
    renamed = data.rename(columns={"output": "short_rate"})
    spec = replace(lp_spec, responses=("short_rate",), variables=(lp_spec.variables[0], VariableSpec("short_rate", "log points"), lp_spec.variables[2]))
    np.testing.assert_array_equal(run_study(renamed, spec).to_frame().estimate, positive.estimate)


def test_compare_like_studies_records_different_lag_samples(data, lp_spec):
    first = run_study(data, lp_spec)
    second = run_study(data, replace(lp_spec, name="three lags", lags=3))
    compared = compare_studies(first, second)
    assert compared.to_frame().groupby("study").n_obs.max().to_dict() == {"baseline": 238, "three lags": 237}
    assert set(compared.to_frame().lags) == {2, 3}
    assert "Fixed 3" in compared.to_markdown()
    assert "tabular" in compared.to_latex() and "#table" in compared.to_typst()
    external = [first, second]
    direct = StudyComparison(external)
    external.clear()
    assert len(direct.studies) == 2
    with pytest.raises(ValueError, match="comparison"):
        StudyComparison([first, run_study(data, var_spec(lp_spec))])


@pytest.mark.parametrize("change", ["method", "size", "unit", "frequency", "data", "sample", "name"])
def test_compare_rejects_unlike_contracts(data, lp_spec, change):
    first = run_study(data, lp_spec)
    spec = replace(lp_spec, name="other")
    altered = data.copy()
    if change == "method":
        spec = var_spec(spec)
    elif change == "size":
        spec = replace(spec, shock_size=1)
    elif change == "unit":
        spec = replace(spec, variables=(VariableSpec("shock", "basis points"), *spec.variables[1:]))
    elif change == "frequency":
        altered.index = pd.period_range("1960-01", periods=len(data), freq="M")
        spec = replace(spec, frequency="M")
    elif change == "data":
        altered.iloc[2, 0] += .1
    elif change == "sample":
        spec = replace(spec, sample_start="1961Q1")
    else:
        spec = lp_spec
    second = run_study(altered, spec)
    with pytest.raises(ValueError, match="comparison"):
        compare_studies(first, second)


@pytest.mark.parametrize("bad", ["gap", "duplicate", "unsorted", "nan", "inf", "object", "complex", "frequency", "short", "rank", "attrs"])
def test_invalid_data_rejected_before_any_estimator(data, lp_spec, bad, monkeypatch):
    import puremacro.lp
    def forbidden(*a, **kw):
        raise AssertionError("estimator ran before input preflight")
    monkeypatch.setattr(puremacro.lp, "lp_hac", forbidden)
    changed = data.copy(deep=True)
    spec = lp_spec
    if bad == "gap": changed = changed.drop(changed.index[30])
    elif bad == "duplicate": changed.index = pd.Index([changed.index[0], *changed.index[:-1]])
    elif bad == "unsorted": changed = changed.iloc[::-1]
    elif bad == "nan": changed.iloc[4, 0] = np.nan
    elif bad == "inf": changed.iloc[4, 0] = np.inf
    elif bad == "object": changed["shock"] = changed.shock.astype(str)
    elif bad == "complex": changed["shock"] = changed.shock.astype(complex) + 1j
    elif bad == "frequency": spec = replace(spec, frequency="M")
    elif bad == "short": changed = changed.iloc[:8]
    elif bad == "rank": changed["price"] = changed.shock
    else: changed.attrs["unserializable"] = object()
    with pytest.raises(ValueError):
        run_study(changed, spec)


def test_global_rng_unchanged_and_repeated_archive_deterministic(data, lp_spec):
    before = np.random.get_state()
    spec = var_spec(lp_spec, n_boot=12, seed=98)
    a = run_study(data, spec)
    b = run_study(data, spec)
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    assert save_study(a) == save_study(b)


@pytest.mark.pyodide_smoke
def test_study_offline_memory_roundtrip(data, lp_spec):
    """Exercise the same portable contracts in an actual browser kernel."""
    for spec in (lp_spec, var_spec(lp_spec)):
        result = run_study(data, spec)
        payload = save_study(result)
        loaded = load_study(payload)
        pd.testing.assert_frame_equal(loaded.to_frame(), result.to_frame())
        assert loaded.inputs.attrs == data.attrs
        assert loaded.manifest == result.manifest


def test_safe_archive_roundtrip_report_and_no_overwrite(data, lp_spec, tmp_path):
    result = run_study(data, lp_spec)
    archive = save_study(result)
    assert isinstance(archive, bytes)
    loaded = load_study(archive)
    assert loaded.manifest == result.manifest
    for name in ("inputs", "data", "audits"):
        pd.testing.assert_frame_equal(getattr(loaded, name), getattr(result, name))
    pd.testing.assert_frame_equal(loaded.to_frame(), result.to_frame())
    p = tmp_path / "test.pmstudy"
    assert save_study(result, p) == p
    with pytest.raises(FileExistsError): save_study(result, p)
    assert load_study(p).manifest == result.manifest
    folder = result.write_report(tmp_path / "report")
    assert set(p.name for p in folder.iterdir()) == {"report.md", "manifest.json", "results.csv", "results.tex", "results.typ", "audits.csv", "study.pmstudy"}
    assert "observed_exogenous" in (folder / "report.md").read_text()
    assert json.loads(result.to_json())["manifest"] == result.manifest
    with pytest.raises(FileExistsError): result.write_report(folder)


def mutate_archive(payload, mutation):
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        members = {name: z.read(name) for name in z.namelist()}
    mutation(members)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        for name, value in members.items(): z.writestr(name, value)
    return out.getvalue()


@pytest.mark.parametrize("mode", ["missing", "extra", "traversal", "payload", "spec", "version", "nonfinite", "duplicate_json"])
def test_archive_rejects_tampering_and_incomplete_inventory(data, lp_spec, mode):
    payload = save_study(run_study(data, lp_spec))
    def mutate(m):
        if mode == "missing": m.pop("data.npz")
        elif mode == "extra": m["extra.py"] = b"raise RuntimeError('must not execute')"
        elif mode == "traversal": m["../outside"] = b"no extraction"
        elif mode == "payload": m["results.npz"] = m["inputs.npz"]
        else:
            manifest = json.loads(m["manifest.json"])
            if mode == "spec": manifest["spec"]["seed"] += 1
            elif mode == "version": manifest["schema_version"] = 99
            elif mode == "nonfinite": manifest["spec"]["shock_size"] = float('nan')
            else:
                m["manifest.json"] = b'{"format":"a","format":"b"}'
                return
            m["manifest.json"] = json.dumps(manifest).encode()
    with pytest.raises(ValueError): load_study(mutate_archive(payload, mutate))


def test_archive_load_never_executes_and_bounds_payloads(data, lp_spec, monkeypatch):
    import puremacro.lp
    result = run_study(data, lp_spec)
    payload = save_study(result)
    def forbidden(*a, **kw): raise AssertionError("loading executed an estimator")
    monkeypatch.setattr(puremacro.lp, "lp_hac", forbidden)
    assert load_study(payload).manifest == result.manifest
    with pytest.raises(ValueError, match="size limit"): load_study(payload, max_bytes=100)
    with pytest.raises(ValueError): load_study(b"not a zip")


@pytest.mark.mechanism_control
def test_estimator_sentinel_positive_control(data, lp_spec, monkeypatch):
    import puremacro.lp
    def forbidden(*a, **kw):
        raise AssertionError("estimator sentinel reached")
    monkeypatch.setattr(puremacro.lp, "lp_hac", forbidden)
    with pytest.raises(AssertionError, match="estimator sentinel reached"):
        run_study(data, lp_spec)


def resign(members):
    manifest = json.loads(members["manifest.json"])
    for name in ("inputs", "data", "results", "audits"):
        payload = members[name + ".npz"]
        manifest["frames"][name] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    encode = lambda obj: json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    manifest["spec_sha256"] = hashlib.sha256(encode(manifest["spec"])).hexdigest()
    manifest.pop("record_sha256")
    manifest["record_sha256"] = hashlib.sha256(encode(manifest)).hexdigest()
    members["manifest.json"] = encode(manifest)


@pytest.mark.parametrize("mode", ["nonfinite_data", "wrong_unit", "wrong_count", "old_codec", "wrong_dates", "wrong_assumption", "huge_horizon"])
def test_rehashed_archive_still_requires_finite_values_and_consistent_provenance(data, lp_spec, mode):
    from puremacro.runtime import store
    payload = save_study(run_study(data, lp_spec))
    def mutate(m):
        if mode in {"nonfinite_data", "wrong_dates", "old_codec"}:
            name = "data.npz"
            frame = store.loads_frame(m[name])
            if mode == "old_codec":
                m[name] = store._dumps_frame(frame, schema_version=1)
            else:
                if mode == "nonfinite_data": frame.iloc[0, 0] = np.inf
                else: frame.index = frame.index[::-1]
                m[name] = store.dumps_frame(frame)
        elif mode in {"wrong_assumption", "huge_horizon"}:
            manifest = json.loads(m["manifest.json"])
            if mode == "wrong_assumption": manifest["assumptions"]["response_convention"] = "level_irf"
            else: manifest["spec"]["horizon"] = 10 ** 30
            m["manifest.json"] = json.dumps(manifest).encode()
        else:
            frame = store.loads_frame(m["results.npz"])
            if mode == "wrong_unit": frame["unit"] = "invented"
            else: frame["n_obs"] = 2
            m["results.npz"] = store.dumps_frame(frame)
        resign(m)
    with pytest.raises(ValueError):
        load_study(mutate_archive(payload, mutate))


def test_npy_header_allocation_and_inner_zip_expansion_are_bounded(data, lp_spec):
    payload = save_study(run_study(data, lp_spec))
    def mutate(m):
        def corrupt(inner):
            header = io.BytesIO()
            np.lib.format.write_array_header_1_0(header, {"descr": "<f8", "fortran_order": False, "shape": (10 ** 12,)})
            inner["c0.npy"] = header.getvalue()  # no payload: allocation must never be attempted
        m["data.npz"] = mutate_archive(m["data.npz"], corrupt)
        resign(m)
    with pytest.raises(ValueError, match="shape exceeds"):
        load_study(mutate_archive(payload, mutate))
    def compressed(m):
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as inner:
            inner.writestr("__puremacro_schema__.npy", b"0" * 200_000)
        m["data.npz"] = out.getvalue()
        resign(m)
    with pytest.raises(ValueError, match="size limit"):
        load_study(mutate_archive(payload, compressed), max_bytes=100_000)


@pytest.mark.parametrize("kwargs", [{"method": "mystery"}, {"lags": True}, {"shock_size": np.inf},
    {"shock_size": 0}, {"ci": 1}, {"responses": "output"}, {"controls": ("output",)},
    {"n_boot": 2}, {"identification": "recursive"}, {"sample_start": "bad date"}])
def test_invalid_spec_rejected(data, lp_spec, kwargs):
    with pytest.raises(ValueError):
        run_study(data, replace(lp_spec, **kwargs))


def scenario_for(study, **overrides):
    spec = study.spec
    model_spec = ScenarioSpec(name="analytical model", model="declared AR illustration",
        variables=spec.variables, responses=spec.responses, shock=spec.shock,
        frequency=spec.frequency, identification=spec.identification,
        response_convention=spec.response_convention, horizon=spec.horizon, shock_size=spec.shock_size)
    model_spec = replace(model_spec, **overrides)
    output = pd.DataFrame([{"response": r, "horizon": h, "estimate": .1 * .7 ** h}
                          for r in model_spec.responses for h in range(model_spec.horizon + 1)])
    output.attrs = {"derivation": "explicit analytical illustration"}
    audit = pd.DataFrame([{"check": "economic_identification", "status": "not_checked",
                           "detail": "User-declared exogeneity; no empirical identification test."}])
    return record_scenario(output, model_spec, inputs=study.inputs, transformed_inputs=study.data,
        model_source="y[h] = 0.1 * 0.7**h  # inert source description",
        parameters={"persistence": .7}, audits=audit, provenance={"source": "test", "vintage": pd.Timestamp("2026-01-01")},
        evidence={"joint_covariance": pd.DataFrame([[1., .4], [.4, 2.]], index=["m1", "m2"], columns=["m1", "m2"]),
                  "held_out": pd.DataFrame({"horizon": [8], "target": [.2], "model": [.15]})})


def test_scenario_record_roundtrip_evidence_and_explicit_comparison(data, lp_spec, tmp_path):
    empirical = run_study(data, lp_spec)
    scenario = scenario_for(empirical)
    assert scenario.to_frame().attrs == {"derivation": "explicit analytical illustration"}
    np.testing.assert_array_equal(scenario.evidence["joint_covariance"], [[1., .4], [.4, 2.]])
    assert scenario.provenance["vintage"] == pd.Timestamp("2026-01-01")
    assert scenario.audits.status.tolist() == ["not_checked"]
    loaded = load_study(save_study(scenario))
    assert isinstance(loaded, ScenarioResult)
    assert loaded.manifest == scenario.manifest and loaded.model_source == scenario.model_source
    assert loaded.provenance == scenario.provenance
    pd.testing.assert_frame_equal(loaded.evidence["joint_covariance"], scenario.evidence["joint_covariance"])
    with pytest.raises(ValueError, match="comparison_note"):
        compare_studies(empirical, scenario)
    compared = compare_studies(empirical, scenario, comparison_note="Matched externally identified shock and pre-shock baseline; illustrative assumptions.")
    assert "illustrative assumptions" in compared.to_markdown()
    assert compared.to_frame().attrs["comparison_note"] == compared.comparison_note
    assert compared.to_frame().groupby("study").size().to_dict() == {"analytical model": 5, "baseline": 5}
    folder = scenario.write_report(tmp_path / "scenario")
    assert (folder / "model_source.txt").read_text() == scenario.model_source
    assert load_study(folder / "study.pmstudy").manifest == scenario.manifest


@pytest.mark.parametrize("changed", [{"response_convention": "level_irf"}, {"identification": "calibrated_scenario"},
    {"shock_size": .5}, {"frequency": "M"}, {"horizon": 3}])
def test_scenario_comparison_rejects_unlike_estimands(data, lp_spec, changed):
    empirical = run_study(data, lp_spec)
    scenario = scenario_for(empirical, **changed)
    with pytest.raises(ValueError, match="unlike comparison"):
        compare_studies(empirical, scenario, comparison_note="A note cannot override inconsistent units/conventions.")


@pytest.mark.pyodide_smoke
def test_static_scenario_does_not_require_an_empirical_time_index_or_execute_source():
    inputs = pd.DataFrame([[.2, .1], [.3, .1]], columns=["sector_a", "sector_b"])
    spec = ScenarioSpec(name="tariff scenario", model="static input-output illustration",
        variables=(VariableSpec("output", "percent"), VariableSpec("tariff", "percentage points")),
        responses=("output",), shock="tariff", frequency="annual_static",
        identification="calibrated_input_output_scenario", response_convention="static_level_change", horizon=0, shock_size=10.)
    scenario = record_scenario(pd.DataFrame({"response": ["output"], "horizon": [0], "estimate": [-.3]}), spec,
        inputs=inputs, model_source="raise RuntimeError('source must never execute')", parameters={"elasticity": 2.},
        audits=pd.DataFrame([{"check": "general_equilibrium", "status": "unsupported", "detail": "Fixed coefficients and prices."}]),
        provenance={"source": "synthetic coefficients"})
    loaded = load_study(save_study(scenario))
    pd.testing.assert_frame_equal(loaded.inputs, inputs)
    assert loaded.audits.status.iloc[0] == "unsupported"
    assert loaded.manifest["format"] == "puremacro-scenario"


@pytest.mark.parametrize("mode", ["source", "missing_evidence", "extra_file", "rehashed_wrong_units", "rehashed_nonfinite"])
def test_scenario_archive_rejects_modified_incomplete_or_invalid_evidence(data, lp_spec, mode):
    payload = save_study(scenario_for(run_study(data, lp_spec)))
    from puremacro.runtime import store
    def mutate(m):
        if mode == "source": m["model_source.txt"] = b"different model"
        elif mode == "missing_evidence": m.pop("evidence.joint_covariance.npz")
        elif mode == "extra_file": m["run.py"] = b"raise RuntimeError('never execute')"
        else:
            key = "results" if mode == "rehashed_wrong_units" else "evidence.joint_covariance"
            frame = store.loads_frame(m[key + ".npz"])
            if mode == "rehashed_wrong_units": frame["unit"] = "wrong units"
            else: frame.iloc[0, 0] = np.nan
            m[key + ".npz"] = store.dumps_frame(frame)
            manifest = json.loads(m["manifest.json"])
            manifest["frames"][key] = {"sha256": hashlib.sha256(m[key + ".npz"]).hexdigest(), "bytes": len(m[key + ".npz"])}
            manifest.pop("record_sha256")
            encode = lambda obj: json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
            manifest["record_sha256"] = hashlib.sha256(encode(manifest)).hexdigest()
            m["manifest.json"] = encode(manifest)
    with pytest.raises(ValueError):
        load_study(mutate_archive(payload, mutate))


def test_scenario_rejects_silent_named_column_metadata_loss(data, lp_spec):
    base = scenario_for(run_study(data, lp_spec))
    named = pd.DataFrame([[1., .2], [.2, 1.]], columns=pd.MultiIndex.from_tuples(
        [("output", 0), ("output", 1)], names=["response", "horizon"]))
    with pytest.raises(ValueError, match="losslessly"):
        record_scenario(base.to_frame().drop(columns=["unit", "shock_unit"]), base.spec,
            inputs=base.inputs, transformed_inputs=base.data, model_source=base.model_source,
            parameters={}, audits=base.audits, provenance={}, evidence={"named_columns": named})
