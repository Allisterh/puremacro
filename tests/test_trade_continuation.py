"""Parameter continuation, multistart and fold helpers: contract, step law and audits."""
from dataclasses import replace
import json
import pickle

import numpy as np
import pytest

from puremacro.trade import solve_policy_equilibrium
from puremacro.trade import continuation as module
from puremacro.trade.continuation import (DirectTargetResult, ParameterContinuationFailure,
    ParameterContinuationResult, continue_parameter, sigma_path, try_starts)
from puremacro.trade.data import generate_synthetic_mrio, package_mrio_to_calibration_result
from puremacro.trade._oecd_icio import condense_final_demand
from puremacro.trade.welfare import _checked_state

TOL = 1e-9


def synthetic_calibration(seed=2):
    """Three-country, three-sector synthetic OECD-layout table under consistent accounting.

    The six raw final uses are condensed to C/I/Cx (zero final-use taxes). Seed 2
    is the first seed whose condensed table has nonnegative expenditure shares;
    the default seed 42 is rejected by consistent accounting.
    """
    raw = generate_synthetic_mrio("oecd", custom_c=3, custom_s=3, seed=seed)
    raw = replace(raw, taxes_less_subsidies_fd=np.zeros(raw.C*raw.K_F))
    return package_mrio_to_calibration_result(condense_final_demand(raw))


def import_tariffs(calib, importer=0, rate=.1):
    """Uniform ad-valorem import tariff by one importer on every foreign origin, all uses."""
    nc, ns, nfd = calib.nc, calib.ns, calib.n_final_demand
    ta, tf = np.ones((ns*nc, ns, nc)), np.ones((ns*nc, nfd, nc))
    for origin in range(nc):
        if origin != importer:
            ta[origin*ns:(origin+1)*ns, :, importer] = 1+rate
            tf[origin*ns:(origin+1)*ns, :, importer] = 1+rate
    return ta, tf


@pytest.fixture(scope="module")
def model():
    calib = synthetic_calibration()
    ta, tf = import_tariffs(calib)
    base = solve_policy_equilibrium(calib, ta, tf, sigma=0., tol=TOL)
    target = solve_policy_equilibrium(calib, ta, tf, sigma=2., tol=TOL)
    return calib, ta, tf, base, target


@pytest.fixture(scope="module")
def replay(model):
    """The sigma=2 root re-solved by warm Newton from itself: its solver record says seed "warm".

    Scripted solvers return copies of this state, so the default warm-start
    rule (allow_fallback=False) accepts them exactly as it accepts a real
    warm-started Newton trial.
    """
    calib, ta, tf, _, target = model
    state = solve_policy_equilibrium(calib, ta, tf, sigma=2., tol=TOL, x0=target.x_sol, method="newton")
    assert state.metadata["policy_solver_attempts"][-1]["seed"] == "warm"
    assert state.metadata["policy_solver_fallback_used"] is False
    return state


def audited_residual(result, calib):
    blocks, _ = _checked_state(result, calib, 1e-8)
    return float(np.max(np.abs(np.r_[blocks["residuals"], blocks["physical_residuals"]])))


# ---------------------------------------------------------------------------
# Certified paths on the synthetic table
# ---------------------------------------------------------------------------
def test_identity_path_audits_start_and_reports_fraction_one(model):
    calib, ta, tf, _, target = model
    seen = []
    result = continue_parameter(calib, lambda f: dict(tau=ta, tau_fd=tf, sigma=2., tol=TOL), target,
                                stage_callback=lambda st, info: seen.append((st, info)),
                                metadata={"policy": "fixture"})
    assert isinstance(result, ParameterContinuationResult)
    assert result.status == "accepted" and result.parameter == "" and result.n_stages == 1
    assert result.fractions.tolist() == [0.] and result.last_fraction == 1.
    assert result.final is target and result.metadata["policy"] == "fixture"
    assert len(result.invariant_id) == 64 and result.metadata["solver_calls"] == 0
    assert seen[0][0] is target and seen[0][1]["audited_max_residual"] <= TOL
    assert not result.fractions.flags.writeable
    assert "not established" in result.qualification


def test_sigma_path_endpoint_matches_direct_solve(model):
    calib, ta, tf, _, target = model
    result = sigma_path(calib, ta, tf, 2., tol=TOL)
    assert result.status == "accepted" and result.last_fraction == 1.
    assert result.parameter == "sigma" and result.parameter_values[0] == 0.
    assert result.parameter_values[-1] == 2. and result.final.metadata["sigma"] == 2.
    assert np.all(np.diff(result.fractions) > 0)
    for stage in result.stages:
        assert stage["audited_max_residual"] <= TOL
        assert stage["requested_tolerance"] == TOL
    np.testing.assert_allclose(result.final.x_sol, target.x_sol, atol=1e-9, rtol=1e-9)
    np.testing.assert_array_equal(result.final.metadata["intermediate_tariff_multipliers"], ta)
    assert result.metadata["sigma_start"] == 0. and result.metadata["sigma_target"] == 2.
    assert result.metadata["continued"] == ["sigma"] and result.metadata["solver"] == "solve_policy_equilibrium"
    assert result.metadata["allow_fallback"] is False and result.metadata["field_tolerance"] == 1e-8
    assert result.metadata["start_source"] == "solved" and result.metadata["start_requested_method"] == "auto"
    for stage in result.stages[1:]:  # every trial is a Newton solve from the previous accepted state
        assert (stage["method"], stage["seed"], stage["fallback_used"]) == ("newton", "warm", False)


def test_resume_from_saved_stage_reproduces_endpoint(model):
    calib, ta, tf, _, _ = model
    saved = []
    first = sigma_path(calib, ta, tf, 2., tol=TOL, keep_states=True,
                       stage_callback=lambda st, info: saved.append((st, info)))
    assert len(first.states) == first.n_stages >= 3
    middle_state, middle = saved[1]
    assert middle["fraction"] == first.fractions[1]
    again = sigma_path(calib, ta, tf, 2., start=middle_state, start_fraction=middle["fraction"], tol=TOL)
    np.testing.assert_allclose(again.final.x_sol, first.final.x_sol, atol=1e-9, rtol=1e-9)
    assert again.fractions[0] == middle["fraction"] and again.start_fraction == middle["fraction"]
    assert again.stages[0]["parameters"] == middle["parameters"]
    assert again.invariant_id == first.invariant_id
    assert again.metadata["start_source"] == "supplied" and again.metadata["start_requested_method"] is None


def test_failure_retains_last_certified_state_not_trial(model):
    calib, ta, tf, base, _ = model
    with pytest.raises(ParameterContinuationFailure, match="unresolved after fraction 0") as failed:
        sigma_path(calib, ta, tf, 2., start=base, tol=TOL, max_iter=1, method="newton",
                   initial_step=.25, min_step=.1)
    # The message and every rejected-trial record name the policy-solver attempt that failed.
    assert "last attempt: newton from the warm seed, converged=False" in str(failed.value)
    result = failed.value.result
    for failure in result.failures:
        (attempt,) = failure["solver_attempts"]
        assert (attempt["method"], attempt["seed"], attempt["solver_converged"]) == ("newton", "warm", False)
        assert attempt["solver_max_residual"] > TOL and attempt["error"]
    np.testing.assert_array_equal(failed.value.state.x_sol, base.x_sol)
    assert failed.value.state is result.final is base
    assert result.status == "unresolved" and result.last_fraction == 0. and result.n_stages == 1
    assert [f["fraction"] for f in result.failures] == [.25, .125]
    assert [f["step"] for f in result.failures] == [.25, .125]
    assert result.failures[0]["parameters"]["sigma"] == .5 != 0.
    assert all(f["exception_type"] == "PolicyEquilibriumError" for f in result.failures)


def test_failure_survives_pickling(model):
    """A path run in a worker process can return its failure and the last audited state."""
    calib, ta, tf, base, _ = model
    with pytest.raises(ParameterContinuationFailure) as failed:
        sigma_path(calib, ta, tf, 2., start=base, tol=TOL, max_iter=1, initial_step=.25, min_step=.1)
    again = pickle.loads(pickle.dumps(failed.value))
    assert type(again) is ParameterContinuationFailure and str(again) == str(failed.value)
    assert again.result.failures == failed.value.result.failures and again.result.status == "unresolved"
    np.testing.assert_array_equal(again.state.x_sol, base.x_sol)


def test_path_that_changes_other_assumptions_raises_before_solving(model, monkeypatch):
    calib, ta, tf, base, _ = model
    monkeypatch.setattr(module, "solve_policy_equilibrium",
                        lambda *a, **k: pytest.fail("solver must not be called"))

    def drifting(f):
        return dict(tau=ta*(1+.1*f), tau_fd=tf, sigma=2.*f, tol=TOL)
    with pytest.raises(ValueError, match="other model assumptions"):
        continue_parameter(calib, drifting, base, continued="sigma")

    def drifting_tolerance(f):
        return dict(tau=ta, tau_fd=tf, sigma=2.*f, tol=TOL*(1+f))
    with pytest.raises(ValueError, match="other model assumptions"):
        continue_parameter(calib, drifting_tolerance, base, continued="sigma")


def test_solver_ignoring_requested_parameter_is_a_contract_error(model):
    calib, ta, tf, base, _ = model
    with pytest.raises(ValueError, match="requested sigma"):
        continue_parameter(lambda kw, x0: base, lambda f: dict(tau=ta, tau_fd=tf, sigma=2.*f, tol=TOL),
                           base, calib=calib)
    with pytest.raises(ValueError, match="requested policy"):
        continue_parameter(lambda kw, x0: base, lambda f: dict(tau=ta*(1+f), tau_fd=tf, sigma=0., tol=TOL),
                           base, calib=calib, continued="tau")
    other_closure = replace(base, metadata={**base.metadata, "fiscal_closure": "baseline"})
    with pytest.raises(ValueError, match="lump-sum"):
        continue_parameter(lambda kw, x0: other_closure,
                           lambda f: dict(tau=ta, tau_fd=tf, sigma=0., tol=TOL, level=f),
                           base, calib=calib, continued="level")


def test_resumed_start_is_audited_at_its_actual_fraction(model):
    calib, ta, tf, base, _ = model
    with pytest.raises(ValueError, match="Start state is not an audited equilibrium at start_fraction 0.5"):
        sigma_path(calib, ta, tf, 2., start=base, start_fraction=.5, tol=TOL)
    stale = replace(base, x_sol=base.x_sol+1e-3)
    with pytest.raises(ValueError, match="Start state is not an audited equilibrium"):
        sigma_path(calib, ta, tf, 2., start=stale, tol=TOL)


@pytest.mark.parametrize("arguments", [
    {"start_fraction": -1.}, {"start_fraction": float("nan")}, {"start_fraction": 1.5},
    {"initial_step": 0.}, {"initial_step": .1, "min_step": .2}, {"max_step": 1.5},
    {"min_step": .3, "initial_step": .3, "max_step": .2}, {"growth": .5},
    {"fast_iterations": -1}, {"fast_iterations": 1.5}, {"fast_iterations": True},
    {"sigma_target": -.1}, {"sigma_start": float("inf")}, {"sigma": 1.},
    {"method": "auto"}, {"method": "hybr"}, {"method": "keller_pac"}, {"x0": "base"},
    {"metadata": {"audit": "none"}}, {"metadata": {"sigma_target": 7.}}, {"metadata": {"solver_calls": 0}},
])
def test_invalid_settings_fail_before_iteration(model, monkeypatch, arguments):
    calib, ta, tf, base, _ = model
    monkeypatch.setattr(module, "solve_policy_equilibrium",
                        lambda *a, **k: pytest.fail("solver must not be called"))
    options = {"sigma_target": 2., "start": base, "tol": TOL, **arguments}
    if isinstance(options.get("x0"), str):
        options["x0"] = base.x_sol  # a seed together with a supplied start is ambiguous
    target = options.pop("sigma_target")
    with pytest.raises(ValueError):
        sigma_path(calib, ta, tf, target, **options)


def test_input_type_contract(model):
    calib, ta, tf, base, _ = model
    with pytest.raises(TypeError, match="TradeCalibrationResult"):
        continue_parameter(lambda kw, x0: base, lambda f: dict(sigma=0.), base)
    with pytest.raises(TypeError, match="start must be"):
        continue_parameter(calib, lambda f: dict(tau=ta, tau_fd=tf, sigma=0., tol=TOL), base.x_sol)
    with pytest.raises(TypeError, match="callable"):
        continue_parameter(calib, {"sigma": 0.}, base)
    with pytest.raises(TypeError, match="mapping"):
        continue_parameter(calib, lambda f: [("sigma", 0.)], base)
    with pytest.raises(ValueError, match="x0"):
        continue_parameter(calib, lambda f: dict(tau=ta, tau_fd=tf, sigma=0., tol=TOL, x0=base.x_sol), base)
    with pytest.raises(ValueError, match="does not return"):
        continue_parameter(calib, lambda f: dict(tau=ta, tau_fd=tf, sigma=0., tol=TOL), base, continued="rho")
    with pytest.raises(ValueError, match="calib must be"):
        continue_parameter(calib, lambda f: dict(sigma=0.), base, calib=synthetic_calibration(3))
    with pytest.raises(TypeError, match="must return a TradeEquilibriumResult"):
        continue_parameter(lambda kw, x0: base.x_sol, lambda f: dict(tau=ta, tau_fd=tf, sigma=2.*f, tol=TOL),
                           base, calib=calib)
    with pytest.raises(TypeError, match="allow_fallback must be a bool"):
        sigma_path(calib, ta, tf, 2., start=base, tol=TOL, allow_fallback="yes")
    with pytest.raises(TypeError, match="allow_fallback must be a bool"):
        try_starts(calib, dict(tau=ta, tau_fd=tf, sigma=2.), [{"name": "a", "x0": base}], allow_fallback=1)
    with pytest.raises(TypeError, match="metadata must be a mapping"):
        continue_parameter(calib, lambda f: dict(tau=ta, tau_fd=tf, sigma=0., tol=TOL), base, metadata=[("a", 1)])


def test_non_consistent_accounting_states_are_rejected(model):
    calib, ta, tf, base, _ = model
    other = replace(base, metadata={**base.metadata, "accounting": "other"})
    with pytest.raises(ValueError, match="consistent"):
        continue_parameter(calib, lambda f: dict(tau=ta, tau_fd=tf, sigma=0., tol=TOL), other)
    # _checked_state would raise NotImplementedError about welfare attribution; the
    # continuation names its own contract instead.
    other_closure = replace(base, metadata={**base.metadata, "fiscal_closure": "baseline"})
    with pytest.raises(ValueError, match="Start state is not an audited equilibrium.*lump-sum"):
        continue_parameter(calib, lambda f: dict(tau=ta, tau_fd=tf, sigma=0., tol=TOL), other_closure)


def test_state_missing_the_requested_tolerance_is_a_rejected_trial(model):
    """A state that satisfies its own recorded tolerance but not the requested one is rejected."""
    calib, ta, tf, base, target = model
    loose = solve_policy_equilibrium(calib, ta, tf, sigma=2., tol=1e-4, method="newton", x0=base.x_sol)
    strict = solve_policy_equilibrium(calib, ta, tf, sigma=2., tol=1e-12)
    assert audited_residual(loose, calib) > 1e-12 >= audited_residual(strict, calib)

    def solver(kwargs, x0):
        return loose
    with pytest.raises(ParameterContinuationFailure) as failed:
        continue_parameter(solver, lambda f: dict(tau=ta, tau_fd=tf, sigma=2., tol=1e-12, level=f),
                           strict, calib=calib, continued="level", initial_step=.5, min_step=.5)
    failure = failed.value.result.failures[0]
    assert "exceeds the requested tolerance" in failure["error"]
    assert failed.value.result.status == "unresolved" and failed.value.state is strict


# ---------------------------------------------------------------------------
# Multistart at one exact target
# ---------------------------------------------------------------------------
def test_try_starts_returns_first_audited_root_and_records_all_attempts(model):
    calib, ta, tf, base, target = model
    events = []
    result = try_starts(calib, dict(tau=ta, tau_fd=tf, sigma=2., tol=TOL), [
        {"name": "bad", "x0": np.full_like(target.x_sol, np.nan)},
        {"name": "certified", "x0": target, "provenance": {"sha256": "fixture"}},
        {"name": "unused", "x0": None}], max_iter=35, progress=events.append)
    assert isinstance(result, DirectTargetResult)
    assert result.status == "accepted" and result.accepted_start == "certified"
    assert [a["status"] for a in result.attempts] == ["unresolved", "accepted"]
    assert result.attempts[0]["error"].startswith("Start vector must be")
    assert result.attempts[1]["provenance"] == {"sha256": "fixture"}
    assert result.attempts[1]["max_iter"] == 35 and result.attempts[1]["iterations"] == 0
    assert result.attempts[1]["seed"] == "warm" and result.attempts[1]["fallback_used"] is False
    np.testing.assert_array_equal(result.final.x_sol, target.x_sol)
    assert result.target["sigma"] == 2. and result.target["max_iter"] == 35
    assert set(result.target["tau"]) == {"shape", "sha256"}
    assert [e["name"] for e in events] == ["bad", "certified"]
    frame = result.to_dataframe()
    assert list(frame.index) == ["bad", "certified"] and frame.loc["certified", "status"] == "accepted"
    assert frame.loc["certified", "seed"] == "warm"
    assert "certified" in result.to_markdown() and "\\begin{tabular}" in result.to_latex()
    assert "#table" in result.to_typst()
    json.dumps(result.attempts)


def test_try_starts_never_returns_uncertified_state(model):
    calib, ta, tf, base, _ = model
    result = try_starts(calib, dict(tau=ta, tau_fd=tf, sigma=2., tol=TOL, method="newton"),
                        [{"name": "wrong_sigma", "x0": base}], max_iter=1)
    assert result.final is None and result.accepted_start is None and result.status == "unresolved"
    assert [a["status"] for a in result.attempts] == ["unresolved"]
    assert result.attempts[0]["exception_type"] == "PolicyEquilibriumError"
    (attempt,) = result.attempts[0]["solver_attempts"]
    assert (attempt["method"], attempt["seed"], attempt["solver_converged"]) == ("newton", "warm", False)
    with pytest.raises(ValueError, match="name and x0"):
        try_starts(calib, dict(tau=ta, tau_fd=tf, sigma=2.), [{"x0": base}])
    with pytest.raises(ValueError, match="max_iter"):
        try_starts(calib, dict(tau=ta, tau_fd=tf, sigma=2.), [{"name": "a", "x0": base}], max_iter=0)
    with pytest.raises(ValueError, match="requested sigma"):
        try_starts(lambda kw, x0: base, dict(tau=ta, tau_fd=tf, sigma=2., tol=TOL),
                   [{"name": "a", "x0": base}], calib=calib)


def test_try_starts_records_wrong_lengths_and_keeps_provenance_verbatim(model):
    """A start of the wrong length is unresolved without a solve; the search goes on."""
    calib, ta, tf, _, target = model
    result = try_starts(calib, dict(tau=ta, tau_fd=tf, sigma=2., tol=TOL), [
        {"name": "short", "x0": target.x_sol[:-1], "provenance": {"ids": [1, 2, 3]}},
        {"name": "certified", "x0": target, "provenance": {"ids": [4, 5], "note": "stage 3"}}])
    assert [a["status"] for a in result.attempts] == ["unresolved", "accepted"]
    assert result.attempts[0]["error"].startswith(f"Start vector has {target.x_sol.size - 1} entries")
    assert result.attempts[0]["provenance"] == {"ids": [1, 2, 3]}
    assert result.attempts[0]["solver_attempts"] is None and "iterations" not in result.attempts[0]
    assert result.attempts[1]["provenance"] == {"ids": [4, 5], "note": "stage 3"}
    assert result.accepted_start == "certified"


# ---------------------------------------------------------------------------
# Warm starts and the allow_fallback opt-in
# ---------------------------------------------------------------------------
def test_non_newton_methods_require_allow_fallback(model, monkeypatch):
    """auto, hybr and keller_pac ignore the warm start, so they are refused before any solve."""
    calib, ta, tf, base, _ = model
    monkeypatch.setattr(module, "solve_policy_equilibrium",
                        lambda *a, **k: pytest.fail("solver must not be called"))
    for method in ("auto", "hybr", "keller_pac"):
        with pytest.raises(ValueError, match="allow_fallback=True"):
            sigma_path(calib, ta, tf, 2., start=base, tol=TOL, method=method)
        with pytest.raises(ValueError, match="allow_fallback=True"):
            continue_parameter(calib, lambda f: dict(tau=ta, tau_fd=tf, sigma=2.*f, tol=TOL, method=method), base)
        with pytest.raises(ValueError, match="does not start from the warm start"):
            try_starts(calib, dict(tau=ta, tau_fd=tf, sigma=2., tol=TOL, method=method), [{"name": "a", "x0": base}])


def test_try_starts_with_allow_fallback_records_the_seed_it_actually_used(model):
    """With allow_fallback=True a Keller root is credited to the named start, but its seed says so."""
    calib, ta, tf, base, _ = model
    result = try_starts(calib, dict(tau=ta, tau_fd=tf, sigma=2., tol=TOL, method="keller_pac"),
                        [{"name": "garbage", "x0": base.x_sol + 5.}], allow_fallback=True)
    assert result.status == "accepted" and result.accepted_start == "garbage"
    attempt = result.attempts[0]
    assert (attempt["method"], attempt["seed"]) == ("keller_pac", "calibrated")
    assert attempt["audited_max_residual"] <= TOL and result.metadata["allow_fallback"] is True


def test_fallback_and_cold_seed_states_are_rejected_unless_allowed(model, replay):
    calib, ta, tf, _, _ = model
    ladder = replace(replay, metadata={**replay.metadata, "policy_solver_fallback_used": True})
    attempts = [dict(a) for a in replay.metadata["policy_solver_attempts"]]
    attempts[-1]["seed"] = "calibrated"
    cold = replace(replay, metadata={**replay.metadata, "policy_solver_attempts": attempts})

    def path(f):
        return dict(tau=ta, tau_fd=tf, sigma=2., tol=TOL, level=f)
    target = dict(tau=ta, tau_fd=tf, sigma=2., tol=TOL)
    for state, message in ((ladder, "fallback ladder"), (cold, "calibrated seed, not from the supplied start")):
        def solver(kwargs, x0, state=state):
            return state
        with pytest.raises(ParameterContinuationFailure) as failed:
            continue_parameter(solver, path, replay, calib=calib, continued="level", initial_step=.5, min_step=.5)
        assert message in failed.value.result.failures[0]["error"]
        assert failed.value.result.metadata["allow_fallback"] is False and failed.value.state is replay
        allowed = continue_parameter(solver, path, replay, calib=calib, continued="level", allow_fallback=True)
        assert allowed.status == "accepted" and allowed.metadata["allow_fallback"] is True
        assert allowed.stages[-1]["fallback_used"] is (state is ladder)
        assert allowed.stages[-1]["seed"] == ("calibrated" if state is cold else "warm")
        direct = try_starts(solver, target, [{"name": "s", "x0": replay}], calib=calib)
        assert direct.status == "unresolved" and message in direct.attempts[0]["error"]
        assert try_starts(solver, target, [{"name": "s", "x0": replay}], calib=calib,
                          allow_fallback=True).status == "accepted"
    # A start without a vector is the calibrated state, so a calibrated seed is its own start.
    direct = try_starts(lambda kw, x0: cold, target, [{"name": "calibrated", "x0": None}], calib=calib)
    assert direct.status == "accepted" and direct.attempts[0]["seed"] == "calibrated"


def test_identity_sigma_path_with_supplied_start_does_not_solve(model, monkeypatch):
    calib, ta, tf, _, target = model
    monkeypatch.setattr(module, "solve_policy_equilibrium",
                        lambda *a, **k: pytest.fail("solver must not be called"))
    result = sigma_path(calib, ta, tf, 2., sigma_start=2., start=target, tol=TOL,
                        metadata={"policy": "10% import tariff by country 0"})
    assert result.status == "accepted" and result.parameter == "" and result.last_fraction == 1.
    assert result.metadata["solver_calls"] == 0 and result.metadata["start_source"] == "supplied"
    assert result.metadata["policy"] == "10% import tariff by country 0"
    assert result.metadata["sigma_target"] == 2. and result.final is target


def test_reserved_metadata_keys_are_refused(model, monkeypatch):
    """User metadata cannot overwrite what the result records about its own audit and target."""
    calib, ta, tf, base, _ = model
    monkeypatch.setattr(module, "solve_policy_equilibrium",
                        lambda *a, **k: pytest.fail("solver must not be called"))
    with pytest.raises(ValueError, match="reserved"):
        continue_parameter(calib, lambda f: dict(tau=ta, tau_fd=tf, sigma=0., tol=TOL), base,
                           metadata={"audit": "none"})
    with pytest.raises(ValueError, match="reserved"):
        sigma_path(calib, ta, tf, 2., start=base, tol=TOL, metadata={"sigma_target": 7.})
    with pytest.raises(ValueError, match="reserved"):
        try_starts(calib, dict(tau=ta, tau_fd=tf, sigma=2., tol=TOL), [{"name": "a", "x0": base}],
                   metadata={"max_iter": 1})


# ---------------------------------------------------------------------------
# Step law (scripted solver)
# ---------------------------------------------------------------------------
def scripted_continuation(calib, ta, tf, target, outcomes, **settings):
    """Drive continue_parameter with a solver whose outcome depends only on the trial fraction.

    ``outcomes`` maps a rounded fraction to an iteration count or ``"fail"``; the
    default outcome is one iteration. The dummy parameter ``level`` is continued
    while the model stays fixed, so every returned state is a genuine audited
    equilibrium and the step law alone determines the accepted fractions.
    """
    calls = []

    def solver(kwargs, x0):
        level = round(float(kwargs["level"]), 9)
        calls.append(level)
        rule = outcomes.get(level, 1)
        if rule == "fail":
            raise RuntimeError(f"Newton iteration limit at fraction {level}")
        return replace(target, iterations=int(rule))

    def path(f):
        return dict(tau=ta, tau_fd=tf, sigma=2., tol=TOL, level=f)
    outcome = None
    try:
        result = continue_parameter(solver, path, replace(target, iterations=0), calib=calib,
                                    continued="level", **settings)
    except ParameterContinuationFailure as exc:
        outcome = exc
        result = exc.result
    return result, calls, outcome


def scenario_table():
    """Scripted scenarios: outcome by trial fraction and step-law settings."""
    grid = [round(k/64, 9) for k in range(65)]
    return {
        "fast_growth": ({}, dict(initial_step=.2)),
        "no_growth": ({round(f, 9): 7 for f in (.2, .4, .6, .8, 1.)}, dict(initial_step=.2)),
        "two_failures": ({.2: "fail", .1: "fail"}, dict(initial_step=.2)),
        "exhaustion": ({f: "fail" for f in grid}, dict(initial_step=.25, min_step=.1)),
        "resume": ({.75: "fail"}, dict(initial_step=.25, start_fraction=.5)),
        "fast_every_stage": ({.2: 2, .5: 2, .95: 2, 1.: 2}, dict(initial_step=.2)),
        # A stage at exactly fast_iterations (6) grows the step; 7 does not.
        "fast_threshold": ({.2: 6, .5: 7}, dict(initial_step=.2)),
        # Growth from .4 is capped at .5, and the capped step is the one recorded by the failure.
        "cap_binds": ({.9: "fail"}, dict(initial_step=.4)),
    }


# self-generated by puremacro 5.1.0 on the scripted scenarios of scenario_table(); not an external reference
# (each row also follows by hand from the step law: halve after a failure, grow by 1.5 after a fast stage,
# cap at max_step=0.5).
EXPECTED = {
    "fast_growth": ([0., .2, .5, .95, 1.], []),
    "no_growth": ([0., .2, .4, .6, .8, 1.], []),
    "two_failures": ([0., .05, .125, .2375, .40625, .659375, 1.], [(.2, 0., .2), (.1, 0., .1)]),
    "resume": ([.5, .625, .8125, 1.], [(.75, .5, .25)]),
    "fast_every_stage": ([0., .2, .5, .95, 1.], []),
    "fast_threshold": ([0., .2, .5, .8, 1.], []),
    "cap_binds": ([0., .4, .65, 1.], [(.9, .4, .5)]),
}


def test_step_law_halves_on_failure_and_grows_on_fast_convergence(model, replay):
    calib, ta, tf, _, _ = model
    target = replay
    table = scenario_table()
    for name, (fractions, failures) in EXPECTED.items():
        outcomes, settings = table[name]
        result, calls, outcome = scripted_continuation(calib, ta, tf, target, outcomes, **settings)
        assert outcome is None and result.status == "accepted", name
        np.testing.assert_allclose(result.fractions, fractions, atol=1e-12, err_msg=name)
        assert [(f["fraction"], f["previous_fraction"], f["step"]) for f in result.failures] == failures, name
        # The recorded step is the proposed step; only the last trial is clipped at one.
        increments = np.diff(fractions).tolist()
        steps = [s["step"] for s in result.stages][1:]
        assert steps[:-1] == pytest.approx(increments[:-1]) and steps[-1] >= increments[-1], name
    outcomes, settings = table["exhaustion"]
    result, calls, outcome = scripted_continuation(calib, ta, tf, target, outcomes, **settings)
    assert isinstance(outcome, ParameterContinuationFailure) and result.status == "unresolved"
    assert result.last_fraction == 0. and result.n_stages == 1
    assert [(f["fraction"], f["step"]) for f in result.failures] == [(.25, .25), (.125, .125)]
    np.testing.assert_array_equal(outcome.state.x_sol, target.x_sol)
    # Growth is capped at max_step and only follows fast stages.
    result, _, _ = scripted_continuation(calib, ta, tf, target, {}, initial_step=.2, max_step=.25)
    np.testing.assert_allclose(result.fractions, [0., .2, .45, .7, .95, 1.])
    result, _, _ = scripted_continuation(calib, ta, tf, target, {.2: 3}, initial_step=.2, fast_iterations=2)
    np.testing.assert_allclose(result.fractions, [0., .2, .4, .7, 1.])


def test_try_starts_stops_at_the_first_accepted_start(model, replay):
    """Two rejected starts, an accepted one and one never tried; then an unresolved search."""
    calib, ta, tf, _, _ = model
    target = replay
    outcomes = {0.: "fail", 1.: "fail", 2.: 3, 3.: 2}

    def direct_solver(kwargs, x0):
        rule = outcomes[float(x0[0])]
        if rule == "fail":
            raise RuntimeError("Newton iteration limit")
        return replace(target, iterations=int(rule))
    starts = [{"name": "bad", "x0": np.r_[0., target.x_sol[1:]]},
              {"name": "worse", "x0": np.r_[1., target.x_sol[1:]]},
              {"name": "good", "x0": np.r_[2., target.x_sol[1:]], "provenance": {"sha256": "x"}},
              {"name": "unused", "x0": np.r_[3., target.x_sol[1:]]}]
    found = try_starts(direct_solver, dict(tau=ta, tau_fd=tf, sigma=2., tol=TOL), starts, calib=calib, max_iter=7)
    assert [a["status"] for a in found.attempts] == ["unresolved", "unresolved", "accepted"]
    assert found.accepted_start == "good" and found.status == "accepted"
    assert [a["max_iter"] for a in found.attempts] == [7, 7, 7]
    unresolved = try_starts(direct_solver, dict(tau=ta, tau_fd=tf, sigma=2., tol=TOL), starts[:1], calib=calib,
                            max_iter=7)
    assert [a["status"] for a in unresolved.attempts] == ["unresolved"]
    assert unresolved.accepted_start is None and unresolved.final is None
    assert unresolved.status == "unresolved"


# ---------------------------------------------------------------------------
# Records, rendering, plotting and callbacks
# ---------------------------------------------------------------------------
def test_records_are_json_serializable_and_render(model):
    calib, ta, tf, _, _ = model
    result = sigma_path(calib, ta, tf, 2., tol=TOL)
    json.dumps(result.stages); json.dumps(result.failures); json.dumps(result.metadata)
    frame = result.to_dataframe()
    assert list(frame.columns[:3]) == ["fraction", "sigma", "step"]
    assert frame.index.name == "stage" and len(frame) == result.n_stages
    assert not frame.isna().any().any()
    markdown = result.to_markdown()
    assert "nan" not in markdown.lower() and "audited_max_residual" in markdown
    assert "\\begin{tabular}" in result.to_latex() and "#table" in result.to_typst()
    summary = result.summary()
    assert "status=accepted" in summary and "not established" in summary
    for stage in result.stages:
        assert set(stage) >= {"fraction", "previous_fraction", "step", "parameters", "method", "seed",
                              "fallback_used", "iterations", "solver_max_residual", "audited_max_residual",
                              "requested_tolerance", "seconds", "minimum_price", "minimum_output_ratio",
                              "minimum_wage", "minimum_final_expenditure"}
        assert stage["minimum_final_expenditure"] > 0 and stage["minimum_price"] > 0


def test_plot_variants_never_show(model, monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda *a, **k: pytest.fail("plot must not call plt.show"))
    calib, ta, tf, _, _ = model
    result = sigma_path(calib, ta, tf, 2., tol=TOL, keep_states=True)
    _, ax = plt.subplots()
    assert result.plot(ax=ax) is ax and ax.get_yscale() == "log" and ax.get_xlabel() == "sigma"
    result.plot("minimum_price", ax=ax, label="price")
    result.plot(lambda state: float(state.gdp[0]), ax=ax)
    result.plot(np.arange(result.n_stages), ax=ax)
    assert len(ax.lines) == 4
    assert result.plot() is not ax
    with pytest.raises(ValueError, match="one number per accepted stage"):
        result.plot([1., 2.], ax=ax)
    bare = sigma_path(calib, ta, tf, 2., tol=TOL)
    with pytest.raises(ValueError, match="keep_states"):
        bare.plot(lambda state: 1., ax=ax)
    plt.close("all")


def test_progress_and_stage_callback_events(model):
    calib, ta, tf, base, _ = model
    events, stages = [], []
    with pytest.raises(ParameterContinuationFailure):
        sigma_path(calib, ta, tf, 2., start=base, tol=TOL, max_iter=1, method="newton",
                   initial_step=.25, min_step=.1, progress=events.append,
                   stage_callback=lambda st, info: stages.append(info))
    assert [e["event"] for e in events] == ["accepted", "trial", "rejected", "trial", "rejected"]
    assert events[1]["fraction"] == .25 and events[1]["parameters"] == {"sigma": .5}
    assert events[2]["step"] == .25 and events[3]["step"] == .125
    assert len(stages) == 1 and stages[0]["fraction"] == 0.
    # Callbacks and progress events receive deep copies; the result keeps its own records.
    seen, events = [], []
    result = sigma_path(calib, ta, tf, 2., tol=TOL, stage_callback=lambda st, info: seen.append(info),
                        progress=events.append)
    seen[0]["fraction"] = 99.
    seen[0]["parameters"]["sigma"] = 99.
    for event in events:
        if "parameters" in event:
            event["parameters"]["sigma"] = -1.
    assert result.stages[0]["fraction"] == 0. and result.stages[0]["parameters"] == {"sigma": 0.}
    assert result.stages[-1]["parameters"] == {"sigma": 2.} and result.parameter_values[0] == 0.


# ---------------------------------------------------------------------------
# OECD 3-region x 3-sector fixture
# ---------------------------------------------------------------------------
def oecd_fixture_calibration():
    from _trade_oecd_fixture import load_oecd_3x3
    return package_mrio_to_calibration_result(condense_final_demand(load_oecd_3x3()))


def usa_tariffs(calib, rate=.1):
    """Uniform import tariff of the fixture's USA region on every foreign origin, all uses."""
    return import_tariffs(calib, importer=list(calib.country_codes).index("USA"), rate=rate)


def test_frozen_oecd_fixture_short_path_matches_direct_solve():
    """OECD 3-region x 3-sector fixture in million USD: explicit tol=1e-5.

    The start comes from the audited ladder (``start_source="solved"``,
    requested method ``"auto"``). Since the consistent-mode stopping rule tests
    the audited physical equations, its first (Newton) attempt is accepted at
    sigma=0 (before that fix Newton stopped at 8.3e-6 with converged=False and
    the start came from a fallback); every later stage is a warm-started Newton
    solve.
    """
    calib = oecd_fixture_calibration()
    tau, tau_fd = usa_tariffs(calib)
    result = sigma_path(calib, tau, tau_fd, .5, tol=1e-5)
    direct = solve_policy_equilibrium(calib, tau, tau_fd, sigma=.5, tol=1e-5)
    assert result.status == "accepted" and result.metadata["unit"] == "M_USD"
    assert result.metadata["start_source"] == "solved" and result.metadata["start_requested_method"] == "auto"
    assert result.stages[0]["method"] in ("newton", "hybr", "keller_pac")
    assert result.n_stages >= 2
    for stage in result.stages[1:]:
        assert (stage["method"], stage["seed"], stage["fallback_used"]) == ("newton", "warm", False)
    assert all(s["audited_max_residual"] <= 1e-5 for s in result.stages)
    np.testing.assert_allclose(result.final.p_sol, direct.p_sol, rtol=1e-6, atol=0)
    np.testing.assert_allclose(result.final.gdp, direct.gdp, rtol=1e-6, atol=0)


@pytest.mark.slow
def test_frozen_oecd_fixture_unit_elasticity_region_is_reported_not_hidden():
    """On this fixture every solver can fail near sigma=1 under the USA tariff.

    The path either completes with audited stages or stops with a certified
    state and recorded failures; it never returns an unaudited endpoint.
    """
    calib = oecd_fixture_calibration()
    tau, tau_fd = usa_tariffs(calib)
    try:
        result = sigma_path(calib, tau, tau_fd, 2., tol=1e-5)
    except ParameterContinuationFailure as exc:
        result = exc.result
        assert result.status == "unresolved" and 0 < result.last_fraction < 1
        assert result.failures and result.final.metadata["sigma"] == result.parameter_values[-1]
        assert audited_residual(exc.state, calib) <= 1e-5
    else:
        assert result.status == "accepted"
    assert all(s["audited_max_residual"] <= 1e-5 for s in result.stages)

# ---------------------------------------------------------------------------
# Fold continuation helpers (renamed from the 4.x names)
# ---------------------------------------------------------------------------
def test_fold_continuation_renamed():
    from puremacro.trade import continuation, solver

    assert hasattr(continuation, "solve_fold_continuation")
    assert hasattr(continuation, "solve_fold_continuation_step")
    assert continuation.solve_fold_continuation_step is solver.solve_fold_continuation_step
    assert "solve_fold_continuation" in continuation.__all__
    assert "solve_fold_continuation_step" in continuation.__all__


def _fold_system(lam, nc, rng_matrix, stiff):
    """Factor-price matrix whose Schur complement in ``stiff`` changes sign at lam = 0.5.

    ``S(lam) = B + (lam - 0.5 - s0) e_c e_c^T`` with ``s0`` the Schur complement of
    ``B`` in coordinate ``c``, so the scalar equation of the stiff coordinate has
    slope ``lam - 0.5``: the linearised system is singular at lam = 0.5.
    """
    B = rng_matrix.copy()
    rest = [i for i in range(nc) if i != stiff]
    s0 = B[stiff, stiff] - B[stiff, rest] @ np.linalg.solve(B[np.ix_(rest, rest)], B[rest, stiff])
    S = B.copy()
    S[stiff, stiff] += (lam - 0.5) - s0
    return S


def test_fold_continuation_on_synthetic():
    """A warm-started continuation through a fold stays on its branch.

    ``F(w, lam) = S(lam) (w - w*(lam)) + (w_c - w*_c(lam))^3 e_c`` has the smooth
    solution branch ``w*(lam) = w0 + lam a``; its Jacobian on the branch is
    ``S(lam)``, singular in the stiff coordinate at lam = 0.5 (a grid point), and
    for lam < 0.5 the stiff equation has two more roots ``+-sqrt(0.5 - lam)``
    that a jumping iteration would land on. Each lam is solved by Newton steps
    from :func:`solve_fold_continuation`, warm-started from the previous state;
    the cold start at lam = 0 is far from the branch, so the clamp engages there.
    """
    from puremacro.trade.continuation import solve_fold_continuation
    from puremacro.trade.data import synthetic_codes

    codes, _sectors = synthetic_codes(5, 4)
    nc = len(codes)
    rng = np.random.default_rng(0)
    B = np.eye(nc) * 3.0 + 0.3 * rng.standard_normal((nc, nc))
    stiff = 2
    w0 = 0.5 + 0.1 * rng.standard_normal(nc)
    a = 0.2 * rng.standard_normal(nc)

    def residual(w, lam):
        e = w - (w0 + lam * a)
        out = _fold_system(lam, nc, B, stiff) @ e
        out[stiff] += e[stiff] ** 3
        return out

    def jacobian(w, lam):
        e = w - (w0 + lam * a)
        J = _fold_system(lam, nc, B, stiff)
        J[stiff, stiff] += 3.0 * e[stiff] ** 2
        return J

    lams = np.linspace(0.0, 1.0, 41)  # lam = 0.5 is a grid point: the fold itself
    assert lams[20] == 0.5
    w = np.zeros(nc)  # cold start: max|w0| > max_disp
    states, clamped_at = [], []
    for lam in lams:
        for _ in range(50):
            r = residual(w, lam)
            if np.max(np.abs(r)) <= 1e-12:
                break
            dw, _res, _conv, info = solve_fold_continuation(
                jacobian(w, lam), -r, idx_stiff=codes[stiff], country_codes=codes,
                max_disp=0.30, return_info=True)
            direction = info["unclamped_step"]
            assert np.all(np.isfinite(dw)) and np.max(np.abs(dw)) <= 0.30 + 1e-12
            # A clamp shortens the Newton direction; it never turns or flips it.
            np.testing.assert_allclose(dw, info["clamp_scale"] * direction, rtol=1e-12, atol=1e-15)
            assert np.sign(dw[stiff]) == np.sign(direction[stiff])
            if info["clamped"]:
                clamped_at.append(lam)
            w = w + dw
        assert np.max(np.abs(residual(w, lam))) <= 1e-12, lam
        states.append(w.copy())

    states = np.asarray(states)
    # The clamp was exercised (cold start) and is idle on the warm-started path.
    assert clamped_at and set(clamped_at) == {0.0}
    # Every state is on the branch w*(lam), the fold included (where the singular
    # Jacobian makes Newton linear, so the tolerance there is looser): no jump to
    # the roots +-sqrt(0.5 - lam) of the stiff equation.
    errors = np.max(np.abs(states - (w0 + lams[:, None] * a)), axis=1)
    assert np.all(np.delete(errors, 20) <= 1e-9)
    assert errors[20] <= 1e-3
    # The branch is continuous: no step between consecutive states, the stiff
    # coordinate included, exceeds 10x the median step.
    steps = np.max(np.abs(np.diff(states, axis=0)), axis=1)
    assert steps.max() < 10.0 * np.median(steps)
    stiff_steps = np.abs(np.diff(states[:, stiff]))
    assert stiff_steps.max() < 10.0 * np.median(stiff_steps)
