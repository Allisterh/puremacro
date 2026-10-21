"""Every course model under ``curso/notebooks/modelos`` solves as its header says.

The MAV course ships seven Dynare ``.mod`` files whose headers invite the reader
to run them directly with :func:`puremacro.dsge.load_mod`. Until this module
nothing in the suite loaded any of them, and ``rbc_canonico.mod`` shipped with
``mu = 0.822516`` under a comment claiming it was calibrated for ``l_ss = 0.33``.
The intratemporal condition ``mu * (1 - l)^(-nu) * c = w`` evaluated at the
``l = 0.33`` steady state in the file's own ``initval`` block gives
``mu = 1.455567``; with the shipped value ``steady;`` walked away from the
``initval`` to ``l_ss = 0.443`` and the one-standard-deviation TFP impact on
hours came out at 0.285% instead of the 0.393% the course quotes.

Checked here, with every file read as UTF-8 (the comments carry accents):

1. the glob finds exactly the seven known files;
2. the six determinate models load as a :class:`LinearModel` and run the
   header's advertised ``stoch_simul(order=1, irf=40)``;
3. ``nk_zlb_cons.mod`` is the documented indeterminate ZLB regime: it raises
   :class:`BlanchardKahnError` under ``strict=True`` and loads with
   ``strict=False``, as lesson T02_D does;
4. every ``initval`` block is the solved steady state;
5. ``rbc_canonico.mod`` targets ``l_ss = 0.33``, its ``mu`` is the value the
   intratemporal condition implies, and the hours impact is ~0.393%;
6. the calibration targets the files' comments state are reproduced.

``initval`` values are parsed with regular expressions, never ``eval``: the
course files only use plain numbers and ``log(<number>)``.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

from puremacro.dsge import BlanchardKahnError, LinearModel, load_mod

MODELOS = Path(__file__).resolve().parents[1] / "curso" / "notebooks" / "modelos"

#: Files whose header advertises a clean first-order solve.
DETERMINATE = (
    "merz_andolfatto.mod",
    "neoclasico_prevision_perfecta.mod",
    "new_keynesian_3eq.mod",
    "nk_zlb_ref.mod",
    "rbc_canonico.mod",
    "rbc_mexico_dual.mod",
)

#: The ZLB regime of the two-regime NK pair: indeterminate on its own by design.
INDETERMINATE = "nk_zlb_cons.mod"

#: Files that carry an ``initval;`` block (the two ``nk_zlb_*`` files use a
#: ``steady_state_model;`` block instead).
WITH_INITVAL = (
    "merz_andolfatto.mod",
    "neoclasico_prevision_perfecta.mod",
    "new_keynesian_3eq.mod",
    "rbc_canonico.mod",
    "rbc_mexico_dual.mod",
)

_NUMBER = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
_INITVAL_BLOCK = re.compile(r"^\s*initval\s*;(.*?)^\s*end\s*;", re.M | re.S)
_ASSIGNMENT = re.compile(r"^\s*(\w+)\s*=\s*([^;]+);", re.M)
_PLAIN_NUMBER = re.compile(rf"^{_NUMBER}$")
_LOG_OF_NUMBER = re.compile(rf"^log\(\s*({_NUMBER})\s*\)$")


def _read(name: str) -> str:
    return (MODELOS / name).read_text(encoding="utf-8")


def _parameter(text: str, name: str) -> float:
    """The top-level ``name = <number>;`` assignment of a parameter."""
    match = re.search(rf"^\s*{name}\s*=\s*({_NUMBER})\s*;", text, re.M)
    assert match is not None, f"no numeric assignment of {name!r} in the file"
    return float(match.group(1))


def _initval(text: str) -> dict[str, float]:
    """``name -> value`` from the ``initval; ... end;`` block, without ``eval``.

    Plain numbers and ``log(<number>)`` (``merz_andolfatto.mod``'s ``lth`` and
    ``lpr``) are the only right-hand sides the course files use. Anything else
    fails loudly so a new expression form is handled here, not skipped.
    """
    match = _INITVAL_BLOCK.search(text)
    assert match is not None, "no initval; ... end; block in the file"
    block = re.sub(r"//.*", "", match.group(1))
    values: dict[str, float] = {}
    for name, rhs in _ASSIGNMENT.findall(block):
        rhs = rhs.strip()
        if _PLAIN_NUMBER.match(rhs):
            values[name] = float(rhs)
            continue
        log_arg = _LOG_OF_NUMBER.match(rhs)
        if log_arg is not None:
            values[name] = math.log(float(log_arg.group(1)))
            continue
        raise ValueError(f"unsupported initval expression: {name} = {rhs!r}")
    assert values, "initval block parsed to nothing"
    return values


def test_seven_mod_files_are_discovered():
    found = sorted(p.name for p in MODELOS.glob("*.mod"))
    assert found == sorted(DETERMINATE + (INDETERMINATE,))


@pytest.mark.parametrize("name", DETERMINATE)
def test_mod_file_solves_at_order_one(name):
    """``load_mod`` + ``stoch_simul(order=1, irf=40)``: the header's own recipe."""
    model = load_mod(MODELOS / name)
    assert isinstance(model, LinearModel)
    assert model.is_determinate
    assert np.isfinite(model.steady_state.to_numpy()).all()
    result = model.stoch_simul(order=1, irf=40)
    assert result.order == 1
    assert result.theoretical_moments is not None
    for var in model.variables:
        for shock in model.shocks:
            irf = result.irfs[f"{var}_{shock}"]
            assert len(irf) == 40 + 1  # impact row h = 0 plus 40 periods
            assert np.isfinite(irf.to_numpy()).all()


def test_zlb_constrained_regime_is_indeterminate_by_design():
    """The clamped-rate regime has no unique stable solution on its own.

    Lesson T02_D pairs it with ``nk_zlb_ref.mod`` through OccBin and loads it
    with ``strict=False``; the file's ``steady_state_model`` block gives
    ``pi = r = -r_ss`` and ``y = (1 - beta) pi / kappa``.
    """
    path = MODELOS / INDETERMINATE
    with pytest.raises(BlanchardKahnError):
        load_mod(path)
    model = load_mod(path, strict=False)
    assert isinstance(model, LinearModel)
    assert model.is_determinate is False
    text = _read(INDETERMINATE)
    r_ss = _parameter(text, "r_ss")
    beta = _parameter(text, "beta")
    kappa = _parameter(text, "kappa")
    assert model.steady_state["r"] == pytest.approx(-r_ss)
    assert model.steady_state["pi"] == pytest.approx(-r_ss)
    assert model.steady_state["y"] == pytest.approx((1 - beta) * (-r_ss) / kappa)
    assert model.steady_state["g"] == pytest.approx(0.0)


@pytest.mark.parametrize("name", WITH_INITVAL)
def test_initval_is_the_steady_state(name):
    """``steady;`` must not walk away from the file's own ``initval`` block."""
    initval = _initval(_read(name))
    model = load_mod(MODELOS / name)
    declared = set(model.variables) | set(model.shocks)
    unknown = sorted(set(initval) - declared)
    assert not unknown, f"{name}: initval names not declared: {unknown}"
    for var in model.variables:
        if var in initval:
            assert float(model.steady_state[var]) == pytest.approx(
                initval[var], abs=1e-5
            ), f"{name}: steady state of {var} differs from its initval"


def test_rbc_canonico_targets_one_third_hours():
    """``mu`` is the value the intratemporal condition implies at ``l_ss = 0.33``.

    The RBC lab page and lesson T02_B quote a 0.39% / +0.393% impact of hours
    to a one-standard-deviation TFP innovation; both recalibrate ``mu`` from
    ``l_ss = 0.33`` themselves, so the file must carry the same value.
    """
    text = _read("rbc_canonico.mod")
    mu = _parameter(text, "mu")
    nu = _parameter(text, "nu")
    sigma_a = _parameter(text, "sigma_a")
    model = load_mod(MODELOS / "rbc_canonico.mod")
    assert model.is_determinate
    ss = model.steady_state
    assert float(ss["l"]) == pytest.approx(0.33, abs=1e-6)
    mu_implied = float(ss["w"] / (ss["c"] * (1 - ss["l"]) ** (-nu)))
    assert mu == pytest.approx(mu_implied, abs=1e-6)
    assert mu == pytest.approx(1.455567, abs=1e-6)
    assert model.units["l"] == "level"
    impact = model.irf("e_a", horizon=1, size=sigma_a)["l"].iloc[0]
    hours_impact_pct = 100.0 * float(impact) / float(ss["l"])
    assert hours_impact_pct == pytest.approx(0.393, abs=0.002)


def test_rbc_mexico_dual_hits_its_documented_targets():
    """Header: ``l_i/l_tot = 0.580``, ``k/y_f = 7.52``; known limits 1) and 2)."""
    text = _read("rbc_mexico_dual.mod")
    ss = load_mod(MODELOS / "rbc_mexico_dual.mod").steady_state
    assert float(ss["l_i"] / ss["l_tot"]) == pytest.approx(0.580, abs=1e-3)
    assert float(ss["k"] / ss["y_f"]) == pytest.approx(7.52, abs=0.01)
    assert float(ss["y_i"] / ss["y"]) == pytest.approx(0.532, abs=1e-3)
    assert float(ss["i"] / ss["y"]) == pytest.approx(0.088, abs=1e-3)
    # Average-productivity ratio pinned by equations 7 and 8 whatever A_i_bar is.
    tau = _parameter(text, "tau")
    alpha = _parameter(text, "alpha")
    gamma = _parameter(text, "gamma")
    ratio = float((ss["y_i"] / ss["l_i"]) / (ss["y_f"] / ss["l_f"]))
    assert ratio == pytest.approx((1 - tau) * (1 - alpha) / gamma, abs=1e-6)
    assert ratio == pytest.approx(0.825, abs=1e-3)


def test_merz_andolfatto_hits_its_documented_targets():
    """Comments: ``u_head`` steady state 39.5%, ``urate`` 4.17%, ``x = 0.0666``."""
    ss = load_mod(MODELOS / "merz_andolfatto.mod").steady_state
    assert float(ss["u_head"]) == pytest.approx(0.395, abs=1e-3)
    assert float(ss["urate"]) == pytest.approx(0.0417, abs=1e-4)
    assert float(ss["x"]) == pytest.approx(0.0666, abs=1e-4)
    assert float(ss["u_head"]) == pytest.approx(1 - ss["n"], abs=1e-9)


def test_neoclasico_capital_output_ratio():
    """Header: annual calibration with ``K/Y = 3.6`` (solved ratio 3.629)."""
    ss = load_mod(MODELOS / "neoclasico_prevision_perfecta.mod").steady_state
    assert float(ss["k"] / ss["y"]) == pytest.approx(3.63, abs=0.01)
    assert float(ss["l"]) == pytest.approx(0.461355, abs=1e-5)
