"""The trade model's gallery cases, and the trade package's no-``scipy.io`` contract.

``puremacro.validation.run_all`` discovers ``puremacro/validation/cases_trade.py``
like every other ``cases_*`` module; this file pins the three trade cases and runs
them directly. An AST scan guards the trade package: puremacro reads no MATLAB
files, and no module of ``puremacro.trade`` may import ``scipy.io``.
"""
import ast
from pathlib import Path

import pytest

import puremacro.trade
from puremacro.validation import run, scorecard
from puremacro.validation.cases_trade import CASES

_RESULTS = [run(case) for case in CASES]

TRADE_CASE_IDS = {
    "trade.oecd_isic_aggregation_conserves_native_world_totals",
    "trade.oecd_isic_columns_balance_and_value_added_is_positive",
    "trade.oecd_isic_base_equilibrium_is_the_calibration_point",
}


@pytest.mark.parametrize("res", _RESULTS, ids=[r.id for r in _RESULTS])
def test_case_passes(res):
    assert res.passed, f"{res.id} failed (margin={res.max_margin}); error={res.error}"


def test_the_three_trade_cases_are_present_and_registered_in_the_gallery():
    assert {case.id for case in CASES} == TRADE_CASE_IDS
    assert {case.subsystem for case in CASES} == {"Trade"}
    assert TRADE_CASE_IDS <= set(scorecard()["id"])


def _scipy_io_imports(tree: ast.AST) -> list[int]:
    """Line numbers of every import of ``scipy.io`` (static or through importlib) in ``tree``."""
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name == "scipy.io" or a.name.startswith("scipy.io.") for a in node.names):
                lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == "scipy.io" or node.module.startswith("scipy.io."):
                lines.append(node.lineno)
            elif node.module == "scipy" and any(a.name == "io" for a in node.names):
                lines.append(node.lineno)
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            target = node.args[0].value
            if name in ("import_module", "__import__") and isinstance(target, str) and (
                    target == "scipy.io" or target.startswith("scipy.io.")):
                lines.append(node.lineno)
    return lines


@pytest.mark.mechanism_control
@pytest.mark.parametrize("source, found", [
    ("import scipy.io", True),
    ("import scipy.io as sio", True),
    ("from scipy import io", True),
    ("from scipy.io import loadmat", True),
    ("from scipy.io.matlab import loadmat", True),
    ("import importlib\nimportlib.import_module('scipy.io')", True),
    ("import scipy.linalg", False),
    ("from scipy import optimize, sparse", False),
    ("from . import io", False),
])
def test_scanner_finds_every_form_of_scipy_io_import(source, found):
    assert bool(_scipy_io_imports(ast.parse(source))) is found


def test_no_trade_module_imports_scipy_io():
    package = Path(puremacro.trade.__file__).resolve().parent
    sources = sorted(package.rglob("*.py"))
    assert len(sources) >= 45, f"only {len(sources)} modules found under {package}"
    offenders = {}
    for path in sources:
        hits = _scipy_io_imports(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        if hits:
            offenders[str(path.relative_to(package.parent))] = hits
    assert not offenders, (
        "puremacro.trade imports scipy.io again. puremacro reads no MATLAB files; a caller "
        f"who has one loads it and passes the mapping. Offending lines: {offenders}"
    )
