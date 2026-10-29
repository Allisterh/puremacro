"""solve_sovereign_default stays bit-identical to the verbatim 6.0.1 solver.

The oracle is ``_legacy_arellano_6_0_1.py`` (with its own copy of
``_monotone``), copied from the merge-base before any refactor. Every array,
scalar and metadata entry must match exactly, dense and monotone.

The legacy copy builds the *live* ``SovereignDefaultResult``, so comparing
``simulate()`` frames between the two solvers cannot catch a change to that
method. ``test_simulate_matches_frozen_6_0_1_frame`` pins one frame generated
from a clean c7002ecd (6.0.1) checkout instead.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from puremacro.sovereign import solve_sovereign_default

from ._legacy_arellano_6_0_1 import solve_sovereign_default as legacy_solve

FROZEN_FRAME = Path(__file__).resolve().parents[1] / "fixtures" / "sovereign_fiscal" / "simulate_6_0_1_defaults_300.json"

# id -> (solver kwargs, policy searches)
CALIBRATIONS: dict[str, tuple[dict[str, Any], tuple[str, ...]]] = {
    "defaults": (dict(n_b=40, n_y=9), ("dense", "monotone")),
    "public": (
        dict(beta=0.95282, r_star=0.017, theta=0.282, b_min=-0.15, b_max=0.33, n_b=101, n_y=9,
             sigma_eps=0.0, y_penalty_type="arellano", enforce_monotonicity=False, tol=1e-10,
             max_iter=10000),
        ("dense", "monotone"),
    ),
    "log_no_saving": (
        dict(gamma=1.0, b_min=0.0, b_max=0.4, n_b=40, n_y=7, sigma_eps=0.01,
             y_penalty_type="arellano", damping=0.3, howard_steps=0),
        ("dense", "monotone"),
    ),
    "long_term": (dict(delta=0.2, kappa=0.03, n_b=30, n_y=5), ("dense",)),
}

CASES = [pytest.param(name, search, id=f"{name}-{search}")
         for name, (_, searches) in CALIBRATIONS.items() for search in searches]


def _assert_values_identical(a: Any, b: Any, where: str) -> None:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        assert isinstance(a, np.ndarray) and isinstance(b, np.ndarray), where
        assert a.dtype == b.dtype, f"{where}: dtype {a.dtype} != {b.dtype}"
        assert a.shape == b.shape, f"{where}: shape {a.shape} != {b.shape}"
        assert np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"), f"{where}: arrays differ"
    elif a is None or b is None:
        assert a is None and b is None, f"{where}: {a!r} vs {b!r}"
    elif isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        assert type(a) is type(b) and a == b, f"{where}: {a!r} vs {b!r}"
    elif isinstance(a, (float, np.floating)) or isinstance(b, (float, np.floating)):
        assert type(a) is type(b), f"{where}: type {type(a)} vs {type(b)}"
        assert a == b or (np.isnan(a) and np.isnan(b)), f"{where}: {a!r} vs {b!r}"
    elif isinstance(a, dict) or isinstance(b, dict):
        assert isinstance(a, dict) and isinstance(b, dict), where
        assert list(a) == list(b), f"{where}: keys {list(a)} vs {list(b)}"
        for key in a:
            _assert_values_identical(a[key], b[key], f"{where}[{key!r}]")
    else:
        assert type(a) is type(b) and a == b, f"{where}: {a!r} vs {b!r}"


def assert_results_identical(a: Any, b: Any) -> None:
    """Exact equality of every dataclass field, metadata entries included."""
    assert type(a).__name__ == type(b).__name__
    for f in dataclasses.fields(a):
        _assert_values_identical(getattr(a, f.name), getattr(b, f.name), f.name)


@pytest.mark.parametrize(("name", "search"), CASES)
def test_solver_bit_identical_to_legacy(name: str, search: str) -> None:
    kwargs = CALIBRATIONS[name][0]
    current = solve_sovereign_default(**kwargs, policy_search=search)
    legacy = legacy_solve(**kwargs, policy_search=search)
    assert_results_identical(current, legacy)
    pd.testing.assert_frame_equal(current.simulate(periods=300), legacy.simulate(periods=300),
                                  check_exact=True)


def test_identity_helper_detects_one_ulp_change() -> None:
    result = legacy_solve(**CALIBRATIONS["defaults"][0])
    assert_results_identical(result, legacy_solve(**CALIBRATIONS["defaults"][0]))

    price = result.bond_price.copy()
    price[3, 2] = np.nextafter(price[3, 2], np.inf)
    with pytest.raises(AssertionError):
        assert_results_identical(dataclasses.replace(result, bond_price=price), result)

    metadata = dict(result.metadata)
    metadata["q_rf"] = float(np.nextafter(metadata["q_rf"], 0.0))
    with pytest.raises(AssertionError):
        assert_results_identical(dataclasses.replace(result, metadata=metadata), result)


def _frozen_frame() -> pd.DataFrame:
    record = json.loads(FROZEN_FRAME.read_text())
    frame = pd.DataFrame({name: record["data"][name] for name in record["columns"]})
    return frame.astype(record["dtypes"])


def test_simulate_matches_frozen_6_0_1_frame() -> None:
    frame = solve_sovereign_default(**CALIBRATIONS["defaults"][0]).simulate(periods=300)
    pd.testing.assert_frame_equal(frame, _frozen_frame(), check_exact=True)


def test_frozen_frame_detects_a_one_row_change() -> None:
    frame = solve_sovereign_default(**CALIBRATIONS["defaults"][0]).simulate(periods=300)
    changed = _frozen_frame()
    changed.loc[150, "consumption"] = np.nextafter(changed.loc[150, "consumption"], np.inf)
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(frame, changed, check_exact=True)
