"""Frozen independent references; no SSJ, Numba or network dependency at runtime."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pytest

from puremacro.models.hank_sequence_space import solve_hank_sequence_space
from puremacro.sovereign import solve_sovereign_default

FIXTURES = Path(__file__).parent / "fixtures" / "research_reliability"


@pytest.mark.parametrize("name", ["hank_ssj_manifest.json", "arellano_manifest.json"])
def test_independent_reference_integrity(name):
    manifest = json.loads((FIXTURES/name).read_text())
    for filename, digest in manifest["fixtures_sha256"].items():
        assert hashlib.sha256((FIXTURES/filename).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("n_a", [50, 100])
def test_one_asset_households_against_authors_ssj(n_a):
    result = solve_hank_sequence_space(T=24, n_a=n_a, a_max=100.0)
    assert result.ss_converged
    assert result.distribution[-1].sum() < 1e-10
    with np.load(FIXTURES/f"hank_ssj_{n_a}.npz") as reference:
        for name in ("policy_c", "policy_a", "distribution"):
            np.testing.assert_allclose(getattr(result, name), reference[name], atol=1e-8, rtol=0)
        assert result.steady_state_consumption == pytest.approx(float(reference["consumption"]), abs=1e-8)
        # Both implementations use h=1e-4. Their treatment of simultaneous
        # return and fiscal-income perturbations differs at finite step size.
        for shock, tolerance in [("w", 1e-7), ("r", 5e-4)]:
            actual = result.fake_news(T=24, shock_input=shock).jacobian
            np.testing.assert_allclose(actual, reference[f"jacobian_{shock}"], atol=tolerance, rtol=0)


@pytest.mark.parametrize("n_b", [41, 81])
@pytest.mark.parametrize("howard_steps", [0, 15])
def test_short_term_deterministic_default_against_independent_equations(n_b, howard_steps):
    result = solve_sovereign_default(n_b=n_b, n_y=9, b_min=-0.4, b_max=0.4,
                                     sigma_eps=0, y_penalty_type="arellano", howard_steps=howard_steps,
                                     enforce_monotonicity=False, tol=1e-10, max_iter=10000)
    assert result.converged
    with np.load(FIXTURES/f"arellano_{n_b}.npz") as reference:
        for name in reference.files:
            np.testing.assert_allclose(getattr(result, name), reference[name], atol=1e-8, rtol=0)
