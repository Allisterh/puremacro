"""Taylor tensor contractions checked against independent Kronecker algebra."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from puremacro.dsge import load_mod
from puremacro.dsge.local_accuracy import diagnose_perturbation, evaluate_perturbation


def polynomial_solution():
    variables, states, shocks = ("x", "z", "c"), ("x", "z"), ("u", "v")
    rng = np.random.default_rng(912)
    attributes = dict(variable_names=variables, state_variables=states, shock_names=shocks,
                      ys=pd.Series([2, 3, 4], index=variables))
    for name, width in [("ghx", 2), ("ghu", 2), ("ghxx", 4), ("ghxu", 4),
                        ("ghuu", 4), ("ghxxx", 8), ("ghxxu", 8), ("ghxuu", 8),
                        ("ghuuu", 8), ("ghxss", 2), ("ghuss", 2)]:
        attributes[name] = pd.DataFrame(rng.normal(size=(3, width)), index=variables)
    attributes["ghs2"] = pd.Series(rng.normal(size=3), index=variables)
    rules = SimpleNamespace(**attributes)
    return SimpleNamespace(decision_rules=lambda: rules), rules


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.pyodide_smoke
def test_taylor_contractions_match_dense_oracle_for_batch_and_single(order):
    solution, rules = polynomial_solution()
    x = np.array([[0.1, -0.2], [0.3, 0.05]])
    u = np.array([[0.02, 0.03], [-0.04, 0.01]])
    sigma = 0.7
    expected = []
    for state, innovation in zip(x, u):
        y = rules.ys.to_numpy() + rules.ghx.to_numpy() @ state + rules.ghu.to_numpy() @ innovation
        if order >= 2:
            y += rules.ghxx.to_numpy() @ np.kron(state, state) / 2
            y += rules.ghxu.to_numpy() @ np.kron(state, innovation)
            y += rules.ghuu.to_numpy() @ np.kron(innovation, innovation) / 2
            y += rules.ghs2.to_numpy() * sigma**2 / 2
        if order >= 3:
            y += rules.ghxxx.to_numpy() @ np.kron(np.kron(state, state), state) / 6
            y += rules.ghxxu.to_numpy() @ np.kron(np.kron(state, state), innovation) / 2
            y += rules.ghxuu.to_numpy() @ np.kron(np.kron(state, innovation), innovation) / 2
            y += rules.ghuuu.to_numpy() @ np.kron(np.kron(innovation, innovation), innovation) / 6
            y += (rules.ghxss.to_numpy() @ state + rules.ghuss.to_numpy() @ innovation) * sigma**2 / 2
        expected.append(y)
    levels = x + np.array([2, 3])
    np.testing.assert_allclose(evaluate_perturbation(solution, levels, u, sigma=sigma, order=order), expected, atol=1e-14)
    np.testing.assert_allclose(evaluate_perturbation(solution, x, u, sigma=sigma, order=order, deviations=True), expected, atol=1e-14)
    np.testing.assert_allclose(evaluate_perturbation(solution, levels[0], u[0], sigma=sigma, order=order), expected[0], atol=1e-14)


def test_raw_rule_matches_first_step_with_zero_risk_and_rejects_invalid_inputs():
    model = load_mod("var x c; varexo e; model; x=0.8*x(-1)+e; c=0.4*x; end; initval; x=0; c=0; end; steady;")
    np.testing.assert_allclose(evaluate_perturbation(model, [2], [0.1]), [1.7, 0.68], atol=1e-12)
    np.testing.assert_allclose(evaluate_perturbation(model, [[2], [1]], [0.1]), [[1.7, 0.68], [0.9, 0.36]], atol=1e-12)
    for kwargs in ({"order": 2}, {"sigma": -1}, {"sigma": np.nan}, {"order": True}):
        with pytest.raises(ValueError):
            evaluate_perturbation(model, [2], [0.1], **kwargs)
    with pytest.raises(ValueError, match="lagged_states"):
        evaluate_perturbation(model, [2, 1])
    with pytest.raises(ValueError, match="shocks"):
        evaluate_perturbation(model, [2], [np.nan])


def test_risk_scale_changes_expectation_terms_without_rescaling_realized_innovations():
    solution, rules = polynomial_solution()
    a = evaluate_perturbation(solution, [2, 3], [0.1, -0.2], sigma=0, order=2)
    b = evaluate_perturbation(solution, [2, 3], [0.1, -0.2], sigma=1, order=2)
    np.testing.assert_allclose(b-a, rules.ghs2.to_numpy()/2, atol=1e-14)


def test_near_unit_roots_and_rank_deficiency_remain_visible():
    model = load_mod("var x c; varexo e; model; x=0.999999*x(-1)+e; c=0.4*x; end; initval; x=0; c=0; end; steady;")
    report = diagnose_perturbation(model)
    assert report.roots.near_unit.all()
    assert report.jacobian_rank == report.equation_count == 2
    assert any("unit circle" in warning for warning in report.warnings)
    deficient = SimpleNamespace(G=np.array([[0.8]]), _A_minus=np.zeros((2, 2)),
                                _A_plus=np.zeros((2, 2)), _A_0=np.array([[1, 0], [2, 0]]))
    report = diagnose_perturbation(deficient)
    assert report.jacobian_rank == 1
    assert any("rank deficient" in warning for warning in report.warnings)
    absent = diagnose_perturbation(SimpleNamespace(G=np.array([[0.8]])))
    assert absent.jacobian_rank is None
    assert any("not checked" in warning for warning in absent.warnings)
    pd.testing.assert_frame_equal(report.to_frame(), report.roots)
    pd.testing.assert_frame_equal(report.to_dataframe(), report.roots)
    assert isinstance(report.to_markdown(), str)
    assert "\\begin{tabular}" in report.to_latex()
    assert "#table(" in report.to_typst()
    frame = report.to_frame()
    frame.loc[0, "modulus"] = -99
    assert report.roots.loc[0, "modulus"] >= 0
