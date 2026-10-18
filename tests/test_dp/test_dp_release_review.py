"""Release regressions for declared discounting in the sequence-space bridge."""
import copy

import numpy as np

from puremacro import dp
from puremacro.models.hank_sequence_space import dp_sequence_jacobian


def test_sequence_space_respects_named_discount_factor():
    m = dp.Model("named_discount")
    m.state("a", np.linspace(0, 20, 40))
    m.exogenous("z", dp.Markov(np.array([0.5, 1.5]), np.array([[0.9, 0.1], [0.1, 0.9]])))
    m.parameters(beta=0.96, gamma=1.0, r=0.01, w=1.0)
    m.local("c = (1+r)*a + w*z - a(+1)")
    m.reward("log(c)")
    m.subject_to("c > 0")
    m.aggregate(C="c", A="a", AA="2*a", AP="a(+1)")
    custom = copy.deepcopy(m)
    custom.parameters(patience=0.96, beta=0.5)
    custom.discount("patience")
    expected = dp_sequence_jacobian(m, ["w"], ["C", "A", "AA", "AP"], T=5)
    actual = dp_sequence_jacobian(custom, ["w"], ["C"], T=5)
    np.testing.assert_allclose(actual["C"]["w"], expected["C"]["w"], atol=1e-10)
    np.testing.assert_allclose(expected["AA"]["w"], 2 * expected["A"]["w"], atol=1e-12)
    np.testing.assert_allclose(expected["A"]["w"][0], 0, atol=1e-12)
    assert np.max(np.abs(expected["AP"]["w"][0])) > 1e-3
