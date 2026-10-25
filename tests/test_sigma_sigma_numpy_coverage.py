"""Test that puremacro.sigma.sigma_numpy.SigmaObject was removed in 6.0.0."""
from __future__ import annotations

import pytest


def test_sigma_numpy_sigmaobject_removed():
    with pytest.raises((ImportError, AttributeError)):
        from puremacro.sigma.sigma_numpy import SigmaObject  # noqa: F401
