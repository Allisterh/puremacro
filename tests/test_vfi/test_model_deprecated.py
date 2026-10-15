"""``vfi.Model`` is deprecated in favour of ``puremacro.dp`` and removed in 6.0."""
from __future__ import annotations

import warnings

import numpy as np

from puremacro.vfi import Model

MESSAGE = ("vfi.Model is deprecated and will be removed in 6.0; declare the model with "
           "puremacro.dp (see docs/dp.md, 'Porting from vfi.Model')")


def test_model_warns_once_with_porting_pointer():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = Model("deprecated")
        m.set_param("beta", 0.95).set_param("gamma", 2.0)
        m.add_state("a", 0.1, 5.0, 10).add_shock("z", 0.9, 0.1, 3)
        m.set_utility(lambda c, p: c ** (1.0 - p["gamma"]) / (1.0 - p["gamma"]))
        m.set_budget(lambda a, ap, z, p: np.exp(z) + 0.9 * a - ap)
    dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(dep) == 1
    assert str(dep[0].message) == MESSAGE
    # stacklevel=2 points the warning at the caller, not at puremacro/vfi/model.py
    assert dep[0].filename == __file__
