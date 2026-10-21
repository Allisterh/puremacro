"""Warning helpers shared across subpackages.

Kept free of third-party imports so any module can use them at import time.
"""
from __future__ import annotations

import os
import sys
from typing import Any

#: The package directory, used to find the first stack frame outside puremacro.
_PUREMACRO_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep


def _caller_stacklevel() -> int:
    """``stacklevel`` for ``warnings.warn`` pointing at the first frame outside puremacro.

    Called from the function that issues the warning. Wrappers (for example
    ``AiyagariContinuousModel.solve``, ``TradePolicySimulator.simulate_*`` or
    ``dp.Model.solve``) forward to the solver one or more frames deeper, so a
    fixed ``stacklevel=2`` would attribute the warning to library code.
    """
    frame: Any = sys._getframe(1)  # the function calling warnings.warn: stacklevel 1
    level = 1
    while frame is not None:
        if not os.path.abspath(frame.f_code.co_filename).startswith(_PUREMACRO_DIR):
            return level
        frame = frame.f_back
        level += 1
    return 2
