"""Frozen result objects for :mod:`puremacro.asset_pricing`.

Every result is a ``@dataclass(frozen=True)`` built by a validating function
(never by a raising ``__post_init__``) and shares one report mixin:
``to_dataframe()`` / ``to_frame()`` and ``to_markdown()`` / ``to_latex()`` /
``to_typst()`` rendered through :mod:`puremacro.reports`, plus ``summary()``
and (where a figure is meaningful) ``plot(ax=None, show=False)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..inference._results import _coef_frame, _ReportMixin

__all__: list[str] = []


def _new_ax(ax: Any):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 3.4))
        return fig, ax, False
    return ax.figure, ax, True


def _finish(fig: Any, ax: Any, given: bool, show: bool):
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    return ax if given else fig
