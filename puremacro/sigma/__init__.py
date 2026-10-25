"""Variance/covariance decompositions and Shapley value attribution."""
from __future__ import annotations

from .shapley import shapley_factors, shapley_table, shapley_var

__all__ = ["shapley_var", "shapley_table", "shapley_factors"]


def __getattr__(name: str) -> None:
    if name == "SigmaObject":
        raise AttributeError(
            "puremacro.sigma.SigmaObject was removed in puremacro 6.0.0; "
            "use puremacro.volatility.sigma.SigmaObject instead."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
