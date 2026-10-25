"""Python port of SigmaObject.m (removed in puremacro 6.0.0)."""
from __future__ import annotations


def __getattr__(name: str) -> None:
    if name == "SigmaObject":
        raise AttributeError(
            "puremacro.sigma.sigma_numpy.SigmaObject was removed in puremacro 6.0.0; "
            "use puremacro.volatility.sigma.SigmaObject instead."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
