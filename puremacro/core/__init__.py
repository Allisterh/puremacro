"""Core unified abstractions, protocols, and adapters for PureMacro v6.0.0."""
from __future__ import annotations

from .protocol import (
    DFMDifferentiableAdapter,
    DSGEDifferentiableAdapter,
    DifferentiableModel,
    HANKDAGDifferentiableAdapter,
    SpatialDifferentiableAdapter,
    StateSpaceProtocol,
    SVARDifferentiableAdapter,
    TradeDifferentiableAdapter,
    wrap_differentiable_model,
)

__all__ = [
    "DifferentiableModel",
    "StateSpaceProtocol",
    "DSGEDifferentiableAdapter",
    "HANKDAGDifferentiableAdapter",
    "DFMDifferentiableAdapter",
    "SVARDifferentiableAdapter",
    "TradeDifferentiableAdapter",
    "SpatialDifferentiableAdapter",
    "wrap_differentiable_model",
]
