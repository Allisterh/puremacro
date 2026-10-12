"""Minimum-distance bridge from empirical moments to structural models."""

from .bridge import MomentTargets, StructuralFitResult, fit_structural
from .lp import lp_moment_targets

__all__ = ["MomentTargets", "StructuralFitResult", "fit_structural", "lp_moment_targets"]
