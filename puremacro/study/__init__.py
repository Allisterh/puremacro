"""Explicit, reproducible time-series studies and comparable result archives."""
from ._core import (
    StudyComparison, StudyResult, StudySpec, VariableSpec,
    compare_studies, run_study,
)
from ._archive import load_study, save_study
from ._scenario import ScenarioSpec, ScenarioResult, record_scenario
from .studio import (
    CounterfactualGrid,
    PolicyStudioCell,
    PolicyStudioResult,
    run_policy_studio,
)

__all__ = [
    "VariableSpec", "StudySpec", "StudyResult", "StudyComparison",
    "run_study", "compare_studies", "save_study", "load_study",
    "ScenarioSpec", "ScenarioResult", "record_scenario",
    "CounterfactualGrid",
    "PolicyStudioCell",
    "PolicyStudioResult",
    "run_policy_studio",
]
