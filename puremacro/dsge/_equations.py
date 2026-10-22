"""Private adapters for optional residual-function evaluation context."""
from __future__ import annotations

import inspect
from typing import Any, Callable


def _steady_state_adapter(equations: Callable) -> Callable:
    """Adapt an explicitly declared ``steady_state`` argument once per solver.

    Legacy residuals may have unrelated optional positional arguments. Never
    overwrite them, or catch a TypeError from the residual body as evidence of
    an incompatible signature. Callables without an inspectable signature use
    the legacy calling convention.
    """
    try:
        parameter = inspect.signature(equations).parameters.get("steady_state")
    except (TypeError, ValueError):
        parameter = None

    if parameter is not None and parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    ):
        def evaluate(*args: Any, steady_state: Any) -> Any:
            return equations(*args, steady_state=steady_state)
    elif parameter is not None and parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
        def evaluate(*args: Any, steady_state: Any) -> Any:
            return equations(*args, steady_state)
    else:
        def evaluate(*args: Any, steady_state: Any) -> Any:
            return equations(*args)
    return evaluate
