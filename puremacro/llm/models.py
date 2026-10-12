"""Model registry and tier picker for :mod:`puremacro.llm`.

One table says which models the package knows, what each costs in memory and by what
name every engine loads it; a few small functions turn a friendly name into that name,
and a machine's memory into a model. Nothing here imports an inference engine, a network
helper or anything beyond the standard library, so the module is as light as the cache:
:func:`device_budget_gb` looks at ``mlx`` only inside the function, and only if it is
installed.

* **The ladder.** Three Qwen3 models, all Apache-2.0, all *thinking* models (the engines
  switch the thinking off and strip what is left of it):

  - ``qwen3-1.7b``: about 2.0 GB, needs a memory budget of 3 GB or more;
  - ``qwen3-4b``: about 3.5 GB, needs 5 GB. It is the *reference* model: shipped caches
    are built with it, and ``default`` names it;
  - ``qwen3-8b``: about 6.0 GB, needs 10 GB. It is never picked automatically.

  The four models of 4.8.1 (``qwen2.5-3b-instruct``, ``gemma2-2b``, ``llama3.2-3b``,
  ``phi3.5``) stay in the table under the same names and with the same ids, so code that
  names them keeps working.
* **Names.** ``default``, ``small`` and ``large`` are aliases of ``qwen3-4b``,
  ``qwen3-1.7b`` and ``qwen3-8b`` (:data:`ALIASES`). :func:`canonical` maps a name to the
  alias the completion cache is keyed on. A name that is neither an alias nor a tier (an
  Ollama tag, a file on disk, a repository id) is not ours to interpret: it passes through
  unchanged to the engine that was asked to load it.
* **Override.** The environment variable ``PUREMACRO_LLM_MODEL`` replaces what ``default``
  means, and nothing else: ``small``, ``large`` and every explicit alias keep their
  meaning. Its value may be a tier name (``small``), a registry alias or an id of an
  engine's own; blank counts as unset.
* **Engine ids.** :func:`resolve_model_id` gives the id one engine loads: the Hugging Face
  repository for MLX, ``(repository, file name)`` for llama.cpp (the exact file, never a
  pattern, because a pattern can match no file or several), the ``ollama pull`` tag, and
  the ``model`` field for OpenAI-compatible servers.
* **Budget.** :func:`device_budget_gb` is the memory, in GB, that a model may use: the
  working set MLX recommends for the GPU when mlx is installed, else 60 % of physical
  memory, else an assumed 8 GB. What is installed decides, never the operating system or
  the processor family. :func:`pick_model` chooses ``qwen3-1.7b`` below 5 GB and
  ``qwen3-4b`` from there up; :func:`check_fits` refuses a model whose minimum budget the
  device does not reach, and says which Qwen3 would fit.
"""
from __future__ import annotations

import importlib.util
import math
import os
from dataclasses import dataclass

__all__ = [
    "ALIASES",
    "MODELS",
    "ModelSpec",
    "ModelTooLarge",
    "canonical",
    "check_fits",
    "device_budget_gb",
    "pick_model",
    "resolve_model_id",
]

_ENV_MODEL = "PUREMACRO_LLM_MODEL"
_ENGINES = ("mlx", "llamacpp", "ollama", "openai")
_RAM_FRACTION = 0.6           # share of physical memory counted as available to a model
_ASSUMED_BUDGET_GB = 8.0      # when nothing reports the device's memory


@dataclass(frozen=True)
class ModelSpec:
    """One model of the registry and the name each engine knows it by.

    Attributes
    ----------
    alias : str
        Canonical name; the completion cache is keyed on it.
    params_b : float
        Parameters, in billions.
    approx_gb : float
        Memory for the 4-bit weights plus headroom.
    min_budget_gb : float
        Smallest memory budget (see :func:`device_budget_gb`) on which the model may be
        loaded; :func:`check_fits` raises below it.
    thinking : bool
        True for models that think before they answer, which the engines must switch off.
    license : str
        Licence identifier, as the Hugging Face model card gives it.
    mlx : str
        Hugging Face repository of the MLX conversion.
    llamacpp : tuple[str, str]
        ``(repository, file name)`` of the GGUF file: the exact name, not a pattern.
    ollama : str
        Tag for ``ollama pull``.
    openai : str
        The ``model`` field sent to OpenAI-compatible servers.
    """

    alias: str
    params_b: float
    approx_gb: float
    min_budget_gb: float
    thinking: bool
    license: str
    mlx: str
    llamacpp: tuple[str, str]
    ollama: str
    openai: str


# The file names below were checked against the Hugging Face API on 2026-10-12 (metadata
# only): each repository holds exactly one Q4_K_M file. The official Qwen repository has no
# Q4_K_M file for the 1.7B, so that rung is the ggml-org conversion.
_SPECS = (
    ModelSpec(
        alias="qwen3-1.7b", params_b=1.7, approx_gb=2.0, min_budget_gb=3.0, thinking=True,
        license="apache-2.0",
        mlx="mlx-community/Qwen3-1.7B-4bit",
        llamacpp=("ggml-org/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q4_K_M.gguf"),
        ollama="qwen3:1.7b", openai="qwen3-1.7b",
    ),
    ModelSpec(
        alias="qwen3-4b", params_b=4.0, approx_gb=3.5, min_budget_gb=5.0, thinking=True,
        license="apache-2.0",
        mlx="mlx-community/Qwen3-4B-4bit",
        llamacpp=("Qwen/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf"),
        ollama="qwen3:4b", openai="qwen3-4b",
    ),
    ModelSpec(
        alias="qwen3-8b", params_b=8.0, approx_gb=6.0, min_budget_gb=10.0, thinking=True,
        license="apache-2.0",
        mlx="mlx-community/Qwen3-8B-4bit",
        llamacpp=("Qwen/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf"),
        ollama="qwen3:8b", openai="qwen3-8b",
    ),
    # The four models of the 4.8.1 table, with the ids they had there; the llama.cpp
    # globs ("*Q4_K_M.gguf") became the exact file names.
    ModelSpec(
        alias="qwen2.5-3b-instruct", params_b=3.0, approx_gb=2.5, min_budget_gb=3.0,
        thinking=False, license="qwen-research",
        mlx="mlx-community/Qwen2.5-3B-Instruct-4bit",
        llamacpp=("Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q4_k_m.gguf"),
        ollama="qwen2.5:3b", openai="qwen2.5-3b-instruct",
    ),
    ModelSpec(
        alias="gemma2-2b", params_b=2.6, approx_gb=2.0, min_budget_gb=3.0,
        thinking=False, license="gemma",
        mlx="mlx-community/gemma-2-2b-it-4bit",
        llamacpp=("bartowski/gemma-2-2b-it-GGUF", "gemma-2-2b-it-Q4_K_M.gguf"),
        ollama="gemma2:2b", openai="gemma2-2b",
    ),
    ModelSpec(
        alias="llama3.2-3b", params_b=3.2, approx_gb=2.5, min_budget_gb=3.0,
        thinking=False, license="llama3.2",
        mlx="mlx-community/Llama-3.2-3B-Instruct-4bit",
        llamacpp=("bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
        ollama="llama3.2:3b", openai="llama3.2-3b",
    ),
    ModelSpec(
        alias="phi3.5", params_b=3.8, approx_gb=3.0, min_budget_gb=3.0,
        thinking=False, license="mit",
        mlx="mlx-community/Phi-3.5-mini-instruct-4bit",
        llamacpp=("bartowski/Phi-3.5-mini-instruct-GGUF", "Phi-3.5-mini-instruct-Q4_K_M.gguf"),
        ollama="phi3.5", openai="phi3.5",
    ),
)

MODELS: dict[str, ModelSpec] = {spec.alias: spec for spec in _SPECS}
"""Every model the package knows, keyed by canonical alias: the Qwen3 ladder, then the
four models of 4.8.1."""

ALIASES: dict[str, str] = {"default": "qwen3-4b", "small": "qwen3-1.7b", "large": "qwen3-8b"}
"""The tier names and the models they stand for. ``qwen3-4b`` is the reference model."""

_LADDER = ("qwen3-1.7b", "qwen3-4b", "qwen3-8b")     # the Qwen3 rungs, smallest first


class ModelTooLarge(RuntimeError):
    """A model needs more memory than the device offers.

    Attributes
    ----------
    alias : str
        The model that was asked for.
    suggested : str | None
        The largest Qwen3 of the ladder that fits the budget, or None if none does.
    budget_gb : float
        The memory budget, in GB, that the model was checked against.
    """

    def __init__(self, alias: str, suggested: str | None, budget_gb: float) -> None:
        # ``args`` carry the three fields rather than a message, so that the exception
        # survives pickling (a worker process re-raises it in the parent).
        super().__init__(alias, suggested, budget_gb)
        self.alias = alias
        self.suggested = suggested
        self.budget_gb = budget_gb

    def __str__(self) -> str:
        spec = MODELS.get(self.alias)
        need = f"at least {spec.min_budget_gb:.1f} GB" if spec is not None else "more memory"
        text = f"{self.alias} needs {need} and this device offers {self.budget_gb:.2f} GB"
        if self.suggested is None:
            return text + "; no model of the ladder fits"
        return text + f"; use {self.suggested!r} instead"


def _override() -> str | None:
    """What ``PUREMACRO_LLM_MODEL`` names, with a tier name resolved, or None if unset."""
    raw = os.environ.get(_ENV_MODEL, "").strip()
    return ALIASES.get(raw, raw) if raw else None


def canonical(model: str) -> str:
    """The canonical name of ``model``: the alias that the completion cache is keyed on.

    ``"default"`` is ``qwen3-4b`` unless ``PUREMACRO_LLM_MODEL`` names another model, in
    which case it is that one (a tier name in the variable is resolved once more, so
    ``PUREMACRO_LLM_MODEL=small`` makes ``default`` mean ``qwen3-1.7b``). ``"small"`` and
    ``"large"`` are never overridden. Every other name, known alias or not, is returned
    unchanged.
    """
    if model == "default":
        return _override() or ALIASES["default"]
    return ALIASES.get(model, model)


def resolve_model_id(model: str, engine_name: str) -> str | tuple[str, str]:
    """The id under which engine ``engine_name`` loads ``model``.

    ``engine_name`` is ``"mlx"``, ``"llamacpp"`` (a ``(repository, file name)`` tuple),
    ``"ollama"`` or ``"openai"``. A name outside the registry is returned unchanged for
    every engine, as an escape hatch for an id the caller already has; a registry name
    with any other ``engine_name`` (a test double) is returned as its canonical alias.
    """
    name = canonical(model)
    spec = MODELS.get(name)
    if spec is None:
        return name
    if engine_name in _ENGINES:
        return getattr(spec, engine_name)
    return spec.alias


def _mlx_working_set_gb() -> float | None:
    """The working set MLX recommends for the default device, in GB; None if unknown."""
    try:
        if importlib.util.find_spec("mlx") is None:
            return None
        import mlx.core as mx

        device_info = getattr(mx, "device_info", None)
        if device_info is None:      # older mlx only has mx.metal.device_info (deprecated in 0.32)
            device_info = mx.metal.device_info
        gb = float(device_info()["max_recommended_working_set_size"]) / 1024 ** 3
    except Exception:
        return None
    return gb if math.isfinite(gb) and gb > 0 else None


def _physical_memory_gb() -> float | None:
    """The share of physical memory that may go to a model, in GB; None if unknown."""
    try:
        from puremacro.runtime import _capabilities

        memory_mb = _capabilities.capabilities().memory_mb
        if memory_mb is None:
            return None
        gb = float(memory_mb) * _RAM_FRACTION / 1024
    except Exception:
        return None
    return gb if math.isfinite(gb) and gb > 0 else None


def device_budget_gb() -> float:
    """The memory, in GB, that a model may use on this machine.

    In order: the working set MLX recommends for the GPU (``max_recommended_working_set_size``)
    when mlx is installed; 60 % of the physical memory ``puremacro.runtime`` reports; an
    assumed 8 GB. A probe that raises, or reports nothing positive, is skipped, so the
    function never raises. Which probe answers depends on what is installed and never on
    the operating system or the processor family.
    """
    for probe in (_mlx_working_set_gb, _physical_memory_gb):
        gb = probe()
        if gb is not None:
            return gb
    return _ASSUMED_BUDGET_GB


def pick_model(tier: str | None = None, *, budget_gb: float | None = None) -> ModelSpec:
    """The registry entry to use.

    With ``tier`` (``"small"``, ``"default"``, ``"large"`` or any registry alias) the
    answer is that entry whatever the budget; ``"default"`` follows ``PUREMACRO_LLM_MODEL``
    like everywhere else. Without it, ``PUREMACRO_LLM_MODEL`` decides if it names a
    registry entry (an override that is an engine's own id has no entry to return here;
    :func:`canonical` still hands it to the engine), and otherwise the budget does:
    ``qwen3-1.7b`` below the minimum budget of ``qwen3-4b`` (5 GB), ``qwen3-4b`` from
    there up. ``qwen3-8b`` is never picked by budget. ``budget_gb`` defaults to
    :func:`device_budget_gb`.

    Raises
    ------
    ValueError
        ``tier`` is neither a tier name nor a registry alias.
    """
    if tier is not None:
        name = canonical(tier)
        spec = MODELS.get(name)
        if spec is None:
            means = f" (it means {name!r})" if name != tier else ""
            raise ValueError(f"unknown model tier {tier!r}{means}; use one of "
                             f"{list(ALIASES)} or a registry alias: {list(MODELS)}")
        return spec
    override = _override()
    if override is not None and override in MODELS:
        return MODELS[override]
    budget = device_budget_gb() if budget_gb is None else budget_gb
    reference = MODELS[ALIASES["default"]]
    return reference if budget >= reference.min_budget_gb else MODELS[ALIASES["small"]]


def check_fits(spec: ModelSpec, budget_gb: float) -> None:
    """Raise :class:`ModelTooLarge` if ``spec`` does not fit in ``budget_gb``.

    A model fits when the budget reaches its ``min_budget_gb``. The exception names the
    largest Qwen3 of the ladder that does fit, or None when not even the smallest does.
    """
    if budget_gb >= spec.min_budget_gb:
        return
    fitting = [alias for alias in _LADDER if MODELS[alias].min_budget_gb <= budget_gb]
    suggested = max(fitting, key=lambda alias: MODELS[alias].params_b) if fitting else None
    raise ModelTooLarge(spec.alias, suggested, budget_gb)
