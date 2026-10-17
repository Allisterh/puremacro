"""Offline AI tutor for the MAV course companion (notebook-side helper).

``tutor(question)`` asks the course's local model through ``_llm.get_llm()``, the path every
lesson uses for the model, so a learner can ask questions or check answers fully offline
($0, no API key). An answer already in the course cache (``data_curso/llm_cache.sqlite``)
comes back with no model installed; with a local model (lesson T00_B explains the install) a
new question is answered live and saved. With no answer the tutor returns a short notice that
points to lesson T00_B and to the lesson's AI-exploration questions.
Never raises — a tutor failure must not break a lesson notebook.
Provides bilingual support (English and Spanish) based on caller notebook or prompt language.

``_llm`` is imported inside :func:`tutor`, not here: a lesson that imports the tutor without
calling it keeps its start-up cost.
"""
from __future__ import annotations

import inspect
import re

_SYSTEM_EN = (
    "You are a concise teaching assistant for a graduate macroeconomics course "
    "that uses the puremacro Python library. Explain clearly, show the key step, "
    "and keep answers short. When asked to check an answer, say whether it is "
    "correct and briefly why."
)

_SYSTEM_ES = (
    "Eres un asistente docente conciso para un curso de posgrado en macroeconomía "
    "que utiliza la librería puremacro. Explica con claridad, muestra el paso clave "
    "y mantén las respuestas breves. Cuando te pidan revisar una respuesta, indica "
    "si es correcta y brevemente por qué."
)

_FALLBACK_ES = (
    "[tutor sin conexión] No hay respuesta del modelo local en esta máquina. "
    "Abre la lección T00_B (configura tu modelo local) para instalar uno, o usa "
    "las preguntas de exploración con IA de más abajo con el asistente que "
    "prefieras. El resto del cuaderno funciona igual sin el tutor."
)

_FALLBACK_EN = (
    "[offline tutor] No answer from a local model on this machine. "
    "Open lesson T00_B (set up your local model) to install one, or use the "
    "AI-exploration questions below with your preferred assistant. "
    "The rest of the notebook works without the tutor."
)

_FALLBACK = _FALLBACK_ES


def _fallback(lang: str) -> str:
    """The notice for ``lang`` (``"es"`` or ``"en"``) when no answer is available."""
    return _FALLBACK_ES if lang == "es" else _FALLBACK_EN


def _detect_lang(question: str = "", context: str = "", lang: str = "auto") -> str:
    """Detect whether Spanish or English should be used for prompt and fallback."""
    if lang in ("en", "es"):
        return lang

    # Inspect call stack for caller notebook filename
    try:
        frame = inspect.currentframe()
        depth = 0
        while frame and depth < 6:
            fname = frame.f_code.co_filename.lower()
            if "_es." in fname or fname.endswith("_es") or "/colab_es/" in fname:
                return "es"
            if any(k in fname for k in ("syllabus.", "business_cycle_facts.", "cross_country_facts.", "high_frequency.")):
                return "en"
            frame = frame.f_back
            depth += 1
    except Exception:
        pass

    # Linguistic heuristic on question and context text
    text = f"{question} {context}".lower()
    es_special = set("¿¡áéíóúüñ")
    if any(ch in es_special for ch in text):
        return "es"

    tokens = re.findall(r"\b\w+\b", text)
    if not tokens:
        return "en"

    es_words = {
        "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "en", "para",
        "por", "con", "sin", "sobre", "que", "qué", "cómo", "cuál", "cuáles", "donde",
        "dónde", "por qué", "porque", "explica", "cuaderno", "desempleo", "inflación",
        "producción", "trabajo", "horas", "tasa", "tasas", "país", "países", "ciclo",
        "modelo", "modelos", "consumo", "inversión", "duradero", "duraderos", "salarios",
        "renta", "brecha", "estados", "unidos", "españa", "méxico", "bienes", "medida",
        "subió", "cayó", "durante", "mientras", "frases", "comportan"
    }
    en_words = {
        "the", "a", "an", "of", "in", "for", "to", "with", "without", "on", "that", "which",
        "what", "how", "why", "explain", "briefly", "notebook", "unemployment", "inflation",
        "output", "labor", "labour", "hours", "rate", "rates", "country", "countries", "cycle",
        "model", "models", "consumption", "investment", "durable", "durables", "wages",
        "income", "gap", "united", "states", "spain", "mexico", "goods", "measure", "measured",
        "rose", "fell", "during", "while", "sentences", "behave", "is", "are", "was", "were"
    }

    es_count = sum(1 for tok in tokens if tok in es_words)
    en_count = sum(1 for tok in tokens if tok in en_words)

    if es_count > en_count:
        return "es"
    if en_count > es_count:
        return "en"
    return "en"


def _build_prompt(question: str, context: str, lang: str) -> str:
    """The prompt for ``question`` in ``lang`` (``"es"`` or ``"en"``): the lesson context, if
    there is one, and then the question.

    The system prompt (``_SYSTEM_ES`` or ``_SYSTEM_EN``) travels separately, as ``system=`` of
    ``try_complete``, so it is not repeated here; the course cache is keyed on both texts.
    """
    es = lang == "es"
    prompt = ""
    if context:
        ctx_label = "Contexto de la lección" if es else "Lesson context"
        prompt += f"{ctx_label}:\n{context}\n\n"
    q_label = "Pregunta" if es else "Question"
    return prompt + f"{q_label}: {question}"


def tutor(question: str, *, context: str = "", model: str = "default",
          engine: str = "auto", lang: str = "auto") -> str:
    """Ask the offline tutor. Returns the answer, or a friendly fallback string
    if no local model answers. Never raises.

    The answer is the course model's (``_llm.get_llm()``): the course cache first, then a
    local model if one is installed (lesson T00_B). ``model`` and ``engine`` are accepted
    for backward compatibility and ignored: the course uses the reference model, and
    ``PUREMACRO_LLM_MODEL`` changes it.

    Supports bilingual English/Spanish fallback and system prompts.
    """
    resolved_lang = "es"
    try:
        resolved_lang = _detect_lang(question, context, lang)
        import _llm     # here and not at the top of the module: see the module docstring
        answer = _llm.get_llm(lang=resolved_lang).try_complete(
            _build_prompt(question, context, resolved_lang),
            system=_SYSTEM_ES if resolved_lang == "es" else _SYSTEM_EN,
            max_tokens=512, temperature=0.0)
    except Exception as exc:  # graceful: any helper/availability error -> fallback
        engine_detail = "detalle del motor" if resolved_lang == "es" else "engine detail"
        return f"{_fallback(resolved_lang)}\n({engine_detail}: {type(exc).__name__})"
    if isinstance(answer, str) and answer.strip():
        return answer
    return _fallback(resolved_lang)
