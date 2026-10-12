"""Local inference engines for :mod:`puremacro.llm`.

Three engines run a model on the user's own machine, and :func:`resolve_engine` picks one:

* :class:`MLXEngine`: Apple-silicon GPU inference through ``mlx-lm`` (the ``[local-llm]``
  extra), on a Mac and under Juno on the iPad, which ships MLX and mlx-lm.
* :class:`LlamaCppEngine`: GGUF inference through ``llama-cpp-python`` (``[local-llm]``) on any
  platform, with every layer offloaded to the GPU when the build has one.
* :class:`HTTPEngine`: a local server over ``urllib``, with no extra dependency: Ollama's
  ``/api/chat`` (``api="ollama"``) or an OpenAI-compatible ``/v1/chat/completions``
  (``api="openai"``: llama-server, llamafile, LM Studio, vLLM, and Ollama as well).

* **Availability.** What is installed decides, never the operating system or the processor
  family: MLX needs ``importlib.util.find_spec`` to find both ``mlx_lm`` and ``mlx``, llama.cpp
  needs ``llama_cpp``. A server is available when its model list (``/api/tags`` for Ollama,
  ``/v1/models`` for OpenAI-compatible servers) answers with a JSON object within three
  seconds; any error, an open port that answers 404 or a body that is not JSON included, means
  no. Nothing is probed when this module is imported or an engine is constructed.
* **Order.** ``resolve_engine("auto")`` tries ``mlx``, ``llamacpp``, then the servers of
  :data:`DEFAULT_SERVERS` in turn: Ollama on port 11434, then OpenAI-compatible servers on 8080
  (llamafile, llama-server) and 1234 (LM Studio). A ``base_url``, or else the environment
  variable ``PUREMACRO_LLM_BASE_URL``, replaces those three with one URL, tried as an Ollama
  server and then as an OpenAI-compatible one.
* **Thinking.** ``complete(..., thinking=True)`` says that the model thinks before it answers
  (``ModelSpec.thinking``) and that the thinking is to be switched off for the call. MLX
  applies the chat template with ``enable_thinking=False``; llama.cpp passes the same switch to
  the template through the model's chat handler, or appends ``/no_think`` to the user turn when
  no handler can be found; Ollama gets ``"think": false``; an OpenAI-compatible server gets both
  ``chat_template_kwargs={"enable_thinking": false}`` and the ``/no_think`` suffix, since
  servers differ in which of the two they honour. Whatever thinking gets through is removed by
  :func:`strip_thinking`, which every engine applies to what it returns.
* **System prompt.** A non-empty ``system`` text becomes a leading ``{"role": "system"}``
  message on every engine.
* **Errors.** :class:`LocalLLMUnavailable`, a :class:`BackendUnavailable`, means that the engine
  could not reach its model: the package is missing, the model cannot be loaded, the server
  cannot be reached or answers with an error status. Callers such as ``score_llm`` let it
  propagate instead of counting it as malformed output.

``mlx``, ``mlx_lm``, ``llama_cpp`` and ``puremacro._http`` are imported inside the functions
that use them, so importing this module needs nothing but the standard library and works under
Pyodide, where a server on ``localhost`` cannot be reached and no engine is available.
"""
from __future__ import annotations

import importlib.util
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "BackendUnavailable",
    "DEFAULT_SERVERS",
    "Engine",
    "HTTPEngine",
    "INSTALL_HINT",
    "LlamaCppEngine",
    "LocalLLMUnavailable",
    "MLXEngine",
    "available_engines",
    "resolve_engine",
    "strip_thinking",
]

_ENV_BASE_URL = "PUREMACRO_LLM_BASE_URL"
_PROBE_TIMEOUT = 3.0          # seconds a server has to answer the probe of its model list
_NO_THINK = "\n/no_think"     # Qwen3's switch in the text of a user turn
_ENGINE_NAMES = ("mlx", "llamacpp", "ollama", "openai")

DEFAULT_SERVERS: tuple[tuple[str, str], ...] = (
    ("ollama", "http://localhost:11434"),
    ("openai", "http://localhost:8080"),
    ("openai", "http://localhost:1234"),
)
"""``(api, base URL)`` of the local servers ``resolve_engine("auto")`` tries, in order: Ollama,
then OpenAI-compatible servers on the ports of llamafile and llama-server (8080) and of LM
Studio (1234)."""

INSTALL_HINT = (
    "No local LLM engine available. Install one with "
    '`pip install "puremacro[local-llm]"` (MLX on Apple silicon, llama-cpp-python anywhere), '
    "or start Ollama (https://ollama.com) and run `ollama pull qwen3:4b`."
)
"""What to do when no engine is usable. Every :class:`LocalLLMUnavailable` raised for a missing
package, an unavailable engine or a server that cannot be reached includes it."""


class BackendUnavailable(RuntimeError):
    """A scoring backend or provider could not reach its model (a connection or setup failure),
    as opposed to the model returning output that cannot be parsed. Callers such as
    ``score_llm`` let it propagate instead of dropping it as malformed."""


class LocalLLMUnavailable(BackendUnavailable):
    """No local inference engine is usable, or the chosen one cannot reach its model."""


@runtime_checkable
class Engine(Protocol):
    """What :func:`resolve_engine` returns: an engine that can run a model.

    Attributes
    ----------
    name : str
        ``"mlx"``, ``"llamacpp"``, ``"ollama"`` or ``"openai"``: the engine name that
        :func:`puremacro.llm.models.resolve_model_id` takes to give this engine's model id.
    """

    name: str

    def available(self) -> bool:
        """Whether the engine can run here: its package is installed, or its server answers."""
        ...

    def complete(self, model_id: Any, prompt: str, *, system: str | None, max_tokens: int,
                 temperature: float, json_mode: bool, thinking: bool) -> str:
        """The model's reply to ``prompt``, with its thinking removed (:func:`strip_thinking`).

        ``model_id`` is already this engine's id for the model. ``system``, when not empty, is
        sent as a leading system message. ``thinking`` is True for a model that thinks before
        it answers; the engine then switches the thinking off for this call.
        """
        ...


_THINK_OPEN, _THINK_CLOSE = "<think>", "</think>"
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text: str) -> str:
    """``text`` without the model's thinking, and without surrounding whitespace.

    Every ``<think>…</think>`` block is removed. If a ``</think>`` is left (thinking whose
    opening tag was part of the prompt), everything up to the last one goes too; if a
    ``<think>`` is left (thinking cut off by the token limit), everything from it on.
    """
    text = _THINK_BLOCK.sub("", text)
    close = text.rfind(_THINK_CLOSE)
    if close != -1:
        text = text[close + len(_THINK_CLOSE):]
    opening = text.find(_THINK_OPEN)
    if opening != -1:
        text = text[:opening]
    return text.strip()


def _installed(name: str) -> bool:
    """Whether ``importlib.util.find_spec`` finds ``name``. A finder that raises (a module
    stubbed in ``sys.modules`` without a spec, an import blocker) counts as not installed."""
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _missing(package: str, exc: ImportError) -> LocalLLMUnavailable:
    return LocalLLMUnavailable(f"{package} cannot be imported ({exc}). {INSTALL_HINT}")


def _messages(prompt: str, system: str | None) -> list[dict[str, str]]:
    """The chat turns of one call: the system text, if any, then the prompt as the user turn.

    An empty ``system`` sends no system turn, like None: the completion cache keys both alike.
    """
    turns = [{"role": "system", "content": system}] if system else []
    turns.append({"role": "user", "content": prompt})
    return turns


def _with_no_think(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """A copy of ``messages`` with ``/no_think`` appended to the last user turn."""
    turns = [dict(turn) for turn in messages]
    for turn in reversed(turns):
        if turn.get("role") == "user":
            turn["content"] = turn["content"] + _NO_THINK
            break
    return turns


def _text(value: object) -> str:
    """``value`` if it is a string, else ``""`` (a missing or null ``content``)."""
    return value if isinstance(value, str) else ""


def _choice_content(reply: Any) -> Any:
    """``reply["choices"][0]["message"]["content"]`` of a chat completion, or None."""
    try:
        return reply["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


class MLXEngine:
    """Apple-silicon GPU inference through ``mlx-lm``.

    Decoding is greedy: ``temperature`` is not used, which keeps extraction deterministic, and
    neither is ``json_mode``. A loaded model and its tokenizer are kept for each model id for
    the life of the instance.

    Parameters
    ----------
    adapter_path : str, optional
        Directory of LoRA adapters that ``mlx_lm.load`` applies to every model this instance
        loads.
    """

    name = "mlx"

    def __init__(self, adapter_path: str | None = None) -> None:
        self.adapter_path = adapter_path
        self._loaded: dict[str, tuple[Any, Any]] = {}

    def available(self) -> bool:
        """True when ``mlx_lm`` and ``mlx`` can both be imported."""
        return _installed("mlx_lm") and _installed("mlx")

    def _load(self, model_id: str) -> tuple[Any, Any]:
        if model_id not in self._loaded:
            try:
                from mlx_lm import load
            except ImportError as exc:
                raise _missing("mlx-lm", exc) from exc
            options = {} if self.adapter_path is None else {"adapter_path": self.adapter_path}
            try:
                self._loaded[model_id] = load(model_id, **options)
            except Exception as exc:
                raise LocalLLMUnavailable(
                    f"MLX could not load the model {model_id!r}: {exc}") from exc
        return self._loaded[model_id]

    def complete(self, model_id: str, prompt: str, *, system: str | None = None,
                 max_tokens: int, temperature: float, json_mode: bool,
                 thinking: bool = False) -> str:
        """The model's reply; see :meth:`Engine.complete`."""
        try:
            from mlx_lm import generate
        except ImportError as exc:
            raise _missing("mlx-lm", exc) from exc
        model, tokenizer = self._load(model_id)
        messages = _messages(prompt, system)
        if thinking:
            templated = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                      enable_thinking=False)
        else:
            templated = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        if isinstance(templated, Mapping):     # a BatchEncoding: transformers 5 returns one
            templated = templated["input_ids"]  # unless the caller asks for plain ids
        # Token ids (what mlx-lm 0.32 returns) and rendered text are both valid prompts.
        reply = generate(model, tokenizer, prompt=templated, max_tokens=max_tokens,
                         verbose=False)
        return strip_thinking(_text(reply))


def _gguf_source(model_id: Any) -> str | tuple[str, str]:
    """Where llama.cpp loads ``model_id`` from: a file path, or a ``(repository, file)`` pair."""
    if isinstance(model_id, (str, os.PathLike)):
        path = os.path.expanduser(os.fspath(model_id))
        if isinstance(path, str) and os.path.isfile(path):
            return path
    elif (isinstance(model_id, Sequence) and len(model_id) == 2
          and all(isinstance(part, str) for part in model_id)):
        return (model_id[0], model_id[1])
    raise LocalLLMUnavailable(
        f"llama.cpp loads a (repository, file name) pair from Hugging Face or the path of a "
        f".gguf file, and {model_id!r} is neither; a model of the registry, such as "
        f"'qwen3-4b', is resolved to its pair.")


def _chat_handler(llm: Any) -> Any:
    """The chat handler ``llm.create_chat_completion`` would use, or None if none is found.

    It is looked for where llama-cpp-python 0.3 looks: ``llm.chat_handler``, then
    ``llm._chat_handlers[llm.chat_format]`` (the handler built from a chat template in the GGUF
    file, as for Qwen3), then the registry of named formats. A step that raises counts as
    finding nothing.
    """
    def registered() -> Any:
        from llama_cpp import llama_chat_format
        return llama_chat_format.get_chat_completion_handler(llm.chat_format)

    for find in (lambda: llm.chat_handler,
                 lambda: llm._chat_handlers.get(llm.chat_format),
                 registered):
        try:
            handler = find()
        except Exception:
            continue
        if callable(handler):
            return handler
    return None


def _thinking_off(handler: Any) -> Any:
    """``handler`` with ``enable_thinking=False`` passed on to the chat template, the way
    ``llama_cpp.server`` applies ``--chat_template_kwargs``."""
    def handler_without_thinking(*args: Any, **kwargs: Any) -> Any:
        return handler(*args, **{"enable_thinking": False, **kwargs})
    return handler_without_thinking


class LlamaCppEngine:
    """GGUF inference through ``llama-cpp-python``, on any platform.

    ``model_id`` is a ``(repository, file name)`` pair, which ``Llama.from_pretrained``
    downloads from Hugging Face the first time, or the path of a ``.gguf`` file. Every layer is
    offloaded to the GPU (``n_gpu_layers=-1``): Metal or CUDA when the build has one, and
    harmless on a build without (llama-cpp-python's own default, 0, would leave a GPU idle). A
    loaded model is kept for each id for the life of the instance.

    Parameters
    ----------
    n_ctx : int
        Context window, in tokens.
    """

    name = "llamacpp"

    def __init__(self, n_ctx: int = 4096) -> None:
        self.n_ctx = n_ctx
        self._loaded: dict[str | tuple[str, str], Any] = {}

    def available(self) -> bool:
        """True when ``llama_cpp`` can be imported."""
        return _installed("llama_cpp")

    def _load(self, model_id: Any) -> Any:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise _missing("llama-cpp-python", exc) from exc
        source = _gguf_source(model_id)
        if source not in self._loaded:
            options = {"n_ctx": self.n_ctx, "n_gpu_layers": -1, "verbose": False}
            try:
                if isinstance(source, str):
                    llm = Llama(model_path=source, **options)
                else:
                    llm = Llama.from_pretrained(repo_id=source[0], filename=source[1], **options)
            except Exception as exc:
                raise LocalLLMUnavailable(
                    f"llama.cpp could not load the model {model_id!r}: {exc}") from exc
            self._loaded[source] = llm
        return self._loaded[source]

    def complete(self, model_id: Any, prompt: str, *, system: str | None = None,
                 max_tokens: int, temperature: float, json_mode: bool,
                 thinking: bool = False) -> str:
        """The model's reply; see :meth:`Engine.complete`.

        With ``thinking``, the chat handler of the model is wrapped for this call so that the
        template gets ``enable_thinking=False``, and put back afterwards; ``/no_think`` is
        appended to the user turn only when no handler can be found.
        """
        llm = self._load(model_id)
        messages = _messages(prompt, system)
        handler = _chat_handler(llm) if thinking else None
        if thinking and handler is None:
            messages = _with_no_think(messages)
        request: dict[str, Any] = {"messages": messages, "temperature": temperature,
                                   "max_tokens": max_tokens}
        if json_mode:
            request["response_format"] = {"type": "json_object"}
        if handler is None:
            reply = llm.create_chat_completion(**request)
        else:
            installed = getattr(llm, "chat_handler", None)
            llm.chat_handler = _thinking_off(handler)
            try:
                reply = llm.create_chat_completion(**request)
            finally:
                llm.chat_handler = installed
        return strip_thinking(_text(_choice_content(reply)))


def _get_json(url: str, *, timeout: float) -> Any:
    """The decoded JSON body of ``GET url``.

    One attempt and no retry, so that a probe costs at most ``timeout`` seconds
    (``puremacro._http`` retries a failed connection once, which doubles the wait on a host that
    drops packets).
    """
    import json
    import urllib.request

    from .._http import USER_AGENT

    request = urllib.request.Request(url, headers={"Accept": "application/json",
                                                   "User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _server_message(exc: Any) -> str:
    """The error message an LLM server put in the body of an error reply, or ``""``.

    Ollama sends ``{"error": "..."}``, OpenAI-compatible servers ``{"error": {"message": ...}}``.
    """
    import json

    try:
        body = json.loads(exc.read().decode("utf-8"))
    except Exception:
        return ""
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        error = error.get("message")
    return " ".join(error.split())[:300] if isinstance(error, str) else ""


class HTTPEngine:
    """A local inference server, reached over ``urllib``.

    Parameters
    ----------
    base_url : str
        Root URL of the server, such as ``http://localhost:11434``; a trailing slash is dropped.
    api : {"ollama", "openai"}
        ``"ollama"`` posts to Ollama's ``/api/chat``; ``"openai"`` to ``/v1/chat/completions``
        (llama-server, llamafile, LM Studio, vLLM; Ollama serves it too). It is also the
        engine's ``name``.
    timeout : float
        Seconds to wait for a completion.
    probe_timeout : float
        Seconds :meth:`available` waits for the server's model list (never more than
        ``timeout``).
    """

    def __init__(self, *, base_url: str, api: str, timeout: float = 120.0,
                 probe_timeout: float = _PROBE_TIMEOUT) -> None:
        if api not in ("ollama", "openai"):
            raise ValueError(f"api must be 'ollama' or 'openai'; got {api!r}")
        self.base_url = base_url.rstrip("/")
        self.api = api
        self.name = api
        self.timeout = timeout
        self.probe_timeout = probe_timeout

    def available(self) -> bool:
        """True when the server's model list answers with a JSON object; never raises."""
        path = "/api/tags" if self.api == "ollama" else "/v1/models"
        try:
            listing = _get_json(self.base_url + path,
                                timeout=min(self.timeout, self.probe_timeout))
        except Exception:
            return False
        return isinstance(listing, dict)

    def complete(self, model_id: Any, prompt: str, *, system: str | None = None,
                 max_tokens: int, temperature: float, json_mode: bool,
                 thinking: bool = False) -> str:
        """The model's reply; see :meth:`Engine.complete`."""
        messages = _messages(prompt, system)
        payload: dict[str, Any]
        if self.api == "ollama":
            path = "/api/chat"
            payload = {"model": model_id, "messages": messages, "stream": False,
                       "options": {"temperature": temperature, "num_predict": max_tokens}}
            if json_mode:
                payload["format"] = "json"
            if thinking:
                payload["think"] = False
        else:
            path = "/v1/chat/completions"
            payload = {"model": model_id,
                       "messages": _with_no_think(messages) if thinking else messages,
                       "temperature": temperature, "max_tokens": max_tokens}
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            if thinking:
                payload["chat_template_kwargs"] = {"enable_thinking": False}
        reply = self._post(path, payload)
        if self.api == "ollama":
            message = reply.get("message") if isinstance(reply, dict) else None
            content = message.get("content") if isinstance(message, dict) else None
        else:
            content = _choice_content(reply)
        return strip_thinking(_text(content))

    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        import urllib.error

        from .._http import post_json

        try:
            return post_json(self.base_url + path, payload, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            said = _server_message(exc)
            raise LocalLLMUnavailable(
                f"The local LLM server at {self.base_url} answered {path} with HTTP "
                f"{exc.code}{': ' + said if said else ''}. {INSTALL_HINT}") from exc
        except OSError as exc:         # URLError, a timeout, a connection dropped mid-reply
            raise LocalLLMUnavailable(
                f"Cannot reach a local LLM server at {self.base_url} ({exc}). "
                f"{INSTALL_HINT}") from exc


def _server_url(base_url: str | None) -> str | None:
    """The one server URL that replaces :data:`DEFAULT_SERVERS`: ``base_url``, else the
    environment variable ``PUREMACRO_LLM_BASE_URL``; None when neither is set (or blank)."""
    for value in (base_url, os.environ.get(_ENV_BASE_URL)):
        if value and value.strip():
            return value.strip()
    return None


def _servers(base_url: str | None) -> tuple[tuple[str, str], ...]:
    """``(api, URL)`` of the servers to try, in order."""
    url = _server_url(base_url)
    return DEFAULT_SERVERS if url is None else (("ollama", url), ("openai", url))


def resolve_engine(engine: str = "auto", *, base_url: str | None = None,
                   timeout: float = 120.0, probe_timeout: float = _PROBE_TIMEOUT) -> Engine:
    """A usable engine, or :class:`LocalLLMUnavailable`.

    ``"auto"`` returns the first available of ``mlx``, ``llamacpp`` and the servers (see the
    module docstring for the order and for how ``base_url`` and ``PUREMACRO_LLM_BASE_URL``
    replace the default servers). An explicit ``"mlx"``, ``"llamacpp"``, ``"ollama"`` or
    ``"openai"`` returns that engine if it is available; ``"openai"`` without a URL tries both
    default ports. ``timeout`` is how long an HTTP engine waits for a completion,
    ``probe_timeout`` how long each server probe waits.

    Raises
    ------
    LocalLLMUnavailable
        No engine (or not the one named) is available.
    ValueError
        ``engine`` is not one of the names above.
    """
    if engine == "auto":
        tried = ["mlx", "llamacpp"]
        for local in (MLXEngine(), LlamaCppEngine()):
            if local.available():
                return local
        for api, url in _servers(base_url):
            server = HTTPEngine(base_url=url, api=api, timeout=timeout, probe_timeout=probe_timeout)
            if server.available():
                return server
            tried.append(f"{api} at {server.base_url}")
        raise LocalLLMUnavailable(f"{INSTALL_HINT} Tried: {', '.join(tried)}.")
    if engine in ("mlx", "llamacpp"):
        local = MLXEngine() if engine == "mlx" else LlamaCppEngine()
        if local.available():
            return local
        raise LocalLLMUnavailable(f"engine {engine!r} is not available. {INSTALL_HINT}")
    if engine in ("ollama", "openai"):
        servers = [HTTPEngine(base_url=url, api=api, timeout=timeout, probe_timeout=probe_timeout)
                   for api, url in _servers(base_url) if api == engine]
        for server in servers:
            if server.available():
                return server
        where = " or ".join(server.base_url for server in servers)
        raise LocalLLMUnavailable(f"engine {engine!r} is not available at {where}. {INSTALL_HINT}")
    raise ValueError(f"unknown engine {engine!r}; expected 'auto' or one of {list(_ENGINE_NAMES)}")


def available_engines(*, probe_servers: bool = True, base_url: str | None = None,
                      probe_timeout: float = _PROBE_TIMEOUT) -> list[str]:
    """Names of the engines available here, without repeats, in the order ``auto`` tries them.

    With ``probe_servers=False`` no server is probed and only ``mlx`` and ``llamacpp`` can be
    listed. A server name is listed once the first of its URLs answers.
    """
    names = [local.name for local in (MLXEngine(), LlamaCppEngine()) if local.available()]
    if probe_servers:
        for api, url in _servers(base_url):
            if api not in names and HTTPEngine(base_url=url, api=api,
                                               probe_timeout=probe_timeout).available():
                names.append(api)
    return names
