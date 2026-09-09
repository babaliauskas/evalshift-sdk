"""Extract the tool calls a model *requested* from an already-returned provider response.

Three different things get called "tools" around a model call, and this module handles exactly
one of them:

* **offered** -- the toolset the call was given (``tools=``; :mod:`evalshift.capture.toolset`);
* **requested** -- what the model asked for in its response (*this module*);
* **executed** -- what the application actually ran (``@capture.tool``).

:func:`extract_requested_tool_calls` turns a provider response into the canonical list of
requested calls -- ``{"name", "arguments", "call_id"}`` per item, in response order -- ready to
hand to ``record_model_call(requested_tool_calls=...)``. Because the offered/requested/executed
triple is what makes a capture answer "did the model ask for a tool it was never given?" and
"did the app run something the model never asked for?", it matters that a list we publish is
*complete*: see the refusal rule below.

Like :mod:`evalshift.capture.toolset`, this is pure duck-typed dict/attribute walking. The SDK
never imports ``openai``, ``anthropic``, or ``google.genai`` -- not even guarded (D-deps) -- so
every shape below is recognised structurally:

* **OpenAI Chat Completions** -- ``choices[0].message.tool_calls[*]`` of
  ``{id, type: "function", function: {name, arguments}}``, where ``arguments`` is a JSON
  *string*; plus the deprecated single-call ``message.function_call`` form.
* **OpenAI Responses** -- top-level ``output[*]`` items with ``type == "function_call"``, of
  ``{name, arguments (JSON string), call_id}``.
* **Anthropic Messages** -- ``content[*]`` blocks with ``type == "tool_use"``, of
  ``{id, name, input}``; ``input`` is already an object, not a string.
* **Gemini** -- ``candidates[0].content.parts[*].functionCall`` (camelCase, as the REST API and
  a serialised response spell it) or ``.function_call`` (snake_case, as the python SDK's
  ``to_dict()`` spells it), of ``{name, args}`` plus an optional newer ``id``.

The input may be a dict (an already-serialised response), an object exposing ``model_dump()`` or
``to_dict()`` (converted to a dict first), or the provider response object itself (walked by
attribute). All three go through the same recognisers: :func:`_get` reads a key from a dict and
an attribute from anything else, so a half-serialised response -- a dict whose leaves are still
objects, say -- works too.

**``[]`` and ``None`` mean different things, on purpose.** ``[]`` is a value: this *was* a
recognised provider response and the model requested no tools. ``None`` is an absence: this did
not look like a provider response at all (or one of its tool calls was unreadable), so nothing
is asserted about what the model requested. Callers must not collapse the two -- recording ``[]``
for an unrecognised response would assert "the model asked for nothing" on no evidence.

**A tool call we cannot read poisons the whole response.** When an entry is positively
identified as a tool call (an OpenAI ``tool_calls`` entry, a ``tool_use`` block, a
``functionCall`` part, a ``function_call`` output item) but has no usable name, the response
resolves to ``None`` rather than to the other calls beside it. Same reasoning as
:func:`evalshift.capture.toolset.normalize_tools`: a requested-call list silently missing one
call reads as complete and is wrong in exactly the direction the capture is used to reason about
("the model never asked for that"), so we refuse instead of publishing a partial list. Blocks
that are *not* tool calls (a text block, a reasoning item, a ``message`` item) are skipped
normally -- they are not gaps.

Degradations short of that are recorded as an empty ``arguments`` dict and a ``debug`` log:
unparseable JSON arguments, JSON that parses to something other than an object, and an
already-parsed ``input``/``args`` that is not an object. Arguments are passed through
:func:`evalshift.capture.generation.jsonable`, which both guarantees the value survives the
sink's plain ``json.dumps`` and copies it, so a later mutation of the caller's response object
cannot change an already-recorded capture.

Never raises. Stdlib only (D-deps).
"""

from __future__ import annotations

import json
from typing import Any

from evalshift import safety
from evalshift.capture.generation import jsonable


def _get(obj: Any, key: str) -> Any:
    """Read ``key`` from a dict, or the same-named attribute from anything else.

    ``None`` when absent, when the lookup itself raises (a provider object's property can), or
    when ``obj`` is not subscriptable/attributed at all. This is what lets one set of
    recognisers walk a serialised dict, a live response object, and any mixture of the two.
    """
    try:
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)
    except Exception:
        return None


def _items(value: Any) -> list[Any] | None:
    """Return ``value`` as a list when it is a list/tuple, else ``None`` (a shape mismatch)."""
    if isinstance(value, (list, tuple)):
        return list(value)
    return None


def _coerce_arguments(value: Any) -> dict[str, Any]:
    """Coerce an already-parsed arguments value to a JSON-safe dict copy.

    A non-dict (Anthropic's ``input`` or Gemini's ``args`` arriving as a scalar or a list, or a
    JSON arguments string that parsed to a non-object) degrades to ``{}`` with a ``debug`` log
    rather than being recorded as-is: every consumer of ``requested_tool_calls`` types
    ``arguments`` as an object. ``None``/missing is the normal "no arguments" case and is not
    logged.

    :func:`~evalshift.capture.generation.jsonable` is reused for the dict case so the result is
    both a deep copy (never aliasing the caller's response) and guaranteed serialisable by the
    sink's plain ``json.dumps``. It recurses, so a self-referential arguments dict raises
    ``RecursionError`` -- caught here, like everything else, in favour of ``{}``.
    """
    if not isinstance(value, dict):
        if value is not None:
            safety.logger.debug(
                "evalshift: requested tool call arguments were not an object (%s); recorded as {}",
                type(value).__name__,
            )
        return {}
    try:
        coerced = jsonable(value)
    except Exception:
        safety.logger.debug(
            "evalshift: requested tool call arguments could not be JSON-coerced; recorded as {}",
            exc_info=True,
        )
        return {}
    return coerced if isinstance(coerced, dict) else {}


def _json_arguments(value: Any) -> dict[str, Any]:
    """Coerce an OpenAI-style ``arguments`` value -- a JSON *string* -- to a dict.

    Both OpenAI surfaces send arguments as text the caller is expected to ``json.loads``. An
    empty or whitespace-only string, and a missing value, are the "no arguments" case (``{}``,
    unlogged); anything unparseable degrades to ``{}`` with a ``debug`` log. A value that is
    already a dict (some gateways and serialisers pre-parse it) is accepted through
    :func:`_coerce_arguments` unchanged.
    """
    if isinstance(value, str):
        if not value.strip():
            return {}
        try:
            parsed = json.loads(value)
        except Exception:
            safety.logger.debug(
                "evalshift: requested tool call arguments were not valid JSON; recorded as {}"
            )
            return {}
        return _coerce_arguments(parsed)
    return _coerce_arguments(value)


def _call(name: Any, arguments: dict[str, Any], call_id: Any) -> dict[str, Any] | None:
    """Build one canonical ``{name, arguments, call_id}`` item, or ``None`` if unreadable.

    A tool call with no usable name is unreadable: ``None`` here propagates all the way out of
    :func:`extract_requested_tool_calls` (see the module docstring's refusal rule). A ``call_id``
    that is not a non-empty string -- Gemini's REST shape has no id at all, and the legacy
    OpenAI ``function_call`` form has none either -- normalises to ``None``.
    """
    if not isinstance(name, str) or not name:
        safety.logger.debug(
            "evalshift: a requested tool call had no usable name; ignoring the whole response"
        )
        return None
    return {
        "name": name,
        "arguments": arguments,
        "call_id": call_id if isinstance(call_id, str) and call_id else None,
    }


def _from_openai_chat(response: Any) -> list[dict[str, Any]] | None:
    """OpenAI Chat Completions: ``choices[0].message`` -- ``tool_calls`` or ``function_call``.

    Only the first choice is read: ``n > 1`` returns independent alternative completions, and
    concatenating their tool calls would invent a turn the model never proposed. ``None`` when
    there is no ``choices`` list at all (not this shape) or the first choice has no message.
    """
    choices = _items(_get(response, "choices"))
    if choices is None:
        return None
    if not choices:
        return []
    message = _get(choices[0], "message")
    if message is None:
        return None

    raw_calls = _get(message, "tool_calls")
    if raw_calls is not None:
        entries = _items(raw_calls)
        if entries is None:
            return None
        calls: list[dict[str, Any]] = []
        for entry in entries:
            function = _get(entry, "function")
            call = _call(
                _get(function, "name"),
                _json_arguments(_get(function, "arguments")),
                _get(entry, "id"),
            )
            if call is None:
                return None
            calls.append(call)
        return calls

    legacy = _get(message, "function_call")
    if legacy is not None:
        call = _call(_get(legacy, "name"), _json_arguments(_get(legacy, "arguments")), None)
        return None if call is None else [call]
    return []


def _from_openai_responses(response: Any) -> list[dict[str, Any]] | None:
    """OpenAI Responses: top-level ``output[*]`` items of ``type == "function_call"``.

    Other item types (``message``, ``reasoning``, built-in tool calls) are skipped rather than
    treated as gaps. The stable identifier to record is ``call_id`` -- the value a
    ``function_call_output`` is later keyed by -- not the item's own ``id``.
    """
    entries = _items(_get(response, "output"))
    if entries is None:
        return None
    calls: list[dict[str, Any]] = []
    for entry in entries:
        if _get(entry, "type") != "function_call":
            continue
        call = _call(
            _get(entry, "name"),
            _json_arguments(_get(entry, "arguments")),
            _get(entry, "call_id"),
        )
        if call is None:
            return None
        calls.append(call)
    return calls


def _from_anthropic(response: Any) -> list[dict[str, Any]] | None:
    """Anthropic Messages: ``content[*]`` blocks of ``type == "tool_use"``.

    ``input`` is already an object (Anthropic parses it server-side), so it goes through
    :func:`_coerce_arguments` rather than the JSON-string path. Text and thinking blocks are
    skipped. ``None`` when ``content`` is not a list -- notably a plain string, which is the
    *request* shape, not a response.
    """
    blocks = _items(_get(response, "content"))
    if blocks is None:
        return None
    calls: list[dict[str, Any]] = []
    for block in blocks:
        if _get(block, "type") != "tool_use":
            continue
        call = _call(
            _get(block, "name"), _coerce_arguments(_get(block, "input")), _get(block, "id")
        )
        if call is None:
            return None
        calls.append(call)
    return calls


def _from_gemini(response: Any) -> list[dict[str, Any]] | None:
    """Gemini: ``candidates[0].content.parts[*]`` carrying a function call.

    Both spellings are accepted -- ``functionCall`` (REST/serialised) and ``function_call``
    (python SDK ``to_dict()`` / response objects) -- since the same response can reach us either
    way. Only the first candidate is read, for the same reason as OpenAI's first choice.
    ``args`` is already an object. There is no call id in the REST shape; the newer optional
    ``id`` field is used when present, otherwise ``call_id`` is ``None``.
    """
    candidates = _items(_get(response, "candidates"))
    if candidates is None:
        return None
    if not candidates:
        return []
    parts = _items(_get(_get(candidates[0], "content"), "parts"))
    if parts is None:
        return None
    calls: list[dict[str, Any]] = []
    for part in parts:
        function_call = _get(part, "functionCall")
        if function_call is None:
            function_call = _get(part, "function_call")
        if function_call is None:
            continue
        call = _call(
            _get(function_call, "name"),
            _coerce_arguments(_get(function_call, "args")),
            _get(function_call, "id"),
        )
        if call is None:
            return None
        calls.append(call)
    return calls


#: Tried in order; the first one that recognises the response wins. The top-level keys they key
#: off (``choices`` / ``output`` / ``content`` / ``candidates``) are disjoint across providers,
#: so the order is documentation, not precedence.
_RECOGNISERS = (_from_openai_chat, _from_openai_responses, _from_anthropic, _from_gemini)


def _as_dict(response: Any) -> Any:
    """Best-effort conversion of a response object to a dict, for the recognisers to walk.

    A dict passes through. Otherwise a ``model_dump()`` (pydantic -- the OpenAI and Anthropic
    SDK response classes) or ``to_dict()`` (google-genai) method is duck-typed via ``getattr``
    and called; both are recognised by shape only, never imported (D-deps). Anything else -- no
    such method, one that raises, one returning a non-dict -- is returned unchanged, and
    :func:`_get` then walks it by attribute instead. Never raises.
    """
    if isinstance(response, dict):
        return response
    for method_name in ("model_dump", "to_dict"):
        try:
            method = getattr(response, method_name, None)
            if not callable(method):
                continue
            dumped = method()
        except Exception:
            continue
        if isinstance(dumped, dict):
            return dumped
    return response


def extract_requested_tool_calls(response: Any) -> list[dict[str, Any]] | None:
    """Extract the tool calls a model requested, from its provider response.

    Args:
        response: A provider response in any of the forms callers actually hold -- an
            already-serialised dict, an object exposing ``model_dump()`` / ``to_dict()``, or the
            provider's own response object (walked by attribute). OpenAI Chat Completions
            (including the deprecated ``function_call`` form), OpenAI Responses, Anthropic
            Messages, and Gemini ``generateContent`` shapes are recognised.

    Returns:
        A list of ``{"name": str, "arguments": dict, "call_id": str | None}`` items in response
        order -- exactly those three keys, always -- ready for
        ``record_model_call(requested_tool_calls=...)``.

        ``[]`` and ``None`` are **not** interchangeable:

        * ``[]`` -- a recognised response in which the model requested no tools. A real,
          deliberate value: "the model asked for nothing."
        * ``None`` -- the value did not look like a provider response at all, or a tool call
          inside a recognised one had no usable name (in which case the whole response is
          refused rather than reported short a call). Nothing is asserted about what the model
          requested; a caller should record no ``requested_tool_calls`` at all, never ``[]``.

        Unparseable JSON arguments, JSON that parses to a non-object, and a non-object
        ``input``/``args`` all degrade to ``{}`` for that one call, logged at ``debug``.

    Never raises: any unexpected shape or exploding attribute resolves to ``None``.
    """
    try:
        source = _as_dict(response)
        for recognise in _RECOGNISERS:
            calls = recognise(source)
            if calls is not None:
                return calls
    except Exception:
        safety.logger.debug(
            "evalshift: could not extract requested tool calls from the response (swallowed)",
            exc_info=True,
        )
        return None
    return None


__all__ = ["extract_requested_tool_calls"]
