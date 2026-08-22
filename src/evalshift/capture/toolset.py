"""Normalise a user-supplied toolset into canonical shape, and fingerprint it.

Every provider shapes "the list of tools an agent was offered" differently: Anthropic's
``{name, description, input_schema}``, OpenAI's ``{type: "function", function: {...}}``, and
Gemini's ``types.Tool(function_declarations=[...])``. :func:`normalize_tools` reduces any of
those -- or a bare sequence (list, tuple, ...) mixing them -- to the one shape the rest of the
product agrees on:
``{name, description, input_schema}``. :func:`fingerprint_tools` then content-addresses an
already-normalised list, so a toolset can be written once (a sidecar, added in a later phase)
and referenced by every ``model_call`` that used it, instead of inlining tens of KB of schema
into every capture.

Tool definitions are config, not payload -- the same class as ``generation_config``
(``evalshift.capture.generation``) -- and are never passed through the redactor. Unlike
``generation_config``, toolsets are **not** allow-listed: an ``input_schema`` is arbitrary user
JSON needed in full to dispatch, so normalisation here only recognises or rejects *shapes*, never
prunes keys *within* a schema.

Gemini support is duck-typed via ``getattr`` only -- this module, like the rest of the SDK, never
imports ``google.genai`` (D-deps).

A schema value itself can also be duck-typed. A real ``google.genai.types.FunctionDeclaration``'s
``parameters`` is a pydantic ``Schema`` object, not a dict, and is not JSON-serialisable as-is --
``json.dumps`` raises ``TypeError: Object of type Schema is not JSON serializable`` on one
directly. :func:`_coerce_schema` recognises a ``model_dump`` method the same duck-typed way
(``getattr``, never an import) and calls it to recover JSON-safe primitives; anything with neither
a dict/list/scalar shape nor ``model_dump`` degrades to ``{}`` rather than being passed through by
reference and later breaking :func:`fingerprint_tools`.

Stdlib only (D-deps).
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable, Sequence
from typing import Any

#: Values already safe to hand to ``json.dumps`` unchanged. Anything else -- notably a
#: duck-typed pydantic model such as a real ``google.genai.types.Schema`` -- needs coercion in
#: :func:`_coerce_schema` before :func:`fingerprint_tools` can serialise it.
_JSON_SAFE_TYPES = (dict, list, str, int, float, bool, type(None))

#: Excluded from the "bare sequence of tools" branch of :func:`normalize_tools` even though each
#: is technically a ``Sequence``: iterating one yields characters/bytes, not tool-shaped items,
#: so ``""`` would otherwise wrongly normalise to ``[]`` (zero characters) instead of being
#: rejected, and a non-empty string/bytes value would be shredded one character at a time.
_NOT_A_TOOL_SEQUENCE = (str, bytes, bytearray)


def _coerce_schema(value: Any) -> Any:
    """Coerce one tool's schema source value (``input_schema`` / ``parameters``) to something
    :func:`fingerprint_tools` -- a plain ``json.dumps`` -- can always serialise, safe to keep
    past the lifetime of the caller's own object.

    A JSON-primitive value (``dict``, ``list``, ``str``, ``int``, ``float``, ``bool``, ``None``)
    is preserved as given -- including a falsy one (``[]``, ``0``, ``False``, ``""``): those are
    real values a caller can genuinely supply for a schema, not a sentinel for "missing". Only
    ``None`` -- the one sentinel every call site's ``.get(key)`` / ``getattr(..., None)`` above
    this function actually uses for "no schema was given at all" -- collapses to ``{}``.

    A ``dict``/``list`` is deep-copied rather than passed through by reference. A caller's
    toolset can outlive normalisation by a whole session (``_current_toolset``) or a LangChain
    handler's whole lifetime; without the copy, an in-place mutation on the caller's side (e.g.
    appending a property to a schema dict after the call that normalised it has already returned)
    would silently change an *already-normalised, already-fingerprinted* toolset too, and a
    sidecar written later from the mutated object would no longer hash to its own filename. A
    scalar (``str`` / ``int`` / ``float`` / ``bool`` / ``None``) is immutable and safe to return
    as-is either way; ``copy.deepcopy`` is applied uniformly across the branch rather than
    special-cased by mutability, since it is a cheap no-op for atomic immutable types.

    ``copy.deepcopy`` itself is called under a ``try``/``except``, same as ``model_dump`` below:
    a ``dict``/``list`` is only JSON-*safe* at its own top level by construction here (that is
    all this ``isinstance`` check inspects) -- something not itself JSON/deepcopy-safe can still
    be nested arbitrarily deep inside it, since ``input_schema`` is unvalidated, un-allow-listed
    user JSON (D-toolset). Most such values (a nested ``set``, ``Enum``, ``datetime``) copy fine
    and are only caught later by :func:`fingerprint_tools`'s ``json.dumps`` -- guarded by its own
    caller, :func:`evalshift.capture.api._normalize_toolset`. A few (a lock, an open file, a
    generator, or nesting deep enough to exhaust the recursion limit) raise on ``deepcopy``
    itself, one step *before* that guard. Catching it here, rather than leaving it to propagate,
    keeps this function's -- and therefore :func:`normalize_tools`'s -- documented "never raises"
    contract true regardless of what turns out to be nested inside a schema. Two call sites in
    ``capture/api.py`` (``record_model_call``, ``_ModelCallRecorder._open``) call
    ``normalize_tools`` with no wrapping guard of their own and depend on that contract holding.

    Neither change affects the fingerprint of any schema the test suite pins a hash against --
    the Anthropic, OpenAI, and dict-``parameters`` Gemini stand-in fixtures all use non-empty,
    non-``None`` dicts, so this branch returns their content unchanged (deep-copied, never
    aliased, but ``json.dumps`` serialises identical bytes either way) -- ``THREE_TOOL_FINGERPRINT``
    / ``EMPTY_TOOLSET_FINGERPRINT`` are unaffected.

    A genuine ``google.genai.types.Schema`` -- what a real ``FunctionDeclaration.parameters``
    actually is -- is a pydantic model, not a dict, and ``json.dumps`` raises ``TypeError:
    Object of type Schema is not JSON serializable`` on one directly. It's duck-typed via
    ``getattr(value, "model_dump", None)`` -- this module never imports ``pydantic`` or
    ``google.genai`` (D-deps) -- and, when present, called with ``mode="json"`` so nested values
    (including nested ``Type`` enum members -- pydantic dumps those recursively) come back as
    plain JSON primitives instead of raising. Its result is a dict ``model_dump`` builds fresh on
    every call, never the caller's own object, so it needs no separate deep copy to avoid aliasing.

    Anything else -- no dict/list/scalar shape, no callable ``model_dump``, or a ``model_dump``
    that itself raises or hands back something that is not a dict -- degrades to ``{}``, the
    same "no usable schema" result normalize_tools already returns for a missing schema, rather
    than passing an opaque, non-JSON-safe object through by reference. Never raises.
    """
    if isinstance(value, _JSON_SAFE_TYPES):
        if value is None:
            return {}
        try:
            return copy.deepcopy(value)
        except Exception:
            return {}
    model_dump = getattr(value, "model_dump", None)
    if not callable(model_dump):
        return {}
    try:
        dumped = model_dump(exclude_none=True, mode="json")
    except Exception:
        return {}
    return dumped if isinstance(dumped, dict) else {}


def _normalize_dict(item: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Normalise one Anthropic- or OpenAI-shaped tool dict. ``None`` if neither shape matches."""
    function = item.get("function")
    if item.get("type") == "function" and isinstance(function, dict):
        name = function.get("name")
        if not isinstance(name, str) or not name:
            return None
        return [
            {
                "name": name,
                "description": function.get("description") or "",
                "input_schema": _coerce_schema(function.get("parameters")),
            }
        ]
    name = item.get("name")
    if isinstance(name, str) and name:
        return [
            {
                "name": name,
                "description": item.get("description") or "",
                "input_schema": _coerce_schema(item.get("input_schema")),
            }
        ]
    return None


def _normalize_gemini_tool(item: Any) -> list[dict[str, Any]] | None:
    """Normalise one duck-typed ``google.genai.types.Tool``. ``None`` if not that shape.

    Recognised solely by an iterable ``function_declarations`` -- the one Gemini ``Tool``
    variant this module handles. A ``Tool`` built around a different built-in tool, such as
    ``google_search``, is expected to leave ``function_declarations`` unset or ``None`` (this
    module never imports ``google.genai`` to confirm that). Either way, ``getattr(item, ...,
    None)`` plus the ``Iterable`` check below rejects it without raising.
    """
    declarations = getattr(item, "function_declarations", None)
    if not isinstance(declarations, Iterable):
        return None
    tools: list[dict[str, Any]] = []
    for declaration in declarations:
        name = getattr(declaration, "name", None)
        if not isinstance(name, str) or not name:
            return None
        tools.append(
            {
                "name": name,
                "description": getattr(declaration, "description", None) or "",
                "input_schema": _coerce_schema(getattr(declaration, "parameters", None)),
            }
        )
    return tools


def _normalize_one(item: Any) -> list[dict[str, Any]] | None:
    """Normalise a single tool-shaped item to zero or more canonical tool dicts.

    Zero is a legitimate result (a Gemini ``Tool`` with an empty ``function_declarations``);
    ``None`` means ``item`` matched none of the recognised shapes at all.
    """
    if isinstance(item, dict):
        return _normalize_dict(item)
    return _normalize_gemini_tool(item)


def normalize_tools(raw: Any) -> list[dict[str, Any]] | None:
    """Normalise a user-supplied toolset to a list of ``{name, description, input_schema}``.

    Accepts the shapes callers actually hold:

    * Anthropic -- ``{name, description, input_schema}``
    * OpenAI -- ``{type: "function", function: {name, description, parameters}}``
    * Gemini ``types.Tool`` with ``function_declarations`` -- duck-typed via ``getattr``, never
      imported (the SDK does not depend on ``google-genai``)
    * a bare sequence (a ``list``, a ``tuple``, or any other non-``str``/``bytes`` ``Sequence``)
      mixing any of the above -- a Gemini ``Tool`` can itself expand into more than one canonical
      tool, one per function declaration. A ``tuple`` is accepted on equal footing with a
      ``list`` -- "the tools I was offered" is a realistic value to hold as either, and rejecting
      one wholesale over the other would be a silent, easy-to-hit data loss trap, not a
      meaningful signal about the caller's intent.

    Args:
        raw: Any value.

    Returns:
        A new list of canonical tool dicts, or ``None`` when ``raw`` -- or, when ``raw`` is a
        sequence, any single element of it -- matches none of the recognised shapes.

        ``[]`` (or ``()``) normalises to ``[]``, not ``None``: an agent offered no tools is a
        real, first-class toolset, not a value this function failed to parse. A sequence
        containing even one unrecognisable element normalises to ``None`` as a whole, on
        purpose: a toolset silently missing one tool is exactly the kind of wrong-but-plausible-
        looking data this module exists to prevent, so we refuse to guess rather than publish a
        partial list as if it were the complete one the agent was offered.

    Never raises.
    """
    if isinstance(raw, Sequence) and not isinstance(raw, _NOT_A_TOOL_SEQUENCE):
        tools: list[dict[str, Any]] = []
        for item in raw:
            one = _normalize_one(item)
            if one is None:
                return None
            tools.extend(one)
        return tools
    return _normalize_one(raw)


def fingerprint_tools(normalized: list[dict[str, Any]]) -> str:
    """Content-address an already-normalised tool list.

    Reuses ``evalshift-server/BUNDLE_SPEC.md``'s Hashing section verbatim, so one hashing rule
    holds product-wide (``evalshift-cli`` ports the identical steps):

    1. (done by the caller) normalise each tool to ``{name, description, input_schema}``.
    2. Sort the tool list by ``name``.
    3. ``json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)``.
    4. SHA-256 the UTF-8 bytes, hex-encode, prefix ``sha256:``.

    Sorting the list (step 2) is required *in addition to* ``sort_keys=True`` -- ``sort_keys``
    only orders each dict's own keys, never list element order, so two logically-identical
    toolsets captured in a different call order would otherwise fingerprint differently.

    Args:
        normalized: A list of ``{name, description, input_schema}`` dicts, e.g. the output of
            :func:`normalize_tools` (never ``None`` -- callers check that first; this function's
            precondition is that normalisation already happened).

    Returns:
        ``"sha256:" + hex digest`` of the tool list's canonical JSON form.
    """
    ordered = sorted(normalized, key=lambda tool: tool["name"])
    canonical = json.dumps(ordered, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


__all__ = ["fingerprint_tools", "normalize_tools"]
