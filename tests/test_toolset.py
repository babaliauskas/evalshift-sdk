"""Unit tests for toolset normalisation + fingerprinting (``evalshift.capture.toolset``).

Two module-level constants below -- ``THREE_TOOL_FINGERPRINT`` and
``EMPTY_TOOLSET_FINGERPRINT`` -- are the shared vector referenced by the implementation plan:
Task 2 (``evalshift-cli``) asserts the identical values, copied verbatim, so the SDK's and the
CLI's independent hashing implementations cannot silently drift apart.
"""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from typing import Any

from evalshift.capture.toolset import fingerprint_tools, normalize_tools

# --- The shared vector ----------------------------------------------------------------------
#
# One tool per real-world shape -- Anthropic (bare dict), OpenAI (type/function wrapper), and a
# duck-typed stand-in for google.genai.types.Tool (never the real class: this module, like the
# rest of the SDK, never imports google.genai -- D-deps). All three tools have different names
# so order-independence tests are meaningful.

ANTHROPIC_TOOL: dict[str, Any] = {
    "name": "get_schedule",
    "description": "Look up a user's schedule for a given date.",
    "input_schema": {
        "type": "object",
        "properties": {"date": {"type": "string"}},
        "required": ["date"],
    },
}

OPENAI_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "add_task",
        "description": "Add a task to the user's to-do list.",
        "parameters": {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
        },
    },
}


def _gemini_tool() -> SimpleNamespace:
    """A fresh duck-typed stand-in per call -- fixtures must not be shared mutable state."""
    declaration = SimpleNamespace(
        name="send_email",
        description="Send an email to a recipient.",
        parameters={
            "type": "object",
            "properties": {"to": {"type": "string"}, "body": {"type": "string"}},
            "required": ["to", "body"],
        },
    )
    return SimpleNamespace(function_declarations=[declaration])


THREE_TOOL_RAW: list[Any] = [ANTHROPIC_TOOL, OPENAI_TOOL, _gemini_tool()]

# What THREE_TOOL_RAW must normalise to -- written out by hand so the test is not just
# "normalize_tools output compared against itself". normalize_tools preserves input order (it
# is fingerprint_tools, not normalize_tools, that sorts) -- so this list stays in
# Anthropic/OpenAI/Gemini input order, not alphabetical.
THREE_TOOL_NORMALIZED: list[dict[str, Any]] = [
    {
        "name": "get_schedule",
        "description": "Look up a user's schedule for a given date.",
        "input_schema": {
            "type": "object",
            "properties": {"date": {"type": "string"}},
            "required": ["date"],
        },
    },
    {
        "name": "add_task",
        "description": "Add a task to the user's to-do list.",
        "input_schema": {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
        },
    },
    {
        "name": "send_email",
        "description": "Send an email to a recipient.",
        "input_schema": {
            "type": "object",
            "properties": {"to": {"type": "string"}, "body": {"type": "string"}},
            "required": ["to", "body"],
        },
    },
]

# Pinned by running the real implementation once and copying its output -- a content hash
# cannot be derived by inspection. Task 2 (evalshift-cli) copies both constants verbatim; if a
# future edit to either implementation changes either hash, one of the two repos goes red.
THREE_TOOL_FINGERPRINT = "sha256:8128183b3b2871d3887b7a21b3ac2e1928d939180e9ea12ec583e93bb1ccebde"
EMPTY_TOOLSET_FINGERPRINT = (
    "sha256:4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
)


# --- normalize_tools: each shape, individually -----------------------------------------------


def test_normalize_anthropic_shape() -> None:
    assert normalize_tools(ANTHROPIC_TOOL) == [
        {
            "name": "get_schedule",
            "description": "Look up a user's schedule for a given date.",
            "input_schema": ANTHROPIC_TOOL["input_schema"],
        }
    ]


def test_normalize_openai_shape() -> None:
    assert normalize_tools(OPENAI_TOOL) == [
        {
            "name": "add_task",
            "description": "Add a task to the user's to-do list.",
            "input_schema": OPENAI_TOOL["function"]["parameters"],
        }
    ]


def test_normalize_gemini_duck_typed_shape() -> None:
    assert normalize_tools(_gemini_tool()) == [
        {
            "name": "send_email",
            "description": "Send an email to a recipient.",
            "input_schema": {
                "type": "object",
                "properties": {"to": {"type": "string"}, "body": {"type": "string"}},
                "required": ["to", "body"],
            },
        }
    ]


def test_normalize_gemini_tool_expands_multiple_function_declarations() -> None:
    """A single Gemini ``Tool`` can bundle several declarations -- each becomes its own tool."""
    tool = SimpleNamespace(
        function_declarations=[
            SimpleNamespace(name="tool_a", description="A", parameters={"a": 1}),
            SimpleNamespace(name="tool_b", description="B", parameters={"b": 2}),
        ]
    )
    assert normalize_tools(tool) == [
        {"name": "tool_a", "description": "A", "input_schema": {"a": 1}},
        {"name": "tool_b", "description": "B", "input_schema": {"b": 2}},
    ]


def test_normalize_gemini_tool_with_no_declarations_contributes_nothing() -> None:
    """An empty ``function_declarations`` list is a recognised, valid, zero-tool contribution."""
    tool = SimpleNamespace(function_declarations=[])
    assert normalize_tools(tool) == []


def test_normalize_bare_list_of_mixed_shapes() -> None:
    assert normalize_tools(THREE_TOOL_RAW) == THREE_TOOL_NORMALIZED


# --- F4: a bare tuple -- "a realistic slip" for a list -- must normalise like a list, not be
#     rejected wholesale as a single unrecognised item -----------------------------------------


def test_normalize_bare_tuple_of_mixed_shapes() -> None:
    """The exact THREE_TOOL_RAW vector, as a tuple instead of a list -- must normalise
    identically. Pre-fix, `isinstance(raw, list)` was False for a tuple, so the whole tuple fell
    through to `_normalize_one`, which cannot recognise a tuple as a single Gemini-shaped tool
    either, and the entire toolset was silently rejected (`None`)."""
    assert normalize_tools(tuple(THREE_TOOL_RAW)) == THREE_TOOL_NORMALIZED


def test_normalize_empty_tuple_returns_empty_list_not_none() -> None:
    result = normalize_tools(())
    assert result == []
    assert result is not None


def test_normalize_tuple_with_one_unrecognisable_item_is_none_not_partial() -> None:
    """Mirrors test_normalize_list_with_one_unrecognisable_item_is_none_not_partial for tuples:
    one bad element still invalidates the whole call, exactly as for a list."""
    assert normalize_tools((ANTHROPIC_TOOL, "garbage")) is None


def test_normalize_does_not_treat_a_string_as_a_sequence_of_characters() -> None:
    """A `str` is technically a `Sequence` -- must stay excluded, or `normalize_tools("")` would
    wrongly normalise to `[]` (iterating zero characters) instead of being rejected as garbage,
    and a non-empty string would be shredded into one-character "tools" instead of being
    rejected as a whole. Both would silently misreport what the agent was offered."""
    assert normalize_tools("") is None
    assert normalize_tools("garbage") is None


def test_normalize_does_not_treat_bytes_as_a_sequence_of_ints() -> None:
    assert normalize_tools(b"") is None
    assert normalize_tools(b"garbage") is None


# --- normalize_tools: a Gemini `parameters` value can itself be duck-typed, not a dict --------
#
# A real `google.genai.types.FunctionDeclaration(...).parameters` is a pydantic `Schema` object,
# not a dict -- confirmed empirically against the real google-genai package, where `json.dumps`
# on one raises `TypeError: Object of type Schema is not JSON serializable`. These stand-ins
# reproduce that duck-typed surface -- no dict interface, only `model_dump` -- without importing
# google-genai or pydantic, which this suite must not depend on any more than the module under
# test does (see its docstring).


class _GeminiSchemaStandIn:
    """Stand-in for a real `google.genai.types.Schema`: no dict interface, only `model_dump`."""

    def __init__(self, dumped: dict[str, Any]) -> None:
        self._dumped = dumped
        self.model_dump_calls: list[dict[str, Any]] = []

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        self.model_dump_calls.append(kwargs)
        return self._dumped


class _RaisingModelDumpStandIn:
    """A `model_dump` that itself raises -- must degrade to `{}`, never propagate."""

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("boom -- simulates a pydantic model_dump failure")


def _gemini_tool_with_parameters(parameters: Any) -> SimpleNamespace:
    declaration = SimpleNamespace(
        name="send_email", description="Send an email to a recipient.", parameters=parameters
    )
    return SimpleNamespace(function_declarations=[declaration])


def test_normalize_gemini_schema_object_is_coerced_via_model_dump() -> None:
    """The common real-world case: `parameters` is a pydantic-like `Schema`, not a dict."""
    dumped: dict[str, Any] = {
        "type": "OBJECT",  # what `mode="json"` resolves a nested `Type` enum member to
        "properties": {"to": {"type": "STRING"}, "body": {"type": "STRING"}},
        "required": ["to", "body"],
    }
    schema = _GeminiSchemaStandIn(dumped)
    result = normalize_tools(_gemini_tool_with_parameters(schema))
    assert result == [
        {
            "name": "send_email",
            "description": "Send an email to a recipient.",
            "input_schema": dumped,
        }
    ]
    # The exact call the reviewer verified resolves nested `Type` enums to JSON primitives.
    assert schema.model_dump_calls == [{"exclude_none": True, "mode": "json"}]


def test_normalize_gemini_schema_object_without_model_dump_degrades_to_empty_schema() -> None:
    """Neither a dict interface nor `model_dump` -- genuinely uncoercible. Must degrade, not
    pass the raw object through by reference (that pass-through is the bug being fixed).
    """
    result = normalize_tools(_gemini_tool_with_parameters(object()))
    assert result == [
        {"name": "send_email", "description": "Send an email to a recipient.", "input_schema": {}}
    ]


def test_normalize_gemini_schema_model_dump_that_raises_degrades_to_empty_schema() -> None:
    """`model_dump` itself raising must not propagate -- normalize_tools never raises."""
    result = normalize_tools(_gemini_tool_with_parameters(_RaisingModelDumpStandIn()))
    assert result == [
        {"name": "send_email", "description": "Send an email to a recipient.", "input_schema": {}}
    ]


def test_fingerprint_tools_of_normalized_gemini_schema_object_does_not_raise() -> None:
    """End-to-end: `fingerprint_tools(normalize_tools(...))` is exactly how Task 6 chains them,
    and is what a real Gemini `FunctionDeclaration.parameters` broke pre-fix -- `normalize_tools`
    alone tolerated the Schema object, but `fingerprint_tools`'s plain `json.dumps` raised on it.
    """
    schema = _GeminiSchemaStandIn({"type": "OBJECT", "properties": {}})
    normalized = normalize_tools(_gemini_tool_with_parameters(schema))
    assert normalized is not None
    fp = fingerprint_tools(normalized)
    assert fp.startswith("sha256:")
    hex_part = fp.removeprefix("sha256:")
    assert len(hex_part) == 64
    int(hex_part, 16)  # raises ValueError if this is not valid hex


# --- normalize_tools: the empty toolset is a value, never an absence -------------------------


def test_normalize_empty_list_returns_empty_list_not_none() -> None:
    """The one thing most likely to go wrong: `[]` must stay `[]`, not collapse to `None`."""
    result = normalize_tools([])
    assert result == []
    assert result is not None


# --- normalize_tools: a missing description becomes "", the key is never dropped -------------


def test_normalize_missing_description_becomes_empty_string_anthropic() -> None:
    tool = {"name": "no_description", "input_schema": {}}
    assert normalize_tools(tool) == [
        {"name": "no_description", "description": "", "input_schema": {}}
    ]


def test_normalize_missing_description_becomes_empty_string_openai() -> None:
    tool = {"type": "function", "function": {"name": "no_description", "parameters": {}}}
    assert normalize_tools(tool) == [
        {"name": "no_description", "description": "", "input_schema": {}}
    ]


def test_normalize_missing_description_becomes_empty_string_gemini() -> None:
    tool = SimpleNamespace(
        function_declarations=[SimpleNamespace(name="no_description", parameters={})]
    )
    assert normalize_tools(tool) == [
        {"name": "no_description", "description": "", "input_schema": {}}
    ]


def test_normalize_missing_input_schema_becomes_empty_dict() -> None:
    tool = {"name": "no_schema"}
    assert normalize_tools(tool) == [{"name": "no_schema", "description": "", "input_schema": {}}]


# --- F5: only a genuinely missing schema (None) defaults to {} -- a falsy-but-real value the
#     caller actually supplied must be preserved verbatim, not collapsed by a truthy check ----


def test_normalize_preserves_empty_list_input_schema_anthropic() -> None:
    assert normalize_tools({"name": "t", "input_schema": []}) == [
        {"name": "t", "description": "", "input_schema": []}
    ]


def test_normalize_preserves_zero_input_schema_anthropic() -> None:
    assert normalize_tools({"name": "t", "input_schema": 0}) == [
        {"name": "t", "description": "", "input_schema": 0}
    ]


def test_normalize_preserves_false_input_schema_anthropic() -> None:
    assert normalize_tools({"name": "t", "input_schema": False}) == [
        {"name": "t", "description": "", "input_schema": False}
    ]


def test_normalize_preserves_empty_string_input_schema_anthropic() -> None:
    assert normalize_tools({"name": "t", "input_schema": ""}) == [
        {"name": "t", "description": "", "input_schema": ""}
    ]


def test_normalize_preserves_falsy_parameters_value_openai() -> None:
    tool = {"type": "function", "function": {"name": "t", "parameters": []}}
    assert normalize_tools(tool) == [{"name": "t", "description": "", "input_schema": []}]


def test_normalize_none_input_schema_still_becomes_empty_dict() -> None:
    """The one falsy value that *should* default to {}: a genuinely absent schema (None is the
    sentinel every call site's `.get(key)` / `getattr(..., None)` uses for "missing")."""
    assert normalize_tools({"name": "t", "input_schema": None}) == [
        {"name": "t", "description": "", "input_schema": {}}
    ]


# --- F3: a schema dict/list must not remain aliased to the caller's own object -- otherwise a
#     host that mutates a tool's schema in place after normalising it (the window is the whole
#     session for _current_toolset, or a LangChain handler's whole lifetime) silently changes
#     the *already-normalised, already-fingerprinted* toolset too, and a sidecar written later
#     no longer hashes to its own filename. ----------------------------------------------------


def test_coerce_schema_deep_copies_dict_so_later_mutation_cannot_alias_it() -> None:
    mutable_schema: dict[str, Any] = {"type": "object", "properties": {"a": {"type": "string"}}}
    tool = {"name": "t", "description": "d", "input_schema": mutable_schema}

    normalized = normalize_tools(tool)
    assert normalized is not None
    fingerprint_before = fingerprint_tools(normalized)

    # The caller mutates its own object after normalising -- a realistic pattern for a session
    # or LangChain-handler toolset that outlives a single call.
    mutable_schema["properties"]["a"] = {"type": "integer"}
    mutable_schema["properties"]["b"] = {"type": "boolean"}

    assert normalized[0]["input_schema"] == {
        "type": "object",
        "properties": {"a": {"type": "string"}},
    }  # unaffected by the mutation above
    assert fingerprint_tools(normalized) == fingerprint_before  # still hashes to its own content


def test_coerce_schema_deep_copies_list_input_schema_so_later_mutation_cannot_alias_it() -> None:
    mutable_schema: list[Any] = [{"a": 1}]
    tool = {"name": "t", "input_schema": mutable_schema}

    normalized = normalize_tools(tool)
    assert normalized is not None

    mutable_schema.append({"b": 2})
    mutable_schema[0]["a"] = 999

    assert normalized[0]["input_schema"] == [{"a": 1}]  # unaffected


def test_coerce_schema_deep_copies_nested_dict_inside_gemini_parameters() -> None:
    """Same aliasing risk via the Gemini duck-typed path -- a plain-dict `parameters` (not a
    `model_dump`-bearing Schema object) goes through the identical by-reference branch."""
    mutable_params: dict[str, Any] = {"type": "object", "properties": {"to": {"type": "string"}}}
    tool = _gemini_tool_with_parameters(mutable_params)

    normalized = normalize_tools(tool)
    assert normalized is not None

    mutable_params["properties"]["to"] = {"type": "integer"}

    assert normalized[0]["input_schema"] == {
        "type": "object",
        "properties": {"to": {"type": "string"}},
    }


def test_coerce_schema_degrades_to_empty_dict_when_deepcopy_itself_raises() -> None:
    """Hardening found while implementing F3's deep copy: a dict/list is only JSON-*safe* at the
    top level `_coerce_schema` inspects -- `input_schema` is unvalidated, un-allow-listed user
    JSON (D-toolset), so something not itself deep-copyable can still be nested arbitrarily deep
    inside it. A `set`/`Enum`/`datetime` copies fine and is only later caught by
    `fingerprint_tools`'s `json.dumps` (guarded by its caller, `_normalize_toolset`); a lock
    (stood in for here), an open file, or a generator raises on `copy.deepcopy` itself, one step
    *before* that guard -- inside `normalize_tools`, which the two per-call sites
    (`record_model_call`, `_ModelCallRecorder._open`) call with no wrapping guard of their own.
    An unguarded deepcopy failure there would silently drop the whole `model_call` event, exactly
    like the bug F1 fixes, just one step earlier in the pipeline and via a different exception."""
    tool = {
        "name": "t",
        "input_schema": {"type": "object", "properties": {"lock": threading.Lock()}},
    }
    assert normalize_tools(tool) == [{"name": "t", "description": "", "input_schema": {}}]


# --- normalize_tools: garbage in, None out -- never raises ------------------------------------


def test_normalize_none_is_unrecognised() -> None:
    assert normalize_tools(None) is None


def test_normalize_string_is_unrecognised() -> None:
    assert normalize_tools("garbage") is None


def test_normalize_int_is_unrecognised() -> None:
    assert normalize_tools(42) is None


def test_normalize_dict_without_name_or_function_is_unrecognised() -> None:
    assert normalize_tools({"foo": "bar"}) is None


def test_normalize_object_without_function_declarations_is_unrecognised() -> None:
    assert normalize_tools(SimpleNamespace(unrelated="attribute")) is None


def test_normalize_gemini_tool_with_non_iterable_declarations_is_unrecognised() -> None:
    assert normalize_tools(SimpleNamespace(function_declarations=42)) is None


def test_normalize_dict_with_nameless_anthropic_shape_is_unrecognised() -> None:
    assert normalize_tools({"description": "no name here", "input_schema": {}}) is None


def test_normalize_list_with_one_unrecognisable_item_is_none_not_partial() -> None:
    """A partially-normalised toolset would misreport what the agent was offered -- worse than
    reporting nothing, so one bad element invalidates the whole call, not just that element.
    """
    assert normalize_tools([ANTHROPIC_TOOL, "garbage"]) is None


def test_normalize_does_not_raise_on_a_variety_of_wrong_types() -> None:
    garbage_values: list[Any] = [
        object(),
        [1, 2, 3],
        {"type": "function", "function": "not-a-dict"},
        True,
        3.14,
        {"type": "function"},
    ]
    for garbage in garbage_values:
        assert normalize_tools(garbage) is None


# --- fingerprint_tools: determinism, and independence from key/list order --------------------


def test_fingerprint_is_deterministic() -> None:
    normalized = normalize_tools(THREE_TOOL_RAW)
    assert normalized is not None
    assert fingerprint_tools(normalized) == fingerprint_tools(normalized)


def test_fingerprint_independent_of_list_order() -> None:
    forward = normalize_tools(THREE_TOOL_RAW)
    backward = normalize_tools(list(reversed(THREE_TOOL_RAW)))
    assert forward is not None
    assert backward is not None
    assert fingerprint_tools(forward) == fingerprint_tools(backward)


def test_fingerprint_independent_of_dict_key_order() -> None:
    reordered_anthropic = {
        "input_schema": ANTHROPIC_TOOL["input_schema"],
        "description": ANTHROPIC_TOOL["description"],
        "name": ANTHROPIC_TOOL["name"],
    }
    original = normalize_tools([ANTHROPIC_TOOL])
    reordered = normalize_tools([reordered_anthropic])
    assert original is not None
    assert reordered is not None
    assert fingerprint_tools(original) == fingerprint_tools(reordered)


def test_fingerprint_tools_ignores_dict_key_order_directly() -> None:
    """Exercises ``sort_keys=True`` itself: normalize_tools always emits dicts with the same
    key order (name, description, input_schema), so a test that only varies *pre-normalisation*
    key order never actually feeds fingerprint_tools two differently-ordered dicts. This test
    does, by constructing the already-normalised dicts by hand.
    """
    forward = [{"name": "x", "description": "d", "input_schema": {"type": "object"}}]
    reordered = [{"input_schema": {"type": "object"}, "description": "d", "name": "x"}]
    assert fingerprint_tools(forward) == fingerprint_tools(reordered)


def test_fingerprint_differs_for_different_toolsets() -> None:
    one = normalize_tools([ANTHROPIC_TOOL])
    two = normalize_tools([OPENAI_TOOL])
    assert one is not None
    assert two is not None
    assert fingerprint_tools(one) != fingerprint_tools(two)


def test_fingerprint_has_sha256_prefix_and_64_char_hex_digest() -> None:
    fp = fingerprint_tools([])
    assert fp.startswith("sha256:")
    hex_part = fp.removeprefix("sha256:")
    assert len(hex_part) == 64
    int(hex_part, 16)  # raises ValueError if this is not valid hex


# --- The shared vector: Task 2 (CLI) asserts these two constants verbatim --------------------


def test_three_tool_vector_matches_pinned_fingerprint() -> None:
    normalized = normalize_tools(THREE_TOOL_RAW)
    assert normalized is not None
    assert normalized == THREE_TOOL_NORMALIZED
    assert fingerprint_tools(normalized) == THREE_TOOL_FINGERPRINT


def test_empty_toolset_vector_matches_pinned_fingerprint() -> None:
    assert fingerprint_tools([]) == EMPTY_TOOLSET_FINGERPRINT


# --- strict (Phase 4, review #8): the one function-envelope key carried besides the three -----


def _strict_openai(strict: Any) -> dict[str, Any]:
    tool: dict[str, Any] = json.loads(json.dumps(OPENAI_TOOL))
    tool["function"]["strict"] = strict
    return tool


def _strict_anthropic(strict: Any) -> dict[str, Any]:
    tool: dict[str, Any] = json.loads(json.dumps(ANTHROPIC_TOOL))
    tool["strict"] = strict
    return tool


def test_openai_strict_true_is_carried_onto_the_canonical_tool() -> None:
    assert normalize_tools(_strict_openai(True)) == [
        {
            "name": "add_task",
            "description": "Add a task to the user's to-do list.",
            "input_schema": OPENAI_TOOL["function"]["parameters"],
            "strict": True,
        }
    ]


def test_anthropic_strict_true_is_carried_onto_the_canonical_tool() -> None:
    assert normalize_tools(_strict_anthropic(True)) == [
        {
            "name": "get_schedule",
            "description": "Look up a user's schedule for a given date.",
            "input_schema": ANTHROPIC_TOOL["input_schema"],
            "strict": True,
        }
    ]


def test_strict_false_or_absent_omits_the_key_entirely() -> None:
    """Absent and falsy must be byte-identical to the pre-strict shape: no `"strict": false`."""
    baseline = normalize_tools(OPENAI_TOOL)
    assert baseline == [
        {
            "name": "add_task",
            "description": "Add a task to the user's to-do list.",
            "input_schema": OPENAI_TOOL["function"]["parameters"],
        }
    ]
    assert normalize_tools(_strict_openai(False)) == baseline
    assert normalize_tools(_strict_openai(None)) == baseline
    assert normalize_tools(_strict_anthropic(False)) == normalize_tools(ANTHROPIC_TOOL)


def test_strict_changes_the_fingerprint() -> None:
    plain = normalize_tools(OPENAI_TOOL)
    strict = normalize_tools(_strict_openai(True))
    assert plain is not None and strict is not None
    assert fingerprint_tools(plain) != fingerprint_tools(strict)


def test_strict_false_keeps_the_pinned_fingerprints_byte_for_byte() -> None:
    """Every capture written before this key existed must fingerprint identically."""
    raw: list[Any] = [_strict_anthropic(False), _strict_openai(False), _gemini_tool()]
    normalized = normalize_tools(raw)
    assert normalized == THREE_TOOL_NORMALIZED
    assert normalized is not None
    assert fingerprint_tools(normalized) == THREE_TOOL_FINGERPRINT
    assert fingerprint_tools([]) == EMPTY_TOOLSET_FINGERPRINT


def test_gemini_declarations_get_no_strict_key() -> None:
    """FunctionDeclaration has no strict flag; a stray attribute must not invent one."""
    tool = _gemini_tool()
    tool.function_declarations[0].strict = True
    normalized = normalize_tools(tool)
    assert normalized is not None
    assert "strict" not in normalized[0]
