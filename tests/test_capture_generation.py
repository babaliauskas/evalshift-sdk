"""Unit tests for the shared generation-config allow-list + JSON coercion."""

from __future__ import annotations

from typing import Any

from evalshift.capture.generation import GENERATION_KEYS, jsonable, sanitize_generation_config


class _Opaque:
    def __repr__(self) -> str:
        return "<opaque>"


class _PydanticLike:
    """Stand-in for a ``google.genai.types.ToolConfig`` — duck-typed, never imported (D-deps)."""

    def __init__(self, dumped: Any = None) -> None:
        self._dumped = {"function_calling_config": {"mode": "ANY"}} if dumped is None else dumped
        self.calls: list[dict[str, Any]] = []

    def model_dump(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._dumped

    def __repr__(self) -> str:
        return "<pydantic-like>"


class _BadDump:
    """A ``model_dump`` that raises — must degrade to ``str()``, never propagate."""

    def model_dump(self, **kwargs: Any) -> Any:
        raise RuntimeError("boom")

    def __repr__(self) -> str:
        return "<bad-dump>"


def test_allowlist_is_the_frozen_canonical_tuple() -> None:
    assert GENERATION_KEYS == (
        "temperature",
        "top_p",
        "response_mime_type",
        "response_schema",
        "response_format",
        "max_output_tokens",
        "max_tokens",
        "tool_choice",
        "parallel_tool_calls",
        "tool_config",
    )


def test_jsonable_passes_primitives_and_recurses() -> None:
    assert jsonable(None) is None
    assert jsonable("x") == "x"
    assert jsonable(1) == 1
    assert jsonable(1.5) == 1.5
    assert jsonable(True) is True
    assert jsonable({"a": (1, _Opaque())}) == {"a": [1, "<opaque>"]}
    assert jsonable([_Opaque()]) == ["<opaque>"]
    assert jsonable(_Opaque()) == "<opaque>"


def test_jsonable_stringifies_non_str_dict_keys() -> None:
    assert jsonable({1: "a"}) == {"1": "a"}


def test_sanitize_returns_none_for_non_dict() -> None:
    assert sanitize_generation_config("nope") is None
    assert sanitize_generation_config(None) is None
    assert sanitize_generation_config([("temperature", 0.0)]) is None


def test_sanitize_returns_none_for_empty_result() -> None:
    assert sanitize_generation_config({}) is None
    assert sanitize_generation_config({"system_instruction": "hi"}) is None
    assert sanitize_generation_config({"temperature": None}) is None


def test_sanitize_keeps_allowlisted_and_coerces() -> None:
    out = sanitize_generation_config(
        {
            "temperature": 0.0,
            "max_tokens": 128,
            "response_schema": _Opaque(),
            "system_instruction": "drop me",
            "safety_settings": ["drop me too"],
        }
    )
    assert out == {"temperature": 0.0, "max_tokens": 128, "response_schema": "<opaque>"}


def test_sanitize_preserves_nested_json_structures() -> None:
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}}
    out = sanitize_generation_config({"response_schema": schema})
    assert out == {"response_schema": schema}


def test_sanitize_does_not_alias_the_input() -> None:
    source = {"temperature": 0.0}
    out = sanitize_generation_config(source)
    assert out is not source


# --- tool_choice / parallel_tool_calls / tool_config (Phase 4, review #8) --------------------


def test_sanitize_records_tool_choice_verbatim() -> None:
    assert sanitize_generation_config({"tool_choice": "required"}) == {"tool_choice": "required"}
    forced = {"type": "function", "function": {"name": "search"}}
    assert sanitize_generation_config({"tool_choice": forced}) == {"tool_choice": forced}


def test_sanitize_keeps_parallel_tool_calls_false() -> None:
    """``False`` is the whole point of the setting — it must not be filtered out as falsy."""
    assert sanitize_generation_config({"parallel_tool_calls": False}) == {
        "parallel_tool_calls": False
    }
    assert sanitize_generation_config({"parallel_tool_calls": True}) == {
        "parallel_tool_calls": True
    }


def test_sanitize_coerces_gemini_tool_config_object_to_a_dict() -> None:
    out = sanitize_generation_config({"tool_config": _PydanticLike()})
    assert out == {"tool_config": {"function_calling_config": {"mode": "ANY"}}}


def test_jsonable_uses_model_dump_when_present() -> None:
    obj = _PydanticLike()
    assert jsonable(obj) == {"function_calling_config": {"mode": "ANY"}}
    assert obj.calls == [{"exclude_none": True, "mode": "json"}]


def test_jsonable_still_stringifies_objects_without_model_dump() -> None:
    assert jsonable(_Opaque()) == "<opaque>"
    assert jsonable({"a": _Opaque()}) == {"a": "<opaque>"}


def test_jsonable_falls_back_to_str_when_model_dump_misbehaves() -> None:
    assert jsonable(_BadDump()) == "<bad-dump>"
    assert jsonable(_PydanticLike(dumped=["not", "a", "dict"])) == "<pydantic-like>"


def test_jsonable_coerces_values_nested_inside_a_model_dump() -> None:
    obj = _PydanticLike(dumped={"mode": _Opaque(), "allowed": (1, 2)})
    assert jsonable(obj) == {"mode": "<opaque>", "allowed": [1, 2]}
