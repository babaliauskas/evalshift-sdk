"""Unit tests for the shared generation-config allow-list + JSON coercion."""

from __future__ import annotations

from evalshift.capture.generation import GENERATION_KEYS, jsonable, sanitize_generation_config


class _Opaque:
    def __repr__(self) -> str:
        return "<opaque>"


def test_allowlist_is_the_frozen_canonical_tuple() -> None:
    assert GENERATION_KEYS == (
        "temperature",
        "top_p",
        "response_mime_type",
        "response_schema",
        "response_format",
        "max_output_tokens",
        "max_tokens",
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
