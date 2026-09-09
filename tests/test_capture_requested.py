"""Unit tests for the provider-response -> requested-tool-call extraction helper.

Every fixture below is hand-built from the *public* API reference of the provider it names --
this SDK never imports a provider SDK (D-deps), so nothing here is generated from one. The
shape's source is cited in a comment above each fixture.

Mirrors ``tests/test_capture_toolset.py``'s duck-typing style (stand-in classes with plain
attributes, never a real provider class) and ``tests/test_safety.py``'s ``caplog`` pattern for
the debug-only degradations.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from evalshift.capture.requested import extract_requested_tool_calls

# --- fixtures: one per provider shape ---------------------------------------------------------

# OpenAI Chat Completions, `chat.completion` object:
# https://platform.openai.com/docs/api-reference/chat/object -- assistant message carries
# `tool_calls[*] = {id, type: "function", function: {name, arguments}}`, `arguments` a JSON
# *string*, and `content` is null on a tool-calling turn.
OPENAI_CHAT: dict[str, Any] = {
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "search_orders",
                            "arguments": '{"customer_id": "c-1"}',
                        },
                    },
                    {
                        "id": "call_def456",
                        "type": "function",
                        "function": {
                            "name": "issue_refund",
                            "arguments": '{"order_id": "o-9", "amount": 12.5}',
                        },
                    },
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
}

# OpenAI Chat Completions, deprecated single-call form:
# https://platform.openai.com/docs/api-reference/chat/object -- `message.function_call =
# {name, arguments}`, no id field at all.
OPENAI_LEGACY_FUNCTION_CALL: dict[str, Any] = {
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "function_call": {"name": "search_orders", "arguments": '{"customer_id": "c-1"}'},
            },
            "finish_reason": "function_call",
        }
    ],
}

# OpenAI Responses API, `response` object:
# https://platform.openai.com/docs/api-reference/responses/object -- tool calls are top-level
# `output[*]` items of `type: "function_call"` with `{name, arguments (JSON string), call_id}`,
# interleaved with other item types (`message`, `reasoning`) that are not tool calls.
OPENAI_RESPONSES: dict[str, Any] = {
    "id": "resp_abc123",
    "object": "response",
    "model": "gpt-4.1",
    "output": [
        {"type": "reasoning", "id": "rs_1", "summary": []},
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_abc123",
            "name": "search_orders",
            "arguments": '{"customer_id": "c-1"}',
            "status": "completed",
        },
    ],
}

# Anthropic Messages, `message` object:
# https://docs.anthropic.com/en/api/messages -- `content[*]` blocks, tool calls are
# `{type: "tool_use", id, name, input}` with `input` an already-parsed object.
ANTHROPIC_MESSAGE: dict[str, Any] = {
    "id": "msg_abc123",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4",
    "stop_reason": "tool_use",
    "content": [
        {"type": "text", "text": "Let me look that up."},
        {
            "type": "tool_use",
            "id": "toolu_abc123",
            "name": "search_orders",
            "input": {"customer_id": "c-1"},
        },
    ],
}

# Gemini generateContent REST response (camelCase on the wire):
# https://ai.google.dev/api/generate-content -- `candidates[0].content.parts[*].functionCall =
# {name, args}`. There is no per-call id in the REST shape.
GEMINI_REST: dict[str, Any] = {
    "candidates": [
        {
            "content": {
                "role": "model",
                "parts": [
                    {"text": "Looking that up."},
                    {"functionCall": {"name": "search_orders", "args": {"customer_id": "c-1"}}},
                ],
            },
            "finishReason": "STOP",
        }
    ]
}

# Gemini via the python SDK's `response.to_dict()` (snake_case keys, same content):
# https://googleapis.github.io/python-genai/ -- `parts[*].function_call = {name, args}`, plus the
# newer optional `id` field.
GEMINI_TO_DICT: dict[str, Any] = {
    "candidates": [
        {
            "content": {
                "role": "model",
                "parts": [
                    {
                        "function_call": {
                            "id": "fc-1",
                            "name": "search_orders",
                            "args": {"customer_id": "c-1"},
                        }
                    }
                ],
            }
        }
    ]
}


# --- duck-typed stand-ins (never a real provider class; the SDK imports no provider SDK) -------


class _Attr:
    """A minimal object whose attributes mirror a provider response class."""

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class _Dumpable:
    """An object exposing only ``model_dump()`` -- the pydantic-response duck type."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def model_dump(self) -> dict[str, Any]:
        return self._payload


class _ToDictable:
    """An object exposing only ``to_dict()`` -- the google-genai response duck type."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def to_dict(self) -> dict[str, Any]:
        return self._payload


class _Exploding:
    """Every access raises -- the helper must still return ``None`` rather than propagate."""

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError("boom")


# --- the recognised shapes --------------------------------------------------------------------


def test_openai_chat_completions_tool_calls() -> None:
    assert extract_requested_tool_calls(OPENAI_CHAT) == [
        {
            "name": "search_orders",
            "arguments": {"customer_id": "c-1"},
            "call_id": "call_abc123",
        },
        {
            "name": "issue_refund",
            "arguments": {"order_id": "o-9", "amount": 12.5},
            "call_id": "call_def456",
        },
    ]


def test_openai_uses_the_first_choice_only() -> None:
    two_choices = {
        "choices": [
            OPENAI_CHAT["choices"][0],
            {
                "index": 1,
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_other",
                            "type": "function",
                            "function": {"name": "other", "arguments": "{}"},
                        }
                    ],
                },
            },
        ]
    }
    names = [call["name"] for call in extract_requested_tool_calls(two_choices) or []]
    assert names == ["search_orders", "issue_refund"]


def test_openai_legacy_function_call() -> None:
    assert extract_requested_tool_calls(OPENAI_LEGACY_FUNCTION_CALL) == [
        {"name": "search_orders", "arguments": {"customer_id": "c-1"}, "call_id": None}
    ]


def test_openai_tool_calls_win_over_a_legacy_function_call() -> None:
    both = {
        "choices": [
            {
                "message": {
                    "tool_calls": OPENAI_CHAT["choices"][0]["message"]["tool_calls"],
                    "function_call": {"name": "legacy", "arguments": "{}"},
                }
            }
        ]
    }
    names = [call["name"] for call in extract_requested_tool_calls(both) or []]
    assert names == ["search_orders", "issue_refund"]


def test_openai_responses_api_function_call_items() -> None:
    assert extract_requested_tool_calls(OPENAI_RESPONSES) == [
        {"name": "search_orders", "arguments": {"customer_id": "c-1"}, "call_id": "call_abc123"}
    ]


def test_anthropic_tool_use_blocks() -> None:
    assert extract_requested_tool_calls(ANTHROPIC_MESSAGE) == [
        {"name": "search_orders", "arguments": {"customer_id": "c-1"}, "call_id": "toolu_abc123"}
    ]


def test_gemini_rest_camel_case_function_call() -> None:
    assert extract_requested_tool_calls(GEMINI_REST) == [
        {"name": "search_orders", "arguments": {"customer_id": "c-1"}, "call_id": None}
    ]


def test_gemini_snake_case_function_call_keeps_an_id_when_present() -> None:
    assert extract_requested_tool_calls(GEMINI_TO_DICT) == [
        {"name": "search_orders", "arguments": {"customer_id": "c-1"}, "call_id": "fc-1"}
    ]


def test_order_is_response_order() -> None:
    multi = {
        "content": [
            {"type": "tool_use", "id": "t3", "name": "c", "input": {}},
            {"type": "text", "text": "..."},
            {"type": "tool_use", "id": "t1", "name": "a", "input": {}},
            {"type": "tool_use", "id": "t2", "name": "b", "input": {}},
        ]
    }
    assert [call["name"] for call in extract_requested_tool_calls(multi) or []] == ["c", "a", "b"]


# --- recognised-but-empty ([]) vs unrecognised (None) ------------------------------------------


def test_recognised_response_with_no_tool_calls_is_empty_list() -> None:
    openai_text = {
        "choices": [{"message": {"role": "assistant", "content": "hi", "tool_calls": None}}]
    }
    anthropic_text = {"content": [{"type": "text", "text": "hi"}]}
    gemini_text = {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}
    responses_text = {"output": [{"type": "message", "content": []}]}
    for response in (openai_text, anthropic_text, gemini_text, responses_text):
        assert extract_requested_tool_calls(response) == []


def test_empty_containers_are_recognised_as_no_tool_calls() -> None:
    assert extract_requested_tool_calls({"choices": []}) == []
    assert extract_requested_tool_calls({"content": []}) == []
    assert extract_requested_tool_calls({"output": []}) == []
    assert extract_requested_tool_calls({"candidates": []}) == []


@pytest.mark.parametrize(
    "response",
    [
        None,
        "nope",
        b"nope",
        42,
        [],
        [{"type": "tool_use", "name": "x", "input": {}}],  # a bare block list, not a response
        {},
        {"foo": "bar"},
        {"choices": "not-a-list"},
        {"content": "just text"},  # Anthropic *request* shape, not a response
        {"candidates": {"not": "a list"}},
        {"output": "text"},
    ],
)
def test_unrecognised_input_is_none(response: Any) -> None:
    assert extract_requested_tool_calls(response) is None


def test_never_raises_on_a_hostile_object() -> None:
    assert extract_requested_tool_calls(_Exploding()) is None


def test_none_and_empty_list_are_distinguishable() -> None:
    # The contract the whole module exists for: [] is a value ("the model requested nothing"),
    # None is an absence ("this did not look like a provider response").
    assert extract_requested_tool_calls({"content": []}) == []
    assert extract_requested_tool_calls({"nonsense": 1}) is None


# --- duck-typed (non-dict) responses -----------------------------------------------------------


def test_duck_typed_object_with_model_dump() -> None:
    assert extract_requested_tool_calls(_Dumpable(ANTHROPIC_MESSAGE)) == [
        {"name": "search_orders", "arguments": {"customer_id": "c-1"}, "call_id": "toolu_abc123"}
    ]


def test_duck_typed_object_with_to_dict() -> None:
    assert extract_requested_tool_calls(_ToDictable(OPENAI_CHAT)) == [
        {"name": "search_orders", "arguments": {"customer_id": "c-1"}, "call_id": "call_abc123"},
        {
            "name": "issue_refund",
            "arguments": {"order_id": "o-9", "amount": 12.5},
            "call_id": "call_def456",
        },
    ]


def test_duck_typed_object_with_a_raising_model_dump_falls_back_to_attributes() -> None:
    class _BadDump(_Attr):
        def model_dump(self) -> dict[str, Any]:
            raise RuntimeError("boom")

    response = _BadDump(
        content=[
            _Attr(type="text", text="hi"),
            _Attr(type="tool_use", id="toolu_1", name="search_orders", input={"q": "x"}),
        ]
    )
    assert extract_requested_tool_calls(response) == [
        {"name": "search_orders", "arguments": {"q": "x"}, "call_id": "toolu_1"}
    ]


def test_attribute_access_openai_response_object() -> None:
    response = _Attr(
        choices=[
            _Attr(
                message=_Attr(
                    content=None,
                    tool_calls=[
                        _Attr(
                            id="call_abc123",
                            type="function",
                            function=_Attr(
                                name="search_orders", arguments='{"customer_id": "c-1"}'
                            ),
                        )
                    ],
                )
            )
        ]
    )
    assert extract_requested_tool_calls(response) == [
        {"name": "search_orders", "arguments": {"customer_id": "c-1"}, "call_id": "call_abc123"}
    ]


def test_attribute_access_gemini_response_object() -> None:
    response = _Attr(
        candidates=[
            _Attr(
                content=_Attr(
                    parts=[_Attr(function_call=_Attr(name="search_orders", args={"q": "x"}))]
                )
            )
        ]
    )
    assert extract_requested_tool_calls(response) == [
        {"name": "search_orders", "arguments": {"q": "x"}, "call_id": None}
    ]


# --- degraded arguments ------------------------------------------------------------------------


def test_unparseable_json_arguments_degrade_to_empty_dict(
    caplog: pytest.LogCaptureFixture,
) -> None:
    broken = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "search_orders", "arguments": "{not json"},
                        }
                    ]
                }
            }
        ]
    }
    with caplog.at_level(logging.DEBUG, logger="evalshift"):
        calls = extract_requested_tool_calls(broken)
    assert calls == [{"name": "search_orders", "arguments": {}, "call_id": "call_1"}]
    assert any("arguments" in record.message for record in caplog.records)


def test_json_arguments_that_are_not_an_object_degrade_to_empty_dict() -> None:
    scalar_args = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "f", "arguments": "[1, 2]"},
                        }
                    ]
                }
            }
        ]
    }
    assert extract_requested_tool_calls(scalar_args) == [
        {"name": "f", "arguments": {}, "call_id": "call_1"}
    ]


def test_missing_or_empty_arguments_are_an_empty_dict() -> None:
    no_args = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"id": "a", "type": "function", "function": {"name": "f"}},
                        {"id": "b", "type": "function", "function": {"name": "g", "arguments": ""}},
                    ]
                }
            }
        ]
    }
    assert extract_requested_tool_calls(no_args) == [
        {"name": "f", "arguments": {}, "call_id": "a"},
        {"name": "g", "arguments": {}, "call_id": "b"},
    ]


def test_non_dict_arguments_degrade_to_empty_dict(caplog: pytest.LogCaptureFixture) -> None:
    # Anthropic `input` and Gemini `args` are already-parsed objects, so a non-dict there is a
    # shape violation rather than a parse failure.
    anthropic = {"content": [{"type": "tool_use", "id": "t1", "name": "f", "input": ["nope"]}]}
    gemini = {"candidates": [{"content": {"parts": [{"functionCall": {"name": "f", "args": 3}}]}}]}
    with caplog.at_level(logging.DEBUG, logger="evalshift"):
        assert extract_requested_tool_calls(anthropic) == [
            {"name": "f", "arguments": {}, "call_id": "t1"}
        ]
        assert extract_requested_tool_calls(gemini) == [
            {"name": "f", "arguments": {}, "call_id": None}
        ]
    assert any("arguments" in record.message for record in caplog.records)


def test_arguments_are_json_coerced_and_never_alias_the_response() -> None:
    source = {"content": [{"type": "tool_use", "id": "t1", "name": "f", "input": {"a": {"b": 1}}}]}
    calls = extract_requested_tool_calls(source)
    assert calls is not None
    calls[0]["arguments"]["a"]["b"] = 2
    assert source["content"][0]["input"] == {"a": {"b": 1}}


def test_self_referential_arguments_degrade_instead_of_raising() -> None:
    args: dict[str, Any] = {}
    args["self"] = args  # JSON coercion recurses -> RecursionError, must be swallowed
    response = {"content": [{"type": "tool_use", "id": "t1", "name": "f", "input": args}]}
    assert extract_requested_tool_calls(response) == [
        {"name": "f", "arguments": {}, "call_id": "t1"}
    ]


# --- a tool call we cannot read makes the whole response unreadable ----------------------------


@pytest.mark.parametrize(
    "response",
    [
        # An OpenAI tool_calls entry with no usable name.
        {"choices": [{"message": {"tool_calls": [{"id": "c1", "type": "function"}]}}]},
        {
            "choices": [
                {"message": {"tool_calls": [{"id": "c1", "function": {"name": "", "args": {}}}]}}
            ]
        },
        {"choices": [{"message": {"tool_calls": "not-a-list"}}]},
        # An Anthropic tool_use block with no name.
        {"content": [{"type": "tool_use", "id": "t1", "input": {}}]},
        # A Gemini functionCall with no name.
        {"candidates": [{"content": {"parts": [{"functionCall": {"args": {}}}]}}]},
        # A Responses function_call item with no name.
        {"output": [{"type": "function_call", "call_id": "c1", "arguments": "{}"}]},
    ],
)
def test_an_unreadable_tool_call_makes_the_response_unrecognised(response: Any) -> None:
    # Same refusal as ``normalize_tools``: a silently-short list of requested calls is worse
    # than no list at all, so we never publish a partial one.
    assert extract_requested_tool_calls(response) is None


def test_a_non_string_call_id_becomes_none() -> None:
    response = {"content": [{"type": "tool_use", "id": 7, "name": "f", "input": {}}]}
    assert extract_requested_tool_calls(response) == [
        {"name": "f", "arguments": {}, "call_id": None}
    ]


# --- the output contract -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "response",
    [
        OPENAI_CHAT,
        OPENAI_LEGACY_FUNCTION_CALL,
        OPENAI_RESPONSES,
        ANTHROPIC_MESSAGE,
        GEMINI_REST,
        GEMINI_TO_DICT,
    ],
)
def test_every_item_has_exactly_the_three_contract_keys(response: dict[str, Any]) -> None:
    calls = extract_requested_tool_calls(response)
    assert calls, "fixture should yield at least one requested call"
    for call in calls:
        assert set(call) == {"name", "arguments", "call_id"}
        assert isinstance(call["name"], str) and call["name"]
        assert isinstance(call["arguments"], dict)
        assert call["call_id"] is None or isinstance(call["call_id"], str)
