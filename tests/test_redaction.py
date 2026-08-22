"""Unit tests for the built-in ``default_redactor`` and the ``redact_tree`` walker."""

from __future__ import annotations

from evalshift.capture.span import SpanTree
from evalshift.redaction import default_redactor, redact_tree


def test_masks_email() -> None:
    out = default_redactor("reach me at jane.doe@example.com please")
    assert "jane.doe@example.com" not in out
    assert "[REDACTED_EMAIL]" in out


def test_masks_openai_key() -> None:
    out = default_redactor("token sk-abcdEFGH1234567890zzzz here")
    assert "sk-abcdEFGH1234567890zzzz" not in out
    assert "[REDACTED_KEY]" in out


def test_masks_bearer_token() -> None:
    out = default_redactor("Authorization: Bearer abc.def.ghi123")
    assert "abc.def.ghi123" not in out
    assert "[REDACTED_KEY]" in out


def test_walks_nested_dict_and_list() -> None:
    payload = {"user": {"email": "a@b.com"}, "items": ["x@y.com", 1, None]}
    out = default_redactor(payload)
    assert "[REDACTED_EMAIL]" in out["user"]["email"]
    assert "[REDACTED_EMAIL]" in out["items"][0]
    assert out["items"][1] == 1
    assert out["items"][2] is None


def test_non_strings_untouched() -> None:
    assert default_redactor(42) == 42
    assert default_redactor(None) is None
    assert default_redactor(3.5) == 3.5
    assert default_redactor(True) is True


def test_clean_string_preserved() -> None:
    assert default_redactor("just a normal sentence") == "just a normal sentence"


def test_does_not_mutate_input() -> None:
    payload = {"email": "a@b.com", "items": ["c@d.com"]}
    default_redactor(payload)
    assert payload == {"email": "a@b.com", "items": ["c@d.com"]}


def test_idempotent() -> None:
    once = default_redactor("a@b.com and sk-abcdEFGH1234567890zzzz")
    assert default_redactor(once) == once


def test_redact_tree_masks_tool_fields() -> None:
    tree = SpanTree()
    span = tree.open_span(
        "tool", span_id="call_1", start_ts=0.0, data={"name": "t", "arguments": {"q": "a@b.com"}}
    )
    tree.close_span(span, end_ts=1.0, result="reply to a@b.com")
    redact_tree(tree, default_redactor)
    assert "[REDACTED_EMAIL]" in span.data["arguments"]["q"]
    assert "[REDACTED_EMAIL]" in span.data["result"]


def test_redact_tree_masks_model_fields() -> None:
    tree = SpanTree()
    span = tree.open_span(
        "model_call",
        span_id="mc_1",
        start_ts=0.0,
        data={"model_id": "m", "input": "to a@b.com", "output": "see a@b.com"},
    )
    tree.close_span(span, end_ts=1.0)
    redact_tree(tree, default_redactor)
    assert "[REDACTED_EMAIL]" in span.data["input"]
    assert "[REDACTED_EMAIL]" in span.data["output"]


def test_redact_tree_leaves_non_payload_fields() -> None:
    tree = SpanTree()
    span = tree.open_span(
        "tool", span_id="call_1", start_ts=0.0, data={"name": "lookup", "arguments": {}}
    )
    tree.close_span(span, end_ts=1.0, result=None)
    redact_tree(tree, default_redactor)
    assert span.data["name"] == "lookup"  # the tool name is structural, never redacted
