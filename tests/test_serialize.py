"""Unit tests for SpanTree -> capture-envelope serialization."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from evalshift.capture.span import SpanTree
from evalshift.trace import schema
from evalshift.trace.serialize import (
    build_capture,
    build_fixture_table,
    canonical_hash,
    capture_filename,
    envelope_to_dict,
)

TS = datetime.fromisoformat("2026-06-16T12:00:00+00:00")


def _events(env_dict: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = env_dict["trace"]["events"]
    return events


def _sample_tree() -> SpanTree:
    tree = SpanTree()
    m = tree.open_span(
        "model_call",
        span_id="m1",
        start_ts=1.0,
        data={
            "model_id": "claude-opus-4-8",
            "input": "hi",
            "output": "yo",
            "input_tokens": 10,
            "output_tokens": 5,
            "cost_usd": 0.01,
        },
    )
    tree.close_span(m, end_ts=1.5)
    t = tree.open_span(
        "tool", span_id="c1", start_ts=2.0, data={"name": "search", "arguments": {"q": "x"}}
    )
    tree.close_span(t, end_ts=2.2, result={"hits": 1})
    tree.open_span("final_output", span_id="f1", start_ts=3.0, data={"text": "done"})
    return tree


# --- canonical_hash ---


def test_canonical_hash_is_deterministic() -> None:
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"a": 1, "b": 2})


def test_canonical_hash_is_key_order_insensitive() -> None:
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})


def test_canonical_hash_distinguishes_values() -> None:
    assert canonical_hash({"a": 1}) != canonical_hash({"a": 2})


# --- event shaping ---


def test_model_call_serializes_to_single_event_with_latency() -> None:
    env = envelope_to_dict(
        build_capture(_sample_tree(), suite="s", agent_input="hi", capture_id="cap_t")
    )
    model_calls = [e for e in _events(env) if e["type"] == "model_call"]
    assert len(model_calls) == 1
    mc = model_calls[0]
    assert mc["model_id"] == "claude-opus-4-8"
    assert mc["latency_ms"] == 500
    assert mc["input_tokens"] == 10


def test_tool_span_serializes_to_call_and_result() -> None:
    env = envelope_to_dict(
        build_capture(_sample_tree(), suite="s", agent_input="hi", capture_id="cap_t")
    )
    calls = [e for e in _events(env) if e["type"] == "tool_call"]
    results = [e for e in _events(env) if e["type"] == "tool_result"]
    assert len(calls) == 1 and len(results) == 1
    assert calls[0]["call_id"] == "c1"
    assert calls[0]["name"] == "search"
    assert calls[0]["arguments"] == {"q": "x"}
    assert results[0]["call_id"] == "c1"
    assert results[0]["result"] == {"hits": 1}


def test_parent_call_id_preserved_on_tool_call() -> None:
    tree = SpanTree()
    p = tree.open_span("tool", span_id="p", start_ts=1.0, data={"name": "outer", "arguments": {}})
    tree.open_span(
        "tool",
        span_id="c",
        parent_call_id=p.span_id,
        start_ts=1.1,
        data={"name": "inner", "arguments": {}},
    )
    env = envelope_to_dict(build_capture(tree, suite="s", agent_input="x", capture_id="cap_t"))
    inner = next(e for e in _events(env) if e["type"] == "tool_call" and e["call_id"] == "c")
    assert inner["parent_call_id"] == "p"


# --- ordering ---


def test_sequence_index_is_dense_and_unique() -> None:
    env = envelope_to_dict(
        build_capture(_sample_tree(), suite="s", agent_input="hi", capture_id="cap_t")
    )
    indices = [e["sequence_index"] for e in _events(env)]
    assert indices == list(range(len(indices)))


def test_concurrent_tools_keep_stable_order_without_collision() -> None:
    tree = SpanTree()
    a = tree.open_span("tool", span_id="a", start_ts=1.0, data={"name": "ta", "arguments": {}})
    b = tree.open_span("tool", span_id="b", start_ts=1.0, data={"name": "tb", "arguments": {}})
    tree.close_span(b, end_ts=2.0, result="rb")
    tree.close_span(a, end_ts=3.0, result="ra")
    env = envelope_to_dict(build_capture(tree, suite="s", agent_input="x", capture_id="cap_t"))
    seq = [(e["type"], e.get("call_id")) for e in _events(env)]
    indices = [e["sequence_index"] for e in _events(env)]
    assert indices == list(range(4))
    assert seq == [
        ("tool_call", "a"),
        ("tool_call", "b"),
        ("tool_result", "b"),
        ("tool_result", "a"),
    ]


# --- concurrency metadata (D-3) ---


def test_events_carry_evalshift_timing_metadata() -> None:
    env = envelope_to_dict(
        build_capture(_sample_tree(), suite="s", agent_input="hi", capture_id="cap_t")
    )
    mc = next(e for e in _events(env) if e["type"] == "model_call")
    block = mc["metadata"]["evalshift"]
    assert block["span_id"] == "m1"
    assert block["start_ts"] == 1.0
    assert block["end_ts"] == 1.5


def test_tool_result_carries_input_hash() -> None:
    env = envelope_to_dict(
        build_capture(_sample_tree(), suite="s", agent_input="hi", capture_id="cap_t")
    )
    res = next(e for e in _events(env) if e["type"] == "tool_result")
    assert res["metadata"]["evalshift"]["input_hash"] == canonical_hash({"q": "x"})


# --- envelope (D-5b) ---


def test_envelope_has_exactly_envelope_keys() -> None:
    env = envelope_to_dict(
        build_capture(
            _sample_tree(), suite="s", agent_input="hi", capture_id="cap_t", created_at=TS
        )
    )
    assert set(env) == set(schema.ENVELOPE_KEYS)
    assert env["schema_version"] == schema.SCHEMA_VERSION
    assert env["created_at"] == "2026-06-16T12:00:00+00:00"


def test_schema_version_not_in_trace() -> None:
    env = envelope_to_dict(
        build_capture(_sample_tree(), suite="s", agent_input="hi", capture_id="cap_t")
    )
    assert "schema_version" not in env["trace"]


def test_input_hash_is_canonical_hash_of_agent_input() -> None:
    agent_input = {"query": "hello"}
    env = envelope_to_dict(
        build_capture(_sample_tree(), suite="s", agent_input=agent_input, capture_id="cap_t")
    )
    assert env["input_hash"] == canonical_hash(agent_input)


def test_identity_defaults_derive_from_capture() -> None:
    env = envelope_to_dict(
        build_capture(_sample_tree(), suite="support_agent", agent_input="x", capture_id="cap_xyz")
    )
    trace = env["trace"]
    assert trace["role"] == "source"
    assert trace["prompt_id"] == "support_agent"
    assert trace["run_id"] == "cap_xyz"
    assert trace["example_id"] == "cap_xyz"


# --- fixtures (D-1) ---


def test_build_fixture_table_keys_by_call_id_and_input_hash() -> None:
    env = build_capture(_sample_tree(), suite="s", agent_input="hi", capture_id="cap_t")
    table = build_fixture_table(env.trace)
    assert table[("c1", canonical_hash({"q": "x"}))] == {"hits": 1}


# --- filename ---


def test_capture_filename() -> None:
    assert capture_filename("cap_abc") == "cap_abc.json"
