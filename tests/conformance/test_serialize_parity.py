"""Conformance: real SDK serializer output validates against the frozen CLI model.

Phase 1 replaces the Phase 0 hand-built placeholder round-trip: a span tree exercising every
event kind is serialized by ``evalshift.trace.serialize`` and the resulting JSON is validated
through the vendored pydantic ``AgentTrace`` / ``TraceEventAdapter`` (the CLI contract). Pydantic
is a dev/test dependency only — never imported by the SDK at runtime.
"""

from __future__ import annotations

import json

from evalshift.capture.span import SpanTree
from evalshift.trace import schema
from evalshift.trace.serialize import build_capture, envelope_to_dict
from tests.conformance.cli_models_vendored import AgentTrace, CaptureEnvelope, TraceEventAdapter


def _full_tree() -> SpanTree:
    tree = SpanTree()
    m = tree.open_span(
        "model_call",
        span_id="m1",
        start_ts=1.0,
        data={"model_id": "claude-opus-4-8", "input": "hi", "output": "yo", "input_tokens": 3},
    )
    tree.close_span(m, end_ts=1.5)
    t = tree.open_span(
        "tool", span_id="c1", start_ts=2.0, data={"name": "search", "arguments": {"q": "x"}}
    )
    tree.close_span(t, end_ts=2.5, result={"hits": 1})
    tree.open_span(
        "retrieval",
        span_id="r1",
        start_ts=3.0,
        data={"source": "kb", "query": "x", "documents": [{"id": "d1"}]},
    )
    tree.open_span("guardrail", span_id="g1", start_ts=4.0, data={"name": "pii", "verdict": "pass"})
    tree.open_span(
        "error", span_id="e1", start_ts=5.0, data={"message": "boom", "category": "runtime"}
    )
    tree.open_span("final_output", span_id="f1", start_ts=6.0, data={"text": "done"})
    return tree


def test_serialized_trace_validates_against_cli_model() -> None:
    env = envelope_to_dict(
        build_capture(_full_tree(), suite="support_agent", agent_input="hi", capture_id="cap_t")
    )
    trace = AgentTrace.model_validate(env["trace"])  # raises if not CLI-valid
    assert [e.type for e in trace.events] == [
        "model_call",
        "tool_call",
        "tool_result",
        "retrieval",
        "guardrail",
        "error",
        "final_output",
    ]


def test_every_event_type_emitted_and_individually_valid() -> None:
    env = envelope_to_dict(
        build_capture(_full_tree(), suite="s", agent_input="hi", capture_id="cap_t")
    )
    emitted = set()
    for event_dict in env["trace"]["events"]:
        validated = TraceEventAdapter.validate_python(event_dict)
        emitted.add(validated.type)
    # all 7 contract event types appear (tool span yields both tool_call and tool_result)
    assert emitted == set(schema.EVENT_TYPES)


def test_serialized_trace_round_trips_through_cli_model() -> None:
    env = envelope_to_dict(
        build_capture(_full_tree(), suite="s", agent_input="hi", capture_id="cap_t")
    )
    trace = AgentTrace.model_validate(env["trace"])
    reloaded = AgentTrace.model_validate(json.loads(trace.model_dump_json()))
    assert reloaded == trace


def test_concurrency_metadata_survives_cli_validation() -> None:
    env = envelope_to_dict(
        build_capture(_full_tree(), suite="s", agent_input="hi", capture_id="cap_t")
    )
    trace = AgentTrace.model_validate(env["trace"])
    model_call = next(e for e in trace.events if e.type == "model_call")
    assert model_call.metadata["evalshift"]["span_id"] == "m1"
    tool_result = next(e for e in trace.events if e.type == "tool_result")
    assert "input_hash" in tool_result.metadata["evalshift"]


def test_envelope_inner_trace_has_no_schema_version() -> None:
    env = envelope_to_dict(
        build_capture(_full_tree(), suite="s", agent_input="hi", capture_id="cap_t")
    )
    assert "schema_version" not in env["trace"]
    # the vendored model rejects schema_version, so a clean validate proves it is absent
    AgentTrace.model_validate(env["trace"])


def test_serialized_envelope_validates_against_cli_capture_envelope_model() -> None:
    env = envelope_to_dict(
        build_capture(_full_tree(), suite="support_agent", agent_input="hi", capture_id="cap_t")
    )
    CaptureEnvelope.model_validate(env)  # raises if not CLI-valid


def test_serialized_envelope_with_conversation_identity_validates_against_cli_model() -> None:
    env = envelope_to_dict(
        build_capture(
            _full_tree(),
            suite="support_agent",
            agent_input="yes",
            capture_id="cap_t",
            conversation_id="conv_1",
            turn_index=3,
            parent_capture_id="cap_prev",
        )
    )
    validated = CaptureEnvelope.model_validate(env)
    assert validated.conversation_id == "conv_1"
    assert validated.turn_index == 3
    assert validated.parent_capture_id == "cap_prev"
