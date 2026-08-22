"""Parity harness: SDK schema constants vs the frozen CLI trace model.

Drift guard: the frozen ``schema.py`` field set must match the vendored CLI model
field-for-field, so a CLI contract change that isn't mirrored here fails the build. The
real-serializer round-trip lives in ``test_serialize_parity.py`` (Phase 1 replaced the Phase 0
hand-built placeholder); the shared ``_event`` builder below still backs the guardrail-verdict
and envelope (D-5b) checks.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from evalshift.trace import migrate, schema
from tests.conformance.cli_models_vendored import (
    AgentTrace,
    CaptureEnvelope,
    ErrorEvent,
    FinalOutputEvent,
    GuardrailEvent,
    ModelCallEvent,
    RetrievalEvent,
    ToolCallEvent,
    ToolResultEvent,
    TraceEventAdapter,
)

TS = "2026-06-16T12:00:00+00:00"

EVENT_MODELS: dict[str, type[BaseModel]] = {
    "model_call": ModelCallEvent,
    "tool_call": ToolCallEvent,
    "tool_result": ToolResultEvent,
    "retrieval": RetrievalEvent,
    "guardrail": GuardrailEvent,
    "final_output": FinalOutputEvent,
    "error": ErrorEvent,
}

_EXTRA: dict[str, dict[str, Any]] = {
    "model_call": {
        "model_id": "claude-opus-4-8",
        "input": "hi",
        "output": "yo",
        "input_tokens": 10,
        "output_tokens": 5,
        "cost_usd": 0.01,
        "latency_ms": 42,
    },
    "tool_call": {"name": "search", "arguments": {"q": "x"}, "call_id": "c1"},
    "tool_result": {"name": "search", "call_id": "c1", "result": {"hits": 1}},
    "retrieval": {"source": "kb", "query": "x", "documents": [{"id": "d1"}]},
    "guardrail": {"name": "pii", "verdict": "pass"},
    "final_output": {"text": "done"},
    "error": {"message": "boom", "category": "runtime"},
}


def _event(event_type: str, seq: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": event_type,
        "sequence_index": seq,
        "timestamp": TS,
        "metadata": {},
    }
    payload.update(_EXTRA[event_type])
    return payload


# --- drift guard: schema.py constants vs the vendored CLI model ---


def test_event_types_match_vendored_union() -> None:
    assert set(schema.EVENT_TYPES) == set(EVENT_MODELS)


@pytest.mark.parametrize("event_type", schema.EVENT_TYPES)
def test_event_fields_match_vendored_model(event_type: str) -> None:
    model = EVENT_MODELS[event_type]
    assert set(schema.EVENT_FIELDS[event_type]) == set(model.model_fields)
    assert set(schema.BASE_EVENT_FIELDS) <= set(model.model_fields)


def test_agenttrace_fields_match_vendored_model() -> None:
    assert set(schema.AGENT_TRACE_FIELDS) == set(AgentTrace.model_fields)


def test_envelope_keys_match_vendored_capture_envelope_model() -> None:
    assert set(schema.ENVELOPE_KEYS) == set(CaptureEnvelope.model_fields)


def test_reconstruction_map_matches_event_types() -> None:
    # The reverse-load dispatch map (Phase 8) must cover exactly the contract event types, so a
    # future event-type addition that isn't mirrored in migrate.py fails the build.
    assert set(migrate._EVENT_CLASS_BY_TYPE) == set(schema.EVENT_TYPES)


def test_current_version_is_supported() -> None:
    assert schema.SCHEMA_VERSION in schema.SUPPORTED_SCHEMA_VERSIONS


def test_trace_roles_match() -> None:
    assert set(schema.TRACE_ROLES) == {"source", "target"}
    for role in schema.TRACE_ROLES:
        AgentTrace.model_validate(
            {"run_id": "r", "prompt_id": "p", "example_id": "e", "role": role, "events": []}
        )


@pytest.mark.parametrize("verdict", schema.GUARDRAIL_VERDICTS)
def test_guardrail_verdicts_accepted(verdict: str) -> None:
    event = _event("guardrail", 0)
    event["verdict"] = verdict
    validated = TraceEventAdapter.validate_python(event)
    assert isinstance(validated, GuardrailEvent)
    assert validated.verdict == verdict


# --- model_call toolset fields (schema 2.0.0) ---


def test_model_call_toolset_fields_accepted_by_vendored_model() -> None:
    event = _event("model_call", 0)
    event["toolset_ref"] = "sha256:" + "a" * 64
    event["tools_offered"] = ["search", "summarize"]
    validated = TraceEventAdapter.validate_python(event)
    assert isinstance(validated, ModelCallEvent)
    assert validated.toolset_ref == "sha256:" + "a" * 64
    assert validated.tools_offered == ["search", "summarize"]


def test_model_call_toolset_fields_default_to_none_on_vendored_model() -> None:
    validated = TraceEventAdapter.validate_python(_event("model_call", 0))
    assert isinstance(validated, ModelCallEvent)
    assert validated.toolset_ref is None
    assert validated.tools_offered is None


# --- D-5b: schema_version lives in the capture envelope, never in the trace ---


def test_schema_version_lives_in_envelope_not_trace() -> None:
    assert "schema_version" in schema.ENVELOPE_KEYS
    assert "schema_version" not in schema.AGENT_TRACE_FIELDS

    envelope = {
        "schema_version": schema.SCHEMA_VERSION,
        "capture_id": "cap_x",
        "suite": "support_agent",
        "input_hash": "deadbeef",
        "code_version": "abc123",
        "created_at": TS,
        "trace": {
            "run_id": "r1",
            "prompt_id": "p1",
            "example_id": "e1",
            "role": "source",
            "events": [_event("final_output", 0)],
        },
        "conversation_id": None,
        "turn_index": None,
        "parent_capture_id": None,
    }
    assert set(envelope) == set(schema.ENVELOPE_KEYS)
    AgentTrace.model_validate(envelope["trace"])  # inner trace stays CLI-valid
    CaptureEnvelope.model_validate(envelope)  # whole envelope stays CLI-valid


def test_envelope_with_conversation_identity_validates_against_vendored_model() -> None:
    envelope = {
        "schema_version": schema.SCHEMA_VERSION,
        "capture_id": "cap_x",
        "suite": "support_agent",
        "input_hash": "deadbeef",
        "code_version": "abc123",
        "created_at": TS,
        "trace": {
            "run_id": "r1",
            "prompt_id": "p1",
            "example_id": "e1",
            "role": "source",
            "events": [_event("final_output", 0)],
        },
        "conversation_id": "conv_1",
        "turn_index": 2,
        "parent_capture_id": "cap_prev",
    }
    validated = CaptureEnvelope.model_validate(envelope)
    assert validated.conversation_id == "conv_1"
    assert validated.turn_index == 2
    assert validated.parent_capture_id == "cap_prev"


def test_schema_version_inside_trace_is_rejected() -> None:
    bad = {
        "run_id": "r1",
        "prompt_id": "p1",
        "example_id": "e1",
        "role": "source",
        "events": [],
        "schema_version": "1.0.0",
    }
    with pytest.raises(ValidationError):
        AgentTrace.model_validate(bad)
