"""Conformance: a capture written end-to-end by @capture.agent validates against the CLI model.

Drives a real decorated agent (two tools + one model call) with the gate on, writing through the
FileSink to disk, then validates the loaded trace through the vendored pydantic ``AgentTrace``
(the CLI contract). Pydantic is a dev/test dependency only — never imported by the SDK at runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from evalshift import capture, record_model_call
from tests.conformance.cli_models_vendored import AgentTrace, CaptureEnvelope, ModelCallEvent
from tests.conftest import CaptureReader


@capture.tool
def _search(q: str) -> dict[str, Any]:
    return {"hits": 1}


@capture.tool
def _summarize(text: str) -> str:
    return f"summary:{text}"


@capture.agent(suite="support_agent", redact=False, tools=[])
def _handle(query: str) -> str:
    hits = _search(query)
    record_model_call(
        model_id="claude-opus-4-8", input=query, output="answer", input_tokens=12, tools=[]
    )
    return _summarize(str(hits))


def test_written_capture_trace_validates_against_cli_model(
    capturing: Path, read_captures: CaptureReader
) -> None:
    _handle("hi")
    cap = read_captures("support_agent")[0]
    trace = AgentTrace.model_validate(cap["trace"])  # raises if not CLI-valid
    types = [e.type for e in trace.events]
    assert types.count("tool_call") == 2
    assert types.count("tool_result") == 2
    assert types.count("model_call") == 1


def test_written_capture_inner_trace_has_no_schema_version(
    capturing: Path, read_captures: CaptureReader
) -> None:
    _handle("hi")
    cap = read_captures("support_agent")[0]
    assert "schema_version" not in cap["trace"]
    AgentTrace.model_validate(cap["trace"])  # vendored model rejects schema_version


def test_failed_agent_capture_is_still_cli_valid(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="support_agent", redact=False, tools=[])
    def boom() -> None:
        record_model_call(model_id="claude-opus-4-8", tools=[])
        raise ValueError("kaboom")

    with pytest.raises(ValueError, match="kaboom"):
        boom()
    cap = read_captures("support_agent")[0]
    trace = AgentTrace.model_validate(cap["trace"])  # error event must stay CLI-valid
    assert any(e.type == "error" for e in trace.events)


def test_conversation_turn_written_via_agent_session_is_cli_valid(
    capturing: Path, read_captures: CaptureReader
) -> None:
    with capture.agent_session(
        suite="support_agent",
        redact=False,
        agent_input="yes",
        conversation_id="conv_1",
        turn_index=2,
        parent_capture_id="cap_prev",
        tools=[],
    ):
        record_model_call(model_id="claude-opus-4-8", input="yes", output="ok", tools=[])
    cap = read_captures("support_agent")[0]
    validated = CaptureEnvelope.model_validate(cap)  # whole envelope, including new fields
    assert validated.conversation_id == "conv_1"
    assert validated.turn_index == 2
    assert validated.parent_capture_id == "cap_prev"


def test_written_requested_tool_calls_validate_against_cli_model(
    capturing: Path, read_captures: CaptureReader
) -> None:
    """A capture carrying what the model *asked for* stays CLI-valid: the CLI's strict
    ``RequestedToolCall`` is ``extra="forbid"``, so the SDK's normalisation to exactly
    ``{name, arguments, call_id}`` is what makes this pass (schema 2.1.0, D-requested)."""

    @capture.agent(suite="support_agent", redact=False, tools=[])
    def requesting() -> None:
        record_model_call(
            model_id="claude-opus-4-8",
            tools=[],
            output="",
            requested_tool_calls=[
                {"type": "tool_use", "name": "search", "arguments": {"q": "x"}, "call_id": "t1"},
                {"name": "summarize"},
            ],
        )

    requesting()
    cap = read_captures("support_agent")[0]
    trace = AgentTrace.model_validate(cap["trace"])
    [model_call] = [e for e in trace.events if isinstance(e, ModelCallEvent)]
    assert model_call.requested_tool_calls is not None
    assert [call.name for call in model_call.requested_tool_calls] == ["search", "summarize"]
    assert model_call.requested_tool_calls[1].arguments == {}
    assert model_call.requested_tool_calls[1].call_id is None
