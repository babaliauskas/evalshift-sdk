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
from tests.conformance.cli_models_vendored import AgentTrace
from tests.conftest import CaptureReader


@capture.tool
def _search(q: str) -> dict[str, Any]:
    return {"hits": 1}


@capture.tool
def _summarize(text: str) -> str:
    return f"summary:{text}"


@capture.agent(suite="support_agent")
def _handle(query: str) -> str:
    hits = _search(query)
    record_model_call(model_id="claude-opus-4-8", input=query, output="answer", input_tokens=12)
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
    @capture.agent(suite="support_agent")
    def boom() -> None:
        record_model_call(model_id="claude-opus-4-8")
        raise ValueError("kaboom")

    with pytest.raises(ValueError, match="kaboom"):
        boom()
    cap = read_captures("support_agent")[0]
    trace = AgentTrace.model_validate(cap["trace"])  # error event must stay CLI-valid
    assert any(e.type == "error" for e in trace.events)
