"""Behavior of @capture.agent and record_model_call when capture is enabled."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evalshift import capture, record_model_call
from evalshift.trace import schema
from evalshift.trace.serialize import canonical_hash
from tests.conftest import CaptureReader


@capture.agent(suite="agent_suite")
def _agent(query: str) -> str:
    record_model_call(model_id="claude-opus-4-8", input=query, output="ok", input_tokens=10)
    return f"handled:{query}"


def _events(capture_dict: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = capture_dict["trace"]["events"]
    return events


def test_written_capture_has_envelope_keys(capturing: Path, read_captures: CaptureReader) -> None:
    _agent("hi")
    cap = read_captures("agent_suite")[0]
    assert set(cap) == set(schema.ENVELOPE_KEYS)
    assert cap["schema_version"] == schema.SCHEMA_VERSION
    assert cap["suite"] == "agent_suite"


def test_input_hash_is_canonical_hash_of_bound_args(
    capturing: Path, read_captures: CaptureReader
) -> None:
    _agent("hello")
    cap = read_captures("agent_suite")[0]
    assert cap["input_hash"] == canonical_hash({"query": "hello"})


def test_non_jsonable_arg_does_not_crash(capturing: Path, read_captures: CaptureReader) -> None:
    @capture.agent(suite="agent_suite")
    def takes_object(obj: object) -> str:
        return "done"

    assert takes_object(object()) == "done"
    assert len(read_captures("agent_suite")) == 1


def test_record_model_call_appears_in_trace(capturing: Path, read_captures: CaptureReader) -> None:
    _agent("hi")
    cap = read_captures("agent_suite")[0]
    model_calls = [e for e in _events(cap) if e["type"] == "model_call"]
    assert len(model_calls) == 1
    assert model_calls[0]["model_id"] == "claude-opus-4-8"
    assert model_calls[0]["input_tokens"] == 10


def test_record_model_call_is_noop_outside_session(tmp_path: Path) -> None:
    # No active agent => no tree => no-op, no crash, nothing written.
    record_model_call(model_id="claude-opus-4-8")
    assert list(tmp_path.rglob("*.json")) == []


def test_return_value_propagates_when_capturing(capturing: Path) -> None:
    sentinel = object()

    @capture.agent(suite="agent_suite")
    def returns_identity() -> object:
        return sentinel

    assert returns_identity() is sentinel


def test_agent_session_context_manager_records(
    capturing: Path, read_captures: CaptureReader
) -> None:
    with capture.agent_session(suite="agent_suite", agent_input={"q": 1}):
        record_model_call(model_id="claude-opus-4-8")
    cap = read_captures("agent_suite")[0]
    assert [e["type"] for e in _events(cap)] == ["model_call"]


def test_agent_session_yields_none_when_gate_off(tmp_path: Path) -> None:
    with capture.agent_session(suite="agent_suite") as tree:
        assert tree is None
    assert list(tmp_path.rglob("*.json")) == []
