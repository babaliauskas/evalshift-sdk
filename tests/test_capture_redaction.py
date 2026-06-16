"""Redaction at capture time: payloads masked in-process before any byte hits disk (Phase 4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evalshift import capture, configure, record_model_call
from evalshift.redaction import default_redactor
from evalshift.trace.serialize import canonical_hash
from tests.conftest import CaptureReader


def _events(capture_dict: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = capture_dict["trace"]["events"]
    return events


def test_decorator_redactor_masks_tool_and_model(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.tool
    def lookup(email: str) -> str:
        return f"found {email}"

    @capture.agent(suite="red", redact=default_redactor)
    def agent() -> None:
        record_model_call(model_id="m", input="ping bob@corp.com", output="pong")
        lookup("bob@corp.com")

    agent()
    events = _events(read_captures("red")[0])
    call = next(e for e in events if e["type"] == "tool_call")
    result = next(e for e in events if e["type"] == "tool_result")
    model = next(e for e in events if e["type"] == "model_call")
    assert "bob@corp.com" not in str(call["arguments"])
    assert "[REDACTED_EMAIL]" in call["arguments"]["email"]
    assert "bob@corp.com" not in str(result["result"])
    assert "bob@corp.com" not in str(model["input"])


def test_global_configure_redactor_applies(capturing: Path, read_captures: CaptureReader) -> None:
    configure(redact=default_redactor)

    @capture.agent(suite="red")
    def agent() -> None:
        record_model_call(model_id="m", input="reach x@y.com", output="ok")

    agent()
    model = next(e for e in _events(read_captures("red")[0]) if e["type"] == "model_call")
    assert "x@y.com" not in str(model["input"])


def test_decorator_redactor_overrides_global(capturing: Path, read_captures: CaptureReader) -> None:
    def tag(value: Any) -> Any:
        return "[DECOR]"

    configure(redact=default_redactor)

    @capture.agent(suite="red", redact=tag)
    def agent() -> None:
        record_model_call(model_id="m", input="x@y.com", output="ok")

    agent()
    model = next(e for e in _events(read_captures("red")[0]) if e["type"] == "model_call")
    assert model["input"] == "[DECOR]"


def test_no_redactor_keeps_payload_verbatim(capturing: Path, read_captures: CaptureReader) -> None:
    @capture.agent(suite="red")
    def agent() -> None:
        record_model_call(model_id="m", input="x@y.com", output="ok")

    agent()
    model = next(e for e in _events(read_captures("red")[0]) if e["type"] == "model_call")
    assert model["input"] == "x@y.com"


def test_tool_input_hash_uses_redacted_arguments(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.tool
    def lookup(email: str) -> str:
        return "ok"

    @capture.agent(suite="red", redact=default_redactor)
    def agent() -> None:
        lookup("bob@corp.com")

    agent()
    events = _events(read_captures("red")[0])
    call = next(e for e in events if e["type"] == "tool_call")
    result = next(e for e in events if e["type"] == "tool_result")
    stored_hash = result["metadata"]["evalshift"]["input_hash"]
    # The fixture key derives from the *redacted* arguments — capture stays self-consistent.
    assert stored_hash == canonical_hash(call["arguments"])


def test_redactor_failure_drops_capture_but_host_returns(
    capturing: Path, read_captures: CaptureReader
) -> None:
    def boom(value: Any) -> Any:
        raise RuntimeError("redactor blew up")

    @capture.agent(suite="red", redact=boom)
    def agent() -> str:
        record_model_call(model_id="m", input="x@y.com", output="ok")
        return "host-result"

    # Fail-closed: redactor crash drops the capture, never writes possibly-unredacted PII...
    assert agent() == "host-result"  # ...but the host agent is untouched.
    assert read_captures("red") == []
