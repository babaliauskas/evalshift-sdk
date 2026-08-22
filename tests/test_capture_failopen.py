"""Fail-open guarantees: capture bookkeeping never breaks or masks the host agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from evalshift import capture, configure, record_model_call
from evalshift.capture import api
from evalshift.capture.span import SpanTree
from evalshift.hygiene import dedup, gc
from tests.conftest import CaptureReader


def _events(capture_dict: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = capture_dict["trace"]["events"]
    return events


# --- user exceptions propagate, capture still written ---


def test_user_exception_propagates(capturing: Path) -> None:
    @capture.agent(suite="fo_suite", redact=False, tools=[])
    def boom() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        boom()


def test_capture_with_error_event_still_written(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="fo_suite", redact=False, tools=[])
    def boom() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError):
        boom()
    cap = read_captures("fo_suite")[0]
    errors = [e for e in _events(cap) if e["type"] == "error"]
    assert len(errors) == 1
    assert errors[0]["message"] == "boom"
    assert errors[0]["category"] == "ValueError"


def test_tool_exception_propagates_and_records_error(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.tool
    def failing() -> None:
        raise RuntimeError("tool boom")

    @capture.agent(suite="fo_suite", redact=False, tools=[])
    def agent() -> None:
        failing()

    with pytest.raises(RuntimeError, match="tool boom"):
        agent()
    result = next(e for e in _events(read_captures("fo_suite")[0]) if e["type"] == "tool_result")
    assert result["error"] == "tool boom"


def test_empty_message_exception_falls_back_to_category(
    capturing: Path, read_captures: CaptureReader
) -> None:
    """str(exc) can be empty (CancelledError, bare KeyboardInterrupt); the trace
    contract requires a non-empty error message, so fall back to the type name."""

    @capture.agent(suite="fo_suite", redact=False, tools=[])
    def cancelled() -> None:
        raise TimeoutError

    with pytest.raises(TimeoutError):
        cancelled()
    cap = read_captures("fo_suite")[0]
    errors = [e for e in _events(cap) if e["type"] == "error"]
    assert errors[0]["message"] == "TimeoutError"
    assert errors[0]["category"] == "TimeoutError"


def test_keyboardinterrupt_propagates_and_is_recorded(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="fo_suite", redact=False, tools=[])
    def boom() -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        boom()
    cap = read_captures("fo_suite")[0]
    errors = [e for e in _events(cap) if e["type"] == "error"]
    assert errors[0]["category"] == "KeyboardInterrupt"


# --- bookkeeping faults never reach the host ---


def test_sink_write_fault_does_not_break_host(
    capturing: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BrokenSink:
        def write(self, envelope: object) -> Path:
            raise OSError("read-only fs")

    monkeypatch.setattr("evalshift.config.active_sink", lambda: BrokenSink())

    @capture.agent(suite="fo_suite", redact=False, tools=[])
    def agent() -> str:
        return "ok"

    assert agent() == "ok"


def test_build_capture_fault_does_not_break_host(
    capturing: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("serialize failed")

    monkeypatch.setattr(api, "build_capture", boom)

    @capture.agent(suite="fo_suite", redact=False, tools=[])
    def agent() -> str:
        return "ok"

    assert agent() == "ok"


# --- Phase 6 hygiene fault matrix: hygiene machinery never breaks or drops the host ---


def test_sink_write_non_oserror_does_not_break_host(
    capturing: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ExplodingSink:
        def write(self, envelope: object) -> Path:
            raise RuntimeError("kaboom")  # not an OSError -> exercises the outer fail_open

    monkeypatch.setattr("evalshift.config.active_sink", lambda: ExplodingSink())

    @capture.agent(suite="fo_suite", redact=False, tools=[])
    def agent() -> str:
        return "ok"

    assert agent() == "ok"


def test_dedup_check_fault_does_not_break_host_or_drop_capture(
    capturing: Path, read_captures: CaptureReader, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure(dedup=True)

    def boom(*_a: object, **_k: object) -> bool:
        raise RuntimeError("dedup registry down")

    monkeypatch.setattr(dedup, "is_duplicate", boom)

    @capture.agent(suite="fo_suite", redact=False, tools=[])
    def agent() -> str:
        return "ok"

    assert agent() == "ok"
    assert len(read_captures("fo_suite")) == 1  # dedup fault must not suppress the write


def test_gc_fault_does_not_break_host(capturing: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    configure(max_captures=1)

    def boom(*_a: object, **_k: object) -> list[Path]:
        raise RuntimeError("gc down")

    monkeypatch.setattr(gc, "evict", boom)

    @capture.agent(suite="fo_suite", redact=False, tools=[])
    def agent() -> str:
        return "ok"

    assert agent() == "ok"


def test_hygiene_sink_construction_fault_does_not_break_host(
    capturing: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure(dedup=True)

    class BadHygiene:
        def __init__(self, *_a: object, **_k: object) -> None:
            raise RuntimeError("ctor down")

    monkeypatch.setattr("evalshift.config.HygieneSink", BadHygiene)

    @capture.agent(suite="fo_suite", redact=False, tools=[])
    def agent() -> str:
        return "ok"

    assert agent() == "ok"


def test_open_span_fault_degrades_to_passthrough(
    capturing: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("span open failed")

    monkeypatch.setattr(SpanTree, "open_span", boom)

    @capture.tool
    def tool() -> str:
        return "tool-ok"

    @capture.agent(suite="fo_suite", redact=False, tools=[])
    def agent() -> str:
        record_model_call(model_id="m", tools=[])
        return tool()

    assert agent() == "tool-ok"
