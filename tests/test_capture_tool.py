"""Behavior of @capture.tool: span shape, argument binding, nesting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evalshift import capture
from tests.conftest import CaptureReader


def _events(capture_dict: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = capture_dict["trace"]["events"]
    return events


def test_tool_records_call_and_result(capturing: Path, read_captures: CaptureReader) -> None:
    @capture.tool
    def search(q: str) -> dict[str, Any]:
        return {"hits": 1}

    @capture.agent(suite="tool_suite", redact=False, tools=[])
    def agent() -> None:
        search("x")

    agent()
    events = _events(read_captures("tool_suite")[0])
    calls = [e for e in events if e["type"] == "tool_call"]
    results = [e for e in events if e["type"] == "tool_result"]
    assert len(calls) == 1 and len(results) == 1
    assert calls[0]["name"] == "search"
    assert calls[0]["call_id"] == results[0]["call_id"]
    assert results[0]["result"] == {"hits": 1}


def test_tool_arguments_bound_by_param_name(capturing: Path, read_captures: CaptureReader) -> None:
    @capture.tool
    def search(q: str, limit: int = 5) -> None:
        return None

    @capture.agent(suite="tool_suite", redact=False, tools=[])
    def agent() -> None:
        search("x")

    agent()
    call = next(e for e in _events(read_captures("tool_suite")[0]) if e["type"] == "tool_call")
    assert call["arguments"] == {"q": "x", "limit": 5}


def test_tool_arguments_fallback_when_unbindable(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.tool
    def variadic(*args: int, **kwargs: int) -> None:
        return None

    @capture.agent(suite="tool_suite", redact=False, tools=[])
    def agent() -> None:
        variadic(1, 2, k=3)

    agent()
    call = next(e for e in _events(read_captures("tool_suite")[0]) if e["type"] == "tool_call")
    # signature.bind succeeds for *args/**kwargs, exposing them under their param names.
    assert call["arguments"] == {"args": [1, 2], "kwargs": {"k": 3}}


def test_nested_tool_parentage(capturing: Path, read_captures: CaptureReader) -> None:
    @capture.tool
    def inner() -> None:
        return None

    @capture.tool
    def outer() -> None:
        inner()

    @capture.agent(suite="tool_suite", redact=False, tools=[])
    def agent() -> None:
        outer()

    agent()
    calls = {
        e["name"]: e for e in _events(read_captures("tool_suite")[0]) if e["type"] == "tool_call"
    }
    assert calls["outer"]["parent_call_id"] is None
    assert calls["inner"]["parent_call_id"] == calls["outer"]["call_id"]


def test_tool_outside_session_is_passthrough(tmp_path: Path) -> None:
    @capture.tool
    def search(q: str) -> str:
        return "result"

    assert search("x") == "result"
    assert list(tmp_path.rglob("*.json")) == []


def test_tool_decorator_bare_and_named_forms(capturing: Path, read_captures: CaptureReader) -> None:
    @capture.tool(name="renamed")
    def original() -> None:
        return None

    @capture.agent(suite="tool_suite", redact=False, tools=[])
    def agent() -> None:
        original()

    agent()
    call = next(e for e in _events(read_captures("tool_suite")[0]) if e["type"] == "tool_call")
    assert call["name"] == "renamed"
