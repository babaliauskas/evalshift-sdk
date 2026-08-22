"""Behavior of @capture.agent / @capture.tool / agent_session_async on async agents (Phase 5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from evalshift import capture, record_model_call
from evalshift.trace import schema
from evalshift.trace.serialize import canonical_hash
from tests.conftest import CaptureReader


def _events(capture_dict: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = capture_dict["trace"]["events"]
    return events


async def test_async_agent_writes_one_schema_valid_capture(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="async_suite", redact=False, tools=[])
    async def handle(query: str) -> str:
        record_model_call(model_id="claude-opus-4-8", input=query, output="ok", tools=[])
        return f"handled:{query}"

    result = await handle("hi")
    assert result == "handled:hi"
    caps = read_captures("async_suite")
    assert len(caps) == 1
    cap = caps[0]
    assert set(cap) == set(schema.ENVELOPE_KEYS)
    assert cap["schema_version"] == schema.SCHEMA_VERSION
    assert cap["suite"] == "async_suite"


async def test_async_agent_return_value_propagates(capturing: Path) -> None:
    sentinel = object()

    @capture.agent(suite="async_suite", redact=False, tools=[])
    async def returns_identity() -> object:
        return sentinel

    assert await returns_identity() is sentinel


async def test_async_agent_input_hash_canonical(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="async_suite", redact=False, tools=[])
    async def handle(query: str) -> None:
        return None

    await handle("hello")
    cap = read_captures("async_suite")[0]
    assert cap["input_hash"] == canonical_hash({"query": "hello"})


async def test_async_tool_call_and_result(capturing: Path, read_captures: CaptureReader) -> None:
    @capture.tool
    async def search(q: str) -> dict[str, Any]:
        return {"hits": 1}

    @capture.agent(suite="async_suite", redact=False, tools=[])
    async def agent() -> None:
        await search("x")

    await agent()
    events = _events(read_captures("async_suite")[0])
    calls = [e for e in events if e["type"] == "tool_call"]
    results = [e for e in events if e["type"] == "tool_result"]
    assert len(calls) == 1 and len(results) == 1
    assert calls[0]["name"] == "search"
    assert calls[0]["call_id"] == results[0]["call_id"]
    assert results[0]["result"] == {"hits": 1}


async def test_async_nested_tool_parentage(capturing: Path, read_captures: CaptureReader) -> None:
    @capture.tool
    async def inner() -> None:
        return None

    @capture.tool
    async def outer() -> None:
        await inner()

    @capture.agent(suite="async_suite", redact=False, tools=[])
    async def agent() -> None:
        await outer()

    await agent()
    calls = {
        e["name"]: e for e in _events(read_captures("async_suite")[0]) if e["type"] == "tool_call"
    }
    assert calls["outer"]["parent_call_id"] is None
    assert calls["inner"]["parent_call_id"] == calls["outer"]["call_id"]


async def test_agent_session_async_records(capturing: Path, read_captures: CaptureReader) -> None:
    async with capture.agent_session_async(
        suite="async_suite", agent_input={"q": 1}, redact=False, tools=[]
    ):
        record_model_call(model_id="claude-opus-4-8", tools=[])
    cap = read_captures("async_suite")[0]
    assert [e["type"] for e in _events(cap)] == ["model_call"]


async def test_agent_session_async_yields_none_when_gate_off(tmp_path: Path) -> None:
    async with capture.agent_session_async(suite="async_suite", redact=False, tools=[]) as tree:
        assert tree is None
    assert list(tmp_path.rglob("*.json")) == []


async def test_agent_session_async_writes_conversation_identity_fields(
    capturing: Path, read_captures: CaptureReader
) -> None:
    async with capture.agent_session_async(
        suite="async_suite",
        redact=False,
        agent_input="yes",
        conversation_id="c1",
        turn_index=1,
        parent_capture_id="cap_x",
        tools=[],
    ):
        record_model_call(model_id="claude-opus-4-8", tools=[])
    cap = read_captures("async_suite")[0]
    assert cap["conversation_id"] == "c1"
    assert cap["turn_index"] == 1
    assert cap["parent_capture_id"] == "cap_x"


async def test_agent_session_async_conversation_identity_written_on_error_path(
    capturing: Path, read_captures: CaptureReader
) -> None:
    with pytest.raises(ValueError, match="boom"):
        async with capture.agent_session_async(
            suite="async_suite",
            redact=False,
            agent_input="yes",
            conversation_id="c1",
            turn_index=2,
            parent_capture_id="cap_x",
            tools=[],
        ):
            raise ValueError("boom")
    cap = read_captures("async_suite")[0]
    assert cap["conversation_id"] == "c1"
    assert cap["turn_index"] == 2
    assert cap["parent_capture_id"] == "cap_x"


async def test_async_agent_decorator_accepts_static_conversation_identity(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.agent(
        suite="async_suite",
        redact=False,
        conversation_id="c_static",
        turn_index=0,
        parent_capture_id="cap_prev",
        tools=[],
    )
    async def handle(query: str) -> str:
        return f"handled:{query}"

    await handle("hi")
    cap = read_captures("async_suite")[0]
    assert cap["conversation_id"] == "c_static"
    assert cap["turn_index"] == 0
    assert cap["parent_capture_id"] == "cap_prev"


async def test_async_agent_gate_off_writes_nothing(tmp_path: Path) -> None:
    @capture.agent(suite="async_suite", redact=False, tools=[])
    async def handle() -> str:
        return "ok"

    assert await handle() == "ok"
    assert list(tmp_path.rglob("*.json")) == []


def test_sync_agent_still_records(capturing: Path, read_captures: CaptureReader) -> None:
    # Regression: the iscoroutinefunction split must not break the sync branch.
    @capture.agent(suite="sync_suite", redact=False, tools=[])
    def handle(query: str) -> str:
        record_model_call(model_id="claude-opus-4-8", input=query, tools=[])
        return f"handled:{query}"

    assert handle("hi") == "handled:hi"
    cap = read_captures("sync_suite")[0]
    assert [e["type"] for e in _events(cap)] == ["model_call"]
