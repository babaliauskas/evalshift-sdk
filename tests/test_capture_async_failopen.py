"""Async fail-open semantics: user exceptions propagate, partial capture still written (Phase 5)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from evalshift import capture, configure
from evalshift.hygiene import dedup, gc
from tests.conftest import CaptureReader


def _events(capture_dict: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = capture_dict["trace"]["events"]
    return events


async def test_async_user_exception_propagates_and_writes_capture(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="fail_suite", redact=False, tools=[])
    async def boom() -> None:
        raise ValueError("kaboom")

    with pytest.raises(ValueError, match="kaboom"):
        await boom()

    # Failed runs are the highest-value telemetry: partial capture + error event still written.
    cap = read_captures("fail_suite")[0]
    errors = [e for e in _events(cap) if e["type"] == "error"]
    assert len(errors) == 1
    assert errors[0]["message"] == "kaboom"
    assert errors[0]["category"] == "ValueError"


async def test_async_tool_exception_records_error_and_propagates(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.tool
    async def flaky() -> None:
        raise RuntimeError("tool down")

    @capture.agent(suite="fail_suite", redact=False, tools=[])
    async def agent() -> None:
        await flaky()

    with pytest.raises(RuntimeError, match="tool down"):
        await agent()

    events = _events(read_captures("fail_suite")[0])
    results = [e for e in events if e["type"] == "tool_result"]
    assert len(results) == 1
    assert results[0]["error"] == "tool down"


async def test_gather_one_tool_raises_records_that_tools_error(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.tool
    async def good() -> str:
        await asyncio.sleep(0)
        return "ok"

    @capture.tool
    async def bad() -> None:
        raise ValueError("nope")

    @capture.agent(suite="fail_suite", redact=False, tools=[])
    async def agent() -> None:
        await asyncio.gather(good(), bad())

    with pytest.raises(ValueError, match="nope"):
        await agent()

    # gather cancels siblings + re-raises; assert only on the raiser + that a capture exists.
    caps = read_captures("fail_suite")
    assert len(caps) == 1
    events = _events(caps[0])
    bad_results = [e for e in events if e["type"] == "tool_result" and e["name"] == "bad"]
    assert len(bad_results) == 1
    assert bad_results[0]["error"] == "nope"
    # The agent-level error event is recorded too.
    assert any(e["type"] == "error" for e in events)


# --- Phase 6 hygiene fault matrix (async mirror): never breaks or drops the host ---


async def test_async_sink_write_non_oserror_does_not_break_host(
    capturing: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ExplodingSink:
        def write(self, envelope: object) -> Path:
            raise RuntimeError("kaboom")

    monkeypatch.setattr("evalshift.config.active_sink", lambda: ExplodingSink())

    @capture.agent(suite="fail_suite", redact=False, tools=[])
    async def agent() -> str:
        return "ok"

    assert await agent() == "ok"


async def test_async_dedup_check_fault_does_not_break_host_or_drop_capture(
    capturing: Path, read_captures: CaptureReader, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure(dedup=True)

    def boom(*_a: object, **_k: object) -> bool:
        raise RuntimeError("dedup registry down")

    monkeypatch.setattr(dedup, "is_duplicate", boom)

    @capture.agent(suite="fail_suite", redact=False, tools=[])
    async def agent() -> str:
        return "ok"

    assert await agent() == "ok"
    assert len(read_captures("fail_suite")) == 1


async def test_async_gc_fault_does_not_break_host(
    capturing: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure(max_captures=1)

    def boom(*_a: object, **_k: object) -> list[Path]:
        raise RuntimeError("gc down")

    monkeypatch.setattr(gc, "evict", boom)

    @capture.agent(suite="fail_suite", redact=False, tools=[])
    async def agent() -> str:
        return "ok"

    assert await agent() == "ok"


async def test_async_hygiene_sink_construction_fault_does_not_break_host(
    capturing: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure(dedup=True)

    class BadHygiene:
        def __init__(self, *_a: object, **_k: object) -> None:
            raise RuntimeError("ctor down")

    monkeypatch.setattr("evalshift.config.HygieneSink", BadHygiene)

    @capture.agent(suite="fail_suite", redact=False, tools=[])
    async def agent() -> str:
        return "ok"

    assert await agent() == "ok"
