"""The opt-in persistence gate at the @capture.agent boundary.

When ``configure(require_model_call=True)`` is set, content-free captures (no model_call
span) are silently dropped instead of written, across every write path: sync/async decorator
and both agent_session context managers, on the success and error branches alike. Real
captures are unaffected. With the flag off (default), the SDK's always-write contract holds.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evalshift import capture, configure, record_model_call
from tests.conftest import CaptureReader


@pytest.fixture(autouse=True)
def _require_model_call() -> None:
    """Every test in this module exercises the gate, so turn it on. Config is reset between
    tests by the autouse ``_isolate_config`` fixture in conftest."""
    configure(require_model_call=True)


def test_noop_agent_writes_nothing(capturing: Path, read_captures: CaptureReader) -> None:
    @capture.agent(suite="gate", redact=False, tools=[])
    def noop(x: str) -> str:
        return f"done:{x}"

    assert noop("hi") == "done:hi"
    assert read_captures("gate") == []


def test_tool_only_agent_writes_nothing(capturing: Path, read_captures: CaptureReader) -> None:
    @capture.tool(name="search")
    def search(q: str) -> str:
        return "hits"

    @capture.agent(suite="gate", redact=False, tools=[])
    def tool_only(x: str) -> str:
        search(x)  # records a tool span, but no model_call
        return "done"

    assert tool_only("hi") == "done"
    assert read_captures("gate") == []


def test_error_only_agent_writes_nothing_and_raises(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="gate", redact=False, tools=[])
    def boom(x: str) -> str:
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        boom("hi")
    assert read_captures("gate") == []


def test_real_agent_writes_one_capture(capturing: Path, read_captures: CaptureReader) -> None:
    @capture.agent(suite="gate", redact=False, tools=[])
    def real(x: str) -> str:
        record_model_call(
            model_id="claude-opus-4-8", input=x, output="ok", input_tokens=5, tools=[]
        )
        return "done"

    real("hi")
    assert len(read_captures("gate")) == 1


def test_model_call_then_error_is_kept_and_raises(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="gate", redact=False, tools=[])
    def call_then_boom(x: str) -> str:
        record_model_call(
            model_id="claude-opus-4-8", input=x, output="ok", input_tokens=5, tools=[]
        )
        raise ValueError("after call")

    with pytest.raises(ValueError, match="after call"):
        call_then_boom("hi")
    caps = read_captures("gate")
    assert len(caps) == 1
    kinds = {e["type"] for e in caps[0]["trace"]["events"]}
    assert "model_call" in kinds
    assert "error" in kinds


async def test_async_noop_agent_writes_nothing(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="gate", redact=False, tools=[])
    async def noop(x: str) -> str:
        return "done"

    assert await noop("hi") == "done"
    assert read_captures("gate") == []


async def test_async_real_agent_writes_one_capture(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="gate", redact=False, tools=[])
    async def real(x: str) -> str:
        record_model_call(
            model_id="claude-opus-4-8", input=x, output="ok", input_tokens=5, tools=[]
        )
        return "done"

    await real("hi")
    assert len(read_captures("gate")) == 1


def test_flag_off_preserves_always_write_contract(
    capturing: Path, read_captures: CaptureReader
) -> None:
    # Override the module fixture: with the gate off (SDK default), a no-op agent still writes.
    configure(require_model_call=False)

    @capture.agent(suite="gate", redact=False, tools=[])
    def noop(x: str) -> str:
        return "done"

    noop("hi")
    assert len(read_captures("gate")) == 1
