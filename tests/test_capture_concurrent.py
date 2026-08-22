"""Concurrent tool calls via asyncio.gather: parentage, ordering, no contextvar bleed (Phase 5)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from evalshift import capture, record_model_call
from tests.conftest import CaptureReader


def _events(capture_dict: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = capture_dict["trace"]["events"]
    return events


async def test_gather_three_tools_all_recorded(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.tool
    async def t1() -> str:
        await asyncio.sleep(0)
        return "1"

    @capture.tool
    async def t2() -> str:
        await asyncio.sleep(0)
        return "2"

    @capture.tool
    async def t3() -> str:
        await asyncio.sleep(0)
        return "3"

    @capture.agent(suite="conc_suite", redact=False, tools=[])
    async def agent() -> None:
        await asyncio.gather(t1(), t2(), t3())

    await agent()
    events = _events(read_captures("conc_suite")[0])
    calls = [e for e in events if e["type"] == "tool_call"]
    results = [e for e in events if e["type"] == "tool_result"]
    assert len(calls) == 3 and len(results) == 3
    assert {e["name"] for e in calls} == {"t1", "t2", "t3"}


async def test_gather_siblings_are_top_level(capturing: Path, read_captures: CaptureReader) -> None:
    @capture.tool
    async def t1() -> None:
        await asyncio.sleep(0)

    @capture.tool
    async def t2() -> None:
        await asyncio.sleep(0)

    @capture.agent(suite="conc_suite", redact=False, tools=[])
    async def agent() -> None:
        await asyncio.gather(t1(), t2())

    await agent()
    calls = [e for e in _events(read_captures("conc_suite")[0]) if e["type"] == "tool_call"]
    # No contextvar bleed: concurrent siblings each opened at agent top level.
    assert all(c["parent_call_id"] is None for c in calls)


async def test_gather_sequence_index_dense_no_collisions(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.tool
    async def tool(n: int) -> int:
        await asyncio.sleep(0)
        return n

    @capture.agent(suite="conc_suite", redact=False, tools=[])
    async def agent() -> None:
        await asyncio.gather(*(tool(i) for i in range(5)))

    await agent()
    events = _events(read_captures("conc_suite")[0])
    seq = sorted(e["sequence_index"] for e in events)
    # Dense 0..N-1, no duplicates — even though spans overlap in wall-clock time.
    assert seq == list(range(len(events)))


async def test_nested_gather_parentage(capturing: Path, read_captures: CaptureReader) -> None:
    @capture.tool
    async def child_a() -> None:
        await asyncio.sleep(0)

    @capture.tool
    async def child_b() -> None:
        await asyncio.sleep(0)

    @capture.tool
    async def parent() -> None:
        await asyncio.gather(child_a(), child_b())

    @capture.agent(suite="conc_suite", redact=False, tools=[])
    async def agent() -> None:
        await parent()

    await agent()
    calls = {
        e["name"]: e for e in _events(read_captures("conc_suite")[0]) if e["type"] == "tool_call"
    }
    assert calls["parent"]["parent_call_id"] is None
    # Both gather children carry the parent tool's call_id — no leak to agent top level.
    assert calls["child_a"]["parent_call_id"] == calls["parent"]["call_id"]
    assert calls["child_b"]["parent_call_id"] == calls["parent"]["call_id"]


async def test_gather_two_agent_sessions_do_not_cross_contaminate_toolsets(
    capturing: Path, read_captures: CaptureReader
) -> None:
    """The regression class D-toolset exists to prevent -- an agent's toolset attributed to a
    *different* agent's call -- is 'the original bug wearing a different hat': the same class of
    contextvar bleed the sibling/nested tests above already pin for ``parent_call_id``.
    ``_current_toolset`` (``capture/state.py``) uses the identical ``ContextVar`` idiom as
    ``_current_tree``/``_current_parent``; this pins that a second, differently-toolset'd session
    running concurrently never leaks its toolset into the first, under real interleaving (the
    ``asyncio.sleep(0)`` forces both sessions to be open at once before either records its call).
    """

    @capture.agent(suite="conc_toolset_a", redact=False, tools=[{"name": "alpha_tool"}])
    async def agent_a() -> None:
        await asyncio.sleep(0)
        record_model_call(model_id="m", tools=None, output="a")

    @capture.agent(suite="conc_toolset_b", redact=False, tools=[{"name": "beta_tool"}])
    async def agent_b() -> None:
        await asyncio.sleep(0)
        record_model_call(model_id="m", tools=None, output="b")

    await asyncio.gather(agent_a(), agent_b())

    [mc_a] = [e for e in _events(read_captures("conc_toolset_a")[0]) if e["type"] == "model_call"]
    [mc_b] = [e for e in _events(read_captures("conc_toolset_b")[0]) if e["type"] == "model_call"]
    assert mc_a["tools_offered"] == ["alpha_tool"]
    assert mc_b["tools_offered"] == ["beta_tool"]
    assert mc_a["toolset_ref"] != mc_b["toolset_ref"]


async def test_no_contextvar_bleed_between_sibling_and_nested(
    capturing: Path, read_captures: CaptureReader
) -> None:
    # One sibling spawns a nested call; the other sibling must NOT inherit that nested parentage.
    @capture.tool
    async def leaf() -> None:
        await asyncio.sleep(0)

    @capture.tool
    async def with_child() -> None:
        await leaf()

    @capture.tool
    async def plain() -> None:
        await asyncio.sleep(0)

    @capture.agent(suite="conc_suite", redact=False, tools=[])
    async def agent() -> None:
        await asyncio.gather(with_child(), plain())

    await agent()
    calls = {
        e["name"]: e for e in _events(read_captures("conc_suite")[0]) if e["type"] == "tool_call"
    }
    assert calls["with_child"]["parent_call_id"] is None
    assert calls["plain"]["parent_call_id"] is None
    assert calls["leaf"]["parent_call_id"] == calls["with_child"]["call_id"]
