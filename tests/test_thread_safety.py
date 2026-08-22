"""Concurrency-safety of SpanTree ordering + MemorySink under real OS threads (Phase 5)."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from evalshift import MemorySink, capture, record_model_call
from evalshift.capture.span import SpanTree
from evalshift.trace.serialize import build_capture
from tests.conftest import CaptureReader


def _events(capture_dict: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = capture_dict["trace"]["events"]
    return events


def test_spantree_concurrent_open_close_unique_orders() -> None:
    # Directly exercise the lock: many threads racing the non-atomic _counter bump.
    tree = SpanTree()
    n = 64

    def record(i: int) -> None:
        span = tree.open_span("tool", span_id=f"s{i}", start_ts=float(i))
        tree.close_span(span, end_ts=float(i) + 1.0)

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(record, range(n)))

    assert len(tree.spans) == n
    orders = [s.start_order for s in tree.spans] + [
        s.end_order for s in tree.spans if s.end_order is not None
    ]
    # 2N monotonic indices, all distinct (no dropped increment from a race).
    assert len(orders) == 2 * n
    assert len(set(orders)) == 2 * n
    assert sorted(orders) == list(range(1, 2 * n + 1))


async def test_threaded_tools_all_recorded(capturing: Path, read_captures: CaptureReader) -> None:
    @capture.tool
    def work(i: int) -> int:
        return i * i

    n = 12

    @capture.agent(suite="thread_suite", redact=False, tools=[])
    async def agent() -> None:
        # asyncio.to_thread copies the contextvars context, so the shared SpanTree is visible
        # in each worker thread; the lock keeps their concurrent open/close from racing.
        await asyncio.gather(*(asyncio.to_thread(work, i) for i in range(n)))

    await agent()
    events = _events(read_captures("thread_suite")[0])
    calls = [e for e in events if e["type"] == "tool_call"]
    results = [e for e in events if e["type"] == "tool_result"]
    assert len(calls) == n and len(results) == n
    seq = sorted(e["sequence_index"] for e in events)
    assert seq == list(range(len(events)))


async def test_threaded_agent_sessions_do_not_cross_contaminate_toolsets(
    capturing: Path, read_captures: CaptureReader
) -> None:
    """Real OS threads, not just interleaved coroutines: two agent sessions with two different
    toolsets, run via ``asyncio.to_thread`` exactly like ``test_threaded_tools_all_recorded``
    above. ``asyncio.to_thread`` copies the *current* contextvars Context into each worker thread
    at call time, before either agent has bound its own ``_current_toolset`` -- so this only
    passes if that binding stays local to the thread that made it, the same guarantee already
    proven for span parentage. There is no known bug; this pins the guarantee against a future
    refactor (e.g. a toolset cache keyed by something other than the contextvar).

    A ``threading.Barrier`` forces both sessions to be simultaneously "open" (both toolsets
    bound) before either records its call -- without it, these tiny synchronous functions can
    each run to completion inside one GIL time slice, so a leak wouldn't reliably show up (a
    plain-global stand-in for the contextvar, tried locally, only failed this test *with* the
    barrier forcing the rendezvous -- without it, the same broken code slipped through)."""
    barrier = threading.Barrier(2)

    @capture.agent(suite="thread_toolset_a", redact=False, tools=[{"name": "alpha_tool"}])
    def agent_a() -> None:
        barrier.wait(timeout=5)
        record_model_call(model_id="m", tools=None, output="a")

    @capture.agent(suite="thread_toolset_b", redact=False, tools=[{"name": "beta_tool"}])
    def agent_b() -> None:
        barrier.wait(timeout=5)
        record_model_call(model_id="m", tools=None, output="b")

    await asyncio.gather(asyncio.to_thread(agent_a), asyncio.to_thread(agent_b))

    [mc_a] = [e for e in _events(read_captures("thread_toolset_a")[0]) if e["type"] == "model_call"]
    [mc_b] = [e for e in _events(read_captures("thread_toolset_b")[0]) if e["type"] == "model_call"]
    assert mc_a["tools_offered"] == ["alpha_tool"]
    assert mc_b["tools_offered"] == ["beta_tool"]
    assert mc_a["toolset_ref"] != mc_b["toolset_ref"]


def test_memory_sink_concurrent_writes() -> None:
    sink = MemorySink()
    envelope = build_capture(SpanTree(), suite="s", agent_input={})
    m = 100

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda _: sink.write(envelope), range(m)))

    assert len(sink.captures) == m
    assert len(sink.flush()) == m
    assert sink.captures == ()
