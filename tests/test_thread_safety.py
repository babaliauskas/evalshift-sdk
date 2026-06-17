"""Concurrency-safety of SpanTree ordering + MemorySink under real OS threads (Phase 5)."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from evalshift import MemorySink, capture
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

    @capture.agent(suite="thread_suite")
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


def test_memory_sink_concurrent_writes() -> None:
    sink = MemorySink()
    envelope = build_capture(SpanTree(), suite="s", agent_input={})
    m = 100

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda _: sink.write(envelope), range(m)))

    assert len(sink.captures) == m
    assert len(sink.flush()) == m
    assert sink.captures == ()
