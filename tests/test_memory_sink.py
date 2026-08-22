"""MemorySink buffers captures in-process; the host flushes them (Phase 3)."""

from __future__ import annotations

from evalshift.capture.span import SpanTree
from evalshift.sinks.memory import MemorySink
from evalshift.trace.models import CaptureEnvelope
from evalshift.trace.serialize import build_capture


def _envelope(capture_id: str = "cap_abc") -> CaptureEnvelope:
    return build_capture(SpanTree(), suite="support_agent", agent_input="hi", capture_id=capture_id)


def test_flush_returns_written_captures_then_clears() -> None:
    sink = MemorySink()
    env = _envelope()
    sink.write(env)
    assert sink.flush() == [env]
    assert sink.flush() == []


def test_multiple_writes_preserve_order() -> None:
    sink = MemorySink()
    a, b = _envelope("cap_a"), _envelope("cap_b")
    sink.write(a)
    sink.write(b)
    assert sink.flush() == [a, b]


def test_captures_view_is_non_draining() -> None:
    sink = MemorySink()
    env = _envelope()
    sink.write(env)
    assert list(sink.captures) == [env]
    # Inspecting via the view must not drain the buffer.
    assert sink.flush() == [env]
