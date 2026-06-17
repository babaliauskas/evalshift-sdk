"""The live recording structure captured in-process: ``Span`` / ``SpanTree``.

A ``Span`` is one timed operation an agent performs (a model call, a tool invocation, a
retrieval, …). ``SpanTree`` collects spans in the order they are opened/closed and stamps each
open/close with a monotonic operation counter so the serializer can derive a stable total order
even when spans overlap in wall-clock time (concurrent tool calls — D-3).

This module holds *no* serialization or schema knowledge; it is the raw recording. The serializer
(``evalshift.trace.serialize``) turns a tree into a CLI-valid ``AgentTrace`` + capture envelope.

Stdlib only (D-deps).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Literal

#: The kind of operation a span records. A ``tool`` span serializes to *two* trace events
#: (``tool_call`` at its start, ``tool_result`` at its close); every other kind serializes to one.
SpanKind = Literal[
    "model_call",
    "tool",
    "retrieval",
    "guardrail",
    "final_output",
    "error",
]


@dataclass
class Span:
    """One timed operation in an agent timeline.

    ``span_id`` doubles as the tool ``call_id`` for ``tool`` spans. ``data`` holds the
    type-specific payload (e.g. ``name``/``arguments``/``result`` for tools, ``model_id``/
    ``input``/``output``/token counts for model calls); the serializer reads from it per kind.
    ``start_order``/``end_order`` are monotonic operation indices assigned by the owning
    ``SpanTree`` — they break ties when wall-clock timestamps collide.
    """

    kind: SpanKind
    span_id: str
    start_ts: float
    start_order: int
    parent_call_id: str | None = None
    end_ts: float | None = None
    end_order: int | None = None
    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpanTree:
    """Ordered collection of spans recorded during one captured agent invocation.

    The ``_lock`` guards order assignment and the span list so concurrent tool calls run from OS
    threads (``asyncio.to_thread`` / ``run_in_executor``) can't race the non-atomic ``_counter``
    bump into duplicate ``start_order`` values (Phase 5). Pure single-event-loop ``asyncio`` never
    contends it; the uncontended acquire is nanoseconds.
    """

    spans: list[Span] = field(default_factory=list)
    _counter: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def __iter__(self) -> Any:
        return iter(self.spans)

    def _next_order_locked(self) -> int:
        """Assign the next monotonic op-index. Caller must already hold ``self._lock``."""
        self._counter += 1
        return self._counter

    def open_span(
        self,
        kind: SpanKind,
        *,
        span_id: str,
        start_ts: float,
        parent_call_id: str | None = None,
        data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Span:
        """Record the start of an operation and register it on the tree."""
        with self._lock:
            span = Span(
                kind=kind,
                span_id=span_id,
                start_ts=start_ts,
                start_order=self._next_order_locked(),
                parent_call_id=parent_call_id,
                data=dict(data) if data is not None else {},
                metadata=dict(metadata) if metadata is not None else {},
            )
            self.spans.append(span)
        return span

    def close_span(self, span: Span, *, end_ts: float, **data_update: Any) -> None:
        """Record the end of an operation, merging any result payload into ``span.data``."""
        with self._lock:
            span.end_ts = end_ts
            span.end_order = self._next_order_locked()
            span.data.update(data_update)
