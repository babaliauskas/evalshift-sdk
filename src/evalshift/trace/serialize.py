"""Turn a recorded :class:`~evalshift.capture.span.SpanTree` into a capture envelope.

Pure functions only — **no disk I/O** (that is Phase 2's ``FileSink``). The serializer:

* flattens spans into ordered trace events with a dense, collision-free ``sequence_index``
  derived from ``(wall-clock ts, monotonic op-order)`` so concurrent spans stay deterministic
  (D-3);
* expands each ``tool`` span into a ``tool_call`` (at its start) and a ``tool_result`` (at its
  close);
* stashes span timing under ``event.metadata["evalshift"]`` — the only home that survives the
  CLI's ``extra="forbid"`` events — preserving concurrency information;
* records each tool's argument hash on its ``tool_result`` so a ``(call_id, input_hash) ->
  result`` replay fixture table can be derived without re-reading the trace (D-1);
* wraps the CLI-valid ``AgentTrace`` in a :class:`CaptureEnvelope` carrying ``schema_version``
  outside the trace (D-5b).

Stdlib only (D-deps).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from evalshift.capture.span import Span, SpanTree
from evalshift.redaction import Redactor, redact_tree
from evalshift.trace.models import (
    AgentTrace,
    CaptureEnvelope,
    ErrorEvent,
    FinalOutputEvent,
    GuardrailEvent,
    ModelCallEvent,
    RetrievalEvent,
    ToolCallEvent,
    ToolResultEvent,
    TraceEvent,
    to_jsonable,
)
from evalshift.trace.schema import SCHEMA_VERSION

#: Namespace key under ``event.metadata`` for SDK-private concurrency/provenance data.
EVALSHIFT_META_KEY = "evalshift"


def canonical_hash(value: Any) -> str:
    """Stable SHA-256 of a JSON-able value. Dict-key-order-insensitive; tolerant of odd types."""
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def capture_filename(capture_id: str) -> str:
    """The on-disk file name for a capture (the FileSink writes this in Phase 2)."""
    return f"{capture_id}.json"


# --- internal: span -> event points ----------------------------------------------------------


class _Point:
    """One event with its sort key ``(timestamp, op-order)``, awaiting a ``sequence_index``."""

    __slots__ = ("make", "order", "ts")

    def __init__(self, ts: float, order: int, make: Any) -> None:
        self.ts = ts
        self.order = order
        self.make = make  # (sequence_index) -> TraceEvent


def _meta_block(span: Span, *, end_ts: float | None) -> dict[str, Any]:
    block: dict[str, Any] = {
        "span_id": span.span_id,
        "start_ts": span.start_ts,
        "end_ts": end_ts,
    }
    if span.parent_call_id is not None:
        block["parent_call_id"] = span.parent_call_id
    return block


def _with_meta(span: Span, block: dict[str, Any]) -> dict[str, Any]:
    meta = dict(span.metadata)
    meta[EVALSHIFT_META_KEY] = block
    return meta


def _ts(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _points_for_span(span: Span) -> list[_Point]:
    data = span.data
    end_ts = span.end_ts
    end_order = span.end_order if span.end_order is not None else span.start_order

    if span.kind == "tool":
        call_meta = _with_meta(span, _meta_block(span, end_ts=end_ts))
        arguments = dict(data.get("arguments", {}))
        result_block = _meta_block(span, end_ts=end_ts)
        result_block["input_hash"] = canonical_hash(arguments)
        result_meta = _with_meta(span, result_block)
        result_ts = end_ts if end_ts is not None else span.start_ts

        def make_call(seq: int) -> TraceEvent:
            return ToolCallEvent(
                name=str(data["name"]),
                sequence_index=seq,
                timestamp=_ts(span.start_ts),
                arguments=arguments,
                call_id=span.span_id,
                parent_call_id=span.parent_call_id,
                metadata=call_meta,
            )

        def make_result(seq: int) -> TraceEvent:
            return ToolResultEvent(
                name=str(data["name"]),
                sequence_index=seq,
                timestamp=_ts(result_ts),
                call_id=span.span_id,
                result=data.get("result"),
                error=data.get("error"),
                metadata=result_meta,
            )

        return [
            _Point(span.start_ts, span.start_order, make_call),
            _Point(result_ts, end_order, make_result),
        ]

    meta = _with_meta(span, _meta_block(span, end_ts=end_ts))

    def make_single(seq: int) -> TraceEvent:
        return _build_single_event(span, seq, meta)

    return [_Point(span.start_ts, span.start_order, make_single)]


def _build_single_event(span: Span, seq: int, meta: dict[str, Any]) -> TraceEvent:
    data = span.data
    ts = _ts(span.start_ts)
    if span.kind == "model_call":
        if span.end_ts is not None and "latency_ms" not in data:
            latency_ms = round((span.end_ts - span.start_ts) * 1000)
        else:
            latency_ms = int(data.get("latency_ms", 0))
        return ModelCallEvent(
            model_id=str(data["model_id"]),
            sequence_index=seq,
            timestamp=ts,
            input=data.get("input"),
            output=data.get("output"),
            input_tokens=int(data.get("input_tokens", 0)),
            output_tokens=int(data.get("output_tokens", 0)),
            cost_usd=float(data.get("cost_usd", 0.0)),
            latency_ms=latency_ms,
            # Stamped into span.data by capture/api.py's recorders (D-toolset), not derived here:
            # the serializer stays a pure pass-through, same as every other model_call field.
            # Absent from data (a hand-built span, or one predating per-call toolset capture)
            # degrades to None via .get(), same as ModelCallEvent's own field default.
            toolset_ref=data.get("toolset_ref"),
            tools_offered=data.get("tools_offered"),
            # Same pass-through contract (schema 2.1.0): capture/api.py normalises and stamps the
            # list into span.data; absent means absent on the event, never an invented [].
            requested_tool_calls=data.get("requested_tool_calls"),
            metadata=meta,
        )
    if span.kind == "retrieval":
        return RetrievalEvent(
            source=str(data["source"]),
            sequence_index=seq,
            timestamp=ts,
            query=str(data.get("query", "")),
            documents=list(data.get("documents", [])),
            metadata=meta,
        )
    if span.kind == "guardrail":
        return GuardrailEvent(
            name=str(data["name"]),
            verdict=str(data["verdict"]),
            sequence_index=seq,
            timestamp=ts,
            reason=data.get("reason"),
            metadata=meta,
        )
    if span.kind == "final_output":
        return FinalOutputEvent(
            sequence_index=seq,
            timestamp=ts,
            text=str(data.get("text", "")),
            metadata=meta,
        )
    if span.kind == "error":
        return ErrorEvent(
            message=str(data["message"]),
            sequence_index=seq,
            timestamp=ts,
            category=data.get("category"),
            metadata=meta,
        )
    raise ValueError(f"unknown span kind: {span.kind}")


def _serialize_events(tree: SpanTree) -> list[TraceEvent]:
    points: list[_Point] = []
    for span in tree:
        points.extend(_points_for_span(span))
    points.sort(key=lambda point: (point.ts, point.order))
    return [point.make(seq) for seq, point in enumerate(points)]


# --- public API -------------------------------------------------------------------------------


def _agent_input_hash(
    agent_input: Any, *, conversation_id: str | None, turn_index: int | None
) -> str:
    """The envelope ``input_hash``.

    Bare ``canonical_hash(agent_input)`` when ``conversation_id`` is unset — byte-identical to the
    pre-1.1.0 behavior, so standalone-capture dedup keys don't shift. When ``conversation_id`` is
    set, the turn identity is folded into the hash so that distinct turns sharing the same short
    user text (e.g. "yes", "1pm") don't collide under the ``(suite, input_hash)`` dedup key.
    """
    if conversation_id is None:
        return canonical_hash(agent_input)
    return canonical_hash(
        {
            "agent_input": agent_input,
            "conversation_id": conversation_id,
            "turn_index": turn_index,
        }
    )


def build_capture(
    tree: SpanTree,
    *,
    suite: str,
    agent_input: Any,
    capture_id: str | None = None,
    code_version: str = "",
    created_at: datetime | str | None = None,
    run_id: str | None = None,
    prompt_id: str | None = None,
    example_id: str | None = None,
    role: str = "source",
    redact: Redactor | None = None,
    conversation_id: str | None = None,
    turn_index: int | None = None,
    parent_capture_id: str | None = None,
) -> CaptureEnvelope:
    """Build a :class:`CaptureEnvelope` from a recorded span tree.

    Capture-time trace identity derives from the capture (Phase 9 promotion rewrites it):
    ``run_id``/``example_id`` default to ``capture_id``, ``prompt_id`` to ``suite``,
    ``role`` to ``"source"`` — all non-empty so the inner trace stays CLI-valid.

    When ``redact`` is given it masks span payloads in place **before** serialization (D-4), so
    both the trace events and the derived tool ``input_hash`` see only redacted values. A redactor
    that raises is left to propagate — the caller's guard drops the capture (fail-closed).

    ``conversation_id`` / ``turn_index`` / ``parent_capture_id`` (schema 1.1.0) stamp multi-turn
    conversation identity onto the envelope; all default to ``None`` for a standalone capture. When
    ``conversation_id`` is given, it (and ``turn_index``) are folded into ``input_hash`` so distinct
    turns with identical ``agent_input`` text hash differently — see :func:`_agent_input_hash`.
    """
    cap_id = capture_id if capture_id is not None else f"cap_{uuid.uuid4().hex}"
    if redact is not None:
        redact_tree(tree, redact)
    trace = AgentTrace(
        run_id=run_id if run_id is not None else cap_id,
        prompt_id=prompt_id if prompt_id is not None else suite,
        example_id=example_id if example_id is not None else cap_id,
        role=role,
        events=_serialize_events(tree),
    )
    return CaptureEnvelope(
        schema_version=SCHEMA_VERSION,
        capture_id=cap_id,
        suite=suite,
        input_hash=_agent_input_hash(
            agent_input, conversation_id=conversation_id, turn_index=turn_index
        ),
        code_version=code_version,
        created_at=created_at if created_at is not None else datetime.now(tz=timezone.utc),
        trace=trace,
        conversation_id=conversation_id,
        turn_index=turn_index,
        parent_capture_id=parent_capture_id,
    )


def envelope_to_dict(envelope: CaptureEnvelope) -> dict[str, Any]:
    """JSON-ready dict for a capture envelope (datetimes ISO-encoded, dataclasses expanded)."""
    result: dict[str, Any] = to_jsonable(envelope)
    return result


def dumps(envelope: CaptureEnvelope) -> str:
    """Serialize a capture envelope to a JSON string."""
    return json.dumps(envelope_to_dict(envelope), ensure_ascii=False)


def build_fixture_table(trace: AgentTrace) -> dict[tuple[str, str], Any]:
    """Derive the replay fixture lookup ``(call_id, input_hash) -> result`` from a trace (D-1).

    Reads the per-tool argument hash stamped on each ``tool_result``'s
    ``metadata["evalshift"]["input_hash"]`` — self-contained, no recomputation needed.
    """
    table: dict[tuple[str, str], Any] = {}
    for event in trace.events:
        if not isinstance(event, ToolResultEvent) or event.call_id is None:
            continue
        block = event.metadata.get(EVALSHIFT_META_KEY, {})
        input_hash = block.get("input_hash")
        if input_hash is None:
            continue
        table[(event.call_id, str(input_hash))] = event.result
    return table


__all__ = [
    "EVALSHIFT_META_KEY",
    "build_capture",
    "build_fixture_table",
    "canonical_hash",
    "capture_filename",
    "dumps",
    "envelope_to_dict",
]
