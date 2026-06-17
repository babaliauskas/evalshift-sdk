"""Tests for the typed reverse path: dict -> dataclasses (inverse of ``to_jsonable``).

No pydantic here. The CLI-validity of a reloaded capture is proven in
``tests/conformance/test_migrate_parity.py``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from evalshift.capture.span import SpanTree
from evalshift.trace import migrate, schema
from evalshift.trace.migrate import (
    UnknownEventTypeError,
    envelope_from_dict,
    event_from_dict,
    load_envelope,
)
from evalshift.trace.models import FinalOutputEvent, to_jsonable
from evalshift.trace.serialize import build_capture, build_fixture_table, dumps, envelope_to_dict

TS = "2026-06-16T12:00:00+00:00"


def _full_tree() -> SpanTree:
    tree = SpanTree()
    m = tree.open_span(
        "model_call",
        span_id="m1",
        start_ts=1.0,
        data={"model_id": "claude-opus-4-8", "input": "hi", "output": "yo", "input_tokens": 3},
    )
    tree.close_span(m, end_ts=1.5)
    t = tree.open_span(
        "tool", span_id="c1", start_ts=2.0, data={"name": "search", "arguments": {"q": "x"}}
    )
    tree.close_span(t, end_ts=2.5, result={"hits": 1})
    tree.open_span(
        "retrieval",
        span_id="r1",
        start_ts=3.0,
        data={"source": "kb", "query": "x", "documents": [{"id": "d1"}]},
    )
    tree.open_span("guardrail", span_id="g1", start_ts=4.0, data={"name": "pii", "verdict": "pass"})
    tree.open_span(
        "error", span_id="e1", start_ts=5.0, data={"message": "boom", "category": "runtime"}
    )
    tree.open_span("final_output", span_id="f1", start_ts=6.0, data={"text": "done"})
    return tree


def _full_envelope_dict() -> dict[str, Any]:
    return envelope_to_dict(
        build_capture(_full_tree(), suite="s", agent_input="hi", capture_id="cap_t", created_at=TS)
    )


# --- event_from_dict --------------------------------------------------------------------------


def test_event_from_dict_dispatches_every_type() -> None:
    seen = set()
    for data in _full_envelope_dict()["trace"]["events"]:
        event = event_from_dict(data)
        assert event.type == data["type"]
        assert isinstance(event.timestamp, datetime)
        seen.add(event.type)
    assert seen == set(schema.EVENT_TYPES)


def test_event_from_dict_unknown_type_raises() -> None:
    with pytest.raises(UnknownEventTypeError):
        event_from_dict({"type": "telepathy", "sequence_index": 0, "timestamp": TS, "metadata": {}})


def test_event_from_dict_missing_type_raises() -> None:
    with pytest.raises(UnknownEventTypeError):
        event_from_dict({"sequence_index": 0, "timestamp": TS, "metadata": {}})


def test_event_from_dict_tolerates_unknown_field() -> None:
    data = {
        "type": "final_output",
        "sequence_index": 0,
        "timestamp": TS,
        "metadata": {},
        "text": "done",
        "future_field": 123,  # a field a newer minor might add
    }
    event = event_from_dict(data)
    assert isinstance(event, FinalOutputEvent)
    assert event.text == "done"
    assert not hasattr(event, "future_field")


# --- envelope_from_dict -----------------------------------------------------------------------


def test_envelope_from_dict_parses_created_at() -> None:
    reloaded = envelope_from_dict(_full_envelope_dict())
    assert isinstance(reloaded.created_at, datetime)


# --- round trips (the "stay promotable/replayable" proof, minus pydantic) ---------------------


def test_round_trip_build_dumps_load_envelope_is_identity() -> None:
    original = build_capture(
        _full_tree(), suite="s", agent_input="hi", capture_id="cap_t", created_at=TS
    )
    original_dict = envelope_to_dict(original)
    reloaded = load_envelope(dumps(original))
    assert to_jsonable(reloaded) == original_dict


def test_reloaded_trace_yields_same_fixture_table() -> None:
    original = build_capture(_full_tree(), suite="s", agent_input="hi", capture_id="cap_t")
    expected = build_fixture_table(original.trace)
    reloaded = load_envelope(dumps(original))
    assert build_fixture_table(reloaded.trace) == expected
    assert expected  # the tool span actually produced a fixture entry


# --- _parse_dt --------------------------------------------------------------------------------


def test_parse_dt_accepts_z_suffix() -> None:
    parsed = migrate._parse_dt("2026-06-16T12:00:00Z")
    assert parsed.utcoffset() == timedelta(0)


def test_parse_dt_passes_datetime_through() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert migrate._parse_dt(now) is now
