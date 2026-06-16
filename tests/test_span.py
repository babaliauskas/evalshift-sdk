"""Unit tests for the live recording structure: Span / SpanTree."""

from __future__ import annotations

from evalshift.capture.span import Span, SpanTree


def test_open_span_registers_and_assigns_start_order() -> None:
    tree = SpanTree()
    span = tree.open_span("tool", span_id="c1", start_ts=1.0, data={"name": "search"})

    assert isinstance(span, Span)
    assert span.kind == "tool"
    assert span.span_id == "c1"
    assert span.start_ts == 1.0
    assert span.start_order == 1
    assert span.end_ts is None
    assert span.end_order is None
    assert span.parent_call_id is None
    assert span.metadata == {}
    assert list(tree) == [span]


def test_close_span_assigns_end_order_greater_than_start() -> None:
    tree = SpanTree()
    span = tree.open_span("tool", span_id="c1", start_ts=1.0)
    tree.close_span(span, end_ts=2.0, result={"hits": 1})

    assert span.end_ts == 2.0
    assert span.end_order is not None
    assert span.end_order > span.start_order
    assert span.data["result"] == {"hits": 1}


def test_order_is_monotonic_across_spans() -> None:
    tree = SpanTree()
    a = tree.open_span("tool", span_id="a", start_ts=1.0)
    tree.close_span(a, end_ts=2.0)
    b = tree.open_span("tool", span_id="b", start_ts=3.0)
    tree.close_span(b, end_ts=4.0)

    assert [a.start_order, a.end_order, b.start_order, b.end_order] == [1, 2, 3, 4]


def test_concurrent_spans_keep_distinct_orders() -> None:
    # A opens, B opens, B closes, A closes (overlapping intervals).
    tree = SpanTree()
    a = tree.open_span("tool", span_id="a", start_ts=1.0)
    b = tree.open_span("tool", span_id="b", start_ts=1.0)
    tree.close_span(b, end_ts=2.0)
    tree.close_span(a, end_ts=3.0)

    orders = [a.start_order, b.start_order, b.end_order, a.end_order]
    assert orders == [1, 2, 3, 4]
    assert len(set(orders)) == 4


def test_parent_call_id_links_child_to_parent() -> None:
    tree = SpanTree()
    parent = tree.open_span("tool", span_id="p", start_ts=1.0)
    child = tree.open_span("tool", span_id="c", parent_call_id=parent.span_id, start_ts=1.5)

    assert child.parent_call_id == "p"
    assert parent.parent_call_id is None


def test_spans_iterate_in_creation_order() -> None:
    tree = SpanTree()
    first = tree.open_span("model_call", span_id="m", start_ts=1.0)
    second = tree.open_span("tool", span_id="t", start_ts=2.0)
    third = tree.open_span("final_output", span_id="f", start_ts=3.0)

    assert list(tree) == [first, second, third]


def test_metadata_passed_through() -> None:
    tree = SpanTree()
    span = tree.open_span("tool", span_id="c1", start_ts=1.0, metadata={"user": "x"})
    assert span.metadata == {"user": "x"}
