"""Unit tests for the live recording structure: Span / SpanTree."""

from __future__ import annotations

from evalshift.capture.span import Span, SpanTree, is_persistable


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


# --- is_persistable: the capture persistence gate ---------------------------
# A capture is worth persisting iff its tree holds >=1 model_call span. Everything
# else (no-op, error-only, tool-only) carries no scoreable ground truth and cannot
# be promoted by the CLI reader, so it must not be written.


def test_is_persistable_false_for_empty_tree() -> None:
    assert is_persistable(SpanTree()) is False


def test_is_persistable_false_for_error_only_tree() -> None:
    tree = SpanTree()
    tree.open_span("error", span_id="e", start_ts=1.0, data={"message": "CancelledError"})
    assert is_persistable(tree) is False


def test_is_persistable_false_for_tool_only_tree() -> None:
    tree = SpanTree()
    tree.open_span("tool", span_id="t", start_ts=1.0, data={"name": "search"})
    assert is_persistable(tree) is False


def test_is_persistable_true_for_model_call_tree() -> None:
    tree = SpanTree()
    tree.open_span("model_call", span_id="m", start_ts=1.0)
    assert is_persistable(tree) is True


def test_is_persistable_true_for_model_call_with_tool() -> None:
    tree = SpanTree()
    tree.open_span("model_call", span_id="m", start_ts=1.0)
    tree.open_span("tool", span_id="t", start_ts=2.0, data={"name": "search"})
    assert is_persistable(tree) is True


def test_is_persistable_true_for_model_call_then_error() -> None:
    # Failed-after-a-call: a model call happened, then the run errored. Highest-value
    # telemetry -- kept.
    tree = SpanTree()
    tree.open_span("model_call", span_id="m", start_ts=1.0)
    tree.open_span("error", span_id="e", start_ts=2.0, data={"message": "boom"})
    assert is_persistable(tree) is True
