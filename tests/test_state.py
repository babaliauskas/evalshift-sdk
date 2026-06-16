"""Unit tests for the async-safe capture-state contextvars."""

from __future__ import annotations

import threading

from evalshift.capture.span import SpanTree
from evalshift.capture.state import (
    current_parent,
    current_tree,
    use_parent,
    use_tree,
)

# --- defaults ---


def test_current_tree_defaults_to_none() -> None:
    assert current_tree() is None


def test_current_parent_defaults_to_none() -> None:
    assert current_parent() is None


# --- use_tree ---


def test_use_tree_binds_and_restores() -> None:
    tree = SpanTree()
    assert current_tree() is None
    with use_tree(tree):
        assert current_tree() is tree
    assert current_tree() is None


def test_use_tree_restores_previous_on_nesting() -> None:
    outer, inner = SpanTree(), SpanTree()
    with use_tree(outer):
        assert current_tree() is outer
        with use_tree(inner):
            assert current_tree() is inner
        assert current_tree() is outer
    assert current_tree() is None


# --- use_parent ---


def test_use_parent_binds_and_restores() -> None:
    assert current_parent() is None
    with use_parent("call_a"):
        assert current_parent() == "call_a"
    assert current_parent() is None


def test_use_parent_restores_previous_on_nesting() -> None:
    with use_parent("outer"):
        assert current_parent() == "outer"
        with use_parent("inner"):
            assert current_parent() == "inner"
        assert current_parent() == "outer"
    assert current_parent() is None


# --- isolation (guards the contextvar choice over globals/thread-locals) ---


def test_state_is_isolated_across_threads() -> None:
    seen: list[SpanTree | None] = []

    def worker() -> None:
        seen.append(current_tree())

    with use_tree(SpanTree()):
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

    # A fresh thread starts a fresh context => it must not see the main thread's binding.
    assert seen == [None]
