"""Unit tests for the async-safe capture-state contextvars."""

from __future__ import annotations

import threading

from evalshift.capture.span import SpanTree
from evalshift.capture.state import (
    current_parent,
    current_toolset,
    current_tree,
    use_parent,
    use_toolset,
    use_tree,
)

# --- defaults ---


def test_current_tree_defaults_to_none() -> None:
    assert current_tree() is None


def test_current_parent_defaults_to_none() -> None:
    assert current_parent() is None


def test_current_toolset_defaults_to_none() -> None:
    assert current_toolset() is None


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


# --- use_toolset ---


def test_use_toolset_binds_and_restores() -> None:
    pair = ([{"name": "search", "description": "", "input_schema": {}}], "sha256:abc")
    assert current_toolset() is None
    with use_toolset(pair):
        assert current_toolset() is pair
    assert current_toolset() is None


def test_use_toolset_restores_previous_on_nesting() -> None:
    outer = ([{"name": "outer_tool", "description": "", "input_schema": {}}], "sha256:outer")
    inner = ([{"name": "inner_tool", "description": "", "input_schema": {}}], "sha256:inner")
    with use_toolset(outer):
        assert current_toolset() is outer
        with use_toolset(inner):
            assert current_toolset() is inner
        assert current_toolset() is outer
    assert current_toolset() is None


def test_use_toolset_accepts_none_for_an_invalid_session_toolset() -> None:
    """A session whose own ``tools=`` didn't normalise binds ``None`` -- same as never binding --
    so a call inheriting it degrades exactly like "no session toolset set" rather than raising.
    """
    with use_toolset(None):
        assert current_toolset() is None


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
