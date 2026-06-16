"""Async-safe current-capture state via :mod:`contextvars`.

Two context variables track the in-flight capture:

* :data:`_current_tree` — the active :class:`~evalshift.capture.span.SpanTree` (the capture
  session), or ``None`` when no ``@capture.agent`` frame is on the stack;
* :data:`_current_parent` — the ``call_id`` of the enclosing tool span, used as
  ``parent_call_id`` for spans opened beneath it (``None`` at the agent top level).

Using :class:`contextvars.ContextVar` (not globals or thread-locals) makes the state correct
across ``await`` boundaries and concurrent tasks by construction — Phase 5 reuses this module
unchanged. The push/pop helpers use reset tokens so nested and re-entrant frames restore the
*previous* value rather than blindly clearing.

Stdlib only (D-deps).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token

from evalshift.capture.span import SpanTree

_current_tree: ContextVar[SpanTree | None] = ContextVar("evalshift_current_tree", default=None)
_current_parent: ContextVar[str | None] = ContextVar("evalshift_current_parent", default=None)


def current_tree() -> SpanTree | None:
    """The active capture session, or ``None`` if capture isn't running in this context."""
    return _current_tree.get()


def current_parent() -> str | None:
    """The ``call_id`` to use as ``parent_call_id`` for a span opened right now."""
    return _current_parent.get()


@contextmanager
def use_tree(tree: SpanTree) -> Iterator[None]:
    """Bind ``tree`` as the active session for the duration of the block."""
    token: Token[SpanTree | None] = _current_tree.set(tree)
    try:
        yield
    finally:
        _current_tree.reset(token)


@contextmanager
def use_parent(call_id: str) -> Iterator[None]:
    """Bind ``call_id`` as the current parent for nested spans for the duration of the block."""
    token: Token[str | None] = _current_parent.set(call_id)
    try:
        yield
    finally:
        _current_parent.reset(token)


__all__ = ["current_parent", "current_tree", "use_parent", "use_tree"]
