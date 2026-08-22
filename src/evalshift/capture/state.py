"""Async-safe current-capture state via :mod:`contextvars`.

Three context variables track the in-flight capture:

* :data:`_current_tree` — the active :class:`~evalshift.capture.span.SpanTree` (the capture
  session), or ``None`` when no ``@capture.agent`` frame is on the stack;
* :data:`_current_parent` — the ``call_id`` of the enclosing tool span, used as
  ``parent_call_id`` for spans opened beneath it (``None`` at the agent top level);
* :data:`_current_toolset` — the enclosing session's own ``tools=`` (D-toolset), already
  normalised to ``(tools, fingerprint)`` by :func:`evalshift.capture.toolset.normalize_tools` /
  :func:`~evalshift.capture.toolset.fingerprint_tools`, or ``None`` when the session has nothing
  usable to offer. Set by the three session entry points (``capture.agent`` / ``agent_session`` /
  ``agent_session_async``) and read by a ``model_call`` recorder whose own ``tools=`` is ``None``
  — a call's own value always wins when it gives one; this is purely the fallback for a call that
  doesn't (``capture/api.py``'s ``_resolve_call_toolset``). The LangChain adapter does not use
  this var — like ``_current_tree``, it keeps its own separate, non-contextvar state (see its
  module docstring) — its constructor-level ``tools=`` plays the same role directly.

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
from typing import Any

from evalshift.capture.span import SpanTree

_current_tree: ContextVar[SpanTree | None] = ContextVar("evalshift_current_tree", default=None)
_current_parent: ContextVar[str | None] = ContextVar("evalshift_current_parent", default=None)
_current_toolset: ContextVar[tuple[list[dict[str, Any]], str] | None] = ContextVar(
    "evalshift_current_toolset", default=None
)


def current_tree() -> SpanTree | None:
    """The active capture session, or ``None`` if capture isn't running in this context."""
    return _current_tree.get()


def current_parent() -> str | None:
    """The ``call_id`` to use as ``parent_call_id`` for a span opened right now."""
    return _current_parent.get()


def current_toolset() -> tuple[list[dict[str, Any]], str] | None:
    """The enclosing session's own ``(normalized_tools, fingerprint)``, or ``None`` when unset or
    the session's ``tools=`` didn't normalise (see the module docstring)."""
    return _current_toolset.get()


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


@contextmanager
def use_toolset(toolset: tuple[list[dict[str, Any]], str] | None) -> Iterator[None]:
    """Bind ``toolset`` as the active session's toolset for the duration of the block."""
    token: Token[tuple[list[dict[str, Any]], str] | None] = _current_toolset.set(toolset)
    try:
        yield
    finally:
        _current_toolset.reset(token)


__all__ = [
    "current_parent",
    "current_toolset",
    "current_tree",
    "use_parent",
    "use_toolset",
    "use_tree",
]
