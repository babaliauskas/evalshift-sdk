"""Per-process input dedup: collapse captures whose agent inputs are identical.

A process-global, lock-guarded registry of ``(suite, input_hash)`` keys. ``input_hash`` is the
agent's entry-argument hash (the prompt), so dedup collapses re-runs of the *same input*
regardless of what happened inside the run. :func:`is_duplicate` (check) and :func:`mark_seen`
(record) are deliberately separate so the sink can check **before** a write and record only
**after** the write is accepted.

Scope and limits (documented in ``docs/DECISIONS.md`` D-6): the registry lives in memory for the
life of the process — there is **no** cross-process or cross-run dedup. :func:`reset_registry` is
called from :func:`evalshift.config.reset_config` so test isolation stays clean.

Stdlib only (D-deps).
"""

from __future__ import annotations

import threading

_LOCK = threading.Lock()
_SEEN: set[tuple[str, str]] = set()


def is_duplicate(suite: str, input_hash: str) -> bool:
    """True iff ``(suite, input_hash)`` was already recorded via :func:`mark_seen`."""
    with _LOCK:
        return (suite, input_hash) in _SEEN


def mark_seen(suite: str, input_hash: str) -> None:
    """Record ``(suite, input_hash)`` so future identical inputs read as duplicates."""
    with _LOCK:
        _SEEN.add((suite, input_hash))


def reset_registry() -> None:
    """Clear all recorded keys (used by ``reset_config`` for test isolation)."""
    with _LOCK:
        _SEEN.clear()


__all__ = ["is_duplicate", "mark_seen", "reset_registry"]
