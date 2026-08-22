"""The fail-open boundary: capture bookkeeping must never break the host agent.

Every SDK-internal operation (opening/closing spans, serializing, writing a sink) runs inside
:func:`fail_open` or :func:`guard`. A failure there is swallowed and logged at ``debug`` — it
never reaches the caller and never spams a production agent's logs. The user's own function call
is **never** wrapped by these; its real return value and exception always propagate (Phase 6
hardens this further).

Stdlib only (D-deps).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TypeVar

#: Library logger. The host owns handler/level config; we never configure it ourselves.
logger = logging.getLogger("evalshift")

T = TypeVar("T")


@contextmanager
def fail_open(action: str) -> Iterator[None]:
    """Swallow-and-log any exception raised inside the block.

    Wrap ONLY SDK-internal, side-effecting bookkeeping with this — never the user's function.
    """
    try:
        yield
    except Exception:
        logger.debug("evalshift: %s failed (swallowed)", action, exc_info=True)


def guard(action: str, fn: Callable[[], T]) -> T | None:
    """Run a value-returning bookkeeping callable under fail-open semantics.

    Returns the callable's value, or ``None`` if it raised (logged at ``debug``).
    """
    try:
        return fn()
    except Exception:
        logger.debug("evalshift: %s failed (swallowed)", action, exc_info=True)
        return None


__all__ = ["fail_open", "guard", "logger"]
