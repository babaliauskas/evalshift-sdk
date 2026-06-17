"""Capture sampling: decide whether to record this agent run.

The decision is made **once, at agent entry** (before any span tree is built), so an unsampled
run degrades to the same near-pure pass-through as the gate-off path — no tree, no redaction, no
serialization. The draw source ``rng`` is injectable so tests are deterministic.

Stdlib only (D-deps).
"""

from __future__ import annotations

import random
from collections.abc import Callable


def should_capture(rate: float | None, *, rng: Callable[[], float] | None = None) -> bool:
    """True iff this run should be captured under ``rate`` (a probability in ``[0, 1]``).

    ``None`` or ``rate >= 1`` always captures; ``rate <= 0`` never captures; otherwise capture
    iff a uniform draw ``rng()`` falls below ``rate`` (strict ``<``, so ``rate == draw`` drops).

    ``rng`` defaults to :func:`random.random`, resolved at call time so tests can monkeypatch it.
    """
    if rate is None or rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    draw = rng if rng is not None else random.random
    return draw() < rate


__all__ = ["should_capture"]
