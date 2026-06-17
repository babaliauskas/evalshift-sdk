"""A sink decorator that adds dedup + GC around any base sink.

:func:`evalshift.config.active_sink` wraps the configured/default sink in a ``HygieneSink`` only
when a hygiene knob is set (``dedup`` / ``max_captures`` / ``capture_ttl``); otherwise the bare
sink is returned unchanged. This keeps the single write seam in ``capture/api.py`` untouched.

Fail-open is **granular here**: the api layer's outer ``fail_open("sink write")`` would drop the
whole capture on any exception, so dedup and GC are guarded *individually inside* ``write`` — a
broken dedup check defaults to "not a duplicate, write it", and a broken GC is swallowed after the
write already succeeded. Neither can suppress the base write.

Dedup short-circuits **before** the base write (so it works for any sink, including
:class:`~evalshift.sinks.memory.MemorySink`, whose ``write`` returns ``None``); the input is marked
seen once the base write has not raised. GC only runs when the base returns a real ``Path``.

Stdlib only (D-deps).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from evalshift import safety
from evalshift.hygiene import dedup as dedup_registry
from evalshift.hygiene import gc as capture_gc
from evalshift.sinks.base import Sink
from evalshift.trace.models import CaptureEnvelope


class HygieneSink:
    """Wrap ``base`` with opt-in input dedup and directory garbage collection."""

    def __init__(
        self,
        base: Sink,
        *,
        dedup: bool,
        max_captures: int | None,
        capture_ttl: float | None,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._base = base
        self._dedup = dedup
        self._max_captures = max_captures
        self._capture_ttl = capture_ttl
        self._now = now

    def write(self, envelope: CaptureEnvelope) -> Path | None:
        if self._dedup:
            seen = safety.guard(
                "dedup check",
                lambda: dedup_registry.is_duplicate(envelope.suite, envelope.input_hash),
            )
            if seen:  # None (guard fault) or False -> not a duplicate, proceed
                return None

        path = self._base.write(envelope)  # may raise; the api layer's guard drops on fault

        if self._dedup:
            safety.guard(
                "dedup mark",
                lambda: dedup_registry.mark_seen(envelope.suite, envelope.input_hash),
            )

        if path is not None and (self._max_captures is not None or self._capture_ttl is not None):
            with safety.fail_open("gc"):
                capture_gc.evict(
                    path.parent,
                    max_captures=self._max_captures,
                    capture_ttl_seconds=self._capture_ttl,
                    now_ts=self._now(),
                )
        return path


__all__ = ["HygieneSink"]
