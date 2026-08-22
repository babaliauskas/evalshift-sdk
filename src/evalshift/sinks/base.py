"""The ``Sink`` protocol: where a finished capture goes.

A sink receives a fully built :class:`~evalshift.trace.models.CaptureEnvelope` and disposes of
it (writes to disk, buffers in memory, ships elsewhere). ``write`` returns the on-disk
:class:`~pathlib.Path` when there is one, or ``None`` when there is not (memory) or when a write
degraded to a no-op. The capture layer always routes through :func:`evalshift.config.active_sink`,
so any object satisfying this protocol can be swapped in via :func:`evalshift.configure`.

``runtime_checkable`` so tests (and defensive callers) can ``isinstance(x, Sink)``.

Stdlib only (D-deps).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from evalshift.trace.models import CaptureEnvelope


@runtime_checkable
class Sink(Protocol):
    def write(self, envelope: CaptureEnvelope) -> Path | None: ...


__all__ = ["Sink"]
