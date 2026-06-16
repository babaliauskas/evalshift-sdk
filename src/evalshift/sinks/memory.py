"""In-memory sink for ephemeral hosts (Lambda, tests) that flush captures themselves.

Captures accumulate in a list; the host calls :meth:`MemorySink.flush` to drain and reset the
buffer, or reads :attr:`MemorySink.captures` to inspect it without draining. Nothing touches
disk, so this is the safe choice on read-only filesystems.

Not yet concurrency-safe — async/threaded recording lands in Phase 5, which adds locking.

Stdlib only (D-deps).
"""

from __future__ import annotations

from evalshift.trace.models import CaptureEnvelope


class MemorySink:
    """Buffer capture envelopes in process memory; the host drains them with :meth:`flush`."""

    def __init__(self) -> None:
        self._captures: list[CaptureEnvelope] = []

    def write(self, envelope: CaptureEnvelope) -> None:
        self._captures.append(envelope)

    def flush(self) -> list[CaptureEnvelope]:
        """Return the buffered captures (in write order) and clear the buffer."""
        drained = self._captures
        self._captures = []
        return drained

    @property
    def captures(self) -> tuple[CaptureEnvelope, ...]:
        """A non-draining snapshot of the buffered captures."""
        return tuple(self._captures)


__all__ = ["MemorySink"]
