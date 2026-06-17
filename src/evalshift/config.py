"""The capture gate and the programmatic configuration surface.

Capture is **off by default**: nothing is recorded or written unless ``EVALSHIFT_CAPTURE`` is set
to a truthy value. The gate is read live on every call (never cached at import) so tests and
long-lived hosts can toggle it mid-process.

:func:`configure` registers process-wide options programmatically. Only ``sink`` is wired this
phase (it swaps where captures go); ``redact`` is consumed in Phase 4, ``sample_rate`` / ``dedup``
in Phase 6 — they are accepted and stored now as forward placeholders. :func:`configure` has
**merge** semantics: only the keyword arguments you pass are changed. :func:`reset_config` clears
everything back to defaults.

:func:`active_sink` is the swap seam the capture layer routes every write through: it returns the
``configure``-registered sink, falling back to a default :class:`~evalshift.sinks.file.FileSink`.

Stdlib only (D-deps).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from evalshift import safety
from evalshift.hygiene import dedup
from evalshift.hygiene.sample import should_capture
from evalshift.redaction import Redactor
from evalshift.sinks.base import Sink
from evalshift.sinks.file import FileSink
from evalshift.sinks.hygiene import HygieneSink

#: Env var that gates capture on/off.
CAPTURE_ENV = "EVALSHIFT_CAPTURE"

#: Values (case-insensitive, stripped) that count as "capture on". Everything else is off.
_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: Sentinel for "argument not provided" so ``configure(...)`` can merge without clobbering.
_UNSET: Any = object()


@dataclass
class _Config:
    """Process-wide capture configuration, mutated by :func:`configure`."""

    sink: Sink | None = None
    redact: Redactor | None = None  # applied in Phase 4
    sample_rate: float | None = None  # capture this fraction of runs (Phase 6, entry gate)
    dedup: bool = False  # collapse identical-input captures (Phase 6)
    max_captures: int | None = None  # cap a suite dir by count, evict oldest (Phase 6 GC)
    capture_ttl: float | None = None  # evict captures older than this many seconds (Phase 6 GC)


#: The live configuration. Reset between tests via :func:`reset_config`.
_CONFIG: _Config = _Config()


def is_capture_enabled() -> bool:
    """True iff ``EVALSHIFT_CAPTURE`` is set to a truthy value (the off-by-default gate)."""
    return os.environ.get(CAPTURE_ENV, "").strip().lower() in _TRUTHY


def configure(
    *,
    sink: Sink | None = _UNSET,
    redact: Redactor | None = _UNSET,
    sample_rate: float | None = _UNSET,
    dedup: bool = _UNSET,
    max_captures: int | None = _UNSET,
    capture_ttl: float | None = _UNSET,
) -> None:
    """Set process-wide capture options. Only the arguments you pass are changed (merge)."""
    if sink is not _UNSET:
        _CONFIG.sink = sink
    if redact is not _UNSET:
        _CONFIG.redact = redact
    if sample_rate is not _UNSET:
        _CONFIG.sample_rate = sample_rate
    if dedup is not _UNSET:
        _CONFIG.dedup = dedup
    if max_captures is not _UNSET:
        _CONFIG.max_captures = max_captures
    if capture_ttl is not _UNSET:
        _CONFIG.capture_ttl = capture_ttl


def reset_config() -> None:
    """Clear all configured options back to defaults (used for test isolation)."""
    _CONFIG.sink = None
    _CONFIG.redact = None
    _CONFIG.sample_rate = None
    _CONFIG.dedup = False
    _CONFIG.max_captures = None
    _CONFIG.capture_ttl = None
    dedup.reset_registry()


def active_sink() -> Sink:
    """Return the sink captures are written to, wrapped with hygiene when any knob is set.

    With no hygiene option configured (``dedup``/``max_captures``/``capture_ttl``) the configured
    or default sink is returned **unchanged** (identity preserved). Otherwise it is wrapped in a
    :class:`~evalshift.sinks.hygiene.HygieneSink` that applies dedup + GC around every write.
    """
    base = _CONFIG.sink if _CONFIG.sink is not None else FileSink()
    if not _CONFIG.dedup and _CONFIG.max_captures is None and _CONFIG.capture_ttl is None:
        return base
    return HygieneSink(
        base,
        dedup=_CONFIG.dedup,
        max_captures=_CONFIG.max_captures,
        capture_ttl=_CONFIG.capture_ttl,
    )


def should_capture_now() -> bool:
    """Sampling decision for one agent run (read at agent entry). Fail-open: capture on fault.

    A configured ``sample_rate`` decides the draw; a fault in the draw defaults to **capturing**
    so a sampling bug never silently disables all telemetry.
    """
    decision = safety.guard("sample decision", lambda: should_capture(_CONFIG.sample_rate))
    return decision is not False


def active_redactor() -> Redactor | None:
    """Return the process-wide redactor registered via ``configure(redact=...)``, else ``None``."""
    return _CONFIG.redact


__all__ = [
    "CAPTURE_ENV",
    "active_redactor",
    "active_sink",
    "configure",
    "is_capture_enabled",
    "reset_config",
    "should_capture_now",
]
