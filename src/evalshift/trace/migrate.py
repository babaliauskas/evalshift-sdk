"""Schema versioning + migration: the SDK's first read/deserialize path.

The capture envelope (``cap_<id>.json``) carries ``schema_version`` (D-5b); the inner
``AgentTrace`` is the frozen CLI contract. This module reads a written capture and **upgrades it
on read** to the current schema version so v1 captures stay promotable/replayable under future
versions. It also reconstructs typed dataclasses (the inverse of
:func:`~evalshift.trace.models.to_jsonable`).

Two layers, used in order:

* **dict migration** — :func:`load_capture` / :func:`upgrade_envelope_dict` parse and walk a
  registered chain of single-step upgrades to the current version. Migrations operate on plain
  dicts (tolerant of unknown fields).
* **typed reconstruction** — :func:`load_envelope` / :func:`envelope_from_dict` rebuild a
  :class:`~evalshift.trace.models.CaptureEnvelope` from an already-current dict.

Unlike the capture hot path (``safety.py`` fail-open), this is **read-side tooling** (the CLI
consumes it in Phase 9): it **raises** typed :class:`MigrationError` subclasses on unreadable or
unsupported input rather than swallowing. Forward-compat policy: a newer minor/patch (same major)
is read best-effort with a warning; a newer major is refused. Stdlib only (D-deps).
"""

from __future__ import annotations

import copy
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

from evalshift.safety import logger
from evalshift.trace.models import (
    AgentTrace,
    CaptureEnvelope,
    ErrorEvent,
    FinalOutputEvent,
    GuardrailEvent,
    ModelCallEvent,
    RetrievalEvent,
    ToolCallEvent,
    ToolResultEvent,
    TraceEvent,
)
from evalshift.trace.schema import (
    AGENT_TRACE_FIELDS,
    ENVELOPE_KEYS,
    EVENT_FIELDS,
    SCHEMA_VERSION,
)

# --- errors (read-side tooling raises these; it does NOT fail open) ---------------------------


class MigrationError(Exception):
    """Base for every read/migrate error. Consumers may ``except MigrationError`` broadly."""


class UnreadableCaptureError(MigrationError):
    """Input is not parseable JSON, not a JSON object, or has an unparseable timestamp."""


class MissingSchemaVersionError(MigrationError):
    """Envelope dict has no ``schema_version`` key and no default was supplied."""


class InvalidSchemaVersionError(MigrationError):
    """``schema_version`` is present but not a ``MAJOR.MINOR.PATCH`` string."""


class UnsupportedSchemaVersionError(MigrationError):
    """Version is a newer MAJOR than this SDK supports (forward-incompatible)."""


class NoMigrationPathError(MigrationError):
    """No registered chain reaches the target version from the input version."""


class UnknownEventTypeError(MigrationError):
    """Typed reconstruction hit an event whose ``type`` is not a known discriminator."""


# --- version value object ---------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class SchemaVersion:
    """A parsed semantic version (``MAJOR.MINOR.PATCH``). Ordered so chains can sort/compare."""

    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, raw: str) -> SchemaVersion:
        """Parse ``'X.Y.Z'``. Raises :class:`InvalidSchemaVersionError` on malformed input."""
        if not isinstance(raw, str):
            raise InvalidSchemaVersionError(
                f"schema_version must be a string, got {type(raw).__name__}"
            )
        parts = raw.split(".")
        if len(parts) != 3 or not all(part.isdigit() for part in parts):
            raise InvalidSchemaVersionError(f"expected MAJOR.MINOR.PATCH, got {raw!r}")
        major, minor, patch = (int(part) for part in parts)
        return cls(major, minor, patch)

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


#: The version this SDK writes and migrates toward (parsed ``schema.SCHEMA_VERSION``).
CURRENT_SCHEMA_VERSION: Final = SchemaVersion.parse(SCHEMA_VERSION)


# --- migration registry / linear chain --------------------------------------------------------

#: A single upgrade step: takes a parsed envelope dict and returns a new one at the next version.
#: Pure, no I/O; must not mutate its argument.
MigrationFn = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class Migration:
    """One registered single-step upgrade from ``from_version`` to ``to_version``."""

    from_version: SchemaVersion
    to_version: SchemaVersion
    apply: MigrationFn
    description: str = ""


_LOCK = threading.Lock()
#: One outgoing edge per version keeps the chain a simple linear path (not a graph). Empty at
#: real import time — there is genuinely only one schema version today.
_REGISTRY: dict[SchemaVersion, Migration] = {}


def register_migration(
    from_version: str,
    to_version: str,
    apply: MigrationFn,
    *,
    description: str = "",
) -> None:
    """Register a single-step upgrade. Future schema bumps add one call here at import time.

    ``to_version`` must move forward, and only one step may leave a given ``from_version``.
    Tests register synthetic migrations and call :func:`reset_migrations` afterwards. Raises
    ``ValueError`` on a backward/same step or a duplicate outgoing edge.
    """
    src = SchemaVersion.parse(from_version)
    dst = SchemaVersion.parse(to_version)
    if dst <= src:
        raise ValueError(f"migration must move forward: {from_version} -> {to_version}")
    with _LOCK:
        if src in _REGISTRY:
            raise ValueError(f"a migration from {from_version} is already registered")
        _REGISTRY[src] = Migration(
            from_version=src, to_version=dst, apply=apply, description=description
        )


def reset_migrations() -> None:
    """Clear the registry back to its built-in (empty) state. Used for test isolation."""
    with _LOCK:
        _REGISTRY.clear()


def registered_migrations() -> tuple[Migration, ...]:
    """Snapshot of registered steps, ordered by ``from_version`` (introspection / tests)."""
    with _LOCK:
        return tuple(sorted(_REGISTRY.values(), key=lambda mig: mig.from_version))


def _build_chain(start: SchemaVersion, target: SchemaVersion) -> list[Migration]:
    """Walk single-step edges from ``start`` up to ``target``. Raises on a gap, overshoot, cycle."""
    with _LOCK:
        registry = dict(_REGISTRY)
    chain: list[Migration] = []
    current = start
    seen: set[SchemaVersion] = set()
    while current < target:
        if current in seen:
            raise NoMigrationPathError(f"migration cycle detected at {current}")
        seen.add(current)
        step = registry.get(current)
        if step is None:
            raise NoMigrationPathError(f"no migration registered from {current} toward {target}")
        if step.to_version > target:
            raise NoMigrationPathError(
                f"migration from {current} overshoots {target} (reaches {step.to_version})"
            )
        chain.append(step)
        current = step.to_version
    return chain


# --- upgrade-on-read (headline API) -----------------------------------------------------------


def detect_version(envelope: dict[str, Any], *, default: str | None = None) -> SchemaVersion:
    """Read and parse ``envelope['schema_version']``.

    Raises :class:`UnreadableCaptureError` if ``envelope`` is not a dict,
    :class:`MissingSchemaVersionError` if the key is absent and ``default`` is None, and
    :class:`InvalidSchemaVersionError` on an unparseable value. ``default`` lets a caller opt
    into a version for captures lacking the key (defensive — there are no real pre-version files).
    """
    if not isinstance(envelope, dict):
        raise UnreadableCaptureError(
            f"capture must be a JSON object, got {type(envelope).__name__}"
        )
    raw = envelope.get("schema_version")
    if raw is None:
        if default is None:
            raise MissingSchemaVersionError("capture envelope has no 'schema_version'")
        raw = default
    return SchemaVersion.parse(raw)


def upgrade_envelope_dict(
    envelope: dict[str, Any],
    *,
    target: str | None = None,
    default_version: str | None = None,
) -> dict[str, Any]:
    """Upgrade a parsed capture-envelope dict to the current (or ``target``) schema version.

    Detects the source version, applies the forward-compat policy (newer minor/patch → warn and
    read best-effort; newer major → :class:`UnsupportedSchemaVersionError`), then runs the
    registered migration chain step by step. The input is never mutated (a deep copy is migrated).
    Same-version input is returned with content unchanged. Raises :class:`MigrationError`
    subclasses; does not fail open.
    """
    target_v = SchemaVersion.parse(target) if target is not None else CURRENT_SCHEMA_VERSION
    source_v = detect_version(envelope, default=default_version)
    result = copy.deepcopy(envelope)

    if source_v == target_v:
        result["schema_version"] = str(target_v)
        return result

    if source_v > target_v:
        if source_v.major > target_v.major:
            raise UnsupportedSchemaVersionError(
                f"capture schema {source_v} is a newer major than supported {target_v}; "
                "refusing to read"
            )
        logger.warning(
            "evalshift: capture schema %s is newer than supported %s; reading best-effort",
            source_v,
            target_v,
        )
        return result  # genuinely newer minor/patch — keep its own version string

    for step in _build_chain(source_v, target_v):
        result = step.apply(result)
    result["schema_version"] = str(target_v)
    return result


def load_capture(
    raw: str | bytes,
    *,
    target: str | None = None,
    default_version: str | None = None,
) -> dict[str, Any]:
    """Parse a capture file's JSON text and upgrade it to current. The read sibling of ``dumps``.

    Raises :class:`UnreadableCaptureError` on bad UTF-8/JSON or a non-object top level.
    """
    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise UnreadableCaptureError(f"capture is not valid UTF-8: {exc}") from exc
    else:
        text = raw
    try:
        payload: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise UnreadableCaptureError(f"capture is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise UnreadableCaptureError(f"capture must be a JSON object, got {type(payload).__name__}")
    return upgrade_envelope_dict(payload, target=target, default_version=default_version)


# --- typed reconstruction (inverse of to_jsonable) --------------------------------------------

#: Discriminated-union reconstruction map: event ``type`` -> dataclass. The single source of which
#: types exist is ``schema.EVENT_TYPES``; a drift guard asserts these keys match it.
_EVENT_CLASS_BY_TYPE: Final[dict[str, type[TraceEvent]]] = {
    "model_call": ModelCallEvent,
    "tool_call": ToolCallEvent,
    "tool_result": ToolResultEvent,
    "retrieval": RetrievalEvent,
    "guardrail": GuardrailEvent,
    "final_output": FinalOutputEvent,
    "error": ErrorEvent,
}


def _parse_dt(value: Any) -> datetime:
    """Parse an ISO-8601 string back to datetime (or pass a datetime through).

    Normalizes a trailing ``Z`` to ``+00:00`` first — ``datetime.fromisoformat`` on py3.10 (the
    floor, D-py) cannot parse ``Z``. Raises :class:`UnreadableCaptureError` on a bad value.
    """
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        raise UnreadableCaptureError(
            f"timestamp must be an ISO-8601 string, got {type(value).__name__}"
        )
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise UnreadableCaptureError(f"invalid timestamp {value!r}: {exc}") from exc


def event_from_dict(data: dict[str, Any]) -> TraceEvent:
    """Reconstruct one :data:`TraceEvent` dataclass, dispatching on ``data['type']``.

    Parses ``timestamp`` back to datetime; passes only declared fields (extra keys from a
    newer-minor capture are dropped, tolerantly). Raises :class:`UnknownEventTypeError` if
    ``type`` is missing or unknown.
    """
    if not isinstance(data, dict):
        raise UnknownEventTypeError(f"event must be a JSON object, got {type(data).__name__}")
    event_type = data.get("type")
    if not isinstance(event_type, str) or event_type not in _EVENT_CLASS_BY_TYPE:
        raise UnknownEventTypeError(f"unknown event type: {event_type!r}")
    cls = _EVENT_CLASS_BY_TYPE[event_type]
    allowed = set(EVENT_FIELDS[event_type])
    kwargs = {key: value for key, value in data.items() if key in allowed}
    if "timestamp" in kwargs:
        kwargs["timestamp"] = _parse_dt(kwargs["timestamp"])
    return cls(**kwargs)


def trace_from_dict(data: dict[str, Any]) -> AgentTrace:
    """Reconstruct an :class:`AgentTrace`, rebuilding each event via :func:`event_from_dict`."""
    if not isinstance(data, dict):
        raise UnreadableCaptureError(f"trace must be a JSON object, got {type(data).__name__}")
    events_raw = data.get("events", [])
    if not isinstance(events_raw, list):
        raise UnreadableCaptureError("trace 'events' must be a list")
    events = [event_from_dict(event) for event in events_raw]
    allowed = set(AGENT_TRACE_FIELDS)
    kwargs = {key: value for key, value in data.items() if key in allowed and key != "events"}
    return AgentTrace(events=events, **kwargs)


def envelope_from_dict(data: dict[str, Any]) -> CaptureEnvelope:
    """Reconstruct a typed :class:`CaptureEnvelope` from an already-current envelope dict.

    Parses ``created_at`` and rebuilds the inner trace. Tolerant of unknown envelope keys
    (dropped). Does **not** migrate — call :func:`upgrade_envelope_dict` first if the input may
    be an older version.
    """
    if not isinstance(data, dict):
        raise UnreadableCaptureError(f"envelope must be a JSON object, got {type(data).__name__}")
    trace_raw = data.get("trace")
    if not isinstance(trace_raw, dict):
        raise UnreadableCaptureError("envelope 'trace' must be a JSON object")
    trace = trace_from_dict(trace_raw)
    created_at_value: datetime | str
    raw_created = data.get("created_at")
    if isinstance(raw_created, datetime):
        created_at_value = raw_created
    elif isinstance(raw_created, str):
        created_at_value = _parse_dt(raw_created)
    else:
        raise UnreadableCaptureError(
            f"envelope 'created_at' must be a string, got {type(raw_created).__name__}"
        )
    allowed = set(ENVELOPE_KEYS)
    rest = {
        key: value
        for key, value in data.items()
        if key in allowed and key not in ("trace", "created_at")
    }
    return CaptureEnvelope(trace=trace, created_at=created_at_value, **rest)


def load_envelope(
    raw: str | bytes,
    *,
    target: str | None = None,
    default_version: str | None = None,
) -> CaptureEnvelope:
    """Full read path: parse → upgrade → reconstruct. Typed sibling of :func:`load_capture`."""
    upgraded = load_capture(raw, target=target, default_version=default_version)
    return envelope_from_dict(upgraded)


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "InvalidSchemaVersionError",
    "Migration",
    "MigrationError",
    "MigrationFn",
    "MissingSchemaVersionError",
    "NoMigrationPathError",
    "SchemaVersion",
    "UnknownEventTypeError",
    "UnreadableCaptureError",
    "UnsupportedSchemaVersionError",
    "detect_version",
    "envelope_from_dict",
    "event_from_dict",
    "load_capture",
    "load_envelope",
    "register_migration",
    "registered_migrations",
    "reset_migrations",
    "trace_from_dict",
    "upgrade_envelope_dict",
]
