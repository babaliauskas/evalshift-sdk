"""Stdlib dataclasses mirroring the CLI trace contract, plus the capture envelope.

These are the *serialized output* shapes. They mirror the CLI's pydantic models in
``evalshift-cli/src/evalshift/traces/models.py`` field-for-field (names, types, defaults); the
parity tests in ``tests/conformance`` assert the emitted JSON validates against the frozen CLI
model. Pydantic is **never** imported at runtime here (D-deps) — these are plain dataclasses.

No dataclass inheritance: each event lists its required fields first, then defaulted fields
(``type`` literal default, optional payload, ``metadata`` factory) so the dataclass machinery
never sees a non-default field after a defaulted one.

``schema_version`` lives only in :class:`CaptureEnvelope`, never inside :class:`AgentTrace`
(D-5b) — embedding it would fail the CLI's ``extra="forbid"`` validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime
from typing import Any


@dataclass
class ModelCallEvent:
    """A model invocation inside an agent timeline."""

    model_id: str
    sequence_index: int
    timestamp: datetime
    input: Any = None
    output: Any = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    type: str = "model_call"


@dataclass
class ToolCallEvent:
    """A tool invocation emitted by an agent."""

    name: str
    sequence_index: int
    timestamp: datetime
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str | None = None
    parent_call_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    type: str = "tool_call"


@dataclass
class ToolResultEvent:
    """The result returned for a previous tool call."""

    name: str
    sequence_index: int
    timestamp: datetime
    call_id: str | None = None
    result: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    type: str = "tool_result"


@dataclass
class RetrievalEvent:
    """A retrieval step in an agent timeline."""

    source: str
    sequence_index: int
    timestamp: datetime
    query: str = ""
    documents: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    type: str = "retrieval"


@dataclass
class GuardrailEvent:
    """A guardrail or policy check."""

    name: str
    verdict: str
    sequence_index: int
    timestamp: datetime
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    type: str = "guardrail"


@dataclass
class FinalOutputEvent:
    """The final response visible to the user."""

    sequence_index: int
    timestamp: datetime
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    type: str = "final_output"


@dataclass
class ErrorEvent:
    """An agent/runtime error event."""

    message: str
    sequence_index: int
    timestamp: datetime
    category: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    type: str = "error"


TraceEvent = (
    ModelCallEvent
    | ToolCallEvent
    | ToolResultEvent
    | RetrievalEvent
    | GuardrailEvent
    | FinalOutputEvent
    | ErrorEvent
)


@dataclass
class AgentTrace:
    """Trace for one prompt/example/model side. Mirrors the CLI ``AgentTrace``."""

    run_id: str
    prompt_id: str
    example_id: str
    role: str
    events: list[TraceEvent] = field(default_factory=list)


@dataclass
class CaptureEnvelope:
    """The ``cap_<id>.json`` wrapper. Keys mirror ``schema.ENVELOPE_KEYS`` exactly."""

    schema_version: str
    capture_id: str
    suite: str
    input_hash: str
    code_version: str
    created_at: datetime | str
    trace: AgentTrace


def to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses / datetimes into JSON-ready primitives.

    Dataclass -> ``dict`` (in field order), ``datetime`` -> ISO 8601 string, containers recursed,
    primitives passed through. Produces the exact JSON the CLI loader consumes.
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {key: to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(item) for item in obj]
    return obj


__all__ = [
    "AgentTrace",
    "CaptureEnvelope",
    "ErrorEvent",
    "FinalOutputEvent",
    "GuardrailEvent",
    "ModelCallEvent",
    "RetrievalEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "TraceEvent",
    "to_jsonable",
]
