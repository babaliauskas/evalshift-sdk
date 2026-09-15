# FROZEN MIRROR — do not edit by hand except to re-sync with the CLI.
# Source: evalshift-cli/src/evalshift_cli/traces/models.py (AgentTrace + TraceEvent union) and
#         evalshift-cli/src/evalshift_cli/captures/models.py (CaptureEnvelope).
# License: the CLI source is Apache-2.0; this mirror is distributed here under this
#          repository's MIT license by Lukas Babaliauskas, the copyright holder of both
#          codebases. Re-syncs must only pull from CLI code with the same sole authorship.
# Purpose: parity target for the SDK's emitted JSON. The SDK must produce JSON that validates
# against these models. Pydantic is a dev/test dependency only — never imported at runtime.
# Adaptations vs source (validation-behavior-preserving only):
#   1. `Self` imported from typing_extensions so this runs on py3.10 (the CLI floor is py3.14
#      where `typing.Self` exists; the SDK floor is py3.10).
#   2. `CaptureEnvelope` here subclasses `_StrictModel` (then overrides `model_config` back to
#      `extra="ignore"`) so it shares the base with the event/`AgentTrace` models in this file;
#      the CLI source instead defines `CaptureEnvelope(BaseModel)` directly. Effective validation
#      behavior (`extra="ignore"`, same fields) is identical.
#   3. Docstrings are reworded (e.g. `CaptureEnvelope`'s forward-compat note and the
#      `conversation_id`/`turn_index`/`parent_capture_id` comment) — content matches, wording
#      does not mirror the CLI source verbatim.
"""Strict models for imported bring-your-own-agent traces."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator
from typing_extensions import Self

TraceRole = Literal["source", "target"]


class _StrictModel(BaseModel):
    """Base model for trace artifacts."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class _BaseEvent(_StrictModel):
    """Fields shared by all imported trace events."""

    type: str
    sequence_index: int = Field(ge=0)
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class RequestedToolCall(_StrictModel):
    """One tool call the model asked for in its response (not necessarily executed)."""

    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    call_id: str | None = None


class ModelCallEvent(_BaseEvent):
    """A model invocation inside an agent timeline."""

    type: Literal["model_call"]
    model_id: str = Field(min_length=1)
    input: Any = None
    output: Any = None
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    latency_ms: int = Field(default=0, ge=0)
    # Content-addressed pointer to the toolset sidecar (``sha256:<hex>``, see
    # ``evalshift_cli.captures.toolset.fingerprint_tools``) plus the cheap,
    # display-only tool-name list. Optional *here* so the parser accepts a
    # capture that predates per-call toolset capture -- the requirement that
    # they be present is enforced at promotion, not at parse time, where a
    # useful error can name the capture instead of a generic parse failure.
    toolset_ref: str | None = None
    tools_offered: list[str] | None = None
    # What the model asked for in this response, as opposed to ``tools_offered`` (what it was
    # allowed to ask for) and the ``tool_call`` events (what the app actually executed).
    # ``None`` on a capture written before schema 2.1.0; ``[]`` means the model requested none.
    requested_tool_calls: list[RequestedToolCall] | None = None


class ToolCallEvent(_BaseEvent):
    """A tool invocation emitted by an agent."""

    type: Literal["tool_call"]
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    call_id: str | None = None
    parent_call_id: str | None = None


class ToolResultEvent(_BaseEvent):
    """The result returned for a previous tool call."""

    type: Literal["tool_result"]
    name: str = Field(min_length=1)
    call_id: str | None = None
    result: Any = None
    error: str | None = None


class RetrievalEvent(_BaseEvent):
    """A retrieval step in an agent timeline."""

    type: Literal["retrieval"]
    source: str = Field(min_length=1)
    query: str = ""
    documents: list[dict[str, Any]] = Field(default_factory=list)


class GuardrailEvent(_BaseEvent):
    """A guardrail or policy check."""

    type: Literal["guardrail"]
    name: str = Field(min_length=1)
    verdict: Literal["pass", "fail", "warn", "skipped"]
    reason: str | None = None


class FinalOutputEvent(_BaseEvent):
    """The final response visible to the user."""

    type: Literal["final_output"]
    text: str = ""


class ErrorEvent(_BaseEvent):
    """An agent/runtime error event."""

    type: Literal["error"]
    message: str = Field(min_length=1)
    category: str | None = None


TraceEvent = Annotated[
    ModelCallEvent
    | ToolCallEvent
    | ToolResultEvent
    | RetrievalEvent
    | GuardrailEvent
    | FinalOutputEvent
    | ErrorEvent,
    Field(discriminator="type"),
]
TraceEventAdapter: TypeAdapter[TraceEvent] = TypeAdapter(TraceEvent)


class AgentTrace(_StrictModel):
    """Imported trace for one prompt/example/model side."""

    run_id: str = Field(min_length=1)
    prompt_id: str = Field(min_length=1)
    example_id: str = Field(min_length=1)
    role: TraceRole
    events: list[TraceEvent] = Field(default_factory=list)

    @property
    def tool_calls(self) -> list[ToolCallEvent]:
        """Return tool-call events in sequence order."""
        return [event for event in self.events if isinstance(event, ToolCallEvent)]

    @property
    def final_outputs(self) -> list[FinalOutputEvent]:
        """Return final-output events in sequence order."""
        return [event for event in self.events if isinstance(event, FinalOutputEvent)]

    @model_validator(mode="after")
    def _normalize_and_validate_events(self) -> Self:
        indices = [event.sequence_index for event in self.events]
        if len(indices) != len(set(indices)):
            duplicates = sorted({index for index in indices if indices.count(index) > 1})
            raise ValueError(f"duplicate sequence_index in trace: {duplicates}")

        self.events.sort(key=lambda event: event.sequence_index)

        seen_call_ids: set[str] = set()
        for event in self.events:
            if isinstance(event, ToolCallEvent) and event.call_id is not None:
                seen_call_ids.add(event.call_id)
            if (
                isinstance(event, ToolResultEvent)
                and event.call_id is not None
                and event.call_id not in seen_call_ids
            ):
                raise ValueError(f"unknown tool_result call_id: {event.call_id}")
        return self


class CaptureEnvelope(_StrictModel):
    """The on-disk capture envelope written by ``evalshift-sdk``.

    Forward-compatible by design (``extra="ignore"``, unlike ``_StrictModel``'s ``forbid``):
    unknown top-level keys from a newer SDK are ignored rather than rejected, so a capture written
    by a newer SDK still loads under a CLI that only knows older fields.
    """

    model_config = ConfigDict(extra="ignore")

    schema_version: str = Field(min_length=1)
    capture_id: str = Field(min_length=1)
    suite: str = Field(min_length=1)
    input_hash: str = ""
    code_version: str = ""
    created_at: str = ""
    trace: AgentTrace
    # Multi-turn provenance (schema 1.1.0) — optional so captures from older SDK releases
    # (which predate conversation tracking) still parse.
    conversation_id: str | None = None
    turn_index: int | None = None
    parent_capture_id: str | None = None


__all__ = [
    "AgentTrace",
    "CaptureEnvelope",
    "ErrorEvent",
    "FinalOutputEvent",
    "GuardrailEvent",
    "ModelCallEvent",
    "RequestedToolCall",
    "RetrievalEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "TraceEvent",
    "TraceEventAdapter",
    "TraceRole",
]
