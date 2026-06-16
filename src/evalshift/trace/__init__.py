"""Trace model, schema freeze, and serialization for the EvalShift SDK."""

from __future__ import annotations

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
from evalshift.trace.schema import SCHEMA_VERSION
from evalshift.trace.serialize import (
    build_capture,
    build_fixture_table,
    canonical_hash,
    capture_filename,
    dumps,
    envelope_to_dict,
)

__all__ = [
    "SCHEMA_VERSION",
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
    "build_capture",
    "build_fixture_table",
    "canonical_hash",
    "capture_filename",
    "dumps",
    "envelope_to_dict",
]
