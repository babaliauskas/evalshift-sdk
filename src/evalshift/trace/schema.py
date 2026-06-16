"""Frozen EvalShift trace + capture-envelope schema (the contract).

Single source of truth for the field set the SDK emits. Mirrors the CLI trace contract in
``evalshift-cli/src/evalshift/traces/models.py`` (``AgentTrace`` plus the discriminated
``TraceEvent`` union). Stdlib only — no pydantic at runtime (D-deps). The parity test in
``tests/conformance`` asserts these constants stay field-for-field aligned with the CLI model.

``SCHEMA_VERSION`` lives in the **capture envelope** (``cap_<id>.json``) only, never inside the
``AgentTrace`` JSONL: the CLI model is ``extra="forbid"`` and has no ``schema_version`` field, so
embedding it in the trace would fail CLI validation (D-5b).
"""

from __future__ import annotations

from typing import Final

#: Frozen SDK trace schema version (sibling to the CLI artifact version). Emitted in the
#: capture envelope, not the trace itself.
SCHEMA_VERSION: Final = "1.0.0"

#: Discriminator values for the ``TraceEvent`` union (CLI ``type`` field).
EVENT_TYPES: Final[tuple[str, ...]] = (
    "model_call",
    "tool_call",
    "tool_result",
    "retrieval",
    "guardrail",
    "final_output",
    "error",
)

#: Allowed ``AgentTrace.role`` values (CLI ``TraceRole``).
TRACE_ROLES: Final[tuple[str, ...]] = ("source", "target")

#: ``AgentTrace`` root fields.
AGENT_TRACE_FIELDS: Final[tuple[str, ...]] = (
    "run_id",
    "prompt_id",
    "example_id",
    "role",
    "events",
)

#: Fields shared by every event (CLI ``_BaseEvent``).
BASE_EVENT_FIELDS: Final[tuple[str, ...]] = (
    "type",
    "sequence_index",
    "timestamp",
    "metadata",
)

#: Per-event field set: base fields plus each type's own fields, mirroring the CLI models
#: exactly. Order matches the CLI class definitions.
EVENT_FIELDS: Final[dict[str, tuple[str, ...]]] = {
    "model_call": (
        *BASE_EVENT_FIELDS,
        "model_id",
        "input",
        "output",
        "input_tokens",
        "output_tokens",
        "cost_usd",
        "latency_ms",
    ),
    "tool_call": (*BASE_EVENT_FIELDS, "name", "arguments", "call_id", "parent_call_id"),
    "tool_result": (*BASE_EVENT_FIELDS, "name", "call_id", "result", "error"),
    "retrieval": (*BASE_EVENT_FIELDS, "source", "query", "documents"),
    "guardrail": (*BASE_EVENT_FIELDS, "name", "verdict", "reason"),
    "final_output": (*BASE_EVENT_FIELDS, "text"),
    "error": (*BASE_EVENT_FIELDS, "message", "category"),
}

#: Allowed ``GuardrailEvent.verdict`` values (CLI literal).
GUARDRAIL_VERDICTS: Final[tuple[str, ...]] = ("pass", "fail", "warn", "skipped")

#: Capture-envelope keys for ``cap_<id>.json`` (written in Phase 1). The inner ``trace`` value is
#: a CLI-valid ``AgentTrace``; ``schema_version`` stays OUTSIDE it.
ENVELOPE_KEYS: Final[tuple[str, ...]] = (
    "schema_version",
    "capture_id",
    "suite",
    "input_hash",
    "code_version",
    "created_at",
    "trace",
)

__all__ = [
    "AGENT_TRACE_FIELDS",
    "BASE_EVENT_FIELDS",
    "ENVELOPE_KEYS",
    "EVENT_FIELDS",
    "EVENT_TYPES",
    "GUARDRAIL_VERDICTS",
    "SCHEMA_VERSION",
    "TRACE_ROLES",
]
