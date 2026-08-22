"""Conformance: a migrated / reloaded capture's inner trace stays CLI-valid.

This is the Phase 8 goal in test form. A synthetic older-version capture is migrated (where a
migration bridges it), and a written capture is read back through the reverse path; in both cases
the inner ``AgentTrace`` must validate against the frozen vendored CLI model. Schema 2.0.0 ships
no migration from the 1.x major, so a real 1.x capture is refused rather than migrated (see
``ObsoleteSchemaVersionError`` below and ``docs/SCHEMA.md``) -- "promotable/replayable" now holds
only within a major this SDK still bridges to. Pydantic is a dev/test dependency only.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from evalshift.capture.span import SpanTree
from evalshift.trace.migrate import (
    ObsoleteSchemaVersionError,
    load_envelope,
    register_migration,
    reset_migrations,
    upgrade_envelope_dict,
)
from evalshift.trace.models import to_jsonable
from evalshift.trace.serialize import build_capture, dumps, envelope_to_dict
from tests.conformance.cli_models_vendored import AgentTrace, CaptureEnvelope


@pytest.fixture(autouse=True)
def _reset_migrations() -> Iterator[None]:
    reset_migrations()
    yield
    reset_migrations()


def _full_tree() -> SpanTree:
    tree = SpanTree()
    m = tree.open_span(
        "model_call",
        span_id="m1",
        start_ts=1.0,
        data={"model_id": "claude-opus-4-8", "input": "hi", "output": "yo", "input_tokens": 3},
    )
    tree.close_span(m, end_ts=1.5)
    t = tree.open_span(
        "tool", span_id="c1", start_ts=2.0, data={"name": "search", "arguments": {"q": "x"}}
    )
    tree.close_span(t, end_ts=2.5, result={"hits": 1})
    tree.open_span("final_output", span_id="f1", start_ts=3.0, data={"text": "done"})
    return tree


def _capture_dict() -> dict[str, Any]:
    return envelope_to_dict(
        build_capture(_full_tree(), suite="support_agent", agent_input="hi", capture_id="cap_t")
    )


def test_migrated_capture_inner_trace_validates_against_cli_model() -> None:
    def rename_suite(env: dict[str, Any]) -> dict[str, Any]:
        upgraded = dict(env)
        upgraded["suite"] = upgraded.pop("legacy_suite")
        return upgraded

    register_migration("0.9.0", "1.0.0", rename_suite, description="legacy_suite -> suite")

    old = _capture_dict()
    old["schema_version"] = "0.9.0"
    old["legacy_suite"] = old.pop("suite")

    # 0.9.0 -> 1.0.0 is registered, but nothing bridges 1.0.0 to the current 2.0.0 major -- the
    # chain stops short of a real schema-version gap this SDK deliberately refuses to migrate.
    migrated = upgrade_envelope_dict(old, target="1.0.0")
    assert migrated["schema_version"] == "1.0.0"
    assert migrated["suite"] == "support_agent"
    AgentTrace.model_validate(migrated["trace"])  # promotable: still the CLI contract
    CaptureEnvelope.model_validate(migrated)  # whole envelope still CLI-valid post-migration


def test_older_major_capture_is_refused_not_silently_migrated() -> None:
    """2.0.0 ships no migration from 1.x -- see ObsoleteSchemaVersionError / docs/SCHEMA.md. A
    migration that invented tools_offered for a legacy capture would assert every one of them ran
    with no tools offered, which is false. Refusing outright is the conformant behavior."""
    old = _capture_dict()
    old["schema_version"] = "1.1.0"

    with pytest.raises(ObsoleteSchemaVersionError) as exc_info:
        upgrade_envelope_dict(old)
    assert str(exc_info.value) == (
        "capture written by an older evalshift-sdk; re-run your agent to re-capture."
    )


def test_reloaded_capture_inner_trace_validates_against_cli_model() -> None:
    original = build_capture(
        _full_tree(), suite="support_agent", agent_input="hi", capture_id="cap_t"
    )
    reloaded = load_envelope(dumps(original))
    AgentTrace.model_validate(to_jsonable(reloaded.trace))  # replayable after a disk round trip


def test_migration_never_leaks_schema_version_into_trace() -> None:
    register_migration("0.9.0", "1.0.0", lambda env: dict(env))
    old = _capture_dict()
    old["schema_version"] = "0.9.0"

    # Explicit target="1.0.0": stays within the registered synthetic edge, same major, so the
    # refusal gate (2.0.0's default target) never enters into it -- see the tests above for that.
    migrated = upgrade_envelope_dict(old, target="1.0.0")
    assert "schema_version" not in migrated["trace"]
    # the vendored model rejects schema_version, so a clean validate proves it is absent
    AgentTrace.model_validate(migrated["trace"])
