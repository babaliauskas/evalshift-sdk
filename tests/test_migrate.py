"""Unit tests for the schema-versioning + migration framework (dict layer).

There is only one real schema version ("1.0.0"), so the chain is exercised with *synthetic*
migrations registered per-test via ``register_migration`` and cleared by the autouse fixture.
No pydantic here — that lives in ``tests/conformance/test_migrate_parity.py``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

import pytest

from evalshift.trace.migrate import (
    CURRENT_SCHEMA_VERSION,
    InvalidSchemaVersionError,
    MissingSchemaVersionError,
    NoMigrationPathError,
    SchemaVersion,
    UnreadableCaptureError,
    UnsupportedSchemaVersionError,
    detect_version,
    load_capture,
    register_migration,
    registered_migrations,
    reset_migrations,
    upgrade_envelope_dict,
)


@pytest.fixture(autouse=True)
def _reset_migrations() -> Iterator[None]:
    """Clear the process-wide migration registry around every test (mirrors conftest)."""
    reset_migrations()
    yield
    reset_migrations()


def _envelope(version: str = "1.0.0", **overrides: Any) -> dict[str, Any]:
    env: dict[str, Any] = {
        "schema_version": version,
        "capture_id": "cap_x",
        "suite": "s",
        "input_hash": "deadbeef",
        "code_version": "",
        "created_at": "2026-06-16T12:00:00+00:00",
        "trace": {
            "run_id": "cap_x",
            "prompt_id": "s",
            "example_id": "cap_x",
            "role": "source",
            "events": [],
        },
    }
    env.update(overrides)
    return env


# --- SchemaVersion ----------------------------------------------------------------------------


def test_schema_version_parse_and_str_round_trip() -> None:
    assert SchemaVersion.parse("1.2.3") == SchemaVersion(1, 2, 3)
    assert str(SchemaVersion.parse("1.2.3")) == "1.2.3"


def test_schema_version_ordering() -> None:
    assert SchemaVersion(1, 0, 0) < SchemaVersion(1, 0, 1)
    assert SchemaVersion(1, 0, 1) < SchemaVersion(1, 1, 0)
    assert SchemaVersion(1, 1, 0) < SchemaVersion(2, 0, 0)


@pytest.mark.parametrize("raw", ["1.0", "1.0.0.0", "abc", "", "1.-1.0", "1.0.x", "v1.0.0"])
def test_schema_version_parse_rejects_malformed(raw: str) -> None:
    with pytest.raises(InvalidSchemaVersionError):
        SchemaVersion.parse(raw)


def test_current_schema_version_matches_constant() -> None:
    assert str(CURRENT_SCHEMA_VERSION) == "1.0.0"


# --- detect_version ---------------------------------------------------------------------------


def test_detect_version_reads_envelope() -> None:
    assert detect_version(_envelope("1.0.0")) == SchemaVersion(1, 0, 0)


def test_detect_version_missing_key_raises() -> None:
    env = _envelope()
    del env["schema_version"]
    with pytest.raises(MissingSchemaVersionError):
        detect_version(env)


def test_detect_version_missing_key_uses_default() -> None:
    env = _envelope()
    del env["schema_version"]
    assert detect_version(env, default="1.0.0") == SchemaVersion(1, 0, 0)


def test_detect_version_invalid_string_raises() -> None:
    with pytest.raises(InvalidSchemaVersionError):
        detect_version(_envelope("nope"))


def test_detect_version_non_string_value_raises() -> None:
    with pytest.raises(InvalidSchemaVersionError):
        detect_version(_envelope(version="1.0.0", schema_version=100))


def test_detect_version_non_dict_raises() -> None:
    with pytest.raises(UnreadableCaptureError):
        detect_version(["not", "a", "dict"])  # type: ignore[arg-type]


# --- registry ---------------------------------------------------------------------------------


def test_registry_starts_empty() -> None:
    assert registered_migrations() == ()


def test_register_duplicate_from_version_raises() -> None:
    register_migration("0.9.0", "1.0.0", lambda env: env)
    with pytest.raises(ValueError, match="already registered"):
        register_migration("0.9.0", "1.0.0", lambda env: env)


def test_register_backward_step_raises() -> None:
    with pytest.raises(ValueError, match="move forward"):
        register_migration("1.0.0", "0.9.0", lambda env: env)


def test_reset_migrations_clears_registry() -> None:
    register_migration("0.9.0", "1.0.0", lambda env: env)
    assert len(registered_migrations()) == 1
    reset_migrations()
    assert registered_migrations() == ()


# --- upgrade_envelope_dict --------------------------------------------------------------------


def test_same_version_is_content_preserving_passthrough() -> None:
    env = _envelope("1.0.0")
    result = upgrade_envelope_dict(env)
    assert result == env
    assert result is not env  # a copy, not the original


def test_single_step_upgrade_runs_migration() -> None:
    def rename_suite(env: dict[str, Any]) -> dict[str, Any]:
        upgraded = dict(env)
        upgraded["suite"] = upgraded.pop("legacy_suite")
        return upgraded

    register_migration("0.9.0", "1.0.0", rename_suite)
    old = _envelope("0.9.0", legacy_suite="support")
    del old["suite"]

    result = upgrade_envelope_dict(old)
    assert result["schema_version"] == "1.0.0"
    assert result["suite"] == "support"
    assert "legacy_suite" not in result


def test_multi_step_chain_runs_in_order() -> None:
    order: list[str] = []

    def step_a(env: dict[str, Any]) -> dict[str, Any]:
        order.append("a")
        return dict(env)

    def step_b(env: dict[str, Any]) -> dict[str, Any]:
        order.append("b")
        return dict(env)

    register_migration("0.8.0", "0.9.0", step_a)
    register_migration("0.9.0", "1.0.0", step_b)

    result = upgrade_envelope_dict(_envelope("0.8.0"))
    assert order == ["a", "b"]
    assert result["schema_version"] == "1.0.0"


def test_missing_step_raises_no_migration_path() -> None:
    register_migration("0.9.0", "1.0.0", lambda env: env)  # gap: nothing from 0.8.0
    with pytest.raises(NoMigrationPathError):
        upgrade_envelope_dict(_envelope("0.8.0"))


def test_newer_minor_is_tolerated_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="evalshift"):
        result = upgrade_envelope_dict(_envelope("1.7.3"))
    assert result["schema_version"] == "1.7.3"  # genuinely newer — keeps its own version
    assert any("newer than supported" in rec.message for rec in caplog.records)


def test_newer_major_is_refused() -> None:
    with pytest.raises(UnsupportedSchemaVersionError):
        upgrade_envelope_dict(_envelope("2.0.0"))


def test_upgrade_does_not_mutate_input() -> None:
    def mutating_step(env: dict[str, Any]) -> dict[str, Any]:
        env["trace"]["events"].append({"type": "error"})  # mutate the copy we were handed
        return env

    register_migration("0.9.0", "1.0.0", mutating_step)
    original = _envelope("0.9.0")
    upgrade_envelope_dict(original)
    assert original["trace"]["events"] == []  # caller's dict is untouched


# --- load_capture -----------------------------------------------------------------------------


def test_load_capture_parses_then_upgrades() -> None:
    text = json.dumps(_envelope("1.0.0"))
    assert load_capture(text) == _envelope("1.0.0")


def test_load_capture_accepts_bytes() -> None:
    raw = json.dumps(_envelope("1.0.0")).encode("utf-8")
    assert load_capture(raw)["schema_version"] == "1.0.0"


@pytest.mark.parametrize("text", ["{not json", "", "[]", "42", '"just a string"'])
def test_load_capture_bad_input_raises_unreadable(text: str) -> None:
    with pytest.raises(UnreadableCaptureError):
        load_capture(text)
