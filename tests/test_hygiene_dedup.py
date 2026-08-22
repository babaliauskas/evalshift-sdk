"""Unit tests for the per-process input dedup registry (Phase 6) and conversation-turn dedup."""

from __future__ import annotations

import threading
from collections.abc import Iterator

import pytest

from evalshift.capture.span import SpanTree
from evalshift.hygiene import dedup
from evalshift.sinks.hygiene import HygieneSink
from evalshift.sinks.memory import MemorySink
from evalshift.trace.models import CaptureEnvelope
from evalshift.trace.serialize import build_capture


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    dedup.reset_registry()
    yield
    dedup.reset_registry()


def test_unseen_input_is_not_duplicate() -> None:
    assert dedup.is_duplicate("suite", "hash_a") is False


def test_marked_input_is_duplicate() -> None:
    dedup.mark_seen("suite", "hash_a")
    assert dedup.is_duplicate("suite", "hash_a") is True


def test_keyed_by_suite_and_hash() -> None:
    dedup.mark_seen("suite_a", "hash_a")
    # Same hash, different suite -> not a duplicate.
    assert dedup.is_duplicate("suite_b", "hash_a") is False
    # Same suite, different hash -> not a duplicate.
    assert dedup.is_duplicate("suite_a", "hash_b") is False


def test_reset_registry_clears_marks() -> None:
    dedup.mark_seen("suite", "hash_a")
    dedup.reset_registry()
    assert dedup.is_duplicate("suite", "hash_a") is False


def test_concurrent_marks_are_not_lost() -> None:
    keys = [("suite", f"hash_{i}") for i in range(200)]

    def worker(start: int) -> None:
        for suite, h in keys[start::4]:
            dedup.mark_seen(suite, h)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(dedup.is_duplicate(suite, h) for suite, h in keys)


# --- end-to-end: HygieneSink dedup vs. conversation turns (schema 1.1.0) ---


def _turn_envelope(
    capture_id: str,
    *,
    agent_input: str,
    conversation_id: str | None,
    turn_index: int | None,
    suite: str = "s",
) -> CaptureEnvelope:
    return build_capture(
        SpanTree(),
        suite=suite,
        agent_input=agent_input,
        capture_id=capture_id,
        conversation_id=conversation_id,
        turn_index=turn_index,
    )


def test_conversation_turns_with_identical_input_are_not_deduped() -> None:
    # Two turns of the same conversation both saying "yes" must NOT collapse under dedup — the
    # input_hash differs per turn_index (build_capture folds conversation_id/turn_index in).
    base = MemorySink()
    sink = HygieneSink(base, dedup=True, max_captures=None, capture_ttl=None)

    sink.write(_turn_envelope("cap_1", agent_input="yes", conversation_id="conv_1", turn_index=1))
    sink.write(_turn_envelope("cap_2", agent_input="yes", conversation_id="conv_1", turn_index=2))

    assert len(base.captures) == 2
    assert {c.capture_id for c in base.captures} == {"cap_1", "cap_2"}


def test_standalone_captures_with_identical_input_and_no_conversation_id_still_deduped() -> None:
    # Regression: standalone (conversation_id=None) captures keep the pre-1.1.0 dedup behavior —
    # identical agent_input still collapses to one write.
    base = MemorySink()
    sink = HygieneSink(base, dedup=True, max_captures=None, capture_ttl=None)

    sink.write(_turn_envelope("cap_1", agent_input="yes", conversation_id=None, turn_index=None))
    sink.write(_turn_envelope("cap_2", agent_input="yes", conversation_id=None, turn_index=None))

    assert len(base.captures) == 1
    assert base.captures[0].capture_id == "cap_1"
