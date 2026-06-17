"""Unit tests for the per-process input dedup registry (Phase 6)."""

from __future__ import annotations

import threading
from collections.abc import Iterator

import pytest

from evalshift.hygiene import dedup


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
