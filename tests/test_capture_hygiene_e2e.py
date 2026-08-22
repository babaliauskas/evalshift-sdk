"""End-to-end hygiene: dedup collapse, GC count/TTL eviction, dedup-of-failures (Phase 6)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from evalshift import capture, configure, record_model_call
from tests.conftest import CaptureReader


def test_dedup_collapses_identical_inputs(capturing: Path, read_captures: CaptureReader) -> None:
    configure(dedup=True)

    @capture.agent(suite="dd", redact=False, tools=[])
    def handle(query: str) -> str:
        return query

    handle("hi")
    handle("hi")  # identical input -> deduped
    assert len(read_captures("dd")) == 1


def test_non_dedup_keeps_identical_inputs(capturing: Path, read_captures: CaptureReader) -> None:
    configure(dedup=False)

    @capture.agent(suite="nd", redact=False, tools=[])
    def handle(query: str) -> str:
        return query

    handle("hi")
    handle("hi")
    assert len(read_captures("nd")) == 2


def test_max_captures_caps_suite_directory(capturing: Path, read_captures: CaptureReader) -> None:
    configure(max_captures=2)

    @capture.agent(suite="cap", redact=False, tools=[])
    def handle(query: str) -> str:
        return query

    for i in range(5):
        handle(f"q{i}")  # distinct inputs so dedup never interferes
    assert len(read_captures("cap")) == 2  # GC trims to the cap after each write


def test_gc_eviction_of_old_captures_never_orphans_a_toolset_sidecar(
    capturing: Path, read_captures: CaptureReader
) -> None:
    """End-to-end companion to ``tests/test_hygiene_gc.py``'s structural regression: drive GC
    through the real pipeline (record_model_call -> ToolsetSink -> HygieneSink -> gc.evict) and
    confirm the toolset sidecar a since-evicted capture referenced is still there -- distinct
    toolsets are few and captures number in the thousands, so an evicted toolset would orphan
    every capture that still references it, not just the one that was just evicted.
    """
    configure(max_captures=2)
    tool = {"name": "search", "description": "", "input_schema": {}}

    @capture.agent(suite="gc_tools", redact=False, tools=[])
    def handle(query: str) -> str:
        record_model_call(model_id="m", tools=[tool], output=query)
        return query

    for i in range(5):
        handle(f"q{i}")  # distinct inputs -> distinct captures; identical toolset every time

    survivors = read_captures("gc_tools")
    assert len(survivors) == 2  # GC trimmed the suite dir, exactly as in the test above
    [mc] = [e for e in survivors[0]["trace"]["events"] if e["type"] == "model_call"]
    fingerprint = mc["toolset_ref"]
    assert fingerprint is not None
    sidecar = capturing / "toolsets" / f"{fingerprint.removeprefix('sha256:')}.json"
    assert sidecar.exists()  # never touched by the GC pass that just trimmed captures/gc_tools/


def test_capture_ttl_evicts_aged_files(capturing: Path) -> None:
    configure(capture_ttl=60.0)
    suite_dir = capturing / "captures" / "ttl"

    @capture.agent(suite="ttl", redact=False, tools=[])
    def handle(query: str) -> str:
        return query

    handle("a")
    aged = next(suite_dir.glob("*.json"))
    os.utime(aged, (100.0, 100.0))  # push age well past the 60s TTL

    handle("b")  # next write triggers GC, which evicts the aged file
    survivors = list(suite_dir.glob("*.json"))
    assert len(survivors) == 1
    assert aged not in survivors


def test_dedup_success_suppresses_later_identical_failure(
    capturing: Path, read_captures: CaptureReader
) -> None:
    configure(dedup=True)
    fail = {"now": False}

    @capture.agent(suite="ddf", redact=False, tools=[])
    def handle(query: str) -> str:
        if fail["now"]:
            raise ValueError("boom")
        return "ok"

    handle("same")  # success -> marks input seen
    fail["now"] = True
    with pytest.raises(ValueError, match="boom"):
        handle("same")  # same input_hash -> deduped, failure NOT captured (best-effort, D-6)

    caps = read_captures("ddf")
    assert len(caps) == 1  # only the first success survived
    assert not [e for e in caps[0]["trace"]["events"] if e["type"] == "error"]
