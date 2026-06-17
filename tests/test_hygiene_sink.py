"""Unit tests for the HygieneSink wrapper composing dedup + GC over a base sink (Phase 6)."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from evalshift.capture.span import SpanTree
from evalshift.hygiene import dedup, gc
from evalshift.sinks.base import Sink
from evalshift.sinks.file import FileSink
from evalshift.sinks.hygiene import HygieneSink
from evalshift.sinks.memory import MemorySink
from evalshift.trace.models import CaptureEnvelope
from evalshift.trace.serialize import build_capture


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    dedup.reset_registry()
    yield
    dedup.reset_registry()


def _env(agent_input: str, capture_id: str, suite: str = "s") -> CaptureEnvelope:
    return build_capture(SpanTree(), suite=suite, agent_input=agent_input, capture_id=capture_id)


def test_satisfies_sink_protocol() -> None:
    sink = HygieneSink(MemorySink(), dedup=True, max_captures=None, capture_ttl=None)
    assert isinstance(sink, Sink)


def test_dedup_skips_second_identical_input() -> None:
    base = MemorySink()
    sink = HygieneSink(base, dedup=True, max_captures=None, capture_ttl=None)
    assert sink.write(_env("hi", "cap_1")) is None  # MemorySink returns None
    sink.write(_env("hi", "cap_2"))  # same input_hash -> deduped
    assert len(base.captures) == 1
    assert base.captures[0].capture_id == "cap_1"


def test_non_dedup_writes_both() -> None:
    base = MemorySink()
    sink = HygieneSink(base, dedup=False, max_captures=None, capture_ttl=None)
    sink.write(_env("hi", "cap_1"))
    sink.write(_env("hi", "cap_2"))
    assert len(base.captures) == 2


def test_different_inputs_both_written_under_dedup() -> None:
    base = MemorySink()
    sink = HygieneSink(base, dedup=True, max_captures=None, capture_ttl=None)
    sink.write(_env("hi", "cap_1"))
    sink.write(_env("bye", "cap_2"))
    assert len(base.captures) == 2


def test_gc_runs_on_filesink_base(tmp_path: Path) -> None:
    sink = HygieneSink(FileSink(base=tmp_path), dedup=False, max_captures=1, capture_ttl=None)
    p1 = sink.write(_env("a", "cap_1"))
    assert p1 is not None
    os.utime(p1, (100.0, 100.0))  # age cap_1 so the count rule evicts it on the next write
    p2 = sink.write(_env("b", "cap_2"))
    assert p2 is not None
    assert not p1.exists()  # evicted by GC
    assert p2.exists()


def test_dedup_fault_does_not_drop_write(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object, **_k: object) -> bool:
        raise RuntimeError("dedup registry exploded")

    monkeypatch.setattr(dedup, "is_duplicate", boom)
    base = MemorySink()
    sink = HygieneSink(base, dedup=True, max_captures=None, capture_ttl=None)
    sink.write(_env("hi", "cap_1"))
    assert len(base.captures) == 1  # write still happened


def test_gc_fault_does_not_drop_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object, **_k: object) -> list[Path]:
        raise RuntimeError("gc exploded")

    monkeypatch.setattr(gc, "evict", boom)
    sink = HygieneSink(FileSink(base=tmp_path), dedup=False, max_captures=1, capture_ttl=None)
    path = sink.write(_env("a", "cap_1"))
    assert path is not None
    assert path.exists()  # capture still written despite GC blowing up
