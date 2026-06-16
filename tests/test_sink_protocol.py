"""The Sink protocol is satisfied by both built-in sinks (Phase 3)."""

from __future__ import annotations

from evalshift.sinks.base import Sink
from evalshift.sinks.file import FileSink
from evalshift.sinks.memory import MemorySink


def test_filesink_satisfies_sink_protocol() -> None:
    assert isinstance(FileSink(), Sink)


def test_memorysink_satisfies_sink_protocol() -> None:
    assert isinstance(MemorySink(), Sink)
