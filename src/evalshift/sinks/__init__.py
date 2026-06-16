"""Capture output sinks: the ``Sink`` protocol plus the built-in ``FileSink`` and ``MemorySink``."""

from __future__ import annotations

from evalshift.sinks.base import Sink
from evalshift.sinks.file import FileSink
from evalshift.sinks.memory import MemorySink

__all__ = ["FileSink", "MemorySink", "Sink"]
