"""Programmatic configure() registers a sink and threads forward-looking placeholders (Phase 3)."""

from __future__ import annotations

from pathlib import Path

from evalshift import capture, configure
from evalshift import config as cfg
from evalshift.config import active_redactor, active_sink, reset_config
from evalshift.sinks.file import FileSink
from evalshift.sinks.memory import MemorySink


def test_configured_sink_is_returned_by_active_sink() -> None:
    sink = MemorySink()
    configure(sink=sink)
    assert active_sink() is sink


def test_reset_config_restores_default_filesink() -> None:
    configure(sink=MemorySink())
    reset_config()
    assert isinstance(active_sink(), FileSink)


def test_configure_merges_without_clobbering_sink() -> None:
    sink = MemorySink()
    configure(sink=sink)
    configure(sample_rate=0.5)
    assert active_sink() is sink


def test_configure_stores_placeholder_fields() -> None:
    def red(value: object) -> object:
        return value

    configure(redact=red, sample_rate=0.25, dedup=True)
    assert cfg._CONFIG.redact is red
    assert cfg._CONFIG.sample_rate == 0.25
    assert cfg._CONFIG.dedup is True


def test_active_redactor_returns_configured_else_none() -> None:
    assert active_redactor() is None

    def red(value: object) -> object:
        return value

    configure(redact=red)
    assert active_redactor() is red


def test_capture_routes_into_configured_sink(capturing: Path) -> None:
    sink = MemorySink()
    configure(sink=sink)

    @capture.agent(suite="cfg_suite")
    def handle(query: str) -> str:
        return f"ok:{query}"

    assert handle("hi") == "ok:hi"
    captured = sink.flush()
    assert len(captured) == 1
    assert captured[0].suite == "cfg_suite"
    # Routed to memory, so nothing hit disk under the capture base.
    assert list(capturing.rglob("*.json")) == []
