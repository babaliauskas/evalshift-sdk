"""Programmatic configure() registers a sink and threads forward-looking placeholders (Phase 3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from evalshift import capture, configure
from evalshift import config as cfg
from evalshift.config import (
    active_redactor,
    active_sink,
    reset_config,
    should_capture_now,
)
from evalshift.hygiene import dedup
from evalshift.sinks.file import FileSink
from evalshift.sinks.hygiene import HygieneSink
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


def test_configure_stores_hygiene_thresholds() -> None:
    configure(max_captures=5, capture_ttl=60.0)
    assert cfg._CONFIG.max_captures == 5
    assert cfg._CONFIG.capture_ttl == 60.0


def test_active_sink_stays_bare_with_only_sample_rate() -> None:
    sink = MemorySink()
    configure(sink=sink, sample_rate=0.5)
    # sample_rate gates at agent entry, not the sink, so it does not trigger wrapping.
    assert active_sink() is sink


def test_active_sink_wraps_when_dedup_set() -> None:
    configure(dedup=True)
    assert isinstance(active_sink(), HygieneSink)


def test_active_sink_wraps_when_max_captures_set() -> None:
    configure(max_captures=10)
    assert isinstance(active_sink(), HygieneSink)


def test_active_sink_wraps_when_capture_ttl_set() -> None:
    configure(capture_ttl=30.0)
    assert isinstance(active_sink(), HygieneSink)


def test_reset_config_clears_hygiene_and_dedup_registry() -> None:
    configure(max_captures=3, capture_ttl=10.0, dedup=True)
    dedup.mark_seen("s", "h")
    reset_config()
    assert cfg._CONFIG.max_captures is None
    assert cfg._CONFIG.capture_ttl is None
    assert dedup.is_duplicate("s", "h") is False  # registry cleared too
    assert isinstance(active_sink(), FileSink)  # back to bare sink


def test_should_capture_now_honors_rate() -> None:
    configure(sample_rate=0.0)
    assert should_capture_now() is False
    configure(sample_rate=1.0)
    assert should_capture_now() is True
    reset_config()
    assert should_capture_now() is True  # no rate -> always


def test_should_capture_now_defaults_to_capture_on_fault(monkeypatch: pytest.MonkeyPatch) -> None:
    configure(sample_rate=0.5)

    def boom() -> float:
        raise RuntimeError("rng broke")

    monkeypatch.setattr("random.random", boom)
    assert should_capture_now() is True  # a sampling fault must not silently drop telemetry


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
