"""Programmatic configure() registers a sink and threads forward-looking placeholders (Phase 3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from evalshift import capture, configure
from evalshift import config as cfg
from evalshift.config import (
    active_sink,
    require_model_call,
    reset_config,
    should_capture_now,
)
from evalshift.hygiene import dedup
from evalshift.sinks.file import FileSink
from evalshift.sinks.hygiene import HygieneSink
from evalshift.sinks.memory import MemorySink


def test_configured_sink_is_returned_by_active_sink() -> None:
    # Default hygiene wraps the sink; disable it to assert the raw registration identity.
    sink = MemorySink()
    configure(sink=sink, dedup=False, max_captures=None, capture_ttl=None)
    assert active_sink() is sink


def test_configured_sink_is_wrapped_under_default_hygiene() -> None:
    sink = MemorySink()
    configure(sink=sink)
    wrapped = active_sink()
    assert isinstance(wrapped, HygieneSink)
    assert wrapped._base is sink


def test_reset_config_restores_default_filesink() -> None:
    configure(sink=MemorySink())
    reset_config()
    # Back to the env-derived default: a HygieneSink wrapping a disk FileSink.
    sink = active_sink()
    assert isinstance(sink, HygieneSink)
    assert isinstance(sink._base, FileSink)


def test_configure_merges_without_clobbering_sink() -> None:
    sink = MemorySink()
    configure(sink=sink, dedup=False, max_captures=None, capture_ttl=None)
    configure(sample_rate=0.5)
    assert active_sink() is sink


def test_configure_stores_placeholder_fields() -> None:
    configure(sample_rate=0.25, dedup=True)
    assert cfg._CONFIG.sample_rate == 0.25
    assert cfg._CONFIG.dedup is True


def test_configure_stores_hygiene_thresholds() -> None:
    configure(max_captures=5, capture_ttl=60.0)
    assert cfg._CONFIG.max_captures == 5
    assert cfg._CONFIG.capture_ttl == 60.0


def test_sample_rate_alone_does_not_trigger_wrapping() -> None:
    sink = MemorySink()
    # sample_rate gates at agent entry, not the sink; with hygiene off it stays bare.
    configure(sink=sink, sample_rate=0.5, dedup=False, max_captures=None, capture_ttl=None)
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


def test_reset_config_restores_env_defaults_and_clears_dedup_registry() -> None:
    configure(max_captures=3, capture_ttl=10.0, dedup=True)
    dedup.mark_seen("s", "h")
    reset_config()
    assert cfg._CONFIG.max_captures == 200  # back to the env-derived built-in default
    assert cfg._CONFIG.dedup is True
    assert cfg._CONFIG.capture_ttl is None
    assert dedup.is_duplicate("s", "h") is False  # registry cleared too
    assert isinstance(active_sink(), HygieneSink)  # default bounded hygiene re-applied


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


def test_require_model_call_defaults_false() -> None:
    assert require_model_call() is False


def test_configure_sets_require_model_call() -> None:
    configure(require_model_call=True)
    assert require_model_call() is True


def test_reset_config_restores_require_model_call_default() -> None:
    configure(require_model_call=True)
    reset_config()
    assert require_model_call() is False


def test_configure_merges_require_model_call_without_clobber() -> None:
    sink = MemorySink()
    configure(sink=sink, dedup=False, max_captures=None, capture_ttl=None)
    configure(require_model_call=True)
    assert active_sink() is sink
    assert require_model_call() is True


def test_capture_routes_into_configured_sink(capturing: Path) -> None:
    sink = MemorySink()
    configure(sink=sink)

    @capture.agent(suite="cfg_suite", redact=False, tools=[])
    def handle(query: str) -> str:
        return f"ok:{query}"

    assert handle("hi") == "ok:hi"
    captured = sink.flush()
    assert len(captured) == 1
    assert captured[0].suite == "cfg_suite"
    # Routed to memory, so nothing hit disk under the capture base.
    assert list(capturing.rglob("*.json")) == []
