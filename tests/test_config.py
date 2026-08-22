"""Unit tests for the capture gate and sink seam."""

from __future__ import annotations

import pytest

from evalshift.config import active_sink, configure, is_capture_enabled
from evalshift.sinks.file import FileSink
from evalshift.sinks.hygiene import HygieneSink


def test_capture_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVALSHIFT_CAPTURE", raising=False)
    assert is_capture_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "True", "yes", "on", " on "])
def test_truthy_values_enable(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVALSHIFT_CAPTURE", value)
    assert is_capture_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "garbage", "2"])
def test_falsy_values_disable(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVALSHIFT_CAPTURE", value)
    assert is_capture_enabled() is False


def test_gate_read_live_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVALSHIFT_CAPTURE", raising=False)
    assert is_capture_enabled() is False
    monkeypatch.setenv("EVALSHIFT_CAPTURE", "1")
    assert is_capture_enabled() is True


def test_active_sink_wraps_filesink_by_default() -> None:
    # Hygiene is bounded-on by default (dedup + max_captures=200), so the default sink is a
    # HygieneSink wrapping the disk FileSink. Isolated here so future seam swaps touch one test.
    sink = active_sink()
    assert isinstance(sink, HygieneSink)
    assert isinstance(sink._base, FileSink)


def test_active_sink_bare_filesink_when_hygiene_disabled() -> None:
    # Explicitly opting out of every hygiene knob restores the identity-preserved bare seam.
    configure(dedup=False, max_captures=None, capture_ttl=None)
    assert isinstance(active_sink(), FileSink)
