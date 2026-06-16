"""Unit tests for the capture gate and sink seam."""

from __future__ import annotations

import pytest

from evalshift.config import active_sink, is_capture_enabled
from evalshift.sinks.file import FileSink


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


def test_active_sink_returns_filesink() -> None:
    # Phase-3 sensitive: this assertion is intentionally isolated here so swapping the seam
    # to a configure()-registered sink later only touches this one test.
    assert isinstance(active_sink(), FileSink)
