"""Unit tests for the capture sampling gate (Phase 6)."""

from __future__ import annotations

from evalshift.hygiene.sample import should_capture


def test_none_rate_captures_everything() -> None:
    assert should_capture(None) is True


def test_rate_at_or_above_one_captures_everything() -> None:
    assert should_capture(1.0) is True
    assert should_capture(2.0) is True


def test_rate_at_or_below_zero_captures_nothing() -> None:
    assert should_capture(0.0) is False
    assert should_capture(-0.5) is False


def test_midrate_keeps_when_draw_below_rate() -> None:
    assert should_capture(0.5, rng=lambda: 0.4) is True


def test_midrate_drops_when_draw_at_or_above_rate() -> None:
    assert should_capture(0.5, rng=lambda: 0.6) is False
    # Boundary: draw == rate drops (strict <), keeping the 0.0 case consistent.
    assert should_capture(0.5, rng=lambda: 0.5) is False
