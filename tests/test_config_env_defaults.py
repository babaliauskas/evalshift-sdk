"""Env vars set the built-in hygiene defaults so a host that never calls ``configure()`` still
keeps ``captures/`` bounded. Precedence: an explicit ``configure(...)`` beats the env default,
which beats the built-in constant.
"""

from __future__ import annotations

import pytest

from evalshift import config as cfg
from evalshift.config import active_sink, configure, reset_config
from evalshift.sinks.file import FileSink
from evalshift.sinks.hygiene import HygieneSink


def test_defaults_are_bounded_out_of_the_box() -> None:
    # No env, no configure(): dedup on, generous count cap, no TTL / sampling.
    assert cfg._CONFIG.dedup is True
    assert cfg._CONFIG.max_captures == 200
    assert cfg._CONFIG.capture_ttl is None
    assert cfg._CONFIG.sample_rate is None
    assert isinstance(active_sink(), HygieneSink)


def test_env_max_captures_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVALSHIFT_MAX_CAPTURES", "50")
    reset_config()
    assert cfg._CONFIG.max_captures == 50


@pytest.mark.parametrize("sentinel", ["0", "none", "unlimited", "off", "NONE", " Unlimited "])
def test_env_max_captures_sentinel_disables_cap(
    sentinel: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVALSHIFT_MAX_CAPTURES", sentinel)
    monkeypatch.setenv("EVALSHIFT_DEDUP", "off")  # also disable dedup to reach the bare seam
    reset_config()
    assert cfg._CONFIG.max_captures is None
    # With every hygiene knob off, the sink is the bare FileSink (unbounded escape hatch).
    assert isinstance(active_sink(), FileSink)


def test_env_max_captures_malformed_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVALSHIFT_MAX_CAPTURES", "not-a-number")
    reset_config()
    assert cfg._CONFIG.max_captures == 200  # fail-open to the built-in default


def test_env_negative_max_captures_treated_as_unlimited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVALSHIFT_MAX_CAPTURES", "-5")
    reset_config()
    assert cfg._CONFIG.max_captures is None


@pytest.mark.parametrize(
    ("value", "expected"), [("off", False), ("0", False), ("1", True), ("yes", True)]
)
def test_env_dedup_parsed(value: str, expected: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVALSHIFT_DEDUP", value)
    reset_config()
    assert cfg._CONFIG.dedup is expected


def test_env_dedup_malformed_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVALSHIFT_DEDUP", "maybe")
    reset_config()
    assert cfg._CONFIG.dedup is True  # built-in default


def test_env_capture_ttl_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVALSHIFT_CAPTURE_TTL", "3600")
    reset_config()
    assert cfg._CONFIG.capture_ttl == 3600.0


def test_env_capture_ttl_none_sentinel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVALSHIFT_CAPTURE_TTL", "none")
    reset_config()
    assert cfg._CONFIG.capture_ttl is None


def test_env_sample_rate_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVALSHIFT_SAMPLE_RATE", "0.25")
    reset_config()
    assert cfg._CONFIG.sample_rate == 0.25


def test_env_sample_rate_malformed_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVALSHIFT_SAMPLE_RATE", "half")
    reset_config()
    assert cfg._CONFIG.sample_rate is None


def test_configure_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVALSHIFT_MAX_CAPTURES", "50")
    reset_config()
    assert cfg._CONFIG.max_captures == 50
    configure(max_captures=999)  # explicit call wins over the env default
    assert cfg._CONFIG.max_captures == 999


def test_reset_config_rereads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_config()
    assert cfg._CONFIG.max_captures == 200
    monkeypatch.setenv("EVALSHIFT_MAX_CAPTURES", "7")
    reset_config()
    assert cfg._CONFIG.max_captures == 7
