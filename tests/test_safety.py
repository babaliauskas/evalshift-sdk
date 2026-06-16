"""Unit tests for the fail-open safety boundary."""

from __future__ import annotations

import logging

import pytest

from evalshift.safety import fail_open, guard


def _boom() -> int:
    raise RuntimeError("boom")


# --- fail_open ---


def test_fail_open_swallows_exception() -> None:
    with fail_open("doing a thing"):
        raise RuntimeError("boom")
    # reaching here means the exception did not propagate


def test_fail_open_yields_for_success() -> None:
    ran = False
    with fail_open("doing a thing"):
        ran = True
    assert ran


def test_fail_open_logs_swallowed_action_at_debug(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger="evalshift"), fail_open("write sink"):
        raise RuntimeError("boom")
    assert any("write sink" in record.message for record in caplog.records)


# --- guard ---


def test_guard_returns_value_on_success() -> None:
    assert guard("compute", lambda: 7) == 7


def test_guard_returns_none_on_failure() -> None:
    assert guard("compute", _boom) is None


def test_guard_logs_swallowed_action_at_debug(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger="evalshift"):
        guard("build capture", _boom)
    assert any("build capture" in record.message for record in caplog.records)
