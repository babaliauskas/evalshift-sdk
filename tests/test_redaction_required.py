"""``redact=`` is a required, explicit choice at every capture point (D-4c).

Covers the resolver (``Redactor | bool`` -> ``Redactor | None``), the four public entry points
that now demand it, and the removal of the process-wide ``configure(redact=...)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from evalshift import capture, configure, record_model_call
from evalshift.redaction import default_redactor, resolve_redactor
from tests.conftest import CaptureReader


def _events(capture_dict: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = capture_dict["trace"]["events"]
    return events


# --- the resolver -------------------------------------------------------------------------


def test_true_resolves_to_default_redactor() -> None:
    assert resolve_redactor(True) is default_redactor


def test_false_resolves_to_none() -> None:
    """``False`` means "skip ``redact_tree`` entirely", not "run an identity redactor"."""
    assert resolve_redactor(False) is None


def test_callable_resolves_to_itself() -> None:
    def custom(value: Any) -> Any:
        return value

    assert resolve_redactor(custom) is custom


def test_none_is_rejected() -> None:
    """``None`` was the 0.2.0 default: it must fail loudly, not read as "no redactor"."""
    with pytest.raises(TypeError) as excinfo:
        resolve_redactor(None)  # type: ignore[arg-type]
    message = str(excinfo.value)
    assert "True" in message
    assert "False" in message
    assert "callable" in message


def test_non_callable_non_bool_is_rejected() -> None:
    with pytest.raises(TypeError):
        resolve_redactor("default")  # type: ignore[arg-type]


# --- the entry points demand it -----------------------------------------------------------


def test_agent_decorator_requires_redact() -> None:
    with pytest.raises(TypeError):
        capture.agent(suite="req", tools=[])  # type: ignore[call-arg]


def test_agent_session_requires_redact() -> None:
    with pytest.raises(TypeError), capture.agent_session(suite="req", tools=[]):  # type: ignore[call-arg]
        pass


async def test_agent_session_async_requires_redact() -> None:
    with pytest.raises(TypeError):
        async with capture.agent_session_async(suite="req", tools=[]):  # type: ignore[call-arg]
            pass


def test_agent_decorator_rejects_none_redact() -> None:
    with pytest.raises(TypeError):
        capture.agent(suite="req", redact=None, tools=[])  # type: ignore[arg-type]


def test_invalid_redact_raises_even_when_gate_is_off() -> None:
    """Gate-independent: a wrong value is wrong in every environment, capture on or off."""
    with pytest.raises(TypeError):
        capture.agent(suite="req", redact="yes", tools=[])  # type: ignore[arg-type]


# --- end-to-end behaviour ------------------------------------------------------------------


def test_true_masks_like_default_redactor(capturing: Path, read_captures: CaptureReader) -> None:
    @capture.agent(suite="req", redact=True, tools=[])
    def agent() -> None:
        record_model_call(model_id="m", input="mail bob@corp.com", output="ok", tools=[])

    agent()
    model = next(e for e in _events(read_captures("req")[0]) if e["type"] == "model_call")
    assert "bob@corp.com" not in str(model["input"])
    assert "[REDACTED_EMAIL]" in str(model["input"])


def test_false_captures_verbatim(capturing: Path, read_captures: CaptureReader) -> None:
    @capture.agent(suite="req", redact=False, tools=[])
    def agent() -> None:
        record_model_call(model_id="m", input="mail bob@corp.com", output="ok", tools=[])

    agent()
    model = next(e for e in _events(read_captures("req")[0]) if e["type"] == "model_call")
    assert "bob@corp.com" in str(model["input"])


def test_session_true_masks(capturing: Path, read_captures: CaptureReader) -> None:
    with capture.agent_session(suite="req", redact=True, tools=[]):
        record_model_call(model_id="m", input="mail bob@corp.com", output="ok", tools=[])

    model = next(e for e in _events(read_captures("req")[0]) if e["type"] == "model_call")
    assert "bob@corp.com" not in str(model["input"])


async def test_session_async_true_masks(capturing: Path, read_captures: CaptureReader) -> None:
    async with capture.agent_session_async(suite="req", redact=True, tools=[]):
        record_model_call(model_id="m", input="mail bob@corp.com", output="ok", tools=[])

    model = next(e for e in _events(read_captures("req")[0]) if e["type"] == "model_call")
    assert "bob@corp.com" not in str(model["input"])


# --- the process-wide setter is gone --------------------------------------------------------


def test_configure_no_longer_accepts_redact() -> None:
    with pytest.raises(TypeError):
        configure(redact=default_redactor)  # type: ignore[call-arg]


def test_active_redactor_is_gone() -> None:
    import evalshift.config as config_module

    assert not hasattr(config_module, "active_redactor")


# --- D-4a still holds ------------------------------------------------------------------------


def test_raising_custom_redactor_still_drops_capture(
    capturing: Path, read_captures: CaptureReader
) -> None:
    """Fail-closed (D-4a) is unchanged by the required-argument shape."""

    def boom(value: Any) -> Any:
        raise RuntimeError("redactor down")

    @capture.agent(suite="req", redact=boom, tools=[])
    def agent() -> str:
        record_model_call(model_id="m", input="secret", output="ok", tools=[])
        return "host result"

    assert agent() == "host result"
    assert read_captures("req") == []
