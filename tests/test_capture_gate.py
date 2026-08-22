"""The off-by-default gate: nothing is written unless capture is enabled."""

from __future__ import annotations

from pathlib import Path

import pytest

from evalshift import capture, record_model_call
from tests.conftest import CaptureReader


@capture.agent(suite="gate_suite", redact=False, tools=[])
def _agent(query: str) -> str:
    record_model_call(model_id="claude-opus-4-8", input=query, output="ok", tools=[])
    return f"handled:{query}"


def test_gate_off_writes_no_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVALSHIFT_CAPTURE", raising=False)
    monkeypatch.setenv("EVALSHIFT_DIR", str(tmp_path))
    result = _agent("hi")
    assert result == "handled:hi"
    assert list(tmp_path.rglob("*.json")) == []


def test_gate_off_returns_real_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVALSHIFT_CAPTURE", raising=False)
    assert _agent("x") == "handled:x"


def test_gate_on_writes_exactly_one_file(capturing: Path, read_captures: CaptureReader) -> None:
    assert _agent("hi") == "handled:hi"
    assert len(read_captures("gate_suite")) == 1
