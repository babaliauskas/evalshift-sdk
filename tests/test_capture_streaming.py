"""Streaming model-call capture via capture.model_call: accumulate, record once (Phase 5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from evalshift import capture, record_model_call
from evalshift.capture import api
from tests.conftest import CaptureReader


def _events(capture_dict: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = capture_dict["trace"]["events"]
    return events


def _model_calls(capture_dict: dict[str, Any]) -> list[dict[str, Any]]:
    return [e for e in _events(capture_dict) if e["type"] == "model_call"]


def test_streaming_model_call_records_once(capturing: Path, read_captures: CaptureReader) -> None:
    @capture.agent(suite="stream_suite")
    def agent() -> None:
        with capture.model_call(model_id="claude-opus-4-8", input="prompt") as mc:
            for chunk in ("a", "b", "c"):
                mc.add_text(chunk)
            mc.set_usage(input_tokens=5, output_tokens=3, cost_usd=0.01)

    agent()
    calls = _model_calls(read_captures("stream_suite")[0])
    assert len(calls) == 1
    assert calls[0]["model_id"] == "claude-opus-4-8"
    assert calls[0]["output"] == "abc"
    assert calls[0]["input_tokens"] == 5
    assert calls[0]["output_tokens"] == 3
    assert calls[0]["cost_usd"] == 0.01


async def test_async_streaming_model_call(capturing: Path, read_captures: CaptureReader) -> None:
    async def fake_stream() -> Any:
        for chunk in ("x", "y", "z"):
            yield chunk

    @capture.agent(suite="stream_suite")
    async def agent() -> None:
        async with capture.model_call(model_id="m", input="p") as mc:
            async for chunk in fake_stream():
                mc.add_text(chunk)
            mc.set_usage(output_tokens=3)

    await agent()
    calls = _model_calls(read_captures("stream_suite")[0])
    assert len(calls) == 1
    assert calls[0]["output"] == "xyz"
    assert calls[0]["output_tokens"] == 3


def test_streaming_without_set_usage_defaults_and_derives_latency(
    capturing: Path, read_captures: CaptureReader, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Clock:
        def __init__(self) -> None:
            self.t = 0.0

        def __call__(self) -> float:
            self.t += 0.5
            return self.t

    monkeypatch.setattr(api, "_now", _Clock())

    @capture.agent(suite="stream_suite")
    def agent() -> None:
        with capture.model_call(model_id="m", input="p") as mc:
            mc.add_text("hello")

    agent()
    call = _model_calls(read_captures("stream_suite")[0])[0]
    assert call["input_tokens"] == 0
    assert call["output_tokens"] == 0
    assert call["cost_usd"] == 0.0
    # latency derived from (end_ts - start_ts): 0.5s opened, 1.0s closed -> 500ms.
    assert call["latency_ms"] == 500


def test_streaming_start_end_ts_reflect_enter_exit(
    capturing: Path, read_captures: CaptureReader, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Clock:
        def __init__(self) -> None:
            self.t = 0.0

        def __call__(self) -> float:
            self.t += 1.0
            return self.t

    monkeypatch.setattr(api, "_now", _Clock())

    @capture.agent(suite="stream_suite")
    def agent() -> None:
        with capture.model_call(model_id="m", input="p") as mc:
            mc.add_text("hi")

    agent()
    call = _model_calls(read_captures("stream_suite")[0])[0]
    block = call["metadata"]["evalshift"]
    # Enter stamps start_ts (1.0), exit stamps end_ts (2.0) — not a single instant.
    assert block["start_ts"] == 1.0
    assert block["end_ts"] == 2.0


def test_streaming_model_call_noop_outside_session(tmp_path: Path) -> None:
    with capture.model_call(model_id="m", input="p") as mc:
        mc.add_text("ignored")
        mc.set_usage(output_tokens=99)
    assert list(tmp_path.rglob("*.json")) == []


def test_record_model_call_still_atomic(capturing: Path, read_captures: CaptureReader) -> None:
    # Regression: the atomic helper is unchanged and records a single-instant model_call.
    @capture.agent(suite="stream_suite")
    def agent() -> None:
        record_model_call(model_id="m", input="p", output="done", output_tokens=2)

    agent()
    calls = _model_calls(read_captures("stream_suite")[0])
    assert len(calls) == 1
    assert calls[0]["output"] == "done"
    assert calls[0]["output_tokens"] == 2
