"""End-to-end capture sampling across all four agent entry points (Phase 6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from evalshift import capture, configure, record_model_call
from evalshift.capture import state
from tests.conftest import CaptureReader


def test_sample_rate_zero_writes_nothing_sync(
    capturing: Path, read_captures: CaptureReader
) -> None:
    configure(sample_rate=0.0)

    @capture.agent(suite="samp")
    def handle(query: str) -> str:
        return f"ok:{query}"

    assert handle("hi") == "ok:hi"  # host result unaffected
    assert read_captures("samp") == []


async def test_sample_rate_zero_writes_nothing_async(
    capturing: Path, read_captures: CaptureReader
) -> None:
    configure(sample_rate=0.0)

    @capture.agent(suite="samp")
    async def handle(query: str) -> str:
        return f"ok:{query}"

    assert await handle("hi") == "ok:hi"
    assert read_captures("samp") == []


def test_sample_rate_zero_session_yields_none(
    capturing: Path, read_captures: CaptureReader
) -> None:
    configure(sample_rate=0.0)
    with capture.agent_session(suite="samp") as tree:
        assert tree is None
    assert read_captures("samp") == []


async def test_sample_rate_zero_async_session_yields_none(
    capturing: Path, read_captures: CaptureReader
) -> None:
    configure(sample_rate=0.0)
    async with capture.agent_session_async(suite="samp") as tree:
        assert tree is None
    assert read_captures("samp") == []


def test_sample_rate_one_writes_one(capturing: Path, read_captures: CaptureReader) -> None:
    configure(sample_rate=1.0)

    @capture.agent(suite="samp")
    def handle() -> str:
        return "ok"

    handle()
    assert len(read_captures("samp")) == 1


def test_mid_rate_keeps_when_draw_below(
    capturing: Path, read_captures: CaptureReader, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("random.random", lambda: 0.1)
    configure(sample_rate=0.5)

    @capture.agent(suite="samp")
    def handle() -> str:
        return "ok"

    handle()
    assert len(read_captures("samp")) == 1


def test_mid_rate_drops_when_draw_above(
    capturing: Path, read_captures: CaptureReader, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("random.random", lambda: 0.9)
    configure(sample_rate=0.5)

    @capture.agent(suite="samp")
    def handle() -> str:
        return "ok"

    handle()
    assert read_captures("samp") == []


def test_unsampled_run_is_true_passthrough(capturing: Path, read_captures: CaptureReader) -> None:
    configure(sample_rate=0.0)
    observed: dict[str, object] = {}

    @capture.tool
    def mytool() -> str:
        observed["tree"] = state.current_tree()
        return "t"

    @capture.agent(suite="samp")
    def handle() -> str:
        record_model_call(model_id="m")
        return mytool()

    assert handle() == "t"
    assert observed["tree"] is None  # no session opened -> tools/model calls no-op
    assert read_captures("samp") == []


def test_sampling_fault_still_captures(
    capturing: Path, read_captures: CaptureReader, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom() -> float:
        raise RuntimeError("rng broke")

    monkeypatch.setattr("random.random", boom)
    configure(sample_rate=0.5)

    @capture.agent(suite="samp")
    def handle() -> str:
        return "ok"

    handle()
    assert len(read_captures("samp")) == 1
