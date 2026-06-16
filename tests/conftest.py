"""Shared fixtures for capture tests."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from evalshift.config import reset_config

CaptureReader = Callable[[str], list[dict[str, Any]]]


@pytest.fixture(autouse=True)
def _isolate_config() -> Iterator[None]:
    """Reset the process-wide capture config around every test (global state must not leak)."""
    reset_config()
    yield
    reset_config()


@pytest.fixture
def capturing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Enable the capture gate and route writes into ``tmp_path``. Returns the capture base dir."""
    monkeypatch.setenv("EVALSHIFT_CAPTURE", "1")
    monkeypatch.setenv("EVALSHIFT_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def read_captures(tmp_path: Path) -> CaptureReader:
    """Return a reader: ``read_captures(suite)`` -> the parsed capture dicts for that suite."""

    def _read(suite: str) -> list[dict[str, Any]]:
        suite_dir = tmp_path / "captures" / suite
        if not suite_dir.exists():
            return []
        return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(suite_dir.glob("*.json"))]

    return _read
