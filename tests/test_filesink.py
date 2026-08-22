"""Unit tests for the basic FileSink (Phase 2 minimal write path)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalshift.capture.span import SpanTree
from evalshift.sinks.file import FileSink
from evalshift.trace.models import CaptureEnvelope
from evalshift.trace.serialize import build_capture, envelope_to_dict


def _envelope(suite: str = "support_agent", capture_id: str = "cap_abc") -> CaptureEnvelope:
    return build_capture(SpanTree(), suite=suite, agent_input="hi", capture_id=capture_id)


def test_explicit_base_path_layout(tmp_path: Path) -> None:
    path = FileSink(base=tmp_path).write(_envelope())
    assert path == tmp_path / "captures" / "support_agent" / "cap_abc.json"
    assert path is not None
    assert path.exists()


def test_base_resolves_from_env_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVALSHIFT_DIR", str(tmp_path))
    path = FileSink().write(_envelope())
    assert path == tmp_path / "captures" / "support_agent" / "cap_abc.json"


def test_base_defaults_to_dot_evalshift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVALSHIFT_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    path = FileSink().write(_envelope())
    assert path == tmp_path / ".evalshift" / "captures" / "support_agent" / "cap_abc.json"


def test_explicit_base_overrides_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVALSHIFT_DIR", str(tmp_path / "env"))
    path = FileSink(base=tmp_path / "explicit").write(_envelope())
    assert path == tmp_path / "explicit" / "captures" / "support_agent" / "cap_abc.json"


def test_write_creates_missing_dirs(tmp_path: Path) -> None:
    base = tmp_path / "deep" / "nested"
    path = FileSink(base=base).write(_envelope())
    assert path is not None
    assert path.exists()


def test_written_json_round_trips(tmp_path: Path) -> None:
    env = _envelope()
    path = FileSink(base=tmp_path).write(env)
    assert path is not None
    assert json.loads(path.read_text(encoding="utf-8")) == envelope_to_dict(env)


def test_suite_path_traversal_stays_under_base(tmp_path: Path) -> None:
    base = tmp_path / "root"
    path = FileSink(base=base).write(_envelope(suite="../escape"))
    assert path is not None
    assert path.is_relative_to(base.absolute())
    assert ".." not in path.parts


def test_write_degrades_to_noop_on_oserror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: object, **_kwargs: object) -> int:
        raise OSError("read-only file system")

    monkeypatch.setattr(Path, "write_text", boom)
    path = FileSink(base=tmp_path).write(_envelope())
    assert path is None
    assert list(tmp_path.rglob("*.json")) == []
