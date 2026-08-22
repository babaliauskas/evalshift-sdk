"""Unit tests for capture garbage collection (Phase 6)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from evalshift.hygiene import gc


def _make(directory: Path, name: str, *, mtime: float) -> Path:
    path = directory / name
    path.write_text("{}", encoding="utf-8")
    os.utime(path, (mtime, mtime))
    return path


def test_count_eviction_keeps_newest_n(tmp_path: Path) -> None:
    files = [_make(tmp_path, f"cap_{i}.json", mtime=100.0 + i) for i in range(5)]
    removed = gc.evict(tmp_path, max_captures=2, capture_ttl_seconds=None, now_ts=1000.0)
    assert set(removed) == set(files[:3])  # three oldest gone
    survivors = set(tmp_path.glob("*.json"))
    assert survivors == set(files[3:])  # two newest kept


def test_max_captures_one_keeps_only_newest(tmp_path: Path) -> None:
    old = _make(tmp_path, "cap_old.json", mtime=100.0)
    new = _make(tmp_path, "cap_new.json", mtime=200.0)
    removed = gc.evict(tmp_path, max_captures=1, capture_ttl_seconds=None, now_ts=1000.0)
    assert removed == [old]
    assert new.exists()


def test_ttl_eviction_removes_aged_files(tmp_path: Path) -> None:
    fresh = _make(tmp_path, "cap_fresh.json", mtime=990.0)
    stale = _make(tmp_path, "cap_stale.json", mtime=100.0)
    removed = gc.evict(tmp_path, max_captures=None, capture_ttl_seconds=60.0, now_ts=1000.0)
    assert removed == [stale]  # age 900 > ttl 60
    assert fresh.exists()  # age 10 < ttl 60


def test_count_and_ttl_rules_union(tmp_path: Path) -> None:
    files = [_make(tmp_path, f"cap_{i}.json", mtime=m) for i, m in enumerate([100.0, 980.0, 995.0])]
    # max_captures=2 would drop the oldest (files[0]); ttl=60 also drops files[0] (age 900).
    removed = gc.evict(tmp_path, max_captures=2, capture_ttl_seconds=60.0, now_ts=1000.0)
    assert set(removed) == {files[0]}
    assert {files[1], files[2]} == set(tmp_path.glob("*.json"))


def test_only_json_files_are_touched(tmp_path: Path) -> None:
    keep_txt = _make(tmp_path, "notes.txt", mtime=100.0)
    old_json = _make(tmp_path, "cap_old.json", mtime=100.0)
    new_json = _make(tmp_path, "cap_new.json", mtime=200.0)
    removed = gc.evict(tmp_path, max_captures=1, capture_ttl_seconds=None, now_ts=1000.0)
    assert removed == [old_json]
    assert keep_txt.exists()  # non-json never considered
    assert new_json.exists()


def test_no_thresholds_evicts_nothing(tmp_path: Path) -> None:
    _make(tmp_path, "cap_a.json", mtime=100.0)
    removed = gc.evict(tmp_path, max_captures=None, capture_ttl_seconds=None, now_ts=1000.0)
    assert removed == []
    assert list(tmp_path.glob("*.json"))


def test_empty_dir_returns_empty(tmp_path: Path) -> None:
    assert gc.evict(tmp_path, max_captures=2, capture_ttl_seconds=60.0, now_ts=1000.0) == []


def test_missing_dir_does_not_raise(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    assert gc.evict(missing, max_captures=1, capture_ttl_seconds=None, now_ts=1000.0) == []


def test_gc_never_touches_a_sibling_toolsets_directory(tmp_path: Path) -> None:
    """Regression for D-toolset: ``evict`` is called with ``<base>/captures/<suite>/`` (see
    ``sinks/hygiene.py``), a sibling of ``<base>/toolsets/`` -- never that directory itself or an
    ancestor of it -- and ``evict`` never recurses into subdirectories (module docstring). Proves
    both halves structurally: pointing ``evict`` at a captures/<suite> dir leaves a toolsets/ dir
    one level up completely untouched, even when it contains files that would sort as "oldest".
    """
    base = tmp_path
    captures_suite_dir = base / "captures" / "demo"
    captures_suite_dir.mkdir(parents=True)
    toolsets_dir = base / "toolsets"
    toolsets_dir.mkdir()

    sidecar = _make(toolsets_dir, "deadbeef.json", mtime=1.0)  # oldest possible mtime
    _make(captures_suite_dir, "cap_a.json", mtime=100.0)
    newer = _make(captures_suite_dir, "cap_b.json", mtime=200.0)

    removed = gc.evict(captures_suite_dir, max_captures=1, capture_ttl_seconds=None, now_ts=1000.0)

    assert removed == [captures_suite_dir / "cap_a.json"]
    assert newer.exists()
    assert sidecar.exists()  # never scanned: not in captures_suite_dir, not recursed into
    assert list(toolsets_dir.glob("*.json")) == [sidecar]


def test_unlink_failure_on_one_file_does_not_abort_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a = _make(tmp_path, "cap_a.json", mtime=100.0)  # oldest
    b = _make(tmp_path, "cap_b.json", mtime=110.0)
    _make(tmp_path, "cap_c.json", mtime=200.0)  # newest, kept

    real_unlink = Path.unlink

    def flaky_unlink(self: Path, missing_ok: bool = False) -> None:
        if self.name == "cap_a.json":
            raise PermissionError("locked")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)
    removed = gc.evict(tmp_path, max_captures=1, capture_ttl_seconds=None, now_ts=1000.0)
    assert removed == [b]  # b removed despite a failing
    assert a.exists()  # the failing one survives, no exception raised
