"""Capture garbage collection: cap a suite directory by count and/or age.

:func:`evict` runs after a successful disk write, on the suite directory just written to. It does
a single :func:`os.scandir` pass and orders by **filesystem mtime** (not the envelope's
``created_at``) — stat only, no JSON parsing, so it stays cheap even when ``max_captures`` is
large. Two rules combine (union):

* **count** — keep the newest ``max_captures`` ``*.json`` files, evict the rest;
* **TTL** — evict any ``*.json`` whose age (``now_ts - mtime``) exceeds ``capture_ttl_seconds``.

Each unlink is individually swallowed, and the whole scan is guarded, so eviction never raises
into the host. Limitation (``docs/DECISIONS.md`` D-6): tools that rewrite mtimes (``cp -p``,
rsync, tar-extract) can perturb eviction order.

Stdlib only (D-deps).
"""

from __future__ import annotations

import os
from pathlib import Path

from evalshift.safety import logger


def evict(
    directory: Path,
    *,
    max_captures: int | None,
    capture_ttl_seconds: float | None,
    now_ts: float,
) -> list[Path]:
    """Evict aged / overflow ``*.json`` captures from ``directory``; return the paths removed.

    No-op (returns ``[]``) when neither threshold is set or the directory is unreadable. Never
    recurses into subdirectories; never raises into the caller.
    """
    if max_captures is None and capture_ttl_seconds is None:
        return []
    try:
        entries = [
            (entry.stat().st_mtime, Path(entry.path))
            for entry in os.scandir(directory)
            if entry.is_file() and entry.name.endswith(".json")
        ]
    except OSError:
        logger.debug("evalshift: gc scan of %s failed (skipped)", directory, exc_info=True)
        return []

    entries.sort(key=lambda item: item[0])  # oldest first
    doomed: set[Path] = set()

    if capture_ttl_seconds is not None:
        doomed.update(path for mtime, path in entries if now_ts - mtime > capture_ttl_seconds)

    if max_captures is not None and len(entries) > max_captures:
        keep = {path for _, path in entries[len(entries) - max_captures :]}
        doomed.update(path for _, path in entries if path not in keep)

    removed: list[Path] = []
    for _, path in entries:  # oldest-first, for a stable removal order
        if path not in doomed:
            continue
        try:
            path.unlink(missing_ok=True)
            removed.append(path)
        except OSError:
            logger.debug("evalshift: gc unlink of %s failed (skipped)", path, exc_info=True)
    return removed


__all__ = ["evict"]
