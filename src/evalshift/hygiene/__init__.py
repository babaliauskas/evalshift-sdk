"""Capture hygiene: sampling, dedup, and garbage collection (Phase 6).

These controls keep a busy agent's capture firehose self-managing. Each is **opt-in** and
**off by default** — with no hygiene knob set, capture behaves exactly as in Phases 0-5.

* :mod:`evalshift.hygiene.sample` — capture a fraction of agent runs (decided at agent entry).
* :mod:`evalshift.hygiene.dedup` — collapse captures with identical inputs (per-process registry).
* :mod:`evalshift.hygiene.gc` — cap a suite directory by count / age, evicting the oldest.

Stdlib only (D-deps).
"""

from __future__ import annotations
