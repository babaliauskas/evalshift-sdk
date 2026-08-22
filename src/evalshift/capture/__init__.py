"""Capture-time machinery: the live recording structures.

Phase 1 ships only the span tree (``span.py``). The public ``capture`` decorator, contextvars
state, and helpers (``api.py`` / ``state.py``) land in Phase 2.
"""

from __future__ import annotations

from evalshift.capture.span import Span, SpanTree

__all__ = ["Span", "SpanTree"]
