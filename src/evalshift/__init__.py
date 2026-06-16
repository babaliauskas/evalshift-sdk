"""EvalShift capture SDK.

Import name ``evalshift`` (distribution ``evalshift-sdk``). Records agent behavior in-process
and writes CLI-valid traces to disk. Capture is off unless ``EVALSHIFT_CAPTURE=1``.

The public ``capture`` / ``configure`` surface is added in later phases; Phase 0 freezes the
schema only.
"""

from __future__ import annotations

from evalshift.trace.schema import SCHEMA_VERSION

__version__ = "0.1.0"

__all__ = ["SCHEMA_VERSION", "__version__"]
