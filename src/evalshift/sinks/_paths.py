"""Shared base-directory resolution for on-disk sinks (:mod:`evalshift.sinks.file`,
:mod:`evalshift.sinks.toolset`).

Both sinks anchor their writes under the same ``base`` directory -- captures under
``<base>/captures/...``, toolset sidecars under ``<base>/toolsets/...`` -- and resolve it in the
same order: an explicit constructor argument wins, else the ``EVALSHIFT_DIR`` env var, else
``.evalshift`` relative to the current working directory. There is deliberately **no** repo-root
walk -- the SDK must not assume it runs inside a checkout (containers, Lambda).

:func:`resolve_base` may return a relative ``Path``; each sink absolutizes it itself, at write
time, once its own subdirectory (``captures`` / ``toolsets``) is appended, so the write target is
unambiguous regardless of how ``base`` was specified.

Private (leading underscore): internal wiring shared between sink implementations, not part of
the SDK's public surface.

Stdlib only (D-deps).
"""

from __future__ import annotations

import os
from pathlib import Path

#: Default root when EVALSHIFT_DIR is unset (relative to CWD; no repo-root walk).
DEFAULT_BASE = ".evalshift"


def resolve_base(base: str | os.PathLike[str] | None) -> Path:
    """Resolve a sink's base directory: explicit ``base``, else ``EVALSHIFT_DIR``, else
    :data:`DEFAULT_BASE`. May return a relative path -- the caller absolutizes it.
    """
    if base is not None:
        return Path(base)
    return Path(os.environ.get("EVALSHIFT_DIR", DEFAULT_BASE))


__all__ = ["DEFAULT_BASE", "resolve_base"]
