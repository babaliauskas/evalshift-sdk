"""In-process redaction applied before any byte hits disk (D-4).

Public surface: the :class:`Redactor` protocol, the :data:`RedactSetting` type accepted by the
required ``redact=`` keyword, the :func:`resolve_redactor` normalizer the capture entry points
run, the :func:`redact_tree` walker the serializer runs, and the ready-made
:func:`default_redactor`.

Stdlib only (D-deps).
"""

from __future__ import annotations

from evalshift.redaction.base import Redactor, redact_tree
from evalshift.redaction.defaults import default_redactor
from evalshift.redaction.resolve import RedactSetting, resolve_redactor

__all__ = [
    "RedactSetting",
    "Redactor",
    "default_redactor",
    "redact_tree",
    "resolve_redactor",
]
