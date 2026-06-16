"""In-process redaction applied before any byte hits disk (D-4).

Public surface: the :class:`Redactor` protocol, the :func:`redact_tree` walker the serializer
runs, and the opt-in :func:`default_redactor`.
"""

from __future__ import annotations

from evalshift.redaction.base import Redactor, redact_tree
from evalshift.redaction.defaults import default_redactor

__all__ = ["Redactor", "default_redactor", "redact_tree"]
