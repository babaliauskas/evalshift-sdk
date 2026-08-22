"""Normalize the public ``redact=`` setting into the internal ``Redactor | None`` (D-4c).

``redact`` is a **required** keyword on every capture entry point: masking is an explicit choice,
never a default. The public type is :data:`RedactSetting` — a bool for the two common answers, or a
callable for a domain-specific redactor:

===============  ===========================  ==================================================
passed           resolves to                  effect
===============  ===========================  ==================================================
``True``         :func:`default_redactor`     masks emails, ``sk-…``, ``AKIA…``, ``Bearer …``
``False``        ``None``                     ``redact_tree`` is skipped entirely — verbatim
a callable       itself                       custom redactor
anything else    —                            :exc:`TypeError`
===============  ===========================  ==================================================

``None`` is rejected on purpose. It was the pre-0.3.0 default and is what a distracted caller (or a
stale example) reaches for to silence the error — accepting it would reopen the hole this rule
closes: an unmasked capture that looks decided but is not.

Stdlib only (D-deps).
"""

from __future__ import annotations

from evalshift.redaction.base import Redactor
from evalshift.redaction.defaults import default_redactor

#: The public ``redact=`` type: ``True`` / ``False`` / a custom redactor.
RedactSetting = Redactor | bool


def resolve_redactor(redact: RedactSetting) -> Redactor | None:
    """Map a public ``redact=`` setting onto the internal redactor (``None`` == capture verbatim).

    Raises :exc:`TypeError` for any other value, including ``None``. Callers invoke this at the
    capture point *before* the ``EVALSHIFT_CAPTURE`` gate is consulted, so a wrong value fails the
    same way whether or not capture is enabled.
    """
    if redact is True:
        return default_redactor
    if redact is False:
        return None
    if callable(redact):
        return redact
    raise TypeError(
        "redact= is required and must be True (mask emails and API keys), False (capture "
        f"verbatim), or a callable (value) -> value; got {redact!r}"
    )
