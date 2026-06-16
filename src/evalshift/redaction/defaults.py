"""A ready-made redactor masking common PII in strings: ``default_redactor``.

**Opt-in.** Nothing redacts unless you pass it explicitly —
``@capture.agent(suite=..., redact=default_redactor)`` or ``configure(redact=default_redactor)``.
The masks are intentionally conservative (precise patterns over greedy ones) to avoid mangling
benign content; they are not a substitute for a domain-specific redactor when your data has
structured secrets. See ``docs/REDACTION.md`` for the data-boundary guarantees.

Stdlib only (D-deps): just :mod:`re`.
"""

from __future__ import annotations

import re
from typing import Any

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_OPENAI_KEY = re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")
_AWS_KEY = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
_BEARER = re.compile(r"\bBearer\s+[A-Za-z0-9._-]+", re.IGNORECASE)


def _mask_string(text: str) -> str:
    text = _EMAIL.sub("[REDACTED_EMAIL]", text)
    text = _BEARER.sub("Bearer [REDACTED_KEY]", text)
    text = _OPENAI_KEY.sub("[REDACTED_KEY]", text)
    text = _AWS_KEY.sub("[REDACTED_KEY]", text)
    return text


def default_redactor(value: Any) -> Any:
    """Recursively mask emails / API keys in strings; pass non-strings through unchanged.

    Returns a redacted copy and never mutates the input. Containers (dict / list / tuple) are
    walked element-wise; anything that is not a str/dict/list/tuple is returned as-is.
    """
    if isinstance(value, str):
        return _mask_string(value)
    if isinstance(value, dict):
        return {key: default_redactor(item) for key, item in value.items()}
    if isinstance(value, list):
        return [default_redactor(item) for item in value]
    if isinstance(value, tuple):
        return tuple(default_redactor(item) for item in value)
    return value
