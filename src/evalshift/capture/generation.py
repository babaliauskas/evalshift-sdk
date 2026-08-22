"""Shared allow-list + JSON coercion for a model call's ``generation_config``.

Every path that stamps ``metadata["generation_config"]`` onto a ``model_call`` span — the manual
API (:func:`evalshift.capture.api.record_model_call`, ``capture.model_call(...)``,
``set_generation_config``) and the LangChain adapter — funnels through
:func:`sanitize_generation_config`. That buys two properties the verbatim ``dict(config)`` did
not:

* **The capture survives.** ``FileSink`` serializes with a plain ``json.dumps`` (no ``default=``)
  and the sink write is fail-open, so one non-serializable value (a google-genai
  ``response_schema`` *class*, say) used to raise at write time and silently drop the whole
  capture. Coercing here keeps the capture intact and the value legible as its ``str()``.
* **Nothing leaks past the redactor.** ``redact_tree`` walks only ``span.data``; ``span.metadata``
  is never redacted. Allow-listing means a user who hands us a whole ``GenerateContentConfig``
  dump cannot accidentally ship ``system_instruction`` / ``safety_settings`` into the capture.

Stdlib only (D-deps).
"""

from __future__ import annotations

from typing import Any

from evalshift import safety

#: The generation params we record. Frozen: config is public API, and every recorded key is one
#: more thing that bypasses redaction, so widening this is a deliberate decision, not a default.
GENERATION_KEYS = (
    "temperature",
    "top_p",
    "response_mime_type",
    "response_schema",
    "response_format",
    "max_output_tokens",
    "max_tokens",
)


def jsonable(value: Any) -> Any:
    """Coerce an arbitrary framework value into JSON-able primitives (objects -> ``str``).

    Primitives pass through untouched, ``dict``/``list``/``tuple`` recurse (so a nested schema
    stays real JSON rather than one opaque string), and anything else degrades to ``str(value)``.
    This mirrors the lossy ``default=str`` already used for input hashing.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return str(value)


def sanitize_generation_config(config: Any) -> dict[str, Any] | None:
    """Reduce a user-supplied generation config to the recordable allow-list.

    Args:
        config: Any value. Non-dicts are rejected outright.

    Returns:
        A new dict holding the :data:`GENERATION_KEYS` entries whose value is not ``None``, each
        JSON-coerced via :func:`jsonable`; or ``None`` when ``config`` is not a dict or nothing
        survives the allow-list (callers then record no ``generation_config`` key at all).

    Never raises and never warns; dropped keys are logged at ``debug`` on the library logger.
    """
    if not isinstance(config, dict):
        return None
    out = {key: jsonable(config[key]) for key in GENERATION_KEYS if config.get(key) is not None}
    dropped = [str(key) for key in config if key not in GENERATION_KEYS]
    if dropped:
        safety.logger.debug(
            "evalshift: dropped non-allow-listed generation_config keys: %s", ", ".join(dropped)
        )
    return out or None


__all__ = ["GENERATION_KEYS", "jsonable", "sanitize_generation_config"]
