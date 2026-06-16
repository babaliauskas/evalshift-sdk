"""The ``Redactor`` protocol and the ``redact_tree`` walker that applies one at capture time.

A redactor is any callable ``(value) -> redacted_value`` (see :class:`Redactor`). It is applied to
each payload-bearing field of every recorded span **before** serialization, so masked values flow
into both the trace events *and* the derived tool ``input_hash`` (the capture stays internally
consistent — D-1).

``redact_tree`` deliberately holds **no** ``try``/``except``: a redactor that raises must propagate
to the build-capture guard in :func:`evalshift.config.active_sink`'s caller, dropping the whole
capture (fail-closed — we never write a possibly-unredacted file).

``runtime_checkable`` so callers can ``isinstance(x, Redactor)`` (mirrors the ``Sink`` protocol).

Stdlib only (D-deps).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from evalshift.capture.span import SpanTree


@runtime_checkable
class Redactor(Protocol):
    def __call__(self, value: Any) -> Any: ...


#: Per span kind, the ``span.data`` fields that may carry user content and must be redacted.
#: Structural fields (``name``, ``model_id``, token counts, …) are intentionally excluded.
_REDACTABLE_FIELDS: dict[str, tuple[str, ...]] = {
    "tool": ("arguments", "result", "error"),
    "model_call": ("input", "output"),
    "retrieval": ("query", "documents"),
    "guardrail": ("reason",),
    "final_output": ("text",),
    "error": ("message",),
}


def redact_tree(tree: SpanTree, redactor: Redactor) -> None:
    """Replace each redactable ``span.data`` field in place with ``redactor(value)``.

    Runs before serialization. No exception handling here on purpose: a raising redactor must
    bubble up so the capture is dropped rather than written unredacted (fail-closed).
    """
    for span in tree:
        for field_name in _REDACTABLE_FIELDS.get(span.kind, ()):
            if field_name in span.data:
                span.data[field_name] = redactor(span.data[field_name])
