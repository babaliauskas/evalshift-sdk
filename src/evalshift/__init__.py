"""EvalShift capture SDK.

Import name ``evalshift`` (distribution ``evalshift-sdk``). Records agent behavior in-process
and writes CLI-valid traces to disk. Capture is off unless ``EVALSHIFT_CAPTURE=1``.

Public surface: the ``capture`` decorator, the ``record_model_call`` helper, the programmatic
``configure`` entry point, the opt-in ``default_redactor``, and the built-in
``FileSink`` / ``MemorySink``.
"""

from __future__ import annotations

from evalshift.capture.api import capture, record_model_call
from evalshift.config import configure
from evalshift.redaction import Redactor, default_redactor
from evalshift.sinks.file import FileSink
from evalshift.sinks.memory import MemorySink
from evalshift.trace.schema import SCHEMA_VERSION

__version__ = "0.1.0"

__all__ = [
    "SCHEMA_VERSION",
    "FileSink",
    "MemorySink",
    "Redactor",
    "__version__",
    "capture",
    "configure",
    "default_redactor",
    "record_model_call",
]
