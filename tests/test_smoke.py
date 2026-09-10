"""Smoke test: the package imports and exposes the frozen schema version."""

from __future__ import annotations

import evalshift
from evalshift.trace import schema


def test_package_exposes_version() -> None:
    assert isinstance(evalshift.__version__, str)
    assert evalshift.__version__


def test_schema_version_frozen() -> None:
    assert evalshift.SCHEMA_VERSION == "2.1.0"
    assert schema.SCHEMA_VERSION == "2.1.0"
