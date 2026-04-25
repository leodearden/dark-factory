"""Unit tests for sandbox_dispatch.set_backend() input contract.

Pins the validation contract of set_backend():
- valid Backend Literal values are accepted and round-trip through get_backend()
- unknown-but-correctly-typed strings (e.g. 'docker') raise ValueError
- non-string inputs (MagicMock, int, None) raise TypeError

A module-scoped autouse fixture saves/restores _preferred around each test so
these tests are fully isolated without relying on any project-wide autouse
fixture.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from orchestrator.agents import sandbox_dispatch


@pytest.fixture(autouse=True)
def _restore_backend():
    """Snapshot _preferred before each test and restore it after."""
    saved = sandbox_dispatch.get_backend()
    yield
    sandbox_dispatch.set_backend(saved)


# ---------------------------------------------------------------------------
# Accept tests — each valid Literal value must be accepted and returned by
# get_backend() immediately after the call.
# ---------------------------------------------------------------------------

class TestSetBackendAcceptsValidValues:
    def test_accepts_auto(self):
        sandbox_dispatch.set_backend('auto')
        assert sandbox_dispatch.get_backend() == 'auto'

    def test_accepts_bwrap(self):
        sandbox_dispatch.set_backend('bwrap')
        assert sandbox_dispatch.get_backend() == 'bwrap'

    def test_accepts_landlock(self):
        sandbox_dispatch.set_backend('landlock')
        assert sandbox_dispatch.get_backend() == 'landlock'

    def test_accepts_none(self):
        sandbox_dispatch.set_backend('none')
        assert sandbox_dispatch.get_backend() == 'none'


# ---------------------------------------------------------------------------
# Reject tests — invalid inputs must raise the appropriate exception.
# Wrong-value-but-correct-type strings raise ValueError; non-strings raise TypeError.
# ---------------------------------------------------------------------------

class TestSetBackendRejectsInvalidInput:
    def test_rejects_docker_string(self):
        with pytest.raises(ValueError):
            sandbox_dispatch.set_backend('docker')  # type: ignore[arg-type]

    def test_rejects_empty_string(self):
        with pytest.raises(ValueError):
            sandbox_dispatch.set_backend('')  # type: ignore[arg-type]

    def test_rejects_magicmock(self):
        with pytest.raises(TypeError):
            sandbox_dispatch.set_backend(MagicMock())  # type: ignore[arg-type]

    def test_rejects_int(self):
        with pytest.raises(TypeError):
            sandbox_dispatch.set_backend(42)  # type: ignore[arg-type]

    def test_rejects_none(self):
        with pytest.raises(TypeError):
            sandbox_dispatch.set_backend(None)  # type: ignore[arg-type]
