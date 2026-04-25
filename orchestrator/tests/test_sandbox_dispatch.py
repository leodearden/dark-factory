"""Tests for sandbox_dispatch.set_backend() input contract.

Two test classes:
- TestSetBackendAcceptsValidValues: one test per Backend Literal member;
  these pass with the current (permissive) implementation.
- TestSetBackendRejectsInvalidInput: five tests asserting TypeError for bad
  inputs; these FAIL before the validator is added in step-2.

Module-local autouse fixture _restore_backend snapshots and restores
_preferred around each test so this file is independent of the
project-wide _reset_sandbox_backend autouse fixture (which step-3 retires).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from orchestrator.agents import sandbox_dispatch


@pytest.fixture(autouse=True)
def _restore_backend():
    """Snapshot and restore sandbox_dispatch._preferred around every test."""
    saved = sandbox_dispatch.get_backend()
    yield
    sandbox_dispatch.set_backend(saved)


class TestSetBackendAcceptsValidValues:
    """set_backend() must accept every member of the Backend Literal."""

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


class TestSetBackendRejectsInvalidInput:
    """set_backend() must raise TypeError for any non-Literal value."""

    def test_rejects_unknown_string(self):
        with pytest.raises(TypeError):
            sandbox_dispatch.set_backend('docker')  # type: ignore[arg-type]

    def test_rejects_empty_string(self):
        with pytest.raises(TypeError):
            sandbox_dispatch.set_backend('')  # type: ignore[arg-type]

    def test_rejects_magicmock(self):
        with pytest.raises(TypeError):
            sandbox_dispatch.set_backend(MagicMock())  # type: ignore[arg-type]

    def test_rejects_int(self):
        with pytest.raises(TypeError):
            sandbox_dispatch.set_backend(42)  # type: ignore[arg-type]

    def test_rejects_none_value(self):
        with pytest.raises(TypeError):
            sandbox_dispatch.set_backend(None)  # type: ignore[arg-type]
