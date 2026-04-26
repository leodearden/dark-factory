"""Tests for sandbox_dispatch.set_backend() input contract.

Two test classes:
- TestSetBackendAcceptsValidValues: one test per Backend Literal member;
  these pass with the current (permissive) implementation.
- TestSetBackendRejectsInvalidInput: five tests asserting the correct
  exception for bad inputs:
    - TypeError  for wrong-type values (not a str): MagicMock, int, None
    - ValueError for right-type-but-invalid strings: 'docker', ''
  This split lets callers distinguish 'caller passed garbage type' from
  'caller passed a typo string'.

Module-local autouse fixture _restore_backend snapshots and restores
_preferred around each test so this file is independent of the
project-wide _reset_sandbox_backend autouse fixture (which was retired
once the validator makes corruption impossible via fail-fast TypeError/
ValueError rather than silent global poisoning).
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
    """set_backend() raises TypeError for wrong-type args, ValueError for bad strings."""

    # --- string inputs (right type, wrong value) → ValueError ---

    def test_rejects_unknown_string(self):
        with pytest.raises(ValueError):
            sandbox_dispatch.set_backend('docker')  # type: ignore[arg-type]

    def test_rejects_empty_string(self):
        with pytest.raises(ValueError):
            sandbox_dispatch.set_backend('')  # type: ignore[arg-type]

    # --- non-string inputs (wrong type entirely) → TypeError ---

    def test_rejects_magicmock(self):
        with pytest.raises(TypeError):
            sandbox_dispatch.set_backend(MagicMock())  # type: ignore[arg-type]

    def test_rejects_int(self):
        with pytest.raises(TypeError):
            sandbox_dispatch.set_backend(42)  # type: ignore[arg-type]

    def test_rejects_none_value(self):
        with pytest.raises(TypeError):
            sandbox_dispatch.set_backend(None)  # type: ignore[arg-type]


class TestResolveActiveBackendCorruptionGuard:
    """resolve_active_backend / wrap_command must fail loudly on corrupted _preferred.

    set_backend's input validator blocks corrupt values from entering via the
    public API, so reaching the unknown-backend branch in either function
    means _preferred was mutated directly past the validator. Pre-fix this
    silently fell through to ``return 'none'`` / ``return inner_cmd``,
    disabling sandboxing on the actual agent run. Post-fix it raises.
    """

    def test_resolve_active_backend_raises_on_corrupted_preferred(self):
        sandbox_dispatch._preferred = 'docker'  # type: ignore[assignment]
        with pytest.raises(RuntimeError, match='_preferred is corrupted'):
            sandbox_dispatch.resolve_active_backend()

    def test_wrap_command_raises_on_corrupted_preferred(self, tmp_path):
        from pathlib import Path
        sandbox_dispatch._preferred = 'docker'  # type: ignore[assignment]
        with pytest.raises(RuntimeError, match='_preferred is corrupted'):
            sandbox_dispatch.wrap_command(
                inner_cmd=['claude'],
                cwd=Path(tmp_path),
                writable_modules=['mod_a'],
            )
