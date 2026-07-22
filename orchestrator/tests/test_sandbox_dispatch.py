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

import logging
from unittest.mock import MagicMock, patch

import pytest

from orchestrator.agents import sandbox_dispatch


@pytest.fixture(autouse=True)
def _restore_backend():
    """Snapshot and restore sandbox_dispatch._preferred around every test."""
    saved = sandbox_dispatch.get_backend()
    yield
    sandbox_dispatch.set_backend(saved)


@pytest.fixture(autouse=True)
def _reset_dedup():
    """Reset the process-global escalation-dedup state around every test (task 2908).

    ``resolve_backend_or_refuse()`` keeps a module-global
    ``_escalated_backend_state`` so the fail-closed escalation fires exactly
    once across N refused invocations (INV-4). Clear it before and after each
    test so the dedup decision never leaks between tests. Uses ``setattr`` so
    the fixture is inert on the RED steps before the global exists.
    """
    setattr(sandbox_dispatch, '_escalated_backend_state', None)
    yield
    setattr(sandbox_dispatch, '_escalated_backend_state', None)


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


class TestResolveBackendOrRefuse:
    """resolve_backend_or_refuse(): fail-CLOSED single-call resolution (task 2908).

    Mirrors recon's RemediationSandboxUnavailable (task 1935): when sandboxing
    is required (config.sandbox.enabled + role.sandboxed) but no OS backend
    resolves, REFUSE rather than silently running unconfined. The one exception
    is the explicit operator escape hatch: backend='none' runs UNSANDBOXED with
    a WARNING and never refuses (same effect as enabled:false, PRD D4).

    The landlock/bwrap availability probes are imported at call-time inside
    resolve_active_backend(), so patching the SOURCE module attribute
    (orchestrator.agents.landlock.is_landlock_available) drives resolution.
    """

    def test_landlock_available_returns_landlock(self):
        sandbox_dispatch.set_backend('landlock')
        with patch(
            'orchestrator.agents.landlock.is_landlock_available',
            return_value=True,
        ):
            assert sandbox_dispatch.resolve_backend_or_refuse() == 'landlock'

    def test_landlock_unavailable_refuses(self):
        sandbox_dispatch.set_backend('landlock')
        with patch(
            'orchestrator.agents.landlock.is_landlock_available',
            return_value=False,
        ):
            with pytest.raises(sandbox_dispatch.SandboxUnavailable) as excinfo:
                sandbox_dispatch.resolve_backend_or_refuse()
        assert excinfo.value.should_escalate is True
        assert excinfo.value.backend_state == 'landlock'

    def test_backend_none_returns_none_with_warning(self, caplog):
        """backend='none' is the operator escape hatch: run unsandboxed, WARN, never refuse."""
        sandbox_dispatch.set_backend('none')
        with caplog.at_level(
            logging.WARNING, logger='orchestrator.agents.sandbox_dispatch',
        ):
            result = sandbox_dispatch.resolve_backend_or_refuse()
        assert result == 'none'
        assert any(r.levelno == logging.WARNING for r in caplog.records)
        assert 'unsandboxed' in caplog.text.lower()

    def test_auto_no_backend_available_refuses(self):
        sandbox_dispatch.set_backend('auto')
        with (
            patch(
                'orchestrator.agents.landlock.is_landlock_available',
                return_value=False,
            ),
            patch(
                'orchestrator.agents.sandbox.is_bwrap_available',
                return_value=False,
            ),
        ):
            with pytest.raises(sandbox_dispatch.SandboxUnavailable) as excinfo:
                sandbox_dispatch.resolve_backend_or_refuse()
        assert excinfo.value.backend_state == 'auto'


class TestResolveBackendDedup:
    """Process-global escalation dedup + re-arm (task 2908, INV-4).

    ``resolve_backend_or_refuse()`` must signal escalate exactly ONCE across N
    consecutive refusals for a given backend state, and re-arm (signal again)
    whenever the backend state changes — a recovery (backend becomes
    available), a config change (set_backend to a different backend), or the
    operator 'none' escape hatch. The dedup lives in a process-global keyed on
    the configured backend, re-armed on ANY non-refusing return.
    """

    def test_dedup_across_n_refusals(self):
        """Exactly one should_escalate=True across N (>=3) refused invocations."""
        sandbox_dispatch.set_backend('landlock')
        flags: list[bool] = []
        with patch(
            'orchestrator.agents.landlock.is_landlock_available',
            return_value=False,
        ):
            for _ in range(4):
                with pytest.raises(sandbox_dispatch.SandboxUnavailable) as excinfo:
                    sandbox_dispatch.resolve_backend_or_refuse()
                flags.append(excinfo.value.should_escalate)
        assert flags[0] is True
        assert all(flag is False for flag in flags[1:]), (
            f'exactly one escalate-signal expected across N refusals; got {flags}'
        )

    def test_rearm_on_recovery(self):
        """A healthy return between two breaks re-arms the escalate signal."""
        sandbox_dispatch.set_backend('landlock')
        with patch(
            'orchestrator.agents.landlock.is_landlock_available',
        ) as mock_ll:
            mock_ll.return_value = False
            with pytest.raises(sandbox_dispatch.SandboxUnavailable) as first:
                sandbox_dispatch.resolve_backend_or_refuse()
            assert first.value.should_escalate is True

            # Recovery: backend becomes available -> healthy return re-arms.
            mock_ll.return_value = True
            assert sandbox_dispatch.resolve_backend_or_refuse() == 'landlock'

            # Re-break: escalate again.
            mock_ll.return_value = False
            with pytest.raises(sandbox_dispatch.SandboxUnavailable) as third:
                sandbox_dispatch.resolve_backend_or_refuse()
            assert third.value.should_escalate is True

    def test_rearm_on_backend_state_change(self):
        """Changing the configured backend (landlock->bwrap) re-arms the signal."""
        sandbox_dispatch.set_backend('landlock')
        with patch(
            'orchestrator.agents.landlock.is_landlock_available',
            return_value=False,
        ):
            with pytest.raises(sandbox_dispatch.SandboxUnavailable) as first:
                sandbox_dispatch.resolve_backend_or_refuse()
            assert first.value.should_escalate is True

        sandbox_dispatch.set_backend('bwrap')
        with patch(
            'orchestrator.agents.sandbox.is_bwrap_available',
            return_value=False,
        ):
            with pytest.raises(sandbox_dispatch.SandboxUnavailable) as second:
                sandbox_dispatch.resolve_backend_or_refuse()
        assert second.value.should_escalate is True
        assert second.value.backend_state == 'bwrap'

    def test_none_escape_path_rearms(self):
        """The 'none' escape hatch is a non-refusing return -> re-arms the signal."""
        sandbox_dispatch.set_backend('landlock')
        with patch(
            'orchestrator.agents.landlock.is_landlock_available',
            return_value=False,
        ):
            with pytest.raises(sandbox_dispatch.SandboxUnavailable) as first:
                sandbox_dispatch.resolve_backend_or_refuse()
            assert first.value.should_escalate is True

        # 'none' escape hatch returns without refusing -> re-arms dedup.
        sandbox_dispatch.set_backend('none')
        assert sandbox_dispatch.resolve_backend_or_refuse() == 'none'

        # Re-break under landlock: escalate again.
        sandbox_dispatch.set_backend('landlock')
        with patch(
            'orchestrator.agents.landlock.is_landlock_available',
            return_value=False,
        ):
            with pytest.raises(sandbox_dispatch.SandboxUnavailable) as third:
                sandbox_dispatch.resolve_backend_or_refuse()
        assert third.value.should_escalate is True
