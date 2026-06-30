"""Tests for the reconciliation sandbox guard (task 1935).

Covers:
  - ReconciliationConfig sandbox field defaults (S3/S4)
  - sandbox_guard.resolve_recon_sandbox_wrap shape and behaviour (S5/S6)
  - fail-closed and bwrap-fallback paths (S7/S8)
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fused_memory.config.schema import ReconciliationConfig

# Deferred import so tests that don't touch sandbox_guard can still run
# when the module doesn't exist yet (S3/S4 tests stay green before S6).
try:
    from fused_memory.reconciliation.sandbox_guard import (
        RemediationSandboxUnavailable,
        resolve_recon_sandbox_wrap,
    )
    _SANDBOX_GUARD_AVAILABLE = True
except ImportError:
    _SANDBOX_GUARD_AVAILABLE = False

try:
    from orchestrator.agents.landlock import is_landlock_available
    _LANDLOCK_IMPORTABLE = True
except ImportError:
    _LANDLOCK_IMPORTABLE = False
    def is_landlock_available() -> bool:  # type: ignore[misc]
        return False


# ── Config defaults (S3 / S4) ─────────────────────────────────────────────────


class TestReconciliationConfigSandboxDefaults:

    def test_sandbox_recon_agents_defaults_true(self) -> None:
        """ReconciliationConfig defaults sandbox_recon_agents to True (fail-safe on)."""
        cfg = ReconciliationConfig()
        assert cfg.sandbox_recon_agents is True

    def test_sandbox_recon_writable_extras_defaults_empty(self) -> None:
        """ReconciliationConfig defaults sandbox_recon_writable_extras to []."""
        cfg = ReconciliationConfig()
        assert cfg.sandbox_recon_writable_extras == []

    def test_sandbox_fields_round_trip(self) -> None:
        """ReconciliationConfig(sandbox_recon_agents=False, sandbox_recon_writable_extras=['/x'])
        round-trips correctly."""
        cfg = ReconciliationConfig(sandbox_recon_agents=False, sandbox_recon_writable_extras=['/x'])
        assert cfg.sandbox_recon_agents is False
        assert cfg.sandbox_recon_writable_extras == ['/x']


# ── sandbox_guard shape, enforcement, network (S5 / S6) ──────────────────────


@pytest.mark.skipif(not _SANDBOX_GUARD_AVAILABLE, reason='sandbox_guard not yet implemented')
class TestSandboxGuardLandlockBranch:
    """Landlock-branch tests for resolve_recon_sandbox_wrap."""

    def test_wrap_shape_landlock(self, tmp_path: Path) -> None:
        """resolve_recon_sandbox_wrap returns a callable whose output is a Landlock argv.

        Checks:
        - wrapped[0] == sys.executable (python interpreter)
        - wrapped[1] ends with 'landlock_exec.py'
        - '--' separator is present; tokens after '--' reproduce the inner cmd
        - writable_extras paths appear as --writable values
        - NO source/module dir (e.g. str(tmp_path/'src')) is in the --writable values
          (writable_modules=[] ⇒ no repo dir writable)
        """
        with patch(
            'orchestrator.agents.landlock.is_landlock_available',
            return_value=True,
        ):
            wrap = resolve_recon_sandbox_wrap(tmp_path, writable_extras=['/var/tmp/extra'])

        inner = ['claude', '--print']
        wrapped = wrap(inner)

        assert wrapped[0] == sys.executable, f'Expected python interpreter; got {wrapped[0]}'
        assert wrapped[1].endswith('landlock_exec.py'), (
            f'Expected landlock_exec.py as second token; got {wrapped[1]}'
        )
        assert '--' in wrapped, 'Expected -- separator in wrapped command'
        sep_idx = wrapped.index('--')
        after_sep = wrapped[sep_idx + 1:]
        assert after_sep == inner, (
            f'Expected original inner cmd after --; got {after_sep}'
        )

        # Extract --writable values
        writable_vals = [
            wrapped[i + 1] for i, tok in enumerate(wrapped) if tok == '--writable'
        ]
        assert '/var/tmp/extra' in writable_vals, (
            f'/var/tmp/extra should be in --writable values; got {writable_vals}'
        )
        # No repo source dir should be writable (writable_modules=[])
        src_dir = str(tmp_path / 'src')
        assert src_dir not in writable_vals, (
            f'Repo src dir {src_dir} must NOT be in --writable values; got {writable_vals}'
        )

    @pytest.mark.skipif(
        not is_landlock_available(),
        reason='landlock not supported on this kernel',
    )
    def test_enforcement_repo_write_denied_tmp_allowed(self, tmp_path: Path) -> None:
        """Landlock denies writes to a /var/tmp 'repo' dir but allows writes to /tmp.

        Uses /var/tmp for the simulated repo because landlock_exec grants blanket
        write access to /tmp (agent scratch). leaf-signal #1.
        """
        base = Path(tempfile.mkdtemp(prefix='recon-landlock-test-', dir='/var/tmp'))
        try:
            repo = base / 'repo'
            repo.mkdir()
            (repo / 'src').mkdir()

            nonce = uuid.uuid4().hex
            # Try to write to repo (should be denied); write to /tmp (should succeed)
            inner = [
                '/bin/sh', '-c',
                (
                    f'touch {repo}/src/prod.py 2>/dev/null || echo repo_denied; '
                    f'touch /tmp/{nonce} && echo tmp_ok'
                ),
            ]
            wrap = resolve_recon_sandbox_wrap(base)
            wrapped = wrap(inner)
            result = subprocess.run(wrapped, capture_output=True, text=True, timeout=15)

            assert not (repo / 'src' / 'prod.py').exists(), (
                'Expected repo write to be denied by Landlock'
            )
            assert Path(f'/tmp/{nonce}').exists(), (
                'Expected /tmp write to succeed (agent scratch)'
            )
            assert 'repo_denied' in result.stdout, f'stdout={result.stdout!r}'
            assert 'tmp_ok' in result.stdout, f'stdout={result.stdout!r}'
        finally:
            shutil.rmtree(base, ignore_errors=True)
            Path(f'/tmp/{nonce}').unlink(missing_ok=True)

    @pytest.mark.skipif(
        not is_landlock_available(),
        reason='landlock not supported on this kernel',
    )
    def test_network_loopback_allowed(self, tmp_path: Path) -> None:
        """Landlock filesystem confinement does not block loopback TCP connections.

        Memory MCP is served over HTTP; this confirms network access survives
        (handled_access_net is not set in landlock_exec). leaf-signal #2.
        """
        # Start a loopback listener on an ephemeral port
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(('127.0.0.1', 0))
        port = srv.getsockname()[1]
        srv.listen(1)

        def _accept_and_close() -> None:
            try:
                conn, _ = srv.accept()
                conn.close()
            except OSError:
                pass
            finally:
                srv.close()

        t = threading.Thread(target=_accept_and_close, daemon=True)
        t.start()

        try:
            inner = [
                'python3', '-c',
                (
                    f"import socket; s=socket.create_connection(('127.0.0.1',{port}),timeout=5); "
                    f"s.close(); print('net_ok')"
                ),
            ]
            wrap = resolve_recon_sandbox_wrap(tmp_path)
            wrapped = wrap(inner)
            result = subprocess.run(wrapped, capture_output=True, text=True, timeout=15)
            assert result.returncode == 0, (
                f'Expected loopback connect to succeed; stderr={result.stderr!r}'
            )
            assert 'net_ok' in result.stdout, f'stdout={result.stdout!r}'
        finally:
            t.join(timeout=2)


# ── Fail-closed and bwrap fallback (S7 / S8) ─────────────────────────────────


@pytest.mark.skipif(not _SANDBOX_GUARD_AVAILABLE, reason='sandbox_guard not yet implemented')
class TestSandboxGuardFailClosedAndBwrap:
    """Fail-closed and bwrap-fallback paths for resolve_recon_sandbox_wrap."""

    def test_fail_closed_when_no_backend(self, tmp_path: Path) -> None:
        """When both Landlock and bwrap are unavailable, raises RemediationSandboxUnavailable.

        An identity/passthrough wrap must never be returned — that would
        silently reopen the write hole.
        """
        with patch(
            'orchestrator.agents.landlock.is_landlock_available',
            return_value=False,
        ), patch(
            'orchestrator.agents.sandbox.is_bwrap_available',
            return_value=False,
        ):
            with pytest.raises(RemediationSandboxUnavailable):
                resolve_recon_sandbox_wrap(tmp_path)

    def test_bwrap_fallback_when_landlock_unavailable(self, tmp_path: Path) -> None:
        """When Landlock is unavailable and bwrap is available, routes to bwrap.

        Confirms writable_modules=[] and writable_extras are forwarded
        to build_bwrap_command.
        """
        sentinel_result = ['bwrap', '--', 'claude']
        with patch(
            'orchestrator.agents.landlock.is_landlock_available',
            return_value=False,
        ), patch(
            'orchestrator.agents.sandbox.is_bwrap_available',
            return_value=True,
        ), patch(
            'orchestrator.agents.sandbox.build_bwrap_command',
            return_value=sentinel_result,
        ) as mock_bwrap:
            wrap = resolve_recon_sandbox_wrap(tmp_path, writable_extras=['/e'])
            result = wrap(['claude'])

        assert result is sentinel_result
        mock_bwrap.assert_called_once()
        call_kwargs = mock_bwrap.call_args
        # Second positional arg is cwd (worktree), third is writable_modules
        # (must be []), and writable_extras must be ['/e']
        _cmd, called_cwd, called_modules = call_kwargs.args[:3]
        assert called_modules == [], (
            f'writable_modules must be [] for deny-repo-writes; got {called_modules}'
        )
        assert call_kwargs.kwargs.get('writable_extras') == ['/e'], (
            f'writable_extras should forward ["/e"]; got {call_kwargs.kwargs}'
        )
