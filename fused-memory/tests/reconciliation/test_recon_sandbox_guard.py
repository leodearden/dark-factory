"""Tests for the reconciliation sandbox guard (task 1935).

Covers:
  - ReconciliationConfig sandbox field defaults (S3/S4)
  - sandbox_guard.resolve_recon_sandbox_wrap shape and behaviour (S5/S6)
  - fail-closed and bwrap-fallback paths (S7/S8)
  - config-dir containment: the INV-1 machine check (task 4003)
"""

from __future__ import annotations

import functools
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

from fused_memory.config.schema import ReconciliationConfig

# Deferred import so tests that don't touch sandbox_guard can still run
# when the module doesn't exist yet (S3/S4 tests stay green before S6).
try:
    from fused_memory.reconciliation.sandbox_guard import (
        RemediationSandboxUnavailable,
        _writable_roots,
        resolve_recon_sandbox_wrap,
    )
    _SANDBOX_GUARD_AVAILABLE = True
except ImportError:
    _SANDBOX_GUARD_AVAILABLE = False

try:
    from orchestrator.agents.landlock import is_landlock_available  # type: ignore[import-not-found]
    _LANDLOCK_IMPORTABLE = True
except ImportError:
    _LANDLOCK_IMPORTABLE = False
    def is_landlock_available() -> bool:  # type: ignore[misc]
        return False


_VAR_TMP_SKIP_REASON = '/var/tmp not writable in this sandbox'


@functools.cache
def _var_tmp_writable() -> bool:
    """Probe whether /var/tmp is actually usable for scratch dirs in this process.

    Catches ANY OS-level refusal, not just permission denial: agent sandboxes
    deny the write via a syscall filter (EACCES) even though the directory's
    own mode (1777) permits it, minimal containers may not ship /var/tmp at all
    (ENOENT), and a read-only bind mount surfaces as EROFS. All three must
    degrade to a skip — an escaping OSError here would abort collection of the
    whole module, the exact failure mode these guards exist to prevent. Must be
    an actual write attempt rather than an os.access/stat-mode check.

    NOTE: duplicated verbatim in orchestrator/tests/test_landlock.py and
    orchestrator/tests/test_sandbox_enforcement_matrix.py — no shared
    test-helper module spans both packages, so keep the three copies in sync.
    """
    try:
        probe = tempfile.mkdtemp(dir='/var/tmp')
    except OSError:
        return False
    shutil.rmtree(probe, ignore_errors=True)
    return True


def _skip_var_tmp() -> bool:
    """Whether /var/tmp-dependent tests should be skipped in this environment.

    Under ``DF_REQUIRE_SANDBOX_TESTS=1`` — set by CI jobs known to have a
    writable /var/tmp and a landlock-capable kernel — an unwritable /var/tmp is
    an environment regression, not a reason to skip: quietly dropping the
    real-kernel enforcement assertions would leave the suite green while they
    stopped running. Fail loudly there instead (mirrors test_landlock.py's copy).
    """
    if _var_tmp_writable():
        return False
    if os.environ.get('DF_REQUIRE_SANDBOX_TESTS') == '1':
        pytest.fail(
            f'DF_REQUIRE_SANDBOX_TESTS=1 but {_VAR_TMP_SKIP_REASON}: refusing to '
            'silently skip the real-kernel sandbox enforcement tests.',
            pytrace=False,
        )
    return True


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
            wrap = resolve_recon_sandbox_wrap(tmp_path, writable_extras=['/var/tmp/extra'])  # type: ignore[possibly-unbound]

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
    @pytest.mark.skipif(_skip_var_tmp(), reason=_VAR_TMP_SKIP_REASON)
    def test_enforcement_repo_write_denied_tmp_allowed(self, tmp_path: Path) -> None:
        """Landlock denies writes to a /var/tmp 'repo' dir but allows writes to /tmp.

        Uses /var/tmp for the simulated repo because landlock_exec grants blanket
        write access to /tmp (agent scratch). leaf-signal #1.
        """
        base = Path(tempfile.mkdtemp(prefix='recon-landlock-test-', dir='/var/tmp'))
        nonce = uuid.uuid4().hex
        try:
            repo = base / 'repo'
            repo.mkdir()
            (repo / 'src').mkdir()

            # Try to write to repo (should be denied); write to /tmp (should succeed)
            inner = [
                '/bin/sh', '-c',
                (
                    f'touch {repo}/src/prod.py 2>/dev/null || echo repo_denied; '
                    f'touch /tmp/{nonce} && echo tmp_ok'
                ),
            ]
            wrap = resolve_recon_sandbox_wrap(base)  # type: ignore[possibly-unbound]
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
            wrap = resolve_recon_sandbox_wrap(tmp_path)  # type: ignore[possibly-unbound]
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
        ), pytest.raises(RemediationSandboxUnavailable):  # type: ignore[possibly-unbound]
            resolve_recon_sandbox_wrap(tmp_path)  # type: ignore[possibly-unbound]

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
            wrap = resolve_recon_sandbox_wrap(tmp_path, writable_extras=['/e'])  # type: ignore[possibly-unbound]
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


@pytest.mark.skipif(not _SANDBOX_GUARD_AVAILABLE, reason='sandbox_guard not yet implemented')
class TestConfigDirContainment:
    """The per-run ``CLAUDE_CONFIG_DIR`` must be inside the writable set (task 4003).

    INV-1 ``contracts-machine-checked``. Before this task the capability
    envelope ("the config dir is writable") lived only in a prose comment in
    ``landlock_exec.py`` plus an empty-by-default config list — and the mismatch
    between comment and ruleset was discovered by failure, three weeks late.
    These tests convert that prose into an enforced check: if a future edit
    drops the computed grant, ``resolve_recon_sandbox_wrap`` refuses to launch
    instead of silently producing a transcript-less stage.

    Containment is asserted against the roots BOTH backends grant (``/tmp``,
    ``<cwd>/.task``, and each extra — verified at landlock.py:69-108 and
    sandbox.py:56-101), so the invariant does not depend on which backend wins
    resolution.
    """

    def test_config_dir_outside_writable_set_fails_closed(self, tmp_path: Path) -> None:
        """A config dir outside every writable root raises RemediationSandboxUnavailable.

        This is the regression that would have caught the 2026-07-18 breakage on
        day one: `<data_dir>/recon-config/claude-config-<run_id>` is neither
        `/tmp` nor `<cwd>/.task` nor an extra, so the grant was absent and the
        CLI's transcript writes were denied — silently.
        """
        orphan = Path('/var/tmp/recon-config/claude-config-x')
        with patch(
            'orchestrator.agents.landlock.is_landlock_available',
            return_value=True,
        ), pytest.raises(RemediationSandboxUnavailable) as excinfo:  # type: ignore[possibly-unbound]
            resolve_recon_sandbox_wrap(  # type: ignore[possibly-unbound]
                tmp_path, [], config_dir=orphan,
            )

        msg = str(excinfo.value)
        # The operator's only signal — a bare exception here halts recon, so the
        # message must name both the offending path and the knob to turn.
        assert str(orphan) in msg, (
            f'Error must name the offending config dir {str(orphan)!r}; got {msg!r}'
        )
        assert 'sandbox_recon_writable_extras' in msg, (
            f'Error must name the config key an operator can act on; got {msg!r}'
        )

    def test_config_dir_inside_extras_is_accepted(self, tmp_path: Path) -> None:
        """A config dir passed in writable_extras is accepted and actually granted."""
        cfg = tmp_path / 'recon-config' / 'claude-config-x'
        cfg.mkdir(parents=True)

        with patch(
            'orchestrator.agents.landlock.is_landlock_available',
            return_value=True,
        ):
            wrap = resolve_recon_sandbox_wrap(  # type: ignore[possibly-unbound]
                tmp_path, [str(cfg)], config_dir=cfg,
            )
            wrapped = wrap(['claude', '--print'])

        assert callable(wrap), f'Expected a callable; got {wrap!r}'
        writable_vals = [
            wrapped[i + 1] for i, tok in enumerate(wrapped) if tok == '--writable'
        ]
        assert str(cfg) in writable_vals, (
            f'Accepted config dir must actually appear in the argv grants; '
            f'got {writable_vals!r}'
        )

    def test_config_dir_under_task_dir_is_accepted(self, tmp_path: Path) -> None:
        """`<cwd>/.task/...` is accepted with no extras — both backends grant `.task`.

        The `_writable_roots` assertion is what makes this leaf mean something:
        pytest's ``tmp_path`` lives under ``/tmp``, which is blanket-writable, so
        acceptance alone would be satisfied by the ``/tmp`` root no matter what
        the ``.task`` logic did. Asserting the `.task` root is present AND
        contains the config dir pins the grant this leaf is named for.
        """
        cfg = tmp_path / '.task' / 'claude-config-x'
        cfg.mkdir(parents=True)

        with patch(
            'orchestrator.agents.landlock.is_landlock_available',
            return_value=True,
        ):
            wrap = resolve_recon_sandbox_wrap(  # type: ignore[possibly-unbound]
                tmp_path, [], config_dir=cfg,
            )

        assert callable(wrap), f'Expected a callable; got {wrap!r}'

        task_root = os.path.realpath(str(tmp_path / '.task'))
        roots = _writable_roots(tmp_path, [])  # type: ignore[possibly-unbound]
        assert task_root in roots, (
            f'`<cwd>/.task` must be one of the writable roots; got {roots!r}'
        )
        assert os.path.realpath(str(cfg)).startswith(task_root + os.sep), (
            f'{cfg} must be inside the .task root {task_root}'
        )

    def test_config_dir_under_tmp_is_accepted(self) -> None:
        """A /tmp config dir is accepted with no extras — /tmp is blanket-writable."""
        cfg = Path(tempfile.mkdtemp(prefix='recon-cfg-', dir='/tmp'))
        try:
            with patch(
                'orchestrator.agents.landlock.is_landlock_available',
                return_value=True,
            ):
                wrap = resolve_recon_sandbox_wrap(  # type: ignore[possibly-unbound]
                    cfg, [], config_dir=cfg,
                )
            assert callable(wrap), f'Expected a callable; got {wrap!r}'
        finally:
            shutil.rmtree(cfg, ignore_errors=True)

    def test_config_dir_none_skips_check(self, tmp_path: Path) -> None:
        """config_dir=None returns a wrap and never raises (back-compat).

        The generic/non-recon call sites pass no config dir; they must keep
        today's behaviour exactly.
        """
        with patch(
            'orchestrator.agents.landlock.is_landlock_available',
            return_value=True,
        ):
            wrap = resolve_recon_sandbox_wrap(tmp_path, [])  # type: ignore[possibly-unbound]

        assert callable(wrap), f'Expected a callable; got {wrap!r}'

    def test_extra_that_does_not_exist_does_not_satisfy_containment(
        self, tmp_path: Path,
    ) -> None:
        """A grant naming a non-existent dir is vacuous, so it must not satisfy the check.

        ``landlock_exec._add_path`` returns SILENTLY for a path that does not
        exist — no rule is added and no error is raised. A containment check
        that trusted such an extra would pass while the write still failed at
        runtime, reproducing the exact silent-degrade class this check exists to
        end.
        """
        # NOT under tmp_path: pytest's tmp_path is itself under /tmp, which both
        # backends grant blanket, so a ghost root there would be rescued by the
        # /tmp rule and this leaf would silently prove nothing. Nothing is ever
        # created here, so no /var/tmp write permission is required.
        ghost_root = Path(f'/var/tmp/df-4003-never-created-{uuid.uuid4().hex}')
        cfg = ghost_root / 'claude-config-x'
        assert not ghost_root.exists(), 'precondition: the root must not exist on disk'

        with patch(
            'orchestrator.agents.landlock.is_landlock_available',
            return_value=True,
        ), pytest.raises(RemediationSandboxUnavailable):  # type: ignore[possibly-unbound]
            resolve_recon_sandbox_wrap(  # type: ignore[possibly-unbound]
                tmp_path, [str(ghost_root)], config_dir=cfg,
            )

    @pytest.mark.skipif(
        not is_landlock_available(),
        reason='landlock not supported on this kernel',
    )
    @pytest.mark.skipif(_skip_var_tmp(), reason=_VAR_TMP_SKIP_REASON)
    def test_enforcement_per_run_config_dir_writable_sibling_denied(self) -> None:
        """Real kernel: this run's config dir is writable, a sibling run's is NOT.

        The credential-isolation invariant proved against an actual Landlock
        ruleset rather than an argv shape. /var/tmp (not /tmp) is mandatory —
        /tmp is blanket-writable in both backends, which would make the
        sibling-denied half of this test vacuous.

        The denial probes must be genuine CONTENT writes. ``touch`` on a file
        that already exists is NOT one: coreutils falls back to ``utimensat()``
        when the ``open(O_WRONLY)`` is refused, and Landlock has no access right
        governing timestamps at all — its FS bits cover EXECUTE / WRITE_FILE /
        READ_FILE / READ_DIR / REMOVE_* / MAKE_* / REFER / TRUNCATE / IOCTL_DEV
        and nothing else. A ``touch``-the-credentials probe therefore succeeds
        under *every* ruleset, including a correct one, and pins nothing.
        """
        base = Path(tempfile.mkdtemp(prefix='recon-cfgdir-test-', dir='/var/tmp'))
        try:
            repo = base / 'repo'
            repo.mkdir()
            cfg_base = base / 'recon-config'
            mine = cfg_base / 'claude-config-mine'
            other = cfg_base / 'claude-config-other'
            mine.mkdir(parents=True)
            other.mkdir(parents=True)
            # A real credential file in the sibling: what must stay unwritable.
            (other / '.credentials.json').write_text('{"token": "sibling"}')

            inner = [
                '/bin/sh', '-c',
                (
                    # `mine` half: s.jsonl does NOT exist yet, so this is a real
                    # O_CREAT (MAKE_REG), the same right the CLI needs to lay
                    # down its session transcript.
                    f'mkdir -p {mine}/projects && touch {mine}/projects/s.jsonl '
                    f'&& echo mine_ok; '
                    # Overwriting the bytes (WRITE_FILE|TRUNCATE) — the exact
                    # capability that must not exist. See the docstring for why
                    # this is not spelled `touch`.
                    f'echo stolen > {other}/.credentials.json 2>/dev/null '
                    f'|| echo sibling_write_denied; '
                    # And no NEW file in a sibling's dir either (MAKE_REG).
                    f'touch {other}/planted 2>/dev/null || echo sibling_create_denied'
                ),
            ]
            wrap = resolve_recon_sandbox_wrap(  # type: ignore[possibly-unbound]
                repo, [str(mine)], config_dir=mine,
            )
            result = subprocess.run(
                wrap(inner), capture_output=True, text=True, timeout=15,
            )

            assert (mine / 'projects' / 's.jsonl').exists(), (
                f'This run\'s config dir must be writable — that transcript write '
                f'is the whole point. stdout={result.stdout!r} stderr={result.stderr!r}'
            )
            assert 'mine_ok' in result.stdout, f'stdout={result.stdout!r}'
            assert 'sibling_write_denied' in result.stdout, (
                f'A sibling run\'s .credentials.json must stay read-only. '
                f'stdout={result.stdout!r} stderr={result.stderr!r}'
            )
            assert 'sibling_create_denied' in result.stdout, (
                f'A sibling run\'s config dir must not accept new files. '
                f'stdout={result.stdout!r} stderr={result.stderr!r}'
            )
            # The ground truth behind both probes: the credential bytes are intact.
            assert (other / '.credentials.json').read_text() == '{"token": "sibling"}', (
                'A sibling run\'s OAuth credentials were modified through the '
                'sandbox — the credential-isolation invariant is broken.'
            )
            assert not (other / 'planted').exists(), (
                'A file was created inside a sibling run\'s config dir.'
            )
        finally:
            shutil.rmtree(base, ignore_errors=True)
