"""Tests for Harness's leftover ``df-verify-*.scope`` reaper wiring (task 2829).

Companion to reify flipping ``verify_use_cgroup_scope: true``. Mirrors
test_harness_interactive_reaper.py's structure:

  step-07/08 — ``_run_leftover_verify_scope_reaper_pass()`` delegates to
               ``verify.reap_leftover_verify_scopes(self.config.project_root)``,
               logs one INFO line per reaped unit + a summary INFO line, and
               never raises.
  step-09/10 — an unconditional sweep runs once at ``run()`` startup, before
               first dispatch.
  step-11    — the task's mandated real-systemd live-sleeper acceptance signal
               (skip-guarded): the tagged scope is stopped while a
               differently-tagged sibling scope SURVIVES (cross-project safety).
"""
from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import time
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator import verify
from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventStore
from orchestrator.harness import Harness
from orchestrator.run_store import RunStore

# ---------------------------------------------------------------------------
# Test factory (mirrors test_harness_interactive_reaper._make_harness)
# ---------------------------------------------------------------------------


def _make_harness(tmp_path: Path) -> tuple[Harness, MagicMock]:
    """Bare Harness with a real config and a spy RunStore."""
    config = OrchestratorConfig(project_root=tmp_path)
    harness = Harness(config)
    mock_run_store = MagicMock(spec=RunStore)
    harness._run_store = mock_run_store
    harness._run_id = 'run-verify-scope-reaper-0001'
    harness.event_store = EventStore(
        tmp_path / 'events.db', 'run-verify-scope-reaper-0001',
    )
    return harness, mock_run_store


# ---------------------------------------------------------------------------
# Step-07: _run_leftover_verify_scope_reaper_pass delegates to
# verify.reap_leftover_verify_scopes, logs per-unit + a summary, never raises.
# ---------------------------------------------------------------------------


class TestRunLeftoverVerifyScopeReaperPass:
    """RED until step-08 adds _run_leftover_verify_scope_reaper_pass to Harness."""

    @pytest.mark.asyncio
    async def test_pass_delegates_and_logs_per_unit_and_summary(
        self, tmp_path: Path, caplog,
    ) -> None:
        """Awaits reap_leftover_verify_scopes(project_root) once; logs one INFO
        line naming each reaped unit + a summary INFO line with the count."""
        harness, _rs = _make_harness(tmp_path)
        units = ['df-verify-proj-1a2b3c4d-abcabcabcabc.scope']
        mock_reap = AsyncMock(return_value=units)

        with (
            patch('orchestrator.verify.reap_leftover_verify_scopes', mock_reap),
            caplog.at_level(logging.INFO, logger='orchestrator.harness'),
        ):
            await harness._run_leftover_verify_scope_reaper_pass()

        mock_reap.assert_awaited_once_with(harness.config.project_root)

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert any(units[0] in r.getMessage() for r in info_records), (
            f'expected an INFO line naming the reaped unit {units[0]!r}; '
            f'got: {[r.getMessage() for r in info_records]}'
        )
        assert any(
            'reaper' in r.getMessage().lower() and str(len(units)) in r.getMessage()
            for r in info_records
        ), (
            'expected a summary INFO line naming the reaped count; '
            f'got: {[r.getMessage() for r in info_records]}'
        )

    @pytest.mark.asyncio
    async def test_pass_logs_debug_when_nothing_reaped(
        self, tmp_path: Path, caplog,
    ) -> None:
        """No leftovers -> no INFO summary noise (DEBUG instead), never raises."""
        harness, _rs = _make_harness(tmp_path)
        mock_reap = AsyncMock(return_value=[])

        with (
            patch('orchestrator.verify.reap_leftover_verify_scopes', mock_reap),
            caplog.at_level(logging.INFO, logger='orchestrator.harness'),
        ):
            await harness._run_leftover_verify_scope_reaper_pass()

        mock_reap.assert_awaited_once_with(harness.config.project_root)
        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert not info_records, (
            f'expected NO INFO lines when nothing was reaped; '
            f'got: {[r.getMessage() for r in info_records]}'
        )

    @pytest.mark.asyncio
    async def test_pass_swallows_exception_and_logs_error(
        self, tmp_path: Path, caplog,
    ) -> None:
        """A raising reap_leftover_verify_scopes() does not propagate; a bounded
        error line is logged instead."""
        harness, _rs = _make_harness(tmp_path)
        mock_reap = AsyncMock(side_effect=RuntimeError('boom'))

        with (
            patch('orchestrator.verify.reap_leftover_verify_scopes', mock_reap),
            caplog.at_level(logging.ERROR, logger='orchestrator.harness'),
        ):
            # Must not raise.
            await harness._run_leftover_verify_scope_reaper_pass()

        mock_reap.assert_awaited_once()
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors, (
            'expected an ERROR log when reap_leftover_verify_scopes raises'
        )


# ---------------------------------------------------------------------------
# Step-09: an unconditional startup sweep runs once at run() startup, before
# first dispatch (crash recovery on boot). Mirrors
# test_harness_interactive_reaper.TestInteractiveReaperStartupSweep.
# ---------------------------------------------------------------------------


def _neutralise_heavy_startup(harness: Harness) -> None:
    """Stub heavyweight startup/shutdown so the real run() loop is drivable.

    Copied from test_harness_interactive_reaper._neutralise_heavy_startup —
    every background-loop starter and external-server touchpoint is stubbed so
    run() can be driven end-to-end against a bare tmp_path.
    """
    harness.mcp = MagicMock()
    harness.mcp.start = AsyncMock()
    harness.mcp.stop = AsyncMock()
    harness.mcp.url = 'http://localhost:0'
    harness.usage_gate = None
    harness.review_checkpoint = None

    harness._build_lifecycle_registry()
    assert harness._lifecycle is not None
    harness._lifecycle.start_all = AsyncMock()
    harness._lifecycle.stop_all = AsyncMock()

    harness._dismiss_stale_escalations = AsyncMock()
    harness._rehydrate_merge_halt = MagicMock()
    harness._file_restored_pause_escalation = MagicMock()
    harness._tag_task_modules = AsyncMock()
    harness._recover_crashed_tasks = AsyncMock()
    harness._reconcile_stranded_in_progress = AsyncMock(return_value=0)
    harness._enforce_cost_ceilings = AsyncMock()


class TestVerifyScopeReaperStartupSweep:
    """An unconditional leftover-verify-scope reaper sweep runs once at
    ``run()`` startup, before the first dispatch — crash recovery on boot.

    RED until step-10 GREEN wires the unconditional startup call into run()
    adjacent to ``_run_interactive_worktree_reaper_pass()``.
    """

    async def _drive_empty_until_idle_run(
        self, harness: Harness, monkeypatch,
    ) -> None:
        """Drive run() to a clean, immediate exit on a drained empty tree."""
        _neutralise_heavy_startup(harness)
        harness.scheduler.acquire_next = AsyncMock(return_value=None)
        harness.scheduler.get_statuses = AsyncMock(
            return_value=({'1': 'pending'}, None),
        )

        async def fake_sleep(_secs, *args, **kwargs):
            return

        monkeypatch.setattr('orchestrator.harness.asyncio.sleep', fake_sleep)

        await harness.run(
            prd_path=None, dry_run=False, force_dirty_start=True,
            until_idle=True,
        )

    @pytest.mark.asyncio
    async def test_startup_sweep_runs_once(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """The leftover-verify-scope reaper pass is awaited exactly once during
        run() startup (before first dispatch / scheduler.finish_startup)."""
        harness, _rs = _make_harness(tmp_path)
        mock_pass = AsyncMock(return_value=None)
        harness._run_leftover_verify_scope_reaper_pass = mock_pass

        await self._drive_empty_until_idle_run(harness, monkeypatch)

        mock_pass.assert_awaited_once()


# ---------------------------------------------------------------------------
# Step-11: the task's mandated real-systemd SIGNAL test (skip-guarded).
# Create a real live-sleeper df-verify-{tag}-*.scope plus a DIFFERENTLY-tagged
# sibling scope; run the reaper; assert the tagged scope + sleeper are dead and
# a summary line is logged, while the sibling SURVIVES (cross-project safety).
# ---------------------------------------------------------------------------


def _systemd_user_available() -> bool:
    """True iff a usable ``systemd --user`` manager can create a transient scope."""
    try:
        r = subprocess.run(
            ['systemd-run', '--user', '--scope', '--collect', 'true'],
            capture_output=True, timeout=15,
        )
        return r.returncode == 0
    except Exception:
        return False


def _sc(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ['systemctl', '--user', *args],
        capture_output=True, text=True, timeout=15,
    )


def _is_active(unit: str) -> str:
    return _sc('is-active', unit).stdout.strip()


def _scope_pids(unit: str) -> list[int]:
    """PIDs in the scope's cgroup. A ``--scope`` unit has no MainPID (it adopts
    processes rather than forking one), so membership is read from cgroup.procs."""
    cg = _sc('show', unit, '--property=ControlGroup', '--value').stdout.strip()
    if not cg:
        return []
    procs = Path('/sys/fs/cgroup') / cg.lstrip('/') / 'cgroup.procs'
    try:
        return [int(x) for x in procs.read_text().split()]
    except Exception:
        return []


def _pid_alive(pid: int) -> bool:
    """True iff *pid* names a live (non-zombie) process.

    A SIGKILL'd sleeper whose parent (this pytest process, via Popen) has not
    yet ``wait()``ed becomes a ZOMBIE — still in the process table, so a bare
    ``os.kill(pid, 0)`` probe would wrongly report it alive. Read the proc
    state and treat 'Z' as dead so the reap assertion reflects real liveness.
    """
    if pid <= 0:
        return False
    try:
        stat = Path(f'/proc/{pid}/stat').read_text()
        # Format: "pid (comm) state ..."; comm may contain spaces/parens, so
        # split past the final ') ' before reading the state field.
        state = stat.rsplit(') ', 1)[1].split(' ', 1)[0]
        return state != 'Z'
    except (ProcessLookupError, FileNotFoundError):
        return False
    except Exception:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True


def _launch_scope(unit: str) -> subprocess.Popen:
    """Launch a live-sleeper transient scope (foreground systemd-run, detached)."""
    return subprocess.Popen(
        ['systemd-run', '--user', '--scope', f'--unit={unit}', 'sleep', '300'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _wait_active(unit: str, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _is_active(unit) == 'active':
            return True
        time.sleep(0.2)
    return _is_active(unit) == 'active'


def _wait_scope_pids(unit: str, timeout: float = 10.0) -> list[int]:
    deadline = time.monotonic() + timeout
    pids = _scope_pids(unit)
    while not pids and time.monotonic() < deadline:
        time.sleep(0.2)
        pids = _scope_pids(unit)
    return pids


@pytest.mark.skipif(
    not _systemd_user_available(),
    reason='systemd --user manager unavailable (CI/session degraded)',
)
class TestVerifyScopeReaperRealSystemd:
    """The task's mandated acceptance signal over REAL systemd --user."""

    @pytest.mark.asyncio
    async def test_tagged_scope_reaped_sibling_survives(
        self, tmp_path: Path, caplog,
    ) -> None:
        # Two distinct project roots -> two distinct tags (unique tmp_path, so
        # the sweep can never touch a real orchestrator's scopes).
        root = tmp_path / 'proj'
        sibling_root = tmp_path / 'sibling'
        root.mkdir()
        sibling_root.mkdir()

        tag = verify._scope_tag_for(root)
        sibling_tag = verify._scope_tag_for(sibling_root)
        assert tag != sibling_tag, (tag, sibling_tag)

        unit = f'df-verify-{tag}-{uuid.uuid4().hex[:12]}.scope'
        sibling_unit = f'df-verify-{sibling_tag}-{uuid.uuid4().hex[:12]}.scope'

        launcher: subprocess.Popen | None = None
        sibling_launcher: subprocess.Popen | None = None
        try:
            launcher = _launch_scope(unit)
            sibling_launcher = _launch_scope(sibling_unit)
            assert _wait_active(unit), f'{unit} never became active'
            assert _wait_active(sibling_unit), f'{sibling_unit} never became active'

            sleeper_pids = _wait_scope_pids(unit)
            assert sleeper_pids and all(_pid_alive(p) for p in sleeper_pids), (
                f'sleeper(s) for {unit} not running (cgroup pids={sleeper_pids})'
            )

            harness, _rs = _make_harness(root)
            with caplog.at_level(logging.INFO, logger='orchestrator.harness'):
                await harness._run_leftover_verify_scope_reaper_pass()

            # The tagged scope + its sleeper(s) are reaped (poll briefly for the
            # SIGKILL/stop to settle).
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and (
                _is_active(unit) == 'active'
                or any(_pid_alive(p) for p in sleeper_pids)
            ):
                time.sleep(0.2)
            assert _is_active(unit) != 'active', (
                f'{unit} still active after reap (state={_is_active(unit)!r})'
            )
            assert not any(_pid_alive(p) for p in sleeper_pids), (
                f'sleeper(s) {sleeper_pids} still alive after reap'
            )

            # A summary line naming the reaped tagged unit was logged; the
            # sibling unit is NEVER named.
            info_msgs = [
                r.getMessage() for r in caplog.records if r.levelno == logging.INFO
            ]
            assert any('reaper' in m.lower() for m in info_msgs), (
                f'expected a summary INFO line; got: {info_msgs}'
            )
            assert any(unit in m for m in info_msgs), (
                f'expected an INFO line naming reaped {unit!r}; got: {info_msgs}'
            )
            assert not any(sibling_unit in m for m in info_msgs), (
                f'sibling {sibling_unit!r} must never be named as reaped; '
                f'got: {info_msgs}'
            )

            # Cross-project safety: the DIFFERENTLY-tagged sibling SURVIVES.
            assert _is_active(sibling_unit) == 'active', (
                f'sibling {sibling_unit} must survive the tag-scoped sweep '
                f'(state={_is_active(sibling_unit)!r})'
            )
        finally:
            for u in (unit, sibling_unit):
                with contextlib.suppress(Exception):
                    _sc('kill', '--signal=SIGKILL', u)
                with contextlib.suppress(Exception):
                    _sc('stop', u)
                with contextlib.suppress(Exception):
                    _sc('reset-failed', u)
            for p in (launcher, sibling_launcher):
                if p is not None:
                    with contextlib.suppress(Exception):
                        p.terminate()
                    with contextlib.suppress(Exception):
                        p.wait(timeout=5)
