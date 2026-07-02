"""Tests for Harness's interactive-worktree (``_iact-*``) reaper wiring — task δ (2012).

Covers the thin harness-side cadence/startup wiring around
``git_ops.reap_interactive_worktrees`` (task δ's git-primitive, see
test_interactive_worktree_reaper.py):

  step-09/10 — ``_run_interactive_worktree_reaper_pass()`` delegates to
               ``git_ops.reap_interactive_worktrees()``, logs one INFO line per
               reaped record, and never raises.
  step-11/12 — the pass is folded into the existing warm-lane GC cadence tick
               (``_run_warm_lane_gc_pass()``).
  step-13/14 — an unconditional sweep runs at ``run()`` startup, independent of
               the ``warm_lane_gc_enabled`` kill-switch (crash recovery on boot).
  step-15/16 — end-to-end over a real repo: the I2 user-observable signal (log
               line + absence from ``git worktree list``).
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventStore
from orchestrator.git_ops import ReapedInteractiveWorktree, _run
from orchestrator.harness import Harness
from orchestrator.run_store import RunStore

# ---------------------------------------------------------------------------
# Test factory (mirrors test_harness_warm_lane_gc._make_harness)
# ---------------------------------------------------------------------------


def _make_harness(tmp_path: Path) -> tuple[Harness, MagicMock]:
    """Bare Harness with a real config and a spy RunStore.

    Mirrors test_harness_warm_lane_gc._make_harness.
    Returns (harness, mock_run_store).
    """
    config = OrchestratorConfig(project_root=tmp_path)
    harness = Harness(config)
    mock_run_store = MagicMock(spec=RunStore)
    harness._run_store = mock_run_store
    harness._run_id = 'run-interactive-reaper-0001'
    harness.event_store = EventStore(tmp_path / 'events.db', 'run-interactive-reaper-0001')
    return harness, mock_run_store


# ---------------------------------------------------------------------------
# Real-repo fixtures + helpers (mirrors test_interactive_worktree_reaper.py)
# ---------------------------------------------------------------------------


async def _init_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


async def _commit_file(repo: Path, name: str, content: str, message: str) -> str:
    """Write+commit a file on the repo's current branch; return the new commit SHA."""
    (repo / name).write_text(content)
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', message], cwd=repo)
    rc, sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    assert rc == 0, f'rev-parse HEAD failed after committing {name!r}'
    return sha.strip()


def _backdate_stamp(path: Path, created_at: datetime) -> None:
    """Rewrite the ``.task/interactive.json`` stamp's ``created_at`` field."""
    import json

    stamp_path = path / '.task' / 'interactive.json'
    stamp = json.loads(stamp_path.read_text())
    stamp['created_at'] = created_at.isoformat()
    stamp_path.write_text(json.dumps(stamp))


async def _registered_worktree_paths(repo: Path) -> set[str]:
    """Return the set of registered worktree paths (resolved) via `git worktree list`."""
    rc, out, _ = await _run(['git', 'worktree', 'list', '--porcelain'], cwd=repo)
    assert rc == 0, 'git worktree list --porcelain failed'
    paths = set()
    for line in out.splitlines():
        if line.startswith('worktree '):
            paths.add(str(Path(line[len('worktree '):].strip()).resolve()))
    return paths


# ---------------------------------------------------------------------------
# Step-09: _run_interactive_worktree_reaper_pass delegates to
# git_ops.reap_interactive_worktrees, logs per-record, and never raises.
# ---------------------------------------------------------------------------


class TestRunInteractiveWorktreeReaperPass:
    """RED until step-10 GREEN adds _run_interactive_worktree_reaper_pass to Harness."""

    @pytest.mark.asyncio
    async def test_pass_delegates_and_logs_one_info_line_per_record(
        self, tmp_path: Path, caplog,
    ) -> None:
        """Awaits reap_interactive_worktrees() once; logs one INFO line per
        reaped record naming slug, branch, and reason."""
        harness, _rs = _make_harness(tmp_path)
        records = [
            ReapedInteractiveWorktree(
                path=Path('/tmp/x/_iact-alpha'), branch='task/alpha',
                slug='alpha', reason='ttl_idle',
            ),
            ReapedInteractiveWorktree(
                path=Path('/tmp/x/_iact-beta'), branch='task/beta',
                slug='beta', reason='landed',
            ),
        ]
        mock_reap = AsyncMock(return_value=records)
        harness.git_ops.reap_interactive_worktrees = mock_reap

        with caplog.at_level(logging.INFO, logger='orchestrator.harness'):
            await harness._run_interactive_worktree_reaper_pass()

        mock_reap.assert_awaited_once()

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        for rec in records:
            assert any(
                rec.slug in r.getMessage()
                and rec.branch in r.getMessage()
                and rec.reason in r.getMessage()
                for r in info_records
            ), (
                f'expected an INFO line naming slug={rec.slug!r} '
                f'branch={rec.branch!r} reason={rec.reason!r}; '
                f'got: {[r.getMessage() for r in info_records]}'
            )

    @pytest.mark.asyncio
    async def test_pass_swallows_exception_and_logs_error(
        self, tmp_path: Path, caplog,
    ) -> None:
        """A raising reap_interactive_worktrees() does not propagate; an
        error is logged instead."""
        harness, _rs = _make_harness(tmp_path)
        mock_reap = AsyncMock(side_effect=RuntimeError('boom'))
        harness.git_ops.reap_interactive_worktrees = mock_reap

        with caplog.at_level(logging.ERROR, logger='orchestrator.harness'):
            # Must not raise.
            await harness._run_interactive_worktree_reaper_pass()

        mock_reap.assert_awaited_once()
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors, 'expected an ERROR log when reap_interactive_worktrees raises'


# ---------------------------------------------------------------------------
# Step-11: the interactive reaper is folded into the existing warm-lane GC
# cadence tick (_run_warm_lane_gc_pass), per PRD δ "folded into the existing
# warm-lane GC cadence" — no new config knob, no new loop.
# ---------------------------------------------------------------------------


class TestWarmLaneGcPassFoldsInInteractiveReaper:
    """_run_warm_lane_gc_pass() also drives the interactive-worktree reaper.

    RED until step-12 GREEN adds the delegation call to
    Harness._run_warm_lane_gc_pass.
    """

    @pytest.mark.asyncio
    async def test_gc_pass_awaits_interactive_reaper_pass_once(
        self, tmp_path: Path,
    ) -> None:
        """Every warm-lane GC cadence tick also runs the interactive reaper."""
        harness, _rs = _make_harness(tmp_path)
        harness.git_ops._run_warm_lane_gc_reclaim = AsyncMock(return_value=0)
        mock_reaper_pass = AsyncMock(return_value=None)
        harness._run_interactive_worktree_reaper_pass = mock_reaper_pass

        await harness._run_warm_lane_gc_pass()

        mock_reaper_pass.assert_awaited_once()


# ---------------------------------------------------------------------------
# Step-13: unconditional startup sweep, independent of warm_lane_gc_enabled
# ---------------------------------------------------------------------------


def _neutralise_heavy_startup(harness: Harness) -> None:
    """Stub heavyweight startup/shutdown so the real run() loop is drivable.

    Mirrors TestHarnessRunForever._neutralise in test_harness_park_stop.py —
    every background-loop starter and external-server touchpoint is stubbed
    so run() can be driven end-to-end against a bare tmp_path (no real MCP
    server, no real git repo needed for this startup-wiring test).
    """
    harness.mcp = MagicMock()
    harness.mcp.start = AsyncMock()
    harness.mcp.stop = AsyncMock()
    harness.mcp.url = 'http://localhost:0'
    harness.usage_gate = None
    harness.review_checkpoint = None

    harness._start_escalation_server = AsyncMock()
    harness._start_merge_worker = AsyncMock()
    harness._dismiss_stale_escalations = AsyncMock()
    harness._rehydrate_merge_halt = MagicMock()
    harness._file_restored_pause_escalation = MagicMock()
    harness._start_orphan_l0_reaper = MagicMock()
    harness._start_terminal_status_watcher = MagicMock()
    harness._start_watcher_supervisor = MagicMock()
    harness._start_stranded_reconcile = MagicMock()
    harness._start_main_tip_sweep = MagicMock()
    harness._start_no_landings_breaker = MagicMock()  # task 1918 loop
    harness._start_warm_lane_gc = MagicMock()  # task 1926 loop — neutralised
    harness._tag_task_modules = AsyncMock()
    harness._recover_crashed_tasks = AsyncMock()
    harness._reconcile_stranded_in_progress = AsyncMock(return_value=0)
    harness._enforce_cost_ceilings = AsyncMock()


class TestInteractiveReaperStartupSweep:
    """An unconditional interactive-worktree reaper sweep runs once at
    ``run()`` startup, independent of the ``warm_lane_gc_enabled``
    kill-switch — crash recovery on boot must not wait for (or depend on)
    the periodic cadence loop.

    RED until step-14 GREEN adds the unconditional startup call in run()
    adjacent to ``_start_warm_lane_gc()``.
    """

    async def _drive_empty_until_idle_run(
        self, harness: Harness, monkeypatch,
    ) -> None:
        """Drive run() to a clean, immediate exit on a drained empty tree.

        Mirrors TestHarnessRunForever.test_until_idle_empty_tree_exits_cleanly:
        with until_idle=True and acquire_next always None, the loop breaks
        on the first completion check without ever reaching idle-sleep.
        """
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
    async def test_startup_sweep_runs_once_when_gc_enabled(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """The baseline case: gc_enabled=True (default) still gets exactly
        one startup sweep — distinct from (and in addition to) the periodic
        cadence, which never fires here since _start_warm_lane_gc is
        neutralised to a MagicMock no-op."""
        harness, _rs = _make_harness(tmp_path)
        harness.config.warm_lane_gc_enabled = True
        mock_pass = AsyncMock(return_value=None)
        harness._run_interactive_worktree_reaper_pass = mock_pass

        await self._drive_empty_until_idle_run(harness, monkeypatch)

        mock_pass.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_startup_sweep_runs_even_when_gc_disabled(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """Crash-recovery sweep at boot is independent of the periodic-cadence
        kill-switch: a crash-leaked _iact-* worktree must still be reaped on
        boot even when warm_lane_gc_enabled=False disables the loop."""
        harness, _rs = _make_harness(tmp_path)
        harness.config.warm_lane_gc_enabled = False
        mock_pass = AsyncMock(return_value=None)
        harness._run_interactive_worktree_reaper_pass = mock_pass

        await self._drive_empty_until_idle_run(harness, monkeypatch)

        mock_pass.assert_awaited_once()


# ---------------------------------------------------------------------------
# Step-15/16: end-to-end I2 user-observable signal over a real repo — the
# stale worktree is gone from `git worktree list`, the live one remains, and
# a summary INFO line reports the reaped count. This is the ζ-facing signal
# that no _iact-* leaks past interactive_worktree_ttl + one sweep interval.
# ---------------------------------------------------------------------------


class TestInteractiveReaperEndToEndSummarySignal:
    """RED until step-16 GREEN adds a summary log line to
    _run_interactive_worktree_reaper_pass.
    """

    @pytest.mark.asyncio
    async def test_stale_reaped_live_preserved_with_summary_log(
        self, tmp_path: Path, caplog,
    ) -> None:
        """Over a real repo: a TTL-expired idle worktree is reaped and gone
        from `git worktree list`; a within-TTL live worktree remains; a
        summary INFO line reports the count of worktrees reaped."""
        repo = tmp_path / 'repo'
        repo.mkdir()
        await _init_repo(repo)

        harness, _rs = _make_harness(repo)
        git_ops = harness.git_ops

        info_stale = await git_ops.create_interactive_worktree('stale')
        info_live = await git_ops.create_interactive_worktree('live')
        await _commit_file(info_live.path, 'work.txt', 'work\n', 'wip on live')

        _backdate_stamp(
            info_stale.path,
            datetime.now(UTC)
            - timedelta(seconds=harness.config.git.interactive_worktree_ttl + 3600),
        )

        with caplog.at_level(logging.INFO, logger='orchestrator.harness'):
            await harness._run_interactive_worktree_reaper_pass()

        registered = await _registered_worktree_paths(repo)
        assert str(info_stale.path.resolve()) not in registered, (
            f'expected the stale worktree to be reaped; registered: {registered}'
        )
        assert str(info_live.path.resolve()) in registered, (
            f'expected the live worktree to remain; registered: {registered}'
        )

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert any(
            'reaped' in r.getMessage().lower() and '1' in r.getMessage()
            for r in info_records
        ), (
            'expected a summary INFO line reporting the count of reaped '
            f'interactive worktrees; got: {[r.getMessage() for r in info_records]}'
        )
