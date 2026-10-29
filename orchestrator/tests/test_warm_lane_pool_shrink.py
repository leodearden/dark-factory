"""Warm-lane pool SHRINK over a real git repo (task 5565).

A restart with a smaller pool (``max_concurrent_tasks + git.spare_warm_lanes``)
leaves lanes ``_lane-N`` and up on disk, some still ASSIGNED to live tasks.
These tests acquire real lanes with a larger pool, then simulate the shrunk
restart by swapping in a smaller ``WarmLanePool`` over the same on-disk
worktrees and durable records, and drive the REAL production seams:
``GitOps.lane_branch_checkouts``, ``Harness._recover_crashed_tasks``,
``Harness._reconcile_lane_checkouts``, ``GitOps.acquire_warm_lane`` and
``GitOps.cleanup_worktree``.

Module-local fixtures only (copied from test_lane_lifecycle_integration.py):
a conftest edit trips verify.py's has_conftest full-suite fallback.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import WorktreeInfo, _run
from orchestrator.harness import Harness
from orchestrator.lane_lifecycle import LaneState as DurableLaneState
from orchestrator.warm_lane_pool import LaneState, WarmLanePool, WarmLanePoolShrink

# ── Module-local fixtures + helpers ─────────────────────────────────────


async def _init_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def ig_git_repo(tmp_path: Path) -> Path:
    """Temporary git repo with an initial commit, a warm base and pool-root sentinel."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    default_base = repo / '.worktrees' / '_merge-verify' / 'target'
    default_base.mkdir(parents=True, exist_ok=True)
    (default_base / '.keep').write_text('warm base sentinel\n')
    (repo / '.worktrees' / '.pool-root').touch()
    return repo


async def _add_warm_lane_scripts(repo: Path, port: int = 39411) -> None:
    """Commit stub seed-warm-lane.sh + setup-worktree-debug-port.sh into repo."""
    scripts_dir = repo / 'scripts'
    scripts_dir.mkdir(parents=True, exist_ok=True)
    seed_script = scripts_dir / 'seed-warm-lane.sh'
    seed_script.write_text(
        '#!/usr/bin/env bash\nmkdir -p "$2/target"\necho "seeded" > "$2/target/seeded.bin"\n'
    )
    seed_script.chmod(0o755)
    debug_script = scripts_dir / 'setup-worktree-debug-port.sh'
    debug_script.write_text(f'#!/usr/bin/env bash\necho {port}\n')
    debug_script.chmod(0o755)
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'add warm-lane scripts'], cwd=repo)


def _make_orch_config(repo: Path, *, max_concurrent_tasks: int) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=repo,
        max_concurrent_tasks=max_concurrent_tasks,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
            push_after_advance=False,
            warm_lane_pool=True,
        ),
    )


def _build_harness(config: OrchestratorConfig) -> Harness:
    """Harness with heavy constructors patched out; GitOps stays REAL."""
    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.Scheduler'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        return Harness(config)


async def _git(cwd: Path, *args: str) -> str:
    rc, out, err = await _run(['git', *args], cwd=cwd)
    assert rc == 0, f'git {" ".join(args)} failed (rc={rc}): {err}'
    return out.strip()


async def _head_is_detached(lane: Path) -> bool:
    rc, _, _ = await _run(['git', 'symbolic-ref', '-q', 'HEAD'], cwd=lane)
    return rc != 0


def _make_plan(done: int, total: int, task_id: str) -> dict:
    steps = [
        {
            'id': f'step-{i + 1}',
            'description': f'Step {i + 1}',
            'status': 'done' if i < done else 'pending',
            'commit': f'abc{i}' if i < done else None,
        }
        for i in range(total)
    ]
    return {'task_id': task_id, 'title': _title(task_id), 'steps': steps}


def _title(task_id: str) -> str:
    return f'Shrink task {task_id}'


def _wire_recovery_scheduler(harness: Harness, statuses: dict[str, str]) -> None:
    """Stub the scheduler surface _recover_crashed_tasks reads.

    Every task not named in *statuses* reads 'pending'; get_task returns the
    title the lane's metadata carries, so the identity guard genuinely matches.
    """
    harness.scheduler.get_status = AsyncMock(
        side_effect=lambda task_id: statuses.get(task_id, 'pending'),
    )
    harness.scheduler.get_task = AsyncMock(
        side_effect=lambda task_id: {'title': _title(task_id)},
    )
    harness.scheduler.get_tasks = AsyncMock(return_value=[])
    harness.scheduler.is_deterministic = MagicMock(return_value=False)
    harness.event_store = MagicMock()


@dataclass(frozen=True)
class _ShrunkHost:
    harness: Harness
    head: str
    base: Path

    def lane(self, k: int) -> Path:
        return self.base / f'_lane-{k}'

    @property
    def pool(self) -> WarmLanePool:
        pool = self.harness.git_ops.warm_lane_pool
        assert pool is not None
        return pool


async def _shrunk_host(repo: Path, *, statuses: dict[str, str] | None = None) -> _ShrunkHost:
    """Three live lanes acquired at N=3, then a restart at N=1.

    '41' -> _lane-0, '42' -> _lane-1, '43' -> _lane-2; '43' carries a plan
    with completed steps and a WIP commit (``wip43.txt``) on task/43.
    """
    await _add_warm_lane_scripts(repo)
    harness = _build_harness(_make_orch_config(repo, max_concurrent_tasks=3))
    _wire_recovery_scheduler(harness, statuses or {})
    head = await _git(repo, 'rev-parse', 'HEAD')
    base = harness.git_ops.worktree_base

    for task_id in ('41', '42', '43'):
        wt = await harness.git_ops.acquire_warm_lane(
            task_id, head, expected_title=_title(task_id),
        )
        assert isinstance(wt, WorktreeInfo), f'acquire for {task_id} failed: {wt!r}'

    lane2 = base / '_lane-2'
    ta = TaskArtifacts(lane2, TaskArtifacts.meta_root_for(base, lane2.name))
    ta.init('43', _title('43'), 'd')
    ta.write_plan(_make_plan(3, 5, '43'))
    (lane2 / 'wip43.txt').write_text('task 43 work in progress\n')
    await _git(lane2, 'add', 'wip43.txt')
    await _git(lane2, 'commit', '-m', 'wip: task 43')

    harness.git_ops.warm_lane_pool = WarmLanePool(worktree_base=base, size=1)
    return _ShrunkHost(harness=harness, head=head, base=base)


async def _wip_survives_on_task_43(repo: Path) -> bool:
    rc, out, _ = await _run(['git', 'show', 'task/43:wip43.txt'], cwd=repo)
    return rc == 0 and 'task 43 work in progress' in out


# ── Over-cap checkouts are visible ──────────────────────────────────────


@pytest.mark.asyncio
async def test_lane_branch_checkouts_include_over_cap_lane(ig_git_repo: Path):
    host = await _shrunk_host(ig_git_repo)

    checkouts = await host.harness.git_ops.lane_branch_checkouts()

    assert checkouts is not None
    assert checkouts.get('43') == host.lane(2)


# ── Recovery adopts, then the lane drains ───────────────────────────────


@pytest.mark.asyncio
async def test_recovery_adopts_over_cap_lane_which_drains_on_cleanup(
    ig_git_repo: Path, caplog: pytest.LogCaptureFixture,
):
    host = await _shrunk_host(ig_git_repo)
    lane2 = host.lane(2)

    with caplog.at_level(logging.INFO, logger='orchestrator.harness'):
        await host.harness._recover_crashed_tasks()

    pool = host.pool
    assert pool.assignment_for('43') == lane2
    assert pool.state(lane2) is LaneState.ASSIGNED
    assert '43' in host.harness._recovered_plans
    record = host.harness.git_ops._lane_lifecycle.read(lane2)
    assert record is not None
    assert record.state is DurableLaneState.ASSIGNED
    assert pool.lane_paths() == [host.lane(0)]
    assert not [
        r for r in caplog.records
        if 'infra-owned' in r.getMessage() and '_lane-2' in r.getMessage()
    ]

    wt = await host.harness.git_ops.acquire_warm_lane(
        '43', host.head, expected_title=_title('43'),
    )

    assert isinstance(wt, WorktreeInfo)
    assert wt.path == lane2
    assert await _wip_survives_on_task_43(ig_git_repo)

    await host.harness.git_ops.cleanup_worktree(lane2, '43')

    assert pool.state(lane2) is None
    record = host.harness.git_ops._lane_lifecycle.read(lane2)
    assert record is not None
    assert record.state is DurableLaneState.RELEASED
    assert lane2.is_dir()
    assert await _head_is_detached(lane2)


@pytest.mark.asyncio
async def test_recovery_releases_over_cap_lane_of_terminal_task(ig_git_repo: Path):
    host = await _shrunk_host(ig_git_repo, statuses={'43': 'done'})
    lane2 = host.lane(2)

    await host.harness._recover_crashed_tasks()

    pool = host.pool
    assert pool.assignment_for('43') is None
    assert pool.state(lane2) is None
    assert lane2.is_dir()
    assert await _head_is_detached(lane2)
    record = host.harness.git_ops._lane_lifecycle.read(lane2)
    assert record is not None
    assert record.state is DurableLaneState.RELEASED


# ── Startup shrink guard ────────────────────────────────────────────────


def _shrink_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage() for r in caplog.records
        if r.name == 'orchestrator.git_ops'
        and r.levelno == logging.WARNING
        and 'shrink' in r.getMessage().lower()
    ]


@pytest.mark.asyncio
async def test_recovery_warns_once_naming_every_over_cap_lane(
    ig_git_repo: Path, caplog: pytest.LogCaptureFixture,
):
    host = await _shrunk_host(ig_git_repo)

    with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
        await host.harness._recover_crashed_tasks()

    warnings = _shrink_warnings(caplog)
    assert len(warnings) == 1, warnings
    for fragment in ('capacity=1', '_lane-1', '_lane-2', '42', '43'):
        assert fragment in warnings[0]

    report = host.harness.git_ops.warn_on_warm_lane_pool_shrink()

    assert isinstance(report, WarmLanePoolShrink)
    assert [lane.name for lane in report.lanes] == ['_lane-1', '_lane-2']


@pytest.mark.asyncio
async def test_shrink_guard_is_silent_without_over_cap_lanes(
    ig_git_repo: Path, caplog: pytest.LogCaptureFixture,
):
    await _add_warm_lane_scripts(ig_git_repo)
    harness = _build_harness(_make_orch_config(ig_git_repo, max_concurrent_tasks=3))
    head = await _git(ig_git_repo, 'rev-parse', 'HEAD')
    wt = await harness.git_ops.acquire_warm_lane('41', head)
    assert isinstance(wt, WorktreeInfo)

    with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
        report = harness.git_ops.warn_on_warm_lane_pool_shrink()

    assert isinstance(report, WarmLanePoolShrink)
    assert report.lanes == ()
    assert not [r for r in caplog.records if r.name == 'orchestrator.git_ops']


# ── Lane-checkout reconciler sees an over-cap checkout ──────────────────


@pytest.mark.asyncio
async def test_reconcile_repins_live_over_cap_checkout_without_record(ig_git_repo: Path):
    harness = _build_harness(_make_orch_config(ig_git_repo, max_concurrent_tasks=1))
    base = harness.git_ops.worktree_base
    lane4 = base / '_lane-4'
    await _git(ig_git_repo, 'worktree', 'add', str(lane4), '-b', 'task/77', 'main')
    (lane4 / 'wip77.txt').write_text('task 77 work\n')
    await _git(lane4, 'add', 'wip77.txt')
    await _git(lane4, 'commit', '-m', 'wip: task 77')
    harness.scheduler.get_statuses = AsyncMock(return_value=({'77': 'pending'}, None))

    await harness._reconcile_lane_checkouts()

    pool = harness.git_ops.warm_lane_pool
    assert pool is not None
    assert pool.capacity == 1
    assert pool.assignment_for('77') == lane4
