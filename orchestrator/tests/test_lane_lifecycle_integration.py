"""B+H integration gate (mechanisms 1+2) — two-way boundary suite B1-B6.

PRD W11 omega (``plans/worktree-lane-lifecycle-prd.md``). TEST-ONLY gate: the
six behaviors below (gamma acquire-writer, delta crash-recovery reader,
epsilon1 contamination structural prevention, epsilon2 clean-survival) are
already merged. These tests drive the REAL writer (GitOps.acquire_warm_lane),
the REAL reader (Harness._recover_crashed_tasks), and the REAL dashboard
reader (dashboard.data.orchestrator.read_task_artifacts) over a real local
git repo, proving they agree on the durable-record location/format/semantics
through their production code paths — not the mocked-git unit tests already
covered by test_crash_recovery.py::TestRecordDrivenRecovery.

Boundary suite:
    B1 — writer->reader round-trip (adopt on exact git-reality match)
    B2 — crash->quarantine (orphaned admin-entry divergence, 2097/2098)
    B3 — illegal transition escalates (born-at-L2, real EscalationQueue)
    B4 — hostile `git add -A` stages ZERO task-meta (STRUCTURAL, load-bearing
         for task theta's later guard deletion)
    B5 — durable record + .task-meta survive `git clean -xfd` + `checkout -f`
    B6 — dashboard reader resolves a relocated lane (new-then-old)

Each test is expected GREEN once its fixture wiring converges: this task's
impl steps exercise the existing seam rather than add production code.

Module-local fixtures only (no conftest.py edit) — a conftest edit trips
verify.py's has_conftest full-suite fallback for merge-time verify.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventType
from orchestrator.git_ops import GitOps, WorktreeInfo, _run
from orchestrator.harness import Harness
from orchestrator.lane_lifecycle import (
    ESCALATION_SENTINEL_ROLE,
    IllegalLaneTransition,
    LaneLifecycle,
)
from orchestrator.lane_lifecycle import LaneState as DurableLaneState
from orchestrator.warm_lane_pool import LaneState, WarmLanePool

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
    """Temporary git repo with an initial commit + a resolvable warm base.

    Pre-creates the default derived warm-lane base (task 2061 gate) so the
    acquire_warm_lane pre-acquire base-health gate sees WarmBaseHealth.OK,
    and marks `.worktrees/.pool-root` present (task 2099 create-once guard) —
    mirrors test_lane_lifecycle_gitops.py's git_repo / test_warm_lane_
    integration_gate.py's ig_git_repo fixtures.
    """
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    default_base = repo / '.worktrees' / '_merge-verify' / 'target'
    default_base.mkdir(parents=True, exist_ok=True)
    (default_base / '.keep').write_text('warm base sentinel\n')
    (repo / '.worktrees' / '.pool-root').touch()
    return repo


async def _add_warm_lane_scripts(repo: Path, port: int = 39411) -> None:
    """Commit stub seed-warm-lane.sh + setup-worktree-debug-port.sh into repo.

    No real ssh/reify — the seed stub only needs to mkdir target/ and drop a
    marker file so acquire_warm_lane's seed step exits 0.
    """
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


def _make_orch_config(repo: Path, *, max_concurrent_tasks: int = 1) -> OrchestratorConfig:
    """Build a minimal real OrchestratorConfig with the warm-lane pool on."""
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
    """Construct a Harness with heavy constructors patched out.

    GitOps is deliberately left REAL (not patched) — this gate drives the
    real writer/reader seam. Mirrors test_warm_lane_integration_gate.py /
    test_harness_warm_lane_wiring.py's _build_harness.
    """
    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.Scheduler'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        return Harness(config)


async def _get_head(repo: Path) -> str:
    """Return the HEAD commit SHA of the repo."""
    rc, out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    assert rc == 0, f'git rev-parse HEAD failed (rc={rc})'
    return out.strip()


def _make_plan(done: int, total: int, task_id: str = 'test') -> dict:
    """Build a plan dict with the given step-completion counts."""
    steps = []
    for i in range(total):
        steps.append({
            'id': f'step-{i + 1}',
            'description': f'Step {i + 1}',
            'status': 'done' if i < done else 'pending',
            'commit': f'abc{i}' if i < done else None,
        })
    return {'task_id': task_id, 'title': 'Test Task', 'steps': steps}


async def _acquire_lane(
    git_ops: GitOps,
    task_id: str,
    start_ref: str,
    *,
    expected_title: str | None = None,
) -> Path:
    """Acquire a warm lane for *task_id*, asserting success, and return its path."""
    wt = await git_ops.acquire_warm_lane(task_id, start_ref, expected_title=expected_title)
    assert isinstance(wt, WorktreeInfo), f'acquire_warm_lane failed: {wt!r}'
    return wt.path


def _wire_recovery_scheduler(harness: Harness) -> None:
    """Stub the scheduler + event_store surface _recover_crashed_tasks reads.

    get_task returns a title-less (but non-None) dict so the identity guard
    (config.worktree_identity_guard_enabled defaults True) fails OPEN
    (identities_match treats an empty title as a match) instead of deferring.
    """
    harness.scheduler.get_status = AsyncMock(return_value=None)
    harness.scheduler.get_task = AsyncMock(return_value={})
    harness.scheduler.get_tasks = AsyncMock(return_value=[])
    harness.scheduler.is_deterministic = MagicMock(return_value=False)
    harness.event_store = MagicMock()


# ── B1 — writer->reader round-trip (adopt) ──────────────────────────────


@pytest.mark.asyncio
class TestB1WriterReaderRoundTrip:
    """B1: acquire a lane for a task, crash (fresh in-memory pool cache,
    durable record + git worktree persist), recover — the reader must adopt
    the lane by reading the SAME durable record the REAL writer produced.
    """

    async def test_acquire_then_recover_adopts_lane(self, ig_git_repo: Path):
        repo = ig_git_repo
        await _add_warm_lane_scripts(repo)
        harness = _build_harness(_make_orch_config(repo))
        _wire_recovery_scheduler(harness)

        # WRITER: real acquire_warm_lane — real `git worktree add`.
        head = await _get_head(repo)
        lane = await _acquire_lane(harness.git_ops, '42', head, expected_title='B1 task')
        assert lane == harness.git_ops.worktree_base / '_lane-0'

        # Write crashed progress under the sibling .task-meta root.
        meta = TaskArtifacts.meta_root_for(harness.git_ops.worktree_base, lane.name)
        ta = TaskArtifacts(lane, meta)
        ta.init('42', 'B1 task', 'd')
        ta.write_plan(_make_plan(3, 5, '42'))

        # Simulate restart: fresh in-memory pool cache; durable record +
        # git worktree persist on disk.
        harness.git_ops.warm_lane_pool = WarmLanePool(
            worktree_base=harness.git_ops.worktree_base,
            size=harness.git_ops.warm_lane_pool.size,
        )

        # READER: real crash recovery.
        await harness._recover_crashed_tasks()

        pool = harness.git_ops.warm_lane_pool
        assert pool.assignment_for('42') == lane
        assert pool.state(lane) == LaneState.ASSIGNED

        rec = harness.git_ops._lane_lifecycle.read(lane)
        assert rec is not None
        assert rec.state == DurableLaneState.ASSIGNED
        assert rec.task_id == '42'

        assert '42' in harness._recovered_plans
        assert lane.exists()
        assert rec.state != DurableLaneState.QUARANTINED
