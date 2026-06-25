"""G5 B+H boundary gate: warm-lane space-safety integration gate (task η / 1861).

Drives the warm-lane pool end-to-end through ONE GitOps/WarmLanePool instance
and asserts all four space-safety behaviors in one run:

1. THIN recycle — prior divergence freed, same lane recycled (α + β)
2. Pool bounded → REQUEUE — WarmLanePoolExhausted, zero cold worktrees (β)
3. Fault ESCALATES — RuntimeError, lane released FREE, never proceeds (β)
4. Low-disk → reclaim → REQUEUE — WarmLaneDiskPressure (γ/δ/ε)

Convention: reify scripts (seed-warm-lane.sh, warm-lane-disk-guard.sh,
warm-lane-gc.sh, setup-worktree-debug-port.sh) are STUBBED — the gate
verifies dark-factory orchestration wiring (β/ε) given contract-honoring
reify primitives.  Extent-sharing (filefrag/CoW) is a reify-α/filesystem
guarantee; DF tests assert only orchestration-observable THIN-ness (planted
divergence freed + same recycled _lane-0).

All assertions are at the create_worktree seam (the real dispatch entry point)
and via direct acquire_warm_lane/WarmLanePool calls.  The workflow.run() →
WorkflowOutcome mapping is already covered by test_workflow_warm_lane_requeue.py.

Prerequisites: tasks 1859 (β) and 1860 (ε) merged.  This gate is expected to
be GREEN with no new production code; impl steps (2/4/6/8/10) touch
git_ops.py/warm_lane_pool.py only if end-to-end composition surfaces a defect.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import (
    GitOps,
    WarmLaneDiskPressure,
    WarmLanePoolExhausted,
    WarmLaneUnavailable,
    WorktreeInfo,
    _run,
)
from orchestrator.harness import Harness, TaskReport
from orchestrator.merge_queue import _classify_branch_presence
from orchestrator.scheduler import TaskAssignment
from orchestrator.warm_lane_pool import LaneState
from orchestrator.workflow import WorkflowOutcome

# ---------------------------------------------------------------------------
# Repo fixture
# ---------------------------------------------------------------------------


async def _init_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def ig_git_repo(tmp_path: Path) -> Path:
    """Temporary git repo with an initial commit for integration-gate tests."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    return repo


# ---------------------------------------------------------------------------
# Stub installers
# ---------------------------------------------------------------------------


def _write_disk_guard_stubs(scripts_dir: Path) -> None:
    """Write warm-lane-disk-guard.sh and warm-lane-gc.sh stubs into scripts_dir.

    guard stub (warm-lane-disk-guard.sh):
      - Appends "check <argv>" to <repo>/.test_disk_call_log.
      - Reads the first exit code from <repo>/.test_disk_check_exits (one per
        line), pops it, and exits with that code.  Exits 0 if absent/empty.

    gc stub (warm-lane-gc.sh):
      - Appends "reclaim <argv>" to <repo>/.test_disk_call_log.
      - Exits 0 always.

    Mirrors _write_disk_guard_stubs from test_warm_lane_disk_guard.py verbatim.
    """
    guard = scripts_dir / 'warm-lane-disk-guard.sh'
    guard.write_text(
        '#!/usr/bin/env bash\n'
        'set -euo pipefail\n'
        'DIR="$(cd "$(dirname "$0")" && pwd)"\n'
        'ROOT="$(dirname "$DIR")"\n'
        '# Append subcommand + full argv to call log\n'
        'echo "check $*" >> "$ROOT/.test_disk_call_log"\n'
        '# Pop first exit code from sequence file\n'
        'if [ -f "$ROOT/.test_disk_check_exits" ] && [ -s "$ROOT/.test_disk_check_exits" ]; then\n'
        '    rc=$(head -1 "$ROOT/.test_disk_check_exits")\n'
        '    tmpf="${ROOT}/.test_disk_check_exits.tmp"\n'
        '    tail -n +2 "$ROOT/.test_disk_check_exits" > "$tmpf"\n'
        '    mv "$tmpf" "$ROOT/.test_disk_check_exits"\n'
        'else\n'
        '    rc=0\n'
        'fi\n'
        'exit "${rc:-0}"\n'
    )
    guard.chmod(0o755)

    gc = scripts_dir / 'warm-lane-gc.sh'
    gc.write_text(
        '#!/usr/bin/env bash\n'
        'set -euo pipefail\n'
        'DIR="$(cd "$(dirname "$0")" && pwd)"\n'
        'ROOT="$(dirname "$DIR")"\n'
        '# Append subcommand + full argv to call log\n'
        'echo "reclaim $*" >> "$ROOT/.test_disk_call_log"\n'
        'exit 0\n'
    )
    gc.chmod(0o755)


async def _add_all_warm_lane_scripts(
    repo: Path,
    *,
    seed_exit_code: int = 0,
    port: int = 39411,
) -> None:
    """Commit stub seed, debug-port, disk-guard, and gc scripts into repo/scripts/.

    seed stub:
      - Exits seed_exit_code (default 0).
      - On exit 0: creates <lane>/target/seeded.bin.
      - On non-zero: exits with that code immediately.

    All four scripts are committed in one commit so any lane worktree checked
    out from main carries them at <lane>/scripts/seed-warm-lane.sh and
    <lane>/scripts/setup-worktree-debug-port.sh.  The disk-guard/gc scripts
    live at <project_root>/scripts/ (run by git_ops directly, not via lane).
    """
    scripts_dir = repo / 'scripts'
    scripts_dir.mkdir(parents=True, exist_ok=True)

    # Seed stub — configurable exit code; on 0 creates target/seeded.bin
    seed = scripts_dir / 'seed-warm-lane.sh'
    if seed_exit_code == 0:
        seed.write_text(
            '#!/usr/bin/env bash\n'
            'mkdir -p "$2/target"\n'
            'echo "seeded" > "$2/target/seeded.bin"\n'
        )
    else:
        seed.write_text(
            f'#!/usr/bin/env bash\necho "seed failure" >&2\nexit {seed_exit_code}\n'
        )
    seed.chmod(0o755)

    # Debug-port stub
    debug = scripts_dir / 'setup-worktree-debug-port.sh'
    debug.write_text(f'#!/usr/bin/env bash\necho {port}\n')
    debug.chmod(0o755)

    # Disk-guard + gc stubs (at project_root/scripts/, run directly by git_ops)
    _write_disk_guard_stubs(scripts_dir)

    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'add all warm-lane stub scripts'], cwd=repo)


# ---------------------------------------------------------------------------
# Exit-sequence + call-log helpers
# ---------------------------------------------------------------------------


def _write_check_exits(repo: Path, exit_codes: list[int]) -> None:
    """Write a guard-script exit-code sequence to <repo>/.test_disk_check_exits."""
    (repo / '.test_disk_check_exits').write_text(
        '\n'.join(str(c) for c in exit_codes) + '\n'
    )


def _clear_check_exits(repo: Path) -> None:
    """Remove the exits file so the disk-guard stub exits 0 (admit)."""
    exits_file = repo / '.test_disk_check_exits'
    if exits_file.exists():
        exits_file.unlink()


def _read_call_log(repo: Path) -> list[str]:
    """Return all non-empty lines from <repo>/.test_disk_call_log."""
    log = repo / '.test_disk_call_log'
    if not log.exists():
        return []
    return [line for line in log.read_text().splitlines() if line.strip()]


def _subcommands(repo: Path) -> list[str]:
    """Return just the first word of each call-log entry (subcommand names)."""
    return [line.split()[0] for line in _read_call_log(repo)]


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def _make_config(**overrides: Any) -> GitConfig:
    """Build a GitConfig with pool + disk-guard enabled and canonical settings."""
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
        warm_lane_pool=True,
        warm_lane_disk_guard=True,
        warm_lane_min_free_gib=50,
        warm_lane_min_free_inodes=500_000,
        **overrides,
    )


# ---------------------------------------------------------------------------
# HEAD helper
# ---------------------------------------------------------------------------


async def _get_head(repo: Path) -> str:
    """Return the HEAD commit SHA of the repo."""
    rc, out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    assert rc == 0, f'git rev-parse HEAD failed (rc={rc})'
    return out.strip()


# ---------------------------------------------------------------------------
# ω-gate scaffolding: harness helpers
# ---------------------------------------------------------------------------


def _build_harness(config: OrchestratorConfig) -> Harness:
    """Construct a Harness with heavy constructors patched out.

    Mirrors _build_harness from test_harness_warm_lane_wiring.py.
    """
    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.Scheduler'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        return Harness(config)


def _make_orch_config(
    repo: Path,
    *,
    max_concurrent_tasks: int = 1,
) -> OrchestratorConfig:
    """Build a minimal OrchestratorConfig for ω-gate tests.

    Sets warm_lane_pool=True; disk guard stays default False so the seam
    tests don't need to stub disk-guard exit codes.
    """
    return OrchestratorConfig(
        project_root=repo,
        max_concurrent_tasks=max_concurrent_tasks,
        git=GitConfig(warm_lane_pool=True),
    )


async def _run_synthetic_hard_cancel_slot(
    harness: Harness,
    task_id: str,
) -> TaskReport:
    """Drive harness._run_slot with TaskWorkflow.run raising asyncio.CancelledError.

    Simulates the T5 hard-cancel scenario (e.g. soft-cancel poll limit exceeded).
    Returns the synthetic CANCELLED report that the harness except-clause builds.

    The caller must pre-assign a lane for *task_id* (via pool.acquire_for(task_id))
    before calling this helper; the β guard being tested requires the lane to be
    ASSIGNED at _run_slot entry.

    Mirrors the setup in
    TestHardCancelLaneRelease.test_synthetic_hard_cancel_retains_branch_and_lane.
    """
    harness.scheduler.carries_substrate_probe.return_value = False
    harness.scheduler.is_deterministic.return_value = False

    with patch('orchestrator.harness.TaskWorkflow') as MockWorkflow:
        mock_wf = MagicMock()
        mock_wf.run = AsyncMock(side_effect=asyncio.CancelledError())
        MockWorkflow.return_value = mock_wf

        assignment = TaskAssignment(
            task_id=task_id,
            task={
                'id': task_id,
                'title': f'ω-gate synthetic-cancel task {task_id}',
                'status': 'pending',
                'metadata': {},
                'dependencies': [],
            },
            modules=[],
        )
        sem = asyncio.Semaphore(1)
        return await harness._run_slot(assignment, sem)


# ===========================================================================
# Step-1 / Step-2: Behavior 1 — THIN recycle (α + β)
#
# Drive create_worktree at the seam: acquire 'A', plant bloat, release,
# acquire 'B' on the same _lane-0.  Assert:
#   (a) same _lane-0 returned (NOT a cold worktree_base/B dir)
#   (b) bloat.bin gone (rm-before-seed freed prior divergence)
#   (c) target/seeded.bin present (warm re-seed ran)
#
# THIN-ness is asserted as orchestration-observable: planted divergence freed
# + same recycled lane.  Literal filefrag/CoW extent-sharing is a reify-α
# guarantee (external dep reify:4715) and is NOT asserted here.
# ===========================================================================


@pytest.mark.asyncio
class TestThinRecycle:
    """Behavior 1: THIN recycle via create_worktree seam."""

    async def test_recycle_returns_same_lane(self, ig_git_repo: Path) -> None:
        """Recycled acquire returns a WorktreeInfo on the SAME _lane-0 path."""
        await _add_all_warm_lane_scripts(ig_git_repo)
        config = _make_config()
        git_ops = GitOps(config, ig_git_repo, warm_lane_pool_size=1)

        # Phase: first acquire (create-once)
        info_a = await git_ops.create_worktree('A')
        assert isinstance(info_a, WorktreeInfo), (
            f'first create_worktree must return WorktreeInfo, got {info_a!r}'
        )
        lane_path = info_a.path

        # Release (mark FREE) so the lane can be recycled
        assert git_ops.warm_lane_pool is not None
        await git_ops.warm_lane_pool.release(lane_path)

        # Phase: recycle for 'B' — pool re-allocates the FREE _lane-0
        info_b = await git_ops.create_worktree('B')
        assert isinstance(info_b, WorktreeInfo), (
            f'recycle create_worktree must return WorktreeInfo, got {info_b!r}'
        )
        assert info_b.path == lane_path, (
            f'Recycled acquire must return the SAME _lane-0 ({lane_path}), '
            f'got {info_b.path} — cold-created dir is the wrong signal'
        )

    async def test_recycle_frees_prior_divergence_bloat(self, ig_git_repo: Path) -> None:
        """rm-before-seed frees stray target/ divergence planted before the recycle."""
        await _add_all_warm_lane_scripts(ig_git_repo)
        config = _make_config()
        git_ops = GitOps(config, ig_git_repo, warm_lane_pool_size=1)

        # First acquire
        info_a = await git_ops.create_worktree('A')
        assert isinstance(info_a, WorktreeInfo)
        lane_path = info_a.path

        # Plant a stray bloat artifact under target/ to simulate retained divergence
        (lane_path / 'target').mkdir(exist_ok=True)
        bloat = lane_path / 'target' / 'bloat.bin'
        bloat.write_bytes(b'\xde\xad\xbe\xef' * 64)
        assert bloat.exists(), 'test setup: stray artifact must exist before recycle'

        # Release and recycle
        assert git_ops.warm_lane_pool is not None
        await git_ops.warm_lane_pool.release(lane_path)
        info_b = await git_ops.create_worktree('B')
        assert isinstance(info_b, WorktreeInfo), (
            f'recycle must succeed (WorktreeInfo), got {info_b!r}'
        )

        # Orchestration-observable THIN-ness: bloat.bin must be gone
        assert not bloat.exists(), (
            'Stray target/ divergence must be removed by rm-before-seed on recycle '
            '(β THIN guarantee, task-1859)'
        )

    async def test_recycle_produces_fresh_seed(self, ig_git_repo: Path) -> None:
        """Re-seed ran: target/seeded.bin present after the recycle."""
        await _add_all_warm_lane_scripts(ig_git_repo)
        config = _make_config()
        git_ops = GitOps(config, ig_git_repo, warm_lane_pool_size=1)

        # First acquire
        info_a = await git_ops.create_worktree('A')
        assert isinstance(info_a, WorktreeInfo)
        lane_path = info_a.path

        # Release and recycle
        assert git_ops.warm_lane_pool is not None
        await git_ops.warm_lane_pool.release(lane_path)
        info_b = await git_ops.create_worktree('B')
        assert isinstance(info_b, WorktreeInfo)

        # Re-seed must have created target/seeded.bin
        assert (lane_path / 'target' / 'seeded.bin').exists(), (
            'target/seeded.bin must exist after recycle re-seed (seed stub ran)'
        )

    async def test_no_cold_worktree_created_for_recycled_branch(
        self, ig_git_repo: Path,
    ) -> None:
        """worktree_base/'B' must not exist — cold path is unreachable when pool enabled."""
        await _add_all_warm_lane_scripts(ig_git_repo)
        config = _make_config()
        git_ops = GitOps(config, ig_git_repo, warm_lane_pool_size=1)

        info_a = await git_ops.create_worktree('A')
        assert isinstance(info_a, WorktreeInfo)
        lane_path = info_a.path

        assert git_ops.warm_lane_pool is not None
        await git_ops.warm_lane_pool.release(lane_path)
        await git_ops.create_worktree('B')

        cold_b = git_ops.worktree_base / 'B'
        assert not cold_b.exists(), (
            f'Cold worktree_base/B must NOT be created when pool is enabled; '
            f'found {cold_b}'
        )


# ===========================================================================
# Step-3 / Step-4: Behavior 2 — pool bounded → REQUEUE + ZERO cold worktrees (β)
#
# size=1: assign _lane-0 to 'X'; try 'Y' → WarmLanePoolExhausted.
# Assert: (a) WarmLanePoolExhausted raised; (b) no cold worktree_base/Y dir;
# (c) _lane-0 remains ASSIGNED to X.
# ===========================================================================


@pytest.mark.asyncio
class TestPoolBoundedExhausted:
    """Behavior 2: pool exhaustion signals REQUEUE; zero cold dirs."""

    async def test_exhausted_raises_warm_lane_pool_exhausted(
        self, ig_git_repo: Path,
    ) -> None:
        """Second create_worktree on a full pool raises WarmLanePoolExhausted."""
        await _add_all_warm_lane_scripts(ig_git_repo)
        config = _make_config()
        git_ops = GitOps(config, ig_git_repo, warm_lane_pool_size=1)

        # Assign the only lane to 'X'
        info_x = await git_ops.create_worktree('X')
        assert isinstance(info_x, WorktreeInfo), (
            f'first create_worktree must succeed, got {info_x!r}'
        )

        # Pool is now exhausted — 'Y' must raise WarmLanePoolExhausted
        with pytest.raises(WarmLanePoolExhausted):
            await git_ops.create_worktree('Y')

    async def test_exhausted_no_cold_worktree_dir(self, ig_git_repo: Path) -> None:
        """Exhausted path: worktree_base/'Y' must NOT be created (cold path unreachable)."""
        await _add_all_warm_lane_scripts(ig_git_repo)
        config = _make_config()
        git_ops = GitOps(config, ig_git_repo, warm_lane_pool_size=1)

        info_x = await git_ops.create_worktree('X')
        assert isinstance(info_x, WorktreeInfo)

        with pytest.raises(WarmLanePoolExhausted):
            await git_ops.create_worktree('Y')

        cold_y = git_ops.worktree_base / 'Y'
        assert not cold_y.exists(), (
            f'No cold worktree_base/Y dir must exist on pool exhaustion; '
            f'found {cold_y} — the β cold-fall-through guard is broken'
        )

    async def test_exhausted_lane_remains_assigned(self, ig_git_repo: Path) -> None:
        """After exhaustion, the first lane (_lane-0) is still ASSIGNED to 'X'."""
        await _add_all_warm_lane_scripts(ig_git_repo)
        config = _make_config()
        git_ops = GitOps(config, ig_git_repo, warm_lane_pool_size=1)

        info_x = await git_ops.create_worktree('X')
        assert isinstance(info_x, WorktreeInfo)
        lane_path = info_x.path  # _lane-0

        with pytest.raises(WarmLanePoolExhausted):
            await git_ops.create_worktree('Y')

        assert git_ops.warm_lane_pool is not None
        assert git_ops.warm_lane_pool.state(lane_path) == LaneState.ASSIGNED, (
            '_lane-0 must remain ASSIGNED to X after exhaustion (not leaked FREE)'
        )

    async def test_exhausted_only_pool_lanes_under_worktree_base(
        self, ig_git_repo: Path,
    ) -> None:
        """Only _lane-* and _merge-verify dirs exist under worktree_base after exhaustion."""
        await _add_all_warm_lane_scripts(ig_git_repo)
        config = _make_config()
        git_ops = GitOps(config, ig_git_repo, warm_lane_pool_size=1)

        info_x = await git_ops.create_worktree('X')
        assert isinstance(info_x, WorktreeInfo)

        with pytest.raises(WarmLanePoolExhausted):
            await git_ops.create_worktree('Y')

        worktree_base = git_ops.worktree_base
        assert worktree_base.exists(), (
            'worktree_base must exist after create_worktree("X") succeeds'
        )
        dirs = [d.name for d in worktree_base.iterdir() if d.is_dir()]
        non_pool_dirs = [
            d for d in dirs
            if not d.startswith('_lane-') and d != '_merge-verify'
        ]
        assert non_pool_dirs == [], (
            f'Only _lane-* and _merge-verify dirs expected; '
            f'found extra dirs (cold worktrees?): {non_pool_dirs}'
        )


# ===========================================================================
# Step-5 / Step-6: Behavior 3 — fault ESCALATES, never proceeds (β)
#
# Install seed that exits 1.  create_worktree('Z') must raise RuntimeError.
# Pool lane is released FREE on fault (no leaked ASSIGNED lane).
# ===========================================================================


@pytest.mark.asyncio
class TestFaultEscalates:
    """Behavior 3: seed/worktree-add fault → RuntimeError; pool lane released FREE."""

    async def test_fault_raises_runtime_error(self, ig_git_repo: Path) -> None:
        """Failing seed (exit 1) → create_worktree raises RuntimeError (FAULT)."""
        await _add_all_warm_lane_scripts(ig_git_repo, seed_exit_code=1)
        config = _make_config()
        git_ops = GitOps(config, ig_git_repo, warm_lane_pool_size=1)

        with pytest.raises(RuntimeError):
            await git_ops.create_worktree('Z')

    async def test_fault_pool_lane_released_free(self, ig_git_repo: Path) -> None:
        """After fault, _lane-0 is FREE (no leaked ASSIGNED lane)."""
        await _add_all_warm_lane_scripts(ig_git_repo, seed_exit_code=1)
        config = _make_config()
        git_ops = GitOps(config, ig_git_repo, warm_lane_pool_size=1)

        with pytest.raises(RuntimeError):
            await git_ops.create_worktree('Z')

        assert git_ops.warm_lane_pool is not None
        lane_path = git_ops.worktree_base / '_lane-0'
        assert git_ops.warm_lane_pool.state(lane_path) == LaneState.FREE, (
            '_lane-0 must be FREE after FAULT (no leaked ASSIGNED lane)'
        )

    async def test_fault_absent_seed_raises_runtime_error(
        self, ig_git_repo: Path,
    ) -> None:
        """Absent seed-warm-lane.sh (rc=127) → create_worktree raises RuntimeError."""
        # No seed committed — absent script returns rc=127 (FAULT)
        config = _make_config()
        git_ops = GitOps(config, ig_git_repo, warm_lane_pool_size=1)

        # Install only disk-guard scripts (no seed, no debug-port)
        scripts_dir = ig_git_repo / 'scripts'
        scripts_dir.mkdir(parents=True, exist_ok=True)
        _write_disk_guard_stubs(scripts_dir)
        await _run(['git', 'add', '-A'], cwd=ig_git_repo)
        await _run(['git', 'commit', '-m', 'disk-guard stubs only (no seed)'], cwd=ig_git_repo)

        with pytest.raises(RuntimeError):
            await git_ops.create_worktree('Z')

    async def test_fault_does_not_return_worktree_info(self, ig_git_repo: Path) -> None:
        """Fault must never return WorktreeInfo (no degraded-warmth proceed-on-retained)."""
        await _add_all_warm_lane_scripts(ig_git_repo, seed_exit_code=1)
        config = _make_config()
        git_ops = GitOps(config, ig_git_repo, warm_lane_pool_size=1)

        with pytest.raises(RuntimeError):
            await git_ops.create_worktree('Z')


# ===========================================================================
# Step-7 / Step-8: Behavior 4 — low-disk → reclaim → REQUEUE exit-75 (ε)
#
# Disk-guard exits [75, 75]: create_worktree('W') raises WarmLaneDiskPressure.
# Assert: (a) WarmLaneDiskPressure raised; (b) call order check→reclaim→check;
# (c) seed never ran (pool lane FREE, no seeded.bin); (d) --mount/--min-free-*
# args present in check invocations.
# ===========================================================================


@pytest.mark.asyncio
class TestDiskPressureRequeue:
    """Behavior 4: disk-guard [75,75] → reclaim → REQUEUE (WarmLaneDiskPressure)."""

    async def test_still_pressured_raises_disk_pressure(
        self, ig_git_repo: Path,
    ) -> None:
        """check→75, reclaim, recheck→75 → create_worktree raises WarmLaneDiskPressure."""
        await _add_all_warm_lane_scripts(ig_git_repo)
        _write_check_exits(ig_git_repo, [75, 75])
        config = _make_config()
        git_ops = GitOps(config, ig_git_repo, warm_lane_pool_size=1)

        with pytest.raises(WarmLaneDiskPressure):
            await git_ops.create_worktree('W')

    async def test_reclaim_invoked_between_checks(self, ig_git_repo: Path) -> None:
        """Call-log order is check→reclaim→check (δ reclaim was invoked)."""
        await _add_all_warm_lane_scripts(ig_git_repo)
        _write_check_exits(ig_git_repo, [75, 75])
        config = _make_config()
        git_ops = GitOps(config, ig_git_repo, warm_lane_pool_size=1)

        with pytest.raises(WarmLaneDiskPressure):
            await git_ops.create_worktree('W')

        assert _subcommands(ig_git_repo) == ['check', 'reclaim', 'check'], (
            f'Expected check→reclaim→check call order (δ reclaim between γ checks); '
            f'got {_subcommands(ig_git_repo)}'
        )

    async def test_seed_never_ran_lane_free(self, ig_git_repo: Path) -> None:
        """Guard runs before acquire_for: no seeded.bin, pool lane stays FREE."""
        await _add_all_warm_lane_scripts(ig_git_repo)
        _write_check_exits(ig_git_repo, [75, 75])
        config = _make_config()
        git_ops = GitOps(config, ig_git_repo, warm_lane_pool_size=1)

        with pytest.raises(WarmLaneDiskPressure):
            await git_ops.create_worktree('W')

        lane_0 = git_ops.worktree_base / '_lane-0'
        assert not (lane_0 / 'target' / 'seeded.bin').exists(), (
            'seeded.bin must NOT exist — seed must not run before disk-pressure blocks'
        )
        assert git_ops.warm_lane_pool is not None
        assert git_ops.warm_lane_pool.state(lane_0) == LaneState.FREE, (
            '_lane-0 must stay FREE — acquire_for must not run before disk-pressure blocks'
        )

    async def test_check_received_mount_and_thresholds(
        self, ig_git_repo: Path,
    ) -> None:
        """CLI contract: check receives --mount, --min-free-gib, --min-free-inodes."""
        await _add_all_warm_lane_scripts(ig_git_repo)
        _write_check_exits(ig_git_repo, [75, 75])
        config = _make_config()
        git_ops = GitOps(config, ig_git_repo, warm_lane_pool_size=1)

        with pytest.raises(WarmLaneDiskPressure):
            await git_ops.create_worktree('W')

        check_lines = [
            line for line in _read_call_log(ig_git_repo)
            if line.split()[0] == 'check'
        ]
        assert len(check_lines) >= 1, 'Expected at least one check invocation'
        first_check = check_lines[0]

        assert '--mount' in first_check, (
            f'check must receive --mount; got: {first_check!r}'
        )
        parts = first_check.split()
        mount_val = parts[parts.index('--mount') + 1]
        assert mount_val == str(git_ops.worktree_base), (
            f'--mount value must be worktree_base; got {mount_val!r}'
        )
        assert '--min-free-gib' in first_check, (
            f'check must receive --min-free-gib; got: {first_check!r}'
        )
        assert '--min-free-inodes' in first_check, (
            f'check must receive --min-free-inodes; got: {first_check!r}'
        )
        gib_val = parts[parts.index('--min-free-gib') + 1]
        inodes_val = parts[parts.index('--min-free-inodes') + 1]
        assert gib_val == '50', f'--min-free-gib must be 50; got {gib_val!r}'
        assert inodes_val == '500000', f'--min-free-inodes must be 500000; got {inodes_val!r}'

    async def test_reclaim_received_mount(self, ig_git_repo: Path) -> None:
        """CLI contract: gc reclaim stub received --mount <worktree_base>."""
        await _add_all_warm_lane_scripts(ig_git_repo)
        _write_check_exits(ig_git_repo, [75, 75])
        config = _make_config()
        git_ops = GitOps(config, ig_git_repo, warm_lane_pool_size=1)

        with pytest.raises(WarmLaneDiskPressure):
            await git_ops.create_worktree('W')

        reclaim_lines = [
            line for line in _read_call_log(ig_git_repo)
            if line.split()[0] == 'reclaim'
        ]
        assert len(reclaim_lines) == 1, (
            f'Expected exactly one reclaim invocation; got {reclaim_lines}'
        )
        reclaim_line = reclaim_lines[0]
        assert '--mount' in reclaim_line, (
            f'reclaim must receive --mount; got: {reclaim_line!r}'
        )
        parts = reclaim_line.split()
        mount_val = parts[parts.index('--mount') + 1]
        assert mount_val == str(git_ops.worktree_base), (
            f'reclaim --mount must be worktree_base; got {mount_val!r}'
        )


# ===========================================================================
# Step-9 / Step-10: G5 combined gate — all four behaviors on ONE GitOps
#
# ONE GitOps + WarmLanePool (size=1, warm_lane_disk_guard=True throughout).
# Phases A-C admit (exits file absent/0); Phase D blocks ([75,75]).
# Lane is released between phases so each starts clean.
#
# Phase A: THIN recycle — acquire 'A', plant bloat, release, recycle 'B'
#           → bloat gone, same _lane-0, release
# Phase B: exhausted — acquire 'X', try 'Y' → WarmLanePoolExhausted + no cold
#           dir, release 'X'
# Phase C: fault — commit failing seed, create 'Z' → RuntimeError, lane FREE
# Phase D: disk-guard [75,75] — create 'W' → WarmLaneDiskPressure,
#           call-log delta = check→reclaim→check
# ===========================================================================


@pytest.mark.asyncio
class TestG5Gate:
    """G5 release criterion: all four behaviors on ONE shared GitOps/WarmLanePool."""

    async def test_all_four_behaviors_on_single_pool(
        self, ig_git_repo: Path,
    ) -> None:
        """Prove all four space-safety behaviors compose on a single shared pool."""
        await _add_all_warm_lane_scripts(ig_git_repo)
        config = _make_config()
        git_ops = GitOps(config, ig_git_repo, warm_lane_pool_size=1)
        assert git_ops.warm_lane_pool is not None

        # ── Phase A: THIN recycle ────────────────────────────────────────────
        info_a = await git_ops.create_worktree('A')
        assert isinstance(info_a, WorktreeInfo), (
            f'Phase A: first acquire must return WorktreeInfo, got {info_a!r}'
        )
        lane_path = info_a.path
        # Cross-phase invariant: the SAME _lane-0 is reused across all phases
        # (implicitly the only lane in a size-1 pool).

        # Plant divergence artifact
        (lane_path / 'target').mkdir(exist_ok=True)
        bloat = lane_path / 'target' / 'bloat.bin'
        bloat.write_bytes(b'\xde\xad\xbe\xef' * 64)

        # Release and recycle
        await git_ops.warm_lane_pool.release(lane_path)

        info_b = await git_ops.create_worktree('B')
        assert isinstance(info_b, WorktreeInfo), (
            f'Phase A: recycle acquire must return WorktreeInfo, got {info_b!r}'
        )
        assert info_b.path == lane_path, (
            f'Phase A: recycled acquire must return same _lane-0, got {info_b.path}'
        )
        assert not bloat.exists(), (
            'Phase A: stray target/bloat.bin must be gone after rm-before-seed (β THIN)'
        )
        # (seeded.bin freshness is already asserted by TestThinRecycle.test_recycle_produces_fresh_seed)

        # Release B's assignment before Phase B
        await git_ops.warm_lane_pool.release(lane_path)

        # ── Phase B: pool exhausted → WarmLanePoolExhausted + ZERO cold dir ─
        info_x = await git_ops.create_worktree('X')
        assert isinstance(info_x, WorktreeInfo), (
            f'Phase B: first acquire must return WorktreeInfo, got {info_x!r}'
        )
        assert info_x.path == lane_path, (
            f'Phase B: expected _lane-0, got {info_x.path}'
        )

        with pytest.raises(WarmLanePoolExhausted):
            await git_ops.create_worktree('Y')

        cold_y = git_ops.worktree_base / 'Y'
        assert not cold_y.exists(), (
            'Phase B: cold worktree_base/Y must NOT be created on exhaustion; '
            'β cold-fall-through guard broken'
        )
        # (ASSIGNED state invariant already asserted by TestPoolBoundedExhausted.test_exhausted_lane_remains_assigned)

        # Release X before Phase C
        await git_ops.warm_lane_pool.release(lane_path)

        # ── Phase C: fault → RuntimeError; pool lane FREE ───────────────────
        # Commit a failing seed script to the repo so reset-in-place picks it up
        fail_seed = ig_git_repo / 'scripts' / 'seed-warm-lane.sh'
        fail_seed.write_text(
            '#!/usr/bin/env bash\necho "forced fault" >&2\nexit 1\n'
        )
        fail_seed.chmod(0o755)
        await _run(['git', 'add', '-A'], cwd=ig_git_repo)
        await _run(['git', 'commit', '-m', 'phase-C: fault seed exits 1'], cwd=ig_git_repo)

        with pytest.raises(RuntimeError):
            await git_ops.create_worktree('Z')

        # Lane must be FREE after fault (auto-released by acquire_warm_lane)
        assert git_ops.warm_lane_pool.state(lane_path) == LaneState.FREE, (
            'Phase C: _lane-0 must be FREE after FAULT (no leaked ASSIGNED lane)'
        )

        # ── Phase D: disk-guard [75,75] → reclaim → WarmLaneDiskPressure ───
        # Capture call-log baseline (Phases A-C had disk-guard checks too)
        log_before_d = _read_call_log(ig_git_repo)

        _write_check_exits(ig_git_repo, [75, 75])

        with pytest.raises(WarmLaneDiskPressure):
            await git_ops.create_worktree('W')

        # Pool lane must stay FREE (guard ran before acquire_for)
        assert git_ops.warm_lane_pool.state(lane_path) == LaneState.FREE, (
            'Phase D: _lane-0 must stay FREE — guard blocks before acquire_for'
        )

        # Phase-D-specific call-log delta must be check→reclaim→check
        new_entries = _read_call_log(ig_git_repo)[len(log_before_d):]
        new_subcmds = [e.split()[0] for e in new_entries]
        assert new_subcmds == ['check', 'reclaim', 'check'], (
            f'Phase D: expected call-log delta check→reclaim→check; '
            f'got {new_subcmds} (full new entries: {new_entries})'
        )


# ===========================================================================
# Step-3: RED — release_lane_for_terminal_task end-to-end seam (tests 8 & 9)
# ===========================================================================


@pytest.mark.asyncio
class TestTerminalLaneRelease:
    """End-to-end seam: release_lane_for_terminal_task frees the lane for reuse.

    Tests 8 and 9 pin the shared primitive from the integration-gate perspective.
    All fail until release_lane_for_terminal_task is added to GitOps (step-4).
    """

    async def test_terminal_release_frees_lane_end_to_end(
        self, ig_git_repo: Path,
    ) -> None:
        """create_worktree('A')→ASSIGNED, release via primitive, re-acquire gets same _lane-0."""
        await _add_all_warm_lane_scripts(ig_git_repo)
        config = _make_config()
        git_ops = GitOps(config, ig_git_repo, warm_lane_pool_size=1)

        # First acquire
        info_a = await git_ops.create_worktree('A')
        assert isinstance(info_a, WorktreeInfo)
        lane_path = info_a.path

        # Write plan.json (mirrors what the workflow does)
        task_dir = lane_path / '.task'
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / 'plan.json').write_text('{"task_id": "A"}')

        # Release via the shared primitive
        assert git_ops.warm_lane_pool is not None
        freed = await git_ops.release_lane_for_terminal_task('A')
        assert freed is True
        assert git_ops.warm_lane_pool.state(lane_path) == LaneState.FREE

        # Fresh create_worktree must reuse the SAME _lane-0
        info_b = await git_ops.create_worktree('B')
        assert isinstance(info_b, WorktreeInfo)
        assert info_b.path == lane_path, (
            f'Fresh acquire after terminal release must reuse {lane_path}, got {info_b.path}'
        )

    async def test_cancelled_then_release_then_reacquire(
        self, ig_git_repo: Path,
    ) -> None:
        """Release (simulating cancelled task) → next create_worktree('B') succeeds."""
        await _add_all_warm_lane_scripts(ig_git_repo)
        config = _make_config()
        git_ops = GitOps(config, ig_git_repo, warm_lane_pool_size=1)

        # Exhaust the pool
        info_a = await git_ops.create_worktree('A')
        assert isinstance(info_a, WorktreeInfo)

        # Release via primitive (simulating CANCELLED terminal exit)
        freed = await git_ops.release_lane_for_terminal_task('A')
        assert freed is True

        # Now create_worktree('B') must succeed rather than raising WarmLanePoolExhausted
        info_b = await git_ops.create_worktree('B')
        assert isinstance(info_b, WorktreeInfo), (
            f'After terminal release, create_worktree must succeed, got {info_b!r}'
        )


# ===========================================================================
# ω-gate: Step-1 (RED β seam) + Step-3 (RED α seam) + Step-5 (RED γ seam) +
#          Step-7 (RED merge seam) + Step-9 (RED combined gate)
#
# One class — TestOmegaIntegrationGate — hosts all ω seam tests plus the
# combined ω gate.  Each seam gets a distinct observable so the gate FAILS
# if ANY of α/β/γ is reverted.  The combined gate is the G2 deliverable.
# ===========================================================================


@pytest.mark.asyncio
class TestOmegaIntegrationGate:
    """ω (task 1915) integration gate: restart mid-pre-merge-rebase → branch
    survives release+restart → merge completes (no unknown_branch), composing α/β/γ.

    Negative-control design:
      β observable: lane stays ASSIGNED after synthetic hard-cancel (B2 skipped)
      α observable: branch retained after real reclaim release (commits > main)
      γ observable: commits preserved on re-dispatch reattach (count == n_before)
      merge: _classify_branch_presence is None AND merge_to_main succeeds
    """

    # -----------------------------------------------------------------------
    # Step-1: RED β seam
    # -----------------------------------------------------------------------

    async def test_beta_synthetic_cancel_lane_stays_assigned(
        self, ig_git_repo: Path,
    ) -> None:
        """β seam: hard-cancel leaves lane ASSIGNED and branch alive.

        Phase 0: create_worktree(id) → lane on task/<id>; commit unmerged WIP.
        Phase 1 (β): synthetic hard-cancel via harness._run_slot.
        ASSERT:
          - report.outcome == CANCELLED and report.synthetic_cancel is True
          - pool.assignment_for(id) is not None AND pool.state(lane) == ASSIGNED
          - git rev-parse --verify refs/heads/task/<id> rc == 0

        Negative control for β: with β reverted, B2 fires release_lane_for_terminal_task
        → lane flips FREE → the ASSIGNED assertion fails.
        """
        task_id = 'omega-beta-1'
        await _add_all_warm_lane_scripts(ig_git_repo)
        config = _make_orch_config(ig_git_repo)
        harness = _build_harness(config)
        pool = harness.git_ops.warm_lane_pool
        assert pool is not None

        # Phase 0: acquire lane + commit unmerged WIP
        info = await harness.git_ops.create_worktree(task_id)
        assert isinstance(info, WorktreeInfo), (
            f'Phase 0: create_worktree must return WorktreeInfo, got {info!r}'
        )
        lane = info.path
        (lane / 'wip_beta.txt').write_text('beta wip content\n')
        await harness.git_ops.commit(lane, 'wip: beta seam unmerged work')

        # Verify WIP is beyond main
        rc, count, _ = await _run(
            ['git', 'rev-list', '--count', f'main..task/{task_id}'],
            cwd=ig_git_repo,
        )
        assert rc == 0 and int(count.strip()) > 0, (
            f'setup: task/{task_id} must have commits beyond main, got count={count!r}'
        )

        # Phase 1 (β): synthetic hard-cancel via harness._run_slot
        report = await _run_synthetic_hard_cancel_slot(harness, task_id)

        # β assertions
        assert report is not None
        assert report.outcome == WorkflowOutcome.CANCELLED, (
            f'synthetic hard-cancel must produce CANCELLED outcome, got {report.outcome!r}'
        )
        assert report.synthetic_cancel is True, (
            'hard-cancel must produce synthetic_cancel=True (β field, harness.py:3911)'
        )

        # Lane stays ASSIGNED (B2 skipped by β guard)
        assert pool.assignment_for(task_id) is not None, (
            'lane assignment must survive hard-cancel (B2 β-guard skipped release)'
        )
        assert pool.state(lane) == LaneState.ASSIGNED, (
            f'lane must stay ASSIGNED after synthetic hard-cancel, got {pool.state(lane)!r}; '
            'β-revert → B2 fires release_lane_for_terminal_task → lane flips FREE'
        )

        # Branch alive after synthetic cancel
        rc_ref, _, _ = await _run(
            ['git', 'rev-parse', '--verify', f'refs/heads/task/{task_id}'],
            cwd=ig_git_repo,
        )
        assert rc_ref == 0, (
            f'task/{task_id} branch must survive hard-cancel (β skips B2 branch delete)'
        )

    # -----------------------------------------------------------------------
    # Step-3: RED α seam
    # -----------------------------------------------------------------------

    async def test_alpha_reclaim_release_retains_branch(
        self, ig_git_repo: Path,
    ) -> None:
        """α seam: branch with commits beyond main survives the real reclaim release.

        Drives release_lane_for_terminal_task (the α-path reclaim β defers to) on
        a lane whose task/<id> branch has an unmerged commit.  Asserts:
          - release_lane_for_terminal_task returns True (reclaim succeeded)
          - git rev-parse --verify refs/heads/task/<id> rc == 0 (branch RETAINED)
          - pool.state(lane) == FREE (cache reclaimed, no leak)

        Negative control for α: with α reverted, release_warm_lane runs an
        unconditional git branch -D task/<id> → ref deleted → rev-parse FAILS.
        """
        task_id = 'omega-alpha-1'
        await _add_all_warm_lane_scripts(ig_git_repo)
        config = _make_orch_config(ig_git_repo)
        harness = _build_harness(config)
        git_ops = harness.git_ops
        pool = git_ops.warm_lane_pool
        assert pool is not None

        # Acquire lane + commit unmerged WIP (1 commit beyond main)
        info = await git_ops.create_worktree(task_id)
        assert isinstance(info, WorktreeInfo)
        lane = info.path
        (lane / 'wip_alpha.txt').write_text('alpha wip content\n')
        await git_ops.commit(lane, 'wip: alpha seam unmerged work')

        rc, count, _ = await _run(
            ['git', 'rev-list', '--count', f'main..task/{task_id}'],
            cwd=ig_git_repo,
        )
        assert rc == 0 and int(count.strip()) == 1, (
            f'setup: must have exactly 1 commit beyond main, got {count!r}'
        )

        # Drive the reclaim release (the path β defers to)
        freed = await git_ops.release_lane_for_terminal_task(task_id)
        assert freed is True, (
            'release_lane_for_terminal_task must return True on successful reclaim'
        )

        # α assertion: branch RETAINED through the real release
        rc_ref, _, _ = await _run(
            ['git', 'rev-parse', '--verify', f'refs/heads/task/{task_id}'],
            cwd=ig_git_repo,
        )
        assert rc_ref == 0, (
            f'task/{task_id} branch must survive release_lane_for_terminal_task '
            f'when it has commits beyond main (α guard in _delete_branch_if_on_main); '
            f'α-revert → unconditional git branch -D → ref deleted → FAIL'
        )

        # Cache reclaimed: lane must be FREE (no leak)
        assert pool.state(lane) == LaneState.FREE, (
            f'lane must be FREE after release_lane_for_terminal_task, '
            f'got {pool.state(lane)!r} (cache leak if still ASSIGNED)'
        )

    # -----------------------------------------------------------------------
    # Step-5: RED γ seam
    # -----------------------------------------------------------------------

    async def test_gamma_reattach_preserves_commits(
        self, ig_git_repo: Path,
    ) -> None:
        """γ seam: re-dispatch acquire reattaches with commits preserved.

        Continues from α-released state (lane FREE + registered, task/<id>
        orphaned with 1 commit beyond main).  Re-dispatch must REATTACH the
        existing task/<id> branch, preserving all commits.

        ASSERT (reattach, not reset/collide):
          - isinstance(info, WorktreeInfo) — no collision/fault
          - lane HEAD on task/<id> (re-attached, not detached/reset-to-main)
          - rev-list count == n_before (commits preserved across reattach+rebase)
          - WIP content reachable from task/<id>

        Negative control for γ:
          - reset-in-place revert → checkout -f -B resets to main → count 0 → FAIL
          - create-once revert → worktree add -b collides → RuntimeError → not WorktreeInfo → FAIL
        """
        task_id = 'omega-gamma-1'
        await _add_all_warm_lane_scripts(ig_git_repo)
        config = _make_orch_config(ig_git_repo)
        harness = _build_harness(config)
        git_ops = harness.git_ops
        pool = git_ops.warm_lane_pool
        assert pool is not None

        # Acquire lane + commit WIP (1 commit beyond main)
        info = await git_ops.create_worktree(task_id)
        assert isinstance(info, WorktreeInfo)
        lane = info.path
        wip_file = f'wip_gamma_{task_id}.txt'
        wip_content = 'gamma wip reattach content\n'
        (lane / wip_file).write_text(wip_content)
        await git_ops.commit(lane, f'wip: gamma seam unmerged work for {task_id}')

        # Capture baseline count
        rc, count_str, _ = await _run(
            ['git', 'rev-list', '--count', f'main..task/{task_id}'],
            cwd=ig_git_repo,
        )
        assert rc == 0
        n_before = int(count_str.strip())
        assert n_before > 0, (
            f'setup: task/{task_id} must have commits beyond main, got {n_before}'
        )

        # α-release: reclaim cache, branch orphaned
        freed = await git_ops.release_lane_for_terminal_task(task_id)
        assert freed is True
        assert pool.state(lane) == LaneState.FREE

        # γ: re-dispatch acquire — must REATTACH
        info2 = await git_ops.create_worktree(task_id)
        assert isinstance(info2, WorktreeInfo), (
            f'γ: re-dispatch must return WorktreeInfo (reattach path), got {info2!r}; '
            f'γ-revert create-once → worktree add -b collides → RuntimeError'
        )

        # Lane HEAD on task/<id> (not detached/reset-to-main)
        rc_br, branch_out, _ = await _run(
            ['git', '-C', str(info2.path), 'rev-parse', '--abbrev-ref', 'HEAD'],
            cwd=ig_git_repo,
        )
        assert rc_br == 0
        assert branch_out.strip() == f'task/{task_id}', (
            f'γ: lane HEAD must be on task/{task_id} after reattach, '
            f'got {branch_out.strip()!r}; γ-revert → HEAD detached/at-main → FAIL'
        )

        # Commits preserved (count == n_before)
        rc2, count_after, _ = await _run(
            ['git', 'rev-list', '--count', f'main..task/{task_id}'],
            cwd=ig_git_repo,
        )
        assert rc2 == 0
        n_after = int(count_after.strip())
        assert n_after == n_before, (
            f'γ: commits must be preserved across reattach+rebase '
            f'(before={n_before}, after={n_after}); '
            f'γ-revert reset-in-place → checkout -f -B resets to main → count 0 → FAIL'
        )

        # WIP content reachable from task/<id>
        rc3, show_out, _ = await _run(
            ['git', 'show', f'task/{task_id}:{wip_file}'],
            cwd=ig_git_repo,
        )
        assert rc3 == 0 and wip_content.strip() in show_out, (
            f'γ: WIP content must be reachable from task/{task_id} after reattach; '
            f'got rc={rc3!r}, out={show_out!r}'
        )
