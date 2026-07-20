"""Warm-lane reseed-contamination verification (task 2854).

A fresh-reseed warm-lane acquire (the ``RECYCLE`` / ``CREATE_ONCE_FRESH``
routes) must GUARANTEE + VERIFY a clean reseed — the lane's checked-out
branch is at the base with ZERO retained prior-occupant commits — BEFORE the
lane is handed out for dispatch. Reify incident 2026-07-20: ``_lane-12`` was
acquired for task 5279 but its ``task/5279`` branch still sat at task 5264's
commits; the rebase-collapse ``BranchResetError`` guard caught it downstream,
but a lane serving another task's tree is a reseed-consistency defect that
must be faulted at acquire time (re-acquire a DIFFERENT lane) rather than
relied on the collapse guard to catch late.

Self-contained, mirroring ``test_lane_lifecycle_gitops.py`` /
``test_warm_lane_abort_teardown.py``: ``conftest.py`` provides no shared
``git_repo`` fixture, so this module owns its own real-git-repo fixture +
stub warm-lane scripts (copied verbatim from ``test_lane_lifecycle_gitops.py``)
rather than depending on another test module's module-level helpers.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from orchestrator.config import GitConfig
from orchestrator.git_ops import (
    GitOps,
    WarmLaneUnavailable,
    WorktreeInfo,
    _run,
)

# Two distinct LaneState enums: the pure in-memory pool state machine
# (FREE/ASSIGNED) that ``warm_lane_pool.state()`` returns, and the durable
# LaneLifecycle record state (SEED/REGISTERED/ASSIGNED/…) that
# ``_lane_lifecycle.read()`` returns. Alias both so the assertions below read
# against the correct one.
from orchestrator.lane_lifecycle import LaneState as RecordLaneState
from orchestrator.warm_lane_pool import LaneState as PoolLaneState

# ---------------------------------------------------------------------------
# Repo fixture (copied from test_lane_lifecycle_gitops.py — self-contained)
# ---------------------------------------------------------------------------


async def _init_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Temporary git repo with an initial commit + a resolvable warm base.

    Pre-creates the default derived warm-lane base (task 2061 gate) so the
    acquire_warm_lane pre-acquire base-health gate sees WarmBaseHealth.OK,
    and marks ``.worktrees/.pool-root`` present (task 2099 create-once guard)
    — mirrors test_lane_lifecycle_gitops.py's git_repo fixture.
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


def _warm_config(**overrides) -> GitConfig:
    """Build a GitConfig with the warm-lane pool enabled (canonical settings)."""
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
        warm_lane_pool=True,
        **overrides,
    )


async def _get_head(repo: Path) -> str:
    """Return the HEAD commit SHA of the repo."""
    rc, out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    assert rc == 0, f'git rev-parse HEAD failed (rc={rc})'
    return out.strip()


async def _add_lane_worktree(repo: Path, lane: Path, branch: str, base: str) -> None:
    """Create a lane worktree checked out on *branch* at *base*."""
    rc, _, err = await _run(
        ['git', 'worktree', 'add', '-b', branch, str(lane), base], cwd=repo,
    )
    assert rc == 0, f'git worktree add failed (rc={rc}): {err}'


# ---------------------------------------------------------------------------
# step-1: RED — GitOps._reseed_verified_clean predicate (not yet implemented)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReseedVerifiedCleanPredicate:
    """``_reseed_verified_clean(lane, full_branch, base_ref)`` returns True iff
    the lane's HEAD is on *full_branch* AND carries ZERO commits beyond
    *base_ref*. Fail-closed: any ambiguity (detached HEAD, wrong branch, git
    error) returns False so a lane that cannot be PROVEN clean is never
    dispatched onto."""

    async def test_clean_reseed_at_base_returns_true(
        self, git_repo: Path, tmp_path: Path,
    ):
        base = await _get_head(git_repo)
        lane = tmp_path / 'lane_clean'
        await _add_lane_worktree(git_repo, lane, 'task/NEW', base)
        git_ops = GitOps(_warm_config(), git_repo)

        assert await git_ops._reseed_verified_clean(lane, 'task/NEW', base) is True

    async def test_one_commit_beyond_base_returns_false(
        self, git_repo: Path, tmp_path: Path,
    ):
        """The incident shape: the lane's branch carries one retained commit
        beyond the base it was supposed to be reset to."""
        base = await _get_head(git_repo)
        lane = tmp_path / 'lane_dirty'
        await _add_lane_worktree(git_repo, lane, 'task/NEW', base)
        # One extra commit on task/NEW in the lane → HEAD is 1 beyond base.
        (lane / 'retained.txt').write_text('prior occupant work\n')
        await _run(['git', 'add', '-A'], cwd=lane)
        await _run(['git', 'commit', '-m', 'retained prior commit'], cwd=lane)
        git_ops = GitOps(_warm_config(), git_repo)

        assert await git_ops._reseed_verified_clean(lane, 'task/NEW', base) is False

    async def test_detached_head_returns_false(
        self, git_repo: Path, tmp_path: Path,
    ):
        base = await _get_head(git_repo)
        lane = tmp_path / 'lane_detached'
        await _add_lane_worktree(git_repo, lane, 'task/NEW', base)
        await _run(['git', 'checkout', '--detach'], cwd=lane)
        git_ops = GitOps(_warm_config(), git_repo)

        assert await git_ops._reseed_verified_clean(lane, 'task/NEW', base) is False

    async def test_different_branch_returns_false(
        self, git_repo: Path, tmp_path: Path,
    ):
        """A lane checked out on a branch OTHER than the one the reseed was
        supposed to switch to is not a verified clean reseed."""
        base = await _get_head(git_repo)
        lane = tmp_path / 'lane_other_branch'
        await _add_lane_worktree(git_repo, lane, 'task/OTHER', base)
        git_ops = GitOps(_warm_config(), git_repo)

        assert await git_ops._reseed_verified_clean(lane, 'task/NEW', base) is False

    async def test_bogus_base_ref_fails_closed_false(
        self, git_repo: Path, tmp_path: Path,
    ):
        """Fail-closed: an unresolvable base_ref makes the rev-list error, and
        an error must return False (cannot prove clean → do not dispatch),
        never fail-open to True."""
        base = await _get_head(git_repo)
        lane = tmp_path / 'lane_bogus_base'
        await _add_lane_worktree(git_repo, lane, 'task/NEW', base)
        git_ops = GitOps(_warm_config(), git_repo)

        assert await git_ops._reseed_verified_clean(
            lane, 'task/NEW', 'nonexistent-ref-deadbeefdeadbeef',
        ) is False


# ---------------------------------------------------------------------------
# step-3: RED — acquire-path wiring (RESEED_CONTAMINATED sentinel + gate)
# ---------------------------------------------------------------------------


async def _contaminating_reset(lane_dir, full_branch, target_commit) -> None:
    """Stand-in for :meth:`GitOps._reset_warm_lane` that reproduces the reify
    incident shape: instead of resetting the lane to base, it leaves
    *full_branch* checked out 1 commit BEYOND main (a retained prior-occupant
    commit). The reset itself "succeeds" (never raises), so the recycle path
    proceeds to the shared tail exactly as it would after a real — but
    contaminated — reseed."""
    await _run(['git', 'checkout', '-f', '-B', full_branch, 'main'], cwd=lane_dir)
    (Path(lane_dir) / 'prior_occupant.txt').write_text('retained prior task work\n')
    await _run(['git', 'add', '-A'], cwd=lane_dir)
    await _run(
        ['git', 'commit', '-m', 'retained prior-occupant commit (contamination)'],
        cwd=lane_dir,
    )


async def _contaminating_seed(lane_dir, mode, *_args, **_kwargs) -> int:
    """Stand-in for :meth:`GitOps._seed_warm_lane` on the CREATE_ONCE_FRESH
    route (the OTHER fresh-reseed route the gate guards). The real seed
    CoW-copies target/ and returns 0 WITHOUT committing, so a genuinely-clean
    ``git worktree add -b <full_branch> <lane> <start_ref>`` leaves full_branch
    AT base. This stub instead leaves the freshly-created full_branch 1 commit
    BEYOND base (the reify incident shape) while still "succeeding" (rc 0):
    the create-once path already checked the lane out on full_branch via
    ``add -b``, so a commit here advances that branch. The seed-rc==0 return
    lets the acquire proceed into the shared-tail reseed gate exactly as a
    real-but-contaminated fresh checkout would."""
    lane = Path(lane_dir)
    (lane / 'prior_occupant.txt').write_text('retained prior task work\n')
    await _run(['git', 'add', '-A'], cwd=lane)
    await _run(
        ['git', 'commit', '-m', 'retained prior-occupant commit (contamination)'],
        cwd=lane,
    )
    return 0


@pytest.mark.asyncio
class TestAcquireReseedContaminationGate:
    """The fresh-reseed acquire routes (RECYCLE / CREATE_ONCE_FRESH) must
    verify the reseed landed clean BEFORE returning a dispatchable
    WorktreeInfo. A contaminated reseed faults to
    ``WarmLaneUnavailable.RESEED_CONTAMINATED`` (lane released FREE, never
    ASSIGNED) instead of dispatching onto the stale tree; a genuinely-clean
    recycle is unaffected (no false positive)."""

    async def test_contaminated_recycle_faults_and_releases_lane(
        self, git_repo: Path, caplog: pytest.LogCaptureFixture,
    ):
        await _add_warm_lane_scripts(git_repo)
        git_ops = GitOps(_warm_config(), git_repo, warm_lane_pool_size=1)
        assert git_ops.warm_lane_pool is not None
        start_ref = await _get_head(git_repo)

        # Seed the pool's only lane via a normal create-once acquire, then
        # bare-release it back to FREE (still a registered worktree on disk) —
        # exactly the state a recycled lane is found in by the next acquire.
        info_old = await git_ops.acquire_warm_lane('OLD', start_ref)
        assert isinstance(info_old, WorktreeInfo), f'seed acquire failed: {info_old!r}'
        lane = info_old.path
        await git_ops.warm_lane_pool.release(lane)

        # Force the reset-in-place recycle to leave task/NEW 1 commit over
        # main (the incident shape) instead of resetting to base.
        with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'), \
                patch.object(
                    git_ops, '_reset_warm_lane',
                    AsyncMock(side_effect=_contaminating_reset),
                ):
            result = await git_ops.acquire_warm_lane('NEW', start_ref)

        # (a) The sentinel, NOT a dispatchable WorktreeInfo.
        assert result is WarmLaneUnavailable.RESEED_CONTAMINATED, (
            f'Expected RESEED_CONTAMINATED sentinel; got {result!r}'
        )
        assert not isinstance(result, WorktreeInfo)

        # (b) The lane is released back to FREE — never handed out ASSIGNED
        # for the contaminating task, so nothing is dispatched onto the stale
        # tree. The pool FREE state is the dispatch-gating invariant (matches
        # the abort-teardown tests). A bare pool release leaves the durable
        # record reflecting the PRIOR 'OLD' assignment; the gate never called
        # _note_assigned_via_route for 'NEW', so the record's task_id must
        # never be the contaminating task.
        assert git_ops.warm_lane_pool.state(lane) == PoolLaneState.FREE, (
            'contaminated lane must be released FREE, not left ASSIGNED'
        )
        record = git_ops._lane_lifecycle.read(lane)
        assert record is None or record.task_id != 'NEW', (
            f'lane must never be ASSIGNED to the contaminating task NEW; got {record!r}'
        )

        # (c) A WARNING naming the reseed contamination (data-integrity signal).
        low = caplog.text.lower()
        assert 'reseed' in low and 'contaminat' in low, (
            f'expected a reseed-contamination WARNING; caplog:\n{caplog.text}'
        )

    async def test_contaminated_create_once_fresh_faults_and_releases_lane(
        self, git_repo: Path, caplog: pytest.LogCaptureFixture,
    ):
        """Route parity for the OTHER fresh-reseed route the gate guards.

        The gate condition is ``route in (RECYCLE, CREATE_ONCE_FRESH)``; the
        recycle case above covers reset-in-place of a released lane, this one
        covers the INITIAL create-once acquire (``git worktree add -b``, no
        prior occupant). A contaminated fresh checkout — full_branch left 1
        commit over base — must fault to the SAME
        ``WarmLaneUnavailable.RESEED_CONTAMINATED`` sentinel with the lane
        released FREE, so a future narrowing of that tuple to only RECYCLE
        would be caught here (mirrors
        test_contaminated_recycle_faults_and_releases_lane)."""
        await _add_warm_lane_scripts(git_repo)
        git_ops = GitOps(_warm_config(), git_repo, warm_lane_pool_size=1)
        assert git_ops.warm_lane_pool is not None
        start_ref = await _get_head(git_repo)

        # First acquire on a brand-new pool: the sole lane does not exist on
        # disk yet, so acquire routes through create-once (CREATE_ONCE_FRESH) —
        # `git worktree add -b task/NEW <lane> start_ref` then seed. Patch the
        # seed to leave the just-created branch 1 commit over base (the incident
        # shape) instead of a clean CoW seed at base.
        seed_mock = AsyncMock(side_effect=_contaminating_seed)
        with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'), \
                patch.object(git_ops, '_seed_warm_lane', seed_mock):
            result = await git_ops.acquire_warm_lane('NEW', start_ref)

        # The patched seed ran → the create-once route was genuinely taken (not
        # a reuse/reattach path), so this exercises CREATE_ONCE_FRESH. Recover
        # the lane path from the seed call (its first positional arg).
        assert seed_mock.await_args is not None, 'seed (create-once route) never ran'
        lane = Path(seed_mock.await_args.args[0])

        # (a) The sentinel, NOT a dispatchable WorktreeInfo.
        assert result is WarmLaneUnavailable.RESEED_CONTAMINATED, (
            f'Expected RESEED_CONTAMINATED sentinel; got {result!r}'
        )
        assert not isinstance(result, WorktreeInfo)

        # (b) The lane is released back to FREE — never handed out ASSIGNED for
        # the contaminating task, so nothing is dispatched onto the stale tree.
        assert git_ops.warm_lane_pool.state(lane) == PoolLaneState.FREE, (
            'contaminated lane must be released FREE, not left ASSIGNED'
        )
        record = git_ops._lane_lifecycle.read(lane)
        assert record is None or record.task_id != 'NEW', (
            f'lane must never be ASSIGNED to the contaminating task NEW; got {record!r}'
        )

        # (c) A WARNING naming the reseed contamination (data-integrity signal).
        low = caplog.text.lower()
        assert 'reseed' in low and 'contaminat' in low, (
            f'expected a reseed-contamination WARNING; caplog:\n{caplog.text}'
        )

    async def test_clean_recycle_still_assigns_no_false_positive(
        self, git_repo: Path,
    ):
        """A normal (unpatched) recycle acquire of a fresh task on a released
        lane still lands a clean reseed → returns a WorktreeInfo with an
        ASSIGNED durable record (the gate must not false-positive)."""
        await _add_warm_lane_scripts(git_repo)
        git_ops = GitOps(_warm_config(), git_repo, warm_lane_pool_size=1)
        assert git_ops.warm_lane_pool is not None
        start_ref = await _get_head(git_repo)

        info_old = await git_ops.acquire_warm_lane('OLD', start_ref)
        assert isinstance(info_old, WorktreeInfo), f'seed acquire failed: {info_old!r}'
        await git_ops.warm_lane_pool.release(info_old.path)

        result = await git_ops.acquire_warm_lane('FRESH', start_ref)

        assert isinstance(result, WorktreeInfo), (
            f'a clean recycle reseed must return a WorktreeInfo; got {result!r}'
        )
        record = git_ops._lane_lifecycle.read(result.path)
        assert record is not None and record.state is RecordLaneState.ASSIGNED, (
            f'a clean recycle must leave an ASSIGNED durable record; got {record!r}'
        )
