"""Tests for GitOps <-> LaneLifecycle integration (W11 gamma):

- acquire_warm_lane / release_warm_lane durable ASSIGNED/RELEASED writes
- the .pool-root sentinel fold (GitOps delegates to LaneLifecycle)
- the .task -> .task-meta relocations (disk-backstop plan.json,
  interactive.json stamp)

Self-contained, mirroring test_warm_lane_abort_teardown.py: conftest.py
provides no shared `git_repo` fixture, so this module owns its own fixture +
stub scripts rather than depending on test_git_ops.py's module-level helpers.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import TASK_META_DIRNAME, GitConfig
from orchestrator.git_ops import (
    PROTECTED_PREFIXES,
    GitOps,
    WorktreeInfo,
    _run,
)
from orchestrator.lane_lifecycle import (
    ACQUIRE_ROUTE_TRANSITIONS,
    LANE_STATE_DIRNAME,
    AcquireRoute,
    LaneState,
)

# ---------------------------------------------------------------------------
# Repo fixture (mirrors test_warm_lane_abort_teardown.py)
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
    and marks `.worktrees/.pool-root` present (task 2099 create-once guard)
    — mirrors test_warm_lane_abort_teardown.py's git_repo fixture.
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


def _lane_admin_dir(lane: Path) -> Path:
    """Parse the ``.git/worktrees/<name>`` admin dir path out of a lane's
    ``.git`` pointer file (``gitdir: <repo>/.git/worktrees/<name>``)."""
    content = (lane / '.git').read_text().strip()
    prefix = 'gitdir:'
    assert content.startswith(prefix), f'unexpected worktree .git pointer: {content!r}'
    return Path(content[len(prefix):].strip())


def _assert_route_logged(caplog: pytest.LogCaptureFixture, route: AcquireRoute) -> None:
    """Assert the route= INFO log fired for *route* — anchored on the
    trailing ' edge=' so a REUSE assertion can't false-positive-match a
    REUSE_REPAIR log line (REUSE is a prefix of REUSE_REPAIR's value)."""
    needle = f'route={route.value} edge='
    assert needle in caplog.text, f'expected {needle!r} in caplog:\n{caplog.text}'


# ---------------------------------------------------------------------------
# acquire_warm_lane -> durable ASSIGNED record
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAcquireWarmLaneWritesAssignedRecord:
    """acquire_warm_lane must write a durable ASSIGNED LaneLifecycle record."""

    async def test_create_once_acquire_writes_assigned_record(self, git_repo: Path):
        await _add_warm_lane_scripts(git_repo)
        git_ops = GitOps(_warm_config(), git_repo, warm_lane_pool_size=1)
        start_ref = await _get_head(git_repo)

        wt = await git_ops.acquire_warm_lane('321', start_ref, expected_title='Fix X')

        assert isinstance(wt, WorktreeInfo), f'Expected WorktreeInfo; got {wt!r}'
        record = git_ops._lane_lifecycle.read(wt.path)
        assert record is not None, 'expected a durable record after acquire'
        assert record.state is LaneState.ASSIGNED, (
            f'expected ASSIGNED, got {record.state!r}'
        )
        assert record.task_id == '321', f'expected task_id==321, got {record.task_id!r}'
        assert record.title == 'Fix X', f'expected title=="Fix X", got {record.title!r}'
        assert record.branch == 'task/321', (
            f'expected branch=="task/321", got {record.branch!r}'
        )

        record_path = git_ops.worktree_base / '.lane-state' / f'{wt.path.name}.json'
        assert record_path.is_file(), (
            f'expected a durable record file to exist at {record_path}'
        )


# ---------------------------------------------------------------------------
# _note_assigned_via_route: route-table-driven ASSIGNED writer (W11 eta)
# ---------------------------------------------------------------------------


def _bare_git_ops(tmp_path: Path) -> GitOps:
    """A GitOps instance over a plain (non-git) directory.

    _note_assigned_via_route only touches self._lane_lifecycle (file-based
    JSON records) and never shells out to git, so — like
    TestPoolStorageSentinelDelegation above — these tests don't need a real
    git repo or the warm-lane scripts/base fixtures.
    """
    project_root = tmp_path / 'project'
    project_root.mkdir()
    return GitOps(_warm_config(), project_root)


class TestNoteAssignedViaRoute:
    """GitOps._note_assigned_via_route must reproduce _lifecycle_note_assigned's
    dynamic FROM-state normalization while reading its terminal target from
    ACQUIRE_ROUTE_TRANSITIONS[route] (making the table load-bearing)."""

    def test_none_origin_bootstraps_to_assigned(self, tmp_path: Path):
        git_ops = _bare_git_ops(tmp_path)
        lane = git_ops.worktree_base / '_lane-0'

        git_ops._note_assigned_via_route(
            lane, AcquireRoute.CREATE_ONCE_FRESH, '321', 'Fix X', 'task/321',
        )

        record = git_ops._lane_lifecycle.read(lane)
        assert record is not None
        assert record.state is LaneState.ASSIGNED
        assert record.task_id == '321'
        assert record.title == 'Fix X'
        assert record.branch == 'task/321'

    def test_registered_transitions_to_assigned(self, tmp_path: Path):
        git_ops = _bare_git_ops(tmp_path)
        lane = git_ops.worktree_base / '_lane-0'
        git_ops._lane_lifecycle.transition(lane, LaneState.SEED, seeded_from_sha='abc')
        git_ops._lane_lifecycle.transition(lane, LaneState.REGISTERED, branch='task/foo')

        git_ops._note_assigned_via_route(
            lane, AcquireRoute.CREATE_ONCE_FRESH, '321', 'Fix X', 'task/321',
        )

        record = git_ops._lane_lifecycle.read(lane)
        assert record is not None
        assert record.state is LaneState.ASSIGNED
        assert record.task_id == '321'

    def test_released_transitions_to_assigned(self, tmp_path: Path):
        git_ops = _bare_git_ops(tmp_path)
        lane = git_ops.worktree_base / '_lane-0'
        git_ops._lane_lifecycle.transition(lane, LaneState.SEED, seeded_from_sha='abc')
        git_ops._lane_lifecycle.transition(lane, LaneState.REGISTERED, branch='task/foo')
        git_ops._lane_lifecycle.transition(
            lane, LaneState.ASSIGNED, task_id='999', title='old',
        )
        git_ops._lane_lifecycle.transition(lane, LaneState.IN_USE)
        git_ops._lane_lifecycle.transition(lane, LaneState.RELEASED)

        git_ops._note_assigned_via_route(
            lane, AcquireRoute.REUSE, '321', 'Fix X', 'task/321',
        )

        record = git_ops._lane_lifecycle.read(lane)
        assert record is not None
        assert record.state is LaneState.ASSIGNED
        assert record.task_id == '321'
        assert record.title == 'Fix X'

    def test_same_task_assigned_is_idempotent_noop(self, tmp_path: Path):
        git_ops = _bare_git_ops(tmp_path)
        lane = git_ops.worktree_base / '_lane-0'
        git_ops._lane_lifecycle.transition(lane, LaneState.SEED, seeded_from_sha='abc')
        git_ops._lane_lifecycle.transition(lane, LaneState.REGISTERED, branch='task/321')
        before = git_ops._lane_lifecycle.transition(
            lane, LaneState.ASSIGNED, task_id='321', title='Fix X', branch='task/321',
        )

        git_ops._note_assigned_via_route(
            lane, AcquireRoute.REUSE, '321', 'Fix X', 'task/321',
        )

        after = git_ops._lane_lifecycle.read(lane)
        assert after == before, (
            f'expected an idempotent no-op for same-task reuse; '
            f'before={before!r} after={after!r}'
        )

    def test_different_task_releases_first_then_assigns(self, tmp_path: Path):
        git_ops = _bare_git_ops(tmp_path)
        lane = git_ops.worktree_base / '_lane-0'
        git_ops._lane_lifecycle.transition(lane, LaneState.SEED, seeded_from_sha='abc')
        git_ops._lane_lifecycle.transition(lane, LaneState.REGISTERED, branch='task/999')
        git_ops._lane_lifecycle.transition(
            lane, LaneState.ASSIGNED, task_id='999', title='old',
        )

        git_ops._note_assigned_via_route(
            lane, AcquireRoute.RECYCLE, '321', 'Fix X', 'task/321',
        )

        record = git_ops._lane_lifecycle.read(lane)
        assert record is not None
        assert record.state is LaneState.ASSIGNED
        assert record.task_id == '321'
        assert record.title == 'Fix X'

    def test_in_use_releases_first_then_assigns(self, tmp_path: Path):
        git_ops = _bare_git_ops(tmp_path)
        lane = git_ops.worktree_base / '_lane-0'
        git_ops._lane_lifecycle.transition(lane, LaneState.SEED, seeded_from_sha='abc')
        git_ops._lane_lifecycle.transition(lane, LaneState.REGISTERED, branch='task/999')
        git_ops._lane_lifecycle.transition(
            lane, LaneState.ASSIGNED, task_id='999', title='old',
        )
        git_ops._lane_lifecycle.transition(lane, LaneState.IN_USE)

        git_ops._note_assigned_via_route(
            lane, AcquireRoute.RECYCLE, '321', 'Fix X', 'task/321',
        )

        record = git_ops._lane_lifecycle.read(lane)
        assert record is not None
        assert record.state is LaneState.ASSIGNED
        assert record.task_id == '321'

    def test_quarantined_lane_never_raises_and_record_unchanged(self, tmp_path: Path):
        git_ops = _bare_git_ops(tmp_path)
        lane = git_ops.worktree_base / '_lane-0'
        git_ops._lane_lifecycle.transition(lane, LaneState.SEED, seeded_from_sha='abc')
        before = git_ops._lane_lifecycle.transition(lane, LaneState.QUARANTINED)

        git_ops._note_assigned_via_route(
            lane, AcquireRoute.CREATE_ONCE_FRESH, '321', 'Fix X', 'task/321',
        )  # must not raise

        after = git_ops._lane_lifecycle.read(lane)
        assert after == before, (
            f'expected the QUARANTINED record to be left untouched; '
            f'before={before!r} after={after!r}'
        )

    def test_write_targets_the_routes_canonical_assigned_edge(self, tmp_path: Path):
        """No route's canonical target is anything other than ASSIGNED (by
        construction, per ACQUIRE_ROUTE_TRANSITIONS) — assert the writer
        actually uses that table lookup (not a hardcoded constant) across
        every route."""
        git_ops = _bare_git_ops(tmp_path)
        for route in AcquireRoute:
            lane = git_ops.worktree_base / f'_lane-{route.value}'
            git_ops._note_assigned_via_route(lane, route, '321', 'Fix X', 'task/321')
            record = git_ops._lane_lifecycle.read(lane)
            assert record is not None
            assert record.state is ACQUIRE_ROUTE_TRANSITIONS[route][1]
            assert record.state is LaneState.ASSIGNED


# ---------------------------------------------------------------------------
# Route classification: each of the 7 acquire_warm_lane routes must record
# ASSIGNED via the correctly-named route (W11 eta)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAcquireRouteClassification:
    """Drives acquire_warm_lane down each of the 7 named routes and asserts
    BOTH the durable record ends ASSIGNED and caplog shows the matching
    route= INFO log line — the user-observable signal that the restructure
    (step 6) actually threads AcquireRoute through every branch instead of
    leaving them on the anonymous _lifecycle_note_assigned chokepoint."""

    async def test_create_once_fresh(self, git_repo: Path, caplog: pytest.LogCaptureFixture):
        """First acquire of a new task on a fresh pool: non-registered lane,
        no orphan branch -> plain `git worktree add -b` + seed."""
        await _add_warm_lane_scripts(git_repo)
        git_ops = GitOps(_warm_config(), git_repo, warm_lane_pool_size=1)
        start_ref = await _get_head(git_repo)

        with caplog.at_level(logging.INFO, logger='orchestrator.git_ops'):
            result = await git_ops.acquire_warm_lane('A', start_ref)

        assert isinstance(result, WorktreeInfo), f'Expected WorktreeInfo; got {result!r}'
        record = git_ops._lane_lifecycle.read(result.path)
        assert record is not None and record.state is LaneState.ASSIGNED
        _assert_route_logged(caplog, AcquireRoute.CREATE_ONCE_FRESH)

    async def test_reuse(self, git_repo: Path, caplog: pytest.LogCaptureFixture):
        """A second same-task acquire with no release between: in-memory map
        hit, lane already registered from the first acquire."""
        await _add_warm_lane_scripts(git_repo)
        git_ops = GitOps(_warm_config(), git_repo, warm_lane_pool_size=1)
        start_ref = await _get_head(git_repo)

        first = await git_ops.acquire_warm_lane('C', start_ref)
        assert isinstance(first, WorktreeInfo), f'Expected WorktreeInfo; got {first!r}'

        with caplog.at_level(logging.INFO, logger='orchestrator.git_ops'):
            result = await git_ops.acquire_warm_lane('C', start_ref)

        assert isinstance(result, WorktreeInfo), f'Expected WorktreeInfo; got {result!r}'
        assert result.path == first.path
        record = git_ops._lane_lifecycle.read(result.path)
        assert record is not None and record.state is LaneState.ASSIGNED
        _assert_route_logged(caplog, AcquireRoute.REUSE)

    async def test_reuse_repair(self, git_repo: Path, caplog: pytest.LogCaptureFixture):
        """A reused lane whose `.git/worktrees/<name>` admin dir survives but
        has a corrupted (stale, non-wiped) gitdir back-pointer: repairable
        per `_repair_orphaned_reuse_lane`'s docstring -> repaired in place,
        not demoted to create-once reattach."""
        await _add_warm_lane_scripts(git_repo)
        git_ops = GitOps(_warm_config(), git_repo, warm_lane_pool_size=1)
        start_ref = await _get_head(git_repo)

        info = await git_ops.acquire_warm_lane('R', start_ref)
        assert isinstance(info, WorktreeInfo), f'Expected WorktreeInfo; got {info!r}'
        lane = info.path
        # DO NOT release — 'R' stays mapped, so the next acquire_for('R')
        # returns reused=True.

        admin_dir = _lane_admin_dir(lane)
        (admin_dir / 'gitdir').write_text('/tmp/bogus/.git\n')
        assert not await git_ops._is_registered_worktree(lane), (
            'lane must appear unregistered once its admin dir gitdir '
            'pointer is corrupted (setup check)'
        )

        with caplog.at_level(logging.INFO, logger='orchestrator.git_ops'):
            result = await git_ops.acquire_warm_lane('R', start_ref)

        assert isinstance(result, WorktreeInfo), (
            f'Expected WorktreeInfo (repaired in place); got {result!r}'
        )
        assert result.path == lane
        record = git_ops._lane_lifecycle.read(result.path)
        assert record is not None and record.state is LaneState.ASSIGNED
        _assert_route_logged(caplog, AcquireRoute.REUSE_REPAIR)

    async def test_create_once_reattach(
        self, git_repo: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ):
        """A non-registered lane whose orphan task/<id> branch already carries
        a commit beyond main: `git worktree add` (no -b) + seed, then the
        reuse tail."""
        await _add_warm_lane_scripts(git_repo)
        start_ref = await _get_head(git_repo)

        tmp_wt = tmp_path / 'orphan_e'
        await _run(
            ['git', 'worktree', 'add', '-b', 'task/E', str(tmp_wt), start_ref],
            cwd=git_repo,
        )
        (tmp_wt / 'wip.txt').write_text('wip\n')
        await _run(['git', 'add', '-A'], cwd=tmp_wt)
        await _run(['git', 'commit', '-m', 'wip'], cwd=tmp_wt)
        await _run(['git', 'worktree', 'remove', '--force', str(tmp_wt)], cwd=git_repo)

        # FRESH pool — _lane-0 never acquired (create-once site).
        git_ops = GitOps(_warm_config(), git_repo, warm_lane_pool_size=1)

        with caplog.at_level(logging.INFO, logger='orchestrator.git_ops'):
            result = await git_ops.acquire_warm_lane('E', start_ref)

        assert isinstance(result, WorktreeInfo), (
            f'Expected WorktreeInfo (reattach); got {result!r}'
        )
        record = git_ops._lane_lifecycle.read(result.path)
        assert record is not None and record.state is LaneState.ASSIGNED
        _assert_route_logged(caplog, AcquireRoute.CREATE_ONCE_REATTACH)

    async def test_disk_backstop_reuse(self, git_repo: Path, caplog: pytest.LogCaptureFixture):
        """A registered FREE lane whose .task/plan.json still names THIS
        task (e.g. after a process restart cleared the in-memory map)."""
        await _add_warm_lane_scripts(git_repo)
        git_ops = GitOps(_warm_config(), git_repo, warm_lane_pool_size=1)
        start_ref = await _get_head(git_repo)

        first = await git_ops.acquire_warm_lane('D', start_ref)
        assert isinstance(first, WorktreeInfo), f'Expected WorktreeInfo; got {first!r}'
        lane = first.path
        (lane / '.task').mkdir(exist_ok=True)
        (lane / '.task' / 'plan.json').write_text('{"task_id": "D"}')
        assert git_ops.warm_lane_pool is not None
        # Bare pool release: drops the in-memory assignment (and frees the
        # pool slot) without touching git or the .task/ scratch on disk —
        # only the on-disk plan.json scan can now discover the match.
        await git_ops.warm_lane_pool.release(lane)

        with caplog.at_level(logging.INFO, logger='orchestrator.git_ops'):
            result = await git_ops.acquire_warm_lane('D', start_ref)

        assert isinstance(result, WorktreeInfo), f'Expected WorktreeInfo; got {result!r}'
        assert result.path == lane
        record = git_ops._lane_lifecycle.read(result.path)
        assert record is not None and record.state is LaneState.ASSIGNED
        assert record.task_id == 'D'
        _assert_route_logged(caplog, AcquireRoute.DISK_BACKSTOP_REUSE)

    async def test_reset_in_place_reattach(
        self, git_repo: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ):
        """An already-registered lane whose orphan task/<id> branch carries a
        commit beyond main: `git checkout -f` + the reuse tail."""
        await _add_warm_lane_scripts(git_repo)
        start_ref = await _get_head(git_repo)

        tmp_wt = tmp_path / 'orphan_f'
        await _run(
            ['git', 'worktree', 'add', '-b', 'task/F', str(tmp_wt), start_ref],
            cwd=git_repo,
        )
        (tmp_wt / 'wip.txt').write_text('wip\n')
        await _run(['git', 'add', '-A'], cwd=tmp_wt)
        await _run(['git', 'commit', '-m', 'wip'], cwd=tmp_wt)
        await _run(['git', 'worktree', 'remove', '--force', str(tmp_wt)], cwd=git_repo)

        # Register+free the pool's only lane as a previous 'seed' occupant.
        git_ops = GitOps(_warm_config(), git_repo, warm_lane_pool_size=1)
        seed = await git_ops.acquire_warm_lane('seed', start_ref)
        assert isinstance(seed, WorktreeInfo), f'Expected WorktreeInfo; got {seed!r}'
        assert git_ops.warm_lane_pool is not None
        await git_ops.warm_lane_pool.release(seed.path)

        with caplog.at_level(logging.INFO, logger='orchestrator.git_ops'):
            result = await git_ops.acquire_warm_lane('F', start_ref)

        assert isinstance(result, WorktreeInfo), (
            f'Expected WorktreeInfo (reattach); got {result!r}'
        )
        assert result.path == seed.path
        record = git_ops._lane_lifecycle.read(result.path)
        assert record is not None and record.state is LaneState.ASSIGNED
        _assert_route_logged(caplog, AcquireRoute.RESET_IN_PLACE_REATTACH)

    async def test_recycle(self, git_repo: Path, caplog: pytest.LogCaptureFixture):
        """A registered FREE lane with no matching on-disk plan.json and no
        orphan-with-commits branch for the new task: fresh reset+reseed via
        `_reset_and_seed_recycled_lane`."""
        await _add_warm_lane_scripts(git_repo)
        git_ops = GitOps(_warm_config(), git_repo, warm_lane_pool_size=1)
        start_ref = await _get_head(git_repo)

        first = await git_ops.acquire_warm_lane('OLD', start_ref)
        assert isinstance(first, WorktreeInfo), f'Expected WorktreeInfo; got {first!r}'
        lane = first.path
        assert git_ops.warm_lane_pool is not None
        # Bare pool release: FREE + registered, no git/file cleanup — 'OLD'
        # left checked out and no .task/plan.json ever written, so the next
        # acquire's disk-backstop scan (task_id 'OLD' != 'NEW') and orphan
        # probe (task/NEW never existed) both miss, landing on recycle.
        await git_ops.warm_lane_pool.release(lane)

        with caplog.at_level(logging.INFO, logger='orchestrator.git_ops'):
            result = await git_ops.acquire_warm_lane('NEW', start_ref)

        assert isinstance(result, WorktreeInfo), f'Expected WorktreeInfo; got {result!r}'
        assert result.path == lane
        record = git_ops._lane_lifecycle.read(result.path)
        assert record is not None and record.state is LaneState.ASSIGNED
        assert record.task_id == 'NEW'
        _assert_route_logged(caplog, AcquireRoute.RECYCLE)


# ---------------------------------------------------------------------------
# Sibling .task-meta contamination on lane recycle (W11 ε1): a DIFFERENT-task
# acquisition via RECYCLE or RESET_IN_PLACE_REATTACH must clear the lane's
# sibling .task-meta/<name> dir, since it lives OUTSIDE the worktree and
# survives `checkout -f -B` / `git clean` / `checkout -f` unscathed — without
# this the incoming task would inherit the PRIOR occupant's plan.json /
# metadata.json (reviewer_comprehensive robustness/data-integrity blocker at
# workflow.py:1736). Same-task disk-backstop reuse must NOT be cleared
# (over-fire guard).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAcquireClearsForeignTaskMeta:
    """DIFFERENT-task RECYCLE / RESET_IN_PLACE_REATTACH acquisitions must
    clear the lane's sibling .task-meta/<name>; SAME-task disk-backstop reuse
    must preserve it."""

    async def test_recycle_clears_foreign_task_meta(
        self, git_repo: Path, caplog: pytest.LogCaptureFixture,
    ):
        """RECYCLE (a registered FREE lane, no orphan-with-commits branch for
        the new task) must clear the PRIOR occupant's .task-meta/<name>."""
        await _add_warm_lane_scripts(git_repo)
        git_ops = GitOps(_warm_config(), git_repo, warm_lane_pool_size=1)
        start_ref = await _get_head(git_repo)

        first = await git_ops.acquire_warm_lane('OLD', start_ref)
        assert isinstance(first, WorktreeInfo), f'Expected WorktreeInfo; got {first!r}'
        lane = first.path

        meta_root = TaskArtifacts.meta_root_for(git_ops.worktree_base, lane.name)
        meta_root.mkdir(parents=True, exist_ok=True)
        (meta_root / 'metadata.json').write_text(
            json.dumps({'task_id': 'OLD', 'base_commit': 'deadbeefdeadbeef'})
        )
        (meta_root / 'plan.json').write_text(json.dumps({'task_id': 'OLD', 'steps': []}))

        assert git_ops.warm_lane_pool is not None
        # Bare pool release: FREE + registered, no git/file cleanup — mirrors
        # test_recycle's setup so 'NEW' lands on the same RECYCLE route.
        await git_ops.warm_lane_pool.release(lane)

        with caplog.at_level(logging.INFO, logger='orchestrator.git_ops'):
            result = await git_ops.acquire_warm_lane('NEW', start_ref)

        assert isinstance(result, WorktreeInfo), f'Expected WorktreeInfo; got {result!r}'
        assert result.path == lane
        _assert_route_logged(caplog, AcquireRoute.RECYCLE)

        arts = TaskArtifacts(
            result.path,
            TaskArtifacts.meta_root_for(git_ops.worktree_base, result.path.name),
        )
        assert arts.read_base_commit() is None, (
            "expected 'NEW' to observe no base_commit — 'OLD' .task-meta leaked"
        )
        assert arts.read_plan() == {}, (
            "expected 'NEW' to observe no plan — 'OLD' .task-meta leaked"
        )
        assert not meta_root.exists(), (
            f'expected {meta_root} to be cleared on RECYCLE, but it still exists'
        )

    async def test_reset_in_place_reattach_clears_foreign_task_meta(
        self, git_repo: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ):
        """RESET_IN_PLACE_REATTACH (an already-registered lane whose orphan
        task/<id> branch carries a commit beyond main) must clear the PRIOR
        occupant's .task-meta/<name> — mirrors test_reset_in_place_reattach's
        setup with a foreign .task-meta seeded onto the reused seed lane."""
        await _add_warm_lane_scripts(git_repo)
        start_ref = await _get_head(git_repo)

        tmp_wt = tmp_path / 'orphan_f'
        await _run(
            ['git', 'worktree', 'add', '-b', 'task/F', str(tmp_wt), start_ref],
            cwd=git_repo,
        )
        (tmp_wt / 'wip.txt').write_text('wip\n')
        await _run(['git', 'add', '-A'], cwd=tmp_wt)
        await _run(['git', 'commit', '-m', 'wip'], cwd=tmp_wt)
        await _run(['git', 'worktree', 'remove', '--force', str(tmp_wt)], cwd=git_repo)

        git_ops = GitOps(_warm_config(), git_repo, warm_lane_pool_size=1)
        seed = await git_ops.acquire_warm_lane('seed', start_ref)
        assert isinstance(seed, WorktreeInfo), f'Expected WorktreeInfo; got {seed!r}'
        lane = seed.path
        assert git_ops.warm_lane_pool is not None
        await git_ops.warm_lane_pool.release(lane)

        meta_root = TaskArtifacts.meta_root_for(git_ops.worktree_base, lane.name)
        meta_root.mkdir(parents=True, exist_ok=True)
        (meta_root / 'metadata.json').write_text(
            json.dumps({'task_id': 'OTHER', 'base_commit': 'feedfacefeedface'})
        )
        (meta_root / 'plan.json').write_text(json.dumps({'task_id': 'OTHER', 'steps': []}))

        with caplog.at_level(logging.INFO, logger='orchestrator.git_ops'):
            result = await git_ops.acquire_warm_lane('F', start_ref)

        assert isinstance(result, WorktreeInfo), (
            f'Expected WorktreeInfo (reattach); got {result!r}'
        )
        assert result.path == lane
        _assert_route_logged(caplog, AcquireRoute.RESET_IN_PLACE_REATTACH)

        arts = TaskArtifacts(
            result.path,
            TaskArtifacts.meta_root_for(git_ops.worktree_base, result.path.name),
        )
        assert arts.read_base_commit() is None, (
            "expected 'F' to observe no base_commit — 'OTHER' .task-meta leaked"
        )
        assert arts.read_plan() == {}, (
            "expected 'F' to observe no plan — 'OTHER' .task-meta leaked"
        )
        assert not meta_root.exists(), (
            f'expected {meta_root} to be cleared on RESET_IN_PLACE_REATTACH, '
            f'but it still exists'
        )

    async def test_same_task_reacquire_preserves_task_meta(
        self, git_repo: Path, caplog: pytest.LogCaptureFixture,
    ):
        """Over-fire guard: re-acquiring the SAME task via the disk-backstop
        reuse route must NOT clear .task-meta/<name> — only the two
        DIFFERENT-task routes above do."""
        await _add_warm_lane_scripts(git_repo)
        git_ops = GitOps(_warm_config(), git_repo, warm_lane_pool_size=1)
        start_ref = await _get_head(git_repo)

        first = await git_ops.acquire_warm_lane('OLD', start_ref)
        assert isinstance(first, WorktreeInfo), f'Expected WorktreeInfo; got {first!r}'
        lane = first.path

        meta_root = TaskArtifacts.meta_root_for(git_ops.worktree_base, lane.name)
        meta_root.mkdir(parents=True, exist_ok=True)
        (meta_root / 'metadata.json').write_text(
            json.dumps({'task_id': 'OLD', 'base_commit': 'cafecafecafe'})
        )
        (meta_root / 'plan.json').write_text(json.dumps({'task_id': 'OLD', 'steps': []}))

        assert git_ops.warm_lane_pool is not None
        await git_ops.warm_lane_pool.release(lane)

        with caplog.at_level(logging.INFO, logger='orchestrator.git_ops'):
            result = await git_ops.acquire_warm_lane('OLD', start_ref)

        assert isinstance(result, WorktreeInfo), f'Expected WorktreeInfo; got {result!r}'
        assert result.path == lane
        _assert_route_logged(caplog, AcquireRoute.DISK_BACKSTOP_REUSE)

        arts = TaskArtifacts(
            result.path,
            TaskArtifacts.meta_root_for(git_ops.worktree_base, result.path.name),
        )
        assert arts.read_base_commit() == 'cafecafecafe', (
            'expected the seeded base_commit to survive same-task reuse; got '
            f'{arts.read_base_commit()!r}'
        )
        assert arts.read_plan().get('task_id') == 'OLD', (
            'expected the seeded plan to survive same-task reuse; got '
            f'{arts.read_plan()!r}'
        )


# ---------------------------------------------------------------------------
# release_warm_lane -> durable RELEASED record
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReleaseWarmLaneWritesReleasedRecord:
    """release_warm_lane must write a durable RELEASED LaneLifecycle record."""

    async def test_release_after_acquire_writes_released_record(self, git_repo: Path):
        await _add_warm_lane_scripts(git_repo)
        git_ops = GitOps(_warm_config(), git_repo, warm_lane_pool_size=1)
        start_ref = await _get_head(git_repo)

        wt = await git_ops.acquire_warm_lane('321', start_ref, expected_title='Fix X')
        assert isinstance(wt, WorktreeInfo), f'Expected WorktreeInfo; got {wt!r}'

        await git_ops.release_warm_lane(wt.path, '321')

        record = git_ops._lane_lifecycle.read(wt.path)
        assert record is not None, 'expected a durable record after release'
        assert record.state is LaneState.RELEASED, (
            f'expected RELEASED, got {record.state!r}'
        )
        assert record.task_id is None, f'expected task_id cleared, got {record.task_id!r}'
        assert record.title is None, f'expected title cleared, got {record.title!r}'

    async def test_release_already_released_lane_is_a_noop(self, git_repo: Path):
        """A second release_warm_lane call on an already-RELEASED lane must
        not raise. RELEASED -> RELEASED is not a legal edge, so
        _lifecycle_note_released must no-op rather than attempt it."""
        await _add_warm_lane_scripts(git_repo)
        git_ops = GitOps(_warm_config(), git_repo, warm_lane_pool_size=1)
        start_ref = await _get_head(git_repo)

        wt = await git_ops.acquire_warm_lane('321', start_ref)
        assert isinstance(wt, WorktreeInfo), f'Expected WorktreeInfo; got {wt!r}'
        await git_ops.release_warm_lane(wt.path, '321')
        before = git_ops._lane_lifecycle.read(wt.path)
        assert before is not None and before.state is LaneState.RELEASED

        await git_ops.release_warm_lane(wt.path, '321')  # must not raise

        after = git_ops._lane_lifecycle.read(wt.path)
        assert after is not None
        assert after.state is LaneState.RELEASED, (
            f'expected RELEASED to remain a legal no-op state, got {after.state!r}'
        )


# ---------------------------------------------------------------------------
# .pool-root sentinel fold: GitOps delegates to LaneLifecycle
# ---------------------------------------------------------------------------


class TestPoolStorageSentinelDelegation:
    """GitOps.pool_storage_present()/mark_pool_storage_present() must delegate
    to the shared LaneLifecycle instance (W11 gamma sentinel fold) — the
    public GitOps API behavior is preserved through the fold."""

    def test_mark_and_present_delegate_to_lane_lifecycle(self, tmp_path: Path):
        project_root = tmp_path / 'project'
        project_root.mkdir()
        git_ops = GitOps(_warm_config(), project_root)
        assert git_ops.pool_storage_present() is False

        git_ops.mark_pool_storage_present()

        sentinel = git_ops.worktree_base / '.pool-root'
        assert sentinel.is_file(), f'expected sentinel at {sentinel}'
        assert git_ops.pool_storage_present() is True
        # Same record LaneLifecycle owns — delegation, not a parallel/
        # duplicate implementation.
        assert git_ops._lane_lifecycle.pool_storage_present() is True


# ---------------------------------------------------------------------------
# Disk-backstop plan.json relocation: new-then-old via TaskArtifacts.meta_root_for
# ---------------------------------------------------------------------------


class TestDiskBackstopPlanJsonRelocation:
    """_find_lane_by_plan_task_id (the on-disk backstop scan) must read
    plan.json from the NEW .task-meta location first, falling back to the
    legacy <lane>/.task/plan.json path (W11 gamma .task -> .task-meta
    relocation, PRD decision 7: new-then-old reads, side-effect-free)."""

    def test_finds_lane_via_new_task_meta_path(self, tmp_path: Path):
        project_root = tmp_path / 'project'
        project_root.mkdir()
        git_ops = GitOps(_warm_config(), project_root, warm_lane_pool_size=1)
        lane = git_ops.worktree_base / '_lane-0'
        lane.mkdir(parents=True)
        meta_root = TaskArtifacts.meta_root_for(git_ops.worktree_base, '_lane-0')
        meta_root.mkdir(parents=True)
        (meta_root / 'plan.json').write_text(json.dumps({'task_id': '777'}))

        found = git_ops._find_lane_by_plan_task_id('777')

        assert found == lane, (
            f'expected the new .task-meta/plan.json to resolve lane {lane}, '
            f'got {found!r}'
        )

    def test_falls_back_to_legacy_task_path(self, tmp_path: Path):
        project_root = tmp_path / 'project'
        project_root.mkdir()
        git_ops = GitOps(_warm_config(), project_root, warm_lane_pool_size=1)
        lane = git_ops.worktree_base / '_lane-0'
        legacy_dir = lane / '.task'
        legacy_dir.mkdir(parents=True)
        (legacy_dir / 'plan.json').write_text(json.dumps({'task_id': '777'}))
        # Deliberately no plan.json at the new .task-meta path at all.

        found = git_ops._find_lane_by_plan_task_id('777')

        assert found == lane, (
            f'expected new-then-old fallback to find legacy plan.json at '
            f'{legacy_dir}, got {found!r}'
        )


# ---------------------------------------------------------------------------
# Interactive worktree stamp relocation: .task-meta write-only via TaskArtifacts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestInteractiveWorktreeStampRelocation:
    """create_interactive_worktree must write its interactive.json stamp ONLY
    to the new .task-meta location (W11 gamma .task -> .task-meta relocation;
    PRD `.task-meta` path-derivation contract: writes new-path-only) — never
    under <iact_worktree>/.task/."""

    async def test_stamp_written_to_new_path_only(self, git_repo: Path):
        git_ops = GitOps(GitConfig(), git_repo)

        info = await git_ops.create_interactive_worktree('stamp-slug')

        meta_root = TaskArtifacts.meta_root_for(git_ops.worktree_base, info.path.name)
        stamp_path = meta_root / 'interactive.json'
        assert stamp_path.is_file(), f'expected stamp at new path {stamp_path}'

        stamp = json.loads(stamp_path.read_text())
        assert stamp['owner'] == 'stamp-slug', (
            f"expected stamp['owner'] == 'stamp-slug', got {stamp.get('owner')!r}"
        )
        assert stamp['branch'] == info.branch, (
            f"expected stamp['branch'] == {info.branch!r}, got {stamp.get('branch')!r}"
        )
        assert stamp['slug'] == 'stamp-slug', (
            f"expected stamp['slug'] == 'stamp-slug', got {stamp.get('slug')!r}"
        )
        datetime.fromisoformat(stamp['created_at'])  # machine-parseable ISO8601

        legacy_path = info.path / '.task' / 'interactive.json'
        assert not legacy_path.exists(), (
            f'expected NO interactive.json written under the legacy path {legacy_path}'
        )


# ---------------------------------------------------------------------------
# Interactive reaper stamp read: new-then-old via TaskArtifacts.meta_root_for
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReapInteractiveWorktreesStampRelocation:
    """reap_interactive_worktrees must resolve the interactive.json stamp
    new-then-old (W11 gamma .task -> .task-meta relocation, PRD decision 7):
    the NEW TaskArtifacts.meta_root_for(...)/interactive.json path first,
    falling back to the legacy <wt>/.task/interactive.json path."""

    async def test_reaps_expired_stamp_found_at_new_path(self, git_repo: Path):
        """create_interactive_worktree writes only to the new path (S10);
        confirm the reaper actually reads created_at from there — a
        'ttl_idle' reason proves a real stamp read, vs. 'stale_no_stamp'
        which would mean the reaper never found it."""
        git_ops = GitOps(GitConfig(), git_repo)
        info = await git_ops.create_interactive_worktree('new-path-expired')

        new_stamp_path = (
            TaskArtifacts.meta_root_for(git_ops.worktree_base, info.path.name)
            / 'interactive.json'
        )
        stamp = json.loads(new_stamp_path.read_text())
        now = datetime.now(UTC)
        stamp['created_at'] = (
            now - timedelta(seconds=git_ops.config.interactive_worktree_ttl + 3600)
        ).isoformat()
        new_stamp_path.write_text(json.dumps(stamp))

        reaped = await git_ops.reap_interactive_worktrees(now=now)

        reaped_paths = {r.path.resolve() for r in reaped}
        assert info.path.resolve() in reaped_paths, (
            f'expected the expired new-path stamp to trigger a ttl_idle reap; '
            f'got {reaped!r}'
        )
        record = next(r for r in reaped if r.path.resolve() == info.path.resolve())
        assert record.reason == 'ttl_idle', (
            f"expected reason == 'ttl_idle' (a real stamp read, not "
            f"'stale_no_stamp'), got {record.reason!r}"
        )

    async def test_falls_back_to_legacy_stamp_when_new_path_absent(
        self, git_repo: Path,
    ):
        """No .task-meta stamp at all (pre-migration lane); a legacy
        <wt>/.task/interactive.json stamp must still be read."""
        git_ops = GitOps(GitConfig(), git_repo)
        info = await git_ops.create_interactive_worktree('legacy-path-expired')

        new_stamp_path = (
            TaskArtifacts.meta_root_for(git_ops.worktree_base, info.path.name)
            / 'interactive.json'
        )
        stamp = json.loads(new_stamp_path.read_text())
        new_stamp_path.unlink()  # simulate a pre-migration lane: no new-path stamp

        now = datetime.now(UTC)
        stamp['created_at'] = (
            now - timedelta(seconds=git_ops.config.interactive_worktree_ttl + 3600)
        ).isoformat()
        legacy_dir = info.path / '.task'
        legacy_dir.mkdir(parents=True, exist_ok=True)
        (legacy_dir / 'interactive.json').write_text(json.dumps(stamp))

        reaped = await git_ops.reap_interactive_worktrees(now=now)

        reaped_paths = {r.path.resolve() for r in reaped}
        assert info.path.resolve() in reaped_paths, (
            f'expected the expired legacy-path stamp to trigger a ttl_idle '
            f'reap via the new-then-old fallback; got {reaped!r}'
        )
        record = next(r for r in reaped if r.path.resolve() == info.path.resolve())
        assert record.reason == 'ttl_idle', (
            f"expected reason == 'ttl_idle', got {record.reason!r}"
        )


@pytest.mark.asyncio
class TestReapInteractiveWorktreesCleansTaskMeta:
    """reap_interactive_worktrees must remove the .task-meta/<name> sibling
    dir once the worktree itself is reaped (resource-leak amendment): the
    interactive.json stamp now lives OUTSIDE the worktree (S10), so `git
    worktree remove` no longer cleans it up incidentally the way it did when
    the stamp lived at <worktree>/.task/interactive.json."""

    async def test_reap_removes_task_meta_sibling_dir(self, git_repo: Path):
        git_ops = GitOps(GitConfig(), git_repo)
        info = await git_ops.create_interactive_worktree('meta-leak-check')

        meta_root = TaskArtifacts.meta_root_for(git_ops.worktree_base, info.path.name)
        assert meta_root.is_dir(), f'expected the stamp dir to exist at {meta_root}'

        stamp_path = meta_root / 'interactive.json'
        stamp = json.loads(stamp_path.read_text())
        now = datetime.now(UTC)
        stamp['created_at'] = (
            now - timedelta(seconds=git_ops.config.interactive_worktree_ttl + 3600)
        ).isoformat()
        stamp_path.write_text(json.dumps(stamp))

        reaped = await git_ops.reap_interactive_worktrees(now=now)

        reaped_paths = {r.path.resolve() for r in reaped}
        assert info.path.resolve() in reaped_paths, (
            f'expected the expired stamp to trigger a ttl_idle reap; got {reaped!r}'
        )
        assert not meta_root.exists(), (
            f'expected {meta_root} to be removed once the worktree was reaped '
            f'— an orphaned .task-meta/<name> dir is an indefinite leak (I2)'
        )


# ---------------------------------------------------------------------------
# PROTECTED_PREFIXES registration: .lane-state / .task-meta bands (W11 gamma)
# ---------------------------------------------------------------------------


class TestLaneStateAndTaskMetaProtectedBands:
    """.lane-state and .task-meta must be registered as PROTECTED bands
    (PRD resolved design decision 1: defense-in-depth) so a destructive
    worktree_base cleanup sweep can never remove either durable-record store
    as an unrecognized foreign band."""

    @pytest.mark.parametrize('dirname', [LANE_STATE_DIRNAME, TASK_META_DIRNAME])
    def test_dirname_is_a_protected_prefixes_key_with_non_empty_owner(
        self, dirname: str,
    ) -> None:
        assert dirname in PROTECTED_PREFIXES, (
            f'expected {dirname!r} to be a key in PROTECTED_PREFIXES; '
            f'got keys={list(PROTECTED_PREFIXES)!r}'
        )
        owner = PROTECTED_PREFIXES[dirname]
        assert isinstance(owner, str) and owner, (
            f'expected a non-empty owner tag for {dirname!r}; got {owner!r}'
        )

    @pytest.mark.parametrize('dirname', [LANE_STATE_DIRNAME, TASK_META_DIRNAME])
    def test_refuse_foreign_band_protects_worktree_base_child(
        self, tmp_path: Path, dirname: str,
    ) -> None:
        project_root = tmp_path / 'project'
        project_root.mkdir()
        git_ops = GitOps(_warm_config(), project_root)
        git_ops.worktree_base.mkdir(parents=True, exist_ok=True)
        path = git_ops.worktree_base / dirname
        path.mkdir()

        result = git_ops._refuse_foreign_band(path, frozenset(), 'test-context')

        assert result is True, (
            f'expected {dirname!r} to be treated as a protected band (refused, '
            f'True) when owned=frozenset(); got {result!r}'
        )
