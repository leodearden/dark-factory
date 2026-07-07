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
from orchestrator.lane_lifecycle import LANE_STATE_DIRNAME, LaneState

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
