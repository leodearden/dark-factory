"""GitOps re-cut ruling primitives (task 3154), against real git.

park_prior_incarnation renames a task's prior incarnation aside and never
resets or deletes; prior_incarnation_evidence lists what survives; the two
provisioning doors refuse a pending ruling while any of it survives.

Self-contained, mirroring test_lane_lifecycle_gitops.py's fixtures.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps, WorktreeInfo, _run
from orchestrator.lane_lifecycle import AcquireRoute
from orchestrator.recut_ruling import (
    RECUT_EXECUTED_KEY,
    RECUT_RULING_KEY,
    GateVerdict,
    RecutRuling,
    RecutRulingViolation,
    enforce_recut_ruling,
    ruling_to_enforce,
)
from orchestrator.warm_lane_pool import LaneState

PRE_RULING_DATE = '2026-07-22T12:00:00Z'
PARKED_NAME = re.compile(r'^task/(?P<id>[^/]+)-\d{8}T\d{6}Z$')
RULING = RecutRuling('esc-2896-3/D11-A', datetime(2026, 8, 30, tzinfo=UTC))
EXECUTED_AT = datetime(2026, 9, 1, 12, 30, tzinfo=UTC)
SPECIMEN_TASK = {
    'id': '2896',
    'metadata': {
        RECUT_RULING_KEY: {'ruling_id': 'esc-2896-3/D11-A', 'ruled_at': '2026-08-30T00:00:00Z'},
    },
}
REUSE_ROUTES = (
    AcquireRoute.REUSE,
    AcquireRoute.DISK_BACKSTOP_REUSE,
    AcquireRoute.RESET_IN_PLACE_REATTACH,
    AcquireRoute.CREATE_ONCE_REATTACH,
)


async def _init_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A repo with an initial commit, a resolvable warm base and `.pool-root`."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    default_base = repo / '.worktrees' / '_merge-verify' / 'target'
    default_base.mkdir(parents=True, exist_ok=True)
    (default_base / '.keep').write_text('warm base sentinel\n')
    (repo / '.worktrees' / '.pool-root').touch()
    return repo


async def _add_warm_lane_scripts(repo: Path, port: int = 39411) -> None:
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


def _config(*, warm_lane_pool: bool) -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
        warm_lane_pool=warm_lane_pool,
    )


async def _get_head(repo: Path) -> str:
    rc, out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    assert rc == 0, f'git rev-parse HEAD failed (rc={rc})'
    return out.strip()


def _route_logged(caplog: pytest.LogCaptureFixture, route: AcquireRoute) -> bool:
    return f'route={route.value} edge=' in caplog.text


def _git(cwd: Path, *args: str, pinned: bool = False) -> str:
    env = None
    if pinned:
        env = {
            **os.environ,
            'GIT_AUTHOR_DATE': PRE_RULING_DATE,
            'GIT_COMMITTER_DATE': PRE_RULING_DATE,
        }
    done = subprocess.run(
        ['git', *args], cwd=cwd, env=env, check=True, capture_output=True, text=True,
    )
    return done.stdout.strip()


def _resolve(repo: Path, ref: str) -> str | None:
    done = subprocess.run(
        ['git', 'rev-parse', '--verify', '--quiet', f'{ref}^{{commit}}'],
        cwd=repo, capture_output=True, text=True,
    )
    return done.stdout.strip() if done.returncode == 0 else None


def _commit_files(worktree: Path, count: int, stem: str) -> None:
    """Commit *count* new files with author and committer dates pinned pre-ruling."""
    for i in range(count):
        name = f'{stem}-{i}.txt'
        (worktree / name).write_text(f'{stem} {i}\n')
        _git(worktree, 'add', name)
        _git(worktree, 'commit', '-m', f'{stem} {i}', pinned=True)


def _branch_with_commits(repo: Path, scratch: Path, branch: str, count: int) -> None:
    """Create *branch* off main carrying *count* commits, with no worktree left."""
    _git(repo, 'worktree', 'add', '-b', branch, str(scratch), 'main')
    _commit_files(scratch, count, 'pre-ruling')
    _git(repo, 'worktree', 'remove', '--force', str(scratch))


def _write_lane_metadata(git_ops: GitOps, lane_name: str, task_id: str) -> Path:
    meta_root = TaskArtifacts.meta_root_for(git_ops.worktree_base, lane_name)
    meta_root.mkdir(parents=True, exist_ok=True)
    (meta_root / 'metadata.json').write_text(json.dumps({'task_id': task_id, 'title': 'T'}))
    return meta_root


def _quarantined_meta_dir(git_ops: GitOps) -> Path:
    return TaskArtifacts.meta_root_for(git_ops.quarantine_base, 'x').parent


async def _pooled_with_task_d(git_repo: Path, pool_size: int = 2) -> tuple[GitOps, Path]:
    """Pooled GitOps with task D on a lane: one committed and one uncommitted change."""
    await _add_warm_lane_scripts(git_repo)
    git_ops = GitOps(_config(warm_lane_pool=True), git_repo, warm_lane_pool_size=pool_size)
    info = await git_ops.acquire_warm_lane('D', await _get_head(git_repo))
    assert isinstance(info, WorktreeInfo), f'Expected WorktreeInfo; got {info!r}'
    _commit_files(info.path, 1, 'committed')
    (info.path / 'README.md').write_text('# uncommitted edit\n')
    return git_ops, info.path


@pytest.mark.asyncio
class TestParkPriorIncarnation:
    async def test_nothing_to_park_returns_none(self, git_repo: Path):
        git_ops = GitOps(_config(warm_lane_pool=False), git_repo)
        refs_before = _git(git_repo, 'for-each-ref', '--format=%(refname)')

        assert await git_ops.park_prior_incarnation('7', reason='recut') is None

        assert _git(git_repo, 'for-each-ref', '--format=%(refname)') == refs_before
        assert not git_ops.quarantine_base.exists()

    async def test_branch_only_is_renamed_aside_intact(self, git_repo: Path, tmp_path: Path):
        _branch_with_commits(git_repo, tmp_path / 'scratch', 'task/7', 3)
        tip = _resolve(git_repo, 'task/7')
        git_ops = GitOps(_config(warm_lane_pool=False), git_repo)

        parked = await git_ops.park_prior_incarnation('7', reason='recut')

        assert parked is not None and parked.parked_branch is not None
        match = PARKED_NAME.match(parked.parked_branch)
        assert match is not None and match['id'] == '7', parked.parked_branch
        assert parked.parked_tip == tip
        assert _resolve(git_repo, parked.parked_branch) == tip
        assert _resolve(git_repo, 'task/7') is None
        assert _git(git_repo, 'rev-list', '--count', f'main..{parked.parked_branch}') == '3'
        assert parked.quarantined_worktree is None

    async def test_cold_worktree_is_quarantined_with_wip_and_meta_root(self, git_repo: Path):
        git_ops = GitOps(_config(warm_lane_pool=False), git_repo)
        worktree = (await git_ops.create_worktree('7')).path
        _commit_files(worktree, 1, 'committed')
        (worktree / 'README.md').write_text('# uncommitted edit\n')
        meta_root = TaskArtifacts.meta_root_for(git_ops.worktree_base, '7')
        meta_root.mkdir(parents=True, exist_ok=True)
        plan_bytes = b'{"task_id": "7", "steps": []}'
        (meta_root / 'plan.json').write_bytes(plan_bytes)

        parked = await git_ops.park_prior_incarnation('7', reason='recut')

        assert parked is not None and parked.parked_branch is not None
        assert parked.quarantined_worktree is not None
        quarantined = Path(parked.quarantined_worktree)
        assert quarantined.parent == git_ops.quarantine_base
        assert quarantined.is_dir() and not worktree.exists()
        assert _git(git_repo, 'show', f'{parked.parked_branch}:README.md') == '# uncommitted edit'
        assert _git(git_repo, 'show', f'{parked.parked_branch}:committed-0.txt') == 'committed 0'
        moved = TaskArtifacts.meta_root_for(git_ops.quarantine_base, quarantined.name)
        assert not meta_root.exists()
        assert (moved / 'plan.json').read_bytes() == plan_bytes
        assert str(moved) in parked.relocated_meta_roots

    async def test_lane_held_branch_is_detached_released_and_renamed(self, git_repo: Path):
        git_ops, lane = await _pooled_with_task_d(git_repo)
        meta_root = _write_lane_metadata(git_ops, lane.name, 'D')

        parked = await git_ops.park_prior_incarnation('D', reason='recut')

        assert parked is not None and parked.parked_branch is not None
        pool = git_ops.warm_lane_pool
        assert pool is not None
        assert pool.assignment_for('D') is None
        assert pool.state(lane) is LaneState.FREE
        assert _git(lane, 'rev-parse', '--abbrev-ref', 'HEAD') == 'HEAD'
        assert _resolve(git_repo, 'task/D') is None
        assert _git(git_repo, 'show', f'{parked.parked_branch}:committed-0.txt') == 'committed 0'
        assert _git(git_repo, 'show', f'{parked.parked_branch}:README.md') == '# uncommitted edit'
        assert not meta_root.exists()
        assert len(parked.relocated_meta_roots) == 1
        moved = Path(parked.relocated_meta_roots[0])
        assert moved.parent == _quarantined_meta_dir(git_ops)
        assert json.loads((moved / 'metadata.json').read_text())['task_id'] == 'D'

    async def test_cross_lane_leftover_roots_move_and_others_stay(self, git_repo: Path):
        git_ops, lane = await _pooled_with_task_d(git_repo)
        own = _write_lane_metadata(git_ops, lane.name, 'D')
        leftover = _write_lane_metadata(git_ops, '_lane-7', 'D')
        unrelated = _write_lane_metadata(git_ops, '_lane-8', 'E')

        parked = await git_ops.park_prior_incarnation('D', reason='recut')

        assert parked is not None
        assert not own.exists() and not leftover.exists()
        assert unrelated.exists()
        assert len(parked.relocated_meta_roots) == 2
        assert len(set(parked.relocated_meta_roots)) == 2
        for moved in parked.relocated_meta_roots:
            assert Path(moved).parent == _quarantined_meta_dir(git_ops)

    async def test_unprovable_checkout_listing_fails_closed(
        self, git_repo: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        git_ops, lane = await _pooled_with_task_d(git_repo)
        meta_root = _write_lane_metadata(git_ops, lane.name, 'D')
        tip = _resolve(git_repo, 'task/D')
        monkeypatch.setattr(git_ops, 'lane_branch_checkouts', AsyncMock(return_value=None))

        with pytest.raises(RuntimeError):
            await git_ops.park_prior_incarnation('D', reason='recut')

        assert _resolve(git_repo, 'task/D') == tip
        assert meta_root.exists()
        assert not _quarantined_meta_dir(git_ops).exists()


@pytest.mark.asyncio
class TestPriorIncarnationEvidence:
    async def test_fresh_id_has_no_evidence(self, git_repo: Path):
        git_ops = GitOps(_config(warm_lane_pool=False), git_repo)
        assert await git_ops.prior_incarnation_evidence('9') == ()

    async def test_names_branch_cold_worktree_and_meta_root(self, git_repo: Path):
        git_ops = GitOps(_config(warm_lane_pool=False), git_repo)
        worktree = (await git_ops.create_worktree('7')).path
        _commit_files(worktree, 1, 'committed')
        tip = _resolve(git_repo, 'task/7')
        assert tip is not None
        meta_root = TaskArtifacts.meta_root_for(git_ops.worktree_base, '7')
        meta_root.mkdir(parents=True, exist_ok=True)

        evidence = await git_ops.prior_incarnation_evidence('7')

        assert any('task/7' in item and tip[:12] in item for item in evidence), evidence
        assert any(str(worktree) in item for item in evidence), evidence
        assert any(str(meta_root) in item for item in evidence), evidence

    async def test_names_lane_assignment_and_lane_meta_root(self, git_repo: Path):
        git_ops, lane = await _pooled_with_task_d(git_repo)
        meta_root = _write_lane_metadata(git_ops, lane.name, 'D')

        evidence = await git_ops.prior_incarnation_evidence('D')

        assert any('task/D' in item for item in evidence), evidence
        assert any(str(lane) in item and 'assign' in item for item in evidence), evidence
        assert any(str(meta_root) in item for item in evidence), evidence

    @pytest.mark.parametrize('pooled', [False, True], ids=['cold', 'pooled'])
    async def test_empty_after_park(self, git_repo: Path, pooled: bool):
        if pooled:
            git_ops, lane = await _pooled_with_task_d(git_repo)
            task_id = 'D'
            _write_lane_metadata(git_ops, lane.name, task_id)
        else:
            git_ops = GitOps(_config(warm_lane_pool=False), git_repo)
            task_id = '7'
            await git_ops.create_worktree(task_id)
            TaskArtifacts.meta_root_for(git_ops.worktree_base, task_id).mkdir(parents=True)

        await git_ops.park_prior_incarnation(task_id, reason='recut')

        assert await git_ops.prior_incarnation_evidence(task_id) == ()



@pytest.mark.asyncio
class TestProvisioningDoorsRefuseUnderPendingRuling:
    async def test_create_worktree_refuses_registered_cold_worktree(self, git_repo: Path):
        git_ops = GitOps(_config(warm_lane_pool=False), git_repo)
        worktree = (await git_ops.create_worktree('7')).path
        _commit_files(worktree, 2, 'pre-ruling')
        tip = _resolve(git_repo, 'task/7')

        with pytest.raises(RecutRulingViolation) as raised:
            await git_ops.create_worktree('7', recut_ruling=RULING)

        assert isinstance(raised.value, RuntimeError)
        assert 'esc-2896-3/D11-A' in str(raised.value)
        assert 'task/7' in str(raised.value)
        assert _resolve(git_repo, 'task/7') == tip
        assert worktree == git_ops.worktree_base / '7' and worktree.is_dir()

    async def test_create_worktree_refuses_branch_only(self, git_repo: Path, tmp_path: Path):
        _branch_with_commits(git_repo, tmp_path / 'scratch', 'task/7', 2)
        git_ops = GitOps(_config(warm_lane_pool=False), git_repo)
        refs_before = _git(git_repo, 'for-each-ref', '--format=%(refname) %(objectname)')

        with pytest.raises(RecutRulingViolation, match='esc-2896-3/D11-A'):
            await git_ops.create_worktree('7', recut_ruling=RULING)

        assert not (git_ops.worktree_base / '7').exists()
        assert _git(git_repo, 'for-each-ref', '--format=%(refname) %(objectname)') == refs_before

    async def test_acquire_warm_lane_refuses_before_any_reuse_route(
        self, git_repo: Path, caplog: pytest.LogCaptureFixture,
    ):
        await _add_warm_lane_scripts(git_repo)
        git_ops = GitOps(_config(warm_lane_pool=True), git_repo, warm_lane_pool_size=1)
        start_ref = await _get_head(git_repo)
        info = await git_ops.acquire_warm_lane('D', start_ref)
        assert isinstance(info, WorktreeInfo), f'Expected WorktreeInfo; got {info!r}'
        _commit_files(info.path, 1, 'pre-ruling')
        _write_lane_metadata(git_ops, info.path.name, 'D')
        assert git_ops.warm_lane_pool is not None
        await git_ops.warm_lane_pool.release(info.path)
        caplog.clear()

        with (
            caplog.at_level(logging.INFO, logger='orchestrator.git_ops'),
            pytest.raises(RecutRulingViolation, match='esc-2896-3/D11-A'),
        ):
            await git_ops.acquire_warm_lane('D', start_ref, recut_ruling=RULING)

        for route in REUSE_ROUTES:
            assert not _route_logged(caplog, route), caplog.text

    async def test_clean_cold_id_is_cut_fresh_off_main(self, git_repo: Path):
        git_ops = GitOps(_config(warm_lane_pool=False), git_repo)

        info = await git_ops.create_worktree('9', recut_ruling=RULING)

        assert info.base_commit == await _get_head(git_repo)
        assert _resolve(git_repo, 'task/9') == await _get_head(git_repo)

    async def test_clean_pooled_id_takes_a_fresh_route(
        self, git_repo: Path, caplog: pytest.LogCaptureFixture,
    ):
        await _add_warm_lane_scripts(git_repo)
        git_ops = GitOps(_config(warm_lane_pool=True), git_repo, warm_lane_pool_size=1)

        with caplog.at_level(logging.INFO, logger='orchestrator.git_ops'):
            info = await git_ops.acquire_warm_lane(
                'N', await _get_head(git_repo), recut_ruling=RULING,
            )

        assert isinstance(info, WorktreeInfo), f'Expected WorktreeInfo; got {info!r}'
        assert any(
            _route_logged(caplog, route)
            for route in (AcquireRoute.CREATE_ONCE_FRESH, AcquireRoute.RECYCLE)
        ), caplog.text

    async def test_no_ruling_reuses_the_existing_worktree(self, git_repo: Path):
        git_ops = GitOps(_config(warm_lane_pool=False), git_repo)
        worktree = (await git_ops.create_worktree('7')).path
        _commit_files(worktree, 2, 'pre-ruling')
        tip = _resolve(git_repo, 'task/7')

        info = await git_ops.create_worktree('7')

        assert info.path == worktree
        assert _resolve(git_repo, 'task/7') == tip
        assert _git(git_repo, 'rev-list', '--count', 'main..task/7') == '2'


class _RecordingPersist:
    def __init__(self) -> None:
        self.patches: list[dict] = []

    async def __call__(self, patch: dict) -> bool:
        self.patches.append(patch)
        return True


def _parked_refs(repo: Path, task_id: str) -> list[str]:
    listing = _git(repo, 'for-each-ref', '--format=%(refname:short)', f'refs/heads/task/{task_id}-*')
    return [line for line in listing.splitlines() if line]


async def _seed_pre_ruling_work(repo: Path, worktree: Path) -> str:
    """Three pre-ruling commits + an uncommitted edit, then main moves on by two."""
    _commit_files(worktree, 3, 'pre-ruling')
    (worktree / 'README.md').write_text('# uncommitted pre-ruling edit\n')
    tip = _resolve(repo, 'task/2896')
    assert tip is not None
    _commit_files(repo, 2, 'main-advance')
    return tip


def _assert_parked_intact(repo: Path, parked_branch: str, early_main: str, original_tip: str) -> None:
    assert _resolve(repo, parked_branch) is not None
    assert _git(repo, 'rev-list', '--count', f'{early_main}..{parked_branch}') == '4'
    assert _git(repo, 'merge-base', parked_branch, 'main') == early_main
    _git(repo, 'merge-base', '--is-ancestor', original_tip, parked_branch)


@pytest.mark.asyncio
class TestSpecimenRecut:
    """Task 2896's shape: work from before ruling D11-A must never be resumed."""

    async def test_cold_specimen_recut_then_redispatch(self, git_repo: Path):
        git_ops = GitOps(_config(warm_lane_pool=False), git_repo)
        early_main = await _get_head(git_repo)
        worktree = (await git_ops.create_worktree('2896')).path
        original_tip = await _seed_pre_ruling_work(git_repo, worktree)
        main_head = await _get_head(git_repo)
        persist = _RecordingPersist()

        result = await enforce_recut_ruling(
            SPECIMEN_TASK, ops=git_ops, persist=persist, now=EXECUTED_AT,
        )
        info = await git_ops.create_worktree('2896', recut_ruling=ruling_to_enforce(SPECIMEN_TASK))

        assert result.verdict is GateVerdict.EXECUTED
        assert result.parked is not None and result.parked.parked_branch is not None
        parked_branch = result.parked.parked_branch
        _assert_parked_intact(git_repo, parked_branch, early_main, original_tip)
        assert _git(git_repo, 'rev-list', '--count', 'main..task/2896') == '0'
        assert info.base_commit == main_head
        assert _resolve(git_repo, 'task/2896') == main_head
        record = persist.patches[0][RECUT_EXECUTED_KEY]
        assert record['parked_branch'] == parked_branch
        assert record['parked_tip'] == _resolve(git_repo, parked_branch)

        redispatch = {
            'id': '2896',
            'metadata': {**SPECIMEN_TASK['metadata'], **persist.patches[0]},
        }
        assert ruling_to_enforce(redispatch) is None
        again = await enforce_recut_ruling(
            redispatch, ops=git_ops, persist=persist, now=EXECUTED_AT,
        )
        info_again = await git_ops.create_worktree('2896', recut_ruling=ruling_to_enforce(redispatch))

        assert again.verdict is GateVerdict.PROCEED
        assert info_again.path == info.path
        assert _resolve(git_repo, 'task/2896') == main_head
        assert _parked_refs(git_repo, '2896') == [parked_branch]

    async def test_pooled_specimen_recut_takes_a_fresh_route(
        self, git_repo: Path, caplog: pytest.LogCaptureFixture,
    ):
        await _add_warm_lane_scripts(git_repo)
        git_ops = GitOps(_config(warm_lane_pool=True), git_repo, warm_lane_pool_size=1)
        early_main = await _get_head(git_repo)
        acquired = await git_ops.acquire_warm_lane('2896', early_main)
        assert isinstance(acquired, WorktreeInfo), f'Expected WorktreeInfo; got {acquired!r}'
        _write_lane_metadata(git_ops, acquired.path.name, '2896')
        original_tip = await _seed_pre_ruling_work(git_repo, acquired.path)
        main_head = await _get_head(git_repo)
        persist = _RecordingPersist()

        result = await enforce_recut_ruling(
            SPECIMEN_TASK, ops=git_ops, persist=persist, now=EXECUTED_AT,
        )
        caplog.clear()
        with caplog.at_level(logging.INFO, logger='orchestrator.git_ops'):
            info = await git_ops.create_worktree(
                '2896', recut_ruling=ruling_to_enforce(SPECIMEN_TASK),
            )

        assert result.verdict is GateVerdict.EXECUTED
        assert result.parked is not None and result.parked.parked_branch is not None
        _assert_parked_intact(git_repo, result.parked.parked_branch, early_main, original_tip)
        assert any(
            _route_logged(caplog, route)
            for route in (AcquireRoute.CREATE_ONCE_FRESH, AcquireRoute.RECYCLE)
        ), caplog.text
        for route in REUSE_ROUTES:
            assert not _route_logged(caplog, route), caplog.text
        assert _git(git_repo, 'rev-list', '--count', 'main..task/2896') == '0'
        assert info.base_commit == main_head
        record = persist.patches[0][RECUT_EXECUTED_KEY]
        assert record['parked_branch'] == result.parked.parked_branch
        assert record['relocated_meta_roots'] == list(result.parked.relocated_meta_roots)
