"""rehydrate_merge_halt re-tests project_root hygiene before re-asserting a preserved halt (task 3876).

Driven through the module's public signature over real git checkouts, so the
predicate is measured exactly as a restarted orchestrator measures it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from _checkout_hygiene_fixtures import dirty_tracked_file, leave_unmerged, run_git
from _git_fixtures import seed_repo
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from test_halt_owner import _FakeMergeWorker, _make_wip_esc

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps
from orchestrator.merge_halt_rehydration import HALT_NOT_REHYDRATED, rehydrate_merge_halt

_GIT_CONFIG = GitConfig(
    main_branch='main', branch_prefix='task/', remote='origin',
    worktree_dir='.worktrees', push_after_advance=False,
)
_LOGGER = 'orchestrator.merge_halt_rehydration'


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return seed_repo(tmp_path / 'repo')


@pytest.fixture
def queue(tmp_path: Path) -> EscalationQueue:
    return EscalationQueue(tmp_path / 'escalations')


async def _rehydrate(
    queue: EscalationQueue, project_root: Path, worker: _FakeMergeWorker,
) -> str | None:
    return await rehydrate_merge_halt(
        queue.get_pending(),
        probe=GitOps(_GIT_CONFIG, project_root),
        worker=worker,
        records=queue,
    )


@pytest.mark.asyncio
async def test_clean_project_root_releases_instead_of_halting(
    repo: Path, queue: EscalationQueue, caplog: pytest.LogCaptureFixture,
):
    esc = _make_wip_esc(queue, '4843')
    worker = _FakeMergeWorker()

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        owner = await _rehydrate(queue, repo, worker)

    assert owner is None
    assert not worker.is_wip_halted
    released = queue.get(esc.id)
    assert released is not None
    assert released.halt_released_at is not None
    assert str(repo) in released.halt_release_note
    assert any(
        r.levelno == logging.WARNING and HALT_NOT_REHYDRATED in r.getMessage()
        and esc.id in r.getMessage()
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]


@pytest.mark.asyncio
async def test_conflict_markers_in_project_root_keep_the_halt(
    repo: Path, queue: EscalationQueue,
):
    esc = _make_wip_esc(queue, '4843')
    leave_unmerged(repo, 'side')
    worker = _FakeMergeWorker()

    owner = await _rehydrate(queue, repo, worker)

    assert owner == esc.id
    assert worker.is_wip_halted
    assert worker.halt_owner_esc_id == esc.id
    held = queue.get(esc.id)
    assert held is not None
    assert held.halt_released_at is None


@pytest.mark.asyncio
async def test_tracked_wip_in_project_root_keeps_the_halt(
    repo: Path, queue: EscalationQueue,
):
    esc = _make_wip_esc(queue, '3457', category='stash_failed')
    dirty_tracked_file(repo)
    worker = _FakeMergeWorker()

    owner = await _rehydrate(queue, repo, worker)

    assert owner == esc.id
    assert worker.is_wip_halted


@pytest.mark.asyncio
async def test_task_worktree_conflict_over_a_clean_project_root_is_released(
    tmp_path: Path, repo: Path, queue: EscalationQueue,
):
    task_worktree = tmp_path / 'task-worktree'
    run_git(repo, 'worktree', 'add', '-q', '-b', 'task/4843', str(task_worktree))
    leave_unmerged(task_worktree, 'side-4843')
    esc = _make_wip_esc(queue, '4843', worktree=str(task_worktree))
    worker = _FakeMergeWorker()

    owner = await _rehydrate(queue, repo, worker)

    assert owner is None
    assert not worker.is_wip_halted
    released = queue.get(esc.id)
    assert released is not None
    assert released.halt_released_at is not None


@pytest.mark.asyncio
async def test_task_worktree_conflict_over_a_dirty_project_root_re_owns_the_halt(
    tmp_path: Path, repo: Path, queue: EscalationQueue,
):
    task_worktree = tmp_path / 'task-worktree'
    run_git(repo, 'worktree', 'add', '-q', '-b', 'task/4843', str(task_worktree))
    leave_unmerged(task_worktree, 'side-4843')
    esc = _make_wip_esc(queue, '4843', worktree=str(task_worktree))
    dirty_tracked_file(repo)
    worker = _FakeMergeWorker()

    owner = await _rehydrate(queue, repo, worker)

    assert owner == esc.id
    assert worker.is_wip_halted
    held = queue.get(esc.id)
    assert held is not None
    assert held.halt_released_at is None


@pytest.mark.asyncio
async def test_unmeasurable_project_root_keeps_the_halt(
    tmp_path: Path, queue: EscalationQueue,
):
    not_a_repo = tmp_path / 'not-a-repo'
    not_a_repo.mkdir()
    esc = _make_wip_esc(queue, '4843')
    worker = _FakeMergeWorker()

    owner = await _rehydrate(queue, not_a_repo, worker)

    assert owner == esc.id
    assert worker.is_wip_halted
    held = queue.get(esc.id)
    assert held is not None
    assert held.halt_released_at is None


@pytest.mark.asyncio
async def test_released_record_never_re_arms(repo: Path, queue: EscalationQueue):
    esc = _make_wip_esc(queue, '4843')
    first = queue.release_merge_halt(esc.id, note='clean at an earlier restart')
    assert first is not None
    dirty_tracked_file(repo)
    worker = _FakeMergeWorker()

    owner = await _rehydrate(queue, repo, worker)

    assert owner is None
    assert not worker.is_wip_halted
    after = queue.get(esc.id)
    assert after is not None
    assert after.halt_released_at == first.halt_released_at
    assert after.halt_release_note == 'clean at an earlier restart'


@pytest.mark.asyncio
async def test_several_holding_records_keep_the_advisory_and_the_newest_owner(
    repo: Path, queue: EscalationQueue, caplog: pytest.LogCaptureFixture,
):
    _make_wip_esc(queue, '7001', timestamp='2026-05-27T09:00:00+00:00')
    newest = _make_wip_esc(queue, '7002', timestamp='2026-05-27T10:00:00+00:00')
    dirty_tracked_file(repo)
    worker = _FakeMergeWorker()

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        owner = await _rehydrate(queue, repo, worker)

    assert owner == newest.id
    assert worker.halt_owner_esc_id == newest.id
    assert any(
        'qualifying level-≥1 record(s) found' in r.getMessage() for r in caplog.records
    ), [r.getMessage() for r in caplog.records]


@pytest.mark.asyncio
async def test_several_records_over_a_clean_project_root_are_all_released(
    repo: Path, queue: EscalationQueue,
):
    older = _make_wip_esc(queue, '7001', timestamp='2026-05-27T09:00:00+00:00')
    newer = _make_wip_esc(queue, '7002', timestamp='2026-05-27T10:00:00+00:00')
    worker = _FakeMergeWorker()

    owner = await _rehydrate(queue, repo, worker)

    assert owner is None
    assert not worker.is_wip_halted
    for esc in (older, newer):
        record = queue.get(esc.id)
        assert record is not None
        assert record.halt_released_at is not None


class _FirstReleaseFails:
    """HaltRecords whose first release raises OSError; it records every id it is asked for."""

    def __init__(self) -> None:
        self.attempted: list[str] = []

    def release_merge_halt(self, escalation_id: str, *, note: str) -> Escalation | None:
        self.attempted.append(escalation_id)
        if len(self.attempted) == 1:
            raise OSError('escalation dir is read-only')
        return None


@pytest.mark.asyncio
async def test_a_failed_release_stamp_is_logged_and_the_rest_are_still_released(
    repo: Path, queue: EscalationQueue, caplog: pytest.LogCaptureFixture,
):
    older = _make_wip_esc(queue, '7001', timestamp='2026-05-27T09:00:00+00:00')
    newer = _make_wip_esc(queue, '7002', timestamp='2026-05-27T10:00:00+00:00')
    records = _FirstReleaseFails()
    worker = _FakeMergeWorker()

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        owner = await rehydrate_merge_halt(
            queue.get_pending(),
            probe=GitOps(_GIT_CONFIG, repo),
            worker=worker,
            records=records,
        )

    assert owner is None
    assert not worker.is_wip_halted
    assert sorted(records.attempted) == sorted([older.id, newer.id])
    assert any(
        'Could not stamp the merge-halt release' in r.getMessage()
        and records.attempted[0] in r.getMessage()
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]


@pytest.mark.asyncio
async def test_no_candidates_touch_no_git(tmp_path: Path, queue: EscalationQueue):
    worker = _FakeMergeWorker()

    owner = await rehydrate_merge_halt(
        [],
        probe=GitOps(_GIT_CONFIG, tmp_path / 'does-not-exist'),
        worker=worker,
        records=queue,
    )

    assert owner is None
    assert not worker.is_wip_halted
