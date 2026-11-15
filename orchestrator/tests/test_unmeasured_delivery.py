"""Why a plan-files not-touched verdict did not measure the task's own work.

Every scene is real git under ``tmp_path`` (see ``_unmeasured_delivery_scenes``)
and every gate result is a :class:`PlanFilesTouchedResult` built directly, so
the diagnosis is exercised through its two public calls alone:
``diagnose_unmeasured_branch`` for causes inside the task's own worktree, and
``find_work_on_sibling_ref`` for declared work left on a ``task/<id>-*`` ref
that no re-cut ruling has superseded, explaining a task branch that carries
no commit of its own. An unreadable touched set comes from the real gate.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from _merged_in_branch import git
from _unmeasured_delivery_scenes import (
    SiblingScene,
    add_worktree,
    commit,
    seeded_repo,
    stopped_on_main_conflict,
    stopped_rebase,
    uncommitted_edits,
    work_on_sibling_ref,
)
from _workflow_helpers import _make_git_ops

from orchestrator.merge_lane.gates import (
    PLAN_FILES_NOT_TOUCHED_REASON_PREFIX,
    PlanFilesTouchedResult,
    _check_plan_files_touched_in_branch,
)
from orchestrator.merge_lane.unmeasured_delivery import (
    PLAN_DELIVERY_UNMEASURED_REASON_PREFIX,
    UnmeasuredCause,
    UnmeasuredDelivery,
    diagnose_unmeasured_branch,
    find_work_on_sibling_ref,
)
from orchestrator.recut_ruling import RecutSupersession
from orchestrator.unblock_types import BlockClass, classify_block_reason

pytestmark = pytest.mark.asyncio

_VANISHED_COMMIT = 'deadbeef' * 5


async def _diagnose(
    worktree: Path, not_touched: list[str], *, task_id: str = '50',
) -> UnmeasuredDelivery | None:
    check = PlanFilesTouchedResult(not_touched=not_touched)
    return await diagnose_unmeasured_branch(check, worktree, task_id=task_id)


async def _find_sibling(
    scene: SiblingScene, not_touched: list[str], *,
    superseded: RecutSupersession | None = None,
) -> UnmeasuredDelivery | None:
    return await find_work_on_sibling_ref(
        PlanFilesTouchedResult(not_touched=not_touched), _make_git_ops(scene.repo),
        branch='task/50', head=git(scene.worktree, 'rev-parse', 'HEAD'),
        task_id='50', superseded=superseded,
    )


async def test_a_genuinely_untouched_file_has_no_unmeasured_cause(tmp_path: Path) -> None:
    repo = seeded_repo(tmp_path, {'a.py': 'a = 1\n', 'b.py': 'b = 1\n'})
    worktree = add_worktree(repo, tmp_path / 'wt', 'task/50')
    commit(worktree, {'a.py': 'a = 2\n'}, 'task work')

    assert await _diagnose(worktree, ['b.py']) is None


async def test_an_unreadable_touched_set_explains_every_entry_in_gits_words(
    tmp_path: Path,
) -> None:
    repo = seeded_repo(tmp_path, {'a.py': 'a = 1\n'})
    worktree = add_worktree(repo, tmp_path / 'wt', 'task/50')
    check = await _check_plan_files_touched_in_branch(
        ['a.py', 'b.py'], _VANISHED_COMMIT, git(worktree, 'rev-parse', 'HEAD'),
        _make_git_ops(repo),
    )

    finding = await diagnose_unmeasured_branch(check, worktree, task_id='50')

    assert finding is not None
    assert finding.cause is UnmeasuredCause.TOUCHED_SET_UNREADABLE
    assert finding.entries == ('a.py', 'b.py')
    assert finding.still_not_touched == ()
    assert f'fatal: Invalid revision range {_VANISHED_COMMIT}..' in finding.block_reason()


async def test_a_probe_that_raises_reports_no_cause_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger='orchestrator.merge_lane.unmeasured_delivery'):
        finding = await _diagnose(tmp_path / 'no-such-worktree', ['a.py'])

    assert finding is None
    assert 'in-branch diagnosis raised for task_id=50' in caplog.text


async def test_a_stopped_rebase_reports_its_progress(tmp_path: Path) -> None:
    scene = stopped_rebase(tmp_path, 'task/60')

    finding = await _diagnose(scene.worktree, ['x.py', 'y.py'], task_id='60')

    assert finding is not None
    assert finding.cause is UnmeasuredCause.OPERATION_IN_PROGRESS
    assert finding.entries == ('x.py', 'y.py')
    assert 'step 1 of 2' in finding.detail
    assert 'task/60' in finding.detail
    assert 'unmerged paths: x.py' in finding.detail


async def test_a_stopped_rebase_wins_over_uncommitted_edits(tmp_path: Path) -> None:
    scene = stopped_rebase(tmp_path, 'task/60')
    (scene.worktree / 'z.py').write_text('z = 2\n')

    finding = await _diagnose(scene.worktree, ['x.py', 'y.py', 'z.py'], task_id='60')

    assert finding is not None
    assert finding.cause is UnmeasuredCause.OPERATION_IN_PROGRESS
    assert finding.entries == ('x.py', 'y.py', 'z.py')


@pytest.mark.parametrize('operation', ['merge', 'cherry-pick'])
async def test_a_stopped_merge_or_cherry_pick_wins_over_uncommitted_edits(
    tmp_path: Path, operation: str,
) -> None:
    scene = stopped_on_main_conflict(tmp_path, operation)
    (scene.worktree / 'y.py').write_text('y = 2\n')

    finding = await _diagnose(scene.worktree, ['x.py', 'y.py'])

    assert finding is not None
    assert finding.cause is UnmeasuredCause.OPERATION_IN_PROGRESS
    assert finding.entries == ('x.py', 'y.py')
    main_tip = git(scene.repo, 'rev-parse', 'main')
    assert f'a {operation} of {main_tip[:10]} stopped' in finding.detail
    assert 'unmerged paths: x.py' in finding.detail


async def test_uncommitted_declared_edits_are_named(tmp_path: Path) -> None:
    scene = uncommitted_edits(tmp_path)

    finding = await _diagnose(scene.worktree, ['a.py', 'new.py', 'pkg', 'b.py'])

    assert finding is not None
    assert finding.cause is UnmeasuredCause.UNCOMMITTED_EDITS
    assert finding.entries == ('a.py', 'new.py', 'pkg')
    assert finding.still_not_touched == ('b.py',)


async def test_declared_work_on_a_sibling_ref_is_named(tmp_path: Path) -> None:
    scene = work_on_sibling_ref(tmp_path)

    finding = await _find_sibling(scene, ['a.py', 'b.py'])

    assert finding is not None
    assert finding.cause is UnmeasuredCause.WORK_ON_SIBLING_REF
    assert finding.entries == ('a.py',)
    assert finding.still_not_touched == ('b.py',)
    assert scene.sibling_ref in finding.detail
    assert scene.sibling_tip[:10] in finding.detail
    assert 'task/500' not in finding.block_reason()


async def test_a_sibling_ref_already_merged_into_main_is_ignored(tmp_path: Path) -> None:
    scene = work_on_sibling_ref(tmp_path)
    git(scene.repo, 'merge', '-q', '--no-ff', '--no-edit', scene.sibling_ref)

    assert await _find_sibling(scene, ['a.py']) is None


async def test_a_sibling_ref_whose_content_landed_by_another_commit_is_ignored(
    tmp_path: Path,
) -> None:
    scene = work_on_sibling_ref(tmp_path)
    commit(scene.repo, {'c.py': 'c = 1\n'}, 'main advances')
    git(scene.repo, 'cherry-pick', scene.sibling_tip)

    assert await _find_sibling(scene, ['a.py']) is None


async def test_a_sibling_never_explains_a_branch_carrying_its_own_work(
    tmp_path: Path,
) -> None:
    scene = work_on_sibling_ref(tmp_path)
    commit(scene.worktree, {'b.py': 'b = 2\n'}, 'own work')

    assert await _find_sibling(scene, ['a.py']) is None


async def test_the_in_branch_diagnosis_never_names_a_sibling(tmp_path: Path) -> None:
    scene = work_on_sibling_ref(tmp_path)

    assert await _diagnose(scene.worktree, ['a.py']) is None


async def test_a_sibling_parked_by_a_recut_is_ignored(tmp_path: Path) -> None:
    scene = work_on_sibling_ref(tmp_path)
    superseded = RecutSupersession(parked_branch=scene.sibling_ref, executed_at=None)

    assert await _find_sibling(scene, ['a.py'], superseded=superseded) is None


async def test_a_sibling_predating_an_executed_recut_is_ignored(tmp_path: Path) -> None:
    scene = work_on_sibling_ref(tmp_path)
    superseded = RecutSupersession(
        parked_branch='task/50-20260101T000000Z',
        executed_at=datetime.now(UTC) + timedelta(seconds=1),
    )

    assert await _find_sibling(scene, ['a.py'], superseded=superseded) is None


async def test_a_sibling_committed_after_the_recut_is_still_named(tmp_path: Path) -> None:
    scene = work_on_sibling_ref(tmp_path)
    superseded = RecutSupersession(
        parked_branch='task/50-20200101T000000Z',
        executed_at=datetime(2020, 1, 1, tzinfo=UTC),
    )

    finding = await _find_sibling(scene, ['a.py'], superseded=superseded)

    assert finding is not None
    assert finding.cause is UnmeasuredCause.WORK_ON_SIBLING_REF
    assert finding.entries == ('a.py',)


def _one_finding_per_cause() -> list[UnmeasuredDelivery]:
    return [
        UnmeasuredDelivery(
            cause=cause, entries=('a.py',), still_not_touched=('b.py',), detail='d',
        )
        for cause in UnmeasuredCause
    ]


@pytest.mark.parametrize('finding', _one_finding_per_cause(), ids=lambda f: f.cause.value)
async def test_the_reason_never_reads_as_a_non_delivery(finding: UnmeasuredDelivery) -> None:
    reason = finding.block_reason()

    assert reason.startswith(PLAN_DELIVERY_UNMEASURED_REASON_PREFIX)
    assert not reason.startswith(PLAN_FILES_NOT_TOUCHED_REASON_PREFIX)
    assert 'has not delivered' not in reason
    assert classify_block_reason(reason) is not BlockClass.MERGE_VERIFY_RED


async def test_each_cause_suggests_its_own_action() -> None:
    actions = [finding.suggested_action for finding in _one_finding_per_cause()]

    assert len(set(actions)) == len(UnmeasuredCause) == 4
    assert 'investigate_and_retry' not in actions
