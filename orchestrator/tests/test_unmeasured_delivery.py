"""Why a plan-files not-touched verdict did not measure the task's own work.

Every scene is real git under ``tmp_path`` (see ``_unmeasured_delivery_scenes``)
and every gate result is a :class:`PlanFilesTouchedResult` built directly, so
the diagnosis is exercised through its public call alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _merged_in_branch import git
from _unmeasured_delivery_scenes import (
    add_worktree,
    commit,
    seeded_repo,
    stopped_rebase,
    uncommitted_edits,
    work_on_sibling_ref,
)

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane.gates import (
    PLAN_FILES_NOT_TOUCHED_REASON_PREFIX,
    PlanFilesTouchedResult,
)
from orchestrator.merge_lane.unmeasured_delivery import (
    PLAN_DELIVERY_UNMEASURED_REASON_PREFIX,
    UnmeasuredCause,
    UnmeasuredDelivery,
    diagnose_unmeasured_delivery,
)
from orchestrator.unblock_types import BlockClass, classify_block_reason

pytestmark = pytest.mark.asyncio


def _git_ops(repo: Path) -> GitOps:
    config = GitConfig(
        main_branch='main', branch_prefix='task/', remote='origin',
        worktree_dir='.worktrees', push_after_advance=False,
    )
    return GitOps(config, repo)


async def _diagnose(
    repo: Path, worktree: Path, not_touched: list[str], *,
    branch: str = 'task/50', touched_set_error: str | None = None,
) -> UnmeasuredDelivery | None:
    check = PlanFilesTouchedResult(
        not_touched=not_touched, touched_set_error=touched_set_error,
    )
    return await diagnose_unmeasured_delivery(
        check, worktree, _git_ops(repo), branch=branch, task_id=branch.split('/')[-1],
    )


async def test_a_genuinely_untouched_file_has_no_unmeasured_cause(tmp_path: Path) -> None:
    repo = seeded_repo(tmp_path, {'a.py': 'a = 1\n', 'b.py': 'b = 1\n'})
    worktree = add_worktree(repo, tmp_path / 'wt', 'task/50')
    commit(worktree, {'a.py': 'a = 2\n'}, 'task work')

    assert await _diagnose(repo, worktree, ['b.py']) is None


async def test_an_unreadable_touched_set_explains_every_entry(tmp_path: Path) -> None:
    repo = seeded_repo(tmp_path, {'a.py': 'a = 1\n'})
    worktree = add_worktree(repo, tmp_path / 'wt', 'task/50')

    finding = await _diagnose(
        repo, worktree, ['a.py', 'b.py'], touched_set_error='fatal: bad revision',
    )

    assert finding is not None
    assert finding.cause is UnmeasuredCause.TOUCHED_SET_UNREADABLE
    assert finding.entries == ('a.py', 'b.py')
    assert finding.still_not_touched == ()
    assert 'fatal: bad revision' in finding.detail


async def test_a_stopped_rebase_reports_its_progress(tmp_path: Path) -> None:
    scene = stopped_rebase(tmp_path, 'task/60')

    finding = await _diagnose(
        scene.repo, scene.worktree, ['x.py', 'y.py'], branch='task/60',
    )

    assert finding is not None
    assert finding.cause is UnmeasuredCause.MID_REBASE
    assert finding.entries == ('x.py', 'y.py')
    assert 'step 1 of 2' in finding.detail
    assert 'task/60' in finding.detail
    assert 'x.py' in finding.detail


async def test_a_stopped_rebase_wins_over_uncommitted_edits(tmp_path: Path) -> None:
    scene = stopped_rebase(tmp_path, 'task/60')
    (scene.worktree / 'z.py').write_text('z = 2\n')

    finding = await _diagnose(
        scene.repo, scene.worktree, ['x.py', 'y.py', 'z.py'], branch='task/60',
    )

    assert finding is not None
    assert finding.cause is UnmeasuredCause.MID_REBASE
    assert finding.entries == ('x.py', 'y.py', 'z.py')


async def test_uncommitted_declared_edits_are_named(tmp_path: Path) -> None:
    scene = uncommitted_edits(tmp_path)

    finding = await _diagnose(
        scene.repo, scene.worktree, ['a.py', 'new.py', 'pkg', 'b.py'],
    )

    assert finding is not None
    assert finding.cause is UnmeasuredCause.UNCOMMITTED_EDITS
    assert finding.entries == ('a.py', 'new.py', 'pkg')
    assert finding.still_not_touched == ('b.py',)


async def test_declared_work_on_a_sibling_ref_is_named(tmp_path: Path) -> None:
    scene = work_on_sibling_ref(tmp_path)

    finding = await _diagnose(scene.repo, scene.worktree, ['a.py', 'b.py'])

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

    assert await _diagnose(scene.repo, scene.worktree, ['a.py']) is None


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
