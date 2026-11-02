"""plan.json carries its own ``_base_commit`` (plans/task-meta-task-keying-prd.md §M2, task 3157)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from test_plan_revalidation_skip import _make

from orchestrator.artifacts import PLAN_BASE_COMMIT_KEY, TaskArtifacts, plan_base_commit
from orchestrator.workflow import WorkflowOutcome


def _one_step_plan() -> dict:
    return {
        'task_id': '101', 'title': 'T', 'analysis': 'a', 'files': ['lib.py'],
        'prerequisites': [],
        'steps': [{'id': 'step-1', 'type': 'impl', 'description': 'x',
                   'status': 'pending', 'commit': None}],
    }


def _dual_root_artifacts(tmp_path: Path) -> TaskArtifacts:
    worktree = tmp_path / 'worktrees' / 'lane-a'
    worktree.mkdir(parents=True)
    artifacts = TaskArtifacts.for_task(worktree, '101')
    artifacts.init('101', 'T', 'd', base_commit='fork111')
    artifacts.write_plan(_one_step_plan())
    return artifacts


def test_plan_base_commit_reads_a_stamped_sha():
    assert plan_base_commit({PLAN_BASE_COMMIT_KEY: 'abc123'}) == 'abc123'


@pytest.mark.parametrize('plan', [
    {},
    {PLAN_BASE_COMMIT_KEY: ''},
    {PLAN_BASE_COMMIT_KEY: None},
    {PLAN_BASE_COMMIT_KEY: 123},
    {PLAN_BASE_COMMIT_KEY: ['abc']},
], ids=['absent', 'empty', 'none', 'int', 'list'])
def test_plan_base_commit_is_none_for_an_absent_or_malformed_stamp(plan: dict):
    assert plan_base_commit(plan) is None


def test_stamp_with_base_writes_the_plan_not_the_dispatch_base(tmp_path: Path):
    artifacts = _dual_root_artifacts(tmp_path)

    artifacts.stamp_plan_provenance('s1', base_commit='plan222')

    assert artifacts.read_plan()[PLAN_BASE_COMMIT_KEY] == 'plan222'
    assert artifacts.read_base_commit() == 'fork111'


def test_stamp_without_base_keeps_the_existing_stamp(tmp_path: Path):
    artifacts = _dual_root_artifacts(tmp_path)
    artifacts.stamp_plan_provenance('s1', base_commit='plan222')

    artifacts.stamp_plan_provenance('s2')
    artifacts.stamp_plan_provenance('s3', task_text_fp='tfp:x')

    assert artifacts.read_plan()[PLAN_BASE_COMMIT_KEY] == 'plan222'


def test_unstamped_plan_stays_unstamped(tmp_path: Path):
    artifacts = _dual_root_artifacts(tmp_path)

    artifacts.stamp_plan_provenance('s1')

    assert PLAN_BASE_COMMIT_KEY not in artifacts.read_plan()


@pytest.mark.parametrize(('plan_base', 'old_plan_base', 'expected_from'), [
    ('planbase789', None, 'planbase789'),
    ('planbase789', 'oldmain123', 'planbase789'),
    (None, 'oldmain123', 'oldmain123'),
], ids=['cross-lane-redispatch-no-lane-base', 'stamp-wins-over-lane-metadata',
        'unstamped-plan-falls-back'])
@pytest.mark.asyncio
async def test_revalidation_diffs_from_the_plans_own_base(
    tmp_path: Path, plan_base: str | None, old_plan_base: str | None, expected_from: str,
):
    f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj', plan_base_commit=plan_base)
    f.wf._old_plan_base = old_plan_base

    outcome = await f.wf._plan()

    assert outcome == WorkflowOutcome.PLANNED
    assert f.invoke_called == []
    f.get_changed_files.assert_awaited_once_with(expected_from, 'newmain456')


@pytest.mark.parametrize('answer', [False, None], ids=['not-ancestor', 'git-error'])
@pytest.mark.asyncio
async def test_plan_stamp_off_main_history_is_not_trusted(tmp_path: Path, answer: bool | None):
    f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj', plan_base_commit='rewritten999')
    f.git_ops.is_ancestor_checked = AsyncMock(return_value=answer)

    await f.wf._plan()

    f.get_changed_files.assert_awaited_once_with('oldmain123', 'newmain456')
    f.git_ops.is_ancestor_checked.assert_awaited()
    assert f.git_ops.is_ancestor_checked.await_args.args[0] == 'rewritten999'
