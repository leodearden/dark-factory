"""The plan stamps that must PRESERVE the task-text fingerprint, never mint one (task 4031).

Only an authoring stamp may record fp(self.task): the per-implementer-iteration
stamp, the post-review replan stamp and the recovered ``initial_plan`` stamp all
run against a plan written before the current task text was read, so minting
there would certify an old plan against amended text and the next dispatch's
amendment check would pass silently.

The harnesses are borrowed from the files that own each seam, not copied.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import pydantic_spec
from shared.task_text_fingerprint import task_text_fingerprint
from test_workflow_metadata_files_reconcile import (
    _blocking_then_pass_reviews,
    _make_replan_workflow,
)
from test_workflow_truthful_commit_stamp import _make_mock_workflow

from orchestrator.agents.invoke import AgentResult
from orchestrator.artifacts import PLAN_TASK_TEXT_FP_KEY, TaskArtifacts
from orchestrator.config import OrchestratorConfig
from orchestrator.scheduler import files_to_modules
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

_AUTHORED_FP = task_text_fingerprint('d', None)


def _plan(task_id: str) -> dict:
    return {
        'task_id': task_id, 'title': 'T', 'analysis': 'a', 'files': ['lib.py'],
        'prerequisites': [],
        'steps': [{'id': 'step-1', 'type': 'impl', 'description': 'x',
                   'status': 'pending', 'commit': None}],
    }


@pytest.mark.asyncio
async def test_iteration_stamp_keeps_the_authored_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    wf = _make_mock_workflow(tmp_path=tmp_path, max_execute_iterations=1)
    artifacts = wf.artifacts
    assert artifacts is not None
    artifacts.write_plan(_plan('2759'))
    artifacts.stamp_plan_provenance(wf.session_id, task_text_fp=_AUTHORED_FP)
    wf.task['description'] = 'd, amended mid-execute'
    stamp = MagicMock(wraps=artifacts.stamp_plan_provenance)
    for target, name, value in (
        (artifacts, 'stamp_plan_provenance', stamp),
        (artifacts, 'get_pending_steps', MagicMock(return_value=[{'id': 'step-1'}])),
        (artifacts, 'validate_plan_owner', MagicMock(return_value=True)),
        (wf, '_invoke', AsyncMock(return_value=AgentResult(
            success=True, output='ok', timed_out=False, turns=2, cost_usd=0.0,
        ))),
        (wf, '_check_escalations', MagicMock(return_value=[])),
        (wf, '_get_head_commit', AsyncMock(return_value='const_sha')),
        (wf.git_ops, 'has_uncommitted_work', AsyncMock(return_value=False)),
        (wf.briefing, 'build_implementer_prompt', AsyncMock(return_value='impl-prompt')),
    ):
        monkeypatch.setattr(target, name, value)

    await wf._execute_iterations()

    stamp.assert_called()
    assert artifacts.read_plan()[PLAN_TASK_TEXT_FP_KEY] == _AUTHORED_FP


def _wiping_replan(wf: TaskWorkflow) -> AsyncMock:
    """A replan whose create_plan() rewrote plan.json from scratch."""
    async def _side_effect(_reviews) -> None:
        assert wf.artifacts is not None
        wf.artifacts.write_plan({**_plan(wf.task_id), '_finalized_at': '2026-10-18T00:00:00+00:00'})

    return AsyncMock(side_effect=_side_effect)


@pytest.mark.parametrize('stamped_fp', [_AUTHORED_FP, None], ids=['stamped', 'unstamped'])
@pytest.mark.asyncio
async def test_post_review_replan_restores_the_prior_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stamped_fp: str | None,
):
    files = ['a/b/c/d/e.py']
    wf, scheduler = _make_replan_workflow(
        tmp_path=tmp_path, initial_files=files, modules=files_to_modules(files, 2),
    )
    assert wf.artifacts is not None
    wf.artifacts.stamp_plan_provenance(wf.session_id, task_text_fp=stamped_fp)
    wf.task['description'] = 'd, amended before review'
    replan = _wiping_replan(wf)
    monkeypatch.setattr(wf, '_execute_iterations', AsyncMock(return_value=WorkflowOutcome.DONE))
    monkeypatch.setattr(wf, '_review', _blocking_then_pass_reviews())
    monkeypatch.setattr(wf, '_replan', replan)

    outcome = await wf._execute_verify_review_loop()

    assert outcome == WorkflowOutcome.DONE
    replan.assert_awaited_once()
    assert wf.artifacts.read_plan().get(PLAN_TASK_TEXT_FP_KEY) == stamped_fp
    assert not [md for _tid, md in scheduler.update_task_calls
                if isinstance(md, dict) and 'plan_text_fp' in md]


class _StopAfterStamp(Exception):
    pass


def _recovering_workflow(
    tmp_path: Path, initial_plan: dict, lock_plan: MagicMock, monkeypatch: pytest.MonkeyPatch,
) -> TaskWorkflow:
    assignment = MagicMock()
    assignment.task_id = '4031'
    assignment.task = {'id': '4031', 'title': 'T', 'description': 'd, amended', 'metadata': {}}
    assignment.modules = ['lib']
    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    config.git.branch_prefix = 'task/'
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0
    config.project_root = tmp_path / 'proj'
    scheduler = MagicMock()
    scheduler.update_task = AsyncMock(return_value=True)
    scheduler.set_task_status = AsyncMock()
    scheduler.get_status = AsyncMock(return_value='blocked')
    scheduler.get_task = AsyncMock(return_value={'id': '4031', 'metadata': {}})
    queue = MagicMock()
    queue.get_by_task = MagicMock(return_value=[])
    wf = TaskWorkflow(
        assignment=assignment, config=config, git_ops=MagicMock(),
        scheduler=scheduler, briefing=MagicMock(), mcp=MagicMock(),
        escalation_queue=queue, initial_plan=initial_plan,
    )
    worktree = tmp_path / 'wt'
    worktree.mkdir(parents=True)
    wf.worktree = worktree
    wf.artifacts = TaskArtifacts(worktree)
    wf.artifacts.init('4031', 'T', 'd')
    monkeypatch.setattr(wf.artifacts, 'lock_plan', lock_plan)
    monkeypatch.setattr(wf, '_mark_blocked', AsyncMock(return_value=WorkflowOutcome.BLOCKED))
    return wf


@pytest.mark.parametrize('recovered_fp', [_AUTHORED_FP, None], ids=['stamped', 'unstamped'])
@pytest.mark.asyncio
async def test_recovered_initial_plan_keeps_its_own_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recovered_fp: str | None,
):
    initial_plan = _plan('4031')
    if recovered_fp is not None:
        initial_plan[PLAN_TASK_TEXT_FP_KEY] = recovered_fp
    lock_plan = MagicMock(side_effect=_StopAfterStamp)
    wf = _recovering_workflow(tmp_path, initial_plan, lock_plan, monkeypatch)

    with (
        patch.object(wf, '_setup_worktree_and_artifacts', new=AsyncMock()),
        patch.object(wf, '_recover_if_already_merged', new=AsyncMock(return_value=None)),
        patch.object(wf, '_resume_merge_retry_if_pending', new=AsyncMock(return_value=None)),
    ):
        await wf.run()

    assert wf.artifacts is not None
    lock_plan.assert_called_once()
    plan = wf.artifacts.read_plan()
    assert plan['_session_id'] == wf.session_id
    assert plan.get(PLAN_TASK_TEXT_FP_KEY) == recovered_fp
