"""A re-dispatch after a branch-swapping quarantine reviews its new branch afresh (task 3663).

The 3404/2432 shape, end to end: a quarantine renamed ``task/<id>`` aside
after an amendment round, and every review record survived in the task root.
Dispatch setup must retire that lineage, so the new branch gets its own
amendment round and the task-2523 adjudicator is not run against suggestions
settled on the parked branch.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import pydantic_spec

from orchestrator.agents.review_briefing import ReviewBriefing
from orchestrator.artifacts import ReviewAggregation, TaskArtifacts
from orchestrator.config import OrchestratorConfig
from orchestrator.git_ops import WorktreeInfo
from orchestrator.quarantine_provenance import stamp_quarantined_from
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

_TASK_ID = '3663'
_BASE_SHA = 'a' * 40
_PARKING_NAME = f'{_TASK_ID}-20261112T000000Z'
_SETTLED = {
    'severity': 'suggestion',
    'location': 'src/foo/helpers.py:3',
    'category': 'naming',
    'description': 'helper name hides its side effect',
    'suggested_fix': 'rename helper',
}
_IN_SCOPE_SUGGESTION = {
    'reviewer': 'analyst',
    'severity': 'suggestion',
    'location': 'src/foo/bar.py:42',
    'category': 'coverage',
    'description': 'Missing edge case',
    'suggested_fix': 'Add a test',
}


def _worktree(tmp_path: Path) -> Path:
    return tmp_path / '.worktrees' / _TASK_ID


def _seed_parked_lineage(tmp_path: Path, *, stamped: bool) -> TaskArtifacts:
    worktree = _worktree(tmp_path)
    worktree.mkdir(parents=True)
    artifacts = TaskArtifacts.for_task(worktree, _TASK_ID)
    artifacts.init(_TASK_ID, 'T', 'd', base_commit=_BASE_SHA)
    artifacts.write_plan({
        'task_id': _TASK_ID,
        'title': 'T',
        'steps': [{'id': 'step-1', 'type': 'impl', 'description': 'x', 'status': 'done', 'commit': 'abc'}],
    })
    artifacts.set_review_counters(amendment_rounds_total=1)
    archive = artifacts.review_archive_dir('amend-1')
    archive.mkdir(parents=True)
    (archive / 'analyst.json').write_text(json.dumps({'verdict': 'ISSUES_FOUND', 'issues': [_SETTLED]}))
    if stamped:
        assert stamp_quarantined_from(artifacts, {
            'branch': f'refs/heads/task/{_PARKING_NAME}',
            'tip_sha': 'b' * 40,
            'unique_commit_count': 3,
            'quarantined_at': datetime.now(UTC).isoformat(),
            'reason': 'orphan-reaper',
            'parking': f'/q/{_PARKING_NAME}',
            'stored_title': None,
        })
    return artifacts


def _make_workflow(tmp_path: Path) -> TaskWorkflow:
    assignment = MagicMock()
    assignment.task_id = _TASK_ID
    assignment.task = {'id': _TASK_ID, 'title': 'T', 'description': 'd'}
    assignment.modules = ['src/foo']

    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.lock_depth = 2
    config.project_root = tmp_path
    config.max_review_cycles = 2
    config.max_amendment_rounds = 1
    config.suppress_resettled_review_suggestions = True
    config.review.reviewer_briefing = ReviewBriefing.DIFF_ONLY
    config.models = MagicMock()
    config.models.reviewer = 'sonnet'

    scheduler = MagicMock()
    scheduler.set_task_status = AsyncMock(return_value=None)
    scheduler.update_task = AsyncMock(return_value=True)
    scheduler.get_status = AsyncMock(return_value=None)

    git_ops = MagicMock()
    git_ops.worktree_base = tmp_path / '.worktrees'
    git_ops.create_worktree = AsyncMock(
        return_value=WorktreeInfo(path=_worktree(tmp_path), base_commit=_BASE_SHA),
    )
    git_ops.branch_tip_if_no_commits_beyond = AsyncMock(return_value=None)
    git_ops.get_head_tree_hash = AsyncMock(return_value='TREE1')
    git_ops.get_new_side_changed_line_ranges = AsyncMock(return_value={})
    git_ops.has_uncommitted_work = AsyncMock(return_value=False)

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
    )
    wf._sync_worktree_venvs = AsyncMock()
    wf._execute_iterations = AsyncMock(return_value=WorkflowOutcome.DONE)
    wf._verify_debugfix_loop = AsyncMock(return_value=WorkflowOutcome.DONE)
    wf._review = AsyncMock(return_value=ReviewAggregation(
        has_blocking_issues=False,
        blocking_issues=[],
        suggestions=[_IN_SCOPE_SUGGESTION],
        reviews={'analyst': {}},
    ))
    wf._route_review_suggestions_to_curator = AsyncMock()
    wf._write_suggestions_to_memory = AsyncMock()
    wf._amend = AsyncMock(return_value=True)
    wf._invoke = AsyncMock()
    return wf


async def _setup(wf: TaskWorkflow) -> TaskArtifacts:
    with patch('orchestrator.workflow._run', new=AsyncMock(return_value=(0, '', ''))):
        await wf._setup_worktree_and_artifacts(f'task/{_TASK_ID}')
    assert wf.artifacts is not None
    return wf.artifacts


@pytest.mark.asyncio
class TestRedispatchAfterQuarantine:
    async def test_setup_retires_the_parked_lineage(self, tmp_path: Path):
        _seed_parked_lineage(tmp_path, stamped=True)
        wf = _make_workflow(tmp_path)

        artifacts = await _setup(wf)

        assert artifacts.get_amendment_rounds_total() == 0
        assert artifacts.review_archive_dirs() == ()
        retired = artifacts.artifact_path(f'retired-reviews/{_PARKING_NAME}/reviews-amend-1/analyst.json')
        assert retired.is_file()

    async def test_the_new_branch_gets_its_own_amendment_round(self, tmp_path: Path):
        _seed_parked_lineage(tmp_path, stamped=True)
        wf = _make_workflow(tmp_path)
        await _setup(wf)

        outcome = await wf._execute_verify_review_loop()

        assert outcome == WorkflowOutcome.DONE
        cast(AsyncMock, wf._amend).assert_awaited_once()
        cast(AsyncMock, wf._invoke).assert_not_called()

    async def test_without_a_stamp_the_parked_lineage_still_governs_the_review(self, tmp_path: Path):
        _seed_parked_lineage(tmp_path, stamped=False)
        wf = _make_workflow(tmp_path)
        artifacts = await _setup(wf)

        assert artifacts.get_amendment_rounds_total() == 1

        outcome = await wf._execute_verify_review_loop()

        assert outcome == WorkflowOutcome.DONE
        cast(AsyncMock, wf._amend).assert_not_called()
        cast(AsyncMock, wf._invoke).assert_awaited()
        assert artifacts.get_amendment_rounds_total() == 1
