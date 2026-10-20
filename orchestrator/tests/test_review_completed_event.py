"""``TaskWorkflow._execute_verify_review_loop`` records one ``review_completed`` row per closed
review cycle (client-feed θ). The loop runs for real against a real EventStore; EXECUTE, VERIFY,
the reviewer, the architect replan and the amendment are stubbed, as in
test_workflow_review_persistence.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec
from _review_fixtures import review_issue

from orchestrator.artifacts import ReviewAggregation, TaskArtifacts
from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventStore
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

pytestmark = pytest.mark.asyncio

TASK_ID = '6967'


def _make_workflow(tmp_path: Path, *, max_review_cycles: int = 2) -> TaskWorkflow:
    assignment = MagicMock()
    assignment.task_id = TASK_ID
    assignment.task = {'id': TASK_ID, 'title': 'T', 'description': 'd'}
    assignment.modules = ['src/foo']

    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.max_review_cycles = max_review_cycles
    config.max_amendment_rounds = 1
    config.lock_depth = 2
    config.project_root = tmp_path / 'proj'
    config.models = MagicMock()
    config.models.reviewer = 'sonnet'
    config.suppress_resettled_review_suggestions = False

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=MagicMock(),
        scheduler=MagicMock(),
        briefing=MagicMock(),
        mcp=MagicMock(),
        event_store=EventStore(tmp_path / 'runs.db', run_id='run-6967'),
    )
    worktree = tmp_path / 'wt'
    worktree.mkdir(parents=True, exist_ok=True)
    artifacts = TaskArtifacts(worktree)
    artifacts.init(TASK_ID, 'T', 'd')
    artifacts.write_plan({
        'task_id': TASK_ID,
        'title': 'T',
        'files': [],
        'analysis': 'a',
        'prerequisites': [],
        'steps': [{'id': 'step-1', 'type': 'impl', 'description': 'd', 'status': 'done',
                   'commit': 'abc123', 'prerequisites': [], 'files': []}],
    })
    artifacts.stamp_plan_provenance(wf.session_id)
    wf.artifacts = artifacts
    wf.worktree = worktree

    wf._execute_iterations = AsyncMock(return_value=WorkflowOutcome.DONE)
    wf._verify_debugfix_loop = AsyncMock(return_value=WorkflowOutcome.DONE)
    wf._get_head_commit = AsyncMock(return_value='tip')
    wf.git_ops.get_head_tree_hash = AsyncMock(return_value='TREE1')
    wf.git_ops.get_new_side_changed_line_ranges = AsyncMock(return_value={})
    wf.git_ops.has_uncommitted_work = AsyncMock(return_value=False)
    wf._route_review_suggestions_to_curator = AsyncMock()
    wf._write_suggestions_to_memory = AsyncMock()
    wf._replan = AsyncMock()
    wf._amend = AsyncMock(return_value=True)
    wf._commit_amendment_wip = AsyncMock()
    wf._suggestions_in_scope = lambda suggestions: []
    return wf


def _aggregation(blocking: list[dict], suggestions: list[dict]) -> ReviewAggregation:
    return ReviewAggregation(
        has_blocking_issues=bool(blocking),
        blocking_issues=list(blocking),
        suggestions=list(suggestions),
        reviews={'reviewer_a': {}},
    )


def _review_completed_rows(wf: TaskWorkflow) -> list[dict]:
    assert wf.event_store is not None
    return wf.event_store.fetch_events_by_type('review_completed')


async def test_first_pass_records_one_row_for_cycle_one(tmp_path: Path):
    wf = _make_workflow(tmp_path)
    wf._review = AsyncMock(return_value=_aggregation([], [review_issue('s1')]))

    outcome = await wf._execute_verify_review_loop()

    assert outcome == WorkflowOutcome.DONE
    [row] = _review_completed_rows(wf)
    assert row['task_id'] == wf.task_id
    assert row['phase'] == 'review'
    assert row['cost_usd'] is None
    assert row['duration_ms'] is None
    data = row['data']
    assert data['cycle'] == 1
    assert data['has_blocking_issues'] is False
    assert data['blocking_count'] == 0
    assert data['suggestion_count'] == 1
    assert data['reviewer_count'] == 1


async def test_block_then_pass_records_both_cycles_without_issue_text(tmp_path: Path):
    wf = _make_workflow(tmp_path)
    blocker, suggestion = review_issue('b1', 'blocking'), review_issue('s1')
    wf._review = AsyncMock(side_effect=[
        _aggregation([blocker], [suggestion]),
        _aggregation([], []),
    ])

    outcome = await wf._execute_verify_review_loop()

    assert outcome == WorkflowOutcome.DONE
    cast(AsyncMock, wf._replan).assert_awaited_once()
    rows = _review_completed_rows(wf)
    assert [r['data']['cycle'] for r in rows] == [1, 2]
    assert [r['data']['has_blocking_issues'] for r in rows] == [True, False]
    assert [r['data']['blocking_count'] for r in rows] == [1, 0]
    for row in rows:
        serialised = json.dumps(row['data'])
        for issue in (blocker, suggestion):
            for field in ('description', 'suggested_fix', 'location', 'reviewer'):
                assert issue[field] not in serialised


async def test_blocking_review_at_the_cap_is_recorded_before_the_exit(tmp_path: Path):
    wf = _make_workflow(tmp_path, max_review_cycles=1)
    wf._review = AsyncMock(return_value=_aggregation([review_issue('b1', 'blocking')], []))
    wf._exit_review_cycles_exhausted = AsyncMock(return_value=WorkflowOutcome.ESCALATED)

    outcome = await wf._execute_verify_review_loop()

    assert outcome == WorkflowOutcome.ESCALATED
    [row] = _review_completed_rows(wf)
    assert row['data']['cycle'] == 1
    assert row['data']['has_blocking_issues'] is True


async def test_cycle_continues_from_the_task_lifetime_blocking_total(tmp_path: Path):
    wf = _make_workflow(tmp_path)
    assert wf.artifacts is not None
    wf.artifacts.set_review_counters(review_cycles_total=2)
    wf._review = AsyncMock(return_value=_aggregation([], []))

    outcome = await wf._execute_verify_review_loop()

    assert outcome == WorkflowOutcome.DONE
    [row] = _review_completed_rows(wf)
    assert row['data']['cycle'] == 3


async def test_amendment_round_closes_one_cycle_with_the_re_review_counts(tmp_path: Path):
    wf = _make_workflow(tmp_path)
    wf._suggestions_in_scope = lambda suggestions: list(suggestions)
    wf._review = AsyncMock(side_effect=[
        _aggregation([], [review_issue('s1')]),
        _aggregation([], []),
    ])

    outcome = await wf._execute_verify_review_loop()

    assert outcome == WorkflowOutcome.DONE
    cast(AsyncMock, wf._amend).assert_awaited_once()
    [row] = _review_completed_rows(wf)
    assert row['data']['cycle'] == 1
    assert row['data']['suggestion_count'] == 0
