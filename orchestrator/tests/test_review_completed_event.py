"""``TaskWorkflow._execute_verify_review_loop`` records one ``review_completed`` row per closed
review cycle (client-feed θ). The loop runs for real against a real EventStore, through
``_review_loop_harness.review_loop_workflow``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock

import pytest
from _review_fixtures import review_issue
from _review_loop_harness import review_loop_workflow

from orchestrator.artifacts import ReviewAggregation
from orchestrator.event_store import EventStore
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

pytestmark = pytest.mark.asyncio

TASK_ID = '6967'


def _recording_workflow(tmp_path: Path, *, max_review_cycles: int = 2) -> TaskWorkflow:
    return review_loop_workflow(
        tmp_path,
        task_id=TASK_ID,
        event_store=EventStore(tmp_path / 'runs.db', run_id='run-6967'),
        max_review_cycles=max_review_cycles,
    )


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
    wf = _recording_workflow(tmp_path)
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
    wf = _recording_workflow(tmp_path)
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
    wf = _recording_workflow(tmp_path, max_review_cycles=1)
    wf._review = AsyncMock(return_value=_aggregation([review_issue('b1', 'blocking')], []))
    wf._exit_review_cycles_exhausted = AsyncMock(return_value=WorkflowOutcome.ESCALATED)

    outcome = await wf._execute_verify_review_loop()

    assert outcome == WorkflowOutcome.ESCALATED
    [row] = _review_completed_rows(wf)
    assert row['data']['cycle'] == 1
    assert row['data']['has_blocking_issues'] is True


async def test_cycle_continues_from_the_task_lifetime_blocking_total(tmp_path: Path):
    wf = _recording_workflow(tmp_path)
    assert wf.artifacts is not None
    wf.artifacts.set_review_counters(review_cycles_total=2)
    wf._review = AsyncMock(return_value=_aggregation([], []))

    outcome = await wf._execute_verify_review_loop()

    assert outcome == WorkflowOutcome.DONE
    [row] = _review_completed_rows(wf)
    assert row['data']['cycle'] == 3


async def test_amendment_round_closes_one_cycle_with_the_re_review_counts(tmp_path: Path):
    wf = _recording_workflow(tmp_path)
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
