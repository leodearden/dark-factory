"""The review-cycles-exhausted exit of ``TaskWorkflow._execute_verify_review_loop``
(task 3415).  When a blocking review arrives with the review-cycle cap already
spent, the loop files a blocking ``review_issues`` escalation.  The suggestions
that came with that review must not be lost on the way out: the exit routes
them to the curator exactly as the non-blocking DONE exit does, and inlines
their content in the escalation together with where they were routed.  These
tests drive the real loop with only the execute/verify/review boundaries
stubbed.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _mcp_transport_harness import RecordingClientFactory, RecordingMcpServer
from _review_fixtures import review_aggregation, review_issue
from _workflow_helpers import _make
from escalation.queue import EscalationQueue

from orchestrator.review_suggestions.disposition import SuggestionDisposition
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

pytestmark = pytest.mark.asyncio

TASK_ID = '3415'

_BLOCKER = review_issue('blocker', 'blocking')


def _make_cap_workflow(
    tmp_path: Path,
    *,
    max_amendment_rounds: int = 1,
    escalation_queue: EscalationQueue | MagicMock | None = None,
) -> TaskWorkflow:
    """A workflow whose next blocking review reaches the review-cycle cap."""
    wf = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj', task_id=TASK_ID).wf
    wf.config.max_review_cycles = 1
    wf.config.max_amendment_rounds = max_amendment_rounds
    wf.config.suppress_resettled_review_suggestions = True
    wf.config.models.reviewer = 'sonnet'
    wf.escalation_queue = escalation_queue
    wf.mcp.url = 'http://localhost:8002'

    wf._execute_iterations = AsyncMock(return_value=WorkflowOutcome.DONE)  # type: ignore[method-assign]
    wf._verify_debugfix_loop = AsyncMock(return_value=WorkflowOutcome.DONE)  # type: ignore[method-assign]
    wf.git_ops.get_head_tree_hash = AsyncMock(return_value='TREE1')
    wf.git_ops.get_new_side_changed_line_ranges = AsyncMock(return_value={})
    wf.git_ops.has_uncommitted_work = AsyncMock(return_value=False)
    wf._replan = AsyncMock()  # type: ignore[method-assign]
    wf._suggestions_in_scope = lambda s: []  # type: ignore[method-assign]
    return wf


def _blocking(suggestions: list[dict]):
    return review_aggregation([_BLOCKER], suggestions)


def _non_blocking(suggestions: list[dict]):
    return review_aggregation([], suggestions)


def _spy_on_router(wf: TaskWorkflow) -> list[SuggestionDisposition]:
    dispositions: list[SuggestionDisposition] = []
    real_route = wf._route_review_suggestions_to_curator

    async def spy(reviews):
        disposition = await real_route(reviews)
        dispositions.append(disposition)
        return disposition

    wf._route_review_suggestions_to_curator = spy  # type: ignore[method-assign]
    return dispositions


async def test_routes_then_escalates_with_the_reported_disposition(tmp_path: Path):
    wf = _make_cap_workflow(tmp_path)
    reviews = _blocking([review_issue('s1')])
    wf._review = AsyncMock(return_value=reviews)  # type: ignore[method-assign]
    manager = MagicMock()
    route = AsyncMock(return_value=SuggestionDisposition.CURATOR)
    escalate = MagicMock()
    manager.attach_mock(route, 'route')
    manager.attach_mock(escalate, 'escalate')
    wf._route_review_suggestions_to_curator = route  # type: ignore[method-assign]
    wf._escalate_review_issues = escalate  # type: ignore[method-assign]

    outcome = await wf._execute_verify_review_loop()

    assert outcome == WorkflowOutcome.ESCALATED
    assert [c[0] for c in manager.mock_calls] == ['route', 'escalate']
    escalate.assert_called_once_with(
        reviews, suggestion_disposition=SuggestionDisposition.CURATOR, n_suggestions_raw=1,
    )
    cast(AsyncMock, wf._replan).assert_not_called()


async def test_no_suggestions_escalates_with_none(tmp_path: Path):
    wf = _make_cap_workflow(tmp_path)
    reviews = _blocking([])
    wf._review = AsyncMock(return_value=reviews)  # type: ignore[method-assign]
    escalate = MagicMock()
    wf._escalate_review_issues = escalate  # type: ignore[method-assign]

    outcome = await wf._execute_verify_review_loop()

    assert outcome == WorkflowOutcome.ESCALATED
    escalate.assert_called_once_with(
        reviews, suggestion_disposition=SuggestionDisposition.NONE, n_suggestions_raw=0,
    )


async def test_routing_failure_still_files_the_escalation(tmp_path: Path, caplog):
    wf = _make_cap_workflow(tmp_path)
    reviews = _blocking([review_issue('s1')])
    wf._review = AsyncMock(return_value=reviews)  # type: ignore[method-assign]
    wf._route_review_suggestions_to_curator = AsyncMock(  # type: ignore[method-assign]
        side_effect=TypeError('not JSON serializable'),
    )
    escalate = MagicMock()
    wf._escalate_review_issues = escalate  # type: ignore[method-assign]

    with caplog.at_level(logging.WARNING, logger='orchestrator.workflow'):
        outcome = await wf._execute_verify_review_loop()

    assert outcome == WorkflowOutcome.ESCALATED
    escalate.assert_called_once()
    assert escalate.call_args.kwargs == {
        'suggestion_disposition': SuggestionDisposition.ERROR, 'n_suggestions_raw': 1,
    }
    assert any(
        r.levelno == logging.WARNING
        and r.name == 'orchestrator.workflow'
        and isinstance(r.args, tuple)
        and wf.task_id in r.args
        for r in caplog.records
    )


async def test_esc_4223_1_shape_one_blocking_eight_suggestions(tmp_path: Path):
    """The shape of esc-4223-1: the stored escalation counted 8 suggestions and
    carried none of them, and none reached the curator."""
    queue = EscalationQueue(tmp_path / 'esc')
    wf = _make_cap_workflow(tmp_path, escalation_queue=queue)
    suggestions = [review_issue(f's{i}') for i in range(1, 9)]
    wf._review = AsyncMock(return_value=_blocking(suggestions))  # type: ignore[method-assign]
    server = RecordingMcpServer()

    with patch('httpx.AsyncClient', RecordingClientFactory(server)):
        outcome = await wf._execute_verify_review_loop()
        await asyncio.gather(*wf._background_tasks, return_exceptions=True)

    assert outcome == WorkflowOutcome.ESCALATED
    [esc] = [e for e in queue.get_by_task(TASK_ID) if e.category == 'review_issues']
    assert '1 blocking issue(s) and 8 suggestion(s)' in esc.summary
    assert '[suggestions → curator]' in esc.summary
    assert 'scoped out' not in esc.summary
    assert esc.detail.startswith('# Review Feedback — Blocking Issues')
    assert _BLOCKER['description'] in esc.detail
    for suggestion in suggestions:
        for key in ('description', 'location', 'suggested_fix'):
            assert suggestion[key] in esc.detail

    submits = server.tool_calls('submit_task')
    assert len(submits) == 8
    metadata = [body['params']['arguments']['metadata'] for body in submits]
    assert {m['escalation_id'] for m in metadata} == {f'review-suggestions-{TASK_ID}'}
    assert len({m['suggestion_hash'] for m in metadata}) == 8


async def test_same_instance_identical_reentry_is_deduped(tmp_path: Path):
    wf = _make_cap_workflow(tmp_path)
    wf._review = AsyncMock(  # type: ignore[method-assign]
        side_effect=[_blocking([review_issue('s1')]), _blocking([review_issue('s1')])],
    )
    dispositions = _spy_on_router(wf)

    with patch.object(wf, '_post_submit_tasks', AsyncMock()) as post:
        first = await wf._execute_verify_review_loop()
        await asyncio.gather(*wf._background_tasks, return_exceptions=True)
        second = await wf._execute_verify_review_loop()
        await asyncio.gather(*wf._background_tasks, return_exceptions=True)

    assert first == second == WorkflowOutcome.ESCALATED
    assert dispositions == [SuggestionDisposition.CURATOR, SuggestionDisposition.DEDUPED]
    assert post.await_count == 1


async def test_blocking_then_done_reentry_reuses_item_keys(tmp_path: Path):
    """After an unblock the re-review passes and re-raises b beside a new c.

    The scalar batch cache cannot absorb [b, c] after [a, b]; the curator R4
    gate can, item by item, because b keeps its (escalation_id,
    suggestion_hash) pair.  A REWORDED b would get a new key and is left to the
    curator's semantic dedup, so this test does not claim to cover it.
    """
    a, b, c = review_issue('s1'), review_issue('s2'), review_issue('s3')
    wf = _make_cap_workflow(tmp_path)
    wf._review = AsyncMock(  # type: ignore[method-assign]
        side_effect=[_blocking([a, b]), _non_blocking([dict(b), c])],
    )
    posted_bodies: list[dict] = []

    async def capture_post(url, *, json: dict, **kwargs):
        posted_bodies.append(json)
        return MagicMock(status_code=200, json=lambda: {'result': {'ticket': 'tkt-1'}})

    with patch('httpx.AsyncClient.post', side_effect=capture_post):
        first = await wf._execute_verify_review_loop()
        await asyncio.gather(*wf._background_tasks, return_exceptions=True)
        run_1_count = len(posted_bodies)
        second = await wf._execute_verify_review_loop()
        await asyncio.gather(*wf._background_tasks, return_exceptions=True)

    def r4_key(body: dict) -> tuple[str, str]:
        meta = body['params']['arguments']['metadata']
        return (meta['escalation_id'], meta['suggestion_hash'])

    assert (first, second) == (WorkflowOutcome.ESCALATED, WorkflowOutcome.DONE)
    assert run_1_count == 2
    run_1_a, run_1_b = (r4_key(body) for body in posted_bodies[:2])
    run_2_b, run_2_c = (r4_key(body) for body in posted_bodies[2:])
    assert run_2_b == run_1_b
    assert run_2_c not in {run_1_a, run_1_b}


async def test_post_amendment_cap_exit_applies_amendment_delta_scope(tmp_path: Path):
    """After an amendment round the cap exit scopes to the amendment delta,
    as the DONE exit does: out-of-delta suggestions go to the curator on
    their own, and the escalation inlines (and counts) only the in-delta ones.
    """
    queue = MagicMock()
    queue.make_id.return_value = f'esc-{TASK_ID}-1'
    wf = _make_cap_workflow(tmp_path, max_amendment_rounds=1, escalation_queue=queue)
    s_in_scope = review_issue('s1')
    s_in_delta = {**review_issue('s2'), 'location': 'src/a.py:10'}
    s_out_of_delta = {**review_issue('s3'), 'location': 'src/z.py:900'}
    wf._suggestions_in_scope = lambda s: list(s)  # type: ignore[method-assign]
    wf._amend = AsyncMock(return_value=True)  # type: ignore[method-assign]
    wf._commit_amendment_wip = AsyncMock()  # type: ignore[method-assign]
    wf._get_head_commit = AsyncMock(return_value='pre-amend')  # type: ignore[method-assign]
    wf._review = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            _non_blocking([s_in_scope]),
            _blocking([s_in_delta, s_out_of_delta]),
        ],
    )
    wf.git_ops.get_new_side_changed_line_ranges = AsyncMock(
        return_value={'src/a.py': [(1, 50)]},
    )
    route = AsyncMock(return_value=SuggestionDisposition.CURATOR)
    wf._route_review_suggestions_to_curator = route  # type: ignore[method-assign]

    outcome = await wf._execute_verify_review_loop()

    assert outcome == WorkflowOutcome.ESCALATED
    assert [c.args[0].suggestions for c in route.await_args_list] == [
        [s_out_of_delta],
        [s_in_delta],
    ]
    esc = queue.submit.call_args[0][0]
    assert s_in_delta['description'] in esc.detail
    assert s_out_of_delta['description'] not in esc.detail
    assert '1 suggestion(s) (+1 scoped out' in esc.summary
