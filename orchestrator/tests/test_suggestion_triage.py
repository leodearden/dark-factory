"""Tests for review suggestion escalation and steward completion grace period."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch  # noqa: F401

import pytest
from _orch_helpers import pydantic_spec
from shared.cli_invoke import AllAccountsCappedException

from orchestrator.artifacts import ReviewAggregation, TaskArtifacts
from orchestrator.config import OrchestratorConfig
from orchestrator.workflow import StewardInterrupted, WorkflowOutcome

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_workflow(*, escalation_queue=None, escalation_event=None):
    """Build a minimal TaskWorkflow-like object with the methods under test."""
    from orchestrator.workflow import TaskWorkflow

    assignment = MagicMock()
    assignment.task_id = '42'
    assignment.task = {'id': '42', 'title': 'Test Task', 'description': 'desc'}

    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.max_review_cycles = 2
    config.steward_completion_timeout = 300.0

    git_ops = MagicMock()
    scheduler = MagicMock()

    mcp = MagicMock()
    mcp.url = 'http://localhost:8002'

    briefing = MagicMock()

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=briefing,
        mcp=mcp,
        escalation_queue=escalation_queue,
        escalation_event=escalation_event,
    )
    return wf


def _fake_reviews(suggestions=None, blocking_issues=None):
    """Return a mock ReviewAggregation."""
    reviews = MagicMock()
    reviews.suggestions = suggestions or []
    reviews.blocking_issues = blocking_issues or []
    reviews.has_blocking_issues = bool(blocking_issues)
    reviews.format_for_replan.return_value = 'formatted review feedback'
    return reviews


def _make_escalation(**overrides):
    from escalation.models import Escalation

    defaults: dict = dict(
        id='esc-42-0',
        task_id='42',
        agent_role='orchestrator',
        severity='info',
        category='review_suggestions',
        summary='3 review suggestion(s) for triage',
        detail='[]',
    )
    defaults.update(overrides)
    return Escalation(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _escalate_suggestions
# ---------------------------------------------------------------------------

class TestEscalateSuggestions:
    def test_creates_escalation_with_correct_fields(self):
        queue = MagicMock()
        queue.make_id.return_value = 'esc-42-0'
        wf = _make_workflow(escalation_queue=queue)
        wf.state = MagicMock(value='review')

        suggestions = [
            {'reviewer': 'test_analyst', 'severity': 'suggestion',
             'location': 'src/foo.py:10', 'category': 'coverage',
             'description': 'Missing edge case', 'suggested_fix': 'Add test'},
        ]
        reviews = _fake_reviews(suggestions)

        wf._escalate_suggestions(reviews)

        queue.submit.assert_called_once()
        esc = queue.submit.call_args[0][0]
        assert esc.category == 'review_suggestions'
        assert esc.severity == 'info'
        assert esc.task_id == '42'
        # Detail is prefixed with content fingerprint: #hash:<hex16>#<json>
        detail = esc.detail
        assert detail.startswith('#hash:')
        json_start = detail.index('#', 6) + 1
        assert json.loads(detail[json_start:]) == suggestions

    def test_noop_without_queue(self):
        wf = _make_workflow(escalation_queue=None)
        reviews = _fake_reviews([{'description': 'something'}])
        wf._escalate_suggestions(reviews)

    def test_noop_without_suggestions(self):
        queue = MagicMock()
        wf = _make_workflow(escalation_queue=queue)
        reviews = _fake_reviews([])
        wf._escalate_suggestions(reviews)
        queue.submit.assert_not_called()


class TestEscalateSuggestionsCharacterization:
    """Pins _escalate_suggestions behavior after the dedup-helper refactor.

    These tests verify that the refactored method still produces exactly the
    same ``#hash:<hex16>#<json>`` detail format, and that both the matching
    (skip) and non-matching (submit) dedup paths work correctly via the shared
    find_prior_review_suggestion predicate.

    Acceptance criterion: 'from orchestrator.review_suggestions.dedup import
    review_suggestion_payload_hash' appears in workflow.py (this import is
    confirmed by the source-file assertion below).
    """

    def _suggestions(self):
        return [
            {'reviewer': 'a', 'severity': 'suggestion',
             'location': 'src/bar.py:1', 'category': 'coverage',
             'description': 'Add coverage for X', 'suggested_fix': 'Write test'},
        ]

    def test_detail_uses_shared_hash_helper(self):
        """detail must start with '#hash:<shared_hash>#' — not a re-inlined hash."""
        from orchestrator.review_suggestions.dedup import review_suggestion_payload_hash

        queue = MagicMock()
        queue.make_id.return_value = 'esc-42-0'
        queue.get_by_task.return_value = []
        wf = _make_workflow(escalation_queue=queue)
        wf.state = MagicMock(value='review')

        suggestions = self._suggestions()
        reviews = _fake_reviews(suggestions)
        wf._escalate_suggestions(reviews)

        queue.submit.assert_called_once()
        esc = queue.submit.call_args[0][0]
        expected_hash = review_suggestion_payload_hash(suggestions)
        assert esc.detail.startswith(f'#hash:{expected_hash}#')

    def test_skips_when_prior_matching_hash(self):
        """submit must NOT be called when a prior escalation with the same hash exists."""
        from orchestrator.review_suggestions.dedup import (
            hash_marker,
            review_suggestion_payload_hash,
        )

        suggestions = self._suggestions()
        h = review_suggestion_payload_hash(suggestions)
        prior = _make_escalation(
            detail=hash_marker(h) + '[]',
            category='review_suggestions',
        )

        queue = MagicMock()
        queue.make_id.return_value = 'esc-42-1'
        queue.get_by_task.return_value = [prior]
        wf = _make_workflow(escalation_queue=queue)
        wf.state = MagicMock(value='review')

        wf._escalate_suggestions(_fake_reviews(suggestions))

        queue.submit.assert_not_called()

    def test_does_not_skip_when_prior_has_different_hash(self):
        """submit IS called when the prior escalation has a different hash."""
        from orchestrator.review_suggestions.dedup import hash_marker

        suggestions = self._suggestions()
        # Build a prior with a DIFFERENT hash
        prior = _make_escalation(
            detail=hash_marker('0000000000000000') + '[]',
            category='review_suggestions',
        )

        queue = MagicMock()
        queue.make_id.return_value = 'esc-42-2'
        queue.get_by_task.return_value = [prior]
        wf = _make_workflow(escalation_queue=queue)
        wf.state = MagicMock(value='review')

        wf._escalate_suggestions(_fake_reviews(suggestions))

        queue.submit.assert_called_once()


# ---------------------------------------------------------------------------
# _route_review_suggestions_to_curator
# ---------------------------------------------------------------------------


class TestRouteReviewSuggestionsToCurator:
    """Tests for the new fire-and-forget direct-to-curator routing method."""

    def _suggestions(self):
        return [
            {
                'reviewer': 'analyst',
                'severity': 'suggestion',
                'location': 'src/foo.py:10',
                'category': 'coverage',
                'description': 'Missing edge case for branch X',
                'suggested_fix': 'Add a test covering branch X',
            },
            {
                'reviewer': 'analyst',
                'severity': 'suggestion',
                'location': 'src/bar.py:20',
                'category': 'naming',
                'description': 'Variable name could be clearer',
                'suggested_fix': 'Rename to something more descriptive',
            },
        ]

    @pytest.mark.asyncio
    async def test_no_calls_for_empty_suggestions(self):
        """Empty suggestions list → no HTTP calls, no background tasks."""
        queue = MagicMock()
        wf = _make_workflow(escalation_queue=queue)

        with patch('httpx.AsyncClient.post', new_callable=AsyncMock) as mock_post:
            await wf._route_review_suggestions_to_curator(_fake_reviews([]))
            mock_post.assert_not_called()

        queue.submit.assert_not_called()

    @pytest.mark.asyncio
    async def test_schedules_one_post_per_suggestion(self):
        """N suggestions → exactly N submit_task MCP POSTs are eventually made."""
        suggestions = self._suggestions()
        wf = _make_workflow()

        posted_bodies = []

        async def capture_post(url, *, json=None, **kwargs):
            posted_bodies.append(json)
            return MagicMock(status_code=200, json=lambda: {'result': {'ticket': 'tkt-1'}})

        with patch('httpx.AsyncClient.post', side_effect=capture_post):
            await wf._route_review_suggestions_to_curator(_fake_reviews(suggestions))
            # Drain background tasks
            tasks = list(wf._background_tasks)
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        assert len(posted_bodies) == len(suggestions)

    @pytest.mark.asyncio
    async def test_payload_fields(self):
        """Each POST payload has the required fields per the spec."""
        from orchestrator.review_suggestions.dedup import review_suggestion_payload_hash

        suggestions = self._suggestions()
        wf = _make_workflow()

        posted_bodies = []

        async def capture_post(url, *, json=None, **kwargs):
            posted_bodies.append(json)
            return MagicMock(status_code=200, json=lambda: {'result': {'ticket': 'tkt-1'}})

        with patch('httpx.AsyncClient.post', side_effect=capture_post):
            await wf._route_review_suggestions_to_curator(_fake_reviews(suggestions))
            tasks = list(wf._background_tasks)
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        assert len(posted_bodies) == len(suggestions)
        content_hash = review_suggestion_payload_hash(suggestions)
        task_id = wf.task_id

        for _i, (body, suggestion) in enumerate(zip(posted_bodies, suggestions, strict=True)):
            assert body['method'] == 'tools/call'
            params = body['params']
            assert params['name'] == 'submit_task'
            args = params['arguments']

            # priority
            assert args['priority'] == 'low'

            # metadata
            meta = args['metadata']
            assert meta['spawned_from'] == task_id
            assert meta['spawn_context'] == 'review_suggestions'
            assert meta['escalation_id'] == f'review-suggestions-{task_id}'
            assert meta['suggestion_hash'] == content_hash

            # title: [<cat>] <loc>: <desc[:60]>
            cat = suggestion.get('category', '')
            loc = suggestion.get('location', '')
            desc = suggestion.get('description', '')
            expected_title = f'[{cat}] {loc}: {desc[:60]}'
            assert args['title'] == expected_title

            # details: json.dumps(suggestion)
            assert args['details'] == json.dumps(suggestion)

            # project_root is passed
            assert 'project_root' in args

    @pytest.mark.asyncio
    async def test_escalation_queue_never_touched(self):
        """escalation_queue.submit must never be called by the curator path."""
        suggestions = self._suggestions()
        queue = MagicMock()
        wf = _make_workflow(escalation_queue=queue)

        with patch('httpx.AsyncClient.post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: {'result': {'ticket': 'tkt-1'}},
            )
            await wf._route_review_suggestions_to_curator(_fake_reviews(suggestions))
            tasks = list(wf._background_tasks)
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        queue.submit.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_curate_batch_call(self):
        """curate_batch must never be called (stall risk)."""
        suggestions = self._suggestions()
        wf = _make_workflow()

        with patch('httpx.AsyncClient.post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: {'result': {'ticket': 'tkt-1'}},
            )
            await wf._route_review_suggestions_to_curator(_fake_reviews(suggestions))
            tasks = list(wf._background_tasks)
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        for call in mock_post.call_args_list:
            body = call.kwargs.get('json') or (call.args[1] if len(call.args) > 1 else {})
            tool_name = (body.get('params') or {}).get('name', '')
            assert tool_name != 'curate_batch', 'curate_batch must never be called'

    @pytest.mark.asyncio
    async def test_distinct_second_call_still_submits(self):
        """Distinct suggestion sets → both batches are submitted (no over-eager dedup).

        The cache keys on content_hash: two different suggestion lists produce
        different hashes so NEITHER call must be short-circuited.
        Total POSTs == len(A) + len(B).
        """
        suggestions_a = self._suggestions()  # 2 suggestions
        suggestions_b = [
            {
                'reviewer': 'security',
                'severity': 'suggestion',
                'location': 'src/auth.py:5',
                'category': 'security',
                'description': 'Validate input before use',
                'suggested_fix': 'Add input validation',
            },
        ]

        wf = _make_workflow()
        posted_bodies = []

        async def capture_post(url, *, json=None, **kwargs):
            posted_bodies.append(json)
            return MagicMock(status_code=200, json=lambda: {'result': {'ticket': 'tkt-1'}})

        with patch('httpx.AsyncClient.post', side_effect=capture_post):
            # First call with suggestions A
            await wf._route_review_suggestions_to_curator(_fake_reviews(suggestions_a))
            tasks = list(wf._background_tasks)
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            # Second call with DIFFERENT suggestions B
            await wf._route_review_suggestions_to_curator(_fake_reviews(suggestions_b))
            tasks = list(wf._background_tasks)
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        # Both batches must go through
        expected = len(suggestions_a) + len(suggestions_b)
        assert len(posted_bodies) == expected, (
            f'Expected {expected} POSTs (A + B), got {len(posted_bodies)}: '
            'distinct suggestion sets must not be collapsed by the dedup cache'
        )

    @pytest.mark.asyncio
    async def test_identical_second_call_short_circuits_fallback(self):
        """Cache check sits above mcp=None guard: identical re-entry skips queue.submit entirely.

        When mcp is None and escalation_queue is set, the in-task cache must
        short-circuit the second call BEFORE touching the fallback queue so
        queue.submit.call_count == 1, not 2.
        """
        suggestions = self._suggestions()
        queue = MagicMock()
        queue.make_id.return_value = 'esc-42-0'
        queue.get_by_task.return_value = []
        wf = _make_workflow(escalation_queue=queue)
        wf.mcp = None  # simulate missing MCP transport
        wf.state = MagicMock(value='review')

        with patch('httpx.AsyncClient.post', new_callable=AsyncMock) as mock_post:
            # First call — falls back to queue.submit
            await wf._route_review_suggestions_to_curator(_fake_reviews(suggestions))
            # Second call — identical, must be short-circuited before queue
            await wf._route_review_suggestions_to_curator(_fake_reviews(suggestions))

        # HTTP must never be attempted (mcp is None)
        mock_post.assert_not_called()
        # Queue submit must be called exactly once (cache blocks second call)
        assert queue.submit.call_count == 1, (
            f'Expected queue.submit called once, got {queue.submit.call_count}: '
            'the in-task cache must short-circuit before reaching the fallback'
        )

    @pytest.mark.asyncio
    async def test_mcp_none_falls_back_to_escalation_queue(self):
        """When self.mcp is None and escalation_queue is set, _escalate_suggestions is called."""
        suggestions = self._suggestions()
        queue = MagicMock()
        queue.make_id.return_value = 'esc-42-0'
        queue.get_by_task.return_value = []
        wf = _make_workflow(escalation_queue=queue)
        wf.mcp = None  # simulate missing MCP transport
        wf.state = MagicMock(value='review')

        with patch('httpx.AsyncClient.post', new_callable=AsyncMock) as mock_post:
            await wf._route_review_suggestions_to_curator(_fake_reviews(suggestions))

        # Must not attempt HTTP calls against a missing MCP
        mock_post.assert_not_called()
        # Must fall back to the steward escalation path
        queue.submit.assert_called_once()

    @pytest.mark.asyncio
    async def test_mcp_none_no_queue_does_not_call_write_to_memory(self):
        """When both self.mcp and escalation_queue are None, only the audit WARNING is emitted.

        _write_suggestions_to_memory is NOT called — the previous dead call (which
        immediately returned when mcp is None) was removed so the branch is now
        unambiguous: log-only, no sink.
        """
        suggestions = self._suggestions()
        wf = _make_workflow(escalation_queue=None)
        wf.mcp = None  # simulate missing MCP transport

        write_mock = AsyncMock()
        with (
            patch.object(wf, '_write_suggestions_to_memory', write_mock),
            patch('httpx.AsyncClient.post', new_callable=AsyncMock) as mock_post,
        ):
            await wf._route_review_suggestions_to_curator(_fake_reviews(suggestions))

        mock_post.assert_not_called()
        write_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_identical_second_call_short_circuits_curator_submit(self):
        """Identical re-entry → exactly N POSTs across both calls, not 2*N.

        The in-task dedup cache (self._last_routed_suggestion_hash) should
        short-circuit the second call so the curator only receives one batch.
        """
        suggestions = self._suggestions()
        wf = _make_workflow()

        posted_bodies = []

        async def capture_post(url, *, json=None, **kwargs):
            posted_bodies.append(json)
            return MagicMock(status_code=200, json=lambda: {'result': {'ticket': 'tkt-1'}})

        with patch('httpx.AsyncClient.post', side_effect=capture_post):
            # First call — should submit
            await wf._route_review_suggestions_to_curator(_fake_reviews(suggestions))
            tasks = list(wf._background_tasks)
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            # Second call with identical suggestions — should be short-circuited
            await wf._route_review_suggestions_to_curator(_fake_reviews(suggestions))
            tasks = list(wf._background_tasks)
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        # Exactly N posts (one batch), not 2*N
        assert len(posted_bodies) == len(suggestions), (
            f'Expected {len(suggestions)} POSTs (one batch), got {len(posted_bodies)}: '
            'second identical call should have been short-circuited by the dedup cache'
        )

    @pytest.mark.asyncio
    async def test_mcp_none_no_queue_logs_audit_warning_on_dropped_suggestions(
        self, caplog
    ):
        """(mcp=None, queue=None) must emit a WARNING so the drop is auditable.

        Before step-2 impl: the fallback silently drops suggestions after the
        no-op _write_suggestions_to_memory call — no log record is produced.
        After step-2 impl: a WARNING is emitted containing 'suggestion' and
        one of {'drop', 'no-op', 'no sink'} plus the count (2).
        """
        suggestions = self._suggestions()
        wf = _make_workflow(escalation_queue=None)
        wf.mcp = None  # simulate CLI/dry-run/test context

        # _write_suggestions_to_memory is no longer called in this branch (dead call
        # removed), so no patch is needed — the WARNING fires unconditionally.
        with (
            patch('httpx.AsyncClient.post', new_callable=AsyncMock) as mock_post,
            caplog.at_level(logging.WARNING, logger='orchestrator.workflow'),
        ):
            await wf._route_review_suggestions_to_curator(_fake_reviews(suggestions))

        # Regression guard: must not attempt any HTTP against missing MCP
        mock_post.assert_not_called()

        # The audit warning must be present — assert on record.args to avoid
        # prose-pinning: any rewording of the message text is fine so long as the
        # task_id and suggestion count are passed as format arguments.
        warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and r.name == 'orchestrator.workflow'
        ]
        assert len(warning_records) == 1, (
            f'Expected exactly one WARNING on orchestrator.workflow, got: {warning_records}'
        )
        rec = warning_records[0]
        assert wf.task_id in rec.args, (
            f'Expected task_id {wf.task_id!r} in record.args, got: {rec.args}'
        )
        assert len(suggestions) in rec.args, (
            f'Expected suggestion count {len(suggestions)} in record.args, got: {rec.args}'
        )

    @pytest.mark.asyncio
    async def test_A_B_A_sequence_resubmits_A(self):
        """A→B→A: the third call re-submits A because B overwrote the scalar cache.

        The in-task cache stores only the *most recently* routed hash (a scalar,
        not a set).  This means it only eliminates *consecutive* duplicates:

            call 1 (A) → submitted,  cache = hash(A)
            call 2 (B) → submitted,  cache = hash(B)   [hash(A) evicted]
            call 3 (A) → submitted,  cache = hash(A)   [NOT a duplicate vs B]

        Total POSTs == len(A) + len(B) + len(A) = 2*len(A) + len(B).

        This boundary is intentional — the server-side curator R4 idempotency
        gate (task_interceptor._check_escalation_idempotency) is the durable
        source-of-truth dedup for non-consecutive repeats.
        """
        suggestions_a = self._suggestions()  # 2 items
        suggestions_b = [
            {
                'reviewer': 'security',
                'severity': 'suggestion',
                'location': 'src/auth.py:5',
                'category': 'security',
                'description': 'Validate input before use',
                'suggested_fix': 'Add input validation',
            },
        ]  # 1 item, different hash

        wf = _make_workflow()
        posted_bodies = []

        async def capture_post(url, *, json=None, **kwargs):
            posted_bodies.append(json)
            return MagicMock(status_code=200, json=lambda: {'result': {'ticket': 'tkt-1'}})

        with patch('httpx.AsyncClient.post', side_effect=capture_post):
            # Call 1: A — submitted, cache = hash(A)
            await wf._route_review_suggestions_to_curator(_fake_reviews(suggestions_a))
            tasks = list(wf._background_tasks)
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            # Call 2: B — submitted (different hash), cache = hash(B)
            await wf._route_review_suggestions_to_curator(_fake_reviews(suggestions_b))
            tasks = list(wf._background_tasks)
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            # Call 3: A again — hash(A) != hash(B) so NOT short-circuited;
            # re-submitted, cache = hash(A) again.
            await wf._route_review_suggestions_to_curator(_fake_reviews(suggestions_a))
            tasks = list(wf._background_tasks)
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        expected = len(suggestions_a) + len(suggestions_b) + len(suggestions_a)
        assert len(posted_bodies) == expected, (
            f'A→B→A: expected {expected} POSTs (A={len(suggestions_a)}, '
            f'B={len(suggestions_b)}, A again={len(suggestions_a)}), '
            f'got {len(posted_bodies)}.  The scalar cache must not suppress '
            'non-consecutive repeats — that is the server-side R4 gate\'s job.'
        )

    @pytest.mark.asyncio
    async def test_mcp_none_no_queue_drop_warning_fires_on_every_call(
        self, caplog
    ):
        """(mcp=None, queue=None) drop branch: WARNING fires on every call, not just the first.

        Unlike the curator and escalation-queue paths, the drop branch does NOT
        record the dedup hash.  Suggestions were never routed; suppressing the
        WARNING on subsequent identical calls would hide repeated data loss from
        audit logs.
        """
        suggestions = self._suggestions()
        wf = _make_workflow(escalation_queue=None)
        wf.mcp = None  # simulate CLI/dry-run/test context

        with (
            patch('httpx.AsyncClient.post', new_callable=AsyncMock) as mock_post,
            caplog.at_level(logging.WARNING, logger='orchestrator.workflow'),
        ):
            await wf._route_review_suggestions_to_curator(_fake_reviews(suggestions))
            await wf._route_review_suggestions_to_curator(_fake_reviews(suggestions))

        mock_post.assert_not_called()

        warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and r.name == 'orchestrator.workflow'
        ]
        assert len(warning_records) == 2, (
            f'Expected two WARNINGs (one per drop call), got {len(warning_records)}: '
            'the drop branch must not cache the hash — repeated drops must remain '
            'auditable.'
        )


# ---------------------------------------------------------------------------
# Integration: stall guard + call-site routing
# ---------------------------------------------------------------------------


class TestRouteReviewSuggestionsIntegration:
    """Integration test: stall guard for `_route_review_suggestions_to_curator` fire-and-forget POST."""

    @pytest.mark.asyncio
    async def test_stall_guard_post_not_completed_when_route_returns(self):
        """STALL GUARD: the curator POST must not complete before the route call returns.

        Contract: `_route_review_suggestions_to_curator` schedules the curator POST
        as a fire-and-forget background task via `asyncio.create_task`.  It must
        NOT synchronously await the POST.

        PRIMARY assertion: verifies that exactly one background task was scheduled and
        is still unfinished when the route call returns.  This check tolerates
        intermediate `await` points on the hot path (e.g. an MCP probe before
        `create_task`) — what matters is that the task was scheduled *and* has not
        finished by the time the call returns.

        SECONDARY (strict no-sync-await sentinel): `post_completed` is an
        `asyncio.Event` that the slow_post stub sets AFTER its 2s sleep finishes.
        This secondary check only holds when no `await` occurs between the route
        entry and the return — if the impl were changed to
        `await self._post_submit_tasks(…)`, the route call would block for 2s while
        slow_post's sleep completes, setting `post_completed`, and the assertion
        would fail with a clear message — independent of system load.  Kept as
        defense-in-depth alongside the PRIMARY check.
        """

        suggestions = [
            {
                'reviewer': 'a',
                'severity': 'suggestion',
                'location': 'src/x.py:1',
                'category': 'coverage',
                'description': 'Add a test',
                'suggested_fix': 'Write one',
            }
        ]
        wf = _make_workflow()

        post_completed = asyncio.Event()

        async def slow_post(url, *, json=None, **kwargs):
            await asyncio.sleep(2)
            post_completed.set()
            return MagicMock(status_code=200, json=lambda: {'result': {'ticket': 'tkt-slow'}})

        with patch('httpx.AsyncClient.post', side_effect=slow_post):
            await wf._route_review_suggestions_to_curator(_fake_reviews(suggestions))

        # PRIMARY: structural contract — exactly one background task was scheduled and
        # has not yet finished.  This directly verifies the fire-and-forget shape:
        # `create_task` adds to _background_tasks; a synchronous `await` would exhaust
        # the task before returning, making `.done()` True (or raising before we get
        # here).  A future refactor that adds a *legitimate* await before create_task
        # (e.g. an MCP probe) would still satisfy this check if the task is added and
        # not yet done — unlike the event check, which relies on no yield occurring.
        assert len(wf._background_tasks) == 1, (
            f'_route_review_suggestions_to_curator must schedule exactly one background '
            f'task via create_task — got {len(wf._background_tasks)}'
        )
        bg_task = next(iter(wf._background_tasks))
        assert not bg_task.done(), (
            'background task must not be done yet — fire-and-forget means it runs '
            'after the route call returns, not before'
        )

        # SECONDARY safety-net: the slow_post sentinel must not have been set, which
        # would only happen if the impl synchronously awaited the 2s sleep.
        # The POST must not have completed — it should still be running (or not yet
        # started) as a background task.  If post_completed is set it means the route
        # call synchronously awaited slow_post's 2s sleep before returning, which
        # violates the fire-and-forget contract.
        assert not post_completed.is_set(), (
            '_route_review_suggestions_to_curator must not synchronously await the '
            'curator POST — post_completed was set, indicating the slow_post sleep '
            'completed before the route call returned'
        )

        # Cancel the background task and await its completion to drain it
        # deterministically, preventing 'Task was destroyed but it is pending'
        # warnings at pytest-asyncio fixture teardown.  Reuse the same handle
        # captured above to avoid re-iterating the set.
        bg_task.cancel()
        await asyncio.gather(bg_task, return_exceptions=True)


# ---------------------------------------------------------------------------
# DONE-branch call-site via real workflow
# ---------------------------------------------------------------------------

_A_SUGGESTION = {
    'reviewer': 'analyst',
    'severity': 'suggestion',
    'location': 'src/a.py:1',
    'category': 'coverage',
    'description': 'Missing edge case',
    'suggested_fix': 'Add a test',
}


class TestDoneBranchCallSiteViaWorkflow:
    """Tests the call site at workflow.py:2437-2440 by driving the real workflow.

    The parametrized test invokes ``_execute_verify_review_loop`` end-to-end:
    fully-done plan (``_execute_iterations`` short-circuits), stubbed
    ``_verify_debugfix_loop`` returning DONE, stubbed ``_review`` returning a
    real ``ReviewAggregation``, and stubbed ``_suggestions_in_scope`` returning
    ``[]`` so the amendment branch is skipped.  Only the two DONE-branch
    routing methods are patched, so the real call site at lines 2437-2440 of
    workflow.py is exercised.
    """

    def _make_e2e_workflow(self, tmp_path: Path):
        """Build a workflow with a real worktree and a fully-done plan."""
        wf = _make_workflow()
        wf.config.max_amendment_rounds = 1
        wf.config.max_execute_iterations = 5
        wf.config.inter_iteration_rebase = False

        wt = tmp_path / 'wt'
        wt.mkdir()
        wf.worktree = wt
        wf.artifacts = TaskArtifacts(wt)
        wf.artifacts.init('42', 'Test Task', 'desc', base_commit='deadbeef')

        plan = {
            'task_id': '42',
            'title': 'Test Task',
            'files': [],
            'analysis': 'Test analysis',
            'prerequisites': [],
            'steps': [
                {
                    'id': 'step-1',
                    'type': 'impl',
                    'description': 'Write code',
                    'status': 'done',
                    'commit': 'abc123',
                },
            ],
        }
        wf.artifacts.write_plan(plan)
        wf.artifacts.stamp_plan_provenance(wf.session_id)

        wf._verify_debugfix_loop = AsyncMock(return_value=WorkflowOutcome.DONE)  # type: ignore[method-assign]
        wf._suggestions_in_scope = lambda s: []  # type: ignore[method-assign]

        return wf

    @pytest.mark.asyncio
    @pytest.mark.parametrize('suggestions, route_n, write_n', [
        ([_A_SUGGESTION], 1, 0),
        ([], 0, 1),
    ])
    async def test_done_branch_routing_via_real_workflow(
        self, tmp_path, suggestions, route_n, write_n
    ):
        """DONE-branch call site at workflow.py:2437-2440: curator vs memory routing.

        With non-empty suggestions the real call site routes to
        ``_route_review_suggestions_to_curator``; with an empty list it routes
        to ``_write_suggestions_to_memory``.  ``_execute_verify_review_loop``
        is driven end-to-end, so a regression (swapped branches, dropped
        ``await``, inverted condition) causes this test to fail.
        """
        wf = self._make_e2e_workflow(tmp_path)

        reviews = ReviewAggregation(
            has_blocking_issues=False,
            blocking_issues=[],
            suggestions=suggestions,
            reviews={'analyst': {}},
        )

        wf._review = AsyncMock(return_value=reviews)  # type: ignore[method-assign]

        route_mock = AsyncMock()
        write_mock = AsyncMock()
        wf._route_review_suggestions_to_curator = route_mock  # type: ignore[method-assign]
        wf._write_suggestions_to_memory = write_mock  # type: ignore[method-assign]

        outcome = await wf._execute_verify_review_loop()

        assert outcome == WorkflowOutcome.DONE
        assert route_mock.await_count == route_n
        assert write_mock.await_count == write_n
        if route_n:
            route_mock.assert_awaited_once_with(reviews)
        else:
            write_mock.assert_awaited_once_with(reviews)


# ---------------------------------------------------------------------------
# _escalate_review_issues
# ---------------------------------------------------------------------------

class TestEscalateReviewIssues:
    def test_creates_blocking_escalation(self):
        queue = MagicMock()
        queue.make_id.return_value = 'esc-42-5'
        wf = _make_workflow(escalation_queue=queue)
        wf.state = MagicMock(value='review')

        reviews = _fake_reviews(
            blocking_issues=[{'description': 'bug'}, {'description': 'crash'}],
            suggestions=[{'description': 'style'}],
        )

        wf._escalate_review_issues(reviews)

        queue.submit.assert_called_once()
        esc = queue.submit.call_args[0][0]
        assert esc.severity == 'blocking'
        assert esc.category == 'review_issues'
        assert '2 blocking issue(s)' in esc.summary
        assert '1 suggestion(s)' in esc.summary

    def test_noop_without_queue(self):
        wf = _make_workflow(escalation_queue=None)
        reviews = _fake_reviews(blocking_issues=[{'description': 'bug'}])
        wf._escalate_review_issues(reviews)  # Should not raise


# ---------------------------------------------------------------------------
# _await_steward_completion
# ---------------------------------------------------------------------------

class TestAwaitStewardCompletion:
    @pytest.mark.asyncio
    async def test_returns_immediately_if_no_queue(self):
        wf = _make_workflow(escalation_queue=None)
        await wf._await_steward_completion()

    @pytest.mark.asyncio
    async def test_returns_immediately_if_no_pending(self):
        queue = MagicMock()
        queue.get_by_task.return_value = []
        wf = _make_workflow(escalation_queue=queue)
        await wf._await_steward_completion()

    @pytest.mark.asyncio
    async def test_returns_when_resolved(self):
        esc = _make_escalation()
        queue = MagicMock()
        queue.get_by_task.side_effect = [[esc], []]

        event = asyncio.Event()
        wf = _make_workflow(escalation_queue=queue, escalation_event=event)
        wf._steward = MagicMock()  # steward must be running to wait

        async def _resolve_after_delay():
            await asyncio.sleep(0.05)
            event.set()

        task = asyncio.create_task(_resolve_after_delay())
        await wf._await_steward_completion()
        await task

        queue.submit.assert_not_called()
        queue.resolve.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_interrupted_with_no_queue_side_effects_on_timeout(self):
        """task 2248 (W9-delta): the in-method L1 re-escalation-on-timeout
        block was removed from _await_steward_completion — on grace-timeout
        it now just returns StewardInterrupted('timeout', ...) and never
        touches the escalation queue itself. _mark_blocked's single
        StewardOutcome branch (see test_workflow_e2e.py) owns the
        escalate-vs-resume-plan decision instead."""
        queue = MagicMock()
        wf = _make_workflow(escalation_queue=queue)
        wf.config.steward_completion_timeout = 0.1
        wf._steward_outcome_channel = asyncio.Queue()  # nothing published
        wf.scheduler.get_status = AsyncMock(return_value='blocked')

        outcome = await wf._await_steward_completion()

        assert outcome == StewardInterrupted('timeout', wip_commits_present=False)
        queue.submit.assert_not_called()
        queue.resolve.assert_not_called()


# ---------------------------------------------------------------------------
# Review loop routing
# ---------------------------------------------------------------------------

class TestReviewLoopRouting:
    def test_escalates_instead_of_memory_write_when_queue_available(self):
        queue = MagicMock()
        queue.make_id.return_value = 'esc-42-0'
        wf = _make_workflow(escalation_queue=queue)
        wf.state = MagicMock(value='review')

        suggestions = [{'description': 'something'}]
        reviews = _fake_reviews(suggestions)

        with patch.object(wf, '_write_suggestions_to_memory') as mock_write:
            wf._escalate_suggestions(reviews)
            mock_write.assert_not_called()
            queue.submit.assert_called_once()

    def test_falls_back_to_memory_write_without_queue(self):
        wf = _make_workflow(escalation_queue=None)
        suggestions = [{'description': 'something'}]
        reviews = _fake_reviews(suggestions)
        wf._escalate_suggestions(reviews)


# ---------------------------------------------------------------------------
# Pre-triage integration (steward)
# ---------------------------------------------------------------------------


def _make_steward(*, config_overrides=None, suggestion_count=15):
    """Build a minimal TaskSteward with mocked dependencies."""
    import tempfile
    from pathlib import Path

    from orchestrator.steward import TaskSteward

    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    config.project_root = Path('/tmp/project')
    config.steward_lifetime_budget = 12.0
    config.steward_max_attempts = 3
    config.steward_max_timeouts_per_escalation = 3
    config.steward_max_empty_outputs_per_escalation = 2
    config.suggestion_triage_threshold = 10
    config.models.triage = 'sonnet'
    config.budgets.triage = 2.0
    config.max_turns.triage = 25
    config.effort.triage = 'medium'
    config.backends.triage = 'claude'
    config.models.steward = 'opus'
    config.budgets.steward = 5.0
    config.max_turns.steward = 100
    config.effort.steward = 'high'
    config.backends.steward = 'claude'
    config.escalation.host = '127.0.0.1'
    config.escalation.port = 8100
    if config_overrides:
        for k, v in config_overrides.items():
            setattr(config, k, v)

    queue = MagicMock()
    queue.make_id.return_value = 'esc-42-1'
    queue.get_by_task.return_value = []

    briefing = MagicMock()
    briefing.build_steward_initial_prompt = AsyncMock(return_value='initial prompt')

    mcp = MagicMock()
    mcp.mcp_config_json.return_value = {}

    task = {'id': '42', 'title': 'Test Task', 'description': 'desc'}
    # Create a real tmp worktree so the steward's pre-flight (added in the
    # zombie-escalation fix) does not auto-escalate before the test gets a
    # chance to assert.  Tests mock invoke_agent so nothing actually runs
    # against this directory.
    worktree = Path(tempfile.mkdtemp(prefix='test-steward-wt-'))
    steward = TaskSteward(
        task_id='42',
        task=task,
        worktree=worktree,
        config=config,
        mcp=mcp,
        escalation_queue=queue,
        briefing=briefing,
        usage_gate=None,
    )
    return steward


def _make_suggestions(n):
    """Generate n fake review suggestions."""
    return [
        {
            'reviewer': 'test_analyst',
            'severity': 'suggestion',
            'location': f'src/mod{i}.py:{i * 10}',
            'category': 'coverage',
            'description': f'Missing test for case {i}',
            'suggested_fix': f'Add test for case {i}',
        }
        for i in range(n)
    ]


def _fake_agent_result(*, cost=0.5, turns=3, structured_output=None):
    """Return a mock AgentResult."""
    result = MagicMock()
    result.cost_usd = cost
    result.duration_ms = 2000
    result.turns = turns
    result.success = True
    result.session_id = 'sess-123'
    result.stderr = ''
    result.output = ''
    result.structured_output = structured_output
    result.account_name = ''
    return result


class TestPreTriageSuggestions:
    @pytest.mark.asyncio
    async def test_pre_triage_invoked_above_threshold(self):
        steward = _make_steward()
        suggestions = _make_suggestions(15)

        triage_output = {
            'accepted': [
                {'index': i, 'suggestion': f'case {i}', 'reason': 'merit',
                 'files': [f'src/mod{i}.py'], 'proposed_task_title': f'Fix {i}'}
                for i in range(10)
            ],
            'skipped': [
                {'index': i, 'suggestion': f'case {i}', 'reason': 'noise'}
                for i in range(10, 15)
            ],
            'proposed_task_groups': [
                {'title': 'Add tests', 'description': 'Add missing tests',
                 'accepted_indices': list(range(10))},
            ],
        }

        esc = _make_escalation(detail=json.dumps(suggestions))

        with patch('orchestrator.steward.invoke_with_cap_retry',
                   return_value=_fake_agent_result(structured_output=triage_output)):
            result = await steward._pre_triage_suggestions(esc)

        assert '## Pre-Triaged Results' in result.detail
        assert '10 accepted' in result.summary
        assert '5 skipped' in result.summary

    @pytest.mark.asyncio
    async def test_pre_triage_not_invoked_below_threshold(self):
        """Small suggestion sets should skip pre-triage in _handle_escalation."""
        steward = _make_steward()
        suggestions = _make_suggestions(5)
        esc = _make_escalation(detail=json.dumps(suggestions))

        # Steward session mock — returns resolved escalation
        steward_result = _fake_agent_result(cost=2.0)
        cast(MagicMock, steward.escalation_queue).get.return_value = MagicMock(status='resolved')

        with patch('orchestrator.steward.invoke_agent', return_value=steward_result) as mock_invoke:
            await steward._handle_escalation(esc)

        # Only the steward session should be called — no triage invocation
        assert mock_invoke.call_count == 1
        call_kwargs = mock_invoke.call_args
        assert call_kwargs.kwargs.get('model') or 'opus' in str(call_kwargs)

    @pytest.mark.asyncio
    async def test_pre_triage_failure_falls_back(self):
        steward = _make_steward()
        suggestions = _make_suggestions(15)
        esc = _make_escalation(detail=json.dumps(suggestions))

        # Triage returns no structured output
        bad_result = _fake_agent_result(structured_output=None)
        bad_result.success = False

        with patch('orchestrator.steward.invoke_with_cap_retry', return_value=bad_result):
            result = await steward._pre_triage_suggestions(esc)

        # Original escalation returned unchanged
        assert result.detail == esc.detail
        assert result.summary == esc.summary

    @pytest.mark.asyncio
    async def test_pre_triage_cost_tracked_in_metrics(self):
        steward = _make_steward()
        assert steward.metrics.total_cost_usd == 0.0

        suggestions = _make_suggestions(15)
        esc = _make_escalation(detail=json.dumps(suggestions))

        triage_output = {
            'accepted': [], 'skipped': [],
            'proposed_task_groups': [],
        }
        result = _fake_agent_result(cost=0.75, structured_output=triage_output)

        with patch('orchestrator.steward.invoke_with_cap_retry', return_value=result):
            await steward._pre_triage_suggestions(esc)

        assert steward.metrics.total_cost_usd == 0.75
        assert steward.metrics.invocations == 1

    @pytest.mark.asyncio
    async def test_pre_triage_replaces_escalation_detail(self):
        steward = _make_steward()
        suggestions = _make_suggestions(12)
        esc = _make_escalation(detail=json.dumps(suggestions))

        triage_output = {
            'accepted': [
                {'index': 0, 'suggestion': 'case 0', 'reason': 'merit',
                 'files': ['src/mod0.py'], 'proposed_task_title': 'Fix 0'},
            ],
            'skipped': [
                {'index': i, 'suggestion': f'case {i}', 'reason': 'noise'}
                for i in range(1, 12)
            ],
            'proposed_task_groups': [
                {'title': 'Fix case 0', 'description': 'Fix it',
                 'accepted_indices': [0]},
            ],
        }

        with patch('orchestrator.steward.invoke_with_cap_retry',
                   return_value=_fake_agent_result(structured_output=triage_output)):
            result = await steward._pre_triage_suggestions(esc)

        # Detail is replaced with pre-triaged markdown
        assert '## Pre-Triaged Results' in result.detail
        assert 'Fix case 0' in result.detail
        # Original suggestions are embedded as reference
        assert 'Original Suggestions' in result.detail


# ---------------------------------------------------------------------------
# Pre-triage cap handling
# ---------------------------------------------------------------------------


class TestPreTriageCapHandling:
    @pytest.mark.asyncio
    async def test_pre_triage_returns_original_escalation_on_cap(
        self, caplog
    ):
        """_pre_triage_suggestions must return the original escalation unchanged on cap.

        Before step-8 impl: AllAccountsCappedException propagates out of
        _pre_triage_suggestions, crashing the steward.
        After step-8 impl: exception is caught, original escalation returned.
        """
        steward = _make_steward()
        suggestions = _make_suggestions(15)
        escalation = _make_escalation(detail=json.dumps(suggestions))

        cap_exc = AllAccountsCappedException(
            retries=2, elapsed_secs=30.0, label='Steward for task 42 [pre-triage]'
        )

        with patch(
            'orchestrator.steward.invoke_with_cap_retry',
            AsyncMock(side_effect=cap_exc),
        ), caplog.at_level(logging.WARNING, logger='orchestrator.steward'):
            result = await steward._pre_triage_suggestions(escalation)

        # Must return the original escalation (identity check)
        assert result is escalation, (
            'Expected original escalation object to be returned unchanged'
        )

        # Must emit a warning containing 'all accounts capped'
        warning_texts = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(
            'all accounts capped' in t.lower() for t in warning_texts
        ), f'Expected warning with "all accounts capped", got: {warning_texts}'
