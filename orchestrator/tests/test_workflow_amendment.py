"""Tests for the L2b in-workflow amendment loop.

Focused on ``TaskWorkflow._suggestions_in_scope`` — the pure-function filter
that decides which reviewer suggestions can be applied by an in-place
implementer amendment pass (vs. escalated as follow-up tasks). The filter
uses ``scheduler.normalize_lock`` against ``self.modules`` (the lock set
granted at schedule time) and must:

- preserve the scheduler's concurrency invariants (no footprint expansion)
- tolerate malformed / missing location fields without raising
- correctly include new files inside a locked module (path doesn't have
  to exist on disk — normalization is pure string manipulation)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec

from orchestrator.artifacts import ReviewAggregation
from orchestrator.config import OrchestratorConfig
from orchestrator.workflow import (
    AmendmentReviewContext,
    TaskWorkflow,
    WorkflowOutcome,
)


def _make_workflow(*, modules: list[str], lock_depth: int = 2) -> TaskWorkflow:
    """Minimal TaskWorkflow instance for filter tests.

    Mirrors the pattern used in ``test_suggestion_triage.py`` but wires up
    the fields ``_suggestions_in_scope`` actually reads: ``self.modules``
    (from ``assignment.modules``) and ``self.config.lock_depth``.
    """
    assignment = MagicMock()
    assignment.task_id = '42'
    assignment.task = {'id': '42', 'title': 'Test Task', 'description': 'd'}
    assignment.modules = modules

    _spec = pydantic_spec(OrchestratorConfig)
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.max_review_cycles = 2
    config.max_amendment_rounds = 1
    config.lock_depth = lock_depth
    config.steward_completion_timeout = 300.0

    return TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=MagicMock(),
        scheduler=MagicMock(),
        briefing=MagicMock(),
        mcp=MagicMock(),
    )


def _sugg(location: str | None, **extras) -> dict:
    """Build a suggestion dict with the fields the filter cares about."""
    s: dict = {
        'reviewer': 'test_analyst',
        'severity': 'suggestion',
        'category': 'coverage',
        'description': 'an opinion',
        'suggested_fix': 'do the thing',
    }
    if location is not None:
        s['location'] = location
    s.update(extras)
    return s


class TestSuggestionsInScope:
    """``_suggestions_in_scope`` — the amendment-pass scope filter."""

    def test_empty_suggestions_returns_empty(self):
        wf = _make_workflow(modules=['crates/reify-types'])
        assert wf._suggestions_in_scope([]) == []

    def test_empty_lock_set_returns_empty_and_warns(self, caplog):
        wf = _make_workflow(modules=[])
        suggestions = [_sugg('crates/reify-types/src/persistent.rs:42')]
        with caplog.at_level(logging.WARNING):
            result = wf._suggestions_in_scope(suggestions)
        assert result == []
        assert any(
            'empty lock set' in rec.message for rec in caplog.records
        ), 'expected a warning when self.modules is empty'

    def test_location_inside_locked_module_included(self):
        wf = _make_workflow(modules=['crates/reify-types'])
        s = _sugg('crates/reify-types/src/persistent.rs:42')
        assert wf._suggestions_in_scope([s]) == [s]

    def test_location_outside_locked_module_excluded(self):
        wf = _make_workflow(modules=['crates/reify-types'])
        s = _sugg('crates/reify-compiler/src/pass.rs:10')
        assert wf._suggestions_in_scope([s]) == []

    def test_mixed_suggestions_partitioned_correctly(self):
        wf = _make_workflow(modules=['crates/reify-types'])
        inside = _sugg('crates/reify-types/src/a.rs:1', category='naming')
        outside = _sugg('crates/reify-eval/src/b.rs:2', category='perf')
        assert wf._suggestions_in_scope([inside, outside]) == [inside]

    def test_new_file_inside_locked_module_included(self):
        """A suggestion pointing at a path that doesn't yet exist on disk
        but normalizes to a locked module is in-scope. The filter is a pure
        string operation — it never stats the filesystem."""
        wf = _make_workflow(modules=['crates/reify-types'])
        s = _sugg('crates/reify-types/src/brand_new_helper.rs:1')
        assert wf._suggestions_in_scope([s]) == [s]

    def test_missing_location_field_excluded(self):
        wf = _make_workflow(modules=['crates/reify-types'])
        s = _sugg(None)  # no 'location' key at all
        assert 'location' not in s
        assert wf._suggestions_in_scope([s]) == []

    def test_empty_location_string_excluded(self):
        wf = _make_workflow(modules=['crates/reify-types'])
        assert wf._suggestions_in_scope([_sugg('')]) == []

    def test_whitespace_only_location_excluded(self):
        wf = _make_workflow(modules=['crates/reify-types'])
        assert wf._suggestions_in_scope([_sugg('   ')]) == []

    def test_location_without_colon_still_parses(self):
        """``location`` without ``:line`` should still be accepted — the
        filter splits on ``:`` and takes the path half, which is the whole
        string when no colon is present."""
        wf = _make_workflow(modules=['crates/reify-types'])
        s = _sugg('crates/reify-types/src/a.rs')
        assert wf._suggestions_in_scope([s]) == [s]

    def test_location_bare_colon_excluded(self):
        """``location`` that's just ``:`` strips to an empty file path and
        should be excluded without raising."""
        wf = _make_workflow(modules=['crates/reify-types'])
        assert wf._suggestions_in_scope([_sugg(':')]) == []

    def test_lock_depth_respected(self):
        """The filter normalizes with ``config.lock_depth``, not
        ``normalize_lock``'s default. With depth=3, the module key for
        ``crates/reify-types/src/persistent.rs`` is ``crates/reify-types/src``
        — which must match exactly against a depth-3 lock."""
        wf = _make_workflow(
            modules=['crates/reify-types/src'], lock_depth=3,
        )
        inside = _sugg('crates/reify-types/src/persistent.rs:1')
        outside = _sugg('crates/reify-types/tests/prop.rs:1')
        result = wf._suggestions_in_scope([inside, outside])
        assert result == [inside]

    def test_multiple_locked_modules(self):
        """A task with multiple module locks accepts suggestions in any
        of them and rejects suggestions in non-locked ones."""
        wf = _make_workflow(
            modules=['crates/reify-types', 'crates/reify-eval'],
        )
        in_types = _sugg('crates/reify-types/src/a.rs:1')
        in_eval = _sugg('crates/reify-eval/src/b.rs:1')
        outside = _sugg('crates/reify-compiler/src/c.rs:1')
        result = wf._suggestions_in_scope([in_types, in_eval, outside])
        assert result == [in_types, in_eval]

    def test_filter_preserves_order(self):
        """Output order matches input order for in-scope items (not
        observed-behavior-dependent, but guards against accidental set()
        round-trips that would lose ordering)."""
        wf = _make_workflow(modules=['crates/reify-types'])
        s1 = _sugg('crates/reify-types/src/a.rs:1', description='first')
        s2 = _sugg('crates/reify-types/src/b.rs:2', description='second')
        s3 = _sugg('crates/reify-types/src/c.rs:3', description='third')
        assert wf._suggestions_in_scope([s1, s2, s3]) == [s1, s2, s3]


# ---------------------------------------------------------------------------
# task 2750: _apply_amendment_delta_scope — deterministic delta partition
# ---------------------------------------------------------------------------


def _make_scope_workflow(delta: dict) -> TaskWorkflow:
    """Workflow wired for _apply_amendment_delta_scope tests.

    ``get_new_side_changed_line_ranges`` returns *delta*; the curator router
    is replaced with an AsyncMock so routing is observable without MCP.
    """
    wf = _make_workflow(modules=['src'])
    wf.worktree = Path('/tmp/wt-2750')
    wf.git_ops.get_new_side_changed_line_ranges = AsyncMock(return_value=delta)
    wf._route_review_suggestions_to_curator = AsyncMock()
    return wf


def _blocking(location: str) -> dict:
    return {
        'reviewer': 'test_analyst',
        'severity': 'blocking',
        'category': 'correctness',
        'location': location,
        'description': 'a real bug',
        'suggested_fix': 'fix it',
    }


@pytest.mark.asyncio
class TestApplyAmendmentDeltaScope:
    """TaskWorkflow._apply_amendment_delta_scope — the enforceable delta scope."""

    async def _ctx(self, suggestions):
        from orchestrator.workflow import AmendmentReviewContext
        return AmendmentReviewContext(
            pre_amendment_head='PRESHA', amended_suggestions=suggestions,
        )

    async def test_partitions_and_routes_out_of_delta(self):
        wf = _make_scope_workflow({'src/foo.py': [(10, 20)]})
        blocking = _blocking('src/other.py:99')
        in_sugg = _sugg('src/foo.py:15')
        out_sugg = _sugg('src/other.py:5')
        reviews = ReviewAggregation(
            has_blocking_issues=True,
            blocking_issues=[blocking],
            suggestions=[in_sugg, out_sugg],
            reviews={'test_analyst': {'verdict': 'APPROVE'}},
            reviewer_errors=[],
        )
        ctx = await self._ctx([in_sugg])

        result = await wf._apply_amendment_delta_scope(reviews, ctx)

        # (i) verdict keeps only the in-delta suggestion.
        assert result.suggestions == [in_sugg]
        # (ii) blocking fields pass through untouched (safety valve).
        assert result.has_blocking_issues is True
        assert result.blocking_issues == [blocking]
        # (iii) the out-of-delta suggestion is routed to the curator, once.
        route_mock = cast(AsyncMock, wf._route_review_suggestions_to_curator)
        route_mock.assert_awaited_once()
        routed = route_mock.call_args.args[0]
        assert routed.suggestions == [out_sugg]
        # the git delta query used the pre-amendment head.
        delta_mock = cast(AsyncMock, wf.git_ops.get_new_side_changed_line_ranges)
        delta_mock.assert_awaited_once()
        gargs = delta_mock.call_args
        assert gargs.args[0] == wf.worktree
        assert gargs.args[1] == 'PRESHA'

    async def test_no_out_of_delta_does_not_route(self):
        wf = _make_scope_workflow({'src/foo.py': [(10, 20)]})
        in1 = _sugg('src/foo.py:12')
        in2 = _sugg('src/foo.py')  # file-level on a delta file → in-delta
        reviews = ReviewAggregation(
            has_blocking_issues=False,
            blocking_issues=[],
            suggestions=[in1, in2],
            reviews={},
            reviewer_errors=[],
        )
        ctx = await self._ctx([in1, in2])

        result = await wf._apply_amendment_delta_scope(reviews, ctx)

        assert result.suggestions == [in1, in2]
        cast(AsyncMock, wf._route_review_suggestions_to_curator).assert_not_awaited()

    async def test_fail_open_on_empty_delta(self, caplog):
        wf = _make_scope_workflow({})  # uncomputable / empty delta
        s1 = _sugg('src/foo.py:15')
        s2 = _sugg('src/other.py:5')
        reviews = ReviewAggregation(
            has_blocking_issues=False,
            blocking_issues=[],
            suggestions=[s1, s2],
            reviews={},
            reviewer_errors=[],
        )
        ctx = await self._ctx([s1])

        with caplog.at_level(logging.WARNING):
            result = await wf._apply_amendment_delta_scope(reviews, ctx)

        # Fail-open: reviews returned unchanged (no suggestion silently dropped).
        assert result is reviews
        assert result.suggestions == [s1, s2]
        cast(AsyncMock, wf._route_review_suggestions_to_curator).assert_not_awaited()
        assert any(rec.levelno == logging.WARNING for rec in caplog.records), (
            'fail-open must log a WARNING for audit'
        )


# ---------------------------------------------------------------------------
# task 2750: post-amendment review wiring in _execute_verify_review_loop
# ---------------------------------------------------------------------------


_PRE_AMENDMENT_SHA = 'preamendsha000000000000000000000000000000'
_POST_AMENDMENT_SHA = 'postamendsha11111111111111111111111111111'


def _nonblocking_reviews(suggestions: list[dict]) -> ReviewAggregation:
    return ReviewAggregation(
        has_blocking_issues=False,
        blocking_issues=[],
        suggestions=suggestions,
        reviews={},
        reviewer_errors=[],
    )


def _review_amendment_ctx(call):
    """Extract the amendment_ctx passed to a mocked _review call (kw or pos)."""
    if 'amendment_ctx' in call.kwargs:
        return call.kwargs['amendment_ctx']
    return call.args[0] if call.args else None


def _make_loop_workflow(tmp_path, *, round0, round1, call_order):
    """A TaskWorkflow wired to drive ONE amendment round through the loop.

    Every EXECUTE/VERIFY helper is stubbed non-blocking; the task-2749
    verdict-cache and task-2752 verify-checkpoint blocks are made inert
    (``event_store=None`` + ``get_head_tree_hash`` → None) so the loop always
    reaches a real REVIEW.  ``_review`` yields *round0* then *round1*;
    ``_suggestions_in_scope`` is the real filter (modules=['src'], depth=1) so
    a ``src/…`` suggestion in *round0* fires exactly one amendment and *round1*
    (no in-scope suggestion) reaches DONE.
    """
    wf = _make_workflow(modules=['src'], lock_depth=1)
    wf.worktree = tmp_path / 'wt'
    wf.event_store = None  # inert task-2752 verify-checkpoint block

    # artifacts: real root (no archive dirs exist → copytree skipped), counters
    # seeded to 0 so amendment_round/review_cycle are ints, verdict cache inert.
    wf.artifacts = MagicMock()
    wf.artifacts.root = tmp_path
    wf.artifacts.get_review_cycles_total.return_value = 0
    wf.artifacts.get_amendment_rounds_total.return_value = 0
    wf.artifacts.get_cached_verdict.return_value = None

    wf.git_ops.get_head_tree_hash = AsyncMock(return_value=None)
    # task 2760: the post-amendment WIP guard awaits git_ops.has_uncommitted_work
    # before re-entering VERIFY/REVIEW.  These loop tests drive the amendment
    # path but do not exercise WIP commit — a clean tree short-circuits the
    # guard to a no-op (its dedicated coverage lives in
    # test_workflow_amendment_wip_guard.py).
    wf.git_ops.has_uncommitted_work = AsyncMock(return_value=False)

    wf._enter_phase = MagicMock()
    wf._execute_iterations = AsyncMock(return_value=WorkflowOutcome.DONE)
    wf._verify_debugfix_loop = AsyncMock(return_value=WorkflowOutcome.DONE)
    wf._write_suggestions_to_memory = AsyncMock()
    wf._route_review_suggestions_to_curator = AsyncMock()

    # HEAD moves once the amendment commits, so a capture taken AFTER _amend
    # would read _POST_AMENDMENT_SHA — the ctx must carry the PRE value.
    state = {'amended': False}

    def _head(*a, **k):
        call_order.append('head')
        return _POST_AMENDMENT_SHA if state['amended'] else _PRE_AMENDMENT_SHA

    def _amend_impl(*a, **k):
        call_order.append('amend')
        state['amended'] = True
        return True

    async def _scope_passthrough(reviews, ctx):
        return reviews

    wf._get_head_commit = AsyncMock(side_effect=_head)
    wf._amend = AsyncMock(side_effect=_amend_impl)
    wf._apply_amendment_delta_scope = AsyncMock(side_effect=_scope_passthrough)
    wf._review = AsyncMock(side_effect=[round0, round1])
    return wf


@pytest.mark.asyncio
class TestPostAmendmentReviewWiring:
    """`_execute_verify_review_loop` scopes the post-amendment review only."""

    async def test_delta_scope_applied_once_consume_once(self, tmp_path):
        call_order: list[str] = []
        in_scope = _sugg('src/foo.py:15')  # normalizes to 'src' → in-scope
        round0 = _nonblocking_reviews([in_scope])
        round1 = _nonblocking_reviews([])  # nothing in-scope → DONE
        wf = _make_loop_workflow(
            tmp_path, round0=round0, round1=round1, call_order=call_order,
        )

        outcome = await wf._execute_verify_review_loop()

        assert outcome == WorkflowOutcome.DONE
        # exactly one amendment fired.
        cast(AsyncMock, wf._amend).assert_awaited_once()

        # (i) pre_amendment_head is captured by _get_head_commit IMMEDIATELY
        # before _amend, so it is the pre-amendment SHA.
        assert 'amend' in call_order
        amend_idx = call_order.index('amend')
        assert call_order[amend_idx - 1] == 'head', (
            'pre_amendment_head must be captured right before _amend'
        )

        # (ii) the post-amendment (round-1) _review is called with a non-None
        # amendment_ctx carrying the pre-amendment SHA and the round-0 in-scope
        # suggestions; the round-0 review is NOT scoped (amendment_ctx=None).
        review_mock = cast(AsyncMock, wf._review)
        assert review_mock.await_count == 2
        assert _review_amendment_ctx(review_mock.await_args_list[0]) is None
        review_ctx = _review_amendment_ctx(review_mock.await_args_list[1])
        assert isinstance(review_ctx, AmendmentReviewContext)
        assert review_ctx.pre_amendment_head == _PRE_AMENDMENT_SHA
        assert review_ctx.amended_suggestions == [in_scope]

        # (iii) _apply_amendment_delta_scope runs EXACTLY ONCE — for the
        # round-1 post-amendment review only (round-0's used_ctx is None).
        scope_mock = cast(AsyncMock, wf._apply_amendment_delta_scope)
        scope_mock.assert_awaited_once()
        assert scope_mock.await_args is not None
        scope_args = scope_mock.await_args.args
        assert scope_args[0] is round1  # scoped the round-1 verdict
        assert scope_args[1] is review_ctx  # with the consume-once ctx


# ---------------------------------------------------------------------------
# task 2640 — Source 2: amendment out-of-scope complement routed at the source
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAmendmentOutOfScopeRouting:
    """`_execute_verify_review_loop` files THIS round's out-of-scope complement.

    When an amendment fires, the in-scope suggestions are applied by ``_amend``
    but the out-of-scope complement of the SAME round's suggestions was
    previously dropped (task 2750 only recovered it downstream IF a re-review
    repeated it as out-of-delta).  Task 2640 files that complement to the
    curator deterministically, at the source, BEFORE ``_amend`` — so follow-up
    work is captured even if the in-scope amendment later escalates.
    """

    async def test_out_of_scope_complement_routed_once_to_curator(self, tmp_path):
        call_order: list[str] = []
        in_scope = _sugg('src/foo.py:15')       # normalizes to 'src' → in-scope
        out_of_scope = _sugg('other/bar.py:5')  # 'other' → out of scope
        round0 = _nonblocking_reviews([in_scope, out_of_scope])
        round1 = _nonblocking_reviews([])        # nothing in-scope → DONE
        wf = _make_loop_workflow(
            tmp_path, round0=round0, round1=round1, call_order=call_order,
        )

        outcome = await wf._execute_verify_review_loop()

        assert outcome == WorkflowOutcome.DONE

        # The in-scope suggestion drives exactly one amendment; only it is
        # passed to _amend (the out-of-scope one is NOT amended).
        amend_mock = cast(AsyncMock, wf._amend)
        amend_mock.assert_awaited_once()
        assert amend_mock.await_args is not None
        assert amend_mock.await_args.args[0] == [in_scope]

        # The out-of-scope complement is routed to the curator EXACTLY ONCE, as
        # a non-blocking ReviewAggregation carrying only that suggestion (parent
        # linkage / priority / dedup all come from _route_review_suggestions_to_curator).
        route_mock = cast(AsyncMock, wf._route_review_suggestions_to_curator)
        route_mock.assert_awaited_once()
        assert route_mock.await_args is not None
        routed = route_mock.await_args.args[0]
        assert isinstance(routed, ReviewAggregation)
        assert routed.suggestions == [out_of_scope]
        assert routed.has_blocking_issues is False

    async def test_identical_out_of_scope_batch_deduped_across_rounds(self):
        """The single-slot ``_last_routed_suggestion_hash`` short-circuits an
        immediately-repeated identical batch.

        This is the round-0-out-of-scope == round-1-out-of-delta double-file
        case the amendment loop's comment (workflow.py, source point 2) relies
        on: round 0 files THIS round's out-of-scope complement, and a
        subsequent post-amendment ``_apply_amendment_delta_scope`` re-routes a
        content-equal batch as out-of-delta.  The dedup slot check sits ABOVE
        every routing branch (curator submit, escalation-queue fallback, no-op)
        and keys on the CONTENT hash, so two consecutive routes of content-equal
        suggestions file the curator exactly ONCE — even when the second batch
        is a fresh, non-identical dict object.  Exercises the REAL router (not
        the fixture AsyncMock), driving the escalation-queue fallback arm so the
        hash is recorded without opening a background HTTP task.
        """
        wf = _make_workflow(modules=['src'])
        wf.mcp = None
        wf.escalation_queue = MagicMock()
        wf._escalate_suggestions = MagicMock()

        # Round 0: file the out-of-scope complement (records the hash).
        await wf._route_review_suggestions_to_curator(
            _nonblocking_reviews([_sugg('other/bar.py:5')])
        )
        # Round 1: the SAME suggestion re-surfaces as a fresh (content-equal)
        # dict — content hash matches → short-circuit → no second file.
        await wf._route_review_suggestions_to_curator(
            _nonblocking_reviews([_sugg('other/bar.py:5')])
        )

        # Despite two route calls, the fallback fires exactly once.
        assert cast(MagicMock, wf._escalate_suggestions).call_count == 1
