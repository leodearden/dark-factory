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
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec

from orchestrator.artifacts import ReviewAggregation
from orchestrator.config import OrchestratorConfig
from orchestrator.workflow import TaskWorkflow


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
        wf._route_review_suggestions_to_curator.assert_awaited_once()
        routed = wf._route_review_suggestions_to_curator.call_args.args[0]
        assert routed.suggestions == [out_sugg]
        # the git delta query used the pre-amendment head.
        wf.git_ops.get_new_side_changed_line_ranges.assert_awaited_once()
        gargs = wf.git_ops.get_new_side_changed_line_ranges.call_args
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
        wf._route_review_suggestions_to_curator.assert_not_awaited()

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
        wf._route_review_suggestions_to_curator.assert_not_awaited()
        assert any(rec.levelno == logging.WARNING for rec in caplog.records), (
            'fail-open must log a WARNING for audit'
        )
