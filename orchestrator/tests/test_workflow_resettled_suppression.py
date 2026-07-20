"""Tests for the prior-round-resolution suppression workflow methods (task 2523).

Covers ``TaskWorkflow._adjudicate_resettled`` (the batched, fail-safe LLM
adjudication) and ``TaskWorkflow._suppress_resettled_suggestions`` (the
``ReviewAggregation`` filter that consumes it), exercised directly with
injected fakes — no real LLM, no live orchestrator loop.  Mirrors the
``MagicMock(spec_set=pydantic_spec(OrchestratorConfig))`` pattern from
test_workflow_amendment.py.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec

from orchestrator.artifacts import ReviewAggregation
from orchestrator.config import OrchestratorConfig
from orchestrator.workflow import TaskWorkflow

# Orchestrator's own pyproject.toml is the pytest rootdir config and does not
# set asyncio_mode, so async tests need the explicit marker (the convention in
# test_provenance_gate_integration.py / test_steward_outcome.py).
pytestmark = pytest.mark.asyncio


def _make_workflow(*, tmp_path=None, suppress: bool = True) -> TaskWorkflow:
    """Minimal TaskWorkflow wired for the suppression method tests.

    Reuses test_workflow_amendment.py's MagicMock(spec_set=...) config pattern,
    additionally setting ``suppress_resettled_review_suggestions`` (the task-2523
    gate) and — when *tmp_path* is given — ``self.worktree``.
    """
    assignment = MagicMock()
    assignment.task_id = '2523'
    assignment.task = {'id': '2523', 'title': 'Test Task', 'description': 'd'}
    assignment.modules = ['orchestrator']

    _spec = pydantic_spec(OrchestratorConfig)
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.max_review_cycles = 2
    config.max_amendment_rounds = 1
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0
    config.suppress_resettled_review_suggestions = suppress

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=MagicMock(),
        scheduler=MagicMock(),
        briefing=MagicMock(),
        mcp=MagicMock(),
    )
    if tmp_path is not None:
        wf.worktree = tmp_path
    return wf


def _issue(severity, description, location='src/foo.py:1', **extras) -> dict:
    d = {
        'severity': severity,
        'location': location,
        'category': 'coverage',
        'description': description,
        'suggested_fix': 'do the thing',
    }
    d.update(extras)
    return d


def _result(*, success=True, timed_out=False, structured_output=None):
    """A minimal AgentResult stand-in exposing the fields _adjudicate reads."""
    return SimpleNamespace(
        success=success,
        timed_out=timed_out,
        structured_output=structured_output,
    )


class TestAdjudicateResettled:
    """``_adjudicate_resettled(current_list, prior_settled)`` — fail-safe matrix."""

    async def test_success_maps_decisions(self, tmp_path):
        """(a) A well-formed decisions payload maps positionally."""
        wf = _make_workflow(tmp_path=tmp_path)
        wf._invoke = AsyncMock(return_value=_result(
            structured_output={'decisions': [
                {'index': 0, 'decision': 'settled'},
                {'index': 1, 'decision': 'not_settled'},
            ]},
        ))
        out = await wf._adjudicate_resettled(
            [_issue('suggestion', 'a'), _issue('suggestion', 'b')],
            [_issue('suggestion', 'p')],
        )
        assert out == ['settled', 'not_settled']

    async def test_invoke_raises_all_not_settled(self, tmp_path):
        """(b) A raised exception fails safe → all NOT_SETTLED."""
        from orchestrator.review_suggestions.prior_round import NOT_SETTLED
        wf = _make_workflow(tmp_path=tmp_path)
        wf._invoke = AsyncMock(side_effect=RuntimeError('boom'))
        current = [_issue('suggestion', 'a'), _issue('suggestion', 'b')]
        out = await wf._adjudicate_resettled(current, [])
        assert out == [NOT_SETTLED] * len(current)

    async def test_non_success_or_timed_out_all_not_settled(self, tmp_path):
        """(c) A non-success or timed-out invoke fails safe → all NOT_SETTLED."""
        from orchestrator.review_suggestions.prior_round import NOT_SETTLED
        current = [_issue('suggestion', 'a')]
        wf1 = _make_workflow(tmp_path=tmp_path)
        wf1._invoke = AsyncMock(return_value=_result(
            success=False,
            structured_output={'decisions': [{'index': 0, 'decision': 'settled'}]},
        ))
        assert await wf1._adjudicate_resettled(current, []) == [NOT_SETTLED]
        wf2 = _make_workflow(tmp_path=tmp_path)
        wf2._invoke = AsyncMock(return_value=_result(
            timed_out=True,
            structured_output={'decisions': [{'index': 0, 'decision': 'settled'}]},
        ))
        assert await wf2._adjudicate_resettled(current, []) == [NOT_SETTLED]

    async def test_unparseable_output_all_not_settled(self, tmp_path):
        """(d) Missing / unparseable structured_output fails safe."""
        from orchestrator.review_suggestions.prior_round import NOT_SETTLED
        current = [_issue('suggestion', 'a'), _issue('suggestion', 'b')]
        wf1 = _make_workflow(tmp_path=tmp_path)
        wf1._invoke = AsyncMock(return_value=_result(structured_output=None))
        assert await wf1._adjudicate_resettled(current, []) == [NOT_SETTLED] * 2
        wf2 = _make_workflow(tmp_path=tmp_path)
        wf2._invoke = AsyncMock(
            return_value=_result(structured_output={'foo': 'bar'})
        )
        assert await wf2._adjudicate_resettled(current, []) == [NOT_SETTLED] * 2

    async def test_inconclusive_preserved_and_omitted_index_defaults(self, tmp_path):
        """(e) An explicit 'inconclusive' is preserved; an omitted index → NOT_SETTLED."""
        from orchestrator.review_suggestions.prior_round import (
            INCONCLUSIVE,
            NOT_SETTLED,
        )
        wf = _make_workflow(tmp_path=tmp_path)
        # Only index 0 is returned; index 1 is omitted by the model.
        wf._invoke = AsyncMock(return_value=_result(
            structured_output={'decisions': [
                {'index': 0, 'decision': 'inconclusive'},
            ]},
        ))
        out = await wf._adjudicate_resettled(
            [_issue('suggestion', 'a'), _issue('suggestion', 'b')], [],
        )
        assert out == [INCONCLUSIVE, NOT_SETTLED]

    async def test_unknown_decision_value_coerced_to_not_settled(self, tmp_path):
        """A verdict outside the vocabulary is coerced to NOT_SETTLED (emit)."""
        from orchestrator.review_suggestions.prior_round import NOT_SETTLED
        wf = _make_workflow(tmp_path=tmp_path)
        wf._invoke = AsyncMock(return_value=_result(
            structured_output={'decisions': [
                {'index': 0, 'decision': 'banana'},
            ]},
        ))
        out = await wf._adjudicate_resettled([_issue('suggestion', 'a')], [])
        assert out == [NOT_SETTLED]

    async def test_empty_current_returns_empty_without_invoke(self, tmp_path):
        """No suggestions → [] and the adjudicator is never invoked."""
        wf = _make_workflow(tmp_path=tmp_path)
        wf._invoke = AsyncMock()
        out = await wf._adjudicate_resettled([], [_issue('suggestion', 'p')])
        assert out == []
        wf._invoke.assert_not_awaited()


class TestSuppressResettledSuggestions:
    """``_suppress_resettled_suggestions(reviews, amendment_round)``."""

    @staticmethod
    def _agg(
        suggestions, *, blocking=None, has_blocking=False, reviews=None,
        errors=None,
    ) -> ReviewAggregation:
        return ReviewAggregation(
            has_blocking_issues=has_blocking,
            blocking_issues=list(blocking or []),
            suggestions=list(suggestions),
            reviews=dict(reviews or {}),
            reviewer_errors=list(errors or []),
        )

    @staticmethod
    def _write_prior_archive(root) -> None:
        """A non-empty reviews-amend-1/ so load_prior_round_suggestions != []."""
        d = root / 'reviews-amend-1'
        d.mkdir(parents=True, exist_ok=True)
        (d / 'reviewer_comprehensive.json').write_text(json.dumps({
            'verdict': 'PASS',
            'issues': [_issue('suggestion', 'prior-settled')],
        }))

    @staticmethod
    def _with_artifacts(wf, root):
        wf.artifacts = MagicMock()
        wf.artifacts.root = root
        return wf

    async def test_round_zero_is_noop(self, tmp_path):
        """(a) amendment_round == 0 → unchanged, adjudicator not called."""
        wf = self._with_artifacts(_make_workflow(tmp_path=tmp_path), tmp_path)
        self._write_prior_archive(tmp_path)
        wf._adjudicate_resettled = AsyncMock()
        reviews = self._agg([_issue('suggestion', 'a')])
        out = await wf._suppress_resettled_suggestions(reviews, 0)
        assert out is reviews
        wf._adjudicate_resettled.assert_not_awaited()

    async def test_flag_off_is_noop(self, tmp_path):
        """(b) suppress_resettled_review_suggestions False → unchanged."""
        wf = self._with_artifacts(
            _make_workflow(tmp_path=tmp_path, suppress=False), tmp_path
        )
        self._write_prior_archive(tmp_path)
        wf._adjudicate_resettled = AsyncMock()
        reviews = self._agg([_issue('suggestion', 'a')])
        out = await wf._suppress_resettled_suggestions(reviews, 1)
        assert out is reviews
        wf._adjudicate_resettled.assert_not_awaited()

    async def test_no_archives_is_noop(self, tmp_path):
        """(c) No reviews-amend-*/ archives → unchanged."""
        wf = self._with_artifacts(_make_workflow(tmp_path=tmp_path), tmp_path)
        wf._adjudicate_resettled = AsyncMock()
        reviews = self._agg([_issue('suggestion', 'a')])
        out = await wf._suppress_resettled_suggestions(reviews, 1)
        assert out is reviews
        wf._adjudicate_resettled.assert_not_awaited()

    async def test_empty_suggestions_is_noop(self, tmp_path):
        """No suggestions → unchanged, adjudicator not called."""
        wf = self._with_artifacts(_make_workflow(tmp_path=tmp_path), tmp_path)
        self._write_prior_archive(tmp_path)
        wf._adjudicate_resettled = AsyncMock()
        reviews = self._agg([])
        out = await wf._suppress_resettled_suggestions(reviews, 1)
        assert out is reviews
        wf._adjudicate_resettled.assert_not_awaited()

    async def test_settled_suppressed_others_preserved(self, tmp_path):
        """(d) SETTLED dropped, others kept; non-suggestion fields verbatim."""
        wf = self._with_artifacts(_make_workflow(tmp_path=tmp_path), tmp_path)
        self._write_prior_archive(tmp_path)
        s0 = _issue('suggestion', 'resettled-one')
        s1 = _issue('suggestion', 'still-open')
        wf._adjudicate_resettled = AsyncMock(
            return_value=['settled', 'not_settled']
        )
        reviews = self._agg(
            [s0, s1],
            blocking=[_issue('blocking', 'blk')],
            has_blocking=True,
            reviews={'reviewer_comprehensive': {'verdict': 'PASS'}},
            errors=['some_reviewer'],
        )
        out = await wf._suppress_resettled_suggestions(reviews, 1)
        assert out.suggestions == [s1]
        assert out.blocking_issues == reviews.blocking_issues
        assert out.reviews == reviews.reviews
        assert out.reviewer_errors == reviews.reviewer_errors
        assert out.has_blocking_issues == reviews.has_blocking_issues

    async def test_blocking_never_passed_to_adjudicator(self, tmp_path):
        """(e) Blocking issues are never handed to the adjudicator nor suppressed."""
        wf = self._with_artifacts(_make_workflow(tmp_path=tmp_path), tmp_path)
        self._write_prior_archive(tmp_path)
        s0 = _issue('suggestion', 's0')
        s1 = _issue('suggestion', 's1')
        blk = _issue('blocking', 'blk')
        wf._adjudicate_resettled = AsyncMock(
            return_value=['not_settled', 'not_settled']
        )
        reviews = self._agg([s0, s1], blocking=[blk], has_blocking=True)
        out = await wf._suppress_resettled_suggestions(reviews, 1)
        passed_current = wf._adjudicate_resettled.await_args.args[0]
        assert passed_current == [s0, s1]
        assert blk not in passed_current
        assert out.blocking_issues == [blk]

    async def test_adjudicator_raises_keeps_all(self, tmp_path):
        """(f) An adjudicator exception fails safe → all suggestions kept."""
        wf = self._with_artifacts(_make_workflow(tmp_path=tmp_path), tmp_path)
        self._write_prior_archive(tmp_path)
        s0 = _issue('suggestion', 's0')
        s1 = _issue('suggestion', 's1')
        wf._adjudicate_resettled = AsyncMock(side_effect=RuntimeError('boom'))
        reviews = self._agg([s0, s1])
        out = await wf._suppress_resettled_suggestions(reviews, 1)
        assert out.suggestions == [s0, s1]
