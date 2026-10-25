"""Tests for reviewer trial scorer — metric computation with canned data (no LLM)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from _eval_refusal import answered
from _orch_helpers import build_usage_gate
from shared.cli_invoke import AgentResult
from shared.config_models import AccountConfig

from orchestrator.evals.measurement import (
    MeasurementStage,
    Scored,
    Unmeasured,
    UnmeasuredCause,
)
from orchestrator.evals.reviewer_trial.corpus import CorpusDiff, GroundTruthIssue
from orchestrator.evals.reviewer_trial.runner import PanelRunResult
from orchestrator.evals.reviewer_trial.scorer import (
    IssueMatch,
    MatchedIssues,
    ScoringResult,
    _deduplicate_issues,
    _normalize_location,
    match_issues,
    score_panel_run,
)


class TestNormalizeLocation:
    def test_strips_line_number(self) -> None:
        assert _normalize_location('src/foo.py:42') == 'src/foo.py'

    def test_strips_line_range(self) -> None:
        assert _normalize_location('src/foo.py:42-50') == 'src/foo.py'

    def test_no_line_number(self) -> None:
        assert _normalize_location('src/foo.py') == 'src/foo.py'

    def test_normalizes_backslash(self) -> None:
        assert _normalize_location('src\\foo.py:10') == 'src/foo.py'

    def test_strips_whitespace(self) -> None:
        assert _normalize_location('  src/foo.py:10  ') == 'src/foo.py'


class TestDeduplicateIssues:
    def test_no_duplicates(self) -> None:
        reviews = {
            'r1': {
                'issues': [
                    {'location': 'a.py:1', 'category': 'bug', 'description': 'Bug A'},
                ]
            },
            'r2': {
                'issues': [
                    {'location': 'b.py:2', 'category': 'style', 'description': 'Style B'},
                ]
            },
        }
        deduped = _deduplicate_issues(reviews)
        assert len(deduped) == 2

    def test_same_location_category_deduped(self) -> None:
        reviews = {
            'r1': {
                'issues': [
                    {'location': 'a.py:1', 'category': 'bug', 'description': 'Short.'},
                ]
            },
            'r2': {
                'issues': [
                    {'location': 'a.py:5', 'category': 'bug', 'description': 'Longer description here.'},
                ]
            },
        }
        deduped = _deduplicate_issues(reviews)
        assert len(deduped) == 1
        # Keeps the longer description
        assert 'Longer' in deduped[0]['description']

    def test_same_location_different_category_not_deduped(self) -> None:
        reviews = {
            'r1': {
                'issues': [
                    {'location': 'a.py:1', 'category': 'bug', 'description': 'Bug.'},
                    {'location': 'a.py:1', 'category': 'style', 'description': 'Style.'},
                ]
            },
        }
        deduped = _deduplicate_issues(reviews)
        assert len(deduped) == 2

    def test_empty_reviews(self) -> None:
        assert _deduplicate_issues({}) == []

    def test_reviews_with_no_issues(self) -> None:
        reviews = {'r1': {'issues': []}}
        assert _deduplicate_issues(reviews) == []

    def test_reviewer_name_preserved(self) -> None:
        reviews = {
            'test_analyst': {
                'issues': [
                    {'location': 'a.py:1', 'category': 'bug', 'description': 'X'},
                ]
            },
        }
        deduped = _deduplicate_issues(reviews)
        assert deduped[0]['reviewer'] == 'test_analyst'


class TestScoringMetrics:
    """Test metric computation with manually constructed ScoringResult data."""

    def test_perfect_recall(self) -> None:
        # 3/3 ground truth matched
        result = ScoringResult(
            variant_name='test',
            diff_id='d1',
            matches=[
                IssueMatch(reviewer_issue={}, ground_truth_id='gt1', match_confidence=0.9, match_reasoning=''),
                IssueMatch(reviewer_issue={}, ground_truth_id='gt2', match_confidence=0.8, match_reasoning=''),
                IssueMatch(reviewer_issue={}, ground_truth_id='gt3', match_confidence=0.7, match_reasoning=''),
            ],
            unmatched_gt=[],
            false_positives=[],
            recall=1.0,
            precision=1.0,
            f1=1.0,
            blocking_recall=1.0,
        )
        assert result.recall == 1.0
        assert result.f1 == 1.0

    def test_zero_recall(self) -> None:
        result = ScoringResult(
            variant_name='test',
            diff_id='d1',
            matches=[],
            unmatched_gt=['gt1', 'gt2'],
            false_positives=[{'description': 'FP'}],
            recall=0.0,
            precision=0.0,
            f1=0.0,
            blocking_recall=0.0,
        )
        assert result.recall == 0.0
        assert result.f1 == 0.0

    def test_partial_metrics(self) -> None:
        # 2/4 GT matched, 2/3 findings are true positives
        recall = 2 / 4
        precision = 2 / 3
        f1 = 2 * precision * recall / (precision + recall)
        result = ScoringResult(
            variant_name='test',
            diff_id='d1',
            matches=[
                IssueMatch(reviewer_issue={}, ground_truth_id='gt1', match_confidence=0.9, match_reasoning=''),
                IssueMatch(reviewer_issue={}, ground_truth_id='gt2', match_confidence=0.8, match_reasoning=''),
            ],
            unmatched_gt=['gt3', 'gt4'],
            false_positives=[{'description': 'FP'}],
            recall=round(recall, 4),
            precision=round(precision, 4),
            f1=round(f1, 4),
            blocking_recall=0.5,
        )
        assert result.recall == pytest.approx(0.5)
        assert result.precision == pytest.approx(0.6667, abs=0.001)
        assert result.f1 > 0


def _make_gt(gt_id: str = 'gt1') -> GroundTruthIssue:
    return GroundTruthIssue(
        id=gt_id,
        location='a.py:1',
        category='bug',
        severity='blocking',
        description='A bug',
        mutation_type='logic',
    )


_ONE_ISSUE = [{'location': 'a.py:1', 'category': 'bug', 'description': 'x'}]


class TestMatchIssues:
    """match_issues is a Scored MatchedIssues with its cost, or an Unmeasured."""

    @pytest.mark.asyncio
    async def test_parseable_answer_is_scored_with_its_cost(self) -> None:
        with patch('orchestrator.evals.reviewer_trial.scorer.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = answered({'matches': []}, cost_usd=0.42)
            outcome = await match_issues(
                reviewer_issues=_ONE_ISSUE,
                ground_truth=[_make_gt('gt1')],
                diff_text='diff',
            )

        assert isinstance(outcome, Scored)
        assert outcome.value == MatchedIssues(matches=[], false_positives=_ONE_ISSUE)
        assert outcome.cost_usd == pytest.approx(0.42)

    @pytest.mark.asyncio
    async def test_a_valid_match_is_kept_and_its_finding_is_not_a_false_positive(self) -> None:
        answer = {'matches': [{
            'reviewer_issue_index': 0, 'ground_truth_id': 'gt1',
            'confidence': 0.9, 'reasoning': 'same bug',
        }]}
        with patch('orchestrator.evals.reviewer_trial.scorer.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = answered(answer, cost_usd=0.1)
            outcome = await match_issues(
                reviewer_issues=_ONE_ISSUE,
                ground_truth=[_make_gt('gt1')],
                diff_text='diff',
            )

        assert isinstance(outcome, Scored)
        assert outcome.value.matches == [IssueMatch(
            reviewer_issue=_ONE_ISSUE[0], ground_truth_id='gt1',
            match_confidence=0.9, match_reasoning='same bug',
        )]
        assert outcome.value.false_positives == []

    @pytest.mark.asyncio
    async def test_empty_inputs_are_scored_without_invoking(self) -> None:
        with patch('orchestrator.evals.reviewer_trial.scorer.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            no_findings = await match_issues(
                reviewer_issues=[],
                ground_truth=[_make_gt('gt1')],
                diff_text='diff',
            )
            no_ground_truth = await match_issues(
                reviewer_issues=_ONE_ISSUE,
                ground_truth=[],
                diff_text='diff',
            )
            mock_invoke.assert_not_called()

        assert no_findings == Scored(MatchedIssues(matches=[], false_positives=[]), cost_usd=0.0)
        assert no_ground_truth == Scored(
            MatchedIssues(matches=[], false_positives=_ONE_ISSUE), cost_usd=0.0,
        )

    @pytest.mark.asyncio
    async def test_unparseable_output_still_reports_cost(self) -> None:
        """PRD D1: an unparseable matcher answer is Unmeasured carrying its real spend."""
        fake_result = AgentResult(
            success=True, output='not json at all', cost_usd=0.42, structured_output=None,
        )

        with patch('orchestrator.evals.reviewer_trial.scorer.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = fake_result
            outcome = await match_issues(
                reviewer_issues=_ONE_ISSUE,
                ground_truth=[_make_gt('gt1')],
                diff_text='diff',
            )

        assert not isinstance(outcome, tuple)
        assert isinstance(outcome, Unmeasured)
        assert outcome.stage is MeasurementStage.MATCHER
        assert outcome.cause is UnmeasuredCause.UNPARSEABLE
        assert outcome.cost_usd == pytest.approx(0.42)

    @pytest.mark.parametrize('output', [
        '{}',
        json.dumps({'matches': 'x'}),
        json.dumps({'matches': ['not-an-object']}),
    ])
    @pytest.mark.asyncio
    async def test_an_answer_without_a_matches_list_is_unparseable(self, output: str) -> None:
        fake_result = AgentResult(success=True, output=output, cost_usd=0.2)

        with patch('orchestrator.evals.reviewer_trial.scorer.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = fake_result
            outcome = await match_issues(
                reviewer_issues=_ONE_ISSUE,
                ground_truth=[_make_gt('gt1')],
                diff_text='diff',
            )

        assert isinstance(outcome, Unmeasured)
        assert outcome.cause is UnmeasuredCause.UNPARSEABLE


class TestScoringResultMatchCost:
    """ScoringResult should carry a match_cost_usd field with default 0.0."""

    def test_default_match_cost_is_zero(self) -> None:
        result = ScoringResult(variant_name='v', diff_id='d')
        assert result.match_cost_usd == 0.0
        assert result.cost_usd == 0.0

    def test_total_cost_usd_property_sums_panel_and_matcher(self) -> None:
        """total_cost_usd property returns cost_usd + match_cost_usd."""
        result = ScoringResult(variant_name='v', diff_id='d', cost_usd=1.25, match_cost_usd=0.37)
        assert result.total_cost_usd == pytest.approx(1.25 + 0.37)

    def test_total_cost_usd_defaults_to_zero(self) -> None:
        """total_cost_usd is 0.0 when both components are at their defaults."""
        result = ScoringResult(variant_name='v', diff_id='d')
        assert result.total_cost_usd == pytest.approx(0.0)


class TestScorePanelRunCost:
    """score_panel_run should populate match_cost_usd from the matcher."""

    @pytest.mark.asyncio
    async def test_match_cost_populated_from_matcher(self) -> None:
        """score_panel_run sets cost_usd=panel cost and match_cost_usd=matcher cost."""
        diff = CorpusDiff(
            diff_id='d1',
            language='python',
            source='synthetic',
            diff_text='--- a/f.py\n+++ b/f.py',
            description='Test diff',
            ground_truth=[],
        )
        run = PanelRunResult(
            variant_name='v1',
            diff_id='d1',
            total_cost_usd=1.25,
            reviews={},
            wall_clock_ms=100,
        )

        with patch('orchestrator.evals.reviewer_trial.scorer.match_issues', new_callable=AsyncMock) as mock_match:
            mock_match.return_value = Scored(MatchedIssues([], []), cost_usd=0.37)
            score = await score_panel_run(run, diff)

        assert score.cost_usd == pytest.approx(1.25)
        assert score.match_cost_usd == pytest.approx(0.37)

    @pytest.mark.asyncio
    async def test_match_cost_and_recall_both_flow_with_nonempty_inputs(self) -> None:
        """score_panel_run propagates non-zero recall/precision alongside match_cost_usd.

        Integration guard: verifies that both cost legs AND the metric pipeline
        fire correctly when reviews + ground_truth are non-empty.
        """
        gt_issue = _make_gt('gt1')
        diff = CorpusDiff(
            diff_id='d2',
            language='python',
            source='synthetic',
            diff_text='--- a/f.py\n+++ b/f.py',
            description='Test diff with GT',
            ground_truth=[gt_issue],
        )
        reviewer_issue = {'location': 'a.py:1', 'category': 'bug', 'description': 'A bug'}
        run = PanelRunResult(
            variant_name='v1',
            diff_id='d2',
            total_cost_usd=2.00,
            reviews={
                'r1': {
                    'verdict': 'NEEDS_CHANGES',
                    'issues': [reviewer_issue],
                }
            },
            wall_clock_ms=200,
        )

        matched = IssueMatch(
            reviewer_issue=reviewer_issue,
            ground_truth_id='gt1',
            match_confidence=0.9,
            match_reasoning='Same location and category.',
        )

        with patch('orchestrator.evals.reviewer_trial.scorer.match_issues', new_callable=AsyncMock) as mock_match:
            # Return one match, no false positives, and a non-zero matcher cost
            mock_match.return_value = Scored(MatchedIssues([matched], []), cost_usd=0.25)
            score = await score_panel_run(run, diff)

        assert score.cost_usd == pytest.approx(2.00)
        assert score.match_cost_usd == pytest.approx(0.25)
        # 1 GT found out of 1 → recall == 1.0
        assert score.recall == pytest.approx(1.0)
        # 1 true positive out of 1 finding → precision == 1.0
        assert score.precision == pytest.approx(1.0)
        assert score.f1 == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_score_panel_run_threads_its_gate_to_the_matcher(self) -> None:
        diff = CorpusDiff(
            diff_id='d3', language='python', source='synthetic',
            diff_text='diff', description='d', ground_truth=[_make_gt('gt1')],
        )
        run = PanelRunResult(
            variant_name='v1', diff_id='d3', total_cost_usd=1.0,
            reviews={'r1': {'verdict': 'NEEDS_CHANGES', 'issues': _ONE_ISSUE}},
        )
        calls: list[dict] = []

        async def recording_invoke(**kwargs: object) -> AgentResult:
            calls.append(kwargs)
            return answered({'matches': []})

        gate = build_usage_gate(
            [AccountConfig(name='max-a', oauth_token_env='CLAUDE_OAUTH_A')], ['tok-a'],
        )
        with patch('orchestrator.evals.reviewer_trial.scorer.invoke_agent', recording_invoke):
            await score_panel_run(run, diff, usage_gate=gate)
            await score_panel_run(run, diff)

        gated, ungated = calls
        assert gated['oauth_token'] == 'tok-a'
        assert ungated.get('oauth_token') is None


class TestMatcherIsolation:
    """The haiku matcher runs at the project root under bypassPermissions;
    it must deny the built-ins and scope MCP to an empty server set so the
    project .mcp.json is never ambient-merged (allowed_tools=[] alone does
    not restrict under bypassPermissions)."""

    @pytest.mark.asyncio
    async def test_matcher_denies_tools_and_strict_scopes_mcp(self) -> None:
        with patch('orchestrator.evals.reviewer_trial.scorer.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = answered({'matches': []})
            await match_issues(
                reviewer_issues=[{'location': 'a.py:1', 'category': 'bug', 'description': 'x'}],
                ground_truth=[_make_gt('gt1')],
                diff_text='diff',
            )
        kwargs = mock_invoke.call_args.kwargs
        assert kwargs['disallowed_tools'] == ['*']
        assert kwargs['strict_mcp_config'] is True
        assert kwargs['mcp_config'] == {'mcpServers': {}}
        assert kwargs['mcp_config'], 'must stay truthy or --strict-mcp-config is never emitted'
