"""Tests for reviewer trial runner — mock invoke_agent, verify structure.

Post-2484/2493: the trial reviewer emits its verdict via the LIVE
verdict-tools transport (the ``submit_review_verdict`` MCP tool), not
``output_schema``/JSON parsing — mirrors
``workflow.TaskWorkflow._run_reviewer``'s read+fail-safe contract. Tests in
``TestRunSingleReviewerLiveTransport`` stub ``invoke_agent`` with a side
effect that writes a verdict envelope into the injected ``meta_root`` (via
``TaskArtifacts.write_verdict``), simulating the agent-side
``submit_review_verdict`` tool call, and independently assert the call was
wired for the live transport (a ``verdict-tools`` MCP server, no
``output_schema``).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from orchestrator.artifacts import TaskArtifacts
from orchestrator.evals.reviewer_trial.corpus import CorpusDiff
from orchestrator.evals.reviewer_trial.runner import (
    REVIEW_SCHEMA,
    PanelRunResult,
    _build_reviewer_prompt,
    _run_single_reviewer,
    run_panel,
)
from orchestrator.evals.reviewer_trial.variants import (
    ReviewerSpec,
    VariantConfig,
)


def _make_review(name: str = 'test', verdict: str = 'PASS') -> dict:
    return {
        'reviewer': name,
        'verdict': verdict,
        'issues': [],
        'summary': 'All good.',
    }


def _make_result(output: str = '', structured: dict | None = None, success: bool = True):
    """Build a mock AgentResult."""
    from types import SimpleNamespace
    return SimpleNamespace(
        success=success,
        output=output,
        cost_usd=0.50,
        duration_ms=5000,
        turns=5,
        session_id='mock-session',
        structured_output=structured,
        stderr='',
    )


def _parse_verdict_tools_target(mcp_config: dict) -> tuple[Path, str]:
    """Extract ``(meta_root, verdict_role)`` from a verdict-tools mcp_config,
    mirroring how the real ``orchestrator.mcp.verdict_tools`` CLI parses its
    own argv (``--meta-root <path> ... --verdict-role <role>``) — lets a
    stub discover WHERE to write regardless of whether ``meta_root`` was
    explicitly injected or auto-generated (``tempfile.mkdtemp()``).
    """
    args = mcp_config['mcpServers']['verdict-tools']['args']
    meta_root = Path(args[args.index('--meta-root') + 1])
    role = args[args.index('--verdict-role') + 1]
    return meta_root, role


def _make_verdict_writing_invoke(verdict: str = 'PASS', success: bool = True, write: bool = True):
    """Build an ``invoke_agent`` side effect that simulates a live reviewer:
    asserts the call was wired for the verdict-tools transport (a
    ``verdict-tools`` MCP server, no ``output_schema``), then — unless
    *write* is False — writes a verdict envelope to the invocation's own
    ``--meta-root`` (via ``TaskArtifacts.write_verdict``, exactly like the
    real ``submit_review_verdict`` tool), before returning a mock
    ``AgentResult`` with the given *success*.
    """
    async def _invoke(**kwargs):
        mcp_config = kwargs.get('mcp_config')
        assert mcp_config is not None, 'invoke_agent must be called with a mcp_config'
        assert 'verdict-tools' in mcp_config.get('mcpServers', {}), (
            'mcp_config must wire a verdict-tools MCP server (live transport)'
        )
        assert kwargs.get('output_schema') is None, (
            'the live transport must NOT pass output_schema — the agent emits '
            'via submit_review_verdict, not structured output capture'
        )
        if write:
            meta_root, role = _parse_verdict_tools_target(mcp_config)
            payload = {'reviewer': role, 'verdict': verdict, 'issues': [], 'summary': 'Trial verdict.'}
            envelope = {
                'role': role,
                'schema_version': 1,
                'session_id': 'mock-session',
                'emitted_at': '2026-07-17T00:00:00+00:00',
                'verdict': payload,
            }
            TaskArtifacts(kwargs['cwd'], meta_root=meta_root).write_verdict(role, envelope)
        return _make_result(success=success)
    return _invoke


class TestBuildReviewerPrompt:
    def test_contains_diff(self) -> None:
        prompt = _build_reviewer_prompt('--- a/f.py\n+++ b/f.py')
        assert '--- a/f.py' in prompt
        assert '```diff' in prompt

    def test_truncates_long_diff(self) -> None:
        long_diff = 'x' * 60_000
        prompt = _build_reviewer_prompt(long_diff)
        assert '... [diff truncated] ...' in prompt
        assert len(prompt) < 55_000

    def test_action_instruction(self) -> None:
        prompt = _build_reviewer_prompt('diff')
        assert 'Review the diff' in prompt

    def test_no_json_output_instruction(self) -> None:
        """Post-2493: the reviewer's SYSTEM prompt (roles.build_reviewer_prompt_spec)
        instructs it to call submit_review_verdict; this action prompt must no
        longer also tell it to "output pure JSON" — that would contradict the
        frozen CONTRACT and leave the agent unsure which channel to use.
        """
        prompt = _build_reviewer_prompt('diff')
        assert 'Output your review as pure JSON' not in prompt


class TestReviewSchema:
    """REVIEW_SCHEMA is retained as the documented shape of a verdict's
    payload (task 2493) even though it's no longer passed to invoke_agent
    as output_schema.
    """

    def test_required_fields(self) -> None:
        assert set(REVIEW_SCHEMA['required']) == {'reviewer', 'verdict', 'issues', 'summary'}

    def test_verdict_enum(self) -> None:
        verdict_props = REVIEW_SCHEMA['properties']['verdict']
        assert set(verdict_props['enum']) == {'PASS', 'ISSUES_FOUND'}


class TestPanelRunResult:
    def test_defaults(self) -> None:
        result = PanelRunResult(variant_name='test', diff_id='d1')
        assert result.reviews == {}
        assert result.total_cost_usd == 0.0
        assert result.errors == []


class TestRunSingleReviewerLiveTransport:
    """(b)/(c): _run_single_reviewer emits via the live verdict-tools
    transport and reads its result back from the verdict artifact, with a
    fail-safe ERROR disposition when no valid verdict was produced —
    mirrors workflow.TaskWorkflow._run_reviewer's read+fail-safe contract.
    """

    @pytest.mark.asyncio
    async def test_valid_verdict_returns_review_keyed_by_spec_name(self, tmp_path) -> None:
        spec = ReviewerSpec(name='r1', model='sonnet', specialization='Testing.')
        diff = CorpusDiff(
            diff_id='d1', language='python', source='synthetic',
            diff_text='diff', description='Test', ground_truth=[],
        )

        with patch(
            'orchestrator.evals.reviewer_trial.runner.invoke_agent',
            side_effect=_make_verdict_writing_invoke(verdict='PASS'),
        ):
            name, review, cost = await _run_single_reviewer(spec, diff, meta_root=tmp_path)

        assert name == 'r1'
        assert review is not None
        assert review['verdict'] in {'PASS', 'ISSUES_FOUND'}
        assert review['reviewer'] == 'reviewer_r1'
        assert cost == 0.50
        # The verdict path is directly observable at the injected meta_root.
        assert (tmp_path / 'verdicts' / 'reviewer_r1.json').exists()

    @pytest.mark.asyncio
    async def test_issues_found_verdict_also_accepted(self, tmp_path) -> None:
        spec = ReviewerSpec(name='r1', model='sonnet', specialization='Testing.')
        diff = CorpusDiff(
            diff_id='d1', language='python', source='synthetic',
            diff_text='diff', description='Test', ground_truth=[],
        )

        with patch(
            'orchestrator.evals.reviewer_trial.runner.invoke_agent',
            side_effect=_make_verdict_writing_invoke(verdict='ISSUES_FOUND'),
        ):
            _name, review, _cost = await _run_single_reviewer(spec, diff, meta_root=tmp_path)

        assert review['verdict'] == 'ISSUES_FOUND'

    @pytest.mark.asyncio
    async def test_missing_verdict_yields_error_disposition(self, tmp_path) -> None:
        """A stub that writes NO verdict (agent never called
        submit_review_verdict) must degrade to the {verdict: 'ERROR'}
        disposition rather than raise or hang.
        """
        spec = ReviewerSpec(name='r1', model='sonnet', specialization='Testing.')
        diff = CorpusDiff(
            diff_id='d1', language='python', source='synthetic',
            diff_text='diff', description='Test', ground_truth=[],
        )

        with patch(
            'orchestrator.evals.reviewer_trial.runner.invoke_agent',
            side_effect=_make_verdict_writing_invoke(write=False),
        ):
            name, review, _cost = await _run_single_reviewer(spec, diff, max_retries=1, meta_root=tmp_path)

        assert name == 'r1'
        assert review is not None
        assert review['verdict'] == 'ERROR'

    @pytest.mark.asyncio
    async def test_unsuccessful_result_yields_error_even_if_verdict_written(self, tmp_path) -> None:
        """An invocation failure (crash / max_turns / budget exhaustion) is
        untrusted even if it happened to write a verdict before failing
        (mirrors workflow._run_reviewer's I-FAIL-SAFE handling).
        """
        spec = ReviewerSpec(name='r1', model='sonnet', specialization='Testing.')
        diff = CorpusDiff(
            diff_id='d1', language='python', source='synthetic',
            diff_text='diff', description='Test', ground_truth=[],
        )

        with patch(
            'orchestrator.evals.reviewer_trial.runner.invoke_agent',
            side_effect=_make_verdict_writing_invoke(verdict='PASS', success=False),
        ):
            _name, review, _cost = await _run_single_reviewer(spec, diff, max_retries=1, meta_root=tmp_path)

        assert review['verdict'] == 'ERROR'

    @pytest.mark.asyncio
    async def test_defaults_to_fresh_meta_root_when_uninjected(self) -> None:
        """Production never injects meta_root — _run_single_reviewer must
        still work, generating its own fresh temp dir per invocation.
        """
        spec = ReviewerSpec(name='r1', model='sonnet', specialization='Testing.')
        diff = CorpusDiff(
            diff_id='d1', language='python', source='synthetic',
            diff_text='diff', description='Test', ground_truth=[],
        )

        with patch(
            'orchestrator.evals.reviewer_trial.runner.invoke_agent',
            side_effect=_make_verdict_writing_invoke(verdict='PASS'),
        ):
            name, review, _cost = await _run_single_reviewer(spec, diff)

        assert name == 'r1'
        assert review['verdict'] == 'PASS'


class TestRunPanel:
    @pytest.mark.asyncio
    async def test_single_reviewer_live_verdict(self) -> None:
        variant = VariantConfig(
            name='test_variant',
            description='Test',
            reviewers=[
                ReviewerSpec(name='r1', model='sonnet', specialization='Testing.'),
            ],
        )
        diff = CorpusDiff(
            diff_id='d1',
            language='python',
            source='synthetic',
            diff_text='--- a/f.py\n+++ b/f.py',
            description='Test',
            ground_truth=[],
        )

        with patch(
            'orchestrator.evals.reviewer_trial.runner.invoke_agent',
            side_effect=_make_verdict_writing_invoke(verdict='PASS'),
        ):
            result = await run_panel(variant, diff, stagger_secs=0)

        assert result.variant_name == 'test_variant'
        assert result.diff_id == 'd1'
        assert 'r1' in result.reviews
        assert result.reviews['r1']['verdict'] == 'PASS'
        assert result.total_cost_usd == 0.50
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_multiple_reviewers(self) -> None:
        variant = VariantConfig(
            name='multi',
            description='Test',
            reviewers=[
                ReviewerSpec(name='r1', model='sonnet', specialization='S1.'),
                ReviewerSpec(name='r2', model='sonnet', specialization='S2.'),
            ],
        )
        diff = CorpusDiff(
            diff_id='d1',
            language='python',
            source='synthetic',
            diff_text='diff',
            description='Test',
            ground_truth=[],
        )

        with patch(
            'orchestrator.evals.reviewer_trial.runner.invoke_agent',
            side_effect=_make_verdict_writing_invoke(verdict='PASS'),
        ):
            result = await run_panel(variant, diff, stagger_secs=0)

        assert len(result.reviews) == 2
        assert result.total_cost_usd == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_reviewer_error_recorded(self) -> None:
        variant = VariantConfig(
            name='error_test',
            description='Test',
            reviewers=[
                ReviewerSpec(name='r1', model='sonnet', specialization='S.'),
            ],
        )
        diff = CorpusDiff(
            diff_id='d1',
            language='python',
            source='synthetic',
            diff_text='diff',
            description='Test',
            ground_truth=[],
        )

        with patch(
            'orchestrator.evals.reviewer_trial.runner.invoke_agent',
            new_callable=AsyncMock,
            side_effect=RuntimeError('API error'),
        ):
            result = await run_panel(variant, diff, stagger_secs=0, max_retries=1)

        assert len(result.errors) == 1
        assert 'r1' not in result.reviews

    @pytest.mark.asyncio
    async def test_missing_verdict_recorded_as_error_review_not_absence(self) -> None:
        """A reviewer that never calls submit_review_verdict still produces
        a (non-None) {verdict: 'ERROR'} review — mirrors production's
        _run_reviewer, which always writes SOME review (never drops the
        reviewer from the aggregation entirely).
        """
        variant = VariantConfig(
            name='no_verdict_test',
            description='Test',
            reviewers=[
                ReviewerSpec(name='r1', model='sonnet', specialization='S.'),
            ],
        )
        diff = CorpusDiff(
            diff_id='d1',
            language='python',
            source='synthetic',
            diff_text='diff',
            description='Test',
            ground_truth=[],
        )

        with patch(
            'orchestrator.evals.reviewer_trial.runner.invoke_agent',
            side_effect=_make_verdict_writing_invoke(write=False),
        ):
            result = await run_panel(variant, diff, stagger_secs=0, max_retries=1)

        assert 'r1' in result.reviews
        assert result.reviews['r1']['verdict'] == 'ERROR'
        assert result.errors == []


class TestRunnerCrossFamilyThreading:
    """runner threads spec.backend/env_overrides + os.environ-resolved
    oauth_token + a prices table into invoke_agent (task 2476)."""

    @pytest.mark.asyncio
    async def test_cross_family_kwargs_threaded(self, monkeypatch) -> None:
        monkeypatch.setenv('TOK', 'secret')
        prices = {'gpt-5.4': {'input_per_1m': 1.0, 'output_per_1m': 2.0}}

        variant = VariantConfig(
            name='xfam_variant',
            description='Test',
            reviewers=[
                ReviewerSpec(
                    name='r1', model='gpt-5.4', specialization='Testing.',
                    backend='codex',
                    env_overrides={'ANTHROPIC_BASE_URL': 'https://x'},
                    oauth_token_env='TOK',
                ),
            ],
        )
        diff = CorpusDiff(
            diff_id='d1', language='python', source='synthetic',
            diff_text='diff', description='Test', ground_truth=[],
        )

        with patch('orchestrator.evals.reviewer_trial.runner.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = _make_result(structured=_make_review('r1'))
            await run_panel(variant, diff, stagger_secs=0, prices=prices)

        mock_invoke.assert_awaited_once()
        kwargs = mock_invoke.call_args.kwargs
        assert kwargs['backend'] == 'codex'
        assert kwargs['env_overrides'] == {'ANTHROPIC_BASE_URL': 'https://x'}
        assert kwargs['oauth_token'] == 'secret'
        assert kwargs['prices'] == prices

    @pytest.mark.asyncio
    async def test_default_claude_backend_no_token_no_prices(self) -> None:
        variant = VariantConfig(
            name='claude_variant',
            description='Test',
            reviewers=[
                ReviewerSpec(name='r1', model='sonnet', specialization='Testing.'),
            ],
        )
        diff = CorpusDiff(
            diff_id='d1', language='python', source='synthetic',
            diff_text='diff', description='Test', ground_truth=[],
        )

        with patch('orchestrator.evals.reviewer_trial.runner.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = _make_result(structured=_make_review('r1'))
            await run_panel(variant, diff, stagger_secs=0)

        mock_invoke.assert_awaited_once()
        kwargs = mock_invoke.call_args.kwargs
        assert kwargs['backend'] == 'claude'
        assert kwargs['oauth_token'] is None
        assert kwargs.get('prices') is None

    @pytest.mark.asyncio
    async def test_missing_oauth_token_env_fails_loudly(self, monkeypatch) -> None:
        """A spec naming oauth_token_env whose var is unset must fail loudly
        (and never dispatch), not silently resolve oauth_token=None and let
        the failure surface as a confusing downstream auth error (task 2476
        review amendment)."""
        monkeypatch.delenv('MISSING_TOK', raising=False)

        variant = VariantConfig(
            name='xfam_missing_token',
            description='Test',
            reviewers=[
                ReviewerSpec(
                    name='r1', model='gpt-5.4', specialization='Testing.',
                    backend='codex',
                    oauth_token_env='MISSING_TOK',
                ),
            ],
        )
        diff = CorpusDiff(
            diff_id='d1', language='python', source='synthetic',
            diff_text='diff', description='Test', ground_truth=[],
        )

        with patch('orchestrator.evals.reviewer_trial.runner.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            result = await run_panel(variant, diff, stagger_secs=0, max_retries=1)

        mock_invoke.assert_not_awaited()
        assert 'r1' not in result.reviews
        assert len(result.errors) == 1
        assert 'MISSING_TOK' in result.errors[0]
