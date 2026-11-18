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

import contextlib
import json
import logging
from collections.abc import Iterator, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest
from _eval_refusal import answered, refused_agent_result
from _orch_helpers import build_usage_gate
from shared.cli_invoke import AgentResult
from shared.config_models import AccountConfig
from shared.usage_gate import UsageGate
from test_reviewer_trial_checkout import _CHANGE_AT_BASE, _commit, _git, _Source

from orchestrator.agents.briefing import BriefingAssembler
from orchestrator.agents.review_briefing import ReviewBriefing
from orchestrator.agents.roles import build_reviewer_role
from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import OrchestratorConfig, default_price_table, project_review_config
from orchestrator.evals.reviewer_trial.checkout import (
    FixtureEnrichmentMissing,
    fixture_checkout,
)
from orchestrator.evals.reviewer_trial.corpus import (
    CorpusDiff,
    CorpusManifest,
    FixtureContext,
    FixtureTaskRecord,
    GroundTruthIssue,
)
from orchestrator.evals.reviewer_trial.runner import (
    DIFF_CAP_CHARS,
    REVIEW_SCHEMA,
    PanelRunKey,
    PanelRunResult,
    _run_single_reviewer,
    run_and_score,
    run_panel,
    run_panel_cached,
    run_trial,
)
from orchestrator.evals.reviewer_trial.scorer import ScoringResult
from orchestrator.evals.reviewer_trial.variants import (
    ReviewerSpec,
    VariantConfig,
    with_briefing,
)
from orchestrator.review_inputs import gather_review_subject
from orchestrator.review_payload import Verdict, normalize_review


@pytest.fixture
def source(tmp_path: Path) -> _Source:
    """A real source repository every panel's scratch clone is cut from."""
    return _Source(tmp_path / 'source')


def _diff_in(
    source: _Source,
    *,
    diff_id: str = 'd1',
    diff_text: str = 'diff',
    context: FixtureContext | None = None,
) -> CorpusDiff:
    return CorpusDiff(
        diff_id=diff_id, language='python', source='synthetic',
        diff_text=diff_text, description='Test', ground_truth=[],
        cwd=source.root, context=context,
    )


def _enriched(
    source: _Source,
    *,
    diff_id: str = 'fx_1',
    base_sha: str | None,
    base_sha_reason: str | None = None,
    plan: dict | None = None,
    plan_reason: str | None = None,
) -> CorpusDiff:
    return _diff_in(
        source,
        diff_id=diff_id,
        diff_text=_CHANGE_AT_BASE,
        context=FixtureContext(
            task=FixtureTaskRecord(
                id='77',
                title='Raise x to ten and add the n module',
                description='x must read 10, and a new module n carries the companion value.',
                test_strategy='Import a and assert x == 10; import new and assert n == 1.',
            ),
            changed_files=['a.py', 'new.py'],
            base_sha=base_sha,
            base_sha_reason=base_sha_reason,
            plan=plan,
            plan_reason=plan_reason,
        ),
    )


def _one_reviewer(name: str = 'r1_panel') -> VariantConfig:
    return VariantConfig(
        name=name, description='Test',
        reviewers=[ReviewerSpec(name='r1', model='sonnet', specialization='Testing.')],
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


def _make_verdict_writing_invoke(
    verdict: str = 'PASS', success: bool = True, write: bool = True, issues: Sequence[dict] = (),
):
    """Build an ``invoke_agent`` side effect that simulates a live reviewer:
    asserts the call was wired for the verdict-tools transport (a
    ``verdict-tools`` MCP server, no ``output_schema``), then — unless
    *write* is False — writes a verdict envelope to the invocation's own
    ``--meta-root`` (via ``TaskArtifacts.write_verdict``, exactly like the
    real ``submit_review_verdict`` tool) carrying *issues*, before returning a
    mock ``AgentResult`` with the given *success*.
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
            payload = {'reviewer': role, 'verdict': verdict, 'issues': list(issues), 'summary': 'Trial verdict.'}
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


def _make_retry_then_succeed_invoke():
    """Build an ``invoke_agent`` side effect that, on its FIRST call, invokes
    successfully but writes NO verdict (forcing a retry), then on the second
    call writes a valid PASS verdict. Both calls report a real ``cost_usd``,
    so a correct runner must return the SUMMED spend across both attempts
    (task 2493 amendment) rather than only the final attempt's cost.
    """
    calls = {'n': 0}
    no_verdict = _make_verdict_writing_invoke(write=False)
    with_verdict = _make_verdict_writing_invoke(verdict='PASS')

    async def _invoke(**kwargs):
        calls['n'] += 1
        delegate = no_verdict if calls['n'] == 1 else with_verdict
        return await delegate(**kwargs)

    return _invoke


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


def _full_panel_result(run_key: PanelRunKey | None) -> PanelRunResult:
    return PanelRunResult(
        variant_name='v1',
        diff_id='d1',
        reviews={'r1': _make_review('r1'), 'r2': _make_review('r2', 'ISSUES_FOUND')},
        total_cost_usd=1.25,
        wall_clock_ms=4321,
        errors=['r3: no result'],
        run_key=run_key,
    )


class TestPanelRunResultFromDict:
    @pytest.mark.parametrize('run_key', [
        PanelRunKey.of(prices=default_price_table(), usage_gate=None, diff_cap_chars=0),
        None,
    ])
    def test_round_trips_asdict(self, run_key: PanelRunKey | None) -> None:
        result = _full_panel_result(run_key)
        assert PanelRunResult.from_dict(asdict(result)) == result

    def test_a_legacy_record_loads_with_defaults_and_no_key(self) -> None:
        loaded = PanelRunResult.from_dict({'variant_name': 'v1', 'diff_id': 'd1'})
        assert loaded == PanelRunResult(variant_name='v1', diff_id='d1')
        assert loaded.run_key is None

    @pytest.mark.parametrize('missing', ['variant_name', 'diff_id'])
    def test_a_record_missing_its_identity_raises(self, missing: str) -> None:
        record = asdict(_full_panel_result(None))
        del record[missing]
        with pytest.raises(KeyError):
            PanelRunResult.from_dict(record)


class TestPanelRunKey:
    def test_equal_price_tables_give_equal_keys(self) -> None:
        first = PanelRunKey.of(prices=default_price_table(), usage_gate=None, diff_cap_chars=DIFF_CAP_CHARS)
        second = PanelRunKey.of(prices=default_price_table(), usage_gate=None, diff_cap_chars=DIFF_CAP_CHARS)
        assert first == second

    @staticmethod
    def _one_rate_changed() -> dict:
        prices = default_price_table()
        model = sorted(prices)[0]
        rate = sorted(prices[model])[0]
        prices[model][rate] += 1.0
        return prices

    @pytest.mark.parametrize('change', [
        pytest.param(lambda: {'prices': None}, id='unpriced'),
        pytest.param(lambda: {'prices': TestPanelRunKey._one_rate_changed()}, id='one-rate-changed'),
        pytest.param(lambda: {'usage_gate': cast(UsageGate, object())}, id='gated'),
        pytest.param(lambda: {'diff_cap_chars': 0}, id='cap-lifted'),
    ])
    def test_each_component_changes_the_key(self, change) -> None:
        baseline = {'prices': default_price_table(), 'usage_gate': None, 'diff_cap_chars': DIFF_CAP_CHARS}
        assert PanelRunKey.of(**{**baseline, **change()}) != PanelRunKey.of(**baseline)

    def test_round_trips_asdict(self) -> None:
        key = PanelRunKey.of(
            prices=default_price_table(), usage_gate=cast(UsageGate, object()), diff_cap_chars=0,
        )
        assert PanelRunKey.from_dict(asdict(key)) == key


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
        assert review.verdict is Verdict.PASS
        assert review.reviewer == 'reviewer_r1'
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

        assert review is not None
        assert review.verdict is Verdict.ISSUES_FOUND

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
        assert review.verdict is Verdict.ERROR
        assert review.issues == ()
        assert review.reviewer == 'reviewer_r1'

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

        assert review is not None
        assert review.verdict is Verdict.ERROR
        assert review.issues == ()
        assert review.reviewer == 'reviewer_r1'

    @pytest.mark.asyncio
    async def test_an_off_contract_inner_verdict_is_no_verdict(self, tmp_path) -> None:
        spec = ReviewerSpec(name='r1', model='sonnet', specialization='Testing.')
        diff = CorpusDiff(
            diff_id='d1', language='python', source='synthetic',
            diff_text='diff', description='Test', ground_truth=[],
        )

        with patch(
            'orchestrator.evals.reviewer_trial.runner.invoke_agent',
            side_effect=_make_verdict_writing_invoke(verdict='NEEDS_CHANGES'),
        ):
            name, review, _cost = await _run_single_reviewer(spec, diff, max_retries=1, meta_root=tmp_path)

        assert name == 'r1'
        assert review is not None
        assert review.verdict is Verdict.ERROR
        assert review.issues == ()
        assert review.reviewer == 'reviewer_r1'

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
        assert review is not None
        assert review.verdict is Verdict.PASS

    @pytest.mark.asyncio
    async def test_cost_accumulates_across_retries(self, tmp_path) -> None:
        """A verdict-less first attempt (which still cost money) followed by a
        successful retry must return the SUM of both attempts' cost, not just
        the last attempt's — otherwise run_panel under-reports trial spend
        (task 2493 amendment).
        """
        spec = ReviewerSpec(name='r1', model='sonnet', specialization='Testing.')
        diff = CorpusDiff(
            diff_id='d1', language='python', source='synthetic',
            diff_text='diff', description='Test', ground_truth=[],
        )

        with patch(
            'orchestrator.evals.reviewer_trial.runner.invoke_agent',
            side_effect=_make_retry_then_succeed_invoke(),
        ):
            name, review, cost = await _run_single_reviewer(
                spec, diff, max_retries=2, meta_root=tmp_path,
            )

        assert name == 'r1'
        assert review is not None
        assert review.verdict is Verdict.PASS
        # attempt 1 (no verdict → retried) + attempt 2 (valid) each cost 0.50;
        # the returned cost must sum BOTH, not report only the final 0.50.
        assert cost == pytest.approx(1.0)


class TestRunPanel:
    @pytest.mark.asyncio
    async def test_single_reviewer_live_verdict(self, source: _Source) -> None:
        variant = VariantConfig(
            name='test_variant',
            description='Test',
            reviewers=[
                ReviewerSpec(name='r1', model='sonnet', specialization='Testing.'),
            ],
        )
        diff = _diff_in(source)

        with patch(
            'orchestrator.evals.reviewer_trial.runner.invoke_agent',
            side_effect=_make_verdict_writing_invoke(verdict='PASS'),
        ):
            result = await run_panel(variant, diff, stagger_secs=0)

        assert result.variant_name == 'test_variant'
        assert result.diff_id == 'd1'
        assert 'r1' in result.reviews
        assert result.reviews['r1']['verdict'] == 'PASS'
        assert result.reviews['r1'] == normalize_review(result.reviews['r1']).to_record()
        assert result.total_cost_usd == 0.50
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_stamps_the_key_it_ran_under(self, source: _Source) -> None:
        with patch(
            'orchestrator.evals.reviewer_trial.runner.invoke_agent',
            side_effect=_make_verdict_writing_invoke(verdict='PASS'),
        ):
            result = await run_panel(_one_reviewer(), _diff_in(source), stagger_secs=0, diff_cap_chars=123)

        assert result.run_key == PanelRunKey.of(prices=None, usage_gate=None, diff_cap_chars=123)

    @pytest.mark.asyncio
    async def test_multiple_reviewers(self, source: _Source) -> None:
        variant = VariantConfig(
            name='multi',
            description='Test',
            reviewers=[
                ReviewerSpec(name='r1', model='sonnet', specialization='S1.'),
                ReviewerSpec(name='r2', model='sonnet', specialization='S2.'),
            ],
        )
        diff = _diff_in(source)

        with patch(
            'orchestrator.evals.reviewer_trial.runner.invoke_agent',
            side_effect=_make_verdict_writing_invoke(verdict='PASS'),
        ):
            result = await run_panel(variant, diff, stagger_secs=0)

        assert len(result.reviews) == 2
        assert result.total_cost_usd == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_reviewer_error_recorded(self, source: _Source) -> None:
        variant = VariantConfig(
            name='error_test',
            description='Test',
            reviewers=[
                ReviewerSpec(name='r1', model='sonnet', specialization='S.'),
            ],
        )
        diff = _diff_in(source)

        with patch(
            'orchestrator.evals.reviewer_trial.runner.invoke_agent',
            new_callable=AsyncMock,
            side_effect=RuntimeError('API error'),
        ):
            result = await run_panel(variant, diff, stagger_secs=0, max_retries=1)

        assert len(result.errors) == 1
        assert 'r1' not in result.reviews

    @pytest.mark.asyncio
    async def test_missing_verdict_recorded_as_error_review_not_absence(self, source: _Source) -> None:
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
        diff = _diff_in(source)

        with patch(
            'orchestrator.evals.reviewer_trial.runner.invoke_agent',
            side_effect=_make_verdict_writing_invoke(write=False),
        ):
            result = await run_panel(variant, diff, stagger_secs=0, max_retries=1)

        assert 'r1' in result.reviews
        assert result.reviews['r1']['verdict'] == 'ERROR'
        assert result.reviews['r1'] == normalize_review(result.reviews['r1']).to_record()
        assert result.errors == []


class TestRunnerCrossFamilyThreading:
    """runner threads spec.backend/env_overrides + os.environ-resolved
    oauth_token + a prices table into invoke_agent (task 2476)."""

    @pytest.mark.asyncio
    async def test_cross_family_kwargs_threaded(self, monkeypatch, source: _Source) -> None:
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
        diff = _diff_in(source)

        with patch('orchestrator.evals.reviewer_trial.runner.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.side_effect = _make_verdict_writing_invoke(verdict='PASS')
            await run_panel(variant, diff, stagger_secs=0, prices=prices)

        mock_invoke.assert_awaited_once()
        kwargs = mock_invoke.call_args.kwargs
        assert kwargs['backend'] == 'codex'
        assert kwargs['env_overrides'] == {'ANTHROPIC_BASE_URL': 'https://x'}
        assert kwargs['oauth_token'] == 'secret'
        assert kwargs['prices'] == prices

    @pytest.mark.asyncio
    async def test_default_claude_backend_no_token_no_prices(self, source: _Source) -> None:
        variant = VariantConfig(
            name='claude_variant',
            description='Test',
            reviewers=[
                ReviewerSpec(name='r1', model='sonnet', specialization='Testing.'),
            ],
        )
        diff = _diff_in(source)

        with patch('orchestrator.evals.reviewer_trial.runner.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.side_effect = _make_verdict_writing_invoke(verdict='PASS')
            await run_panel(variant, diff, stagger_secs=0)

        mock_invoke.assert_awaited_once()
        kwargs = mock_invoke.call_args.kwargs
        assert kwargs['backend'] == 'claude'
        assert kwargs['oauth_token'] is None
        assert kwargs.get('prices') is None

    @pytest.mark.asyncio
    async def test_missing_oauth_token_env_fails_loudly(self, monkeypatch, source: _Source) -> None:
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
        diff = _diff_in(source)

        with patch('orchestrator.evals.reviewer_trial.runner.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            result = await run_panel(variant, diff, stagger_secs=0, max_retries=1)

        mock_invoke.assert_not_awaited()
        assert 'r1' not in result.reviews
        assert len(result.errors) == 1
        assert 'MISSING_TOK' in result.errors[0]


class TestRunnerIsolation:
    """The trial runs under bypassPermissions at a cwd that carries the
    project .mcp.json (live escalation + fused-memory servers). Only
    ``--strict-mcp-config`` keeps the reviewer scoped to its verdict-tools
    server, so the runner must pass it and the argv it builds must carry it.
    """

    @staticmethod
    def _variant() -> VariantConfig:
        return VariantConfig(
            name='iso_variant', description='Test',
            reviewers=[ReviewerSpec(name='r1', model='opus', specialization='Testing.')],
        )

    @pytest.mark.asyncio
    async def test_strict_mcp_config_reaches_the_built_argv(self, source: _Source) -> None:
        from shared.cli_invoke import build_claude_argv

        with patch('orchestrator.evals.reviewer_trial.runner.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.side_effect = _make_verdict_writing_invoke(verdict='PASS')
            await run_panel(self._variant(), _diff_in(source), stagger_secs=0)

        kwargs = mock_invoke.call_args.kwargs
        assert kwargs['strict_mcp_config'] is True
        assert {'Edit', 'Write'} <= set(kwargs['disallowed_tools'])

        cmd, temp_files = build_claude_argv(
            model=kwargs['model'], max_budget_usd=kwargs['max_budget_usd'],
            system_prompt=kwargs['system_prompt'], max_turns=kwargs['max_turns'],
            permission_mode='bypassPermissions',
            allowed_tools=kwargs['allowed_tools'], disallowed_tools=kwargs['disallowed_tools'],
            mcp_config=kwargs['mcp_config'], output_schema=None, effort=kwargs['effort'],
            resume_session_id=None, session_id=None,
            strict_mcp_config=kwargs['strict_mcp_config'],
        )
        for f in temp_files:
            Path(f).unlink(missing_ok=True)
        assert '--strict-mcp-config' in cmd
        assert cmd.index('--strict-mcp-config') == cmd.index('--mcp-config') + 2

    @pytest.mark.asyncio
    async def test_usage_gate_routes_through_cap_retry(self, source: _Source) -> None:
        """With a gate the reviewer draws on the account pool via
        invoke_with_cap_retry (failover on cap hits) instead of the CLI's
        default login; the isolation kwarg rides along."""
        gate = cast(UsageGate, object())
        captured: dict = {}

        async def _cap_retry(usage_gate, label, *, invoke_fn, backend, **kwargs):
            captured.update(usage_gate=usage_gate, label=label, invoke_fn=invoke_fn, backend=backend, **kwargs)
            return await _make_verdict_writing_invoke(verdict='PASS')(**kwargs)

        with (
            patch('orchestrator.evals.reviewer_trial.runner.invoke_with_cap_retry', side_effect=_cap_retry),
            patch('orchestrator.evals.reviewer_trial.runner.invoke_agent', new_callable=AsyncMock) as mock_invoke,
        ):
            result = await run_panel(self._variant(), _diff_in(source), stagger_secs=0, usage_gate=gate)

        mock_invoke.assert_not_awaited()
        assert captured['usage_gate'] is gate
        assert captured['backend'] == 'claude'
        assert captured['strict_mcp_config'] is True
        assert 'oauth_token' not in captured
        assert result.reviews['r1']['verdict'] == 'PASS'


def _capturing_invoke(seen: list[dict]):
    """A live-transport reviewer stub that also records what the clone looked like."""
    write_verdict = _make_verdict_writing_invoke(verdict='PASS')

    async def _invoke(**kwargs):
        clone = Path(kwargs['cwd'])
        seen.append({**kwargs, 'head_parent': _git(clone, 'rev-parse', 'HEAD^')})
        return await write_verdict(**kwargs)

    return _invoke


class TestPanelBriefsLikeProduction:
    """Every arm is briefed by the production builders, in a scratch clone of the fixture."""

    @pytest.mark.asyncio
    async def test_a_diff_only_arm_gets_the_production_prompt_in_a_scratch_clone(
        self, source: _Source,
    ) -> None:
        variant = _one_reviewer()
        diff = _enriched(source, base_sha=source.base)
        seen: list[dict] = []

        with patch(
            'orchestrator.evals.reviewer_trial.runner.invoke_agent',
            side_effect=_capturing_invoke(seen),
        ):
            result = await run_panel(variant, diff, stagger_secs=0)

        [call] = seen
        assert call['prompt'] == await BriefingAssembler(OrchestratorConfig()).build_reviewer_prompt(
            'reviewer_r1', diff.diff_text, context='', diff_cap_chars=DIFF_CAP_CHARS,
        )
        clone = Path(call['cwd'])
        assert clone.resolve() != source.root.resolve()
        assert call['head_parent'] == source.base
        assert not clone.exists()
        assert result.reviews['r1']['verdict'] == 'PASS'

    @pytest.mark.parametrize('cap', [DIFF_CAP_CHARS, 0, 1_000])
    @pytest.mark.asyncio
    async def test_the_diff_cap_reaches_the_production_builder(
        self, source: _Source, cap: int,
    ) -> None:
        big = 'x' * 60_000
        diff = _diff_in(source, diff_text=big)

        with patch(
            'orchestrator.evals.reviewer_trial.runner.invoke_agent', new_callable=AsyncMock,
        ) as mock_invoke:
            mock_invoke.side_effect = _make_verdict_writing_invoke(verdict='PASS')
            await run_panel(_one_reviewer(), diff, stagger_secs=0, diff_cap_chars=cap)

        prompt = mock_invoke.call_args.kwargs['prompt']
        assert prompt == await BriefingAssembler(OrchestratorConfig()).build_reviewer_prompt(
            'reviewer_r1', big, context='', diff_cap_chars=cap,
        )
        assert ('[diff truncated]' in prompt) == (cap != 0)

    @pytest.mark.parametrize(
        ('plan', 'plan_reason'),
        [
            pytest.param({'title': 'Raise x', 'files': ['a.py', 'new.py']}, None, id='plan'),
            pytest.param(None, 'plan.json was reclaimed with its worktree', id='no-plan'),
        ],
    )
    @pytest.mark.asyncio
    async def test_a_d2_arm_gets_the_production_context_complete_prompt(
        self, source: _Source, plan: dict | None, plan_reason: str | None,
    ) -> None:
        variant = with_briefing(_one_reviewer(), ReviewBriefing.CONTEXT_COMPLETE)
        diff = _enriched(source, base_sha=source.base, plan=plan, plan_reason=plan_reason)
        assert diff.context is not None
        seen: list[dict] = []

        with patch(
            'orchestrator.evals.reviewer_trial.runner.invoke_agent',
            side_effect=_capturing_invoke(seen),
        ):
            await run_panel(variant, diff, stagger_secs=0)

        async with fixture_checkout(diff) as checkout:
            assert checkout.change_base == source.base
            subject = await gather_review_subject(
                checkout.root, source.base,
                task=diff.context.task.as_task_dict(),
                plan=diff.context.plan,
                plan_absent_reason=diff.context.plan_reason or '',
                conventions_relpath=project_review_config(source.root).briefing_path,
            )
        expected = await BriefingAssembler(OrchestratorConfig()).build_context_complete_reviewer_prompt(
            'reviewer_r1', subject, context='',
        )
        [call] = seen
        assert call['prompt'] == expected
        assert diff.context.task.title in call['prompt']
        assert diff.context.task.test_strategy in call['prompt']
        for path in ('a.py', 'new.py'):
            assert f'`{path}`' in call['prompt']
        assert call['head_parent'] == source.base
        spec = variant.reviewers[0]
        assert call['system_prompt'] == build_reviewer_role(
            spec.name, spec.specialization, ReviewBriefing.CONTEXT_COMPLETE,
        ).system_prompt

    @pytest.mark.asyncio
    async def test_a_d2_arm_reads_conventions_where_the_fixture_project_configures_them(
        self, source: _Source,
    ) -> None:
        _git(source.root, 'checkout', '-q', '-b', 'conventions', source.base)
        base = _commit(source.root, {
            'review/briefing.yaml': _conventions_yaml('DEFAULT-PATH RULE'),
            'docs/review-conventions.yaml': _conventions_yaml('PROJECT-PATH RULE'),
        }, 'conventions at two paths')
        (source.root / 'dark-factory-orchestrator.yaml').write_text(
            'review:\n  briefing_path: docs/review-conventions.yaml\n',
        )
        variant = with_briefing(_one_reviewer(), ReviewBriefing.CONTEXT_COMPLETE)
        seen: list[dict] = []

        with patch(
            'orchestrator.evals.reviewer_trial.runner.invoke_agent',
            side_effect=_capturing_invoke(seen),
        ):
            await run_panel(variant, _enriched(source, base_sha=base), stagger_secs=0)

        [call] = seen
        assert 'PROJECT-PATH RULE' in call['prompt']
        assert 'DEFAULT-PATH RULE' not in call['prompt']


def _conventions_yaml(rule: str) -> str:
    return f'conventions:\n  - rule: {rule}\n    why: the project says so\n'


class TestTrialFixtureReadiness:
    """A D2 arm fails loudly on unenriched fixtures and skips unmaterialisable ones."""

    @pytest.mark.asyncio
    async def test_a_d2_arm_refuses_a_corpus_missing_enrichment_before_any_spend(
        self, source: _Source, tmp_path: Path,
    ) -> None:
        corpus = CorpusManifest(diffs=[
            _diff_in(source, diff_id='bare_one'),
            _enriched(source, diff_id='enriched', base_sha=source.base),
            _diff_in(source, diff_id='bare_two'),
        ])
        arms = [_one_reviewer(), with_briefing(_one_reviewer(), ReviewBriefing.CONTEXT_COMPLETE)]
        results_dir = tmp_path / 'results'

        with patch(
            'orchestrator.evals.reviewer_trial.runner.invoke_agent', new_callable=AsyncMock,
        ) as mock_invoke, pytest.raises(FixtureEnrichmentMissing) as raised:
            await run_trial(arms, corpus, results_dir=results_dir)

        mock_invoke.assert_not_called()
        assert 'bare_one' in str(raised.value)
        assert 'bare_two' in str(raised.value)
        assert 'enriched' not in str(raised.value)
        assert not list(results_dir.glob('*.json'))

    @pytest.mark.asyncio
    async def test_a_diff_only_trial_still_runs_unenriched_fixtures(
        self, source: _Source, tmp_path: Path,
    ) -> None:
        corpus = CorpusManifest(diffs=[_diff_in(source, diff_id='bare_one')])

        with patch(
            'orchestrator.evals.reviewer_trial.runner.invoke_agent',
            side_effect=_make_verdict_writing_invoke(verdict='PASS'),
        ):
            results = await run_trial([_one_reviewer()], corpus, results_dir=tmp_path / 'results')

        assert [(r.variant_name, r.diff_id) for r in results] == [('r1_panel', 'bare_one')]

    @pytest.mark.asyncio
    async def test_an_unmaterialisable_fixture_is_skipped_for_the_d2_arm_only(
        self, source: _Source, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        reason = 'synthetic diff never applied to any commit'
        corpus = CorpusManifest(diffs=[
            _enriched(source, diff_id='fx_nobase', base_sha=None, base_sha_reason=reason),
        ])
        diff_only = _one_reviewer()
        d2 = with_briefing(diff_only, ReviewBriefing.CONTEXT_COMPLETE)
        results_dir = tmp_path / 'results'

        with (
            caplog.at_level(logging.WARNING, logger='orchestrator.evals.reviewer_trial.runner'),
            patch(
                'orchestrator.evals.reviewer_trial.runner.invoke_agent', new_callable=AsyncMock,
            ) as mock_invoke,
        ):
            mock_invoke.side_effect = _make_verdict_writing_invoke(verdict='PASS')
            results = await run_trial([diff_only, d2], corpus, results_dir=results_dir)

        assert [(r.variant_name, r.diff_id) for r in results] == [(diff_only.name, 'fx_nobase')]
        assert mock_invoke.await_count == 1
        assert [(p.variant_name, p.diff_id) for p in _stored_panels(results_dir)] == [
            (diff_only.name, 'fx_nobase'),
        ]
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any(reason in message and d2.name in message for message in warnings), warnings


_GATE = cast(UsageGate, object())


@contextlib.contextmanager
def _counted_reviewers() -> Iterator[list[str]]:
    """Patch both reviewer dispatch paths with a PASS-verdict writer; yield every dispatch, in order."""
    dispatched: list[str] = []
    write_verdict = _make_verdict_writing_invoke(verdict='PASS')

    async def _invoke(**kwargs):
        dispatched.append('ungated')
        return await write_verdict(**kwargs)

    async def _cap_retry(usage_gate, label, *, invoke_fn, backend, **kwargs):
        dispatched.append('gated')
        return await write_verdict(**kwargs)

    with (
        patch('orchestrator.evals.reviewer_trial.runner.invoke_agent', side_effect=_invoke),
        patch('orchestrator.evals.reviewer_trial.runner.invoke_with_cap_retry', side_effect=_cap_retry),
    ):
        yield dispatched


def _full_style() -> dict:
    return {'prices': default_price_table(), 'usage_gate': _GATE, 'diff_cap_chars': 0}


def _stored_panels(results_dir: Path) -> list[PanelRunResult]:
    return [PanelRunResult.from_dict(json.loads(p.read_text())) for p in sorted(results_dir.glob('*.json'))]


def _stored_scores(score_dir: Path) -> list[ScoringResult]:
    return [ScoringResult.from_dict(json.loads(p.read_text())) for p in sorted(score_dir.glob('*.json'))]


def _only_entry(directory: Path) -> Path:
    [entry] = directory.glob('*.json')
    return entry


_SMOKE_KEY = PanelRunKey.of(prices=None, usage_gate=None, diff_cap_chars=DIFF_CAP_CHARS)


def _write_legacy_panel(results_dir: Path, legacy: PanelRunResult) -> Path:
    """Store *legacy* as a panel written before task 6741: no run_key, at the bare cell name."""
    results_dir.mkdir(parents=True, exist_ok=True)
    record = asdict(legacy)
    del record['run_key']
    entry = results_dir / f'{legacy.variant_name}__{legacy.diff_id}.json'
    entry.write_text(json.dumps(record))
    return entry


class TestPanelCache:
    """run_panel_cached reuses a stored panel only when it was produced under the same key."""

    @pytest.mark.asyncio
    async def test_a_full_style_run_never_reuses_a_smoke_entry(self, source: _Source, tmp_path: Path) -> None:
        variant, diff, results_dir = _one_reviewer(), _diff_in(source), tmp_path / 'results'

        with _counted_reviewers() as dispatched:
            await run_panel_cached(variant, diff, results_dir=results_dir, stagger_secs=0)
            await run_panel_cached(variant, diff, results_dir=results_dir, stagger_secs=0, **_full_style())

        assert dispatched == ['ungated', 'gated']
        assert {p.run_key for p in _stored_panels(results_dir)} == {_SMOKE_KEY, PanelRunKey.of(**_full_style())}

    @pytest.mark.asyncio
    async def test_each_key_keeps_its_own_slot(self, source: _Source, tmp_path: Path) -> None:
        variant, diff, results_dir = _one_reviewer(), _diff_in(source), tmp_path / 'results'

        with _counted_reviewers() as dispatched:
            for _ in range(2):
                await run_panel_cached(variant, diff, results_dir=results_dir, stagger_secs=0)
                await run_panel_cached(variant, diff, results_dir=results_dir, stagger_secs=0, **_full_style())

        assert dispatched == ['ungated', 'gated']

    @pytest.mark.asyncio
    async def test_an_identical_call_is_served_from_the_cache(self, source: _Source, tmp_path: Path) -> None:
        variant, diff, results_dir = _one_reviewer(), _diff_in(source), tmp_path / 'results'

        with _counted_reviewers() as dispatched:
            first = await run_panel_cached(variant, diff, results_dir=results_dir, stagger_secs=0)
            second = await run_panel_cached(variant, diff, results_dir=results_dir, stagger_secs=0)

        assert dispatched == ['ungated']
        assert second == first

    @pytest.mark.parametrize('change', [
        pytest.param(lambda: {'prices': default_price_table()}, id='prices'),
        pytest.param(lambda: {'usage_gate': _GATE}, id='gate'),
        pytest.param(lambda: {'diff_cap_chars': 0}, id='cap'),
    ])
    @pytest.mark.asyncio
    async def test_any_one_changed_component_reruns(self, source: _Source, tmp_path: Path, change) -> None:
        variant, diff, results_dir = _one_reviewer(), _diff_in(source), tmp_path / 'results'

        with _counted_reviewers() as dispatched:
            await run_panel_cached(variant, diff, results_dir=results_dir, stagger_secs=0)
            await run_panel_cached(variant, diff, results_dir=results_dir, stagger_secs=0, **change())

        assert len(dispatched) == 2

    @pytest.mark.asyncio
    async def test_a_keyless_legacy_entry_is_honoured_and_named_quietly(
        self, source: _Source, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        variant, diff, results_dir = _one_reviewer(), _diff_in(source), tmp_path / 'results'
        legacy = PanelRunResult(
            variant_name=variant.name, diff_id=diff.diff_id,
            reviews={'r1': _make_review('r1')}, total_cost_usd=0.75, wall_clock_ms=10,
        )
        entry = _write_legacy_panel(results_dir, legacy)

        with (
            caplog.at_level(logging.INFO, logger='orchestrator.evals.reviewer_trial.runner'),
            _counted_reviewers() as dispatched,
        ):
            result = await run_panel_cached(variant, diff, results_dir=results_dir, stagger_secs=0, **_full_style())

        assert dispatched == []
        assert result == legacy
        infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        assert any(str(entry) in message for message in infos), infos
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []

    @pytest.mark.asyncio
    async def test_a_corrupt_legacy_entry_is_rerun(self, source: _Source, tmp_path: Path) -> None:
        variant, diff, results_dir = _one_reviewer(), _diff_in(source), tmp_path / 'results'
        results_dir.mkdir()
        (results_dir / f'{variant.name}__{diff.diff_id}.json').write_text('{not json')

        with _counted_reviewers() as dispatched:
            result = await run_panel_cached(variant, diff, results_dir=results_dir, stagger_secs=0)

        assert dispatched == ['ungated']
        assert result.run_key == _SMOKE_KEY

    @pytest.mark.parametrize('stored', [
        pytest.param(lambda variant, diff: '{not json', id='corrupt'),
        pytest.param(
            lambda variant, diff: json.dumps(asdict(PanelRunResult(
                variant_name='another_variant', diff_id=diff.diff_id, run_key=_SMOKE_KEY,
            ))),
            id='foreign-variant',
        ),
        pytest.param(
            lambda variant, diff: json.dumps(asdict(PanelRunResult(
                variant_name=variant.name, diff_id='another_diff', run_key=_SMOKE_KEY,
            ))),
            id='foreign-diff',
        ),
    ])
    @pytest.mark.asyncio
    async def test_a_corrupt_or_foreign_entry_is_rerun_and_overwritten(
        self, source: _Source, tmp_path: Path, stored,
    ) -> None:
        variant, diff, results_dir = _one_reviewer(), _diff_in(source), tmp_path / 'results'

        with _counted_reviewers() as dispatched:
            await run_panel_cached(variant, diff, results_dir=results_dir, stagger_secs=0)
            entry = _only_entry(results_dir)
            entry.write_text(stored(variant, diff))
            result = await run_panel_cached(variant, diff, results_dir=results_dir, stagger_secs=0)

        assert dispatched == ['ungated', 'ungated']
        assert PanelRunResult.from_dict(json.loads(entry.read_text())) == result
        assert (result.variant_name, result.diff_id) == (variant.name, diff.diff_id)

    @pytest.mark.asyncio
    async def test_run_trial_reruns_a_smoke_entry_under_a_full_style_key(
        self, source: _Source, tmp_path: Path,
    ) -> None:
        variant, diff, results_dir = _one_reviewer(), _diff_in(source), tmp_path / 'results'
        corpus = CorpusManifest(diffs=[diff])

        with _counted_reviewers() as dispatched:
            await run_panel_cached(variant, diff, results_dir=results_dir, stagger_secs=0)
            await run_trial([variant], corpus, results_dir=results_dir, **_full_style())
            assert dispatched == ['ungated', 'gated']
            await run_trial([variant], corpus, results_dir=results_dir, **_full_style())

        assert dispatched == ['ungated', 'gated']

    @pytest.mark.asyncio
    async def test_run_trial_summarises_its_legacy_hits_in_one_warning(
        self, source: _Source, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        variant, results_dir = _one_reviewer(), tmp_path / 'results'
        diffs = [_diff_in(source, diff_id='d1'), _diff_in(source, diff_id='d2')]
        for diff in diffs:
            _write_legacy_panel(results_dir, PanelRunResult(variant_name=variant.name, diff_id=diff.diff_id))

        with (
            caplog.at_level(logging.INFO, logger='orchestrator.evals.reviewer_trial.runner'),
            _counted_reviewers() as dispatched,
        ):
            await run_trial([variant], CorpusManifest(diffs=diffs), results_dir=results_dir, stagger_secs=0)

        assert dispatched == []
        [warning] = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert 'Reused 2 legacy panel' in warning and str(results_dir) in warning, warning

    @pytest.mark.asyncio
    async def test_run_trial_threads_its_stagger_to_each_panel(self, source: _Source, tmp_path: Path) -> None:
        with (
            _counted_reviewers(),
            patch('orchestrator.evals.reviewer_trial.runner.run_panel', wraps=run_panel) as spy,
        ):
            await run_trial(
                [_one_reviewer()], CorpusManifest(diffs=[_diff_in(source)]),
                results_dir=tmp_path / 'results', stagger_secs=0,
            )

        assert spy.await_args is not None
        assert spy.await_args.kwargs['stagger_secs'] == 0


_BLOCKING_ISSUE = {
    'severity': 'blocking', 'location': 'a.py:1', 'category': 'bug', 'description': 'x is wrong',
}


def _scored_diff(source: _Source, diff_id: str, marker: str) -> CorpusDiff:
    """A fixture with one planted blocking issue, whose matcher prompt carries *marker*."""
    return CorpusDiff(
        diff_id=diff_id, language='python', source='synthetic', diff_text=marker,
        description='Test', cwd=source.root,
        ground_truth=[GroundTruthIssue(
            id=f'{diff_id}_gt', location='a.py:1', category='bug', severity='blocking',
            description='x is wrong', mutation_type='logic',
        )],
    )


class TestRunAndScore:
    """run_and_score runs every panel through the cache, then scores each, caching measured scores only."""

    @staticmethod
    def _corpus(source: _Source) -> CorpusManifest:
        return CorpusManifest(diffs=[
            _scored_diff(source, 'd_ok', 'OK_MARKER'),
            _scored_diff(source, 'd_refused', 'REFUSED_MARKER'),
        ])

    @staticmethod
    @contextlib.contextmanager
    def _panels_and_matcher() -> Iterator[list[str]]:
        """Reviewers find the blocking issue; the matcher matches OK_MARKER's and refuses REFUSED_MARKER's.

        Yields the marker of every matcher prompt, in order.
        """
        matched: list[str] = []

        async def matcher(**kwargs: object):
            refused = 'REFUSED_MARKER' in str(kwargs['prompt'])
            matched.append('REFUSED_MARKER' if refused else 'OK_MARKER')
            if refused:
                return refused_agent_result()
            return answered({'matches': [{
                'reviewer_issue_index': 0, 'ground_truth_id': 'd_ok_gt',
                'confidence': 0.9, 'reasoning': 'same bug',
            }]}, cost_usd=0.1)

        with (
            patch(
                'orchestrator.evals.reviewer_trial.runner.invoke_agent',
                side_effect=_make_verdict_writing_invoke('ISSUES_FOUND', issues=[_BLOCKING_ISSUE]),
            ),
            patch('orchestrator.evals.reviewer_trial.scorer.invoke_agent', matcher),
        ):
            yield matched

    @pytest.mark.asyncio
    async def test_one_score_per_panel_in_trial_order(self, source: _Source, tmp_path: Path) -> None:
        variants = [_one_reviewer('a_panel'), _one_reviewer('b_panel')]
        seen: list[ScoringResult] = []

        with self._panels_and_matcher():
            scores = await run_and_score(
                variants, self._corpus(source), results_dir=tmp_path / 'results',
                stagger_secs=0, on_scored=seen.append,
            )

        assert [(s.variant_name, s.diff_id) for s in scores] == [
            ('a_panel', 'd_ok'), ('a_panel', 'd_refused'), ('b_panel', 'd_ok'), ('b_panel', 'd_refused'),
        ]
        assert seen == scores
        assert scores[0].f1 == 1.0 and scores[0].unmeasured is None
        assert scores[1].unmeasured is not None

    @pytest.mark.asyncio
    async def test_a_score_dir_caches_measured_scores_only(self, source: _Source, tmp_path: Path) -> None:
        variant, corpus = _one_reviewer(), self._corpus(source)
        results_dir, score_dir = tmp_path / 'results', tmp_path / 'scores'

        with self._panels_and_matcher() as matched:
            first = await run_and_score(
                [variant], corpus, results_dir=results_dir, score_dir=score_dir, stagger_secs=0,
            )
            assert matched == ['OK_MARKER', 'REFUSED_MARKER']
            second = await run_and_score(
                [variant], corpus, results_dir=results_dir, score_dir=score_dir, stagger_secs=0,
            )

        assert _stored_scores(score_dir) == [first[0]]
        assert matched == ['OK_MARKER', 'REFUSED_MARKER', 'REFUSED_MARKER']
        assert second[0] == first[0]
        assert second[1].unmeasured is not None

    @pytest.mark.asyncio
    async def test_a_corrupt_stored_score_is_rescored_and_overwritten(self, source: _Source, tmp_path: Path) -> None:
        variant, corpus = _one_reviewer(), self._corpus(source)
        results_dir, score_dir = tmp_path / 'results', tmp_path / 'scores'

        with self._panels_and_matcher() as matched:
            await run_and_score([variant], corpus, results_dir=results_dir, score_dir=score_dir, stagger_secs=0)
            _only_entry(score_dir).write_text('{"variant_name": "r1_pa')
            second = await run_and_score(
                [variant], corpus, results_dir=results_dir, score_dir=score_dir, stagger_secs=0,
            )

        assert matched == ['OK_MARKER', 'REFUSED_MARKER', 'OK_MARKER', 'REFUSED_MARKER']
        assert second[0].unmeasured is None
        assert _stored_scores(score_dir) == [second[0]]

    @pytest.mark.asyncio
    async def test_a_panel_rerun_under_a_new_key_is_rescored(self, source: _Source, tmp_path: Path) -> None:
        variant, corpus = _one_reviewer(), CorpusManifest(diffs=[_scored_diff(source, 'd_ok', 'OK_MARKER')])
        dirs = {'results_dir': tmp_path / 'results', 'score_dir': tmp_path / 'scores', 'stagger_secs': 0}

        with self._panels_and_matcher() as matched:
            await run_and_score([variant], corpus, **dirs)
            await run_and_score([variant], corpus, diff_cap_chars=0, **dirs)
            assert matched == ['OK_MARKER', 'OK_MARKER']
            await run_and_score([variant], corpus, **dirs)
            await run_and_score([variant], corpus, diff_cap_chars=0, **dirs)

        assert matched == ['OK_MARKER', 'OK_MARKER']
        assert len(_stored_scores(dirs['score_dir'])) == 2

    @pytest.mark.asyncio
    async def test_a_legacy_panel_reuses_its_legacy_score(self, source: _Source, tmp_path: Path) -> None:
        variant, results_dir, score_dir = _one_reviewer(), tmp_path / 'results', tmp_path / 'scores'
        _write_legacy_panel(results_dir, PanelRunResult(variant_name=variant.name, diff_id='d_ok'))
        score_dir.mkdir()
        legacy_score = ScoringResult(variant_name=variant.name, diff_id='d_ok', f1=0.25, cost_usd=1.5)
        (score_dir / f'{variant.name}__d_ok.json').write_text(json.dumps(asdict(legacy_score)))

        with self._panels_and_matcher() as matched, _counted_reviewers() as dispatched:
            scores = await run_and_score(
                [variant], CorpusManifest(diffs=[_scored_diff(source, 'd_ok', 'OK_MARKER')]),
                results_dir=results_dir, score_dir=score_dir, stagger_secs=0,
            )

        assert (dispatched, matched) == ([], [])
        assert scores == [legacy_score]

    @pytest.mark.asyncio
    async def test_without_a_score_dir_every_run_rescores(self, source: _Source, tmp_path: Path) -> None:
        variant, corpus, results_dir = _one_reviewer(), self._corpus(source), tmp_path / 'results'

        with self._panels_and_matcher() as matched:
            await run_and_score([variant], corpus, results_dir=results_dir, stagger_secs=0)
            await run_and_score([variant], corpus, results_dir=results_dir, stagger_secs=0)

        assert matched == ['OK_MARKER', 'REFUSED_MARKER', 'OK_MARKER', 'REFUSED_MARKER']
        assert {p.parent for p in tmp_path.rglob('*.json')} == {results_dir}

    @pytest.mark.asyncio
    async def test_a_usage_gate_reaches_the_panels_and_the_matcher(self, source: _Source, tmp_path: Path) -> None:
        gate = build_usage_gate([AccountConfig(name='max-a', oauth_token_env='CLAUDE_OAUTH_A')], ['tok-a'])
        reviewer_calls: list[dict] = []
        matcher_calls: list[dict] = []
        write_verdict = _make_verdict_writing_invoke('ISSUES_FOUND', issues=[_BLOCKING_ISSUE])

        async def reviewer(**kwargs):
            reviewer_calls.append(kwargs)
            await write_verdict(**kwargs)
            return AgentResult(success=True, output='', cost_usd=0.5, duration_ms=1000, turns=1)

        async def matcher(**kwargs: object):
            matcher_calls.append(kwargs)
            return answered({'matches': []})

        with (
            patch('orchestrator.evals.reviewer_trial.runner.invoke_agent', side_effect=reviewer),
            patch('orchestrator.evals.reviewer_trial.scorer.invoke_agent', matcher),
        ):
            await run_and_score(
                [_one_reviewer()], CorpusManifest(diffs=[_scored_diff(source, 'd_ok', 'OK_MARKER')]),
                results_dir=tmp_path / 'results', stagger_secs=0, usage_gate=gate,
            )

        assert [c['oauth_token'] for c in reviewer_calls] == ['tok-a']
        assert [c['oauth_token'] for c in matcher_calls] == ['tok-a']
