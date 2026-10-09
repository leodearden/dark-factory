"""The steward's prompt builder re-reads its target escalation live (task 5730).

``TaskSteward._invoke_with_session`` hands ``invoke_with_cap_retry`` a no-arg
builder that the shared loop awaits before every re-dispatch.  It must:

1. decline (``None``) once the escalation it is handling is no longer pending,
   so a retry never re-runs finished work;
2. render the LIVE record (amendments made after handling began), keeping the
   handled copy's ``detail``/``summary`` overlay that pre-triage applied;
3. reach a fresh retry after a pre-turn CLI rejection, so a rejected
   continuation is not resent bare into a brand-new session.
"""

from __future__ import annotations

import dataclasses
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from escalation.queue import EscalationQueue
from shared.cli_invoke import AgentResult
from shared.testing import make_gate_mock
from test_steward import _make_escalation, _make_result

_SLEEP_PATCH = 'shared.cli_invoke.asyncio.sleep'


@pytest.fixture
def queue(tmp_path):
    return EscalationQueue(tmp_path / 'escalations')


@pytest.fixture
def steward(make_steward, queue):
    return make_steward(escalation_queue=queue)


async def _captured_builder(steward, escalation):
    with patch(
        'orchestrator.steward.invoke_with_cap_retry', new_callable=AsyncMock,
    ) as iwcr:
        iwcr.return_value = _make_result(session_id='sess-x')
        await steward._invoke_with_session(
            prompt='handle this escalation',
            cwd=steward.worktree,
            mcp_config={'mcpServers': {}},
            per_invocation_budget=3.0,
            escalation=escalation,
        )
    return iwcr.call_args.kwargs['rebuild_prompt']


def _cli_rejected() -> AgentResult:
    return AgentResult(
        success=False,
        output='',
        subtype='error_cli_input_rejected',
        stderr=(
            'Error: Input must be provided either through stdin or as a prompt '
            'argument when using --print\n'
        ),
        turns=0,
        cost_usd=0.0,
        duration_ms=2_100,
    )


@pytest.mark.asyncio
class TestBuilderDeclinesFinishedWork:
    async def test_resolved_escalation_declines(self, steward, queue):
        esc = _make_escalation()
        queue.submit(esc)
        builder = await _captured_builder(steward, esc)

        queue.resolve(esc.id, 'resolved by the run that just capped')

        assert await builder() is None
        steward.briefing.build_steward_initial_prompt.assert_not_called()

    async def test_missing_escalation_declines(self, steward):
        esc = _make_escalation()
        builder = await _captured_builder(steward, esc)

        assert await builder() is None
        steward.briefing.build_steward_initial_prompt.assert_not_called()


@pytest.mark.asyncio
class TestBuilderRendersTheLiveRecord:
    async def test_amended_record_rendered_with_the_handled_overlay(self, steward, queue):
        esc = _make_escalation()
        queue.submit(esc)
        handled = dataclasses.replace(
            esc, detail='PRE-TRIAGED DETAIL', summary='3 suggestions pre-triaged',
        )
        builder = await _captured_builder(steward, handled)

        queue.amend(esc.id, root_cause='ruled after handling began', agent_role='steward')
        queue.submit(_make_escalation(id='esc-42-2', summary='filed meanwhile'))

        prompt = await builder()

        live = queue.get(esc.id)
        assert live is not None and live.amendments
        steward.briefing.build_steward_initial_prompt.assert_awaited_once_with(
            task=steward.task,
            escalation={
                **live.to_dict(),
                'detail': 'PRE-TRIAGED DETAIL',
                'summary': '3 suggestions pre-triaged',
            },
            pending_escalations=[
                e.to_dict() for e in queue.get_by_task(steward.task_id, status='pending')
            ],
            worktree=steward.worktree,
        )
        assert prompt == steward.briefing.build_steward_initial_prompt.return_value


@pytest.mark.asyncio
class TestBuilderThroughTheRealRetryLoop:
    async def test_rejected_continuation_retries_fresh_with_the_initial_briefing(
        self, steward, queue,
    ):
        esc = _make_escalation()
        queue.submit(esc)
        steward._session_id = 'sess-prev'
        steward.usage_gate = make_gate_mock(account_count=2)
        dispatches: list[dict] = []

        async def invoke_agent(**kwargs):
            dispatches.append(kwargs)
            return _cli_rejected() if len(dispatches) == 1 else _make_result()

        with (
            patch('orchestrator.steward.invoke_agent', side_effect=invoke_agent),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
        ):
            await steward._invoke_with_session(
                prompt='continuation prompt',
                cwd=steward.worktree,
                mcp_config={'mcpServers': {}},
                per_invocation_budget=3.0,
                escalation=esc,
            )

        assert len(dispatches) == 2
        assert dispatches[0]['resume_session_id'] == 'sess-prev'
        assert dispatches[0]['prompt'] == 'continuation prompt'
        assert not dispatches[1].get('resume_session_id')
        assert dispatches[1]['prompt'] == (
            steward.briefing.build_steward_initial_prompt.return_value
        )

    async def test_escalation_resolved_by_the_capped_run_cancels_the_rerun(
        self, steward, queue,
    ):
        esc = _make_escalation()
        queue.submit(esc)
        steward.usage_gate = make_gate_mock(
            account_count=2, detect_cap_hit=MagicMock(side_effect=[True]),
        )

        async def invoke_agent(**_kwargs):
            queue.resolve(esc.id, 'fixed before the cap hit')
            return AgentResult(success=False, output='usage limit', cost_usd=0.4)

        with (
            patch('orchestrator.steward.invoke_agent', side_effect=invoke_agent) as inv,
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
        ):
            result = await steward._invoke_with_session(
                prompt='initial prompt',
                cwd=steward.worktree,
                mcp_config={'mcpServers': {}},
                per_invocation_budget=3.0,
                escalation=esc,
            )

        assert inv.await_count == 1
        assert result.retry_aborted is True
