"""How ``steward_session_mode`` shapes the steward's session across L0s (task 5819).

``fresh_per_l0`` starts every handling in a new session with the full briefing
plus the digest of the L0s this steward already concluded; ``resume`` keeps one
resumed session across L0s and is pinned here byte-for-byte as the shadow-eval
comparison arm. The within-L0 cap-hit resume is shared by both.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _steward_helpers import _make_escalation, _make_result
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from shared.cli_invoke import AgentResult
from shared.testing import make_gate_mock

from orchestrator.agents.briefing import BriefingAssembler
from orchestrator.agents.steward_digest import EarlierAttempts, HandledL0
from orchestrator.config import OrchestratorConfig
from orchestrator.workflow_types import StewardInterrupted, StewardResolved

_SLEEP_PATCH = 'shared.cli_invoke.asyncio.sleep'
_RESUME_KEYS = ('resume_session_id', 'resume_prior_total_cost_usd')
_DIGEST_HEADING = '## Escalations This Steward Already Handled'
_ATTEMPTS_HEADING = '## Earlier Attempts at This Escalation'


def _resolution(escalation_id: str) -> str:
    return f'handled {escalation_id}'


class _Driver:
    """Stands in for the agent CLI: records each dispatch and, by default,
    resolves the escalation being handled, returning session ``sess-<n>``."""

    def __init__(self, queue: EscalationQueue):
        self.queue = queue
        self.dispatches: list[dict] = []
        self.handling: str | None = None
        self.scripted: list[AgentResult | None] = []

    async def invoke_agent(self, **kwargs) -> AgentResult:
        self.dispatches.append(kwargs)
        scripted = self.scripted.pop(0) if self.scripted else None
        if scripted is not None:
            return scripted
        assert self.handling is not None
        self.queue.resolve(self.handling, _resolution(self.handling))
        return _make_result(session_id=f'sess-{len(self.dispatches)}')

    async def handle(self, steward, escalation: Escalation) -> None:
        self.handling = escalation.id
        with patch('orchestrator.steward.invoke_agent', side_effect=self.invoke_agent):
            await steward._handle_escalation(escalation)


@pytest.fixture
def queue(tmp_path: Path) -> EscalationQueue:
    return EscalationQueue(tmp_path / 'escalations')


@pytest.fixture
def driver(queue: EscalationQueue) -> _Driver:
    return _Driver(queue)


@pytest.fixture
def two_l0s(queue: EscalationQueue) -> tuple[Escalation, Escalation]:
    first = _make_escalation(id=queue.make_id('42'), summary='first problem')
    second = _make_escalation(id=queue.make_id('42'), summary='second problem')
    assert (first.id, second.id) == ('esc-42-1', 'esc-42-2')
    queue.submit(first)
    queue.submit(second)
    return first, second


def _build(make_steward, queue: EscalationQueue, mode: str, **overrides):
    return make_steward(
        escalation_queue=queue,
        config_overrides={'steward_session_mode': mode, **overrides},
    )


def _pending_dicts(queue: EscalationQueue) -> list[dict]:
    return [e.to_dict() for e in queue.get_by_task('42', status='pending')]


def _resolved_entry(escalation: Escalation) -> HandledL0:
    return HandledL0(
        escalation.id, escalation.category,
        StewardResolved(resolution_text=_resolution(escalation.id)),
    )


def _carries_no_resume(dispatch: dict) -> bool:
    return not any(key in dispatch for key in _RESUME_KEYS)


def _timed_out(session_id: str) -> AgentResult:
    return _make_result(
        session_id=session_id, success=False, timed_out=True, duration_ms=900_000,
    )


@pytest.mark.asyncio
class TestFreshPerL0:
    async def test_every_l0_gets_a_new_session_and_the_full_briefing(
        self, make_steward, queue, driver, two_l0s,
    ):
        first, second = two_l0s
        steward = _build(make_steward, queue, 'fresh_per_l0')
        initial = steward.briefing.build_steward_initial_prompt

        await driver.handle(steward, first)
        pending_at_second = _pending_dicts(queue)
        await driver.handle(steward, second)

        assert len(driver.dispatches) == 2
        assert all(_carries_no_resume(d) for d in driver.dispatches)
        assert initial.await_count == 2
        steward.briefing.build_steward_continuation_prompt.assert_not_awaited()
        first_call, second_call = initial.await_args_list
        assert first_call.kwargs['handled'] == ()
        assert second_call.kwargs['handled'] == (_resolved_entry(first),)
        assert second_call.kwargs['pending_escalations'] == pending_at_second
        assert first_call.kwargs['earlier_attempts'] == EarlierAttempts()
        assert second_call.kwargs['earlier_attempts'] == EarlierAttempts()

    async def test_the_real_briefing_carries_the_digest_into_the_second_prompt(
        self, make_steward, queue, driver, two_l0s, tmp_path,
    ):
        first, second = two_l0s
        steward = _build(make_steward, queue, 'fresh_per_l0')
        steward.briefing = BriefingAssembler(OrchestratorConfig(project_root=tmp_path))

        with patch.object(
            BriefingAssembler, '_get_memory_context', return_value='# Context\n\n_stub_',
        ):
            await driver.handle(steward, first)
            await driver.handle(steward, second)

        first_prompt, second_prompt = (d['prompt'] for d in driver.dispatches)
        assert _DIGEST_HEADING not in first_prompt
        assert _DIGEST_HEADING in second_prompt
        assert 'esc-42-1' in second_prompt
        assert _resolution('esc-42-1') in second_prompt

    async def test_a_timeout_re_handle_starts_fresh_told_of_its_timed_out_attempt(
        self, make_steward, queue, driver, two_l0s,
    ):
        first, _ = two_l0s
        steward = _build(make_steward, queue, 'fresh_per_l0')
        driver.scripted = [_timed_out('sess-timed-out')]

        await driver.handle(steward, first)
        assert queue.get(first.id).status == 'pending'
        await driver.handle(steward, first)

        assert len(driver.dispatches) == 2
        assert _carries_no_resume(driver.dispatches[1])
        first_call, second_call = steward.briefing.build_steward_initial_prompt.await_args_list
        assert first_call.kwargs['earlier_attempts'] == EarlierAttempts()
        assert second_call.kwargs['handled'] == ()
        assert second_call.kwargs['earlier_attempts'] == EarlierAttempts(timed_out=1)

    async def test_the_real_briefing_tells_a_re_handle_about_its_timed_out_attempt(
        self, make_steward, queue, driver, two_l0s, tmp_path,
    ):
        first, _ = two_l0s
        steward = _build(make_steward, queue, 'fresh_per_l0')
        steward.briefing = BriefingAssembler(OrchestratorConfig(project_root=tmp_path))
        driver.scripted = [_timed_out('sess-timed-out')]

        with patch.object(
            BriefingAssembler, '_get_memory_context', return_value='# Context\n\n_stub_',
        ):
            await driver.handle(steward, first)
            await driver.handle(steward, first)

        first_prompt, second_prompt = (d['prompt'] for d in driver.dispatches)
        assert _ATTEMPTS_HEADING not in first_prompt
        assert _ATTEMPTS_HEADING in second_prompt
        assert 'time limit' in second_prompt

    async def test_in_ladder_cap_hit_recovery_still_resumes_the_capped_attempt(
        self, make_steward, queue, driver, two_l0s,
    ):
        first, second = two_l0s
        steward = _build(make_steward, queue, 'fresh_per_l0')
        await driver.handle(steward, first)

        steward.usage_gate = make_gate_mock(
            account_count=2, detect_cap_hit=MagicMock(side_effect=[True, False]),
        )
        driver.scripted = [
            AgentResult(
                success=False, output='usage limit', cost_usd=0.4, session_id='sess-cap',
            ),
        ]
        with patch(_SLEEP_PATCH, new_callable=AsyncMock):
            await driver.handle(steward, second)

        assert len(driver.dispatches) == 3
        assert driver.dispatches[2]['resume_session_id'] == 'sess-cap'

    async def test_a_fresh_retry_inside_a_re_handle_rebuilds_the_briefing_with_both_histories(
        self, make_steward, queue, driver, two_l0s,
    ):
        first, second = two_l0s
        steward = _build(make_steward, queue, 'fresh_per_l0')
        await driver.handle(steward, first)
        driver.scripted = [_timed_out('')]
        await driver.handle(steward, second)

        steward.usage_gate = make_gate_mock(
            account_count=2, detect_cap_hit=MagicMock(side_effect=[True, False]),
        )
        driver.scripted = [
            AgentResult(success=False, output='usage limit', cost_usd=0.4, session_id=''),
        ]
        with patch(_SLEEP_PATCH, new_callable=AsyncMock):
            await driver.handle(steward, second)

        assert len(driver.dispatches) == 4
        assert _carries_no_resume(driver.dispatches[3])
        initial = steward.briefing.build_steward_initial_prompt
        assert initial.await_count == 4
        rebuilt = initial.await_args_list[3]
        assert rebuilt.kwargs['handled'] == (_resolved_entry(first),)
        assert rebuilt.kwargs['earlier_attempts'] == EarlierAttempts(timed_out=1)

    async def test_a_give_up_is_digested_for_the_next_l0(
        self, make_steward, queue, driver, two_l0s,
    ):
        first, second = two_l0s
        steward = _build(make_steward, queue, 'fresh_per_l0', steward_max_attempts=1)
        driver.scripted = [_make_result(session_id='sess-left-pending')]

        await driver.handle(steward, first)
        assert queue.get(first.id).status == 'pending'
        await driver.handle(steward, first)
        await driver.handle(steward, second)

        last_call = steward.briefing.build_steward_initial_prompt.await_args_list[-1]
        assert last_call.kwargs['handled'] == (
            HandledL0(
                first.id, first.category,
                StewardInterrupted(reason='attempt_cap', wip_commits_present=False),
            ),
        )


@pytest.mark.asyncio
class TestResumeParity:
    async def test_one_session_is_resumed_across_l0s_with_the_continuation_prompt(
        self, make_steward, queue, driver, two_l0s,
    ):
        first, second = two_l0s
        steward = _build(make_steward, queue, 'resume')
        pending_at_first = _pending_dicts(queue)

        await driver.handle(steward, first)
        await driver.handle(steward, second)

        first_dispatch, second_dispatch = driver.dispatches
        assert _carries_no_resume(first_dispatch)
        assert first_dispatch['prompt'] == 'initial prompt'
        assert second_dispatch == {
            **first_dispatch,
            'prompt': 'continuation prompt',
            'resume_session_id': 'sess-1',
        }
        steward.briefing.build_steward_initial_prompt.assert_awaited_once_with(
            task=steward.task,
            escalation=first.to_dict(),
            pending_escalations=pending_at_first,
            worktree=steward.worktree,
            handled=(),
            earlier_attempts=EarlierAttempts(),
        )
        steward.briefing.build_steward_continuation_prompt.assert_awaited_once_with(
            task=steward.task,
            escalation=second.to_dict(),
        )

    async def test_a_sessionless_re_handle_gets_the_full_briefing_with_no_history(
        self, make_steward, queue, driver, two_l0s,
    ):
        first, _ = two_l0s
        steward = _build(make_steward, queue, 'resume')
        driver.scripted = [_timed_out('')]

        await driver.handle(steward, first)
        await driver.handle(steward, first)

        second_call = steward.briefing.build_steward_initial_prompt.await_args_list[1]
        assert second_call.kwargs['handled'] == ()
        assert second_call.kwargs['earlier_attempts'] == EarlierAttempts()


@pytest.mark.asyncio
class TestModeIsReadAtUseTime:
    async def test_a_flip_to_fresh_drops_the_carried_session(
        self, make_steward, queue, driver, two_l0s,
    ):
        first, second = two_l0s
        steward = _build(make_steward, queue, 'resume')

        await driver.handle(steward, first)
        steward.config.steward_session_mode = 'fresh_per_l0'
        await driver.handle(steward, second)

        assert _carries_no_resume(driver.dispatches[1])
        second_call = steward.briefing.build_steward_initial_prompt.await_args_list[1]
        assert second_call.kwargs['handled'] == (_resolved_entry(first),)

    async def test_a_flip_to_resume_resumes_the_last_session(
        self, make_steward, queue, driver, two_l0s,
    ):
        first, second = two_l0s
        steward = _build(make_steward, queue, 'fresh_per_l0')

        await driver.handle(steward, first)
        steward.config.steward_session_mode = 'resume'
        await driver.handle(steward, second)

        assert driver.dispatches[1]['resume_session_id'] == 'sess-1'
        assert driver.dispatches[1]['prompt'] == 'continuation prompt'
