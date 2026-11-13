"""Task 5579: only a pre-first-turn resume REJECTION retries fresh.

A resumed invocation that ran turns and then ended non-success did real work;
``invoke_with_cap_retry`` returns it to the caller like any fresh result.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.cap_markers import REAL_CLI_CAP_HIT_MESSAGES
from shared.cli_invoke import AgentResult, invoke_with_cap_retry, is_resume_rejected
from shared.testing import make_gate_mock
from shared.testing_resume_rejection import measured_resume_rejection


def _rejection(**changes: object):
    return dataclasses.replace(measured_resume_rejection('sess-armed'), **changes)


class TestIsResumeRejected:
    def test_the_measured_rejection_is_a_rejection(self):
        assert is_resume_rejected(measured_resume_rejection()) is True

    def test_a_success_is_not_a_rejection(self):
        assert is_resume_rejected(_rejection(success=True)) is False

    def test_a_timeout_is_not_a_rejection(self):
        assert is_resume_rejected(_rejection(timed_out=True)) is False

    @pytest.mark.parametrize(
        'changes',
        [pytest.param({'turns': 1}, id='took-a-turn'), pytest.param({'cost_usd': 0.01}, id='billed')],
    )
    def test_evidence_of_work_vetoes_the_marker(self, changes):
        assert is_resume_rejected(_rejection(**changes)) is False

    @pytest.mark.parametrize(
        'stderr',
        [pytest.param('Segmentation fault', id='other-stderr'), pytest.param('', id='empty-stderr')],
    )
    def test_a_pre_turn_failure_without_the_marker_is_not_a_rejection(self, stderr):
        result = _rejection(stderr=stderr)
        assert result.subtype == 'error_empty_output'
        assert is_resume_rejected(result) is False

    def test_the_marker_counts_only_on_stderr(self):
        marker_text = measured_resume_rejection('sess-armed').stderr
        assert is_resume_rejected(_rejection(stderr='', output=marker_text)) is False

    @pytest.mark.parametrize(
        'stderr',
        [
            pytest.param('NO CONVERSATION FOUND WITH SESSION ID: sess-armed', id='upper-case'),
            pytest.param(
                'Warning: something unrelated\nNo conversation found with session ID: sess-armed\n',
                id='one-line-of-several',
            ),
        ],
    )
    def test_the_marker_matches_case_insensitively_on_any_line(self, stderr):
        assert is_resume_rejected(_rejection(stderr=stderr)) is True


ARMED = 'sess-armed'


def _did_work_shapes() -> list:
    return [
        pytest.param(
            dict(
                subtype='success', stop_reason='end_turn', ended_awaiting_background=True,
                turns=40, cost_usd=2.195, transcript_turns=120,
            ),
            id='ended_awaiting_background',
        ),
        pytest.param(dict(subtype='error_max_budget_usd', turns=1, cost_usd=4.96), id='budget_kill'),
        pytest.param(
            dict(subtype='error_during_execution', turns=7, cost_usd=0.8),
            id='generic_non_success_after_turns',
        ),
    ]


def _resumed_non_success(session_id: str = ARMED, **fields: Any) -> AgentResult:
    return AgentResult(success=False, output='', session_id=session_id, **fields)


async def _drive(side_effect: list, *, detect_cap_hit: MagicMock | None = None, **kwargs: Any):
    gate = make_gate_mock(
        account_count=2,
        before_invoke=AsyncMock(side_effect=['tok-a', 'tok-b', 'tok-a']),
        detect_cap_hit=detect_cap_hit or MagicMock(return_value=False),
    )
    with (
        patch('shared.cli_invoke.invoke_claude_agent', new_callable=AsyncMock, side_effect=side_effect) as mock_invoke,
        patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock),
    ):
        got = await invoke_with_cap_retry(gate, 'test-label', prompt='real task', **kwargs)
    return got, mock_invoke


class TestAResumedRunThatDidWorkIsReturned:
    @pytest.mark.parametrize('fields', _did_work_shapes())
    async def test_the_caller_armed_resume_is_returned_unretried(self, fields, caplog):
        did_work = _resumed_non_success(**fields)
        caplog.set_level(logging.WARNING, logger='shared.cli_invoke')

        got, mock_invoke = await _drive([did_work], resume_session_id=ARMED)

        mock_invoke.assert_awaited_once()
        assert mock_invoke.await_args_list[0].kwargs['resume_session_id'] == ARMED
        assert got is did_work
        assert got.resume_fallbacks == 0
        assert got.resume_fallback_session_ids == ()
        assert got.session_id == ARMED
        assert 'retrying fresh' not in caplog.text

    async def test_the_returned_run_s_spend_is_recorded(self):
        did_work = _resumed_non_success(
            subtype='success', stop_reason='end_turn', ended_awaiting_background=True,
            turns=40, cost_usd=2.195,
        )
        cost_store = MagicMock()
        cost_store.save_invocation = AsyncMock()

        got, _ = await _drive([did_work], resume_session_id=ARMED, cost_store=cost_store)

        assert got.resume_fallbacks == 0
        cost_store.save_invocation.assert_awaited_once()
        assert cost_store.save_invocation.await_args.kwargs['cost_usd'] == 2.195

    async def test_an_unproven_pre_turn_failure_is_returned_unretried(self, caplog):
        unproven = _resumed_non_success(
            subtype='error_empty_output', stderr='Segmentation fault', duration_ms=30000,
        )
        caplog.set_level(logging.WARNING, logger='shared.cli_invoke')

        got, mock_invoke = await _drive([unproven], resume_session_id=ARMED)

        mock_invoke.assert_awaited_once()
        assert got is unproven
        assert got.resume_fallbacks == 0
        assert 'retrying fresh' not in caplog.text

    async def test_a_loop_rearmed_resume_that_did_work_is_returned_uncounted(self):
        capped = AgentResult(
            success=False, output=REAL_CLI_CAP_HIT_MESSAGES[0], session_id='sess-capped', cost_usd=0.5,
        )
        did_work = _resumed_non_success(
            session_id='sess-capped', subtype='error_during_execution', turns=7, cost_usd=0.8,
        )

        got, mock_invoke = await _drive(
            [capped, did_work], detect_cap_hit=MagicMock(side_effect=[True, False]),
        )

        assert mock_invoke.await_count == 2
        assert mock_invoke.await_args_list[1].kwargs['resume_session_id'] == 'sess-capped'
        assert got is did_work
        assert got.resume_fallbacks == 0


class TestAProvenRejectionStillRetriesFresh:
    async def test_the_rejected_session_is_dropped_and_the_real_prompt_replayed(self, caplog):
        rejection = measured_resume_rejection(ARMED)
        ok = AgentResult(success=True, output='ok', cost_usd=0.5)
        caplog.set_level(logging.WARNING, logger='shared.cli_invoke')

        got, mock_invoke = await _drive(
            [rejection, ok], resume_session_id=ARMED, session_id=ARMED,
        )

        assert mock_invoke.await_count == 2
        fresh = mock_invoke.await_args_list[1].kwargs
        assert 'resume_session_id' not in fresh
        assert fresh['prompt'] == 'real task'
        assert fresh['session_id'] != ARMED
        assert got.resume_fallbacks == 1
        assert got.resume_fallback_session_ids == (ARMED,)
        retry_lines = [r.getMessage() for r in caplog.records if 'retrying fresh' in r.getMessage()]
        assert len(retry_lines) == 1
        assert rejection.stderr in retry_lines[0]
