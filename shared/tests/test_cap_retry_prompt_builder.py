"""The caller's prompt BUILDER contract of ``invoke_with_cap_retry`` (task 5730).

``rebuild_prompt`` is a no-arg async callable returning ``str | None``.  The
retry loop consults it at ONE point: immediately before every RE-dispatch,
after the cooldown and after the next slot is acquired, never before the first
dispatch.  A fresh retry sends the built string; a resumed retry keeps its own
continuation prompt but still lets the builder cancel.  ``None`` cancels: the
last attempt's result comes back unretried, flagged ``retry_aborted`` and never
a success.  A builder that raises or returns a blank string degrades to the
original prompt.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from shared.cli_invoke import (
    CAP_HIT_RESUME_PROMPT,
    AgentResult,
    invoke_with_cap_retry,
    is_cli_invocation_rejected,
)
from shared.config_dir import TaskConfigDir
from shared.testing import make_gate_mock

_INVOKE_PATCH = 'shared.cli_invoke.invoke_claude_agent'
_SLEEP_PATCH = 'shared.cli_invoke.asyncio.sleep'

_ORIGINAL = 'the original prompt'
_BUILT = 'BUILT FROM LIVE STATE'


def _ok() -> AgentResult:
    return AgentResult(success=True, output='done', cost_usd=0.5, turns=3, duration_ms=9_000)


def _cap_banner(session_id: str = '') -> AgentResult:
    return AgentResult(success=False, output='usage limit', cost_usd=0.5, session_id=session_id)


def _cli_rejected() -> AgentResult:
    result = AgentResult(
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
    assert is_cli_invocation_rejected(result)
    return result


def _write_progressed_transcript(base: Path, session_id: str) -> None:
    records = [
        {'type': 'user', 'content': 'hi'},
        {
            'type': 'assistant',
            'message': {
                'role': 'assistant',
                'content': [
                    {'type': 'tool_use', 'name': 'Read', 'input': {'file_path': '/tmp/x.py'}},
                ],
            },
        },
    ]
    slug_dir = base / 'projects' / 'myproject'
    slug_dir.mkdir(parents=True, exist_ok=True)
    (slug_dir / f'{session_id}.jsonl').write_text(
        '\n'.join(json.dumps(r) for r in records) + '\n',
    )


@dataclass
class _Trigger:
    """One retry trigger: what the first attempt returns and how the loop is set up."""

    first: AgentResult
    detect_cap_hit: list[bool] = field(default_factory=lambda: [False, False])
    invoke_kwargs: dict[str, Any] = field(default_factory=dict)
    gated: bool = True
    lease_is_current: bool = False
    transcript_session: str = ''
    expect_resume: str = ''


_FRESH_TRIGGERS = {
    'cap-hit-no-session': _Trigger(first=_cap_banner(), detect_cap_hit=[True, False]),
    'failed-resume': _Trigger(
        first=AgentResult(success=False, output='resume broke', cost_usd=0.5),
        invoke_kwargs={'resume_session_id': 'sess-caller'},
    ),
    'auth-failed': _Trigger(
        first=AgentResult(success=False, output='', api_error_status=401, cost_usd=0.0),
    ),
    'zero-output-wedge-on-resume': _Trigger(
        first=AgentResult(
            success=False, output='', cost_usd=0.0, duration_ms=300_000,
            turns=0, session_id='sess-caller', timed_out=True,
        ),
        invoke_kwargs={'resume_session_id': 'sess-caller'},
    ),
    'heuristic-instant-cap': _Trigger(
        first=AgentResult(
            success=False, output='weird-unrecognised', cost_usd=0.0,
            turns=0, duration_ms=100,
        ),
        lease_is_current=True,
    ),
    'cli-input-rejected-gated': _Trigger(first=_cli_rejected()),
    'cli-input-rejected-no-gate': _Trigger(first=_cli_rejected(), gated=False),
}

_RESUME_TRIGGER = _Trigger(
    first=_cap_banner(session_id='sess-capped'),
    detect_cap_hit=[True, False],
    transcript_session='sess-capped',
    expect_resume='sess-capped',
)


async def _drive(
    trigger: _Trigger, builder: Any, tmp_path: Path,
) -> tuple[AgentResult, AsyncMock]:
    gate = None
    if trigger.gated:
        gate = make_gate_mock(
            account_count=2,
            before_invoke=AsyncMock(side_effect=['tok-a', 'tok-b']),
            detect_cap_hit=MagicMock(side_effect=trigger.detect_cap_hit),
        )
        gate.lease_is_current = MagicMock(return_value=trigger.lease_is_current)
    config_dir = None
    if trigger.transcript_session:
        config_dir = TaskConfigDir('5730-builder-resume', base_dir=tmp_path)
        _write_progressed_transcript(config_dir.path, trigger.transcript_session)
    with (
        patch(_INVOKE_PATCH, new_callable=AsyncMock, side_effect=[trigger.first, _ok()]) as inv,
        patch(_SLEEP_PATCH, new_callable=AsyncMock),
    ):
        result = await invoke_with_cap_retry(
            gate, 'lbl-builder', config_dir=config_dir,
            prompt=_ORIGINAL, rebuild_prompt=builder, **trigger.invoke_kwargs,
        )
    return result, inv


@pytest.mark.asyncio
class TestBuilderConsultedBeforeEveryRedispatch:
    @pytest.mark.parametrize('name', sorted(_FRESH_TRIGGERS))
    async def test_fresh_retry_dispatches_the_built_prompt(self, name, tmp_path):
        builder = AsyncMock(return_value=_BUILT)

        result, inv = await _drive(_FRESH_TRIGGERS[name], builder, tmp_path)

        assert inv.await_count == 2
        builder.assert_awaited_once_with()
        retry = inv.call_args_list[1].kwargs
        assert 'resume_session_id' not in retry
        assert retry['prompt'] == _BUILT
        assert result.success is True

    async def test_resumed_retry_consults_the_builder_but_keeps_the_resume_prompt(
        self, tmp_path,
    ):
        builder = AsyncMock(return_value=_BUILT)

        _result, inv = await _drive(_RESUME_TRIGGER, builder, tmp_path)

        builder.assert_awaited_once_with()
        retry = inv.call_args_list[1].kwargs
        assert retry['resume_session_id'] == 'sess-capped'
        assert retry['prompt'] == CAP_HIT_RESUME_PROMPT

    async def test_every_retry_reconsults_the_builder(self):
        gate = make_gate_mock(
            account_count=3,
            before_invoke=AsyncMock(side_effect=['tok-a', 'tok-b', 'tok-c']),
            detect_cap_hit=MagicMock(side_effect=[True, True, False]),
        )
        builder = AsyncMock(side_effect=['BUILT-1', 'BUILT-2'])
        with (
            patch(
                _INVOKE_PATCH, new_callable=AsyncMock,
                side_effect=[_cap_banner(), _cap_banner(), _ok()],
            ) as inv,
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
        ):
            await invoke_with_cap_retry(
                gate, 'lbl', prompt=_ORIGINAL, rebuild_prompt=builder,
            )

        assert builder.await_args_list == [call(), call()]
        assert [c.kwargs['prompt'] for c in inv.call_args_list] == [
            _ORIGINAL, 'BUILT-1', 'BUILT-2',
        ]

    async def test_builder_runs_after_the_wait_immediately_before_dispatch(self):
        order: list[str] = []

        async def acquire(**_kw):
            order.append('acquire')
            return 'tok'

        results = iter([_cap_banner(), _ok()])

        async def dispatch(**_kw):
            order.append('dispatch')
            return next(results)

        async def builder():
            order.append('build')
            return _BUILT

        gate = make_gate_mock(
            account_count=2,
            before_invoke=AsyncMock(side_effect=acquire),
            detect_cap_hit=MagicMock(side_effect=[True, False]),
        )
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, side_effect=dispatch),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
        ):
            await invoke_with_cap_retry(
                gate, 'lbl', prompt=_ORIGINAL, rebuild_prompt=builder,
            )

        assert order == ['acquire', 'dispatch', 'acquire', 'build', 'dispatch']


@pytest.mark.asyncio
class TestBuilderReturningNoneCancelsTheRetry:
    @staticmethod
    def _two_account_gate() -> MagicMock:
        leases = iter([('tok-first', 'acct-first'), ('tok-second', 'acct-second')])

        async def acquire(**_kw):
            token, name = next(leases)
            gate.active_account_name = name
            return token

        gate = make_gate_mock(
            account_count=2,
            before_invoke=AsyncMock(side_effect=acquire),
            detect_cap_hit=MagicMock(side_effect=[True]),
        )
        return gate

    async def test_none_returns_the_last_attempt_unretried_and_never_a_success(self):
        gate = self._two_account_gate()
        exit_zero_banner = AgentResult(success=True, output='usage limit', cost_usd=0.0)
        builder = AsyncMock(return_value=None)
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, side_effect=[exit_zero_banner]) as inv,
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
        ):
            result = await invoke_with_cap_retry(
                gate, 'lbl', prompt=_ORIGINAL, rebuild_prompt=builder,
            )

        assert inv.await_count == 1
        builder.assert_awaited_once_with()
        assert result is exit_zero_banner
        assert result.retry_aborted is True
        assert result.success is False
        assert result.account_name == 'acct-first'

    async def test_the_cancelled_slot_is_released_not_settled(self):
        gate = self._two_account_gate()
        builder = AsyncMock(return_value=None)
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, side_effect=[_cap_banner()]),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
        ):
            await invoke_with_cap_retry(
                gate, 'lbl', prompt=_ORIGINAL, rebuild_prompt=builder,
            )

        assert call('tok-second') not in gate.confirm_account_ok.call_args_list
        gate.on_agent_complete.assert_not_called()
        assert call('tok-second') in gate.release_probe_slot.call_args_list

    async def test_none_cancels_on_the_no_gate_path(self):
        builder = AsyncMock(return_value=None)
        rejected = _cli_rejected()
        with patch(_INVOKE_PATCH, new_callable=AsyncMock, side_effect=[rejected]) as inv:
            result = await invoke_with_cap_retry(
                None, 'lbl', prompt=_ORIGINAL, rebuild_prompt=builder,
            )

        assert inv.await_count == 1
        assert result is rejected
        assert result.retry_aborted is True
        assert result.success is False

    async def test_no_builder_leaves_retry_aborted_false(self, tmp_path):
        result, _inv = await _drive(_FRESH_TRIGGERS['cap-hit-no-session'], None, tmp_path)

        assert result.retry_aborted is False

    async def test_builder_that_never_declines_leaves_retry_aborted_false(self, tmp_path):
        builder = AsyncMock(return_value=_BUILT)

        result, _inv = await _drive(_FRESH_TRIGGERS['cap-hit-no-session'], builder, tmp_path)

        assert result.retry_aborted is False
        assert result.success is True


@pytest.mark.asyncio
class TestBuilderFailureDegradesToTheOriginalPrompt:
    @pytest.mark.parametrize(
        'builder',
        [
            AsyncMock(side_effect=RuntimeError('transient MCP failure')),
            AsyncMock(return_value=''),
            AsyncMock(return_value='   \n'),
        ],
        ids=['raises', 'empty', 'whitespace'],
    )
    async def test_fresh_retry_dispatches_the_original_prompt(
        self, builder, tmp_path, caplog,
    ):
        with caplog.at_level(logging.WARNING, logger='shared.cli_invoke'):
            result, inv = await _drive(
                _FRESH_TRIGGERS['cap-hit-no-session'], builder, tmp_path,
            )

        assert inv.await_count == 2
        assert inv.call_args_list[1].kwargs['prompt'] == _ORIGINAL
        assert result.success is True
        assert result.retry_aborted is False
        assert any(
            r.levelno == logging.WARNING
            and 'lbl-builder' in r.getMessage()
            and 'prompt builder' in r.getMessage()
            for r in caplog.records
        )


@pytest.mark.asyncio
class TestNoBuilderReplaysTheOriginalPrompt:
    @pytest.mark.parametrize('name', sorted(_FRESH_TRIGGERS))
    async def test_fresh_retry_replays_the_original(self, name, tmp_path):
        _result, inv = await _drive(_FRESH_TRIGGERS[name], None, tmp_path)

        assert inv.call_args_list[1].kwargs['prompt'] == _ORIGINAL
