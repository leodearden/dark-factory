"""A caller's opt-out from cap-hit resume on ``invoke_with_cap_retry`` (task 4837).

``resume_on_cap_hit=False`` makes every cap-hit re-dispatch FRESH: the real
prompt, no ``--resume``, whatever the capped session's transcript shows.  It is
for stateless callers (the escalation-watcher-auto rotation, the dry-run
unblock investigation) that hold no mid-work state worth preserving, where a
resume would only replay a continuation prompt into a session that may no
longer carry the caller's charter.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.cli_invoke import CAP_HIT_RESUME_PROMPT, AgentResult, invoke_with_cap_retry
from shared.config_dir import TaskConfigDir
from shared.testing import make_gate_mock

_SLEEP_PATCH = 'shared.cli_invoke.asyncio.sleep'

_ORIGINAL = 'the original prompt'


def _ok() -> AgentResult:
    return AgentResult(success=True, output='done', cost_usd=0.5, turns=3, duration_ms=9_000)


def _capped(session_id: str) -> AgentResult:
    return AgentResult(success=False, output='usage limit', cost_usd=0.5, session_id=session_id)


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


def _gate(caps: int) -> MagicMock:
    """A gate with one more account than cap hits: *caps* caps, then a clean call."""
    return make_gate_mock(
        account_count=caps + 1,
        before_invoke=AsyncMock(side_effect=[f'tok-{i}' for i in range(caps + 1)]),
        detect_cap_hit=MagicMock(side_effect=[True] * caps + [False]),
        active_account_name='acct-b',
    )


@pytest.mark.asyncio
class TestResumeOnCapHitOptOut:
    async def test_opt_out_retries_fresh_without_config_dir(self):
        inv = AsyncMock(side_effect=[_capped('sess-1'), _ok()])
        with patch(_SLEEP_PATCH, new_callable=AsyncMock):
            result = await invoke_with_cap_retry(
                _gate(1), 'lbl-opt-out', invoke_fn=inv,
                resume_on_cap_hit=False, prompt=_ORIGINAL,
            )

        assert result.success is True
        assert inv.await_count == 2
        retry = inv.call_args_list[1].kwargs
        assert 'resume_session_id' not in retry
        assert retry['prompt'] == _ORIGINAL

    async def test_opt_out_retries_fresh_even_when_transcript_shows_progress(
        self, tmp_path,
    ):
        config_dir = TaskConfigDir('opt-out', base_dir=tmp_path)
        _write_progressed_transcript(config_dir.path, 'sess-1')
        inv = AsyncMock(side_effect=[_capped('sess-1'), _ok()])
        with patch(_SLEEP_PATCH, new_callable=AsyncMock):
            await invoke_with_cap_retry(
                _gate(1), 'lbl-opt-out', config_dir=config_dir, invoke_fn=inv,
                resume_on_cap_hit=False, prompt=_ORIGINAL, session_id='sess-1',
            )

        retry = inv.call_args_list[1].kwargs
        assert 'resume_session_id' not in retry
        assert retry['prompt'] == _ORIGINAL
        assert retry['session_id'] != 'sess-1'
        assert str(uuid.UUID(retry['session_id'])) == retry['session_id']

    async def test_opt_out_every_consecutive_cap_hit_is_fresh(self):
        inv = AsyncMock(side_effect=[
            _capped('sess-1'), _capped('sess-2'), _capped('sess-3'), _ok(),
        ])
        with patch(_SLEEP_PATCH, new_callable=AsyncMock):
            await invoke_with_cap_retry(
                _gate(3), 'lbl-opt-out', invoke_fn=inv,
                resume_on_cap_hit=False, prompt=_ORIGINAL,
            )

        assert inv.await_count == 4
        retries = [c.kwargs for c in inv.call_args_list[1:]]
        assert [r.get('resume_session_id') for r in retries] == [None, None, None]
        assert CAP_HIT_RESUME_PROMPT not in [r['prompt'] for r in retries]

    async def test_opt_out_cap_hit_log_names_the_reason(self, caplog):
        inv = AsyncMock(side_effect=[_capped('sess-1'), _ok()])
        with (
            caplog.at_level(logging.WARNING, logger='shared.cli_invoke'),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
        ):
            await invoke_with_cap_retry(
                _gate(1), 'lbl-opt-out', invoke_fn=inv,
                resume_on_cap_hit=False, prompt=_ORIGINAL,
            )

        cap_warnings = [
            r.getMessage() for r in caplog.records
            if r.name == 'shared.cli_invoke'
            and r.levelno == logging.WARNING
            and 'cap hit' in r.getMessage()
        ]
        assert any('opted out' in m for m in cap_warnings), cap_warnings

    async def test_default_still_resumes(self):
        inv = AsyncMock(side_effect=[_capped('sess-1'), _ok()])
        with patch(_SLEEP_PATCH, new_callable=AsyncMock):
            await invoke_with_cap_retry(
                _gate(1), 'lbl-default', invoke_fn=inv, prompt=_ORIGINAL,
            )

        retry = inv.call_args_list[1].kwargs
        assert retry['resume_session_id'] == 'sess-1'
        assert retry['prompt'] == CAP_HIT_RESUME_PROMPT
