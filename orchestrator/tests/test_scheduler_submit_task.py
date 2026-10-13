"""``Scheduler.submit_task`` against the shared fused-memory response fixture.

``shared/tests/fixtures/submit_task_responses.json`` pins each response the
``submit_task`` tool returns to the typed outcome the orchestrator must see; the
live cases are asserted against the real server in
``fused-memory/tests/test_task_interceptor.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _orch_helpers import mcp_tool_envelope
from shared.task_submit import SubmitFailed, SubmitOutcome, SubmittedTask, SubmittedTicket

from orchestrator.config import OrchestratorConfig
from orchestrator.scheduler import Scheduler

_FIXTURE = (
    Path(__file__).parents[2] / 'shared' / 'tests' / 'fixtures' / 'submit_task_responses.json'
)
_CASES: dict[str, dict] = json.loads(_FIXTURE.read_text())['cases']


class _CannedMcpSession:
    """Answers every ``call_tool`` with one canned body, or raises ``error``."""

    def __init__(self, body: dict | None = None, error: Exception | None = None) -> None:
        self.body = body or {}
        self.error = error
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict, timeout: float = 30) -> dict:
        self.calls.append((name, arguments))
        if self.error is not None:
            raise self.error
        return self.body


def _wire_body(case: dict) -> dict:
    if 'jsonrpc_result' in case:
        return {'jsonrpc': '2.0', 'id': 1, 'result': case['jsonrpc_result']}
    return mcp_tool_envelope(case['payload'])


def _assert_matches(outcome: SubmitOutcome, expected: dict) -> None:
    kind = expected['kind']
    if kind == 'SubmittedTicket':
        assert outcome == SubmittedTicket(expected['ticket_id'])
    elif kind == 'SubmittedTask':
        assert outcome == SubmittedTask(
            expected['task_id'], expected['status'], expected['combined'],
        )
    else:
        assert kind == 'SubmitFailed'
        assert isinstance(outcome, SubmitFailed), outcome
        assert expected['reason_contains'] in outcome.reason


def _scheduler(tmp_path: Path, session: _CannedMcpSession) -> Scheduler:
    return Scheduler(OrchestratorConfig(project_root=tmp_path), mcp_session=session)


class TestSubmitTaskOutcomes:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('case', _CASES.values(), ids=list(_CASES))
    async def test_fixture_response_yields_its_outcome(self, case, tmp_path):
        scheduler = _scheduler(tmp_path, _CannedMcpSession(_wire_body(case)))

        outcome = await scheduler.submit_task({'title': 'T'})

        _assert_matches(outcome, case['outcome'])

    @pytest.mark.asyncio
    async def test_bare_unwrapped_dict_is_a_failure(self, tmp_path):
        scheduler = _scheduler(tmp_path, _CannedMcpSession({'task_id': '1'}))

        outcome = await scheduler.submit_task({'title': 'T'})

        assert isinstance(outcome, SubmitFailed)

    @pytest.mark.asyncio
    async def test_dispatch_exception_is_a_failure_naming_it(self, tmp_path):
        scheduler = _scheduler(tmp_path, _CannedMcpSession(error=ConnectionError('mcp down')))

        outcome = await scheduler.submit_task({'title': 'T'})

        assert isinstance(outcome, SubmitFailed)
        assert 'ConnectionError' in outcome.reason
        assert 'mcp down' in outcome.reason


class TestSubmitTaskArguments:
    _OK = mcp_tool_envelope({'ticket': 'tkt_abc'})

    @pytest.mark.asyncio
    async def test_project_root_defaults_to_the_schedulers(self, tmp_path):
        session = _CannedMcpSession(self._OK)

        await _scheduler(tmp_path, session).submit_task({'title': 'T'})

        assert session.calls == [
            ('submit_task', {'title': 'T', 'project_root': str(tmp_path)}),
        ]

    @pytest.mark.asyncio
    async def test_caller_project_root_is_kept(self, tmp_path):
        session = _CannedMcpSession(self._OK)

        await _scheduler(tmp_path, session).submit_task(
            {'title': 'T', 'project_root': '/elsewhere'},
        )

        assert session.calls[0][1]['project_root'] == '/elsewhere'

    @pytest.mark.asyncio
    async def test_callers_dict_is_not_mutated(self, tmp_path):
        arguments = {'title': 'T'}

        await _scheduler(tmp_path, _CannedMcpSession(self._OK)).submit_task(arguments)

        assert arguments == {'title': 'T'}
