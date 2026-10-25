"""``submit_task_batch`` over a real httpx client: what each ``submit_task`` became.

Driven through ``patch('httpx.AsyncClient', ...)``, the constructor seam
``shared.mcp_post.open_mcp_client`` documents, so every reply below crossed a
real client and the real ``shared.mcp_post.call_mcp_tool`` decode.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import httpx
import pytest
from _mcp_transport_harness import (
    RecordingClientFactory,
    RecordingMcpServer,
    tool_reply_envelope,
)

from orchestrator.curator_submit import (
    SubmitBatchResult,
    SubmitOutcome,
    submit_task_batch,
)

BASE_URL = 'http://127.0.0.1:8002'
CONTEXT = 'curator submit_task for task 7165'
LOGGER_NAME = 'orchestrator.curator_submit'
REFUSAL = {
    'error': 'Error executing tool submit_task: 1 validation error for submit_taskArguments',
    'error_type': 'ToolError',
}

#: Captured before any patch is active, so a local factory builds a real client.
_REAL_ASYNC_CLIENT = httpx.AsyncClient


def _arguments(title: str) -> dict:
    return {
        'title': title,
        'description': f'{title} description',
        'priority': 'low',
        'project_root': '/srv/project',
    }


async def _submit_to(server: RecordingMcpServer, arguments_list: list[dict]) -> SubmitBatchResult:
    with patch('httpx.AsyncClient', RecordingClientFactory(server)):
        return await submit_task_batch(BASE_URL, arguments_list, context=CONTEXT)


def _curator_submit_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage() for record in caplog.records
        if record.name == LOGGER_NAME and record.levelno == logging.WARNING
    ]


@pytest.mark.asyncio
async def test_a_ticket_reply_is_a_delivery() -> None:
    server = RecordingMcpServer(tool_replies={'submit_task': {'ticket': 'tkt_A'}})
    arguments = _arguments('first')

    result = await _submit_to(server, [arguments])

    assert [(o.ticket, o.failure) for o in result.outcomes] == [('tkt_A', None)]
    assert result.all_delivered is True
    assert result.tickets == ('tkt_A',)
    assert result.failures == ()
    assert [call['params']['arguments'] for call in server.tool_calls('submit_task')] == [
        arguments,
    ]


@pytest.mark.asyncio
async def test_a_tool_refusal_is_a_loud_failure(caplog: pytest.LogCaptureFixture) -> None:
    server = RecordingMcpServer(tool_replies={'submit_task': REFUSAL})

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        result = await _submit_to(server, [_arguments('refused one')])

    [outcome] = result.outcomes
    assert outcome.ticket is None
    assert outcome.failure is not None and 'ToolError' in outcome.failure
    assert result.all_delivered is False
    assert result.tickets == ()
    [warning] = _curator_submit_warnings(caplog)
    assert CONTEXT in warning
    assert 'refused one' in warning


@pytest.mark.asyncio
async def test_a_reply_without_a_ticket_is_not_a_delivery() -> None:
    result = await _submit_to(RecordingMcpServer(), [_arguments('no ticket')])

    [outcome] = result.outcomes
    assert outcome.ticket is None
    assert outcome.failure is not None
    assert result.all_delivered is False


@pytest.mark.asyncio
async def test_a_transport_failure_fails_only_its_own_submission() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 2:
            raise httpx.ConnectError('connection refused', request=request)
        return httpx.Response(200, json=tool_reply_envelope({'ticket': 'tkt_A'}))

    def client_factory(*args, **kwargs) -> httpx.AsyncClient:
        kwargs.setdefault('transport', httpx.MockTransport(handler))
        return _REAL_ASYNC_CLIENT(*args, **kwargs)

    with patch('httpx.AsyncClient', client_factory):
        result = await submit_task_batch(
            BASE_URL, [_arguments('delivered'), _arguments('lost')], context=CONTEXT,
        )

    delivered, lost = result.outcomes
    assert (delivered.ticket, delivered.failure) == ('tkt_A', None)
    assert lost.ticket is None
    assert lost.failure is not None and 'ConnectError' in lost.failure
    assert result.tickets == ('tkt_A',)
    assert result.all_delivered is False


@pytest.mark.asyncio
async def test_outcomes_follow_argument_order_and_carry_each_title() -> None:
    server = RecordingMcpServer(tool_replies={'submit_task': {'ticket': 'tkt_A'}})
    titles = ['one', 'two', 'three']

    result = await _submit_to(server, [_arguments(title) for title in titles])

    assert [outcome.title for outcome in result.outcomes] == titles


def test_an_outcome_with_neither_ticket_nor_failure_is_refused() -> None:
    with pytest.raises(ValueError) as excinfo:
        SubmitOutcome(title='orphan')

    message = str(excinfo.value)
    assert 'ticket' in message and 'failure' in message


def test_an_outcome_with_both_ticket_and_failure_is_refused() -> None:
    with pytest.raises(ValueError) as excinfo:
        SubmitOutcome(title='torn', ticket='tkt_A', failure='refused')

    message = str(excinfo.value)
    assert 'tkt_A' in message and 'refused' in message
