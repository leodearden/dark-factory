"""``submit_task_batch`` over a real httpx client: what each ``submit_task`` became.

Driven through ``patch('httpx.AsyncClient', ...)``, the constructor seam
``shared.mcp_post.open_mcp_client`` documents, so every reply below crossed a
real client and the real ``shared.mcp_post.call_mcp_tool`` decode.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
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


def _client_factory(handler: Callable) -> Callable[..., httpx.AsyncClient]:
    """An ``httpx.AsyncClient`` stand-in whose every request *handler* answers."""
    def build(*args, **kwargs) -> httpx.AsyncClient:
        kwargs.setdefault('transport', httpx.MockTransport(handler))
        return _REAL_ASYNC_CLIENT(*args, **kwargs)
    return build


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


@pytest.mark.parametrize('reply', [{'success': True}, {'task_id': '1'}])
@pytest.mark.asyncio
async def test_a_reply_without_a_ticket_is_not_a_delivery(reply: dict) -> None:
    server = RecordingMcpServer(tool_replies={'submit_task': reply})

    result = await _submit_to(server, [_arguments('no ticket')])

    [outcome] = result.outcomes
    assert outcome.ticket is None
    assert outcome.failure is not None and 'no ticket' in outcome.failure
    assert result.all_delivered is False


@pytest.mark.asyncio
async def test_an_empty_envelope_is_not_a_delivery() -> None:
    result = await _submit_to(RecordingMcpServer(), [_arguments('empty envelope')])

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

    with patch('httpx.AsyncClient', _client_factory(handler)):
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
async def test_a_hung_server_fails_every_unanswered_submission_at_the_deadline(
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: list[httpx.Request] = []
    never_answered = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 2:
            await never_answered.wait()
        return httpx.Response(200, json=tool_reply_envelope({'ticket': 'tkt_A'}))

    titles = ['delivered', 'in flight', 'never sent']
    with (
        caplog.at_level(logging.WARNING, logger=LOGGER_NAME),
        patch('httpx.AsyncClient', _client_factory(handler)),
    ):
        result = await asyncio.wait_for(
            submit_task_batch(
                BASE_URL, [_arguments(title) for title in titles],
                context=CONTEXT, deadline_secs=0.2,
            ),
            timeout=5,
        )

    assert [outcome.title for outcome in result.outcomes] == titles
    assert result.tickets == ('tkt_A',)
    assert [
        outcome.failure is not None and 'deadline' in outcome.failure
        for outcome in result.failures
    ] == [True, True]
    assert len(calls) == 2
    assert len(_curator_submit_warnings(caplog)) == 2


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
