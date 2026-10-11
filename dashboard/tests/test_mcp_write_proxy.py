"""Every MCP-write route renders tool results and failures identically.

Driven through the public HTTP seam only: each route is posted a valid body
while ``memory.mcp_tool_call`` is scripted to answer that route's tool.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock, patch

import httpx
import pytest

_PATCH_TARGET = 'dashboard.data.memory.mcp_tool_call'

_WRITE_ROUTES = [
    pytest.param(
        '/api/v2/dashboard/curator/cancel',
        {'ticket_id': 'tkt_abc'},
        'cancel_ticket',
        id='curator-cancel',
    ),
    pytest.param(
        '/api/v2/dashboard/scheduler/override',
        {'task_id': 'T1', 'project_root': '/proj'},
        'set_task_priority_override',
        id='scheduler-override',
    ),
    pytest.param(
        '/api/v2/dashboard/scheduler/clear-override',
        {'task_id': 'T1', 'project_root': '/proj'},
        'clear_task_priority_override',
        id='scheduler-clear-override',
    ),
    pytest.param(
        '/api/v2/dashboard/scheduler/reorder-pin-queue',
        {'task_ids': ['T1'], 'project_root': '/proj'},
        'reorder_pin_queue',
        id='scheduler-reorder-pin-queue',
    ),
    pytest.param(
        '/api/v2/dashboard/scheduler/evict-park',
        {'task_id': 'T1', 'project_root': '/proj'},
        'request_park_eviction',
        id='scheduler-evict-park',
    ),
]

_DETAIL_CAUSE_CHAR_CAP = 200


def _http_status_error(message: str) -> httpx.HTTPStatusError:
    request = httpx.Request('POST', 'http://x')
    return httpx.HTTPStatusError(
        message, request=request, response=httpx.Response(500, request=request)
    )


_RENDERED_FAILURES = [
    pytest.param(httpx.ConnectError('refused'), 'ConnectError: refused', id='ConnectError'),
    pytest.param(
        httpx.TimeoutException('timed out'),
        'TimeoutException: timed out',
        id='TimeoutException',
    ),
    pytest.param(
        _http_status_error('server error'), 'HTTPStatusError: server error', id='HTTPStatusError'
    ),
    pytest.param(ValueError('X' * 300), 'ValueError: ' + 'X' * 300, id='long-ValueError'),
    pytest.param(
        _http_status_error('X' * 300),
        'HTTPStatusError: ' + 'X' * 300,
        id='long-HTTPStatusError',
    ),
]


def _scripted(tool_name: str, outcome: object) -> Callable[..., Awaitable[object]]:
    """Answer *tool_name* with *outcome* (raised if an exception); every other tool gets ``{}``.

    The lifespan's background samplers dial the same patched seam, so they are
    kept inert rather than sharing the route's outcome.
    """

    async def _scripted_mcp_tool_call(
        client: object, url: str, called_tool: str, arguments: object, **kwargs: object
    ) -> object:
        if called_tool != tool_name:
            return {}
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    return _scripted_mcp_tool_call


def _post_with_tool_outcome(
    client, path: str, body: dict, tool_name: str, outcome: object
) -> httpx.Response:
    with patch(_PATCH_TARGET, new=AsyncMock(side_effect=_scripted(tool_name, outcome))):
        return client.post(path, json=body)


def _call_site_warnings(caplog: pytest.LogCaptureFixture, tool_name: str) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == 'dashboard.app'
        and r.levelno == logging.WARNING
        and r.getMessage().startswith(f'{tool_name} failed for ')
    ]


def _only_fused_memory_url(client) -> str:
    (url,) = client.app.state.config.fused_memory_urls
    return url


@pytest.mark.parametrize(('path', 'body', 'tool_name'), _WRITE_ROUTES)
def test_non_dict_tool_result_is_forwarded_verbatim(client, path, body, tool_name):
    resp = _post_with_tool_outcome(client, path, body, tool_name, ['unexpected', 'list'])

    assert resp.status_code == 200
    assert resp.json() == ['unexpected', 'list']


@pytest.mark.parametrize(('path', 'body', 'tool_name'), _WRITE_ROUTES)
def test_failure_with_empty_message_names_the_exception_type(
    client, caplog, path, body, tool_name
):
    url = _only_fused_memory_url(client)
    with caplog.at_level(logging.DEBUG):
        resp = _post_with_tool_outcome(client, path, body, tool_name, httpx.PoolTimeout(''))

    assert resp.status_code == 502
    assert resp.json()['error'] == 'fused_memory_unreachable'
    assert resp.json()['detail'] == f'{url}: PoolTimeout'
    assert _call_site_warnings(caplog, tool_name) == [f'{tool_name} failed for {url}: PoolTimeout']
    assert not [
        r
        for r in caplog.records
        if r.name == 'dashboard.data.mcp_fanout' and tool_name in r.getMessage()
    ], 'log_failures=False must leave reporting entirely to the call site'


@pytest.mark.parametrize(('exc', 'rendered_cause'), _RENDERED_FAILURES)
@pytest.mark.parametrize(('path', 'body', 'tool_name'), _WRITE_ROUTES)
def test_failure_names_its_type_once_capped_in_the_502_detail_but_whole_in_the_warning(
    client, caplog, path, body, tool_name, exc, rendered_cause
):
    url = _only_fused_memory_url(client)
    with caplog.at_level(logging.WARNING, logger='dashboard.app'):
        resp = _post_with_tool_outcome(client, path, body, tool_name, exc)

    assert resp.status_code == 502
    assert resp.json()['detail'] == f'{url}: {rendered_cause[:_DETAIL_CAUSE_CHAR_CAP]}'
    assert _call_site_warnings(caplog, tool_name) == [
        f'{tool_name} failed for {url}: {rendered_cause}'
    ]
