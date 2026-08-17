"""Tests for shared.mcp_post — the slash-less MCP endpoint + loud response check.

Task 4023.  Five raw ``httpx`` POSTs in the orchestrator targeted
``f'{base}/mcp/'`` on a bare ``AsyncClient``.  The fused-memory server
307-redirects ``/mcp/`` -> ``/mcp``, and a bare client does NOT follow
redirects — so every one of those writes was silently discarded for months
while the HTTP exchange itself reported perfect success.

TDD pair 1: ``mcp_endpoint_url`` canonicalization (GREEN on impl step-2).
TDD pair 2: ``check_mcp_post_response`` loud-and-never-raising (GREEN on step-4).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx
import pytest

from shared.mcp_post import check_mcp_post_response, mcp_endpoint_url

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Every WARNING this module emits must be attributable to the shared
#: primitive, not to whichever call site happened to invoke it.
LOGGER_NAME = 'shared.mcp_post'


# ---------------------------------------------------------------------------
# Pair 1 — mcp_endpoint_url (step-1 RED / step-2 GREEN)
# ---------------------------------------------------------------------------


def test_endpoint_url_has_no_trailing_slash():
    """The canonical endpoint is ``/mcp`` — the trailing slash is the defect."""
    assert mcp_endpoint_url('http://localhost:8002') == 'http://localhost:8002/mcp'


def test_endpoint_url_never_ends_in_slash_for_any_base():
    """No base shape may produce a trailing slash — that is what 307s."""
    for base in (
        'http://localhost:8002',
        'http://localhost:8002/',
        'http://127.0.0.1:8102',
        'https://memory.example.test',
    ):
        url = mcp_endpoint_url(base)
        assert not url.endswith('/'), f'{base!r} produced a redirect-triggering {url!r}'
        assert url.endswith('/mcp')


def test_endpoint_url_collapses_a_base_that_already_ends_in_slash():
    """A configured base ending in ``/`` must not yield ``//mcp``.

    Mirrors ``McpSession.__init__``'s ``base_url.rstrip('/')`` — the rstrip is
    load-bearing, not cosmetic.
    """
    assert mcp_endpoint_url('http://localhost:8002/') == 'http://localhost:8002/mcp'
    assert '//mcp' not in mcp_endpoint_url('http://localhost:8002/')
    # Multiple trailing slashes collapse too.
    assert mcp_endpoint_url('http://localhost:8002///') == 'http://localhost:8002/mcp'


def test_endpoint_url_matches_the_form_mcp_json_declares():
    """The result must equal the slash-less URL ``.mcp.json`` already declares.

    ``.mcp.json`` is the convergence target: the Claude CLI's own MCP client
    reaches fused-memory at ``http://127.0.0.1:8002/mcp`` with no trailing
    slash, and so must every raw POST.
    """
    declared = json.loads((REPO_ROOT / '.mcp.json').read_text())['mcpServers']
    urls = [
        server['url']
        for server in declared.values()
        if isinstance(server, dict) and str(server.get('url', '')).endswith('/mcp')
    ]
    assert urls, 'expected .mcp.json to declare at least one HTTP MCP server'
    for url in urls:
        base = url[: -len('/mcp')].rstrip('/')
        assert mcp_endpoint_url(base) == url, (
            f'helper must reproduce the declared endpoint {url!r} exactly'
        )


# ---------------------------------------------------------------------------
# Pair 2 — check_mcp_post_response (step-3 RED / step-4 GREEN)
# ---------------------------------------------------------------------------


def _response(status, *, json_body=None, text=None, headers=None):
    """A real ``httpx.Response`` bound to a real POST request."""
    request = httpx.Request('POST', 'http://memory.test:8002/mcp')
    kwargs = {'request': request, 'headers': headers or {}}
    if json_body is not None:
        kwargs['json'] = json_body
    elif text is not None:
        kwargs['text'] = text
    return httpx.Response(status, **kwargs)


def _warnings(caplog):
    return [
        r for r in caplog.records
        if r.levelno >= logging.WARNING and r.name == LOGGER_NAME
    ]


def test_redirect_returns_false_and_warns_about_the_unfollowed_redirect(caplog):
    """A 307 is THIS TASK'S defect and gets its own distinct WARNING.

    A redirect is a perfectly successful HTTP exchange, which is exactly why
    the original bug was invisible: nothing raised, nothing logged, and the
    payload went nowhere.
    """
    resp = _response(307, headers={'Location': 'http://memory.test:8002/mcp'})

    with caplog.at_level(logging.WARNING):
        ok = check_mcp_post_response(resp, context='completion memory write for task 4023')

    assert ok is False, 'a 307 must not be reported as a delivered POST'
    warnings = _warnings(caplog)
    assert warnings, (
        f'Expected a WARNING from {LOGGER_NAME}; '
        f'got records={[(r.name, r.getMessage()) for r in caplog.records]!r}'
    )
    message = ' '.join(r.getMessage() for r in warnings).lower()
    assert 'redirect' in message, (
        'the redirect case needs its OWN signal, not a generic transport warning'
    )
    assert '307' in message, 'the warning must name the status actually received'


def test_redirect_warning_names_its_call_site_context(caplog):
    """``context`` is interpolated so a warning identifies which of the five sites fired."""
    resp = _response(308, headers={'Location': '/mcp'})

    with caplog.at_level(logging.WARNING):
        check_mcp_post_response(resp, context='curator submit_task for task 4023')

    message = ' '.join(r.getMessage() for r in _warnings(caplog))
    assert 'curator submit_task for task 4023' in message


def test_server_error_returns_false_and_warns(caplog):
    """Any >=400 is a failed write and must be loud."""
    resp = _response(500, text='boom')

    with caplog.at_level(logging.WARNING):
        ok = check_mcp_post_response(resp, context='decisions memory write')

    assert ok is False
    warnings = _warnings(caplog)
    assert warnings, 'a 500 must emit a WARNING'
    assert '500' in ' '.join(r.getMessage() for r in warnings)


def test_jsonrpc_error_member_returns_false_and_warns(caplog):
    """A 200 whose JSON-RPC body carries ``error`` is still a failed write."""
    resp = _response(
        200,
        json_body={'jsonrpc': '2.0', 'id': 1, 'error': {'code': -32602, 'message': 'bad params'}},
    )

    with caplog.at_level(logging.WARNING):
        ok = check_mcp_post_response(resp, context='suggestions memory write')

    assert ok is False, 'HTTP 200 + JSON-RPC error is a failure, not a success'
    warnings = _warnings(caplog)
    assert warnings, 'a JSON-RPC error member must emit a WARNING'
    assert 'bad params' in ' '.join(r.getMessage() for r in warnings), (
        'the warning must carry the server-supplied error so it is diagnosable'
    )


def test_successful_result_returns_true_and_logs_nothing(caplog):
    """The happy path must be silent — otherwise the warnings are noise."""
    resp = _response(200, json_body={'jsonrpc': '2.0', 'id': 1, 'result': {'content': []}})

    with caplog.at_level(logging.WARNING):
        ok = check_mcp_post_response(resp, context='completion memory write')

    assert ok is True
    assert _warnings(caplog) == [], 'a delivered POST must not warn'


def test_undecodable_body_returns_false_and_warns_instead_of_raising(caplog):
    """A 2xx with a non-JSON body warns; it never propagates a JSONDecodeError."""
    resp = _response(200, text='<html>not json at all</html>')

    with caplog.at_level(logging.WARNING):
        ok = check_mcp_post_response(resp, context='completion memory write')

    assert ok is False
    assert _warnings(caplog), 'an undecodable body must emit a WARNING'


@pytest.mark.parametrize(
    'resp',
    [
        None,
        object(),
        'not a response',
        123,
        _response(204),
        _response(200, json_body=['a', 'list', 'not', 'a', 'dict']),
        _response(200, text=''),
        _response(301, headers={}),  # a redirect with no Location header
        _response(200, json_body={'error': None}),  # explicit null error member
        _response(404, text='no such route'),
    ],
    ids=[
        'none', 'bare-object', 'str', 'int', 'no-content',
        'list-body', 'empty-body', 'redirect-without-location',
        'null-error-member', 'not-found',
    ],
)
def test_no_input_shape_ever_raises(resp, caplog):
    """THE contract: all five call sites are fire-and-forget.

    ``_post_submit_tasks`` runs under ``asyncio.create_task`` and the memory
    writes are best-effort side channels whose failure must never fail a task.
    A checker that raises on a duck-typed or malformed response would convert
    a silent no-op into a new failure mode.
    """
    with caplog.at_level(logging.WARNING):
        result = check_mcp_post_response(resp, context='fuzz')

    assert isinstance(result, bool), 'must always return a bool, never None'


def test_a_null_jsonrpc_error_member_is_not_treated_as_an_error(caplog):
    """``{'error': None}`` is the JSON-RPC absent-error spelling, not a failure."""
    resp = _response(200, json_body={'jsonrpc': '2.0', 'id': 1, 'error': None, 'result': {}})

    with caplog.at_level(logging.WARNING):
        ok = check_mcp_post_response(resp, context='completion memory write')

    assert ok is True
    assert _warnings(caplog) == []
