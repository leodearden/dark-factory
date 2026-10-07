"""Tests for shared.mcp_post — the slash-less MCP endpoint + loud response check.

Task 4023.  Five raw ``httpx`` POSTs in the orchestrator targeted
``f'{base}/mcp/'`` on a bare ``AsyncClient``.  The fused-memory server
307-redirects ``/mcp/`` -> ``/mcp``, and a bare client does NOT follow
redirects — so every one of those writes was silently discarded for months
while the HTTP exchange itself reported perfect success.

TDD pair 1: ``mcp_endpoint_url`` canonicalization (GREEN on impl step-2).
TDD pair 2: ``check_mcp_post_response`` loud-and-never-raising (GREEN on step-4).
Pair 2b + 3: the SSE decode path and the composed ``post_mcp_tool_call`` /
``open_mcp_client`` primitives (amendment pass, reviewer suggestions 3 and 4).
Pair 4: ``decode_mcp_response_body``, the public decoder ``McpSession``
consumes (task 4819).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx
import pytest

from shared.mcp_post import (
    MCP_POST_HEADERS,
    call_mcp_tool,
    check_mcp_post_response,
    decode_mcp_response_body,
    mcp_endpoint_url,
    mcp_tool_call_payload,
    open_mcp_client,
    post_mcp_tool_call,
)

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


# ---------------------------------------------------------------------------
# Pair 2b — the SSE decode path (amendment: reviewer suggestion 3)
# ---------------------------------------------------------------------------
#
# WHY THIS BRANCH IS NOT OPTIONAL COVERAGE.  The fix adds
# ``Accept: application/json, text/event-stream`` to all five POSTs — that
# header is what INVITES FastMCP to answer in SSE at all.  So the change made
# ``decode_mcp_response_body``'s ``text/event-stream`` branch newly reachable in
# production while nothing exercised it.  If ``_parse_sse`` were wrong, every
# SUCCESSFUL write would emit a "could not be inspected" WARNING: the exact
# inverse of this module's loud-only-on-real-failure contract, and
# indistinguishable in the logs from an actually lost write.
#
# ``test_no_input_shape_ever_raises`` above does NOT close this — none of its
# ten cases carries an SSE content-type.

SSE_HEADERS = {'content-type': 'text/event-stream'}


def _sse_body(payload: dict, *, prefix: str = 'data: ') -> str:
    """An SSE frame the way FastMCP writes one: an ``event:`` line then ``data:``."""
    return f'event: message\n{prefix}{json.dumps(payload)}\n\n'


def test_sse_success_returns_true_and_logs_nothing(caplog):
    """A successful write answered in SSE must be silent, not warn.

    This is the case that turns a wrong ``_parse_sse`` into log noise on every
    healthy write rather than into a visible failure.
    """
    resp = _response(
        200,
        text=_sse_body({'jsonrpc': '2.0', 'id': 1, 'result': {'content': []}}),
        headers=SSE_HEADERS,
    )

    with caplog.at_level(logging.WARNING):
        ok = check_mcp_post_response(resp, context='completion memory write for task 4023')

    assert ok is True, 'an SSE-framed JSON-RPC result is a DELIVERED write'
    assert _warnings(caplog) == [], (
        'a successful SSE answer must not warn — that warning would be '
        'indistinguishable from a genuinely lost write'
    )


def test_sse_frame_without_the_space_after_data_is_still_parsed(caplog):
    """``data:{...}`` — the no-space spelling — must decode identically.

    ``_parse_sse`` carries a ``'data: '`` / ``'data:'`` prefix pair; only the
    first was exercised.  The SSE spec makes the single leading space optional,
    so a server emitting the tight form would otherwise warn on every success.
    """
    resp = _response(
        200,
        text=_sse_body({'jsonrpc': '2.0', 'id': 1, 'result': {}}, prefix='data:'),
        headers=SSE_HEADERS,
    )

    with caplog.at_level(logging.WARNING):
        ok = check_mcp_post_response(resp, context='completion memory write')

    assert ok is True
    assert _warnings(caplog) == []


def test_sse_jsonrpc_error_member_returns_false_and_warns(caplog):
    """An application-level failure delivered over SSE is still a failed write.

    Pins that the SSE branch feeds the SAME error inspection as the JSON
    branch — a decoder that returned the raw text would make every SSE-framed
    JSON-RPC error read as success.
    """
    resp = _response(
        200,
        text=_sse_body({
            'jsonrpc': '2.0',
            'id': 1,
            'error': {'code': -32602, 'message': 'bad params over sse'},
        }),
        headers=SSE_HEADERS,
    )

    with caplog.at_level(logging.WARNING):
        ok = check_mcp_post_response(resp, context='suggestions memory write')

    assert ok is False
    assert 'bad params over sse' in ' '.join(r.getMessage() for r in _warnings(caplog))


def test_sse_last_data_line_wins(caplog):
    """FastMCP may emit several frames; the JSON-RPC answer is the LAST one."""
    resp = _response(
        200,
        text=(
            'event: message\ndata: {"jsonrpc":"2.0","id":1,"result":{"partial":true}}\n\n'
            'event: message\ndata: {"jsonrpc":"2.0","id":1,'
            '"error":{"code":-1,"message":"final frame lost"}}\n\n'
        ),
        headers=SSE_HEADERS,
    )

    with caplog.at_level(logging.WARNING):
        ok = check_mcp_post_response(resp, context='decisions memory write')

    assert ok is False, 'the LAST data frame decides the outcome, not the first'
    assert 'final frame lost' in ' '.join(r.getMessage() for r in _warnings(caplog))


def test_sse_labelled_body_with_no_data_line_warns_instead_of_raising(caplog):
    """``_parse_sse``'s ValueError must surface as a WARNING, never propagate.

    An SSE content-type with no ``data:`` line at all is malformed; the
    never-raises contract still holds because every call site is
    fire-and-forget.
    """
    resp = _response(
        200,
        text='event: ping\nretry: 3000\n\n',
        headers=SSE_HEADERS,
    )

    with caplog.at_level(logging.WARNING):
        ok = check_mcp_post_response(resp, context='completion memory write')

    assert ok is False
    warnings = _warnings(caplog)
    assert warnings, 'a malformed SSE body must warn'
    assert 'could not be inspected' in ' '.join(r.getMessage() for r in warnings)


def test_an_unlabelled_sse_body_is_still_decoded(caplog):
    """A body that IS SSE but is not labelled falls back to the SSE spelling.

    ``decode_mcp_response_body`` tries ``resp.json()`` first and only then ``_parse_sse``.
    Pins that fallback so a server sending SSE under ``application/json`` (or
    no content-type at all) does not warn on a delivered write.
    """
    resp = _response(
        200,
        text=_sse_body({'jsonrpc': '2.0', 'id': 1, 'result': {}}),
    )

    with caplog.at_level(logging.WARNING):
        ok = check_mcp_post_response(resp, context='completion memory write')

    assert ok is True
    assert _warnings(caplog) == []


# ---------------------------------------------------------------------------
# Pair 3 — the composed primitives (amendment: reviewer suggestion 4)
# ---------------------------------------------------------------------------
#
# The three ingredients alone left the failure this module exists to close
# still reachable: a sixth call site applying THREE of the four parts (a
# slash-less URL and follow_redirects, but no Accept header) still discards
# its payload — to a silent 406 instead of a silent 307.  These pin that the
# composed form applies all four as a unit, so they cannot be partially
# applied.


class _RecordingClient:
    """Minimal ``AsyncClient`` stand-in recording exactly what was sent."""

    def __init__(self, response=None, *, follow_redirects=True):
        self.follow_redirects = follow_redirects
        self.calls: list[dict] = []
        self._response = response or _response(
            200, json_body={'jsonrpc': '2.0', 'id': 1, 'result': {}}
        )

    async def post(self, url, *, headers=None, json=None, timeout=None):
        self.calls.append(
            {'url': url, 'headers': headers, 'json': json, 'timeout': timeout}
        )
        return self._response


def test_tool_call_payload_is_a_wellformed_jsonrpc_envelope():
    """The envelope shape all five call sites used to hand-write."""
    payload = mcp_tool_call_payload('add_memory', {'content': 'hi'})

    assert payload == {
        'jsonrpc': '2.0',
        'id': 1,
        'method': 'tools/call',
        'params': {'name': 'add_memory', 'arguments': {'content': 'hi'}},
    }


@pytest.mark.asyncio
async def test_post_mcp_tool_call_applies_all_four_parts_at_once():
    """THE point of the composed primitive: URL, headers and check together.

    A caller cannot obtain the slash-less URL without also obtaining the
    ``Accept`` header, which is what makes a partially-applied fix (the 406
    variant of this task's defect) unreachable through this path.
    """
    client = _RecordingClient()

    ok = await post_mcp_tool_call(
        client, 'http://memory.test:8002', 'submit_task', {'title': 't'},
        context='curator submit_task for task 4023',
    )

    assert ok is True
    assert len(client.calls) == 1
    sent = client.calls[0]
    assert sent['url'] == 'http://memory.test:8002/mcp', 'part 1: no trailing slash'
    assert sent['headers'] == MCP_POST_HEADERS, 'part 2: the Accept header'
    assert sent['json']['params'] == {'name': 'submit_task', 'arguments': {'title': 't'}}
    assert sent['timeout'] == 10


@pytest.mark.asyncio
async def test_post_mcp_tool_call_collapses_a_base_that_ends_in_a_slash():
    """The composed form inherits ``mcp_endpoint_url``'s canonicalization."""
    client = _RecordingClient()

    await post_mcp_tool_call(
        client, 'http://memory.test:8002/', 'add_memory', {}, context='fuzz',
    )

    assert client.calls[0]['url'] == 'http://memory.test:8002/mcp'


@pytest.mark.asyncio
async def test_post_mcp_tool_call_reports_a_failed_write(caplog):
    """Part 4 is applied too: a 307 answer returns False and warns."""
    client = _RecordingClient(
        response=_response(307, headers={'Location': '/mcp'}),
    )

    with caplog.at_level(logging.WARNING):
        ok = await post_mcp_tool_call(
            client, 'http://memory.test:8002', 'add_memory', {},
            context='completion memory write for task 4023',
        )

    assert ok is False
    assert 'redirect' in ' '.join(r.getMessage() for r in _warnings(caplog)).lower()


@pytest.mark.asyncio
async def test_post_mcp_tool_call_warns_when_the_client_ignores_redirects(caplog):
    """Part 3 lives on the client, so a client missing it must be LOUD.

    Client construction is deliberately a separate call — three of the five
    sites share one connection pool across a batch — so this warning is what
    keeps splitting it out from reopening the partial-application hole.
    """
    client = _RecordingClient(follow_redirects=False)

    with caplog.at_level(logging.WARNING):
        await post_mcp_tool_call(
            client, 'http://memory.test:8002', 'add_memory', {}, context='fuzz',
        )

    message = ' '.join(r.getMessage() for r in _warnings(caplog))
    assert 'follow_redirects' in message
    assert 'open_mcp_client' in message, 'the warning must name the fix'


@pytest.mark.asyncio
async def test_post_mcp_tool_call_does_not_warn_for_a_stub_without_the_attribute(caplog):
    """A duck-typed double with no ``follow_redirects`` must not warn spuriously.

    Several existing test doubles are plain objects; a warning on every one of
    them would be noise that trains readers to ignore this logger.
    """
    class _Bare:
        async def post(self, url, *, headers=None, json=None, timeout=None):
            return _response(200, json_body={'jsonrpc': '2.0', 'id': 1, 'result': {}})

    with caplog.at_level(logging.WARNING):
        ok = await post_mcp_tool_call(
            _Bare(), 'http://memory.test:8002', 'add_memory', {}, context='fuzz',
        )

    assert ok is True
    assert _warnings(caplog) == []


@pytest.mark.asyncio
async def test_open_mcp_client_follows_redirects_by_default():
    """Part 3, applied by construction rather than remembered per call site.

    Asserted on a REAL ``httpx.AsyncClient`` (not the monkeypatched factory
    below) so this stays true of httpx's actual constructor semantics, and
    closed via ``async with`` so no transport is left open.
    """
    async with open_mcp_client() as client:
        assert isinstance(client, httpx.AsyncClient)
        assert client.follow_redirects is True


def test_open_mcp_client_resolves_asyncclient_at_call_time(monkeypatch):
    """``patch('httpx.AsyncClient', ...)`` must still intercept construction.

    That constructor seam is what every transport test in this repo drives.  A
    future rewrite binding ``AsyncClient`` at import time — ``from httpx import
    AsyncClient`` at module scope — would silently blind all of them while
    every test kept passing, so the late lookup is pinned here rather than
    left as a property of how the import happens to be spelled.
    """
    built: list[dict] = []

    def _factory(**kwargs):
        built.append(kwargs)
        return 'sentinel-client'

    monkeypatch.setattr(httpx, 'AsyncClient', _factory)

    assert open_mcp_client() == 'sentinel-client'
    assert built == [{'follow_redirects': True}]


def test_open_mcp_client_forwards_extra_kwargs(monkeypatch):
    """``follow_redirects`` is a DEFAULT, and other kwargs pass through."""
    built: list[dict] = []
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kw: built.append(kw))

    open_mcp_client(timeout=5, follow_redirects=False)

    assert built == [{'timeout': 5, 'follow_redirects': False}]


# ---------------------------------------------------------------------------
# Pair 4 — decode_mcp_response_body, the public decoder McpSession consumes (task 4819)
# ---------------------------------------------------------------------------

_RESULT = {'jsonrpc': '2.0', 'id': 1, 'result': {'content': []}}


def test_decode_returns_a_json_labelled_json_body():
    """A JSON body labelled application/json decodes to that object."""
    assert decode_mcp_response_body(_response(200, json_body=_RESULT)) == _RESULT


@pytest.mark.parametrize('prefix', ['data: ', 'data:'], ids=['spaced', 'tight'])
def test_decode_parses_an_sse_body_with_either_data_prefix(prefix):
    """An SSE body decodes to its data frame, with or without the optional space."""
    resp = _response(200, text=_sse_body(_RESULT, prefix=prefix), headers=SSE_HEADERS)

    assert decode_mcp_response_body(resp) == _RESULT


def test_decode_takes_the_last_sse_data_frame():
    """Of several SSE frames, the last data frame is the answer."""
    final = {'jsonrpc': '2.0', 'id': 1, 'result': {'final': True}}
    resp = _response(
        200,
        text=_sse_body({'jsonrpc': '2.0', 'id': 1, 'result': {'partial': True}})
        + _sse_body(final),
        headers=SSE_HEADERS,
    )

    assert decode_mcp_response_body(resp) == final


@pytest.mark.parametrize(
    'headers',
    [None, {'content-type': 'application/json'}],
    ids=['unlabelled', 'mislabelled-as-json'],
)
def test_decode_falls_back_to_sse_for_an_unlabelled_or_mislabelled_body(headers):
    """An SSE body without the SSE label still decodes via the SSE fallback."""
    resp = _response(200, text=_sse_body(_RESULT), headers=headers)

    assert decode_mcp_response_body(resp) == _RESULT


@pytest.mark.parametrize(
    ('text', 'headers'),
    [
        ('event: ping\nretry: 3000\n\n', SSE_HEADERS),
        ('not json and not sse', None),
    ],
    ids=['sse-without-data', 'neither-json-nor-sse'],
)
def test_decode_raises_value_error_when_no_data_line(text, headers):
    """The decoder RAISES on an undecodable body; only the checker swallows it."""
    with pytest.raises(ValueError):
        decode_mcp_response_body(_response(200, text=text, headers=headers))


# ---------------------------------------------------------------------------
# Pair 5 — call_mcp_tool, the reply-returning twin of post_mcp_tool_call
# (task 6181)
# ---------------------------------------------------------------------------
#
# post_mcp_tool_call answers only "did it land", so a caller that needs the
# tool's reply (a read's payload, a write's error_type) had no composed path.
# call_mcp_tool shares the same four-part send and returns the decoded reply.

#: The decoder (shared.mcp_envelope) warns under its own logger name.
SHARED_LOGGERS = {LOGGER_NAME, 'shared.mcp_envelope'}


def _shared_warnings(caplog):
    return [
        r for r in caplog.records
        if r.levelno >= logging.WARNING and r.name in SHARED_LOGGERS
    ]


def _tool_reply(text: str, *, is_error: bool = False) -> dict:
    result: dict = {'content': [{'type': 'text', 'text': text}]}
    if is_error:
        result['isError'] = True
    return {'jsonrpc': '2.0', 'id': 1, 'result': result}


_FOUND = {'found': True, 'content': 'x'}


@pytest.mark.asyncio
async def test_call_mcp_tool_applies_all_four_parts_at_once():
    client = _RecordingClient(
        response=_response(200, json_body=_tool_reply(json.dumps(_FOUND))),
    )

    await call_mcp_tool(
        client, 'http://memory.test:8002/', 'get_memory_by_id', {'memory_id': 'm'},
        context='link-heal get_memory_by_id', request_id=7, timeout=3,
    )

    assert len(client.calls) == 1
    sent = client.calls[0]
    assert sent['url'] == 'http://memory.test:8002/mcp'
    assert sent['headers'] == MCP_POST_HEADERS
    assert sent['json'] == mcp_tool_call_payload(
        'get_memory_by_id', {'memory_id': 'm'}, request_id=7,
    )
    assert sent['timeout'] == 3


@pytest.mark.asyncio
@pytest.mark.parametrize('framing', ['json', 'sse'])
async def test_call_mcp_tool_returns_the_tool_reply_dict(framing, caplog):
    envelope = _tool_reply(json.dumps(_FOUND))
    if framing == 'json':
        resp = _response(200, json_body=envelope)
    else:
        resp = _response(200, text=_sse_body(envelope), headers=SSE_HEADERS)

    with caplog.at_level(logging.WARNING):
        reply = await call_mcp_tool(
            _RecordingClient(response=resp), 'http://memory.test:8002',
            'get_memory_by_id', {}, context='fuzz',
        )

    assert reply == _FOUND
    assert _shared_warnings(caplog) == []


@pytest.mark.asyncio
async def test_call_mcp_tool_returns_an_error_type_reply_unchanged():
    """A tool-level rejection is a successful call; its reply must survive so
    the caller can read the rejection class."""
    rejection = {'error': 'no', 'error_type': 'Mem0UpdateNotAuthorized'}
    client = _RecordingClient(
        response=_response(200, json_body=_tool_reply(json.dumps(rejection))),
    )

    reply = await call_mcp_tool(
        client, 'http://memory.test:8002', 'update_memory', {}, context='fuzz',
    )

    assert reply == rejection


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'resp',
    [
        _response(307, headers={'Location': '/mcp'}),
        _response(406, json_body={'error': {'code': -32600, 'message': 'Not Acceptable'}}),
        _response(
            200,
            json_body={'jsonrpc': '2.0', 'id': 1, 'error': {'code': -32601, 'message': 'x'}},
        ),
        _response(200, json_body=_tool_reply('Error calling tool', is_error=True)),
        _response(200, json_body=_tool_reply('not json at all')),
        _response(200, json_body=_tool_reply(json.dumps(['a', 'list']))),
    ],
    ids=[
        'redirect-307', 'http-406', 'jsonrpc-error', 'result-isError',
        'text-not-json', 'json-not-an-object',
    ],
)
async def test_call_mcp_tool_returns_none_and_warns_when_no_reply(resp, caplog):
    with caplog.at_level(logging.WARNING):
        reply = await call_mcp_tool(
            _RecordingClient(response=resp), 'http://memory.test:8002',
            'get_memory_by_id', {}, context='link-heal get_memory_by_id',
        )

    assert reply is None
    assert _shared_warnings(caplog), 'a missing reply must never be silent'


@pytest.mark.asyncio
async def test_call_mcp_tool_propagates_a_transport_exception():
    class _Down:
        follow_redirects = True

        async def post(self, url, *, headers=None, json=None, timeout=None):
            raise httpx.ConnectError('connection refused')

    with pytest.raises(httpx.ConnectError):
        await call_mcp_tool(
            _Down(), 'http://memory.test:8002', 'get_memory_by_id', {}, context='fuzz',
        )


@pytest.mark.asyncio
async def test_call_mcp_tool_warns_when_the_client_ignores_redirects(caplog):
    client = _RecordingClient(
        response=_response(200, json_body=_tool_reply(json.dumps(_FOUND))),
        follow_redirects=False,
    )

    with caplog.at_level(logging.WARNING):
        await call_mcp_tool(
            client, 'http://memory.test:8002', 'get_memory_by_id', {}, context='fuzz',
        )

    message = ' '.join(r.getMessage() for r in _warnings(caplog))
    assert 'follow_redirects' in message
    assert 'open_mcp_client' in message
