"""PrefixResolverClient: the escalation servers' C2 port over POST /resolve-prefix.

Driven through ``httpx.MockTransport``, passed as the client's public
``transport`` field, so every assertion is about what crosses the wire. The
other side of the wire is pinned against the REAL route in
``fused-memory/tests/test_prefix_resolver_client_contract.py``.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import ssl
import time
from collections.abc import Callable, Coroutine
from typing import Any

import httpx
import pytest
from shared.mcp_session import McpSession, SyncMcpSession
from shared.uuid_prefix_guard import (
    MAX_CONCURRENT_RESOLUTIONS,
    MAX_DISTINCT_PREFIXES_PER_CALL,
    Candidate,
    Resolution,
    ResolverUnavailable,
    UnavailableReason,
)

from escalation import prefix_resolver_client
from escalation.prefix_resolver_client import PrefixResolverClient

_BASE_URL = 'http://fused-memory.test'
_URL = f'{_BASE_URL}/resolve-prefix'
_PROJECT = 'reify'
_PREFIX = 'bff81530'
_FULL = 'bff81530-70e5-4ab6-939e-507299d1bcd2'
_SIBLING = 'bff81530-0000-4000-8000-000000000002'

Handler = (
    Callable[[httpx.Request], httpx.Response]
    | Callable[[httpx.Request], Coroutine[Any, Any, httpx.Response]]
)


class _Recorder:
    """A MockTransport handler answering one fixed response and keeping every request."""

    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.response


def _client(handler: Handler, *, base_url: str = _BASE_URL, **kwargs: float) -> PrefixResolverClient:
    return PrefixResolverClient(base_url=base_url, transport=httpx.MockTransport(handler), **kwargs)


async def _resolve(client: PrefixResolverClient) -> Resolution:
    return await client.resolve_prefix(_PROJECT, _PREFIX)


def _raise_connect_error(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError('connection refused', request=request)


def _raise_read_timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout('slow', request=request)


def _client_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record for record in caplog.records
        if record.name == prefix_resolver_client.__name__ and record.levelno == logging.WARNING
    ]


class TestRequest:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('base_url', [_BASE_URL, f'{_BASE_URL}/'])
    async def test_one_post_to_resolve_prefix_carrying_exactly_project_and_prefix(self, base_url):
        recorder = _Recorder(httpx.Response(200, json={'outcome': 'none', 'candidates': []}))

        await _resolve(_client(recorder, base_url=base_url))

        assert len(recorder.requests) == 1
        request = recorder.requests[0]
        assert request.method == 'POST'
        assert str(request.url) == _URL
        assert json.loads(request.content) == {'project_id': _PROJECT, 'prefix': _PREFIX}


class TestResolutions:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(('body', 'expected'), [
        (
            {'outcome': 'unique', 'candidates': [{'namespace': 'mem0', 'id': _FULL, 'preview': 'p'}]},
            Resolution('unique', (Candidate('mem0', _FULL, 'p'),)),
        ),
        (
            {'outcome': 'ambiguous', 'candidates': [
                {'namespace': 'graphiti_edge', 'id': _SIBLING, 'preview': 'edge'},
                {'namespace': 'mem0', 'id': _FULL, 'preview': 'memory'},
            ]},
            Resolution('ambiguous', (
                Candidate('graphiti_edge', _SIBLING, 'edge'),
                Candidate('mem0', _FULL, 'memory'),
            )),
        ),
        ({'outcome': 'none', 'candidates': []}, Resolution('none', ())),
    ], ids=['unique', 'ambiguous', 'none'])
    async def test_a_200_decodes_to_the_resolution_preserving_candidate_order(self, body, expected):
        resolution = await _resolve(_client(_Recorder(httpx.Response(200, json=body))))

        assert resolution == expected
        assert all(isinstance(candidate, Candidate) for candidate in resolution.candidates)


class TestUnavailable:
    @pytest.mark.asyncio
    async def test_a_503_names_the_store_from_its_body(self):
        response = httpx.Response(503, json={'error_type': 'resolver_unavailable', 'store': 'graphiti'})

        with pytest.raises(ResolverUnavailable) as raised:
            await _resolve(_client(_Recorder(response)))

        assert (raised.value.store, raised.value.reason) == ('graphiti', UnavailableReason.UNREACHABLE)

    @pytest.mark.asyncio
    @pytest.mark.parametrize('body', [
        {'error_type': 'resolver_unavailable'},
        {'error_type': 'resolver_unavailable', 'store': 7},
        ['graphiti'],
    ])
    async def test_a_503_without_a_string_store_names_the_service(self, body):
        with pytest.raises(ResolverUnavailable) as raised:
            await _resolve(_client(_Recorder(httpx.Response(503, json=body))))

        assert (raised.value.store, raised.value.reason) == ('fused-memory', UnavailableReason.UNREACHABLE)

    @pytest.mark.asyncio
    async def test_a_503_with_a_non_json_body_names_the_service(self):
        with pytest.raises(ResolverUnavailable) as raised:
            await _resolve(_client(_Recorder(httpx.Response(503, content=b'upstream gone'))))

        assert (raised.value.store, raised.value.reason) == ('fused-memory', UnavailableReason.UNREACHABLE)

    @pytest.mark.asyncio
    @pytest.mark.parametrize('status', [400, 404, 500])
    async def test_any_other_status_is_a_bad_response_never_a_none(self, status):
        body = {'outcome': 'none', 'candidates': [], 'store': 'graphiti'}

        with pytest.raises(ResolverUnavailable) as raised:
            await _resolve(_client(_Recorder(httpx.Response(status, json=body))))

        assert (raised.value.store, raised.value.reason) == ('fused-memory', UnavailableReason.BAD_RESPONSE)

    @pytest.mark.asyncio
    async def test_a_transport_connect_error_is_unreachable(self):
        with pytest.raises(ResolverUnavailable) as raised:
            await _resolve(_client(_raise_connect_error))

        assert (raised.value.store, raised.value.reason) == ('fused-memory', UnavailableReason.UNREACHABLE)

    @pytest.mark.asyncio
    async def test_a_real_refused_connection_is_unreachable(self):
        client = PrefixResolverClient(base_url='http://127.0.0.1:1')

        with pytest.raises(ResolverUnavailable) as raised:
            await _resolve(client)

        assert (raised.value.store, raised.value.reason) == ('fused-memory', UnavailableReason.UNREACHABLE)

    @pytest.mark.asyncio
    @pytest.mark.parametrize('base_url', [
        'http://localhost:abc',
        'http://[::1',
        'http://127.0.0.1:80020',
    ], ids=['invalid-port', 'unclosed-ipv6', 'port-out-of-range'])
    async def test_an_unusable_base_url_is_unreachable(self, base_url):
        client = PrefixResolverClient(base_url=base_url)

        with pytest.raises(ResolverUnavailable) as raised:
            await _resolve(client)

        assert (raised.value.store, raised.value.reason) == ('fused-memory', UnavailableReason.UNREACHABLE)


class TestDeadline:
    @staticmethod
    async def _hang(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(30)
        return httpx.Response(200, json={'outcome': 'none', 'candidates': []})

    @pytest.mark.asyncio
    async def test_a_stalled_resolver_is_abandoned_at_the_configured_deadline_as_a_timeout(self):
        started = time.monotonic()

        with pytest.raises(ResolverUnavailable) as raised:
            await _resolve(_client(self._hang, timeout_seconds=0.05))

        assert time.monotonic() - started < 1.0
        assert raised.value.reason is UnavailableReason.TIMED_OUT

    def test_the_default_deadline_keeps_a_guarded_call_inside_the_client_budget(self):
        """INV-8: the guard awaits at most ceil(16 / 8) rounds of this deadline,
        which must leave an escalation tool call inside its callers' budget."""
        budget = min(
            inspect.signature(McpSession.call_tool).parameters['timeout'].default,
            inspect.signature(SyncMcpSession).parameters['timeout'].default,
        )
        rounds = math.ceil(MAX_DISTINCT_PREFIXES_PER_CALL / MAX_CONCURRENT_RESOLUTIONS)

        occupancy = rounds * PrefixResolverClient(base_url=_BASE_URL).timeout_seconds

        assert occupancy < budget


class TestSlowButAnswering:
    """Measured /resolve-prefix latency reaches ~2.4 s under the guard's own
    8-wide fan-out, so a resolver that answers in 1.5 s is healthy, not down."""

    @pytest.mark.asyncio
    async def test_a_resolver_answering_after_1_5s_resolves_under_the_default_deadline(self, caplog):
        caplog.set_level(logging.WARNING, logger=prefix_resolver_client.__name__)

        async def slow(request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(1.5)
            return httpx.Response(200, json={
                'outcome': 'unique', 'candidates': [{'namespace': 'mem0', 'id': _FULL, 'preview': 'p'}],
            })

        resolution = await _resolve(_client(slow))

        assert resolution == Resolution('unique', (Candidate('mem0', _FULL, 'p'),))
        assert _client_warnings(caplog) == []


def _raise_group(*leaves: Callable[[httpx.Request], Exception]) -> Handler:
    """A handler raising an ExceptionGroup, as anyio does, of one error per *leaves* builder."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise ExceptionGroup('unhandled errors in a TaskGroup', [leaf(request) for leaf in leaves])

    return handler


class TestTimeoutReason:
    """A deadline, ours or httpx's, is ``timed_out`` — a slow store, not a down one.

    The other reasons are asserted beside each failure shape above.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(('handler', 'reason'), [
        (_raise_read_timeout, UnavailableReason.TIMED_OUT),
        (
            _raise_group(lambda r: httpx.ReadTimeout('a', request=r), lambda r: TimeoutError()),
            UnavailableReason.TIMED_OUT,
        ),
        (
            _raise_group(
                lambda r: httpx.ReadTimeout('a', request=r),
                lambda r: httpx.ConnectError('refused', request=r),
            ),
            UnavailableReason.UNREACHABLE,
        ),
    ], ids=['httpx-read-timeout', 'group-of-only-timeouts', 'group-with-a-refusal'])
    async def test_a_timeout_is_timed_out_and_a_group_only_when_every_leaf_is(self, handler, reason):
        with pytest.raises(ResolverUnavailable) as raised:
            await _resolve(_client(handler))

        assert (raised.value.store, raised.value.reason) == ('fused-memory', reason)


class TestCancellation:
    @pytest.mark.asyncio
    async def test_cancellation_propagates_and_is_never_unavailability(self, caplog):
        caplog.set_level(logging.WARNING, logger=prefix_resolver_client.__name__)
        in_flight = asyncio.Event()

        async def hang(request: httpx.Request) -> httpx.Response:
            in_flight.set()
            await asyncio.sleep(30)
            return httpx.Response(200, json={'outcome': 'none', 'candidates': []})

        resolving = asyncio.create_task(_resolve(_client(hang, timeout_seconds=60)))
        async with asyncio.timeout(5):
            await in_flight.wait()
        resolving.cancel()

        with pytest.raises(asyncio.CancelledError):
            await resolving
        assert _client_warnings(caplog) == []


class TestEventLoopCost:
    @pytest.mark.asyncio
    async def test_resolutions_build_the_tls_context_at_most_once(self, monkeypatch):
        built: list[ssl.SSLContext] = []
        create_default_context = ssl.create_default_context

        def recording(*args: Any, **kwargs: Any) -> ssl.SSLContext:
            built.append(create_default_context(*args, **kwargs))
            return built[-1]

        monkeypatch.setattr(ssl, 'create_default_context', recording)
        client = PrefixResolverClient(base_url='http://127.0.0.1:1')

        for _ in range(3):
            with pytest.raises(ResolverUnavailable):
                await _resolve(client)

        assert len(built) <= 1


class TestMalformedSuccess:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('response', [
        httpx.Response(200, content=b'not json'),
        httpx.Response(200, json=['unique']),
        httpx.Response(200, json={'outcome': 'maybe', 'candidates': []}),
        httpx.Response(200, json={'outcome': 'unique', 'candidates': 'mem0'}),
        httpx.Response(200, json={'outcome': 'unique', 'candidates': [
            {'namespace': 'qdrant', 'id': _FULL, 'preview': 'p'},
        ]}),
        httpx.Response(200, json={'outcome': 'unique', 'candidates': [
            {'namespace': 'mem0', 'preview': 'p'},
        ]}),
    ], ids=[
        'non-json', 'json-list', 'unknown-outcome', 'candidates-not-a-list',
        'unknown-namespace', 'candidate-missing-id',
    ])
    async def test_a_malformed_200_is_a_bad_response_never_a_resolution(self, response):
        with pytest.raises(ResolverUnavailable) as raised:
            await _resolve(_client(_Recorder(response)))

        assert (raised.value.store, raised.value.reason) == ('fused-memory', UnavailableReason.BAD_RESPONSE)


class TestFailureWarning:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(('handler', 'reason'), [
        (_Recorder(httpx.Response(503, json={'store': 'mem0'})), '503'),
        (_Recorder(httpx.Response(500)), '500'),
        (_Recorder(httpx.Response(200, json={'outcome': 'maybe', 'candidates': []})), '200'),
        (_raise_connect_error, 'ConnectError'),
    ], ids=['503', '500', 'malformed-200', 'connect-error'])
    async def test_every_failure_logs_exactly_one_warning_naming_the_call(self, caplog, handler, reason):
        caplog.set_level(logging.WARNING, logger=prefix_resolver_client.__name__)

        with pytest.raises(ResolverUnavailable):
            await _resolve(_client(handler))

        warnings = _client_warnings(caplog)
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        for fragment in (_URL, _PROJECT, _PREFIX, reason):
            assert fragment in message

    @pytest.mark.asyncio
    @pytest.mark.parametrize(('base_url', 'error'), [
        ('http://localhost:abc', 'InvalidURL'),
        ('http://127.0.0.1:80020', 'OverflowError'),
    ], ids=['invalid-port', 'port-out-of-range'])
    async def test_an_unusable_base_url_logs_one_warning_naming_the_error(self, caplog, base_url, error):
        caplog.set_level(logging.WARNING, logger=prefix_resolver_client.__name__)

        with pytest.raises(ResolverUnavailable):
            await _resolve(PrefixResolverClient(base_url=base_url))

        warnings = _client_warnings(caplog)
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        for fragment in (f'{base_url}/resolve-prefix', _PROJECT, _PREFIX, error):
            assert fragment in message

    @pytest.mark.asyncio
    async def test_a_success_logs_no_warning(self, caplog):
        caplog.set_level(logging.WARNING, logger=prefix_resolver_client.__name__)

        await _resolve(_client(_Recorder(httpx.Response(200, json={'outcome': 'none', 'candidates': []}))))

        assert _client_warnings(caplog) == []


class TestConstruction:
    @pytest.mark.parametrize('timeout_seconds', [0, -1])
    def test_a_non_positive_deadline_is_refused_naming_the_value(self, timeout_seconds):
        with pytest.raises(ValueError, match=str(timeout_seconds)):
            PrefixResolverClient(base_url=_BASE_URL, timeout_seconds=timeout_seconds)
