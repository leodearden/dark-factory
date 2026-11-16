"""first_success's ``ambiguous_result`` mode: never repeat a write that may have landed.

Every configured fused-memory URL is a network path to ONE logical store
(DESIGN.md "Curator Tickets & Routing Invariant"), so failing a mutating call
over to the next URL after a failure that may have reached the server applies
it twice. With ``ambiguous_result`` given, only failures that prove the request
never left the client fall through; any other caught failure ends the fan-out
with ``ambiguous_result(errors)``. Without it, read callers fail over exactly as
before.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from dashboard.data import memory
from dashboard.data.mcp_fanout import PreformattedFanoutError, first_success

_URLS = ['http://a', 'http://b']


def _offline(errors: list[str]) -> dict:
    return {'offline': errors}


def _ambiguous(errors: list[str]) -> dict:
    return {'ambiguous': errors}


def _http_status_error(url: str = 'http://a') -> httpx.HTTPStatusError:
    request = httpx.Request('POST', f'{url}/mcp')
    response = httpx.Response(500, request=request)
    return httpx.HTTPStatusError('Server Error', request=request, response=response)


def _failing_on_a(exc: BaseException, attempted: list[str]):
    async def call(url: str) -> str:
        attempted.append(url)
        if url == 'http://a':
            raise exc
        return f'{url}-payload'

    return call


@pytest.fixture
def invalidated(monkeypatch) -> list[str]:
    recorded: list[str] = []
    monkeypatch.setattr(memory, 'invalidate_session', recorded.append)
    return recorded


@pytest.mark.parametrize(
    'exc',
    [
        httpx.ReadTimeout('rt'),
        httpx.WriteTimeout('wt'),
        _http_status_error(),
        ValueError('bad'),
        TimeoutError('slow'),
    ],
    ids=['ReadTimeout', 'WriteTimeout', 'HTTPStatusError', 'ValueError', 'TimeoutError'],
)
async def test_possibly_delivered_failure_stops_at_the_failing_url(exc, invalidated):
    attempted: list[str] = []

    result = await first_success(
        _URLS, _failing_on_a(exc, attempted),
        log_label='test', offline_result=_offline, ambiguous_result=_ambiguous,
    )

    assert isinstance(result, dict)
    (error,) = result['ambiguous']
    assert error.startswith('http://a: ')
    assert type(exc).__name__ in error
    assert attempted == ['http://a']
    assert invalidated == ['http://a']


@pytest.mark.parametrize(
    'exc',
    [httpx.ConnectError('refused'), httpx.ConnectTimeout('ct'), httpx.PoolTimeout('')],
    ids=['ConnectError', 'ConnectTimeout', 'PoolTimeout'],
)
async def test_undelivered_failure_falls_through_to_the_next_url(exc):
    attempted: list[str] = []

    result = await first_success(
        _URLS, _failing_on_a(exc, attempted),
        log_label='test', offline_result=_offline, ambiguous_result=_ambiguous,
    )

    assert result == 'http://b-payload'
    assert attempted == ['http://a', 'http://b']


async def test_ambiguous_failure_after_an_undelivered_one_stops_and_names_both():
    attempted: list[str] = []
    failures = {'http://a': httpx.ConnectError('refused'), 'http://b': httpx.ReadTimeout('rt')}

    async def call(url: str) -> str:
        attempted.append(url)
        if url in failures:
            raise failures[url]
        return f'{url}-payload'

    result = await first_success(
        ['http://a', 'http://b', 'http://c'], call,
        log_label='test', offline_result=_offline, ambiguous_result=_ambiguous,
    )

    assert result == {'ambiguous': ['http://a: ConnectError: refused', 'http://b: ReadTimeout: rt']}
    assert attempted == ['http://a', 'http://b']


def _preformatted_from(cause: Exception) -> PreformattedFanoutError:
    try:
        raise PreformattedFanoutError(f'{type(cause).__name__}: {cause}') from cause
    except PreformattedFanoutError as wrapped:
        return wrapped


async def test_preformatted_wrapper_of_a_possibly_delivered_cause_stops():
    attempted: list[str] = []

    result = await first_success(
        _URLS, _failing_on_a(_preformatted_from(httpx.ReadTimeout('rt')), attempted),
        log_label='test', offline_result=_offline, ambiguous_result=_ambiguous,
    )

    assert result == {'ambiguous': ['http://a: ReadTimeout: rt']}
    assert attempted == ['http://a']


async def test_preformatted_wrapper_of_an_undelivered_cause_falls_through():
    attempted: list[str] = []

    result = await first_success(
        _URLS, _failing_on_a(_preformatted_from(httpx.ConnectError('refused')), attempted),
        log_label='test', offline_result=_offline, ambiguous_result=_ambiguous,
    )

    assert result == 'http://b-payload'
    assert attempted == ['http://a', 'http://b']


async def test_whole_operation_deadline_is_ambiguous():
    attempted: list[str] = []
    never = asyncio.Event()

    async def call(url: str) -> str:
        attempted.append(url)
        if url == 'http://a':
            await never.wait()
        return f'{url}-payload'

    result = await asyncio.wait_for(
        first_success(
            _URLS, call,
            log_label='test', offline_result=_offline, ambiguous_result=_ambiguous,
            per_url_timeout=0.05,
        ),
        5,
    )

    assert isinstance(result, dict)
    (error,) = result['ambiguous']
    assert error.startswith('http://a: TimeoutError')
    assert attempted == ['http://a']


async def test_every_url_undelivered_returns_the_offline_result():
    async def call(url: str) -> str:
        raise httpx.ConnectError('refused')

    result = await first_success(
        _URLS, call,
        log_label='test', offline_result=_offline, ambiguous_result=_ambiguous,
    )

    assert result == {
        'offline': ['http://a: ConnectError: refused', 'http://b: ConnectError: refused'],
    }


async def test_without_ambiguous_result_a_read_timeout_still_fails_over():
    attempted: list[str] = []

    result = await first_success(
        _URLS, _failing_on_a(httpx.ReadTimeout('rt'), attempted),
        log_label='test', offline_result=_offline,
    )

    assert result == 'http://b-payload'
    assert attempted == ['http://a', 'http://b']
