"""Hermetic guards for the shared "provider never reached" skip policy, on a live
call and on a triage fail-open that call caused.

They drive the REAL openai SDK over ``httpx.MockTransport``, so the exception
chain the policy keys on is the SDK's own, not a hand-built stand-in; the
fail-open guards also go through the REAL ``triage_write`` fail-open.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock

import httpx
import openai
import pytest
from _real_openai_sdk import connect_error, connect_timeout, read_timeout, real_sdk_over
from _unreached_provider import FailOpenCauses, skip_if_provider_unreached
from server.write_triage_harness import configure_triage_config, search_candidate

from fused_memory.server.write_triage import (
    OUTCOME_STORED,
    BandDecision,
    TriageFailOpenCounter,
    triage_write,
)
from fused_memory.services.memory_service import SearchResults


async def _sdk_call(handler: Callable[[httpx.Request], httpx.Response]) -> None:
    """Make one real SDK call over *handler*."""
    async with real_sdk_over(handler)(max_retries=0) as client:
        await client.chat.completions.create(
            model='gpt-4o-mini', messages=[{'role': 'user', 'content': 'x'}]
        )


async def _raised_by(awaitable: Awaitable[object]) -> BaseException:
    """What awaiting *awaitable* raised, a skip or a fail included: neither is an
    Exception, so pytest.raises would pass it through and report the guard skipped
    or failed for the wrong reason.
    """
    try:
        await awaitable
    except (Exception, pytest.skip.Exception, pytest.fail.Exception) as raised:
        return raised
    pytest.fail('the awaited call raised nothing')


def _model_not_found(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        404,
        json={
            'error': {
                'message': 'The model `gpt-4.1-nano` has been deprecated',
                'type': 'invalid_request_error',
                'param': None,
                'code': 'model_not_found',
            }
        },
        request=request,
    )


def _write_timeout(request: httpx.Request) -> httpx.Response:
    """The request was sent past connect: the provider was reached."""
    raise httpx.WriteTimeout('timed out', request=request)


def _read_error(request: httpx.Request) -> httpx.Response:
    """The connection dropped after connect: an APIConnectionError the provider did see."""
    raise httpx.ReadError('connection reset by peer', request=request)


def _server_error(request: httpx.Request) -> httpx.Response:
    return httpx.Response(500, request=request)


_CONNECT_PHASE_FAILURES = [
    pytest.param(connect_timeout, 'ConnectTimeout', id='connect-timeout'),
    pytest.param(connect_error, 'ConnectError', id='connect-error'),
]


@pytest.mark.parametrize(('handler', 'cause'), _CONNECT_PHASE_FAILURES)
@pytest.mark.asyncio
async def test_a_connect_phase_failure_skips_the_live_call(handler, cause):
    raised = await _raised_by(skip_if_provider_unreached(_sdk_call(handler)))

    assert isinstance(raised, pytest.skip.Exception), raised
    assert 'api.openai.com' in str(raised)
    assert cause in str(raised)


@pytest.mark.parametrize(
    'handler',
    [
        pytest.param(read_timeout, id='read-timeout'),
        pytest.param(_write_timeout, id='write-timeout'),
        pytest.param(_read_error, id='read-error'),
        pytest.param(_server_error, id='provider-5xx'),
        pytest.param(_model_not_found, id='model-not-found'),
    ],
)
@pytest.mark.asyncio
async def test_a_failure_after_the_connect_phase_stays_red(handler):
    raised = await _raised_by(skip_if_provider_unreached(_sdk_call(handler)))

    assert isinstance(raised, openai.APIError), raised


def _middle_band_service() -> AsyncMock:
    """A triage service whose one candidate (cosine 0.80) sits in the judge band."""
    service = AsyncMock()
    service.search = AsyncMock(return_value=SearchResults([search_candidate(score=0.80)]))
    configure_triage_config(service)
    return service


async def _triage_failing_open_on(seam: Callable[..., Awaitable[object]]) -> BandDecision:
    """`triage_write` judging through *seam*, then the fail-open policy a live test applies."""
    counter = TriageFailOpenCounter()
    causes = FailOpenCauses()
    decision = await triage_write(
        _middle_band_service(), content='c', project_id='p',
        counter=counter, judge=causes.recording(seam),
    )
    causes.assert_no_fail_open_unless_unreached(counter.live_count())
    return decision


def _judge_calling(
    handler: Callable[[httpx.Request], httpx.Response],
) -> Callable[..., Awaitable[None]]:
    """A judge seam whose work is one real SDK call over *handler*."""

    async def _judge(**_: object) -> None:
        await _sdk_call(handler)

    return _judge


async def _judge_broke(**_: object) -> None:
    raise RuntimeError('judge broke')


async def _judge_stores(**_: object) -> str:
    return OUTCOME_STORED


@pytest.mark.parametrize(('handler', 'cause'), _CONNECT_PHASE_FAILURES)
@pytest.mark.asyncio
async def test_a_fail_open_caused_by_an_unreached_provider_skips(handler, cause):
    raised = await _raised_by(_triage_failing_open_on(_judge_calling(handler)))

    assert isinstance(raised, pytest.skip.Exception), raised
    assert 'api.openai.com' in str(raised)
    assert cause in str(raised)


@pytest.mark.parametrize(
    ('seam', 'recorded'),
    [
        pytest.param(_judge_calling(read_timeout), 'APITimeoutError', id='read-timeout'),
        pytest.param(_judge_calling(_server_error), 'InternalServerError', id='provider-5xx'),
        pytest.param(_judge_calling(_model_not_found), 'NotFoundError', id='model-not-found'),
        pytest.param(_judge_broke, 'RuntimeError', id='judge-broke'),
    ],
)
@pytest.mark.asyncio
async def test_a_fail_open_with_any_other_cause_fails(seam, recorded):
    raised = await _raised_by(_triage_failing_open_on(seam))

    assert isinstance(raised, pytest.fail.Exception), raised
    assert recorded in str(raised)


@pytest.mark.asyncio
async def test_a_judge_that_answers_passes_without_skipping():
    decision = await _triage_failing_open_on(_judge_stores)

    assert decision.outcome == OUTCOME_STORED


@pytest.mark.parametrize(
    ('unreached_recorded', 'fail_open_count', 'named'),
    [
        pytest.param(0, 1, 'no recorded cause', id='nothing-recorded'),
        pytest.param(1, 2, 'APIConnectionError', id='more-counted-than-recorded'),
        pytest.param(2, 1, 'APIConnectionError', id='fewer-counted-than-recorded'),
    ],
)
@pytest.mark.asyncio
async def test_a_fail_open_count_the_recorded_causes_do_not_match_fails(
    unreached_recorded, fail_open_count, named,
):
    causes = FailOpenCauses()
    seam = causes.recording(_judge_calling(connect_error))
    for _ in range(unreached_recorded):
        with pytest.raises(openai.APIConnectionError):
            await seam()

    with pytest.raises((pytest.fail.Exception, pytest.skip.Exception)) as excinfo:
        causes.assert_no_fail_open_unless_unreached(fail_open_count)

    assert isinstance(excinfo.value, pytest.fail.Exception), excinfo.value
    assert named in str(excinfo.value)


@pytest.mark.asyncio
async def test_the_recorded_seam_re_raises_the_original_exception():
    error = RuntimeError('judge broke')

    async def _seam(**_: object) -> None:
        raise error

    with pytest.raises(RuntimeError) as excinfo:
        await FailOpenCauses().recording(_seam)()

    assert excinfo.value is error
