"""Hermetic guards for the shared "provider never reached" skip policy.

They drive the REAL openai SDK over ``httpx.MockTransport``, so the exception
chain the policy keys on is the SDK's own, not a hand-built stand-in.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import httpx
import openai
import pytest
from _real_openai_sdk import connect_error, connect_timeout, read_timeout, real_sdk_over
from _unreached_provider import skip_if_provider_unreached


async def _sdk_call(handler: Callable[[httpx.Request], httpx.Response]) -> None:
    """Make one real SDK call over *handler*."""
    async with real_sdk_over(handler)(max_retries=0) as client:
        await client.chat.completions.create(
            model='gpt-4o-mini', messages=[{'role': 'user', 'content': 'x'}]
        )


async def _raised_by(awaitable: Awaitable[object]) -> BaseException:
    """What awaiting *awaitable* raised, a skip included: a skip is no Exception,
    so pytest.raises would pass it through and report the guard skipped, not red.
    """
    try:
        await awaitable
    except (Exception, pytest.skip.Exception) as raised:
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


@pytest.mark.asyncio
async def test_a_connect_timeout_skips_the_live_call():
    raised = await _raised_by(skip_if_provider_unreached(_sdk_call(connect_timeout)))

    assert isinstance(raised, pytest.skip.Exception), raised
    assert 'api.openai.com' in str(raised)
    assert 'ConnectTimeout' in str(raised)


@pytest.mark.parametrize(
    'handler',
    [
        pytest.param(connect_error, id='connect-error'),
        pytest.param(read_timeout, id='read-timeout'),
        pytest.param(_model_not_found, id='model-not-found'),
    ],
)
@pytest.mark.asyncio
async def test_a_provider_failure_other_than_a_connect_timeout_stays_red(handler):
    raised = await _raised_by(skip_if_provider_unreached(_sdk_call(handler)))

    assert isinstance(raised, openai.APIError), raised
