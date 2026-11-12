"""A live OpenAI-SDK test skips only when its call failed in the connect phase,
before any byte reached the provider.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import TypeVar

import httpx
import openai
import pytest

_CONNECT_PHASE_CAUSES = (httpx.ConnectError, httpx.ConnectTimeout)


def unreached_provider_skip_reason(exc: BaseException) -> str | None:
    """The skip reason when *exc* shows the provider's connect phase failed
    (ConnectError or ConnectTimeout) before any response; None for every other failure.
    """
    cause = exc.__cause__
    if not (isinstance(exc, openai.APIConnectionError) and isinstance(cause, _CONNECT_PHASE_CAUSES)):
        return None
    return (
        f'{exc.request.url.host} was not reached: {type(cause).__name__} before any '
        f'response, so the live call observed nothing; a lasting outage is '
        f"scripts/lane_dependency_probe.py's to report"
    )


_T = TypeVar('_T')


async def skip_if_provider_unreached(call: Awaitable[_T]) -> _T:
    """Await *call*, skipping the test only when the provider was never reached."""
    try:
        return await call
    except openai.APIError as exc:
        if (reason := unreached_provider_skip_reason(exc)) is None:
            raise
        pytest.skip(reason)
