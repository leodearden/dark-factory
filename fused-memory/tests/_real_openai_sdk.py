"""The REAL openai SDK client over an ``httpx.MockTransport``, and the transport failures to feed it."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import openai

_REAL_ASYNC_OPENAI = openai.AsyncOpenAI


def real_sdk_over(
    handler: Callable[[httpx.Request], httpx.Response],
) -> Callable[..., openai.AsyncOpenAI]:
    """A constructor of the real `openai.AsyncOpenAI` whose every request *handler* answers.

    It also serves as ``patch('openai.AsyncOpenAI', side_effect=...)``, since the class
    is captured before any patch. The real client raises the SDK's own exception chain;
    a hand-built `openai.APITimeoutError` carries no `__cause__`, so it cannot stand in.
    The hermetic key and base URL are defaults the caller's kwargs override, so neither
    OPENAI_API_KEY nor OPENAI_BASE_URL from the environment reaches the client.
    """

    def _construct(**kwargs: Any) -> openai.AsyncOpenAI:
        return _REAL_ASYNC_OPENAI(
            **{
                'api_key': 'sk-hermetic-not-a-real-key',
                'base_url': 'https://api.openai.com/v1',
                **kwargs,
            },
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

    return _construct


def connect_timeout(request: httpx.Request) -> httpx.Response:
    """The connect phase timed out: no byte reached the provider."""
    raise httpx.ConnectTimeout('timed out', request=request)


def connect_error(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError('[Errno -3] Temporary failure in name resolution', request=request)


def read_timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout('timed out', request=request)
