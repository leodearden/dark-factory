"""Stand-ins for the async SDK clients (AsyncOpenAI, AsyncAnthropic) fused-memory takes or constructs."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock


def sdk_client_double() -> MagicMock:
    """A client that is its own async context manager and has an awaitable close().

    The real SDK clients' ``__aenter__`` returns self and their ``close()`` is a
    coroutine; a bare ``MagicMock`` binds a FRESH child on ``async with`` and its
    ``close()`` cannot be awaited.
    """
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.close = AsyncMock()
    return client


def completion_client(content: str) -> MagicMock:
    """A client whose chat completion answers with *content*."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    client = sdk_client_double()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


def raising_client(exc: Exception) -> MagicMock:
    """A client whose chat completion raises *exc*."""
    client = sdk_client_double()
    client.chat.completions.create = AsyncMock(side_effect=exc)
    return client
