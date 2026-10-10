"""Stand-ins for the AsyncOpenAI client that ``routing/classifier.py::WriteClassifier`` takes."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock


def completion_client(content: str) -> MagicMock:
    """A client whose chat completion answers with *content*."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


def raising_client(exc: Exception) -> MagicMock:
    """A client whose chat completion raises *exc*."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=exc)
    return client
