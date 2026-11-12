"""Attribute graphiti LLM token usage to the asyncio context that spent it.

``measure_llm_tokens`` opens a window; every ``record()`` made on an
``AttributingTokenUsageTracker`` from inside that window's context, including
from tasks spawned within it (asyncio copies the context at task creation), is
credited to it. Records from concurrent writes in other tasks never are, even
though they share the same client and tracker.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from graphiti_core.llm_client.token_tracker import TokenUsageTracker


@dataclass(frozen=True)
class LlmTokenUsage:
    """The LLM usage recorded inside one measured window. ``llm_calls`` counts records."""

    input_tokens: int
    output_tokens: int
    llm_calls: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_journal_dict(self) -> dict[str, int]:
        return {
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'total_tokens': self.total_tokens,
            'llm_calls': self.llm_calls,
        }


class _Sink:
    def __init__(self) -> None:
        self._input_tokens = 0
        self._output_tokens = 0
        self._llm_calls = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self._input_tokens += input_tokens
        self._output_tokens += output_tokens
        self._llm_calls += 1

    def freeze(self) -> LlmTokenUsage:
        return LlmTokenUsage(self._input_tokens, self._output_tokens, self._llm_calls)


_ACTIVE_SINKS: ContextVar[tuple[_Sink, ...]] = ContextVar('active_llm_token_sinks', default=())


class AttributingTokenUsageTracker(TokenUsageTracker):
    """Upstream's cumulative tracker that also credits every open window."""

    def record(self, prompt_name: str | None, input_tokens: int, output_tokens: int) -> None:
        super().record(prompt_name, input_tokens, output_tokens)
        for sink in _ACTIVE_SINKS.get():
            sink.add(input_tokens, output_tokens)


@dataclass
class TokenMeasurement:
    """Set when its window closes; ``None`` means the client was not measurable."""

    usage: LlmTokenUsage | None = None


@contextlib.asynccontextmanager
async def measure_llm_tokens(llm_client: Any) -> AsyncIterator[TokenMeasurement]:
    measurement = TokenMeasurement()
    if not isinstance(getattr(llm_client, 'token_tracker', None), AttributingTokenUsageTracker):
        yield measurement
        return
    sink = _Sink()
    reset_token = _ACTIVE_SINKS.set((*_ACTIVE_SINKS.get(), sink))
    try:
        yield measurement
    finally:
        _ACTIVE_SINKS.reset(reset_token)
        measurement.usage = sink.freeze()
