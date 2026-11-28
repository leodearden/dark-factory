"""openai SDK clients whose every attempt has a deadline, so their own retries fit inside a backend budget."""

from __future__ import annotations

from typing import TypedDict

import httpx
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from openai import DEFAULT_TIMEOUT, AsyncOpenAI

from fused_memory.config.request_attempts import RequestAttemptsConfig


class SdkAttemptOptions(TypedDict):
    """The kwargs both an openai client constructor and its ``with_options`` accept."""

    timeout: httpx.Timeout
    max_retries: int


def _capped(phase_s: float | None, deadline_s: float) -> float:
    return deadline_s if phase_s is None else min(phase_s, deadline_s)


def request_timeout_within(deadline_s: float) -> httpx.Timeout:
    """The SDK's default timeout with every phase capped at *deadline_s*, none lengthened."""
    return httpx.Timeout(
        connect=_capped(DEFAULT_TIMEOUT.connect, deadline_s),
        read=_capped(DEFAULT_TIMEOUT.read, deadline_s),
        write=_capped(DEFAULT_TIMEOUT.write, deadline_s),
        pool=_capped(DEFAULT_TIMEOUT.pool, deadline_s),
    )


def bounded_attempts(attempts: RequestAttemptsConfig) -> SdkAttemptOptions:
    return SdkAttemptOptions(
        timeout=request_timeout_within(attempts.per_attempt_timeout_seconds),
        max_retries=attempts.max_attempts - 1,
    )


def bounded_openai_embedder(
    config: OpenAIEmbedderConfig, attempts: RequestAttemptsConfig,
) -> OpenAIEmbedder:
    """A graphiti embedder whose client takes its endpoint from *config* alone."""
    client = AsyncOpenAI(
        api_key=config.api_key, base_url=config.base_url, **bounded_attempts(attempts),
    )
    return OpenAIEmbedder(config=config, client=client)
