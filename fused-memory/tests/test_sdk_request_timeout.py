"""The shared helper that gives every openai SDK attempt a deadline (task 10241)."""

from __future__ import annotations

import httpx
import openai
import pytest
from fused_memory.backends.sdk_request_timeout import (
    bounded_attempts,
    bounded_openai_embedder,
    request_timeout_within,
)
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

from fused_memory.config.request_attempts import RequestAttemptsConfig

_URL = 'http://127.0.0.1:9/v1'


class TestRequestTimeoutWithin:
    def test_every_long_phase_is_capped_at_the_deadline(self):
        timeout = request_timeout_within(8.0)

        assert isinstance(timeout, httpx.Timeout)
        assert (timeout.read, timeout.write, timeout.pool) == (8.0, 8.0, 8.0)
        assert timeout.connect == openai.DEFAULT_TIMEOUT.connect == 5.0

    def test_a_short_deadline_caps_connect_too(self):
        assert request_timeout_within(2.0) == httpx.Timeout(2.0)

    def test_no_phase_is_ever_lengthened(self):
        assert request_timeout_within(10_000) == openai.DEFAULT_TIMEOUT


class TestBoundedAttempts:
    def test_it_sets_the_deadline_and_retries_after_the_first_attempt(self):
        attempts = RequestAttemptsConfig(per_attempt_timeout_seconds=8, max_attempts=3)

        assert bounded_attempts(attempts) == {
            'timeout': request_timeout_within(8),
            'max_retries': 2,
        }

    def test_a_single_attempt_never_retries(self):
        assert bounded_attempts(RequestAttemptsConfig(max_attempts=1))['max_retries'] == 0


@pytest.mark.asyncio
async def test_the_bounded_embedder_carries_its_config_and_the_attempts():
    config = OpenAIEmbedderConfig(
        api_key='k', base_url=_URL, embedding_model='m', embedding_dim=1024,
    )
    attempts = RequestAttemptsConfig(per_attempt_timeout_seconds=3.0, max_attempts=4)

    embedder = bounded_openai_embedder(config, attempts)
    try:
        assert isinstance(embedder, OpenAIEmbedder)
        assert embedder.config is config
        client = embedder.client
        assert isinstance(client, openai.AsyncOpenAI)
        assert str(client.base_url).rstrip('/') == _URL
        assert client.api_key == 'k'
        assert client.timeout == request_timeout_within(3.0)
        assert client.max_retries == 3
    finally:
        await embedder.client.close()
