"""Shared test fixtures."""

import pytest

from fused_memory.config.schema import (
    EmbedderConfig,
    EmbedderProvidersConfig,
    FusedMemoryConfig,
    LLMConfig,
    LLMProvidersConfig,
    OpenAIProviderConfig,
    QueueConfig,
    RoutingConfig,
)


@pytest.fixture
def mock_config(tmp_path) -> FusedMemoryConfig:
    """A FusedMemoryConfig that doesn't require real API keys or services."""
    return FusedMemoryConfig(
        llm=LLMConfig(
            provider='openai',
            model='gpt-4o-mini',
            providers=LLMProvidersConfig(
                openai=OpenAIProviderConfig(api_key='test-key'),
            ),
        ),
        embedder=EmbedderConfig(
            provider='openai',
            model='text-embedding-3-small',
            providers=EmbedderProvidersConfig(
                openai=OpenAIProviderConfig(api_key='test-key'),
            ),
        ),
        routing=RoutingConfig(
            use_heuristics=True,
            llm_fallback=False,
            confidence_threshold=0.7,
        ),
        queue=QueueConfig(
            semaphore_limit=5,
            workers_per_group=2,
            max_attempts=3,
            retry_base_seconds=0.05,
            write_timeout_seconds=2.0,
            data_dir=str(tmp_path / 'queue'),
        ),
    )
