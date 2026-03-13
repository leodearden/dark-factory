"""Shared test fixtures."""

import pytest

from fused_memory.config.schema import FusedMemoryConfig, LLMConfig, LLMProvidersConfig, OpenAIProviderConfig, EmbedderConfig, EmbedderProvidersConfig, RoutingConfig


@pytest.fixture
def mock_config() -> FusedMemoryConfig:
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
    )
