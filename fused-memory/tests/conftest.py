"""Shared test fixtures."""

from dataclasses import dataclass, field

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


@dataclass
class MockNode:
    """Simulates a Graphiti entity node (source/target of an edge)."""

    name: str


@dataclass
class MockEdge:
    """Simulates a Graphiti entity edge returned from add_episode or search."""

    fact: str
    uuid: str = ''
    name: str | None = None
    source_node: MockNode | None = None
    target_node: MockNode | None = None
    episodes: list[str] = field(default_factory=list)
    valid_at: str | None = None
    invalid_at: str | None = None


@dataclass
class EdgeWithoutUuid:
    """Simulates a Graphiti edge that lacks a uuid attribute.

    Deliberately omits the uuid field so that getattr(e, 'uuid', None) returns
    None — modelling real-world edge objects that do not expose a uuid property.
    """

    fact: str = 'some fact'
    name: str | None = None
    source_node: MockNode | None = None
    target_node: MockNode | None = None
    episodes: list[str] = field(default_factory=list)
    valid_at: str | None = None
    invalid_at: str | None = None


@dataclass
class EdgeWithNoneEpisodes:
    """Simulates a Graphiti edge where the episodes attribute is present but None.

    Tests the is-not-None semantic: None episodes must produce empty provenance,
    same as a missing episodes attribute.
    """

    uuid: str = 'u-ep-none'
    fact: str = 'fact with none episodes'
    name: str | None = None
    source_node: MockNode | None = None
    target_node: MockNode | None = None
    episodes: list[str] | None = None
    valid_at: str | None = None
    invalid_at: str | None = None


@dataclass
class MockAddEpisodeResult:
    """Simulates the AddEpisodeResults returned by Graphiti's add_episode."""

    entity_edges: list[MockEdge] = field(default_factory=list)


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
