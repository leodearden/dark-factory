"""Tests for the read router heuristics."""

import pytest

from fused_memory.models.enums import QueryType, SourceStore
from fused_memory.routing.router import ReadRouter


@pytest.fixture
def router(mock_config):
    return ReadRouter(mock_config)


class TestHeuristicRouting:
    """Heuristic-only tests — no LLM needed."""

    def test_entity_lookup(self, router):
        result = router._heuristic_route('What is the auth service?')
        assert result is not None
        assert result.query_type == QueryType.entity_lookup
        assert result.primary_store == SourceStore.graphiti

    def test_temporal_query(self, router):
        result = router._heuristic_route('What changed in the API last week?')
        assert result is not None
        assert result.query_type == QueryType.temporal
        assert SourceStore.graphiti in result.stores

    def test_relational_query(self, router):
        result = router._heuristic_route('What depends on the database?')
        assert result is not None
        assert result.query_type == QueryType.relational
        assert result.primary_store == SourceStore.graphiti

    def test_preference_query(self, router):
        result = router._heuristic_route("What's the convention for naming?")
        assert result is not None
        assert result.query_type == QueryType.preference
        assert result.primary_store == SourceStore.mem0

    def test_procedural_query(self, router):
        result = router._heuristic_route('How do I deploy to staging?')
        assert result is not None
        assert result.query_type == QueryType.procedural
        assert result.primary_store == SourceStore.mem0

    def test_broad_query(self, router):
        result = router._heuristic_route('Give me an overview of the system')
        assert result is not None
        assert result.query_type == QueryType.broad

    def test_no_match_returns_none(self, router):
        result = router._heuristic_route('foo bar baz')
        assert result is None


class TestRouteAsync:
    """Test the full route() path."""

    @pytest.mark.asyncio
    async def test_stores_override(self, router):
        result = await router.route(
            'anything', stores_override=[SourceStore.mem0]
        )
        assert result.stores == [SourceStore.mem0]
        assert result.primary_store == SourceStore.mem0

    @pytest.mark.asyncio
    async def test_clear_match(self, router):
        result = await router.route('What is the billing service?')
        assert result.query_type == QueryType.entity_lookup
        assert result.primary_store == SourceStore.graphiti

    @pytest.mark.asyncio
    async def test_no_match_falls_back_to_broad(self, router):
        # LLM fallback disabled → defaults to broad
        result = await router.route('foo bar baz')
        assert result.query_type == QueryType.broad
