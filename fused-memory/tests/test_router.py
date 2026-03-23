"""Tests for the read router heuristics."""

from unittest.mock import AsyncMock, MagicMock

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


class TestLLMRouting:
    """Test _llm_route with a mocked OpenAI client."""

    def _make_mock_client(self, content: str) -> MagicMock:
        """Return a mock AsyncOpenAI client that yields *content* as LLM output."""
        mock_msg = MagicMock()
        mock_msg.content = content
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        return mock_client

    @pytest.mark.asyncio
    async def test_nested_braces_in_reasoning_parsed(self, router):
        """_llm_route correctly parses JSON whose value contains nested braces."""
        # The broken regex r'\{[^}]+\}' fails here — stops at first '}' inside
        # the nested {entity lookup} value, producing invalid JSON for json.loads.
        llm_output = (
            '{"query_type": "entity_lookup", '
            '"stores": ["graphiti"], '
            '"primary_store": "graphiti", '
            '"reasoning": "matches {entity lookup} pattern"}'
        )
        router._openai_client = self._make_mock_client(llm_output)

        result = await router._llm_route('What is the billing service?')

        assert result.query_type == QueryType.entity_lookup
        assert result.primary_store == SourceStore.graphiti
        assert SourceStore.graphiti in result.stores

    @pytest.mark.asyncio
    async def test_markdown_fence_response_parsed(self, router):
        """_llm_route correctly parses JSON wrapped in ```json...``` code fences."""
        llm_output = (
            '```json\n'
            '{"query_type": "procedural", '
            '"stores": ["mem0"], '
            '"primary_store": "mem0", '
            '"reasoning": "how-to query → {procedural store}"}\n'
            '```'
        )
        router._openai_client = self._make_mock_client(llm_output)

        result = await router._llm_route('How do I deploy to production?')

        assert result.query_type == QueryType.procedural
        assert result.primary_store == SourceStore.mem0

    @pytest.mark.asyncio
    async def test_no_json_falls_back_to_broad(self, router):
        """_llm_route returns the broad default when the LLM returns no JSON."""
        router._openai_client = self._make_mock_client('Sorry, I cannot classify this.')

        result = await router._llm_route('some query')

        assert result.query_type == QueryType.broad
