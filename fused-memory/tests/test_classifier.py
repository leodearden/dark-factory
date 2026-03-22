"""Tests for the write classifier heuristics."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.models.enums import MemoryCategory
from fused_memory.routing.classifier import WriteClassifier


@pytest.fixture
def classifier(mock_config):
    return WriteClassifier(mock_config)


class TestHeuristicClassification:
    """Heuristic-only tests — no LLM needed."""

    def test_entities_and_relations(self, classifier):
        result = classifier._heuristic_classify(
            'The auth service depends on the user database'
        )
        assert result is not None
        assert result.primary == MemoryCategory.entities_and_relations

    def test_temporal_facts(self, classifier):
        result = classifier._heuristic_classify(
            'The API was deprecated since version 3.0'
        )
        assert result is not None
        assert result.primary == MemoryCategory.temporal_facts

    def test_decisions_and_rationale(self, classifier):
        result = classifier._heuristic_classify(
            'We chose PostgreSQL because of its JSON support'
        )
        assert result is not None
        assert result.primary == MemoryCategory.decisions_and_rationale

    def test_preferences_and_norms(self, classifier):
        result = classifier._heuristic_classify(
            'Always use snake_case for Python variable names'
        )
        assert result is not None
        assert result.primary == MemoryCategory.preferences_and_norms

    def test_procedural_knowledge(self, classifier):
        result = classifier._heuristic_classify(
            'To deploy, first run the build script, then push to staging'
        )
        assert result is not None
        assert result.primary == MemoryCategory.procedural_knowledge

    def test_observations_and_summaries(self, classifier):
        result = classifier._heuristic_classify(
            'Overall the migration went smoothly with a clear pattern of improvement'
        )
        assert result is not None
        assert result.primary == MemoryCategory.observations_and_summaries

    def test_no_match_returns_none(self, classifier):
        result = classifier._heuristic_classify('Hello world')
        assert result is None

    def test_ambiguous_returns_low_confidence(self, classifier):
        # Contains both temporal and decision markers
        result = classifier._heuristic_classify(
            'We decided to migrate before the deadline changed'
        )
        assert result is not None
        assert result.confidence < 0.7
        assert result.secondary is not None


class TestClassifyAsync:
    """Test the full classify() path with heuristic-only config."""

    @pytest.mark.asyncio
    async def test_clear_match_skips_llm(self, classifier):
        result = await classifier.classify(
            'The payment service depends on the billing API'
        )
        assert result.primary == MemoryCategory.entities_and_relations
        assert result.confidence >= 0.7

    @pytest.mark.asyncio
    async def test_no_match_falls_back_to_default(self, classifier):
        # LLM fallback disabled in mock_config, so should get default
        result = await classifier.classify('Hello world')
        assert result.primary == MemoryCategory.observations_and_summaries
        assert result.confidence < 0.5


class TestLLMClassification:
    """Test _llm_classify with a mocked OpenAI client."""

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
    async def test_nested_braces_in_reasoning_parsed(self, classifier):
        """_llm_classify correctly parses JSON whose reasoning value contains nested braces."""
        # The broken regex r'\{[^}]+\}' fails on this because it stops at the first '}'
        # inside the nested {braces here} value, producing invalid JSON for json.loads.
        llm_output = (
            '{"primary": "decisions_and_rationale", "secondary": null, '
            '"confidence": 0.9, '
            '"reasoning": "chose PostgreSQL because {it has ACID + JSON support}"}'
        )
        classifier._openai_client = self._make_mock_client(llm_output)

        result = await classifier._llm_classify('chose PostgreSQL')

        assert result.primary == MemoryCategory.decisions_and_rationale
        assert result.confidence == pytest.approx(0.9)
        assert 'LLM:' in result.reasoning

    @pytest.mark.asyncio
    async def test_markdown_fence_response_parsed(self, classifier):
        """_llm_classify correctly parses JSON wrapped in ```json...``` code fences."""
        llm_output = (
            '```json\n'
            '{"primary": "temporal_facts", "secondary": null, '
            '"confidence": 0.85, '
            '"reasoning": "time references like {since v3.0} detected"}\n'
            '```'
        )
        classifier._openai_client = self._make_mock_client(llm_output)

        result = await classifier._llm_classify('The API was deprecated since v3.0')

        assert result.primary == MemoryCategory.temporal_facts
        assert result.confidence == pytest.approx(0.85)

    @pytest.mark.asyncio
    async def test_no_json_falls_back_to_default(self, classifier):
        """_llm_classify returns the default when the LLM returns no JSON at all."""
        classifier._openai_client = self._make_mock_client('I cannot classify this.')

        result = await classifier._llm_classify('some content')

        assert result.primary == MemoryCategory.observations_and_summaries
        assert result.confidence < 0.5
