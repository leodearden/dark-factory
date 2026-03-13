"""Tests for the write classifier heuristics."""

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
