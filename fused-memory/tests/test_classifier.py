"""Tests for the write classifier heuristics."""

import pytest
from _mock_openai_server import mock_openai_server
from _sdk_client_stubs import completion_client, raising_client

from fused_memory.models.enums import (
    LLM_CLASSIFIER_FAILURES,
    ClassificationFallback,
    MemoryCategory,
)
from fused_memory.routing.classifier import CLASSIFICATION_SYSTEM_PROMPT, WriteClassifier
from fused_memory.routing.openai_client import routing_chat_params

_NO_HEURISTIC_MATCH = 'Hello world'


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


@pytest.fixture
def llm_only_config(mock_config):
    mock_config.routing.use_heuristics = False
    mock_config.routing.llm_fallback = True
    return mock_config


class TestLLMClassification:
    """classify() with heuristics off, over an injected OpenAI client."""

    @pytest.mark.asyncio
    async def test_nested_braces_in_reasoning_parsed(self, llm_only_config):
        """classify() correctly parses JSON whose reasoning value contains nested braces."""
        # The broken regex r'\{[^}]+\}' fails on this because it stops at the first '}'
        # inside the nested {braces here} value, producing invalid JSON for json.loads.
        llm_output = (
            '{"primary": "decisions_and_rationale", "secondary": null, '
            '"confidence": 0.9, '
            '"reasoning": "chose PostgreSQL because {it has ACID + JSON support}"}'
        )
        classifier = WriteClassifier(llm_only_config, openai_client=completion_client(llm_output))

        result = await classifier.classify('chose PostgreSQL')

        assert result.primary == MemoryCategory.decisions_and_rationale
        assert result.confidence == pytest.approx(0.9)
        assert 'LLM:' in result.reasoning

    @pytest.mark.asyncio
    async def test_markdown_fence_response_parsed(self, llm_only_config):
        """classify() correctly parses JSON wrapped in ```json...``` code fences."""
        llm_output = (
            '```json\n'
            '{"primary": "temporal_facts", "secondary": null, '
            '"confidence": 0.85, '
            '"reasoning": "time references like {since v3.0} detected"}\n'
            '```'
        )
        classifier = WriteClassifier(llm_only_config, openai_client=completion_client(llm_output))

        result = await classifier.classify('The API was deprecated since v3.0')

        assert result.primary == MemoryCategory.temporal_facts
        assert result.confidence == pytest.approx(0.85)

    @pytest.mark.asyncio
    async def test_no_json_falls_back_to_default(self, llm_only_config):
        """classify() returns the default when the LLM returns no JSON at all."""
        classifier = WriteClassifier(
            llm_only_config, openai_client=completion_client('I cannot classify this.'),
        )

        result = await classifier.classify('some content')

        assert result.primary == MemoryCategory.observations_and_summaries
        assert result.confidence < 0.5


class TestClassificationFallbackReason:
    """classify() labels every default category with WHY it is a default."""

    @pytest.fixture
    def llm_config(self, mock_config):
        mock_config.routing.llm_fallback = True
        return mock_config

    @pytest.mark.asyncio
    async def test_llm_call_raising_is_llm_error(self, llm_config):
        classifier = WriteClassifier(
            llm_config, openai_client=raising_client(RuntimeError('connection refused')),
        )

        result = await classifier.classify(_NO_HEURISTIC_MATCH)

        assert result.primary == MemoryCategory.observations_and_summaries
        assert result.fallback is ClassificationFallback.llm_error
        assert result.fallback in LLM_CLASSIFIER_FAILURES

    @pytest.mark.asyncio
    async def test_llm_returning_no_json_is_llm_no_json(self, llm_config):
        classifier = WriteClassifier(
            llm_config, openai_client=completion_client('I cannot classify this.'),
        )

        result = await classifier.classify(_NO_HEURISTIC_MATCH)

        assert result.primary == MemoryCategory.observations_and_summaries
        assert result.fallback is ClassificationFallback.llm_no_json
        assert result.fallback in LLM_CLASSIFIER_FAILURES

    @pytest.mark.asyncio
    async def test_llm_naming_an_invalid_category_is_llm_error(self, llm_config):
        classifier = WriteClassifier(
            llm_config, openai_client=completion_client('{"primary": "nonsense"}'),
        )

        result = await classifier.classify(_NO_HEURISTIC_MATCH)

        assert result.primary == MemoryCategory.observations_and_summaries
        assert result.fallback is ClassificationFallback.llm_error

    @pytest.mark.asyncio
    async def test_successful_llm_classification_is_not_a_fallback(self, llm_config):
        classifier = WriteClassifier(
            llm_config,
            openai_client=completion_client(
                '{"primary": "decisions_and_rationale", "secondary": null, '
                '"confidence": 0.9, "reasoning": "a choice"}'
            ),
        )

        result = await classifier.classify(_NO_HEURISTIC_MATCH)

        assert result.primary == MemoryCategory.decisions_and_rationale
        assert result.fallback is None

    @pytest.mark.asyncio
    async def test_heuristic_only_no_match_is_no_confident_match(self, classifier):
        result = await classifier.classify(_NO_HEURISTIC_MATCH)

        assert result.primary == MemoryCategory.observations_and_summaries
        assert result.fallback is ClassificationFallback.no_confident_match
        assert result.fallback not in LLM_CLASSIFIER_FAILURES

    @pytest.mark.asyncio
    async def test_confident_heuristic_match_is_not_a_fallback(self, classifier):
        result = await classifier.classify(
            'The payment service depends on the billing API'
        )

        assert result.primary == MemoryCategory.entities_and_relations
        assert result.fallback is None

    @pytest.mark.asyncio
    async def test_best_effort_heuristic_match_is_not_a_fallback(self, classifier):
        result = await classifier.classify(
            'We decided to migrate before the deadline changed'
        )

        assert result.confidence < 0.7
        assert result.fallback is None


_LUNA_REQUEST = {'model': 'gpt-6-luna', 'reasoning_effort': 'low', 'max_completion_tokens': 256}
_TODAYS_REQUEST = {'model': 'gpt-4o-mini', 'temperature': 0.0, 'max_tokens': 256}


class TestTheRoutingRequest:
    """classify() sends routing_chat_params(config) and its messages, nothing else (task 9945)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('pin', 'expected'),
        [pytest.param({}, _TODAYS_REQUEST, id='today'),
         pytest.param({'model': 'gpt-6-luna', 'reasoning_effort': 'low'}, _LUNA_REQUEST, id='luna-low')],
    )
    async def test_the_request_is_the_routing_call_and_the_messages(self, llm_only_config, pin, expected):
        for leaf, value in pin.items():
            setattr(llm_only_config.routing, leaf, value)
        client = completion_client('{"primary": "temporal_facts", "confidence": 0.9}')
        classifier = WriteClassifier(llm_only_config, openai_client=client)

        await classifier.classify('x' * 2500)

        messages = [
            {'role': 'system', 'content': CLASSIFICATION_SYSTEM_PROMPT},
            {'role': 'user', 'content': 'x' * 2000},
        ]
        sent = client.chat.completions.create.call_args.kwargs
        assert sent == expected | {'messages': messages}
        assert sent == routing_chat_params(llm_only_config) | {'messages': messages}

    @pytest.mark.asyncio
    async def test_a_pinned_model_without_its_endpoint_degrades_and_sends_nothing(self, llm_only_config):
        """The leaves unset (a missing ${OPENAI_PLAN_PROXY_*}) never fall back onto the llm provider."""
        with mock_openai_server() as server:
            llm_only_config.llm.providers.openai.api_url = server.base_url
            llm_only_config.routing.model = 'gpt-6-luna'
            classifier = WriteClassifier(llm_only_config)

            result = await classifier.classify(_NO_HEURISTIC_MATCH)
            await classifier.close()

            assert server.requests_to('/chat/completions') == []
        assert result.fallback is ClassificationFallback.llm_error
        assert 'routing.openai_api_url' in result.reasoning
