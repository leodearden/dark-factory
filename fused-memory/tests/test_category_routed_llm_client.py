"""CategoryRoutedLLMClient: which client a graphiti call lands on (task 9943).

The router is driven through its public surface with two recording clients;
nothing inside it is mocked.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig as GraphitiLLMConfig
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.prompts.models import Message
from graphiti_core.tracer import NoOpTracer
from pydantic import BaseModel

from fused_memory.backends.llm_clients import (
    EXTRACTION_PROMPT_NAMES,
    CategoryRoutedLLMClient,
    extraction_category_scope,
    wrapped_llm_clients,
)
from fused_memory.models.enums import MemoryCategory

_EXTRACTION = (
    'extract_nodes.extract_message',
    'extract_nodes.extract_text',
    'extract_nodes.extract_json',
    'extract_edges.edge',
)

_NOT_EXTRACTION = (
    'dedupe_edges.resolve_edge',
    'dedupe_nodes.nodes',
    'extract_nodes.extract_summaries_batch',
    'extract_nodes.extract_attributes',
    'extract_edges.extract_attributes',
    None,
)


class _Answer(BaseModel):
    name: str


class _RecordingClient(LLMClient):
    """Answers with its own name and records every call it receives."""

    def __init__(self, name: str) -> None:
        super().__init__(GraphitiLLMConfig(model=f'{name}-model'))
        self.name = name
        self.calls: list[dict[str, Any]] = []
        self.tracers: list[Any] = []

    def set_tracer(self, tracer: Any) -> None:
        self.tracers.append(tracer)
        super().set_tracer(tracer)

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
        prompt_name: str | None = None,
    ) -> dict[str, Any]:
        await asyncio.sleep(0)
        self.calls.append({
            'messages': messages, 'response_model': response_model, 'max_tokens': max_tokens,
            'model_size': model_size, 'group_id': group_id, 'prompt_name': prompt_name,
        })
        return {'client': self.name}

    async def _generate_response(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError('generate_response is overridden')


@pytest.fixture
def default() -> _RecordingClient:
    return _RecordingClient('default')


@pytest.fixture
def route() -> _RecordingClient:
    return _RecordingClient('route')


@pytest.fixture
def router(default: _RecordingClient, route: _RecordingClient) -> CategoryRoutedLLMClient:
    return CategoryRoutedLLMClient(default, {MemoryCategory.decisions_and_rationale: route})


async def _answered_by(router: CategoryRoutedLLMClient, prompt_name: str | None) -> str:
    result = await router.generate_response(
        [Message(role='user', content='x')], prompt_name=prompt_name,
    )
    return result['client']


class TestAScopedExtractionPromptRidesItsCategorysRoute:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('prompt_name', _EXTRACTION)
    async def test_an_extraction_prompt_reaches_the_route(
        self, router: CategoryRoutedLLMClient, route: _RecordingClient, prompt_name: str,
    ) -> None:
        messages = [Message(role='user', content='the record')]
        with extraction_category_scope(MemoryCategory.decisions_and_rationale):
            result = await router.generate_response(
                messages, response_model=_Answer, max_tokens=123, group_id='g1', prompt_name=prompt_name,
            )

        assert result == {'client': 'route'}
        assert route.calls == [{
            'messages': messages, 'response_model': _Answer, 'max_tokens': 123,
            'model_size': ModelSize.medium, 'group_id': 'g1', 'prompt_name': prompt_name,
        }]

    @pytest.mark.asyncio
    @pytest.mark.parametrize('prompt_name', _NOT_EXTRACTION)
    async def test_any_other_prompt_stays_on_the_default(
        self, router: CategoryRoutedLLMClient, prompt_name: str | None,
    ) -> None:
        with extraction_category_scope(MemoryCategory.decisions_and_rationale):
            assert await _answered_by(router, prompt_name) == 'default'


class TestAnUnroutedCategoryStaysOnTheDefault:
    @pytest.mark.asyncio
    async def test_outside_any_scope(self, router: CategoryRoutedLLMClient) -> None:
        assert await _answered_by(router, 'extract_edges.edge') == 'default'

    @pytest.mark.asyncio
    @pytest.mark.parametrize('category', [MemoryCategory.entities_and_relations, None])
    async def test_inside_a_scope_without_a_route(
        self, router: CategoryRoutedLLMClient, category: MemoryCategory | None,
    ) -> None:
        with extraction_category_scope(category):
            assert await _answered_by(router, 'extract_edges.edge') == 'default'


class TestTheScope:
    @pytest.mark.asyncio
    async def test_it_crosses_task_creation(self, router: CategoryRoutedLLMClient) -> None:
        with extraction_category_scope(MemoryCategory.decisions_and_rationale):
            answers = await asyncio.gather(
                _answered_by(router, 'extract_nodes.extract_text'),
                _answered_by(router, 'extract_edges.edge'),
            )

        assert answers == ['route', 'route']

    @pytest.mark.asyncio
    async def test_it_is_cleared_on_exit_and_nested_scopes_restore_the_outer(
        self, router: CategoryRoutedLLMClient,
    ) -> None:
        with extraction_category_scope(MemoryCategory.decisions_and_rationale):
            with extraction_category_scope(MemoryCategory.entities_and_relations):
                assert await _answered_by(router, 'extract_edges.edge') == 'default'
            assert await _answered_by(router, 'extract_edges.edge') == 'route'

        assert await _answered_by(router, 'extract_edges.edge') == 'default'


class TestTheRoutersSurface:
    def test_set_tracer_reaches_every_inner_client(
        self, router: CategoryRoutedLLMClient, default: _RecordingClient, route: _RecordingClient,
    ) -> None:
        tracer = NoOpTracer()

        router.set_tracer(tracer)

        assert default.tracers == [tracer]
        assert route.tracers == [tracer]

    def test_wrapped_clients_are_the_default_then_its_routes(
        self, router: CategoryRoutedLLMClient, default: _RecordingClient, route: _RecordingClient,
    ) -> None:
        assert wrapped_llm_clients(router) == (default, route)
        assert wrapped_llm_clients(default) == (default,)

    def test_it_presents_the_defaults_model_and_tracker(
        self, router: CategoryRoutedLLMClient, default: _RecordingClient,
    ) -> None:
        assert router.model == default.model
        assert router.token_tracker is default.token_tracker

    def test_the_routed_prompts_are_graphitis_four_extraction_calls(self) -> None:
        assert frozenset(_EXTRACTION) == EXTRACTION_PROMPT_NAMES
