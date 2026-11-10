"""Tests for scripts/replay_overreach_extraction.py (task 9040).

The replay calls graphiti_core's own extract_nodes and extract_edges, so these
tests drive the real extraction functions with a fake LLM that answers by
prompt name. No FalkorDB handle and no driver double exists anywhere here:
the replay never needs one.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from _fm_helpers import load_script_module
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig as GraphitiLLMConfig
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.prompts.models import Message
from pydantic import BaseModel

from fused_memory.backends.llm_token_usage import AttributingTokenUsageTracker, LlmTokenUsage
from fused_memory.config.schema import FusedMemoryConfig

SCRIPTS = Path(__file__).parent.parent / 'scripts'
mod = load_script_module(SCRIPTS / 'replay_overreach_extraction.py', mod_name='replay_overreach_extraction')

NODE_PROMPT, EDGE_PROMPT = 'extract_nodes.extract_text', 'extract_edges.edge'
VALID_AT = datetime(2026, 9, 17, 20, 31, 43, 521296, tzinfo=UTC)
PREVIOUS = ('Earlier the team weighed Gamma for the queue.', 'Later Delta shipped to production.')


class _FakeLLM(LLMClient):
    """Answers the node prompt with *entities* and the edge prompt with *edges*; records every call."""

    def __init__(
        self, entities: tuple[str, ...] = ('Alpha', 'Beta'),
        edges: tuple[tuple[str, str, str], ...] = (('Alpha', 'Beta', 'Alpha adopts Beta'),),
        *, model: str = 'fake-extractor',
    ) -> None:
        super().__init__(GraphitiLLMConfig(model=model))
        self.token_tracker = AttributingTokenUsageTracker()
        self.entities = entities
        self.edges = edges
        self.calls: list[tuple[str | None, list[Message]]] = []

    async def _generate_response(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError('the replay calls generate_response, never the private hook')

    async def generate_response(
        self, messages: list[Message], response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None, model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None, prompt_name: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append((prompt_name, messages))
        if prompt_name == NODE_PROMPT:
            self.token_tracker.record(prompt_name, 100, 10)
            return {'extracted_entities': [{'name': name, 'entity_type_id': 0} for name in self.entities]}
        if prompt_name == EDGE_PROMPT:
            self.token_tracker.record(prompt_name, 300, 40)
            return {'edges': [
                {'source_entity_name': source, 'target_entity_name': target,
                 'relation_type': 'RELATES', 'fact': fact}
                for source, target, fact in self.edges
            ]}
        raise AssertionError(f'unexpected prompt {prompt_name!r}')

    def prompt(self, prompt_name: str) -> str:
        (messages,) = [messages for name, messages in self.calls if name == prompt_name]
        return '\n'.join(message.content for message in messages)


def _input(**overrides: Any) -> Any:
    fields: dict[str, Any] = {
        'graph': 'dark_factory', 'uuid': 'ep-1', 'stratum': 'ruling_lexeme',
        'content': 'Ruling: Alpha adopts Beta for the merge queue.', 'source': 'text',
        'source_description': 'add_memory:decisions_and_rationale', 'valid_at': VALID_AT,
        'previous': PREVIOUS,
    }
    return mod.ReplayInput(**(fields | overrides))


async def _replay(llm: _FakeLLM, arm: str = 'control', **overrides: Any) -> Any:
    return await mod.replay_episode(llm, _input(**overrides), mod.ARMS[arm])


class TestTheArms:
    @pytest.mark.asyncio
    async def test_o1_instructions_reach_both_the_node_and_the_edge_prompt(self) -> None:
        llm = _FakeLLM()
        await _replay(llm, 'o1')
        assert mod.O1_INSTRUCTIONS in llm.prompt(NODE_PROMPT)
        assert mod.O1_INSTRUCTIONS in llm.prompt(EDGE_PROMPT)

    @pytest.mark.asyncio
    async def test_the_control_arm_carries_no_instructions(self) -> None:
        llm = _FakeLLM()
        await _replay(llm, 'control')
        first_line = mod.O1_INSTRUCTIONS.strip().splitlines()[0]
        assert first_line not in llm.prompt(NODE_PROMPT)
        assert first_line not in llm.prompt(EDGE_PROMPT)
        assert mod.ARMS['control'].instructions is None

    def test_the_o1_version_is_a_hash_of_its_text(self) -> None:
        assert mod.O1_INSTRUCTIONS_VERSION.startswith('o1-v1-')
        assert mod.ARMS['o1'].instructions_version == mod.O1_INSTRUCTIONS_VERSION
        assert mod.ARMS['control'].instructions_version == mod.NO_INSTRUCTIONS


class TestTheEpisodeContext:
    @pytest.mark.asyncio
    async def test_a_text_episode_uses_graphitis_text_node_prompt(self) -> None:
        llm = _FakeLLM()
        await _replay(llm)
        assert [name for name, _ in llm.calls] == [NODE_PROMPT, EDGE_PROMPT]

    @pytest.mark.asyncio
    async def test_the_episode_body_reaches_both_prompts(self) -> None:
        llm = _FakeLLM()
        await _replay(llm)
        assert 'Alpha adopts Beta for the merge queue.' in llm.prompt(NODE_PROMPT)
        assert 'Alpha adopts Beta for the merge queue.' in llm.prompt(EDGE_PROMPT)

    @pytest.mark.asyncio
    async def test_previous_episodes_reach_the_edge_prompt_in_order(self) -> None:
        llm = _FakeLLM()
        await _replay(llm)
        edge_prompt = llm.prompt(EDGE_PROMPT)
        assert all(previous in edge_prompt for previous in PREVIOUS)
        assert edge_prompt.index(PREVIOUS[0]) < edge_prompt.index(PREVIOUS[1])

    @pytest.mark.asyncio
    async def test_the_episodes_valid_at_is_the_edge_reference_time(self) -> None:
        llm = _FakeLLM()
        await _replay(llm)
        assert str(VALID_AT) in llm.prompt(EDGE_PROMPT)


class TestMinting:
    @pytest.mark.asyncio
    async def test_facts_equal_under_graphitis_exact_normalisation_collapse(self) -> None:
        llm = _FakeLLM(edges=(
            ('Alpha', 'Beta', 'Alpha adopts Beta'),
            ('Alpha', 'Beta', '  alpha   ADOPTS beta '),
            ('Alpha', 'Beta', 'Alpha rejects Beta'),
            ('Beta', 'Alpha', 'Alpha adopts Beta'),
        ))
        result = await _replay(llm)
        assert [(m.source_name, m.target_name, m.fact) for m in result.minted] == [
            ('Alpha', 'Beta', 'Alpha adopts Beta'),
            ('Alpha', 'Beta', 'Alpha rejects Beta'),
            ('Beta', 'Alpha', 'Alpha adopts Beta'),
        ]

    @pytest.mark.asyncio
    async def test_an_edge_naming_an_unextracted_entity_is_not_minted(self) -> None:
        llm = _FakeLLM(edges=(('Alpha', 'Zeta', 'Alpha needs Zeta'), ('Alpha', 'Beta', 'Alpha adopts Beta')))
        result = await _replay(llm)
        assert [m.fact for m in result.minted] == ['Alpha adopts Beta']

    @pytest.mark.asyncio
    async def test_every_minted_edge_has_its_own_uuid(self) -> None:
        llm = _FakeLLM(edges=(('Alpha', 'Beta', 'Alpha adopts Beta'), ('Beta', 'Alpha', 'Beta serves Alpha')))
        result = await _replay(llm)
        uuids = [m.edge_uuid for m in result.minted]
        assert len(set(uuids)) == 2 and all(isinstance(u, str) and u for u in uuids)


class TestTheResult:
    @pytest.mark.asyncio
    async def test_the_result_names_its_arm_model_and_episode(self) -> None:
        result = await _replay(_FakeLLM(model='m-small'), 'o1')
        assert (result.arm, result.model, result.instructions_version) == ('o1', 'm-small', mod.O1_INSTRUCTIONS_VERSION)
        assert (result.graph, result.uuid, result.stratum) == ('dark_factory', 'ep-1', 'ruling_lexeme')

    @pytest.mark.asyncio
    async def test_the_result_records_the_extracted_entity_names(self) -> None:
        result = await _replay(_FakeLLM(entities=('Alpha', 'Beta', 'Gamma')))
        assert result.entities == ('Alpha', 'Beta', 'Gamma')

    @pytest.mark.asyncio
    async def test_tokens_are_measured_per_prompt(self) -> None:
        result = await _replay(_FakeLLM())
        assert result.tokens == {
            'extract_nodes': LlmTokenUsage(input_tokens=100, output_tokens=10, llm_calls=1),
            'extract_edges': LlmTokenUsage(input_tokens=300, output_tokens=40, llm_calls=1),
        }


class TestBuildArmLlm:
    def test_a_model_override_builds_that_model_and_leaves_the_config_alone(
        self, mock_config: FusedMemoryConfig,
    ) -> None:
        before = mock_config.llm.model
        client = mod.build_arm_llm(mock_config, model='m-strong')
        assert client is not None and client.model == 'm-strong'
        assert mock_config.llm.model == before

    def test_no_override_keeps_the_configured_model(self, mock_config: FusedMemoryConfig) -> None:
        client = mod.build_arm_llm(mock_config, model=None)
        assert client is not None and client.model == mock_config.llm.model

    def test_the_client_attributes_its_tokens(self, mock_config: FusedMemoryConfig) -> None:
        client = mod.build_arm_llm(mock_config, model=None)
        assert client is not None and isinstance(client.token_tracker, AttributingTokenUsageTracker)
