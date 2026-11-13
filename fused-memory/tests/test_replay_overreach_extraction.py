"""Tests for scripts/replay_overreach_extraction.py (task 9040).

The replay calls graphiti_core's own extract_nodes and extract_edges, so these
tests drive the real extraction functions with a fake LLM that answers by
prompt name. No FalkorDB handle and no driver double exists anywhere here:
the replay never needs one.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from _fm_helpers import load_script_module
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig as GraphitiLLMConfig
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.llm_client.token_tracker import TokenUsageTracker
from graphiti_core.prompts.extract_edges import ExtractedEdges
from graphiti_core.prompts.extract_nodes import ExtractedEntities
from graphiti_core.prompts.models import Message
from graphiti_core.search.search_utils import RELEVANT_SCHEMA_LIMIT
from pydantic import BaseModel
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

from fused_memory.backends.falkor_read_only import ReadOnlyGraphReader
from fused_memory.backends.falkor_string_bounds import exact_string_upper_bound
from fused_memory.backends.llm_token_usage import (
    AttributingTokenUsageTracker,
    LlmTokenUsage,
    measure_llm_tokens,
)
from fused_memory.config.schema import FusedMemoryConfig

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

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
        *, model: str = 'fake-extractor', reasoning_effort: str | None = None,
    ) -> None:
        super().__init__(GraphitiLLMConfig(model=model))
        self.token_tracker = AttributingTokenUsageTracker()
        self.reasoning_effort = reasoning_effort
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

    def test_the_o1_version_is_the_pre_registered_hash_of_its_text(self) -> None:
        """Tripwire: design.md §2.1 pre-registered this version, and every o1 record in replay.jsonl
        carries it. Changing O1_INSTRUCTIONS makes a new arm, with a new version and registration."""
        text_hash = hashlib.sha256(mod.O1_INSTRUCTIONS.encode()).hexdigest()[:12]
        assert mod.O1_INSTRUCTIONS_VERSION == 'o1-v1-99b813d73cba' == f'o1-v1-{text_hash}'
        assert mod.ARMS['o1'].instructions_version == mod.O1_INSTRUCTIONS_VERSION
        assert mod.ARMS['control'].instructions_version == mod.NO_INSTRUCTIONS

    def test_o6_is_two_arms_one_per_reasoning_effort(self) -> None:
        """esc-9065-2: 'o6' could only ever mean one effort, so each measured effort is its own arm."""
        assert sorted(mod.ARMS) == ['control', 'o1', 'o6-low', 'o6-medium']

    @pytest.mark.parametrize('effort', ['low', 'medium'])
    def test_each_o6_arm_is_the_controls_instructions_on_the_stronger_model_at_its_effort(
        self, effort: str,
    ) -> None:
        o6 = mod.ARMS[f'o6-{effort}']
        assert o6.llm == mod.ExtractionLlm(mod.O6_MODEL, effort)
        assert (o6.instructions, o6.instructions_version) == (None, mod.NO_INSTRUCTIONS)

    @pytest.mark.parametrize('arm', ['control', 'o1'])
    def test_the_control_and_o1_extract_on_the_configured_client(self, arm: str) -> None:
        llm = mod.ARMS[arm].llm
        assert llm == mod.ExtractionLlm()
        assert (llm.model, llm.reasoning_effort) == (None, None)

    def test_an_extraction_llm_is_a_frozen_hashable_key(self) -> None:
        llm = mod.ExtractionLlm(mod.O6_MODEL, 'low')
        assert {llm: 1}[mod.ExtractionLlm(mod.O6_MODEL, 'low')] == 1
        for field in dataclasses.fields(llm):
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(llm, field.name, 'medium')

    @pytest.mark.asyncio
    async def test_the_o6_low_arm_splices_nothing_and_runs_on_its_client(self) -> None:
        llm = _FakeLLM(model=mod.O6_MODEL, reasoning_effort='low')
        result = await _replay(llm, 'o6-low')
        first_line = mod.O1_INSTRUCTIONS.strip().splitlines()[0]
        assert first_line not in llm.prompt(NODE_PROMPT)
        assert first_line not in llm.prompt(EDGE_PROMPT)
        assert (result.arm, result.model, result.reasoning_effort) == ('o6-low', mod.O6_MODEL, 'low')

    @pytest.mark.asyncio
    async def test_an_arm_refuses_a_client_at_another_effort_before_any_call(self) -> None:
        llm = _FakeLLM(model=mod.O6_MODEL, reasoning_effort='medium')
        with pytest.raises(mod.ArmClientMismatch) as raised:
            await _replay(llm, 'o6-low')
        assert all(name in str(raised.value) for name in ('o6-low', "'low'", "'medium'"))
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_an_effort_arm_refuses_a_client_sending_no_effort(self) -> None:
        """A client sending no effort runs at the API's default, which nobody chose."""
        llm = _FakeLLM(model=mod.O6_MODEL)
        with pytest.raises(mod.ArmClientMismatch):
            await _replay(llm, 'o6-low')
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_an_arm_refuses_a_client_on_another_model_before_any_call(self) -> None:
        llm = _FakeLLM(model='m-small', reasoning_effort='medium')
        with pytest.raises(mod.ArmClientMismatch) as raised:
            await _replay(llm, 'o6-medium')
        assert all(name in str(raised.value) for name in ('o6-medium', mod.O6_MODEL, 'm-small'))
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_the_control_refuses_an_effort_client(self) -> None:
        """The control's records claim the configured client, so an effort client must not run it."""
        llm = _FakeLLM(reasoning_effort='low')
        with pytest.raises(mod.ArmClientMismatch):
            await _replay(llm, 'control')
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_a_client_whose_tokens_cannot_be_attributed_is_refused_before_any_call(self) -> None:
        llm = _FakeLLM()
        llm.token_tracker = TokenUsageTracker()
        with pytest.raises(mod.UnmeasurableClient):
            await _replay(llm)
        assert llm.calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize('arm', ['control', 'o1'])
    async def test_an_arm_naming_no_model_takes_any_client(self, arm: str) -> None:
        result = await _replay(_FakeLLM(model='m-anything'), arm)
        assert result.model == 'm-anything'


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
    async def test_a_replayed_record_carries_the_clients_reasoning_effort(self) -> None:
        o6_low = await _replay(_FakeLLM(model=mod.O6_MODEL, reasoning_effort='low'), 'o6-low')
        control = await _replay(_FakeLLM(), 'control')
        assert mod.replayed_record(o6_low, _input())['reasoning_effort'] == 'low'
        assert mod.replayed_record(control, _input())['reasoning_effort'] is None

    def test_a_failed_record_carries_the_clients_model_and_reasoning_effort(self) -> None:
        llm = _FakeLLM(model=mod.O6_MODEL, reasoning_effort='medium')
        record = mod.failed_record(mod.ARMS['o6-medium'], llm, _worksheet_row(), RuntimeError('provider down'))
        assert (record['arm'], record['model'], record['reasoning_effort']) == ('o6-medium', mod.O6_MODEL, 'medium')
        assert (record['kind'], record['error_type']) == ('failed', 'RuntimeError')

    @pytest.mark.asyncio
    async def test_tokens_are_measured_per_prompt(self) -> None:
        result = await _replay(_FakeLLM())
        assert result.tokens == {
            'extract_nodes': LlmTokenUsage(input_tokens=100, output_tokens=10, llm_calls=1),
            'extract_edges': LlmTokenUsage(input_tokens=300, output_tokens=40, llm_calls=1),
        }


class _Answer(BaseModel):
    answer: str


class _RecordingSdk:
    """A fake AsyncOpenAI: records the kwargs of every Responses and chat call it is sent, and
    answers each Responses call with the JSON *answers* holds for its text_format."""

    def __init__(self, answers: dict[type[BaseModel], dict[str, Any]] | None = None) -> None:
        self.answers = answers if answers is not None else {_Answer: {'answer': 'yes'}}
        self.sent: list[dict[str, Any]] = []
        self.responses = SimpleNamespace(parse=self._parse)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _parse(self, **kwargs: Any) -> Any:
        self.sent.append(kwargs)
        return SimpleNamespace(
            output_text=json.dumps(self.answers[kwargs['text_format']]),
            usage=SimpleNamespace(input_tokens=11, output_tokens=7),
        )

    async def _create(self, **kwargs: Any) -> Any:
        self.sent.append(kwargs)
        raise AssertionError('a JSON-mode chat completion cannot carry the effort')


EFFORT_MAX_TOKENS = 4096
"""config.yaml's llm.max_tokens: the client's own cap, which extract_nodes uses because it passes none."""
EXTRACT_EDGES_MAX_TOKENS = 16384
"""graphiti_core 0.28.2 edge_operations.py::extract_edges' own cap, passed on every edge call."""
EXTRACTION_ANSWERS: dict[type[BaseModel], dict[str, Any]] = {
    ExtractedEntities: {'extracted_entities': [
        {'name': 'Alpha', 'entity_type_id': 0}, {'name': 'Beta', 'entity_type_id': 0},
    ]},
    ExtractedEdges: {'edges': [
        {'source_entity_name': 'Alpha', 'target_entity_name': 'Beta', 'relation_type': 'ADOPTS',
         'fact': 'Alpha adopts Beta'},
    ]},
}


def _effort_client(
    effort: str = 'low', sdk: _RecordingSdk | None = None,
) -> tuple[Any, _RecordingSdk, OpenAIClient]:
    """An OpenAIClient as build_llm_client builds one for O6_MODEL, wrapped at *effort*."""
    sdk = sdk if sdk is not None else _RecordingSdk()
    built = OpenAIClient(
        config=GraphitiLLMConfig(api_key='test', model=mod.O6_MODEL, small_model=mod.O6_MODEL, temperature=0.0),
        client=sdk, max_tokens=EFFORT_MAX_TOKENS,
    )
    built.token_tracker = AttributingTokenUsageTracker()
    return mod.ExplicitEffortOpenAIClient(built, effort), sdk, built


async def _structured_call(client: Any, response_model: type[BaseModel] | None = _Answer) -> Any:
    messages = [Message(role='system', content='extract'), Message(role='user', content='the record')]
    return await client.generate_response(
        messages, response_model=response_model, prompt_name='extract_nodes.extract_text',
    )


class TestTheExplicitEffortClient:
    @pytest.mark.asyncio
    async def test_it_sends_the_effort_and_no_other_sampling_control(self) -> None:
        client, sdk, _ = _effort_client('low')
        assert await _structured_call(client) == {'answer': 'yes'}
        (sent,) = sdk.sent
        assert 'temperature' not in sent and 'text' not in sent
        assert sent['reasoning'] == {'effort': 'low'}
        assert sent['model'] == mod.O6_MODEL
        assert sent['max_output_tokens'] == client.max_tokens == EFFORT_MAX_TOKENS
        assert set(sent) == {'model', 'input', 'max_output_tokens', 'text_format', 'reasoning'}

    @pytest.mark.asyncio
    async def test_it_keeps_the_built_clients_token_tracker_so_the_call_is_attributed(self) -> None:
        client, _, built = _effort_client()
        assert client.token_tracker is built.token_tracker
        async with measure_llm_tokens(client) as measured:
            await _structured_call(client)
        assert measured.usage == LlmTokenUsage(input_tokens=11, output_tokens=7, llm_calls=1)

    @pytest.mark.parametrize('effort', ['low', 'medium'])
    def test_it_declares_the_effort_it_sends(self, effort: str) -> None:
        client, _, _ = _effort_client(effort)
        assert client.reasoning_effort == effort
        assert mod.client_reasoning_effort(client) == effort

    @pytest.mark.asyncio
    async def test_a_json_mode_call_is_refused_before_anything_is_sent(self) -> None:
        """A JSON-mode chat completion cannot carry reasoning.effort, so it is never sent without it."""
        client, sdk, _ = _effort_client()
        with pytest.raises(NotImplementedError, match='reasoning.effort'):
            await _structured_call(client, response_model=None)
        assert sdk.sent == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize('message', [
        {'role': 'assistant', 'content': 'an earlier answer'},
        {'role': 'user', 'content': [{'type': 'text', 'text': 'a content part'}]},
    ])
    async def test_a_message_graphiti_never_emits_is_refused_before_anything_is_sent(self, message: Any) -> None:
        """graphiti's message conversion emits only system and user text, so through
        generate_response this is unreachable; the hook graphiti calls is where it is refused."""
        client, sdk, _ = _effort_client()
        messages: list[Any] = [{'role': 'system', 'content': 'extract'}, message]
        with pytest.raises(TypeError, match='is not Responses input'):
            await client._create_structured_completion(
                model=mod.O6_MODEL, messages=messages, temperature=None,
                max_tokens=EFFORT_MAX_TOKENS, response_model=_Answer,
            )
        assert sdk.sent == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize('effort', ['low', 'medium'])
    async def test_graphitis_own_extraction_sends_the_effort_and_each_stages_cap(self, effort: str) -> None:
        """design.md §2.2: both stages go through the structured path, so both carry the effort and
        no temperature, and each carries its own output cap."""
        sdk = _RecordingSdk(EXTRACTION_ANSWERS)
        client, _, _ = _effort_client(effort, sdk)
        result = await mod.replay_episode(client, _input(), mod.ARMS[f'o6-{effort}'])
        assert [sent['text_format'] for sent in sdk.sent] == [ExtractedEntities, ExtractedEdges]
        assert [sent['max_output_tokens'] for sent in sdk.sent] == [EFFORT_MAX_TOKENS, EXTRACT_EDGES_MAX_TOKENS]
        for sent in sdk.sent:
            assert set(sent) == {'model', 'input', 'max_output_tokens', 'text_format', 'reasoning'}
            assert sent['reasoning'] == {'effort': effort}
            assert [message['role'] for message in sent['input']] == ['system', 'user']
        assert [(m.source_name, m.target_name, m.fact) for m in result.minted] == [
            ('Alpha', 'Beta', 'Alpha adopts Beta'),
        ]
        assert (result.model, result.reasoning_effort) == (mod.O6_MODEL, effort)


class TestBuildArmLlm:
    def test_the_default_client_is_build_llm_clients_own_on_the_configured_model(
        self, mock_config: FusedMemoryConfig,
    ) -> None:
        client = mod.build_arm_llm(mock_config, mod.ExtractionLlm())
        assert type(client) is OpenAIClient and client.model == mock_config.llm.model
        assert not hasattr(client, 'reasoning_effort')
        assert mod.client_reasoning_effort(client) is None

    def test_a_model_override_builds_that_model_and_leaves_the_config_alone(
        self, mock_config: FusedMemoryConfig,
    ) -> None:
        before = mock_config.llm.model
        client = mod.build_arm_llm(mock_config, mod.ExtractionLlm('m-strong'))
        assert client is not None and client.model == 'm-strong'
        assert mock_config.llm.model == before

    def test_the_client_attributes_its_tokens(self, mock_config: FusedMemoryConfig) -> None:
        client = mod.build_arm_llm(mock_config, mod.ExtractionLlm())
        assert client is not None and isinstance(client.token_tracker, AttributingTokenUsageTracker)

    def test_a_set_effort_wraps_the_built_client_in_the_effort_client(self, mock_config: FusedMemoryConfig) -> None:
        before = mock_config.llm.model
        client = mod.build_arm_llm(mock_config, mod.ExtractionLlm(mod.O6_MODEL, 'medium'))
        assert isinstance(client, mod.ExplicitEffortOpenAIClient)
        assert (client.model, client.small_model) == (mod.O6_MODEL, mod.O6_MODEL)
        assert client.reasoning_effort == 'medium'
        assert isinstance(client.token_tracker, AttributingTokenUsageTracker)
        assert mock_config.llm.model == before

    def test_a_set_effort_on_a_chat_completions_only_client_class_is_refused(
        self, mock_config: FusedMemoryConfig,
    ) -> None:
        generic = mock_config.model_copy(deep=True)
        generic.llm.client_class = 'openai_generic'
        with pytest.raises(mod.ArmClientMismatch) as raised:
            mod.build_arm_llm(generic, mod.ExtractionLlm(mod.O6_MODEL, 'low'))
        assert 'openai_generic' in str(raised.value) and 'low' in str(raised.value)


# --------------------------------------------------------------------------- #
# The read side: an episode's production inputs, read-only
# --------------------------------------------------------------------------- #

STORED_VALID_AT = '2026-09-17T20:31:43.521296+00:00'


class _Graph:
    """A graph double answering each Cypher constant with canned rows; records what it was sent."""

    name = 'dark_factory'

    def __init__(self, answers: dict[str, list[list[Any]]]) -> None:
        self.answers = answers
        self.sent: list[tuple[str, dict[str, Any] | None]] = []

    async def ro_query(self, cypher: str, params: dict[str, Any] | None = None) -> Any:
        self.sent.append((cypher, params))
        return SimpleNamespace(result_set=self.answers[cypher])


def _episode_graph(previous_desc: list[list[Any]] | None = None, *, own: list[list[Any]] | None = None) -> _Graph:
    return _Graph({
        mod.EPISODE_CYPHER: own if own is not None else [
            [STORED_VALID_AT, 'text', 'add_memory:decisions_and_rationale', 'dark_factory'],
        ],
        mod.PREVIOUS_EPISODES_CYPHER: previous_desc if previous_desc is not None else [
            ['third', '2026-09-17T19:00:00+00:00'], ['second', '2026-09-17T18:00:00+00:00'],
            ['first', '2026-09-17T17:00:00+00:00'],
        ],
    })


def _worksheet_row(uuid: str = 'ep-1', **overrides: Any) -> dict[str, Any]:
    return {'graph': 'dark_factory', 'uuid': uuid, 'stratum': 'ruling_lexeme',
            'content': f'the body of {uuid}', 'minted': []} | overrides


async def _read(graph: _Graph, row: dict[str, Any] | None = None) -> Any:
    reader = mod.EpisodeContextReader(ReadOnlyGraphReader(graph_name='dark_factory', graph=graph))
    return await reader.read(row or _worksheet_row())


class TestEpisodeContextReader:
    @pytest.mark.asyncio
    async def test_the_episodes_own_fields_come_from_its_own_row(self) -> None:
        replay_input = await _read(_episode_graph())
        assert (replay_input.graph, replay_input.uuid, replay_input.stratum) == ('dark_factory', 'ep-1', 'ruling_lexeme')
        assert replay_input.content == 'the body of ep-1'
        assert (replay_input.source, replay_input.source_description) == ('text', 'add_memory:decisions_and_rationale')
        assert replay_input.valid_at == datetime(2026, 9, 17, 20, 31, 43, 521296, tzinfo=UTC)

    @pytest.mark.asyncio
    async def test_previous_episodes_are_put_back_in_chronological_order(self) -> None:
        replay_input = await _read(_episode_graph())
        assert replay_input.previous == ('first', 'second', 'third')

    @pytest.mark.asyncio
    async def test_the_previous_episode_query_is_graphitis_retrieve_episodes_rule(self) -> None:
        graph = _episode_graph()
        await _read(graph)
        (own_params,) = [params for cypher, params in graph.sent if cypher == mod.EPISODE_CYPHER]
        (previous_params,) = [params for cypher, params in graph.sent if cypher == mod.PREVIOUS_EPISODES_CYPHER]
        assert own_params == {'uuid': 'ep-1'}
        assert previous_params == {
            'reference_time': STORED_VALID_AT, 'group_ids': ['dark_factory'], 'source': 'text',
            'uuid': 'ep-1', 'num_episodes': RELEVANT_SCHEMA_LIMIT,
        }
        assert mod.RELEVANT_SCHEMA_LIMIT is RELEVANT_SCHEMA_LIMIT and RELEVANT_SCHEMA_LIMIT == 10

    def test_the_valid_at_bound_is_one_no_range_index_can_serve(self) -> None:
        assert exact_string_upper_bound('e.valid_at', 'reference_time', inclusive=True) in mod.PREVIOUS_EPISODES_CYPHER

    @pytest.mark.asyncio
    async def test_an_episode_the_graph_no_longer_holds_is_a_named_error(self) -> None:
        with pytest.raises(mod.EpisodeMissing, match='ep-gone'):
            await _read(_episode_graph(own=[]), _worksheet_row('ep-gone'))

    @pytest.mark.asyncio
    async def test_only_the_read_primitive_reaches_the_graph(self) -> None:
        graph = _episode_graph()
        await _read(graph)
        assert {cypher for cypher, _ in graph.sent} == {mod.EPISODE_CYPHER, mod.PREVIOUS_EPISODES_CYPHER}


def _sizes(**episodes: int) -> dict[str, Any]:
    sizes: dict[str, dict[str, dict[str, int]]] = {}
    for cell, count in episodes.items():
        graph, stratum = cell.split('__')
        sizes.setdefault(graph, {})[stratum] = {'episodes': count, 'minted': 0, 'corroborated': 0}
    return {'sample': {'sizes': sizes}}


class TestCheckWorksheet:
    ROWS = [
        _worksheet_row('a1'), _worksheet_row('a2'),
        _worksheet_row('b1', graph='reify', stratum='other_decisions'),
    ]

    def _check(self, **episodes: int) -> None:
        mod.check_worksheet(self.ROWS, mod.baseline_cells(_sizes(**episodes)))

    def test_matching_cell_counts_pass(self) -> None:
        self._check(dark_factory__ruling_lexeme=2, reify__other_decisions=1)

    def test_a_short_cell_is_named(self) -> None:
        with pytest.raises(mod.WorksheetMismatch, match='dark_factory/ruling_lexeme'):
            self._check(dark_factory__ruling_lexeme=3, reify__other_decisions=1)

    def test_a_cell_the_baseline_lacks_is_named(self) -> None:
        with pytest.raises(mod.WorksheetMismatch, match='reify/other_decisions'):
            self._check(dark_factory__ruling_lexeme=2)

    def test_a_baseline_cell_the_worksheet_lacks_is_named(self) -> None:
        with pytest.raises(mod.WorksheetMismatch, match='reify/decision_anchor'):
            self._check(dark_factory__ruling_lexeme=2, reify__other_decisions=1, reify__decision_anchor=8)

    @pytest.mark.parametrize('report', [{}, {'sample': {}}, {'sample': {'sizes': {'dark_factory': {'x': {}}}}}])
    def test_a_report_without_sample_sizes_is_a_replay_file_error(self, report: dict[str, Any]) -> None:
        with pytest.raises(mod.ReplayFileError, match='sample.sizes'):
            mod.baseline_cells(report)


# --------------------------------------------------------------------------- #
# The CLI: replay and worksheet, end to end in tmp_path
# --------------------------------------------------------------------------- #

CLI_ROWS = [
    _worksheet_row('ep-a'), _worksheet_row('ep-b'),
    _worksheet_row('ep-c', graph='reify', stratum='other_decisions'),
]
CLI_BASELINE = _sizes(dark_factory__ruling_lexeme=2, reify__other_decisions=1)


class _CountingLLM(_FakeLLM):
    """A _FakeLLM that fails for any episode whose body names one of *failing*."""

    def __init__(self, failing: frozenset[str] = frozenset(), **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.failing = failing

    async def generate_response(self, messages: list[Message], *args: Any, **kwargs: Any) -> dict[str, Any]:
        text = '\n'.join(message.content for message in messages)
        if any(f'the body of {uuid}' in text for uuid in self.failing):
            raise RuntimeError('provider down')
        return await super().generate_response(messages, *args, **kwargs)

    def episodes_called(self) -> list[str]:
        return [
            uuid for name, messages in self.calls if name == NODE_PROMPT
            for uuid in ('ep-a', 'ep-b', 'ep-c') if f'the body of {uuid}' in messages[1].content
        ]


def _cli_inputs(tmp_path: Path, rows: list[dict[str, Any]] = CLI_ROWS,
                baseline: dict[str, Any] = CLI_BASELINE) -> dict[str, Path]:
    paths = {
        'worksheet': tmp_path / 'worksheet-4716.jsonl', 'baseline': tmp_path / 'report.json',
        'replay': tmp_path / 'out' / 'replay.jsonl', 'blind': tmp_path / 'out' / 'blind.jsonl',
    }
    paths['worksheet'].write_text(''.join(json.dumps(row) + '\n' for row in rows))
    paths['baseline'].write_text(json.dumps(baseline))
    return paths


def _replay_argv(paths: dict[str, Path], *arms: str) -> list[str]:
    argv = ['replay', '--worksheet', str(paths['worksheet']), '--baseline-report', str(paths['baseline']),
            '--out', str(paths['replay'])]
    for arm in arms or ('control', 'o1'):
        argv += ['--arm', arm]
    return argv


def _run_replay_on(paths: dict[str, Path], factory: Any, *arms: str) -> int:
    return mod.run(
        _replay_argv(paths, *arms), llm_factory=factory,
        reader_factory=lambda graph: ReadOnlyGraphReader(graph_name=graph, graph=_episode_graph()),
    )


def _run_replay(paths: dict[str, Path], llm: _FakeLLM, *arms: str) -> int:
    return _run_replay_on(paths, lambda llm_spec: llm, *arms)


CONFIGURED = 'm-configured'


CONTROL_LLM = mod.ExtractionLlm()
O6_LOW_LLM = mod.ExtractionLlm(mod.O6_MODEL, 'low')
O6_MEDIUM_LLM = mod.ExtractionLlm(mod.O6_MODEL, 'medium')


class _ClientsByLlm:
    """An llm_factory building one _CountingLLM per ExtractionLlm it is asked for, on that model
    (None is CONFIGURED) and declaring that effort."""

    def __init__(
        self, *, unavailable: frozenset[Any] = frozenset(),
        edges: dict[Any, tuple[tuple[str, str, str], ...]] | None = None,
    ) -> None:
        self.unavailable = unavailable
        self.edges = edges or {}
        self.asked: list[Any] = []
        self.clients: dict[Any, _CountingLLM] = {}

    def __call__(self, spec: Any) -> _CountingLLM | None:
        self.asked.append(spec)
        if spec in self.unavailable:
            return None
        fields: dict[str, Any] = {'model': spec.model or CONFIGURED, 'reasoning_effort': spec.reasoning_effort}
        if spec in self.edges:
            fields['edges'] = self.edges[spec]
        self.clients[spec] = _CountingLLM(**fields)
        return self.clients[spec]


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


class TestReplayCommand:
    def test_one_record_per_arm_and_episode(self, tmp_path: Path) -> None:
        paths = _cli_inputs(tmp_path)
        assert _run_replay(paths, _CountingLLM()) == 0
        records = _records(paths['replay'])
        assert sorted((r['arm'], r['uuid']) for r in records) == sorted(
            (arm, row['uuid']) for arm in ('control', 'o1') for row in CLI_ROWS
        )
        assert all(r['kind'] == 'replayed' for r in records)

    def test_a_record_carries_its_arm_model_episode_edges_and_tokens(self, tmp_path: Path) -> None:
        paths = _cli_inputs(tmp_path)
        assert _run_replay(paths, _CountingLLM(model='m-small'), 'o1') == 0
        record = _records(paths['replay'])[0]
        assert (record['arm'], record['model'], record['instructions_version']) == (
            'o1', 'm-small', mod.O1_INSTRUCTIONS_VERSION,
        )
        assert (record['graph'], record['uuid'], record['stratum']) == ('dark_factory', 'ep-a', 'ruling_lexeme')
        assert record['entities'] == ['Alpha', 'Beta']
        assert [(m['fact'], m['source_name'], m['target_name']) for m in record['minted']] == [
            ('Alpha adopts Beta', 'Alpha', 'Beta'),
        ]
        assert record['previous_count'] == 3
        assert record['tokens']['extract_nodes'] == {'input_tokens': 100, 'output_tokens': 10, 'llm_calls': 1}

    def test_no_episode_text_is_recorded(self, tmp_path: Path) -> None:
        paths = _cli_inputs(tmp_path)
        assert _run_replay(paths, _CountingLLM()) == 0
        text = paths['replay'].read_text()
        assert 'the body of' not in text
        assert not any(previous in text for previous in ('first', 'second', 'third'))

    def test_a_rerun_skips_what_is_recorded(self, tmp_path: Path) -> None:
        paths = _cli_inputs(tmp_path)
        assert _run_replay(paths, _CountingLLM(), 'control') == 0
        second = _CountingLLM()
        assert _run_replay(paths, second, 'control', 'o1') == 0
        assert sorted(second.episodes_called()) == ['ep-a', 'ep-b', 'ep-c']
        assert len(_records(paths['replay'])) == 6

    def test_a_failed_episode_is_recorded_and_the_run_finishes_then_exits_1(self, tmp_path: Path) -> None:
        paths = _cli_inputs(tmp_path)
        llm = _CountingLLM(failing=frozenset({'ep-b'}))
        assert _run_replay(paths, llm, 'control') == 1
        records = _records(paths['replay'])
        assert [(r['kind'], r['uuid']) for r in records] == [
            ('replayed', 'ep-a'), ('failed', 'ep-b'), ('replayed', 'ep-c'),
        ]
        assert records[1]['error_type'] == 'RuntimeError'

    def test_a_resumed_run_retries_only_the_failure_and_drops_its_line(self, tmp_path: Path) -> None:
        paths = _cli_inputs(tmp_path)
        assert _run_replay(paths, _CountingLLM(failing=frozenset({'ep-b'})), 'control') == 1
        retry = _CountingLLM()
        assert _run_replay(paths, retry, 'control') == 0
        assert retry.episodes_called() == ['ep-b']
        records = _records(paths['replay'])
        assert all(r['kind'] == 'replayed' for r in records) and len(records) == 3

    def test_a_worksheet_off_the_baseline_sample_exits_2_and_writes_nothing(self, tmp_path: Path) -> None:
        paths = _cli_inputs(tmp_path, baseline=_sizes(dark_factory__ruling_lexeme=8, reify__other_decisions=1))
        llm = _CountingLLM()
        assert _run_replay(paths, llm) == 2
        assert not paths['replay'].exists() and llm.calls == []

    def test_an_unmeasurable_client_exits_2_before_any_call(self, tmp_path: Path) -> None:
        paths = _cli_inputs(tmp_path)
        llm = _CountingLLM()
        llm.token_tracker = TokenUsageTracker()
        assert _run_replay(paths, llm) == 2
        assert not paths['replay'].exists() and llm.calls == []

    def test_a_client_on_another_model_than_its_arms_exits_2_before_any_call(self, tmp_path: Path) -> None:
        paths = _cli_inputs(tmp_path)
        llm = _CountingLLM(model='m-small')
        assert _run_replay(paths, llm, 'control', 'o6-low') == 2
        assert not paths['replay'].exists() and llm.calls == []

    def test_a_defect_in_an_episode_read_is_raised_not_recorded_as_a_failure(self, tmp_path: Path) -> None:
        class _Broken:
            async def ro_query(self, cypher: str, params: dict[str, Any] | None = None) -> Any:
                raise KeyError('a defect, not an outage')

        paths = _cli_inputs(tmp_path)
        with pytest.raises(KeyError):
            mod.run(_replay_argv(paths), llm_factory=lambda llm_spec: _CountingLLM(),
                    reader_factory=lambda graph: ReadOnlyGraphReader(graph_name=graph, graph=_Broken()))
        assert _records(paths['replay']) == []

    def test_a_missing_client_exits_1(self, tmp_path: Path) -> None:
        paths = _cli_inputs(tmp_path)
        code = mod.run(_replay_argv(paths), llm_factory=lambda llm_spec: None,
                       reader_factory=lambda graph: ReadOnlyGraphReader(graph_name=graph, graph=_episode_graph()))
        assert code == 1 and not paths['replay'].exists()

    def test_each_distinct_arm_client_gets_one_client_and_every_arm_one_file(self, tmp_path: Path) -> None:
        paths = _cli_inputs(tmp_path)
        factory = _ClientsByLlm()
        assert _run_replay_on(paths, factory, 'control', 'o1', 'o6-low', 'o6-medium') == 0
        assert factory.asked == [CONTROL_LLM, O6_LOW_LLM, O6_MEDIUM_LLM]
        records = _records(paths['replay'])
        assert sorted((r['arm'], r['uuid']) for r in records) == sorted(
            (arm, row['uuid']) for arm in ('control', 'o1', 'o6-low', 'o6-medium') for row in CLI_ROWS
        )
        clients = {arm: {(r['model'], r['reasoning_effort']) for r in records if r['arm'] == arm}
                   for arm in ('control', 'o1', 'o6-low', 'o6-medium')}
        assert clients == {
            'control': {(CONFIGURED, None)}, 'o1': {(CONFIGURED, None)},
            'o6-low': {(mod.O6_MODEL, 'low')}, 'o6-medium': {(mod.O6_MODEL, 'medium')},
        }

    def test_a_resume_adds_both_o6_arms_beside_a_recorded_control(self, tmp_path: Path) -> None:
        paths = _cli_inputs(tmp_path)
        assert _run_replay_on(paths, _ClientsByLlm(), 'control') == 0
        second = _ClientsByLlm()
        assert _run_replay_on(paths, second, 'control', 'o6-low', 'o6-medium') == 0
        assert second.clients[CONTROL_LLM].calls == []
        for spec in (O6_LOW_LLM, O6_MEDIUM_LLM):
            assert sorted(second.clients[spec].episodes_called()) == ['ep-a', 'ep-b', 'ep-c']
        assert len(_records(paths['replay'])) == 9

    def test_a_recorded_effort_arm_whose_client_now_declares_another_effort_exits_2(self, tmp_path: Path) -> None:
        paths = _cli_inputs(tmp_path)
        assert _run_replay_on(paths, _ClientsByLlm(), 'control', 'o6-low') == 0
        before = paths['replay'].read_bytes()

        def medium_for_low(spec: Any) -> _CountingLLM:
            if spec == O6_LOW_LLM:
                return _CountingLLM(model=mod.O6_MODEL, reasoning_effort='medium')
            return _CountingLLM(model=CONFIGURED)

        assert _run_replay_on(paths, medium_for_low, 'control', 'o6-low') == 2
        assert paths['replay'].read_bytes() == before

    def test_a_recorded_record_at_another_effort_than_its_arms_client_exits_2(self, tmp_path: Path) -> None:
        paths = _cli_inputs(tmp_path)
        assert _run_replay_on(paths, _ClientsByLlm(), 'o6-low') == 0
        rewritten = [r | {'reasoning_effort': 'medium'} for r in _records(paths['replay'])]
        paths['replay'].write_text(''.join(json.dumps(r) + '\n' for r in rewritten))
        before = paths['replay'].read_bytes()
        factory = _ClientsByLlm()
        assert _run_replay_on(paths, factory, 'o6-low') == 2
        assert paths['replay'].read_bytes() == before
        assert all(client.calls == [] for client in factory.clients.values())

    def test_a_second_model_for_a_recorded_arm_exits_2(self, tmp_path: Path) -> None:
        paths = _cli_inputs(tmp_path)
        assert _run_replay(paths, _CountingLLM(model='m-small'), 'control') == 0
        before = paths['replay'].read_text()
        assert _run_replay(paths, _CountingLLM(model='m-other'), 'control') == 2
        assert paths['replay'].read_text() == before

    def test_a_record_written_before_9065_resumes_as_effort_none(self, tmp_path: Path) -> None:
        """A 9040-era record has no 'reasoning_effort' key; it reads as the configured client's None."""
        paths = _cli_inputs(tmp_path)
        assert _run_replay_on(paths, _ClientsByLlm(), 'control') == 0
        legacy = [{k: v for k, v in r.items() if k != 'reasoning_effort'} for r in _records(paths['replay'])]
        paths['replay'].write_text(''.join(json.dumps(r) + '\n' for r in legacy))
        second = _ClientsByLlm()
        assert _run_replay_on(paths, second, 'control', 'o6-low') == 0
        assert second.clients[CONTROL_LLM].calls == []
        assert len(_records(paths['replay'])) == 6

    def test_a_factory_refusing_an_arms_client_exits_2_and_writes_nothing(self, tmp_path: Path) -> None:
        paths = _cli_inputs(tmp_path)
        built: list[_CountingLLM] = []

        def refusing(spec: Any) -> _CountingLLM:
            if spec.reasoning_effort is not None:
                raise mod.ArmClientMismatch('this client class cannot send an effort')
            built.append(_CountingLLM(model=CONFIGURED))
            return built[-1]

        assert _run_replay_on(paths, refusing, 'control', 'o6-low') == 2
        assert not paths['replay'].exists()
        assert all(client.calls == [] for client in built)

    def test_an_arm_client_with_no_client_exits_1_before_any_call(self, tmp_path: Path) -> None:
        paths = _cli_inputs(tmp_path)
        factory = _ClientsByLlm(unavailable=frozenset({O6_LOW_LLM}))
        assert _run_replay_on(paths, factory, 'control', 'o6-low') == 1
        assert not paths['replay'].exists()
        assert all(client.calls == [] for client in factory.clients.values())

    def test_the_retired_o6_arm_is_refused(self, tmp_path: Path) -> None:
        paths = _cli_inputs(tmp_path)
        with pytest.raises(SystemExit) as raised:
            _run_replay_on(paths, _ClientsByLlm(), 'control', 'o6')
        assert raised.value.code == 2


def _worksheet_argv(paths: dict[str, Path]) -> list[str]:
    return ['worksheet', '--replay', str(paths['replay']), '--worksheet', str(paths['worksheet']),
            '--out', str(paths['blind'])]


class TestWorksheetCommand:
    def _replayed(self, tmp_path: Path) -> dict[str, Path]:
        paths = _cli_inputs(tmp_path)
        llm = _CountingLLM(edges=(('Alpha', 'Beta', 'Alpha adopts Beta'), ('Beta', 'Alpha', 'Beta serves Alpha')))
        assert _run_replay(paths, llm) == 0
        return paths

    def test_one_row_per_episode_with_its_body_and_every_arms_edges(self, tmp_path: Path) -> None:
        paths = self._replayed(tmp_path)
        assert mod.run(_worksheet_argv(paths)) == 0
        rows = _records(paths['blind'])
        assert [(r['graph'], r['uuid'], r['stratum']) for r in rows] == [
            (row['graph'], row['uuid'], row['stratum']) for row in CLI_ROWS
        ]
        assert rows[0]['content'] == 'the body of ep-a'
        replayed = {(r['uuid'], m['edge_uuid']) for r in _records(paths['replay']) for m in r['minted']}
        assert {(r['uuid'], m['edge_uuid']) for r in rows for m in r['minted']} == replayed
        assert all(len(r['minted']) == 4 for r in rows)

    def test_edges_are_interleaved_by_hash_and_carry_no_arm(self, tmp_path: Path) -> None:
        paths = self._replayed(tmp_path)
        assert mod.run(_worksheet_argv(paths)) == 0
        for row in _records(paths['blind']):
            hashes = [hashlib.sha256(m['edge_uuid'].encode()).hexdigest() for m in row['minted']]
            assert hashes == sorted(hashes)
            assert 'arm' not in row and all('arm' not in m for m in row['minted'])
        assert 'control' not in paths['blind'].read_text()

    def test_the_worksheet_is_byte_identical_across_runs(self, tmp_path: Path) -> None:
        paths = self._replayed(tmp_path)
        assert mod.run(_worksheet_argv(paths)) == 0
        first = paths['blind'].read_bytes()
        assert mod.run(_worksheet_argv(paths)) == 0
        assert paths['blind'].read_bytes() == first

    def test_an_incomplete_replay_exits_2_and_writes_nothing(self, tmp_path: Path) -> None:
        paths = _cli_inputs(tmp_path)
        assert _run_replay(paths, _CountingLLM(failing=frozenset({'ep-b'})), 'control') == 1
        assert mod.run(_worksheet_argv(paths)) == 2
        assert not paths['blind'].exists()


# --------------------------------------------------------------------------- #
# The score: each arm at the 4716 cell weights, the O1 decision, fidelity
# --------------------------------------------------------------------------- #

audit = load_script_module(SCRIPTS / 'audit_ruling_overreach.py', mod_name='audit_ruling_overreach')

DF, REIFY = 'dark_factory', 'reify'
RULING, OTHER = 'ruling_lexeme', 'other_decisions'
SCORE_EPISODES = ((DF, 'A1', RULING), (DF, 'A2', RULING), (REIFY, 'B1', OTHER), (REIFY, 'B2', OTHER))
"""Window populations 20 and 4 over two sampled episodes each: cell weights 10 and 2, over 2 days."""

BASELINE_OVERREACH = {'rate': 0.1238, 'ci': [0.0876, 0.1721], 'effective_n': 234.8}

PASSING = {
    'control': {DF: ('overreach', 'overreach', 'holding', 'holding', 'context'),
                REIFY: ('overreach', 'holding', 'holding', 'holding', 'context')},
    'o1': {DF: ('holding', 'holding', 'context', 'context', 'bookkeeping'),
           REIFY: ('holding', 'holding', 'holding', 'context', 'bookkeeping')},
}
DILUTING_O1 = {DF: ('overreach',) * 2 + ('holding',) * 4 + ('context',) * 6,
               REIFY: ('overreach',) + ('holding',) * 6 + ('context',) * 5}
"""Under half the control's overreach per minted edge, but as much per day: it only mints more."""
HOLDING_LOSING_O1 = {DF: ('holding',) + ('context',) * 3 + ('bookkeeping',),
                     REIFY: ('holding',) * 2 + ('context',) * 2 + ('bookkeeping',)}
HALVING_O1 = {DF: ('overreach', 'holding', 'holding', 'context', 'bookkeeping'), REIFY: PASSING['o1'][REIFY]}
"""20 weighted overreach edges to the control's 44, as many minted and as many holdings."""

ENTITIES = ('Alpha', 'Beta', 'Gamma', 'Delta')
O1_ENTITIES_OF_A1 = ('alpha', '  BETA ', 'Epsilon')
TOKENS = {'control': {'extract_nodes': (100, 10, 1), 'extract_edges': (300, 40, 1)},
          'o1': {'extract_nodes': (150, 10, 1), 'extract_edges': (350, 40, 2)}}


def _score_baseline(overreach: dict[str, Any] = BASELINE_OVERREACH, *, window_days: float = 2.0) -> dict[str, Any]:
    cell = {'minted': 0, 'corroborated': 0, 'episodes': 2}
    return {
        'strata': {DF: {RULING: {'population': 90, 'window_population': 20}},
                   REIFY: {OTHER: {'population': 90, 'window_population': 4}}},
        'sample': {'sizes': {DF: {RULING: cell}, REIFY: {OTHER: cell}}, 'window_days': window_days},
        'adjudicated': {'population_weighted': {'overreach_rate_minted': overreach}},
    }


def _score_records(
    labels_by_arm: dict[str, dict[str, tuple[str, ...]]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], str]]:
    records, labels = [], {}
    for arm, by_graph in labels_by_arm.items():
        for graph, uuid, stratum in SCORE_EPISODES:
            minted = []
            for index, label in enumerate(by_graph[graph]):
                edge_uuid = f'{arm}-{uuid}-{index}'
                minted.append({'edge_uuid': edge_uuid, 'fact': f'fact {edge_uuid}',
                               'source_name': 'S', 'target_name': 'T'})
                labels[(graph, edge_uuid)] = label
            entities = O1_ENTITIES_OF_A1 if (arm, uuid) == ('o1', 'A1') else ENTITIES
            records.append({
                'kind': 'replayed', 'arm': arm, 'model': 'gpt-4o-mini',
                'instructions_version': mod.ARMS[arm].instructions_version,
                'graph': graph, 'uuid': uuid, 'stratum': stratum, 'previous_count': 10,
                'entities': list(entities), 'minted': minted,
                'tokens': {stage: {'input_tokens': i, 'output_tokens': o, 'llm_calls': c}
                           for stage, (i, o, c) in TOKENS[arm].items()},
            })
    return records, labels


def _score_verdicts(records: list[dict[str, Any]], labels: dict[tuple[str, str], str]) -> dict[str, Any]:
    return {
        'sample': audit.SampleDefinition().to_dict(),
        'verdicts': [
            {'graph': r['graph'], 'episode_uuid': r['uuid'], 'edge_uuid': m['edge_uuid'],
             'fact': m['fact'], 'label': labels[(r['graph'], m['edge_uuid'])],
             'rationale': f"the span behind {m['edge_uuid']}"}
            for r in records for m in r['minted']
        ],
    }


def _run_score(
    tmp_path: Path, records: list[dict[str, Any]], verdicts: dict[str, Any],
    baseline: dict[str, Any] | None = None,
) -> tuple[int, Path]:
    replay, verdict_file = tmp_path / 'replay.jsonl', tmp_path / 'verdicts.json'
    baseline_file, out = tmp_path / 'report-4716.json', tmp_path / 'out' / 'report.json'
    replay.write_text(''.join(json.dumps(record) + '\n' for record in records))
    verdict_file.write_text(json.dumps(verdicts))
    baseline_file.write_text(json.dumps(baseline or _score_baseline()))
    code = mod.run(['score', '--replay', str(replay), '--verdicts', str(verdict_file),
                    '--baseline-report', str(baseline_file), '--out', str(out)])
    return code, out


def _scored(
    tmp_path: Path, labels_by_arm: dict[str, dict[str, tuple[str, ...]]] = PASSING,
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    records, labels = _score_records(labels_by_arm)
    code, out = _run_score(tmp_path, records, _score_verdicts(records, labels), baseline)
    assert code == 0
    return json.loads(out.read_text())


class TestOneCopyOfTheEstimators:
    def test_validation_and_weighting_are_the_audits_own(self) -> None:
        assert mod.load_verdicts is audit.load_verdicts
        assert mod.adjudicated_rates is audit.adjudicated_rates
        assert mod.population_weighted_rates is audit.population_weighted_rates
        assert mod.weighted_rate is audit.weighted_rate


class TestScorePerArm:
    def test_overreach_is_weighted_by_the_baseline_cells(self, tmp_path: Path) -> None:
        weighted = _scored(tmp_path)['arms']['control']['population_weighted']
        assert weighted['overreach_rate_minted'] == audit.weighted_rate([(10.0, 4, 10), (2.0, 2, 10)])
        assert weighted['overreach_rate_minted']['rate'] == 0.3667
        assert (weighted['overreach_per_day'], weighted['minted_per_day']) == (22.0, 60.0)
        assert weighted['weights'] == {DF: {RULING: 10.0}, REIFY: {OTHER: 2.0}}

    def test_an_arm_without_overreach_rates_zero(self, tmp_path: Path) -> None:
        weighted = _scored(tmp_path)['arms']['o1']['population_weighted']
        assert weighted['overreach_rate_minted']['rate'] == 0.0
        assert (weighted['overreach_per_day'], weighted['minted_per_day']) == (0.0, 60.0)

    def test_the_window_counts_are_unrounded_weighted_edge_counts(self, tmp_path: Path) -> None:
        weighted = _scored(tmp_path)['arms']['control']['population_weighted']
        assert weighted['window_counts'] == {'minted': 120.0, 'overreach': 44.0, 'holding': 52.0}

    def test_holding_share_and_holdings_per_day_are_weighted_too(self, tmp_path: Path) -> None:
        weighted = _scored(tmp_path)['arms']['control']['population_weighted']
        assert weighted['holding_share'] == audit.weighted_rate([(10.0, 4, 10), (2.0, 6, 10)])
        assert weighted['holding_share']['rate'] == 0.4333
        assert weighted['holdings_per_day'] == 26.0

    def test_unweighted_label_counts(self, tmp_path: Path) -> None:
        rates = _scored(tmp_path)['arms']['control']['rates']
        assert rates['all']['label_counts'] == {
            'holding': 10, 'bookkeeping': 0, 'context': 4, 'overreach': 6, 'misbound': 0, 'unjudgeable': 0,
        }
        assert rates[RULING]['label_counts']['overreach'] == 4

    def test_measures_with_no_meaning_for_a_replayed_edge_are_left_out(self, tmp_path: Path) -> None:
        rates = _scored(tmp_path)['arms']['control']['rates']
        assert set(rates['all']) == {
            'label_counts', 'overreach_rate_minted', 'overreach_rate_substantive',
            'episode_hit_rate', 'holding_share', 'misbound_count',
        }

    def test_entities_per_episode_and_overlap_with_the_control(self, tmp_path: Path) -> None:
        arms = _scored(tmp_path)['arms']
        assert arms['control']['entities'] == {'mean_per_episode': 4.0, 'mean_share_of_control': 1.0}
        assert arms['o1']['entities'] == {'mean_per_episode': 3.75, 'mean_share_of_control': 0.875}

    def test_tokens_are_totalled_per_prompt(self, tmp_path: Path) -> None:
        arms = _scored(tmp_path)['arms']
        assert arms['control']['tokens'] == {
            'extract_nodes': {'input_tokens': 400, 'output_tokens': 40, 'llm_calls': 4},
            'extract_edges': {'input_tokens': 1200, 'output_tokens': 160, 'llm_calls': 4},
        }
        assert arms['o1']['tokens']['extract_edges'] == {'input_tokens': 1400, 'output_tokens': 160, 'llm_calls': 8}

    def test_a_record_without_a_reasoning_effort_scores_as_effort_none(self, tmp_path: Path) -> None:
        records, _ = _score_records(PASSING)
        assert not any('reasoning_effort' in record for record in records)
        assert {arm: block['reasoning_effort'] for arm, block in _scored(tmp_path)['arms'].items()} == {
            'control': None, 'o1': None,
        }

    def test_each_arm_names_its_model_and_instructions(self, tmp_path: Path) -> None:
        arms = _scored(tmp_path)['arms']
        assert (arms['o1']['model'], arms['o1']['instructions_version']) == ('gpt-4o-mini', mod.O1_INSTRUCTIONS_VERSION)
        assert (arms['o1']['episodes'], arms['o1']['minted']) == (4, 20)


class TestTheO1Decision:
    def test_the_bars_are_pre_registered(self) -> None:
        assert (mod.OVERREACH_RATIO_MAX, mod.HOLDING_RETENTION_MIN) == (0.5, 0.8)

    def test_o1_passes_when_it_halves_overreach_and_keeps_its_holdings(self, tmp_path: Path) -> None:
        decision = _scored(tmp_path)['comparison']['o1']
        assert decision['checks'] == {
            'halves_overreach_per_minted': True, 'halves_overreach_per_day': True,
            'upper_bound_below_control_point': True, 'retains_holdings': True,
        }
        assert decision['passes'] is True
        assert (decision['overreach_ratio_minted'], decision['overreach_ratio_per_day'],
                decision['holding_retention']) == (0.0, 0.0, 1.0)

    def test_o1_fails_when_it_only_dilutes_overreach_by_minting_more(self, tmp_path: Path) -> None:
        decision = _scored(tmp_path, PASSING | {'o1': DILUTING_O1})['comparison']['o1']
        assert decision['checks'] == {
            'halves_overreach_per_minted': True, 'halves_overreach_per_day': False,
            'upper_bound_below_control_point': True, 'retains_holdings': True,
        }
        assert decision['passes'] is False

    def test_o1_fails_when_it_loses_holdings(self, tmp_path: Path) -> None:
        decision = _scored(tmp_path, PASSING | {'o1': HOLDING_LOSING_O1})['comparison']['o1']
        assert decision['checks'] == {
            'halves_overreach_per_minted': True, 'halves_overreach_per_day': True,
            'upper_bound_below_control_point': True, 'retains_holdings': False,
        }
        assert decision['passes'] is False
        assert decision['holding_retention'] == round(14.0 / 26.0, 4)

    def test_it_compares_unrounded_counts_where_rounded_per_day_rates_collide(self, tmp_path: Path) -> None:
        report = _scored(tmp_path, PASSING | {'o1': HALVING_O1}, baseline=_score_baseline(window_days=300.0))
        control, o1 = (report['arms'][arm]['population_weighted'] for arm in ('control', 'o1'))
        assert control['overreach_per_day'] == o1['overreach_per_day'] == 0.1
        decision = report['comparison']['o1']
        assert decision['overreach_ratio_per_day'] == decision['overreach_ratio_minted'] == round(20 / 44, 4)
        assert decision['checks']['halves_overreach_per_day'] is True
        assert decision['holding_retention'] == 1.0

    def test_the_comparison_states_its_bars(self, tmp_path: Path) -> None:
        decision = _scored(tmp_path)['comparison']['o1']
        assert decision['bars'] == {'overreach_ratio_max': 0.5, 'holding_retention_min': 0.8}


class TestFidelity:
    def test_the_baseline_interval_is_reported_beside_the_controls(self, tmp_path: Path) -> None:
        report = _scored(tmp_path)
        fidelity = report['fidelity']
        assert fidelity['baseline'] == {'rate': 0.1238, 'ci': [0.0876, 0.1721]}
        control = report['arms']['control']['population_weighted']['overreach_rate_minted']
        assert fidelity['control'] == {'rate': control['rate'], 'ci': control['ci']}

    @pytest.mark.parametrize(('baseline_ci', 'overlap'), [([0.30, 0.40], True), ([0.85, 0.95], False)])
    def test_whether_the_intervals_overlap(self, tmp_path: Path, baseline_ci: list[float], overlap: bool) -> None:
        baseline = _score_baseline({'rate': sum(baseline_ci) / 2, 'ci': baseline_ci, 'effective_n': 100.0})
        assert _scored(tmp_path, baseline=baseline)['fidelity']['intervals_overlap'] is overlap


class TestScoreFailsClosed:
    def _assert_rejected(self, tmp_path: Path, records: list[dict[str, Any]], verdicts: dict[str, Any]) -> None:
        code, out = _run_score(tmp_path, records, verdicts)
        assert code == 2
        assert not out.exists() and not out.parent.exists()

    def test_a_replayed_edge_without_a_verdict(self, tmp_path: Path) -> None:
        records, labels = _score_records(PASSING)
        verdicts = _score_verdicts(records, labels)
        verdicts['verdicts'] = verdicts['verdicts'][1:]
        self._assert_rejected(tmp_path, records, verdicts)

    def test_a_verdict_on_an_edge_no_arm_minted(self, tmp_path: Path) -> None:
        records, labels = _score_records(PASSING)
        verdicts = _score_verdicts(records, labels)
        verdicts['verdicts'].append({'graph': DF, 'episode_uuid': 'A1', 'edge_uuid': 'ghost', 'fact': 'f',
                                     'label': 'holding', 'rationale': 'a span'})
        self._assert_rejected(tmp_path, records, verdicts)

    def test_an_arm_missing_an_episode(self, tmp_path: Path) -> None:
        records, labels = _score_records(PASSING)
        short = [r for r in records if (r['arm'], r['uuid']) != ('o1', 'B2')]
        self._assert_rejected(tmp_path, short, _score_verdicts(short, labels))

    def test_a_replay_holding_a_failure_line(self, tmp_path: Path) -> None:
        records, labels = _score_records(PASSING)
        failed = {'kind': 'failed', 'arm': 'o1', 'model': 'gpt-4o-mini', 'graph': REIFY, 'uuid': 'B2',
                  'stratum': OTHER, 'error_type': 'RuntimeError', 'error': 'provider down'}
        kept = [r for r in records if (r['arm'], r['uuid']) != ('o1', 'B2')]
        self._assert_rejected(tmp_path, [*kept, failed], _score_verdicts(kept, labels))

    def test_a_verdict_file_adjudicated_under_another_sample(self, tmp_path: Path) -> None:
        records, labels = _score_records(PASSING)
        verdicts = _score_verdicts(records, labels)
        verdicts['sample'] = audit.SampleDefinition(cap=20).to_dict()
        self._assert_rejected(tmp_path, records, verdicts)

    def test_a_baseline_report_without_its_sample_sizes(self, tmp_path: Path) -> None:
        records, labels = _score_records(PASSING)
        baseline = _score_baseline()
        del baseline['sample']['sizes']
        code, out = _run_score(tmp_path, records, _score_verdicts(records, labels), baseline)
        assert code == 2
        assert not out.exists() and not out.parent.exists()


# --------------------------------------------------------------------------- #
# One replay file, one blind worksheet and one score across arm models
# --------------------------------------------------------------------------- #

CLI_SCORE_BASELINE = {
    'strata': {DF: {RULING: {'population': 90, 'window_population': 20}},
               REIFY: {OTHER: {'population': 90, 'window_population': 4}}},
    'sample': CLI_BASELINE['sample'] | {'window_days': 2.0},
    'adjudicated': {'population_weighted': {'overreach_rate_minted': BASELINE_OVERREACH}},
}
OVERREACHING_FACT = 'Beta overrides Alpha'
CONFIGURED_EDGES = (('Alpha', 'Beta', 'Alpha adopts Beta'), ('Beta', 'Alpha', OVERREACHING_FACT))
O6_LOW_EDGES = tuple(('Alpha', 'Beta', f'Alpha adopts Beta for queue {index}') for index in range(8))
"""Eight holdings per episode and no overreach: enough edges that its Wilson upper bound clears
the control's point, so o6-low passes."""
O6_MEDIUM_EDGES = CONFIGURED_EDGES
"""The control's overreach rate exactly, so o6-medium fails."""
EDGES_PER_EPISODE = len(CONFIGURED_EDGES) + len(O6_LOW_EDGES) + len(O6_MEDIUM_EDGES)


class TestOneReplayAcrossArmClients:
    def _replayed(self, tmp_path: Path) -> dict[str, Path]:
        paths = _cli_inputs(tmp_path, baseline=CLI_SCORE_BASELINE)
        factory = _ClientsByLlm(edges={
            CONTROL_LLM: CONFIGURED_EDGES, O6_LOW_LLM: O6_LOW_EDGES, O6_MEDIUM_LLM: O6_MEDIUM_EDGES,
        })
        assert _run_replay_on(paths, factory, 'control', 'o6-low', 'o6-medium') == 0
        return paths

    def _report(self, tmp_path: Path) -> dict[str, Any]:
        paths = self._replayed(tmp_path)
        records = _records(paths['replay'])
        labels = {
            (r['graph'], m['edge_uuid']): 'overreach' if m['fact'] == OVERREACHING_FACT else 'holding'
            for r in records for m in r['minted']
        }
        verdict_file, out = tmp_path / 'verdicts.json', tmp_path / 'out' / 'report.json'
        verdict_file.write_text(json.dumps(_score_verdicts(records, labels)))
        assert mod.run(['score', '--replay', str(paths['replay']), '--verdicts', str(verdict_file),
                        '--baseline-report', str(paths['baseline']), '--out', str(out)]) == 0
        return json.loads(out.read_text())

    def test_the_worksheet_interleaves_every_clients_edges_naming_no_arm(self, tmp_path: Path) -> None:
        paths = self._replayed(tmp_path)
        assert mod.run(_worksheet_argv(paths)) == 0
        records = _records(paths['replay'])
        for row in _records(paths['blind']):
            replayed = {m['edge_uuid'] for r in records if r['uuid'] == row['uuid'] for m in r['minted']}
            assert {m['edge_uuid'] for m in row['minted']} == replayed and len(replayed) == EDGES_PER_EPISODE
            hashes = [hashlib.sha256(m['edge_uuid'].encode()).hexdigest() for m in row['minted']]
            assert hashes == sorted(hashes)
            assert row.keys() == {'graph', 'uuid', 'stratum', 'content', 'minted'}
            assert all(m.keys() == {'edge_uuid', 'fact', 'source_name', 'target_name'} for m in row['minted'])
        text = paths['blind'].read_text()
        assert mod.O6_MODEL not in text and CONFIGURED not in text
        assert not any(arm in text for arm in ('o6-low', 'o6-medium', 'control'))

    def test_each_arm_block_names_its_client(self, tmp_path: Path) -> None:
        arms = self._report(tmp_path)['arms']
        assert {arm: (block['model'], block['reasoning_effort']) for arm, block in arms.items()} == {
            'control': (CONFIGURED, None), 'o6-low': (mod.O6_MODEL, 'low'), 'o6-medium': (mod.O6_MODEL, 'medium'),
        }
        assert {arms[arm]['instructions_version'] for arm in ('o6-low', 'o6-medium')} == {mod.NO_INSTRUCTIONS}

    def test_each_o6_arm_is_judged_against_the_same_files_control(self, tmp_path: Path) -> None:
        report = self._report(tmp_path)
        control = report['arms']['control']['population_weighted']
        assert control['overreach_rate_minted']['rate'] == 0.5
        assert set(report['comparison']) == {'o6-low', 'o6-medium'}
        for arm in ('o6-low', 'o6-medium'):
            assert report['comparison'][arm] == mod.lever_decision(control, report['arms'][arm]['population_weighted'])

    def test_the_two_efforts_are_decided_independently(self, tmp_path: Path) -> None:
        comparison = self._report(tmp_path)['comparison']
        assert (comparison['o6-low']['passes'], comparison['o6-medium']['passes']) == (True, False)


# --------------------------------------------------------------------------- #
# 9040's committed evidence still scores to its committed report
# --------------------------------------------------------------------------- #

REPO = Path(__file__).resolve().parents[2]
EVIDENCE_9040 = REPO / 'docs' / 'ruling-overreach-lever-reopen-2026-11-10'
BASELINE_4716 = REPO / 'docs' / 'ruling-overreach-guard-design-2026-10-01' / 'report.json'


class TestThe9040EvidenceRescoresUnchanged:
    """The 2026-11-10 design.md documents re-scoring its committed files, so a 9040-era replay
    (no reasoning_effort key) must still score, and its decision-bearing output must not move."""

    @pytest.fixture(scope='class')
    def rescored(self, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
        out = tmp_path_factory.mktemp('rescore-9040') / 'report.json'
        assert mod.run([
            'score', '--replay', str(EVIDENCE_9040 / 'replay.jsonl'),
            '--verdicts', str(EVIDENCE_9040 / 'verdicts.json'),
            '--baseline-report', str(BASELINE_4716), '--out', str(out),
        ]) == 0
        return json.loads(out.read_text())

    @pytest.fixture(scope='class')
    def committed(self) -> dict[str, Any]:
        return json.loads((EVIDENCE_9040 / 'report.json').read_text())

    def test_the_comparison_and_fidelity_are_unchanged(
        self, rescored: dict[str, Any], committed: dict[str, Any],
    ) -> None:
        assert rescored['comparison'] == committed['comparison']
        assert rescored['fidelity'] == committed['fidelity']

    def test_each_arm_block_gains_only_reasoning_effort_none(
        self, rescored: dict[str, Any], committed: dict[str, Any],
    ) -> None:
        assert set(rescored['arms']) == set(committed['arms']) == {'control', 'o1'}
        for arm, block in committed['arms'].items():
            assert rescored['arms'][arm] == block | {'reasoning_effort': None}
