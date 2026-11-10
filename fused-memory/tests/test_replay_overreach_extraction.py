"""Tests for scripts/replay_overreach_extraction.py (task 9040).

The replay calls graphiti_core's own extract_nodes and extract_edges, so these
tests drive the real extraction functions with a fake LLM that answers by
prompt name. No FalkorDB handle and no driver double exists anywhere here:
the replay never needs one.
"""
from __future__ import annotations

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
from graphiti_core.prompts.models import Message
from graphiti_core.search.search_utils import RELEVANT_SCHEMA_LIMIT
from pydantic import BaseModel

from fused_memory.backends.falkor_read_only import ReadOnlyGraphReader
from fused_memory.backends.falkor_string_bounds import exact_string_upper_bound
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

    def test_matching_cell_counts_pass(self) -> None:
        mod.check_worksheet(self.ROWS, _sizes(dark_factory__ruling_lexeme=2, reify__other_decisions=1))

    def test_a_short_cell_is_named(self) -> None:
        with pytest.raises(mod.WorksheetMismatch, match='dark_factory/ruling_lexeme'):
            mod.check_worksheet(self.ROWS, _sizes(dark_factory__ruling_lexeme=3, reify__other_decisions=1))

    def test_a_cell_the_baseline_lacks_is_named(self) -> None:
        with pytest.raises(mod.WorksheetMismatch, match='reify/other_decisions'):
            mod.check_worksheet(self.ROWS, _sizes(dark_factory__ruling_lexeme=2))

    def test_a_baseline_cell_the_worksheet_lacks_is_named(self) -> None:
        with pytest.raises(mod.WorksheetMismatch, match='reify/decision_anchor'):
            mod.check_worksheet(self.ROWS, _sizes(
                dark_factory__ruling_lexeme=2, reify__other_decisions=1, reify__decision_anchor=8,
            ))


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


def _run_replay(paths: dict[str, Path], llm: _FakeLLM, *arms: str) -> int:
    return mod.run(
        _replay_argv(paths, *arms), llm_factory=lambda model: llm,
        reader_factory=lambda graph: ReadOnlyGraphReader(graph_name=graph, graph=_episode_graph()),
    )


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

    def test_a_missing_client_exits_1(self, tmp_path: Path) -> None:
        paths = _cli_inputs(tmp_path)
        code = mod.run(_replay_argv(paths), llm_factory=lambda model: None,
                       reader_factory=lambda graph: ReadOnlyGraphReader(graph_name=graph, graph=_episode_graph()))
        assert code == 1 and not paths['replay'].exists()

    def test_the_model_axis_reaches_the_factory(self, tmp_path: Path) -> None:
        paths = _cli_inputs(tmp_path)
        asked: list[str | None] = []

        def factory(model: str | None) -> _FakeLLM:
            asked.append(model)
            return _CountingLLM(model=model or 'configured')

        code = mod.run([*_replay_argv(paths, 'control'), '--model', 'm-strong'], llm_factory=factory,
                       reader_factory=lambda graph: ReadOnlyGraphReader(graph_name=graph, graph=_episode_graph()))
        assert code == 0 and asked == ['m-strong']
        assert {r['model'] for r in _records(paths['replay'])} == {'m-strong'}

    def test_a_second_model_for_a_recorded_arm_exits_2(self, tmp_path: Path) -> None:
        paths = _cli_inputs(tmp_path)
        assert _run_replay(paths, _CountingLLM(model='m-small'), 'control') == 0
        before = paths['replay'].read_text()
        assert _run_replay(paths, _CountingLLM(model='m-other'), 'control') == 2
        assert paths['replay'].read_text() == before


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
