"""Tests for scripts/luna_migration_eval.py (task 9945): the CLI over the luna-eval frame, sites and score.

Every graph read goes through a ReadOnlyGraphReader over one in-memory graph
that answers by Cypher, the journal is a temp sqlite file with the production
schema, the graphiti sites run on fake LLM clients built per arm, and the
routing sites send their chat completions to an httpx MockTransport. Nothing
here reaches a FalkorDB, a journal in data/ or a network.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest
from _fm_helpers import load_script_module
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig as GraphitiLLMConfig
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.prompts.models import Message
from pydantic import BaseModel

from fused_memory.backends.falkor_read_only import ReadOnlyGraphReader
from fused_memory.backends.llm_token_usage import AttributingTokenUsageTracker
from fused_memory.models.enums import MemoryCategory, QueryType
from fused_memory.services.write_journal import SCHEMA_SQL

SCRIPTS = Path(__file__).parent.parent / 'scripts'
cli = load_script_module(SCRIPTS / 'luna_migration_eval.py', mod_name='luna_migration_eval')
replay = load_script_module(SCRIPTS / 'replay_overreach_extraction.py', mod_name='replay_overreach_extraction')
score_mod = load_script_module(SCRIPTS / 'luna_eval_score.py', mod_name='luna_eval_score')

DF = 'dark_factory'
LUNA, MINI = replay.LUNA_MODEL, 'gpt-4o-mini'
PROXY_URL, PROXY_KEY = 'http://127.0.0.1:8317/v1', 'proxy-test-key'
PROXY_ENV = {'OPENAI_PLAN_PROXY_URL': PROXY_URL, 'OPENAI_PLAN_PROXY_KEY': PROXY_KEY}
ARMS = ('control', 'luna-low')
SITES = ('resolve_edge', 'summaries', 'classifier', 'router')

RESOLVE_PROMPT, SUMMARY_PROMPT = 'dedupe_edges.resolve_edge', 'extract_nodes.extract_summaries_batch'

# --------------------------------------------------------------------------- #
# One graph, answered by Cypher
# --------------------------------------------------------------------------- #

_SKIP_LIMIT_RE = re.compile(r'SKIP\s+(\d+)\s+LIMIT\s+(\d+)', re.IGNORECASE)
_CENSUS_RE = re.compile(r'RETURN\s+count\((\w+)\)\s*$', re.IGNORECASE)

EPISODES = {
    'ep-0': ('2026-09-01T00:00:00+00:00', 'add_memory:temporal_facts'),
    'ep-1': ('2026-10-05T00:00:00+00:00', 'add_memory:temporal_facts'),
    'ep-2': ('2026-10-06T00:00:00+00:00', 'add_memory:procedural_knowledge'),
}
"""uuid -> (created_at = valid_at, source_description); each body is 'the body of <uuid>'."""

LONG_SUMMARY = 'Alpha runs the queue. ' * 100
"""Past 4 * MAX_SUMMARY_CHARS, so graphiti summarises Alpha by LLM."""

ENTITIES = {'A': ('Alpha', LONG_SUMMARY), 'B': ('Beta', 'Beta is short.')}

EDGES = {
    'sib': ('A', 'B', 'Alpha uses Beta for the queue', '2026-09-01T00:00:00+00:00', 'ep-0'),
    'new': ('A', 'B', 'Alpha adopts Beta for the queue', '2026-10-05T00:00:00+00:00', 'ep-1'),
}
"""uuid -> (source, target, fact, created_at, minting episode)."""

MENTIONS = [('m-1', 'ep-1', 'A'), ('m-2', 'ep-1', 'B')]


def _body(uuid: str) -> str:
    return f'the body of {uuid}'


class _Result:
    def __init__(self, result_set: list[list[Any]]) -> None:
        self.result_set = result_set


class _Graph:
    """Episodic, RELATES_TO, MENTIONS and Entity pages with their census; text reads by uuid; the
    replay's own-episode and previous-episodes reads. ``summaries`` overrides an entity's summary."""

    def __init__(self, *, summaries: dict[str, str] | None = None) -> None:
        self.summaries = {uuid: summary for uuid, (_, summary) in ENTITIES.items()} | (summaries or {})
        self.populations = {
            ':MENTIONS': sorted([list(row) for row in MENTIONS]),
            ':RELATES_TO': sorted(
                [uuid, src, tgt, fact, at, at, None, None, [episode]]
                for uuid, (src, tgt, fact, at, episode) in EDGES.items()
            ),
            ':Episodic': sorted([uuid, at, desc, at] for uuid, (at, desc) in EPISODES.items()),
            ':Entity': sorted([uuid, name, self.summaries[uuid]] for uuid, (name, _) in ENTITIES.items()),
        }

    def _by_uuid(self, cypher: str, wanted: Sequence[str]) -> list[list[Any]]:
        if ':RELATES_TO' in cypher:
            return [[uuid, EDGES[uuid][0], EDGES[uuid][1], EDGES[uuid][2]] for uuid in wanted if uuid in EDGES]
        if ':Entity' in cypher:
            return [[uuid, self.summaries[uuid]] for uuid in wanted if uuid in self.summaries]
        return [[uuid, _body(uuid)] for uuid in wanted if uuid in EPISODES]

    def _previous(self, params: dict[str, Any]) -> list[list[Any]]:
        older = sorted(
            ((at, uuid) for uuid, (at, _) in EPISODES.items()
             if at <= params['reference_time'] and uuid != params['uuid']),
            reverse=True,
        )
        return [[_body(uuid), at] for at, uuid in older[: params['num_episodes']]]

    async def ro_query(self, cypher: str, params: dict[str, Any] | None = None) -> _Result:
        params = params or {}
        if 'IN $uuids' in cypher:
            return _Result(self._by_uuid(cypher, list(params['uuids'])))
        if '{uuid: $uuid}' in cypher:
            at, desc = EPISODES[params['uuid']]
            return _Result([[at, 'text', desc, DF]])
        if '$reference_time' in cypher:
            return _Result(self._previous(params))
        rows = next(rows for marker, rows in self.populations.items() if marker in cypher.split('RETURN')[0])
        if _CENSUS_RE.search(cypher.strip()):
            return _Result([[len(rows)]])
        match = _SKIP_LIMIT_RE.search(cypher)
        assert match, cypher
        skip, limit = int(match.group(1)), int(match.group(2))
        return _Result(rows[skip: skip + limit])

    async def query(self, cypher: str, params: dict[str, Any] | None = None) -> Any:
        raise AssertionError('the eval is read-only: it may never issue query()')


class _Unreachable:
    async def ro_query(self, cypher: str, params: dict[str, Any] | None = None) -> Any:
        raise ConnectionError('FalkorDB is down')


def _readers(graph: Any) -> Any:
    return lambda name: ReadOnlyGraphReader(graph_name=name, graph=graph, page_size=2, resultset_size=50)


# --------------------------------------------------------------------------- #
# The journal
# --------------------------------------------------------------------------- #

J_START, J_END, J_IN = '2026-11-04T00:00:00Z', '2026-11-18T00:00:00Z', '2026-11-10T12:00:00+00:00'
JOURNAL_ROWS = [
    ('w-1', 'add_memory', 'write', {'content': 'Hello world', 'category': 'observations_and_summaries'}),
    ('w-2', 'add_memory', 'write',
     {'content': 'We decided to migrate before the deadline changed', 'category': 'temporal_facts'}),
    ('q-1', 'search', 'read', {'query': 'foo bar baz', 'query_truncated': False, 'limit': 10}),
]
"""Inputs the production classifier and router send to their LLM."""


def _journal(tmp_path: Path) -> Path:
    path = tmp_path / 'write_journal.db'
    with sqlite3.connect(path) as db:
        db.executescript(SCHEMA_SQL)
        db.executemany(
            'INSERT INTO write_ops (id, operation, kind, params, created_at) VALUES (?, ?, ?, ?, ?)',
            [(row_id, op, kind, json.dumps(params), J_IN) for row_id, op, kind, params in JOURNAL_ROWS],
        )
    return path


# --------------------------------------------------------------------------- #
# The arms' clients
# --------------------------------------------------------------------------- #


class _SiteLLM(LLMClient):
    """Answers each prompt by name, records each call and its tokens, and declares its client;
    raises for every call once *failing* is set."""

    def __init__(self, answers: dict[str, dict[str, Any]], *, model: str, reasoning_effort: str | None,
                 endpoint: str, failing: bool = False) -> None:
        super().__init__(GraphitiLLMConfig(model=model))
        self.token_tracker = AttributingTokenUsageTracker()
        self.reasoning_effort = reasoning_effort
        self.endpoint = endpoint
        self.answers = answers
        self.failing = failing
        self.calls: list[tuple[str | None, list[Message]]] = []

    async def _generate_response(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError('the runners call generate_response, never the private hook')

    async def generate_response(
        self, messages: list[Message], response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None, model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None, prompt_name: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append((prompt_name, messages))
        if self.failing:
            raise RuntimeError('proxy down')
        self.token_tracker.record(prompt_name or '', 120, 30)
        return self.answers[prompt_name or '']

    def prompts(self, prompt_name: str) -> list[str]:
        return ['\n'.join(m.content for m in messages) for name, messages in self.calls if name == prompt_name]


def _answers(model: str) -> dict[str, dict[str, Any]]:
    """The control merges the new edge into its sibling and luna does not; each summarises Alpha."""
    return {
        RESOLVE_PROMPT: {'duplicate_facts': [0] if model == MINI else [], 'contradicted_facts': []},
        SUMMARY_PROMPT: {'summaries': [{'name': 'Alpha', 'summary': f'Alpha summary {len(model)}'}]},
    }


class _Factory:
    """An llm_factory: one _SiteLLM per ExtractionLlm, on that model (None is gpt-4o-mini) and
    declaring its effort and endpoint; *failing* arms' models raise, *missing* ones are None."""

    def __init__(self, *, failing: frozenset[str] = frozenset(), missing: frozenset[str] = frozenset(),
                 model: str | None = None) -> None:
        self.failing, self.missing, self.model = failing, missing, model
        self.clients: list[_SiteLLM] = []

    def __call__(self, spec: Any) -> _SiteLLM | None:
        model = self.model or spec.model or MINI
        if model in self.missing:
            return None
        client = _SiteLLM(_answers(model), model=model, reasoning_effort=spec.reasoning_effort,
                          endpoint=spec.endpoint, failing=model in self.failing)
        self.clients.append(client)
        return client

    def calls(self) -> int:
        return sum(len(client.calls) for client in self.clients)


class _Chat:
    """Answers every chat completion with a label both routing parsers read, by the request's model,
    so the two arms disagree on every item; usage 11/7."""

    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        luna = body['model'] == LUNA
        answer = {
            'primary': 'temporal_facts' if luna else 'observations_and_summaries', 'confidence': 0.8,
            'query_type': 'temporal' if luna else 'broad', 'stores': ['graphiti'], 'primary_store': 'graphiti',
        }
        return httpx.Response(200, json={
            'id': 'c', 'object': 'chat.completion', 'created': 0, 'model': body['model'],
            'choices': [{'index': 0, 'finish_reason': 'stop',
                         'message': {'role': 'assistant', 'content': json.dumps(answer)}}],
            'usage': {'prompt_tokens': 11, 'completion_tokens': 7, 'total_tokens': 18},
        })


@pytest.fixture
def routing_cfg(mock_config: Any) -> Any:
    mock_config.routing.llm_fallback = True
    return mock_config


# --------------------------------------------------------------------------- #
# The argv of each phase
# --------------------------------------------------------------------------- #


class _Eval:
    """One eval's paths: the committed docs dir, and the private files that carry bodies."""

    def __init__(self, tmp_path: Path) -> None:
        self.dir = tmp_path / 'docs' / 'luna-migration-eval-2026-11'
        self.private = tmp_path / 'private'
        self.worksheet = self.private / 'worksheet.jsonl'
        self.summary_inputs = self.private / 'summary-inputs.jsonl'
        self.runs = self.dir / 'runs.jsonl'
        self.journal = _journal(tmp_path)

    def frame_argv(self, **overrides: str) -> list[str]:
        paths = {'worksheet': str(self.worksheet), 'summary_inputs': str(self.summary_inputs)} | overrides
        return [
            'frame', '--graph', DF, '--dir', str(self.dir),
            '--worksheet', paths['worksheet'], '--summary-inputs', paths['summary_inputs'],
            '--extraction-window', '2026-08-01T00:00:00Z', '2026-11-18T00:00:00Z', '--cap', '3',
            '--fixture-window', '2026-10-01T00:00:00Z', '2026-11-18T00:00:00Z',
            '--resolve-n', '200', '--summaries-n', '80', '--nodes-per-episode', '2',
            '--journal', str(self.journal), '--journal-window', J_START, J_END,
            '--classifier-per-category', '50', '--router-n', '300',
        ]

    def run_argv(self, site: str, *arms: str) -> list[str]:
        argv = ['run', '--site', site, '--dir', str(self.dir), '--out', str(self.runs)]
        if site == 'summaries':
            argv += ['--summary-inputs', str(self.summary_inputs)]
        for arm in arms or ARMS:
            argv += ['--arm', arm]
        return argv

    def file(self, name: str) -> Path:
        return self.dir / name

    def lines(self, path: Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def committed_text(self) -> str:
        return ''.join(path.read_text() for path in sorted(self.dir.iterdir()))


def _frame(ev: _Eval, graph: Any = None) -> int:
    return cli.run(ev.frame_argv(), reader_factory=_readers(graph or _Graph()), environ=PROXY_ENV)


def _run(ev: _Eval, site: str, factory: Any, chat: _Chat | None, cfg: Any, *arms: str,
         environ: dict[str, str] = PROXY_ENV) -> int:
    return cli.run(
        ev.run_argv(site, *arms), llm_factory=factory, environ=environ, config=cfg,
        routing_transport=httpx.MockTransport(chat or _Chat()),
    )


@pytest.fixture
def drawn(tmp_path: Path) -> _Eval:
    ev = _Eval(tmp_path)
    assert _frame(ev) == 0
    return ev


@pytest.fixture
def ran(drawn: _Eval, routing_cfg: Any) -> _Eval:
    for site in SITES:
        assert _run(drawn, site, _Factory(), _Chat(), routing_cfg) == 0
    return drawn


# --------------------------------------------------------------------------- #
# frame
# --------------------------------------------------------------------------- #

FIXTURE_FILES = (
    'fixtures-resolve-edge.jsonl', 'fixtures-summaries.json', 'fixtures-classifier.jsonl', 'fixtures-router.jsonl',
)


class TestFrame:
    def test_it_writes_the_frame_and_the_four_fixture_files_into_the_dir(self, drawn: _Eval) -> None:
        assert sorted(path.name for path in drawn.dir.iterdir()) == sorted(('frame.json', *FIXTURE_FILES))
        frame = json.loads(drawn.file('frame.json').read_text())
        assert frame['graphs'] == [DF]
        assert frame['sample']['definition']['cap'] == 3
        assert frame['sample']['definition']['strata'] == ['temporal_facts', 'entities_and_relations',
                                                           'other_categories']

    def test_each_fixture_file_holds_its_sites_fixtures(self, drawn: _Eval) -> None:
        assert [f['edge_uuid'] for f in drawn.lines(drawn.file('fixtures-resolve-edge.jsonl'))] == ['new']
        (item,) = json.loads(drawn.file('fixtures-summaries.json').read_text())
        assert (item['episode_uuid'], item['node_uuid']) == ('ep-1', 'A')
        assert sorted(f['id'] for f in drawn.lines(drawn.file('fixtures-classifier.jsonl'))) == ['w-1', 'w-2']
        assert [f['id'] for f in drawn.lines(drawn.file('fixtures-router.jsonl'))] == ['q-1']

    def test_the_worksheet_is_the_one_the_replay_scores_against_the_frame(self, drawn: _Eval) -> None:
        rows = drawn.lines(drawn.worksheet)
        frame = json.loads(drawn.file('frame.json').read_text())
        replay.check_worksheet(rows, replay.baseline_cells(frame))
        assert replay.Baseline.from_report(frame).window_population[DF]['temporal_facts'] == 2
        assert {row['uuid']: row['content'] for row in rows} == {uuid: _body(uuid) for uuid in EPISODES}

    def test_the_summary_inputs_restore_the_items_text_and_previous_episodes(self, drawn: _Eval) -> None:
        (inputs,) = drawn.lines(drawn.summary_inputs)
        assert inputs['episode_content'] == _body('ep-1')
        assert inputs['nodes'] == [{'node_uuid': 'A', 'node_name': 'Alpha', 'summary': LONG_SUMMARY}]
        assert [(e['uuid'], e['fact']) for e in inputs['edges']] == [('new', 'Alpha adopts Beta for the queue')]
        assert inputs['previous'] == [_body('ep-0')]

    def test_no_committed_file_carries_an_episode_body_or_a_summary(self, drawn: _Eval) -> None:
        committed = drawn.committed_text()
        assert 'the body of' not in committed and 'Alpha runs the queue.' not in committed

    @pytest.mark.parametrize('private', ['worksheet', 'summary_inputs'])
    def test_a_private_file_inside_the_dir_exits_2_and_writes_nothing(self, tmp_path: Path, private: str) -> None:
        ev = _Eval(tmp_path)
        argv = ev.frame_argv(**{private: str(ev.dir / 'bodies.jsonl')})
        assert cli.run(argv, reader_factory=_readers(_Graph()), environ=PROXY_ENV) == 2
        assert not ev.dir.exists() and not ev.private.exists()

    def test_an_unreachable_graph_exits_1_and_writes_nothing(self, tmp_path: Path) -> None:
        ev = _Eval(tmp_path)
        assert _frame(ev, _Unreachable()) == 1
        assert not ev.dir.exists() and not ev.private.exists()

    def test_a_missing_journal_exits_2_and_writes_nothing(self, tmp_path: Path) -> None:
        ev = _Eval(tmp_path)
        ev.journal.unlink()
        assert _frame(ev) == 2
        assert not ev.dir.exists() and not ev.journal.exists()

    @pytest.mark.parametrize('journal', [
        pytest.param(lambda path: sqlite3.connect(path).executescript('CREATE TABLE other (id TEXT);').close(),
                     id='another-schema'),
        pytest.param(lambda path: path.write_bytes(b'not a sqlite database, only bytes ' * 64), id='corrupt'),
    ])
    def test_an_unreadable_journal_exits_2_and_writes_nothing(self, tmp_path: Path, journal: Any) -> None:
        ev = _Eval(tmp_path)
        ev.journal.unlink()
        journal(ev.journal)
        assert _frame(ev) == 2
        assert not ev.dir.exists() and not ev.private.exists()

    def test_summaries_the_live_graph_rewrites_during_the_draw_still_restore_the_manifest(
        self, tmp_path: Path, routing_cfg: Any,
    ) -> None:
        class _Rewriting(_Graph):
            """Every Entity read answers a newer summary, as a graph still being written does."""

            def __init__(self) -> None:
                super().__init__()
                self.edits = 0

            async def ro_query(self, cypher: str, params: dict[str, Any] | None = None) -> _Result:
                result = await super().ro_query(cypher, params)
                pattern = cypher.split('RETURN')[0]
                if ':Entity' not in pattern or '-[' in pattern:
                    return result
                self.edits += 1
                return _Result([[*row[:-1], f'{row[-1]} edit {self.edits}'] if len(row) > 1 else row
                                for row in result.result_set])

        ev = _Eval(tmp_path)
        assert _frame(ev, _Rewriting()) == 0
        (inputs,) = ev.lines(ev.summary_inputs)
        assert inputs['nodes'][0]['summary'].startswith(LONG_SUMMARY)
        assert _run(ev, 'summaries', _Factory(), None, routing_cfg) == 0


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #


def _keys(ev: _Eval, kind: str = 'replayed') -> set[tuple[str, str, str]]:
    return {(r['site'], r['arm'], r['fixture_id']) for r in ev.lines(ev.runs) if r['kind'] == kind}


EXPECTED_KEYS = {
    (site, arm, fid) for arm in ARMS for site, fid in (
        ('resolve_edge', f'{DF}/new'), ('summaries', f'{DF}/ep-1'),
        ('classifier', 'w-1'), ('classifier', 'w-2'), ('router', 'q-1'),
    )
}


class TestRun:
    def test_every_site_runs_both_arms_over_its_fixtures_into_one_file(self, ran: _Eval) -> None:
        assert _keys(ran) == EXPECTED_KEYS

    def test_each_record_names_the_client_its_arm_ran_on(self, ran: _Eval) -> None:
        clients = {(r['site'], r['arm'], r['model'], r['reasoning_effort'], r['endpoint']) for r in ran.lines(ran.runs)}
        assert clients == {
            (site, arm, model, effort, endpoint) for site in SITES for arm, model, effort, endpoint in (
                ('control', MINI, None, 'configured'), ('luna-low', LUNA, 'low', 'plan_proxy'),
            )
        }

    def test_the_arms_answers_are_recorded(self, ran: _Eval) -> None:
        by_key = {(r['site'], r['arm'], r['fixture_id']): r for r in ran.lines(ran.runs)}
        assert by_key[('resolve_edge', 'control', f'{DF}/new')]['duplicate_uuids'] == ['sib']
        assert by_key[('resolve_edge', 'luna-low', f'{DF}/new')]['duplicate_uuids'] == []
        assert by_key[('classifier', 'luna-low', 'w-1')]['label'] == 'temporal_facts'
        assert by_key[('router', 'control', 'q-1')]['label'] == 'broad'

    def test_summaries_run_over_the_restored_episode_and_its_previous_episodes(
        self, drawn: _Eval, routing_cfg: Any,
    ) -> None:
        factory = _Factory()
        assert _run(drawn, 'summaries', factory, None, routing_cfg) == 0
        prompts = [prompt for client in factory.clients for prompt in client.prompts(SUMMARY_PROMPT)]
        assert len(prompts) == 2
        assert all(_body('ep-1') in p and _body('ep-0') in p and 'Alpha runs the queue.' in p for p in prompts)

    def test_the_routing_sites_send_each_arms_request(self, drawn: _Eval, routing_cfg: Any) -> None:
        chat = _Chat()
        assert _run(drawn, 'router', _Factory(), chat, routing_cfg) == 0
        assert sorted((b['model'], b.get('reasoning_effort'), b.get('temperature')) for b in chat.bodies) == [
            (MINI, None, 0.0), (LUNA, 'low', None),
        ]

    def test_a_rerun_calls_nothing_already_recorded(self, ran: _Eval, routing_cfg: Any) -> None:
        factory, chat = _Factory(), _Chat()
        for site in SITES:
            assert _run(ran, site, factory, chat, routing_cfg) == 0
        assert factory.calls() == 0 and chat.bodies == []
        assert len(ran.lines(ran.runs)) == len(EXPECTED_KEYS)

    def test_a_failure_is_a_line_that_exits_1_and_a_rerun_retries_only_it(
        self, drawn: _Eval, routing_cfg: Any,
    ) -> None:
        assert _run(drawn, 'resolve_edge', _Factory(failing=frozenset({LUNA})), None, routing_cfg) == 1
        assert _keys(drawn, 'failed') == {('resolve_edge', 'luna-low', f'{DF}/new')}
        retry = _Factory()
        assert _run(drawn, 'resolve_edge', retry, None, routing_cfg) == 0
        assert [client.model for client in retry.clients if client.calls] == [LUNA]
        assert _keys(drawn, 'failed') == set()

    def test_running_another_site_keeps_this_sites_failure_lines(self, drawn: _Eval, routing_cfg: Any) -> None:
        assert _run(drawn, 'resolve_edge', _Factory(failing=frozenset({LUNA})), None, routing_cfg) == 1
        assert _run(drawn, 'router', _Factory(), None, routing_cfg) == 0
        assert _keys(drawn, 'failed') == {('resolve_edge', 'luna-low', f'{DF}/new')}

    @pytest.mark.parametrize('site', SITES)
    @pytest.mark.parametrize('missing', sorted(PROXY_ENV))
    def test_a_missing_plan_proxy_variable_exits_2_and_writes_nothing(
        self, drawn: _Eval, routing_cfg: Any, site: str, missing: str,
    ) -> None:
        environ = {name: value for name, value in PROXY_ENV.items() if name != missing}
        assert _run(drawn, site, None, None, routing_cfg, environ=environ) == 2
        assert not drawn.runs.exists()

    def test_a_client_on_another_model_than_its_arm_exits_2_before_any_call(
        self, drawn: _Eval, routing_cfg: Any,
    ) -> None:
        factory = _Factory(model='gpt-4.1-nano')
        assert _run(drawn, 'resolve_edge', factory, None, routing_cfg) == 2
        assert factory.calls() == 0 and not drawn.runs.exists()

    def test_a_missing_client_exits_1_and_writes_nothing(self, drawn: _Eval, routing_cfg: Any) -> None:
        assert _run(drawn, 'summaries', _Factory(missing=frozenset({LUNA})), None, routing_cfg) == 1
        assert not drawn.runs.exists()

    def test_summary_inputs_that_differ_from_the_manifest_exit_2_and_write_nothing(
        self, drawn: _Eval, routing_cfg: Any,
    ) -> None:
        (inputs,) = drawn.lines(drawn.summary_inputs)
        inputs['episode_content'] = 'another body'
        drawn.summary_inputs.write_text(json.dumps(inputs) + '\n')
        factory = _Factory()
        assert _run(drawn, 'summaries', factory, None, routing_cfg) == 2
        assert factory.calls() == 0 and not drawn.runs.exists()

    def test_a_run_before_the_frame_exits_2_and_writes_nothing(self, tmp_path: Path, routing_cfg: Any) -> None:
        ev = _Eval(tmp_path)
        assert _run(ev, 'router', _Factory(), None, routing_cfg) == 2
        assert not ev.runs.exists()


# --------------------------------------------------------------------------- #
# worksheet
# --------------------------------------------------------------------------- #

ARM_NAMES = ('control', 'luna', MINI, LUNA, 'plan_proxy', 'configured')


def _worksheet(ev: _Eval, kind: str, out: Path, *extra: str) -> int:
    return cli.run(['worksheet', '--kind', kind, '--dir', str(ev.dir), '--out', str(out), *extra])


def _summary_worksheet(ev: _Eval, out: Path, key: Path) -> int:
    return _worksheet(ev, 'summaries', out, '--runs', str(ev.runs), '--summary-inputs', str(ev.summary_inputs),
                      '--key-out', str(key))


class TestWorksheet:
    def test_the_resolve_gold_worksheet_offers_each_fixture_and_its_candidates_only(self, ran: _Eval) -> None:
        out = ran.private / 'resolve-gold.jsonl'
        assert _worksheet(ran, 'resolve_gold', out) == 0
        (row,) = ran.lines(out)
        assert row['fixture_id'] == f'{DF}/new'
        assert [edge['uuid'] for edge in row['related']] == ['sib']
        assert not any(name in out.read_text() for name in ARM_NAMES)

    def test_the_summary_worksheet_shows_both_summaries_unattributed_and_keys_them_apart(self, ran: _Eval) -> None:
        out, key = ran.private / 'summaries.jsonl', ran.private / 'summaries-key.json'
        assert _summary_worksheet(ran, out, key) == 0
        (row,) = ran.lines(out)
        assert (row['fixture_id'], row['node_uuid']) == (f'{DF}/ep-1', 'A')
        assert sorted(row['summaries'].values()) == sorted(f'Alpha summary {len(m)}' for m in (MINI, LUNA))
        assert not any(name in out.read_text() for name in ARM_NAMES)
        (entry,) = json.loads(key.read_text())
        assert {entry['A'], entry['B']} == set(ARMS)

    def test_a_summary_worksheet_inside_the_dir_exits_2_and_writes_nothing(self, ran: _Eval) -> None:
        out, key = ran.dir / 'summaries.jsonl', ran.private / 'summaries-key.json'
        assert _summary_worksheet(ran, out, key) == 2
        assert not out.exists() and not key.exists()

    def test_the_routing_worksheet_offers_every_discordant_item_with_its_sites_vocabulary(self, ran: _Eval) -> None:
        out = ran.private / 'routing.jsonl'
        assert _worksheet(ran, 'routing', out, '--runs', str(ran.runs)) == 0
        rows = ran.lines(out)
        assert sorted((row['site'], row['fixture_id']) for row in rows) == [
            ('classifier', 'w-1'), ('classifier', 'w-2'), ('router', 'q-1'),
        ]
        vocabulary = {row['site']: row['vocabulary'] for row in rows}
        assert vocabulary == {'classifier': [c.value for c in MemoryCategory], 'router': [q.value for q in QueryType]}
        assert {row['text'] for row in rows if row['site'] == 'router'} == {'foo bar baz'}

    def test_a_worksheet_over_runs_missing_a_site_exits_2(self, drawn: _Eval, routing_cfg: Any) -> None:
        assert _run(drawn, 'router', _Factory(), None, routing_cfg) == 0
        out = drawn.private / 'routing.jsonl'
        assert _worksheet(drawn, 'routing', out, '--runs', str(drawn.runs)) == 2
        assert not out.exists()


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #

CALLS_PER_DAY = {'extraction': 400.0, 'resolve_edge': 300.0, 'summaries': 200.0, 'classifier': 500.0, 'router': 900.0}


def _replay_records(ev: _Eval) -> list[dict[str, Any]]:
    """What replay_overreach_extraction.py replay writes over the frame's worksheet: one holding edge
    per episode under each arm."""
    clients = {'control': (MINI, None, 'configured'), 'luna-low': (LUNA, 'low', 'plan_proxy')}
    return [
        {
            'kind': 'replayed', 'arm': arm, 'model': model, 'reasoning_effort': effort, 'endpoint': endpoint,
            'instructions_version': replay.NO_INSTRUCTIONS, 'graph': row['graph'], 'uuid': row['uuid'],
            'stratum': row['stratum'], 'previous_count': 0, 'entities': ['Alpha', 'Beta'],
            'minted': [{'edge_uuid': f"{arm}-{row['uuid']}", 'fact': f"fact of {row['uuid']}",
                        'source_name': 'Alpha', 'target_name': 'Beta'}],
            'tokens': {'extract_nodes': {'input_tokens': 100, 'output_tokens': 10, 'llm_calls': 1},
                       'extract_edges': {'input_tokens': 300, 'output_tokens': 40, 'llm_calls': 1}},
        }
        for row in ev.lines(ev.worksheet) for arm, (model, effort, endpoint) in clients.items()
    ]


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj))


def _adjudicate(ev: _Eval) -> None:
    """Every verdict file, written from the CLI's own worksheets."""
    records = _replay_records(ev)
    ev.file('replay.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    frame = json.loads(ev.file('frame.json').read_text())
    _write_json(ev.file('verdicts-extraction.json'), {
        'sample': frame['sample']['definition'],
        'verdicts': [
            {'graph': r['graph'], 'episode_uuid': r['uuid'], 'edge_uuid': m['edge_uuid'], 'fact': m['fact'],
             'label': 'holding', 'rationale': 'states the record'}
            for r in records for m in r['minted']
        ],
    })
    gold, summaries = ev.private / 'resolve-gold.jsonl', ev.private / 'summaries.jsonl'
    routing, key = ev.private / 'routing.jsonl', ev.file('ratings-summaries-key.json')
    assert _worksheet(ev, 'resolve_gold', gold) == 0
    assert _summary_worksheet(ev, summaries, key) == 0
    assert _worksheet(ev, 'routing', routing, '--runs', str(ev.runs)) == 0
    _write_json(ev.file('gold-resolve-edge.json'), {'gold': [
        {'fixture_id': row['fixture_id'], 'duplicates': [], 'contradicted': [], 'judgeable': True,
         'rationale': 'adopts is not uses'} for row in ev.lines(gold)
    ]})
    _write_json(ev.file('ratings-summaries.json'), {'ratings': [
        {'fixture_id': row['fixture_id'], 'node_uuid': row['node_uuid'], 'summary': letter, 'rating': 'faithful',
         'rationale': 'it says what the inputs say'} for row in ev.lines(summaries) for letter in ('A', 'B')
    ]})
    _write_json(ev.file('verdicts-routing.json'), {'verdicts': [
        {'site': row['site'], 'fixture_id': row['fixture_id'], 'label': row['labels'][0], 'rationale': 'fits'}
        for row in ev.lines(routing)
    ]})


def _score(ev: _Eval, out: Path) -> int:
    calls = ev.private / 'calls-per-day.json'
    _write_json(calls, CALLS_PER_DAY)
    return cli.run([
        'score', '--dir', str(ev.dir), '--calls-per-day', str(calls),
        '--plan-window-usd-per-week', '280', '470', '--out', str(out),
    ])


@pytest.fixture
def adjudicated(ran: _Eval) -> _Eval:
    _adjudicate(ran)
    return ran


class TestScore:
    def test_it_writes_every_sites_decision_the_routing_layer_and_cost(self, adjudicated: _Eval) -> None:
        out = adjudicated.file('report.json')
        assert _score(adjudicated, out) == 0
        report = json.loads(out.read_text())
        assert set(report['sites']) == set(SITES)
        assert report['sites']['resolve_edge']['decision']['statistics']['merge_accuracy']['candidate_only'] == 1
        assert report['routing_layer']['checks'].keys() == {
            'C1_classifier_label_accuracy', 'Q1_router_label_accuracy',
        }
        assert set(report['cost']['sites']) == {'extraction', *SITES}
        assert report['cost']['plan_window']['window_usd_per_week'] == [280.0, 470.0]

    def test_the_extraction_leg_is_the_replays_own_score_over_the_frame(self, adjudicated: _Eval) -> None:
        out = adjudicated.file('report.json')
        assert _score(adjudicated, out) == 0
        frame_text = adjudicated.file('frame.json').read_text()
        expected = score_mod.score_extraction(
            _replay_records(adjudicated), json.loads(adjudicated.file('verdicts-extraction.json').read_text()),
            json.loads(frame_text),
        )
        report = json.loads(out.read_text())
        assert report['extraction'] == json.loads(json.dumps(expected))
        assert set(report['extraction']['comparison']['luna-low']['checks']) == {
            'E1_overreach_upper_within_margin', 'E2_retains_holdings',
        }
        assert report['sample']['frame_sha256'] == score_mod.build_report(
            frame_text=frame_text, extraction_replay=_replay_records(adjudicated),
            extraction_verdicts=json.loads(adjudicated.file('verdicts-extraction.json').read_text()),
            sites={}, routing_layer={}, cost={},
        )['sample']['frame_sha256']

    @pytest.mark.parametrize('missing', [
        'verdicts-extraction.json', 'gold-resolve-edge.json', 'ratings-summaries.json',
        'ratings-summaries-key.json', 'verdicts-routing.json', 'replay.jsonl', 'runs.jsonl',
    ])
    def test_a_missing_input_exits_2_and_writes_nothing(self, adjudicated: _Eval, missing: str) -> None:
        adjudicated.file(missing).unlink()
        out = adjudicated.private / 'report.json'
        assert _score(adjudicated, out) == 2
        assert not out.exists()

    def test_runs_missing_a_pre_registered_fixture_exit_2_naming_it(
        self, adjudicated: _Eval, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A run killed between fixtures leaves no failure line: the decision must not be taken
        over the fixtures that happened to run."""
        lines = [r for r in adjudicated.lines(adjudicated.runs) if r['fixture_id'] != 'w-2']
        adjudicated.runs.write_text(''.join(json.dumps(r) + '\n' for r in lines))
        out = adjudicated.private / 'report.json'
        with caplog.at_level(logging.ERROR):
            assert _score(adjudicated, out) == 2
        assert 'RunFileError' in caplog.text and "['w-2']" in caplog.text
        assert not out.exists()

    def test_runs_over_a_fixture_not_pre_registered_exit_2_naming_it(
        self, adjudicated: _Eval, caplog: pytest.LogCaptureFixture,
    ) -> None:
        fixtures = adjudicated.file('fixtures-classifier.jsonl')
        kept = [fixture for fixture in adjudicated.lines(fixtures) if fixture['id'] != 'w-2']
        fixtures.write_text(''.join(json.dumps(fixture) + '\n' for fixture in kept))
        out = adjudicated.private / 'report.json'
        with caplog.at_level(logging.ERROR):
            assert _score(adjudicated, out) == 2
        assert 'RunFileError' in caplog.text and "not pre-registered: ['w-2']" in caplog.text
        assert not out.exists()

    def test_a_summary_key_that_swaps_the_arms_exits_2(self, adjudicated: _Eval, caplog: pytest.LogCaptureFixture) -> None:
        key = json.loads(adjudicated.file('ratings-summaries-key.json').read_text())
        _write_json(adjudicated.file('ratings-summaries-key.json'), [entry | {'A': entry['B'], 'B': entry['A']}
                                                                     for entry in key])
        out = adjudicated.private / 'report.json'
        with caplog.at_level(logging.ERROR):
            assert _score(adjudicated, out) == 2
        assert 'MalformedVerdicts' in caplog.text and f'{DF}/ep-1/A' in caplog.text
        assert not out.exists()

    def test_a_verdict_file_the_loader_refuses_exits_2(self, adjudicated: _Eval) -> None:
        verdicts = json.loads(adjudicated.file('verdicts-routing.json').read_text())
        verdicts['verdicts'] = verdicts['verdicts'][1:]
        _write_json(adjudicated.file('verdicts-routing.json'), verdicts)
        out = adjudicated.private / 'report.json'
        assert _score(adjudicated, out) == 2
        assert not out.exists()
