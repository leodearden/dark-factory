"""Tests for scripts/luna_eval_frame.py (task 9945): the frozen, read-only luna-eval sample frame.

Every read goes through a ReadOnlyGraphReader over a fake graph that answers by
Cypher, as test_replay_overreach_extraction.py and test_audit_ruling_overreach.py
do; nothing here can reach a FalkorDB.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from _fm_helpers import load_script_module

from fused_memory.backends.falkor_read_only import ReadOnlyGraphReader

SCRIPTS = Path(__file__).parent.parent / 'scripts'
frame_mod = load_script_module(SCRIPTS / 'luna_eval_frame.py', mod_name='luna_eval_frame')
audit = load_script_module(SCRIPTS / 'audit_ruling_overreach.py', mod_name='audit_ruling_overreach')
replay = load_script_module(SCRIPTS / 'replay_overreach_extraction.py', mod_name='replay_overreach_extraction')

DF, REIFY = 'dark_factory', 'reify'
TEMPORAL, RELATIONS, OTHER = 'temporal_facts', 'entities_and_relations', 'other_categories'
START, END = '2026-08-01T00:00:00+00:00', '2026-11-18T00:00:00+00:00'
INSIDE = '2026-09-10T12:00:00+00:00'

_SKIP_LIMIT_RE = re.compile(r'SKIP\s+(\d+)\s+LIMIT\s+(\d+)', re.IGNORECASE)
_CENSUS_RE = re.compile(r'RETURN\s+count\((\w+)\)\s*$', re.IGNORECASE)


class _Result:
    def __init__(self, result_set: list[list[Any]]) -> None:
        self.result_set = result_set


class _EpisodeGraph:
    """Episodic nodes: metadata pages with their census, and a content read by uuid.

    Records every content read so a test can prove only the sample's bodies are read.
    ``census_extra`` makes the census claim more rows than the pages serve: a short read.
    """

    def __init__(self, episodes: list[dict[str, Any]], *, census_extra: int = 0) -> None:
        self.episodes = sorted(episodes, key=lambda e: e['uuid'])
        self.census_extra = census_extra
        self.queries: list[str] = []
        self.content_reads: list[str] = []

    async def ro_query(self, cypher: str, params: dict[str, Any] | None = None) -> _Result:
        self.queries.append(cypher)
        if 'e.content' in cypher:
            wanted = set(params['uuids']) if params else set()
            self.content_reads.extend(sorted(wanted))
            return _Result([[e['uuid'], e['content']] for e in self.episodes if e['uuid'] in wanted])
        if _CENSUS_RE.search(cypher.strip()):
            return _Result([[len(self.episodes) + self.census_extra]])
        match = _SKIP_LIMIT_RE.search(cypher)
        assert match, cypher
        skip, limit = int(match.group(1)), int(match.group(2))
        return _Result([
            [e['uuid'], e['created_at'], e['source_description']]
            for e in self.episodes[skip: skip + limit]
        ])

    async def query(self, cypher: str, params: dict[str, Any] | None = None) -> Any:
        raise AssertionError('the frame is read-only: it may never issue query()')


def _episode(uuid: str, source_description: str | None, created_at: str = INSIDE) -> dict[str, Any]:
    return {'uuid': uuid, 'created_at': created_at, 'source_description': source_description,
            'content': f'the body of {uuid}'}


def _readers(graphs: dict[str, _EpisodeGraph]) -> dict[str, ReadOnlyGraphReader]:
    return {
        name: ReadOnlyGraphReader(graph_name=name, graph=graph, page_size=2, resultset_size=5)
        for name, graph in graphs.items()
    }


def _definition(cap: int = 3) -> Any:
    return audit.SampleDefinition(window_start=START, window_end=END, cap=cap, strata=frame_mod.EXTRACTION_STRATA)


def _key(graph: str, uuid: str) -> str:
    return hashlib.sha256(f'{graph}:{uuid}'.encode()).hexdigest()


async def _frame(graphs: dict[str, _EpisodeGraph], cap: int = 3) -> Any:
    return await frame_mod.extraction_frame(_readers(graphs), _definition(cap))


MIXED = [
    _episode('d-plain', 'add_memory:decisions_and_rationale'),
    _episode('d-tagged', '[temporal:2026-09] [unverified_claim] add_memory:decisions_and_rationale'),
    _episode('t-1', 'add_memory:temporal_facts'),
    _episode('t-tagged', '[temporal:2026-09] add_memory:temporal_facts'),
    _episode('r-1', 'add_memory:entities_and_relations'),
    _episode('p-1', 'add_memory:procedural_knowledge'),
    _episode('n-1', 'add_memory:preferences_and_norms'),
    _episode('x-none', None),
    _episode('x-text', 'a raw add_episode'),
]


class TestTheStrata:
    def test_the_extraction_strata(self) -> None:
        assert frame_mod.EXTRACTION_STRATA == (TEMPORAL, RELATIONS, OTHER)

    @pytest.mark.asyncio
    async def test_each_category_maps_to_its_stratum_and_decisions_are_excluded(self) -> None:
        drawn = await _frame({DF: _EpisodeGraph(MIXED)}, cap=10)
        assert {row['uuid']: row['stratum'] for row in drawn.worksheet} == {
            't-1': TEMPORAL, 't-tagged': TEMPORAL, 'r-1': RELATIONS,
            'p-1': OTHER, 'n-1': OTHER, 'x-none': OTHER, 'x-text': OTHER,
        }

    @pytest.mark.asyncio
    async def test_a_row_carries_its_parsed_category(self) -> None:
        drawn = await _frame({DF: _EpisodeGraph(MIXED)}, cap=10)
        categories = {row['uuid']: row['category'] for row in drawn.worksheet}
        assert (categories['t-tagged'], categories['p-1'], categories['x-text']) == (
            TEMPORAL, 'procedural_knowledge', None,
        )

    @pytest.mark.asyncio
    async def test_a_definition_with_other_strata_is_refused(self) -> None:
        with pytest.raises(ValueError, match='strata'):
            await frame_mod.extraction_frame(
                _readers({DF: _EpisodeGraph(MIXED)}),
                audit.SampleDefinition(window_start=START, window_end=END, cap=3),
            )


class TestTheWindowAndTheCap:
    @pytest.mark.asyncio
    async def test_only_episodes_created_inside_the_half_open_window_count(self) -> None:
        graph = _EpisodeGraph([
            _episode('at-start', 'add_memory:temporal_facts', START),
            _episode('before', 'add_memory:temporal_facts', '2026-07-31T23:59:59+00:00'),
            _episode('at-end', 'add_memory:temporal_facts', END),
            _episode('naive-inside', 'add_memory:temporal_facts', '2026-09-01T00:00:00'),
        ])
        drawn = await _frame({DF: graph}, cap=10)
        assert sorted(row['uuid'] for row in drawn.worksheet) == ['at-start', 'naive-inside']
        assert drawn.frame['strata'][DF][TEMPORAL]['window_population'] == 2

    @pytest.mark.asyncio
    async def test_each_cell_keeps_its_first_cap_by_the_audits_hash(self) -> None:
        uuids = [f't-{index}' for index in range(7)]
        graph = _EpisodeGraph([_episode(uuid, 'add_memory:temporal_facts') for uuid in uuids])
        drawn = await _frame({DF: graph}, cap=3)
        expected = sorted(uuids, key=lambda uuid: audit.sample_key(DF, uuid))[:3]
        assert [row['uuid'] for row in drawn.worksheet] == expected
        assert audit.sample_key(DF, 't-0') == _key(DF, 't-0')
        assert drawn.frame['strata'][DF][TEMPORAL]['window_population'] == 7
        assert drawn.frame['sample']['sizes'][DF][TEMPORAL]['episodes'] == 3

    @pytest.mark.asyncio
    async def test_a_short_metadata_read_refuses_to_draw_a_frame(self) -> None:
        with pytest.raises(frame_mod.FrameReadIncomplete, match=DF):
            await _frame({DF: _EpisodeGraph(MIXED, census_extra=3)})


class TestTheFrame:
    @pytest.mark.asyncio
    async def test_the_frame_is_what_the_replays_score_reads(self) -> None:
        drawn = await _frame({DF: _EpisodeGraph(MIXED), REIFY: _EpisodeGraph(MIXED[2:4])})
        baseline = replay.Baseline.from_report(drawn.frame)
        assert baseline.definition == _definition()
        assert baseline.overreach_rate_minted is None
        assert baseline.window_days == pytest.approx(109.0)
        assert baseline.window_population == {
            DF: {TEMPORAL: 2, RELATIONS: 1, OTHER: 4}, REIFY: {TEMPORAL: 2, RELATIONS: 0, OTHER: 0},
        }
        assert dict(baseline.sample_cells) == {
            f'{DF}/{TEMPORAL}': 2, f'{DF}/{RELATIONS}': 1, f'{DF}/{OTHER}': 3,
            f'{REIFY}/{TEMPORAL}': 2, f'{REIFY}/{RELATIONS}': 0, f'{REIFY}/{OTHER}': 0,
        }

    @pytest.mark.asyncio
    async def test_the_worksheet_matches_the_frame_and_carries_the_body(self) -> None:
        drawn = await _frame({DF: _EpisodeGraph(MIXED), REIFY: _EpisodeGraph(MIXED[2:4])})
        replay.check_worksheet(drawn.worksheet, replay.baseline_cells(drawn.frame))
        for row in drawn.worksheet:
            assert row.keys() == {'graph', 'uuid', 'stratum', 'category', 'created_at', 'content'}
            assert row['content'] == f"the body of {row['uuid']}"
            assert row['created_at'] == INSIDE

    @pytest.mark.asyncio
    async def test_the_frame_holds_no_episode_body(self) -> None:
        drawn = await _frame({DF: _EpisodeGraph(MIXED)})
        assert 'the body of' not in repr(drawn.frame)

    @pytest.mark.asyncio
    async def test_content_is_read_only_for_the_sampled_episodes(self) -> None:
        graph = _EpisodeGraph([_episode(f't-{index}', 'add_memory:temporal_facts') for index in range(7)])
        drawn = await _frame({DF: graph}, cap=3)
        assert sorted(graph.content_reads) == sorted(row['uuid'] for row in drawn.worksheet)
        paged = [cypher for cypher in graph.queries if 'SKIP' in cypher or 'count(' in cypher]
        assert paged and not any('e.content' in cypher for cypher in paged)


# --------------------------------------------------------------------------- #
# Graph-derived fixtures: resolve_edge and summaries (step 9)
# --------------------------------------------------------------------------- #

WINDOW = ('2026-10-01T00:00:00+00:00', '2026-11-18T00:00:00+00:00')
BEFORE, IN_1, IN_2, IN_3 = (
    '2026-09-20T00:00:00+00:00', '2026-10-05T00:00:00+00:00',
    '2026-10-06T00:00:00+00:00', '2026-10-07T00:00:00+00:00',
)


class _Store:
    """A whole graph served by population: RELATES_TO, Episodic, MENTIONS and Entity pages, each
    with its census, plus episode bodies by uuid. Records every body read."""

    def __init__(self, *, edges: Sequence[list[Any]] = (), episodes: Sequence[list[Any]] = (),
                 mentions: Sequence[list[Any]] = (), entities: Sequence[list[Any]] = (),
                 bodies: dict[str, str] | None = None) -> None:
        self.populations = {
            ':MENTIONS': sorted(mentions), ':RELATES_TO': sorted(edges),
            ':Episodic': sorted(episodes), ':Entity': sorted(entities),
        }
        self.bodies = bodies or {}
        self.body_reads: list[str] = []

    def _population(self, cypher: str) -> list[list[Any]]:
        match = cypher.split('RETURN')[0]
        return next(rows for marker, rows in self.populations.items() if marker in match)

    async def ro_query(self, cypher: str, params: dict[str, Any] | None = None) -> _Result:
        if 'e.content' in cypher:
            wanted = list(params['uuids']) if params else []
            self.body_reads.extend(wanted)
            return _Result([[uuid, self.bodies[uuid]] for uuid in wanted if uuid in self.bodies])
        rows = self._population(cypher)
        if _CENSUS_RE.search(cypher.strip()):
            return _Result([[len(rows)]])
        match = _SKIP_LIMIT_RE.search(cypher)
        assert match, cypher
        skip, limit = int(match.group(1)), int(match.group(2))
        return _Result(rows[skip: skip + limit])

    async def query(self, cypher: str, params: dict[str, Any] | None = None) -> Any:
        raise AssertionError('the fixtures are read-only: they may never issue query()')


def _edge(uuid: str, source: str, target: str, fact: str, created_at: str, episodes: tuple[str, ...] = ('ep-1',),
          *, invalid_at: str | None = None) -> list[Any]:
    """A RELATES_TO row in the order the frame projects it."""
    return [uuid, source, target, fact, created_at, created_at, invalid_at, None, list(episodes)]


def _store_readers(store: _Store, graph: str = DF) -> dict[str, ReadOnlyGraphReader]:
    return {graph: ReadOnlyGraphReader(graph_name=graph, graph=store, page_size=2, resultset_size=5)}


EPISODES = [['ep-1', IN_1, 'add_memory:temporal_facts', '2026-10-05T01:00:00+00:00']]


def _resolve_store(*edges: list[Any]) -> _Store:
    return _Store(edges=list(edges), episodes=EPISODES, bodies={'ep-1': 'the body of ep-1'})


async def _resolve(store: _Store, n: int = 10, **caps: int) -> list[dict[str, Any]]:
    return await frame_mod.resolve_edge_fixtures(_store_readers(store), WINDOW, n, **caps)


class TestResolveEdgeFixtures:
    @pytest.mark.asyncio
    async def test_a_candidate_is_resolved_against_the_older_edges_between_its_pair_newest_first(self) -> None:
        store = _resolve_store(
            _edge('c', 'A', 'B', 'Alpha adopts Beta for the queue', IN_2),
            _edge('s-old', 'A', 'B', 'Alpha weighs Beta', BEFORE),
            _edge('s-new', 'A', 'B', 'Alpha trials Beta', IN_1),
            _edge('s-later', 'A', 'B', 'Alpha drops Beta', IN_3),
            _edge('reverse', 'B', 'A', 'Beta serves Alpha', BEFORE),
        )
        fixtures = await _resolve(store)
        (fixture,) = [f for f in fixtures if f['edge_uuid'] == 'c']
        assert [edge['uuid'] for edge in fixture['related']] == ['s-new', 's-old']

    @pytest.mark.asyncio
    async def test_siblings_are_capped(self) -> None:
        siblings = [_edge(f's-{index}', 'A', 'B', f'fact {index}', BEFORE) for index in range(4)]
        fixtures = await _resolve(_resolve_store(_edge('c', 'A', 'B', 'new fact', IN_2), *siblings), sibling_cap=2)
        (fixture,) = [f for f in fixtures if f['edge_uuid'] == 'c']
        assert len(fixture['related']) == 2

    @pytest.mark.asyncio
    async def test_an_edge_with_no_older_sibling_or_a_verbatim_one_is_skipped(self) -> None:
        store = _resolve_store(
            _edge('lonely', 'C', 'D', 'Gamma feeds Delta', IN_2),
            _edge('verbatim', 'A', 'B', '  ALPHA adopts   beta ', IN_2),
            _edge('s', 'A', 'B', 'Alpha adopts Beta', BEFORE),
        )
        assert [f['edge_uuid'] for f in await _resolve(store)] == []

    @pytest.mark.asyncio
    async def test_candidates_are_the_windows_edges_in_hash_order_up_to_n(self) -> None:
        candidates = [_edge(f'c-{index}', 'A', 'B', f'new fact {index}', IN_2) for index in range(5)]
        store = _resolve_store(*candidates, _edge('s', 'A', 'B', 'old fact', BEFORE),
                               _edge('outside', 'A', 'B', 'too old', '2026-09-25T00:00:00+00:00'))
        fixtures = await _resolve(store, n=3)
        expected = sorted((f'c-{index}' for index in range(5)), key=lambda uuid: _key(DF, uuid))[:3]
        assert [f['edge_uuid'] for f in fixtures] == expected

    @pytest.mark.asyncio
    async def test_invalidation_candidates_are_older_edges_on_either_endpoint_off_the_pair(self) -> None:
        store = _resolve_store(
            _edge('c', 'A', 'B', 'Alpha adopts Beta', IN_2),
            _edge('s', 'A', 'B', 'Alpha weighs Beta', BEFORE),
            _edge('on-a', 'A', 'X', 'Alpha uses Xi', BEFORE),
            _edge('on-b', 'Y', 'B', 'Psi feeds Beta', BEFORE),
            _edge('reverse', 'B', 'A', 'Beta serves Alpha', BEFORE),
            _edge('newer', 'A', 'Z', 'Alpha needs Zeta', IN_3),
            _edge('elsewhere', 'P', 'Q', 'Pi meets Qi', BEFORE),
        )
        (fixture, *_) = [f for f in await _resolve(store) if f['edge_uuid'] == 'c']
        expected = sorted(['on-a', 'on-b', 'reverse'], key=lambda uuid: _key(DF, uuid))
        assert [edge['uuid'] for edge in fixture['invalidation_candidates']] == expected

    @pytest.mark.asyncio
    async def test_invalidation_candidates_are_capped(self) -> None:
        incident = [_edge(f'i-{index}', 'A', f'X{index}', f'fact {index}', BEFORE) for index in range(4)]
        store = _resolve_store(_edge('c', 'A', 'B', 'new', IN_2), _edge('s', 'A', 'B', 'old', BEFORE), *incident)
        (fixture, *_) = [f for f in await _resolve(store, candidate_cap=2) if f['edge_uuid'] == 'c']
        assert len(fixture['invalidation_candidates']) == 2

    @pytest.mark.asyncio
    async def test_a_fixture_carries_ids_facts_times_and_its_episode_but_no_body(self) -> None:
        store = _resolve_store(_edge('c', 'A', 'B', 'Alpha adopts Beta', IN_2, ('ep-1', 'ep-2')),
                               _edge('s', 'A', 'B', 'Alpha weighs Beta', BEFORE, invalid_at=IN_1))
        (fixture,) = await _resolve(store)
        assert {k: fixture[k] for k in ('graph', 'edge_uuid', 'fact', 'source_node_uuid', 'target_node_uuid')} == {
            'graph': DF, 'edge_uuid': 'c', 'fact': 'Alpha adopts Beta', 'source_node_uuid': 'A', 'target_node_uuid': 'B',
        }
        assert (fixture['created_at'], fixture['valid_at'], fixture['invalid_at'], fixture['expired_at']) == (
            IN_2, IN_2, None, None,
        )
        assert (fixture['episode_uuid'], fixture['episode_valid_at']) == ('ep-1', '2026-10-05T01:00:00+00:00')
        assert fixture['related'] == [{
            'uuid': 's', 'fact': 'Alpha weighs Beta', 'source_node_uuid': 'A', 'target_node_uuid': 'B',
            'created_at': BEFORE, 'valid_at': BEFORE, 'invalid_at': IN_1, 'expired_at': None,
        }]
        assert 'the body of' not in repr(fixture) and store.body_reads == []


def _long(prefix: str, length: int) -> str:
    return (prefix + ' ') * (length // (len(prefix) + 1)) + 'x' * (length % (len(prefix) + 1))


LIMIT = 4 * 500
"""4 * graphiti_core.utils.text_utils.MAX_SUMMARY_CHARS in graphiti_core 0.28.2."""


def _summary_store(summaries: dict[str, str], *, minted: tuple[tuple[str, str, str], ...] = (),
                   episodes: tuple[str, ...] = ('ep-1',)) -> _Store:
    """Each episode mentions every node in *summaries*; *minted* edges are (uuid, node, fact) of ep-1."""
    return _Store(
        episodes=[[uuid, IN_1, 'add_memory:temporal_facts', IN_1] for uuid in episodes],
        mentions=[[f'm-{ep}-{node}', ep, node] for ep in episodes for node in summaries],
        entities=[[node, f'name of {node}', summary] for node, summary in summaries.items()],
        edges=[_edge(uuid, node, 'OTHER', fact, IN_1, ('ep-1',)) for uuid, node, fact in minted],
        bodies={uuid: f'the body of {uuid}' for uuid in episodes},
    )


async def _summaries(store: _Store, n: int = 10, **kwargs: int) -> list[dict[str, Any]]:
    return await frame_mod.summary_fixtures(_store_readers(store), WINDOW, n, **kwargs)


class TestSummaryFixtures:
    def test_the_threshold_is_graphitis_own_constant(self) -> None:
        from graphiti_core.utils.text_utils import MAX_SUMMARY_CHARS
        assert frame_mod.MAX_SUMMARY_CHARS is MAX_SUMMARY_CHARS and 4 * MAX_SUMMARY_CHARS == LIMIT

    @pytest.mark.asyncio
    async def test_a_node_is_an_item_only_past_four_times_max_summary_chars(self) -> None:
        store = _summary_store({'at-limit': _long('kept', LIMIT), 'past-limit': _long('long', LIMIT + 1)})
        assert [item['node_uuid'] for item in await _summaries(store)] == ['past-limit']

    @pytest.mark.asyncio
    async def test_the_episodes_minted_incident_facts_count_toward_the_length(self) -> None:
        summary = _long('base', LIMIT - 10)
        store = _summary_store({'N': summary}, minted=(('r1', 'N', 'a fact that tips it over'),))
        (item,) = await _summaries(store)
        assert item['edge_uuids'] == ['r1']
        assert item['summary_with_edges_chars'] == len(f'{summary}\na fact that tips it over'.strip())

    @pytest.mark.asyncio
    async def test_an_edge_the_episode_only_corroborates_does_not_count(self) -> None:
        store = _summary_store({'N': _long('base', LIMIT - 10)})
        store.populations[':RELATES_TO'] = [_edge('r1', 'N', 'OTHER', 'a long corroborated fact', IN_1, ('ep-0', 'ep-1'))]
        assert await _summaries(store) == []

    @pytest.mark.asyncio
    async def test_at_most_nodes_per_episode_by_hash_and_at_most_n(self) -> None:
        nodes = {f'N{index}': _long('long', LIMIT + 50) for index in range(4)}
        store = _summary_store(nodes, episodes=('ep-1', 'ep-2', 'ep-3'))
        items = await _summaries(store, n=5, nodes_per_episode=2)
        assert len(items) == 5
        by_episode: dict[str, list[str]] = {}
        for item in items:
            by_episode.setdefault(item['episode_uuid'], []).append(item['node_uuid'])
        episode_order = sorted(('ep-1', 'ep-2', 'ep-3'), key=lambda uuid: _key(DF, uuid))
        assert list(by_episode) == episode_order
        node_order = sorted(nodes, key=lambda uuid: _key(DF, uuid))[:2]
        assert all(chosen == node_order[: len(chosen)] for chosen in by_episode.values())

    @pytest.mark.asyncio
    async def test_an_episode_outside_the_window_offers_no_item(self) -> None:
        store = _summary_store({'N': _long('long', LIMIT + 1)})
        store.populations[':Episodic'] = [['ep-1', BEFORE, 'add_memory:temporal_facts', BEFORE]]
        assert await _summaries(store) == []

    @pytest.mark.asyncio
    async def test_the_manifest_carries_ids_and_hashes_not_bodies(self) -> None:
        summary = _long('long', LIMIT + 1)
        (item,) = await _summaries(_summary_store({'N': summary}))
        assert item['graph'] == DF and (item['episode_uuid'], item['node_uuid']) == ('ep-1', 'N')
        assert item['node_name'] == 'name of N' and item['episode_valid_at'] == IN_1
        assert item['episode_content_sha256'] == hashlib.sha256(b'the body of ep-1').hexdigest()
        assert item['summary_sha256'] == hashlib.sha256(summary.encode()).hexdigest()
        assert summary not in repr(item) and 'the body of' not in repr(item)


# --------------------------------------------------------------------------- #
# Journal fixtures: the classifier and the router (step 11)
# --------------------------------------------------------------------------- #

JOURNAL_WINDOW = ('2026-11-04T00:00:00+00:00', '2026-11-18T00:00:00+00:00')
J_IN, J_OUT = '2026-11-10T12:00:00+00:00', '2026-11-01T12:00:00+00:00'
CONFIDENT_WRITE = 'The auth service depends on the user database'
"""One heuristic category at 0.85: classified with no LLM call."""
AMBIGUOUS_WRITE = 'We decided to migrate before the deadline changed'
CONFIDENT_QUERY = 'What is the auth service?'
"""One heuristic query type: routed with no LLM call."""


def _journal(tmp_path: Path, rows: Sequence[tuple[str, str, str, dict[str, Any], str]]) -> Path:
    """A write journal with the production schema; each row is (id, operation, kind, params, created_at)."""
    import json
    import sqlite3

    from fused_memory.services.write_journal import SCHEMA_SQL

    path = tmp_path / 'write_journal.db'
    with sqlite3.connect(path) as db:
        db.executescript(SCHEMA_SQL)
        db.executemany(
            'INSERT INTO write_ops (id, operation, kind, params, created_at) VALUES (?, ?, ?, ?, ?)',
            [(row_id, op, kind, json.dumps(params), at) for row_id, op, kind, params, at in rows],
        )
    return path


def _write(row_id: str, content: str, category: str, at: str = J_IN) -> tuple[str, str, str, dict[str, Any], str]:
    return (row_id, 'add_memory', 'write', {'content': content, 'category': category}, at)


def _search(row_id: str, query: str, at: str = J_IN, *, truncated: bool = False) -> tuple[str, str, str, dict[str, Any], str]:
    return (row_id, 'search', 'read', {'query': query, 'query_truncated': truncated, 'limit': 10}, at)


class TestClassifierFixtures:
    @pytest.mark.asyncio
    async def test_only_window_inputs_that_reach_the_llm_are_kept(self, tmp_path: Path) -> None:
        journal = _journal(tmp_path, [
            _write('w-confident', CONFIDENT_WRITE, RELATIONS),
            _write('w-ambiguous', AMBIGUOUS_WRITE, TEMPORAL),
            _write('w-nomatch', 'Hello world', 'observations_and_summaries'),
            _write('w-outside', 'Hello again', 'observations_and_summaries', J_OUT),
            ('s-1', 'search', 'read', {'query': 'Hello world', 'query_truncated': False}, J_IN),
        ])
        fixtures = await frame_mod.classifier_fixtures(journal, JOURNAL_WINDOW, per_category=10)
        assert sorted(f['id'] for f in fixtures) == ['w-ambiguous', 'w-nomatch']

    @pytest.mark.asyncio
    async def test_a_fixture_carries_its_row_input_and_reference_category(self, tmp_path: Path) -> None:
        journal = _journal(tmp_path, [_write('w-1', AMBIGUOUS_WRITE, TEMPORAL)])
        (fixture,) = await frame_mod.classifier_fixtures(journal, JOURNAL_WINDOW, per_category=10)
        assert fixture == {'id': 'w-1', 'created_at': J_IN, 'input': AMBIGUOUS_WRITE, 'reference_category': TEMPORAL}

    @pytest.mark.asyncio
    async def test_the_first_per_category_by_row_hash(self, tmp_path: Path) -> None:
        rows = [_write(f'w-{index}', f'Hello world {index}', TEMPORAL) for index in range(5)]
        rows += [_write('r-1', 'Hello there', RELATIONS)]
        fixtures = await frame_mod.classifier_fixtures(_journal(tmp_path, rows), JOURNAL_WINDOW, per_category=2)
        temporal = sorted((f'w-{index}' for index in range(5)), key=lambda row_id: hashlib.sha256(row_id.encode()).hexdigest())
        assert [f['id'] for f in fixtures if f['reference_category'] == TEMPORAL] == temporal[:2]
        assert [f['id'] for f in fixtures if f['reference_category'] == RELATIONS] == ['r-1']


class TestRouterFixtures:
    @pytest.mark.asyncio
    async def test_only_untruncated_window_queries_that_reach_the_llm_are_kept(self, tmp_path: Path) -> None:
        journal = _journal(tmp_path, [
            _search('q-confident', CONFIDENT_QUERY),
            _search('q-nomatch', 'foo bar baz'),
            _search('q-truncated', 'foo bar qux', truncated=True),
            _search('q-outside', 'foo bar quux', J_OUT),
            _write('w-1', 'foo bar baz', TEMPORAL),
        ])
        fixtures = await frame_mod.router_fixtures(journal, JOURNAL_WINDOW, n=10)
        assert fixtures == [{'id': 'q-nomatch', 'created_at': J_IN, 'query': 'foo bar baz'}]

    @pytest.mark.asyncio
    async def test_identical_queries_are_deduped_and_the_first_n_by_row_hash_kept(self, tmp_path: Path) -> None:
        rows = [_search(f'q-{index}', f'foo bar {index}') for index in range(5)]
        rows += [_search('q-dup', 'foo bar 0')]
        fixtures = await frame_mod.router_fixtures(_journal(tmp_path, rows), JOURNAL_WINDOW, n=3)
        assert len({f['query'] for f in fixtures}) == len(fixtures) == 3
        kept = sorted({f['id'] for f in fixtures})
        ranked = sorted((r[0] for r in rows), key=lambda row_id: hashlib.sha256(row_id.encode()).hexdigest())
        first_per_query: dict[str, str] = {}
        for row_id in ranked:
            first_per_query.setdefault(next(r[3]['query'] for r in rows if r[0] == row_id), row_id)
        assert kept == sorted(sorted(first_per_query.values(), key=lambda row_id: hashlib.sha256(row_id.encode()).hexdigest())[:3])


class TestJournalRowsThatAreNotWrites:
    @pytest.mark.asyncio
    async def test_a_row_whose_params_are_not_a_json_object_is_skipped_and_counted(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging
        import sqlite3

        journal = _journal(tmp_path, [_write('w-1', 'Hello world', TEMPORAL)])
        with sqlite3.connect(journal) as db:
            db.executemany(
                'INSERT INTO write_ops (id, operation, kind, params, created_at) VALUES (?, ?, ?, ?, ?)',
                [('w-garbled', 'add_memory', 'write', '{not json', J_IN), ('w-list', 'add_memory', 'write', '[1]', J_IN)],
            )
        with caplog.at_level(logging.WARNING):
            fixtures = await frame_mod.classifier_fixtures(journal, JOURNAL_WINDOW, per_category=10)
        assert [fixture['id'] for fixture in fixtures] == ['w-1']
        assert '2 add_memory row(s)' in caplog.text

    @pytest.mark.asyncio
    async def test_a_journal_of_readable_rows_warns_nothing(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        with caplog.at_level(logging.WARNING):
            await frame_mod.router_fixtures(_journal(tmp_path, [_search('q-1', 'foo bar baz')]), JOURNAL_WINDOW, n=3)
        assert 'row(s)' not in caplog.text


class TestTheJournalIsReadOnly:
    def test_a_write_through_the_journal_connection_fails(self, tmp_path: Path) -> None:
        import sqlite3

        journal = _journal(tmp_path, [_write('w-1', 'Hello world', TEMPORAL)])
        with frame_mod.open_journal(journal) as db, pytest.raises(sqlite3.OperationalError):
            db.execute("DELETE FROM write_ops WHERE id = 'w-1'")

    def test_a_missing_journal_is_refused_and_not_created(self, tmp_path: Path) -> None:
        missing = tmp_path / 'absent.db'
        with pytest.raises(FileNotFoundError):
            frame_mod.open_journal(missing)
        assert not missing.exists()
