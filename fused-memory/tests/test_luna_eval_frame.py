"""Tests for scripts/luna_eval_frame.py (task 9945): the frozen, read-only luna-eval sample frame.

Every read goes through a ReadOnlyGraphReader over a fake graph that answers by
Cypher, as test_replay_overreach_extraction.py and test_audit_ruling_overreach.py
do; nothing here can reach a FalkorDB.
"""
from __future__ import annotations

import hashlib
import re
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
