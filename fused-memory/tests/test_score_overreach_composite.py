"""Tests for scripts/score_overreach_composite.py (task 9040).

The composite is Stage 1's retire/keep decision on the overreach flags the
judge handed it. These tests seed a real sidecar through OverreachShadowStore
and answer the graph reads with a double behind the read-only seam.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from _fm_helpers import load_script_module
from _overreach_shadow_fixtures import judged_edge, judged_episode

from fused_memory.backends.falkor_read_only import ReadOnlyGraphReader
from fused_memory.backends.graphiti_client import _RESULTSET_SIZE
from fused_memory.services.overreach_guard import PROMPT_VERSION
from fused_memory.services.overreach_shadow import SIDECAR_FILENAME, OverreachShadowStore

SCRIPTS = Path(__file__).parent.parent / 'scripts'
mod = load_script_module(SCRIPTS / 'score_overreach_composite.py', mod_name='score_overreach_composite')
audit = load_script_module(SCRIPTS / 'audit_ruling_overreach.py', mod_name='audit_ruling_overreach')

DF, REIFY, THIRD = 'dark_factory', 'reify', 'third'
WINDOW_START, WINDOW_END = '2026-10-20T00:00:00+00:00', '2026-10-30T00:00:00+00:00'
JUDGED = datetime(2026, 10, 19, 6, 0, tzinfo=UTC)
BEFORE = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)
IN_WINDOW = datetime(2026, 10, 25, 6, 0, tzinfo=UTC)
RETIRED_AT = '2026-10-25T06:05:00+00:00'

EDGES: dict[str, dict[str, tuple[str | None, str | None]]] = {
    DF: {'r1': (RETIRED_AT, None), 'r2': (RETIRED_AT, None), 'r3': (RETIRED_AT, None),
         'k1': (None, None), 'k2': (None, RETIRED_AT), 's1': (RETIRED_AT, RETIRED_AT),
         'h1': (None, None), 'o1': (RETIRED_AT, None), 'p1': (None, None)},
    REIFY: {'rr1': (RETIRED_AT, None), 'rk1': (None, None)},
    THIRD: {'x1': (RETIRED_AT, None)},
}
"""``(invalid_at, expired_at)`` per live edge; g1 is gone from dark_factory."""

EPISODES = {
    DF: {'ep-r': 'the body of ep-r', 'ep-k': 'the body of ep-k', 'ep-s': 'the body of ep-s',
         'ep-o': 'the body of ep-o'},
    REIFY: {'ep-rr': 'the body of ep-rr'},
    THIRD: {'ep-x': 'the body of ep-x'},
}


class _SetClock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


async def _seed_sidecar(data_dir: Path) -> Path:
    """dark_factory: r1-r3 retired, k1-k2 kept, s1 superseded, g1 gone, all overreach and handed
    in the window; h1 a handed holding; o1 handed before the window; p1 never handed.
    reify: rr1 retired, rk1 kept. third: x1, a graph nobody asks for."""
    clock = _SetClock(JUDGED)
    store = await OverreachShadowStore.open(data_dir, clock=clock)
    try:
        await store.record(episode=judged_episode('ep-r', graph=DF), judgements=[
            judged_edge(uuid, 'overreach') for uuid in ('r1', 'r2', 'r3')
        ])
        await store.record(episode=judged_episode('ep-k', graph=DF), judgements=[
            judged_edge('k1', 'overreach'), judged_edge('k2', 'overreach'), judged_edge('h1', 'holding'),
        ])
        await store.record(episode=judged_episode('ep-s', graph=DF), judgements=[
            judged_edge('s1', 'overreach'), judged_edge('g1', 'overreach'),
        ])
        await store.record(episode=judged_episode('ep-o', graph=DF), judgements=[
            judged_edge('o1', 'overreach'), judged_edge('p1', 'overreach'),
        ])
        await store.record(episode=judged_episode('ep-rr', graph=REIFY), judgements=[
            judged_edge('rr1', 'overreach'), judged_edge('rk1', 'overreach'),
        ])
        await store.record(episode=judged_episode('ep-x', graph=THIRD), judgements=[judged_edge('x1', 'overreach')])
        clock.now = BEFORE
        await store.record_handoffs(DF, ['o1'])
        clock.now = IN_WINDOW
        await store.record_handoffs(DF, ['r1', 'r2', 'r3', 'k1', 'k2', 'h1', 's1', 'g1'])
        await store.record_handoffs(REIFY, ['rr1', 'rk1'])
        await store.record_handoffs(THIRD, ['x1'])
    finally:
        await store.close()
    return data_dir / SIDECAR_FILENAME


@pytest.fixture
def sidecar(tmp_path: Path) -> Path:
    return asyncio.run(_seed_sidecar(tmp_path / 'data'))


class _StateGraph:
    """Answers the edge-state and episode-content reads for one graph; records what it was sent."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.sent: list[tuple[str, dict[str, Any]]] = []

    async def ro_query(self, cypher: str, params: dict[str, Any] | None = None) -> Any:
        assert params is not None
        self.sent.append((cypher, params))
        if cypher == mod.EDGE_STATE_CYPHER:
            edges = EDGES[self.name]
            rows = [[uuid, *edges[uuid], f'source of {uuid}', f'target of {uuid}']
                    for uuid in params['uuids'] if uuid in edges]
        elif cypher == mod.EPISODE_CONTENT_CYPHER:
            episodes = EPISODES[self.name]
            rows = [[uuid, episodes[uuid]] for uuid in params['uuids'] if uuid in episodes]
        else:
            raise AssertionError(f'unexpected read {cypher!r}')
        return SimpleNamespace(result_set=rows)


class _Graphs:
    def __init__(self) -> None:
        self.opened: dict[str, _StateGraph] = {}

    def __call__(self, graph: str) -> ReadOnlyGraphReader:
        self.opened[graph] = _StateGraph(graph)
        return ReadOnlyGraphReader(graph_name=graph, graph=self.opened[graph])


def _paths(tmp_path: Path, tag: str = 'run') -> dict[str, Path]:
    return {'sample': tmp_path / tag / 'composite-sample.json', 'worksheet': tmp_path / tag / 'worksheet.jsonl'}


def _sample_argv(sidecar: Path, paths: dict[str, Path], *, cap: int = 2,
                 window: tuple[str, str] = (WINDOW_START, WINDOW_END)) -> list[str]:
    return ['sample', '--sidecar', str(sidecar), '--graph', DF, '--graph', REIFY,
            '--window-start', window[0], '--window-end', window[1], '--cap', str(cap),
            '--out-sample', str(paths['sample']), '--out-worksheet', str(paths['worksheet'])]


def _sampled(sidecar: Path, tmp_path: Path, graphs: _Graphs | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = _paths(tmp_path)
    assert mod.run(_sample_argv(sidecar, paths), reader_factory=graphs or _Graphs()) == 0
    worksheet = [json.loads(line) for line in paths['worksheet'].read_text().splitlines()]
    return json.loads(paths['sample'].read_text()), worksheet


def _first_by_hash(graph: str, uuids: list[str], cap: int) -> list[str]:
    return sorted(uuids, key=lambda uuid: audit.sample_key(graph, uuid))[:cap]


class TestDecisionOf:
    @pytest.mark.parametrize(('invalid_at', 'expired_at', 'decision'), [
        (RETIRED_AT, None, 'retired'),
        (RETIRED_AT, RETIRED_AT, 'superseded'),
        (None, None, 'kept'),
        (None, RETIRED_AT, 'kept'),
    ])
    def test_a_live_edge_is_classified_by_its_stamps(
        self, invalid_at: str | None, expired_at: str | None, decision: str,
    ) -> None:
        state = mod.EdgeState(invalid_at=invalid_at, expired_at=expired_at, source_name='S', target_name='T')
        assert mod.decision_of(state) == decision

    def test_an_absent_edge_is_gone(self) -> None:
        assert mod.decision_of(None) == 'gone'


class TestTheSample:
    def test_only_handed_overreach_flags_in_the_window_and_the_requested_graphs(
        self, sidecar: Path, tmp_path: Path,
    ) -> None:
        sample, _ = _sampled(sidecar, tmp_path)
        sampled = {(f['graph'], f['edge_uuid']) for f in sample['flags']}
        assert sampled == {
            *((DF, uuid) for uuid in _first_by_hash(DF, ['r1', 'r2', 'r3'], 2)),
            (DF, 'k1'), (DF, 'k2'), (REIFY, 'rr1'), (REIFY, 'rk1'),
        }

    def test_each_cell_takes_the_first_cap_by_the_audits_sample_key(
        self, sidecar: Path, tmp_path: Path,
    ) -> None:
        sample, _ = _sampled(sidecar, tmp_path)
        retired = [f['edge_uuid'] for f in sample['flags'] if (f['graph'], f['decision']) == (DF, 'retired')]
        assert sorted(retired) == sorted(_first_by_hash(DF, ['r1', 'r2', 'r3'], 2))
        assert mod.sample_key is audit.sample_key

    def test_each_sampled_flag_carries_its_decision_and_handoff(self, sidecar: Path, tmp_path: Path) -> None:
        sample, _ = _sampled(sidecar, tmp_path)
        by_edge = {(f['graph'], f['edge_uuid']): f for f in sample['flags']}
        assert by_edge[(DF, 'k2')] == {
            'graph': DF, 'edge_uuid': 'k2', 'episode_uuid': 'ep-k', 'fact': 'fact of k2',
            'decision': 'kept', 'handed_at': IN_WINDOW.isoformat(),
        }
        assert by_edge[(REIFY, 'rr1')]['decision'] == 'retired'

    def test_superseded_and_gone_flags_are_counted_but_never_sampled(
        self, sidecar: Path, tmp_path: Path,
    ) -> None:
        sample, _ = _sampled(sidecar, tmp_path)
        assert {f['decision'] for f in sample['flags']} == {'retired', 'kept'}
        census = sample['census']['by_graph'][DF]
        assert (census['superseded']['flags'], census['gone']['flags']) == (1, 1)

    def test_the_census_counts_every_handed_flag_per_graph_and_decision_per_day(
        self, sidecar: Path, tmp_path: Path,
    ) -> None:
        census = _sampled(sidecar, tmp_path)[0]['census']
        assert census['window_days'] == 10.0
        assert census['by_graph'] == {
            DF: {'handed': 7, 'retired': {'flags': 3, 'per_day': 0.3}, 'kept': {'flags': 2, 'per_day': 0.2},
                 'superseded': {'flags': 1, 'per_day': 0.1}, 'gone': {'flags': 1, 'per_day': 0.1}},
            REIFY: {'handed': 2, 'retired': {'flags': 1, 'per_day': 0.1}, 'kept': {'flags': 1, 'per_day': 0.1},
                    'superseded': {'flags': 0, 'per_day': 0.0}, 'gone': {'flags': 0, 'per_day': 0.0}},
        }
        assert census['prompt_versions'] == [PROMPT_VERSION]

    def test_the_sample_definition_is_frozen_into_the_file(self, sidecar: Path, tmp_path: Path) -> None:
        sample, _ = _sampled(sidecar, tmp_path)
        assert sample['sample'] == audit.SampleDefinition(
            window_start=WINDOW_START, window_end=WINDOW_END, cap=2,
            strata=('retired', 'kept'), hash_rule='sha256(graph:edge_uuid)',
        ).to_dict()
        assert mod.HASH_RULE == 'sha256(graph:edge_uuid)' and mod.SAMPLED_DECISIONS == ('retired', 'kept')
        assert sample['graphs'] == [DF, REIFY]
        assert datetime.fromisoformat(sample['sampled_at']).tzinfo is not None

    def test_the_sample_file_holds_no_episode_body_and_no_judge_rationale(
        self, sidecar: Path, tmp_path: Path,
    ) -> None:
        _sampled(sidecar, tmp_path)
        text = _paths(tmp_path)['sample'].read_text()
        assert 'the body of' not in text and 'span for' not in text

    def test_a_rerun_is_byte_identical_apart_from_sampled_at(self, sidecar: Path, tmp_path: Path) -> None:
        first, second = _paths(tmp_path, 'first'), _paths(tmp_path, 'second')
        assert mod.run(_sample_argv(sidecar, first), reader_factory=_Graphs()) == 0
        assert mod.run(_sample_argv(sidecar, second), reader_factory=_Graphs()) == 0
        assert first['worksheet'].read_bytes() == second['worksheet'].read_bytes()
        one, other = (json.loads(p['sample'].read_text()) for p in (first, second))
        for document in (one, other):
            document.pop('sampled_at')
        assert one == other

    def test_small_read_batches_give_the_same_sample(
        self, sidecar: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        whole, _ = _sampled(sidecar, tmp_path / 'whole')
        monkeypatch.setattr(mod, 'UUID_BATCH', 2)
        graphs = _Graphs()
        batched, _ = _sampled(sidecar, tmp_path / 'batched', graphs)
        assert batched['flags'] == whole['flags'] and batched['census'] == whole['census']
        assert all(len(params['uuids']) <= 2 for g in graphs.opened.values() for _, params in g.sent)

    def test_a_read_batch_stays_well_under_the_result_set_cap(self) -> None:
        assert mod.UUID_BATCH * 10 <= _RESULTSET_SIZE

    def test_only_the_requested_graphs_are_read_and_only_by_the_two_reads(
        self, sidecar: Path, tmp_path: Path,
    ) -> None:
        graphs = _Graphs()
        _sampled(sidecar, tmp_path, graphs)
        assert set(graphs.opened) == {DF, REIFY}
        assert {cypher for g in graphs.opened.values() for cypher, _ in g.sent} == {
            mod.EDGE_STATE_CYPHER, mod.EPISODE_CONTENT_CYPHER,
        }


class TestTheWorksheet:
    def test_one_row_per_sampled_episode_with_its_body_and_flagged_facts(
        self, sidecar: Path, tmp_path: Path,
    ) -> None:
        sample, worksheet = _sampled(sidecar, tmp_path)
        assert {(r['graph'], e['edge_uuid']) for r in worksheet for e in r['edges']} == {
            (f['graph'], f['edge_uuid']) for f in sample['flags']
        }
        rows = {(r['graph'], r['uuid']): r for r in worksheet}
        assert rows[(DF, 'ep-k')]['content'] == 'the body of ep-k'
        assert {e['fact'] for e in rows[(DF, 'ep-k')]['edges']} == {'fact of k1', 'fact of k2'}
        assert rows[(DF, 'ep-k')]['edges'][0]['source_name'].startswith('source of ')

    def test_it_names_no_decision_and_no_judge_rationale(self, sidecar: Path, tmp_path: Path) -> None:
        _, worksheet = _sampled(sidecar, tmp_path)
        text = _paths(tmp_path)['worksheet'].read_text()
        assert 'span for' not in text and 'retired' not in text and 'kept' not in text
        assert all('decision' not in row and all('decision' not in e for e in row['edges']) for row in worksheet)

    def test_rows_and_edges_are_ordered_by_hash_not_by_cell(self, sidecar: Path, tmp_path: Path) -> None:
        _, worksheet = _sampled(sidecar, tmp_path)
        assert [(r['graph'], r['uuid']) for r in worksheet] == sorted(
            ((r['graph'], r['uuid']) for r in worksheet), key=lambda key: audit.sample_key(*key),
        )
        for row in worksheet:
            hashes = [hashlib.sha256(f"{row['graph']}:{e['edge_uuid']}".encode()).hexdigest() for e in row['edges']]
            assert hashes == sorted(hashes)


class TestSampleFailsClosed:
    def _assert_nothing_written(self, paths: dict[str, Path]) -> None:
        assert not paths['sample'].exists() and not paths['worksheet'].exists()

    def test_a_missing_sidecar_exits_1(self, tmp_path: Path) -> None:
        paths = _paths(tmp_path)
        assert mod.run(_sample_argv(tmp_path / 'absent.db', paths), reader_factory=_Graphs()) == 1
        self._assert_nothing_written(paths)

    def test_an_unreadable_graph_exits_1(self, sidecar: Path, tmp_path: Path) -> None:
        class _Down:
            name = DF

            async def ro_query(self, cypher: str, params: dict[str, Any] | None = None) -> Any:
                raise ConnectionError('falkordb down')

        paths = _paths(tmp_path)
        code = mod.run(_sample_argv(sidecar, paths),
                       reader_factory=lambda graph: ReadOnlyGraphReader(graph_name=graph, graph=_Down()))
        assert code == 1
        self._assert_nothing_written(paths)

    @pytest.mark.parametrize('window', [
        (WINDOW_END, WINDOW_START),
        (WINDOW_START, WINDOW_START),
        ('2026-10-20T00:00:00', WINDOW_END),
        ('not a time', WINDOW_END),
    ])
    def test_an_invalid_window_exits_2(self, sidecar: Path, tmp_path: Path, window: tuple[str, str]) -> None:
        paths = _paths(tmp_path)
        assert mod.run(_sample_argv(sidecar, paths, window=window), reader_factory=_Graphs()) == 2
        self._assert_nothing_written(paths)

    def test_a_cap_below_one_exits_2(self, sidecar: Path, tmp_path: Path) -> None:
        paths = _paths(tmp_path)
        assert mod.run(_sample_argv(sidecar, paths, cap=0), reader_factory=_Graphs()) == 2
        self._assert_nothing_written(paths)
