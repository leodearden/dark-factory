"""Tests for rebuild_write_triage_live_slates.py — the soak's write-time slates, rebuilt.

Synthetic only: no soak bundle, no dump on disk outside tmp_path, no Qdrant.
Vectors are 1536-d with a few non-zero axes, so every cosine is chosen.
"""
from __future__ import annotations

import functools
import json
import math
import sqlite3
import types
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _fm_helpers import load_script_module

SCRIPTS = Path(__file__).parent.parent / 'scripts'
DIMS = 1536

WRITE_AT = '2026-10-12T00:00:00+00:00'
EARLIER = '2026-10-11T20:00:00+00:00'
LATER = '2026-10-12T00:00:01+00:00'
STORED_AT = '2026-10-12T00:00:00.001000+00:00'


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'rebuild_write_triage_live_slates.py', mod_name='rebuild_write_triage_live_slates',
    )


@functools.cache
def _rounds() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'write_triage_rating_rounds.py', mod_name='write_triage_rating_rounds',
    )


@functools.cache
def _arms() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'run_write_triage_population_arms.py', 'run_write_triage_population_arms',
    )


def _unit(axis: int) -> np.ndarray:
    vector = np.zeros(DIMS, dtype=np.float32)
    vector[axis] = 1.0
    return vector


def _at_cosine(cosine: float, axis: int, toward: int = 0) -> np.ndarray:
    """A unit vector at *cosine* to ``_unit(toward)``, its remainder on *axis*."""
    return cosine * _unit(toward) + math.sqrt(1 - cosine ** 2) * _unit(axis)


def _payload(text: str, created_at: str | None, category: str = 'procedural_knowledge',
             **fields: Any) -> dict[str, Any]:
    return {'data': text, 'category': category, 'created_at': created_at, 'hash': 'h',
            'user_id': 'df', **fields}


Point = tuple[str, np.ndarray, dict[str, Any]]


def _table(points: list[Point]) -> Any:
    ids = [point_id for point_id, _, _ in points]
    return _mod().VectorTable(ids, np.stack([vector for _, vector, _ in points]),
                              [payload for _, _, payload in points])


def _journal(intent: str, *, at: str = WRITE_AT, source: str = 'mcp_tool',
             category: str | None = 'procedural_knowledge', memory_ids: tuple[str, ...] | None = None,
             metadata: dict[str, Any] | None = None, project: str = 'df') -> dict[str, Any]:
    """A duplicates/journal_add_memory.jsonl row."""
    return {
        'write_op_id': f'op-{intent}', 'project_id': project, 'agent_id': 'claude-test',
        'source': source, 'intent_id': intent, 'intent_created_at': at,
        'intent_category': category, 'intent_content': f'text of {intent}',
        'intent_metadata': {'category': category, **(metadata or {})},
        'result_summary': {'memory_ids': list((f'm-{intent}',) if memory_ids is None else memory_ids),
                           'stores': ['mem0']},
    }


def _attach(intent: str, outcome: str, parent: str, *, triage_sig: bool = True) -> dict[str, Any]:
    """A correctness/attaches_soak.json row."""
    return {'intent_id': intent, 'outcome': outcome, 'meta': {'parent_id': parent},
            'triage_sig': triage_sig, 'project': 'df'}


def _window(end: datetime | None = None) -> Any:
    return _mod().Window(start=datetime(2026, 10, 11, 13, 35, tzinfo=UTC), end=end)


def _writes(rows: list[dict[str, Any]], attaches: list[dict[str, Any]] | None = None,
            window: Any = None) -> list[Any]:
    return _mod().soak_writes(rows, attaches or [], window or _window())


def _history_db(path: Path, rows: list[tuple[str, str | None, str, str, str, str]]) -> Path:
    """mem0's history table; each row is (memory_id, old, new, event, created_at, updated_at)."""
    connection = sqlite3.connect(path)
    connection.execute(
        'CREATE TABLE history (id TEXT PRIMARY KEY, memory_id TEXT, old_memory TEXT,'
        ' new_memory TEXT, event TEXT, created_at DATETIME, updated_at DATETIME,'
        ' is_deleted INTEGER, actor_id TEXT, role TEXT)'
    )
    for index, row in enumerate(rows):
        connection.execute('INSERT INTO history VALUES (?, ?, ?, ?, ?, ?, ?, 0, NULL, NULL)',
                           (f'h{index}', *row))
    connection.commit()
    connection.close()
    return path


class TestTheSoakConstants:
    def test_are_the_soak_configuration(self) -> None:
        rebuild = _mod()
        assert rebuild.SOAK_EMBEDDING_DIMS == 1536
        assert rebuild.SOAK_T_HIGH == 0.8868282526657318
        assert rebuild.SOAK_T_LOW == 0.5229672433064231
        assert rebuild.CANDIDATE_K == 20
        assert rebuild.SOAK_WINDOW.start == datetime(2026, 10, 11, 13, 35, tzinfo=UTC)
        assert rebuild.SOAK_WINDOW.end is None


class TestSoakWrites:
    def test_keeps_a_triageable_write_with_its_instant_and_production_outcome(self) -> None:
        [write] = _writes([_journal('i1')], [_attach('i1', 'amended', 'P')])
        assert (write.memory_id, write.project_id, write.category, write.created_at) == (
            'm-i1', 'df', 'procedural_knowledge', WRITE_AT)
        assert write.instant == datetime.fromisoformat(WRITE_AT)
        assert write.content == 'text of i1'
        assert write.metadata == {'category': 'procedural_knowledge'}
        assert (write.production_outcome, write.production_target) == ('amended', 'P')

    def test_a_write_without_an_attach_record_was_stored(self) -> None:
        [write] = _writes([_journal('i1')])
        assert (write.production_outcome, write.production_target) == ('stored', None)

    @pytest.mark.parametrize('row', [
        pytest.param(_journal('x', at='2026-10-11T13:34:59+00:00'), id='before-the-window'),
        pytest.param(_journal('x', source='targeted_recon'), id='a-recon-source'),
        pytest.param(_journal('x', source='stage2_task_count_snapshot_write'), id='a-snapshot-source'),
        pytest.param(_journal('x', category='decisions_and_rationale'), id='graphiti-primary'),
        pytest.param(_journal('x', category=None), id='no-category'),
        pytest.param({**_journal('x'), 'intent_id': None}, id='no-intent'),
        pytest.param(_journal('x', memory_ids=()), id='no-memory-id'),
        pytest.param(_journal('x', memory_ids=('m1', 'm2')), id='two-memory-ids'),
        pytest.param(_journal('x', metadata={'kind': 'sighting', 'parent_id': 'P'}),
                     id='declares-attach-keys'),
    ])
    def test_drops_a_write_triage_never_decided(self, row: dict[str, Any]) -> None:
        assert _writes([row]) == []

    def test_keeps_a_full_recon_write(self) -> None:
        assert [w.memory_id for w in _writes([_journal('x', source='full_recon')])] == ['m-x']

    def test_keeps_a_triage_attach_whose_journal_carries_the_keys_triage_wrote(self) -> None:
        journaled = _journal('i1', metadata={'kind': 'sighting', 'parent_id': 'P', 'topic': 't'})
        [write] = _writes([journaled], [_attach('i1', 'restated', 'P')])
        assert write.metadata == {'category': 'procedural_knowledge', 'topic': 't'}
        assert (write.production_outcome, write.production_target) == ('restated', 'P')

    def test_drops_a_caller_owned_attach(self) -> None:
        assert _writes([_journal('x')], [_attach('x', 'amended', 'P', triage_sig=False)]) == []

    def test_drops_a_write_after_the_window_end(self) -> None:
        end = datetime(2026, 10, 12, tzinfo=UTC)
        rows = [_journal('in', at='2026-10-11T23:00:00+00:00'), _journal('out', at=WRITE_AT)]
        assert [w.memory_id for w in _writes(rows, window=_window(end))] == ['m-in']


def _dump(directory: Path, project: str, points: list[Point]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / f'{project}.npy', np.stack([vector for _, vector, _ in points]))
    (directory / f'{project}.jsonl').write_text(''.join(
        json.dumps({'id': point_id, 'payload': payload}) + '\n' for point_id, _, payload in points))
    return directory


class _Scroll:
    """A fake Qdrant HTTP POST: answers points/scroll in pages of two."""

    def __init__(self, points: list[Point]) -> None:
        self.points = points
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((url, body))
        start = body.get('offset') or 0
        page = self.points[start:start + 2]
        following = start + 2 if start + 2 < len(self.points) else None
        return {'result': {
            'points': [{'id': point_id, 'vector': vector.tolist(), 'payload': payload}
                       for point_id, vector, payload in page],
            'next_page_offset': following,
        }}


_POINTS: list[Point] = [
    ('a', _unit(1), _payload('text of a', EARLIER)),
    ('b', _unit(2), _payload('text of b', EARLIER)),
    ('c', _unit(3), _payload('text of c', EARLIER)),
]


class TestTheVectorTable:
    def test_load_dump_reads_a_projects_vectors_and_payloads(self, tmp_path: Path) -> None:
        table = _mod().load_dump(_dump(tmp_path / 'dump', 'df', _POINTS), 'df')
        assert table.ids == ('a', 'b', 'c')
        assert table.payload('b') == _payload('text of b', EARLIER)
        assert table.payload('missing') is None
        np.testing.assert_allclose(table.vector('c'), _unit(3))
        assert table.vector('missing') is None

    def test_load_qdrant_scrolls_every_page_read_only(self) -> None:
        scroll = _Scroll(_POINTS)
        table = _mod().load_qdrant('http://qdrant:6333', 'fused_df', post=scroll)
        assert table.ids == ('a', 'b', 'c')
        assert table.payload('c') == _payload('text of c', EARLIER)
        assert [url for url, _ in scroll.calls] == [
            'http://qdrant:6333/collections/fused_df/points/scroll'] * 2
        assert all(body['with_vector'] is True and body['with_payload'] is True
                   for _, body in scroll.calls)

    @pytest.mark.parametrize('width', [768, 3072])
    def test_a_width_other_than_the_soaks_is_refused_by_both_loaders(
        self, tmp_path: Path, width: int,
    ) -> None:
        points = [(point_id, np.ones(width, dtype=np.float32), payload)
                  for point_id, _, payload in _POINTS]
        with pytest.raises(ValueError, match=str(width)):
            _mod().load_dump(_dump(tmp_path / 'dump', 'df', points), 'df')
        with pytest.raises(ValueError, match=str(width)):
            _mod().load_qdrant('http://qdrant:6333', 'fused_df', post=_Scroll(points))


def _slate_points() -> list[Point]:
    """The write m-i1 on axis 0, and neighbours at chosen cosines around it."""
    return [
        ('m-i1', _unit(0), _payload('text of i1', STORED_AT)),
        ('a', _at_cosine(0.80, 1), _payload('a now', EARLIER)),
        ('b', _at_cosine(0.75, 2), _payload('z' * 5000, EARLIER, 'observations_and_summaries')),
        ('late', _at_cosine(0.95, 3), _payload('late', LATER)),
        ('same', _at_cosine(0.94, 4), _payload('same', WRITE_AT)),
        ('graph', _at_cosine(0.93, 5), _payload('graph', EARLIER, 'decisions_and_rationale')),
        ('plan', _at_cosine(0.92, 6), _payload('plan', EARLIER, planned=True)),
        ('naive', _at_cosine(0.91, 7), _payload('naive', '2026-10-11T20:00:00')),
        ('undated', _at_cosine(0.90, 8), _payload('undated', None)),
        ('offset', _at_cosine(0.70, 9), _payload('offset', '2026-10-12T01:00:00+02:00')),
        ('kid', _at_cosine(0.65, 10), _payload('kid', EARLIER, kind='sighting', parent_id='P')),
        ('P', _at_cosine(0.30, 11), _payload('P now', EARLIER)),
    ]


@pytest.fixture
def history(tmp_path: Path) -> Any:
    """'a' and 'P' were edited after the write; 'b' was edited before it."""
    path = _history_db(tmp_path / 'history.db', [
        ('a', 'a then', 'a now', 'UPDATE', EARLIER, '2026-10-13T00:00:00+00:00'),
        ('P', 'P then', 'P now', 'UPDATE', EARLIER, '2026-10-14T00:00:00+00:00'),
        ('b', 'b older', 'z' * 5000, 'UPDATE', EARLIER, '2026-10-11T21:00:00+00:00'),
    ])
    return _rounds().MemoryHistory.read(path)


def _slate(history: Any, k: int = 20, points: list[Point] | None = None) -> dict[str, Any]:
    [write] = _writes([_journal('i1')], [_attach('i1', 'amended', 'P')])
    return _mod().write_time_slate(write, _table(points or _slate_points()), history, k=k)


class TestTheWriteTimeSlate:
    def test_is_the_nearest_earlier_mem0_primary_unplanned_records(self, history: Any) -> None:
        assert [c['memory_id'] for c in _slate(history)['candidates']] == [
            'a', 'b', 'offset', 'kid', 'P',
        ]

    def test_holds_at_most_k_candidates(self, history: Any) -> None:
        assert [c['memory_id'] for c in _slate(history, k=3)['candidates']] == ['a', 'b', 'offset']

    def test_each_candidate_carries_its_cosine_canonical_and_write_time_text(
        self, history: Any,
    ) -> None:
        candidates = {c['memory_id']: c for c in _slate(history)['candidates']}
        assert candidates['a']['store_score'] == pytest.approx(0.80, abs=1e-6)
        assert candidates['a']['metadata']['store_score'] == candidates['a']['store_score']
        assert candidates['a']['created_at'] == EARLIER
        assert (candidates['a']['content'], candidates['a']['edited_after_write']) == ('a then', True)
        assert candidates['b']['content'] == _rounds().capped('z' * 5000)
        assert candidates['b']['edited_after_write'] is False
        assert candidates['b']['category'] == 'observations_and_summaries'
        assert candidates['kid']['canonical_id'] == 'P'
        assert candidates['a']['canonical_id'] == 'a'
        assert 'data' not in candidates['a']['metadata']

    def test_the_band_is_re_decided_at_the_soak_edges(self, history: Any) -> None:
        slate = _slate(history)
        assert (slate['band'], slate['band_winner_id']) == ('judge', 'a')
        assert slate['similarity'] == pytest.approx(0.80, abs=1e-6)
        restated = [*_slate_points(), ('twin', _at_cosine(0.89, 12), _payload('twin', EARLIER))]
        assert _slate(history, points=restated)['band'] == 'restated'
        alone = [_slate_points()[0], ('far', _at_cosine(0.50, 12), _payload('far', EARLIER))]
        assert _slate(history, points=alone)['band'] == 'stored'

    def test_a_hoisted_canonical_off_the_slate_gets_its_write_time_text(self, history: Any) -> None:
        slate = _slate(history, k=4)
        assert 'P' not in [c['memory_id'] for c in slate['candidates']]
        assert slate['targets'] == {'P': {'content': 'P then', 'created_at': EARLIER,
                                          'edited_after_write': True}}
        assert _arms().target_text(slate['targets'], slate, 'P') == 'P then'

    def test_a_hoisted_canonical_missing_from_the_table_has_no_text(self, history: Any) -> None:
        points = [point for point in _slate_points() if point[0] != 'P']
        assert _slate(history, k=4, points=points)['targets'] == {
            'P': {'content': None, 'created_at': None, 'edited_after_write': True},
        }

    def test_the_row_is_snapshot_shaped_so_the_arms_selector_reads_it(self, history: Any) -> None:
        slate = _slate(history)
        assert {'memory_id', 'project_id', 'category', 'created_at', 'content', 'metadata', 'band',
                'band_winner_id', 'similarity', 'candidates', 'targets', 'production_outcome',
                'production_target'} <= set(slate)
        assert (slate['memory_id'], slate['production_outcome'], slate['production_target']) == (
            'm-i1', 'amended', 'P')
        assert _arms().top_targets(slate, 5) == ['a', 'b', 'offset', 'P']
        assert _arms().top_targets(slate, 2) == ['a', 'b']

    def test_a_write_whose_own_vector_is_absent_is_unrecoverable(self, history: Any) -> None:
        points = [point for point in _slate_points() if point[0] != 'm-i1']
        with pytest.raises(_mod().Unrecoverable, match='m-i1'):
            _slate(history, points=points)


POPULATIONS = {'a': [f'a{i:02d}' for i in range(10)], 'b': ['b0', 'b1', 'b2'],
               'c': [f'c{i:02d}' for i in range(10)]}


class TestDrawLiveSample:
    def _draw(self, total: int, seed: int = 7) -> Any:
        return _mod().draw_live_sample(POPULATIONS, total=total, seed=seed)

    def test_allocates_equally_capped_at_each_population_and_redistributes(self) -> None:
        sample = self._draw(15)
        assert sample.allocation == {'a': 6, 'b': 3, 'c': 6}
        assert sample.strata == {'a': 10, 'b': 3, 'c': 10}
        assert len(sample.drawn) == len(set(sample.drawn)) == 15
        for project, count in sample.allocation.items():
            assert sum(drawn in POPULATIONS[project] for drawn in sample.drawn) == count

    def test_the_draw_is_seeded_and_deterministic(self) -> None:
        assert self._draw(15).drawn == self._draw(15).drawn
        assert self._draw(15).drawn != self._draw(15, seed=8).drawn
        shuffled = {project: list(reversed(ids)) for project, ids in POPULATIONS.items()}
        assert _mod().draw_live_sample(shuffled, total=15, seed=7).drawn == self._draw(15).drawn

    def test_a_larger_total_extends_a_smaller_one(self) -> None:
        assert self._draw(16).drawn[:15] == self._draw(15).drawn
        assert self._draw(23).drawn[:16] == self._draw(16).drawn

    def test_a_total_beyond_the_populations_is_refused(self) -> None:
        with pytest.raises(ValueError, match='24'):
            self._draw(24)


def _judge_band_point(index: int) -> Point:
    """A write near the shared neighbour 'n' (cosine 0.8): judge band."""
    vector = (0.8 * _unit(0) + 0.6 * _unit(100 + index)).astype(np.float32)
    return (f'm-j{index}', vector, _payload(f'text of j{index}', STORED_AT))


def _rebuild_inputs() -> tuple[list[Any], dict[str, Any]]:
    points = [('n', _unit(0), _payload('the neighbour', EARLIER))]
    points += [_judge_band_point(index) for index in range(4)]
    points.append(('m-left', _at_cosine(0.2, 300), _payload('text of left', STORED_AT)))
    points.append(('m-quiet', _at_cosine(0.2, 301), _payload('text of quiet', STORED_AT)))
    rows = [_journal(f'j{index}') for index in range(4)]
    rows += [_journal('left'), _journal('quiet'), _journal('gone')]
    writes = _writes(rows, [_attach('left', 'amended', 'n')])
    return writes, {'df': _table(points)}


class TestRebuild:
    def test_draws_only_judge_band_writes_and_records_leavers_and_unrecoverables(
        self, history: Any,
    ) -> None:
        writes, tables = _rebuild_inputs()
        rebuilt = _mod().rebuild(writes, tables, history, total=3, seed=7)
        assert rebuilt.strata == {'df': 4}
        assert rebuilt.allocation == {'df': 3}
        assert len(rebuilt.rows) == 3
        assert {row['memory_id'] for row in rebuilt.rows} < {f'm-j{index}' for index in range(4)}
        assert all(row['band'] == 'judge' and row['band_winner_id'] == 'n' for row in rebuilt.rows)
        assert rebuilt.leavers == [{'memory_id': 'm-left', 'project_id': 'df',
                                    'production_outcome': 'amended', 'band': 'stored'}]
        [unrecoverable] = rebuilt.unrecoverable
        assert (unrecoverable['memory_id'], unrecoverable['project_id']) == ('m-gone', 'df')
        assert 'm-gone' in unrecoverable['reason']

    def test_rows_follow_the_draw_order(self, history: Any) -> None:
        writes, tables = _rebuild_inputs()
        rebuilt = _mod().rebuild(writes, tables, history, total=4, seed=7)
        sample = _mod().draw_live_sample({'df': [f'm-j{index}' for index in range(4)]},
                                         total=4, seed=7)
        assert [row['memory_id'] for row in rebuilt.rows] == list(sample.drawn)


class TestTheCli:
    def _inputs(self, tmp_path: Path) -> list[str]:
        bundle = tmp_path / 'bundle'
        (bundle / 'duplicates').mkdir(parents=True)
        (bundle / 'correctness').mkdir()
        rows = [_journal(f'j{index}') for index in range(4)] + [_journal('left')]
        journal = bundle / 'duplicates' / 'journal_add_memory.jsonl'
        journal.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        attaches = bundle / 'correctness' / 'attaches_soak.json'
        attaches.write_text(json.dumps([_attach('left', 'amended', 'n')]))
        points = [('n', _unit(0), _payload('the neighbour', EARLIER))]
        points += [_judge_band_point(index) for index in range(4)]
        points.append(('m-left', _at_cosine(0.2, 300), _payload('text of left', STORED_AT)))
        dump = _dump(tmp_path / 'dump', 'df', points)
        history = _history_db(tmp_path / 'history.db', [])
        return ['--journal', str(journal), '--attaches', str(attaches),
                '--history-db', str(history), '--vectors', f'dump:{dump}',
                '--projects', 'df', '--total', '3', '--seed', '7']

    def test_writes_a_header_line_then_one_line_per_sampled_write(self, tmp_path: Path) -> None:
        out = tmp_path / 'live_slates.jsonl'
        assert _mod().main([*self._inputs(tmp_path), '--out', str(out)]) == 0
        header, *rows = [json.loads(line) for line in out.read_text().splitlines()]
        assert {'built_at', 'source', 't_high', 't_low', 'candidate_k', 'seed', 'strata',
                'allocation', 'leavers', 'unrecoverable', 'edited_candidates',
                'deleted_candidates'} <= set(header)
        assert header['source']['kind'] == 'dump'
        assert set(header['source']['sha256']) == {'df.npy', 'df.jsonl'}
        assert (header['t_high'], header['t_low'], header['candidate_k'], header['seed']) == (
            0.8868282526657318, 0.5229672433064231, 20, 7)
        assert header['strata'] == {'df': 4}
        assert header['allocation'] == {'df': 3}
        assert [leaver['memory_id'] for leaver in header['leavers']] == ['m-left']
        assert len(rows) == 3
        assert all(_arms().top_targets(row, 5) == ['n'] for row in rows)

    def test_refuses_to_write_inside_the_read_only_bundle(self, tmp_path: Path) -> None:
        inputs = self._inputs(tmp_path)
        out = tmp_path / 'bundle' / 'live_slates.jsonl'
        with pytest.raises(SystemExit):
            _mod().main([*inputs, '--out', str(out)])
        assert not out.exists()
