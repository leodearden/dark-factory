"""Tests for freeze_write_triage_population.py — the frozen π population (task 6151).

Driven through the script's public functions against a LOCAL store double: the
three reads the freeze makes (``mem0.scroll_all_by_metadata``, ``search`` and
``get_memory_by_id``) and nothing else. No Qdrant, no network.
"""
from __future__ import annotations

import functools
import hashlib
import json
import types
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from _fm_helpers import load_script_module

from fused_memory.models.enums import MEM0_PRIMARY, SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.reconciliation.prompts import (
    FLAG_FOR_STAGE2_MARKER_KIND,
    STAGE2_SUPPRESS_GUARD_KIND,
)
from fused_memory.server.write_triage import decide_band, declares_attach_keys
from fused_memory.services.memory_service import SearchResults

SCRIPTS = Path(__file__).parent.parent / 'scripts'

SINCE = datetime(2026, 9, 29, tzinfo=UTC)
FROZEN_AT = datetime(2026, 10, 5, 12, tzinfo=UTC)
CATEGORIES = sorted(c.value for c in MEM0_PRIMARY)


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'freeze_write_triage_population.py', 'freeze_write_triage_population',
    )


class _Mem0:
    """``Mem0Backend.scroll_all_by_metadata``: a plain call returning an async stream."""

    def __init__(self, scrolled: dict[tuple[str, str], list[dict]]) -> None:
        self._scrolled = scrolled
        self.calls: list[tuple[str, dict]] = []

    def scroll_all_by_metadata(self, scope: Any, filters: dict) -> AsyncIterator[dict]:
        self.calls.append((scope.project_id, dict(filters)))
        rows = self._scrolled.get((scope.project_id, filters['category']), [])

        async def _stream() -> AsyncIterator[dict]:
            for row in rows:
                yield row

        return _stream()


class _Store:
    """The memory service the freeze reads through."""

    def __init__(
        self,
        scrolled: dict[tuple[str, str], list[dict]] | None = None,
        by_id: dict[tuple[str, str], dict] | None = None,
    ) -> None:
        self.mem0 = _Mem0(scrolled or {})
        self._by_id = by_id or {}
        self.search = AsyncMock()

    async def get_memory_by_id(self, project_id: str, memory_id: str) -> dict | None:
        return self._by_id.get((project_id, memory_id))


def _scrolled(memory_id: str, created_at: str | None, category: str) -> dict:
    return {'id': memory_id, 'created_at': created_at, 'metadata': {'category': category}}


def _stored(memory_id: str, **metadata: Any) -> dict:
    return {'id': memory_id, 'content': f'content of {memory_id}', 'metadata': metadata}


def _enumerate(store: _Store, projects=('dark_factory', 'reify')):
    import asyncio  # noqa: PLC0415

    return asyncio.run(_mod().enumerate_population(
        store, projects=projects, since=SINCE, frozen_at=FROZEN_AT,
    ))


def _one_project_store(rows: list[tuple[str, str | None, dict]]) -> _Store:
    """``procedural_knowledge`` rows in dark_factory, each live with *metadata*."""
    return _Store(
        scrolled={('dark_factory', 'procedural_knowledge'): [
            _scrolled(memory_id, created_at, 'procedural_knowledge')
            for memory_id, created_at, _ in rows
        ]},
        by_id={
            ('dark_factory', memory_id): _stored(memory_id, **metadata)
            for memory_id, _, metadata in rows
        },
    )


def _ids(writes: list[dict]) -> list[str]:
    return [write['memory_id'] for write in writes]


class TestEnumeration:
    def test_every_project_scrolls_every_mem0_primary_category(self) -> None:
        store = _Store()
        _enumerate(store)
        assert store.mem0.calls == [
            (project, {'category': category})
            for project in ('dark_factory', 'reify')
            for category in CATEGORIES
        ]

    def test_the_window_is_compared_tz_aware_and_inclusive_of_since(self) -> None:
        store = _one_project_store([
            ('pacific-evening', '2026-09-28T18:30:00-07:00', {}),
            ('utc-before', '2026-09-28T23:59:59+00:00', {}),
            ('at-since', '2026-09-29T00:00:00+00:00', {}),
            ('after-frozen', '2026-10-05T12:00:01+00:00', {}),
            ('at-frozen', '2026-10-05T12:00:00+00:00', {}),
        ])
        writes, _ = _enumerate(store)
        assert set(_ids(writes)) == {'pacific-evening', 'at-since', 'at-frozen'}

    @pytest.mark.parametrize('created_at', [
        '2026-09-30T10:00:00', None, 'not-a-date', '',
    ])
    def test_an_undated_record_is_excluded_and_counted_never_guessed(
        self, created_at: str | None,
    ) -> None:
        store = _one_project_store([
            ('undated', created_at, {}),
            ('dated', '2026-09-30T10:00:00+00:00', {}),
        ])
        writes, excluded = _enumerate(store)
        assert _ids(writes) == ['dated']
        assert excluded['undated'] == 1

    def test_a_record_that_vanished_before_its_read_is_excluded_and_counted(self) -> None:
        store = _Store(
            scrolled={('reify', 'observations_and_summaries'): [
                _scrolled('gone', '2026-09-30T10:00:00+00:00', 'observations_and_summaries'),
                _scrolled('here', '2026-09-30T11:00:00+00:00', 'observations_and_summaries'),
            ]},
            by_id={('reify', 'here'): _stored('here')},
        )
        writes, excluded = _enumerate(store)
        assert _ids(writes) == ['here']
        assert excluded['vanished'] == 1

    def test_content_and_metadata_come_from_the_point_read(self) -> None:
        store = _one_project_store([
            ('m', '2026-09-30T10:00:00+00:00', {'category': 'procedural_knowledge', 'topic': 't'}),
        ])
        [write] = _enumerate(store)[0]
        assert write['content'] == 'content of m'
        assert write['metadata'] == {'category': 'procedural_knowledge', 'topic': 't'}

    @pytest.mark.parametrize(('metadata', 'marker'), [
        ({'flag_for_stage2': True}, True),
        ({'stage2_suppress': True}, True),
        ({'kind': FLAG_FOR_STAGE2_MARKER_KIND}, True),
        ({'kind': STAGE2_SUPPRESS_GUARD_KIND}, True),
        ({'flag_for_stage2': False}, False),
        ({'stage2_suppress': 'true'}, False),
        ({'kind': 'sighting'}, False),
        ({}, False),
    ])
    def test_a_recon_marker_is_recognised_by_flag_or_kind(
        self, metadata: dict, marker: bool,
    ) -> None:
        store = _one_project_store([('m', '2026-09-30T10:00:00+00:00', metadata)])
        [write] = _enumerate(store)[0]
        assert write['recon_marker'] is marker

    @pytest.mark.parametrize('metadata', [
        {}, {'kind': 'sighting'}, {'parent_id': 'p'}, {'x_contested': False}, {'topic': 't'},
    ])
    def test_declares_attach_keys_is_the_shipped_predicate(self, metadata: dict) -> None:
        store = _one_project_store([('m', '2026-09-30T10:00:00+00:00', metadata)])
        [write] = _enumerate(store)[0]
        assert write['declares_attach_keys'] is declares_attach_keys(metadata)

    def test_each_write_carries_its_identity(self) -> None:
        store = _Store(
            scrolled={('reify', 'preferences_and_norms'): [
                _scrolled('m', '2026-09-28T18:30:00-07:00', 'preferences_and_norms'),
            ]},
            by_id={('reify', 'm'): _stored('m')},
        )
        [write] = _enumerate(store)[0]
        assert (
            write['memory_id'], write['project_id'], write['category'], write['created_at'],
        ) == ('m', 'reify', 'preferences_and_norms', '2026-09-28T18:30:00-07:00')

    def test_the_order_is_project_then_instant_then_id(self) -> None:
        rows = {
            ('reify', 'procedural_knowledge'): [
                _scrolled('r1', '2026-09-30T00:00:00+00:00', 'procedural_knowledge'),
            ],
            ('dark_factory', 'procedural_knowledge'): [
                _scrolled('late', '2026-09-28T18:30:00-07:00', 'procedural_knowledge'),
                _scrolled('b-tie', '2026-09-30T00:00:00+00:00', 'procedural_knowledge'),
            ],
            ('dark_factory', 'observations_and_summaries'): [
                _scrolled('early', '2026-09-29T01:00:00+00:00', 'observations_and_summaries'),
                _scrolled('a-tie', '2026-09-30T00:00:00+00:00', 'observations_and_summaries'),
            ],
        }
        by_id = {
            (project, row['id']): _stored(row['id'])
            for (project, _), scrolled in rows.items() for row in scrolled
        }
        writes, _ = _enumerate(_Store(scrolled=rows, by_id=by_id))
        assert _ids(writes) == ['early', 'late', 'a-tie', 'b-tie', 'r1']


# ---------------------------------------------------------------------------
# Freeze and snapshot
# ---------------------------------------------------------------------------

T_HIGH = 0.9
T_LOW = 0.5
K = 4
_LATER = '2026-10-01T00:00:00+00:00'


def _retrieval() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'eval_write_triage_retrieval.py', 'eval_write_triage_retrieval',
    )


def _row(
    memory_id: str, cosine: float | None, created_at: str = _LATER, **metadata: Any,
) -> MemoryResult:
    """A store row as ``MemoryService.search`` returns it, scored for one query."""
    if cosine is not None:
        metadata['store_score'] = cosine
    return MemoryResult(
        id=memory_id, content=f'content of {memory_id}', source_store=SourceStore.mem0,
        metadata=metadata, created_at=created_at,
    )


def _write(memory_id: str, project_id: str = 'dark_factory') -> dict:
    return {
        'memory_id': memory_id, 'project_id': project_id,
        'category': 'procedural_knowledge', 'created_at': '2026-09-30T00:00:00+00:00',
        'content': f'content of {memory_id}', 'metadata': {},
        'recon_marker': False, 'declares_attach_keys': False,
    }


def _searching_store(
    rows_by_query: dict[str, SearchResults | list[MemoryResult]],
    by_id: dict[tuple[str, str], dict] | None = None,
) -> _Store:
    store = _Store(by_id=by_id)
    store.search = AsyncMock(side_effect=lambda **kwargs: (
        rows if isinstance(rows := rows_by_query[kwargs['query']], SearchResults)
        else SearchResults(rows)
    ))
    return store


def _freeze(store: _Store, writes: list[dict]) -> dict:
    import asyncio  # noqa: PLC0415

    return asyncio.run(_mod().freeze_population(
        store, writes, k=K, t_high=T_HIGH, t_low=T_LOW,
    ))


def _frozen_write(frozen: dict, memory_id: str) -> dict:
    [write] = [w for w in frozen['writes'] if w['memory_id'] == memory_id]
    return write


def _slate_ids(write: dict) -> list[str]:
    return [candidate['memory_id'] for candidate in write['candidates']]


class TestFreezePopulation:
    def test_one_production_retrieval_per_write_with_the_own_id_dropped(self) -> None:
        store = _searching_store({
            'content of w1': [_row('w1', 0.99), _row('x', 0.7)],
            'content of w2': [_row('y', 0.6)],
        })
        frozen = _freeze(store, [_write('w1'), _write('w2')])

        assert store.search.await_count == 2
        for call in store.search.await_args_list:
            assert call.kwargs['limit'] == K
            assert call.kwargs['stores'] == ['mem0']
            assert call.kwargs['anchor_topics'] is False
        w1 = _frozen_write(frozen, 'w1')
        assert _slate_ids(w1) == ['x']
        assert w1['self_retrieved'] is True
        assert _frozen_write(frozen, 'w2')['self_retrieved'] is False
        assert frozen['excluded']['self_retrieved'] == 1

    def test_a_retrieved_child_of_the_write_is_dropped_before_banding_and_counted(
        self,
    ) -> None:
        store = _searching_store({'content of w': [
            _row('own-sighting', 0.97, kind='sighting', parent_id='w'),
            _row('x', 0.6),
        ]})
        frozen = _freeze(store, [_write('w')])
        w = _frozen_write(frozen, 'w')
        assert _slate_ids(w) == ['x']
        assert (w['band'], w['band_winner_id'], w['similarity']) == ('judge', 'x', 0.6)
        assert w['own_children_dropped'] == 1
        assert frozen['excluded']['own_children_dropped'] == 1

    @pytest.mark.parametrize(('cosines', 'band'), [
        ((0.95, 0.7), 'restated'),
        ((0.9, 0.7), 'restated'),
        ((0.8, 0.7), 'judge'),
        ((0.5, 0.2), 'judge'),
        ((0.49, 0.2), 'stored'),
    ])
    def test_the_band_is_the_shipped_decide_band(
        self, cosines: tuple[float, float], band: str,
    ) -> None:
        rows = [_row('a', cosines[0]), _row('b', cosines[1])]
        frozen = _freeze(_searching_store({'content of w': rows}), [_write('w')])
        w = _frozen_write(frozen, 'w')
        decision = decide_band(rows, t_high=T_HIGH, t_low=T_LOW)
        assert w['band'] == decision.outcome == band
        assert w['band_winner_id'] == decision.canonical_id
        assert w['similarity'] == decision.similarity
        assert w['retrieved_count'] == 2

    def test_the_slate_keeps_every_comparable_row_up_to_k_in_cosine_order(self) -> None:
        rows = [
            _row('c3', 0.55, created_at='2026-09-29T08:00:00-07:00'),
            _row('pin', None),
            _row('c1', 0.8),
            _row('c5', 0.51),
            _row('c2', 0.6),
            _row('c4', 0.52),
        ]
        frozen = _freeze(_searching_store({'content of w': rows}), [_write('w')])
        w = _frozen_write(frozen, 'w')
        by_id = {row.id: row for row in rows}
        assert w['candidates'] == [
            {**_retrieval().normalize(by_id[i]), 'created_at': by_id[i].created_at}
            for i in ('c1', 'c2', 'c3', 'c4')
        ]

    def test_a_hoisted_target_off_the_slate_is_fetched_or_recorded_gone(self) -> None:
        store = _searching_store(
            {'content of w': [
                _row('amend', 0.8, kind='amendment', parent_id='parent-live'),
                _row('sight', 0.7, kind='sighting', parent_id='parent-gone'),
                _row('on-slate-child', 0.65, kind='sighting', parent_id='plain'),
                _row('plain', 0.6),
            ]},
            by_id={('reify', 'parent-live'): {
                'id': 'parent-live', 'content': 'the parent', 'metadata': {},
            }},
        )
        frozen = _freeze(store, [_write('w', project_id='reify')])
        assert frozen['targets'] == {
            'parent-live': {'project_id': 'reify', 'content': 'the parent'},
            'parent-gone': {'project_id': 'reify', 'content': None},
        }
        assert _frozen_write(frozen, 'w')['band_winner_id'] == 'parent-live'

    def test_a_degraded_retrieval_refuses_the_freeze_naming_the_write(self) -> None:
        store = _searching_store({
            'content of ok': [_row('x', 0.7)],
            'content of blip': SearchResults([], degraded=True, failed_stores=['mem0']),
        })
        with pytest.raises(ValueError, match='blip'):
            _freeze(store, [_write('ok'), _write('blip')])


def _snapshot(frozen_at: str = '2026-10-05T07:00:00+00:00') -> dict:
    return {
        'schema_version': 1, 'frozen_at': frozen_at, 'writes': [
            {'memory_id': 'm', 'content': 'café — naïve'},
        ], 'targets': {},
    }


class TestSnapshotFiles:
    def test_the_directory_is_named_for_the_utc_freeze_date(self, tmp_path: Path) -> None:
        path = _mod().write_snapshot(tmp_path, _snapshot('2026-10-04T20:00:00-07:00'))
        assert path == tmp_path / 'write-triage-population-2026-10-05' / 'snapshot.json'

    def test_the_bytes_are_deterministic_and_the_sidecar_is_their_digest(
        self, tmp_path: Path,
    ) -> None:
        snapshot = _snapshot()
        path = _mod().write_snapshot(tmp_path / 'a', snapshot)
        assert path.read_bytes() == json.dumps(
            snapshot, sort_keys=True, ensure_ascii=False,
        ).encode('utf-8')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert (path.parent / 'snapshot.sha256').read_text().strip() == digest

    def test_load_round_trips_with_the_same_digest(self, tmp_path: Path) -> None:
        path = _mod().write_snapshot(tmp_path, _snapshot())
        loaded, sha = _mod().load_snapshot(path)
        assert loaded == _snapshot()
        assert sha == hashlib.sha256(path.read_bytes()).hexdigest()

    def test_an_edited_snapshot_is_refused_naming_both_digests(self, tmp_path: Path) -> None:
        path = _mod().write_snapshot(tmp_path, _snapshot())
        recorded = (path.parent / 'snapshot.sha256').read_text().strip()
        path.write_bytes(path.read_bytes().replace(b'"m"', b'"n"'))
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        with pytest.raises(ValueError, match=recorded) as excinfo:
            _mod().load_snapshot(path)
        assert actual in str(excinfo.value)

    def test_a_frozen_population_is_never_re_frozen_in_place(self, tmp_path: Path) -> None:
        _mod().write_snapshot(tmp_path, _snapshot())
        with pytest.raises(FileExistsError):
            _mod().write_snapshot(tmp_path, _snapshot('2026-10-05T23:00:00+00:00'))


class TestFreezeSnapshot:
    def test_the_top_level_records_what_was_frozen_and_how(self) -> None:
        import asyncio  # noqa: PLC0415

        store = _searching_store(
            {'content of m': [_row('x', 0.7)]},
            by_id={('reify', 'm'): _stored('m')},
        )
        store.mem0 = _Mem0({('reify', 'procedural_knowledge'): [
            _scrolled('m', '2026-09-30T10:00:00+00:00', 'procedural_knowledge'),
            _scrolled('undated', None, 'procedural_knowledge'),
        ]})
        snapshot = asyncio.run(_mod().freeze_snapshot(
            store, projects=('reify',), since=SINCE, frozen_at=FROZEN_AT,
            k=K, t_high=T_HIGH, t_low=T_LOW,
        ))
        assert {key: snapshot[key] for key in (
            'schema_version', 'frozen_at', 'since', 'projects', 'categories',
            'candidate_k', 't_high', 't_low',
        )} == {
            'schema_version': 1,
            'frozen_at': '2026-10-05T12:00:00+00:00',
            'since': '2026-09-29T00:00:00+00:00',
            'projects': ['reify'],
            'categories': CATEGORIES,
            'candidate_k': K, 't_high': T_HIGH, 't_low': T_LOW,
        }
        assert snapshot['excluded'] == {
            'undated': 1, 'vanished': 0, 'own_children_dropped': 0, 'self_retrieved': 0,
        }
        assert _slate_ids(_frozen_write(snapshot, 'm')) == ['x']
        assert snapshot['targets'] == {}
