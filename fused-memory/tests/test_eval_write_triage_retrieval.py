"""Tests for eval_write_triage_retrieval.py — the judge eval's live-store edge.

The module's one claim is that a retrieved slate is the one production would
build. So every test drives the SHIPPED retrieval, banding and trim through a
fake MemoryService, and where production's own shape is the expectation it is
obtained from the shipped function rather than restated as a literal.
"""
from __future__ import annotations

import functools
import logging
import types
from collections.abc import Iterable
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _fm_helpers import load_script_module

from fused_memory.models.enums import MemoryCategory, SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.server.grouped_read import PARENT_ID_KEY, SIGHTING_KIND
from fused_memory.server.write_triage import (
    OUTCOME_JUDGE,
    OUTCOME_RESTATED,
    OUTCOME_STORED,
    decide_band,
    retrieve_candidates,
)
from fused_memory.server.write_triage_judge import select_judge_candidates
from fused_memory.services.memory_service import SearchResults

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'eval_write_triage_retrieval.py'

T_HIGH = 0.9
T_LOW = 0.5


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(SCRIPT_PATH, 'eval_write_triage_retrieval')


def _record(memory_id: str, cluster_id: str) -> dict:
    return {
        'memory_id': memory_id, 'cluster_id': cluster_id, 'label': 'duplicate',
        'content': f'the text of {memory_id}',
    }


def _row(memory_id: str, cosine: float, **metadata) -> MemoryResult:
    """A post-RRF search row: the cosine lives in ``metadata['store_score']``."""
    return MemoryResult(
        id=memory_id, content=f'the text of {memory_id}',
        category=MemoryCategory.procedural_knowledge, source_store=SourceStore.mem0,
        metadata={'store_score': cosine, **metadata},
    )


def _child(memory_id: str, cosine: float, parent_id: str) -> MemoryResult:
    return _row(memory_id, cosine, kind=SIGHTING_KIND, **{PARENT_ID_KEY: parent_id})


class _FakeMemoryService:
    """The two MemoryService reads the edge makes; ids in ``live`` resolve."""

    def __init__(
        self,
        rows: Iterable[MemoryResult] = (),
        *,
        live: Iterable[str] = (),
        degraded: bool = False,
    ) -> None:
        self.search = AsyncMock(return_value=SearchResults(list(rows), degraded=degraded))
        self.live = frozenset(live)
        self.probed: list[str] = []

    async def get_memory_by_id(self, project_id: str, memory_id: str) -> dict | None:
        self.probed.append(memory_id)
        if memory_id not in self.live:
            return None
        return {'id': memory_id, 'content': '', 'metadata': {}}


async def _prefetch(service: _FakeMemoryService, records: list[dict], k: int = 5) -> dict:
    return await _mod().prefetch_retrievals(service, records, project_id='reify', k=k)


def _slates(records: list[dict], retrievals: dict, count: int = 3) -> list:
    return _mod().retrieved_slates(
        records, retrievals, t_high=T_HIGH, t_low=T_LOW, judge_candidate_count=count,
    )


class TestPrefetchRetrievals:
    @pytest.mark.asyncio
    async def test_searches_exactly_as_the_shipped_retrieve_candidates_does(self) -> None:
        record = _record('d1', 'c1')
        ours = _FakeMemoryService(live={'c1'})
        await _prefetch(ours, [record], k=7)
        shipped = _FakeMemoryService()
        await retrieve_candidates(shipped, record['content'], 'reify', 7)
        assert ours.search.await_args_list == shipped.search.await_args_list

    @pytest.mark.asyncio
    async def test_the_records_own_id_is_dropped_and_reported(self) -> None:
        records = [_record('d1', 'c1'), _record('d2', 'c1')]
        service = _FakeMemoryService([_row('d1', 0.99), _row('c1', 0.8)], live={'c1'})
        got = await _prefetch(service, records)
        assert [row.id for row in got['d1']['results']] == ['c1']
        assert got['d1']['self_retrieved'] is True
        assert [row.id for row in got['d2']['results']] == ['d1', 'c1']
        assert got['d2']['self_retrieved'] is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize('degraded', [True, False])
    async def test_degraded_is_read_off_the_search_results(self, degraded: bool) -> None:
        service = _FakeMemoryService([_row('c1', 0.8)], live={'c1'}, degraded=degraded)
        got = await _prefetch(service, [_record('d1', 'c1')])
        assert got['d1']['degraded'] is degraded

    @pytest.mark.asyncio
    async def test_each_clusters_canonical_is_probed_once(self) -> None:
        records = [_record('d1', 'c1'), _record('d2', 'c1'), _record('d3', 'c2')]
        service = _FakeMemoryService(live={'c1', 'c2'})
        await _prefetch(service, records)
        assert sorted(service.probed) == ['c1', 'c2']

    @pytest.mark.asyncio
    async def test_a_missing_canonical_is_recorded_absent(self) -> None:
        records = [_record('d1', 'c1'), _record('d2', 'gone')]
        got = await _prefetch(_FakeMemoryService(live={'c1'}), records)
        assert got['d1']['canonical_present'] is True
        assert got['d2']['canonical_present'] is False


class TestRetrievedSlates:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(('top_cosine', 'band', 'attach_target_id'), [
        (0.95, OUTCOME_RESTATED, 'c1'),
        (0.7, OUTCOME_JUDGE, 'c1'),
        (0.3, OUTCOME_STORED, None),
    ])
    async def test_the_band_is_decide_bands_at_the_given_edges(
        self, top_cosine: float, band: str, attach_target_id: str | None,
    ) -> None:
        record = _record('d1', 'c1')
        rows = [_row('c1', top_cosine), _row('x', top_cosine - 0.1)]
        [slate] = _slates([record], await _prefetch(_FakeMemoryService(rows), [record]))
        assert slate.band == band
        assert slate.similarity == (top_cosine if band != OUTCOME_STORED else None)
        assert slate.attach_target_id == attach_target_id

    @pytest.mark.asyncio
    async def test_the_slate_is_the_shipped_trim_in_prompt_order(self) -> None:
        record = _record('d1', 'c1')
        rows = [_row('a', 0.6), _row('b', 0.8), _row('c', 0.7), _row('d', 0.55), _row('e', 0.65)]
        [slate] = _slates([record], await _prefetch(_FakeMemoryService(rows), [record]), count=3)
        decision = decide_band(rows, t_high=T_HIGH, t_low=T_LOW)
        shipped = select_judge_candidates(rows, 3, canonical_id=decision.canonical_id)
        assert [c['memory_id'] for c in slate.candidates] == [row.id for row in shipped]
        assert len(slate.candidates) == 3
        assert slate.retrieved_count == len(rows)

    @pytest.mark.asyncio
    async def test_a_child_winner_attaches_to_its_parent(self) -> None:
        record = _record('d1', 'parent')
        rows = [_child('child', 0.95, 'parent'), _row('x', 0.6)]
        [slate] = _slates([record], await _prefetch(_FakeMemoryService(rows), [record]))
        assert slate.band == OUTCOME_RESTATED
        assert slate.attach_target_id == 'parent'

    @pytest.mark.asyncio
    async def test_a_degraded_retrieval_is_warned_about_by_memory_id(self, caplog) -> None:
        degraded, healthy = _record('d-degraded', 'c1'), _record('d-healthy', 'c1')
        retrievals = {
            **await _prefetch(_FakeMemoryService(live={'c1'}, degraded=True), [degraded]),
            **await _prefetch(_FakeMemoryService(live={'c1'}), [healthy]),
        }
        with caplog.at_level(logging.WARNING):
            _slates([degraded, healthy], retrievals)
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any('d-degraded' in message for message in warnings)
        assert not any('d-healthy' in message for message in warnings)

    @pytest.mark.asyncio
    async def test_a_healthy_run_warns_about_nothing(self, caplog) -> None:
        record = _record('d1', 'c1')
        retrievals = await _prefetch(_FakeMemoryService([_row('c1', 0.7)], live={'c1'}), [record])
        with caplog.at_level(logging.WARNING):
            _slates([record], retrievals)
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


class TestNormalize:
    def test_metadata_is_kept_verbatim(self) -> None:
        row = _child('child', 0.8, 'parent')
        row.metadata['topic'] = 'some-topic'
        got = _mod().normalize(row)
        assert got['metadata'] == row.metadata
        assert got['store_score'] == 0.8
        assert (got['memory_id'], got['content']) == (row.id, row.content)

    def test_a_child_is_hoisted_to_its_parent(self) -> None:
        assert _mod().normalize(_child('child', 0.8, 'parent'))['canonical_id'] == 'parent'

    def test_a_non_child_is_its_own_canonical(self) -> None:
        assert _mod().normalize(_row('plain', 0.8))['canonical_id'] == 'plain'
