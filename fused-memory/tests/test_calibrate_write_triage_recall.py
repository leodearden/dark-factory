"""Tests for the recall side of calibrate_write_triage.py.

Recall measures the SHIPPED retrieval, so the ``production`` mode is pinned in
lockstep with ``write_triage.retrieve_candidates`` through a fake
MemoryService, never against a restated literal. ``legacy`` is the historical
search shape, and a literal is exactly what pins a historical shape.
"""
from __future__ import annotations

import functools
import logging
import types
from collections.abc import Iterable
from pathlib import Path
from unittest.mock import AsyncMock, call

import pytest
from _fm_helpers import load_script_module

from fused_memory.models.enums import MemoryCategory, SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.server.grouped_read import AMENDMENT_KIND, PARENT_ID_KEY, SIGHTING_KIND
from fused_memory.server.write_triage import retrieve_candidates
from fused_memory.services.memory_service import SearchResults

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'calibrate_write_triage.py'
ALIASES_PATH = (
    Path(__file__).parent / 'fixtures' / 'write_triage_calibration.canonical_aliases.json'
)
MODES = ('legacy', 'production')


@functools.cache
def _calib() -> types.ModuleType:
    return load_script_module(SCRIPT_PATH, mod_name='calibrate_write_triage')


def _record(memory_id: str, cluster_id: str, label: str = 'duplicate') -> dict:
    return {
        'memory_id': memory_id, 'cluster_id': cluster_id, 'label': label,
        'content': f'the text of {memory_id}', 'category': 'procedural_knowledge',
    }


def _row(memory_id: str, **metadata) -> MemoryResult:
    return MemoryResult(
        id=memory_id, content=f'the text of {memory_id}',
        category=MemoryCategory.procedural_knowledge, source_store=SourceStore.mem0,
        metadata={'store_score': 0.5, **metadata},
    )


class _FakeMemoryService:
    """The two MemoryService reads recall makes; ids in ``live`` resolve."""

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


async def _hit(service: _FakeMemoryService, record: dict, mode: str, k: int = 20) -> dict:
    return await _calib().fetch_recall_hit(
        service, record, project_id='reify', k=k, retrieval_mode=mode,
    )


class TestFetchRecallHit:
    @pytest.mark.asyncio
    async def test_production_searches_exactly_as_retrieve_candidates_does(self) -> None:
        record = _record('d1', 'c1')
        ours = _FakeMemoryService(live={'c1'})
        await _hit(ours, record, 'production', k=20)
        shipped = _FakeMemoryService()
        await retrieve_candidates(shipped, record['content'], 'reify', 20)
        assert ours.search.await_args_list == shipped.search.await_args_list

    @pytest.mark.asyncio
    async def test_legacy_keeps_the_historical_search_shape(self) -> None:
        record = _record('d1', 'c1')
        service = _FakeMemoryService(live={'c1'})
        await _hit(service, record, 'legacy', k=20)
        assert service.search.await_args_list == [
            call(query=record['content'], project_id='reify', limit=20, stores=['mem0']),
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize('mode', MODES)
    async def test_candidates_are_in_rank_order(self, mode: str) -> None:
        service = _FakeMemoryService([_row('r3'), _row('r1'), _row('r2')], live={'c1'})
        got = await _hit(service, _record('d1', 'c1'), mode)
        assert got['candidates'] == ['r3', 'r1', 'r2']

    @pytest.mark.asyncio
    @pytest.mark.parametrize('mode', MODES)
    async def test_a_child_candidate_maps_to_its_parent(self, mode: str) -> None:
        rows = [
            _row('sighting', kind=SIGHTING_KIND, **{PARENT_ID_KEY: 'p1'}),
            _row('amendment', kind=AMENDMENT_KIND, **{PARENT_ID_KEY: 'p2'}),
            _row('plain'),
            _row('not-a-child-kind', kind='completion_note', **{PARENT_ID_KEY: 'p3'}),
            _row('no-parent-link', kind=SIGHTING_KIND),
            _row('null-parent-link', kind=SIGHTING_KIND, **{PARENT_ID_KEY: None}),
            _row('empty-parent-link', kind=SIGHTING_KIND, **{PARENT_ID_KEY: ''}),
            _row('non-str-parent-link', kind=AMENDMENT_KIND, **{PARENT_ID_KEY: 7}),
        ]
        got = await _hit(_FakeMemoryService(rows, live={'c1'}), _record('d1', 'c1'), mode)
        assert got['candidate_parents'] == {'sighting': 'p1', 'amendment': 'p2'}

    @pytest.mark.asyncio
    @pytest.mark.parametrize('mode', MODES)
    async def test_canonical_presence_is_read_from_the_store(self, mode: str) -> None:
        service = _FakeMemoryService(live={'c1'})
        present = await _hit(service, _record('d1', 'c1'), mode)
        absent = await _hit(service, _record('d2', 'gone'), mode)
        assert (present['canonical_present'], absent['canonical_present']) == (True, False)
        assert service.probed == ['c1', 'gone']

    @pytest.mark.asyncio
    @pytest.mark.parametrize('mode', MODES)
    @pytest.mark.parametrize('degraded', [True, False])
    async def test_degraded_is_read_off_the_search_results(
        self, mode: str, degraded: bool,
    ) -> None:
        service = _FakeMemoryService([_row('c1')], live={'c1'}, degraded=degraded)
        got = await _hit(service, _record('d1', 'c1'), mode)
        assert got['degraded'] is degraded

    @pytest.mark.asyncio
    async def test_an_unknown_mode_raises_naming_the_vocabulary(self) -> None:
        service = _FakeMemoryService(live={'c1'})
        with pytest.raises(ValueError, match='bogus') as excinfo:
            await _hit(service, _record('d1', 'c1'), 'bogus')
        for mode in _calib().RETRIEVAL_MODES:
            assert mode in str(excinfo.value)
        service.search.assert_not_awaited()
        assert service.probed == []


class TestRecallProvenance:
    def test_production_names_the_shipped_seam(self) -> None:
        got = _calib().recall_provenance(
            project_id='reify', retrieval_mode='production',
            canonical_aliases_path=ALIASES_PATH,
            aliases=_calib().load_canonical_aliases(ALIASES_PATH),
        )
        assert got == {
            'project_id': 'reify',
            'retrieval_mode': 'production',
            'retrieval_call': (
                f'{retrieve_candidates.__module__}::{retrieve_candidates.__qualname__}'
            ),
            'canonical_aliases_path': (
                'tests/fixtures/write_triage_calibration.canonical_aliases.json'
            ),
            'canonical_aliases_count': 3,
        }

    def test_legacy_names_the_memory_service_search_call(self) -> None:
        got = _calib().recall_provenance(
            project_id='reify', retrieval_mode='legacy',
            canonical_aliases_path=None, aliases=None,
        )
        assert 'MemoryService.search' in got['retrieval_call']
        assert (got['canonical_aliases_path'], got['canonical_aliases_count']) == (None, 0)

    @pytest.mark.parametrize('mode', MODES)
    def test_no_restated_search_kwargs_are_published(self, mode: str) -> None:
        """The branch's own production run published ``search_categories: all``
        beside a three-category search: a restated shape drifts from the call."""
        got = _calib().recall_provenance(
            project_id='reify', retrieval_mode=mode,
            canonical_aliases_path=None, aliases=None,
        )
        assert set(got) == {
            'project_id', 'retrieval_mode', 'retrieval_call',
            'canonical_aliases_path', 'canonical_aliases_count',
        }

    def test_an_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match='bogus'):
            _calib().recall_provenance(
                project_id='reify', retrieval_mode='bogus',
                canonical_aliases_path=None, aliases=None,
            )


_VECTORS = {
    'c1': [1.0, 0.0], 'd1': [0.99, 0.14],
    'c2': [0.0, 1.0], 'd2': [0.14, 0.99],
}


def _calibrate(tmp_path: Path, degraded_ids: frozenset[str]) -> dict:
    records = [
        _record('c1', 'c1', 'canonical'), _record('d1', 'c1'),
        _record('c2', 'c2', 'canonical'), _record('d2', 'c2'),
    ]

    def search(record: dict, k: int) -> dict:
        return {
            'candidates': [record['cluster_id']],
            'canonical_present': True,
            'degraded': record['memory_id'] in degraded_ids,
        }

    return _calib().run_calibration(
        records=records,
        embed_fn=lambda memory_id, content: _VECTORS[memory_id],
        search_fn=search,
        report_path=tmp_path / 'report.json',
        ks=[1],
        provenance={'fixture_path': 'x.jsonl'},
    )


class TestRunCalibrationCountsDegradedRetrievals:
    def test_a_degraded_retrieval_is_counted_and_warned_about(
        self, tmp_path: Path, caplog,
    ) -> None:
        with caplog.at_level(logging.WARNING):
            got = _calibrate(tmp_path, frozenset({'d1'}))
        assert got['report']['provenance']['degraded_retrievals'] == 1
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any('d1' in message for message in warnings)

    def test_a_healthy_run_records_zero(self, tmp_path: Path) -> None:
        got = _calibrate(tmp_path, frozenset())
        assert got['report']['provenance']['degraded_retrievals'] == 0
