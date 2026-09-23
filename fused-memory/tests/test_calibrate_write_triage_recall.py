"""Tests for the recall side of calibrate_write_triage.py.

Recall measures the SHIPPED retrieval, so the ``production`` mode is pinned in
lockstep with ``write_triage.retrieve_candidates`` through a fake
MemoryService, never against a restated literal. ``legacy`` is the historical
search shape, and a literal is exactly what pins a historical shape.
"""
from __future__ import annotations

import functools
import json
import logging
import sys
import types
from collections.abc import Iterable, Mapping
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
_FIXTURE = 'tests/fixtures/x.jsonl'
_BANDS_FROM = SCRIPT_PATH.parent.parent / 'calibration' / 'write_triage_calibration_report.json'


def _records() -> list[dict]:
    return [
        _record('c1', 'c1', 'canonical'), _record('d1', 'c1'),
        _record('c2', 'c2', 'canonical'), _record('d2', 'c2'),
    ]


def _search(
    hits: Mapping[str, list[str]],
    *,
    absent: frozenset[str] = frozenset(),
    degraded: frozenset[str] = frozenset(),
    searched: list[str] | None = None,
):
    def search(record: dict, k: int) -> dict:
        if searched is not None:
            searched.append(record['memory_id'])
        return {
            'candidates': list(hits.get(record['memory_id'], [])),
            'canonical_present': record['cluster_id'] not in absent,
            'degraded': record['memory_id'] in degraded,
        }

    return search


def _calibrate(tmp_path: Path, search, *, provenance: dict | None = None) -> dict:
    return _calib().run_calibration(
        records=_records(),
        embed_fn=lambda memory_id, content: _VECTORS[memory_id],
        search_fn=search,
        report_path=tmp_path / 'report.json',
        ks=[1],
        provenance=provenance if provenance is not None else {'fixture_path': _FIXTURE},
    )


class TestRunCalibrationCountsDegradedRetrievals:
    def test_a_degraded_retrieval_is_counted_and_warned_about(
        self, tmp_path: Path, caplog,
    ) -> None:
        with caplog.at_level(logging.WARNING):
            got = _calibrate(tmp_path, _search({}, degraded=frozenset({'d1'})))
        assert got['report']['provenance']['degraded_retrievals'] == 1
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any('d1' in message for message in warnings)

    def test_a_healthy_run_records_zero(self, tmp_path: Path) -> None:
        got = _calibrate(tmp_path, _search({}))
        assert got['report']['provenance']['degraded_retrievals'] == 0


class TestRecallStatesItsPopulation:
    RETRIEVALS = [
        {'memory_id': 'd1', 'canonical_id': 'c1', 'canonical_present': True,
         'candidates': ['c1']},
        {'memory_id': 'd2', 'canonical_id': 'gone', 'canonical_present': False,
         'candidates': ['z']},
    ]

    @staticmethod
    def _absent_line(recall: dict) -> str:
        md = _calib().render_markdown(_calib().build_report(
            scores_by_class={}, t_high=None, t_low=None, reason=None,
            recall=recall, provenance={},
        ))
        return next(
            line for line in md.splitlines()
            if line.startswith('Canonicals absent from the corpus')
        )

    def test_the_default_population_excludes_absent_canonicals(self) -> None:
        recall = _calib().compute_recall_at_k(self.RETRIEVALS, [1])
        assert recall['absent_in_denominator'] is False
        assert recall['per_k'][0]['total'] == 1
        line = self._absent_line(recall)
        assert 'excluded from the denominator' in line
        assert line.endswith(': 1')

    def test_counting_absent_canonicals_says_they_are_counted(self) -> None:
        recall = _calib().compute_recall_at_k(self.RETRIEVALS, [1], count_absent_as_miss=True)
        assert recall['absent_in_denominator'] is True
        assert recall['per_k'][0]['total'] == 2
        line = self._absent_line(recall)
        assert 'excluded' not in line
        assert 'counted in the denominator' in line
        assert 'alias' in line and 'hoisted child' in line
        assert line.endswith(': 1')

    def test_a_recall_section_that_does_not_state_its_population_is_refused(self) -> None:
        recall = _calib().compute_recall_at_k(self.RETRIEVALS, [1])
        del recall['absent_in_denominator']
        with pytest.raises(KeyError, match='absent_in_denominator'):
            self._absent_line(recall)


_BAND_SECTION = (
    'chosen_t_high', 'chosen_t_low', 'reason', 'deterministic_band_false_positives',
    'distributions', 'per_band', 'per_category',
)
_ALIASES = {'c2': 'c2-new'}


class TestRemeasureRecall:
    """A recall-only run carries the band section of record verbatim (PRD D2)."""

    @staticmethod
    def _base(tmp_path: Path) -> dict:
        base_dir = tmp_path / 'base'
        base_dir.mkdir()
        return _calibrate(
            base_dir,
            _search({'d1': ['c1'], 'd2': ['z']}, absent=frozenset({'c2'}),
                    degraded=frozenset({'d2'})),
            provenance={
                'fixture_path': _FIXTURE, 'project_id': 'reify',
                'embedder_model': 'text-embedding-3-small', 'embedder_dimensions': 1536,
                'search_categories': 'all',
            },
        )['report']

    @staticmethod
    def _this_run(**overrides) -> dict:
        return {
            'fixture_path': _FIXTURE,
            'embedder_model': 'a-different-embedder',
            'embedder_dimensions': 1536,
            **_calib().recall_provenance(
                project_id='reify', retrieval_mode='production',
                canonical_aliases_path=None, aliases=_ALIASES,
            ),
            **overrides,
        }

    @staticmethod
    def _remeasure(tmp_path: Path, base: dict, *, records=None, provenance=None,
                   searched: list[str] | None = None) -> dict:
        return _calib().remeasure_recall(
            base_report=base,
            records=records if records is not None else _records(),
            search_fn=_search(
                {'d1': ['c1'], 'd2': ['z', 'c2-new']}, absent=frozenset({'c2'}),
                searched=searched,
            ),
            report_path=tmp_path / 'report.json',
            ks=[1, 20],
            provenance=provenance if provenance is not None else TestRemeasureRecall._this_run(),
            bands_from=_BANDS_FROM,
            aliases=_ALIASES,
        )

    def test_the_band_section_is_the_base_reports(self, tmp_path: Path) -> None:
        base = self._base(tmp_path)
        report = self._remeasure(tmp_path, base)['report']
        for key in _BAND_SECTION:
            assert report[key] == base[key], key
        assert list(report) == list(base), 'the base report\'s key order is kept'

    def test_recall_is_the_new_measurement_over_the_full_population(
        self, tmp_path: Path,
    ) -> None:
        base = self._base(tmp_path)
        assert [(r['k'], r['total']) for r in base['recall_at_k']['per_k']] == [(1, 1)]
        recall = self._remeasure(tmp_path, base)['report']['recall_at_k']
        assert [(r['k'], r['hits'], r['total']) for r in recall['per_k']] == [
            (1, 1, 2), (20, 2, 2),
        ]
        assert recall['absent_in_denominator'] is True

    def test_provenance_is_the_bases_band_side_plus_this_runs_recall_side(
        self, tmp_path: Path,
    ) -> None:
        base = self._base(tmp_path)
        provenance = self._remeasure(tmp_path, base)['report']['provenance']
        band_side = (
            'fixture_path', 'embedder_model', 'embedder_dimensions', 'record_count',
            'cluster_count', 'per_category_record_counts', 'cross_category_dropped',
            'pair_counts', 'per_category_pair_counts',
        )
        recall_side = (
            'project_id', 'retrieval_mode', 'retrieval_call',
            'canonical_aliases_path', 'canonical_aliases_count',
        )
        assert set(provenance) == {*band_side, *recall_side, 'degraded_retrievals', 'bands_from'}
        for key in band_side:
            assert provenance[key] == base['provenance'][key], key
        assert provenance['embedder_model'] == 'text-embedding-3-small'
        for key in recall_side:
            assert provenance[key] == self._this_run()[key], key
        assert base['provenance']['degraded_retrievals'] == 1
        assert provenance['degraded_retrievals'] == 0
        assert provenance['bands_from'] == 'calibration/write_triage_calibration_report.json'

    def test_a_base_measured_on_another_fixture_is_refused(self, tmp_path: Path) -> None:
        base = self._base(tmp_path)
        searched: list[str] = []
        with pytest.raises(ValueError) as excinfo:
            self._remeasure(
                tmp_path, base, searched=searched,
                provenance=self._this_run(fixture_path='tests/fixtures/other.jsonl'),
            )
        assert _FIXTURE in str(excinfo.value)
        assert 'tests/fixtures/other.jsonl' in str(excinfo.value)
        assert searched == []
        assert not (tmp_path / 'report.json').exists()

    def test_a_base_measured_on_another_population_is_refused(self, tmp_path: Path) -> None:
        base = self._base(tmp_path)
        with pytest.raises(ValueError, match='record'):
            self._remeasure(tmp_path, base, records=_records()[:-1])
        assert not (tmp_path / 'report.json').exists()

    def test_the_json_and_its_markdown_render_are_written(self, tmp_path: Path) -> None:
        got = self._remeasure(tmp_path, self._base(tmp_path))
        written = json.loads((tmp_path / 'report.json').read_text())
        assert written == got['report']
        assert (tmp_path / 'report.md').read_text() == _calib().render_markdown(written)
        assert {r['memory_id'] for r in got['retrievals']} == {'d1', 'd2'}


class TestBandsFromCli:
    def test_a_recall_only_run_can_never_write_config(
        self, tmp_path: Path, monkeypatch, capsys,
    ) -> None:
        monkeypatch.setattr(sys, 'argv', [
            'calibrate_write_triage.py', '--bands-from', str(tmp_path / 'base.json'),
            '--write-config',
        ])
        with pytest.raises(SystemExit) as excinfo:
            _calib().main()
        assert excinfo.value.code == 2
        assert 'not allowed with argument' in capsys.readouterr().err
