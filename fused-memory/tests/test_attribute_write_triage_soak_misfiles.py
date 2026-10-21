"""Tests for attribute_write_triage_soak_misfiles.py — where soak misfiles come from.

Reached only through the script's public surface: ``load_soak_writes``, the
``ORIGIN_AXES`` classifiers and the data types they read. The script is loaded
by path (``scripts/`` is not a package). Every evidence directory here is a
tiny synthetic one in ``tmp_path`` written in the measured row shapes of
``duplicates/records.jsonl`` and ``team/A/attaches_axes.json``; the real soak
evidence is never read.
"""
from __future__ import annotations

import functools
import json
import types
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from _fm_helpers import load_script_module

from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.server.write_triage import AttachGuard

SCRIPTS = Path(__file__).parent.parent / 'scripts'

_PARENT = '11111111-1111-4111-8111-111111111111'
_CHILD_OF_PARENT = '22222222-2222-4222-8222-222222222222'
_OTHER = '33333333-3333-4333-8333-333333333333'
_WRITE = '44444444-4444-4444-8444-444444444444'
_STANDALONE = '55555555-5555-4555-8555-555555555555'


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'attribute_write_triage_soak_misfiles.py',
        mod_name='attribute_write_triage_soak_misfiles',
    )


# --- synthetic evidence -----------------------------------------------------


def _record(
    memory_id: str, *, cls: str = 'standalone', window: str = 'soak',
    project: str = 'solar_challenge', category: str = 'procedural_knowledge',
    nn_triage: float | None = 0.7,
) -> dict[str, Any]:
    return {
        'project': project, 'window': window, 'memory_id': memory_id,
        'category': category, 'class': cls, 'kind': None, 'parent_id': None,
        'nn_triage': nn_triage,
    }


def _slate_row(memory_id: str, cos: float, *, canon: str | None = None, kind: str | None = None) -> dict:
    return {'id': memory_id, 'cos': cos, 'canon': canon or memory_id, 'kind': kind}


def _attach(
    child_id: str, *, parent_id: str = _PARENT, outcome: str = 'amended',
    project: str = 'solar_challenge', cos_parent: float = 0.7,
    slate: list[dict] | None = None, rt_named: str | None = None,
    verdict: str | None = None, child_source: str | None = None,
    child_task_id: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        'child_id': child_id, 'parent_id': parent_id, 'project': project,
        'outcome': outcome, 'child_cat': 'procedural_knowledge',
        'child_task_id': child_task_id, 'child_source': child_source,
        'cos_parent': cos_parent, 'rated': verdict not in (None, 'unclear', 'pending'),
        'verdict': verdict, 'retarget': rt_named is not None,
        'slate20': slate if slate is not None else [_slate_row(parent_id, cos_parent)],
    }
    if rt_named is not None:
        row.update(rt_named=rt_named, rt_named_hoisted=False)
    return row


def _evidence(tmp_path: Path, records: list[dict], attaches: list[dict]) -> Path:
    (tmp_path / 'duplicates').mkdir()
    (tmp_path / 'team' / 'A').mkdir(parents=True)
    (tmp_path / 'duplicates' / 'records.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in records),
    )
    (tmp_path / 'team' / 'A' / 'attaches_axes.json').write_text(json.dumps(attaches))
    return tmp_path


def _by_id(population: Any) -> dict[str, Any]:
    return {write.memory_id: write for write in population.writes}


# --- loader -----------------------------------------------------------------


class TestLoadSoakWrites:
    def test_keeps_only_soak_standalone_and_child_rows_that_carry_a_triage_cosine(self, tmp_path):
        evidence = _evidence(tmp_path, [
            _record(_STANDALONE),
            _record(_WRITE, cls='child'),
            _record(_OTHER, window='base'),
            _record(_PARENT, cls='caller_owned_other'),
            _record(_CHILD_OF_PARENT, nn_triage=None),
        ], [_attach(_WRITE)])

        population = _mod().load_soak_writes(evidence)

        assert set(_by_id(population)) == {_STANDALONE, _WRITE}

    def test_a_row_with_no_triage_cosine_is_tallied_not_silently_dropped(self, tmp_path):
        evidence = _evidence(tmp_path, [
            _record(_STANDALONE, nn_triage=None),
            _record(_WRITE, cls='child', nn_triage=None),
        ], [_attach(_WRITE)])

        population = _mod().load_soak_writes(evidence)

        assert population.writes == ()
        assert set(population.excluded[_mod().Exclusion.NO_TRIAGE_COSINE]) == {_STANDALONE, _WRITE}

    def test_a_standalone_write_is_stored_with_no_attach_fields(self, tmp_path):
        evidence = _evidence(tmp_path, [_record(_STANDALONE, nn_triage=0.41)], [])

        write = _by_id(_mod().load_soak_writes(evidence))[_STANDALONE]

        assert write.outcome == 'stored'
        assert write.max_cosine == 0.41
        assert (write.parent_id, write.cos_parent, write.verdict, write.retarget_named_id) == (
            None, None, None, None,
        )
        assert write.slate == ()

    def test_a_child_write_joins_its_attach_row_on_memory_id(self, tmp_path):
        mod = _mod()
        evidence = _evidence(tmp_path, [
            _record(_WRITE, cls='child', project='reify', category='preferences_and_norms', nn_triage=0.77),
        ], [_attach(
            _WRITE, outcome='contested', project='reify', cos_parent=0.66,
            slate=[_slate_row(_OTHER, 0.77), _slate_row(_CHILD_OF_PARENT, 0.7, canon=_PARENT, kind='sighting')],
            rt_named=_CHILD_OF_PARENT, verdict='contested_misfile',
            child_source='orchestrator_completion', child_task_id='42',
        )])

        write = _by_id(mod.load_soak_writes(evidence))[_WRITE]

        assert write.project == 'reify'
        assert write.category == 'preferences_and_norms'
        assert write.outcome == 'contested'
        assert write.max_cosine == 0.77
        assert write.parent_id == _PARENT
        assert write.cos_parent == 0.66
        assert write.retarget_named_id == _CHILD_OF_PARENT
        assert write.verdict is mod.Verdict.CONTESTED_MISFILE
        assert (write.child_source, write.child_task_id) == ('orchestrator_completion', '42')
        assert write.slate == (
            mod.SlateEntry(id=_OTHER, cosine=0.77, canonical_id=_OTHER, kind=None),
            mod.SlateEntry(id=_CHILD_OF_PARENT, cosine=0.7, canonical_id=_PARENT, kind='sighting'),
        )

    def test_an_unretargeted_attach_has_no_named_id(self, tmp_path):
        evidence = _evidence(tmp_path, [_record(_WRITE, cls='child')], [_attach(_WRITE)])

        assert _by_id(_mod().load_soak_writes(evidence))[_WRITE].retarget_named_id is None

    @pytest.mark.parametrize('verdict', ['unclear', 'pending', 'correct', 'misfile'])
    def test_every_adjudication_verdict_parses(self, tmp_path, verdict):
        evidence = _evidence(tmp_path, [_record(_WRITE, cls='child')], [_attach(_WRITE, verdict=verdict)])

        write = _by_id(_mod().load_soak_writes(evidence))[_WRITE]

        assert write.verdict == _mod().Verdict(verdict)

    def test_an_unknown_verdict_is_refused_naming_the_value_and_the_row(self, tmp_path):
        evidence = _evidence(tmp_path, [_record(_WRITE, cls='child')], [_attach(_WRITE, verdict='half_right')])

        with pytest.raises(ValueError, match='half_right') as refused:
            _mod().load_soak_writes(evidence)
        assert _WRITE in str(refused.value)

    def test_an_unknown_attach_outcome_is_refused_naming_the_value_and_the_row(self, tmp_path):
        evidence = _evidence(tmp_path, [_record(_WRITE, cls='child')], [_attach(_WRITE, outcome='judge')])

        with pytest.raises(ValueError, match='judge') as refused:
            _mod().load_soak_writes(evidence)
        assert _WRITE in str(refused.value)

    def test_an_attach_row_with_no_soak_child_record_is_refused_naming_the_id(self, tmp_path):
        evidence = _evidence(tmp_path, [_record(_STANDALONE)], [_attach(_WRITE)])

        with pytest.raises(ValueError, match=_WRITE):
            _mod().load_soak_writes(evidence)

    def test_an_attach_row_whose_record_is_standalone_is_refused(self, tmp_path):
        evidence = _evidence(tmp_path, [_record(_WRITE)], [_attach(_WRITE)])

        with pytest.raises(ValueError, match=_WRITE):
            _mod().load_soak_writes(evidence)

    def test_a_child_row_with_no_attach_row_is_a_caller_declared_child_tallied_apart(self, tmp_path):
        # The caller set parent_id/kind itself, so triage force-stored it
        # untriaged: measured on the real evidence, 47 such soak children are
        # exactly the 47 without the triage metadata signature.
        evidence = _evidence(tmp_path, [_record(_WRITE, cls='child'), _record(_STANDALONE)], [])

        population = _mod().load_soak_writes(evidence)

        assert set(_by_id(population)) == {_STANDALONE}
        assert population.excluded[_mod().Exclusion.CALLER_DECLARED_CHILD] == (_WRITE,)

    def test_the_loaded_population_is_immutable(self, tmp_path):
        evidence = _evidence(tmp_path, [_record(_STANDALONE)], [])
        population = _mod().load_soak_writes(evidence)

        with pytest.raises(AttributeError):
            population.writes[0].outcome = 'restated'
        with pytest.raises(TypeError):
            population.excluded[_mod().Exclusion.NO_TRIAGE_COSINE] = ()


# --- axes -------------------------------------------------------------------


def _routing(*, t_high: float | None = 0.88, t_low: float | None = 0.52, judge_candidate_count: int = 5) -> Any:
    return _mod().Routing(t_high=t_high, t_low=t_low, judge_candidate_count=judge_candidate_count)


def _entry(memory_id: str, cosine: float, *, parent: str | None = None, kind: str | None = None) -> Any:
    return _mod().SlateEntry(id=memory_id, cosine=cosine, canonical_id=parent or memory_id, kind=kind)


def _write(**overrides: Any) -> Any:
    base = _mod().SoakWrite(
        memory_id=_WRITE, project='solar_challenge', category='procedural_knowledge',
        outcome='amended', max_cosine=0.7, slate=(_entry(_PARENT, 0.7),), parent_id=_PARENT,
        cos_parent=0.7, retarget_named_id=None, verdict=None, child_source=None,
        child_task_id=None,
    )
    return replace(base, **overrides)


def _stored(**overrides: Any) -> Any:
    return _write(
        outcome='stored', slate=(), parent_id=None, cos_parent=None, **overrides,
    )


def _axis(name: str, write: Any, routing: Any | None = None) -> Any:
    return _mod().ORIGIN_AXES[name](write, routing or _routing())


class TestOriginAxes:
    def test_the_axes_are_exactly_the_five_named_ones(self):
        assert set(_mod().ORIGIN_AXES) == {'band', 'hoist', 'retarget', 'record_class', 'cos_parent'}


class TestBandAxis:
    @pytest.mark.parametrize(('max_cosine', 'band'), [
        (0.95, 'RESTATED'),
        (0.88, 'RESTATED'),
        (0.8799, 'JUDGE'),
        (0.52, 'JUDGE'),
        (0.5199, 'STORED'),
    ])
    def test_both_edges_are_inclusive_as_decide_band_reads_them(self, max_cosine, band):
        assert _axis('band', _write(max_cosine=max_cosine)) is getattr(_mod().Band, band)

    def test_an_empty_restate_band_sends_everything_above_t_low_to_the_judge(self):
        assert _axis('band', _write(max_cosine=0.99), _routing(t_high=None)) is _mod().Band.JUDGE

    def test_an_uncalibrated_lower_edge_stores_everything(self):
        assert _axis('band', _write(max_cosine=0.99), _routing(t_low=None)) is _mod().Band.STORED

    def test_the_band_reads_the_write_max_cosine_not_its_slate(self):
        write = _write(max_cosine=0.9, slate=(_entry(_PARENT, 0.6),))

        assert _axis('band', write) is _mod().Band.RESTATED


class TestHoistAxis:
    def test_a_retarget_naming_a_child_of_the_parent_is_via_child(self):
        write = _write(
            retarget_named_id=_CHILD_OF_PARENT,
            slate=(_entry(_OTHER, 0.7), _entry(_CHILD_OF_PARENT, 0.66, parent=_PARENT, kind='sighting')),
        )

        assert _axis('hoist', write) is _mod().Hoist.VIA_CHILD

    def test_a_retarget_naming_the_parent_itself_is_direct(self):
        write = _write(
            retarget_named_id=_PARENT, slate=(_entry(_OTHER, 0.7), _entry(_PARENT, 0.66)),
        )

        assert _axis('hoist', write) is _mod().Hoist.DIRECT

    def test_a_judge_band_write_whose_parent_is_on_the_judge_slate_is_direct(self):
        slate = (
            _entry(_OTHER, 0.7), _entry('a', 0.69), _entry(_PARENT, 0.68),
            _entry('b', 0.67), _entry('c', 0.66), _entry('d', 0.65),
        )

        assert _axis('hoist', _write(max_cosine=0.7, slate=slate)) is _mod().Hoist.DIRECT

    def test_a_judge_band_write_whose_parent_fell_off_the_judge_slate_is_via_child(self):
        # The parent is only reachable through a child below the top five.
        slate = (
            _entry(_OTHER, 0.7), _entry('a', 0.69), _entry('b', 0.68),
            _entry('c', 0.67), _entry('d', 0.66),
            _entry(_CHILD_OF_PARENT, 0.6, parent=_PARENT, kind='amendment'),
        )

        assert _axis('hoist', _write(max_cosine=0.7, slate=slate)) is _mod().Hoist.VIA_CHILD

    def test_the_judge_slate_width_is_the_routing_judge_candidate_count(self):
        slate = (_entry(_OTHER, 0.7), _entry('a', 0.69), _entry(_PARENT, 0.68))
        write = _write(max_cosine=0.7, slate=slate)

        assert _axis('hoist', write, _routing(judge_candidate_count=2)) is _mod().Hoist.VIA_CHILD
        assert _axis('hoist', write, _routing(judge_candidate_count=3)) is _mod().Hoist.DIRECT

    def test_a_judge_band_winner_that_is_a_child_keeps_the_child_not_the_unretrieved_parent(self):
        # select_judge_candidates guarantees the hoisted winner a seat; when the
        # parent was never retrieved, the child holding the evidence is kept.
        slate = (_entry(_CHILD_OF_PARENT, 0.7, parent=_PARENT, kind='sighting'), _entry(_OTHER, 0.6))

        assert _axis('hoist', _write(max_cosine=0.7, slate=slate)) is _mod().Hoist.VIA_CHILD

    def test_a_restate_band_write_whose_top_entry_is_a_child_is_via_child(self):
        write = _write(
            outcome='restated', max_cosine=0.95,
            slate=(_entry(_CHILD_OF_PARENT, 0.95, parent=_PARENT, kind='sighting'), _entry(_PARENT, 0.9)),
        )

        assert _axis('hoist', write) is _mod().Hoist.VIA_CHILD

    def test_a_restate_band_write_whose_top_entry_is_the_parent_is_direct(self):
        write = _write(outcome='restated', max_cosine=0.95, slate=(_entry(_PARENT, 0.95),))

        assert _axis('hoist', write) is _mod().Hoist.DIRECT

    def test_a_restate_band_write_filed_away_from_its_winner_is_via_child_whatever_the_slate_says(self):
        # The slate was re-read after the soak; the filed parent is the soak's own record.
        write = _write(outcome='restated', max_cosine=0.95, slate=(_entry(_OTHER, 0.95),))

        assert _axis('hoist', write) is _mod().Hoist.VIA_CHILD

    def test_a_child_kind_without_a_parent_link_is_not_hoisted(self):
        write = _write(outcome='restated', max_cosine=0.95, slate=(_entry(_PARENT, 0.95, kind='amendment'),))

        assert _axis('hoist', write) is _mod().Hoist.DIRECT

    def test_a_stored_write_was_not_attached(self):
        assert _axis('hoist', _stored()) is _mod().Hoist.NOT_ATTACHED


class TestRetargetAxis:
    def test_a_named_retarget_moved(self):
        assert _axis('retarget', _write(retarget_named_id=_OTHER)) is _mod().Retarget.MOVED

    def test_an_attach_with_no_retarget_kept_the_winner(self):
        assert _axis('retarget', _write()) is _mod().Retarget.KEPT

    def test_a_stored_write_was_not_attached(self):
        assert _axis('retarget', _stored()) is _mod().Retarget.NOT_ATTACHED


class TestRecordClassAxis:
    def test_an_orchestrator_completion_is_a_completion_ledger_record_even_when_task_stamped(self):
        write = _write(child_source='orchestrator_completion', child_task_id='7')

        assert _axis('record_class', write) is _mod().RecordClass.COMPLETION_LEDGER

    def test_a_task_id_without_the_completion_source_is_task_stamped(self):
        write = _write(child_source='implementer', child_task_id='7')

        assert _axis('record_class', write) is _mod().RecordClass.TASK_STAMPED

    def test_no_task_id_is_untasked(self):
        assert _axis('record_class', _write(child_source='agent')) is _mod().RecordClass.UNTASKED


class TestCosParentAxis:
    @pytest.mark.parametrize(('cos_parent', 'bucket'), [
        (0.5199, 'BELOW_T_LOW'),
        (0.52, 'T_LOW_TO_0_65'),
        (0.6499, 'T_LOW_TO_0_65'),
        (0.65, 'FROM_0_65_TO_0_75'),
        (0.7499, 'FROM_0_65_TO_0_75'),
        (0.75, 'FROM_0_75_TO_T_HIGH'),
        (0.8799, 'FROM_0_75_TO_T_HIGH'),
        (0.88, 'AT_OR_ABOVE_T_HIGH'),
        (1.0, 'AT_OR_ABOVE_T_HIGH'),
    ])
    def test_buckets_are_lower_inclusive_on_the_routing_edges(self, cos_parent, bucket):
        assert _axis('cos_parent', _write(cos_parent=cos_parent)) is getattr(_mod().CosParentBucket, bucket)

    def test_an_empty_restate_band_leaves_the_top_bucket_open(self):
        write = _write(cos_parent=0.99)

        assert _axis('cos_parent', write, _routing(t_high=None)) is _mod().CosParentBucket.FROM_0_75_TO_T_HIGH

    def test_a_stored_write_has_no_parent_cosine(self):
        assert _axis('cos_parent', _stored()) is _mod().CosParentBucket.NOT_ATTACHED


# --- origin tables, census fidelity, guard residual -------------------------


@functools.cache
def _stats() -> types.ModuleType:
    return load_script_module(SCRIPTS / 'write_triage_stats.py', mod_name='write_triage_stats')


def _kn(k: int, n: int) -> dict[str, Any]:
    return {'k': k, 'n': n, 'rate': _stats().rate(k, n), 'wilson95': _stats().wilson95(k, n)}


def _verdict(name: str | None) -> Any:
    return None if name is None else _mod().Verdict(name)


_GROUPS = {'solar': frozenset({'solar_challenge', 'solar_challenge_platform'}), 'df': frozenset({'dark_factory'})}


def _census_population() -> list[Any]:
    """Judge-band attaches unless noted; w6 is restate-band; w10 is stored."""
    return [
        _write(memory_id='w1', outcome='contested', verdict=_verdict('contested_misfile'), retarget_named_id=_PARENT),
        _write(memory_id='w2', outcome='contested', verdict=_verdict('correct')),
        _write(memory_id='w3', outcome='contested', verdict=_verdict('unclear')),
        _write(memory_id='w4', outcome='contested'),
        _write(memory_id='w5', outcome='amended', verdict=_verdict('misfile')),
        _write(memory_id='w6', outcome='restated', max_cosine=0.95, slate=(_entry(_PARENT, 0.95),),
               verdict=_verdict('correct')),
        _write(memory_id='w7', project='solar_challenge_platform', outcome='contested',
               verdict=_verdict('false_contested')),
        _write(memory_id='w8', project='solar_challenge_platform', outcome='contested', verdict=_verdict('pending')),
        _write(memory_id='w9', project='dark_factory', outcome='contested', verdict=_verdict('correct')),
        _stored(memory_id='w10', project='dark_factory'),
    ]


class TestOriginTables:
    def _tables(self) -> dict[str, Any]:
        return _mod().origin_tables(_census_population(), _GROUPS, _routing())

    def test_every_origin_axis_has_a_table_per_group(self):
        tables = self._tables()

        assert set(tables) == set(_mod().ORIGIN_AXES)
        assert all(set(by_group) == set(_GROUPS) for by_group in tables.values())

    def test_misfile_counts_only_rated_attaches_per_outcome_and_cell(self):
        kept = self._tables()['retarget']['solar']['kept']

        # w2 correct + w7 false_contested rated; w3 unclear, w4 unrated, w8 pending are not.
        assert kept['contested'] == {'attaches': 5, 'misfile': _kn(0, 2)}
        assert kept['amended'] == {'attaches': 1, 'misfile': _kn(1, 1)}
        assert kept['restated'] == {'attaches': 1, 'misfile': _kn(0, 1)}

    def test_contested_misfile_and_misfile_both_count_as_misfile(self):
        tables = self._tables()

        assert tables['retarget']['solar']['moved']['contested']['misfile'] == _kn(1, 1)
        assert tables['band']['solar']['judge']['amended']['misfile'] == _kn(1, 1)

    def test_an_empty_cell_reports_no_rate_rather_than_zero(self):
        moved = self._tables()['retarget']['solar']['moved']

        assert moved['amended'] == {'attaches': 0, 'misfile': {'k': 0, 'n': 0, 'rate': None, 'wilson95': None}}

    def test_every_interval_is_the_wilson_interval_of_its_own_counts(self):
        band = self._tables()['band']['solar']['judge']['contested']['misfile']

        assert band == _kn(1, 3)
        assert band['wilson95'] == _stats().wilson95(1, 3)

    def test_stored_writes_are_not_attaches_and_open_no_cell(self):
        tables = self._tables()

        assert set(tables['retarget']['df']) == {'kept'}
        assert tables['band']['df']['judge']['contested'] == {'attaches': 1, 'misfile': _kn(0, 1)}

    def test_the_restate_band_cell_holds_the_restate_band_attach(self):
        restated = self._tables()['band']['solar']['restated']

        assert restated['restated'] == {'attaches': 1, 'misfile': _kn(0, 1)}
        assert restated['contested']['attaches'] == 0


class TestCensusFidelity:
    def test_contested_misfile_over_rated_contested_per_group(self):
        census = _mod().census_fidelity(_census_population(), _GROUPS)

        assert census == {'solar': _kn(1, 3), 'df': _kn(0, 1)}


class TestGroups:
    def test_the_census_groups_are_solar_with_scoping_intake_and_dark_factory_with_reify(self):
        groups = _mod().GROUPS

        assert groups == {
            'solar': frozenset({'solar_challenge', 'solar_challenge_platform', 'scoping_intake'}),
            'dark_factory_reify': frozenset({'dark_factory', 'reify'}),
        }

    def test_every_project_gets_a_row_of_its_own_after_the_named_groups(self):
        writes = [_write(project='reify'), _stored(project='pump_web_ui')]

        groups = _mod().groups_with_projects(_GROUPS, writes)

        assert list(groups) == ['solar', 'df', 'pump_web_ui', 'reify']
        assert groups['reify'] == frozenset({'reify'})

    def test_a_group_name_that_shadows_a_project_is_refused(self):
        with pytest.raises(ValueError, match='reify'):
            _mod().groups_with_projects({'reify': frozenset({'dark_factory'})}, [_write(project='reify')])


def _guard(*, floor: float | None = 0.60, gap: float | None = 0.05) -> AttachGuard:
    return AttachGuard(min_parent_cosine=floor, hoist_max_cosine_gap=gap)


def _below_floor(**overrides: Any) -> Any:
    return _write(
        outcome='amended', max_cosine=0.55, slate=(_entry(_PARENT, 0.55),), cos_parent=0.55, **overrides,
    )


def _judged_hoist_past_the_gap(**overrides: Any) -> Any:
    return _write(
        outcome='contested', max_cosine=0.75, retarget_named_id=_CHILD_OF_PARENT, cos_parent=0.62,
        slate=(_entry(_OTHER, 0.75), _entry(_CHILD_OF_PARENT, 0.72, parent=_PARENT, kind='sighting')),
        **overrides,
    )


def _restated_hoist_past_the_gap(**overrides: Any) -> Any:
    return _write(
        outcome='restated', max_cosine=0.95, cos_parent=0.85,
        slate=(_entry(_CHILD_OF_PARENT, 0.95, parent=_PARENT, kind='sighting'),), **overrides,
    )


def _residual(writes: list[Any], guard: Any | None = None) -> dict[str, Any]:
    return _mod().guard_residual(writes, guard or _guard(), _routing(), {'solar': frozenset({'solar_challenge'})})['solar']


class TestGuardResidual:
    def test_a_rated_misfile_below_the_parent_floor_is_removed(self):
        amended = _residual([_below_floor(verdict=_verdict('misfile'))])['amended']

        assert amended['removed'] == 1
        assert amended['removed_by_reason'] == {'below_min_parent_cosine': 1}
        assert amended['rated_misfiles_removed'] == 1
        assert amended['misfile_after'] == _kn(0, 0)

    def test_a_judged_hoist_more_than_the_gap_below_its_source_is_removed(self):
        contested = _residual([_judged_hoist_past_the_gap(verdict=_verdict('contested_misfile'))])['contested']

        assert contested['removed'] == 1
        assert contested['removed_by_reason'] == {'hoist_cosine_gap': 1}

    def test_the_same_gap_in_the_restate_band_is_not_removed(self):
        restated = _residual([_restated_hoist_past_the_gap(verdict=_verdict('misfile'))])['restated']

        assert restated['removed'] == 0
        assert restated['misfile_after'] == _kn(1, 1)

    def test_a_guard_rule_switched_off_removes_nothing(self):
        contested = _residual([_judged_hoist_past_the_gap()], _guard(gap=None))['contested']

        assert contested['removed'] == 0

    def test_the_residual_is_misfile_k_of_n_over_the_rated_attaches_that_survive(self):
        residual = _residual([
            _below_floor(memory_id='g1', verdict=_verdict('misfile')),
            _judged_hoist_past_the_gap(memory_id='g2', verdict=_verdict('contested_misfile')),
            _restated_hoist_past_the_gap(memory_id='g3', verdict=_verdict('misfile')),
            _write(memory_id='g4', outcome='contested', verdict=_verdict('correct')),
            _write(memory_id='g5', outcome='contested', verdict=_verdict('contested_misfile')),
            _below_floor(memory_id='g6'),
            _write(memory_id='g7', outcome='contested', cos_parent=0.5, verdict=_verdict('correct')),
            _stored(memory_id='g8'),
        ])

        assert residual['contested'] == {
            'attaches': 4, 'removed': 2,
            'removed_by_reason': {'below_min_parent_cosine': 1, 'hoist_cosine_gap': 1},
            'rated_misfiles_removed': 1, 'rated_non_misfiles_removed': 1,
            'misfile_before': _kn(2, 4), 'misfile_after': _kn(1, 2),
        }
        assert residual['amended']['attaches'] == 2
        assert residual['amended']['removed'] == 2
        assert residual['amended']['misfile_before'] == _kn(1, 1)
        assert residual['restated']['misfile_after'] == _kn(1, 1)


class TestShippedWritePath:
    def test_the_routing_and_guard_are_read_from_the_shipped_write_triage_config(self):
        triage = FusedMemoryConfig().write_triage

        routing, guard = _mod().shipped_write_path()

        assert routing == _mod().Routing(
            t_high=triage.t_high, t_low=triage.t_low, judge_candidate_count=triage.judge_candidate_count,
        )
        assert guard == AttachGuard(
            min_parent_cosine=triage.min_parent_cosine, hoist_max_cosine_gap=triage.hoist_max_cosine_gap,
        )


# --- per-project band arms and the replay -----------------------------------


def _pooled() -> Any:
    return _routing(t_high=0.88, t_low=0.52)


def _report(
    *, t_high: float | None = None, t_low: float | None = None,
    per_category: dict[str, tuple[float | None, float | None]] | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {'chosen_t_high': t_high, 'chosen_t_low': t_low}
    if per_category is not None:
        report['per_category'] = {
            category: {'t_high': high, 't_low': low, 'reason': None if low is not None else 'empty_class: n_duplicate=0'}
            for category, (high, low) in per_category.items()
        }
    return report


def _edges(arm: Any, category: str) -> tuple[tuple[float | None, str], tuple[float | None, str]]:
    edges = arm.edges_for(category)
    return (edges.t_high.value, edges.t_high.source.value), (edges.t_low.value, edges.t_low.source.value)


_FALLBACK = ((0.88, 'pooled_fallback'), (0.52, 'pooled_fallback'))


class TestProjectBandArms:
    def test_the_two_arms_are_project_and_project_category(self):
        arms = _mod().project_band_arms(_report(), _pooled())

        assert set(arms) == {'project', 'project_category'}

    def test_a_fully_refused_report_falls_back_to_pooled_on_every_edge(self):
        report = _report(per_category={'procedural_knowledge': (None, None), 'preferences_and_norms': (None, None)})

        arms = _mod().project_band_arms(report, _pooled())

        for arm in arms.values():
            for category in ('procedural_knowledge', 'preferences_and_norms', 'observations_and_summaries'):
                assert _edges(arm, category) == _FALLBACK

    def test_a_derived_category_overrides_only_that_category(self):
        report = _report(per_category={'observations_and_summaries': (0.85, 0.63), 'preferences_and_norms': (None, None)})

        arm = _mod().project_band_arms(report, _pooled())['project_category']

        assert _edges(arm, 'observations_and_summaries') == ((0.85, 'report'), (0.63, 'report'))
        assert _edges(arm, 'preferences_and_norms') == _FALLBACK
        assert _edges(arm, 'procedural_knowledge') == _FALLBACK

    def test_a_derived_category_leaves_the_project_arm_on_its_own_chosen_edges(self):
        report = _report(per_category={'observations_and_summaries': (0.85, 0.63)})

        arm = _mod().project_band_arms(report, _pooled())['project']

        assert _edges(arm, 'observations_and_summaries') == _FALLBACK

    def test_the_project_arm_takes_the_chosen_edges_for_every_category(self):
        arm = _mod().project_band_arms(_report(t_high=0.8, t_low=0.7), _pooled())['project']

        assert _edges(arm, 'procedural_knowledge') == ((0.8, 'report'), (0.7, 'report'))

    def test_each_edge_falls_back_on_its_own(self):
        arm = _mod().project_band_arms(_report(t_low=0.6), _pooled())['project']

        assert _edges(arm, 'procedural_knowledge') == ((0.88, 'pooled_fallback'), (0.6, 'report'))

    def test_a_missing_per_category_section_is_all_fallback(self):
        arm = _mod().project_band_arms(_report(t_high=0.8, t_low=0.7), _pooled())['project_category']

        assert _edges(arm, 'procedural_knowledge') == _FALLBACK


def _move(memory_id: str, max_cosine: float, **overrides: Any) -> Any:
    return _write(memory_id=memory_id, max_cosine=max_cosine, **overrides)


def _replay(writes: list[Any], report: dict[str, Any], arm: str = 'project') -> dict[str, Any]:
    arms = {'solar_challenge': _mod().project_band_arms(report, _pooled())}
    return _mod().replay_bands(writes, _pooled(), arms)['solar_challenge'][arm]


def _matrix(**cells: int) -> dict[str, dict[str, int]]:
    bands = ('restated', 'judge', 'stored')
    matrix = {source: dict.fromkeys(bands, 0) for source in bands}
    for cell, count in cells.items():
        source, target = cell.split('__')
        matrix[source][target] = count
    return matrix


class TestReplayBands:
    def _raised_floor_lowered_ceiling(self) -> dict[str, Any]:
        # Arm 0.80 / 0.70 against pooled 0.88 / 0.52.
        return _replay([
            _move('a', 0.60, verdict=_verdict('misfile')),
            _move('b', 0.65, outcome='contested', verdict=_verdict('correct')),
            _move('c', 0.68),
            _move('d', 0.82, verdict=_verdict('correct')),
            _move('e', 0.75, verdict=_verdict('unclear')),
            _stored(memory_id='f', max_cosine=0.40),
            _move('g', 0.95, outcome='restated'),
            _move('h', 0.60, project='dark_factory', verdict=_verdict('misfile')),
        ], _report(t_high=0.80, t_low=0.70))

    def test_the_matrix_counts_every_write_once_from_its_pooled_band_to_its_arm_band(self):
        assert self._raised_floor_lowered_ceiling()['matrix'] == _matrix(
            restated__restated=1, judge__stored=3, judge__restated=1, judge__judge=1, stored__stored=1,
        )

    def test_moving_into_stored_removes_rated_misfiles_and_loses_rated_non_misfiles(self):
        assert self._raised_floor_lowered_ceiling()['into_stored'] == {
            'movers': 3, 'rated_misfiles_removed': 1, 'rated_non_misfiles_lost': 1, 'unrated': 1,
        }

    def test_any_other_move_reports_soak_outcomes_and_verdicts_and_an_unknown_counterfactual(self):
        assert self._raised_floor_lowered_ceiling()['other_moves'] == [{
            'from': 'judge', 'to': 'restated', 'movers': 1,
            'soak_outcomes': {'amended': 1}, 'rated_verdicts': {'correct': 1}, 'unrated': 0,
            'counterfactual_verdict': 'unknown',
        }]

    def test_a_lowered_floor_moves_stored_writes_into_the_judge_band(self):
        replay = _replay(
            [_stored(memory_id='s', max_cosine=0.48), _stored(memory_id='t', max_cosine=0.40)],
            _report(t_low=0.45),
        )

        assert replay['matrix'] == _matrix(judge__stored=0, stored__judge=1, stored__stored=1)
        assert replay['other_moves'] == [{
            'from': 'stored', 'to': 'judge', 'movers': 1, 'soak_outcomes': {'stored': 1},
            'rated_verdicts': {}, 'unrated': 1, 'counterfactual_verdict': 'unknown',
        }]

    def test_a_fully_refused_report_replays_to_the_diagonal_only(self):
        replay = _replay(
            [_move('a', 0.60, verdict=_verdict('misfile')), _stored(memory_id='f', max_cosine=0.40)],
            _report(per_category={'procedural_knowledge': (None, None)}), arm='project_category',
        )

        assert replay['matrix'] == _matrix(judge__judge=1, stored__stored=1)
        assert replay['into_stored'] == {
            'movers': 0, 'rated_misfiles_removed': 0, 'rated_non_misfiles_lost': 0, 'unrated': 0,
        }
        assert replay['other_moves'] == []

    def test_the_category_arm_routes_each_write_on_its_own_category(self):
        report = _report(per_category={'observations_and_summaries': (0.85, 0.63)})

        replay = _replay([
            _move('o', 0.60, category='observations_and_summaries'),
            _move('p', 0.60, category='procedural_knowledge'),
        ], report, arm='project_category')

        assert replay['matrix'] == _matrix(judge__stored=1, judge__judge=1)

    def test_each_arm_echoes_its_edges_and_where_each_came_from(self):
        report = _report(per_category={'observations_and_summaries': (0.85, 0.63)})

        edges = _replay([], report, arm='project_category')['edges']

        assert edges == {
            'default': {
                't_high': {'value': 0.88, 'source': 'pooled_fallback'},
                't_low': {'value': 0.52, 'source': 'pooled_fallback'},
            },
            'by_category': {'observations_and_summaries': {
                't_high': {'value': 0.85, 'source': 'report'},
                't_low': {'value': 0.63, 'source': 'report'},
            }},
        }

    def test_a_project_with_no_report_is_not_replayed(self):
        arms = {'solar_challenge': _mod().project_band_arms(_report(), _pooled())}

        replay = _mod().replay_bands([_move('h', 0.6, project='dark_factory')], _pooled(), arms)

        assert set(replay) == {'solar_challenge'}
