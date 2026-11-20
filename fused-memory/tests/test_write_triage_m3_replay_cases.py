"""Tests for write_triage_m3_replay_cases.py — the soak corpus as replay cases (task 7519).

Small synthetic inputs only; no soak data. The script is loaded by path, as
its siblings are.
"""
from __future__ import annotations

import functools
import types
from pathlib import Path
from typing import Any

import pytest
from _fm_helpers import load_script_module

from fused_memory.memory_metadata import ORCHESTRATOR_COMPLETION_SOURCE
from fused_memory.server.write_triage import AttachGuard

SCRIPTS = Path(__file__).parent.parent / 'scripts'


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'write_triage_m3_replay_cases.py', mod_name='write_triage_m3_replay_cases',
    )


@functools.cache
def _scorer() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'score_write_triage_reader.py', mod_name='score_write_triage_reader',
    )


_CHILD = 'The new entry says the cache is now flushed hourly, not daily.'
_PARENT = 'The cache is flushed daily by the maintenance job.'


def _item(
    entry: str = 'E1', target: str = 'P1', *, outcome: str = 'amended', band: str | None = 'judge',
    stratum: str = 'amended-sample', supersedes_target: bool = False,
    via_sibling: bool | None = False, **extra: Any,
) -> dict[str, Any]:
    return {
        'entry_id': entry, 'target_id': target, 'outcome': outcome, 'band': band,
        'project': 'dark_factory', 'stratum': stratum, 'earlier_ids': [],
        'superseded_marks': [], 'supersedes_target': supersedes_target,
        'via_sibling': via_sibling, 'cross_category': False,
        'parent_edited_after_attach': False, **extra,
    }


def _text(
    entry: str = 'E1', target: str = 'P1', *, basis: str = 'current',
    child: str = _CHILD, parent: str = _PARENT,
) -> dict[str, Any]:
    return {
        'entry_id': entry, 'target_id': target, 'basis': basis,
        'child_text': child, 'parent_text': parent,
    }


def _cos(cos_parent: float = 0.8, sibling: float | None = None) -> dict[str, Any]:
    return {'cos_parent': cos_parent, 'max_cos_earlier_sibling': sibling}


def _build(
    items: list[dict[str, Any]] | None = None,
    texts: list[dict[str, Any]] | None = None,
    cosines: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[Any, ...]:
    return _mod().build_cases(
        items if items is not None else [_item()],
        texts if texts is not None else [_text()],
        cosines if cosines is not None else {'E1': _cos()},
        metadata if metadata is not None else {'E1': {'source': 'agent'}},
    )


def _case(**item: Any) -> Any:
    metadata = item.pop('metadata', {'source': 'agent'})
    cosine = item.pop('cosine', _cos())
    texts = item.pop('texts', [_text()])
    (case,) = _build([_item(**item)], texts, {'E1': cosine}, {'E1': metadata})
    return case


_GUARD = AttachGuard(min_parent_cosine=0.60, hoist_max_cosine_gap=0.05)
_NO_GUARD = AttachGuard(min_parent_cosine=None, hoist_max_cosine_gap=None)


class TestBuildCases:
    def test_joins_each_item_to_its_current_text_and_cosines(self) -> None:
        (case,) = _build()
        features = case.features
        assert (features.entry_id, features.target_id) == ('E1', 'P1')
        assert (features.stratum, features.project) == ('amended-sample', 'dark_factory')
        assert (features.soak_outcome, features.soak_band) == ('amended', 'judge')
        assert (case.child_text, case.parent_text) == (_CHILD, _PARENT)
        assert features.cos_parent == 0.8
        assert features.metadata_recovered is True
        assert features.cosine_recovered is True

    def test_an_attach_time_text_row_is_not_the_current_text(self) -> None:
        texts = [_text(basis='attach_time', child='an older parent text'), _text()]
        (case,) = _build(texts=texts)
        assert case.child_text == _CHILD
        with pytest.raises(ValueError, match='E1'):
            _build(texts=[_text(basis='attach_time')])

    def test_an_item_without_a_text_row_is_refused_naming_it(self) -> None:
        items = [_item(), _item('E2', 'P2')]
        with pytest.raises(ValueError, match='E2') as refusal:
            _build(items, [_text()], {'E1': _cos(), 'E2': _cos()})
        assert 'P2' in str(refusal.value)

    def test_a_text_row_for_another_target_does_not_join(self) -> None:
        with pytest.raises(ValueError, match='E1'):
            _build(texts=[_text(target='P9')])

    def test_an_item_absent_from_the_cosines_is_refused_naming_it(self) -> None:
        items = [_item(), _item('E2', 'P2')]
        texts = [_text(), _text('E2', 'P2')]
        with pytest.raises(ValueError, match='E2'):
            _build(items, texts, {'E1': _cos()})

    def test_an_explicitly_unrecoverable_cosine_is_flagged_not_refused(self) -> None:
        case = _case(cosine=None)
        assert case.features.cosine_recovered is False
        assert case.features.cos_parent is None
        assert case.features.source_cosine is None

    def test_cases_are_sorted_by_entry_id(self) -> None:
        items = [_item('E2', 'P2'), _item()]
        texts = [_text('E2', 'P2'), _text()]
        cases = _build(items, texts, {'E1': _cos(), 'E2': _cos()}, {})
        assert [case.features.entry_id for case in cases] == ['E1', 'E2']


class TestSourceCosine:
    def test_a_direct_attach_measures_its_source_at_the_parent(self) -> None:
        case = _case(via_sibling=False, cosine=_cos(0.7, 0.9))
        assert case.features.source_cosine == 0.7

    def test_a_via_sibling_attach_measures_its_source_at_the_sibling(self) -> None:
        case = _case(via_sibling=True, cosine=_cos(0.7, 0.9))
        assert case.features.source_cosine == 0.9


class TestRoute:
    def test_a_supersession_of_the_target_routes_first(self) -> None:
        case = _case(supersedes_target=True, metadata={'kind': 'note'},
                     texts=[_text(child=_PARENT)])
        assert case.features.route is _mod().Route.SUPERSESSION

    def test_caller_owned_attach_keys_force_store(self) -> None:
        case = _case(metadata={'kind': 'note'}, texts=[_text(child=_PARENT)])
        assert case.features.route is _mod().Route.FORCE_STORED

    def test_a_completion_record_force_stores(self) -> None:
        case = _case(metadata={'source': ORCHESTRATOR_COMPLETION_SOURCE})
        assert case.features.route is _mod().Route.FORCE_STORED

    def test_a_supersession_of_another_record_force_stores_without_marking_the_target(
        self,
    ) -> None:
        case = _case(metadata={'supersedes': ['P7']})
        assert case.features.route is _mod().Route.FORCE_STORED

    def test_a_near_verbatim_write_is_restated_without_a_call(self) -> None:
        case = _case(texts=[_text(child='the CACHE is flushed   daily by the maintenance job .')])
        assert case.features.route is _mod().Route.NEAR_VERBATIM

    def test_everything_else_is_judged(self) -> None:
        assert _case().features.route is _mod().Route.JUDGE

    def test_unrecovered_metadata_routes_as_if_no_force_store_applies(self) -> None:
        (case,) = _build(metadata={})
        assert case.features.route is _mod().Route.JUDGE
        assert case.features.metadata_recovered is False


class TestCallerMetadata:
    def test_strips_the_keys_an_attach_wrote(self) -> None:
        stored = {'parent_id': 'P1', 'kind': 'amendment', 'x_contested': True,
                  'source': 'agent', 'topic': 'cache'}
        assert _mod().caller_metadata(stored) == {'source': 'agent', 'topic': 'cache'}


class TestRoutedOutcome:
    @pytest.mark.parametrize('route_item', [
        {'supersedes_target': True}, {'metadata': {'kind': 'note'}},
    ])
    def test_a_force_stored_route_is_stored_whatever_the_verdict(
        self, route_item: dict[str, Any],
    ) -> None:
        case = _case(**route_item)
        assert _mod().routed_outcome(case.features, None, _GUARD) == 'stored'

    def test_a_near_verbatim_route_is_restated(self) -> None:
        case = _case(texts=[_text(child=_PARENT)])
        assert _mod().routed_outcome(case.features, None, _GUARD) == 'restated'

    @pytest.mark.parametrize('verdict', ['restated', 'amended', 'contested', 'stored'])
    def test_a_judged_route_keeps_its_verdict_when_the_guard_passes(self, verdict: str) -> None:
        case = _case(cosine=_cos(0.60))
        assert _mod().routed_outcome(case.features, verdict, _GUARD) == verdict

    def test_the_floor_refuses_an_attach_below_it(self) -> None:
        case = _case(cosine=_cos(0.59))
        assert _mod().routed_outcome(case.features, 'contested', _GUARD) == 'stored'
        assert _mod().routed_outcome(case.features, 'contested', _NO_GUARD) == 'contested'

    def test_the_hoist_gap_refuses_a_judged_hoist(self) -> None:
        case = _case(via_sibling=True, cosine=_cos(0.70, 0.76))
        assert _mod().routed_outcome(case.features, 'amended', _GUARD) == 'stored'
        within = _case(via_sibling=True, cosine=_cos(0.72, 0.76))
        assert _mod().routed_outcome(within.features, 'amended', _GUARD) == 'amended'

    def test_an_unrecovered_cosine_keeps_the_verdict(self) -> None:
        case = _case(cosine=None)
        assert _mod().routed_outcome(case.features, 'amended', _GUARD) == 'amended'

    def test_a_judged_route_needs_a_verdict_in_the_vocabulary(self) -> None:
        with pytest.raises(ValueError, match='distinct'):
            _mod().routed_outcome(_case().features, 'distinct', _GUARD)
        with pytest.raises(ValueError, match='None'):
            _mod().routed_outcome(_case().features, None, _GUARD)


class TestItemRow:
    @pytest.mark.parametrize('outcome', ['restated', 'amended', 'contested'])
    def test_an_attach_keeps_its_target(self, outcome: str) -> None:
        row = _mod().item_row(_case().features, outcome)
        assert row['outcome'] == outcome
        assert row['target_id'] == 'P1'
        assert row['earlier_ids'] == []
        assert row['superseded_marks'] == []
        _scorer().ReaderItem.from_row(0, row)

    def test_a_stored_write_lists_its_target_as_an_earlier_record(self) -> None:
        row = _mod().item_row(_case().features, 'stored')
        assert row['target_id'] is None
        assert row['earlier_ids'] == ['P1']
        assert row['superseded_marks'] == []
        assert row['supersedes_target'] is False
        _scorer().ReaderItem.from_row(0, row)

    def test_only_a_supersession_marks_its_target(self) -> None:
        row = _mod().item_row(_case(supersedes_target=True).features, 'stored')
        assert row['superseded_marks'] == ['P1']
        assert row['supersedes_target'] is False
        _scorer().ReaderItem.from_row(0, row)

    def test_the_row_keeps_the_items_facts_and_names_the_routes_band(self) -> None:
        judged = _mod().item_row(_case().features, 'amended')
        assert (judged['entry_id'], judged['project'], judged['stratum']) == (
            'E1', 'dark_factory', 'amended-sample',
        )
        assert (judged['via_sibling'], judged['cross_category']) == (False, False)
        assert judged['band'] == 'judge'
        verbatim = _mod().item_row(_case(texts=[_text(child=_PARENT)]).features, 'restated')
        assert verbatim['band'] == 'restated'
        forced = _mod().item_row(_case(metadata={'kind': 'note'}).features, 'stored')
        assert forced['band'] == 'stored'


class TestFeaturesRow:
    def test_carries_no_text(self) -> None:
        row = _case().features.to_row()
        assert 'child_text' not in row
        assert 'parent_text' not in row
        assert _CHILD not in str(row)

    def test_carries_the_route_cosines_recovery_and_text_lengths(self) -> None:
        row = _case(via_sibling=True, cosine=_cos(0.7, 0.9)).features.to_row()
        assert row['route'] == 'judge'
        assert (row['cos_parent'], row['source_cosine']) == (0.7, 0.9)
        assert row['metadata_recovered'] is True
        assert row['cosine_recovered'] is True
        assert (row['child_chars'], row['parent_chars']) == (len(_CHILD), len(_PARENT))

    def test_round_trips(self) -> None:
        features = _case(via_sibling=True, cosine=_cos(0.7, 0.9)).features
        assert type(features).from_row(features.to_row()) == features

    def test_a_case_round_trips_with_its_texts(self) -> None:
        case = _case()
        row = case.to_row()
        assert (row['child_text'], row['parent_text']) == (_CHILD, _PARENT)
        assert type(case).from_row(row) == case
