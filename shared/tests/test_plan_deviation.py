"""Tests for shared.plan_deviation — the per-step plan deviation vocabulary.

PRD ``plans/plan-deviation-recording-prd.md`` decision **D4**: a deviation
record's ``kind`` is a closed vocabulary that three packages consume (the
plan-tools server validates against it, ``roles.py`` renders the doctrine from
it, the nightly census reads it). ``DEVIATION_KINDS`` is that cross-package
contract, so it is pinned by set equality, the precedent of
``escalation/tests/test_models.py::test_resolution_classes_contains_exactly_the_legal_values``.
"""
from __future__ import annotations

import dataclasses
import json

import pytest

from shared.plan_deviation import DEVIATION_KINDS, DeviationRecord, UnknownDeviationKind

_RECORD_KEYS = {
    'kind',
    'observed',
    'expected',
    'action_taken',
    'prose_only',
    'note',
    'recorded_at',
    'recorded_by',
}


def _record(kind: str = 'other', **overrides) -> DeviationRecord:
    fields = {
        'kind': kind,
        'observed': 'The helper already existed.',
        'expected': 'The step said to write the helper.',
        'action_taken': 'Reused the existing helper.',
        'prose_only': False,
        'note': '',
        'recorded_at': '2026-10-15T07:00:00+00:00',
        'recorded_by': None,
        **overrides,
    }
    return DeviationRecord(**fields)


def test_deviation_kinds_is_a_frozenset():
    assert isinstance(DEVIATION_KINDS, frozenset)


def test_deviation_kinds_contains_exactly_the_legal_values():
    assert DEVIATION_KINDS == frozenset({
        'never_checked_fact',
        'wrong_inference',
        'plan_internal_contradiction',
        'inherited_from_task_text',
        'narrowing_refuted_by_measurement',
        'prediction_falsified',
        'other',
    })


@pytest.mark.parametrize('kind', sorted(DEVIATION_KINDS))
def test_every_kind_in_the_vocabulary_constructs(kind):
    assert _record(kind).kind == kind


def test_unknown_kind_is_refused_naming_the_value_and_the_vocabulary():
    with pytest.raises(UnknownDeviationKind) as caught:
        _record('typo')

    exc = caught.value
    assert isinstance(exc, ValueError)
    assert exc.kind == 'typo'
    assert exc.allowed == tuple(sorted(DEVIATION_KINDS))
    message = str(exc)
    assert 'typo' in message
    for kind in DEVIATION_KINDS:
        assert kind in message


@pytest.mark.parametrize('field', [f.name for f in dataclasses.fields(DeviationRecord)])
def test_record_is_frozen(field):
    record = _record()
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(record, field, 'replacement')


def test_to_dict_carries_exactly_the_eight_fields_and_survives_json():
    record = _record(
        'prediction_falsified',
        observed='Four tests failed, not two.',
        expected='Two tests fail until step-4.',
        action_taken='Greened all four in step-4.',
        prose_only=True,
        note='CHANGELOG claim adjusted.',
        recorded_by='claude-sess-1',
    )

    as_dict = record.to_dict()

    assert set(as_dict) == _RECORD_KEYS
    assert as_dict == {
        'kind': 'prediction_falsified',
        'observed': 'Four tests failed, not two.',
        'expected': 'Two tests fail until step-4.',
        'action_taken': 'Greened all four in step-4.',
        'prose_only': True,
        'note': 'CHANGELOG claim adjusted.',
        'recorded_at': '2026-10-15T07:00:00+00:00',
        'recorded_by': 'claude-sess-1',
    }
    assert json.loads(json.dumps(as_dict)) == as_dict
