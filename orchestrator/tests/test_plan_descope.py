"""The done-step integrity invariant and descope record (task 4032, β).

``plans/task-amendment-delivery-prd.md`` C4 (the descope record) and C5 (the
invariant a revalidated plan must satisfy). Only the public functions of
``orchestrator.plan_descope`` are driven here.
"""
from __future__ import annotations

import json

import pytest

from orchestrator.plan_descope import (
    DESCOPE_DISPOSITIONS,
    DESCOPED_STEPS_KEY,
    RESCOPED_AT_KEY,
    RESCOPED_BY_KEY,
    RESCOPED_NOTE_KEY,
    DescopedStep,
    InvalidDescopeRecord,
    check_done_step_integrity,
    commits_match,
    descope_record_of,
    done_step_items,
    done_step_records,
    same_done_record,
)


def _step(step_id: str, status: str = 'pending', commit: str | None = None) -> dict:
    item: dict = {'id': step_id, 'description': f'{step_id} work', 'status': status}
    if commit is not None:
        item['commit'] = commit
    return item


def _plan(steps: list[dict], prerequisites: list[dict] | None = None, **extra) -> dict:
    return {'steps': steps, 'prerequisites': prerequisites or [], **extra}


def _descope(step_id: str, commit: str, **overrides) -> dict:
    entry = {
        'step_id': step_id,
        'commit': commit,
        'disposition': 'orphan',
        'authority': 'esc-2971-6',
        'note': 'superseded by the ratified code-only scope',
    }
    entry.update(overrides)
    return entry


# --- (a) the 2971 replay ---------------------------------------------------

_BEFORE_2971 = _plan([_step('step-1', 'done', '309c22ab6f'), _step('step-2', 'pending')])


def test_2971_replay_recorded_drop_passes():
    after = _plan(
        [_step('step-1', 'done', 'c0609049a7')],
        **{DESCOPED_STEPS_KEY: [_descope('step-1', '309c22ab6f')]},
    )
    verdict = check_done_step_integrity(_BEFORE_2971, after)
    assert verdict.passed
    assert verdict.undisposed_drops == ()


def test_2971_replay_without_record_is_an_undisposed_drop_though_count_holds():
    after = _plan([_step('step-1', 'done', 'c0609049a7')])
    verdict = check_done_step_integrity(_BEFORE_2971, after)
    assert not verdict.passed
    assert verdict.undisposed_drops == (('step-1', '309c22ab6f'),)


# --- (b) the 3143 replay ---------------------------------------------------

_BEFORE_3143 = _plan([_step(f'step-{n}', 'done', f'{n:02d}' + 'f' * 8) for n in range(1, 20)])


def _records_3143() -> list[dict]:
    return [
        _descope(item['id'], item['commit'], authority='esc-3143-4')
        for item in _BEFORE_3143['steps']
    ]


def test_3143_replay_all_nineteen_recorded_passes():
    after = _plan([_step('step-1', 'pending')], **{DESCOPED_STEPS_KEY: _records_3143()})
    assert check_done_step_integrity(_BEFORE_3143, after).passed


def test_3143_replay_one_record_missing_names_exactly_that_pair():
    records = _records_3143()
    missing = records.pop(6)
    after = _plan([_step('step-1', 'pending')], **{DESCOPED_STEPS_KEY: records})
    verdict = check_done_step_integrity(_BEFORE_3143, after)
    assert not verdict.passed
    assert verdict.undisposed_drops == ((missing['step_id'], missing['commit']),)


# --- (c) the silent create_plan wipe ---------------------------------------

def test_silent_wipe_to_pending_reports_every_before_pair():
    before = _plan([
        _step('step-1', 'done', 'aaaaaaa111'),
        _step('step-2', 'done', 'bbbbbbb222'),
        _step('step-3', 'pending'),
    ])
    after = _plan([_step('step-1'), _step('step-2'), _step('step-3')])
    verdict = check_done_step_integrity(before, after)
    assert not verdict.passed
    assert set(verdict.undisposed_drops) == {
        ('step-1', 'aaaaaaa111'),
        ('step-2', 'bbbbbbb222'),
    }


# --- (d) invalid descope entries do not dispose a drop ---------------------

@pytest.mark.parametrize(
    ('override', 'field'),
    [
        ({'disposition': 'delete'}, 'disposition'),
        ({'authority': '   '}, 'authority'),
    ],
)
def test_invalid_descope_entry_does_not_dispose_and_is_reported(override, field):
    after = _plan(
        [_step('step-1', 'done', 'c0609049a7')],
        **{DESCOPED_STEPS_KEY: [_descope('step-1', '309c22ab6f', **override)]},
    )
    verdict = check_done_step_integrity(_BEFORE_2971, after)
    assert not verdict.passed
    assert verdict.undisposed_drops == (('step-1', '309c22ab6f'),)
    assert len(verdict.invalid_descope_records) == 1
    assert field in verdict.invalid_descope_records[0]


# --- (e) prefix equivalence ------------------------------------------------

_SHORT = 'abcdef1234567'
_FULL = _SHORT + '0' * (40 - len(_SHORT))


def test_full_sha_re_mark_of_a_recorded_prefix_is_kept():
    before = _plan([_step('step-1', 'done', _SHORT)])
    after = _plan([_step('step-1', 'done', _FULL)])
    assert check_done_step_integrity(before, after).passed


def test_commits_match_prefix_floor():
    assert commits_match(_SHORT, _FULL)
    assert commits_match(_FULL.upper(), _SHORT)
    assert not commits_match('abcd', _FULL)
    assert not commits_match('abcdef1', 'abcdef2')


def test_four_char_prefix_does_not_keep_the_step():
    before = _plan([_step('step-1', 'done', 'abcd')])
    after = _plan([_step('step-1', 'done', _FULL)])
    verdict = check_done_step_integrity(before, after)
    assert verdict.undisposed_drops == (('step-1', 'abcd'),)


def test_same_done_record_needs_the_step_id_and_a_commit_match():
    assert same_done_record(('step-1', _SHORT), ('step-1', _FULL))
    assert not same_done_record(('step-1', _SHORT), ('step-2', _FULL))
    assert not same_done_record(('step-1', 'abcd'), ('step-1', _FULL))


# --- (f) duplicates ---------------------------------------------------------

def test_introduced_duplicate_commit_is_refused_naming_both_steps():
    before = _plan([_step('step-1', 'done', 'aaaaaaa111'), _step('step-2')])
    after = _plan([_step('step-1', 'done', 'aaaaaaa111'), _step('step-2', 'done', 'aaaaaaa111')])
    verdict = check_done_step_integrity(before, after)
    assert not verdict.passed
    assert len(verdict.introduced_duplicates) == 1
    duplicate = verdict.introduced_duplicates[0]
    assert duplicate.commit == 'aaaaaaa111'
    assert set(duplicate.step_ids) == {'step-1', 'step-2'}


def test_pre_existing_shared_commit_is_grandfathered():
    before = _plan(
        [_step('step-1', 'done', 'aaaaaaa111')],
        prerequisites=[_step('pre-1', 'done', 'aaaaaaa111')],
    )
    after = json.loads(json.dumps(before))
    verdict = check_done_step_integrity(before, after)
    assert verdict.passed
    assert verdict.introduced_duplicates == ()


# --- (g) prerequisites are done records like steps -------------------------

def test_prerequisites_and_steps_are_both_done_records():
    plan = _plan(
        [_step('step-1', 'done', 'bbbbbbb222'), _step('step-2', 'done'), _step('step-3', 'pending', 'c' * 10)],
        prerequisites=[_step('pre-1', 'done', 'aaaaaaa111')],
    )
    assert set(done_step_records(plan)) == {('pre-1', 'aaaaaaa111'), ('step-1', 'bbbbbbb222')}


def test_done_step_items_are_the_plans_own_dicts():
    plan = _plan(
        [_step('step-1', 'done', 'bbbbbbb222'), _step('step-2', 'done')],
        prerequisites=[_step('pre-1', 'done', 'aaaaaaa111')],
    )
    items = done_step_items(plan)
    assert [item['id'] for item in items] == ['pre-1', 'step-1']
    assert items[0] is plan['prerequisites'][0]
    assert items[1] is plan['steps'][0]


def test_done_step_items_tolerate_a_null_column():
    assert done_step_items({'steps': None, 'prerequisites': None}) == []


def test_dropped_prerequisite_is_undisposed():
    before = _plan([], prerequisites=[_step('pre-1', 'done', 'aaaaaaa111')])
    after = _plan([], prerequisites=[_step('pre-1', 'pending')])
    verdict = check_done_step_integrity(before, after)
    assert verdict.undisposed_drops == (('pre-1', 'aaaaaaa111'),)


# --- (h) the verdict serialises ---------------------------------------------

def test_verdict_to_dict_is_json_serialisable_with_pair_mappings():
    before = _plan([_step('step-1', 'done', 'aaaaaaa111'), _step('step-2')])
    after = _plan(
        [_step('step-2', 'done', 'ccccccc333'), _step('step-3', 'done', 'ccccccc333')],
        **{DESCOPED_STEPS_KEY: [_descope('step-9', 'ddddddd444', disposition='delete')]},
    )
    data = check_done_step_integrity(before, after).to_dict()
    round_tripped = json.loads(json.dumps(data))
    assert round_tripped['undisposed_drops'] == [{'step_id': 'step-1', 'commit': 'aaaaaaa111'}]
    assert round_tripped['introduced_duplicates'][0]['commit'] == 'ccccccc333'
    assert set(round_tripped['introduced_duplicates'][0]['step_ids']) == {'step-2', 'step-3'}
    assert len(round_tripped['invalid_descope_records']) == 1


def test_verdict_reason_is_one_line():
    verdict = check_done_step_integrity(_BEFORE_2971, _plan([]))
    assert '\n' not in verdict.reason()
    assert verdict.reason()


# --- (i) the record validates itself ---------------------------------------

def test_out_of_vocabulary_disposition_names_field_value_and_allowed_set():
    with pytest.raises(InvalidDescopeRecord) as excinfo:
        DescopedStep(**_descope('step-1', '309c22ab6f', disposition='delete'))
    message = str(excinfo.value)
    assert excinfo.value.field == 'disposition'
    assert excinfo.value.value == 'delete'
    assert 'disposition' in message
    assert 'delete' in message
    for allowed in DESCOPE_DISPOSITIONS:
        assert allowed in message


@pytest.mark.parametrize('field', ['step_id', 'commit', 'authority', 'note'])
def test_blank_required_field_is_refused(field):
    entry = {**_descope('step-1', '309c22ab6f'), field: ' '}
    with pytest.raises(InvalidDescopeRecord) as excinfo:
        DescopedStep(**entry)
    assert excinfo.value.field == field
    assert field in str(excinfo.value)


def test_from_mapping_round_trips_and_refuses_a_non_mapping():
    entry = _descope('step-1', '309c22ab6f')
    assert DescopedStep.from_mapping(entry).to_dict() == entry
    with pytest.raises(InvalidDescopeRecord):
        DescopedStep.from_mapping('step-1')


@pytest.mark.parametrize('field', ['step_id', 'commit', 'disposition', 'authority', 'note'])
def test_from_mapping_names_a_missing_field(field):
    entry = _descope('step-1', '309c22ab6f')
    del entry[field]
    with pytest.raises(InvalidDescopeRecord) as excinfo:
        DescopedStep.from_mapping(entry)
    assert excinfo.value.field == field
    assert excinfo.value.value is None


def test_dispositions_are_the_closed_vocabulary():
    assert frozenset({'orphan', 'revert', 'ride-along'}) == DESCOPE_DISPOSITIONS


# --- (j) descope_record_of --------------------------------------------------

def test_descope_record_of_returns_exactly_the_present_keys():
    plan = _plan(
        [_step('step-1')],
        **{
            RESCOPED_AT_KEY: '2026-10-30T00:00:00+00:00',
            RESCOPED_NOTE_KEY: 'why',
            DESCOPED_STEPS_KEY: [_descope('step-1', '309c22ab6f')],
            '_session_id': 'unrelated',
        },
    )
    assert descope_record_of(plan) == {
        RESCOPED_AT_KEY: '2026-10-30T00:00:00+00:00',
        RESCOPED_NOTE_KEY: 'why',
        DESCOPED_STEPS_KEY: [_descope('step-1', '309c22ab6f')],
    }
    assert RESCOPED_BY_KEY not in descope_record_of(plan)


def test_descope_record_of_a_plan_without_the_keys_is_empty():
    assert descope_record_of(_plan([_step('step-1')])) == {}
