"""The supersession record of a plan design decision (task 3865).

Only the public seam of ``orchestrator.plan_decisions`` is driven here.
"""
from __future__ import annotations

import copy

import pytest

from orchestrator.plan_decisions import (
    SUPERSEDED_VERSIONS_KEY,
    InvalidSupersession,
    supersede_decision,
)

_DECISIONS = [
    {'decision': 'Key the sweep on text', 'rationale': 'Titles are stable'},
    {'decision': 'Page the store', 'rationale': 'It is large'},
]
_NEW = {'decision': 'Key the sweep on metadata', 'rationale': 'Text keys drift'}
_REASON = 'titles were measured to drift'
_AT = '2026-11-28T05:00:00+00:00'


def _supersede(decisions, index=0, **overrides):
    arguments = {**_NEW, 'reason': _REASON, 'at': _AT, **overrides}
    return supersede_decision(decisions, index, **arguments)


def test_a_supersession_rewrites_the_entry_and_keeps_the_prior_pair():
    before = copy.deepcopy(_DECISIONS)

    after = _supersede(_DECISIONS, 0)

    assert after is not _DECISIONS
    assert after[0]['decision'] == _NEW['decision']
    assert after[0]['rationale'] == _NEW['rationale']
    assert after[0][SUPERSEDED_VERSIONS_KEY] == [
        {
            'decision': 'Key the sweep on text',
            'rationale': 'Titles are stable',
            'reason': _REASON,
            'superseded_at': _AT,
        },
    ]
    assert after[1] == _DECISIONS[1]
    assert before == _DECISIONS


def test_a_second_supersession_appends_oldest_first():
    once = _supersede(_DECISIONS, 1)
    twice = supersede_decision(
        once, 1, decision='Stream the store', rationale='Paging is slow', reason='paging measured slow',
        at='2026-11-29T09:30:00+00:00',
    )

    assert twice[1]['decision'] == 'Stream the store'
    assert twice[1][SUPERSEDED_VERSIONS_KEY] == [
        {'decision': 'Page the store', 'rationale': 'It is large', 'reason': _REASON, 'superseded_at': _AT},
        {
            'decision': _NEW['decision'],
            'rationale': _NEW['rationale'],
            'reason': 'paging measured slow',
            'superseded_at': '2026-11-29T09:30:00+00:00',
        },
    ]
    assert len(once[1][SUPERSEDED_VERSIONS_KEY]) == 1


def test_an_unknown_key_on_the_entry_survives():
    decisions = [{**_DECISIONS[0], 'origin': 'revalidation'}]

    [entry] = _supersede(decisions, 0)

    assert entry['origin'] == 'revalidation'


def test_changing_only_the_rationale_is_a_supersession():
    [entry, _] = _supersede(_DECISIONS, 0, decision=_DECISIONS[0]['decision'])

    assert entry['decision'] == _DECISIONS[0]['decision']
    assert entry['rationale'] == _NEW['rationale']
    assert len(entry[SUPERSEDED_VERSIONS_KEY]) == 1


def _refusal(decisions, index=0, **overrides) -> InvalidSupersession:
    before = copy.deepcopy(decisions)
    with pytest.raises(InvalidSupersession) as caught:
        _supersede(decisions, index, **overrides)
    assert decisions == before
    return caught.value


def _assert_names(exc: InvalidSupersession, field: str, value: object) -> None:
    assert exc.field == field
    assert exc.value == value
    assert repr(field) in str(exc)
    assert repr(value) in str(exc)


@pytest.mark.parametrize('field', ['decision', 'rationale', 'reason'])
@pytest.mark.parametrize('blank', ['', '   ', '\n\t'])
def test_blank_prose_is_refused(field, blank):
    exc = _refusal(_DECISIONS, 0, **{field: blank})

    _assert_names(exc, field, blank)


@pytest.mark.parametrize('index', [-1, 2, 7])
def test_an_out_of_range_index_is_refused_and_states_the_count(index):
    exc = _refusal(_DECISIONS, index)

    _assert_names(exc, 'decision_index', index)
    assert '2 design decisions' in str(exc)


@pytest.mark.parametrize('index', [True, False, '0', 0.0, None])
def test_a_non_int_index_is_refused(index):
    exc = _refusal(_DECISIONS, index)

    _assert_names(exc, 'decision_index', index)
    assert '2 design decisions' in str(exc)


def test_an_empty_list_has_no_decision_to_address():
    exc = _refusal([], 0)

    _assert_names(exc, 'decision_index', 0)
    assert '0 design decisions' in str(exc)


@pytest.mark.parametrize('decisions', [None, {'decision': 'x'}, 'decision'])
def test_a_plan_with_no_decision_list_is_refused(decisions):
    with pytest.raises(InvalidSupersession) as caught:
        _supersede(decisions, 0)

    _assert_names(caught.value, 'decision_index', 0)
    assert 'no design decision' in str(caught.value)


def test_restating_the_current_pair_is_refused():
    exc = _refusal(_DECISIONS, 1, **_DECISIONS[1])

    _assert_names(exc, 'decision', _DECISIONS[1]['decision'])
    assert 'nothing to supersede' in str(exc)


@pytest.mark.parametrize('entry', ['a bare string', None, ['decision', 'rationale']])
def test_a_stored_entry_that_is_not_a_dict_is_refused(entry):
    exc = _refusal([_DECISIONS[0], entry], 1)

    _assert_names(exc, 'design_decisions[1]', entry)


@pytest.mark.parametrize('key', ['decision', 'rationale'])
@pytest.mark.parametrize('stored', [None, 3, ['text']])
def test_a_stored_entry_whose_prose_is_not_a_str_is_refused(key, stored):
    exc = _refusal([{**_DECISIONS[0], key: stored}], 0)

    _assert_names(exc, f'design_decisions[0].{key}', stored)


@pytest.mark.parametrize('stored', ['one old version', {'decision': 'x'}, None])
def test_a_stored_history_that_is_not_a_list_is_refused_and_kept(stored):
    decisions = [{**_DECISIONS[0], SUPERSEDED_VERSIONS_KEY: stored}]

    exc = _refusal(decisions, 0)

    _assert_names(exc, f'design_decisions[0].{SUPERSEDED_VERSIONS_KEY}', stored)
    assert decisions[0][SUPERSEDED_VERSIONS_KEY] == stored
