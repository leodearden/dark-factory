"""Tests for reading a non-raising Mem0 add the one way every write path reads it."""

from __future__ import annotations

import dataclasses

import pytest

from fused_memory.models.enums import SourceStore
from fused_memory.services.mem0_add_outcome import (
    RAISED_MEM0_ADD,
    Mem0AddOutcome,
    Mem0LegResult,
    read_mem0_add_result,
)


def test_landed_result_claims_mem0_with_its_ids():
    outcome = read_mem0_add_result(
        {'results': [{'id': 'm-1'}]}, operation='add', zero_ids_anomalous=True
    )

    assert outcome.memory_ids == ('m-1',)
    assert outcome.empty_result_error is None
    assert outcome.landed is True
    assert outcome.stores_written == [SourceStore.mem0]


@pytest.mark.parametrize('operation', ['add', 'add_system_record'])
def test_anomalous_zero_ids_claims_nothing_and_names_the_operation(operation):
    outcome = read_mem0_add_result(
        {'results': []}, operation=operation, zero_ids_anomalous=True
    )

    assert outcome.memory_ids == ()
    assert outcome.landed is False
    assert outcome.stores_written == []
    assert outcome.empty_result_error == (
        f'empty_result: mem0 {operation} returned zero memory_ids'
    )


def test_legitimate_zero_ids_claims_nothing_without_a_diagnostic():
    outcome = read_mem0_add_result(
        {'results': []}, operation='add', zero_ids_anomalous=False
    )

    assert outcome.memory_ids == ()
    assert outcome.empty_result_error is None
    assert outcome.landed is False
    assert outcome.stores_written == []


@pytest.mark.parametrize(
    ('result', 'expected_ids'),
    [
        (None, ()),
        ({}, ()),
        ({'results': ['x', {'event': 'NONE'}, {'id': 'm-2'}]}, ('m-2',)),
    ],
    ids=['none', 'no_results_key', 'mixed_entries'],
)
def test_malformed_entries_do_not_count_as_ids(result, expected_ids):
    outcome = read_mem0_add_result(result, operation='add', zero_ids_anomalous=True)

    assert outcome.memory_ids == expected_ids


def test_raised_add_claims_nothing_and_leaves_the_diagnostic_to_the_exception():
    assert RAISED_MEM0_ADD.memory_ids == ()
    assert RAISED_MEM0_ADD.empty_result_error is None
    assert RAISED_MEM0_ADD.landed is False
    assert RAISED_MEM0_ADD.stores_written == []


def test_a_diagnostic_beside_ids_is_rejected_naming_both_values():
    with pytest.raises(ValueError) as excinfo:
        Mem0AddOutcome(memory_ids=('m-1',), empty_result_error='x')

    message = str(excinfo.value)
    assert "'m-1'" in message
    assert "'x'" in message


def test_outcome_is_frozen():
    outcome = read_mem0_add_result(
        {'results': [{'id': 'm-1'}]}, operation='add', zero_ids_anomalous=True
    )

    for field in dataclasses.fields(outcome):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(outcome, field.name, None)


_LANDED = Mem0AddOutcome(memory_ids=('m1',), empty_result_error=None)
_EMPTY = Mem0AddOutcome(
    memory_ids=(), empty_result_error='empty_result: mem0 add returned zero memory_ids'
)


@pytest.mark.parametrize(
    'leg',
    [
        Mem0LegResult(_LANDED),
        Mem0LegResult(_EMPTY),
        Mem0LegResult(RAISED_MEM0_ADD, error='boom'),
        Mem0LegResult(RAISED_MEM0_ADD, pending_retry_reason='rate_limited: RateLimitError'),
    ],
    ids=['landed', 'empty-result', 'raised', 'deferred'],
)
def test_a_consistent_leg_constructs(leg):
    assert isinstance(leg, Mem0LegResult)


def test_a_leg_cannot_both_fail_and_be_deferred():
    with pytest.raises(ValueError) as excinfo:
        Mem0LegResult(RAISED_MEM0_ADD, error='boom', pending_retry_reason='rate_limited')

    message = str(excinfo.value)
    assert "'boom'" in message
    assert "'rate_limited'" in message


@pytest.mark.parametrize(
    'raised_field', [{'error': 'boom'}, {'pending_retry_reason': 'rate_limited'}]
)
def test_a_landed_leg_cannot_carry_a_raise(raised_field):
    with pytest.raises(ValueError, match="'m1'"):
        Mem0LegResult(_LANDED, **raised_field)


@pytest.mark.parametrize(
    'raised_field', [{'error': 'boom'}, {'pending_retry_reason': 'rate_limited'}]
)
def test_a_raised_leg_is_exactly_the_raised_outcome(raised_field):
    with pytest.raises(ValueError, match='empty_result'):
        Mem0LegResult(_EMPTY, **raised_field)


@pytest.mark.parametrize(
    ('leg', 'pending'),
    [
        (Mem0LegResult(_LANDED), False),
        (Mem0LegResult(RAISED_MEM0_ADD, error='boom'), False),
        (Mem0LegResult(RAISED_MEM0_ADD, pending_retry_reason='rate_limited'), True),
    ],
    ids=['landed', 'raised', 'deferred'],
)
def test_only_a_deferred_leg_is_pending_retry(leg, pending):
    assert leg.pending_retry is pending
