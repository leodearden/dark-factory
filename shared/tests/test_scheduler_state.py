"""Tests for shared.scheduler_state: the scheduler snapshot's one typed shape.

The literal key set and record shapes below are the contract
(plans/review-all-s14-shared-leaves-prd.md, decision 12 and invariant I4): the
producer writes exactly these keys and every reader parses them.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from shared.scheduler_state import (
    OverrideRecord,
    ParkRecord,
    PinQueueEntry,
    RequeueCooldown,
    SchedulerStateSnapshot,
)

_SNAPSHOT_KEYS = {
    'skip_counts',
    'parks',
    'park_stacks',
    'pin_reservations',
    'effective_priorities',
    'pin_queue',
    'overrides',
    'current_holders',
    'lock_depth',
    'is_paused',
    'pause_reason',
    'requeue_cooldowns',
    'snapshot_at',
}

_INSTALLED_AT = '2026-10-15T12:00:00+00:00'

_PIN_STACK_ENTRY = {
    'owner': 'P',
    'rank': -1,
    'shadowed': False,
    'installed_at': _INSTALLED_AT,
    'source': 'pin',
    'pin_order': 1,
}
_FAIRNESS_STACK_ENTRY = {
    'owner': 'T3',
    'rank': 2,
    'shadowed': True,
    'installed_at': '',
    'source': 'fairness',
    'tier': 'medium',
}


def _populated() -> SchedulerStateSnapshot:
    return SchedulerStateSnapshot(
        skip_counts={'T3': 2},
        parks={'P': ParkRecord(modules=('pin/mod',), installed_at=_INSTALLED_AT)},
        park_stacks={'pin/mod': (_FAIRNESS_STACK_ENTRY, _PIN_STACK_ENTRY)},
        pin_reservations={'P': ParkRecord(modules=('pin/mod', 'pin/other'), installed_at=_INSTALLED_AT)},
        effective_priorities={'T3': 'high'},
        pin_queue=(PinQueueEntry(task_id='P', order=1),),
        overrides={
            'T1': OverrideRecord(
                boost_tier='high',
                pinned=True,
                reserve_now=True,
                ttl_until='2026-10-16T00:00:00+00:00',
            ),
        },
        current_holders={'mod/a': 'T7'},
        lock_depth=3,
        is_paused=True,
        pause_reason='operator halt',
        requeue_cooldowns={
            'T1': RequeueCooldown(
                armed_secs=31.25,
                envelope_secs=60.0,
                transient=True,
                transient_count=1,
                armed_at=_INSTALLED_AT,
                expires_at='2026-10-15T12:00:31.250000+00:00',
            ),
        },
        snapshot_at=datetime.now(UTC),
    )


class TestEmpty:
    def test_no_producer_has_written(self) -> None:
        assert SchedulerStateSnapshot.empty().snapshot_at is None

    def test_wire_form_has_exactly_the_contract_keys(self) -> None:
        assert set(SchedulerStateSnapshot.empty().to_json()) == _SNAPSHOT_KEYS

    def test_wire_form_values_are_empty(self) -> None:
        assert SchedulerStateSnapshot.empty().to_json() == {
            'skip_counts': {},
            'parks': {},
            'park_stacks': {},
            'pin_reservations': {},
            'effective_priorities': {},
            'pin_queue': [],
            'overrides': {},
            'current_holders': {},
            'lock_depth': None,
            'is_paused': False,
            'pause_reason': None,
            'requeue_cooldowns': {},
            'snapshot_at': None,
        }


class TestRoundTrip:
    def test_from_json_of_to_json_is_identity(self) -> None:
        snap = _populated()
        assert SchedulerStateSnapshot.from_json(snap.to_json()) == snap

    def test_round_trip_survives_a_json_text_hop(self) -> None:
        snap = _populated()
        text = json.dumps(snap.to_json())
        assert SchedulerStateSnapshot.from_json(json.loads(text)) == snap

    def test_populated_wire_form_has_exactly_the_contract_keys(self) -> None:
        assert set(_populated().to_json()) == _SNAPSHOT_KEYS


class TestWireShape:
    def test_snapshot_at_is_isoformat(self) -> None:
        snap = _populated()
        assert snap.snapshot_at is not None
        assert snap.to_json()['snapshot_at'] == snap.snapshot_at.isoformat()

    def test_park_record(self) -> None:
        assert _populated().to_json()['parks'] == {
            'P': {'modules': ['pin/mod'], 'installed_at': _INSTALLED_AT},
        }

    def test_pin_reservation_record(self) -> None:
        assert _populated().to_json()['pin_reservations'] == {
            'P': {'modules': ['pin/mod', 'pin/other'], 'installed_at': _INSTALLED_AT},
        }

    def test_pin_queue(self) -> None:
        assert _populated().to_json()['pin_queue'] == [{'task_id': 'P', 'order': 1}]

    def test_override_record(self) -> None:
        assert _populated().to_json()['overrides'] == {
            'T1': {
                'boost_tier': 'high',
                'pinned': True,
                'reserve_now': True,
                'ttl_until': '2026-10-16T00:00:00+00:00',
            },
        }

    def test_requeue_cooldown(self) -> None:
        assert _populated().to_json()['requeue_cooldowns'] == {
            'T1': {
                'armed_secs': 31.25,
                'envelope_secs': 60.0,
                'transient': True,
                'transient_count': 1,
                'armed_at': _INSTALLED_AT,
                'expires_at': '2026-10-15T12:00:31.250000+00:00',
            },
        }

    def test_park_stack_entries_pass_through_key_for_key(self) -> None:
        stack = _populated().to_json()['park_stacks']['pin/mod']
        assert stack == [_FAIRNESS_STACK_ENTRY, _PIN_STACK_ENTRY]
        assert 'tier' not in stack[1]

    def test_scalars(self) -> None:
        wire = _populated().to_json()
        assert wire['skip_counts'] == {'T3': 2}
        assert wire['effective_priorities'] == {'T3': 'high'}
        assert wire['current_holders'] == {'mod/a': 'T7'}
        assert wire['lock_depth'] == 3
        assert wire['is_paused'] is True
        assert wire['pause_reason'] == 'operator halt'


class TestRollingRestart:
    def test_unknown_keys_from_a_newer_producer_are_ignored(self) -> None:
        wire = _populated().to_json()
        newer = {
            **wire,
            'added_by_a_newer_producer': {'x': 1},
            'parks': {'P': {**wire['parks']['P'], 'added_nested': True}},
        }
        parsed = SchedulerStateSnapshot.from_json(newer)
        assert parsed == SchedulerStateSnapshot.from_json(wire)
        assert 'added_by_a_newer_producer' not in parsed.to_json()
        assert 'added_nested' not in parsed.to_json()['parks']['P']

    def test_older_producer_missing_keys_fill_with_defaults(self) -> None:
        parsed = SchedulerStateSnapshot.from_json({'current_holders': {'m': '7'}})
        assert parsed.snapshot_at is None
        assert parsed.to_json() == {
            **SchedulerStateSnapshot.empty().to_json(),
            'current_holders': {'m': '7'},
        }


class TestMalformedInput:
    @pytest.mark.parametrize('body', [[], 'x', None, 3])
    def test_non_mapping_body(self, body: Any) -> None:
        with pytest.raises(ValidationError):
            SchedulerStateSnapshot.from_json(body)

    def test_naive_snapshot_at(self) -> None:
        with pytest.raises(ValidationError):
            SchedulerStateSnapshot.from_json({'snapshot_at': '2026-01-01T00:00:00'})

    @pytest.mark.parametrize('depth', [True, '4', 4.0])
    def test_lock_depth_is_a_strict_int(self, depth: object) -> None:
        with pytest.raises(ValidationError):
            SchedulerStateSnapshot.from_json({'lock_depth': depth})

    def test_park_installed_at_must_be_a_string(self) -> None:
        with pytest.raises(ValidationError):
            SchedulerStateSnapshot.from_json(
                {'parks': {'P': {'modules': ['m'], 'installed_at': 12345}}},
            )

    def test_current_holder_must_be_a_string(self) -> None:
        with pytest.raises(ValidationError):
            SchedulerStateSnapshot.from_json({'current_holders': {'m': 7}})


class TestFrozen:
    def test_field_assignment_is_refused(self) -> None:
        snap = SchedulerStateSnapshot.empty()
        with pytest.raises(ValidationError):
            snap.is_paused = True
