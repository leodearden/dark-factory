"""Composed-path admission scenarios of plans/verify-admission-fifo-prd.md, task α.

The record carriers (the test CheckRun's ``admission``, VerifyResult's
``admission_legs`` and the three EventTypes), then the gate composed with the
REAL flock and real subprocess commands through ``run_verification``,
``run_scoped_verification`` and ``run_full_verification``.
"""

from __future__ import annotations

import dataclasses

from orchestrator.event_store import EventType
from orchestrator.verify import VerifyResult

_LEG = {
    'plan_id': 'p', 'module': 'pkg', 'scope_kind': None, 'order': 'arrival',
    'arrived_at': '2026-10-12T00:00:00+00:00', 'granted_at': '2026-10-12T00:00:01+00:00',
    'waited_secs': 1.0, 'depth_at_arrival': 0, 'slot_held': True, 'held_secs': 2.5,
    'rc': 0, 'timed_out': False,
}


def _result(**overrides) -> VerifyResult:
    return VerifyResult(
        passed=True, test_output='', lint_output='', type_output='', summary='All checks passed',
        **overrides,
    )


class TestRecordCarriers:
    def test_admission_legs_default_to_none(self):
        assert _result().admission_legs is None

    def test_admission_legs_do_not_take_part_in_equality(self):
        assert _result(admission_legs=[_LEG]) == _result(admission_legs=None)

    def test_admission_legs_round_trip_through_the_wire_codec(self):
        result = _result(admission_legs=[_LEG, {**_LEG, 'module': 'other'}])

        decoded = VerifyResult(**dataclasses.asdict(result))

        assert decoded.admission_legs == result.admission_legs

    def test_the_admission_event_types_exist(self):
        assert {
            EventType('verify_admission_plan'),
            EventType('verify_admission_heartbeat'),
            EventType('verify_admission_wait_aged'),
        }
