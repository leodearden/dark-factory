"""A record that gates MERGE entry also pins the stranded-blocked sweep (task 6145).

Two predicates read the same open records and must not disagree:

* ``orchestrator.workflow._is_gating_escalation`` decides whether a record
  stops the workflow at merge entry, which blocks the task;
* ``orchestrator.recovery_pins.records_pin_blocked_recovery``, as
  ``Scheduler._phase_redispatch_stranded_blocked`` consumes it with
  ``live_claimant=False``, decides whether that blocked task stays blocked.

A record that gates without pinning is a block -> re-pend -> re-dispatch loop:
the sweep re-pends the task, the re-dispatch reaches the same gate, and the
gate blocks it again.  The grid below states the ONE permitted disagreement,
a blocking L0, exactly, so any new gate-without-pin row fails here.
"""

from __future__ import annotations

import itertools

import pytest
from escalation.models import Escalation
from escalation.pins import needs_human

from orchestrator.recovery_pins import records_pin_blocked_recovery
from orchestrator.workflow import _is_gating_escalation

_TID = '6145'

_SEVERITIES = ('info', 'INFO', ' info ', 'blocking', 'critical', 'urgent', 'warn', '')
_LEVELS = (0, 1, 2, 3)
_CATEGORIES = ('stranded_blocked', 'preexisting_main_break', 'task_failure', 'design_concern')
_FILERS = (None, 'run-a/sess-a/pid=101')

_Row = tuple[str, int, str, str | None]


def _record(row: _Row, esc_id: str) -> Escalation:
    severity, level, category, filer = row
    return Escalation(
        id=esc_id,
        task_id=_TID,
        agent_role='harness-test',
        severity=severity,
        category=category,
        summary='grid',
        level=level,
        filing_claimant_run_id=filer,
    )


def _row_id(row: _Row) -> str:
    severity, level, category, filer = row
    return f'{severity!r}-L{level}-{category}-{"composed" if filer else "no-filer"}'


_GRID: list[_Row] = list(itertools.product(_SEVERITIES, _LEVELS, _CATEGORIES, _FILERS))
_GRID_RECORDS: dict[_Row, Escalation] = {
    row: _record(row, f'esc-{_TID}-{n}') for n, row in enumerate(_GRID, start=1)
}

#: The one intended gate-without-pin shape.  Its re-dispatch reaches the
#: merge-entry steward wait, which is real work, and every park exit of that
#: wait leaves a record that pins.
_PERMITTED: frozenset[_Row] = frozenset(
    ('blocking', 0, category, filer) for category in _CATEGORIES for filer in _FILERS
)

_NEEDS_HUMAN_ROWS: list[_Row] = [row for row in _GRID if needs_human(_GRID_RECORDS[row])]

_COMPANIONS: dict[str, list[Escalation]] = {
    'alone': [],
    'stranded-blocked-l1': [
        _record(('blocking', 1, 'stranded_blocked', None), f'esc-{_TID}-c1'),
    ],
    'info-l2': [_record(('info', 2, 'stranded_blocked', None), f'esc-{_TID}-c2')],
    'dead-blocking-l0': [
        _record(('blocking', 0, 'stranded_blocked', None), f'esc-{_TID}-c3'),
    ],
}


def _sweep_pins(records: list[Escalation]) -> bool:
    return records_pin_blocked_recovery(_TID, records, live_claimant=False)


def test_the_only_gate_without_pin_rows_are_blocking_l0s() -> None:
    gate_without_pin = {
        row
        for row, record in _GRID_RECORDS.items()
        if _is_gating_escalation(record) and not _sweep_pins([record])
    }

    unexpected = sorted(gate_without_pin - _PERMITTED, key=_row_id)
    missing = sorted(_PERMITTED - gate_without_pin, key=_row_id)
    assert gate_without_pin == _PERMITTED, (
        f'gate-without-pin rows outside the blocking-L0 set: '
        f'{[_row_id(row) for row in unexpected]}; '
        f'permitted rows that no longer disagree: {[_row_id(row) for row in missing]}'
    )


@pytest.mark.parametrize('row', sorted(_PERMITTED, key=_row_id), ids=_row_id)
def test_the_permitted_rows_never_need_a_human(row: _Row) -> None:
    """Why the permitted row is safe: a record no human holds routes the
    re-dispatch into the merge-entry steward wait, not a zero-work hand-off."""
    assert needs_human(_GRID_RECORDS[row]) is False


@pytest.mark.parametrize('companion', list(_COMPANIONS), ids=list(_COMPANIONS))
@pytest.mark.parametrize('row', _NEEDS_HUMAN_ROWS, ids=_row_id)
def test_a_record_that_needs_a_human_pins_whatever_sits_beside_it(
    row: _Row, companion: str,
) -> None:
    records = [_GRID_RECORDS[row], *_COMPANIONS[companion]]

    assert _sweep_pins(records) is True
