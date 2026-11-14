"""record_strand_conversion: the stamp and the event a convert-to-blocked leaves behind (task 6476)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from shared.strand_conversion import StrandConverted
from shared.task_statuses import TaskStatus

from orchestrator.event_store import EventStore, EventType
from orchestrator.strand_conversion import record_strand_conversion
from orchestrator.task_ground_truth import (
    BranchState,
    BranchStateKind,
    EscalationRef,
    TruthReport,
    recovery_shape_str,
)

_TID = 'T9'
_NOW = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)
_LOGGER = 'orchestrator.strand_conversion'

_HANDOFF = EscalationRef(
    id='esc-9-2', level=1, category='task_failure', severity='blocking',
    created_at='2026-10-13T11:00:00+00:00',
)
_DEAD_L0 = EscalationRef(
    id='esc-9-1', level=0, category='task_failure', severity='blocking',
    filing_claimant_run_id='run-x/sess-y/pid=1',
)
_INFO = EscalationRef(id='esc-9-3', level=1, category='design_concern', severity='info')


def _report(*escalations: EscalationRef) -> TruthReport:
    return TruthReport(
        db_status=TaskStatus.IN_PROGRESS,
        live_claimant=None,
        branch_state=BranchState(BranchStateKind.ON_MAIN, 'sha-on-main'),
        worktree_present=True,
        open_escalations=list(escalations),
        deploy_phase=None,
    )


_CONVERTING = _report(_HANDOFF, _DEAD_L0, _INFO)


@pytest.fixture
def store(tmp_path) -> EventStore:
    return EventStore(tmp_path / 'runs.db', 'run-test')


@pytest.fixture
def update_task() -> AsyncMock:
    return AsyncMock(return_value=True)


def _rows(store: EventStore) -> list[dict]:
    return store.fetch_events_by_type(EventType.strand_converted)


async def _record(
    store: EventStore | None, update_task: AsyncMock, report: TruthReport = _CONVERTING,
) -> StrandConverted | None:
    return await record_strand_conversion(
        _TID, report, event_store=store, update_task=update_task, now=_NOW,
    )


def _logged(caplog, level: int) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno >= level]


@pytest.mark.asyncio
async def test_stamps_the_row_once_with_only_the_queue_handoff_pin(store, update_task):
    await _record(store, update_task)

    update_task.assert_awaited_once()
    args, kwargs = update_task.await_args
    assert args[0] == _TID
    assert set(args[1]) == {'strand_converted'}
    assert kwargs == {'metadata_mode': 'merge'}
    stamp = StrandConverted.model_validate(args[1]['strand_converted'])
    assert stamp.pinning_escalation_ids == ('esc-9-2',)
    assert stamp.shape == recovery_shape_str(_CONVERTING)
    assert stamp.at == _NOW


@pytest.mark.asyncio
async def test_emits_one_event_derived_from_the_same_stamp(store, update_task):
    await _record(store, update_task)

    rows = _rows(store)
    assert len(rows) == 1
    assert rows[0]['task_id'] == _TID
    data = rows[0]['data']
    assert set(data) == {
        'task_id', 'shape', 'escalation_ids', 'pinning_escalation_ids',
        'ages_secs', 'measured_at', 'metadata_stamped',
    }
    assert data['task_id'] == _TID
    assert data['shape'] == recovery_shape_str(_CONVERTING)
    assert data['pinning_escalation_ids'] == ['esc-9-2']
    assert data['measured_at'] == _NOW.isoformat()
    assert data['escalation_ids'] == {
        'dead_l0': ['esc-9-1'], 'queue_handoff': ['esc-9-2'], 'non_pinning': ['esc-9-3'],
    }
    assert data['ages_secs'] == {'esc-9-2': 3600.0, 'esc-9-1': None, 'esc-9-3': None}
    assert data['metadata_stamped'] is True


@pytest.mark.asyncio
async def test_returns_the_stamp_it_recorded(store, update_task):
    stamp = await _record(store, update_task)

    assert stamp == StrandConverted.model_validate(update_task.await_args.args[1]['strand_converted'])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'refusal',
    [{'return_value': False}, {'side_effect': RuntimeError('boom')}],
    ids=['refused', 'raised'],
)
async def test_a_failed_stamp_warns_and_still_emits(store, caplog, refusal):
    caplog.set_level(logging.INFO, logger=_LOGGER)
    update_task = AsyncMock(**refusal)

    stamp = await _record(store, update_task)

    assert stamp is not None
    warnings = _logged(caplog, logging.WARNING)
    assert len(warnings) == 1
    assert _TID in warnings[0].getMessage()
    rows = _rows(store)
    assert len(rows) == 1
    assert rows[0]['data']['metadata_stamped'] is False


@pytest.mark.asyncio
async def test_without_an_event_store_the_stamp_is_still_written(update_task):
    stamp = await _record(None, update_task)

    assert stamp is not None
    update_task.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_conversion_nothing_pins_records_nothing_and_logs_an_error(
    store, update_task, caplog,
):
    caplog.set_level(logging.INFO, logger=_LOGGER)

    stamp = await _record(store, update_task, _report(_DEAD_L0))

    assert stamp is None
    update_task.assert_not_awaited()
    assert _rows(store) == []
    errors = _logged(caplog, logging.ERROR)
    assert len(errors) == 1
    assert _TID in errors[0].getMessage()
    assert 'pinning' in errors[0].getMessage()


@pytest.mark.asyncio
async def test_the_success_path_logs_nothing_at_warning_or_above(store, update_task, caplog):
    caplog.set_level(logging.INFO, logger=_LOGGER)

    await _record(store, update_task)

    assert _logged(caplog, logging.WARNING) == []
