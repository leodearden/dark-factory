"""The coordinator's settle path (plans/live-shadow-eval-prd.md C2/C4, rows 4 and 5).

A cell settles exactly once: on a landing, on a non-landing production
terminal, or by a late-reference catch-up when its run completes after
production's outcome is already recorded.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest
import pytest_asyncio
from _orch_helpers import wait_responsive
from _shadow_coordinator_doubles import (
    PROJECT_ID,
    RUN_ID,
    START,
    CoordinatorRig,
    build_coordinator,
    completed,
    make_task,
)
from shared.cost_store import CostStore

from orchestrator.event_store import EventType
from orchestrator.harness import TaskReport
from orchestrator.run_store import ShadowCell, ShadowCellQuery
from orchestrator.shadow_scoring import ShadowReference, score_implementer_cell
from orchestrator.shadow_types import ReferenceKind, ShadowCellState, ShadowShape
from orchestrator.workflow_types import WorkflowOutcome, WorkflowState

_S = ShadowCellState
_SHA = 'f' * 40
_CANDIDATE = 'claude-sonnet-max'


@pytest_asyncio.fixture
async def costs(tmp_path):
    async with CostStore(tmp_path / 'runs.db') as store:
        yield store


@pytest_asyncio.fixture
async def rig(tmp_path, costs) -> CoordinatorRig:
    return build_coordinator(tmp_path, costs)


def _cells(rig: CoordinatorRig, task_id: str | None = None) -> list[ShadowCell]:
    return rig.store.find_shadow_cells(ShadowCellQuery(project_id=PROJECT_ID, task_id=task_id))


async def _awaiting_cell(rig: CoordinatorRig, task_id: str = '5387') -> ShadowCell:
    [cell_id] = await rig.coordinator.on_phase(make_task(task_id), WorkflowState.PLAN, entering=False)
    await rig.coordinator.join()
    cell = rig.store.get_shadow_cell(cell_id)
    assert cell is not None
    assert cell.state is _S.AWAITING_REFERENCE
    return cell


def _merge_payload(task_id: str, state: str, merge_sha: str | None = _SHA) -> dict:
    return {
        'request_id': f'req-{task_id}',
        'task_id': task_id,
        'branch': f'task/{task_id}',
        'state': state,
        'merge_sha': merge_sha,
        'superseded_by': None,
    }


def _settled_events(rig: CoordinatorRig) -> list[dict]:
    return rig.events_of(EventType.shadow_cell_settled)


@pytest.mark.parametrize('state', ['done', 'done_wip_recovery'])
@pytest.mark.asyncio
async def test_a_landing_settles_the_awaiting_cell_landed(rig, state):
    cell = await _awaiting_cell(rig)
    rig.clock.advance(timedelta(hours=2))

    settled = await rig.coordinator.on_merge_finalized(_merge_payload('5387', state))

    assert settled == [cell.cell_id]
    row = rig.store.get_shadow_cell(cell.cell_id)
    assert row is not None
    assert row.state is _S.SETTLED
    assert (row.reference_kind, row.reference_sha) == (ReferenceKind.LANDED, _SHA)
    assert row.settled_at == rig.clock.now()
    [event] = _settled_events(rig)
    assert event['task_id'] == '5387'
    assert event['data'] == {
        'cell_id': cell.cell_id,
        'shape': ShadowShape.IMPLEMENTER.value,
        'candidate': _CANDIDATE,
        'reference_kind': 'landed',
        'reference_sha': _SHA,
    }


@pytest.mark.asyncio
async def test_a_cell_settles_exactly_once(rig):
    cell = await _awaiting_cell(rig)
    await rig.coordinator.on_merge_finalized(_merge_payload('5387', 'done'))
    first = rig.store.get_shadow_cell(cell.cell_id)
    rig.clock.advance(timedelta(hours=1))

    again = await rig.coordinator.on_merge_finalized(_merge_payload('5387', 'done'))

    assert again == []
    assert rig.store.get_shadow_cell(cell.cell_id) == first
    assert len(_settled_events(rig)) == 1


@pytest.mark.parametrize(
    'state,merge_sha',
    [
        ('conflict', None),
        ('blocked', None),
        ('superseded', None),
        ('abandoned', None),
        ('error', None),
        ('already_merged', None),
        ('done', None),
        ('done', '  '),
    ],
)
@pytest.mark.asyncio
async def test_a_non_landing_merge_state_settles_nothing(rig, state, merge_sha):
    cell = await _awaiting_cell(rig)

    settled = await rig.coordinator.on_merge_finalized(_merge_payload('5387', state, merge_sha))

    assert settled == []
    row = rig.store.get_shadow_cell(cell.cell_id)
    assert row is not None
    assert row.state is _S.AWAITING_REFERENCE
    assert _settled_events(rig) == []


@pytest.mark.parametrize(
    'payload',
    [
        {'state': 'done', 'merge_sha': _SHA},
        {'task_id': '', 'state': 'done', 'merge_sha': _SHA},
        {'task_id': '5387', 'merge_sha': _SHA},
        {'task_id': 5387, 'state': 'done', 'merge_sha': _SHA},
    ],
    ids=['no-task-id', 'blank-task-id', 'no-state', 'non-str-task-id'],
)
@pytest.mark.asyncio
async def test_a_malformed_payload_settles_nothing_and_never_raises(rig, payload):
    cell = await _awaiting_cell(rig)

    assert await rig.coordinator.on_merge_finalized(payload) == []
    row = rig.store.get_shadow_cell(cell.cell_id)
    assert row is not None
    assert row.state is _S.AWAITING_REFERENCE


@pytest.mark.parametrize('outcome', [WorkflowOutcome.CANCELLED, WorkflowOutcome.BLOCKED])
@pytest.mark.asyncio
async def test_a_production_terminal_without_a_landing_settles_none(rig, outcome):
    cell = await _awaiting_cell(rig)

    settled = await rig.coordinator.on_production_terminal('5387', outcome)

    assert settled == [cell.cell_id]
    row = rig.store.get_shadow_cell(cell.cell_id)
    assert row is not None
    assert row.state is _S.SETTLED
    assert (row.reference_kind, row.reference_sha) == (ReferenceKind.NONE, None)
    assert row.settled_at == rig.clock.now()
    metrics = rig.runner.default.metrics
    assert metrics is not None
    assert score_implementer_cell(metrics, ShadowReference.none()).composite_score is None
    [event] = _settled_events(rig)
    assert (event['data']['reference_kind'], event['data']['reference_sha']) == ('none', None)


@pytest.mark.parametrize(
    'outcome',
    [
        WorkflowOutcome.DONE,
        WorkflowOutcome.REQUEUED,
        WorkflowOutcome.SOFT_CANCELLED,
        WorkflowOutcome.MERGE_DEFERRED,
        WorkflowOutcome.ESCALATED,
        WorkflowOutcome.PLANNED,
    ],
)
@pytest.mark.asyncio
async def test_other_production_outcomes_settle_nothing(rig, outcome):
    cell = await _awaiting_cell(rig)

    assert await rig.coordinator.on_production_terminal('5387', outcome) == []
    row = rig.store.get_shadow_cell(cell.cell_id)
    assert row is not None
    assert row.state is _S.AWAITING_REFERENCE


@pytest.mark.asyncio
async def test_a_running_cell_is_never_settled(rig):
    release = asyncio.Event()
    rig.runner.script(ShadowShape.IMPLEMENTER, _CANDIDATE, block=release)
    [cell_id] = await rig.coordinator.on_phase(make_task('5387'), WorkflowState.PLAN, entering=False)
    await wait_responsive(rig.runner.run_started.wait(), label='the shadow implementer run starts')

    assert await rig.coordinator.on_production_terminal('5387', WorkflowOutcome.CANCELLED) == []
    row = rig.store.get_shadow_cell(cell_id)
    assert row is not None
    assert row.state is _S.RUNNING

    release.set()
    await rig.coordinator.join()


@pytest.mark.asyncio
async def test_a_failed_cell_is_never_settled(rig):
    rig.runner.script(ShadowShape.IMPLEMENTER, _CANDIDATE, run_raises=RuntimeError('boom'))
    [cell_id] = await rig.coordinator.on_phase(make_task('5387'), WorkflowState.PLAN, entering=False)
    await rig.coordinator.join()

    assert await rig.coordinator.on_merge_finalized(_merge_payload('5387', 'done')) == []
    row = rig.store.get_shadow_cell(cell_id)
    assert row is not None
    assert row.state is _S.FAILED
    assert row.settled_at is None


@pytest.mark.asyncio
async def test_a_landing_recorded_while_the_cell_ran_settles_it_at_completion(rig):
    def land_production(request) -> None:
        rig.events.emit(
            EventType.merge_finalized,
            task_id=request.cell.task_id,
            phase='merge',
            data=_merge_payload(request.cell.task_id, 'done'),
        )

    rig.runner.script(
        ShadowShape.IMPLEMENTER, _CANDIDATE, outcome=completed(), before_return=land_production,
    )

    [cell_id] = await rig.coordinator.on_phase(make_task('5387'), WorkflowState.PLAN, entering=False)
    await rig.coordinator.join()

    row = rig.store.get_shadow_cell(cell_id)
    assert row is not None
    assert row.state is _S.SETTLED
    assert (row.reference_kind, row.reference_sha) == (ReferenceKind.LANDED, _SHA)
    assert len(_settled_events(rig)) == 1


def _record_production_outcome(rig: CoordinatorRig, task_id: str, completed_at) -> None:
    rig.store.start_run(RUN_ID, PROJECT_ID, START.isoformat())
    rig.store.save_task_result(
        RUN_ID,
        TaskReport(
            task_id=task_id,
            title='t',
            outcome=WorkflowOutcome.CANCELLED,
            completed_at=completed_at.isoformat(),
        ),
        PROJECT_ID,
    )


@pytest.mark.asyncio
async def test_a_terminal_recorded_while_the_cell_ran_settles_it_none_at_completion(rig):
    rig.runner.script(
        ShadowShape.IMPLEMENTER, _CANDIDATE, outcome=completed(),
        before_return=lambda request: _record_production_outcome(
            rig, request.cell.task_id, START + timedelta(hours=1),
        ),
    )

    [cell_id] = await rig.coordinator.on_phase(make_task('5387'), WorkflowState.PLAN, entering=False)
    await rig.coordinator.join()

    row = rig.store.get_shadow_cell(cell_id)
    assert row is not None
    assert row.state is _S.SETTLED
    assert (row.reference_kind, row.reference_sha) == (ReferenceKind.NONE, None)


@pytest.mark.asyncio
async def test_a_terminal_recorded_before_the_cell_opened_is_ignored(rig):
    _record_production_outcome(rig, '5387', START - timedelta(hours=1))

    cell = await _awaiting_cell(rig)

    assert cell.state is _S.AWAITING_REFERENCE
    assert _settled_events(rig) == []


@pytest.mark.asyncio
async def test_settling_one_task_never_touches_another(rig):
    cell_t = await _awaiting_cell(rig, '5387')
    cell_u = await _awaiting_cell(rig, '5388')

    assert await rig.coordinator.on_merge_finalized(_merge_payload('5387', 'done')) == [cell_t.cell_id]
    assert await rig.coordinator.on_production_terminal('5387', WorkflowOutcome.CANCELLED) == []

    row_u = rig.store.get_shadow_cell(cell_u.cell_id)
    assert row_u is not None
    assert row_u.state is _S.AWAITING_REFERENCE
    assert [c.state for c in _cells(rig, '5387')] == [_S.SETTLED]
