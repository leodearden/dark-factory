"""Expiry and restart reconciliation (plans/live-shadow-eval-prd.md C1/C2, row 6, INV-6/7/8).

No cell is left open: an awaiting cell past its deadline expires through a
bounded sweep that names what it deferred, and a restart fails every held cell
whose run is gone.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
import pytest_asyncio
from _shadow_coordinator_doubles import (
    INCUMBENTS,
    PROJECT_ID,
    RUN_ID,
    START,
    CoordinatorRig,
    build_coordinator,
    make_task,
)
from shared.cost_store import CostStore

from orchestrator.event_store import EventType
from orchestrator.run_store import ShadowCell, ShadowCellQuery
from orchestrator.shadow_eval import EXPIRE_BATCH_LIMIT, ExpireSweep
from orchestrator.shadow_types import CostBasis, PlanSource, ShadowCellState, ShadowShape
from orchestrator.workflow_types import WorkflowState

_S = ShadowCellState
_HELD = frozenset(state for state in ShadowCellState if state.held_by_run)


@pytest_asyncio.fixture
async def costs(tmp_path):
    async with CostStore(tmp_path / 'runs.db') as store:
        yield store


@pytest_asyncio.fixture
async def rig(tmp_path, costs) -> CoordinatorRig:
    return build_coordinator(tmp_path, costs)


def _seeded(index: int, *, state: ShadowCellState, run_id: str | None = 'run-dead') -> ShadowCell:
    opened_at = START - timedelta(hours=200) + timedelta(minutes=index)
    return ShadowCell(
        cell_id=f'01JSEED{index:03d}',
        project_id=PROJECT_ID,
        task_id=f'task-{index:03d}',
        shape=ShadowShape.IMPLEMENTER,
        candidate='claude-sonnet-max',
        incumbent=INCUMBENTS[ShadowShape.IMPLEMENTER],
        stratum=('df', 'feature', 'full'),
        base_sha='a' * 40,
        state=state,
        cost_basis=CostBasis.METERED,
        plan_source=PlanSource.PRODUCTION,
        run_id=run_id,
        opened_at=opened_at,
        settle_deadline=opened_at + timedelta(hours=168),
    )


def _row(rig: CoordinatorRig, cell_id: str) -> ShadowCell:
    cell = rig.store.get_shadow_cell(cell_id)
    assert cell is not None
    return cell


@pytest.mark.asyncio
async def test_an_overdue_awaiting_cell_expires_and_is_never_reopened(rig):
    task = make_task('5387')
    [cell_id] = await rig.coordinator.on_phase(task, WorkflowState.PLAN, entering=False)
    await rig.coordinator.join()
    deadline = _row(rig, cell_id).settle_deadline
    rig.clock.advance(deadline - rig.clock.now() + timedelta(seconds=1))
    now = rig.clock.now()

    sweep = await rig.coordinator.expire(now)

    assert sweep == ExpireSweep(expired=(cell_id,), deferred=())
    row = _row(rig, cell_id)
    assert row.state is _S.EXPIRED
    assert row.reason == 'settle_deadline'
    assert row.settled_at == now
    [event] = rig.events_of(EventType.shadow_cell_settled)
    assert event['data']['cell_id'] == cell_id
    assert (event['data']['state'], event['data']['reason']) == ('expired', 'settle_deadline')

    landing = {'task_id': '5387', 'state': 'done', 'merge_sha': 'f' * 40}
    assert await rig.coordinator.on_merge_finalized(landing) == []
    assert await rig.coordinator.on_phase(task, WorkflowState.PLAN, entering=False) == []
    skipped = rig.events_of(EventType.shadow_cell_skipped)
    assert [row['data']['reason'] for row in skipped] == ['already_open']
    assert len(rig.store.find_shadow_cells(ShadowCellQuery(project_id=PROJECT_ID))) == 1


@pytest.mark.asyncio
async def test_a_cell_inside_its_deadline_is_untouched(rig):
    [cell_id] = await rig.coordinator.on_phase(make_task('5387'), WorkflowState.PLAN, entering=False)
    await rig.coordinator.join()
    before = _row(rig, cell_id)

    sweep = await rig.coordinator.expire(before.settle_deadline - timedelta(seconds=1))

    assert sweep == ExpireSweep(expired=(), deferred=())
    assert _row(rig, cell_id) == before


@pytest.mark.asyncio
async def test_the_sweep_is_bounded_and_names_what_it_deferred(rig):
    seeded = [
        _seeded(i, state=_S.AWAITING_REFERENCE) for i in range(EXPIRE_BATCH_LIMIT + 3)
    ]
    for cell in reversed(seeded):
        rig.store.insert_shadow_cell(cell)
    now = START

    first = await rig.coordinator.expire(now)

    ids = [cell.cell_id for cell in seeded]
    assert first.expired == tuple(ids[:EXPIRE_BATCH_LIMIT])
    assert first.deferred == tuple(ids[EXPIRE_BATCH_LIMIT:])
    assert all(_row(rig, cell_id).state is _S.AWAITING_REFERENCE for cell_id in first.deferred)

    second = await rig.coordinator.expire(now)

    assert second == ExpireSweep(expired=tuple(ids[EXPIRE_BATCH_LIMIT:]), deferred=())
    assert all(_row(rig, cell_id).state is _S.EXPIRED for cell_id in ids)
    assert len(rig.events_of(EventType.shadow_cell_settled)) == len(ids)


@pytest.fixture
def restart_rows(rig) -> dict[str, ShadowCell]:
    rows = {
        'dead-sampled': _seeded(0, state=_S.SAMPLED),
        'dead-opened': _seeded(1, state=_S.OPENED),
        'dead-running': _seeded(2, state=_S.RUNNING),
        'no-run-id': _seeded(3, state=_S.RUNNING, run_id=None),
        'dead-awaiting': _seeded(4, state=_S.AWAITING_REFERENCE),
        'own-running': _seeded(5, state=_S.RUNNING, run_id=RUN_ID),
    }
    for cell in rows.values():
        rig.store.insert_shadow_cell(cell)
    return rows


_ORPHANS = ('dead-sampled', 'dead-opened', 'dead-running', 'no-run-id')


@pytest.mark.asyncio
async def test_a_restart_fails_every_held_cell_of_a_gone_run(rig, restart_rows):
    reconciled = await rig.coordinator.reconcile_on_start()

    assert sorted(reconciled) == sorted(restart_rows[name].cell_id for name in _ORPHANS)
    for name in _ORPHANS:
        row = _row(rig, restart_rows[name].cell_id)
        assert row.state is _S.FAILED
        assert row.reason == 'orphaned_by_restart'
        assert row.settled_at is None
    ended = rig.events_of(EventType.shadow_cell_ended)
    assert sorted(event['data']['cell_id'] for event in ended) == sorted(reconciled)
    assert {event['data']['reason'] for event in ended} == {'orphaned_by_restart'}


@pytest.mark.asyncio
async def test_settle_and_expire_own_an_awaiting_cell_not_its_run(rig, restart_rows):
    await rig.coordinator.reconcile_on_start()

    assert _row(rig, restart_rows['dead-awaiting'].cell_id) == restart_rows['dead-awaiting']
    assert _row(rig, restart_rows['own-running'].cell_id) == restart_rows['own-running']


@pytest.mark.asyncio
async def test_after_reconcile_only_this_run_holds_cells(rig, restart_rows):
    await rig.coordinator.reconcile_on_start()

    held = rig.store.find_shadow_cells(ShadowCellQuery(project_id=PROJECT_ID, states=_HELD))

    assert [cell.run_id for cell in held] == [RUN_ID]


@pytest.mark.asyncio
async def test_reconcile_is_idempotent(rig, restart_rows):
    await rig.coordinator.reconcile_on_start()

    assert await rig.coordinator.reconcile_on_start() == []
    assert len(rig.events_of(EventType.shadow_cell_ended)) == len(_ORPHANS)
