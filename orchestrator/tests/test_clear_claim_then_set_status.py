"""Scheduler.clear_claim_then_set_status and its two clear-then-flip call sites.

PRD docs/prds/claimant-invariant-enforcement.md task δ (boundary tests
B11/B12).  Every test drives a REAL ``Scheduler`` whose MCP session is
``_TaskRowStore``: a stateful in-process double that models one task row, so
assertions read the row back through ``Scheduler.get_task`` and inspect the
snapshot taken after every write, rather than mock call order.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from shared.task_claimant import is_stranded

from orchestrator.config import OrchestratorConfig
from orchestrator.scheduler import Scheduler, SetTaskStatusRejected, TickContext

TID = '4621'
DEAD_CLAIMANT = 'run-dead0000/4621-x/pid=999999'
STALE_HEARTBEAT = '2026-08-19T21:57:20+00:00'
FIXED_DT = datetime(2026, 10, 10, 12, 0, 0, tzinfo=UTC)


class _TaskRowStore:
    """``McpSessionLike`` double holding one task row and every write applied to it.

    Same JSON-RPC envelope, ``call_tool`` signature and loud failure on unknown
    tools as ``_orch_helpers.HermeticMcpSession``, plus the per-row state and
    fault injection that stateless double deliberately lacks.
    """

    def __init__(self, row: dict) -> None:
        self.row = row
        self.writes: list[tuple[str, dict]] = []
        self.fail_status_write = False
        self.fail_claimant_write = False
        self._request_id = 0

    def _envelope(self, payload: dict) -> dict:
        self._request_id += 1
        return {
            'jsonrpc': '2.0',
            'id': self._request_id,
            'result': {'content': [{'type': 'text', 'text': json.dumps(payload)}]},
        }

    def _record(self, name: str) -> None:
        self.writes.append((name, dict(self.row)))

    async def call_tool(self, name: str, arguments: dict, timeout: float = 30) -> dict:
        if name == 'get_task':
            return self._envelope(dict(self.row))
        if name == 'set_task_claimant':
            if self.fail_claimant_write:
                raise ConnectionError('injected set_task_claimant fault')
            for column in ('claimant_run_id', 'heartbeat_at'):
                if column in arguments:
                    self.row[column] = arguments[column]
            self._record(name)
            return self._envelope({})
        if name == 'set_task_status':
            if self.fail_status_write:
                return self._envelope({'error': 'injected_fault', 'error_type': 'InjectedFault'})
            self.row['status'] = arguments['status']
            self._record(name)
            return self._envelope({'id': arguments['id'], 'status': arguments['status']})
        raise NotImplementedError(
            f'_TaskRowStore: unknown tool {name!r} — add a branch in call_tool '
            'if this tool is needed by the test'
        )

    def pending_with_stale_claimant(self) -> list[tuple[str, dict]]:
        return [
            (name, row) for name, row in self.writes
            if row['status'] == 'pending' and row['claimant_run_id'] is not None
        ]


def _claimed_row(status: str) -> dict:
    return {
        'id': TID,
        'status': status,
        'metadata': {},
        'dependencies': [],
        'claimant_run_id': DEAD_CLAIMANT,
        'heartbeat_at': STALE_HEARTBEAT,
    }


@pytest.fixture
def store() -> _TaskRowStore:
    return _TaskRowStore(_claimed_row('in-progress'))


@pytest.fixture
def scheduler(tmp_path: Path, store: _TaskRowStore) -> Scheduler:
    return Scheduler(OrchestratorConfig(project_root=tmp_path), mcp_session=store)


@pytest.mark.asyncio
async def test_clears_claimant_then_writes_status(
    scheduler: Scheduler, store: _TaskRowStore, forbid_live_mcp,
) -> None:
    await scheduler.clear_claim_then_set_status(TID, 'pending')

    row = await scheduler.get_task(TID)
    assert row is not None
    assert (row['status'], row['claimant_run_id'], row['heartbeat_at']) == ('pending', None, None)
    assert [name for name, _ in store.writes] == ['set_task_claimant', 'set_task_status']
    first = store.writes[0][1]
    assert (first['status'], first['claimant_run_id']) == ('in-progress', None)


@pytest.mark.asyncio
async def test_failed_clear_does_not_block_the_flip(
    scheduler: Scheduler, store: _TaskRowStore, forbid_live_mcp,
) -> None:
    store.fail_claimant_write = True

    await scheduler.clear_claim_then_set_status(TID, 'pending')

    row = await scheduler.get_task(TID)
    assert row is not None
    assert row['status'] == 'pending'


@pytest.mark.asyncio
async def test_rejected_flip_leaves_cleared_row_never_pending_with_stale_claimant(
    scheduler: Scheduler, store: _TaskRowStore, forbid_live_mcp,
) -> None:
    store.fail_status_write = True

    with pytest.raises(SetTaskStatusRejected):
        await scheduler.clear_claim_then_set_status(TID, 'pending')

    row = await scheduler.get_task(TID)
    assert row is not None
    assert (row['status'], row['claimant_run_id']) == ('in-progress', None)
    assert store.pending_with_stale_claimant() == []
    assert is_stranded(row, datetime.now(UTC), timedelta(seconds=300))


class _NoOpenEscalations:
    def get_by_task(self, task_id: str, status: str | None = None) -> list:
        return []


@pytest.mark.asyncio
async def test_stranded_blocked_sweep_leaves_pending_with_null_claimant(
    tmp_path: Path, forbid_live_mcp,
) -> None:
    store = _TaskRowStore(_claimed_row('blocked'))
    scheduler = Scheduler(
        OrchestratorConfig(project_root=tmp_path),
        mcp_session=store,
        wall_time_source=lambda: FIXED_DT,
    )
    scheduler.escalation_queue = _NoOpenEscalations()  # type: ignore[assignment]
    task = dict(store.row)
    ctx = TickContext(tasks=[task], status_map={TID: 'blocked'}, tasks_by_id={TID: task})

    await scheduler._phase_redispatch_stranded_blocked(ctx)

    row = await scheduler.get_task(TID)
    assert row is not None
    assert (row['status'], row['claimant_run_id']) == ('pending', None)
    assert [name for name, _ in store.writes] == ['set_task_claimant', 'set_task_status']
