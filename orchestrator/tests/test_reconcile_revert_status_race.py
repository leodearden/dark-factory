"""A reconcile revert that races a row going terminal must not write.

Any refusal the revert does surface names the transition actually attempted.
Section 1 pins the Scheduler half that supplies that transition. Every test
drives a REAL ``Scheduler`` whose MCP session is ``_ServerGatedRowStore``: one
stateful task row behind fused-memory's own terminal-exit gate, answering in
the real rejection payload shapes.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from _orch_helpers import HermeticMcpSession

from orchestrator.config import OrchestratorConfig
from orchestrator.scheduler import Scheduler, SetTaskStatusRejected, TerminalExitRejection
from orchestrator.task_status import TERMINAL_STATUSES

TID = '42'
DEAD_CLAIMANT = 'run-dead0000/42-x/pid=999999'
STALE_HEARTBEAT = '2026-08-19T21:57:20+00:00'


class _ServerGatedRowStore(HermeticMcpSession):
    """One task row whose ``set_task_status`` gates like fused-memory's interceptor.

    ``refusal`` (a payload dict) answers the next status writes verbatim;
    otherwise a non-terminal write onto a terminal row without
    ``reopen_reason`` gets the real terminal-exit payload. ``writes`` records
    each applied write with a snapshot of the row after it.
    """

    def __init__(self, row: dict) -> None:
        super().__init__()
        self.row = row
        self.writes: list[tuple[str, dict]] = []
        self.refusal: dict | None = None
        self.on_claimant_clear: Callable[[], None] | None = None

    def _reply(self, payload: dict) -> dict:
        return self._envelope(json.dumps(payload))

    def _record(self, name: str) -> None:
        self.writes.append((name, dict(self.row)))

    def _status_write(self, arguments: dict) -> dict:
        target = arguments['status']
        if self.refusal is not None:
            return self.refusal
        current = self.row['status']
        if (
            current in TERMINAL_STATUSES
            and target not in TERMINAL_STATUSES
            and 'reopen_reason' not in arguments
        ):
            return {
                'success': False,
                'error': 'terminal_exit_rejected',
                'task_id': self.row['id'],
                'from_status': current,
                'to_status': target,
            }
        self.row['status'] = target
        self._record('set_task_status')
        return {'id': self.row['id'], 'status': target}

    async def call_tool(self, name: str, arguments: dict, timeout: float = 30) -> dict:
        if name == 'get_task':
            return self._reply(dict(self.row))
        if name == 'get_statuses':
            return self._reply({'statuses': {self.row['id']: self.row['status']}})
        if name == 'set_task_claimant':
            for column in ('claimant_run_id', 'heartbeat_at'):
                if column in arguments:
                    self.row[column] = arguments[column]
            self._record(name)
            if self.on_claimant_clear is not None:
                self.on_claimant_clear()
            return self._reply({})
        if name == 'set_task_status':
            return self._reply(self._status_write(arguments))
        return await super().call_tool(name, arguments, timeout)


def _claimed_row(status: str) -> dict:
    return {
        'id': TID,
        'status': status,
        'metadata': {},
        'dependencies': [],
        'claimant_run_id': DEAD_CLAIMANT,
        'heartbeat_at': STALE_HEARTBEAT,
    }


def _illegal_transition(from_status: str, to_status: str) -> dict:
    return {
        'success': False,
        'error': 'illegal_transition',
        'task_id': TID,
        'from_status': from_status,
        'to_status': to_status,
        'actor': 'orchestrator',
    }


@pytest.fixture
def store() -> _ServerGatedRowStore:
    return _ServerGatedRowStore(_claimed_row('in-progress'))


@pytest.fixture
def scheduler(tmp_path: Path, store: _ServerGatedRowStore) -> Scheduler:
    return Scheduler(OrchestratorConfig(project_root=tmp_path), mcp_session=store)


# --- Section 1: the refusal carries the attempted transition ---------------


@pytest.mark.asyncio
async def test_illegal_transition_refusal_names_the_server_reported_transition(
    scheduler: Scheduler, store: _ServerGatedRowStore, forbid_live_mcp,
) -> None:
    store.refusal = _illegal_transition('in-progress', 'pending')

    with pytest.raises(SetTaskStatusRejected) as info:
        await scheduler.set_task_status(TID, 'pending')

    exc = info.value
    assert exc.error_code == 'illegal_transition'
    assert (exc.old_status, exc.target_status) == ('in-progress', 'pending')
    assert exc.attempted_transition == 'in-progress -> pending'
    assert exc.row_already_terminal is False


@pytest.mark.asyncio
async def test_refusal_without_from_status_renders_an_unknown_origin(
    scheduler: Scheduler, store: _ServerGatedRowStore, forbid_live_mcp,
) -> None:
    store.refusal = {'success': False, 'error': 'something_unexpected'}

    with pytest.raises(SetTaskStatusRejected) as info:
        await scheduler.set_task_status(TID, 'in-progress')

    assert info.value.old_status is None
    assert info.value.attempted_transition == 'unknown -> in-progress'


@pytest.mark.asyncio
async def test_terminal_exit_refusal_reports_the_row_already_terminal(
    tmp_path: Path, forbid_live_mcp,
) -> None:
    store = _ServerGatedRowStore(_claimed_row('done'))
    scheduler = Scheduler(OrchestratorConfig(project_root=tmp_path), mcp_session=store)

    with pytest.raises(TerminalExitRejection) as info:
        await scheduler.set_task_status(TID, 'pending')

    assert info.value.attempted_transition == 'done -> pending'
    assert info.value.row_already_terminal is True


@pytest.mark.asyncio
@pytest.mark.parametrize('refusal', [
    {'success': False, 'error': 'done_gate_missing_files', 'missing_files': ['gone.py']},
    {'success': False, 'error': 'done_provenance_invalid'},
    {
        'success': False,
        'error': 'done_evidence_stale',
        'evidence_commit': 'abc1234',
        'evidence_committed_at': '2026-10-01T00:00:00+00:00',
        'reopen_at': '2026-10-02T00:00:00+00:00',
        'agent_id': 'orchestrator',
    },
], ids=lambda refusal: refusal['error'])
async def test_done_family_refusal_names_a_done_target(
    scheduler: Scheduler, store: _ServerGatedRowStore, refusal: dict, forbid_live_mcp,
) -> None:
    store.refusal = refusal

    with pytest.raises(SetTaskStatusRejected) as info:
        await scheduler.set_task_status(TID, 'done')

    assert info.value.target_status == 'done'
    assert info.value.attempted_transition == 'unknown -> done'
