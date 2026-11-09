"""A scheduled self-deploy holds its dependents until its restart lands (task 7276).

A self-target deterministic deploy goes ``done`` the moment its detached restart
is merely scheduled.  Task 3189 depended on 7238, such a deploy, and dispatched
onto pre-restart code.  These tests drive the REAL scheduler tick through
``acquire_next()`` against a backend-free session double and observe dispatch
only through its returned assignment.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from test_delivered_check_gate_e2e import (
    _build_harness,
    _LocalDepMcpSession,
    _register_dependent,
    _run_tick,
)

SELF_DEPLOY_ID = '7238'
DEPENDENT_ID = '3189'


class _DeploySession(_LocalDepMcpSession):
    """``_LocalDepMcpSession`` plus the metadata writes and lookups a settle needs."""

    def __init__(self) -> None:
        super().__init__()
        self.metadata_writes: list[tuple[str, dict]] = []

    def task(self, task_id: str) -> dict:
        return next(t for t in self.tasks if str(t.get('id')) == task_id)

    async def call_tool(self, name: str, arguments: dict, timeout: float = 30) -> dict:
        if name == 'update_task':
            return self._update_task(arguments)
        if name == 'find_tasks_by_metadata':
            return self._find_tasks_by_metadata(arguments)
        return await super().call_tool(name, arguments, timeout)

    def _update_task(self, arguments: dict) -> dict:
        task_id = str(arguments['id'])
        metadata = json.loads(arguments['metadata'])
        self.metadata_writes.append((task_id, metadata))
        self.task(task_id).setdefault('metadata', {}).update(metadata)
        return self._envelope(json.dumps({'id': task_id, 'success': True}))

    def _find_tasks_by_metadata(self, arguments: dict) -> dict:
        key = arguments['key']
        statuses = arguments.get('statuses')
        rows = []
        for task in self.tasks:
            metadata = task.get('metadata') or {}
            if key not in metadata:
                continue
            if 'value' in arguments and metadata[key] != arguments['value']:
                continue
            if statuses is not None and task.get('status') not in statuses:
                continue
            row = {
                'task_id': task['id'],
                'title': task.get('title', ''),
                'status': task.get('status'),
                'priority': task.get('priority', 'medium'),
                'updated_at': None,
            }
            if arguments.get('include_value'):
                row['matched_value'] = metadata[key]
            rows.append(row)
        pagination = {'has_more': False, 'offset': arguments.get('offset', 0), 'returned': len(rows)}
        return self._envelope(json.dumps({'results': rows, 'pagination': pagination}))


def _register_scheduled_deploy(
    session: _DeploySession,
    task_id: str,
    phase: str,
    *,
    scheduled_at: datetime,
) -> dict:
    """A done self-target deploy whose restart was scheduled at *scheduled_at* (the 7238 shape)."""
    transient = f'orch-redeploy-restart-{task_id}.service'
    task: dict = {
        'id': task_id,
        'title': f'Self-deploy {task_id}',
        'status': 'done',
        'dependencies': [],
        'metadata': {
            'task_kind': 'deterministic',
            'files': [f'local_dep/self_deploy_{task_id}.py'],
            'done_provenance': {
                'kind': 'deterministic-deploy-scheduled',
                'unit': 'orchestrator-dark-factory.service',
                'transient_unit': transient,
                'fire_delay_secs': 60,
            },
            'before_done': {'target_unit': 'orchestrator-dark-factory.service'},
            'before_done_scheduled_at': {
                'at': scheduled_at.isoformat(),
                'transient_unit': transient,
                'fire_delay_secs': 60,
            },
            'deploy_state': {
                'phase': phase,
                'ran_at': scheduled_at.isoformat(),
                'verify_baseline': None,
                'verified_at': None,
                'escalated_at': None,
            },
        },
    }
    session.tasks.append(task)
    return task


def _set_phase(session: _DeploySession, task_id: str, phase: str) -> None:
    session.task(task_id)['metadata']['deploy_state']['phase'] = phase


def _anchor_writes(session: _DeploySession, task_id: str) -> list[dict]:
    return [
        metadata for written_id, metadata in session.metadata_writes
        if written_id == task_id and 'milestone_deps_satisfied_at' in metadata
    ]


@pytest.fixture
def session() -> _DeploySession:
    session = _DeploySession()
    _register_scheduled_deploy(
        session, SELF_DEPLOY_ID, 'scheduled',
        scheduled_at=datetime.now(UTC) - timedelta(hours=2),
    )
    return session


class TestDispatchHold:
    @pytest.mark.asyncio
    async def test_dependent_of_a_scheduled_self_deploy_is_not_dispatched(
        self, tmp_path: Path, session: _DeploySession,
    ) -> None:
        _register_dependent(session, DEPENDENT_ID, dep_id=SELF_DEPLOY_ID)
        harness = _build_harness(tmp_path, session, tmp_path / 'escalations')

        assert await _run_tick(harness) is None
        assert await _run_tick(harness) is None

    @pytest.mark.asyncio
    async def test_dependent_dispatches_once_the_phase_reads_verified(
        self, tmp_path: Path, session: _DeploySession,
    ) -> None:
        _register_dependent(session, DEPENDENT_ID, dep_id=SELF_DEPLOY_ID)
        harness = _build_harness(tmp_path, session, tmp_path / 'escalations')

        assert await _run_tick(harness) is None
        _set_phase(session, SELF_DEPLOY_ID, 'verified')

        assert await _run_tick(harness) == DEPENDENT_ID

    @pytest.mark.asyncio
    async def test_delayed_milestone_anchor_waits_for_the_restart(
        self, tmp_path: Path, session: _DeploySession,
    ) -> None:
        _register_dependent(
            session, DEPENDENT_ID, dep_id=SELF_DEPLOY_ID,
            metadata={'milestone': {'mode': 'delayed', 'after_secs': 3600}},
        )
        harness = _build_harness(tmp_path, session, tmp_path / 'escalations')

        await _run_tick(harness)
        assert _anchor_writes(session, DEPENDENT_ID) == []

        _set_phase(session, SELF_DEPLOY_ID, 'verified')
        await _run_tick(harness)
        assert len(_anchor_writes(session, DEPENDENT_ID)) == 1
