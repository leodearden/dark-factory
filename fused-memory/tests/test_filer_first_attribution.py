"""Acceptance: a path under the filing project's own root never makes its task cross-repo.

Task 7515. Drives the real ``TaskInterceptor.submit_task`` against tmp_path
project roots laid out like the stream_touch incident
(``_filer_first_layout.stale_snapshot_registry``).
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from _filer_first_layout import FILER_PROJECT, TASK_27_FILES, stale_snapshot_registry

from fused_memory.middleware.scope_violation_escalator import ScopeViolationEscalator
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.middleware.ticket_store import TicketStore
from fused_memory.reconciliation.event_buffer import EventBuffer

PROSE = (
    'Record the starter plan into sessions/plans/lp-starter-v1.toml and the '
    'trace into corpus/fixtures/a.trace.toml'
)


class _RecordingEscalator(ScopeViolationEscalator):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, Any]] = []

    def report_rejection(self, **kwargs: Any) -> str | None:
        self.calls.append(kwargs)
        return None


class _Stack:
    def __init__(
        self, interceptor: TaskInterceptor, store: TicketStore,
        escalator: _RecordingEscalator, filer_root: str,
    ) -> None:
        self.interceptor = interceptor
        self.store = store
        self.escalator = escalator
        self.filer_root = filer_root

    async def submitted_metadata(self, **kwargs: Any) -> dict[str, Any]:
        result = await self.interceptor.submit_task(
            project_root=self.filer_root, **kwargs,
        )
        assert 'error_type' not in result, result
        assert result['ticket'].startswith('tkt_')
        row = await self.store.get(result['ticket'])
        assert row is not None
        return json.loads(row['candidate_json'])['metadata'] or {}


@pytest_asyncio.fixture
async def stack(tmp_path):
    projects = tmp_path / 'projects'
    projects.mkdir()
    registry = stale_snapshot_registry(projects)
    filer_root = registry.root_for_project(FILER_PROJECT)
    assert filer_root is not None
    buffer = EventBuffer(db_path=tmp_path / 'events.db', buffer_size_threshold=100)
    await buffer.initialize()
    store = TicketStore(tmp_path / 'tickets.db')
    await store.initialize()
    escalator = _RecordingEscalator()
    taskmaster = AsyncMock()
    taskmaster.add_task = AsyncMock(return_value={'id': '1'})
    interceptor = TaskInterceptor(
        taskmaster, AsyncMock(), buffer,
        ticket_store=store,
        prefix_registry=registry,
        scope_violation_escalator=escalator,
    )
    try:
        yield _Stack(interceptor, store, escalator, filer_root)
    finally:
        await interceptor.close()
        await buffer.close()


@pytest.mark.asyncio
async def test_task_27_shape_is_neither_cross_repo_nor_a_scope_mismatch(stack):
    metadata = await stack.submitted_metadata(
        title='Starter session plan', description=PROSE,
        metadata={'files': TASK_27_FILES},
    )
    assert 'cross_repo' not in metadata
    assert 'cross_repo_project' not in metadata
    assert 'possible_scope_mismatch' not in metadata
    assert stack.escalator.calls == []


@pytest.mark.asyncio
async def test_fileless_prose_citing_own_dirs_is_not_a_scope_mismatch(stack):
    metadata = await stack.submitted_metadata(
        title='Starter session plan', description=PROSE,
    )
    assert 'possible_scope_mismatch' not in metadata
    assert stack.escalator.calls == []


@pytest.mark.asyncio
async def test_files_only_another_project_has_stay_cross_repo(stack):
    metadata = await stack.submitted_metadata(
        title='Harness change', description='Adjust the harness.',
        metadata={'files': [
            'orchestrator/src/orchestrator/x.py', 'orchestrator/tests/test_x.py',
        ]},
    )
    assert metadata['cross_repo'] is True
    assert metadata['cross_repo_project'] == 'dark_factory'
