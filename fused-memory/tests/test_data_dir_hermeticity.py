"""A test's DARK_FACTORY_DATA_DIR writes land in its own tmp_path, never under the pytest CWD.

Task 6456.  ``conftest.py::_isolate_fm_data_dir`` holds the invariant; the
measurement behind it is in ``plans/fused-memory-config-cwd-leak-rca-2026-09-13.md``.
Driven through the public seams only: a curator combine goes in through
``submit_task`` and comes out through ``resolve_ticket``.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from _curator_helpers import curator_returning

from fused_memory.middleware.curator_types import CuratorDecision, RewrittenTask
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.middleware.ticket_store import TicketStore
from fused_memory.reconciliation.event_buffer import EventBuffer

AUDIT_FILE = 'combine_audit.jsonl'


@pytest.fixture
def taskmaster():
    tm = AsyncMock()
    tm.get_tasks = AsyncMock(return_value={'tasks': []})
    tm.get_task = AsyncMock(return_value={
        'id': '50', 'title': 'Target', 'status': 'pending', 'metadata': {},
    })
    tm.add_task = AsyncMock(return_value={'id': '42', 'title': 'New Task'})
    tm.update_task = AsyncMock(return_value={'success': True})
    tm.set_task_status = AsyncMock(return_value={'success': True})
    tm.ensure_connected = AsyncMock()
    return tm


@pytest_asyncio.fixture
async def event_buffer(tmp_path):
    buf = EventBuffer(db_path=tmp_path / 'events.db', buffer_size_threshold=100)
    await buf.initialize()
    yield buf
    await buf.close()


@pytest_asyncio.fixture
async def ticket_store(tmp_path):
    store = TicketStore(tmp_path / 'tickets.db')
    await store.initialize()
    yield store
    await store.close()


async def _combine_once(taskmaster, event_buffer, ticket_store) -> dict:
    """Run one curator combine onto task 50 and return the resolved ticket."""
    interceptor = TaskInterceptor(
        taskmaster, None, event_buffer, ticket_store=ticket_store,
        curator=curator_returning(CuratorDecision(
            action='combine',
            target_id='50',
            target_fingerprint='Target',
            rewritten_task=RewrittenTask(
                title='Combined', description='d', details='', files_to_modify=[],
                priority='medium',
            ),
            justification='same concern',
        )),
    )
    try:
        submitted = await interceptor.submit_task(
            project_root='/project', agent_id='hermeticity-probe',
            title='Candidate', description='d',
        )
        return await interceptor.resolve_ticket(
            submitted['ticket'], '/project', timeout_seconds=10,
        )
    finally:
        await interceptor.close()


def _audit_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.mark.asyncio
async def test_a_combine_audit_lands_in_the_tests_tmp_path_from_any_cwd(
    monkeypatch, tmp_path, tmp_path_factory, taskmaster, event_buffer, ticket_store,
):
    """The chdir target is a SIBLING of ``tmp_path``, so a CWD-relative write
    can never land on the pinned root by coincidence."""
    elsewhere = tmp_path_factory.mktemp('elsewhere')
    monkeypatch.chdir(elsewhere)

    resolved = await _combine_once(taskmaster, event_buffer, ticket_store)

    assert resolved['status'] == 'combined'
    assert list(elsewhere.iterdir()) == []
    [record] = _audit_records(tmp_path / 'data' / AUDIT_FILE)
    assert record['target_id'] == '50'


class TestAnInheritedDataDirCannotRedirectTheAudit:
    """Planting AMBIENTLY models a variable inherited from the launching shell:
    the class-scoped plant is set up before the function-scoped conftest pins,
    exactly as in ``test_config_hermeticity.py::TestAmbientInterpolationCannotRedirectTheConfig``."""

    @pytest.fixture(scope='class', autouse=True)
    def planted_data_dir(self, tmp_path_factory):
        planted = tmp_path_factory.mktemp('ambient')
        ambient = pytest.MonkeyPatch()
        ambient.setenv('DARK_FACTORY_DATA_DIR', str(planted))
        yield planted
        ambient.undo()

    @pytest.mark.asyncio
    async def test_an_inherited_data_dir_cannot_redirect_the_combine_audit(
        self, planted_data_dir, tmp_path, taskmaster, event_buffer, ticket_store,
    ):
        resolved = await _combine_once(taskmaster, event_buffer, ticket_store)

        assert resolved['status'] == 'combined'
        assert not (planted_data_dir / AUDIT_FILE).exists()
        [record] = _audit_records(tmp_path / 'data' / AUDIT_FILE)
        assert record['target_id'] == '50'
