"""Internal task writers that never cross MCP dispatch meet the storage gate legibly.

The MCP boundary guard (``server/markup_guard.py::install_markup_guard``)
cannot see these writers: they run below it, in-process, and reach
``SqliteTaskBackend`` directly. Task 4419 guards the backend's two write
sinks; these tests drive the two riskiest internal writers over a real
backend, with no MCP call anywhere in the loop, and require that each one
reports the refusal as a markup leak with its structured facts rather than
as an opaque generic failure.

Specimens are composed from ``shared.toolcall_markup``'s spellings, never
written raw (the Sentinel-literal hazard in ``shared/src/shared/toolcall_markup.py``).
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pytest
import pytest_asyncio
from shared.toolcall_markup import CANONICAL_OPENER_PREFIX, closer_for

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import TaskmasterConfig
from fused_memory.middleware.task_curator import CuratorDecision, RewrittenTask
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.models.scope import ProjectRoot
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.reconciliation.stages.task_knowledge_sync import (
    _queue_briefing_refresh_tasks,
)

# Task 4358: the description closer swallowed priority='low'.
TASK_4358_FRAGMENT = (
    closer_for('description') + '\n' + CANONICAL_OPENER_PREFIX + '"priority">low'
)

# A curator rewrite copying a candidate's leaked description. The rewrite
# always supplies its own priority, so the swallowed argument here is one it
# does not supply; otherwise nothing would be provably swallowed.
REWRITE_LEAKED_FRAGMENT = (
    closer_for('description') + '\n' + CANONICAL_OPENER_PREFIX + '"dependencies">4358'
)
REWRITE_DESCRIPTION = 'Merged description of the two duplicates.\n' + REWRITE_LEAKED_FRAGMENT


@pytest_asyncio.fixture
async def backend(tmp_path):
    b = SqliteTaskBackend(TaskmasterConfig(project_root=str(tmp_path)))
    await b.start()
    yield b
    await b.close()


@pytest.fixture
def project_root(tmp_path) -> str:
    return str(tmp_path / 'proj')


@pytest_asyncio.fixture
async def interceptor(backend, tmp_path, monkeypatch):
    monkeypatch.setenv('DARK_FACTORY_DATA_DIR', str(tmp_path / 'data'))
    buffer = EventBuffer(db_path=tmp_path / 'events.db', buffer_size_threshold=100)
    await buffer.initialize()
    yield TaskInterceptor(backend, None, buffer)
    await buffer.close()


def _stored_rows(project_root: str) -> list[dict]:
    db_path = Path(project_root) / '.taskmaster' / 'tasks' / 'tasks.db'
    conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute('SELECT * FROM tasks ORDER BY id')]
    finally:
        conn.close()


_LEAK_FACTS = ('column', 'fragment', 'recovered')


def _leak_reports(caplog) -> list[tuple[str, dict]]:
    """Each log record carrying the leak's structured facts, as (message, facts)."""
    return [
        (record.getMessage(), {key: record.__dict__[key] for key in _LEAK_FACTS})
        for record in caplog.records
        if all(key in record.__dict__ for key in _LEAK_FACTS)
    ]


@pytest.mark.asyncio
async def test_combine_refuses_a_leaked_rewrite_and_reports_it_as_a_markup_leak(
    interceptor, backend, project_root, caplog,
):
    seeded = await backend.add_task(
        project_root, title='Live task', description='original description',
        details='original details',
    )
    before = _stored_rows(project_root)
    decision = CuratorDecision(
        action='combine',
        target_id=seeded['id'],
        target_fingerprint='Live task',
        rewritten_task=RewrittenTask(
            title='Live task', description=REWRITE_DESCRIPTION, details='merged',
            files_to_modify=[], priority='medium',
        ),
        justification='duplicate of the live task',
    )

    with caplog.at_level(logging.WARNING):
        result = await interceptor._execute_combine(project_root, decision)

    assert result is None
    assert _stored_rows(project_root) == before
    [(message, facts)] = _leak_reports(caplog)
    assert 'markup' in message
    assert facts == {
        'column': 'description',
        'fragment': REWRITE_LEAKED_FRAGMENT,
        'recovered': {'dependencies': '4358'},
    }


@pytest.mark.asyncio
async def test_briefing_refresh_refuses_a_leaked_gap_and_reports_it_as_a_markup_leak(
    backend, project_root, caplog,
):
    mismatch = {
        'task_id': '17',
        'subproject': 'fused-memory',
        'title': 'Some gap-carrying task',
        'what': 'real description prose\n' + TASK_4358_FRAGMENT,
    }

    with caplog.at_level(logging.WARNING):
        summary = await _queue_briefing_refresh_tasks(
            backend, ProjectRoot(project_root), [mismatch], existing_tasks=[],
        )

    assert summary == {'created': [], 'skipped': [], 'failed': ['17']}
    assert _stored_rows(project_root) == []
    [(message, facts)] = _leak_reports(caplog)
    assert 'markup' in message
    assert facts == {
        'column': 'description',
        'fragment': TASK_4358_FRAGMENT,
        'recovered': {'priority': 'low'},
    }
