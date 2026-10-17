"""Internal task writers that never cross MCP dispatch meet the storage gate legibly.

The MCP boundary guard (``server/markup_guard.py::install_markup_guard``)
cannot see these writers: they run below it, in-process, and reach
``SqliteTaskBackend`` directly. Task 4419 guards the backend's two write
sinks; these tests drive the two riskiest internal writers over a real
backend, with no MCP call anywhere in the loop, and require that each one
reports the refusal as a markup leak with its structured facts rather than
as an opaque generic failure.

Both writers are called through private entry points because neither has a
public seam that isolates it. The combine is reachable publicly only through
the ticket worker with the private ``_get_curator`` patched, and the briefing
refresh only through Stage 2's ``run()``, which also runs the LLM stage. That
is a missing seam, filed as a follow-up of task 4419, not a pattern to copy.

Specimens are composed from ``shared.toolcall_markup``'s spellings, never
written raw (the Sentinel-literal hazard in ``shared/src/shared/toolcall_markup.py``).
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pytest
import pytest_asyncio
from _curator_helpers import curator_returning
from shared.toolcall_markup import CANONICAL_OPENER_PREFIX, closer_for

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import TaskmasterConfig
from fused_memory.middleware.task_curator import CuratorDecision, RewrittenTask
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.middleware.ticket_store import TicketStore
from fused_memory.models.scope import ProjectRoot
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.reconciliation.stages.task_knowledge_sync import (
    queue_briefing_refresh_tasks,
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
async def ticket_store(tmp_path):
    store = TicketStore(tmp_path / 'tickets.db')
    await store.initialize()
    yield store
    await store.close()


@pytest_asyncio.fixture
async def build_interceptor(backend, ticket_store, tmp_path, monkeypatch):
    """Build TaskInterceptors around an injected curator; each is closed at teardown."""
    monkeypatch.setenv('DARK_FACTORY_DATA_DIR', str(tmp_path / 'data'))
    buffer = EventBuffer(db_path=tmp_path / 'events.db', buffer_size_threshold=100)
    await buffer.initialize()
    built: list[TaskInterceptor] = []

    def _build(curator) -> TaskInterceptor:
        interceptor = TaskInterceptor(
            backend, None, buffer, ticket_store=ticket_store, curator=curator,
        )
        built.append(interceptor)
        return interceptor

    yield _build
    for interceptor in built:
        await interceptor.close()
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
    build_interceptor, backend, project_root, caplog,
):
    seeded = await backend.add_task(
        project_root, title='Live task', description='original description',
        details='original details',
    )
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
    interceptor = build_interceptor(curator_returning(decision))
    [target_before] = _stored_rows(project_root)

    with caplog.at_level(logging.WARNING):
        submitted = await interceptor.submit_task(
            project_root=project_root, title='Candidate duplicate',
            description='candidate description',
        )
        resolved = await interceptor.resolve_ticket(
            submitted['ticket'], project_root, timeout_seconds=10,
        )

    assert resolved['status'] == 'created'
    assert resolved['reason'].startswith('create: combine-failed')
    rows = _stored_rows(project_root)
    assert rows[0] == target_before
    assert [(r['id'], r['title'], r['description']) for r in rows[1:]] == [
        (int(resolved['task_id']), 'Candidate duplicate', 'candidate description'),
    ]
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
        summary = await queue_briefing_refresh_tasks(
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
