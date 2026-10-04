"""Recurrence-carrier minting on terminal status writes (task 4866 r2).

Completing a recurrence carrier ``done`` through the public interceptor mints
exactly one pending successor link; every outcome is read back through the
real SqliteTaskBackend's ``get_task``/``get_tasks``. Contract:
docs/prds/recurring-deterministic-tasks.md C-2/C-3/C-6.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.backends.task_backend_types import AddTaskResult
from fused_memory.config.schema import TaskmasterConfig
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.models.reconciliation import EventType
from fused_memory.models.scope import resolve_project_id
from fused_memory.reconciliation.event_buffer import EventBuffer

KEY = 'nightly-check'
INTERVAL_SECS = 86400
BEFORE_DONE = {'kind': 'predicate', 'script': 'scripts/check.sh', 'timeout_secs': 60}
MINT_LOGGER = 'fused_memory.middleware.recurrence_mint'


class _AddTaskFails(SqliteTaskBackend):
    """Backend whose insert of a minted successor fails; every other insert succeeds."""

    async def add_task(
        self,
        project_root: str,
        prompt: str | None = None,
        title: str | None = None,
        description: str | None = None,
        details: str | None = None,
        dependencies: str | None = None,
        priority: str | None = None,
        metadata: str | None = None,
        tag: str | None = None,
        status: str = 'pending',
    ) -> AddTaskResult:
        if metadata is not None and 'recurrence-mint' in metadata:
            raise RuntimeError('boom')
        return await super().add_task(
            project_root, prompt, title, description, details, dependencies,
            priority, metadata, tag, status,
        )


@pytest_asyncio.fixture
async def event_buffer(tmp_path):
    buf = EventBuffer(db_path=tmp_path / 'recurrence_mint_eb.db', buffer_size_threshold=100)
    await buf.initialize()
    yield buf
    await buf.close()


@pytest_asyncio.fixture
async def backend(tmp_path):
    b = SqliteTaskBackend(TaskmasterConfig(project_root=str(tmp_path)))
    await b.start()
    yield b
    await b.close()


@pytest.fixture
def interceptor(backend, event_buffer):
    return TaskInterceptor(backend, None, event_buffer)


@pytest.fixture
def root(tmp_path) -> str:
    return str(tmp_path)


def _write_check_script(project_root: str) -> Path:
    script = Path(project_root) / 'scripts' / 'check.sh'
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text('#!/bin/sh\nexit 0\n')
    script.chmod(0o755)
    return script


async def _seed_carrier(
    interceptor: TaskInterceptor,
    backend: SqliteTaskBackend,
    project_root: str,
    *,
    key: str = KEY,
    interval_secs: int = INTERVAL_SECS,
    title: str = 'Nightly check',
    status: str = 'in-progress',
) -> str:
    _write_check_script(project_root)
    metadata = {
        'task_kind': 'deterministic',
        'before_done': BEFORE_DONE,
        'milestone': {'mode': 'dated', 'at': '2026-09-01T00:00:00+00:00'},
        'recurrence': {'key': key, 'interval_secs': interval_secs},
        'files': ['scripts/check.sh'],
    }
    added = await backend.add_task(
        project_root=project_root,
        title=title,
        description='Run the nightly check',
        details='details text',
        priority='high',
        metadata=json.dumps(metadata),
    )
    task_id = str(added['id'])
    if status != 'pending':
        result = await interceptor.set_task_status(task_id, status, project_root)
        assert 'error' not in result, result
    return task_id


async def _all_tasks(backend: SqliteTaskBackend, project_root: str) -> list[dict[str, Any]]:
    return list((await backend.get_tasks(project_root))['tasks'])


async def _chain_links(
    backend: SqliteTaskBackend, project_root: str, *, excluding: str, key: str = KEY,
) -> list[dict[str, Any]]:
    return [
        t for t in await _all_tasks(backend, project_root)
        if (t.get('metadata') or {}).get('recurrence', {}).get('key') == key
        and str(t['id']) != excluding
    ]


async def _next_wall_clock_second() -> None:
    """Sleep past the current wall-clock second.

    Successive links' milestone.at values are whole seconds, so two links of
    one chain completed inside the same second would mint identical titles.
    """
    now = datetime.now(UTC)
    await asyncio.sleep(1.01 - now.microsecond / 1_000_000)


@pytest.mark.asyncio
async def test_done_mints_exactly_one_successor_per_c3(interceptor, backend, root):
    pid = await _seed_carrier(interceptor, backend, root)

    t0 = datetime.now(UTC)
    result = await interceptor.set_task_status(pid, 'done', root)
    t1 = datetime.now(UTC)

    assert 'error' not in result, result
    successors = await _chain_links(backend, root, excluding=pid)
    assert len(successors) == 1, successors
    successor = successors[0]
    assert successor['status'] == 'pending'
    assert successor['description'] == 'Run the nightly check'
    assert successor['details'] == 'details text'
    assert successor['priority'] == 'high'
    md = successor['metadata']
    assert md['task_kind'] == 'deterministic'
    assert md['before_done'] == BEFORE_DONE
    assert md['files'] == ['scripts/check.sh']
    assert md['recurrence'] == {'key': KEY, 'interval_secs': INTERVAL_SECS, 'minted_from': pid}
    assert md['source'] == 'recurrence-mint'
    assert md['milestone']['mode'] == 'dated'
    at = datetime.fromisoformat(md['milestone']['at'])
    interval = timedelta(seconds=INTERVAL_SECS)
    assert t0.replace(microsecond=0) + interval <= at <= t1 + interval
    assert successor['title'] == f"Nightly check [due {md['milestone']['at']}]"
    predecessor = await backend.get_task(pid, project_root=root)
    assert predecessor['status'] == 'done'


@pytest.mark.asyncio
async def test_chain_titles_do_not_accumulate_run_labels(interceptor, backend, root):
    pid = await _seed_carrier(interceptor, backend, root)
    assert 'error' not in await interceptor.set_task_status(pid, 'done', root)
    [link2] = await _chain_links(backend, root, excluding=pid)
    link2_id = str(link2['id'])

    await _next_wall_clock_second()
    assert 'error' not in await interceptor.set_task_status(link2_id, 'in-progress', root)
    assert 'error' not in await interceptor.set_task_status(link2_id, 'done', root)

    link3s = [
        t for t in await _chain_links(backend, root, excluding=pid)
        if t['metadata']['recurrence'].get('minted_from') == link2_id
    ]
    assert len(link3s) == 1, link3s
    link3 = link3s[0]
    assert link3['title'] == f"Nightly check [due {link3['metadata']['milestone']['at']}]"
    assert (await backend.get_task(link2_id, project_root=root))['status'] == 'done'


@pytest.mark.asyncio
async def test_cancelled_carrier_mints_nothing(interceptor, backend, root):
    pid = await _seed_carrier(interceptor, backend, root)
    count_before = len(await _all_tasks(backend, root))

    assert 'error' not in await interceptor.set_task_status(pid, 'cancelled', root)

    assert len(await _all_tasks(backend, root)) == count_before
    assert await _chain_links(backend, root, excluding=pid) == []


@pytest.mark.asyncio
async def test_non_terminal_transition_mints_nothing(interceptor, backend, root):
    pid = await _seed_carrier(interceptor, backend, root)
    count_before = len(await _all_tasks(backend, root))

    assert 'error' not in await interceptor.set_task_status(pid, 'blocked', root)

    assert len(await _all_tasks(backend, root)) == count_before


@pytest.mark.asyncio
async def test_non_carrier_done_mints_nothing(interceptor, backend, root):
    added = await backend.add_task(project_root=root, title='plain task')
    task_id = str(added['id'])
    assert 'error' not in await interceptor.set_task_status(task_id, 'in-progress', root)
    count_before = len(await _all_tasks(backend, root))

    assert 'error' not in await interceptor.set_task_status(task_id, 'done', root)

    assert len(await _all_tasks(backend, root)) == count_before


@pytest.mark.asyncio
async def test_existing_non_terminal_link_suppresses_mint(interceptor, backend, root, caplog):
    pid = await _seed_carrier(interceptor, backend, root)
    existing_id = await _seed_carrier(
        interceptor, backend, root, title='Nightly check (manual)', status='pending',
    )
    count_before = len(await _all_tasks(backend, root))

    with caplog.at_level(logging.INFO, logger=MINT_LOGGER):
        assert 'error' not in await interceptor.set_task_status(pid, 'done', root)

    links = await _chain_links(backend, root, excluding=pid)
    assert [str(t['id']) for t in links] == [existing_id]
    assert len(await _all_tasks(backend, root)) == count_before
    assert any('recurrence_mint_skipped' in r.getMessage() for r in caplog.records), [
        r.getMessage() for r in caplog.records
    ]

    replay = await interceptor.set_task_status(pid, 'done', root)

    assert replay.get('no_op') is True, replay
    assert len(await _all_tasks(backend, root)) == count_before


def _mint_failures(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage() for r in caplog.records
        if r.levelno == logging.ERROR and r.getMessage().startswith('recurrence_mint_failed:')
    ]


@pytest.mark.asyncio
async def test_mint_reverifies_carrier_and_fails_soft(interceptor, backend, root, caplog):
    pid = await _seed_carrier(interceptor, backend, root)
    os.chmod(Path(root) / 'scripts' / 'check.sh', 0o644)

    with caplog.at_level(logging.ERROR, logger=MINT_LOGGER):
        result = await interceptor.set_task_status(pid, 'done', root)

    assert 'error' not in result, result
    assert (await backend.get_task(pid, project_root=root))['status'] == 'done'
    assert await _chain_links(backend, root, excluding=pid) == []
    failures = _mint_failures(caplog)
    assert len(failures) == 1, failures
    assert pid in failures[0]
    assert KEY in failures[0]


@pytest.mark.asyncio
async def test_backend_add_failure_fails_soft(tmp_path, event_buffer, caplog):
    project_root = str(tmp_path)
    failing_backend = _AddTaskFails(TaskmasterConfig(project_root=project_root))
    await failing_backend.start()
    try:
        interceptor = TaskInterceptor(failing_backend, None, event_buffer)
        pid = await _seed_carrier(interceptor, failing_backend, project_root)

        with caplog.at_level(logging.ERROR, logger=MINT_LOGGER):
            result = await interceptor.set_task_status(pid, 'done', project_root)

        assert 'error' not in result, result
        predecessor = await failing_backend.get_task(pid, project_root=project_root)
        assert predecessor['status'] == 'done'
        assert await _chain_links(failing_backend, project_root, excluding=pid) == []
    finally:
        await failing_backend.close()
    failures = _mint_failures(caplog)
    assert len(failures) == 1, failures


@pytest.mark.asyncio
async def test_mint_journals_task_created_event(interceptor, backend, root, event_buffer):
    pid = await _seed_carrier(interceptor, backend, root)
    assert 'error' not in await interceptor.set_task_status(pid, 'done', root)
    [successor] = await _chain_links(backend, root, excluding=pid)

    events = await event_buffer.peek_buffered(resolve_project_id(root), limit=100)

    created = [
        e for e in events
        if e.type == EventType.task_created and e.payload.get('task_id') == str(successor['id'])
    ]
    assert len(created) == 1, events
    assert created[0].payload['minted_from'] == pid
    assert created[0].payload['source'] == 'recurrence-mint'
