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
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from _fm_helpers import poll_until

from fused_memory.backends import sqlite_task_backend
from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.backends.task_backend_types import AddTaskResult
from fused_memory.config.schema import TaskmasterConfig
from fused_memory.middleware.recurrence_mint import (
    RECURRENCE_MINT_SOURCE,
    MintOutcomeKind,
    broken_chains,
    mint_successor,
    remint_broken_chain,
    run_remint_sweep_loop,
)
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.models.reconciliation import EventType
from fused_memory.models.scope import resolve_project_id
from fused_memory.reconciliation.event_buffer import EventBuffer

KEY = 'nightly-check'
INTERVAL_SECS = 86400
BEFORE_DONE = {'kind': 'predicate', 'script': 'scripts/check.sh', 'timeout_secs': 60}
FILES = ['scripts/check.sh']
FEATURE_TAG = 'feature'
MINT_LOGGER = 'fused_memory.middleware.recurrence_mint'
STORE_CLOCK_START = datetime(2026, 9, 1, 12, 0, 0, 250000, tzinfo=UTC)


def _is_mint_insert(metadata: str | None) -> bool:
    return metadata is not None and RECURRENCE_MINT_SOURCE in metadata


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
        if _is_mint_insert(metadata):
            raise RuntimeError('boom')
        return await super().add_task(
            project_root, prompt, title, description, details, dependencies,
            priority, metadata, tag, status,
        )


class _GatedMintInsert(SqliteTaskBackend):
    """Backend whose insert of a minted successor waits until the test releases it."""

    def __init__(self, config: TaskmasterConfig) -> None:
        super().__init__(config)
        self.mint_entered = asyncio.Event()
        self.mint_released = asyncio.Event()

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
        if _is_mint_insert(metadata):
            self.mint_entered.set()
            await self.mint_released.wait()
        return await super().add_task(
            project_root, prompt, title, description, details, dependencies,
            priority, metadata, tag, status,
        )


class _FailingMintInsert(SqliteTaskBackend):
    """Backend whose insert of a minted successor raises when ``_fails`` says so."""

    def _fails(self, successor_metadata: dict[str, Any]) -> bool:
        raise NotImplementedError

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
        if metadata is not None and _is_mint_insert(metadata) and self._fails(json.loads(metadata)):
            raise RuntimeError('boom')
        return await super().add_task(
            project_root, prompt, title, description, details, dependencies,
            priority, metadata, tag, status,
        )


class _FirstMintInsertFails(_FailingMintInsert):
    def __init__(self, config: TaskmasterConfig) -> None:
        super().__init__(config)
        self._mint_inserts = 0

    def _fails(self, successor_metadata: dict[str, Any]) -> bool:
        self._mint_inserts += 1
        return self._mint_inserts == 1


class _MintInsertFailsForKey(_FailingMintInsert):
    def __init__(self, config: TaskmasterConfig, key: str) -> None:
        super().__init__(config)
        self._key = key

    def _fails(self, successor_metadata: dict[str, Any]) -> bool:
        return successor_metadata['recurrence']['key'] == self._key


class _StoreClock:
    """Stands in for the task store's clock, in its ``updatedAt`` string shape."""

    def __init__(self) -> None:
        self.now = STORE_CLOCK_START

    def advance(self, delta: timedelta) -> None:
        self.now += delta

    def __call__(self) -> str:
        return self.now.strftime('%Y-%m-%dT%H:%M:%S.') + f'{self.now.microsecond // 1000:03d}Z'


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


@pytest_asyncio.fixture
async def gated_backend(tmp_path):
    b = _GatedMintInsert(TaskmasterConfig(project_root=str(tmp_path)))
    await b.start()
    yield b
    await b.close()


@pytest_asyncio.fixture
async def first_mint_fails_backend(tmp_path):
    b = _FirstMintInsertFails(TaskmasterConfig(project_root=str(tmp_path)))
    await b.start()
    yield b
    await b.close()


@pytest_asyncio.fixture
async def chain_a_fails_backend(tmp_path):
    b = _MintInsertFailsForKey(TaskmasterConfig(project_root=str(tmp_path)), 'chain-a')
    await b.start()
    yield b
    await b.close()


@pytest.fixture
def root(tmp_path) -> str:
    return str(tmp_path)


@pytest.fixture
def store_clock(monkeypatch) -> _StoreClock:
    clock = _StoreClock()
    monkeypatch.setattr(sqlite_task_backend, 'task_timestamp_now', clock)
    return clock


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
    tag: str | None = None,
    files: list[str] | None = FILES,
    recurrence: dict[str, Any] | None = None,
) -> str:
    _write_check_script(project_root)
    metadata: dict[str, Any] = {
        'task_kind': 'deterministic',
        'before_done': BEFORE_DONE,
        'milestone': {'mode': 'dated', 'at': '2026-09-01T00:00:00+00:00'},
        'recurrence': recurrence or {'key': key, 'interval_secs': interval_secs},
    }
    if files is not None:
        metadata['files'] = files
    added = await backend.add_task(
        project_root=project_root,
        title=title,
        description='Run the nightly check',
        details='details text',
        priority='high',
        metadata=json.dumps(metadata),
        tag=tag,
    )
    task_id = str(added['id'])
    if status != 'pending':
        result = await interceptor.set_task_status(task_id, status, project_root, tag=tag)
        assert 'error' not in result, result
    return task_id


async def _all_tasks(
    backend: SqliteTaskBackend, project_root: str, tag: str | None = None,
) -> list[dict[str, Any]]:
    return list((await backend.get_tasks(project_root, tag))['tasks'])


async def _chain_links(
    backend: SqliteTaskBackend,
    project_root: str,
    *,
    excluding: str,
    key: str = KEY,
    tag: str | None = None,
) -> list[dict[str, Any]]:
    return [
        t for t in await _all_tasks(backend, project_root, tag)
        if (t.get('metadata') or {}).get('recurrence', {}).get('key') == key
        and str(t['id']) != excluding
    ]


@pytest.mark.asyncio
async def test_done_mints_exactly_one_successor_per_c3(interceptor, backend, root, store_clock):
    pid = await _seed_carrier(interceptor, backend, root)

    result = await interceptor.set_task_status(pid, 'done', root)

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
    assert md['milestone'] == {'mode': 'dated', 'at': '2026-09-02T12:00:00+00:00'}
    assert successor['title'] == 'Nightly check [due 2026-09-02T12:00:00+00:00]'
    predecessor = await backend.get_task(pid, project_root=root)
    assert predecessor['status'] == 'done'
    terminal_time = datetime.fromisoformat(predecessor['updatedAt'])
    assert datetime.fromisoformat(md['milestone']['at']) == (
        terminal_time + timedelta(seconds=INTERVAL_SECS)
    ).replace(microsecond=0)


@pytest.mark.asyncio
async def test_chain_titles_do_not_accumulate_run_labels(interceptor, backend, root, store_clock):
    pid = await _seed_carrier(interceptor, backend, root)
    assert 'error' not in await interceptor.set_task_status(pid, 'done', root)
    [link2] = await _chain_links(backend, root, excluding=pid)
    link2_id = str(link2['id'])
    assert link2['title'] == 'Nightly check [due 2026-09-02T12:00:00+00:00]'

    store_clock.advance(timedelta(days=1))
    assert 'error' not in await interceptor.set_task_status(link2_id, 'in-progress', root)
    assert 'error' not in await interceptor.set_task_status(link2_id, 'done', root)

    link3s = [
        t for t in await _chain_links(backend, root, excluding=pid)
        if t['metadata']['recurrence'].get('minted_from') == link2_id
    ]
    assert len(link3s) == 1, link3s
    assert link3s[0]['title'] == 'Nightly check [due 2026-09-03T12:00:00+00:00]'
    assert (await backend.get_task(link2_id, project_root=root))['status'] == 'done'


@pytest.mark.asyncio
@pytest.mark.parametrize('files', [None, FILES])
async def test_successor_lands_in_the_carriers_tag_copying_files_only_when_present(
    files, interceptor, backend, root,
):
    pid = await _seed_carrier(interceptor, backend, root, tag=FEATURE_TAG, files=files)

    assert 'error' not in await interceptor.set_task_status(pid, 'done', root, tag=FEATURE_TAG)

    [successor] = await _chain_links(backend, root, excluding=pid, tag=FEATURE_TAG)
    assert successor['status'] == 'pending'
    assert successor['metadata']['recurrence']['minted_from'] == pid
    copied_files = {k: v for k, v in successor['metadata'].items() if k == 'files'}
    assert copied_files == ({} if files is None else {'files': files})
    assert await _chain_links(backend, root, excluding=pid) == []


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
@pytest.mark.parametrize('tag', [None, FEATURE_TAG])
async def test_existing_non_terminal_link_suppresses_mint(tag, interceptor, backend, root, caplog):
    pid = await _seed_carrier(interceptor, backend, root, tag=tag)
    existing_id = await _seed_carrier(
        interceptor, backend, root, title='Nightly check (manual)', status='pending', tag=tag,
    )
    count_before = len(await _all_tasks(backend, root, tag))

    with caplog.at_level(logging.INFO, logger=MINT_LOGGER):
        assert 'error' not in await interceptor.set_task_status(pid, 'done', root, tag=tag)

    links = await _chain_links(backend, root, excluding=pid, tag=tag)
    assert [str(t['id']) for t in links] == [existing_id]
    assert len(await _all_tasks(backend, root, tag)) == count_before
    assert any('recurrence_mint_skipped' in r.getMessage() for r in caplog.records), [
        r.getMessage() for r in caplog.records
    ]

    replay = await interceptor.set_task_status(pid, 'done', root, tag=tag)

    assert replay.get('no_op') is True, replay
    assert len(await _all_tasks(backend, root, tag)) == count_before


def _mint_failures(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage() for r in caplog.records
        if r.levelno == logging.ERROR and r.getMessage().startswith('recurrence_mint_failed:')
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'recurrence',
    [{'interval_secs': INTERVAL_SECS}, {'key': '', 'interval_secs': INTERVAL_SECS}],
    ids=['no-key', 'empty-key'],
)
async def test_a_recurrence_without_a_chain_key_is_not_a_carrier(
    recurrence, interceptor, backend, root, caplog,
):
    task_id = await _seed_carrier(interceptor, backend, root, recurrence=recurrence)
    count_before = len(await _all_tasks(backend, root))

    with caplog.at_level(logging.ERROR, logger=MINT_LOGGER):
        assert 'error' not in await interceptor.set_task_status(task_id, 'done', root)
        assert await interceptor.remint_broken_recurrence_chains(root) == ()

    assert len(await _all_tasks(backend, root)) == count_before
    assert _mint_failures(caplog) == []


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
async def test_link_completed_in_its_predecessors_second_fails_soft(
    interceptor, backend, root, store_clock, caplog,
):
    pid = await _seed_carrier(interceptor, backend, root)
    assert 'error' not in await interceptor.set_task_status(pid, 'done', root)
    [link2] = await _chain_links(backend, root, excluding=pid)
    link2_id = str(link2['id'])
    assert 'error' not in await interceptor.set_task_status(link2_id, 'in-progress', root)

    with caplog.at_level(logging.ERROR, logger=MINT_LOGGER):
        result = await interceptor.set_task_status(link2_id, 'done', root)

    assert 'error' not in result, result
    assert (await backend.get_task(link2_id, project_root=root))['status'] == 'done'
    assert [str(t['id']) for t in await _chain_links(backend, root, excluding=pid)] == [link2_id]
    failures = _mint_failures(caplog)
    assert len(failures) == 1, failures
    assert f'predecessor={link2_id}' in failures[0]
    assert 'DuplicateCandidateKeyError' in failures[0]


@pytest.mark.asyncio
async def test_cancelled_request_still_mints_the_successor(gated_backend, event_buffer, root):
    interceptor = TaskInterceptor(gated_backend, None, event_buffer)
    pid = await _seed_carrier(interceptor, gated_backend, root)
    request = asyncio.create_task(interceptor.set_task_status(pid, 'done', root))
    await poll_until(gated_backend.mint_entered.is_set)

    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    gated_backend.mint_released.set()
    await interceptor.drain()

    assert (await gated_backend.get_task(pid, project_root=root))['status'] == 'done'
    assert len(await _chain_links(gated_backend, root, excluding=pid)) == 1


@pytest.mark.asyncio
async def test_cancelled_mint_logs_the_failure_and_reraises(
    gated_backend, event_buffer, root, caplog,
):
    interceptor = TaskInterceptor(gated_backend, None, event_buffer)
    pid = await _seed_carrier(interceptor, gated_backend, root)
    await gated_backend.set_task_status(pid, 'done', root)
    mint = asyncio.create_task(mint_successor(
        gated_backend, write_lock=asyncio.Lock(), predecessor_id=pid, project_root=root, tag=None,
    ))
    await poll_until(gated_backend.mint_entered.is_set)

    with caplog.at_level(logging.ERROR, logger=MINT_LOGGER):
        mint.cancel()
        with pytest.raises(asyncio.CancelledError):
            await mint

    failures = _mint_failures(caplog)
    assert len(failures) == 1, failures
    assert f'predecessor={pid}' in failures[0]
    assert KEY in failures[0]
    assert 'reason=cancelled' in failures[0]
    assert await _chain_links(gated_backend, root, excluding=pid) == []


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
    assert created[0].payload['trigger'] == 'done_write'


# ── Re-mint sweep (task 6572): repair chains whose current link is done ──


def _remints(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage() for r in caplog.records
        if r.levelno == logging.WARNING and r.getMessage().startswith('recurrence_remint:')
    ]


async def _done_without_mint(
    backend: SqliteTaskBackend, task_id: str, project_root: str, tag: str | None = None,
) -> None:
    """Commit ``done`` straight through the store, as a process that died before minting."""
    await backend.set_task_status(task_id, 'done', project_root, tag)


@pytest.mark.asyncio
async def test_sweep_repairs_a_chain_whose_mint_failed(
    first_mint_fails_backend, event_buffer, root, caplog,
):
    backend = first_mint_fails_backend
    interceptor = TaskInterceptor(backend, None, event_buffer)
    pid = await _seed_carrier(interceptor, backend, root)

    with caplog.at_level(logging.INFO, logger=MINT_LOGGER):
        assert 'error' not in await interceptor.set_task_status(pid, 'done', root)
        assert len(_mint_failures(caplog)) == 1
        assert await _chain_links(backend, root, excluding=pid) == []

        outcomes = await interceptor.remint_broken_recurrence_chains(root)

    [successor] = await _chain_links(backend, root, excluding=pid)
    successor_id = str(successor['id'])
    assert [(o.kind, o.successor_id) for o in outcomes] == [(MintOutcomeKind.MINTED, successor_id)]
    assert successor['status'] == 'pending'
    md = successor['metadata']
    assert md['recurrence'] == {'key': KEY, 'interval_secs': INTERVAL_SECS, 'minted_from': pid}
    assert md['source'] == 'recurrence-mint'
    assert successor['title'] == f"Nightly check [due {md['milestone']['at']}]"
    remints = _remints(caplog)
    assert len(remints) == 1, remints
    assert f'predecessor={pid}' in remints[0]
    assert KEY in remints[0]
    assert f'successor={successor_id}' in remints[0]


@pytest.mark.asyncio
async def test_sweep_after_restart_repairs_a_done_link_that_never_minted(
    interceptor, backend, root, event_buffer,
):
    pid = await _seed_carrier(interceptor, backend, root)
    await _done_without_mint(backend, pid, root)
    restarted = TaskInterceptor(backend, None, event_buffer)

    await restarted.remint_broken_recurrence_chains(root)

    [successor] = await _chain_links(backend, root, excluding=pid)
    assert successor['status'] == 'pending'
    assert successor['metadata']['recurrence']['minted_from'] == pid


@pytest.mark.asyncio
async def test_swept_successor_is_anchored_on_the_persisted_terminal_time(
    interceptor, backend, root, store_clock,
):
    pid = await _seed_carrier(interceptor, backend, root)
    await _done_without_mint(backend, pid, root)
    store_clock.advance(timedelta(hours=3))

    await interceptor.remint_broken_recurrence_chains(root)

    [successor] = await _chain_links(backend, root, excluding=pid)
    terminal_time = datetime.fromisoformat((await backend.get_task(pid, root))['updatedAt'])
    at = successor['metadata']['milestone']['at']
    assert at == '2026-09-02T12:00:00+00:00'
    assert datetime.fromisoformat(at) == (
        terminal_time + timedelta(seconds=INTERVAL_SECS)
    ).replace(microsecond=0)


@pytest.mark.asyncio
async def test_rerunning_the_sweep_does_not_double_mint(interceptor, backend, root, caplog):
    pid = await _seed_carrier(interceptor, backend, root)
    await _done_without_mint(backend, pid, root)
    await interceptor.remint_broken_recurrence_chains(root)
    count_after_repair = len(await _all_tasks(backend, root))
    caplog.clear()

    with caplog.at_level(logging.INFO, logger=MINT_LOGGER):
        outcomes = await interceptor.remint_broken_recurrence_chains(root)

    assert MintOutcomeKind.MINTED not in {o.kind for o in outcomes}
    assert len(await _all_tasks(backend, root)) == count_after_repair
    assert _remints(caplog) == []


@pytest.mark.asyncio
async def test_sweep_leaves_a_healthy_chain_alone(interceptor, backend, root, caplog):
    pid = await _seed_carrier(interceptor, backend, root)
    assert 'error' not in await interceptor.set_task_status(pid, 'done', root)
    count_before = len(await _all_tasks(backend, root))

    with caplog.at_level(logging.INFO, logger=MINT_LOGGER):
        outcomes = await interceptor.remint_broken_recurrence_chains(root)

    assert outcomes == ()
    assert len(await _all_tasks(backend, root)) == count_before
    assert _remints(caplog) == []


@pytest.mark.asyncio
async def test_sweep_leaves_a_cancelled_carrier_ended(interceptor, backend, root):
    pid = await _seed_carrier(interceptor, backend, root)
    assert 'error' not in await interceptor.set_task_status(pid, 'cancelled', root)
    count_before = len(await _all_tasks(backend, root))

    assert await interceptor.remint_broken_recurrence_chains(root) == ()

    assert len(await _all_tasks(backend, root)) == count_before


@pytest.mark.asyncio
async def test_sweep_does_not_renew_a_done_link_whose_successor_was_cancelled(
    interceptor, backend, root,
):
    pid = await _seed_carrier(interceptor, backend, root)
    assert 'error' not in await interceptor.set_task_status(pid, 'done', root)
    [link2] = await _chain_links(backend, root, excluding=pid)
    assert 'error' not in await interceptor.set_task_status(str(link2['id']), 'cancelled', root)
    count_before = len(await _all_tasks(backend, root))

    assert await interceptor.remint_broken_recurrence_chains(root) == ()

    assert len(await _all_tasks(backend, root)) == count_before


@pytest.mark.asyncio
async def test_a_stale_sweep_snapshot_does_not_revive_a_chain_cancelled_since(
    interceptor, backend, root, caplog,
):
    pid = await _seed_carrier(interceptor, backend, root)
    await _done_without_mint(backend, pid, root)
    [stale] = await broken_chains(backend, root)
    await interceptor.remint_broken_recurrence_chains(root)
    [link2] = await _chain_links(backend, root, excluding=pid)
    assert 'error' not in await interceptor.set_task_status(str(link2['id']), 'cancelled', root)
    count_before = len(await _all_tasks(backend, root))
    caplog.clear()

    with caplog.at_level(logging.INFO, logger=MINT_LOGGER):
        outcome = await remint_broken_chain(
            backend, write_lock=asyncio.Lock(), chain=stale, project_root=root,
        )

    assert outcome.kind is MintOutcomeKind.SKIPPED_NOT_BROKEN
    assert len(await _all_tasks(backend, root)) == count_before
    assert _remints(caplog) == []
    assert _mint_failures(caplog) == []


@pytest.mark.asyncio
async def test_older_live_link_keeps_a_newer_done_link_from_reading_broken(
    interceptor, backend, root,
):
    existing_id = await _seed_carrier(
        interceptor, backend, root, title='Nightly check (manual)', status='pending',
    )
    pid = await _seed_carrier(interceptor, backend, root)
    assert 'error' not in await interceptor.set_task_status(pid, 'done', root)
    count_before = len(await _all_tasks(backend, root))

    assert await interceptor.remint_broken_recurrence_chains(root) == ()

    assert len(await _all_tasks(backend, root)) == count_before
    links = await _chain_links(backend, root, excluding=pid)
    assert [str(t['id']) for t in links] == [existing_id]


@pytest.mark.asyncio
async def test_sweep_repairs_a_broken_chain_in_its_own_non_default_tag(
    interceptor, backend, root,
):
    pid = await _seed_carrier(interceptor, backend, root, tag=FEATURE_TAG)
    await _done_without_mint(backend, pid, root, FEATURE_TAG)

    await interceptor.remint_broken_recurrence_chains(root)

    [successor] = await _chain_links(backend, root, excluding=pid, tag=FEATURE_TAG)
    assert successor['metadata']['recurrence']['minted_from'] == pid
    assert await _all_tasks(backend, root) == []


@pytest.mark.asyncio
async def test_sweep_ignores_done_tasks_outside_any_chain(interceptor, backend, root):
    added = await backend.add_task(project_root=root, title='plain task')
    await backend.set_task_status(str(added['id']), 'done', root)
    count_before = len(await _all_tasks(backend, root))

    assert await interceptor.remint_broken_recurrence_chains(root) == ()

    assert len(await _all_tasks(backend, root)) == count_before


@pytest.mark.asyncio
async def test_one_unmintable_chain_does_not_stop_the_others(
    chain_a_fails_backend, event_buffer, root, caplog,
):
    backend = chain_a_fails_backend
    interceptor = TaskInterceptor(backend, None, event_buffer)
    a_id = await _seed_carrier(interceptor, backend, root, key='chain-a', title='Chain A')
    b_id = await _seed_carrier(interceptor, backend, root, key='chain-b', title='Chain B')
    await _done_without_mint(backend, a_id, root)
    await _done_without_mint(backend, b_id, root)

    with caplog.at_level(logging.ERROR, logger=MINT_LOGGER):
        outcomes = await interceptor.remint_broken_recurrence_chains(root)

    assert sorted(o.kind for o in outcomes) == sorted(
        [MintOutcomeKind.FAILED, MintOutcomeKind.MINTED],
    )
    [b_successor] = await _chain_links(backend, root, excluding=b_id, key='chain-b')
    assert b_successor['metadata']['recurrence']['minted_from'] == b_id
    assert await _chain_links(backend, root, excluding=a_id, key='chain-a') == []
    failures = _mint_failures(caplog)
    assert len(failures) == 1, failures
    assert 'chain-a' in failures[0]


@pytest.mark.asyncio
async def test_sweep_journals_the_repair_as_a_task_created_event(
    interceptor, backend, root, event_buffer,
):
    pid = await _seed_carrier(interceptor, backend, root)
    await _done_without_mint(backend, pid, root)

    await interceptor.remint_broken_recurrence_chains(root)

    [successor] = await _chain_links(backend, root, excluding=pid)
    events = await event_buffer.peek_buffered(resolve_project_id(root), limit=100)
    created = [
        e for e in events
        if e.type == EventType.task_created and e.payload.get('task_id') == str(successor['id'])
    ]
    assert len(created) == 1, events
    assert created[0].payload['minted_from'] == pid
    assert created[0].payload['source'] == 'recurrence-mint'
    assert created[0].payload['trigger'] == 'remint_sweep'



# ── Re-mint sweep loop: startup sweep, then every interval ──


class _RecordingRemint:
    """Stands in for the interceptor's sweep: records each root it is handed."""

    def __init__(
        self, *, roots: Sequence[str], sweeps: int, failing_root: str | None = None,
    ) -> None:
        self.seen: list[str] = []
        self.swept_enough = asyncio.Event()
        self._roots = roots
        self._sweeps = sweeps
        self._failing_root = failing_root

    async def __call__(self, root: str) -> None:
        self.seen.append(root)
        if all(self.seen.count(r) >= self._sweeps for r in self._roots):
            self.swept_enough.set()
        if root == self._failing_root:
            raise RuntimeError('boom')


def _real_roots(tmp_path: Path) -> tuple[str, str]:
    a, b = tmp_path / 'a', tmp_path / 'b'
    a.mkdir()
    b.mkdir()
    return str(a), str(b)


async def _run_loop_until_swept(
    remint: _RecordingRemint, project_roots: Sequence[str], *, interval_secs: float,
) -> None:
    loop = asyncio.create_task(
        run_remint_sweep_loop(remint, project_roots, interval_secs=interval_secs),
    )
    await poll_until(remint.swept_enough.is_set)
    loop.cancel()
    with pytest.raises(asyncio.CancelledError):
        await loop


@pytest.mark.asyncio
async def test_loop_sweeps_every_root_at_startup_before_its_first_sleep(tmp_path):
    a, b = _real_roots(tmp_path)
    remint = _RecordingRemint(roots=[a, b], sweeps=1)

    await _run_loop_until_swept(remint, [a, b], interval_secs=3600)

    assert remint.seen == [a, b]


@pytest.mark.asyncio
async def test_loop_sweeps_again_every_interval(tmp_path):
    a, b = _real_roots(tmp_path)
    remint = _RecordingRemint(roots=[a, b], sweeps=2)

    await _run_loop_until_swept(remint, [a, b], interval_secs=0.01)

    assert remint.seen[:4] == [a, b, a, b]


@pytest.mark.asyncio
async def test_one_roots_failure_does_not_stop_the_sweep(tmp_path, caplog):
    a, b = _real_roots(tmp_path)
    remint = _RecordingRemint(roots=[a, b], sweeps=2, failing_root=a)

    with caplog.at_level(logging.ERROR, logger=MINT_LOGGER):
        await _run_loop_until_swept(remint, [a, b], interval_secs=0.01)

    assert remint.seen[:4] == [a, b, a, b]
    failures = [
        r for r in caplog.records
        if r.levelno == logging.ERROR
        and r.getMessage().startswith('recurrence_remint_sweep_failed:')
    ]
    assert failures, [r.getMessage() for r in caplog.records]
    assert a in failures[0].getMessage()
    assert failures[0].exc_info is not None


@pytest.mark.asyncio
async def test_loop_never_sweeps_a_phantom_root(tmp_path):
    a, b = _real_roots(tmp_path)
    gone = tmp_path / 'gone'
    remint = _RecordingRemint(roots=[a, b], sweeps=2)

    await _run_loop_until_swept(remint, [a, str(gone), b], interval_secs=0.01)

    assert str(gone) not in remint.seen
    assert not gone.exists()
