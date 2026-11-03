"""The curator's sibling stream, driven through TaskCurator.prepare_candidate over a real SQLite store.

An agent-followup candidate's pool always holds the parent's other
non-cancelled follow-ups, whatever files they touch (task 6227). The
embedding stream is switched off through the public config by pointing
``mem0.qdrant_url`` at a closed loopback port, so these cases never patch a
private name and never reach a live Qdrant.
"""

from __future__ import annotations

import itertools
import json
import logging
import subprocess
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from _curator_helpers import make_config

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import FusedMemoryConfig, TaskmasterConfig
from fused_memory.middleware.curator_types import (
    CandidateTask,
    PoolEntry,
    PreparedCandidate,
)
from fused_memory.middleware.task_curator import TaskCurator

_CLOSED_LOOPBACK_URL = 'http://127.0.0.1:9'
_FOLLOWUP = 'agent_followup'
_CANDIDATE_FILE = 'pkg_a/sub/lms_slate_run.py'
_MODULE_FILE = 'pkg_a/sub/neighbour.py'
_SIBLING_FILE = 'pkg_b/sub/lms_slate.py'
_SIBLING_TITLE_PREFIX = 'sibling '
# Wider than SQLite's INTEGER, so only its text form can be looked up.
_PARENT_ID_PAST_SQLITE_INTEGER = str(2**64)


@dataclass(frozen=True)
class SiblingStack:
    root: str
    backend: SqliteTaskBackend
    config: FusedMemoryConfig
    curator: TaskCurator
    # The store refuses a second task with the same (title, files) key.
    _serial: Iterator[int] = field(default_factory=lambda: itertools.count(1))

    async def seed(
        self, title: str, *, status: str = 'pending', metadata: dict[str, Any] | None = None,
    ) -> str:
        added = await self.backend.add_task(
            project_root=self.root,
            title=title,
            description=f'{title} (sibling-stream fixture)',
            metadata=json.dumps(metadata) if metadata is not None else None,
            status=status,
        )
        return str(added['id'])

    async def seed_sibling(self, parent_id: str | int, *, status: str = 'pending') -> str:
        return await self.seed(f'{_SIBLING_TITLE_PREFIX}{next(self._serial)}', status=status, metadata={
            'spawned_from': parent_id,
            'spawn_context': _FOLLOWUP,
            'files': [_SIBLING_FILE],
        })

    async def seed_module_task(self) -> str:
        return await self.seed(f'module neighbour {next(self._serial)}', metadata={
            'files': [_MODULE_FILE],
        })

    async def prepare(self, candidate: CandidateTask) -> PreparedCandidate:
        prepared = await self.curator.prepare_candidate(candidate, 'p', self.root)
        assert prepared.corpus_error is None
        return prepared


class _LookupFailsBackend(SqliteTaskBackend):
    async def find_tasks_by_metadata(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError('metadata lookup unavailable')


class _TextLookupFailsBackend(SqliteTaskBackend):
    async def find_tasks_by_metadata(
        self, project_root: str, key: str, value: str | int | None = None, **kwargs: Any,
    ) -> Any:
        if isinstance(value, str):
            raise RuntimeError('text metadata lookup unavailable')
        return await super().find_tasks_by_metadata(project_root, key, value, **kwargs)


class _SiblingFetchFailsBackend(SqliteTaskBackend):
    async def get_task(self, task_id: str, project_root: str, tag: str | None = None) -> dict:
        task = await super().get_task(task_id, project_root, tag)
        if str(task.get('title', '')).startswith(_SIBLING_TITLE_PREFIX):
            raise RuntimeError('task store unavailable')
        return task


@asynccontextmanager
async def _open_stack(
    project_dir: Path, backend_cls: type[SqliteTaskBackend],
) -> AsyncIterator[SiblingStack]:
    project_dir.mkdir(parents=True)
    subprocess.run(['git', 'init', '-q', str(project_dir)], check=True, capture_output=True)
    root = str(project_dir.resolve())
    backend = backend_cls(TaskmasterConfig(project_root=root))
    await backend.start()
    config = make_config()
    config.mem0.qdrant_url = _CLOSED_LOOPBACK_URL
    curator = TaskCurator(config=config, taskmaster=backend)
    try:
        yield SiblingStack(root, backend, config, curator)
    finally:
        await curator.close()
        await backend.close()


@pytest_asyncio.fixture
async def stack(tmp_path: Path) -> AsyncIterator[SiblingStack]:
    async with _open_stack(tmp_path / 'project', SqliteTaskBackend) as opened:
        yield opened


@pytest_asyncio.fixture
async def failing_lookup_stack(tmp_path: Path) -> AsyncIterator[SiblingStack]:
    async with _open_stack(tmp_path / 'project', _LookupFailsBackend) as opened:
        yield opened


def _followup(parent_id: str, *, files: list[str] | None = None) -> CandidateTask:
    return CandidateTask(
        title='Add a regression test for the slate runner',
        files_to_modify=[_CANDIDATE_FILE] if files is None else files,
        spawned_from=parent_id,
        spawn_context=_FOLLOWUP,
    )


def _entries(pool: list[PoolEntry], task_id: str) -> list[PoolEntry]:
    return [e for e in pool if e.task_id == task_id]


def _ids_from(pool: list[PoolEntry], source: str) -> set[str]:
    return {e.task_id for e in pool if e.source == source}


@pytest.mark.asyncio
async def test_an_agent_followup_pools_a_same_parent_sibling_filed_at_a_different_site(stack):
    parent_id = await stack.seed('parent')
    sibling_id = await stack.seed_sibling(parent_id)

    prepared = await stack.prepare(_followup(parent_id))

    assert [e.source for e in _entries(prepared.pool, sibling_id)] == ['sibling']
    assert prepared.pool_sizes['sibling'] == 1
    assert [e.source for e in _entries(prepared.pool, parent_id)] == ['anchor']


@pytest.mark.asyncio
async def test_siblings_are_pooled_when_the_candidate_has_no_module_keys(stack):
    parent_id = await stack.seed('parent')
    sibling_id = await stack.seed_sibling(parent_id)

    prepared = await stack.prepare(_followup(parent_id, files=[]))

    assert _ids_from(prepared.pool, 'sibling') == {sibling_id}


@pytest.mark.asyncio
async def test_a_cancelled_sibling_is_not_pooled(stack):
    parent_id = await stack.seed('parent')
    cancelled_id = await stack.seed_sibling(parent_id, status='cancelled')
    live_id = await stack.seed_sibling(parent_id)

    prepared = await stack.prepare(_followup(parent_id))

    assert _entries(prepared.pool, cancelled_id) == []
    assert _ids_from(prepared.pool, 'sibling') == {live_id}
    assert prepared.withheld is not None
    assert prepared.withheld.by_source['sibling'] == 0


@pytest.mark.asyncio
async def test_a_candidate_that_is_not_an_agent_followup_gets_no_sibling_stream(stack):
    parent_id = await stack.seed('parent')
    sibling_id = await stack.seed_sibling(parent_id)
    candidate = CandidateTask(
        title='Review finding on the slate runner',
        files_to_modify=[_CANDIDATE_FILE],
        spawned_from=parent_id,
        spawn_context='review',
    )

    prepared = await stack.prepare(candidate)

    assert _entries(prepared.pool, sibling_id) == []
    assert prepared.pool_sizes['sibling'] == 0
    assert _ids_from(prepared.pool, 'anchor') == {parent_id}


@pytest.mark.asyncio
async def test_a_sibling_the_module_stream_also_matches_is_pooled_once_as_a_sibling(stack):
    stack.config.curator.pool_module_cap = 2
    parent_id = await stack.seed('parent')
    module_ids = {await stack.seed_module_task() for _ in range(2)}
    sibling_id = await stack.seed('sibling', metadata={
        'spawned_from': parent_id, 'spawn_context': _FOLLOWUP, 'files': [_MODULE_FILE],
    })

    prepared = await stack.prepare(_followup(parent_id))

    assert [e.source for e in _entries(prepared.pool, sibling_id)] == ['sibling']
    assert _ids_from(prepared.pool, 'module') == module_ids
    assert prepared.pool_sizes['module'] == 2
    assert prepared.withheld is not None
    assert prepared.withheld.by_source['module'] == 0


@pytest.mark.asyncio
async def test_the_sibling_cap_admits_live_siblings_before_done_ones(stack):
    stack.config.curator.pool_sibling_cap = 2
    parent_id = await stack.seed('parent')
    pending_ids = {await stack.seed_sibling(parent_id) for _ in range(2)}
    for _ in range(3):
        await stack.seed_sibling(parent_id, status='done')

    prepared = await stack.prepare(_followup(parent_id))

    assert _ids_from(prepared.pool, 'sibling') == pending_ids
    assert prepared.withheld is not None
    assert prepared.withheld.by_source['sibling'] == 3
    assert prepared.withheld.caps['sibling'] == 2


@pytest.mark.asyncio
async def test_the_sibling_cap_admits_the_newest_among_equals(stack):
    stack.config.curator.pool_sibling_cap = 2
    parent_id = await stack.seed('parent')
    sibling_ids = [await stack.seed_sibling(parent_id) for _ in range(3)]

    prepared = await stack.prepare(_followup(parent_id))

    newest_two = set(sorted(sibling_ids, key=int)[1:])
    assert _ids_from(prepared.pool, 'sibling') == newest_two
    assert prepared.withheld is not None
    assert prepared.withheld.by_source['sibling'] == 1


@pytest.mark.asyncio
async def test_siblings_outlast_every_other_stream_when_the_total_cap_trims(stack):
    stack.config.curator.pool_module_cap = 5
    stack.config.curator.pool_total_cap = 4
    parent_id = await stack.seed('parent')
    sibling_ids = {await stack.seed_sibling(parent_id) for _ in range(2)}
    for _ in range(5):
        await stack.seed_module_task()

    prepared = await stack.prepare(_followup(parent_id))

    pooled = {e.task_id for e in prepared.pool}
    assert sibling_ids | {parent_id} <= pooled
    assert len(prepared.pool) == 4
    assert prepared.withheld is not None
    assert prepared.withheld.by_source['total_cap'] == 4


@pytest.mark.asyncio
async def test_an_integer_spawned_from_names_the_same_parent(stack):
    parent_id = await stack.seed('parent')
    sibling_id = await stack.seed_sibling(int(parent_id))

    prepared = await stack.prepare(_followup(parent_id))

    assert _ids_from(prepared.pool, 'sibling') == {sibling_id}


@pytest.mark.asyncio
async def test_a_failed_sibling_lookup_leaves_the_rest_of_the_pool(failing_lookup_stack, caplog):
    stack = failing_lookup_stack
    parent_id = await stack.seed('parent')
    await stack.seed_sibling(parent_id)
    caplog.set_level(logging.WARNING, logger='fused_memory.middleware.task_curator')

    prepared = await stack.prepare(_followup(parent_id))

    assert _ids_from(prepared.pool, 'anchor') == {parent_id}
    assert prepared.pool_sizes['sibling'] == 0
    assert any(
        record.name == 'fused_memory.middleware.task_curator'
        and record.levelno == logging.WARNING
        and 'sibling' in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_a_zero_sibling_cap_disables_the_stream(stack):
    stack.config.curator.pool_sibling_cap = 0
    parent_id = await stack.seed('parent')
    sibling_id = await stack.seed_sibling(parent_id)

    prepared = await stack.prepare(_followup(parent_id))

    assert _entries(prepared.pool, sibling_id) == []
    assert prepared.pool_sizes['sibling'] == 0
    assert prepared.withheld is not None
    assert prepared.withheld.caps['sibling'] == 0


@pytest.mark.asyncio
async def test_a_parent_id_too_wide_for_an_integer_still_pools_its_text_stored_siblings(stack):
    sibling_id = await stack.seed_sibling(_PARENT_ID_PAST_SQLITE_INTEGER)

    prepared = await stack.prepare(_followup(_PARENT_ID_PAST_SQLITE_INTEGER))

    assert _ids_from(prepared.pool, 'sibling') == {sibling_id}


@pytest.mark.asyncio
async def test_a_failed_text_lookup_keeps_the_integer_stored_siblings(tmp_path, caplog):
    caplog.set_level(logging.WARNING, logger='fused_memory.middleware.task_curator')
    async with _open_stack(tmp_path / 'project', _TextLookupFailsBackend) as stack:
        parent_id = await stack.seed('parent')
        sibling_id = await stack.seed_sibling(int(parent_id))

        prepared = await stack.prepare(_followup(parent_id))

    assert _ids_from(prepared.pool, 'sibling') == {sibling_id}
    assert any(
        record.levelno == logging.WARNING and repr(parent_id) in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_a_sibling_whose_full_fetch_fails_is_pooled_thin_from_its_lookup_row(tmp_path):
    async with _open_stack(tmp_path / 'project', _SiblingFetchFailsBackend) as stack:
        parent_id = await stack.seed('parent')
        sibling_id = await stack.seed_sibling(parent_id)

        prepared = await stack.prepare(_followup(parent_id))

    [entry] = _entries(prepared.pool, sibling_id)
    assert entry.source == 'sibling'
    assert entry.title.startswith(_SIBLING_TITLE_PREFIX)
    assert entry.status == 'unknown'
    assert entry.combine_eligible is False
