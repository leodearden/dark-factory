"""The curator's corpus streams, driven through TaskCurator.prepare_candidate over a real SQLite store.

An agent-followup candidate's pool always holds the parent's other
non-cancelled follow-ups, whatever files they touch (task 6227). The module
and dependency streams read only the open backlog (task 6708). The
embedding stream is switched off through the public config by pointing
``mem0.qdrant_url`` at a closed loopback port, so these cases never patch a
private name and never reach a live Qdrant; the embedding stream is
therefore unavailable in every case here.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import subprocess
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from _curator_helpers import agent_result, invoke_by_role, make_config, must_not_invoke, tripwire
from shared.cli_invoke import AgentResult
from shared.task_statuses import ACTIVE
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import FusedMemoryConfig, TaskmasterConfig
from fused_memory.middleware.curation_deferral import DeferReason
from fused_memory.middleware.curator_types import (
    CandidateTask,
    PoolEntry,
    PreparedCandidate,
)
from fused_memory.middleware.task_curator import TaskCurator
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.middleware.ticket_store import TicketStatus, TicketStore
from fused_memory.reconciliation.event_buffer import EventBuffer

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

_CLOSED_LOOPBACK_URL = 'http://127.0.0.1:9'
_FOLLOWUP = 'agent_followup'
_CANDIDATE_FILE = 'pkg_a/sub/lms_slate_run.py'
_MODULE_FILE = 'pkg_a/sub/neighbour.py'
_SIBLING_FILE = 'pkg_b/sub/lms_slate.py'
_SIBLING_TITLE_PREFIX = 'sibling '
# Wider than SQLite's INTEGER, so only its text form can be looked up.
_PARENT_ID_PAST_SQLITE_INTEGER = str(2**64)


@dataclass(frozen=True)
class CorpusStack:
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
            description=f'{title} (corpus-stream fixture)',
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


class _RecordingBackend(SqliteTaskBackend):
    """Records the statuses every get_tasks call asked for and the id every get_task call read."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.get_tasks_statuses: list[list[str] | None] = []
        self.get_task_ids: list[str] = []

    async def get_tasks(
        self, project_root: str, tag: str | None = None,
        statuses: list[str] | None = None, fields: Any = None,
    ) -> Any:
        self.get_tasks_statuses.append(statuses)
        return await super().get_tasks(project_root, tag, statuses=statuses, fields=fields)

    async def get_task(self, task_id: str, project_root: str, tag: str | None = None) -> dict:
        self.get_task_ids.append(str(task_id))
        return await super().get_task(task_id, project_root, tag)


class _ModuleReadFailsBackend(SqliteTaskBackend):
    async def get_tasks(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError('task tree read unavailable')


class _DependencyReadFailsBackend(SqliteTaskBackend):
    async def get_dependency_edges(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError('dependency edges unavailable')


class _FetchFailsBackend(SqliteTaskBackend):
    """A store whose get_task raises for every id in ``failing_ids``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.failing_ids: set[str] = set()

    async def get_task(self, task_id: str, project_root: str, tag: str | None = None) -> dict:
        if str(task_id) in self.failing_ids:
            raise RuntimeError('task store unavailable')
        return await super().get_task(task_id, project_root, tag)


class _SiblingFetchFailsBackend(SqliteTaskBackend):
    async def get_task(self, task_id: str, project_root: str, tag: str | None = None) -> dict:
        task = await super().get_task(task_id, project_root, tag)
        if str(task.get('title', '')).startswith(_SIBLING_TITLE_PREFIX):
            raise RuntimeError('task store unavailable')
        return task


@asynccontextmanager
async def _open_stack(
    project_dir: Path,
    backend_cls: type[SqliteTaskBackend],
    *,
    invoke: AsyncMock | None = None,
) -> AsyncIterator[CorpusStack]:
    project_dir.mkdir(parents=True)
    subprocess.run(['git', 'init', '-q', str(project_dir)], check=True, capture_output=True)
    root = str(project_dir.resolve())
    backend = backend_cls(TaskmasterConfig(project_root=root))
    await backend.start()
    config = make_config()
    config.mem0.qdrant_url = _CLOSED_LOOPBACK_URL
    curator = TaskCurator(
        config=config, taskmaster=backend,
        invoke=invoke if invoke is not None else must_not_invoke(),
    )
    try:
        yield CorpusStack(root, backend, config, curator)
    finally:
        await curator.close()
        await backend.close()


@pytest_asyncio.fixture
async def stack(tmp_path: Path) -> AsyncIterator[CorpusStack]:
    async with _open_stack(tmp_path / 'project', SqliteTaskBackend) as opened:
        yield opened


@pytest_asyncio.fixture
async def recording_stack(tmp_path: Path) -> AsyncIterator[CorpusStack]:
    async with _open_stack(tmp_path / 'project', _RecordingBackend) as opened:
        yield opened


@pytest_asyncio.fixture
async def failing_lookup_stack(tmp_path: Path) -> AsyncIterator[CorpusStack]:
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
    assert prepared.withheld is not None
    assert 'sibling' in prepared.withheld.unavailable_streams
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
    assert prepared.withheld is not None
    assert 'sibling' not in prepared.withheld.unavailable_streams
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


def _recorded(stack: CorpusStack) -> _RecordingBackend:
    assert isinstance(stack.backend, _RecordingBackend)
    return stack.backend


def _review_of(anchor_id: str) -> CandidateTask:
    """A review candidate with no files: only the anchor, embedding and dependency streams apply."""
    return CandidateTask(
        title='Review finding on the anchor', spawned_from=anchor_id, spawn_context='review',
    )


@pytest.mark.asyncio
async def test_the_module_stream_reads_only_the_open_backlog(recording_stack):
    stack = recording_stack
    module = {'files': [_MODULE_FILE]}
    pending_id = await stack.seed('pending neighbour', metadata=module)
    in_progress_id = await stack.seed('in-progress neighbour', status='in-progress', metadata=module)
    done_id = await stack.seed('done neighbour', status='done', metadata=module)

    prepared = await stack.prepare(CandidateTask(title='t', files_to_modify=[_CANDIDATE_FILE]))

    recorded = _recorded(stack).get_tasks_statuses
    assert recorded
    open_backlog = {status.value for status in ACTIVE}
    assert all(statuses is not None and set(statuses) == open_backlog for statuses in recorded)
    assert _ids_from(prepared.pool, 'module') == {pending_id, in_progress_id}
    assert _entries(prepared.pool, done_id) == []


@pytest.mark.asyncio
async def test_the_dependency_stream_pools_the_anchors_open_neighbours(recording_stack):
    stack = recording_stack
    anchor_id = await stack.seed('anchor')
    pending_dependency = await stack.seed('pending dependency')
    done_dependency = await stack.seed('done dependency', status='done')
    in_progress_dependent = await stack.seed('in-progress dependent', status='in-progress')
    cancelled_dependent = await stack.seed('cancelled dependent', status='cancelled')
    await stack.backend.add_dependency(anchor_id, pending_dependency, stack.root)
    await stack.backend.add_dependency(anchor_id, done_dependency, stack.root)
    await stack.backend.add_dependency(in_progress_dependent, anchor_id, stack.root)
    await stack.backend.add_dependency(cancelled_dependent, anchor_id, stack.root)

    prepared = await stack.prepare(_review_of(anchor_id))

    assert _ids_from(prepared.pool, 'dependency') == {pending_dependency, in_progress_dependent}
    assert _recorded(stack).get_tasks_statuses == []


@pytest.mark.asyncio
async def test_the_dependency_cap_bounds_the_full_row_reads(recording_stack):
    stack = recording_stack
    stack.config.curator.pool_dependency_cap = 1
    anchor_id = await stack.seed('anchor')
    dependents = {await stack.seed(f'dependent {n}') for n in range(3)}
    for dependent in dependents:
        await stack.backend.add_dependency(dependent, anchor_id, stack.root)

    prepared = await stack.prepare(_review_of(anchor_id))

    assert len(_ids_from(prepared.pool, 'dependency')) == 1
    assert prepared.withheld is not None
    assert prepared.withheld.by_source['dependency'] == 2
    assert len([tid for tid in _recorded(stack).get_task_ids if tid in dependents]) <= 1


_CURATOR_LOGGER = 'fused_memory.middleware.task_curator'
_EMBEDDING_UNAVAILABLE = 'stream embedding unavailable: absence is not proof of no overlap'


def _creating_and_recording(prompts: list[str]) -> Any:
    """A single-spec invoke arm that records each prompt and answers 'create'."""
    def answer(kwargs: dict[str, Any]) -> AgentResult:
        prompts.append(kwargs['prompt'])
        return agent_result({'action': 'create', 'justification': 'no overlap in the pool'})

    return answer


def _unavailable(prepared: PreparedCandidate) -> tuple[str, ...]:
    assert prepared.withheld is not None
    return prepared.withheld.unavailable_streams


def _anchored(anchor_id: str) -> CandidateTask:
    """A candidate with files under *anchor_id*: anchor, module, embedding and dependency apply."""
    return CandidateTask(
        title='Fix the slate runner', files_to_modify=[_CANDIDATE_FILE], spawned_from=anchor_id,
    )


@pytest.mark.asyncio
async def test_a_failed_embedding_stream_is_named_in_the_prompt_and_the_decision(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=_CURATOR_LOGGER)
    prompts: list[str] = []
    invoke = invoke_by_role(
        single=_creating_and_recording(prompts),
        batch=tripwire('one candidate must take the single spec'),
    )
    async with _open_stack(tmp_path / 'project', SqliteTaskBackend, invoke=invoke) as stack:
        await stack.seed_module_task()

        prepared = await stack.prepare(CandidateTask(title='t', files_to_modify=[_CANDIDATE_FILE]))
        [decision] = await stack.curator.curate_prepared([prepared], 'p', stack.root)

    assert prepared.corpus_error is None
    assert _unavailable(prepared) == ('embedding',)
    [prompt] = prompts
    assert _EMBEDDING_UNAVAILABLE in prompt.splitlines()
    assert decision.action == 'create'
    assert decision.to_log_fields()['unavailable_streams'] == ['embedding']
    curator_records = [r for r in caplog.records if r.name == _CURATOR_LOGGER]
    assert any(
        r.levelno == logging.WARNING and 'embedding' in r.getMessage()
        and 'unavailable' in r.getMessage()
        for r in curator_records
    )
    assert any(
        r.levelno == logging.INFO and 'decision' in r.getMessage()
        and 'unavailable_streams' in r.getMessage() and "'embedding'" in r.getMessage()
        for r in curator_records
    )


@pytest.mark.asyncio
async def test_a_failed_module_read_is_named_when_other_streams_answered(tmp_path):
    async with _open_stack(tmp_path / 'project', _ModuleReadFailsBackend) as stack:
        anchor_id = await stack.seed('anchor')

        prepared = await stack.prepare(_anchored(anchor_id))

    assert _ids_from(prepared.pool, 'anchor') == {anchor_id}
    assert _unavailable(prepared) == ('module', 'embedding')


@pytest.mark.asyncio
async def test_a_failed_anchor_fetch_is_named(tmp_path):
    async with _open_stack(tmp_path / 'project', _FetchFailsBackend) as stack:
        anchor_id = await stack.seed('anchor')
        module_id = await stack.seed_module_task()
        assert isinstance(stack.backend, _FetchFailsBackend)
        stack.backend.failing_ids.add(anchor_id)

        prepared = await stack.prepare(_anchored(anchor_id))

    assert 'anchor' in _unavailable(prepared)
    assert _ids_from(prepared.pool, 'module') == {module_id}


@pytest.mark.asyncio
async def test_a_spawned_from_naming_no_task_is_absent_not_unavailable(stack):
    module_id = await stack.seed_module_task()

    prepared = await stack.prepare(_anchored('987654'))

    assert _ids_from(prepared.pool, 'anchor') == set()
    assert 'anchor' not in _unavailable(prepared)
    assert _ids_from(prepared.pool, 'module') == {module_id}


@pytest.mark.asyncio
async def test_a_failed_dependency_read_is_named(tmp_path):
    async with _open_stack(tmp_path / 'project', _DependencyReadFailsBackend) as stack:
        anchor_id = await stack.seed('anchor')

        prepared = await stack.prepare(_anchored(anchor_id))

    assert _ids_from(prepared.pool, 'anchor') == {anchor_id}
    assert 'dependency' in _unavailable(prepared)


_RESOLVE_DEADLINE_SECONDS = 30.0


@asynccontextmanager
async def _interceptor_over(
    stack: CorpusStack, state_dir: Path,
) -> AsyncIterator[tuple[TaskInterceptor, TicketStore]]:
    """A real interceptor, and its ticket store, whose worker curates through *stack*'s curator."""
    store = TicketStore(state_dir / 'tickets.db')
    await store.initialize()
    buffer = EventBuffer(db_path=state_dir / 'events.db', buffer_size_threshold=100)
    await buffer.initialize()
    interceptor = TaskInterceptor(
        stack.backend, None, buffer, ticket_store=store, curator=stack.curator,
    )
    try:
        yield interceptor, store
    finally:
        await interceptor.close()
        await buffer.close()
        await store.close()


@pytest.mark.asyncio
async def test_a_candidate_whose_every_stream_failed_is_corpus_failed(tmp_path):
    async with _open_stack(tmp_path / 'project', _ModuleReadFailsBackend) as stack:
        prepared = await stack.curator.prepare_candidate(
            CandidateTask(title='t', files_to_modify=[_CANDIDATE_FILE]), 'p', stack.root,
        )
        [decision] = await stack.curator.curate_prepared([prepared], 'p', stack.root)

    assert prepared.corpus_error is not None
    assert 'module' in prepared.corpus_error
    assert 'embedding' in prepared.corpus_error
    assert _unavailable(prepared) == ('module', 'embedding')
    assert decision.action == 'defer'
    assert decision.defer_reason == DeferReason.CORPUS_FAILED
    assert decision.to_log_fields()['unavailable_streams'] == ['module', 'embedding']


@pytest.mark.asyncio
async def test_a_candidate_with_only_the_embedding_stream_and_it_failed_is_corpus_failed(stack):
    prepared = await stack.curator.prepare_candidate(CandidateTask(title='t'), 'p', stack.root)

    assert prepared.corpus_error is not None
    assert _unavailable(prepared) == ('embedding',)


@pytest.mark.asyncio
async def test_the_ticket_of_an_all_streams_failed_candidate_parks_awaiting_curation(
    tmp_path,
):
    async with (
        _open_stack(tmp_path / 'project', _ModuleReadFailsBackend) as stack,
        _interceptor_over(stack, tmp_path) as (interceptor, store),
    ):
        submitted = await interceptor.submit_task(
            stack.root, title='Fix the slate runner', metadata={'files': [_CANDIDATE_FILE]},
        )
        ticket_id = submitted['ticket']

        answer = await asyncio.wait_for(
            interceptor.resolve_ticket(
                ticket_id, stack.root, timeout_seconds=_RESOLVE_DEADLINE_SECONDS,
            ),
            timeout=_RESOLVE_DEADLINE_SECONDS + 5,
        )
        row = await store.get(ticket_id)
        stored = await stack.backend.get_statuses_raw(stack.root)

    assert answer == {
        'status': TicketStatus.AWAITING_CURATION,
        'ticket': ticket_id,
        'reason': f'curation-deferred: {DeferReason.CORPUS_FAILED}',
    }
    assert row is not None
    assert row['defer_reason'] == DeferReason.CORPUS_FAILED.value
    assert stored == {}


@pytest.mark.asyncio
async def test_an_integer_spawned_from_submitted_through_submit_task_resolves_created(tmp_path):
    prompts: list[str] = []
    invoke = invoke_by_role(
        single=_creating_and_recording(prompts),
        batch=tripwire('one candidate must take the single spec'),
    )
    async with (
        _open_stack(tmp_path / 'project', SqliteTaskBackend, invoke=invoke) as stack,
        _interceptor_over(stack, tmp_path) as (interceptor, _store),
    ):
        parent_id = await stack.seed('parent')
        submitted = await interceptor.submit_task(
            stack.root, title='Fix the slate runner', metadata={
                'files': [_CANDIDATE_FILE],
                'spawned_from': int(parent_id),
                'spawn_context': _FOLLOWUP,
            },
        )

        answer = await asyncio.wait_for(
            interceptor.resolve_ticket(
                submitted['ticket'], stack.root, timeout_seconds=_RESOLVE_DEADLINE_SECONDS,
            ),
            timeout=_RESOLVE_DEADLINE_SECONDS + 5,
        )
        created = await stack.backend.get_task(str(answer['task_id']), stack.root)

    assert answer['status'] == TicketStatus.CREATED
    assert created['metadata']['spawned_from'] == int(parent_id)
    [prompt] = prompts
    assert any(
        line.startswith(f'[Task {parent_id}]') and 'source=anchor' in line
        for line in prompt.splitlines()
    )
