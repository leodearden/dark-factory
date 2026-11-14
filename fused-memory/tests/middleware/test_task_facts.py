"""``read_task_facts`` (plans/live-task-facts-at-recall-prd.md C1) over a real interceptor and store."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from _out_of_band_task_writes import write_task_out_of_band

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import TaskmasterConfig
from fused_memory.middleware.task_facts import (
    Lookup,
    TaskFacts,
    TaskFactsReader,
    TaskRef,
    read_task_facts,
)
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.reconciliation.event_buffer import EventBuffer

ALPHA = 'alpha_proj'
BETA = 'beta_proj'


@dataclass
class Store:
    backend: SqliteTaskBackend
    interceptor: TaskInterceptor
    known_projects: dict[str, str]

    async def add(self, project: str, title: str) -> TaskRef:
        added = await self.backend.add_task(project_root=self.known_projects[project], title=title)
        return TaskRef(project, added['id'])

    async def depend(self, dependent: TaskRef, on: TaskRef) -> None:
        await self.backend.add_dependency(
            dependent.task_id, on.task_id, self.known_projects[dependent.project_id],
        )

    def write(self, ref: TaskRef, *, status: str | None = None, metadata: str | None = None) -> None:
        write_task_out_of_band(
            self.known_projects[ref.project_id], ref.task_id, status=status, metadata=metadata,
        )

    async def read(self, refs: list[TaskRef], reader: TaskFactsReader | None = None) -> dict[TaskRef, TaskFacts]:
        return await read_task_facts(
            refs, reader=self.interceptor if reader is None else reader,
            known_projects=self.known_projects,
        )


@pytest_asyncio.fixture
async def store(tmp_path: Path) -> AsyncIterator[Store]:
    backend = SqliteTaskBackend(TaskmasterConfig(project_root=str(tmp_path)))
    await backend.start()
    interceptor = TaskInterceptor(backend, None, EventBuffer(db_path=None))
    known = {ALPHA: str(tmp_path / 'alpha'), BETA: str(tmp_path / 'beta')}
    try:
        yield Store(backend, interceptor, known)
    finally:
        await interceptor.close()
        await backend.close()


_DELEGATE = object()


@dataclass
class RecordingReader:
    """A ``TaskFactsReader`` that records every call and replies per project root.

    A root in ``scripted`` gets that reply, raised when it is an exception;
    any other root is forwarded to ``delegate``.
    """

    delegate: TaskFactsReader | None = None
    scripted: dict[str, object] = field(default_factory=dict)
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)

    async def get_task_facts(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        reply = self.scripted.get(kwargs.get('project_root', ''), _DELEGATE)
        if isinstance(reply, Exception):
            raise reply
        if reply is _DELEGATE:
            assert self.delegate is not None
            return await self.delegate.get_task_facts(*args, **kwargs)
        return reply


def _found(ref: TaskRef, status: str, title: str, *,
           waits_on: tuple[tuple[TaskRef, str], ...] = (),
           finished_how: str | None = None, finished_commit: str | None = None) -> TaskFacts:
    return TaskFacts(
        ref=ref, lookup=Lookup.FOUND, reason=None, status=status, title=title,
        waits_on=waits_on, finished_how=finished_how, finished_commit=finished_commit,
    )


def _provenance(kind: str, commit: str) -> str:
    return json.dumps({'done_provenance': {'kind': kind, 'commit': commit}})


def _row(status: object, title: str = 'Scripted') -> dict[str, object]:
    return {
        'status': status, 'title': title, 'dependencies': [],
        'done_provenance_kind': None, 'done_provenance_commit': None,
    }


# ── 1. FOUND ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_found_carries_status_title_and_done_provenance(store: Store):
    pending = await store.add(ALPHA, 'Pending one')
    finished = await store.add(ALPHA, 'Finished with provenance')
    store.write(finished, status='done', metadata=_provenance('merged', 'abc123'))
    bare_done = await store.add(ALPHA, 'Finished without provenance')
    store.write(bare_done, status='done')
    cancelled = await store.add(ALPHA, 'Cancelled with provenance')
    store.write(cancelled, status='cancelled', metadata=_provenance('merged', 'def456'))

    facts = await store.read([pending, finished, bare_done, cancelled])

    assert facts[pending] == _found(pending, 'pending', 'Pending one')
    assert facts[finished] == _found(
        finished, 'done', 'Finished with provenance',
        finished_how='merged', finished_commit='abc123',
    )
    assert facts[bare_done] == _found(bare_done, 'done', 'Finished without provenance')
    assert facts[cancelled] == _found(cancelled, 'cancelled', 'Cancelled with provenance')


# ── 2. waits_on ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_waits_on_lists_only_unfinished_dependencies_in_id_order(store: Store):
    task = await store.add(ALPHA, 'Waiting task')
    unfinished_low = await store.add(ALPHA, 'Unfinished low')
    finished = await store.add(ALPHA, 'Finished dependency')
    unfinished_high = await store.add(ALPHA, 'Unfinished high')
    store.write(finished, status='done')
    store.write(unfinished_high, status='in-progress')
    await store.depend(task, unfinished_high)
    await store.depend(task, finished)
    await store.depend(task, unfinished_low)

    facts = await store.read([task])

    assert facts[task].waits_on == (
        (unfinished_low, 'pending'),
        (unfinished_high, 'in-progress'),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('terminal', ['done', 'cancelled'])
async def test_a_terminal_task_waits_on_nothing(store: Store, terminal: str):
    task = await store.add(ALPHA, 'Terminal task')
    dependency = await store.add(ALPHA, 'Still pending')
    await store.depend(task, dependency)
    store.write(task, status=terminal)

    facts = await store.read([task])

    assert facts[task].waits_on == ()
    assert facts[task].status == terminal


@pytest.mark.asyncio
async def test_a_dangling_dependency_waits_as_no_such_task(store: Store):
    task = await store.add(ALPHA, 'Waiting task')
    gone = await store.add(ALPHA, 'Removed dependency')
    await store.depend(task, gone)
    await store.backend.remove_tasks([gone.task_id], store.known_projects[ALPHA])

    facts = await store.read([task])

    assert facts[task].waits_on == ((gone, 'no_such_task'),)


# ── 3. NO_SUCH_TASK ────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize('task_id', ['999', 'abc'])
async def test_an_id_the_store_lacks_is_no_such_task(store: Store, task_id: str):
    await store.add(ALPHA, 'Unrelated')
    ref = TaskRef(ALPHA, task_id)

    facts = await store.read([ref])

    assert facts[ref] == TaskFacts(
        ref=ref, lookup=Lookup.NO_SUCH_TASK, reason=None, status=None, title=None,
        waits_on=(), finished_how=None, finished_commit=None,
    )


# ── 4/5. COULD_NOT_CHECK without a read ────────────────────────────────


@pytest.mark.asyncio
async def test_an_unregistered_project_could_not_check_and_is_never_read(store: Store):
    ref = TaskRef('unregistered_proj', '1')
    reader = RecordingReader(delegate=store.interceptor)

    facts = await store.read([ref], reader=reader)

    assert facts[ref].lookup is Lookup.COULD_NOT_CHECK
    assert facts[ref].reason == 'project not registered'
    assert reader.calls == []


@pytest.mark.asyncio
async def test_no_reader_could_not_check_every_ref(store: Store):
    refs = [TaskRef(ALPHA, '1'), TaskRef(BETA, '2'), TaskRef('unregistered_proj', '3')]

    facts = await read_task_facts(refs, reader=None, known_projects=store.known_projects)

    assert {ref: f.lookup for ref, f in facts.items()} == dict.fromkeys(refs, Lookup.COULD_NOT_CHECK)
    assert facts[refs[0]].reason == 'no task store configured'
    assert facts[refs[1]].reason == 'no task store configured'


# ── 6. Totality and batching ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_every_ref_has_one_entry_and_each_project_is_read_once_by_keyword(store: Store):
    alpha = [await store.add(ALPHA, f'Alpha {n}') for n in range(3)]
    beta = await store.add(BETA, 'Beta 0')
    refs = [alpha[2], alpha[0], alpha[2], TaskRef(ALPHA, '404'), alpha[1], beta, beta]
    reader = RecordingReader(delegate=store.interceptor)

    facts = await store.read(refs, reader=reader)

    assert set(facts) == set(refs)
    assert len(facts) == len(set(refs))
    assert all(facts[ref].ref == ref for ref in refs)
    assert sorted(reader.calls, key=lambda call: call[1]['project_root']) == [
        ((), {'ids': sorted({r.task_id for r in alpha} | {'404'}),
              'project_root': store.known_projects[ALPHA]}),
        ((), {'ids': [beta.task_id], 'project_root': store.known_projects[BETA]}),
    ]


@pytest.mark.asyncio
async def test_an_empty_ref_collection_reads_nothing(store: Store):
    reader = RecordingReader(delegate=store.interceptor)

    assert await store.read([], reader=reader) == {}
    assert reader.calls == []


# ── 7. Containment ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_raising_project_could_not_check_and_spares_the_others(store: Store):
    alpha = await store.add(ALPHA, 'Alpha task')
    beta = await store.add(BETA, 'Beta task')
    reader = RecordingReader(
        delegate=store.interceptor,
        scripted={store.known_projects[ALPHA]: OSError('disk unplugged')},
    )

    facts = await store.read([alpha, beta], reader=reader)

    assert facts[alpha].lookup is Lookup.COULD_NOT_CHECK
    assert facts[alpha].reason is not None
    assert 'OSError' in facts[alpha].reason
    assert 'disk unplugged' in facts[alpha].reason
    assert facts[beta] == _found(beta, 'pending', 'Beta task')


# ── 8. Malformed replies ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_non_mapping_reply_could_not_check_that_project(store: Store):
    alpha = TaskRef(ALPHA, '1')
    beta = await store.add(BETA, 'Beta task')
    reader = RecordingReader(
        delegate=store.interceptor, scripted={store.known_projects[ALPHA]: ['1']},
    )

    facts = await store.read([alpha, beta], reader=reader)

    assert facts[alpha].lookup is Lookup.COULD_NOT_CHECK
    assert facts[beta].lookup is Lookup.FOUND


@pytest.mark.asyncio
async def test_a_row_without_a_str_status_could_not_check_that_ref_only(store: Store):
    broken, sound = TaskRef(ALPHA, '7'), TaskRef(ALPHA, '8')
    reader = RecordingReader(scripted={
        store.known_projects[ALPHA]: {'7': _row(None), '8': _row('pending', 'Sound')},
    })

    facts = await store.read([broken, sound], reader=reader)

    assert facts[broken].lookup is Lookup.COULD_NOT_CHECK
    assert facts[broken].reason == 'task store returned no usable status'
    assert facts[sound] == _found(sound, 'pending', 'Sound')


# ── 9. An unconfigured interceptor ─────────────────────────────────────


@pytest.mark.asyncio
async def test_an_unconfigured_interceptor_could_not_check(store: Store):
    ref = TaskRef(ALPHA, '1')
    bare = TaskInterceptor(None, None, EventBuffer(db_path=None))

    facts = await store.read([ref], reader=bare)

    assert facts[ref].lookup is Lookup.COULD_NOT_CHECK
    assert facts[ref].reason is not None
    assert 'RuntimeError' in facts[ref].reason


# ── 10. Invariants ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ('project_id', 'task_id', 'field_name'),
    [('', '1', 'project_id'), ('p', '', 'task_id')],
)
def test_a_task_ref_refuses_an_empty_field(project_id: str, task_id: str, field_name: str):
    with pytest.raises(ValueError, match=field_name):
        TaskRef(project_id, task_id)


@pytest.mark.parametrize(
    'fields',
    [
        pytest.param({'lookup': Lookup.COULD_NOT_CHECK, 'reason': None}, id='could-not-check-without-reason'),
        pytest.param({'lookup': Lookup.FOUND, 'status': None}, id='found-without-status'),
        pytest.param({'lookup': Lookup.FOUND, 'status': 'done', 'reason': 'x'}, id='found-with-reason'),
        pytest.param({'lookup': Lookup.NO_SUCH_TASK, 'title': 'T'}, id='no-such-task-with-title'),
        pytest.param({'lookup': Lookup.FOUND, 'status': 'pending', 'finished_how': 'merged'},
                     id='finished-how-on-a-pending-task'),
        pytest.param({'lookup': Lookup.FOUND, 'status': 'done',
                      'waits_on': ((TaskRef('p', '2'), 'pending'),)}, id='terminal-task-waiting'),
    ],
)
def test_task_facts_refuses_a_broken_lookup_invariant(fields: Mapping[str, Any]):
    base: dict[str, Any] = {
        'ref': TaskRef('p', '1'), 'lookup': Lookup.NO_SUCH_TASK, 'reason': None, 'status': None,
        'title': None, 'waits_on': (), 'finished_how': None, 'finished_commit': None,
    }
    with pytest.raises(ValueError):
        TaskFacts(**(base | dict(fields)))
