"""R4, the escalation-level idempotency check, driven through submit_task/resolve_ticket over a real SQLite store.

A ticket whose metadata carries an ``(escalation_id, suggestion_hash)`` pair
that an existing non-cancelled task already carries resolves ``combined``
onto that task without curation (PRD B11). The check looks the hash up in
the store's metadata index and confirms the escalation id on each match, so
its cost is bounded by the matching rows, never by the task tree.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from _curator_helpers import curator_returning

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import TaskmasterConfig
from fused_memory.middleware.curator_types import CuratorDecision
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.middleware.ticket_store import TicketStatus, TicketStore
from fused_memory.reconciliation.event_buffer import EventBuffer

_INTERCEPTOR_LOGGER = 'fused_memory.middleware.task_interceptor'
_ESCALATION_ID = 'esc-6708-1'
_SUGGESTION_HASH = 'a1b2c3d4e5f60718'
_LOOKUP_PAGE = 100
_RESOLVE_DEADLINE_SECONDS = 30.0


class _RecordingBackend(SqliteTaskBackend):
    """Records every get_tasks call and the arguments of every metadata lookup."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.get_tasks_calls = 0
        self.lookups: list[dict[str, Any]] = []

    async def get_tasks(self, *args: Any, **kwargs: Any) -> Any:
        self.get_tasks_calls += 1
        return await super().get_tasks(*args, **kwargs)

    async def find_tasks_by_metadata(
        self, project_root: str, key: str, value: str | int | None = None, **kwargs: Any,
    ) -> Any:
        self.lookups.append({'key': key, 'value': value, **kwargs})
        return await super().find_tasks_by_metadata(project_root, key, value, **kwargs)


class _LookupFailsBackend(_RecordingBackend):
    async def find_tasks_by_metadata(
        self, project_root: str, key: str, value: str | int | None = None, **kwargs: Any,
    ) -> Any:
        self.lookups.append({'key': key, 'value': value, **kwargs})
        raise RuntimeError('metadata lookup unavailable')


@dataclass(frozen=True)
class R4Stack:
    root: str
    backend: _RecordingBackend
    interceptor: TaskInterceptor
    curator: MagicMock

    async def seed(
        self,
        title: str,
        *,
        escalation_id: str = _ESCALATION_ID,
        suggestion_hash: str = _SUGGESTION_HASH,
        status: str = 'pending',
    ) -> str:
        added = await self.backend.add_task(
            project_root=self.root,
            title=title,
            description=f'{title} (R4 fixture)',
            metadata=json.dumps({
                'escalation_id': escalation_id, 'suggestion_hash': suggestion_hash,
            }),
            status=status,
        )
        return str(added['id'])

    async def submit(self, metadata: dict[str, Any] | str) -> dict[str, Any]:
        submitted = await self.interceptor.submit_task(
            self.root, title='Fix the escalated defect', description='d', metadata=metadata,
        )
        return await asyncio.wait_for(
            self.interceptor.resolve_ticket(
                submitted['ticket'], self.root, timeout_seconds=_RESOLVE_DEADLINE_SECONDS,
            ),
            timeout=_RESOLVE_DEADLINE_SECONDS + 5,
        )

    def hash_lookups(self) -> list[dict[str, Any]]:
        return [lookup for lookup in self.backend.lookups if lookup['key'] == 'suggestion_hash']

    async def task_ids(self) -> set[str]:
        return set(await self.backend.get_statuses_raw(self.root))


@asynccontextmanager
async def _open_stack(
    tmp_path: Path, backend_cls: type[_RecordingBackend] = _RecordingBackend,
) -> AsyncIterator[R4Stack]:
    project_dir = tmp_path / 'project'
    project_dir.mkdir()
    root = str(project_dir.resolve())
    backend = backend_cls(TaskmasterConfig(project_root=root))
    await backend.start()
    buffer = EventBuffer(db_path=tmp_path / 'events.db', buffer_size_threshold=100)
    await buffer.initialize()
    store = TicketStore(tmp_path / 'tickets.db')
    await store.initialize()
    curator = curator_returning(
        CuratorDecision(action='create', degraded=False, justification='novel'),
    )
    interceptor = TaskInterceptor(backend, None, buffer, ticket_store=store, curator=curator)
    try:
        yield R4Stack(root, backend, interceptor, curator)
    finally:
        await interceptor.close()
        await store.close()
        await buffer.close()
        await backend.close()


def _pair(escalation_id: str = _ESCALATION_ID) -> dict[str, str]:
    return {'escalation_id': escalation_id, 'suggestion_hash': _SUGGESTION_HASH}


def _combined_onto(task_id: str) -> dict[str, Any]:
    return {'status': TicketStatus.COMBINED, 'task_id': task_id, 'reason': 'idempotency_hit'}


@pytest.mark.asyncio
async def test_a_resubmitted_pair_resolves_combined_without_reading_the_task_tree(tmp_path):
    async with _open_stack(tmp_path) as stack:
        existing = await stack.seed('existing')

        answer = await stack.submit(_pair())

        assert answer == _combined_onto(existing)
        lookups = stack.hash_lookups()
        assert lookups
        assert all(lookup['value'] == _SUGGESTION_HASH for lookup in lookups)
        assert stack.backend.get_tasks_calls == 0
        assert stack.curator.curate_prepared.await_count == 0
        assert await stack.task_ids() == {existing}


@pytest.mark.asyncio
async def test_a_done_task_with_the_pair_does_not_hit(tmp_path):
    async with _open_stack(tmp_path) as stack:
        await stack.seed('existing', status='done')

        answer = await stack.submit(_pair())

        assert answer['status'] == TicketStatus.CREATED


@pytest.mark.asyncio
async def test_a_cancelled_task_with_the_pair_does_not_hit(tmp_path):
    async with _open_stack(tmp_path) as stack:
        await stack.seed('existing', status='cancelled')

        answer = await stack.submit(_pair())

        assert answer['status'] == TicketStatus.CREATED


@pytest.mark.asyncio
async def test_the_same_hash_under_another_escalation_does_not_hit(tmp_path):
    async with _open_stack(tmp_path) as stack:
        await stack.seed('existing', escalation_id='esc-6708-other')

        answer = await stack.submit(_pair())

        assert answer['status'] == TicketStatus.CREATED


@pytest.mark.asyncio
async def test_metadata_submitted_as_a_json_string_hits_like_a_dict(tmp_path):
    async with _open_stack(tmp_path) as stack:
        existing = await stack.seed('existing')

        assert await stack.submit(json.dumps(_pair())) == _combined_onto(existing)


@pytest.mark.asyncio
async def test_metadata_without_a_suggestion_hash_never_queries_the_lookup(tmp_path):
    async with _open_stack(tmp_path) as stack:
        await stack.seed('existing')

        answer = await stack.submit({'escalation_id': _ESCALATION_ID})

        assert answer['status'] == TicketStatus.CREATED
        assert stack.hash_lookups() == []


@pytest.mark.asyncio
async def test_a_failed_lookup_falls_through_to_the_curator(tmp_path, caplog):
    caplog.set_level(logging.WARNING, logger=_INTERCEPTOR_LOGGER)
    async with _open_stack(tmp_path, _LookupFailsBackend) as stack:
        await stack.seed('existing')

        answer = await stack.submit(_pair())

        assert answer['status'] == TicketStatus.CREATED
        assert stack.curator.curate_prepared.await_count == 1
    assert any(
        record.name == _INTERCEPTOR_LOGGER and record.levelno == logging.WARNING
        and record.getMessage().startswith('r4:')
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_a_confirming_match_past_the_first_lookup_page_still_hits(tmp_path):
    async with _open_stack(tmp_path) as stack:
        for n in range(_LOOKUP_PAGE):
            await stack.seed(f'other escalation {n}', escalation_id=f'esc-6708-other-{n}')
        existing = await stack.seed('existing')

        assert await stack.submit(_pair()) == _combined_onto(existing)
