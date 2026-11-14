"""Shared fakes for the auto-consolidation executor tests (task 5240).

Not collected (no ``test_`` prefix); imported as
``from _auto_consolidation_fakes import ...``, the
``tests/_tool_surface_classification.py`` precedent, so task eta's integration
gate can reuse it rather than copy it.

:class:`FakeMemoryService` is a STATEFUL in-memory Mem0: writes land and later
reads see them, so "the incumbent was not touched" and "no orphan remains" are
questions about the fake's world rather than restatements of a canned row.
Every method is an ``AsyncMock`` with a side effect, so awaits stay
inspectable. The mem0_update and consolidation_auto leaves are REAL config
models: the authorization resolver rejects a Mock leaf, and a case would pass
for the wrong reason.
"""

from __future__ import annotations

import copy
import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from fused_memory.backends.task_backend_errors import DuplicateCandidateKeyError
from fused_memory.config.schema import ConsolidationAutoConfig, Mem0UpdateConfig
from fused_memory.models.memory import AddMemoryResponse
from fused_memory.reconciliation.consolidation_proposal import (
    ConsolidationProposal,
    record_proposal,
)

PROJECT_ID = 'dark_factory'
CATEGORY = 'procedural_knowledge'
NOW = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
PROPOSED_AT = NOW - timedelta(hours=1)
VALID_CLAIM = 'Stage one keeps reproposing the identical unaddressed duplicate cluster'
CREATED_AT = '2026-06-01T00:00:00+00:00'


def new_id() -> str:
    return str(uuid.uuid4())


def enabled_config(**overrides: Any) -> ConsolidationAutoConfig:
    fields: dict[str, Any] = {'enabled': True, 'enabled_projects': [PROJECT_ID]}
    fields.update(overrides)
    return ConsolidationAutoConfig(**fields)


class FakeMemoryService:
    """A stateful Mem0 over ``{id: {'id', 'content', 'metadata'}}``.

    Injection points: ``read_errors`` (id -> exception ``get_memory_by_id``
    raises), ``update_returns`` (id -> the dict ``update_memory`` returns
    instead of landing the patch), ``delete_errors`` (id -> exception
    ``delete_memory`` raises), ``add_error`` (raised by every ``add_memory``),
    ``count_error`` and ``scroll_error``.
    """

    def __init__(
        self,
        ledger: Any,
        *,
        consolidation_auto: ConsolidationAutoConfig | None = None,
        mem0_update: Mem0UpdateConfig | None = None,
    ) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self.system_records: list[dict[str, Any]] = []
        self.read_errors: dict[str, BaseException] = {}
        self.update_returns: dict[str, dict[str, Any]] = {}
        self.delete_errors: dict[str, BaseException] = {}
        self.add_error: BaseException | None = None
        self.count_error: BaseException | None = None
        self.scroll_error: BaseException | None = None
        self.config = SimpleNamespace(
            mem0_update=mem0_update or Mem0UpdateConfig(),
            consolidation_auto=consolidation_auto or enabled_config(),
        )
        self.recon_ledger = ledger
        self.get_memory_by_id = AsyncMock(side_effect=self._get)
        self.update_memory = AsyncMock(side_effect=self._update)
        self.add_memory = AsyncMock(side_effect=self._add)
        self.delete_memory = AsyncMock(side_effect=self._delete)
        self.count_memories_by_metadata = AsyncMock(side_effect=self._count)
        self.get_memories_by_metadata = AsyncMock(side_effect=self._scroll)
        self.add_system_record = AsyncMock(side_effect=self._add_system_record)
        self.search = AsyncMock(return_value=[])

    # ---- seeding and inspection ------------------------------------------

    def seed(self, *, content: str | None = None, **metadata: Any) -> str:
        memory_id = new_id()
        self.records[memory_id] = {
            'id': memory_id,
            'content': content or f'record {memory_id}',
            'metadata': {'category': CATEGORY, **metadata},
        }
        return memory_id

    def seed_many(self, n: int, **metadata: Any) -> list[str]:
        return [self.seed(**metadata) for _ in range(n)]

    def metadata_of(self, memory_id: str) -> dict[str, Any]:
        return self.records[memory_id]['metadata']

    def snapshot(self, memory_id: str) -> dict[str, Any]:
        return copy.deepcopy(self.records[memory_id])

    def canonical_ids(self) -> list[str]:
        return [
            r['id'] for r in self.records.values() if r['metadata'].get('canonical') is True
        ]

    def write_awaits(self) -> tuple[int, int, int, int]:
        return (
            self.update_memory.await_count,
            self.add_memory.await_count,
            self.delete_memory.await_count,
            self.add_system_record.await_count,
        )

    # ---- the MemoryService surface ---------------------------------------

    async def _get(self, project_id: str, memory_id: str) -> dict[str, Any] | None:
        if memory_id in self.read_errors:
            raise self.read_errors[memory_id]
        record = self.records.get(memory_id)
        return copy.deepcopy(record) if record is not None else None

    async def _update(
        self,
        memory_id: str,
        project_id: str = 'main',
        content: str | None = None,
        metadata_patch: dict[str, Any] | None = None,
        metadata_delete_keys: list[str] | None = None,
        metadata_mode: str = 'merge',
        **_: Any,
    ) -> dict[str, Any]:
        if memory_id in self.update_returns:
            return self.update_returns[memory_id]
        record = self.records.get(memory_id)
        if record is None:
            return {'error': f'{memory_id} not found', 'error_type': 'MemoryNotFound'}
        record['metadata'].update(metadata_patch or {})
        for key in metadata_delete_keys or []:
            record['metadata'].pop(key, None)
        if content is not None:
            record['content'] = content
        return {'status': 'updated', 'store': 'mem0', 'id': memory_id}

    async def _add(
        self,
        content: str,
        category: str | None = None,
        project_id: str = 'main',
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        **_: Any,
    ) -> AddMemoryResponse:
        if self.add_error is not None:
            raise self.add_error
        memory_id = new_id()
        self.records[memory_id] = {
            'id': memory_id,
            'content': content,
            'metadata': {**(metadata or {}), 'category': category},
        }
        return AddMemoryResponse(memory_ids=[memory_id], message='ok')

    async def _delete(
        self, memory_id: str, store: str, project_id: str = 'main', **_: Any,
    ) -> dict[str, Any]:
        if memory_id in self.delete_errors:
            raise self.delete_errors[memory_id]
        self.records.pop(memory_id, None)
        return {'status': 'deleted', 'id': memory_id, 'store': store}

    def _matching(self, filters: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                'id': r['id'],
                'content': r['content'],
                'created_at': CREATED_AT,
                'metadata': copy.deepcopy(r['metadata']),
            }
            for r in self.records.values()
            if all(r['metadata'].get(k) == v for k, v in filters.items())
        ]

    async def _count(self, project_id: str, filters: dict[str, Any]) -> int:
        if self.count_error is not None:
            raise self.count_error
        return len(self._matching(filters))

    async def _scroll(
        self, project_id: str, filters: dict[str, Any], limit: int = 1000,
    ) -> list[dict[str, Any]]:
        if self.scroll_error is not None:
            raise self.scroll_error
        return self._matching(filters)[:limit]

    async def _add_system_record(
        self,
        content: str,
        *,
        project_id: str,
        agent_id: str | None,
        category: str,
        metadata: dict[str, Any] | None = None,
        **_: Any,
    ) -> AddMemoryResponse:
        memory_id = new_id()
        record = {
            'id': memory_id,
            'content': content,
            'metadata': {**(metadata or {}), 'category': category},
        }
        self.records[memory_id] = record
        self.system_records.append(record)
        return AddMemoryResponse(memory_ids=[memory_id], message='ok')


class FakeTaskBackend:
    """Records ``add_task`` kwargs; returns ``{'id': N}`` or raises as configured."""

    def __init__(
        self,
        *,
        duplicate_of: int | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self._duplicate_of = duplicate_of
        self._error = error
        self._next_id = 1000
        self.add_task = AsyncMock(side_effect=self._add_task)

    async def _add_task(self, **kwargs: Any) -> dict[str, str]:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        if self._duplicate_of is not None:
            raise DuplicateCandidateKeyError(existing_id=self._duplicate_of)
        self._next_id += 1
        return {'id': str(self._next_id), 'message': 'added'}


async def propose(
    ledger: Any,
    *,
    topic: str,
    member_ids: list[str],
    run_id: str = 'run-p1',
    category: str = CATEGORY,
    claim: str = VALID_CLAIM,
    project_id: str = PROJECT_ID,
    now: datetime = PROPOSED_AT,
    ttl_hours: int = 168,
) -> ConsolidationProposal:
    """Record a proposal row the way ``propose_consolidation`` would."""
    proposal = ConsolidationProposal(
        project_id=project_id,
        run_id=run_id,
        topic=topic,
        member_ids=tuple(member_ids),
        category=category,
        claim=claim,
        rationale='These records state the same rule in different words.',
    )
    assert await record_proposal(ledger, proposal, now=now, ttl_hours=ttl_hours)
    return proposal
