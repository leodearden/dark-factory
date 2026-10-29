"""The one place fused-memory tests construct a ``ReconciliationHarness`` (task 6075).

Import as ``from reconciliation.harness_fixtures import make_harness, ...``; the
package-import mechanics are those described in ``consolidator_fixtures.py``'s
docstring. This is a plain module, not a ``conftest.py``, so it adds no implicit
directory-wide fixture names: each consumer's import line says what it uses.
``reconciliation/test_harness_construction_guard.py`` keeps every other test
module from calling the constructor directly.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest

from fused_memory.config.schema import FusedMemoryConfig, ReconciliationConfig
from fused_memory.models.reconciliation import StageReport
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.reconciliation.harness import ReconciliationHarness
from fused_memory.reconciliation.journal import ReconciliationJournal
from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator
from fused_memory.reconciliation.stages.task_knowledge_sync import (
    IntegrityCheck,
    TaskKnowledgeSync,
)

if TYPE_CHECKING:
    from fused_memory.backends.task_backend_protocol import TaskBackendProtocol
    from fused_memory.services.memory_service import MemoryService

_PIPELINE_STAGE_CLASSES = (MemoryConsolidator, TaskKnowledgeSync, IntegrityCheck)

_BASELINE_RECONCILIATION_FIELDS: dict[str, Any] = {
    'enabled': True,
    'explore_codebase_root': '/tmp/test',
    'agent_llm_provider': 'anthropic',
    'agent_llm_model': 'claude-sonnet-4-20250514',
}

_BASELINE_EVENT_BUFFER_FIELDS: dict[str, Any] = {
    'buffer_size_threshold': 2,
    'max_staleness_seconds': 3600,
}


def make_memory_service() -> AsyncMock:
    """A memory-service double whose reads a reconciliation cycle makes all come back empty."""
    service = AsyncMock()
    service.search = AsyncMock(return_value=[])
    service.get_episodes = AsyncMock(return_value=[])
    service.get_status = AsyncMock(
        return_value={
            'graphiti': {'connected': True},
            'mem0': {'connected': True},
            'projects': {},
        }
    )
    service.get_entity = AsyncMock(return_value={'nodes': [], 'edges': []})
    service.get_memories_by_metadata = AsyncMock(return_value=[])
    service.mem0 = AsyncMock()
    service.mem0.get_all = AsyncMock(return_value={'results': []})
    return service


def make_reconciliation_config(**overrides: Any) -> ReconciliationConfig:
    """The baseline reconciliation section; ``judge_enabled`` keeps its schema default unless overridden."""
    return ReconciliationConfig(**{**_BASELINE_RECONCILIATION_FIELDS, **overrides})


def make_harness_config(**reconciliation_overrides: Any) -> FusedMemoryConfig:
    """A ``FusedMemoryConfig`` whose only non-default section is ``make_reconciliation_config(...)``."""
    return FusedMemoryConfig(reconciliation=make_reconciliation_config(**reconciliation_overrides))


@asynccontextmanager
async def open_journal(data_dir: Path) -> AsyncIterator[ReconciliationJournal]:
    journal = ReconciliationJournal(data_dir)
    await journal.initialize()
    try:
        yield journal
    finally:
        await journal.close()


@asynccontextmanager
async def open_event_buffer(db_path: Path, **overrides: Any) -> AsyncIterator[EventBuffer]:
    buffer = EventBuffer(db_path=db_path, **{**_BASELINE_EVENT_BUFFER_FIELDS, **overrides})
    await buffer.initialize()
    try:
        yield buffer
    finally:
        await buffer.close()


def make_harness(
    *,
    journal: ReconciliationJournal,
    event_buffer: EventBuffer,
    taskmaster: TaskBackendProtocol | None,
    memory_service: MemoryService | None = None,
    config: FusedMemoryConfig | None = None,
    **harness_kwargs: Any,
) -> ReconciliationHarness:
    """Build a harness; constructor kwargs not named here are forwarded only when given.

    ``taskmaster`` is required because ``None`` ("no taskmaster") is a meaningful
    test input, not a default.
    """
    return ReconciliationHarness(
        memory_service=make_memory_service() if memory_service is None else memory_service,
        taskmaster=taskmaster,
        journal=journal,
        event_buffer=event_buffer,
        config=make_harness_config() if config is None else config,
        **harness_kwargs,
    )


@asynccontextmanager
async def open_harness(
    tmp_path: Path,
    *,
    taskmaster: TaskBackendProtocol | None,
    **make_harness_kwargs: Any,
) -> AsyncIterator[ReconciliationHarness]:
    """A harness on real, initialised stores under *tmp_path*: ``harness.journal`` and ``harness.buffer``."""
    async with (
        open_journal(tmp_path / 'journal') as journal,
        open_event_buffer(tmp_path / 'events.db') as event_buffer,
    ):
        yield make_harness(
            journal=journal,
            event_buffer=event_buffer,
            taskmaster=taskmaster,
            **make_harness_kwargs,
        )


def stub_stage_runs(
    monkeypatch: pytest.MonkeyPatch,
    fake_run: Callable[..., Awaitable[StageReport]],
) -> None:
    """Replace ``run`` on every pipeline stage class with *fake_run* until test teardown.

    *fake_run* has the signature
    ``(self, events, watermark, prior_reports, run_id, model=None, resume_session_id=None)``.
    """
    for stage_class in _PIPELINE_STAGE_CLASSES:
        monkeypatch.setattr(stage_class, 'run', fake_run)
