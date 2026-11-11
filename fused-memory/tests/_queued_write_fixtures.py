"""Test helpers for the durable queue's typed values (``fused_memory.services.queued_writes``).

Imported by name (``from _queued_write_fixtures import queued``), like
``_fm_helpers``. ``queued`` turns a legacy payload dict into a value through
the public row decoder, so a fixture written in the old dict shape also
exercises the legacy-decode path.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import aiosqlite

from fused_memory.services.queued_writes import (
    QueueCallback,
    QueuedEpisodeWrite,
    QueuedGraphitiMemoryWrite,
    QueuedMem0Fact,
    QueuedWrite,
    WriteEnvelope,
    decode_queued_write,
)

#: The flat keys an ``add_episode`` row carries today (PRD S26 boundary row 1).
EPISODE_WIRE_KEYS = frozenset({
    'name', 'content', 'correlation_id', 'source', 'group_id',
    'source_description', 'project_id', 'agent_id', 'session_id',
    '_causation_id', '_write_op_id', 'temporal_context', 'unverified_claim',
    'reference_time', 'referents',
})


def queued(operation: str, payload: Mapping[str, Any], *, group_id: str | None = None) -> QueuedWrite:
    """*payload* decoded as a row of *operation* would be; the group defaults to the payload's own."""
    row_group = (
        group_id
        or payload.get('group_id')
        or f"mem0_{payload.get('project_id', 'main')}"
    )
    return decode_queued_write(operation, json.dumps(payload), group_id=row_group, row_id=0)


def envelope(**overrides: Any) -> WriteEnvelope:
    fields: dict[str, Any] = {'project_id': 'test', 'group_id': 'test'} | overrides
    return WriteEnvelope(**fields)


def episode_write(**overrides: Any) -> QueuedEpisodeWrite:
    fields: dict[str, Any] = {
        'envelope': envelope(),
        'name': 'episode_test',
        'content': 'test content',
        'source_description': 'notes',
    } | overrides
    return QueuedEpisodeWrite(**fields)


def graphiti_memory_write(**overrides: Any) -> QueuedGraphitiMemoryWrite:
    fields: dict[str, Any] = {
        'envelope': envelope(),
        'name': 'memory_test',
        'content': 'test content',
    } | overrides
    return QueuedGraphitiMemoryWrite(**fields)


def mem0_fact(**overrides: Any) -> QueuedMem0Fact:
    fields: dict[str, Any] = {
        'envelope': envelope(group_id='mem0_test'),
        'fact_text': 'a fact',
    } | overrides
    return QueuedMem0Fact(**fields)


def enqueued(mock: AsyncMock) -> list[tuple[QueuedWrite, QueueCallback | None]]:
    """Every ``(value, callback)`` an ``enqueue`` or ``enqueue_batch`` AsyncMock was awaited with, in order."""
    pairs: list[tuple[QueuedWrite, QueueCallback | None]] = []
    for call in mock.await_args_list:
        first = call.args[0]
        if isinstance(first, list | tuple):
            pairs.extend(first)
        else:
            pairs.append((first, call.kwargs.get('callback')))
    return pairs


async def insert_raw_row(
    db_path: Path,
    *,
    group_id: str,
    operation: str,
    payload_text: str,
    callback_type: str | None = None,
    status: str = 'pending',
    max_attempts: int = 5,
) -> int:
    """Hand-insert one ``write_queue`` row, bypassing ``enqueue``'s typed API. Returns its id."""
    async with aiosqlite.connect(str(db_path)) as db:
        cursor = await db.execute(
            'INSERT INTO write_queue '
            '(group_id, operation, payload, callback_type, status, attempts, '
            ' max_attempts, next_retry_at, created_at, executed) '
            'VALUES (?, ?, ?, ?, ?, 0, ?, 0, ?, 0)',
            (group_id, operation, payload_text, callback_type, status, max_attempts, time.time()),
        )
        await db.commit()
    assert cursor.lastrowid is not None
    return cursor.lastrowid
