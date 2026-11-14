"""Seed and read ``mem0_intents`` journal rows, and build the openai errors that defer them.

Shared by the tests of the Mem0 retry path (task 7397). Every row is moved to
its status through the journal's own API, so a seeded row is one the server
could have produced. Import with ``from _mem0_intent_fixtures import ...``:
``tests/conftest.py`` puts this directory on ``sys.path``.
"""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import closing
from datetime import UTC, datetime

import httpx
import openai

from fused_memory.services.write_journal import Mem0IntentStatus, WriteJournal

EMBEDDINGS_REQUEST = httpx.Request('POST', 'https://api.openai.com/v1/embeddings')


def openai_status_error(
    cls: type[openai.APIStatusError], status: int
) -> openai.APIStatusError:
    return cls(
        f'Error code: {status} - no credits',
        response=httpx.Response(status, request=EMBEDDINGS_REQUEST),
        body=None,
    )


def rate_limit_error() -> openai.RateLimitError:
    error = openai_status_error(openai.RateLimitError, 429)
    assert isinstance(error, openai.RateLimitError)
    return error


async def seed_mem0_intent(
    journal: WriteJournal,
    *,
    status: Mem0IntentStatus = Mem0IntentStatus.PENDING,
    reason: str | None = None,
    retry_at: datetime | None = None,
    created_at: datetime | None = None,
    write_op_id: str | None = None,
    content: str = 'seeded',
    metadata: dict | None = None,
    project_id: str | None = None,
    agent_id: str | None = None,
    session_id: str | None = None,
    category: str | None = None,
) -> str:
    """Journal one intent, move it to ``status``, and return its id.

    A ``retrying`` row is scheduled for ``retry_at`` (default: now).
    """
    intent_id = str(uuid.uuid4())
    await journal.log_mem0_intent(
        intent_id=intent_id,
        write_op_id=write_op_id or str(uuid.uuid4()),
        project_id=project_id,
        agent_id=agent_id,
        session_id=session_id,
        category=category,
        content=content,
        metadata=metadata,
        payload_digest='d',
    )
    if status == Mem0IntentStatus.RETRYING:
        assert await journal.schedule_mem0_retry(
            intent_id,
            reason=reason or 'rate_limited',
            retry_at=retry_at or datetime.now(UTC),
        )
    elif status != Mem0IntentStatus.PENDING:
        assert await journal.resolve_mem0_intent(intent_id, status, reason=reason)
    if created_at is not None:
        set_mem0_intent_created_at(journal, intent_id, created_at)
    return intent_id


def set_mem0_intent_created_at(
    journal: WriteJournal, intent_id: str, created_at: datetime
) -> None:
    """Backdate a row; the journal API always stamps ``created_at`` with now."""
    with closing(sqlite3.connect(journal.db_path)) as conn, conn:
        conn.execute(
            'UPDATE mem0_intents SET created_at = ? WHERE id = ?',
            (created_at.isoformat(), intent_id),
        )


async def mem0_intent_row(journal: WriteJournal, intent_id: str) -> dict:
    (row,) = [r for r in await journal.get_mem0_intents() if r['id'] == intent_id]
    return row
