"""Tests for the startup retention sweep over every write_journal.db table."""

import contextlib
import sqlite3
import uuid
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio

from fused_memory.config.schema import WriteJournalConfig
from fused_memory.services.journal_retention import (
    TABLES_WITHOUT_RETENTION,
    prune_every_table_at_startup,
)
from fused_memory.services.write_journal import WriteJournal

_LONG_AGO = '2000-01-01T00:00:00+00:00'


@pytest_asyncio.fixture
async def journal(tmp_path):
    j = WriteJournal(tmp_path / 'test_wj')
    await j.initialize()
    yield j
    await j.close()


def _backdate(journal: WriteJournal, table: str, key_column: str, key: str,
              *, columns: tuple[str, ...] = ('created_at',), to: str = _LONG_AGO) -> None:
    assignments = ', '.join(f'{column} = ?' for column in columns)
    with contextlib.closing(sqlite3.connect(journal.db_path)) as conn:
        conn.execute(
            f'UPDATE {table} SET {assignments} WHERE {key_column} = ?',
            (*(to for _ in columns), key),
        )
        conn.commit()


def _row_count(journal: WriteJournal, table: str) -> int:
    with contextlib.closing(sqlite3.connect(journal.db_path)) as conn:
        return conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]


def _keys(journal: WriteJournal, table: str, key_column: str) -> set[str]:
    with contextlib.closing(sqlite3.connect(journal.db_path)) as conn:
        return {row[0] for row in conn.execute(f'SELECT {key_column} FROM {table}')}


@pytest.mark.asyncio
async def test_the_startup_sweep_decides_retention_for_every_table(journal):
    with contextlib.closing(sqlite3.connect(journal.db_path)) as conn:
        tables = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }

    result = await prune_every_table_at_startup(journal, WriteJournalConfig())

    assert set(result).isdisjoint(TABLES_WITHOUT_RETENTION)
    assert set(result) | TABLES_WITHOUT_RETENTION == tables, (
        f'write_journal.db tables {sorted(tables - set(result) - TABLES_WITHOUT_RETENTION)} '
        'have no retention decision. Give each one in services/journal_retention.py: '
        'a prune in prune_every_table_at_startup, or an entry in TABLES_WITHOUT_RETENTION.'
    )
    assert all(isinstance(deleted, int) for deleted in result.values()), result


async def _seed_one_row_per_table(journal: WriteJournal) -> dict[str, tuple[str, str]]:
    """One row in each pruned table; returns ``{table: (key_column, key)}``."""
    write_op_id = str(uuid.uuid4())
    await journal.log_write_op(write_op_id=write_op_id, operation='add_memory', kind='write')
    backend_write_op_id = str(uuid.uuid4())
    await journal.log_backend_op(write_op_id=backend_write_op_id, backend='mem0', operation='add')
    intent_id = str(uuid.uuid4())
    await journal.log_mem0_intent(
        intent_id=intent_id, write_op_id=str(uuid.uuid4()), content='c', payload_digest=None,
    )
    await journal.resolve_mem0_intent(intent_id, 'completed')
    client_op_id = str(uuid.uuid4())
    await journal.record_idempotent_result(client_op_id, 'update_task', {})
    episode_uuid = str(uuid.uuid4())
    await journal.log_referent_findings([{'k': 1}], group_id='g', episode_uuid=episode_uuid)
    return {
        'write_ops': ('id', write_op_id),
        'backend_ops': ('write_op_id', backend_write_op_id),
        'mem0_intents': ('id', intent_id),
        'idempotent_ops': ('client_op_id', client_op_id),
        'referent_findings': ('episode_uuid', episode_uuid),
    }


@pytest.mark.asyncio
async def test_the_sweep_ages_out_each_table_and_keeps_fresh_rows(journal):
    fresh = await _seed_one_row_per_table(journal)
    aged = await _seed_one_row_per_table(journal)
    for table, (key_column, key) in aged.items():
        columns = ('created_at', 'resolved_at') if table == 'mem0_intents' else ('created_at',)
        _backdate(journal, table, key_column, key, columns=columns)

    result = await prune_every_table_at_startup(journal, WriteJournalConfig())

    assert result == {table: 1 for table in fresh}
    for table, (key_column, key) in fresh.items():
        assert _keys(journal, table, key_column) == {key}, table


@pytest.mark.asyncio
async def test_the_sweep_ages_out_mem0_intents_and_idempotent_ops_at_their_configured_horizons(
    journal,
):
    ten_days_ago = (datetime.now(UTC) - timedelta(days=10)).isoformat()
    intent_id = str(uuid.uuid4())
    await journal.log_mem0_intent(
        intent_id=intent_id, write_op_id=str(uuid.uuid4()), content='c', payload_digest=None,
    )
    await journal.resolve_mem0_intent(intent_id, 'completed')
    _backdate(journal, 'mem0_intents', 'id', intent_id,
              columns=('created_at', 'resolved_at'), to=ten_days_ago)
    client_op_id = str(uuid.uuid4())
    await journal.record_idempotent_result(client_op_id, 'update_task', {})
    _backdate(journal, 'idempotent_ops', 'client_op_id', client_op_id, to=ten_days_ago)

    result = await prune_every_table_at_startup(
        journal,
        WriteJournalConfig(mem0_intents_retention_days=5, idempotent_ops_retention_days=15),
    )

    assert (result['mem0_intents'], result['idempotent_ops']) == (1, 0), result
    assert _row_count(journal, 'mem0_intents') == 0
    assert _row_count(journal, 'idempotent_ops') == 1


@pytest.mark.asyncio
async def test_write_ops_and_backend_ops_share_one_row_budget(journal):
    aged = (datetime.now(UTC) - timedelta(days=800)).isoformat()
    for _ in range(3):
        write_op_id = str(uuid.uuid4())
        await journal.log_write_op(write_op_id=write_op_id, operation='add_memory', kind='write')
        _backdate(journal, 'write_ops', 'id', write_op_id, to=aged)
        await journal.log_backend_op(write_op_id=write_op_id, backend='mem0', operation='add')
        _backdate(journal, 'backend_ops', 'write_op_id', write_op_id, to=aged)

    result = await prune_every_table_at_startup(
        journal, WriteJournalConfig(prune_batch_size=1, prune_max_rows_per_run=4),
    )

    assert result['write_ops'] + result['backend_ops'] == 4, result
    assert _row_count(journal, 'write_ops') + _row_count(journal, 'backend_ops') == 2


class _TickingClock:
    """Each reading is one second after the last, so every deadline check costs a second."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 1.0
        return self.now


@pytest.mark.asyncio
async def test_write_ops_and_backend_ops_share_one_deadline(journal, monkeypatch):
    from fused_memory.services import write_journal as wj

    aged = (datetime.now(UTC) - timedelta(days=800)).isoformat()
    for _ in range(3):
        write_op_id = str(uuid.uuid4())
        await journal.log_write_op(write_op_id=write_op_id, operation='add_memory', kind='write')
        _backdate(journal, 'write_ops', 'id', write_op_id, to=aged)
        await journal.log_backend_op(write_op_id=write_op_id, backend='mem0', operation='add')
        _backdate(journal, 'backend_ops', 'write_op_id', write_op_id, to=aged)
    monkeypatch.setattr(wj, '_monotonic', _TickingClock())

    result = await prune_every_table_at_startup(
        journal, WriteJournalConfig(prune_batch_size=1, prune_max_seconds=2.0),
    )

    assert result['backend_ops'] == 1, (
        f'write_ops spent the shared deadline, so backend_ops gets only the one batch '
        f'a between-batches deadline check allows: {result}'
    )
    assert _row_count(journal, 'backend_ops') == 2


@pytest.mark.asyncio
async def test_the_sweep_never_raises(journal):
    await journal.close()

    result = await prune_every_table_at_startup(journal, WriteJournalConfig())

    assert result
    assert all(deleted == 0 for deleted in result.values()), result
