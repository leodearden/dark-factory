"""Tests for the write_ops retention policy: its horizons and the pass that runs them."""

import contextlib
import sqlite3

import pytest
import pytest_asyncio

from fused_memory.services.write_journal import (
    READ_HORIZON_PREDICATE,
    SEARCH_HORIZON_PREDICATE,
    WRITE_HORIZON_PREDICATE,
    WriteJournal,
)

_KIND_TIME_SEEK = 'SEARCH write_ops USING INDEX idx_wo_kind_time (kind=? AND created_at<?)'


@pytest_asyncio.fixture
async def journal(tmp_path):
    j = WriteJournal(tmp_path / 'test_wj')
    await j.initialize()
    yield j
    await j.close()


@pytest.mark.parametrize(
    'predicate',
    [READ_HORIZON_PREDICATE, SEARCH_HORIZON_PREDICATE, WRITE_HORIZON_PREDICATE],
    ids=['read', 'search', 'write'],
)
@pytest.mark.asyncio
async def test_every_horizon_is_an_index_seek(journal, predicate):
    with contextlib.closing(sqlite3.connect(journal.db_path)) as conn:
        plan = conn.execute(
            'EXPLAIN QUERY PLAN SELECT id FROM write_ops INDEXED BY idx_wo_kind_time '
            f'WHERE {predicate} LIMIT ?',
            ('2026-01-01T00:00:00+00:00', 1),
        ).fetchall()
    details = [row[3] for row in plan]
    assert _KIND_TIME_SEEK in details, (
        f'{predicate!r} plans as {details}, not a (kind, created_at) range seek. '
        'A SCAN here is the 22 s single-statement lock hold measured on the live '
        'journal, which the between-batches deadline cannot interrupt.'
    )
