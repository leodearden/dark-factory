"""Tests for fused-memory/scripts/memory_eval_write_after_miss.py (task 3213).

The producer reads the write journal, so the fixture journal is built through
the REAL producers of its rows (``WriteJournal.log_write_op`` plus the
``read_telemetry`` summarisers). The consumer side of the search-telemetry
contract therefore parses genuine rows, not a hand-copied shape.

``asyncio_mode = "strict"``: every async test carries an explicit
``@pytest.mark.asyncio``.
"""

from __future__ import annotations

import contextlib
import functools
import itertools
import re
import sqlite3
import types
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _fm_helpers import load_script_module

from fused_memory.models.enums import SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.services.read_telemetry import (
    fallback_search_summary,
    summarize_search_query,
    summarize_search_results,
)
from fused_memory.services.write_journal import WriteJournal

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'memory_eval_write_after_miss.py'

PROJECT = 'dark_factory'
CALLER_A = 'claude-task-101-implementer'
CALLER_B = 'claude-task-202-implementer'


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(SCRIPT_PATH, mod_name='memory_eval_write_after_miss')


def _hit(memory_id: str, *, score: float = 0.5, category: str = 'procedural_knowledge',
         metadata: dict | None = None, created_at: str | None = None) -> MemoryResult:
    return MemoryResult(
        id=memory_id,
        content=f'content of {memory_id}',
        category=category,
        source_store=SourceStore.mem0,
        metadata={'store_score': score, **(metadata or {})},
        created_at=created_at,
    )


def _grouped(parent_id: str, *, children: Sequence[str] = (), amendments: Sequence[str] = ()) -> dict:
    """A grouped MCP payload entry, the shape ``server/grouped_read.py`` folds children into."""
    return {
        'id': parent_id,
        'content': f'content of {parent_id}',
        'source_store': 'mem0',
        'metadata': {'store_score': 0.5},
        'grouped': {
            'matched_children': [{'id': child} for child in children],
            'amendments': [{'id': amendment} for amendment in amendments],
        },
    }


class _FixtureJournal:
    """Logs rows in sequence through the real journal, so ``created_at`` orders them."""

    def __init__(self, journal: WriteJournal) -> None:
        self._journal = journal
        self._ids = itertools.count(1)
        self.db_path = journal.db_path

    def _next_id(self, kind: str) -> str:
        return f'row-{next(self._ids):03d}-{kind}'

    async def search(
        self,
        caller: str | None,
        shown: Sequence[Any] = (),
        *,
        project_id: str = PROJECT,
        result_summary: dict | None = None,
    ) -> str:
        row_id = self._next_id('search')
        params: dict[str, Any] = {**summarize_search_query(f'query for {row_id}'), 'limit': 5}
        if caller is not None:
            params['caller_agent_id'] = caller
        await self._journal.log_write_op(
            write_op_id=row_id,
            operation='search',
            kind='read',
            project_id=project_id,
            params=params,
            result_summary=(
                result_summary if result_summary is not None else summarize_search_results(shown)
            ),
        )
        return row_id

    async def write(
        self,
        caller: str,
        *,
        content: str,
        category: str = 'procedural_knowledge',
        memory_ids: Sequence[str] = (),
        project_id: str = PROJECT,
    ) -> str:
        row_id = self._next_id('write')
        await self._journal.log_write_op(
            write_op_id=row_id,
            operation='add_memory',
            kind='write',
            project_id=project_id,
            agent_id=caller,
            params={'content': content[:200], 'category': category},
            result_summary={
                'memory_ids': list(memory_ids),
                'stores': ['mem0'],
                'stores_pending': [],
            },
        )
        return row_id


@contextlib.asynccontextmanager
async def _fixture_journal(tmp_path: Path) -> AsyncIterator[_FixtureJournal]:
    journal = WriteJournal(tmp_path / 'journal')
    await journal.initialize()
    try:
        yield _FixtureJournal(journal)
    finally:
        await journal.close()


def _period_around_now() -> Any:
    now = datetime.now(UTC)
    return _mod().Period(now - timedelta(hours=1), now + timedelta(hours=1))


def _read(db_path: Path, period: Any = None, *, project_id: str = PROJECT) -> Any:
    return _mod().read_journal(
        db_path, project_id=project_id, period=period or _period_around_now(),
    )


class TestJournalReader:
    """``read_journal`` parses rows at the boundary into frozen values, read-only."""

    @pytest.mark.asyncio
    async def test_a_write_row_carries_its_caller_category_ids_and_aware_time(self, tmp_path):
        async with _fixture_journal(tmp_path) as fixture:
            row_id = await fixture.write(
                CALLER_A, content='a gotcha', category='procedural_knowledge',
                memory_ids=['m-1', 'm-2'],
            )

        window = _read(fixture.db_path)

        assert len(window.writes) == 1
        write = window.writes[0]
        assert isinstance(write, _mod().WriteRow)
        assert write.row_id == row_id
        assert write.project_id == PROJECT
        assert write.caller_agent_id == CALLER_A
        assert write.category == 'procedural_knowledge'
        assert write.memory_ids == ('m-1', 'm-2')
        assert write.created_at.tzinfo is not None

    @pytest.mark.asyncio
    async def test_a_search_row_shows_result_ids_and_every_folded_child(self, tmp_path):
        async with _fixture_journal(tmp_path) as fixture:
            row_id = await fixture.search(
                CALLER_A,
                [_hit('hit-1'), _grouped('parent-1', children=['child-1'], amendments=['amend-1'])],
            )

        window = _read(fixture.db_path)

        assert len(window.searches) == 1
        search = window.searches[0]
        assert isinstance(search, _mod().SearchRow)
        assert search.row_id == row_id
        assert search.project_id == PROJECT
        assert search.caller_agent_id == CALLER_A
        assert search.shown_ids == frozenset({'hit-1', 'parent-1', 'child-1', 'amend-1'})
        assert search.complete is True
        assert search.created_at.tzinfo is not None

    @pytest.mark.asyncio
    async def test_a_telemetry_error_or_truncated_search_is_incomplete(self, tmp_path):
        async with _fixture_journal(tmp_path) as fixture:
            errored = await fixture.search(CALLER_A, result_summary=fallback_search_summary(3))
            truncated = await fixture.search(
                CALLER_A,
                result_summary=summarize_search_results(
                    [_hit('t-1'), _hit('t-2'), _hit('t-3')], max_results=1,
                ),
            )

        window = _read(fixture.db_path)

        completeness = {search.row_id: search.complete for search in window.searches}
        assert completeness == {errored: False, truncated: False}

    @pytest.mark.asyncio
    async def test_an_unattributed_search_is_disclosed_not_returned(self, tmp_path):
        async with _fixture_journal(tmp_path) as fixture:
            await fixture.search(None, [_hit('hit-1')])
            attributed = await fixture.search(CALLER_A, [_hit('hit-2')])

        window = _read(fixture.db_path)

        assert [search.row_id for search in window.searches] == [attributed]
        assert window.unattributed_searches == 1

    @pytest.mark.asyncio
    async def test_other_projects_and_rows_outside_the_period_are_excluded(self, tmp_path):
        async with _fixture_journal(tmp_path) as fixture:
            await fixture.write(CALLER_A, content='before', memory_ids=['m-0'])
            period_start = datetime.now(UTC)
            inside = await fixture.write(CALLER_A, content='inside', memory_ids=['m-1'])
            await fixture.write(
                CALLER_A, content='elsewhere', memory_ids=['m-2'], project_id='other_project',
            )
            await fixture.search(CALLER_A, [_hit('hit-1')], project_id='other_project')
            period_end = datetime.now(UTC)
            await fixture.write(CALLER_A, content='after', memory_ids=['m-3'])

        window = _read(fixture.db_path, _mod().Period(period_start, period_end))

        assert [write.row_id for write in window.writes] == [inside]
        assert window.searches == ()

    def test_a_missing_journal_is_refused_by_name(self, tmp_path):
        missing = tmp_path / 'absent' / 'write_journal.db'
        with pytest.raises(_mod().JournalUnavailableError, match=re.escape(str(missing))):
            _read(missing)

    def test_a_sqlite_file_without_write_ops_is_refused_by_name(self, tmp_path):
        not_a_journal = tmp_path / 'other.db'
        with contextlib.closing(sqlite3.connect(not_a_journal)) as connection:
            connection.execute('CREATE TABLE something_else (x INTEGER)')
            connection.commit()
        with pytest.raises(_mod().JournalUnavailableError, match=re.escape(str(not_a_journal))):
            _read(not_a_journal)
