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
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from _fm_helpers import load_script_module

from fused_memory.models.enums import MemoryCategory, SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.services.memory_service import SearchResults
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
THRESHOLD = 0.9


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(SCRIPT_PATH, mod_name='memory_eval_write_after_miss')


def _hit(memory_id: str, *, score: float = 0.5, category: str = 'procedural_knowledge',
         metadata: dict | None = None, created_at: str | None = None) -> MemoryResult:
    return MemoryResult(
        id=memory_id,
        content=f'content of {memory_id}',
        category=MemoryCategory(category),
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


class FakeMemory:
    """A read-only ``MemoryService`` double: the seams the producer reads, and write methods that raise.

    Candidates are keyed by the query content, which is the write's stored
    content, so a test states "searching W1's content returns these entries".
    The threshold is set on the config path the production resolver reads.
    """

    def __init__(
        self,
        *,
        contents: Mapping[str, str],
        candidates: Mapping[str, Sequence[MemoryResult]] | None = None,
        degraded: bool = False,
        threshold: float = THRESHOLD,
    ) -> None:
        self.config = SimpleNamespace(
            reconciliation=SimpleNamespace(procedural_knowledge_near_dup_threshold=threshold),
        )
        self._contents = dict(contents)
        self._candidates = {query: list(results) for query, results in (candidates or {}).items()}
        self._degraded = degraded
        self.search_calls: list[dict[str, Any]] = []

    async def get_memory_by_id(self, project_id: str, memory_id: str) -> dict | None:
        content = self._contents.get(memory_id)
        if content is None:
            return None
        return {'id': memory_id, 'content': content, 'metadata': {}}

    async def search(self, **kwargs: Any) -> SearchResults:
        self.search_calls.append(kwargs)
        return SearchResults(
            self._candidates.get(kwargs['query'], []),
            degraded=self._degraded,
            failed_stores=['mem0'] if self._degraded else None,
        )

    async def add_memory(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError('the write-after-miss producer must never write the corpus')

    async def update_memory(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError('the write-after-miss producer must never write the corpus')

    async def delete_memory(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError('the write-after-miss producer must never write the corpus')


def _before(write: Any) -> str:
    return (write.created_at - timedelta(minutes=5)).isoformat()


def _after(write: Any) -> str:
    return (write.created_at + timedelta(minutes=5)).isoformat()


def _write_by_row(window: Any, row_id: str) -> Any:
    return next(write for write in window.writes if write.row_id == row_id)


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


async def _headline_fixture(tmp_path: Path) -> tuple[Any, FakeMemory, dict[str, str]]:
    """Caller A misses an existing entry and re-writes it; caller B is shown it and still writes.

    Returns the parsed window, the memory double, and the row ids by role.
    """
    async with _fixture_journal(tmp_path) as fixture:
        search_a = await fixture.search(CALLER_A, [_hit('x')])
        write_1 = await fixture.write(CALLER_A, content='W1 content', memory_ids=['w1'])
        search_b = await fixture.search(CALLER_B, [_hit('y2')])
        write_2 = await fixture.write(CALLER_B, content='W2 content', memory_ids=['w2'])
    window = _read(fixture.db_path)
    w1 = _write_by_row(window, write_1)
    w2 = _write_by_row(window, write_2)
    memory = FakeMemory(
        contents={'w1': 'W1 content', 'w2': 'W2 content'},
        candidates={
            'W1 content': [_hit('w1', score=1.0, created_at=w1.created_at.isoformat()),
                           _hit('y', score=0.95, created_at=_before(w1))],
            'W2 content': [_hit('w2', score=1.0, created_at=w2.created_at.isoformat()),
                           _hit('y2', score=0.96, created_at=_before(w2))],
        },
    )
    rows = {'search_a': search_a, 'write_1': write_1, 'search_b': search_b, 'write_2': write_2}
    return window, memory, rows


async def _single_write_fixture(tmp_path: Path, *, shown: Sequence[Any] = (_hit('x'),)) -> Any:
    """Caller A searches (shown *shown*) and then writes ``W content`` as memory ``w``."""
    async with _fixture_journal(tmp_path) as fixture:
        await fixture.search(CALLER_A, list(shown))
        await fixture.write(CALLER_A, content='W content', memory_ids=['w'])
    return _read(fixture.db_path)


async def _adjudicate(window: Any, memory: FakeMemory) -> Any:
    return await _mod().adjudicate(window, memory, project_id=PROJECT)


class TestAdjudication:
    """One incident per write whose guard-selected near-duplicate its window was never shown."""

    @pytest.mark.asyncio
    async def test_a_missed_near_duplicate_is_one_incident_and_a_shown_one_is_none(self, tmp_path):
        window, memory, rows = await _headline_fixture(tmp_path)
        w1 = _write_by_row(window, rows['write_1'])

        adjudication = await _adjudicate(window, memory)

        assert adjudication.adjudicated == 2
        assert len(adjudication.incidents) == 1
        incident = adjudication.incidents[0]
        assert isinstance(incident, _mod().Incident)
        assert incident.write_row_id == rows['write_1']
        assert incident.search_row_ids == (rows['search_a'],)
        assert incident.project_id == PROJECT
        assert incident.caller_agent_id == CALLER_A
        assert incident.matched_memory_id == 'y'
        assert incident.similarity == pytest.approx(0.95)
        assert incident.threshold == pytest.approx(THRESHOLD)
        assert incident.category == 'procedural_knowledge'
        assert incident.write_created_at == w1.created_at

    @pytest.mark.asyncio
    async def test_the_writes_own_ids_are_never_its_duplicate(self, tmp_path):
        window = await _single_write_fixture(tmp_path)
        write = window.writes[0]
        memory = FakeMemory(
            contents={'w': 'W content'},
            candidates={'W content': [_hit('w', score=1.0, created_at=_before(write))]},
        )

        adjudication = await _adjudicate(window, memory)

        assert adjudication.adjudicated == 1
        assert adjudication.incidents == ()

    @pytest.mark.asyncio
    async def test_a_candidate_created_after_the_write_is_ignored(self, tmp_path):
        window = await _single_write_fixture(tmp_path)
        write = window.writes[0]
        memory = FakeMemory(
            contents={'w': 'W content'},
            candidates={'W content': [_hit('later', score=0.99, created_at=_after(write))]},
        )

        adjudication = await _adjudicate(window, memory)

        assert adjudication.adjudicated == 1
        assert adjudication.incidents == ()

    @pytest.mark.asyncio
    async def test_a_candidate_below_the_configured_threshold_is_no_incident(self, tmp_path):
        window = await _single_write_fixture(tmp_path)
        write = window.writes[0]
        memory = FakeMemory(
            contents={'w': 'W content'},
            candidates={'W content': [_hit('near', score=0.85, created_at=_before(write))]},
            threshold=0.9,
        )

        adjudication = await _adjudicate(window, memory)

        assert adjudication.adjudicated == 1
        assert adjudication.incidents == ()

    @pytest.mark.asyncio
    async def test_a_child_whose_grouped_parent_was_shown_counts_as_shown(self, tmp_path):
        window = await _single_write_fixture(tmp_path, shown=[_hit('parent')])
        write = window.writes[0]
        child = _hit(
            'child', score=0.97, created_at=_before(write),
            metadata={'parent_id': 'parent', 'kind': 'amendment'},
        )
        memory = FakeMemory(contents={'w': 'W content'}, candidates={'W content': [child]})

        adjudication = await _adjudicate(window, memory)

        assert adjudication.adjudicated == 1
        assert adjudication.incidents == ()

    @pytest.mark.asyncio
    async def test_every_candidate_search_has_the_guards_shape_and_journals_nothing(self, tmp_path):
        window, memory, _rows = await _headline_fixture(tmp_path)

        await _adjudicate(window, memory)

        assert len(memory.search_calls) == 2
        for call in memory.search_calls:
            assert call['stores'] == ['mem0']
            assert call['categories'] == ['procedural_knowledge']
            assert call['anchor_topics'] is False
            assert call['project_id'] == PROJECT
            assert 'causation_id' not in call
        assert {call['query'] for call in memory.search_calls} == {'W1 content', 'W2 content'}

    @pytest.mark.asyncio
    async def test_a_write_with_no_earlier_same_caller_search_is_not_adjudicated(self, tmp_path):
        async with _fixture_journal(tmp_path) as fixture:
            await fixture.search(CALLER_A, [_hit('x')])
            await fixture.write(CALLER_B, content='W content', memory_ids=['w'])
            await fixture.search(CALLER_B, [_hit('x')])
        window = _read(fixture.db_path)
        memory = FakeMemory(
            contents={'w': 'W content'},
            candidates={'W content': [_hit('y', score=0.99, created_at=_before(window.writes[0]))]},
        )

        adjudication = await _adjudicate(window, memory)

        assert adjudication.adjudicated == 0
        assert adjudication.incidents == ()
        assert memory.search_calls == []
