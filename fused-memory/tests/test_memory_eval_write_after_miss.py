"""Tests for fused-memory/scripts/memory_eval_write_after_miss.py (task 3213).

The producer reads the write journal, so the fixture journal is built through
the REAL producers of its rows (``WriteJournal.log_write_op`` plus the
``read_telemetry`` summarisers). The consumer side of the search-telemetry
contract therefore parses genuine rows, not a hand-copied shape.

``asyncio_mode = "strict"``: every async test carries an explicit
``@pytest.mark.asyncio``.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import itertools
import json
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
from shared.memory_eval_metrics import (
    RUN_STAMP_ENV_VAR,
    SCHEMA_VERSION,
    Corpus,
    MetricSeries,
    format_stamp,
    load_metric_series,
    stamp_datetime,
    write_metric_series,
)

from fused_memory.models.enums import MemoryCategory, SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.services.memory_service import SearchResults
from fused_memory.services.read_telemetry import (
    SEARCH_TELEMETRY_SCHEMA_VERSION,
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
STAMP = '20261109T220000Z'


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
        caller: str | None,
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

    async def raw_write(self, *, params: dict, result_summary: dict | str) -> str:
        """An ``add_memory`` row with blobs of any shape, such as the durable queue's ``str(result)``."""
        row_id = self._next_id('raw-write')
        await self._journal.log_write_op(
            write_op_id=row_id,
            operation='add_memory',
            kind='write',
            project_id=PROJECT,
            agent_id=CALLER_A,
            params=params,
            result_summary=result_summary,
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
    content, so a test states "searching W1's content returns these entries",
    best first; a search returns the first ``limit`` of them, as a vector store does.
    The threshold is set on the config path ``resolve_near_dup_threshold`` reads.
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
        self.write_calls: list[str] = []

    async def get_memory_by_id(self, project_id: str, memory_id: str) -> dict | None:
        content = self._contents.get(memory_id)
        if content is None:
            return None
        return {'id': memory_id, 'content': content, 'metadata': {}}

    async def search(self, **kwargs: Any) -> SearchResults:
        self.search_calls.append(kwargs)
        return SearchResults(
            self._candidates.get(kwargs['query'], [])[:kwargs['limit']],
            degraded=self._degraded,
            failed_stores=['mem0'] if self._degraded else None,
        )

    async def add_memory(self, *args: Any, **kwargs: Any) -> Any:
        self.write_calls.append('add_memory')
        raise AssertionError('the write-after-miss producer must never write the corpus')

    async def update_memory(self, *args: Any, **kwargs: Any) -> Any:
        self.write_calls.append('update_memory')
        raise AssertionError('the write-after-miss producer must never write the corpus')

    async def delete_memory(self, *args: Any, **kwargs: Any) -> Any:
        self.write_calls.append('delete_memory')
        raise AssertionError('the write-after-miss producer must never write the corpus')

    async def close(self) -> None:
        self.closed = True


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
    async def test_a_search_row_of_another_telemetry_schema_version_is_incomplete(self, tmp_path):
        reshaped = {
            **summarize_search_results([_hit('hit-1')]),
            'schema_version': SEARCH_TELEMETRY_SCHEMA_VERSION + 1,
        }
        async with _fixture_journal(tmp_path) as fixture:
            row_id = await fixture.search(CALLER_A, result_summary=reshaped)

        window = _read(fixture.db_path)

        assert [(search.row_id, search.complete) for search in window.searches] == [(row_id, False)]

    @pytest.mark.asyncio
    async def test_rows_without_their_producers_shape_are_counted_not_raised(self, tmp_path):
        async with _fixture_journal(tmp_path) as fixture:
            await fixture.raw_write(
                params={'content': 'queued', 'category': 'procedural_knowledge'},
                result_summary="{'memory_ids': ['m-1']}",
            )
            await fixture.raw_write(
                params={'content': 'no category'},
                result_summary={'memory_ids': ['m-2'], 'stores': ['mem0'], 'stores_pending': []},
            )

        window = _read(fixture.db_path)

        assert window.writes == ()
        assert window.unparsed_rows == 2

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


async def _headline_fixture(tmp_path: Path) -> tuple[Any, FakeMemory, dict[str, str], Path]:
    """Caller A misses an existing entry and re-writes it; caller B is shown it and still writes.

    Returns the parsed window, the memory double, the row ids by role, and the journal path.
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
    return window, memory, rows, fixture.db_path


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
        window, memory, rows, _journal = await _headline_fixture(tmp_path)
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
    async def test_an_undated_candidate_is_dropped_and_counted(self, tmp_path):
        window = await _single_write_fixture(tmp_path)
        memory = FakeMemory(
            contents={'w': 'W content'},
            candidates={'W content': [_hit('undated', score=0.99, created_at=None)]},
        )

        adjudication = await _adjudicate(window, memory)

        assert adjudication.adjudicated == 1
        assert adjudication.incidents == ()
        assert adjudication.undated_candidates == 1

    @pytest.mark.asyncio
    async def test_later_entries_cannot_crowd_out_a_pre_existing_match(self, tmp_path):
        window = await _single_write_fixture(tmp_path)
        write = window.writes[0]
        depth = _mod().CANDIDATE_DEPTH
        restatements = [
            _hit(f'later-{index}', score=0.99 - index * 0.001, created_at=_after(write))
            for index in range(depth + 1)
        ]
        memory = FakeMemory(
            contents={'w': 'W content'},
            candidates={'W content': [
                _hit('w', score=1.0, created_at=write.created_at.isoformat()),
                *restatements,
                _hit('y', score=0.95, created_at=_before(write)),
            ]},
        )

        adjudication = await _adjudicate(window, memory)

        assert [incident.matched_memory_id for incident in adjudication.incidents] == ['y']
        limits = [call['limit'] for call in memory.search_calls]
        assert limits == sorted(set(limits)) and len(limits) > 1

    @pytest.mark.asyncio
    async def test_a_candidate_below_the_configured_threshold_is_no_incident(self, tmp_path):
        window = await _single_write_fixture(tmp_path)
        write = window.writes[0]
        memory = FakeMemory(
            contents={'w': 'W content'},
            candidates={'W content': [_hit('near', score=0.93, created_at=_before(write))]},
            threshold=0.95,
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
        window, memory, _rows, _journal = await _headline_fixture(tmp_path)

        await _adjudicate(window, memory)

        assert len(memory.search_calls) == 2
        for call in memory.search_calls:
            assert call['stores'] == ['mem0']
            assert call['categories'] == ['procedural_knowledge']
            assert call['anchor_topics'] is False
            assert call['project_id'] == PROJECT
            assert call['limit'] == _mod().CANDIDATE_DEPTH + 1
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


def _assert_only_skipped_as(adjudication: Any, reason_name: str) -> None:
    reason = getattr(_mod().SkipReason, reason_name)
    assert adjudication.adjudicated == 0
    assert adjudication.incidents == ()
    assert adjudication.skipped[reason] == 1
    assert sum(adjudication.skipped.values()) == 1


class TestUnadjudicableWritesAreDisclosed:
    """No write vanishes: every one is adjudicated or counted under exactly one SkipReason."""

    @pytest.mark.asyncio
    async def test_unresolvable_content_is_content_unavailable(self, tmp_path):
        window = await _single_write_fixture(tmp_path)

        adjudication = await _adjudicate(window, FakeMemory(contents={}))

        _assert_only_skipped_as(adjudication, 'CONTENT_UNAVAILABLE')

    @pytest.mark.asyncio
    async def test_a_degraded_candidate_search_is_disclosed(self, tmp_path):
        window = await _single_write_fixture(tmp_path)

        adjudication = await _adjudicate(
            window, FakeMemory(contents={'w': 'W content'}, degraded=True),
        )

        _assert_only_skipped_as(adjudication, 'CANDIDATE_SEARCH_DEGRADED')

    @pytest.mark.asyncio
    async def test_a_window_still_crowded_at_the_ceiling_is_skipped_not_clean(self, tmp_path):
        window = await _single_write_fixture(tmp_path)
        write = window.writes[0]
        memory = FakeMemory(
            contents={'w': 'W content'},
            candidates={'W content': [
                _hit(f'later-{index}', score=0.8, created_at=_after(write))
                for index in range(_mod().CANDIDATE_FETCH_CEILING)
            ]},
        )

        adjudication = await _adjudicate(window, memory)

        _assert_only_skipped_as(adjudication, 'CANDIDATES_CROWDED_OUT')
        assert memory.search_calls[-1]['limit'] == _mod().CANDIDATE_FETCH_CEILING

    @pytest.mark.asyncio
    async def test_an_unseen_match_behind_an_incomplete_search_is_never_an_incident(self, tmp_path):
        async with _fixture_journal(tmp_path) as fixture:
            await fixture.search(CALLER_A, result_summary=fallback_search_summary(4))
            await fixture.write(CALLER_A, content='W content', memory_ids=['w'])
        window = _read(fixture.db_path)
        memory = FakeMemory(
            contents={'w': 'W content'},
            candidates={'W content': [_hit('y', score=0.97, created_at=_before(window.writes[0]))]},
        )

        adjudication = await _adjudicate(window, memory)

        _assert_only_skipped_as(adjudication, 'SHOWN_SET_INCOMPLETE')

    @pytest.mark.asyncio
    async def test_a_match_a_complete_sibling_showed_is_adjudicated_clean(self, tmp_path):
        async with _fixture_journal(tmp_path) as fixture:
            await fixture.search(CALLER_A, result_summary=fallback_search_summary(4))
            await fixture.search(CALLER_A, [_hit('y')])
            await fixture.write(CALLER_A, content='W content', memory_ids=['w'])
        window = _read(fixture.db_path)
        memory = FakeMemory(
            contents={'w': 'W content'},
            candidates={'W content': [_hit('y', score=0.97, created_at=_before(window.writes[0]))]},
        )

        adjudication = await _adjudicate(window, memory)

        assert adjudication.adjudicated == 1
        assert adjudication.incidents == ()
        assert sum(adjudication.skipped.values()) == 0

    @pytest.mark.asyncio
    async def test_a_graphiti_primary_write_is_category_not_comparable(self, tmp_path):
        async with _fixture_journal(tmp_path) as fixture:
            await fixture.search(CALLER_A, [_hit('x')])
            await fixture.write(
                CALLER_A, content='W content', category='decisions_and_rationale',
                memory_ids=['w'],
            )
        window = _read(fixture.db_path)

        adjudication = await _adjudicate(window, FakeMemory(contents={'w': 'W content'}))

        _assert_only_skipped_as(adjudication, 'CATEGORY_NOT_COMPARABLE')

    @pytest.mark.asyncio
    async def test_a_write_that_landed_no_memory_id_is_disclosed(self, tmp_path):
        async with _fixture_journal(tmp_path) as fixture:
            await fixture.search(CALLER_A, [_hit('x')])
            await fixture.write(CALLER_A, content='W content', memory_ids=[])
        window = _read(fixture.db_path)

        adjudication = await _adjudicate(window, FakeMemory(contents={}))

        _assert_only_skipped_as(adjudication, 'NO_MEMORY_ID')

    @pytest.mark.asyncio
    async def test_a_write_with_no_prior_search_is_disclosed(self, tmp_path):
        async with _fixture_journal(tmp_path) as fixture:
            await fixture.write(CALLER_A, content='W content', memory_ids=['w'])
            await fixture.search(CALLER_A, [_hit('x')])
        window = _read(fixture.db_path)

        adjudication = await _adjudicate(window, FakeMemory(contents={'w': 'W content'}))

        _assert_only_skipped_as(adjudication, 'NO_PRIOR_SEARCH')

    @pytest.mark.asyncio
    async def test_a_write_naming_no_agent_is_no_caller_not_no_prior_search(self, tmp_path):
        async with _fixture_journal(tmp_path) as fixture:
            await fixture.search(CALLER_A, [_hit('x')])
            await fixture.write(
                None, content='W content', category='observations_and_summaries',
                memory_ids=['w'],
            )
        window = _read(fixture.db_path)

        adjudication = await _adjudicate(window, FakeMemory(contents={'w': 'W content'}))

        _assert_only_skipped_as(adjudication, 'NO_CALLER')


async def _run(memory: FakeMemory, journal_path: Path, out_root: Path) -> Any:
    return await _mod().run_write_after_miss(
        memory,
        journal_path=journal_path,
        out_root=out_root,
        project_id=PROJECT,
        stamp=STAMP,
        period=_period_around_now(),
    )


class TestSeriesAndArtifacts:
    """One M1 proportion, its structured incident evidence, and every narrowing disclosed."""

    @pytest.mark.asyncio
    async def test_the_series_holds_one_proportion_of_incidents_over_adjudicated(self, tmp_path):
        _window, memory, _rows, journal = await _headline_fixture(tmp_path)
        out_root = tmp_path / 'evals'

        await _run(memory, journal, out_root)

        metrics_path = out_root / 'e7-write-after-miss' / f'metrics-{STAMP}.json'
        series = load_metric_series(metrics_path)
        assert series.eval_id == 'e7-write-after-miss'
        assert series.run_stamp == STAMP
        assert len(series.metrics) == 1
        metric = series.metrics[0]
        assert metric.metric_id == 'write-after-miss'
        assert metric.kind == 'proportion'
        assert metric.direction == 'higher_is_worse'
        assert metric.denominator == metric.n == 2
        assert metric.value == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_details_path_resolves_to_the_structured_incident_evidence(self, tmp_path):
        _window, memory, rows, journal = await _headline_fixture(tmp_path)
        out_root = tmp_path / 'evals'
        period = _period_around_now()

        outcome = await _mod().run_write_after_miss(
            memory, journal_path=journal, out_root=out_root, project_id=PROJECT,
            stamp=STAMP, period=period,
        )

        eval_root = out_root / 'e7-write-after-miss'
        metric = outcome.series.metrics[0]
        incidents_path = eval_root / f'incidents-{STAMP}.json'
        assert metric.details_path is not None
        assert outcome.metrics_path.parent / metric.details_path == incidents_path
        assert outcome.incidents_path == incidents_path
        payload = json.loads(incidents_path.read_text(encoding='utf-8'))
        assert payload['period_start'] == period.start.isoformat()
        assert payload['period_end'] == period.end.isoformat()
        assert payload['skipped'] == {reason.value: 0 for reason in _mod().SkipReason}
        assert len(payload['incidents']) == 1
        incident = payload['incidents'][0]
        assert set(incident) == {
            'write_row_id', 'search_row_ids', 'project_id', 'caller_agent_id',
            'matched_memory_id', 'similarity', 'threshold', 'category', 'write_created_at',
        }
        assert incident['write_row_id'] == rows['write_1']
        assert incident['search_row_ids'] == [rows['search_a']]
        assert incident['matched_memory_id'] == 'y'

    @pytest.mark.asyncio
    async def test_corpus_counts_disclose_every_narrowing_even_at_zero(self, tmp_path):
        _window, memory, _rows, journal = await _headline_fixture(tmp_path)

        outcome = await _run(memory, journal, tmp_path / 'evals')

        corpus = outcome.series.corpus
        assert corpus.project_id == PROJECT
        expected = {
            'writes_in_period': 2,
            'adjudicated': 2,
            'incidents': 1,
            'unattributed_searches': 0,
            'unparsed_rows': 0,
            'undated_candidates': 0,
            'uncovered_period_seconds': 0,
            **{f'skipped_{reason.value}': 0 for reason in _mod().SkipReason},
        }
        assert {key: corpus.counts.get(key) for key in expected} == expected

    @pytest.mark.asyncio
    async def test_hours_a_capped_period_left_uncovered_are_disclosed(self, tmp_path):
        _window, memory, _rows, journal = await _headline_fixture(tmp_path)
        covered = _period_around_now()
        uncovered_start = covered.start - timedelta(hours=30)
        period = _mod().Period(covered.start, covered.end, uncovered_start=uncovered_start)

        outcome = await _mod().run_write_after_miss(
            memory, journal_path=journal, out_root=tmp_path / 'evals', project_id=PROJECT,
            stamp=STAMP, period=period,
        )

        assert outcome.series.corpus.counts['uncovered_period_seconds'] == 30 * 3600
        uncovered_line = next(
            line for line in outcome.report.splitlines() if line.startswith('uncovered:')
        )
        assert uncovered_start.isoformat() in uncovered_line
        payload = json.loads(outcome.incidents_path.read_text(encoding='utf-8'))
        assert payload['uncovered_start'] == uncovered_start.isoformat()

    @pytest.mark.asyncio
    async def test_nothing_adjudicated_writes_a_series_with_no_metric(self, tmp_path):
        async with _fixture_journal(tmp_path) as fixture:
            await fixture.write(CALLER_A, content='W content', memory_ids=['w'])
        memory = FakeMemory(contents={'w': 'W content'})

        outcome = await _run(memory, fixture.db_path, tmp_path / 'evals')

        series = load_metric_series(outcome.metrics_path)
        assert series.metrics == []
        assert series.corpus.counts['skipped_no_prior_search'] == 1
        report = outcome.report_path.read_text(encoding='utf-8')
        assert 'write-after-miss: not measured' in report
        assert 'no_prior_search' in report

    @pytest.mark.asyncio
    async def test_the_report_names_each_incidents_write_and_match(self, tmp_path):
        _window, memory, rows, journal = await _headline_fixture(tmp_path)

        outcome = await _run(memory, journal, tmp_path / 'evals')

        report = outcome.report_path.read_text(encoding='utf-8')
        assert report == outcome.report
        incident_line = next(line for line in report.splitlines() if rows['write_1'] in line)
        assert 'y' in incident_line.replace(',', ' ').split()
        assert rows['write_2'] not in report

    @pytest.mark.asyncio
    async def test_a_run_never_writes_the_corpus(self, tmp_path):
        _window, memory, _rows, journal = await _headline_fixture(tmp_path)

        await _run(memory, journal, tmp_path / 'evals')

        assert memory.write_calls == []


EVAL_ID = _mod().EVAL_ID


def _empty_series(stamp: str) -> MetricSeries:
    return MetricSeries(
        schema_version=SCHEMA_VERSION, eval_id=EVAL_ID, run_stamp=stamp,
        corpus=Corpus(project_id=PROJECT), metrics=[],
    )


def _sweep_argv(root: Path) -> list[str]:
    """The runner's ``_sweep_argv`` shape for this producer."""
    return ['--metrics-root', str(root), '--eval-id', EVAL_ID, '--project-id', PROJECT]


def _stamp_an_hour_from_now() -> str:
    return format_stamp(datetime.now(UTC) + timedelta(hours=1))


class TestPeriodAndCli:
    """Periods tile on run stamps; the CLI takes the runner's argv and fails loudly."""

    def test_the_period_starts_at_the_previous_runs_stamp(self, tmp_path):
        previous, current = '20261108T220000Z', '20261109T220000Z'
        write_metric_series(_empty_series(previous), tmp_path)

        period = _mod().evaluation_period(tmp_path, EVAL_ID, current, initial_period_hours=24, max_period_hours=72)

        assert period == _mod().Period(stamp_datetime(previous), stamp_datetime(current))

    def test_a_period_past_the_cap_keeps_its_latest_hours_and_names_the_gap(self, tmp_path):
        previous, current = '20261104T220000Z', '20261109T220000Z'
        write_metric_series(_empty_series(previous), tmp_path)

        period = _mod().evaluation_period(
            tmp_path, EVAL_ID, current, initial_period_hours=24, max_period_hours=72,
        )

        end = stamp_datetime(current)
        assert period == _mod().Period(
            end - timedelta(hours=72), end, uncovered_start=stamp_datetime(previous),
        )

    def test_the_first_period_is_the_bootstrap_window(self, tmp_path):
        current = '20261109T220000Z'

        period = _mod().evaluation_period(tmp_path, EVAL_ID, current, initial_period_hours=6, max_period_hours=72)

        end = stamp_datetime(current)
        assert period == _mod().Period(end - timedelta(hours=6), end)

    def test_an_artifact_at_or_after_this_stamp_is_not_a_previous_run(self, tmp_path):
        current = '20261109T220000Z'
        write_metric_series(_empty_series(current), tmp_path)
        write_metric_series(_empty_series('20261110T220000Z'), tmp_path)

        period = _mod().evaluation_period(tmp_path, EVAL_ID, current, initial_period_hours=24, max_period_hours=72)

        end = stamp_datetime(current)
        assert period == _mod().Period(end - timedelta(hours=24), end)

    def test_the_parser_takes_the_runners_argv_and_the_producers_own_flags(self, tmp_path):
        parser = _mod().build_parser()

        args = parser.parse_args([
            *_sweep_argv(tmp_path),
            '--journal', str(tmp_path / 'write_journal.db'),
            '--initial-period-hours', '12',
            '--max-period-hours', '48',
            '--config', str(tmp_path / 'config.yaml'),
        ])

        assert args.metrics_root == str(tmp_path)
        assert args.eval_id == EVAL_ID
        assert args.project_id == PROJECT
        assert args.journal == str(tmp_path / 'write_journal.db')
        assert args.initial_period_hours == 12
        assert args.max_period_hours == 48
        assert args.config == str(tmp_path / 'config.yaml')
        defaults = parser.parse_args(_sweep_argv(tmp_path))
        assert (defaults.initial_period_hours, defaults.max_period_hours) == (24, 72)

    def test_a_missing_journal_exits_non_zero_and_writes_no_metrics(
        self, tmp_path, monkeypatch, capsys,
    ):
        stamp = _stamp_an_hour_from_now()
        monkeypatch.setenv(RUN_STAMP_ENV_VAR, stamp)
        opened: list[Any] = []

        async def open_fake_memory(config: Any) -> FakeMemory:
            opened.append(config)
            return FakeMemory(contents={})

        missing = tmp_path / 'absent' / 'write_journal.db'
        root = tmp_path / 'evals'

        exit_code = _mod().main(
            [*_sweep_argv(root), '--journal', str(missing)], open_memory=open_fake_memory,
        )

        assert exit_code != 0
        assert str(missing) in capsys.readouterr().err
        assert not (root / EVAL_ID / f'metrics-{stamp}.json').exists()
        assert opened == []

    def test_a_full_cli_run_writes_the_pinned_stamps_metrics(self, tmp_path, monkeypatch):
        _window, memory, _rows, journal = asyncio.run(_headline_fixture(tmp_path))
        stamp = _stamp_an_hour_from_now()
        monkeypatch.setenv(RUN_STAMP_ENV_VAR, stamp)

        async def open_fake_memory(config: Any) -> FakeMemory:
            return memory

        root = tmp_path / 'evals'

        exit_code = _mod().main(
            [*_sweep_argv(root), '--journal', str(journal)], open_memory=open_fake_memory,
        )

        assert exit_code == 0
        series = load_metric_series(root / EVAL_ID / f'metrics-{stamp}.json')
        assert [(metric.metric_id, metric.n) for metric in series.metrics] == [('write-after-miss', 2)]
        assert memory.write_calls == []
        assert memory.closed is True


class _InitRecordingService:
    """A ``MemoryService`` stand-in that records which lifecycle methods ran."""

    def __init__(self, config: Any, *, fail_with: Exception | None = None) -> None:
        self.config = config
        self.calls: list[str] = []
        self._fail_with = fail_with

    async def initialize(self) -> None:
        self.calls.append('initialize')

    async def initialize_for_reads(self) -> None:
        self.calls.append('initialize_for_reads')
        if self._fail_with is not None:
            raise self._fail_with

    async def close(self) -> None:
        self.calls.append('close')


class TestTheLiveFactoryOpensForReadsOnly:
    def test_it_never_starts_the_durable_queue_or_graphiti_maintenance(self):
        memory = asyncio.run(_mod().open_memory(SimpleNamespace(), service=_InitRecordingService))

        assert memory.calls == ['initialize_for_reads']

    def test_an_unopenable_store_closes_the_service_and_raises_a_named_error(self):
        opened: list[_InitRecordingService] = []

        def failing_service(config: Any) -> _InitRecordingService:
            service = _InitRecordingService(config, fail_with=sqlite3.OperationalError('locked'))
            opened.append(service)
            return service

        with pytest.raises(_mod().MemoryStoreUnavailableError, match='OperationalError'):
            asyncio.run(_mod().open_memory(SimpleNamespace(), service=failing_service))

        assert opened[0].calls == ['initialize_for_reads', 'close']
