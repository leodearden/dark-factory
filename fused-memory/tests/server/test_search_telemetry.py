"""Integration tests for search journal telemetry (task 3212, items 1/2/5).

The user-observable signal: a search issued through the MCP tool yields a
`write_ops` row carrying result ids + relevance scores + per-result content
sizes, the FULL query text, and — when the caller passes them — the
caller-identity params.  Drives the REAL tool through
`server._tool_manager.call_tool` against a REAL `WriteJournal`, so an
undeclared kwarg fails loudly instead of being absorbed by a patch.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from fused_memory.models.enums import SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.server.tools import create_mcp_server
from fused_memory.services.memory_service import SearchResults
from fused_memory.services.write_journal import WriteJournal

_PROJECT_ID = 'dark_factory'

# Deliberately > 200 characters: the search journal used to truncate the query
# at 200, which throws away the second half of every real briefing query.
_LONG_QUERY = (
    'what did the architect decide about the write journal retention horizons '
    'for search rows versus task-read rows, and why does the prune need to be '
    'batched rather than a single unbounded DELETE at startup on a table that '
    'has never once been pruned in 157 days ENDMARKER'
)
assert len(_LONG_QUERY) > 200, 'fixture must exceed the old 200-char truncation'


@pytest_asyncio.fixture
async def write_journal(tmp_path):
    journal = WriteJournal(tmp_path / 'wj_search_telemetry')
    await journal.initialize()
    yield journal
    await journal.close()


def _make_result(
    result_id: str,
    content: str,
    relevance_score: float,
    store_rank: int,
    store_score: float | None,
    source: SourceStore = SourceStore.mem0,
) -> MemoryResult:
    return MemoryResult(
        id=result_id,
        content=content,
        source_store=source,
        relevance_score=relevance_score,
        metadata={'store_rank': store_rank, 'store_score': store_score},
    )


_RESULT_A_ID = '11111111-1111-4111-8111-111111111111'
_RESULT_B_ID = '22222222-2222-4222-8222-222222222222'
_CONTENT_A = 'alpha body ' * 4      # 44 chars
_CONTENT_B = 'beta body ' * 3       # 30 chars


def _two_results() -> list[MemoryResult]:
    return [
        _make_result(_RESULT_A_ID, _CONTENT_A, 0.91, 1, 0.62),
        _make_result(_RESULT_B_ID, _CONTENT_B, 0.33, 2, None, SourceStore.graphiti),
    ]


def _make_server(write_journal, *, results=None, raises: Exception | None = None):
    """create_mcp_server over a mocked service and a REAL journal."""
    mock_service = AsyncMock()
    if raises is not None:
        mock_service.search = AsyncMock(side_effect=raises)
    else:
        mock_service.search = AsyncMock(
            return_value=results if results is not None else SearchResults(_two_results())
        )
    # Grouping (task 3129) runs at this boundary and asks the service how many
    # children a hit has; 0 keeps the grouped payload equal to the base entries
    # so these tests isolate TELEMETRY rather than re-testing grouped_read.
    mock_service.count_memories_by_metadata = AsyncMock(return_value=0)
    return mock_service, create_mcp_server(mock_service, None, write_journal)


async def _search_rows(journal: WriteJournal) -> list[dict]:
    db = journal._db
    assert db is not None
    async with db.execute(
        "SELECT * FROM write_ops WHERE operation = 'search' ORDER BY created_at"
    ) as cursor:
        rows = [dict(r) for r in await cursor.fetchall()]
    for row in rows:
        row['params'] = json.loads(row['params'] or '{}')
        row['result_summary'] = json.loads(row['result_summary'] or '{}')
    return rows


async def _one_search_row(journal: WriteJournal) -> dict:
    rows = await _search_rows(journal)
    assert len(rows) == 1, f'Expected exactly one journalled search row, got {rows!r}'
    return rows[0]


class TestSearchRowCarriesResultDetail:
    """Item (1): result ids + scores + sizes, not a bare count."""

    @pytest.mark.asyncio
    async def test_result_summary_carries_ids_scores_and_sizes(self, write_journal):
        _, server = _make_server(write_journal)

        await server._tool_manager.call_tool(
            'search', {'query': _LONG_QUERY, 'project_id': _PROJECT_ID}
        )

        row = await _one_search_row(write_journal)
        summary = row['result_summary']

        assert summary.get('count') == 2, (
            f'count must survive the widening, got {summary!r}. RED: row not widened.'
        )
        entries = summary.get('results') or []
        assert [e['id'] for e in entries] == [_RESULT_A_ID, _RESULT_B_ID], (
            'Both result IDs must reach the journal — leaf eta (3213) cannot ask '
            f'"was the agent shown this?" without them. got {summary!r}. '
            "RED: result_summary is still {'count': N}."
        )
        assert [e['relevance_score'] for e in entries] == [0.91, 0.33], (
            f'Both relevance SCORES must reach the journal, got {summary!r}. RED: not logged.'
        )
        assert [e['content_size'] for e in entries] == [len(_CONTENT_A), len(_CONTENT_B)], (
            f'Per-result content SIZES must reach the journal, got {summary!r}. RED: not logged.'
        )

    @pytest.mark.asyncio
    async def test_result_summary_names_its_size_unit(self, write_journal):
        _, server = _make_server(write_journal)

        await server._tool_manager.call_tool(
            'search', {'query': _LONG_QUERY, 'project_id': _PROJECT_ID}
        )

        summary = (await _one_search_row(write_journal))['result_summary']
        assert summary.get('size_unit') == 'chars', (
            'The size unit must be NAMED in the row, not implied by a field name, '
            f'got {summary!r}. RED: unit absent.'
        )

    @pytest.mark.asyncio
    async def test_result_summary_carries_store_rank_and_score(self, write_journal):
        """Task 3658 put per-store truth in metadata; the journal must carry it."""
        _, server = _make_server(write_journal)

        await server._tool_manager.call_tool(
            'search', {'query': _LONG_QUERY, 'project_id': _PROJECT_ID}
        )

        entries = (await _one_search_row(write_journal))['result_summary']['results']
        assert [e['store_rank'] for e in entries] == [1, 2], (
            f'store_rank must be lifted out of metadata into the row, got {entries!r}. '
            'RED: 3658 fields not logged.'
        )
        assert [e['store_score'] for e in entries] == [0.62, None], (
            "Graphiti's store_score is None by contract and must survive the round trip "
            f'verbatim, got {entries!r}. RED: dropped or coerced.'
        )

    @pytest.mark.asyncio
    async def test_kind_and_operation_are_unchanged(self, write_journal):
        """The widening must not disturb the columns existing readers filter on."""
        _, server = _make_server(write_journal)

        await server._tool_manager.call_tool(
            'search', {'query': _LONG_QUERY, 'project_id': _PROJECT_ID}
        )

        row = await _one_search_row(write_journal)
        assert row['kind'] == 'read', f'RED: kind changed, got {row["kind"]!r}'
        assert row['operation'] == 'search', f'RED: operation changed, got {row["operation"]!r}'


class TestFullQueryText:
    """Item (1): drop the 200-char truncation for SEARCH rows."""

    @pytest.mark.asyncio
    async def test_query_is_journalled_untruncated(self, write_journal):
        _, server = _make_server(write_journal)

        await server._tool_manager.call_tool(
            'search', {'query': _LONG_QUERY, 'project_id': _PROJECT_ID}
        )

        params = (await _one_search_row(write_journal))['params']
        assert params.get('query') == _LONG_QUERY, (
            'The FULL query text must be journalled — a retrieval-quality metric '
            'computed from half a query measures the wrong thing. '
            f'got {params.get("query")!r}. RED: still truncated.'
        )
        assert params['query'] != _LONG_QUERY[:200], (
            f'RED: query was cut at exactly 200 chars ({len(params["query"])} logged).'
        )
        assert params['query'].endswith('ENDMARKER'), (
            f'RED: the query tail was discarded, got ...{params["query"][-40:]!r}'
        )

    @pytest.mark.asyncio
    async def test_limit_is_still_journalled(self, write_journal):
        _, server = _make_server(write_journal)

        await server._tool_manager.call_tool(
            'search', {'query': _LONG_QUERY, 'project_id': _PROJECT_ID, 'limit': 7}
        )

        params = (await _one_search_row(write_journal))['params']
        assert params.get('limit') == 7, (
            f'limit was already recorded and must not be lost, got {params!r}.'
        )


class TestErrorPath:
    """A failed search is still a read that happened — and still attributable."""

    @pytest.mark.asyncio
    async def test_error_row_carries_the_full_query(self, write_journal):
        _, server = _make_server(write_journal, raises=RuntimeError('mem0 exploded'))

        await server._tool_manager.call_tool(
            'search', {'query': _LONG_QUERY, 'project_id': _PROJECT_ID}
        )

        row = await _one_search_row(write_journal)
        assert row['success'] == 0, f'RED: the failed search was journalled as success: {row!r}'
        assert row['params'].get('query') == _LONG_QUERY, (
            'The error path truncated the query at 200 too — a failed search is '
            f'exactly when the full query matters. got {row["params"]!r}. RED: still truncated.'
        )


class TestDegradedFactsSurvive:
    """The widening must not DROP what the row already recorded."""

    @pytest.mark.asyncio
    async def test_failed_stores_survives_alongside_the_new_keys(self, write_journal):
        _, server = _make_server(
            write_journal,
            results=SearchResults([], degraded=True, failed_stores=['mem0']),
        )

        await server._tool_manager.call_tool(
            'search', {'query': _LONG_QUERY, 'project_id': _PROJECT_ID}
        )

        summary = (await _one_search_row(write_journal))['result_summary']
        assert summary.get('failed_stores') == ['mem0'], (
            'failed_stores was already recorded on the degraded path and must survive '
            f'the widening, got {summary!r}. RED: pre-existing fact dropped.'
        )
        assert summary.get('count') == 0, f'RED: got {summary!r}'
        assert summary.get('size_unit') == 'chars', (
            f'The widened keys must be present on the degraded path too, got {summary!r}.'
        )


class TestGroupedPayloadIsWhatIsSummarised:
    """Leaf eta asks what the agent was SHOWN, and grouping happens at this boundary.

    Pinned at the SEAM rather than through grouped_read's internals: this test
    owns "the tool summarises the grouped list", not "grouping folds correctly"
    (which is tests/server/test_grouped_read.py's job).
    """

    @pytest.mark.asyncio
    async def test_journal_summarises_the_grouped_list_not_the_raw_one(
        self, write_journal, monkeypatch
    ):
        import fused_memory.server.tools as tools_mod

        parent_id = '44444444-4444-4444-8444-444444444444'
        folded_child_id = '55555555-5555-4555-8555-555555555555'

        async def _fake_group(_service, _project_id, _results):
            return [
                {
                    'id': parent_id,
                    'content': 'canonical body',
                    'source_store': 'mem0',
                    'relevance_score': 0.7,
                    'metadata': {},
                    'topic_anchored': False,
                    'grouped': {
                        'matched_children': [
                            {'id': folded_child_id, 'content': 'amendment body',
                             'kind': 'amendment', 'matched': True}
                        ],
                        'amendment_count': 1,
                        'sighting_count': 0,
                    },
                }
            ]

        monkeypatch.setattr(tools_mod, 'group_search_results', _fake_group)
        _, server = _make_server(write_journal)

        await server._tool_manager.call_tool(
            'search', {'query': _LONG_QUERY, 'project_id': _PROJECT_ID}
        )

        summary = (await _one_search_row(write_journal))['result_summary']
        assert [e['id'] for e in summary['results']] == [parent_id], (
            'The row must summarise the GROUPED payload the agent was shown. The raw '
            f'service list held two DIFFERENT ids, so this is decisive. got {summary!r}. '
            'RED: the raw pre-grouping list was summarised.'
        )
        assert folded_child_id in (summary['results'][0].get('folded_child_ids') or []), (
            'A child folded INTO the parent was shown to the agent as part of it, so '
            f'its id must reach the journal, got {summary!r}. RED: folded id lost.'
        )


class TestTelemetryFaultCannotBreakSearch:
    """A telemetry fault must never turn a working search into an error."""

    @pytest.mark.asyncio
    async def test_summariser_fault_degrades_to_a_count(self, write_journal, monkeypatch):
        import fused_memory.server.tools as tools_mod

        def _boom(*_args, **_kwargs):
            raise ValueError('summariser exploded')

        monkeypatch.setattr(tools_mod, 'summarize_search_results', _boom)
        _, server = _make_server(write_journal)

        result = await server._tool_manager.call_tool(
            'search', {'query': _LONG_QUERY, 'project_id': _PROJECT_ID}
        )

        assert 'error' not in result, (
            f'A telemetry fault must not fail the search, got {result!r}. '
            'RED: summarise call is unguarded.'
        )
        assert len(result['results']) == 2, f'RED: results lost, got {result!r}'
        summary = (await _one_search_row(write_journal))['result_summary']
        assert summary.get('count') == 2, (
            f'The fallback must still record the count, got {summary!r}. RED: row lost entirely.'
        )
