"""Tests for the token-budget context assembler."""

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.memory import MemoryCategory, MemoryResult, SourceStore
from fused_memory.models.reconciliation import (
    ContextItem,
    EventSource,
    EventType,
    ReconciliationEvent,
    Watermark,
)
from fused_memory.reconciliation.context_assembler import (
    ContextAssembler,
    _format_task,  # Intentional private-API coupling: targeted unit coverage of
    #                the display_id parameter that the integration tests alone
    #                cannot isolate as cleanly.
    estimate_tokens,
    format_event,
)


def _make_event(
    event_type: EventType = EventType.memory_added,
    project_id: str = 'test-project',
    payload: dict | None = None,
    timestamp: datetime | None = None,
) -> ReconciliationEvent:
    return ReconciliationEvent(
        id=str(uuid.uuid4()),
        type=event_type,
        source=EventSource.agent,
        project_id=project_id,
        timestamp=timestamp or datetime.now(UTC),
        payload=payload or {},
    )


def _make_memory_result(
    content: str = 'test memory',
    source: SourceStore = SourceStore.mem0,
    category: MemoryCategory = MemoryCategory.observations_and_summaries,
) -> MemoryResult:
    return MemoryResult(
        id=str(uuid.uuid4()),
        content=content,
        source_store=source,
        category=category,
        relevance_score=0.8,
    )


def _make_watermark(
    last_run: datetime | None = None,
) -> Watermark:
    return Watermark(
        project_id='test-project',
        last_full_run_completed=last_run or datetime.now(UTC) - timedelta(hours=1),
    )


def _make_config(**overrides: int) -> ReconciliationConfig:
    return ReconciliationConfig(
        token_budget=overrides.get('token_budget', 65_000),
        context_search_limit=overrides.get('context_search_limit', 5),
        context_fetch_batch_size=overrides.get('context_fetch_batch_size', 10),
    )


@pytest_asyncio.fixture
async def mock_memory():
    svc = AsyncMock()
    svc.search = AsyncMock(return_value=[])
    return svc


@pytest_asyncio.fixture
async def mock_taskmaster():
    tm = AsyncMock()
    tm.get_task = AsyncMock(return_value={})
    return tm


def _make_assembler(
    memory_service=None,
    taskmaster=None,
    config=None,
    project_root='/test',
):
    return ContextAssembler(
        memory_service=memory_service or AsyncMock(search=AsyncMock(return_value=[])),
        taskmaster=taskmaster,
        config=config or _make_config(),
        project_root=project_root,
    )


# ── estimate_tokens / format_event ─────────────────────────────────


def test_estimate_tokens():
    assert estimate_tokens('') == 0
    assert estimate_tokens('a' * 100) == 25
    assert estimate_tokens('a' * 4) == 1


def test_format_event():
    event = _make_event(payload={'task_id': '42'})
    formatted = format_event(event)
    assert '[memory_added]' in formatted
    assert '"task_id": "42"' in formatted


# ── ContextItem ─────────────────────────────────────────────────────


def test_context_item_auto_token_estimate():
    item = ContextItem(id='x', source='mem0', formatted='a' * 100)
    assert item.token_estimate == 25


def test_context_item_explicit_token_estimate():
    item = ContextItem(id='x', source='mem0', formatted='a' * 100, token_estimate=50)
    assert item.token_estimate == 50


# ── assemble() — budget control ─────────────────────────────────────


@pytest.mark.asyncio
async def test_budget_stops_accumulation(mock_memory):
    """Assembler stops adding events when budget is reached."""
    config = _make_config(token_budget=1_450)  # overhead (1400) + room for ~1 event (~50 tokens)
    assembler = _make_assembler(memory_service=mock_memory, config=config)

    events = [_make_event(payload={'content_preview': f'content {i}'}) for i in range(10)]
    watermark = _make_watermark()

    result = await assembler.assemble(events, watermark, 'test-project')

    assert len(result.events) < len(events)
    assert result.events_remaining > 0
    assert result.total_tokens <= 1_600


@pytest.mark.asyncio
async def test_all_events_fit_when_under_budget(mock_memory):
    """When total tokens < budget, all events are included."""
    config = _make_config(token_budget=65_000)
    assembler = _make_assembler(memory_service=mock_memory, config=config)

    events = [_make_event() for _ in range(5)]
    watermark = _make_watermark()

    result = await assembler.assemble(events, watermark, 'test-project')

    assert len(result.events) == 5
    assert result.events_remaining == 0


@pytest.mark.asyncio
async def test_empty_events():
    """Empty event list returns empty payload."""
    assembler = _make_assembler()
    watermark = _make_watermark()

    result = await assembler.assemble([], watermark, 'test-project')

    assert len(result.events) == 0
    assert len(result.context_items) == 0
    assert result.events_remaining == 0


# ── Context deduplication ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_context_deduplication(mock_memory):
    """Same context item from multiple events is counted once."""
    shared_result = _make_memory_result(content='shared memory')

    # Both events return the same memory result
    mock_memory.search = AsyncMock(return_value=[shared_result])

    assembler = _make_assembler(memory_service=mock_memory)
    events = [
        _make_event(payload={'content_preview': 'event 1'}),
        _make_event(payload={'content_preview': 'event 2'}),
    ]
    watermark = _make_watermark()

    result = await assembler.assemble(events, watermark, 'test-project')

    assert len(result.events) == 2
    # The shared result should appear only once in context_items
    assert len(result.context_items) == 1
    assert shared_result.id in result.context_items


# ── Per-event-type dispatch ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_memory_added_searches_by_preview(mock_memory):
    """memory_added events trigger search with content_preview."""
    mock_memory.search = AsyncMock(return_value=[_make_memory_result()])

    assembler = _make_assembler(memory_service=mock_memory)
    events = [
        _make_event(
            event_type=EventType.memory_added,
            payload={
                'content_preview': 'decision about auth',
                'category': 'decisions_and_rationale',
            },
        ),
    ]
    watermark = _make_watermark()

    await assembler.assemble(events, watermark, 'test-project')

    mock_memory.search.assert_called_once()
    call_kwargs = mock_memory.search.call_args
    assert (
        call_kwargs.kwargs.get('query') == 'decision about auth'
        or call_kwargs[1].get('query') == 'decision about auth'
    )


@pytest.mark.asyncio
async def test_task_status_changed_fetches_task(mock_memory, mock_taskmaster):
    """task_status_changed events fetch the task and search memory hints."""
    mock_taskmaster.get_task = AsyncMock(
        return_value={
            'id': '42',
            'title': 'Implement auth',
            'status': 'done',
            'dependencies': [],
            'metadata': {
                'memory_hints': {
                    'queries': ['auth implementation decisions'],
                    'entities': ['AuthService'],
                },
            },
        }
    )
    mock_memory.search = AsyncMock(return_value=[_make_memory_result()])

    assembler = _make_assembler(
        memory_service=mock_memory,
        taskmaster=mock_taskmaster,
    )
    events = [
        _make_event(
            event_type=EventType.task_status_changed,
            payload={'task_id': '42', 'old_status': 'in-progress', 'new_status': 'done'},
        ),
    ]
    watermark = _make_watermark()

    result = await assembler.assemble(events, watermark, 'test-project')

    mock_taskmaster.get_task.assert_called_once()
    # Should have task context + memory hint search results
    assert 'task:42' in result.context_items
    # Search called for the hint query
    assert mock_memory.search.call_count >= 1


@pytest.mark.asyncio
async def test_memory_deleted_searches_references(mock_memory):
    """memory_deleted events search for references to the deleted memory."""
    mock_memory.search = AsyncMock(return_value=[])

    assembler = _make_assembler(memory_service=mock_memory)
    events = [
        _make_event(
            event_type=EventType.memory_deleted,
            payload={'memory_id': 'mem-123', 'store': 'mem0'},
        ),
    ]
    watermark = _make_watermark()

    await assembler.assemble(events, watermark, 'test-project')

    mock_memory.search.assert_called_once()
    call_args = mock_memory.search.call_args
    query = call_args.kwargs.get('query') or call_args[1].get('query') or call_args[0][0]
    assert 'mem-123' in query


@pytest.mark.asyncio
async def test_episode_added_searches_by_preview(mock_memory):
    """episode_added events search for related entities."""
    mock_memory.search = AsyncMock(return_value=[])

    assembler = _make_assembler(memory_service=mock_memory)
    events = [
        _make_event(
            event_type=EventType.episode_added,
            payload={'content_preview': 'refactored the auth module'},
        ),
    ]
    watermark = _make_watermark()

    await assembler.assemble(events, watermark, 'test-project')

    mock_memory.search.assert_called_once()


# ── Context fetch failure ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_context_fetch_failure_includes_event(mock_memory):
    """Failed context fetch still includes the event (degrades to event-only)."""
    mock_memory.search = AsyncMock(side_effect=RuntimeError('search failed'))

    assembler = _make_assembler(memory_service=mock_memory)
    events = [
        _make_event(payload={'content_preview': 'something'}),
    ]
    watermark = _make_watermark()

    result = await assembler.assemble(events, watermark, 'test-project')

    assert len(result.events) == 1
    assert len(result.context_items) == 0


# ── Effective watermark ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_effective_watermark_uses_min_of_run_and_event():
    """Watermark aligns with the earliest event if it predates last run."""
    assembler = _make_assembler()

    last_run = datetime(2026, 4, 2, 10, 0, tzinfo=UTC)
    old_event_time = datetime(2026, 4, 1, 8, 0, tzinfo=UTC)

    events = [_make_event(timestamp=old_event_time)]
    watermark = _make_watermark(last_run=last_run)

    result = await assembler.assemble(events, watermark, 'test-project')

    assert result.effective_watermark == old_event_time


@pytest.mark.asyncio
async def test_effective_watermark_uses_last_run_when_events_are_recent():
    """Watermark uses last_run when all events are more recent."""
    assembler = _make_assembler()

    last_run = datetime(2026, 4, 1, 8, 0, tzinfo=UTC)
    recent_event = datetime(2026, 4, 2, 10, 0, tzinfo=UTC)

    events = [_make_event(timestamp=recent_event)]
    watermark = _make_watermark(last_run=last_run)

    result = await assembler.assemble(events, watermark, 'test-project')

    assert result.effective_watermark == last_run


# ── Batch processing ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_batch_processing(mock_memory):
    """Events are processed in batches of context_fetch_batch_size."""
    call_count = 0

    async def counting_search(**kwargs):
        nonlocal call_count
        call_count += 1
        return []

    mock_memory.search = counting_search

    config = _make_config(context_fetch_batch_size=3)
    assembler = _make_assembler(memory_service=mock_memory, config=config)

    events = [_make_event(payload={'content_preview': f'event {i}'}) for i in range(7)]
    watermark = _make_watermark()

    result = await assembler.assemble(events, watermark, 'test-project')

    assert len(result.events) == 7
    assert call_count == 7  # one search per event


@pytest.mark.asyncio
async def test_task_event_without_taskmaster(mock_memory):
    """Task events with no taskmaster client return no context."""
    assembler = _make_assembler(memory_service=mock_memory, taskmaster=None)

    events = [
        _make_event(
            event_type=EventType.task_status_changed,
            payload={'task_id': '42', 'old_status': 'pending', 'new_status': 'done'},
        ),
    ]
    watermark = _make_watermark()

    result = await assembler.assemble(events, watermark, 'test-project')

    assert len(result.events) == 1
    assert len(result.context_items) == 0


@pytest.mark.asyncio
async def test_memory_added_without_preview(mock_memory):
    """memory_added with no content_preview returns no context."""
    assembler = _make_assembler(memory_service=mock_memory)

    events = [
        _make_event(
            event_type=EventType.memory_added,
            payload={'category': 'observations_and_summaries'},
        ),
    ]
    watermark = _make_watermark()

    result = await assembler.assemble(events, watermark, 'test-project')

    assert len(result.events) == 1
    assert len(result.context_items) == 0
    mock_memory.search.assert_not_called()


@pytest.mark.asyncio
async def test_first_event_always_included_even_if_over_budget(mock_memory):
    """At least one event is included even if it exceeds the budget."""
    # Very small context items so the event itself is what matters
    big_results = [_make_memory_result(content='x' * 2000) for _ in range(5)]
    mock_memory.search = AsyncMock(return_value=big_results)

    # Budget barely above fixed overhead
    config = _make_config(token_budget=1_500)
    assembler = _make_assembler(memory_service=mock_memory, config=config)

    events = [
        _make_event(payload={'content_preview': 'something'}),
        _make_event(payload={'content_preview': 'something else'}),
    ]
    watermark = _make_watermark()

    result = await assembler.assemble(events, watermark, 'test-project')

    # First event should be included even though it blows the budget
    assert len(result.events) >= 1
    # Second event should not be (budget already exceeded)
    assert result.events_remaining >= 1


# ---------------------------------------------------------------------------
# task-512: CancelledError must propagate out of assemble(), not be swallowed
# ---------------------------------------------------------------------------


class TestContextAssemblerCancellation:
    """CancelledError raised inside gather must propagate, not be silently captured.

    asyncio.gather(return_exceptions=True) captures *all* BaseException subclasses as
    result values — including asyncio.CancelledError, which is a BaseException but NOT
    an Exception in Python 3.8+.  The buggy guard ``isinstance(ctx_result, BaseException)``
    therefore treats cancellation signals as ordinary per-event failures and converts them
    into empty context lists (``ctx_result = []``), silently swallowing the shutdown signal
    and letting assemble() return a normal payload instead of propagating the cancel.

    The fix uses the shared 'two-tier check' via propagate_cancellations
    (fused_memory.utils.async_utils), the same Pass 1 guard used by all
    gather(return_exceptions=True) callsites in this codebase.

    Two passes:
      - Pass 1 (propagate_cancellations): re-raise any value that is a BaseException
        but NOT an Exception (CancelledError, KeyboardInterrupt, SystemExit).
      - Pass 2 (accumulation): the per-event zip loop uses ``isinstance(ctx_result, Exception)``
        so only application-level failures degrade to empty context.
    """

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates_from_context_fetch(self, mock_memory):
        """CancelledError raised by memory.search must propagate out of assemble().

        The production path:
          mock_memory.search raises CancelledError
          → _ctx_memory_added calls self.memory.search → CancelledError propagates
          → _fetch_context's ``except Exception`` does NOT catch BaseException subclasses
            that are not Exception (i.e., CancelledError bypasses the except-clause)
          → asyncio.gather(return_exceptions=True) captures it as a value in batch_contexts
          → the buggy ``isinstance(ctx_result, BaseException)`` guard converts it to []
            (current behaviour: assemble() returns normally — test DID NOT RAISE)
          → the fixed propagation pass detects it and re-raises → pytest.raises passes.
        """
        mock_memory.search = AsyncMock(side_effect=asyncio.CancelledError())
        assembler = _make_assembler(memory_service=mock_memory)

        events = [
            _make_event(
                event_type=EventType.memory_added,
                payload={'content_preview': 'some content'},
            ),
        ]
        watermark = _make_watermark()

        with pytest.raises(asyncio.CancelledError):
            await assembler.assemble(events, watermark, 'test-project')

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates_alongside_exception(self, mock_memory):
        """CancelledError must take precedence over per-event RuntimeErrors in the same batch.

        When gather(return_exceptions=True) returns a mix of Exception subclasses and
        CancelledError, the propagation pass (which runs before per-event accumulation)
        must still re-raise the CancelledError even if a RuntimeError appears first in
        the results list. This guards against a regression where someone reorders the
        guard branches and accidentally promotes RuntimeError accounting before the
        cancellation check.

        Patches assembler._fetch_context via patch.object to emit a RuntimeError for
        event_one and a CancelledError for event_two — bypassing _fetch_context's
        internal ``except Exception`` wrapper and forcing the exact mix into
        batch_contexts (in deterministic slot order, since gather preserves input-order
        in its result list regardless of scheduling) that the propagation pass must
        handle correctly.

        Sister test: TestRebuildEntitySummariesCancellation::
            test_cancelled_error_propagates_alongside_other_errors
        """
        call_count = 0

        async def patched_fetch_context(event, project_id):
            nonlocal call_count
            call_count += 1
            if event.payload['content_preview'] == 'event one':
                raise RuntimeError('per-event failure')
            raise asyncio.CancelledError()

        assembler = _make_assembler(memory_service=mock_memory)

        with patch.object(assembler, '_fetch_context', new=patched_fetch_context):
            events = [
                _make_event(
                    event_type=EventType.memory_added,
                    payload={'content_preview': 'event one'},
                ),
                _make_event(
                    event_type=EventType.memory_added,
                    payload={'content_preview': 'event two'},
                ),
            ]
            watermark = _make_watermark()

            with pytest.raises(asyncio.CancelledError):
                await assembler.assemble(events, watermark, 'test-project')
            assert call_count == 2, (
                'mixed-batch scenario requires both tasks to dispatch — '
                'the RuntimeError/CancelledError ordering premise was not exercised'
            )

    @pytest.mark.asyncio
    async def test_exception_only_batch_does_not_propagate(self, mock_memory):
        """Plain RuntimeErrors must be caught by Pass 2, not re-raised by Pass 1.

        Task 575 — guards the ``and not isinstance(ctx_result, Exception)`` clause
        in the Pass 1 propagation check (context_assembler.py:134).  If that clause
        were dropped (or the check widened to bare ``isinstance(ctx_result,
        BaseException)``), a RuntimeError would be re-raised by Pass 1 instead of
        being swallowed by Pass 2 and degraded to an event-only entry.

        Directly patches assembler._fetch_context — bypassing _fetch_context's own
        ``except Exception`` wrapper (lines 221-225) — so the RuntimeError lands in
        batch_contexts exactly as the Pass 1 guard sees it.  assemble() must return
        normally with all events present in degraded event-only mode and no context
        items.
        """
        call_count = 0

        async def patched_fetch_context(event, project_id):
            nonlocal call_count
            call_count += 1
            raise RuntimeError('per-event failure')

        assembler = _make_assembler(memory_service=mock_memory)

        events = [
            _make_event(
                event_type=EventType.memory_added,
                payload={'content_preview': f'event {i}'},
            )
            for i in range(3)
        ]
        watermark = _make_watermark()

        with patch.object(assembler, '_fetch_context', new=patched_fetch_context):
            result = await assembler.assemble(events, watermark, 'test-project')

        assert len(result.events) == 3, 'all events must appear in degraded event-only payload'
        assert len(result.context_items) == 0, 'no context items expected when every fetch fails'
        assert call_count == 3, 'patched _fetch_context must be called once per event'

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates_from_second_batch(self, mock_memory):
        """CancelledError raised on the second batch iteration must still propagate.

        Task 575 — guards the while-loop structure in assemble()
        (context_assembler.py:113-166).  The Pass 1 propagation check runs inside
        the loop body on every iteration; this test forces two loop iterations and
        verifies that Pass 1 fires on the second one.  A regression such as an
        early-continue that skips the propagation check on subsequent iterations
        would cause assemble() to return normally instead of raising.

        Uses context_fetch_batch_size=10 with 12 events so that the first batch
        (calls 1-10) succeeds and the CancelledError is raised on call 11 — the
        first call of the second batch.

        Sister test: test_cancelled_error_propagates_from_context_fetch (single-batch
        variant) and test_cancelled_error_propagates_alongside_exception (mixed-batch
        variant in the first batch).
        """
        call_count = 0

        async def patched_fetch_context(event, project_id):
            nonlocal call_count
            call_count += 1
            if call_count >= 11:
                raise asyncio.CancelledError()
            return []

        config = _make_config(context_fetch_batch_size=10)
        assembler = _make_assembler(memory_service=mock_memory, config=config)

        events = [
            _make_event(
                event_type=EventType.memory_added,
                payload={'content_preview': f'event {i}'},
            )
            for i in range(12)
        ]
        watermark = _make_watermark()

        with (
            patch.object(assembler, '_fetch_context', new=patched_fetch_context),
            pytest.raises(asyncio.CancelledError),
        ):
            await assembler.assemble(events, watermark, 'test-project')

        assert call_count >= 11, (
            'first batch (calls 1-10) must complete before the second batch '
            'dispatches and triggers the CancelledError on call 11'
        )

    @pytest.mark.asyncio
    async def test_pass2_guard_narrowed_to_exception(self, mock_memory):
        """Pass 2 guard must use isinstance(ctx_result, Exception), not BaseException.

        Defense-in-depth test: injects a KeyboardInterrupt() instance directly into
        batch_contexts by patching asyncio.gather to return it as a value (avoiding the
        asyncio task re-raise that would occur if a coroutine actually raised it).
        Pass 1 is bypassed by patching propagate_cancellations to a no-op, so Pass 2
        sees the bare BaseException value directly.

        With the current ``isinstance(ctx_result, BaseException)`` guard,
        KeyboardInterrupt() IS caught and degraded to [] — assemble() returns normally.
        With the narrowed ``isinstance(ctx_result, Exception)`` guard, KeyboardInterrupt()
        is NOT caught and falls through to the list comprehension
        ``[item for item in ctx_result ...]``, causing
        ``TypeError: 'KeyboardInterrupt' object is not iterable``.

        This test would FAIL with the old ``isinstance(ctx_result, BaseException)`` guard
        (assemble() would return normally) and PASSES with the narrowed
        ``isinstance(ctx_result, Exception)`` guard.

        Sister test: test_exception_only_batch_does_not_propagate — verifies that real
        Exception subclasses ARE caught by Pass 2 and degraded gracefully.
        """
        assembler = _make_assembler(memory_service=mock_memory)
        events = [
            _make_event(
                event_type=EventType.memory_added,
                payload={'content_preview': 'some content'},
            ),
        ]
        watermark = _make_watermark()

        async def _mock_gather(*args, **kwargs):
            # Return KeyboardInterrupt() as a value — bypasses asyncio task re-raise
            return [KeyboardInterrupt()]

        with (
            patch('fused_memory.reconciliation.context_assembler.asyncio.gather', new=_mock_gather),
            patch(
                'fused_memory.reconciliation.context_assembler.propagate_cancellations',
                new=lambda results: None,
            ),
            pytest.raises(TypeError),
        ):
            await assembler.assemble(events, watermark, 'test-project')


# ── _format_task display_id ──────────────────────────────────────────


def test_format_task_display_id_overrides_bare_int_id():
    """When display_id='450.2' is passed, output uses qualified ID not bare int."""
    task = {
        'id': 2,
        'title': 'Fix subtask',
        'status': 'in-progress',
        'dependencies': [],
    }
    result = _format_task(task, display_id='450.2')
    assert '[task:450.2]' in result
    assert '[task:2]' not in result


def test_format_task_no_display_id_uses_task_id_field():
    """Without display_id, output uses task.get('id') — backward-compatible."""
    task = {
        'id': '42',
        'title': 'Top-level task',
        'status': 'done',
        'dependencies': [1, 3],
    }
    result = _format_task(task)
    assert '[task:42]' in result


# ── _ctx_task_event subtask qualification ────────────────────────────


@pytest.mark.asyncio
async def test_ctx_task_event_subtask_uses_qualified_id(mock_memory, mock_taskmaster):
    """_ctx_task_event for subtask event uses qualified task_id as display ID.

    When the event payload carries task_id='450.2' and get_task returns a dict
    with bare id=2, the ContextItem's formatted text must contain '[task:450.2]'
    — not '[task:2]' — so Stage 1 can cross-reference with the Active Task Tree.
    """
    mock_taskmaster.get_task = AsyncMock(
        return_value={
            'id': 2,
            'title': 'Subtask fix',
            'status': 'in-progress',
            'dependencies': [],
            'metadata': {},
        }
    )

    assembler = _make_assembler(
        memory_service=mock_memory,
        taskmaster=mock_taskmaster,
    )
    events = [
        _make_event(
            event_type=EventType.task_status_changed,
            payload={'task_id': '450.2', 'old_status': 'pending', 'new_status': 'in-progress'},
        ),
    ]
    watermark = _make_watermark()

    result = await assembler.assemble(events, watermark, 'test-project')

    # ContextItem key must be the qualified ID
    assert 'task:450.2' in result.context_items
    # Formatted text must show qualified ID, not bare int
    item = result.context_items['task:450.2']
    assert '[task:450.2]' in item.formatted
    assert '[task:2]' not in item.formatted


# ── _ctx_tasks_bulk_created parent_id qualification ──────────────────


@pytest.mark.asyncio
async def test_ctx_tasks_bulk_created_uses_parent_id_as_display_id(
    mock_memory, mock_taskmaster
):
    """_ctx_tasks_bulk_created passes parent_id as display_id to _format_task.

    When the event payload carries parent_task_id='7' but get_task returns a
    dict where id is None (e.g., a Taskmaster quirk), the ContextItem's
    formatted text must still contain '[task:7]' — taken from the event's
    parent_task_id — rather than '[task:None]' from the bare task dict.
    """
    mock_taskmaster.get_task = AsyncMock(
        return_value={
            'id': None,   # id missing/null from Taskmaster response
            'title': 'Parent task',
            'status': 'in-progress',
            'dependencies': [],
            'metadata': {},
        }
    )

    assembler = _make_assembler(
        memory_service=mock_memory,
        taskmaster=mock_taskmaster,
    )
    events = [
        _make_event(
            event_type=EventType.tasks_bulk_created,
            payload={'parent_task_id': '7', 'count': 3},
        ),
    ]
    watermark = _make_watermark()

    result = await assembler.assemble(events, watermark, 'test-project')

    assert 'task:7' in result.context_items
    item = result.context_items['task:7']
    assert '[task:7]' in item.formatted
    assert '[task:None]' not in item.formatted


# ── Integration: full assemble() with subtask status-change event ────


@pytest.mark.asyncio
async def test_assemble_subtask_event_qualified_id_end_to_end(
    mock_memory, mock_taskmaster
):
    """End-to-end: assemble() with subtask status-change event emits qualified ID.

    Validates the complete pipeline from event ingestion through payload assembly:
    - Event payload has task_id='450.2'
    - get_task returns a dict with bare id=2
    - Assembled context_items must have key 'task:450.2'
    - That item's formatted text must contain '[task:450.2]', not '[task:2]'

    This is the regression test for the Stage 1 cross-reference gap: Stage 1
    sees '[450.2]' in the Active Task Tree but was seeing '[task:2]' in Related
    Context before this fix.
    """
    mock_taskmaster.get_task = AsyncMock(
        return_value={
            'id': 2,
            'title': 'Wire up auth endpoint',
            'status': 'done',
            'dependencies': [],
            'metadata': {
                'memory_hints': {},
            },
        }
    )
    mock_memory.search = AsyncMock(return_value=[])

    assembler = _make_assembler(
        memory_service=mock_memory,
        taskmaster=mock_taskmaster,
    )

    events = [
        _make_event(
            event_type=EventType.task_status_changed,
            payload={
                'task_id': '450.2',
                'old_status': 'in-progress',
                'new_status': 'done',
            },
        ),
    ]
    watermark = _make_watermark()

    result = await assembler.assemble(events, watermark, 'test-project')

    # The assembled payload must include the event
    assert len(result.events) == 1

    # The context item key must be the qualified ID from the event payload
    assert 'task:450.2' in result.context_items, (
        f"Expected 'task:450.2' key in context_items, got: {list(result.context_items.keys())}"
    )

    # The formatted text must use the qualified ID, not the bare int from task dict
    item = result.context_items['task:450.2']
    assert '[task:450.2]' in item.formatted, (
        f"Expected '[task:450.2]' in formatted text, got: {item.formatted!r}"
    )
    assert '[task:2]' not in item.formatted, (
        f"Must not contain bare '[task:2]', got: {item.formatted!r}"
    )
