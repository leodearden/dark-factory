"""Write-ahead intent for the dual-store add_memory path (task 2710).

Exercises MemoryService.add_memory intent bracketing and
recover_mem0_intents against a real WriteJournal (SQLite on tmp_path) with a
mocked mem0 backend, so the "kill mid-Mem0-write is never SILENT" guarantee
is proven by runtime behavior — the intent must be durably committed
BEFORE the mem0 await, and every crash-mid-write outcome resolves to a
visible terminal row.

Cloned from test_e2e_durable_queue.integrated_service, additionally wiring a
real WriteJournal via set_write_journal.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from fused_memory.models.enums import SourceStore
from fused_memory.services.memory_service import MemoryService
from fused_memory.services.write_journal import WriteJournal


@pytest_asyncio.fixture
async def recovery_service(mock_config, tmp_path):
    """MemoryService with real DurableWriteQueue + real WriteJournal, mocked backends."""
    svc = MemoryService(mock_config)

    # Mock backends so we never hit real FalkorDB/Qdrant
    svc.graphiti = MagicMock()
    svc.graphiti.initialize = AsyncMock()
    svc.graphiti.add_episode = AsyncMock(return_value=None)
    svc.graphiti.close = AsyncMock()
    svc.graphiti._require_client = MagicMock()

    svc.mem0 = MagicMock()
    svc.mem0.search = AsyncMock(return_value={'results': []})
    svc.mem0.add = AsyncMock(return_value={'results': [{'id': 'mem0-1'}]})
    svc.mem0.get_all = AsyncMock(return_value={'results': []})
    svc.mem0.delete = AsyncMock(return_value={'message': 'deleted'})
    svc.mem0.close = AsyncMock()

    # Real initialize — creates DurableWriteQueue with real SQLite
    await svc.initialize()

    # Real WriteJournal so intent bracketing + recovery hit real SQLite
    journal = WriteJournal(tmp_path / 'wj')
    await journal.initialize()
    svc.set_write_journal(journal)

    yield svc

    # svc.close() closes the wired write_journal too (see MemoryService.close)
    await svc.close()


# ---------------------------------------------------------------------------
# add_memory intent bracketing (durable-before-await)
# ---------------------------------------------------------------------------


class TestAddMemoryIntentBracketing:
    @pytest.mark.asyncio
    async def test_intent_precedes_await_and_completes(self, recovery_service):
        """The intent is durably 'pending' BEFORE mem0.add is awaited, then completed."""
        svc = recovery_service
        journal = svc._write_journal
        assert journal is not None

        seen: dict = {}

        async def _assert_pending_at_call_time(*args, **kwargs):
            # The write-ahead guarantee: the intent must already be committed
            # (pending) on disk at the moment mem0.add is entered.
            seen['pending'] = await journal.get_incomplete_mem0_intents()
            return {'results': [{'id': 'mem0-1'}]}

        svc.mem0.add = AsyncMock(side_effect=_assert_pending_at_call_time)

        long_content = 'remember to always pin dependencies exactly ' * 8  # > 200 chars
        result = await svc.add_memory(
            content=long_content,
            category='preferences_and_norms',  # mem0-primary
            project_id='proj-x',
            agent_id='agent-a',
            session_id='sess-1',
        )

        # A pending intent existed at mem0.add call time, with full content + digest.
        assert len(seen['pending']) == 1
        pending_row = seen['pending'][0]
        assert pending_row['status'] == 'pending'
        assert pending_row['payload_digest']  # non-empty digest
        assert pending_row['content'] == long_content  # FULL, untruncated
        assert len(pending_row['content']) > 200

        # After the call the intent is completed and nothing is left pending.
        assert await journal.get_incomplete_mem0_intents() == []
        completed = await journal.get_mem0_intents(status='completed')
        assert len(completed) == 1
        assert completed[0]['content'] == long_content
        assert SourceStore.mem0 in result.stores_written

    @pytest.mark.asyncio
    async def test_mem0_error_marks_intent_failed(self, recovery_service):
        """mem0.add raising resolves the intent 'failed' with the error reason; add_memory returns."""
        svc = recovery_service
        journal = svc._write_journal
        assert journal is not None

        svc.mem0.add = AsyncMock(side_effect=RuntimeError('mem0 down'))

        result = await svc.add_memory(
            content='always run black before commit',
            category='preferences_and_norms',
            project_id='proj-x',
        )

        # add_memory still returns, surfacing the mem0 error to the caller.
        assert 'mem0_error' in result.message
        assert 'mem0 down' in result.message

        # The intent is resolved 'failed' (not left pending) with the reason.
        assert await journal.get_incomplete_mem0_intents() == []
        failed = await journal.get_mem0_intents(status='failed')
        assert len(failed) == 1
        assert 'mem0 down' in (failed[0]['reason'] or '')
