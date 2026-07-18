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

import uuid
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


# ---------------------------------------------------------------------------
# recover_mem0_intents: startup reconciliation of crash-mid-write intents
# ---------------------------------------------------------------------------


async def _seed_pending_intent(
    journal,
    *,
    write_op_id,
    intent_id,
    content='seeded content',
    metadata=None,
    payload_digest='dig',
    project_id='proj-x',
    agent_id='agent-a',
    session_id='sess-1',
):
    await journal.log_mem0_intent(
        intent_id=intent_id,
        write_op_id=write_op_id,
        project_id=project_id,
        agent_id=agent_id,
        session_id=session_id,
        category='preferences_and_norms',
        content=content,
        metadata=metadata if metadata is not None else {'category': 'preferences_and_norms'},
        payload_digest=payload_digest,
    )


class TestRecoverMem0Intents:
    @pytest.mark.asyncio
    async def test_success_beop_marks_completed_no_reissue(self, recovery_service):
        """A SUCCESS mem0 backend_op proves the write landed → completed, no re-issue."""
        svc = recovery_service
        journal = svc._write_journal
        assert journal is not None
        write_op_id = str(uuid.uuid4())
        intent_id = str(uuid.uuid4())
        await _seed_pending_intent(
            journal, write_op_id=write_op_id, intent_id=intent_id
        )
        # A successful mem0 backend_op for the same write_op_id.
        await journal.log_backend_op(
            write_op_id=write_op_id, backend='mem0', operation='add', success=True
        )
        svc.mem0.add.reset_mock()

        summary = await svc.recover_mem0_intents()

        svc.mem0.add.assert_not_called()
        assert await journal.get_incomplete_mem0_intents() == []
        completed = await journal.get_mem0_intents(status='completed')
        assert [r['id'] for r in completed] == [intent_id]
        assert summary['scanned'] == 1
        assert summary['reconciled'] == 1
        assert summary['reissued'] == 0
        assert summary['dead_lettered'] == 0

    @pytest.mark.asyncio
    async def test_failed_only_beop_reissues(self, recovery_service):
        """Only a FAILED mem0 backend_op → the add() provably raised → safe re-issue."""
        svc = recovery_service
        journal = svc._write_journal
        assert journal is not None
        write_op_id = str(uuid.uuid4())
        intent_id = str(uuid.uuid4())
        meta = {'category': 'preferences_and_norms', 'kind': 'note'}
        await _seed_pending_intent(
            journal,
            write_op_id=write_op_id,
            intent_id=intent_id,
            content='reissue me faithfully',
            metadata=meta,
            payload_digest='dig-2',
        )
        await journal.log_backend_op(
            write_op_id=write_op_id,
            backend='mem0',
            operation='add',
            success=False,
            error='timeout',
        )
        svc.mem0.add.reset_mock()
        svc.mem0.add.return_value = {'results': [{'id': 'reissued-1'}]}

        summary = await svc.recover_mem0_intents()

        # Re-issued exactly once with the reconstructed content / Scope / metadata.
        svc.mem0.add.assert_called_once()
        call_kwargs = svc.mem0.add.call_args.kwargs
        assert call_kwargs['content'] == 'reissue me faithfully'
        assert call_kwargs['scope'].project_id == 'proj-x'
        assert call_kwargs['scope'].agent_id == 'agent-a'
        assert call_kwargs['scope'].session_id == 'sess-1'
        assert call_kwargs['metadata'] == meta

        assert await journal.get_incomplete_mem0_intents() == []
        completed = await journal.get_mem0_intents(status='completed')
        assert [r['id'] for r in completed] == [intent_id]
        assert summary['scanned'] == 1
        assert summary['reconciled'] == 0
        assert summary['reissued'] == 1
        assert summary['dead_lettered'] == 0

    @pytest.mark.asyncio
    async def test_no_beop_dead_letters_headline(self, recovery_service):
        """HEADLINE: NO backend_op → outcome UNKNOWN → dead-letter, never a silent orphan."""
        svc = recovery_service
        journal = svc._write_journal
        assert journal is not None
        write_op_id = str(uuid.uuid4())
        intent_id = str(uuid.uuid4())
        await _seed_pending_intent(
            journal,
            write_op_id=write_op_id,
            intent_id=intent_id,
            content='ambiguous outcome',
            payload_digest='dig-headline',
        )
        # NO backend_op at all — killed before/inside the await.
        svc.mem0.add.reset_mock()

        summary = await svc.recover_mem0_intents()

        # A non-idempotent add is NOT blindly re-issued.
        svc.mem0.add.assert_not_called()
        assert await journal.get_incomplete_mem0_intents() == []
        dead = await journal.get_mem0_intents(status='dead')
        assert [r['id'] for r in dead] == [intent_id]
        # Structured, non-empty reason embedding the payload_digest.
        assert dead[0]['reason']
        assert 'dig-headline' in dead[0]['reason']
        assert summary['scanned'] == 1
        assert summary['reconciled'] == 0
        assert summary['reissued'] == 0
        assert summary['dead_lettered'] == 1
