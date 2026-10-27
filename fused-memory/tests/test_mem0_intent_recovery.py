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

import datetime
import json
import logging
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from _mem0_intent_fixtures import rate_limit_error

from fused_memory.models.enums import SourceStore
from fused_memory.models.scope import Scope
from fused_memory.services.memory_service import MemoryService
from fused_memory.services.write_journal import WriteJournal


@pytest_asyncio.fixture
async def recovery_journal(tmp_path):
    """Real WriteJournal (SQLite on tmp_path); closed by recovery_service's teardown."""
    journal = WriteJournal(tmp_path / 'wj')
    await journal.initialize()
    yield journal


async def _service_with_mocked_backends(mock_config) -> MemoryService:
    """MemoryService with a real DurableWriteQueue and mocked backends, no journal wired."""
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
    return svc


@pytest_asyncio.fixture
async def recovery_service(mock_config, recovery_journal):
    """MemoryService with real DurableWriteQueue + real WriteJournal, mocked backends."""
    svc = await _service_with_mocked_backends(mock_config)

    # Real WriteJournal so intent bracketing + recovery hit real SQLite
    svc.set_write_journal(recovery_journal)

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

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'meta',
        [
            pytest.param({'x_tags': {'a', 'b'}}, id='set-value'),
            pytest.param({'x_when': datetime.datetime(2026, 1, 1)}, id='datetime-value'),
            pytest.param({'x_nested': {1: 'a', 'b': 'c'}}, id='nested-mixed-keys'),
        ],
    )
    async def test_unjournalable_metadata_fails_open_past_the_digest(
        self, recovery_service, recovery_journal, caplog, meta
    ):
        """A digest that cannot serialize the metadata never strands the Graphiti twin."""
        svc = recovery_service

        with caplog.at_level(logging.ERROR, logger='fused_memory.services.memory_service'):
            result = await svc.add_memory(
                content='always pin dependencies exactly',
                category='preferences_and_norms',
                project_id='proj-x',
                metadata=dict(meta),
                dual_write=True,
                causation_id='cause-digest',
            )

        assert svc.mem0.add.await_count == 1
        assert SourceStore.graphiti in result.stores_written
        assert SourceStore.mem0 in result.stores_written
        mem0_beops = [
            op
            for op in await recovery_journal.get_ops_by_causation('cause-digest')
            if op['layer'] == 'backend_op' and op['backend'] == 'mem0'
        ]
        assert [op['success'] for op in mem0_beops] == [1]
        assert any(
            r.levelno == logging.ERROR
            and r.name == 'fused_memory.services.memory_service'
            for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_digest_failure_still_journals_intent_without_digest(
        self, recovery_service, recovery_journal
    ):
        """Nested mixed-type keys break only the sorted digest; the journal still stores the intent."""
        svc = recovery_service
        content = 'always pin dependencies exactly'

        await svc.add_memory(
            content=content,
            category='preferences_and_norms',
            project_id='proj-x',
            metadata={'x_nested': {1: 'a', 'b': 'c'}},
        )

        completed = await recovery_journal.get_mem0_intents(status='completed')
        assert len(completed) == 1
        assert completed[0]['payload_digest'] is None
        assert completed[0]['content'] == content


# ---------------------------------------------------------------------------
# add_memory defers a transient Mem0 failure for server-side retry (task 7397)
# ---------------------------------------------------------------------------


class _IntentDroppingJournal(WriteJournal):
    """A journal whose write-ahead intent silently does not land."""

    async def log_mem0_intent(self, **kwargs: Any) -> None:
        return None


_DEFERRED_CONTENT = 'always pin dependencies exactly, the full untruncated text ' * 5


class TestAddMemoryTransientDeferral:
    @pytest.mark.asyncio
    async def test_a_rate_limited_write_is_queued_for_retry(
        self, recovery_service, recovery_journal
    ):
        svc = recovery_service
        svc.mem0.add = AsyncMock(side_effect=rate_limit_error())

        result = await svc.add_memory(
            content=_DEFERRED_CONTENT,
            category='procedural_knowledge',
            project_id='dark_factory',
            agent_id='agent-a',
        )

        assert result.memory_ids == []
        assert SourceStore.mem0 not in result.stores_written
        assert result.stores_pending_retry == [SourceStore.mem0]
        assert 'mem0_pending_retry' in result.message
        assert 'mem0_error' not in result.message
        (intent,) = await recovery_journal.get_mem0_intents()
        assert intent['status'] == 'retrying'
        assert intent['attempts'] == 1
        assert datetime.datetime.fromisoformat(
            intent['next_attempt_at']
        ) > datetime.datetime.fromisoformat(intent['created_at'])
        assert intent['content'] == _DEFERRED_CONTENT
        write_op = await recovery_journal.get_write_op(intent['write_op_id'])
        assert write_op is not None
        assert json.loads(write_op['result_summary'])['stores_pending'] == ['mem0']

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'error',
        [RuntimeError('qdrant down'), TimeoutError('Mem0 add timed out after 30.0s')],
        ids=['runtime-error', 'write-budget-timeout'],
    )
    async def test_a_non_transient_failure_keeps_the_failed_path(
        self, recovery_service, recovery_journal, error
    ):
        svc = recovery_service
        svc.mem0.add = AsyncMock(side_effect=error)

        result = await svc.add_memory(
            content='always run black before commit',
            category='procedural_knowledge',
            project_id='dark_factory',
        )

        assert 'mem0_error' in result.message
        assert result.stores_pending_retry == []
        (intent,) = await recovery_journal.get_mem0_intents()
        assert intent['status'] == 'failed'

    @pytest.mark.asyncio
    @pytest.mark.parametrize('journal_kind', ['intent-dropping', 'none'])
    async def test_no_durable_intent_means_no_pending_claim(
        self, mock_config, tmp_path, journal_kind
    ):
        svc = await _service_with_mocked_backends(mock_config)
        journal: WriteJournal | None = None
        if journal_kind == 'intent-dropping':
            journal = _IntentDroppingJournal(tmp_path / 'dropping-wj')
            await journal.initialize()
            svc.set_write_journal(journal)
        svc.mem0.add = AsyncMock(side_effect=rate_limit_error())
        try:
            result = await svc.add_memory(
                content='always run black before commit',
                category='procedural_knowledge',
                project_id='dark_factory',
            )
            assert 'mem0_error' in result.message
            assert result.stores_pending_retry == []
            if journal is not None:
                assert await journal.get_mem0_intents(status='retrying') == []
        finally:
            await svc.close()

    @pytest.mark.asyncio
    async def test_a_dual_write_keeps_its_graphiti_leg_and_defers_mem0(
        self, recovery_service
    ):
        svc = recovery_service
        svc.mem0.add = AsyncMock(side_effect=rate_limit_error())

        result = await svc.add_memory(
            content='we chose sqlite for the journal because it is local',
            category='decisions_and_rationale',
            project_id='dark_factory',
            dual_write=True,
        )

        assert result.stores_written == [SourceStore.graphiti]
        assert result.stores_pending_retry == [SourceStore.mem0]

    @pytest.mark.asyncio
    async def test_the_deferred_write_lands_when_the_retry_pass_runs(
        self, recovery_service, recovery_journal
    ):
        svc = recovery_service
        failing_add = AsyncMock(side_effect=rate_limit_error())
        svc.mem0.add = failing_add
        await svc.add_memory(
            content=_DEFERRED_CONTENT,
            category='procedural_knowledge',
            project_id='dark_factory',
            agent_id='agent-a',
            session_id='sess-1',
            metadata={'x_note': 'kept'},
        )
        original = failing_add.call_args.kwargs
        (intent,) = await recovery_journal.get_mem0_intents()
        landing_add = AsyncMock(return_value={'results': [{'id': 'mem0-late'}]})
        svc.mem0.add = landing_add

        summary = await svc.retry_due_mem0_intents(
            now=datetime.datetime.fromisoformat(intent['next_attempt_at'])
            + datetime.timedelta(seconds=1)
        )

        assert summary.landed == 1
        landing_add.assert_awaited_once()
        retried = landing_add.call_args.kwargs
        assert retried['content'] == _DEFERRED_CONTENT
        assert retried['scope'] == Scope(
            project_id='dark_factory', agent_id='agent-a', session_id='sess-1'
        )
        assert retried['metadata'] == original['metadata']
        assert retried['metadata']['x_note'] == 'kept'
        (after,) = await recovery_journal.get_mem0_intents()
        assert after['status'] == 'completed'
        mem0_ops = [
            op
            for op in await recovery_journal.get_backend_ops_for_write_op(
                intent['write_op_id']
            )
            if op['backend'] == 'mem0'
        ]
        assert [op['success'] for op in mem0_ops] == [0, 1]

    @pytest.mark.asyncio
    async def test_the_tool_payload_carries_the_pending_store(self, recovery_service):
        svc = recovery_service
        svc.mem0.add = AsyncMock(side_effect=rate_limit_error())

        result = await svc.add_memory(
            content='always run black before commit',
            category='procedural_knowledge',
            project_id='dark_factory',
        )

        assert result.model_dump()['stores_pending_retry'] == ['mem0']


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
    causation_id=None,
):
    await journal.log_mem0_intent(
        intent_id=intent_id,
        write_op_id=write_op_id,
        causation_id=causation_id,
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
        # Scope canonicalizes project_id (dashes -> underscores), exactly as the
        # original add_memory write did, so the re-issue is faithful to it.
        svc.mem0.add.assert_called_once()
        call_kwargs = svc.mem0.add.call_args.kwargs
        assert call_kwargs['content'] == 'reissue me faithfully'
        assert call_kwargs['scope'].project_id == 'proj_x'
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

    @pytest.mark.asyncio
    async def test_reissue_exception_dead_letters_not_left_pending(self, recovery_service):
        """failed-only backend_op but re-issue raises → dead-lettered (not left pending)."""
        svc = recovery_service
        journal = svc._write_journal
        assert journal is not None
        write_op_id = str(uuid.uuid4())
        intent_id = str(uuid.uuid4())
        await _seed_pending_intent(
            journal,
            write_op_id=write_op_id,
            intent_id=intent_id,
            content='cannot reissue',
            payload_digest='dig-x',
        )
        await journal.log_backend_op(
            write_op_id=write_op_id,
            backend='mem0',
            operation='add',
            success=False,
            error='first failure',
        )
        svc.mem0.add = AsyncMock(side_effect=RuntimeError('reissue boom'))

        summary = await svc.recover_mem0_intents()

        # Not left pending — resolved 'dead' with the re-issue error captured.
        assert await journal.get_incomplete_mem0_intents() == []
        dead = await journal.get_mem0_intents(status='dead')
        assert [r['id'] for r in dead] == [intent_id]
        assert 'reissue boom' in (dead[0]['reason'] or '')
        assert summary['scanned'] == 1
        assert summary['reissued'] == 0
        assert summary['dead_lettered'] == 1

        # The failed re-issue is itself journaled as a mem0 backend_op.
        mem0_beops = [
            b
            for b in await journal.get_backend_ops_for_write_op(write_op_id)
            if b['backend'] == 'mem0'
        ]
        assert [b['success'] for b in mem0_beops] == [0, 0]
        assert 'reissue boom' in (mem0_beops[1]['error'] or '')

    @pytest.mark.asyncio
    async def test_a_transient_reissue_failure_is_deferred_not_dead_lettered(
        self, recovery_service, recovery_journal
    ):
        """A provider still rate-limited at restart must not dead-letter the crash-recovered write."""
        svc = recovery_service
        write_op_id = str(uuid.uuid4())
        intent_id = str(uuid.uuid4())
        await _seed_pending_intent(
            recovery_journal,
            write_op_id=write_op_id,
            intent_id=intent_id,
            content='reissue me later',
        )
        await recovery_journal.log_backend_op(
            write_op_id=write_op_id,
            backend='mem0',
            operation='add',
            success=False,
            error='RateLimitError: 429',
        )
        svc.mem0.add = AsyncMock(side_effect=rate_limit_error())

        summary = await svc.recover_mem0_intents()

        (intent,) = await recovery_journal.get_mem0_intents()
        assert intent['id'] == intent_id
        assert intent['status'] == 'retrying'
        assert intent['attempts'] == 1
        assert intent['next_attempt_at'] is not None
        assert summary['deferred'] == 1
        assert summary['dead_lettered'] == 0

    @pytest.mark.asyncio
    async def test_reissue_journals_success_backend_op_under_same_write_op_id(
        self, recovery_service, recovery_journal
    ):
        """A successful re-issue leaves a SUCCESS mem0 backend_op under the intent's write_op_id."""
        svc = recovery_service
        write_op_id = str(uuid.uuid4())
        await _seed_pending_intent(
            recovery_journal,
            write_op_id=write_op_id,
            intent_id=str(uuid.uuid4()),
            causation_id='cause-reissue',
        )
        await recovery_journal.log_backend_op(
            write_op_id=write_op_id,
            backend='mem0',
            operation='add',
            success=False,
            error='timeout',
        )

        summary = await svc.recover_mem0_intents()

        rows = [
            b
            for b in await recovery_journal.get_backend_ops_for_write_op(write_op_id)
            if b['backend'] == 'mem0'
        ]
        assert [b['success'] for b in rows] == [0, 1]
        assert rows[-1]['operation'] == 'add'
        assert rows[-1]['causation_id'] == 'cause-reissue'
        assert summary['reissued'] == 1

    @pytest.mark.asyncio
    async def test_crash_after_successful_reissue_is_reconciled_not_reissued_again(
        self, recovery_service, recovery_journal, monkeypatch
    ):
        """HEADLINE: a crash between a successful re-issue and its stamp never re-issues twice."""
        svc = recovery_service
        write_op_id = str(uuid.uuid4())
        intent_id = str(uuid.uuid4())
        await _seed_pending_intent(
            recovery_journal, write_op_id=write_op_id, intent_id=intent_id
        )
        await recovery_journal.log_backend_op(
            write_op_id=write_op_id,
            backend='mem0',
            operation='add',
            success=False,
            error='timeout',
        )
        svc.mem0.add.reset_mock()

        # Process dies after the re-issue lands but before the 'completed' stamp.
        monkeypatch.setattr(recovery_journal, 'resolve_mem0_intent', AsyncMock())
        await svc.recover_mem0_intents()

        assert svc.mem0.add.await_count == 1
        pending = await recovery_journal.get_incomplete_mem0_intents()
        assert [r['id'] for r in pending] == [intent_id]

        # Restart: the real stamp is back.
        monkeypatch.undo()
        second = await svc.recover_mem0_intents()

        assert svc.mem0.add.await_count == 1
        assert second == {
            'scanned': 1,
            'reconciled': 1,
            'reissued': 0,
            'deferred': 0,
            'dead_lettered': 0,
        }
        completed = await recovery_journal.get_mem0_intents(status='completed')
        assert [r['id'] for r in completed] == [intent_id]

    @pytest.mark.asyncio
    async def test_second_recover_is_zero_work(self, recovery_service):
        """A second recover after all intents resolve is a zero-work no-op."""
        svc = recovery_service
        journal = svc._write_journal
        assert journal is not None
        write_op_id = str(uuid.uuid4())
        intent_id = str(uuid.uuid4())
        await _seed_pending_intent(
            journal, write_op_id=write_op_id, intent_id=intent_id
        )
        await journal.log_backend_op(
            write_op_id=write_op_id, backend='mem0', operation='add', success=True
        )

        first = await svc.recover_mem0_intents()
        assert first['scanned'] == 1

        svc.mem0.add.reset_mock()
        second = await svc.recover_mem0_intents()

        assert second == {
            'scanned': 0,
            'reconciled': 0,
            'reissued': 0,
            'deferred': 0,
            'dead_lettered': 0,
        }
        svc.mem0.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_journal_unset_returns_zeroed_summary(self, mock_config):
        """A MemoryService without a wired journal returns a zeroed summary, no error."""
        svc = MemoryService(mock_config)
        svc.mem0 = MagicMock()
        svc.mem0.add = AsyncMock()
        # No set_write_journal call → self._write_journal is None.
        assert svc._write_journal is None

        summary = await svc.recover_mem0_intents()

        assert summary == {
            'scanned': 0,
            'reconciled': 0,
            'reissued': 0,
            'deferred': 0,
            'dead_lettered': 0,
        }
        svc.mem0.add.assert_not_called()
