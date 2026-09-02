"""The MemoryService half of ensure_entity_node (task 4932): lock, pre-read, journal.

Guards 1, 2 and the journal half of 7. The wrapper composes three EXISTING
backend primitives and adds no new Cypher:

  * ``_identity_lock_for`` — the per-group_id write-time-identity lock, acquired
    with a BOUNDED wait (guard 1). This is the second acquisition site in all of
    ``fused-memory/src/``; the first is
    ``MemoryService._execute_graphiti_write``, which holds it across a full LLM
    extraction plus ``_reconcile_episode_identity``, which is exactly why an
    unbounded ``async with`` would block an MCP request for tens of seconds.
  * ``get_nodes_by_exact_name`` — the exact-name PRE-READ under that lock
    (guard 2), which is what makes ``_resolve_or_create_entity``'s irreversible
    >=2-match COLLAPSE arm structurally unreachable through this tool.
  * ``ensure_entity_node`` — called unchanged, on the 0-match branch only.

TEST POSTURE: backend doubles plus ``install_identity_mocks``, which installs a
REAL per-group_id ``asyncio.Lock`` registry. A FalkorDriver is NEVER constructed
— doing so fires ``build_indices_and_constraints`` and would destroy protected
index evidence.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from _fm_helpers import install_identity_mocks

from fused_memory.services.memory_service import MemoryService

_PROJECT = 'dark_factory'
_NAME = 'Task 3222'


@pytest.fixture
def service(mock_config):
    """MemoryService with fully-mocked backends.

    Copies ``tests/test_referent_verification.py``'s fixture in shape.
    ``install_identity_mocks`` is REQUIRED, not decorative: the wrapper acquires
    ``self.graphiti._identity_lock_for(project_id)``, and a bare MagicMock hands
    back a child mock with no ``acquire``.
    """
    svc = MemoryService(mock_config)
    svc.graphiti = MagicMock()
    svc.graphiti.get_nodes_by_exact_name = AsyncMock(return_value=[])
    svc.graphiti.ensure_entity_node = AsyncMock(return_value='uuid-new')
    svc.graphiti.merge_entities = AsyncMock()
    install_identity_mocks(svc.graphiti)
    return svc


def _node(uuid: str, name: str = _NAME) -> dict:
    return {'uuid': uuid, 'name': name, 'summary': '', 'labels': ['Entity']}


class TestMintAndResolve:
    """Guard 2: the pre-read decides mint vs. resolve vs. refuse."""

    @pytest.mark.asyncio
    async def test_zero_matches_mints_via_the_backend(self, service):
        result = await service.ensure_entity_node(
            name=_NAME, project_id=_PROJECT, agent_id='curator-x',
        )

        service.graphiti.ensure_entity_node.assert_awaited_once()
        args, kwargs = service.graphiti.ensure_entity_node.await_args
        assert args[0] == _NAME
        assert kwargs['group_id'] == _PROJECT
        assert result['status'] == 'minted'
        assert result['minted'] is True
        assert result['uuid'] == 'uuid-new'
        assert result['name'] == _NAME

    @pytest.mark.asyncio
    async def test_one_match_resolves_without_any_backend_write(self, service):
        """A pure resolve does no writes at all — not even a backend call."""
        service.graphiti.get_nodes_by_exact_name = AsyncMock(
            return_value=[_node('uuid-existing')],
        )

        result = await service.ensure_entity_node(
            name=_NAME, project_id=_PROJECT, agent_id='curator-x',
        )

        assert result['status'] == 'resolved'
        assert result['minted'] is False
        assert result['uuid'] == 'uuid-existing'
        service.graphiti.ensure_entity_node.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_two_matches_refuse_and_merge_nothing(self, service):
        """The irreversible collapse arm of _resolve_or_create_entity must not
        be reachable through this tool.

        merge_entities is irreversible, has no type check, discards the
        deprecated summary and hard-deletes parallel duplicates leaving only a
        count in a best-effort journal — which is why the referent-fidelity PRD
        puts the duplicate-name keys OUT OF SCOPE. The refusal names every
        matching uuid so an operator can adjudicate them by hand.
        """
        service.graphiti.get_nodes_by_exact_name = AsyncMock(
            return_value=[_node('uuid-a'), _node('uuid-b')],
        )

        result = await service.ensure_entity_node(
            name=_NAME, project_id=_PROJECT, agent_id='curator-x',
        )

        assert result['status'] == 'refused'
        assert result['error_type'] == 'EntityMintAmbiguousName', result
        assert isinstance(result.get('error'), str) and result['error']
        assert set(result['uuids']) == {'uuid-a', 'uuid-b'}, result
        service.graphiti.ensure_entity_node.assert_not_awaited()
        service.graphiti.merge_entities.assert_not_awaited()


class TestIdentityLock:
    """Guard 1: held across the pre-read AND the mint, and bounded."""

    @pytest.mark.asyncio
    async def test_lock_is_held_across_the_pre_read_and_released_on_return(
        self, service,
    ):
        lock = service.graphiti._identity_lock_for(_PROJECT)
        observed = {}

        async def _probe(name, *, group_id):
            observed['locked'] = lock.locked()
            observed['same_lock'] = (
                service.graphiti._identity_lock_for(group_id) is lock
            )
            return []

        service.graphiti.get_nodes_by_exact_name = AsyncMock(side_effect=_probe)

        await service.ensure_entity_node(
            name=_NAME, project_id=_PROJECT, agent_id='curator-x',
        )

        assert observed['locked'] is True, (
            'the pre-read must run INSIDE the critical section, or a concurrent '
            'writer could mint the same name between the read and the mint'
        )
        assert observed['same_lock'] is True
        assert lock.locked() is False, 'the lock must be released on return'

    @pytest.mark.asyncio
    async def test_lock_is_held_across_the_mint_itself(self, service):
        lock = service.graphiti._identity_lock_for(_PROJECT)
        observed = {}

        async def _probe(name, *, group_id, summary=''):
            observed['locked'] = lock.locked()
            return 'uuid-new'

        service.graphiti.ensure_entity_node = AsyncMock(side_effect=_probe)

        await service.ensure_entity_node(
            name=_NAME, project_id=_PROJECT, agent_id='curator-x',
        )

        assert observed['locked'] is True
        assert lock.locked() is False

    @pytest.mark.asyncio
    async def test_a_busy_lock_returns_a_refusal_value_rather_than_blocking(
        self, service,
    ):
        """The acquire is BOUNDED by entity_mint.lock_timeout_seconds.

        A refusal VALUE, not a raise and not an unbounded block: the other
        holder keeps this lock across a full LLM extraction, so an MCP request
        would otherwise hang for tens of seconds.
        """
        service.config.entity_mint.lock_timeout_seconds = 0.05
        lock = service.graphiti._identity_lock_for(_PROJECT)
        await lock.acquire()
        try:
            result = await service.ensure_entity_node(
                name=_NAME, project_id=_PROJECT, agent_id='curator-x',
            )
        finally:
            lock.release()

        assert result['status'] == 'refused'
        assert result['error_type'] == 'EntityMintLockBusy', result
        assert isinstance(result.get('error'), str) and result['error']
        service.graphiti.ensure_entity_node.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_lock_timeout_is_read_live_per_call(self, service):
        """Proving the leaf is green-tier rather than restart-only in disguise:
        a timeout captured at construction could not observe this mutation."""
        service.config.entity_mint.lock_timeout_seconds = 0.05
        lock = service.graphiti._identity_lock_for(_PROJECT)
        await lock.acquire()
        try:
            busy = await service.ensure_entity_node(
                name=_NAME, project_id=_PROJECT, agent_id='curator-x',
            )
            assert busy['error_type'] == 'EntityMintLockBusy'
        finally:
            lock.release()

        # Uncontended again: the SAME service object now mints.
        ok = await service.ensure_entity_node(
            name=_NAME, project_id=_PROJECT, agent_id='curator-x',
        )
        assert ok['minted'] is True, ok


class TestWriteJournal:
    """Every outcome is journalled — the evidence trail a storm alarm points at."""

    @staticmethod
    def _journal(service) -> AsyncMock:
        journal = AsyncMock()
        service.set_write_journal(journal)
        return journal

    @pytest.mark.asyncio
    async def test_a_mint_is_journalled_as_minted(self, service):
        journal = self._journal(service)

        await service.ensure_entity_node(
            name=_NAME, project_id=_PROJECT, agent_id='curator-x',
        )

        journal.log_write_op.assert_awaited_once()
        kwargs = journal.log_write_op.await_args.kwargs
        assert kwargs['operation'] == 'ensure_entity_node'
        assert kwargs['params']['name'] == _NAME
        assert kwargs['success'] is True
        assert kwargs['result_summary']['minted'] is True
        assert kwargs['result_summary']['status'] == 'minted'

    @pytest.mark.asyncio
    async def test_a_resolve_is_journalled_as_not_minted(self, service):
        service.graphiti.get_nodes_by_exact_name = AsyncMock(
            return_value=[_node('uuid-existing')],
        )
        journal = self._journal(service)

        await service.ensure_entity_node(
            name=_NAME, project_id=_PROJECT, agent_id='curator-x',
        )

        kwargs = journal.log_write_op.await_args.kwargs
        assert kwargs['success'] is True
        assert kwargs['result_summary']['minted'] is False
        assert kwargs['result_summary']['status'] == 'resolved'

    @pytest.mark.asyncio
    async def test_an_ambiguous_refusal_is_journalled_as_refused(self, service):
        service.graphiti.get_nodes_by_exact_name = AsyncMock(
            return_value=[_node('uuid-a'), _node('uuid-b')],
        )
        journal = self._journal(service)

        await service.ensure_entity_node(
            name=_NAME, project_id=_PROJECT, agent_id='curator-x',
        )

        kwargs = journal.log_write_op.await_args.kwargs
        assert kwargs['success'] is False
        assert kwargs['result_summary']['status'] == 'refused'
        assert kwargs['result_summary']['error_type'] == 'EntityMintAmbiguousName'

    @pytest.mark.asyncio
    async def test_a_lock_busy_refusal_is_journalled_as_refused(self, service):
        service.config.entity_mint.lock_timeout_seconds = 0.05
        journal = self._journal(service)
        lock = service.graphiti._identity_lock_for(_PROJECT)
        await lock.acquire()
        try:
            await service.ensure_entity_node(
                name=_NAME, project_id=_PROJECT, agent_id='curator-x',
            )
        finally:
            lock.release()

        kwargs = journal.log_write_op.await_args.kwargs
        assert kwargs['success'] is False
        assert kwargs['result_summary']['error_type'] == 'EntityMintLockBusy'

    @pytest.mark.asyncio
    async def test_a_backend_failure_is_journalled_and_re_raised(self, service):
        service.graphiti.ensure_entity_node = AsyncMock(side_effect=RuntimeError('boom'))
        journal = self._journal(service)

        with pytest.raises(RuntimeError, match='boom'):
            await service.ensure_entity_node(
                name=_NAME, project_id=_PROJECT, agent_id='curator-x',
            )

        kwargs = journal.log_write_op.await_args.kwargs
        assert kwargs['success'] is False
        assert 'boom' in kwargs['error']

    @pytest.mark.asyncio
    async def test_a_failing_journal_never_breaks_the_mint(self, service):
        journal = AsyncMock()
        journal.log_write_op = AsyncMock(side_effect=RuntimeError('journal down'))
        service.set_write_journal(journal)

        result = await service.ensure_entity_node(
            name=_NAME, project_id=_PROJECT, agent_id='curator-x',
        )

        assert result['minted'] is True, (
            'the journal is best-effort — a journal failure must not turn a '
            'landed mint into an error'
        )

    @pytest.mark.asyncio
    async def test_the_lock_is_released_even_when_the_backend_raises(self, service):
        service.graphiti.ensure_entity_node = AsyncMock(side_effect=RuntimeError('boom'))
        lock = service.graphiti._identity_lock_for(_PROJECT)

        with pytest.raises(RuntimeError):
            await service.ensure_entity_node(
                name=_NAME, project_id=_PROJECT, agent_id='curator-x',
            )

        assert lock.locked() is False, (
            'a leaked identity lock would wedge every subsequent write to this '
            'group_id, including the episode-ingest path'
        )
