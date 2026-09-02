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

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from _fm_helpers import install_identity_mocks

from fused_memory.services.memory_service import (
    _ENTITY_MINT_DEFAULT_LOCK_TIMEOUT_SECONDS,
    MemoryService,
)

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

    @pytest.mark.parametrize('corrupt', [None, 'x', 0, 0.0, -1.0, True, False])
    @pytest.mark.asyncio
    async def test_a_corrupt_lock_timeout_falls_back_to_the_module_default(
        self, service, monkeypatch, corrupt,
    ):
        """A missing / non-numeric / bool / non-positive leaf must not become an
        UNBOUNDED wait, and must not crash the mint either.

        The equivalent corrupt-leaf branches for ``storm_threshold`` and
        ``storm_window_seconds`` are already pinned below; without this one the
        omission is asymmetric — and this is the leaf whose failure mode is the
        worst of the three, because losing the BOUND (rather than losing an
        alarm) is what hangs an MCP request behind the episode-ingest path's
        multi-second hold.

        ``True`` is load-bearing among the parameters: ``isinstance(True, int)``
        is True and ``True > 0``, so a naive numeric check would accept it and
        bound the acquire at one second on a config typo.
        """
        object.__setattr__(
            service.config.entity_mint, 'lock_timeout_seconds', corrupt,
        )
        captured: dict = {}
        real_wait_for = asyncio.wait_for

        async def _spy(awaitable, timeout):
            captured['timeout'] = timeout
            return await real_wait_for(awaitable, timeout)

        monkeypatch.setattr(asyncio, 'wait_for', _spy)

        result = await service.ensure_entity_node(
            name=_NAME, project_id=_PROJECT, agent_id='curator-x',
        )

        assert result['minted'] is True, result
        assert captured['timeout'] == _ENTITY_MINT_DEFAULT_LOCK_TIMEOUT_SECONDS
        assert _ENTITY_MINT_DEFAULT_LOCK_TIMEOUT_SECONDS == 5.0, (
            'the fallback is anchored to the schema default; if one moves the '
            'other must, or a corrupt leaf silently changes the bound'
        )

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

        # Without this the leg is VACUOUS: if a refactor ever moved the journal
        # call behind a condition that stopped matching, the failing-journal
        # path would go unexercised and this test would still be green.
        journal.log_write_op.assert_awaited_once()
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


class _FakeClock:
    """Advanceable clock — the 3600s window without sleeping.

    Copies ``tests/test_memory_service.py::_FakeClock``, the idiom the sibling
    ``mem0_update`` storm alarm is tested with.
    """

    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestEntityMintStormAlarm:
    """Guard 7: the MINT-path burst alarm (INV-4 storm escape).

    Minting is the one branch of this tool that CREATES something, and nothing
    sweeps orphan minted nodes — so a caller stuck in a mint loop leaves a
    growing pile of junk identity nodes and no other signal. This counter is
    that signal.

    A monitoring alarm, NEVER a rate limiter: crossing the threshold must not
    fail the mint that crossed it, or a legitimate repair batch would break
    mid-run over its own success count.

    The emitter is monkeypatched at the ``services.memory_service`` module
    symbol rather than asserted through a real escalation file: the
    ``escalation`` package is a DEFENSIVE OPTIONAL import and is absent in
    minimal envs, so a test that filed for real would be environment-coupled and
    would fail for a reason that has nothing to do with this alarm.
    """

    @pytest.fixture
    def stormy(self, service, monkeypatch):
        """Service + fake clock + stubbed emitter."""
        clock = _FakeClock()
        service._entity_mint_storm_time_provider = clock
        # The escalator resolves project_root from `_known_projects` and
        # escalates NOTHING when it cannot (never guessing at cwd), so the
        # registry has to be populated or every leg here would pass vacuously.
        service.set_known_projects({_PROJECT: '/tmp/df-root'})
        emitter = MagicMock(return_value='esc-entity-mint-storm-1')
        monkeypatch.setattr(
            'fused_memory.services.memory_service.emit_entity_mint_storm_escalation',
            emitter,
        )
        return service, clock, emitter

    @staticmethod
    async def _mint(service, agent_id: str | None = 'curator-x'):
        return await service.ensure_entity_node(
            name=_NAME, project_id=_PROJECT, agent_id=agent_id,
        )

    @pytest.mark.asyncio
    async def test_a_burst_of_mints_fires_the_emitter_exactly_once(self, stormy):
        service, _clock, emitter = stormy
        threshold = service.config.entity_mint.storm_threshold

        for _ in range(threshold - 1):
            await self._mint(service)
        assert emitter.call_count == 0, (
            f'{threshold - 1} mints is BELOW the bar; firing here would page an '
            'operator for a batch that never breached'
        )

        await self._mint(service)

        emitter.assert_called_once()
        kwargs = emitter.call_args.kwargs
        assert kwargs['agent_id'] == 'curator-x'
        assert kwargs['project_id'] == _PROJECT
        assert kwargs['count'] == threshold
        assert kwargs['threshold'] == threshold
        assert kwargs['window_seconds'] == service.config.entity_mint.storm_window_seconds

        # A sustained storm keeps minting but does not keep paging.
        for _ in range(threshold):
            await self._mint(service)
        emitter.assert_called_once()

    @pytest.mark.asyncio
    async def test_the_project_root_is_the_referents_own_project(self, stormy):
        """Filed into the project's OWN queue, never the server cwd."""
        service, _clock, emitter = stormy

        for _ in range(service.config.entity_mint.storm_threshold):
            await self._mint(service)

        emitter.assert_called_once()
        args, kwargs = emitter.call_args
        assert (args[0] if args else kwargs.get('project_root')) == '/tmp/df-root'

    @pytest.mark.asyncio
    async def test_resolves_never_count(self, stormy):
        """Only MINTS count. A resolve creates nothing to sweep."""
        service, _clock, emitter = stormy
        service.graphiti.get_nodes_by_exact_name = AsyncMock(
            return_value=[_node('uuid-existing')],
        )

        for _ in range(service.config.entity_mint.storm_threshold * 2):
            await self._mint(service)

        emitter.assert_not_called()

    @pytest.mark.asyncio
    async def test_ambiguous_refusals_never_count(self, stormy):
        service, _clock, emitter = stormy
        service.graphiti.get_nodes_by_exact_name = AsyncMock(
            return_value=[_node('uuid-a'), _node('uuid-b')],
        )

        for _ in range(service.config.entity_mint.storm_threshold * 2):
            await self._mint(service)

        emitter.assert_not_called()

    @pytest.mark.asyncio
    async def test_lock_busy_refusals_never_count(self, stormy):
        """A refused mint minted nothing, so it is not evidence of a mint storm."""
        service, _clock, emitter = stormy
        service.config.entity_mint.lock_timeout_seconds = 0.01
        lock = service.graphiti._identity_lock_for(_PROJECT)
        await lock.acquire()
        try:
            for _ in range(service.config.entity_mint.storm_threshold * 2):
                busy = await self._mint(service)
                assert busy['error_type'] == 'EntityMintLockBusy'
        finally:
            lock.release()

        emitter.assert_not_called()

    @pytest.mark.asyncio
    async def test_threshold_and_window_are_read_live(self, stormy):
        """What EARNS the green-tier classification of both leaves.

        A threshold captured at construction could not observe this in-place
        mutation, which would leave `entity_mint.storm_threshold` restart-only
        while sitting in RELOADABLE_FIELDS as if it were hot-reloadable.
        """
        service, _clock, emitter = stormy

        await self._mint(service)
        await self._mint(service)
        emitter.assert_not_called()

        # reload_config mutates the shared config object IN PLACE, exactly so.
        service.config.entity_mint.storm_threshold = 3
        service.config.entity_mint.storm_window_seconds = 60.0

        await self._mint(service)

        emitter.assert_called_once()
        kwargs = emitter.call_args.kwargs
        assert kwargs['threshold'] == 3, 'the NEW threshold must be what decided'
        assert kwargs['count'] == 3
        assert kwargs['window_seconds'] == 60.0

    @pytest.mark.asyncio
    async def test_mints_outside_the_window_are_evicted(self, stormy):
        """A slow steady trickle never trips it — the window is a real window."""
        service, clock, emitter = stormy
        threshold = service.config.entity_mint.storm_threshold
        window = service.config.entity_mint.storm_window_seconds

        for _ in range(threshold * 3):
            await self._mint(service)
            clock.advance(window)

        emitter.assert_not_called()

    @pytest.mark.asyncio
    async def test_an_emitter_that_raises_never_breaks_the_mint(self, stormy):
        """The mint has ALREADY landed by the time the alarm runs.

        Turning a completed mint into an exception because the COMPLAINT about
        it failed would be strictly worse than losing the signal.
        """
        service, _clock, emitter = stormy
        emitter.side_effect = RuntimeError('escalation queue down')
        threshold = service.config.entity_mint.storm_threshold

        result = None
        for _ in range(threshold):
            result = await self._mint(service)

        # Without this the leg is VACUOUS: none of the assertions below depend
        # on the emitter ever being called, so a silently-dead alarm (a bad
        # threshold read, an absent project_root, a counter that never records)
        # would leave this green while proving nothing about the swallow.
        emitter.assert_called_once()
        assert result is not None
        assert result['minted'] is True, result
        assert result['status'] == 'minted'
        assert service.graphiti.ensure_entity_node.await_count == threshold

    @pytest.mark.asyncio
    async def test_an_unregistered_project_escalates_nothing_and_says_so(
        self, stormy, caplog,
    ):
        """NO FALLBACK: a silent misfile is strictly worse than a logged refusal.

        The escalator files into the affected project's OWN
        ``data/escalations`` queue, resolved from ``_known_projects``. There is
        deliberately no fall back to ``config.taskmaster.project_root``, which
        defaults to ``'.'`` — that would file into the server's cwd, where no
        operator watches, and report success doing it, destroying the evidence
        that the alarm ever fired.

        The ``stormy`` fixture always populates the registry, so this branch
        would otherwise never be exercised.
        """
        service, _clock, emitter = stormy
        service.set_known_projects({})

        with caplog.at_level(
            logging.WARNING, logger='fused_memory.services.memory_service',
        ):
            result = None
            for _ in range(service.config.entity_mint.storm_threshold):
                result = await self._mint(service)

        assert result is not None
        assert result['minted'] is True, (
            'the alarm is additive to the write — an unresolvable project_root '
            'must cost the SIGNAL, never the mint that already landed'
        )
        emitter.assert_not_called()
        text = caplog.text
        assert _PROJECT in text, f'the WARN must name the project, got {text!r}'
        assert 'curator-x' in text, f'the WARN must name the agent, got {text!r}'
        assert '_known_projects' in text, (
            f'the WARN must name what to wire to restore the alarm, got {text!r}'
        )

    @pytest.mark.asyncio
    async def test_a_non_numeric_threshold_skips_the_alarm_without_raising(
        self, stormy,
    ):
        """A corrupt config leaf costs the alarm, never the write."""
        service, _clock, emitter = stormy
        service.config.entity_mint.storm_threshold = 'ten'

        result = await self._mint(service)

        assert result['minted'] is True, result
        emitter.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_non_numeric_window_skips_the_alarm_without_raising(self, stormy):
        service, _clock, emitter = stormy
        service.config.entity_mint.storm_window_seconds = None

        result = await self._mint(service)

        assert result['minted'] is True, result
        emitter.assert_not_called()

    @pytest.mark.asyncio
    async def test_counters_are_keyed_per_agent(self, stormy):
        """Two independently-busy agents must not sum into a false alarm."""
        service, _clock, emitter = stormy
        threshold = service.config.entity_mint.storm_threshold

        # (threshold - 1) each: individually innocent, jointly well past the
        # bar. A shared counter would page here.
        for i in range((threshold - 1) * 2):
            await self._mint(service, agent_id=f'curator-{i % 2}')

        emitter.assert_not_called()
        assert set(service._entity_mint_storm_counters) == {'curator-0', 'curator-1'}

    @pytest.mark.asyncio
    async def test_an_unattributed_mint_labels_as_unattributed(self, stormy):
        """A missing agent_id still counts — there is simply nothing to name it."""
        service, _clock, emitter = stormy

        for _ in range(service.config.entity_mint.storm_threshold):
            await self._mint(service, agent_id=None)

        emitter.assert_called_once()
        assert emitter.call_args.kwargs['agent_id'] == '<unattributed>'
        assert set(service._entity_mint_storm_counters) == {'<unattributed>'}

    @pytest.mark.asyncio
    async def test_dormant_counters_are_evicted(self, stormy):
        """agent_id is caller-supplied and unbounded in cardinality.

        Each counter self-prunes its own deque, but nothing would drop the
        counter OBJECT — so a server running for weeks between restarts would
        accumulate one dead counter per agent it ever saw.
        """
        service, clock, _emitter = stormy
        window = service.config.entity_mint.storm_window_seconds

        await self._mint(service, agent_id='curator-gone')
        assert 'curator-gone' in service._entity_mint_storm_counters

        clock.advance(window * 2)
        await self._mint(service, agent_id='curator-live')

        assert set(service._entity_mint_storm_counters) == {'curator-live'}, (
            'a counter whose window has gone empty must be dropped, not merely '
            'self-pruned'
        )
