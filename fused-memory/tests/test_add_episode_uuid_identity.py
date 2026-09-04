"""Episode-uuid identity across the add_episode write path (task 3561).

The incident: ``MemoryService.add_episode`` minted a fresh uuid4, stamped it
into the durable-queue payload as ``'uuid'``, and ``_execute_graphiti_write``
forwarded it to ``graphiti_core``.  But upstream treats a caller-supplied uuid
as a LOAD instruction, not a create-with-this-id instruction::

    episode = (await EpisodicNode.get_by_uuid(self.driver, uuid)  # LOAD (raises)
               if uuid is not None
               else EpisodicNode(...))                            # CREATE

so a *freshly minted* uuid is unconditionally ``NodeNotFoundError``.  Every
``add_episode`` write failed this way (304 attempts, 0 successes, 0 of 28
historical episode_ids resolving to a real node).

Why the existing suite could never have caught it
-------------------------------------------------
Every graphiti mock in the suite is a permissive ``AsyncMock(return_value=None)``
(test_memory_service.py, test_temporal_context.py, test_temporal_guards.py,
test_journaling_integration.py, test_e2e_durable_queue.py).  A mock that accepts
any ``uuid=`` can never surface a load-or-raise contract.  The missing seam —
and the centre of this module — is :class:`FakeGraphitiClient`, a STATEFUL fake
with a real episodic store that raises on an unknown uuid.

The fake also encodes the two traps this fix must not fall into:

TRAP 1 — "pre-create the EpisodicNode so the caller's uuid resolves".  When the
uuid IS found, graphiti_core saves the STORED node back and the caller's
``episode_body`` serves only as retrieval context.  The fake therefore returns
the stored node and *ignores* the passed body, so a pre-create fix makes every
call succeed while silently discarding the episode content — trading a loud
dead-letter for silent data loss.  The content assertions here fail in that
case.

TRAP 2 — "just delete the payload uuid".  Planning registration read the same
payload key, and its guard block is an ``if/elif`` with no ``else``, so removing
the key alone disables registration with zero signal.  The registration
assertions here fail in that case.
"""

from __future__ import annotations

import logging
import re
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from graphiti_core.errors import NodeNotFoundError

from fused_memory.backends.graphiti_client import GraphitiBackend
from fused_memory.services.memory_service import MemoryService
from fused_memory.services.planned_episode_registry import PlannedEpisodeRegistry

# ---------------------------------------------------------------------------
# The stateful fake — the seam the suite was missing
# ---------------------------------------------------------------------------


class FakeGraphitiClient:
    """Reproduces graphiti_core 0.28.2's real ``uuid=`` load-or-raise contract.

    Deliberately NOT an ``AsyncMock``: the whole defect is that a permissive
    mock accepts any ``uuid=``.  This fake holds a real episodic store and
    behaves as upstream does:

    * ``uuid`` given and UNKNOWN  -> raise :class:`NodeNotFoundError` (the
      literal production failure).
    * ``uuid`` given and KNOWN    -> return the STORED node untouched, ignoring
      the passed ``episode_body`` (TRAP 1: content is silently discarded).
    * ``uuid is None``            -> mint a deterministic uuid, store a node
      carrying the caller's fields, and return an ``AddEpisodeResults``-shaped
      object whose ``.episode`` IS that stored node.

    Uuids are minted deterministically (``real-uuid-N``) rather than randomly:
    tests assert on the exact minted value, and determinism keeps them
    reproducible.
    """

    def __init__(self) -> None:
        self.episodes: dict[str, SimpleNamespace] = {}
        self.calls: list[dict[str, Any]] = []
        # Edges the next add_episode call should attribute to the episode it
        # mints — lets a test simulate extraction provenance without a real LLM.
        self.next_edges: list[Any] = []

    async def add_episode(
        self,
        *,
        name: str = '',
        episode_body: str = '',
        source: Any = None,
        group_id: str = 'main',
        source_description: str = '',
        reference_time: Any = None,
        entity_types: Any = None,
        uuid: str | None = None,
        **kwargs: Any,
    ) -> SimpleNamespace:
        self.calls.append({
            'name': name,
            'episode_body': episode_body,
            'source': source,
            'group_id': group_id,
            'source_description': source_description,
            'reference_time': reference_time,
            'uuid': uuid,
        })

        if uuid is not None:
            stored = self.episodes.get(uuid)
            if stored is None:
                # The exact production failure this task fixes.
                raise NodeNotFoundError(uuid)
            # TRAP 1: upstream saves the STORED node back — the caller's
            # episode_body never lands. Returning it untouched means a
            # "pre-create the node" fix passes the no-raise assertion and
            # FAILS the content assertion, which is the point.
            return self._results_for(stored, edges=[])

        minted = f'real-uuid-{len(self.episodes) + 1}'
        node = SimpleNamespace(
            uuid=minted,
            name=name,
            content=episode_body,
            source=source,
            source_description=source_description,
            group_id=group_id,
            valid_at=reference_time,
            entity_edges=[],
        )
        self.episodes[minted] = node
        edges, self.next_edges = self.next_edges, []
        for edge in edges:
            # Attribute extracted edges to the uuid that actually exists.
            edge.episodes = [minted]
        return self._results_for(node, edges=edges)

    @staticmethod
    def _results_for(episode: SimpleNamespace, *, edges: list[Any]) -> SimpleNamespace:
        """Shape-match ``graphiti_core.graphiti.AddEpisodeResults``."""
        return SimpleNamespace(
            episode=episode,
            episodic_edges=[],
            nodes=[],
            edges=list(edges),
            communities=[],
            community_edges=[],
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _backend_with_fake(mock_config) -> tuple[GraphitiBackend, FakeGraphitiClient]:
    """A REAL GraphitiBackend whose graphiti_core client is the stateful fake.

    Mirrors the ``b._client_for = MagicMock(return_value=mock_client)`` idiom in
    test_temporal_context.py:35-46 — the real backend is kept in the path so the
    production ``uuid=`` forwarding, group canonicalisation and write timeout are
    all exercised; only the graphiti_core client is replaced.
    """
    from _fm_helpers import install_identity_mocks

    backend = GraphitiBackend(mock_config)
    fake = FakeGraphitiClient()
    backend.client = fake  # type: ignore[assignment]
    backend._client_for = MagicMock(return_value=fake)  # type: ignore[method-assign]
    # Real _identity_lock_for already works on a real backend; this additionally
    # no-ops _resolve_or_create_entity so the post-write reconcile sweeps cannot
    # reach a (nonexistent) FalkorDB driver.
    install_identity_mocks(backend)  # type: ignore[arg-type]
    return backend, fake


@pytest.fixture
def svc_and_fake(mock_config):
    """MemoryService with a real GraphitiBackend over the stateful fake."""
    svc = MemoryService(mock_config)
    backend, fake = _backend_with_fake(mock_config)
    svc.graphiti = backend

    svc.mem0 = MagicMock()
    svc.mem0.search = AsyncMock(return_value={'results': []})
    svc.mem0.add = AsyncMock(return_value={'results': [{'id': 'mem0-1'}]})

    svc.durable_queue = MagicMock()
    svc.durable_queue.enqueue = AsyncMock(return_value=1)
    svc.durable_queue.enqueue_batch = AsyncMock(return_value=[1])

    return svc, fake


@pytest_asyncio.fixture
async def svc_fake_registry(svc_and_fake, tmp_path):
    """``svc_and_fake`` plus a REAL PlannedEpisodeRegistry.

    Real, not mocked, because acceptance criterion 2 is that registration
    actually round-trips — ``register()`` was called is not the same claim as
    ``is_planned()`` returns True for the uuid that really exists.
    """
    svc, fake = svc_and_fake
    reg = PlannedEpisodeRegistry(data_dir=tmp_path / 'planned-reg')
    await reg.initialize()
    svc.planned_episode_registry = reg
    yield svc, fake, reg
    await reg.close()


def _enqueued_payload(svc) -> dict[str, Any]:
    """The payload ``add_episode`` handed to the durable queue."""
    svc.durable_queue.enqueue.assert_called_once()
    return svc.durable_queue.enqueue.call_args[1]['payload']


# ---------------------------------------------------------------------------
# step-01: incident reproduction + TRAP 2 guard
# ---------------------------------------------------------------------------


class TestAddEpisodeWriteReachesTheGraph:
    """Acceptance 1 + 3: the write succeeds and the episode content is stored."""

    @pytest.mark.asyncio
    async def test_queued_episode_write_creates_a_node_with_the_passed_content(
        self, svc_and_fake
    ):
        """The full enqueue -> execute path must not raise, and must store the body.

        RED before the fix with the exact production error::

            NodeNotFoundError: node <fresh uuid4> not found

        The content assertion is what closes TRAP 1: a fix that pre-creates the
        node would clear the no-raise assertion but return the pre-created
        (empty-bodied) node here.
        """
        svc, fake = svc_and_fake
        content = 'Task 165 depends on Task 166'

        await svc.add_episode(content=content, project_id='p')
        payload = _enqueued_payload(svc)

        # Must not raise — this is the 304-attempt/0-success production failure.
        await svc._execute_graphiti_write('add_episode', payload)

        assert len(fake.episodes) == 1, (
            'Exactly one episode node should have been created in the graph; '
            f'store holds {list(fake.episodes)}'
        )
        (node,) = fake.episodes.values()
        assert node.content == content, (
            "The stored node must carry the caller's episode body verbatim "
            '(TRAP 1: a pre-created node would silently discard it); got '
            f'{node.content!r}'
        )

    @pytest.mark.asyncio
    async def test_backend_is_never_asked_to_load_a_freshly_minted_uuid(
        self, svc_and_fake
    ):
        """graphiti_core must be called with uuid=None so it takes the CREATE branch."""
        svc, fake = svc_and_fake

        await svc.add_episode(content='some episode', project_id='p')
        await svc._execute_graphiti_write('add_episode', _enqueued_payload(svc))

        assert len(fake.calls) == 1
        assert fake.calls[0]['uuid'] is None, (
            'A caller-supplied uuid means "LOAD this existing episode" upstream; '
            'passing a fresh one is unconditionally NodeNotFoundError. '
            f'Got uuid={fake.calls[0]["uuid"]!r}'
        )


class TestPlanningRegistrationKeysOnTheRealUuid:
    """Acceptance 2 / TRAP 2: registration must follow the uuid that exists."""

    @pytest.mark.asyncio
    async def test_planning_episode_registered_under_the_minted_uuid(
        self, svc_fake_registry
    ):
        """is_planned() must be True for the uuid graphiti_core actually minted.

        This is the assertion that fails *silently* if the fix removes the
        payload uuid without rerouting registration onto ``result.episode.uuid``
        — the registration block is an ``if/elif`` with no ``else``.
        """
        svc, fake, reg = svc_fake_registry

        await svc.add_episode(
            content='We plan to modularize the merge queue',
            project_id='p',
            temporal_context='planning',
        )
        await svc._execute_graphiti_write('add_episode', _enqueued_payload(svc))

        assert len(fake.episodes) == 1
        minted_uuid = next(iter(fake.episodes))
        assert await reg.is_planned(minted_uuid) is True, (
            f'Planning episode should be registered under the minted uuid '
            f'{minted_uuid!r} — the uuid that actually names a graph node. '
            'Registering anything else makes the search filter permanently '
            'vacuous.'
        )


# ---------------------------------------------------------------------------
# step-03: a pre-fix payload executed by post-fix code
# ---------------------------------------------------------------------------

LEGACY_UUID = 'stale-uuid-from-before-the-fix'


def _pre_fix_payload(*, uuid: str | None = LEGACY_UUID) -> dict[str, Any]:
    """A durable-queue payload in the shape ``add_episode`` produced BEFORE the fix.

    Reproduced verbatim (minus the fields added since) rather than derived from
    today's ``add_episode``, because the whole point is that these rows were
    serialized by the OLD code and outlive it.
    """
    payload: dict[str, Any] = {
        'name': 'episode_stale123',
        'content': 'A row enqueued before the fix, drained after it',
        'source': 'text',
        'group_id': 'legacy-group',
        'source_description': '',
        'project_id': 'p',
        '_causation_id': 'causation-legacy-1',
        '_write_op_id': 'write-op-legacy-1',
        'temporal_context': None,
        'reference_time': None,
    }
    if uuid is not None:
        payload['uuid'] = uuid
    return payload


class TestLegacyPayloadDrainedAfterTheFix:
    """The durable queue outlives the deploy that fixed the producer.

    Step-02 stopped PRODUCING ``'uuid'``, but a row enqueued before that deploy
    and executed after it still carries the key.  If execution still forwarded
    whatever the payload holds, those rows would reproduce the identical
    ``NodeNotFoundError`` — the defect would survive its own fix for as long as
    the backlog does.  (This is also what governs task 3584's replay gate.)
    """

    @pytest.mark.asyncio
    async def test_legacy_uuid_payload_still_creates_the_episode(
        self, svc_and_fake, caplog
    ):
        """A pre-fix row must execute successfully and keep its content."""
        svc, fake = svc_and_fake
        payload = _pre_fix_payload()
        expected_content = payload['content']

        with caplog.at_level(logging.WARNING):
            # Must not raise: the stale uuid names no node, so forwarding it
            # would be NodeNotFoundError all over again.
            await svc._execute_graphiti_write('add_episode', payload)

        assert len(fake.episodes) == 1, (
            "The legacy row's episode must still be written, not silently "
            f'dropped; store holds {list(fake.episodes)}'
        )
        (minted_uuid,) = fake.episodes
        node = fake.episodes[minted_uuid]

        assert node.content == expected_content, (
            "The legacy row's content must survive the ignored uuid; got "
            f'{node.content!r}'
        )
        assert minted_uuid != LEGACY_UUID, (
            'The node must carry the uuid graphiti_core minted, never the '
            f'stale payload one; got {minted_uuid!r}'
        )
        assert fake.calls[0]['uuid'] is None, (
            'The backend must be reached with uuid=None even when the payload '
            f'carries a legacy one; got {fake.calls[0]["uuid"]!r}'
        )

    @pytest.mark.asyncio
    async def test_legacy_uuid_payload_warns_naming_the_ignored_key(
        self, svc_and_fake, caplog
    ):
        """Draining a pre-fix backlog must be visible in logs, not only in a graph diff."""
        svc, fake = svc_and_fake

        with caplog.at_level(logging.WARNING):
            await svc._execute_graphiti_write('add_episode', _pre_fix_payload())

        warnings = [
            r.getMessage()
            for r in caplog.records
            if r.levelno >= logging.WARNING and LEGACY_UUID in r.getMessage()
        ]
        assert warnings, (
            'An operator draining pre-fix rows must see a WARNING naming the '
            'ignored uuid; otherwise the only evidence is a graph diff. '
            f'Records seen: {[r.getMessage() for r in caplog.records]}'
        )
        message = warnings[0]
        assert 'uuid' in message, f"The warning must name the ignored key; got {message!r}"
        assert 'legacy-group' in message, (
            f'The warning must name the group_id so the row is locatable; got {message!r}'
        )

    @pytest.mark.asyncio
    async def test_clean_payload_emits_no_legacy_warning(self, svc_and_fake, caplog):
        """The warning must stay a real signal, not per-write noise.

        Every post-fix write goes through this path, so a warning that fired
        unconditionally would be worthless the moment a backlog existed.
        """
        svc, fake = svc_and_fake

        with caplog.at_level(logging.WARNING):
            await svc._execute_graphiti_write(
                'add_episode', _pre_fix_payload(uuid=None)
            )

        offenders = [
            r.getMessage()
            for r in caplog.records
            if r.levelno >= logging.WARNING and 'legacy' in r.getMessage().lower()
        ]
        assert not offenders, (
            'A clean post-fix payload carries no uuid key, so it must not warn; '
            f'got {offenders}'
        )
        assert len(fake.episodes) == 1


# ---------------------------------------------------------------------------
# step-05: no silent fallthrough when the real uuid is unavailable
# ---------------------------------------------------------------------------

#: Results that carry no usable episode uuid. Step-02 moved registration onto
#: ``result.episode.uuid``, which reintroduces TRAP 2's exact failure shape one
#: layer up: each of these leaves ``episode_uuid`` falsy, and the guard block is
#: an ``if/elif`` with no ``else``.
DEGENERATE_RESULTS = [
    pytest.param(None, id='result-is-None'),
    pytest.param(SimpleNamespace(edges=[], nodes=[]), id='result-has-no-episode'),
    pytest.param(
        SimpleNamespace(episode=SimpleNamespace(uuid=None), edges=[], nodes=[]),
        id='episode-uuid-is-None',
    ),
]

CORRELATION_ID = 'corr_degenerate-write-1'


def _planning_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        'name': 'episode_degen01',
        'content': 'We plan to split the merge queue',
        'source': 'text',
        'group_id': 'degenerate-group',
        'source_description': '[temporal:planning] PRD',
        'correlation_id': CORRELATION_ID,
        'temporal_context': 'planning',
    }
    payload.update(overrides)
    return payload


class TestDegenerateResultNeverRegistersSilently:
    """A missed planning registration must be diagnosable from logs alone.

    Planning registration is only ever OBSERVABLE via search results, so a miss
    does not surface as an error — it surfaces, much later, as aspirational PRD
    content leaking into default factual search.  That is why this path must
    never fall through quietly.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize('degenerate_result', DEGENERATE_RESULTS)
    async def test_degenerate_result_warns_instead_of_registering(
        self, svc_fake_registry, caplog, degenerate_result
    ):
        svc, _fake, reg = svc_fake_registry
        svc.graphiti.add_episode = AsyncMock(return_value=degenerate_result)
        register_spy = AsyncMock(wraps=reg.register)
        reg.register = register_spy  # type: ignore[method-assign]

        with caplog.at_level(logging.WARNING):
            # Must not raise — preserves the existing no-crash contract
            # (test_execute_graphiti_write_none_result_no_crash).
            await svc._execute_graphiti_write('add_episode', _planning_payload())

        register_spy.assert_not_called()

        warnings = [
            r.getMessage()
            for r in caplog.records
            if r.levelno >= logging.WARNING and 'degenerate-group' in r.getMessage()
        ]
        assert warnings, (
            'A planning episode that could not be registered must WARN — the '
            'miss is otherwise invisible until aspirational content shows up in '
            'factual search. '
            f'Records seen: {[r.getMessage() for r in caplog.records]}'
        )
        message = warnings[0]
        assert CORRELATION_ID in message, (
            'The warning must name the correlation id so the miss can be tied '
            f'back to the originating write; got {message!r}'
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize('degenerate_result', DEGENERATE_RESULTS)
    async def test_degenerate_result_is_silent_when_not_planning(
        self, svc_fake_registry, caplog, degenerate_result
    ):
        """Registration is genuinely not wanted here, so warning would be noise."""
        svc, _fake, reg = svc_fake_registry
        svc.graphiti.add_episode = AsyncMock(return_value=degenerate_result)
        register_spy = AsyncMock(wraps=reg.register)
        reg.register = register_spy  # type: ignore[method-assign]

        with caplog.at_level(logging.WARNING):
            await svc._execute_graphiti_write(
                'add_episode', _planning_payload(temporal_context=None)
            )

        register_spy.assert_not_called()
        offenders = [
            r.getMessage()
            for r in caplog.records
            if r.levelno >= logging.WARNING and 'degenerate-group' in r.getMessage()
        ]
        assert not offenders, (
            'A non-planning write is not meant to register, so the registration '
            f'path must stay silent; got {offenders}'
        )


# ---------------------------------------------------------------------------
# step-07: AddEpisodeResponse.episode_id demoted to a correlation id
# ---------------------------------------------------------------------------

#: The payload name must stay derived from the RAW uuid4, not the prefixed
#: correlation id — episode names in the graph are unaffected by the demotion.
_EPISODE_NAME_RE = re.compile(r'^episode_[0-9a-f]{8}$')


class TestEpisodeIdIsACorrelationId:
    """``episode_id`` is returned at ENQUEUE time, before any node uuid exists.

    Leaving it silently returning a uuid that matches nothing is the status quo
    this task removes: today a caller who copies ``episode_id`` into
    ``delete_episode`` gets a silent no-op against a nonexistent node.  The
    resolution is to demote it to a correlation id, enforced at RUNTIME by a
    ``corr_`` prefix rather than by a docstring promise — so a ``corr_``-prefixed
    id fails self-describingly instead.
    """

    @pytest.mark.asyncio
    async def test_returned_id_is_prefixed_and_distinct_from_the_minted_uuid(
        self, svc_fake_registry, caplog
    ):
        svc, fake, reg = svc_fake_registry

        with caplog.at_level(logging.INFO):
            response = await svc.add_episode(
                content='We plan to extract the merge queue',
                project_id='p',
                temporal_context='planning',
            )
            payload = _enqueued_payload(svc)
            await svc._execute_graphiti_write('add_episode', dict(payload))

        # (a) A caller or log reader can tell at a glance it is not a node uuid.
        assert response.episode_id.startswith('corr_'), (
            'episode_id is minted before the node exists, so it must announce '
            f'itself as a correlation id; got {response.episode_id!r}'
        )

        # (b) It travels on a key that can never reach graphiti_core.
        assert payload['correlation_id'] == response.episode_id
        assert 'uuid' not in payload, (
            "The enqueue payload must carry no 'uuid' key at all; got "
            f'{payload.get("uuid")!r}'
        )

        # (c) Episode NAMES in the graph are unaffected by the demotion — the
        # name stays derived from the raw uuid4, not from 'corr_...'.
        assert _EPISODE_NAME_RE.match(payload['name']), (
            "Episode name must stay 'episode_<8 hex>' off the RAW uuid4, so the "
            f'prefix does not leak into the graph; got {payload["name"]!r}'
        )

        # (d) The two identities are provably distinct, and the registry holds
        #     the one that names a real node.
        assert len(fake.episodes) == 1
        minted_uuid = next(iter(fake.episodes))
        assert response.episode_id != minted_uuid
        assert await reg.is_planned(minted_uuid) is True, (
            f'The registry must hold the minted uuid {minted_uuid!r}'
        )
        assert await reg.is_planned(response.episode_id) is False, (
            'The correlation id must never be registered as an episode uuid'
        )

    @pytest.mark.asyncio
    async def test_an_info_log_ties_the_correlation_id_to_the_real_uuid(
        self, svc_fake_registry, caplog
    ):
        """The mapping must be recoverable from logs alone.

        This is what tasks 3583/3584 can key off without this task duplicating
        their work: nothing else records which queued write became which node.
        """
        svc, fake, _reg = svc_fake_registry

        with caplog.at_level(logging.INFO):
            response = await svc.add_episode(content='an episode', project_id='p')
            await svc._execute_graphiti_write(
                'add_episode', dict(_enqueued_payload(svc))
            )

        minted_uuid = next(iter(fake.episodes))
        tying = [
            r.getMessage()
            for r in caplog.records
            if response.episode_id in r.getMessage() and minted_uuid in r.getMessage()
        ]
        assert tying, (
            'An INFO log must name BOTH the correlation id '
            f'({response.episode_id!r}) and the real episode uuid '
            f'({minted_uuid!r}); otherwise the mapping is unrecoverable. '
            f'Records seen: {[r.getMessage() for r in caplog.records]}'
        )
