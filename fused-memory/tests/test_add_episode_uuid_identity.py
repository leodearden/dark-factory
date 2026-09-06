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

import contextlib
import logging
import os
import re
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from _fm_helpers import FALKOR_HOST, FALKOR_PORT, falkor_skipif, unique_graph_name
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
        # Edges retained per group, so ``search`` returns what ``add_episode``
        # actually attributed (step-09). Keeping them keyed by group is what
        # makes the group-scoping identity — registration writes
        # ``payload['group_id']``, the filter reads ``scope.graphiti_group_id``
        # — an assertion rather than an assumption.
        self.edges_by_group: dict[str, list[Any]] = {}

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
        self.edges_by_group.setdefault(group_id, []).extend(edges)
        return self._results_for(node, edges=edges)

    async def search(
        self,
        *,
        query: str = '',
        group_ids: list[str] | None = None,
        num_results: int = 10,
        **kwargs: Any,
    ) -> list[Any]:
        """Return the edges ``add_episode`` attributed, scoped by group.

        Stateful for the same reason ``add_episode`` is: the point of step-09
        is that the uuid the registry holds and the uuid on the edge's
        ``episodes`` provenance are the SAME minted uuid.  A canned
        ``AsyncMock(return_value=[MockEdge(episodes=['whatever'])])`` would let
        the test assert that identity into existence instead of observing it.
        """
        out: list[Any] = []
        for gid in group_ids or list(self.edges_by_group):
            out.extend(self.edges_by_group.get(gid, []))
        return out[:num_results]

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
    # GraphitiBackend.search resolves a per-group driver CLONE off the real
    # FalkorDB driver before delegating to the client, and there is no real
    # driver here. The clone is only ever forwarded to client.search(driver=...),
    # which the fake ignores — so stubbing it keeps the real backend (its
    # @_canonicalize_group_args entry, group_ids plumbing and read timeout) in
    # the path while making the read seam reachable without a live graph.
    backend._driver_for = MagicMock(return_value=None)  # type: ignore[method-assign]
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


def _pre_fix_payload(
    *, uuid: str | None = LEGACY_UUID, temporal_context: str | None = None
) -> dict[str, Any]:
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
        'temporal_context': temporal_context,
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
        # The stated reason execution POPS rather than GETS: `_dual_write_callback`
        # receives this same dict, so a key left in place would leak onward to a
        # consumer that has no reason to expect a dead uuid. Pinned here because
        # `get` would satisfy every other assertion in this test.
        assert 'uuid' not in payload, (
            "The legacy key must be POPPED from the payload, not merely ignored, "
            f'so it cannot leak onward; still holds {payload.get("uuid")!r}'
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
        # The QUOTED spelling, not the bare word: `warnings` was filtered on
        # LEGACY_UUID, which itself contains the substring 'uuid', so
        # `'uuid' in message` could never fail and pinned nothing.
        assert "'uuid'" in message, (
            f'The warning must name the ignored key by name; got {message!r}'
        )
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

        # 'Ignoring legacy', not the bare word 'legacy': the payload's group_id
        # is literally 'legacy-group', so any unrelated warning naming the group
        # would fail this test with a thoroughly misleading message.
        offenders = [
            r.getMessage()
            for r in caplog.records
            if r.levelno >= logging.WARNING
            and 'ignoring legacy' in r.getMessage().lower()
        ]
        assert not offenders, (
            'A clean post-fix payload carries no uuid key, so it must not warn; '
            f'got {offenders}'
        )
        assert len(fake.episodes) == 1

    @pytest.mark.asyncio
    async def test_legacy_planning_row_registers_the_minted_uuid_not_the_stale_one(
        self, svc_fake_registry
    ):
        """The one cell where the two guards interact, and the only one that can
        resurrect the ORIGINAL defect.

        A pre-fix backlog row that is also a planning episode exercises the
        legacy-uuid pop and the registration keying at once.  A regression that
        re-reads ``payload['uuid']`` for registration passes every other test in
        this module — the write still succeeds, the warning still fires — while
        registering a uuid that names no node, which is exactly the vacuous
        registration task 3561 exists to end.
        """
        svc, fake, reg = svc_fake_registry

        await svc._execute_graphiti_write(
            'add_episode', _pre_fix_payload(temporal_context='planning')
        )

        assert len(fake.episodes) == 1
        (minted_uuid,) = fake.episodes
        assert minted_uuid != LEGACY_UUID

        assert await reg.is_planned(minted_uuid) is True, (
            'A legacy planning row must register under the uuid graphiti_core '
            f'minted ({minted_uuid!r}); the search filter matches registered '
            'uuids against edge episode provenance, so anything else is inert.'
        )
        assert await reg.is_planned(LEGACY_UUID) is False, (
            'The stale payload uuid names no graph node and must never reach '
            'the registry — registering it is the pre-3561 vacuous behaviour.'
        )


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

    The status quo this task removes is a bare uuid4 that matches no node while
    looking exactly like one that does: a caller who copied it into
    ``delete_episode`` got a ``NodeNotFoundError`` naming a plausible-looking
    uuid, with nothing to say the id had never been resolvable at all.  The
    resolution is to demote it to a correlation id carrying a ``corr_`` prefix
    — a legibility marker, not a validated guard (nothing anywhere rejects a
    ``corr_`` id, and ``remove_episode`` raised for the bare uuid4 just as
    loudly).  The tests below pin the prefix, the ``correlation_id`` payload
    key and the INFO tie-back, because those are what a regression would
    silently undo.
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

    @pytest.mark.asyncio
    async def test_no_mapping_log_for_the_fallthrough_operations(
        self, svc_and_fake, caplog
    ):
        """The mapping channel stays one line per EPISODE, not per Graphiti write.

        ``_execute_graphiti_write`` is the fallthrough dispatch target for every
        queued operation other than mem0's, so ``add_memory_graphiti`` lands here
        too — carrying no ``correlation_id``.  An ungated emit would file those
        under ``add_episode ... correlation_id=None``, mislabelling them and
        diluting the exact channel tasks 3583/3584 have to grep.
        """
        svc, fake = svc_and_fake

        with caplog.at_level(logging.INFO):
            await svc._execute_graphiti_write(
                'add_memory_graphiti', _pre_fix_payload(uuid=None)
            )

        assert len(fake.episodes) == 1, 'the write itself must still happen'
        stray = [
            r.getMessage()
            for r in caplog.records
            if 'write executed' in r.getMessage()
        ]
        assert not stray, (
            'Only add_episode rows carry a correlation id, so only they may '
            f'emit the mapping line; got {stray}'
        )


# ---------------------------------------------------------------------------
# step-09: acceptance criterion 2 with teeth — the planning episode is actually
# FILTERED OUT of default search, not merely registered
# ---------------------------------------------------------------------------


async def _execute_planning_round_trip(
    svc, fake, *, content: str, edge_fact: str, temporal_context: str | None
) -> tuple[str, Any]:
    """Enqueue + execute one episode carrying one extracted edge.

    Returns ``(minted_uuid, edge)``.  The edge is attributed by the fake to the
    uuid graphiti_core actually minted, exactly as upstream attributes real
    extraction provenance.
    """
    from _fm_helpers import MockEdge

    edge = MockEdge(fact=edge_fact, uuid='edge-under-test')
    fake.next_edges = [edge]

    await svc.add_episode(
        content=content, project_id='p', temporal_context=temporal_context
    )
    await svc._execute_graphiti_write('add_episode', _enqueued_payload(svc))

    assert len(fake.episodes) == 1
    return next(iter(fake.episodes)), edge


async def _graphiti_only_search(svc, query: str, **kwargs: Any) -> list[Any]:
    """``svc.search`` restricted to the seam under test.

    ``stores=['graphiti']`` takes ``ReadRouter.route``'s override branch, so no
    classification (and no LLM) runs, and ``_search_mem0`` is never scheduled.
    ``anchor_topics=False`` skips the topic-anchoring pin, whose Qdrant
    round-trip is unrelated to planned-episode filtering.
    """
    return await svc.search(
        query, project_id='p', stores=['graphiti'], anchor_topics=False, **kwargs
    )


class TestPlanningEpisodeIsFilteredFromSearch:
    """Acceptance 2, end to end: registered under the uuid the FILTER reads.

    The three tests in :class:`TestPlanningRegistrationKeysOnTheRealUuid` prove
    ``is_planned(minted_uuid)``.  That is necessary but not sufficient: before
    this task, registration stored a uuid naming no graph node, so
    ``_search_graphiti`` compared ``get_planned_uuids(...)`` against edge
    episode provenance and never matched — the criterion passed as a
    registration fact while failing as a *filtering* fact, silently, which is
    exactly the shape the task description calls out.  These tests close that
    gap by asserting the registry key and the edge's provenance are literally
    the same string, and then that the filter acts on it.
    """

    @pytest.mark.asyncio
    async def test_planning_edge_is_excluded_from_default_search(
        self, svc_fake_registry
    ):
        svc, fake, reg = svc_fake_registry

        minted_uuid, edge = await _execute_planning_round_trip(
            svc, fake,
            content='We plan to modularize the merge queue',
            edge_fact='PRD: MergeQueue is decomposed into a planner and a worker',
            temporal_context='planning',
        )

        # THE IDENTITY, asserted directly — this is the whole fix. Registration
        # keys on result.episode.uuid and provenance carries the same minted
        # uuid, so the set-membership test in _search_graphiti can match.
        assert edge.episodes == [minted_uuid]
        assert await reg.get_planned_uuids('p') == {minted_uuid}, (
            'The registry must hold exactly the minted uuid, under the group the '
            'filter reads (scope.graphiti_group_id). Registration writes '
            "payload['group_id']; if those two ever diverge the filter silently "
            'sees an empty planned set.'
        )

        results = await _graphiti_only_search(svc, 'merge queue')

        assert results == [], (
            'A planning episode’s edge must not appear in default search '
            f'results. Got {[r.content for r in results]}'
        )

    @pytest.mark.asyncio
    async def test_planning_edge_surfaces_and_is_marked_with_include_planned(
        self, svc_fake_registry
    ):
        svc, fake, _reg = svc_fake_registry

        minted_uuid, _edge = await _execute_planning_round_trip(
            svc, fake,
            content='We plan to modularize the merge queue',
            edge_fact='PRD: MergeQueue is decomposed into a planner and a worker',
            temporal_context='planning',
        )

        results = await _graphiti_only_search(
            svc, 'merge queue', include_planned=True
        )

        assert len(results) == 1, (
            'include_planned=True must surface the planning edge that the '
            f'default search excludes. Got {results!r}'
        )
        assert results[0].metadata.get('planned') is True, (
            "A surfaced planning edge must be marked metadata['planned'] = True "
            'so the caller can tell aspiration from fact; got '
            f'{results[0].metadata!r}'
        )
        assert results[0].provenance == [minted_uuid], (
            'The surfaced edge must carry the minted uuid as its provenance — '
            'that is the identity the filter matched on.'
        )

    @pytest.mark.asyncio
    async def test_non_planning_edge_is_not_excluded(self, svc_fake_registry):
        """The filter must be narrow: only planning episodes are withheld.

        Without this, a fix that registered EVERY episode would pass both tests
        above while making default search return nothing at all.
        """
        svc, fake, reg = svc_fake_registry

        minted_uuid, edge = await _execute_planning_round_trip(
            svc, fake,
            content='The merge queue was modularized in commit abc123',
            edge_fact='MergeQueue was decomposed into a planner and a worker',
            temporal_context=None,
        )

        assert edge.episodes == [minted_uuid]
        assert await reg.get_planned_uuids('p') == set(), (
            'A non-planning episode must never be registered as planned.'
        )

        results = await _graphiti_only_search(svc, 'merge queue')

        assert len(results) == 1, (
            'A factual episode’s edge must survive the default filter; got '
            f'{results!r}'
        )
        assert results[0].metadata.get('planned') is not True
        assert results[0].provenance == [minted_uuid]


# ---------------------------------------------------------------------------
# step-11: acceptance criterion 1 verified DIRECTLY in FalkorDB
#
# Everything above runs against the fake. The fake is a faithful transcription
# of graphiti_core 0.28.2's contract, but it is still a transcription — and the
# defect this task fixes is precisely a case where the suite's model of the
# backend and the backend disagreed for five months. This is the one test that
# asks the real graph.
#
# Deselected by default (pyproject.toml's `-m 'not integration'`), so it
# burdens neither CI nor a routine local run with OPENAI credits.
# ---------------------------------------------------------------------------

_LIVE_CONTENT = 'Task 3561 fixed the self-referential add_episode NodeNotFoundError.'


async def _live_episodic_count(graph_name: str, episode_uuid: str) -> tuple[int, str | None]:
    """Return ``(count, content)`` for the Episodic node named by *episode_uuid*.

    READ-ONLY by construction: issued through FalkorDB's ``ro_query``, which the
    server itself refuses to run a mutating statement on.  So this verification
    cannot alter the graph it is inspecting — a real risk here, since the whole
    claim under test is "a node with this uuid EXISTS", and a verification that
    could write one would prove nothing.

    DEVIATION from step-11 as written, recorded here and in esc-3561-10: the
    step names ``_fm_helpers.assert_ro_query_only`` for this. That helper
    installs a MagicMock over ``backend._driver._get_graph`` and asserts a
    backend METHOD used ro_query; against a live graph it would return canned
    rows and make this test vacuous — the exact opposite of its purpose. Its
    stated rationale ("so the test cannot mutate the graph it is inspecting")
    is what ``ro_query`` delivers directly, so the rationale is honoured and
    only the mechanism differs.
    """
    from falkordb.asyncio import FalkorDB

    client = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
    try:
        graph = client.select_graph(graph_name)
        result = await graph.ro_query(
            'MATCH (e:Episodic {uuid: $uuid}) RETURN count(e), e.content',
            {'uuid': episode_uuid},
        )
        rows = list(getattr(result, 'result_set', None) or [])
        if not rows:
            return 0, None
        count, content = rows[0][0], rows[0][1]
        return int(count), content
    finally:
        await client.aclose()


@falkor_skipif()
@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'),
    reason='a real add_episode extracts entities via OpenAI',
)
# A real add_episode is an LLM round trip per extraction pass, well past the
# suite's 60s default. Under `timeout_method = "thread"` an under-budget
# timeout os._exit(1)s the whole xdist worker, so it would read as an
# infrastructure crash rather than a slow test.
@pytest.mark.timeout(300)
@pytest.mark.asyncio
async def test_the_write_creates_a_real_episodic_node_in_falkordb(mock_config):
    """Acceptance 1 + 3, against the REAL graph: the node exists AND carries the body.

    The task description's measurement was that 0 of 28 historical episode_ids
    resolved to a node — ``MATCH (e:Episodic {uuid: $uuid}) RETURN count(e)``
    returned 0 every time.  This asserts it returns 1.

    The content assertion is not redundant with it: it is what closes TRAP 1
    end-to-end.  A "pre-create the EpisodicNode so the caller's uuid resolves"
    fix would satisfy the count assertion — the node exists, by construction —
    and fail this one, because when graphiti_core FINDS the uuid it saves the
    stored node back and the caller's ``episode_body`` never lands.
    """
    from falkordb.asyncio import FalkorDB

    graph_name = unique_graph_name('3561_episode_uuid')

    config = mock_config.model_copy(deep=True)
    # mock_config's api_keys are the literal string 'test-key'. Clearing them
    # makes the OpenAI clients fall back to the real OPENAI_API_KEY env var —
    # the idiom test_recon_dedup_premise.py:130-133 established for the two
    # tests in this suite that need real embeddings.
    config.llm.providers.openai.api_key = None
    config.embedder.providers.openai.api_key = None
    config.graphiti.falkordb.uri = f'redis://{FALKOR_HOST}:{FALKOR_PORT}'

    backend = GraphitiBackend(config)
    # skip_maintenance=True is load-bearing, not an optimisation: the default
    # path enumerates EVERY graph on the server and runs an index build plus a
    # dup-uuid-edge REPAIR (a write) over each. A test must never sweep the
    # real project graphs sharing this FalkorDB instance.
    await backend.initialize(skip_maintenance=True)
    try:
        # The scratch graph is virgin, and graphiti's entity-dedup search needs
        # its indices. _ensure_indices is a deliberate no-op (task 3707);
        # ensure_indices is the real provisioning path.
        await backend.ensure_indices(group_id=graph_name)

        svc = MemoryService(config)
        svc.graphiti = backend
        svc.mem0 = MagicMock()
        svc.mem0.add = AsyncMock(return_value={'results': []})
        svc.durable_queue = MagicMock()
        svc.durable_queue.enqueue = AsyncMock(return_value=1)

        # The real production route: enqueue, then execute the queued payload.
        await svc.add_episode(content=_LIVE_CONTENT, project_id=graph_name)
        result = await svc._execute_graphiti_write(
            'add_episode', _enqueued_payload(svc)
        )

        real_uuid = result.episode.uuid
        assert real_uuid, 'graphiti_core must report the uuid it minted'

        count, stored_content = await _live_episodic_count(graph_name, real_uuid)

        assert count == 1, (
            f'MATCH (e:Episodic {{uuid: {real_uuid!r}}}) RETURN count(e) must be '
            f'1. This query returned 0 for all 28 historical episode_ids — the '
            f'measurement this task exists to correct. Got {count}.'
        )
        assert stored_content == _LIVE_CONTENT, (
            'The persisted node must carry the content that was passed. A '
            'pre-created-node fix (TRAP 1) satisfies the count assertion above '
            f'and fails here. Got {stored_content!r}'
        )
    finally:
        await backend.close()
        cleanup = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
        try:
            with contextlib.suppress(Exception):
                await cleanup.select_graph(graph_name).delete()
        finally:
            await cleanup.aclose()
