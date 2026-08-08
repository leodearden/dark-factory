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
