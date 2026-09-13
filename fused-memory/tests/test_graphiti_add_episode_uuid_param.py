"""The ``uuid=`` contract at ``GraphitiBackend.add_episode`` (task 3568).

graphiti_core 0.28.2 treats a caller-supplied ``uuid`` as a LOAD instruction,
never as create-with-this-id (``graphiti_core/graphiti.py::Graphiti.add_episode``)::

    episode = (await EpisodicNode.get_by_uuid(self.driver, uuid)  # LOAD (raises)
               if uuid is not None
               else EpisodicNode(name=..., content=episode_body, ...))  # CREATE

``EpisodicNode.get_by_uuid`` ends ``if len(episodes) == 0: raise
NodeNotFoundError(uuid)``, so a caller who passes a FRESHLY MINTED uuid gets
``node <their own uuid> not found`` — an error that names the caller's own
input and explains nothing.  That self-referential opacity is why the original
incident read as a mysterious retry storm rather than a precondition violation.

Task 3561 settled the contract but stated it as prose in a DIFFERENT module
(``services/memory_service.py::_execute_graphiti_write``, the ANTI-REGRESSION
paragraph: the ``uuid`` parameter is meaningful only for "naming an episode
that ALREADY exists", and "no caller may pass it a freshly minted one ... this
function is the only production caller; keep it that way").  An invariant
enforced by a comment one file away from the seam it governs is enforced by
nothing.  This module mechanises it AT the seam.

Two failures live here, and both are about the same misconception — a caller
believing ``uuid=`` means "create under this id":

  * the uuid does NOT resolve -> upstream's cryptic self-referential error,
    which the backend must translate into one naming the uuid, the group and
    the LOAD-not-CREATE contract, without over-claiming a not-found it cannot
    prove is about that uuid;
  * the uuid DOES resolve -> upstream adopts the stored node and silently
    discards the ``content`` just handed to it.

The seam is ``_fm_helpers.FakeGraphitiClient`` (promoted from task 3561's
test_add_episode_uuid_identity.py): a stateful fake that raises the REAL
``graphiti_core.errors.NodeNotFoundError`` for an unknown uuid and returns the
stored node for a known one.  Every permissive ``AsyncMock`` in the suite
accepts any ``uuid=``, which is exactly why none of this was ever observable.

pytest runs with ``asyncio_mode='strict'``, so every async test needs an
explicit ``@pytest.mark.asyncio``.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest
from _fm_helpers import FakeGraphitiClient, backend_with_fake_graphiti
from graphiti_core.errors import NodeNotFoundError as GraphitiCoreNodeNotFoundError

from fused_memory.backends import graphiti_client

# Never stored by the fake, so its ``uuid=`` branch raises the genuine
# upstream NodeNotFoundError — the literal production failure.
FRESH = 'fresh-uuid-never-stored'
# Distinctive enough that "the error names the group" is a real assertion
# (``'g' in message`` would pass on the word "group" alone), and hyphenated
# so the assertion also pins that it is the CANONICALIZED group that gets
# named — @_canonicalize_group_args maps 'grp-3568' -> 'grp_3568' before the
# method body runs, and naming the raw form would misdirect an operator
# grepping FalkorDB for the graph key.
GROUP = 'grp-3568'
CANONICAL_GROUP = 'grp_3568'


def _discard_warnings(caplog) -> list[str]:
    """WARNINGs about a discarded episode body, by the fact they state.

    Selected on 'content' + 'stored' rather than on record count, so an
    unrelated warning from elsewhere in the backend cannot make a
    no-warning assertion fail for the wrong reason.
    """
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
        and 'content' in r.getMessage().lower()
        and 'stored' in r.getMessage().lower()
    ]


@pytest.fixture
def backend_and_fake(mock_config):
    """A REAL GraphitiBackend over the stateful graphiti_core fake."""
    return backend_with_fake_graphiti(mock_config)


class TestFreshUuidIsRejectedLoudly:
    """A uuid that does not resolve must fail naming the contract it broke."""

    @pytest.mark.asyncio
    async def test_fresh_uuid_raises_the_backend_s_own_not_found_error(
        self, backend_and_fake
    ):
        backend, _fake = backend_and_fake

        with pytest.raises(graphiti_client.NodeNotFoundError) as caught:
            await backend.add_episode(name='n', content='c', group_id=GROUP, uuid=FRESH)

        # The module-local class is a plain Exception, NOT a subclass of the
        # upstream one aliased at graphiti_client.py::GraphitiCoreNodeNotFoundError.
        # Pinned so a later reader can never confuse the translated error with
        # the untranslated one that still escapes on the fail-open path.
        assert not isinstance(caught.value, GraphitiCoreNodeNotFoundError), (
            f'The backend-owned NodeNotFoundError must stay distinguishable '
            f'from graphiti_core\'s; got {type(caught.value).__mro__}.'
        )

    @pytest.mark.asyncio
    async def test_the_error_explains_the_load_not_create_contract(self, backend_and_fake):
        backend, _fake = backend_and_fake

        with pytest.raises(graphiti_client.NodeNotFoundError) as caught:
            await backend.add_episode(name='n', content='c', group_id=GROUP, uuid=FRESH)

        message = str(caught.value)
        assert FRESH in message, f'The error must name the uuid; got {message!r}.'
        assert CANONICAL_GROUP in message, (
            f'The error must name the canonical group; got {message!r}.'
        )
        # The one fact that exists only as prose: a caller-supplied uuid selects
        # the LOAD branch, so it can only name an episode that already exists.
        assert 'already exist' in message.lower(), (
            f'The error must state that uuid= names an episode that must '
            f'ALREADY exist, not one to create; got {message!r}.'
        )

    @pytest.mark.asyncio
    async def test_the_cryptic_upstream_message_no_longer_escapes_and_is_preserved_as_cause(
        self, backend_and_fake
    ):
        backend, _fake = backend_and_fake

        with pytest.raises(graphiti_client.NodeNotFoundError) as caught:
            await backend.add_episode(name='n', content='c', group_id=GROUP, uuid=FRESH)

        # Built from the real class, never from a copied literal: an upstream
        # reword must turn this RED rather than pass against a stale string.
        upstream_message = str(GraphitiCoreNodeNotFoundError(FRESH))
        assert str(caught.value) != upstream_message, (
            f'The self-referential upstream message {upstream_message!r} must '
            f'no longer escape verbatim.'
        )
        assert isinstance(caught.value.__cause__, GraphitiCoreNodeNotFoundError), (
            f'The original diagnosis must survive as __cause__; got '
            f'{caught.value.__cause__!r}.'
        )
        assert str(caught.value.__cause__) == upstream_message


class TestGuardFailsOpen:
    """The guard may only claim the failure it can actually prove.

    ``uuid is not None`` is NOT sufficient grounds to relabel an upstream
    not-found as "your uuid does not exist": graphiti_core also raises
    NodeNotFoundError from entity/edge resolution AFTER the episode loaded
    fine, and attaching the LOAD-not-CREATE explanation to that would be an
    actively misleading diagnosis.  Anything the guard cannot prove is about
    the caller's own uuid must propagate untouched — degrading to exactly
    today's behaviour rather than to a confident wrong claim.
    """

    @staticmethod
    def _backend_raising(mock_config, exc: BaseException):
        """A backend whose fake client raises *exc* from every add_episode.

        Subclassing the promoted fake — rather than adding a raises= knob to
        it — keeps this fail-open probing local to the module that needs it;
        the shared double stays a faithful model of upstream's contract.
        """

        class _RaisingFake(FakeGraphitiClient):
            async def add_episode(self, **kwargs):  # type: ignore[override]
                raise exc

        backend, _fake = backend_with_fake_graphiti(mock_config)
        raising = _RaisingFake()
        backend.client = raising
        backend._client_for = MagicMock(return_value=raising)
        return backend

    @pytest.mark.asyncio
    async def test_not_found_naming_a_different_node_propagates_untouched(self, mock_config):
        backend = self._backend_raising(
            mock_config, GraphitiCoreNodeNotFoundError('some-unrelated-node-uuid')
        )

        with pytest.raises(GraphitiCoreNodeNotFoundError) as caught:
            await backend.add_episode(name='n', content='c', group_id=GROUP, uuid=FRESH)

        assert not isinstance(caught.value, graphiti_client.NodeNotFoundError), (
            f'A not-found about a DIFFERENT node must not be relabelled as '
            f'"your uuid does not exist"; got {str(caught.value)!r}.'
        )

    @pytest.mark.asyncio
    async def test_a_reworded_upstream_message_propagates_untouched(self, mock_config):
        class _RewordedNotFound(GraphitiCoreNodeNotFoundError):
            def __str__(self) -> str:
                return f'episodic node {FRESH} could not be located'

        backend = self._backend_raising(mock_config, _RewordedNotFound(FRESH))

        with pytest.raises(GraphitiCoreNodeNotFoundError) as caught:
            await backend.add_episode(name='n', content='c', group_id=GROUP, uuid=FRESH)

        # Still OUR uuid — but the guard can no longer prove it, so it must
        # decline to claim it. An upstream reword degrades to today's
        # behaviour, never to a wrong explanation.
        assert not isinstance(caught.value, graphiti_client.NodeNotFoundError), (
            f'A reworded upstream message must fail OPEN; got '
            f'{str(caught.value)!r}.'
        )

    @pytest.mark.asyncio
    async def test_uuid_none_never_translates(self, mock_config):
        backend = self._backend_raising(
            mock_config, GraphitiCoreNodeNotFoundError('anything')
        )

        with pytest.raises(GraphitiCoreNodeNotFoundError) as caught:
            await backend.add_episode(name='n', content='c', group_id=GROUP)

        assert not isinstance(caught.value, graphiti_client.NodeNotFoundError), (
            f'Nothing on the uuid=None production path may be reinterpreted '
            f'by this guard; got {str(caught.value)!r}.'
        )


class TestResolvingUuidDiscardsContentLoudly:
    """The same misconception, with the strictly worse outcome.

    When the uuid DOES resolve there is no error at all: graphiti_core adopts
    the STORED node and the ``content`` just handed to it never lands (task
    3561's TRAP 1).  A caller who passes both ``content`` and a resolving
    ``uuid=`` holds exactly the create-with-this-id misconception this module
    exists to correct — so instead of a cryptic error they get silent data
    loss.  That must be loud, but a WARNING rather than a raise: naming an
    already-stored episode is the parameter's one legitimate documented use.
    """

    @staticmethod
    async def _mint(backend) -> str:
        """A resolvable uuid, obtained through the fake's own create path.

        Minting it rather than reaching into ``fake.episodes`` keeps the test
        honest: the uuid it later passes is one production really could hold.
        """
        result = await backend.add_episode(
            name='n', content='original text', group_id=GROUP
        )
        return result.episode.uuid

    @pytest.mark.asyncio
    async def test_resolving_uuid_returns_the_stored_episode_not_the_passed_content(
        self, backend_and_fake
    ):
        backend, fake = backend_and_fake
        stored_uuid = await self._mint(backend)

        result = await backend.add_episode(
            name='n', content='replacement text', group_id=GROUP, uuid=stored_uuid
        )

        # The pass-through is intact — the uuid really does reach upstream.
        assert fake.calls[-1]['uuid'] == stored_uuid
        # ...and upstream really did discard the caller's content. This
        # characterises the upstream behaviour, so the warning below is
        # provably about something real rather than hypothetical.
        assert result.episode.content == 'original text', (
            f'Expected upstream to adopt the STORED episode body; got '
            f'{result.episode.content!r}.'
        )

    @pytest.mark.asyncio
    async def test_a_warning_names_the_uuid_and_the_discarded_content(
        self, backend_and_fake, caplog
    ):
        backend, _fake = backend_and_fake
        stored_uuid = await self._mint(backend)

        with caplog.at_level(logging.WARNING, logger=graphiti_client.logger.name):
            await backend.add_episode(
                name='n', content='replacement text', group_id=GROUP, uuid=stored_uuid
            )

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        matching = [r for r in warnings if stored_uuid in r.getMessage()]
        assert matching, (
            f'A resolving uuid= silently discards content — that must warn, '
            f'naming the uuid. Warnings seen: {[r.getMessage() for r in warnings]}.'
        )
        message = matching[0].getMessage().lower()
        assert 'content' in message, (
            f'The warning must name what was lost; got {matching[0].getMessage()!r}.'
        )
        assert 'not' in message and 'stored' in message, (
            f'The warning must say the content was NOT stored; got '
            f'{matching[0].getMessage()!r}.'
        )

    @pytest.mark.asyncio
    async def test_no_warning_when_uuid_is_none(self, backend_and_fake, caplog):
        backend, _fake = backend_and_fake

        with caplog.at_level(logging.WARNING, logger=graphiti_client.logger.name):
            await backend.add_episode(name='n', content='c', group_id=GROUP)

        assert not _discard_warnings(caplog), (
            f'The uuid=None production path must gain no log noise; got '
            f'{_discard_warnings(caplog)}.'
        )

    @pytest.mark.asyncio
    async def test_no_warning_when_no_content_was_offered(self, backend_and_fake, caplog):
        backend, _fake = backend_and_fake
        stored_uuid = await self._mint(backend)

        with caplog.at_level(logging.WARNING, logger=graphiti_client.logger.name):
            await backend.add_episode(
                name='n', content='', group_id=GROUP, uuid=stored_uuid
            )

        # A caller offering nothing to store is not making the
        # create-with-this-id mistake, so there is nothing to warn about.
        assert not _discard_warnings(caplog), (
            f'An empty content= must not be nagged; got {_discard_warnings(caplog)}.'
        )
