"""The ``uuid=`` contract at ``GraphitiBackend.add_episode`` (task 3568).

graphiti_core 0.28.2 treats a caller-supplied ``uuid`` as a LOAD instruction,
never as create-with-this-id (``graphiti_core/graphiti.py:906-920``)::

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

import pytest
from _fm_helpers import backend_with_fake_graphiti
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
