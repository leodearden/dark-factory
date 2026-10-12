"""A stateful stand-in for the graphiti_core client (tasks 3561, 3568).

One purpose: model graphiti_core 0.28.2's ``uuid=`` load-or-raise contract
faithfully enough that ``GraphitiBackend``'s behaviour at that seam becomes
observable.  Every permissive ``AsyncMock`` in the suite accepts any ``uuid=``,
which is exactly why nothing about that contract was ever testable.

It lives here rather than in ``_fm_helpers`` — the suite-wide miscellany most
test modules import — so that the double is shared without either enlarging a
grab-bag module or being duplicated.  ``conftest.py``'s
``make_backend_over_fake_graphiti`` wires it into a real ``GraphitiBackend``;
modules that subclass it or wire it onto a live driver import the class
directly.  The ``tests/_*.py`` name follows the established helper convention
(``_fm_helpers``, ``_git_root_helper``, ``_mock_openai_server``): a module name
unique per subproject, so a ``from _graphiti_fake import X`` cannot collide
with a sibling subproject's helpers when root-level pytest loads several of
them in one process.
"""

from types import SimpleNamespace
from typing import Any

from graphiti_core.errors import NodeNotFoundError


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

