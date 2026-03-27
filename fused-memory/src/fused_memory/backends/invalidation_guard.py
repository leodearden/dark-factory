"""Post-write invalidation guard for Graphiti.

Graphiti's resolve_extracted_edges() uses a broad semantic search with no
entity-node filtering when gathering invalidation candidates. This causes
spurious temporal invalidations when edges from unrelated entities share
surface-level semantic similarity (e.g., 'Task 208 is in-progress' gets
invalidated when writing an episode about 'Task 481 is in-progress').

This module provides a post-write guard that:
1. Detects spurious invalidations by checking entity-UUID overlap
2. Reverses them by clearing invalid_at/expired_at via Cypher

Usage::

    guard = InvalidationGuard(backend)
    restored = await guard.guard(results)  # returns list of restored UUIDs
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol: minimal shape of AddEpisodeResults we need
# ---------------------------------------------------------------------------


class _HasUUID(Protocol):
    uuid: str


class _HasEntityUUIDs(Protocol):
    source_node_uuid: str
    target_node_uuid: str
    expired_at: Any  # datetime | None
    uuid: str
    fact: str


class _HasNodesAndEdges(Protocol):
    nodes: list[Any]
    edges: list[Any]


# ---------------------------------------------------------------------------
# Core detection function
# ---------------------------------------------------------------------------


def detect_spurious_invalidations(results: Any) -> list[Any]:
    """Return edges that were spuriously invalidated by Graphiti.

    An edge is considered spuriously invalidated when:
    1. Its ``expired_at`` field is not None (i.e. it was invalidated by this
       episode write), AND
    2. Neither its ``source_node_uuid`` nor its ``target_node_uuid`` matches
       any UUID in the episode's extracted entity nodes.

    An invalidated edge that shares at least one entity endpoint with the
    episode is considered a *legitimate* invalidation (same entity, different
    state), and is NOT returned.

    Args:
        results: An AddEpisodeResults-like object with:
            - nodes: list of objects with a ``uuid`` attribute
            - edges: list of objects with ``expired_at``, ``source_node_uuid``,
              ``target_node_uuid``, and ``uuid`` attributes

    Returns:
        List of edge objects that are spuriously invalidated (should be restored).
    """
    # Build the set of entity UUIDs in this episode
    episode_entity_uuids: set[str] = {
        node.uuid for node in (results.nodes or []) if node.uuid
    }

    spurious: list[Any] = []
    for edge in results.edges or []:
        # Only consider invalidated edges
        if edge.expired_at is None:
            continue
        # Check if either endpoint is in the episode's entity set
        if (
            edge.source_node_uuid in episode_entity_uuids
            or edge.target_node_uuid in episode_entity_uuids
        ):
            # Legitimate invalidation: same entity, different state
            continue
        # No entity overlap → spurious invalidation
        spurious.append(edge)
    return spurious


# ---------------------------------------------------------------------------
# InvalidationGuard: orchestrates detection + restoration
# ---------------------------------------------------------------------------


class InvalidationGuard:
    """Detects and reverses spurious edge invalidations after Graphiti writes.

    Args:
        backend: A GraphitiBackend instance that provides ``bulk_restore_edge_validity()``.
    """

    def __init__(self, backend: Any) -> None:
        self._backend = backend

    async def guard(self, results: Any) -> list[str]:
        """Run the guard against AddEpisodeResults.

        Detects spurious invalidations and calls bulk_restore_edge_validity()
        to reverse them.

        Args:
            results: AddEpisodeResults from Graphiti's add_episode() call.

        Returns:
            List of edge UUIDs that were restored (empty if none were spurious).
        """
        spurious = detect_spurious_invalidations(results)
        if not spurious:
            return []

        uuids = [e.uuid for e in spurious]
        count = await self._backend.bulk_restore_edge_validity(uuids)
        for edge in spurious:
            logger.warning(
                'Restored spuriously invalidated edge uuid=%s fact=%r',
                edge.uuid,
                getattr(edge, 'fact', ''),
            )
        logger.info('InvalidationGuard: restored %d/%d spurious edges', count, len(uuids))
        return uuids
