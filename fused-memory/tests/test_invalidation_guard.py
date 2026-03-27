"""Tests for the post-write invalidation guard.

The guard detects spurious edge invalidations that Graphiti's broad semantic
search may produce, and reverses them by clearing invalid_at/expired_at.

Organised by concern:
  TestDetectSpuriousInvalidations    — step-1 through step-6
  TestRestoreEdgeValidity            — step-7 / step-8
  TestBulkRestoreEdgeValidity        — step-9 / step-10
  TestInvalidationGuardOrchestration — step-11 through step-16
  TestGuardIntegration               — step-17 through step-22
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fused_memory.backends.graphiti_client import GraphitiBackend
from fused_memory.backends.invalidation_guard import (
    InvalidationGuard,
    detect_spurious_invalidations,
)


# ---------------------------------------------------------------------------
# Local mock helpers
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass
class FakeNode:
    """Minimal stand-in for an EntityNode."""

    uuid: str
    name: str = ''


@dataclass
class FakeEdge:
    """Minimal stand-in for an EntityEdge."""

    uuid: str
    fact: str = ''
    source_node_uuid: str = ''
    target_node_uuid: str = ''
    expired_at: datetime | None = None


@dataclass
class FakeAddEpisodeResults:
    """Minimal stand-in for AddEpisodeResults."""

    nodes: list[FakeNode] = field(default_factory=list)
    edges: list[FakeEdge] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Step-1 / Step-2: detect_spurious_invalidations — no-overlap → flagged
# ---------------------------------------------------------------------------


class TestDetectSpuriousInvalidations:
    """detect_spurious_invalidations() identifies invalidated edges with no entity overlap."""

    def test_invalidated_edge_with_no_entity_overlap_is_flagged(self):
        """Edge with expired_at set and no shared entity UUID is flagged as spurious."""
        # Episode about Task-481 → node uuid=node-481
        nodes = [FakeNode(uuid='node-481', name='Task 481')]
        # Edge about Task-208, no connection to episode entities
        edges = [
            FakeEdge(
                uuid='edge-208',
                fact='Task 208 has status in-progress',
                source_node_uuid='node-208',
                target_node_uuid='node-status',
                expired_at=_now(),
            )
        ]
        results = FakeAddEpisodeResults(nodes=nodes, edges=edges)
        spurious = detect_spurious_invalidations(results)
        assert len(spurious) == 1
        assert spurious[0].uuid == 'edge-208'

    def test_multiple_spurious_edges_all_flagged(self):
        """Multiple invalidated edges with no entity overlap are all flagged."""
        nodes = [FakeNode(uuid='node-A', name='Entity A')]
        edges = [
            FakeEdge(
                uuid='edge-X',
                fact='Edge X fact',
                source_node_uuid='node-X1',
                target_node_uuid='node-X2',
                expired_at=_now(),
            ),
            FakeEdge(
                uuid='edge-Y',
                fact='Edge Y fact',
                source_node_uuid='node-Y1',
                target_node_uuid='node-Y2',
                expired_at=_now(),
            ),
        ]
        results = FakeAddEpisodeResults(nodes=nodes, edges=edges)
        spurious = detect_spurious_invalidations(results)
        uuids = {e.uuid for e in spurious}
        assert uuids == {'edge-X', 'edge-Y'}

    def test_empty_nodes_all_invalidated_edges_flagged(self):
        """When episode has no entities, all invalidated edges are spurious."""
        nodes: list[FakeNode] = []
        edges = [
            FakeEdge(
                uuid='edge-Z',
                fact='Some fact',
                source_node_uuid='node-1',
                target_node_uuid='node-2',
                expired_at=_now(),
            )
        ]
        results = FakeAddEpisodeResults(nodes=nodes, edges=edges)
        spurious = detect_spurious_invalidations(results)
        assert len(spurious) == 1

    # -----------------------------------------------------------------
    # Step-3 / Step-4: edges with entity overlap are NOT flagged
    # -----------------------------------------------------------------

    def test_invalidated_edge_sharing_source_entity_not_flagged(self):
        """Invalidated edge sharing source_node_uuid with episode is NOT spurious."""
        nodes = [FakeNode(uuid='node-task', name='Task 481')]
        edges = [
            FakeEdge(
                uuid='edge-old-status',
                fact='Task 481 has status pending',
                source_node_uuid='node-task',  # SAME as episode entity
                target_node_uuid='node-status',
                expired_at=_now(),
            )
        ]
        results = FakeAddEpisodeResults(nodes=nodes, edges=edges)
        spurious = detect_spurious_invalidations(results)
        assert spurious == []

    def test_invalidated_edge_sharing_target_entity_not_flagged(self):
        """Invalidated edge sharing target_node_uuid with episode is NOT spurious."""
        nodes = [FakeNode(uuid='node-task', name='Task 481')]
        edges = [
            FakeEdge(
                uuid='edge-old-status',
                fact='Some project is about Task 481',
                source_node_uuid='node-project',
                target_node_uuid='node-task',  # SAME as episode entity
                expired_at=_now(),
            )
        ]
        results = FakeAddEpisodeResults(nodes=nodes, edges=edges)
        spurious = detect_spurious_invalidations(results)
        assert spurious == []

    def test_non_invalidated_edges_ignored_even_without_entity_overlap(self):
        """Edges without expired_at are NOT flagged regardless of entity overlap."""
        nodes = [FakeNode(uuid='node-A')]
        edges = [
            FakeEdge(
                uuid='edge-active',
                fact='Some active fact',
                source_node_uuid='node-X',
                target_node_uuid='node-Y',
                expired_at=None,  # NOT invalidated
            )
        ]
        results = FakeAddEpisodeResults(nodes=nodes, edges=edges)
        spurious = detect_spurious_invalidations(results)
        assert spurious == []

    def test_empty_edges_returns_empty_list(self):
        """No edges → empty list returned."""
        results = FakeAddEpisodeResults(nodes=[], edges=[])
        assert detect_spurious_invalidations(results) == []

    # -----------------------------------------------------------------
    # Step-5 / Step-6: non-invalidated edges pass through
    # -----------------------------------------------------------------

    def test_mix_of_invalidated_and_active_only_flags_invalidated(self):
        """Mix of active and invalidated edges: only expired ones without overlap flagged."""
        nodes = [FakeNode(uuid='node-A')]
        edges = [
            FakeEdge(
                uuid='edge-active',
                fact='Active fact',
                source_node_uuid='node-A',
                target_node_uuid='node-B',
                expired_at=None,
            ),
            FakeEdge(
                uuid='edge-spurious',
                fact='Spurious invalidated',
                source_node_uuid='node-X',
                target_node_uuid='node-Y',
                expired_at=_now(),
            ),
            FakeEdge(
                uuid='edge-legit',
                fact='Legitimate invalidation',
                source_node_uuid='node-A',  # same as episode
                target_node_uuid='node-Z',
                expired_at=_now(),
            ),
        ]
        results = FakeAddEpisodeResults(nodes=nodes, edges=edges)
        spurious = detect_spurious_invalidations(results)
        assert len(spurious) == 1
        assert spurious[0].uuid == 'edge-spurious'
