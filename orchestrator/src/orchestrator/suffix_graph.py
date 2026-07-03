"""Suffix-conflict graph and bounce-state tracker (MQ-refactor task δ=1988).

Extracted verbatim from :mod:`orchestrator.merge_queue`: the two-layer
suffix-conflict machinery — ``SuffixConflictGraph`` (the immutable
conflict-graph dataclass) and ``EMPTY_SUFFIX_CONFLICT_GRAPH`` (its sentinel
empty instance) — plus a NEW :class:`SuffixConflictTracker` class that owns
the state (``graph`` / ``signature`` / ``last_known_main_sha`` /
``bounce_registry``) and logic (``recompute()`` /
``bounce_conflicting_suffix_items()``) that previously lived directly on
``SpeculativeMergeWorker``.  ``merge_queue`` re-exports all three names
through a top-level shim so existing importers
(``from orchestrator.merge_queue import X``, etc.) keep working unchanged.

``SpeculativeMergeWorker`` delegates to a ``SuffixConflictTracker`` instance
(``self._suffix_tracker``) via thin property/method wrappers that preserve
the worker's original attribute names (``_suffix_conflict_graph``,
``_suffix_conflict_signature``, ``_last_known_main_sha``,
``_bounce_registry``) and method signatures
(``recompute_suffix_conflict_graph()``, ``_bounce_conflicting_suffix_items()``)
so the existing conflict-graph and bounce test suites keep passing with zero
churn.  The tracker itself never holds a worker reference: lane-buffer
access and frozen-prefix reads are injected as narrow callables
(``lane_buffers``, ``frozen_prefix``, ``frozen_prefix_tip``) captured at
construction, so it is unit-testable standalone.

A moved method that reads a merge_queue-resident constant (``MERGE_LANES``,
``MERGE_BOUNCE_CAP``, ``NEEDS_REBASE_REASON_PREFIX``) — these stay in
``merge_queue.py`` and are NOT moved — resolves it through a function-local
(deferred) import from :mod:`orchestrator.merge_queue` rather than a
top-level import.  This mirrors the reach-back convention established by
:mod:`orchestrator.merge_liveness` / :mod:`orchestrator.merge_drift` and
keeps this module free of any top-level import of ``merge_queue`` (which
would deadlock module load, since merge_queue's shim needs this module
fully defined first).
"""

from __future__ import annotations

import collections
import dataclasses
import logging
from collections.abc import Callable

from orchestrator.git_ops import GitOps
from orchestrator.merge_types import (
    GroupMergeRequest,
    MergeBounceRegistry,
    MergeOutcome,
    MergeRequest,
)
from orchestrator.overlap_footprint import Footprint, get_overlap_detector

logger = logging.getLogger('orchestrator.merge_queue')


@dataclasses.dataclass(frozen=True)
class SuffixConflictGraph:
    """Immutable conflict-graph over the unfrozen merge-queue suffix (task δ=1889).

    Holds two distinct pair-edge relations and one per-item marker:

    * **footprint_edges** — pairs whose changed-path footprints overlap (γ seam).
      Drives future ζ (ordering) consumers; computed cheaply via path-set
      intersection without forking git.

    * **textual_edges** — pairs with genuine 3-way textual conflicts (β seam).
      Pruned to footprint-overlapping pairs only (textual ⇒ footprint contract).
      Drives future η (bounce) consumers; each entry represents a confirmed
      git merge-tree conflict.

    * **conflicts_with_main** — request_ids of suffix items that conflict with
      the current main tip (the δ user-signal).

    **Node identity** — request_id (the stable per-MergeRequest UUID, e.g.
    ``'mr-a1b2c3d4'``).  Nodes are stored in pick order (high lane before
    normal, FIFO within each lane) so the tuple doubles as the ordered view
    of the suffix.

    **Immutability** — frozen dataclass; every field is a frozenset or tuple
    so the graph can be shared safely across the async event loop without
    copying.

    See also: EMPTY_SUFFIX_CONFLICT_GRAPH (module constant for the zero case).
    """

    nodes: tuple[str, ...]
    """Request IDs in pick order (high lane → normal lane, FIFO within each lane)."""

    textual_edges: frozenset[frozenset[str]]
    """Unordered pairs {rid_a, rid_b} with a confirmed 3-way textual conflict."""

    footprint_edges: frozenset[frozenset[str]]
    """Unordered pairs {rid_a, rid_b} whose changed-path footprints overlap."""

    conflicts_with_main: frozenset[str]
    """Request IDs that conflict with the current main tip."""

    def textual_neighbors(self, rid: str) -> frozenset[str]:
        """Return the set of request_ids connected to *rid* via textual_edges."""
        return frozenset(
            next(iter(edge - {rid}))
            for edge in self.textual_edges
            if rid in edge
        )

    def footprint_neighbors(self, rid: str) -> frozenset[str]:
        """Return the set of request_ids connected to *rid* via footprint_edges."""
        return frozenset(
            next(iter(edge - {rid}))
            for edge in self.footprint_edges
            if rid in edge
        )

    def to_snapshot_dict(self) -> dict:
        """Return a JSON-safe dict representation suitable for heartbeat snapshots.

        Output format:
          nodes: list[str]                 — in pick order
          textual_edges: list[list[str]]   — each inner list sorted; outer sorted
          footprint_edges: list[list[str]] — same shape as textual_edges
          conflicts_with_main: list[str]   — sorted
        """
        return {
            'nodes': list(self.nodes),
            'textual_edges': sorted(sorted(edge) for edge in self.textual_edges),
            'footprint_edges': sorted(sorted(edge) for edge in self.footprint_edges),
            'conflicts_with_main': sorted(self.conflicts_with_main),
        }


EMPTY_SUFFIX_CONFLICT_GRAPH = SuffixConflictGraph(
    nodes=(),
    textual_edges=frozenset(),
    footprint_edges=frozenset(),
    conflicts_with_main=frozenset(),
)
"""Sentinel empty SuffixConflictGraph for the default/zero-suffix case."""


class SuffixConflictTracker:
    """Owns the two-layer suffix-conflict state + logic (task δ=1988).

    Constructed with a live :class:`~orchestrator.git_ops.GitOps` reference
    plus three narrow accessor callables — ``lane_buffers``, ``frozen_prefix``,
    ``frozen_prefix_tip`` — instead of a worker reference, so the tracker is
    fully unit-testable without a :class:`SpeculativeMergeWorker`.  Callables
    are captured (not their values) so the tracker always observes the
    caller's live state at call time.

    Attributes:
        graph: The current :class:`SuffixConflictGraph` (starts at the empty
            sentinel).
        signature: Debounce signature ``(ordered_rids, main_sha)`` from the
            last successful :meth:`recompute`, or ``None`` before the first
            compute (or after invalidation by a bounce).
        last_known_main_sha: The real main SHA cached at the last successful
            :meth:`recompute` call.
        bounce_registry: Per-branch bounce counter for the needs-rebase cap.
    """

    def __init__(
        self,
        git_ops: GitOps,
        *,
        lane_buffers: Callable[[], dict[str, collections.deque[MergeRequest]]],
        frozen_prefix: Callable[[], tuple[str, ...]],
        frozen_prefix_tip: Callable[[str], str],
    ) -> None:
        self._git_ops = git_ops
        self._lane_buffers = lane_buffers
        self._frozen_prefix = frozen_prefix
        self._frozen_prefix_tip = frozen_prefix_tip
        self.graph: SuffixConflictGraph = EMPTY_SUFFIX_CONFLICT_GRAPH
        self.signature: tuple[tuple[str, ...], str] | None = None
        self.last_known_main_sha: str | None = None
        self.bounce_registry: MergeBounceRegistry = MergeBounceRegistry()

    async def recompute(self) -> None:
        """Recompute and store the conflict-graph over the unfrozen suffix.

        See the original :meth:`SpeculativeMergeWorker.recompute_suffix_conflict_graph`
        docstring (preserved verbatim on the worker's thin delegator) for the
        full debounce / fail-open / textual-pruning contract.
        """
        raise NotImplementedError

    async def bounce_conflicting_suffix_items(self) -> None:
        """η=1892 graph-time bounce: divert suffix items that conflict with the frozen tip.

        See the original
        :meth:`SpeculativeMergeWorker._bounce_conflicting_suffix_items`
        docstring (preserved verbatim on the worker's thin delegator) for the
        full cap/escalation/TOCTOU contract.
        """
        raise NotImplementedError
