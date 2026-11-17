"""Count the nodes that leaked across graphs — read-only.

A node whose ``group_id`` property names a graph other than the one it lives in,
or names no graph, is task 2115's cross-graph leak. Every group-scoped read —
``task_family_census`` among them — filters on ``group_id`` and so cannot see
such a node; this module counts them per graph and per label. It is read-only by
construction: every call it makes is ``GraphHandle.ro_query``. The per-node
manifest and the remedy are ``scripts/migrate_cross_graph_leak.py``'s dry run.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from fused_memory.backends.graph_access import GraphAccess, GraphHandle, OpBoundReader
from fused_memory.backends.paged_read import (
    INCOMPLETE_CENSUS_UNAVAILABLE,
    INCOMPLETE_KINDS,
    INCOMPLETE_SHORT_READ,
    census_count,
)

logger = logging.getLogger(__name__)

LEAK_LABELS: tuple[str, ...] = ('Entity', 'Episodic')
"""The node labels counted, in report order."""

_LEAKED_MATCH = 'MATCH (n:{label}) WHERE n.group_id IS NULL OR n.group_id <> $graph_key'
_COUNT_CYPHER = {
    label: _LEAKED_MATCH.format(label=label) + ' RETURN count(n)' for label in LEAK_LABELS
}
_BREAKDOWN_CYPHER = {
    label: _LEAKED_MATCH.format(label=label) + ' RETURN n.group_id, count(n)'
    for label in LEAK_LABELS
}


@dataclass(frozen=True)
class ForeignGroup:
    """How many leaked nodes of one label carry one ``group_id``; None means none at all."""

    node_group_id: str | None
    nodes: int


@dataclass(frozen=True)
class LabelLeak:
    """One label's leaked nodes in one graph, and whether that count is whole.

    ``leaked`` is the proven total from a single-row ``count(n)``, which the
    server's row cap cannot truncate; None means the store gave no usable
    count. ``by_group_id`` is a second, grouped read, largest group first.
    ``incomplete_kind`` is one of ``paged_read.INCOMPLETE_KINDS``, and is None
    exactly when ``complete``.
    """

    label: str
    leaked: int | None
    by_group_id: tuple[ForeignGroup, ...]
    complete: bool
    incomplete_kind: str | None

    def __post_init__(self) -> None:
        if (self.incomplete_kind is None) != self.complete:
            raise ValueError(
                'LabelLeak.incomplete_kind must be None exactly when complete; got '
                f'complete={self.complete} incomplete_kind={self.incomplete_kind!r} '
                f'for label {self.label!r}'
            )
        if self.incomplete_kind is not None and self.incomplete_kind not in INCOMPLETE_KINDS:
            raise ValueError(
                f'LabelLeak.incomplete_kind must be one of {sorted(INCOMPLETE_KINDS)}; got '
                f'incomplete_kind={self.incomplete_kind!r} complete={self.complete} '
                f'for label {self.label!r}'
            )


@dataclass(frozen=True)
class GraphLeaks:
    """Every label's leak count in one graph; ``complete`` is False if any label's is not."""

    group_id: str
    labels: tuple[LabelLeak, ...]
    complete: bool


def _describe_breakdown(groups: tuple[ForeignGroup, ...]) -> str:
    return ', '.join(
        f'{"<none>" if g.node_group_id is None else g.node_group_id}={g.nodes}' for g in groups
    ) or '(empty breakdown)'


def _incompleteness(leaked: int | None, breakdown_sum: int) -> str | None:
    """Why a label's read is not whole, or None when it is.

    A breakdown summing ABOVE the proven total is growth between the two reads,
    not a truncation — the same ``>=`` rule ``paged_ro_query`` applies.
    """
    if leaked is None:
        return INCOMPLETE_CENSUS_UNAVAILABLE
    if breakdown_sum < leaked:
        return INCOMPLETE_SHORT_READ
    return None


def _log_label(group_id: str, leak: LabelLeak) -> None:
    breakdown = _describe_breakdown(leak.by_group_id)
    if leak.leaked:
        logger.warning(
            'cross-graph leak in graph %r: %d %s node(s) carry another graph\'s group_id '
            'or none — %s',
            group_id, leak.leaked, leak.label, breakdown,
        )
    if not leak.complete:
        logger.error(
            'COVERAGE SHORTFALL: graph %r %s leak count is incomplete (%s): proven total %s, '
            'breakdown sums to %d — %s',
            group_id, leak.label, leak.incomplete_kind, leak.leaked,
            sum(g.nodes for g in leak.by_group_id), breakdown,
        )


class CrossGraphLeakCensus:
    """Counts leaked nodes graph by graph. Never writes.

    Each graph is read through ``graphs.graph(key)`` with the RAW graph key,
    never a canonicalized one: the comparison against that key is the whole
    point, and an alias-named graph would otherwise be compared against a
    different graph's name.
    """

    def __init__(self, *, graphs: GraphAccess):
        self.graphs = graphs

    async def run(self, group_id: str) -> GraphLeaks:
        """Count one graph's leaked nodes, one ``LabelLeak`` per entry in ``LEAK_LABELS``."""
        handle = self.graphs.graph(group_id)
        labels = []
        for label in LEAK_LABELS:
            leak = await self._read_label(handle, group_id, label)
            _log_label(group_id, leak)
            labels.append(leak)
        return GraphLeaks(
            group_id=group_id,
            labels=tuple(labels),
            complete=all(leak.complete for leak in labels),
        )

    @staticmethod
    async def _read_label(handle: GraphHandle, group_id: str, label: str) -> LabelLeak:
        """The proven total FIRST, then the grouped breakdown checked against it."""
        params = {'graph_key': group_id}
        leaked = await census_count(
            OpBoundReader(handle, 'cross_graph_leak_count'), _COUNT_CYPHER[label], params,
        )
        result = await handle.ro_query(
            _BREAKDOWN_CYPHER[label], params, op='cross_graph_leak_breakdown',
        )
        groups = tuple(sorted(
            (
                ForeignGroup(
                    node_group_id=None if node_group_id is None else str(node_group_id),
                    nodes=int(nodes),
                )
                for node_group_id, nodes in getattr(result, 'result_set', None) or []
            ),
            key=lambda g: (-g.nodes, g.node_group_id is None, g.node_group_id or ''),
        ))
        kind = _incompleteness(leaked, sum(g.nodes for g in groups))
        return LabelLeak(
            label=label,
            leaked=leaked,
            by_group_id=groups,
            complete=kind is None,
            incomplete_kind=kind,
        )
