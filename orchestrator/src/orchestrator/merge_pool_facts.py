"""The facts the merge lane records for calibrating a priority-ordered merge pool offline.

Each is an additive key on an existing event: who was queued at which tier
(``merge_queued``), what was waiting at each pick and which of those pairs the
suffix conflict graph held as conflicting (``merge_dequeued``), and which
requests one verify covered (``merge_verify``). The graph may predate the pick,
so each request also says whether it was in the suffix at the graph's last
recompute (``in_graph``), though its pairs may still be unassessed; pairs are
counted only among the listed requests. Nothing here
decides anything; a would-be pick order is computed offline from these rows,
never in-process.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING

from orchestrator.merge_verify_facts import ChainMembers, MergeVerifyDispatchFacts
from orchestrator.task_priority import TaskPriority, TaskPriorityLookup, read_task_priority

if TYPE_CHECKING:
    from orchestrator.merge_lane.types import ChainResult, MergeRequest
    from orchestrator.suffix_graph import SuffixConflictGraph

logger = logging.getLogger('orchestrator.merge_lane')

WAITING_CAP = 200
"""Most waiting requests one ``merge_dequeued`` row lists; the rest are counted."""

PICK_RECORD_CAP = 16
"""Most pick records held for an emit that never comes (an abandoned pick)."""


def queued_request_fields(req: MergeRequest, priority: TaskPriority) -> dict[str, object]:
    """The keys ``merge_queued`` adds for *req*: its identity, lane, tier, pin and age."""
    return {
        'request_id': req.request_id,
        'lane': req.lane,
        'tier': priority.tier,
        'pinned': priority.pinned,
        'head_sha': req.snapshot_tip,
        'first_enqueued_at': req.first_enqueued_at,
    }


class PickRecords:
    """What the lane buffers held at each pick, kept until the pick's
    ``merge_dequeued`` row takes it.

    Recording only reads: it never touches a buffer, a request or the graph,
    and a failure is logged and leaves no record rather than reaching the pick.
    """

    def __init__(
        self,
        *,
        lanes: tuple[str, ...],
        is_reserved: Callable[[str], bool],
        is_lane_halted: Callable[[str], bool],
        task_priority: TaskPriorityLookup | None,
    ) -> None:
        self._lanes = lanes
        self._is_reserved = is_reserved
        self._is_lane_halted = is_lane_halted
        self._task_priority = task_priority
        self._records: dict[str, dict[str, object]] = {}

    def record(
        self,
        picked: MergeRequest,
        lane: str,
        buffers: Mapping[str, Iterable[MergeRequest]],
        graph: SuffixConflictGraph,
    ) -> None:
        """Note that *picked* left *lane* while *buffers* held the rest."""
        try:
            snapshot = self._snapshot(picked, lane, buffers, graph)
        except Exception:  # noqa: BLE001
            logger.warning(
                'merge pool facts: no pick record for %s', picked.request_id, exc_info=True,
            )
            return
        self._records.pop(picked.request_id, None)
        self._records[picked.request_id] = snapshot
        while len(self._records) > PICK_RECORD_CAP:
            del self._records[next(iter(self._records))]

    def take(self, request_id: str) -> dict[str, object]:
        """The record for *request_id*'s latest pick, removed; empty when there is none."""
        return self._records.pop(request_id, {})

    def _snapshot(
        self,
        picked: MergeRequest,
        lane: str,
        buffers: Mapping[str, Iterable[MergeRequest]],
        graph: SuffixConflictGraph,
    ) -> dict[str, object]:
        waiting = [(ln, req) for ln in self._lanes for req in buffers[ln]]
        listed = waiting[:WAITING_CAP]
        members = frozenset({picked.request_id, *(req.request_id for _, req in listed)})
        priority = self._priority(picked)
        in_graph = frozenset(graph.nodes)
        return {
            'request_id': picked.request_id,
            'lane': lane,
            'tier': priority.tier,
            'pinned': priority.pinned,
            'in_graph': picked.request_id in in_graph,
            'waiting': [self._waiting_entry(ln, req, in_graph) for ln, req in listed],
            'waiting_dropped': len(waiting) - len(listed),
            'halted_lanes': [ln for ln in self._lanes if self._is_lane_halted(ln)],
            'textual_conflict_pairs': _pairs_within(graph.textual_edges, members),
            'footprint_pair_count': sum(1 for pair in graph.footprint_edges if pair <= members),
        }

    def _waiting_entry(
        self, lane: str, req: MergeRequest, in_graph: frozenset[str],
    ) -> dict[str, object]:
        priority = self._priority(req)
        return {
            'request_id': req.request_id,
            'task_id': req.task_id,
            'lane': lane,
            'tier': priority.tier,
            'pinned': priority.pinned,
            'first_enqueued_at': req.first_enqueued_at,
            'reserved': self._is_reserved(req.request_id),
            'in_graph': req.request_id in in_graph,
        }

    def _priority(self, req: MergeRequest) -> TaskPriority:
        return read_task_priority(self._task_priority, req.task_id)


def _pairs_within(edges: frozenset[frozenset[str]], members: frozenset[str]) -> list[list[str]]:
    return sorted(sorted(pair) for pair in edges if pair <= members)


def with_chain_members(
    facts: MergeVerifyDispatchFacts | None, req: MergeRequest, chain: ChainResult | None,
) -> MergeVerifyDispatchFacts:
    """*facts* also naming the requests a verify of *req* covers, as built.

    *req* alone when nothing chained; *req* then each link when *chain* did.
    Left without members when the chain's links cannot be named, and never raises.
    """
    base = facts if facts is not None else MergeVerifyDispatchFacts()
    try:
        members = _chain_members(req, chain)
    except Exception:  # noqa: BLE001
        logger.warning('merge pool facts: no chain members for %s', req.request_id, exc_info=True)
        return base
    return base if members is None else base.covering(members)


def _chain_members(req: MergeRequest, chain: ChainResult | None) -> ChainMembers | None:
    if chain is None:
        return ChainMembers((req.request_id,), (req.task_id,))
    if len(chain.link_request_ids) != len(chain.links):
        return None
    return ChainMembers(
        (req.request_id, *chain.link_request_ids),
        (req.task_id, *(task_id for task_id, _ in chain.links)),
    )
