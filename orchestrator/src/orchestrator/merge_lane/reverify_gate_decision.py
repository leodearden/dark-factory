"""The post-rebase re-verify gate's decision, and its durable ``merge_reverify_gate`` row.

``gates.py::_reverify_rebased_tree`` decides, for a request whose verified
merge was rebased onto a moved main, whether that rebased tree may advance
without a fresh verify. ``ReverifyGateReason`` names every way the decision
can go, and is the one home of which reasons skip the verify and which
``GateClearance`` a skip yields. ``ReverifyGateDecision`` records the facts
the decision was made on; ``event_data`` renders them as the row's payload.

The row is written BEFORE any re-verify runs, so it states what the gate
decided, not how a fired re-verify ended: that verdict is the ``merge_verify``
row with the same task id and merge sha, and the clearance a request advanced
on is its ``merge_attempt{gate_retry}`` row.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from enum import StrEnum

from orchestrator.event_store import EventStore, EventType

MAX_RECORDED_OVERLAP_PATHS = 10
MAX_LOGGED_OVERLAP_PATHS = 5


class GateClearance(StrEnum):
    """Why a rebased tree cleared ``gates.py::_reverify_rebased_tree`` (task 6647)."""

    same_tree = 'same_tree'
    """The rebased tree is byte-identical to the tree whose verify passed."""
    disjoint = 'disjoint'
    """Footprint-disjoint from trusted drift; see ``gates.py::_disjoint_skip_blockers``."""
    reverified = 'reverified'
    """A full re-verify of the rebased tree passed."""


class ReverifyGateReason(StrEnum):
    """How the gate decided one rebased tree; only the skips map to a ``GateClearance``."""

    same_tree = 'same_tree'
    """The rebased tree is the verified tree (only under ``merge_gate_history_pure``)."""
    disjoint = 'disjoint'
    """No path overlap, and the disjointness premises hold."""
    disjoint_untrusted = 'disjoint_untrusted'
    """No path overlap, but a premise of the disjointness inference failed."""
    overlap = 'overlap'
    """The branch and the intervening main delta touched a common path."""
    overlap_unknown = 'overlap_unknown'
    """The overlap probe failed, so the overlap is unknown."""

    @property
    def skip_clearance(self) -> GateClearance | None:
        return _SKIP_CLEARANCE.get(self)

    @property
    def reverifies(self) -> bool:
        return self not in _SKIP_CLEARANCE


_SKIP_CLEARANCE: Mapping[ReverifyGateReason, GateClearance] = {
    ReverifyGateReason.same_tree: GateClearance.same_tree,
    ReverifyGateReason.disjoint: GateClearance.disjoint,
}


@dataclasses.dataclass(frozen=True)
class ReverifyGateDecision:
    task_id: str
    reason: ReverifyGateReason
    merge_sha: str
    verified_merge_sha: str
    rebased_from: str
    rebased_onto: str
    overlap: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        overlap_fits = bool(self.overlap) == (self.reason is ReverifyGateReason.overlap)
        blockers_fit = bool(self.blockers) == (
            self.reason is ReverifyGateReason.disjoint_untrusted
        )
        if not (overlap_fits and blockers_fit):
            raise ValueError(
                f'ReverifyGateDecision for task {self.task_id}: reason {self.reason} '
                f'requires overlap paths exactly for overlap and blockers exactly for '
                f'disjoint_untrusted; got overlap={self.overlap!r}, '
                f'blockers={self.blockers!r}'
            )

    @property
    def reverifies(self) -> bool:
        return self.reason.reverifies

    def event_data(self) -> dict:
        return {
            'reason': str(self.reason),
            'reverify': self.reverifies,
            'merge_sha': self.merge_sha,
            'verified_merge_sha': self.verified_merge_sha,
            'rebased_from': self.rebased_from,
            'rebased_onto': self.rebased_onto,
            'overlap': list(self.overlap[:MAX_RECORDED_OVERLAP_PATHS]),
            'overlap_total': len(self.overlap),
            'blockers': list(self.blockers),
        }


def emit_reverify_gate_decision(
    event_store: EventStore | None, decision: ReverifyGateDecision,
) -> None:
    if event_store is None:
        return
    event_store.emit(
        EventType.merge_reverify_gate,
        task_id=decision.task_id,
        phase='merge',
        data=decision.event_data(),
    )
