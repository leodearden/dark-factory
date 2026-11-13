"""The INV-3 dead-base ledger: merge commits that will never reach main, what killed
each, and where a verdict built on one was voided.

Rationale: ``plans/merge-verdict-integrity-prd.md`` §1 INV-3 for the ledger,
``plans/merge-lane-policy-prd.md`` D10 for the killer provenance.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from enum import StrEnum


class DeadBaseSite(StrEnum):
    """The worker site whose lane event recorded a commit dead."""

    #: The head-failure cascade's failed head.
    HEAD_FAILED = 'head_failed'
    #: A downstream entry the head-failure cascade re-merged.
    CASCADE_REMERGE = 'cascade_remerge'
    #: A verdict voided at adoption and re-merged.
    ADOPTION_VOID = 'adoption_void'
    #: An entry re-merged after its host reported RUNNER_UNAVAILABLE.
    RUNNER_UNAVAILABLE = 'runner_unavailable'
    #: A head that landed rebased, superseding its pre-rebase commit.
    GATE_REBASED = 'gate_rebased'
    #: A successor torn down so it re-anchors on its head's gate-rebased commit.
    GATE_REANCHOR = 'gate_reanchor'
    #: A gate-rebased commit whose head left the lane without landing it.
    GATE_UNLANDED = 'gate_unlanded'
    #: A dead-base straggler discarded and re-merged at dispatch.
    DISPATCH_VOID = 'dispatch_void'
    #: A stale merge discarded at dispatch (Mechanism 2).
    STALE_DISCARD = 'stale_discard'
    #: A deep chain tip whose reserved links did not all land (task 7720).
    CHAIN_RELEASED = 'chain_released'


class VoidPoint(StrEnum):
    """The INV-3 enforcement point that voided a verdict built on a dead base."""

    DISPATCH = 'dispatch'
    ADOPTION = 'adoption'


@dataclasses.dataclass(frozen=True)
class DeadBaseKiller:
    """The site and task whose lane event first recorded a commit dead."""

    site: DeadBaseSite
    task_id: str

    def to_wire(self) -> dict[str, str]:
        return {'site': self.site.value, 'task_id': self.task_id}


class DeadBaseLedger:
    """A bounded FIFO of dead commits, each mapped to its first killer.

    Past *cap* the oldest commit is evicted. A re-record of a present commit
    keeps its first killer and its FIFO position: first death is the
    provenance. Blank SHAs are ignored and every SHA is whitespace-stripped.
    """

    def __init__(self, *, cap: int) -> None:
        if cap < 1:
            raise ValueError(f'DeadBaseLedger cap must be >= 1, got cap={cap}')
        self._cap = cap
        self._killers: dict[str, DeadBaseKiller] = {}

    def record(self, sha: str, killer: DeadBaseKiller) -> None:
        key = sha.strip()
        if not key or key in self._killers:
            return
        self._killers[key] = killer
        while len(self._killers) > self._cap:
            del self._killers[next(iter(self._killers))]

    def killer_of(self, sha: str) -> DeadBaseKiller | None:
        return self._killers.get(sha.strip())

    def __contains__(self, sha: object) -> bool:
        return isinstance(sha, str) and sha.strip() in self._killers

    def __len__(self) -> int:
        return len(self._killers)

    def __bool__(self) -> bool:
        return bool(self._killers)

    def __iter__(self) -> Iterator[str]:
        return iter(self._killers)
