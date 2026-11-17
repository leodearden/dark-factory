"""Where a recovery re-merge is anchored: the live frontier tip, or main.

A replacement built by a recovery re-merge stacks on the frozen-prefix tip
only while that tip can still land, and otherwise falls to main with the
reason recorded (plans/merge-lane-policy-prd.md D1; contract §6).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Container, Sequence

from orchestrator.merge_lane.speculation_controller import PermitLedger
from orchestrator.merge_lane.types import (
    InflightEntry,
    InflightVerifyResult,
    RealMergeItem,
    RemergeBase,
    RemergeReason,
    SpecPermit,
)


def entry_commit(entry: InflightEntry) -> str | None:
    """*entry*'s merge commit, or None when its item built none."""
    item = entry.item
    if isinstance(item, RealMergeItem) and item.merge_result.merge_commit:
        return item.merge_result.merge_commit.strip() or None
    return None


def entry_successor_base(entry: InflightEntry) -> str | None:
    """What a successor of *entry* stacks on: the chain tip when *entry*'s
    dispatch chained links (task 7720), else *entry*'s own merge commit."""
    chain = entry.chain
    if chain is not None and chain.links:
        return chain.tip.strip() or None
    return entry_commit(entry)


def entry_bases(entry: InflightEntry) -> frozenset[str]:
    """Every commit *entry* stands for: its own merge commit and its successor base."""
    return frozenset(filter(None, (entry_commit(entry), entry_successor_base(entry))))


def frontier_tip(entries: Sequence[InflightEntry]) -> str | None:
    """The newest successor base among *entries*, or None when there is none."""
    for entry in reversed(entries):
        base = entry_successor_base(entry)
        if base is not None:
            return base
    return None


def _verify_alive(entry: InflightEntry) -> bool:
    task = entry.verify_task
    if entry.gate is not None or task is None or task.cancelled():
        return False
    if not task.done():
        return True
    if task.exception() is not None:
        return False
    result = task.result()
    return (
        isinstance(result, InflightVerifyResult)
        and result.outcome is None
        and result.status is None
    )


def _breaks_chain(entry: InflightEntry, main_sha: str, dead_bases: Container[str]) -> bool:
    """INV-3's base-dead-and-not-main rule (worker.py::_chain_dead_link), plus
    the entry's own commit or its successor base being dead."""
    item = entry.item
    if not isinstance(item, RealMergeItem):
        return False
    base = item.base_sha.strip()
    if base and base != main_sha and base in dead_bases:
        return True
    return any(sha in dead_bases for sha in entry_bases(entry))


def _owner_index(tip: str, frontier: Sequence[InflightEntry]) -> int | None:
    return next(
        (i for i, entry in enumerate(frontier) if entry_successor_base(entry) == tip), None,
    )


def _fall_to_main_reason(
    main_sha: str, tip: str, dead_bases: Container[str], frontier: Sequence[InflightEntry],
) -> RemergeReason | None:
    if not tip or tip == main_sha:
        return RemergeReason.NO_TIP
    if tip in dead_bases:
        return RemergeReason.TIP_DEAD
    owner_at = _owner_index(tip, frontier)
    if owner_at is None:
        return RemergeReason.TIP_VERIFY_DEAD
    chain = frontier[:owner_at + 1]
    if any(_breaks_chain(entry, main_sha, dead_bases) for entry in chain):
        return RemergeReason.TIP_DEAD
    if any(entry.vacated for entry in chain):
        return RemergeReason.TIP_VACATED
    if not _verify_alive(frontier[owner_at]):
        return RemergeReason.TIP_VERIFY_DEAD
    return None


def choose_remerge_base(
    main_sha: str,
    frozen_tip: str,
    dead_bases: Container[str],
    frontier: Sequence[InflightEntry],
    *,
    permit_available: bool,
) -> RemergeBase:
    """Anchor on *frozen_tip* when it is live, else fall to main with the reason.

    Live means: not main, not dead, owned by a *frontier* entry whose verify
    is alive, with every entry up to that owner chain-intact and not vacated.  *frontier* is ordered oldest first.  A live tip still falls
    to main when no permit is available, since a stacked replacement must
    hold one.
    """
    tip = frozen_tip.strip()
    reason = _fall_to_main_reason(main_sha.strip(), tip, dead_bases, frontier)
    if reason is None and not permit_available:
        reason = RemergeReason.NO_PERMIT
    if reason is not None:
        return RemergeBase(None, reason)
    return RemergeBase(tip, RemergeReason.LIVE_TIP)


@dataclasses.dataclass(frozen=True)
class RemergeClaim:
    """A re-anchor decision and the permit the replacement leaves with: a
    speculative anchor holds exactly one permit, a fall to main holds none."""

    anchor: RemergeBase
    permit: SpecPermit | None

    def __post_init__(self) -> None:
        if self.permit is not None and self.permit.released:
            raise ValueError(
                'RemergeClaim must not hold a released permit; got '
                f'reason={self.anchor.reason.value!r} permit={self.permit!r}',
            )
        if self.anchor.speculative != (self.permit is not None):
            raise ValueError(
                'RemergeClaim holds a permit iff its anchor is speculative; got '
                f'reason={self.anchor.reason.value!r} permit={self.permit!r}',
            )


def claim_remerge_base(
    *,
    main_sha: str,
    frozen_tip: str,
    dead_bases: Container[str],
    frontier: Sequence[InflightEntry],
    held_permit: SpecPermit | None,
    ledger: PermitLedger[SpecPermit],
) -> RemergeClaim:
    """Decide the anchor once, with the permit already in hand.

    The replacement keeps *held_permit* while it is still live, or takes a
    free slot without waiting; a permit it cannot use (a fall to main) is
    released here.
    """
    held = held_permit if held_permit is not None and not held_permit.released else None
    taken = held if held is not None else ledger.try_acquire()
    anchor = choose_remerge_base(
        main_sha, frozen_tip, dead_bases, frontier, permit_available=taken is not None,
    )
    if taken is not None and not anchor.speculative:
        ledger.release(taken)
        taken = None
    return RemergeClaim(anchor, taken)
