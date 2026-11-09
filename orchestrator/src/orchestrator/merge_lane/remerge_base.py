"""Where a recovery re-merge is anchored: the live frontier tip, or main.

A replacement built by a recovery re-merge stacks on the frozen-prefix tip
only while that tip can still land, and otherwise falls to main with the
reason recorded (plans/merge-lane-policy-prd.md D1; contract §6).
"""

from __future__ import annotations

from collections.abc import Container, Sequence

from orchestrator.merge_lane.types import (
    InflightEntry,
    InflightVerifyResult,
    RealMergeItem,
    RemergeBase,
    RemergeReason,
)


def _commit(entry: InflightEntry) -> str | None:
    item = entry.item
    if isinstance(item, RealMergeItem) and item.merge_result.merge_commit:
        return item.merge_result.merge_commit.strip() or None
    return None


def frontier_tip(entries: Sequence[InflightEntry]) -> str | None:
    """The newest real merge commit among *entries*, or None when there is none."""
    for entry in reversed(entries):
        commit = _commit(entry)
        if commit is not None:
            return commit
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
    the entry's own commit being dead."""
    item = entry.item
    if not isinstance(item, RealMergeItem):
        return False
    base = item.base_sha.strip()
    if base and base != main_sha and base in dead_bases:
        return True
    commit = _commit(entry)
    return commit is not None and commit in dead_bases


def _owner_index(tip: str, frontier: Sequence[InflightEntry]) -> int | None:
    return next((i for i, entry in enumerate(frontier) if _commit(entry) == tip), None)


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
    if any(_breaks_chain(entry, main_sha, dead_bases) for entry in frontier[:owner_at + 1]):
        return RemergeReason.TIP_DEAD
    owner = frontier[owner_at]
    if owner.vacated:
        return RemergeReason.TIP_VACATED
    if not _verify_alive(owner):
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

    Live means: not main, not dead, owned by a *frontier* entry that is not
    vacated and whose verify is alive, with every entry up to that owner
    chain-intact.  *frontier* is ordered oldest first.  A live tip still falls
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
