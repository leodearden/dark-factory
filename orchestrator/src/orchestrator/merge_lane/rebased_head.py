"""The gate-rebased head as an ordinary dispatch item, and how the lane recognises work on it.

When a head's gate re-verify becomes a dispatch (plans/merge-lane-policy-prd.md
D3), the head is rebuilt as an item on its rebased commit and verified like any
other dispatch. Its successors re-anchor on that commit at their own dispatch,
which is a planned re-anchor rather than a void.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from pathlib import Path

from orchestrator.merge_lane.dead_base_ledger import DeadBaseKiller, DeadBaseSite
from orchestrator.merge_lane.remerge_base import entry_commit
from orchestrator.merge_lane.types import (
    CapPermit,
    InflightEntry,
    LandingCandidate,
    RealMergeItem,
    RemergeBase,
)

_GATE_KILL_SITES = frozenset({DeadBaseSite.GATE_REBASED, DeadBaseSite.GATE_REANCHOR})


def rebased_head_item(
    candidate: LandingCandidate,
    *,
    rebased_sha: str,
    rebased_onto: str,
    merge_wt: Path,
    cap_permit: CapPermit | None,
) -> RealMergeItem:
    """*candidate*'s item rebuilt as a main-based dispatch of its rebased commit.

    ``dataclasses.replace`` keeps every field not named here, the request, the
    merged branch tip and the start clock among them, so none is silently dropped.
    """
    item = candidate.item
    return dataclasses.replace(
        item,
        merge_result=dataclasses.replace(
            item.merge_result, merge_commit=rebased_sha, merge_worktree=merge_wt,
        ),
        merge_wt=merge_wt,
        base_sha=rebased_onto,
        speculative=False,
        permit=None,
        cap_permit=cap_permit,
        remerge_base=None,
        gate_rebase_of=candidate.sha,
    )


def _is_rebased_head(entry: InflightEntry) -> bool:
    return isinstance(entry.item, RealMergeItem) and entry.item.gate_rebase_of is not None


def rebased_head_commits(entries: Iterable[InflightEntry]) -> frozenset[str]:
    """The merge commits of the gate-rebased heads among *entries*."""
    commits = (entry_commit(entry) for entry in entries if _is_rebased_head(entry))
    return frozenset(commit for commit in commits if commit is not None)


def is_gate_reanchor(killer: DeadBaseKiller | None, anchor: RemergeBase) -> bool:
    """True when a gate killed the base and the re-merge anchored on a live tip.

    A gate kills a base two ways: the head's rebase supersedes its pre-rebase
    commit (``GATE_REBASED``), or a successor is torn down behind a dispatched
    rebased head (``GATE_REANCHOR``).  Re-anchoring an item stacked on either
    onto a live tip is planned work, not a voided verdict.
    """
    return killer is not None and killer.site in _GATE_KILL_SITES and anchor.speculative
