"""The ports the verify dispatcher reaches its not-yet-extracted neighbours through.

Each port groups the operations of one concern the dispatcher does not own,
named for the module that will own it: request and entry custody (lifecycle
and landing), speculation, merge-worktree ownership, and merge-verify-lane
contention, plus the lane-wide context every post-merge verify inherits.
The worker builds each as callables that re-read its live members on every
call (the ``merge_lane/intake_buffers.py::IntakeLifecycle`` precedent), so a
member replaced on the worker instance is still the one the dispatcher
reaches; a later extraction replaces one port with its module object.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Collection, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from orchestrator.event_store import EventStore
from orchestrator.merge_lane.dead_base_ledger import DeadBaseSite, VoidPoint
from orchestrator.merge_lane.main_health_probe import MainHealthProbeHandles
from orchestrator.merge_lane.remerge_base import RemergeClaim
from orchestrator.merge_lane.types import (
    CapPermit,
    DeepPlacement,
    InflightEntrySlot,
    InflightVerifyResult,
    ItemLifecycleState,
    MergeOutcome,
    MergeRequest,
    RealMergeItem,
    RemergeBase,
    SpecPermit,
    SpeculativeItem,
    VerifyBaseDivergence,
)
from orchestrator.merge_lane.verify_failure_outcome import DryRunInvestigationHandles
from orchestrator.merge_verify_facts import ChainDepthReason
from orchestrator.verify_runner import HostLease, RunnerUnavailable


@dataclass(frozen=True)
class DispatchCustody:
    """Who holds a dispatched request and its in-flight entry, and how a verify
    hands either back: drop, requeue, bench its host, or adopt its head."""

    request_abandoned: Callable[[MergeRequest], bool]
    note_verify_completed: Callable[[object, InflightEntrySlot | None, HostLease | None], None]
    teardown_verify_task: Callable[..., Awaitable[None]]
    """``(lease, verify_task, task_id, *, cancel, shutdown_defensive=False)``:
    the one sanctioned cancel-and-reap of a verify task."""
    drop_from_verify: Callable[[RealMergeItem, InflightEntrySlot | None], Awaitable[None]]
    requeue_from_verify: Callable[[RealMergeItem, InflightEntrySlot | None], Awaitable[None]]
    bench_from_verify: Callable[[InflightEntrySlot | None, RunnerUnavailable], Awaitable[None]]
    bench_unavailable_runner: Callable[[str, RunnerUnavailable], None]
    adopt_head_on_tip_authority: Callable[[RealMergeItem], Awaitable[object]]
    admission_halted: Callable[[], bool]
    requeue_request: Callable[[MergeRequest], None]
    retire_item: Callable[[str], None]
    record_unlanded_rebased_head: Callable[[SpeculativeItem], None]
    note_transition: Callable[
        [str, ItemLifecycleState, ItemLifecycleState, SpeculativeItem | MergeRequest], None,
    ]
    """``(request_id, before, after, live_obj)``: one lifecycle-registry move."""
    has_inflight_verify: Callable[[], bool]
    """Whether any entry is in the verify frontier."""
    release_at_verdict: Callable[
        [Callable[[], Awaitable[InflightVerifyResult]], InflightEntrySlot],
        Awaitable[InflightVerifyResult],
    ]
    """Run a verify, then free its host at the verdict: the body of every verify task."""


@dataclass(frozen=True)
class DispatchSpeculation:
    """The deep-chain policy state a verify's verdict feeds."""

    mark_next_dispatch_unchained: Callable[[Iterable[str]], None]
    end_chain_reservation: Callable[[str], None]
    """End the owner's chain reservation as not all landed."""
    note_tip_result: Callable[[ChainDepthReason | None, int, MergeOutcome | None], None]
    release_cap_permit: Callable[[CapPermit], None]
    release_permit: Callable[[SpecPermit], None]
    previous_failed: Callable[[], bool]
    """Whether the last head result was a failure."""
    has_dead_bases: Callable[[], bool]
    main_sha_or_empty: Callable[[str], Awaitable[str]]
    """``(reader)``: main, or '' with the failed read logged against *reader*."""
    chain_dead_link: Callable[[SpeculativeItem, str], str | None]
    claim_remerge_base: Callable[[RealMergeItem, str, SpecPermit | None], RemergeClaim]
    emit_dead_base_discard: Callable[
        [MergeRequest, SpeculativeItem, str, VoidPoint, RemergeBase], None,
    ]
    """``(req, item, dead_link, point, anchor)``."""
    emit_speculative: Callable[..., None]
    """``(event_type, task_id, **data)``: one speculation event row."""
    record_item_commit_dead: Callable[[SpeculativeItem, DeadBaseSite, bool], None]
    """``(item, site, notify_merger)``."""
    remerge: Callable[[MergeRequest, float | None, RemergeClaim], Awaitable[SpeculativeItem]]
    """``(req, started_monotonic, claim)``: re-merge *req* onto the claim's anchor."""
    notify_merger_of_dead: Callable[[Collection[str]], None]
    reanchor_merger_on: Callable[[RealMergeItem, SpeculativeItem], None]
    """``(superseded, replacement)``: re-point a merger holding the superseded commit."""
    note_main_sha: Callable[[str], None]
    verify_base_divergence: Callable[[SpeculativeItem, str], VerifyBaseDivergence | None]
    """The frozen-tip detector: it observes, logs and escalates, never refuses."""
    frontier_base_commits: Callable[[], tuple[set[str], set[str]]]
    """``(stacked_on, rebased_heads)`` of the frozen verify frontier."""
    probe_placement: Callable[[RealMergeItem], ProbeAttribution | None]
    frontier_depth: Callable[[], int]
    deep_chain_placement: Callable[[RealMergeItem, int, bool], Awaitable[DeepPlacement]]
    """``(item, frontier_depth, verify_is_local)``; an external cancel propagates."""


class ProbeAttribution(Protocol):
    """A firing probe's facts: the depth label and base it attributes a dispatch to."""

    @property
    def depth(self) -> int: ...

    @property
    def base(self) -> str: ...


@dataclass(frozen=True)
class DispatchWorktrees:
    """The worker's merge-worktree ownership ledger."""

    cleanup_owned: Callable[[Path | None], Awaitable[None]]
    release_or_cleanup: Callable[[Path | None, bool], Awaitable[None]]
    """``(merge_wt, spec_warm)``: release a warm ``_spec-`` lane or clean up cold."""
    deregister_owned: Callable[[Path | None], None]
    owned: Callable[[], set[Path]]
    """A fresh copy of the owned set, for a verify's ``keep_worktrees``."""


@dataclass(frozen=True)
class LeaseContention:
    """The contended merge-verify-lane defer streak, per task."""

    advance_streak: Callable[[str, BaseException], tuple[int, float]]
    clear_streak: Callable[[str], None]
    next_cap_out: Callable[[], int]
    """Count one more terminal cap-out and return its ordinal."""


@dataclass(frozen=True)
class VerifyRetryBudgets:
    """The per-task post-merge verify retry ledgers and their caps, shared by
    reference with the train path and finalize."""

    timeouts: dict[str, int]
    enospc_retries: dict[str, int]
    narrowed_retries: dict[str, int]
    max_timeouts: int
    max_enospc: int
    max_narrowed: int


@dataclass(frozen=True)
class LaneVerifyContext:
    """What every post-merge verify of the lane inherits."""

    retry_budgets: Callable[[], VerifyRetryBudgets]
    event_store: Callable[[], EventStore | None]
    escalation_queue: Callable[[], Any]
    quarantine_divergent: Callable[[str], None]
    halt_hook: Callable[[str], None]
    """The synchronous all-lane halt a concluded cross-check divergence fires,
    so the queue stops before any further adoption (task 2886)."""
    dry_run_handles: Callable[[], DryRunInvestigationHandles]
    main_health_probe: Callable[[], MainHealthProbeHandles]

