"""The early-start selection's vocabulary and rules: who qualifies, why a pair is refused, the claim.

plans/early-start-on-predecessor-branch-prd.md §5.1 (who qualifies, the tier condition) and §5.4
(the selection's steps). Everything here is a value type or a pure rule over its arguments and the
lock table's read views; the entry is ``Scheduler.early_start_shortlist`` and
``Scheduler.select_early_start``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from orchestrator.delivered_checks import ChecksAtCommit
from orchestrator.dependency_verdict import DepsVerdict, deps_verdict
from orchestrator.dispatch_row import FreshRow
from orchestrator.id_tie_break import id_tie_break
from orchestrator.task_picture import TaskPicture
from orchestrator.task_priority import TaskPriority, tier_order


class EarlyStartRefusalReason(StrEnum):
    NOT_ONLY_DEPENDENCY = 'not_only_dependency'
    NOT_ELIGIBLE = 'not_eligible'
    HAS_BRANCH = 'has_branch'
    NOT_CODE_TASK = 'not_code_task'
    HAS_MILESTONE = 'has_milestone'
    DELIVERED_CHECK_SCRIPT = 'delivered_check_script'
    DELIVERED_CHECK_FAILED = 'delivered_check_failed'
    KEY_UNAVAILABLE = 'key_unavailable'
    KEY_RESERVED = 'key_reserved'
    HIGHER_TIER_WAITING = 'higher_tier_waiting'
    PREDECESSOR_HAS_DEPENDENT = 'predecessor_has_dependent'

    DISABLED = 'disabled'
    HOST_LOAD = 'host_load'
    SCHEDULER_PAUSED = 'scheduler_paused'
    REVIEW_CHECKPOINT = 'review_checkpoint'
    WARM_BASE_DOWN = 'warm_base_down'
    MERGE_QUEUE_HALTED = 'merge_queue_halted'
    USAGE_PAUSED = 'usage_paused'
    COST_CEILING = 'cost_ceiling'

    @property
    def is_hold(self) -> bool:
        """A system hold refuses every pair at once; every other reason refuses one pair."""
        return self in _HOLD_REASONS


_HOLD_REASONS = frozenset({
    EarlyStartRefusalReason.DISABLED,
    EarlyStartRefusalReason.HOST_LOAD,
    EarlyStartRefusalReason.SCHEDULER_PAUSED,
    EarlyStartRefusalReason.REVIEW_CHECKPOINT,
    EarlyStartRefusalReason.WARM_BASE_DOWN,
    EarlyStartRefusalReason.MERGE_QUEUE_HALTED,
    EarlyStartRefusalReason.USAGE_PAUSED,
    EarlyStartRefusalReason.COST_CEILING,
})


@dataclass(frozen=True)
class EarlyStartRefusal:
    """Why a pair, or with a hold every pair, was not started early.

    A hold names no pair; a pair reason names both members; ``blocking_task`` names the
    higher-tier task exactly for ``higher_tier_waiting``.
    """

    reason: EarlyStartRefusalReason
    dependent: str | None = None
    predecessor: str | None = None
    blocking_task: str | None = None

    def __post_init__(self) -> None:
        pair = (self.dependent, self.predecessor)
        if self.reason.is_hold and pair != (None, None):
            raise ValueError(f'hold {self.reason} names no pair, got {pair!r}')
        if not self.reason.is_hold and None in pair:
            raise ValueError(f'pair reason {self.reason} names both members, got {pair!r}')
        names_blocker = self.reason is EarlyStartRefusalReason.HIGHER_TIER_WAITING
        if names_blocker != (self.blocking_task is not None):
            raise ValueError(
                f'blocking_task is set exactly for higher_tier_waiting, got '
                f'{self.reason} with blocking_task={self.blocking_task!r}'
            )


RankKey = tuple[tuple[int, int], tuple[float, str]]


def rank_key(priority: TaskPriority, task_id: str) -> RankKey:
    """The one tier order, then the id tie-break; smaller ranks higher."""
    return (tier_order(priority), id_tie_break(task_id))


@dataclass(frozen=True)
class EarlyStartCandidate:
    """A dependent that qualifies to start early on *predecessor*'s branch, with its keys and priority."""

    dependent: str
    predecessor: str
    task: Mapping[str, Any]
    keys: tuple[str, ...]
    priority: TaskPriority

    @property
    def rank_key(self) -> RankKey:
        return rank_key(self.priority, self.dependent)


@dataclass(frozen=True)
class EarlyStartShortlist:
    """Step (a)'s answer: the candidates in rank order, and the refused pairs."""

    candidates: tuple[EarlyStartCandidate, ...]
    refusals: tuple[EarlyStartRefusal, ...]


class DependentBranch(StrEnum):
    """What step (b) found of the dependent's task branch."""

    ABSENT = 'absent'
    STACKED = 'stacked'
    UNSTACKED = 'unstacked'
    UNREADABLE = 'unreadable'


@dataclass(frozen=True)
class PairReads:
    """Step (b)'s reads for one pair, made by the caller before the claim.

    ``dependent_row`` is the dependent's store row and its dependency statuses, re-read through
    ``orchestrator/src/orchestrator/dispatch_row.py::refresh_dispatch_row`` as every dispatch does.
    """

    predecessor: str
    predecessor_tip: str
    predecessor_checks: ChecksAtCommit
    dependent_branch: DependentBranch
    dependent_row: FreshRow

    def __post_init__(self) -> None:
        if not self.predecessor:
            raise ValueError(f'reads name no predecessor, got {self.predecessor!r}')
        if not isinstance(self.predecessor_tip, str) or not self.predecessor_tip:
            raise ValueError(
                f'reads for predecessor {self.predecessor!r} carry no tip, got {self.predecessor_tip!r}'
            )


@dataclass(frozen=True)
class StepReads:
    """Step (b)'s live statuses and its reads per pair, keyed by dependent."""

    statuses: Mapping[str, str]
    pairs: Mapping[str, PairReads]

    def __post_init__(self) -> None:
        for dependent, pair in self.pairs.items():
            row_id = str(pair.dependent_row.task.get('id'))
            if row_id != dependent:
                raise ValueError(f'reads keyed by {dependent!r} carry the row of {row_id!r}')


@dataclass(frozen=True)
class EarlyStartHolds:
    """The system holds the harness owns, every one stated by the caller."""

    review_checkpoint: bool
    merge_queue_halted: bool
    usage_paused: bool
    cost_ceiling_breached: bool


@dataclass(frozen=True)
class EarlyStartClaim:
    """A dependent started on its predecessor's branch: its keys, the commit it stacks on, its tier and row.

    ``task`` is the store row step (b) re-read, so the launch acts on it as an ordinary start does.
    """

    dependent: str
    predecessor: str
    keys: tuple[str, ...]
    stacked_commit: str
    tier: str
    task: dict[str, Any]


@dataclass(frozen=True)
class EarlyStartSelection:
    """Step (c)'s answer: the claim, if any, and every refused pair or hold."""

    claim: EarlyStartClaim | None
    refusals: tuple[EarlyStartRefusal, ...]


def picture_deps_verdict(
    task: Mapping[str, Any],
    picture: TaskPicture,
    status_map: dict[str, str],
    *,
    waiting_predecessors: frozenset[str] = frozenset(),
) -> DepsVerdict:
    """The dependency predicate for *task* over *status_map* and the picture's dependency caches."""
    return deps_verdict(
        dict(task), status_map, picture.tasks_by_id,
        external_status_cache=picture.external_statuses,
        external_resolver_failed=picture.external_resolver_failed,
        delivered_check_cache=picture.delivered_check_cache,
        terminal_dep_records=picture.terminal_dep_records,
        deferred_terminal_deps=picture.deferred_terminal_deps,
        waiting_predecessors=waiting_predecessors,
    )
