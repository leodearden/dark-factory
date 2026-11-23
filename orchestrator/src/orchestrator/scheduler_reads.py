"""The values the read parts of the scheduler's ordinary pass return.

Task 10476, plans/early-start-on-predecessor-branch-prd.md §5.4: each writing
step of ``Scheduler.acquire_next`` is a read part, which consults the pass's
caches only by key and writes nothing, followed by a write part. The read
parts return these frozen values; ``Scheduler.build_task_picture`` assembles
them into a ``TaskPicture`` without running any write part.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field

from shared.mcp_envelope import resolver_failed

from orchestrator.overrides import OverrideRow, OverrideSource


@dataclass(frozen=True)
class TaskIndex:
    """The status and id indices of one task fetch, and its highest numeric id."""

    status_map: dict[str, str]
    tasks_by_id: dict[str, dict]
    max_id: int


def index_tasks(tasks: Iterable[dict]) -> TaskIndex:
    status_map: dict[str, str] = {}
    tasks_by_id: dict[str, dict] = {}
    max_id = 0
    for t in tasks:
        tid = str(t.get('id', ''))
        if not tid:
            continue
        status_map[tid] = t.get('status', 'unknown')
        tasks_by_id[tid] = t
        if tid.isdecimal():
            max_id = max(max_id, int(tid))
    return TaskIndex(status_map, tasks_by_id, max_id)


@dataclass(frozen=True)
class DepStatusRead:
    """The ``get_statuses`` answer for the dep ids a task fetch left out."""

    requested: tuple[str, ...] = ()
    answered: Mapping[str, str] = field(default_factory=dict)
    error: Exception | None = None

    @property
    def degraded(self) -> bool:
        return bool(self.requested) and resolver_failed(self.answered, self.error)

    @property
    def resolved(self) -> Mapping[str, str]:
        """The statuses to merge into the status map: none from a degraded read."""
        return {} if self.degraded else self.answered

    @property
    def still_missing(self) -> frozenset[str]:
        """Requested ids a clean answer left out."""
        return frozenset() if self.degraded else frozenset(self.requested).difference(self.answered)


@dataclass(frozen=True)
class TerminalRecordRead:
    """Terminal dep records: served from the cross-tick cache, or fetched within the budget.

    ``held``, ``new`` and ``revalidate`` are the fetch candidates by group,
    ``ordered`` the fetches the budget allowed, and ``fetched`` each of those
    ids' fresh record (``None``, or absent, when unreadable).
    """

    served: Mapping[str, dict] = field(default_factory=dict)
    held: frozenset[str] = frozenset()
    new: frozenset[str] = frozenset()
    revalidate: frozenset[str] = frozenset()
    ordered: tuple[str, ...] = ()
    fetched: Mapping[str, dict | None] = field(default_factory=dict)

    @property
    def records(self) -> dict[str, dict]:
        records = dict(self.served)
        records.update((dep_id, rec) for dep_id, rec in self.fetched.items() if rec is not None)
        return records

    @property
    def deferred(self) -> frozenset[str]:
        return (self.held | self.new | self.revalidate).difference(self.ordered)

    @property
    def deferred_new(self) -> frozenset[str]:
        """Never-cached deps the budget deferred: their dependents wait."""
        return self.deferred & self.new


@dataclass(frozen=True)
class ExternalStatusRead:
    """Pending tasks with cross-project deps, and the one batched status answer for them."""

    pending_with_external: tuple[dict, ...] = ()
    statuses: Mapping[str, str] = field(default_factory=dict)
    error: Exception | None = None


@dataclass(frozen=True)
class ErroredDep:
    """A checked dep left unresolved because a check ERRORED and none FAILED."""

    dep_id: str
    check_names: tuple[str | None, ...]
    unreadable: tuple[object, ...]


@dataclass(frozen=True)
class DeliveredCheckSweep:
    """One evaluation of the checked terminal deps at main.

    ``main_sha`` is ``None`` when there was nothing to check or main did not
    resolve; the projection is then empty.  ``digests`` names each checked
    dep's current descriptor digest, ``delivered_keys`` the cache keys found
    DELIVERED by a run this sweep, and ``resolved`` every dep whose verdict
    was definitive (served, delivered or failed).
    """

    main_sha: str | None = None
    checked_deps: Mapping[str, dict] = field(default_factory=dict)
    deps_by_dependent: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    digests: Mapping[str, str] = field(default_factory=dict)
    projection: Mapping[str, bool] = field(default_factory=dict)
    fail_details: Mapping[str, dict] = field(default_factory=dict)
    delivered_keys: tuple[tuple[str, str, str], ...] = ()
    resolved: tuple[str, ...] = ()
    errored: tuple[ErroredDep, ...] = ()


@dataclass(frozen=True)
class OverrideRead:
    """The stored override rows, and the ids whose rows the clean-up hands ``clear_terminal``.

    ``terminal_or_missing`` names the terminal statuses, and rows whose task
    the fetch lacks.
    """

    rows: Mapping[str, OverrideRow] = field(default_factory=dict)
    terminal_or_missing: frozenset[str] = frozenset()

    def in_force(self, status_map: Mapping[str, str], expired: Collection[str]) -> dict[str, OverrideRow]:
        """The rows the clean-up leaves: not terminal or missing, not TTL-expired, no unpending watchdog pin."""
        return {
            tid: row for tid, row in self.rows.items()
            if tid not in self.terminal_or_missing
            and tid not in expired
            and not is_unpending_watchdog_pin(row, status_map.get(tid))
        }


@dataclass(frozen=True)
class PriorityRead:
    effective_priorities: Mapping[str, str]
    transitive_counts: Mapping[str, int]


def is_unpending_watchdog_pin(row: OverrideRow, status: str | None) -> bool:
    """A starvation-watchdog pin whose task is no longer pending: its pin goes."""
    return row.source is OverrideSource.STARVATION_WATCHDOG and status != 'pending'
