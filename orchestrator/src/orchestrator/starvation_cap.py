"""What the auto-pin cap L1 says about the waiting cohort, and when it is filed, refreshed or resolved.

The cohort is every pin-gate firing the auto-pin cap turned away, waiting for
a watchdog pin slot.  The L1 reports a CHANGE in that cohort, not the
standing backlog, and only when the watchdog can point at something a human
should look at: a blocker whose holder is not live, or a cohort none of whose
members has a recorded blocker while some task holds a lock.  An empty lock
table (the first ticks after a restart, before holders re-dispatch) explains
nothing either way, so it alarms nothing.  The rule, its thresholds and the liveness
walk are justified in plans/scheduler-dispatch-scoring-and-lock-layer-prd.md
C4 Bounds.

- FILE when no cap L1 is open, the cohort alarms, and it is news against the
  last announcement: a new non-live holder, a new first-blocker module, or
  growth to ``size + max(GROWTH_FLOOR, ceil(GROWTH_RATIO * size))``.
- REFRESH the open L1 in place whenever the cohort's rendering changes.
- RESOLVE the open L1 once the cohort no longer alarms.

The rendering carries no ages or counts (each task's ``starvation_detected``
event holds those), so a refresh happens only on a membership, module or
wedge change.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from escalation.categories import EscalationCategory
from escalation.file_or_refresh import RefreshableAlarm, file_or_refresh

from orchestrator.blocker_liveness import LockWorld, NonLiveHolder, non_live_holder
from orchestrator.starvation_state import CapBaseline, WatchdogState
from orchestrator.starvation_watchdog import (
    AUTO_PIN_CAP_SENTINEL,
    STARVATION_WATCHDOG_MARKER,
    STARVATION_WATCHDOG_ROLE,
    StarvationGate,
)

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue

logger = logging.getLogger(__name__)

GROWTH_RATIO = 0.5
GROWTH_FLOOR = 10
_SUMMARY_LIMIT = 200
_NO_BLOCKER = 'no recorded blocker'

CAP_ALARM = RefreshableAlarm(
    task_id=AUTO_PIN_CAP_SENTINEL,
    level=1,
    agent_role=STARVATION_WATCHDOG_ROLE,
    severity='blocking',
    category=EscalationCategory.RISK_IDENTIFIED,
    suggested_action=(
        'Check the named non-live holders: finish, unblock or release a wedged '
        'one. Pin by hand any waiting task that must go first.'
    ),
)


@dataclass(frozen=True, order=True)
class CohortMember:
    task_id: str
    gate: StarvationGate
    first_module: str | None


@dataclass(frozen=True)
class Wedge:
    """A non-live holder and the cohort members waiting behind it."""

    holder: NonLiveHolder
    waiters: tuple[str, ...]


@dataclass(frozen=True)
class CapReport:
    cap: int
    members: tuple[CohortMember, ...]
    wedges: tuple[Wedge, ...]
    locks_held: bool

    @property
    def unexplained(self) -> bool:
        """Members wait while locks are held, yet none has a recorded blocker the watchdog could name."""
        return (
            self.locks_held
            and bool(self.members)
            and all(m.first_module is None for m in self.members)
        )

    @property
    def alarming(self) -> bool:
        return bool(self.wedges) or self.unexplained

    @property
    def modules(self) -> frozenset[str]:
        return frozenset(m.first_module for m in self.members if m.first_module is not None)

    @property
    def subjects(self) -> frozenset[str]:
        """Who a reader should look at: the non-live holders, or the members when unexplained."""
        owners = frozenset(w.holder.blocker.owner for w in self.wedges)
        if self.unexplained:
            return owners | frozenset(m.task_id for m in self.members)
        return owners

    def announced(self) -> CapBaseline:
        return CapBaseline(size=len(self.members), modules=self.modules, subjects=self.subjects)


class CapAction(StrEnum):
    NONE = 'none'
    FILE = 'file'
    REFRESH = 'refresh'
    RESOLVE = 'resolve'


def cap_report(
    waiting: Mapping[str, StarvationGate], world: LockWorld, cap: int, *, locks_held: bool,
) -> CapReport:
    """The cohort waiting on a full cap, with every non-live holder behind any member's blocker.

    *locks_held* is whether any task holds a lock this tick.
    """
    members: list[CohortMember] = []
    waiters: dict[NonLiveHolder, list[str]] = {}
    for tid in sorted(waiting):
        blockers = world.blockers_of(tid)
        members.append(
            CohortMember(tid, waiting[tid], blockers[0].module if blockers else None),
        )
        for blocker in blockers:
            found = non_live_holder(blocker, world)
            if found is not None and tid not in waiters.setdefault(found, []):
                waiters[found].append(tid)
    wedges = tuple(
        Wedge(holder, tuple(tids))
        for holder, tids in sorted(waiters.items(), key=lambda kv: (kv[0].blocker, kv[0].reason))
    )
    return CapReport(cap=cap, members=tuple(members), wedges=wedges, locks_held=locks_held)


def is_news(report: CapReport, baseline: CapBaseline) -> bool:
    """Whether *report* differs from what was last announced by more than drift."""
    growth_bound = baseline.size + max(GROWTH_FLOOR, math.ceil(GROWTH_RATIO * baseline.size))
    return (
        not report.subjects <= baseline.subjects
        or not report.modules <= baseline.modules
        or len(report.members) >= growth_bound
    )


def cap_action(report: CapReport, baseline: CapBaseline, open_detail: str | None) -> CapAction:
    """What to do with the cap L1, given its open record's detail (None when none is open)."""
    if not report.alarming:
        return CapAction.NONE if open_detail is None else CapAction.RESOLVE
    if open_detail is not None:
        return CapAction.NONE if cap_text(report)[1] == open_detail else CapAction.REFRESH
    return CapAction.FILE if is_news(report, baseline) else CapAction.NONE


def next_baseline(action: CapAction, report: CapReport, baseline: CapBaseline) -> CapBaseline:
    """An announcement replaces the baseline; an empty cohort ends the episode."""
    if action in (CapAction.FILE, CapAction.REFRESH):
        return report.announced()
    if not report.members:
        return CapBaseline.EMPTY
    return baseline


def cap_text(report: CapReport) -> tuple[str, str]:
    """The cap L1's (summary, detail): no ages or counts, so it changes only with the cohort."""
    size = len(report.members)
    if report.wedges:
        holders = ', '.join(
            f'task {w.holder.blocker.owner} ({w.holder.reason.value})' for w in report.wedges
        )
        summary = (
            f'{STARVATION_WATCHDOG_MARKER}: auto-pin cap full; {size} starving task(s) '
            f'wait behind non-live holder(s) {holders}'
        )
    else:
        summary = (
            f'{STARVATION_WATCHDOG_MARKER}: auto-pin cap full; {size} starving task(s) '
            f'wait with no recorded blocker'
        )
    detail = (
        f'The starvation watchdog fired on these tasks, but '
        f'starvation_watchdog.auto_pin_max_concurrent={report.cap} watchdog pins '
        f'already exist, so each waits for a pin slot and none has a per-task '
        f'escalation.  Per-task ages and counts are in its starvation_detected '
        f'event.\n\nNon-live holders:\n{_wedge_lines(report)}\n\n'
        f'Waiting tasks by first blocker module:\n{_member_lines(report)}'
    )
    return _bounded(summary), detail


def _wedge_lines(report: CapReport) -> str:
    if not report.wedges:
        return '  (none: no waiting task has a recorded blocker, so the watchdog cannot name one)'
    return '\n'.join(
        f'  - task {w.holder.blocker.owner} ({w.holder.reason.value}), '
        f'{w.holder.blocker.kind.value} module {w.holder.blocker.module}; waiting: '
        + ', '.join(f'task {tid}' for tid in w.waiters)
        for w in report.wedges
    )


def _member_lines(report: CapReport) -> str:
    groups: dict[str, list[CohortMember]] = {}
    for member in report.members:
        groups.setdefault(member.first_module or _NO_BLOCKER, []).append(member)
    return '\n'.join(
        f'  - {module}: '
        + ', '.join(f'task {m.task_id} ({m.gate.value})' for m in groups[module])
        for module in sorted(groups)
    )


def _bounded(summary: str) -> str:
    if len(summary) <= _SUMMARY_LIMIT:
        return summary
    return summary[: _SUMMARY_LIMIT - 3] + '...'


def _resolution(report: CapReport) -> str:
    if not report.members:
        return 'auto-resolved: no starving task waits on a watchdog pin slot'
    return 'auto-resolved: every blocker of the waiting tasks has a live holder'


def sync_cap_l1(queue: EscalationQueue | None, report: CapReport, state: WatchdogState) -> bool:
    """Bring the cap L1 in line with *report*; whether the queue now reflects it.

    The baseline is recorded only after the queue write succeeds, so a failed
    write is retried on the next call.  Any open level-1 cap record counts,
    whoever filed it, so one a previous deploy left is refreshed or resolved
    rather than duplicated.  Never raises.
    """
    if queue is None:
        return True
    try:
        open_records = queue.get_by_task(AUTO_PIN_CAP_SENTINEL, status='pending', level=1)
        open_detail = open_records[0].detail if open_records else None
        action = cap_action(report, state.baseline(), open_detail)
        if action in (CapAction.FILE, CapAction.REFRESH):
            summary, detail = cap_text(report)
            file_or_refresh(
                queue, CAP_ALARM, summary=summary, detail=detail, filing_claimant_run_id=None,
            )
        elif action is CapAction.RESOLVE:
            for esc in open_records:
                queue.resolve(esc.id, _resolution(report), resolved_by=STARVATION_WATCHDOG_ROLE)
    except Exception:
        logger.warning(
            'Starvation watchdog: cannot bring the auto-pin cap L1 in line; retrying '
            'next tick', exc_info=True,
        )
        return False
    state.record_baseline(next_baseline(action, report, state.baseline()))
    return True
