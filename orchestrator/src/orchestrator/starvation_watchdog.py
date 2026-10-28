"""The starvation watchdog's predicate, its lock-refusal counter and its filing text.

The predicate (plans/scheduler-dispatch-scoring-and-lock-layer-prd.md C4, as
retuned by esc-4851-3): a task is starving only when it has BOTH a durable
eligibility age past its gate's floor AND a stuck signal observed while it was
a dispatch candidate.  Age alone is what every task in a deep backlog has, so
age alone never fires.

- ``dual`` — top-candidate skips >= ``skip_threshold`` and age >= ``idle_secs``.
- ``lock_refused`` — lock refusals >= ``skip_threshold`` and age >=
  ``idle_only_secs``.  A refusal is a scan in which the task was refused its
  locks while a LOWER-ranked candidate dispatched, so priority does not explain
  the wait.  A task the scan never reaches accrues none.
- ``park_blocked`` — the task's live park has a module held by another task,
  and the park is at least ``park_blocked_age_secs`` old.  It takes precedence:
  a task parked behind a holder is never routed to a pinning gate.

This module owns the predicate and the text; ``Scheduler`` gathers the facts
and delivers the consequence (event, pin, escalation).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from orchestrator.config import StarvationWatchdogConfig
from orchestrator.pin_reservation import Blocker
from orchestrator.sentinel_l1 import SentinelL1

STARVATION_WATCHDOG_ROLE = 'orchestrator-starvation-watchdog'
# Grep anchor leading every watchdog summary; tests and classifiers key on it.
STARVATION_WATCHDOG_MARKER = 'STARVATION_WATCHDOG'
AUTO_PIN_CAP_SENTINEL = '__starvation_auto_pin_cap__'
_SUMMARY_LIMIT = 200


class StarvationGate(StrEnum):
    DUAL = 'dual'
    LOCK_REFUSED = 'lock_refused'
    PARK_BLOCKED = 'park_blocked'

    @property
    def pins(self) -> bool:
        """Whether a firing earns an auto-pin; a park-blocked task needs its holder to finish."""
        return self is not StarvationGate.PARK_BLOCKED


class StarvationDisposition(StrEnum):
    AUTO_PINNED = 'auto_pinned'
    OPERATOR_OVERRIDE = 'operator_override'
    CAP_REACHED = 'cap_reached'
    ESCALATED_L1 = 'escalated_l1'


@dataclass(frozen=True)
class ParkBlock:
    blocking_holder: str
    blocking_module: str
    park_age_secs: float


@dataclass(frozen=True)
class CandidateFacts:
    task_id: str
    eligible_age_secs: float
    skip_count: int
    lock_refusals: int
    park_block: ParkBlock | None


@dataclass(frozen=True)
class Finding:
    facts: CandidateFacts
    gate: StarvationGate
    blockers: tuple[Blocker, ...]


def classify(facts: CandidateFacts, cfg: StarvationWatchdogConfig) -> StarvationGate | None:
    if facts.park_block is not None:
        if facts.park_block.park_age_secs >= cfg.park_blocked_age_secs:
            return StarvationGate.PARK_BLOCKED
        return None
    age = facts.eligible_age_secs
    if facts.skip_count >= cfg.skip_threshold and age >= cfg.idle_secs:
        return StarvationGate.DUAL
    if facts.lock_refusals >= cfg.skip_threshold and age >= cfg.idle_only_secs:
        return StarvationGate.LOCK_REFUSED
    return None


class LockRefusals:
    """Per-task count of scans that refused the task its locks while a lower-ranked task dispatched."""

    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()

    def note_scan(self, refused: Iterable[str]) -> None:
        self._counts.update(set(refused))

    def count(self, task_id: str) -> int:
        return self._counts.get(task_id, 0)

    def forget(self, task_id: str) -> None:
        self._counts.pop(task_id, None)

    def task_ids(self) -> frozenset[str]:
        return frozenset(self._counts)


def _hours(secs: float) -> str:
    return f'{secs / 3600:.1f}h'


def _first_blocker(blockers: Sequence[Blocker]) -> str:
    if not blockers:
        return 'no live blocker recorded'
    first = blockers[0]
    return f'{first.kind.value} by task {first.owner}'


def _blocker_lines(blockers: Sequence[Blocker]) -> str:
    if not blockers:
        return '  (none: the lock table shows no holder or park on its modules now)'
    return '\n'.join(f'  - {b.module}: {b.kind.value} by task {b.owner}' for b in blockers)


def escalation_text(
    facts: CandidateFacts,
    gate: StarvationGate,
    cfg: StarvationWatchdogConfig,
    blockers: Sequence[Blocker],
) -> tuple[str, str]:
    """The (summary, detail) of one task's watchdog escalation."""
    tid, age = facts.task_id, facts.eligible_age_secs
    head = f'{STARVATION_WATCHDOG_MARKER} [{gate.value}]: task {tid}'
    counts = (
        f'eligible_age_secs={age:.0f}, skip_count={facts.skip_count}, '
        f'lock_refusals={facts.lock_refusals}, skip_threshold={cfg.skip_threshold}'
    )
    if gate is StarvationGate.PARK_BLOCKED:
        park = facts.park_block
        assert park is not None
        summary = (
            f'{head} parked {_hours(park.park_age_secs)} behind live holder '
            f'task {park.blocking_holder}'
        )
        detail = (
            f'Task {tid} owns a live fairness park, and its module '
            f'{park.blocking_module} is held by task {park.blocking_holder}.  The '
            f'park is {park.park_age_secs:.0f}s old (park_blocked_age_secs='
            f'{cfg.park_blocked_age_secs}).  An auto-pin cannot help: the task '
            f'waits on that holder finishing, so check whether task '
            f'{park.blocking_holder} is progressing or wedged.\n\n'
            f'{counts}\nBlockers:\n{_blocker_lines(blockers)}'
        )
    elif gate is StarvationGate.DUAL:
        summary = (
            f'{head} eligible {_hours(age)}, {facts.skip_count} top-skips '
            f'(threshold {cfg.skip_threshold}); first blocker {_first_blocker(blockers)}'
        )
        detail = (
            f'Task {tid} has been dispatch-eligible for {age:.0f}s (idle_secs='
            f'{cfg.idle_secs}) and was skipped {facts.skip_count} times as the '
            f'top-scored candidate because it could not take its module locks.  '
            f'The fairness parks have not resolved the contention.\n\n'
            f'{counts}\nBlockers:\n{_blocker_lines(blockers)}'
        )
    else:
        summary = (
            f'{head} eligible {_hours(age)}, refused its locks in '
            f'{facts.lock_refusals} scans while lower-ranked tasks dispatched; '
            f'first blocker {_first_blocker(blockers)}'
        )
        detail = (
            f'Task {tid} has been dispatch-eligible for {age:.0f}s (idle_only_secs='
            f'{cfg.idle_only_secs}) and in {facts.lock_refusals} scans was refused '
            f'its module locks while a lower-ranked candidate dispatched, so its '
            f'priority does not explain the wait: its locks do.\n\n'
            f'{counts}\nBlockers:\n{_blocker_lines(blockers)}'
        )
    return summary[:_SUMMARY_LIMIT], detail


def cap_escalation(cohort: Sequence[Finding], cfg: StarvationWatchdogConfig) -> SentinelL1:
    """One L1 naming every pin-worthy firing the auto-pin cap turned away."""
    lines = '\n'.join(
        f'- task {f.facts.task_id}: gate {f.gate.value}, eligible_age_secs='
        f'{f.facts.eligible_age_secs:.0f}, skip_count={f.facts.skip_count}, '
        f'lock_refusals={f.facts.lock_refusals}, first blocker '
        + (f'{f.blockers[0].module} {_first_blocker(f.blockers)}' if f.blockers else 'none recorded')
        for f in cohort
    )
    return SentinelL1(
        AUTO_PIN_CAP_SENTINEL,
        category='risk_identified',
        summary=(
            f'{STARVATION_WATCHDOG_MARKER}: {len(cohort)} starving task(s) not '
            f'auto-pinned: the cap of {cfg.auto_pin_max_concurrent} watchdog pins is full'
        ),
        detail=(
            f'The starvation watchdog fired on these tasks, but '
            f'starvation_watchdog.auto_pin_max_concurrent='
            f'{cfg.auto_pin_max_concurrent} watchdog-owned pins already exist, so '
            f'none was pinned and no per-task escalation was filed:\n{lines}\n\n'
            f'Several tasks starving at once usually share one long lock holder.'
        ),
        suggested_action=(
            'Check the first blockers for a wedged holder; pin by hand any task '
            'that must go first, or raise auto_pin_max_concurrent.'
        ),
    )
