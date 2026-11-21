"""Counts consecutive unknown-tier picks at one priority gate; verify admission and the merge pool call ``note_pick`` once per pick."""
from __future__ import annotations

import logging
from collections.abc import Callable
from enum import StrEnum

from orchestrator.sentinel_l1 import SentinelL1
from orchestrator.task_priority import TaskPriority, TaskPriorityLookup, gate_rank

logger = logging.getLogger(__name__)

UNKNOWN_TIER_STREAK_LIMIT = 10


class PriorityGate(StrEnum):
    MERGE_POOL = 'merge_pool'
    VERIFY_ADMISSION = 'verify_admission'


class UnknownTierStreak:
    """Files one L1 from the tenth consecutive unknown-tier pick, and resolves it on the next known one.

    A no-op where no lookup is wired: an unknown tier there is expected, not a broken lookup.
    """

    def __init__(
        self,
        gate: PriorityGate,
        lookup: TaskPriorityLookup | None,
        *,
        escalate: Callable[[SentinelL1], None],
        resolve: Callable[[str, str], None],
    ) -> None:
        self._gate = gate
        self._wired = lookup is not None
        self._escalate = escalate
        self._resolve = resolve
        self._sentinel = f'__unknown_tier:{gate}__'
        self._count = 0
        # True at start: a previous process may have stopped with its escalation open.
        self._escalation_may_be_open = True

    def note_pick(self, priority: TaskPriority) -> None:
        if not self._wired:
            return
        if gate_rank(priority.tier) is None:
            self._count += 1
            if self._count >= UNKNOWN_TIER_STREAK_LIMIT:
                self._file()
            return
        self._count = 0
        if self._escalation_may_be_open:
            self._escalation_may_be_open = False
            resolution = f'the {self._gate} gate picked a task of known tier again'
            self._deliver(lambda: self._resolve(self._sentinel, resolution))

    def _file(self) -> None:
        self._escalation_may_be_open = True
        l1 = SentinelL1(
            sentinel=self._sentinel,
            category='infra_issue',
            summary=f'{self._count} consecutive picks at the {self._gate} gate had an unknown tier',
            detail=(
                f'Every one of those picks ranked as medium, so the {self._gate} gate\'s '
                'priority order degraded to age order. The task priority lookup is broken.'
            ),
            suggested_action='restore_the_task_priority_lookup',
        )
        self._deliver(lambda: self._escalate(l1))

    def _deliver(self, call: Callable[[], None]) -> None:
        try:
            call()
        except Exception:
            logger.warning('unknown-tier streak at the %s gate: delivery failed', self._gate, exc_info=True)
