#!/usr/bin/env python3
"""Whether a findings run has landed enough of its filed work for the next run.

``--run R --threshold T --project-root P`` weighs the tasks carrying
``x_finding_run=R`` by priority and prints a compact JSON verdict as its last
line. It exits 0 when the landed share reaches T, 75 (``os.EX_TEMPFAIL``) when
it has not yet, and 2 when it could not check. ``--pending`` reports the open
trigger chains. The measure is docs/quality-findings-contract.md §11; the
design is plans/completion-driven-triggers-prd.md C3.
"""

from __future__ import annotations

import enum
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Final, NamedTuple

from shared.task_statuses import TaskStatus

PRIORITY_WEIGHTS: Final[Mapping[str, int]] = MappingProxyType(
    {'critical': 4, 'high': 3, 'medium': 2, 'low': 1, 'polish': 1}
)
MISSING_PRIORITY: Final = 'medium'
SHARE_DIGITS: Final = 4


class Verdict(enum.StrEnum):
    PASS = 'pass'
    NOT_YET = 'not_yet'
    ERROR = 'error'


class FailureCode(enum.StrEnum):
    """Why a check could not conclude; the ``error`` field of an error verdict."""

    EMPTY_RUN = 'empty_run'
    UNKNOWN_PRIORITY = 'unknown_priority'
    STORE_UNREACHABLE = 'store_unreachable'
    TOOL_ERROR = 'tool_error'
    MALFORMED_REPLY = 'malformed_reply'


class UnknownPriority(ValueError):
    def __init__(self, task_id: object, value: object) -> None:
        super().__init__(f'task {task_id} has priority {value!r}, not one of {", ".join(PRIORITY_WEIGHTS)}')
        self.task_id = task_id
        self.value = value


class TierCounts(NamedTuple):
    done: int
    open: int
    cancelled: int


def threshold_error(value: float) -> str | None:
    """None iff *value* is a usable threshold: a fraction in (0, 1], so never nan."""
    if 0 < value <= 1:
        return None
    return f'threshold {value!r} is outside (0, 1]'


@dataclass(frozen=True)
class CompletionVerdict:
    run_id: str
    threshold: float
    share: float | None
    landed: int
    pending: int
    total: int
    vacuous: bool
    passed: bool
    by_priority: Mapping[str, TierCounts]

    @property
    def verdict(self) -> Verdict:
        if self.passed:
            return Verdict.PASS
        return Verdict.ERROR if self.total == 0 else Verdict.NOT_YET

    @property
    def error(self) -> FailureCode | None:
        return FailureCode.EMPTY_RUN if self.total == 0 else None

    def to_dict(self) -> dict[str, Any]:
        rendered: dict[str, Any] = {
            'run': self.run_id,
            'verdict': self.verdict.value,
            'share': None if self.share is None else round(self.share, SHARE_DIGITS),
            'threshold': self.threshold,
            'landed': self.landed,
            'pending': self.pending,
            'total': self.total,
            'vacuous': self.vacuous,
            'by_priority': {tier: list(counts) for tier, counts in self.by_priority.items()},
        }
        if self.error is not None:
            rendered['error'] = self.error.value
        return rendered

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(',', ':'))


def _tier(row: Mapping[str, object]) -> str:
    value = row.get('priority')
    if value is None:
        return MISSING_PRIORITY
    if not isinstance(value, str) or value not in PRIORITY_WEIGHTS:
        raise UnknownPriority(row.get('task_id'), value)
    return value


def _slot(status: object) -> int:
    if status == TaskStatus.DONE:
        return 0
    return 2 if status == TaskStatus.CANCELLED else 1


def _tally(tasks: Iterable[Mapping[str, object]]) -> Mapping[str, TierCounts]:
    counts = {tier: [0, 0, 0] for tier in PRIORITY_WEIGHTS}
    for row in tasks:
        counts[_tier(row)][_slot(row.get('status'))] += 1
    return MappingProxyType({tier: TierCounts(*slots) for tier, slots in counts.items() if any(slots)})


def weighted_completion(
    tasks: Iterable[Mapping[str, object]], threshold: float, *, run_id: str,
) -> CompletionVerdict:
    """The priority-weighted landed share of *tasks*, judged against *threshold*."""
    problem = threshold_error(threshold)
    if problem is not None:
        raise ValueError(problem)
    by_priority = _tally(tasks)
    landed = sum(PRIORITY_WEIGHTS[tier] * counts.done for tier, counts in by_priority.items())
    pending = sum(PRIORITY_WEIGHTS[tier] * counts.open for tier, counts in by_priority.items())
    total = sum(sum(counts) for counts in by_priority.values())
    weighed = landed + pending
    vacuous = total > 0 and weighed == 0
    return CompletionVerdict(
        run_id=run_id,
        threshold=threshold,
        share=landed / weighed if weighed else None,
        landed=landed,
        pending=pending,
        total=total,
        vacuous=vacuous,
        passed=vacuous or (weighed > 0 and Fraction(landed, weighed) >= Fraction(str(threshold))),
        by_priority=by_priority,
    )
