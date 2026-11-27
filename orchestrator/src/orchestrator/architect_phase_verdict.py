"""What one architect phase concluded, read from its on-disk artifacts: whether it confirmed a plan, and whether that plan conflicts with an exit report."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TypeGuard

from orchestrator.artifacts import Plan, PlanState, TaskArtifacts

_DETAIL_LIMIT = 2000
_TRUNCATED = '…'
_CONFIRMED_STATES = frozenset({PlanState.CONFIRMED, PlanState.STAMPED})


def is_confirmed(plan: Plan | None) -> TypeGuard[Plan]:
    """Whether *plan* has steps and its typed lifecycle reads confirmed or stamped."""
    return plan is not None and plan.state in _CONFIRMED_STATES


def newly_confirmed(plan: Plan | None, before: Plan | None) -> TypeGuard[Plan]:
    """Whether *plan* is confirmed and carries a confirmation *before*, the plan as it was earlier (None if absent), did not."""
    if not is_confirmed(plan):
        return False
    return before is None or plan.confirmed_since(before)


class ExitReport(StrEnum):
    """An exit the architect writes instead of a plan."""

    UNACTIONABLE_TASK = 'unactionable_task'
    FALSE_PREMISE = 'false_premise'
    ALREADY_DONE = 'already_done'
    READY_TO_MERGE = 'ready_to_merge'
    BLOCKING_DEPENDENCY = 'blocking_dependency'


_Reader = Callable[[TaskArtifacts], dict | None]
_Clearer = Callable[[TaskArtifacts], None]

_ACCESSORS: Mapping[ExitReport, tuple[_Reader, _Clearer]] = MappingProxyType({
    ExitReport.UNACTIONABLE_TASK: (TaskArtifacts.read_unactionable_task, TaskArtifacts.clear_unactionable_task),
    ExitReport.FALSE_PREMISE: (TaskArtifacts.read_false_premise, TaskArtifacts.clear_false_premise),
    ExitReport.ALREADY_DONE: (TaskArtifacts.read_already_done, TaskArtifacts.clear_already_done),
    ExitReport.READY_TO_MERGE: (TaskArtifacts.read_ready_to_merge, TaskArtifacts.clear_ready_to_merge),
    ExitReport.BLOCKING_DEPENDENCY: (
        TaskArtifacts.read_blocking_dependency, TaskArtifacts.clear_blocking_dependency,
    ),
})


def read_exit_reports(artifacts: TaskArtifacts) -> dict[ExitReport, dict]:
    """Every exit report on disk, keyed by its kind."""
    present = {report: read(artifacts) for report, (read, _) in _ACCESSORS.items()}
    return {report: payload for report, payload in present.items() if payload is not None}


def clear_exit_reports(artifacts: TaskArtifacts, reports: Iterable[ExitReport]) -> None:
    for report in reports:
        _, clear = _ACCESSORS[report]
        clear(artifacts)


@dataclass(frozen=True)
class PlanExitConflict:
    """One architect phase both confirmed a plan and wrote exit reports: two verdicts that need a ruling."""

    plan: Plan
    reports: Mapping[ExitReport, Mapping]

    def summary(self) -> str:
        kinds = ', '.join(report.value for report in self.reports)
        return (
            f'Conflicting architect verdicts: exit report(s) {kinds} written, '
            f'yet the same phase confirmed a {len(self.plan.steps)}-step plan'
        )

    def detail(self) -> str:
        """Both verdicts within the detail budget: every report quoted, and the plan's steps, since plan.json may not outlive the ruling."""
        header = (
            f'The architect phase confirmed a {len(self.plan.steps)}-step plan '
            f'(_finalized_at {self.plan.lifecycle.confirmed_at}) and also wrote '
            f'{len(self.reports)} exit report(s).'
        )
        report_lines = [
            f'{report.value}: {json.dumps(dict(payload), ensure_ascii=False, default=str)}'
            for report, payload in self.reports.items()
        ]
        step_lines = [
            f"{step.get('id')}: {' '.join(str(step.get('description', '')).split())}" for step in self.plan.steps
        ]
        headings = ('Exit reports:', 'Confirmed plan steps:')
        room = _DETAIL_LIMIT - len(header) - sum(len(heading) + 1 for heading in headings) - 1
        reports = _fitted(report_lines, room - min(len('\n'.join(step_lines)), room // 2))
        steps = _fitted(step_lines, room - len('\n'.join(reports)))
        return '\n'.join([header, headings[0], *reports, headings[1], *steps])[:_DETAIL_LIMIT]


def _fitted(lines: list[str], budget: int) -> list[str]:
    """*lines*, each cut to an equal share of *budget* characters, the newlines joining them included."""
    if not lines:
        return []
    share = max(budget // len(lines) - 1, 0)
    return [_truncated(line, share) for line in lines]


def _truncated(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:max(limit - len(_TRUNCATED), 0)] + _TRUNCATED


def find_plan_exit_conflict(
    plan: Plan | None, before: Plan | None, reports: Mapping[ExitReport, Mapping],
) -> PlanExitConflict | None:
    """The conflict when *plan* is newly confirmed since *before* and *reports* is non-empty, else None."""
    if not reports or not newly_confirmed(plan, before):
        return None
    return PlanExitConflict(plan=plan, reports=MappingProxyType(dict(reports)))
