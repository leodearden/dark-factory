"""What one architect phase concluded, read from its on-disk artifacts: whether it confirmed a plan, and whether that plan conflicts with an exit report."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from orchestrator.artifacts import Plan, PlanState, TaskArtifacts

_DETAIL_LIMIT = 2000
_TRUNCATED = '…'


def confirmed_at(plan: Mapping[str, object]) -> str | None:
    """The confirm stamp of *plan*; None unless its typed lifecycle reads confirmed or stamped."""
    parsed = Plan.from_json(plan)
    if parsed.state in (PlanState.CONFIRMED, PlanState.STAMPED):
        return parsed.lifecycle.confirmed_at
    return None


def confirmed_since(plan: Mapping[str, object], marker: str | None) -> bool:
    """Whether *plan* is confirmed with a stamp other than *marker*."""
    stamp = confirmed_at(plan)
    return stamp is not None and stamp != marker


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

    confirmed_at: str
    step_count: int
    reports: Mapping[ExitReport, Mapping]

    def summary(self) -> str:
        kinds = ', '.join(report.value for report in self.reports)
        return (
            f'Conflicting architect verdicts: exit report(s) {kinds} written, '
            f'yet the same phase confirmed a {self.step_count}-step plan'
        )

    def detail(self) -> str:
        header = (
            f'The architect phase confirmed a {self.step_count}-step plan (_finalized_at {self.confirmed_at}) '
            f'and also wrote {len(self.reports)} exit report(s), quoted below.'
        )
        labels = [f'{report.value}: ' for report in self.reports]
        room = _DETAIL_LIMIT - len(header) - sum(len(label) + 1 for label in labels)
        per_report = max(room // max(len(labels), 1), 0)
        lines = [
            label + _truncated(json.dumps(dict(payload), ensure_ascii=False, default=str), per_report)
            for label, payload in zip(labels, self.reports.values(), strict=True)
        ]
        return '\n'.join([header, *lines])[:_DETAIL_LIMIT]


def _truncated(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:max(limit - len(_TRUNCATED), 0)] + _TRUNCATED


def find_plan_exit_conflict(
    plan: Mapping[str, object], phase_marker: str | None, reports: Mapping[ExitReport, Mapping],
) -> PlanExitConflict | None:
    """The conflict when *plan* was confirmed since *phase_marker* and *reports* is non-empty, else None."""
    stamp = confirmed_at(plan)
    if not reports or stamp is None or stamp == phase_marker:
        return None
    return PlanExitConflict(
        confirmed_at=stamp,
        step_count=len(Plan.from_json(plan).steps),
        reports=MappingProxyType(dict(reports)),
    )
