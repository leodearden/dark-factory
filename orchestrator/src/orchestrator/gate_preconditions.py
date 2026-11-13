"""What a pure gate waited on when it escalated, and whether it now waits on more.

A resolution that held a gate behind a new precondition is "not yet", not an
approval. Rationale: docs/deterministic-runner.md §3 "Resuming a pure gate".
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

from orchestrator.dependency_verdict import task_dependency_ids, task_external_deps

logger = logging.getLogger(__name__)

GATE_ESCALATED_PRECONDITIONS_KEY = 'gate_escalated_preconditions'
_STAMP_SHAPE = (
    "{'dependencies': list[str], 'external_deps': list[str], 'milestone': <milestone or None>}"
)


@dataclass(frozen=True)
class GatePreconditions:
    """The local dependencies, external dependencies and milestone a task waits on."""

    dependencies: frozenset[str]
    external_deps: frozenset[str]
    milestone: object

    @classmethod
    def of_task(cls, task: dict) -> GatePreconditions:
        return cls(
            dependencies=frozenset(task_dependency_ids(task)),
            external_deps=frozenset(task_external_deps(task)),
            milestone=copy.deepcopy((task.get('metadata') or {}).get('milestone')),
        )

    def to_stamp(self) -> dict:
        return {
            'dependencies': sorted(self.dependencies),
            'external_deps': sorted(self.external_deps),
            'milestone': self.milestone,
        }

    @classmethod
    def from_stamp(cls, stamp: object) -> GatePreconditions:
        if not isinstance(stamp, dict) or 'milestone' not in stamp:
            raise ValueError(
                f'metadata.{GATE_ESCALATED_PRECONDITIONS_KEY} must be {_STAMP_SHAPE}; '
                f'got {stamp!r}'
            )
        return cls(
            dependencies=_string_set(stamp, 'dependencies'),
            external_deps=_string_set(stamp, 'external_deps'),
            milestone=copy.deepcopy(stamp['milestone']),
        )

    def waits_on_more_than(self, earlier: GatePreconditions) -> bool:
        return (
            not self.dependencies <= earlier.dependencies
            or not self.external_deps <= earlier.external_deps
            or (self.milestone is not None and self.milestone != earlier.milestone)
        )


def _string_set(stamp: dict, field: str) -> frozenset[str]:
    values = stamp.get(field)
    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
        raise ValueError(
            f'metadata.{GATE_ESCALATED_PRECONDITIONS_KEY}.{field} must be a list of '
            f'strings (whole record: {_STAMP_SHAPE}); got {values!r}'
        )
    return frozenset(values)


def gate_preconditions_stamp(task: dict) -> dict:
    """The metadata a pure gate's filing writes beside ``gate_escalated_at``."""
    return {GATE_ESCALATED_PRECONDITIONS_KEY: GatePreconditions.of_task(task).to_stamp()}


def gate_held_since_escalation(task: dict) -> bool:
    """Whether *task* now waits on more than its gate escalated under.

    A gate stamped before task 9520 carries no record and is never held. An
    unreadable record counts as held: re-asking the human beats skipping one.
    """
    stamp = (task.get('metadata') or {}).get(GATE_ESCALATED_PRECONDITIONS_KEY)
    if stamp is None:
        return False
    try:
        armed = GatePreconditions.from_stamp(stamp)
    except ValueError as exc:
        logger.warning(
            'Task %s: unreadable gate preconditions record, treating the gate as held: %s',
            task.get('id', '?'), exc,
        )
        return True
    return GatePreconditions.of_task(task).waits_on_more_than(armed)
