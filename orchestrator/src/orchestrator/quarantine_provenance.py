"""What a quarantine leaves behind, so a parked lane can be found and read later.

``orchestrator/src/orchestrator/git_ops.py::GitOps.quarantine_worktree``
parks a lane under ``quarantine_base`` and describes the parking as a
:class:`QuarantineOutcome`. This module turns that outcome into three records:

- the ``worktree_quarantined`` event payload (:meth:`QuarantineOutcome.event_data`);
- a point-in-time copy of the task's artifacts inside the parking
  (:func:`snapshot_into_parking`), because the lane's ``.task/plan.json`` is a
  symlink to the live task-root plan and so captures nothing by itself;
- a ``_quarantined_from`` back-pointer appended to the live task-root
  plan.json (:func:`stamp_quarantined_from`), which the next dispatch and a
  replanning architect read, and which ``create_plan`` carries forward
  through :func:`quarantined_from_of`. A parking of an earlier task's work
  under a recycled id gets no back-pointer: the live plan is not that task's.

Built only on ``orchestrator.artifacts``; git_ops, harness, lane_lifecycle and
plan_tools import this module, never the other way round.
"""
from __future__ import annotations

import logging
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType, NoneType
from typing import Any

from orchestrator.artifacts import Plan, TaskArtifacts

logger = logging.getLogger(__name__)

QUARANTINED_FROM_KEY = '_quarantined_from'

SNAPSHOT_ARTIFACTS: tuple[str, ...] = ('plan.json', 'metadata.json', 'review_state.json', 'iterations.jsonl')

STAMP_FIELDS: Mapping[str, tuple[type, ...]] = MappingProxyType({
    'branch': (str, NoneType),
    'tip_sha': (str, NoneType),
    'unique_commit_count': (int, NoneType),
    'quarantined_at': (str,),
    'reason': (str,),
    'parking': (str,),
})
"""Every ``_quarantined_from`` entry's keys, each with the value types it accepts. No field accepts a bool."""


@dataclass(frozen=True)
class QuarantineOutcome:
    """Where a quarantine parked a lane, and the task branch it renamed aside.

    ``dest`` is None when nothing moved. ``branch`` is the task's branch
    ``<prefix><bare>`` as renamed aside (full ref); None when no branch was
    renamed, because it did not exist or another registered worktree held
    it. It need not be the parking's checkout: a detached or foreign-branch
    lane moves with its own HEAD. With a None ``dest``, ``branch`` is set
    only when the move failed and the branch could not be renamed back: it
    names the ref the work is stranded on. The rule's one home is
    ``orchestrator/src/orchestrator/git_ops.py::GitOps.quarantine_worktree``.
    ``tip_sha`` and ``unique_commit_count`` describe ``branch``; the count is
    its commits not on main.
    """

    dest: Path | None
    branch: str | None = None
    tip_sha: str | None = None
    unique_commit_count: int | None = None

    def event_data(self, reason: str) -> dict[str, Any]:
        """The ``worktree_quarantined`` event payload for a quarantine done for *reason*."""
        return {
            'reason': reason,
            'dest': str(self.dest) if self.dest is not None else None,
            'branch': self.branch,
            'tip_sha': self.tip_sha,
            'unique_commit_count': self.unique_commit_count,
        }


NOT_RELOCATED = QuarantineOutcome(dest=None)


def snapshot_into_parking(artifacts: TaskArtifacts, parking: Path) -> tuple[str, ...]:
    """Copy each existing ``SNAPSHOT_ARTIFACTS`` file into ``<parking>/.task/`` as a regular file.

    A symlink already at a destination, or at ``<parking>/.task`` itself, is
    replaced rather than written through, so the live artifacts are never
    touched. Returns the names copied.
    """
    parked_task = _real_directory(parking / '.task')
    copied = []
    for name in SNAPSHOT_ARTIFACTS:
        source = artifacts.artifact_path(name)
        if not source.is_file():
            continue
        destination = parked_task / name
        destination.unlink(missing_ok=True)
        shutil.copyfile(source, destination)
        copied.append(name)
    return tuple(copied)


def _real_directory(path: Path) -> Path:
    if path.is_symlink():
        path.unlink()
    path.mkdir(parents=True, exist_ok=True)
    return path


def stamp_quarantined_from(artifacts: TaskArtifacts, stamp: Mapping[str, Any]) -> bool:
    """Append *stamp* to the plan's ``_quarantined_from`` list; False, writing nothing, when there is no plan.

    A body-only edit: the plan's steps and lifecycle, its confirmation
    included, are kept. Raises ValueError, writing nothing, for a stamp that
    does not match ``STAMP_FIELDS``, since :func:`quarantined_from_of` would
    silently drop it at the next re-plan.
    """
    problems = _stamp_problems(stamp)
    if problems:
        raise ValueError(f'malformed {QUARANTINED_FROM_KEY} stamp: {"; ".join(problems)}')
    with artifacts.plan_mutex():
        plan = artifacts.read_plan_model()
        if plan is None:
            return False
        document = plan.to_json()
        existing = document.get(QUARANTINED_FROM_KEY)
        document[QUARANTINED_FROM_KEY] = [*(existing if isinstance(existing, list) else []), dict(stamp)]
        artifacts.write_plan_model(Plan.from_json(document), mutex_held=True)
    return True


def record_quarantine_provenance(
    worktree: Path, task_id: str, outcome: QuarantineOutcome, reason: str, *, recycled_id: bool = False,
) -> None:
    """Snapshot the task's artifacts into the parking, then stamp the live plan. Never raises.

    *worktree* is the lane's path BEFORE the move, which is what its lane
    ``.task-meta`` root is named after. The two records are independent: a
    failed snapshot is logged and the stamp is still attempted. A
    *recycled_id* parking holds an earlier task's work under an id another
    task now holds, so it is snapshotted but never stamped onto that task's plan.
    """
    if outcome.dest is None:
        return
    parking = outcome.dest
    try:
        artifacts = TaskArtifacts.for_task(worktree, task_id)
    except Exception:
        logger.warning(_UNRECORDED, 'the task artifacts', worktree, task_id, parking, exc_info=True)
        return
    try:
        snapshot_into_parking(artifacts, parking)
    except Exception:
        logger.warning(_UNRECORDED, 'the artifact snapshot', worktree, task_id, parking, exc_info=True)
    if recycled_id:
        return
    try:
        stamp_quarantined_from(artifacts, _stamp_for(outcome, parking, reason))
    except Exception:
        logger.warning(
            _UNRECORDED, f'the {QUARANTINED_FROM_KEY} stamp', worktree, task_id, parking, exc_info=True,
        )


_UNRECORDED = 'quarantine provenance: could not record %s for the parking of %s (task %s) at %s'


def _stamp_for(outcome: QuarantineOutcome, parking: Path, reason: str) -> dict[str, Any]:
    return {
        'branch': outcome.branch,
        'tip_sha': outcome.tip_sha,
        'unique_commit_count': outcome.unique_commit_count,
        'quarantined_at': datetime.now(UTC).isoformat(),
        'reason': reason,
        'parking': str(parking),
    }


def quarantined_from_of(plan: Mapping[str, Any]) -> dict[str, Any]:
    """The plan's well-formed ``_quarantined_from`` entries, projected to ``STAMP_FIELDS``; {} when none survive."""
    entries = plan.get(QUARANTINED_FROM_KEY)
    if not isinstance(entries, list):
        return {}
    kept = [_projected(entry) for entry in entries if _well_formed(entry)]
    return {QUARANTINED_FROM_KEY: kept} if kept else {}


def _well_formed(entry: object) -> bool:
    return not _stamp_problems(entry)


def _stamp_problems(entry: object) -> list[str]:
    """Each way *entry* fails ``STAMP_FIELDS``: a missing key, or a value of a type it does not accept."""
    if not isinstance(entry, Mapping):
        return [f'not a mapping: {type(entry).__name__}']
    problems = []
    for key, accepted in STAMP_FIELDS.items():
        if key not in entry:
            problems.append(f'missing {key!r}')
        elif not isinstance(entry[key], accepted) or isinstance(entry[key], bool):
            problems.append(f'{key!r} is a {type(entry[key]).__name__}')
    return problems


def _projected(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {key: entry[key] for key in STAMP_FIELDS}
