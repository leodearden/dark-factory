"""Typed values for a task's scope moving past its declared files.

A writer agent that finds the plan's ``files`` list merely incomplete files a
:class:`ScopeRequest` (``plan_tools.py::report_missing_plan_file``) instead of
escalating. At the next writer boundary the workflow tries to lock each
requested path, and :func:`build_scope_drift_report` folds the outcome into a
:class:`ScopeDriftReport`: the paths whose locks were taken, and the paths
another task holds, each with its holder. That report is the ``scope_drift``
event's payload and the next briefing's input. Task 4371 extends this module
with its own ``reason`` value rather than adding a parallel one.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from shared.locking import is_file_path

from orchestrator.module_charter import derive_modules
from orchestrator.pin_reservation import Blocker

IMPLEMENT_SCOPE_EXPANDED = 'implement_scope_expanded'
IMPLEMENT_BLAST_RADIUS_LOCK_CONFLICT = 'implement_blast_radius_lock_conflict'

_DOT_SLASH = './'
_TASK_STATE_DIR = '.task'


class InvalidScopeRequest(ValueError):
    """A scope request whose ``field`` holds a value the boundary refuses."""

    def __init__(self, field: str, message: str) -> None:
        super().__init__(message)
        self.field = field


def _normalized_path(path: str) -> str:
    while path.startswith(_DOT_SLASH):
        path = path[len(_DOT_SLASH):]
    return path


def _path_refusal(path: str) -> str | None:
    segments = path.split('/')
    if not path:
        return 'is empty'
    if path.startswith('/'):
        return 'is absolute; give a repo-relative path'
    if '..' in segments:
        return "contains a '..' segment"
    if segments[0] == _TASK_STATE_DIR:
        return 'is orchestrator state under .task/'
    if not is_file_path(path):
        return 'is directory-shaped; name a file'
    return None


@dataclass(frozen=True)
class ScopeRequest:
    """An agent's request to add one existing file to its task's declared scope."""

    path: str
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, str):
            raise InvalidScopeRequest('path', f'path {self.path!r} is not a string')
        if not isinstance(self.reason, str):
            raise InvalidScopeRequest('reason', f'reason {self.reason!r} is not a string')
        path = _normalized_path(self.path)
        refusal = _path_refusal(path)
        if refusal is not None:
            raise InvalidScopeRequest('path', f'path {self.path!r} {refusal}')
        if not self.reason.strip():
            raise InvalidScopeRequest('reason', f'reason {self.reason!r} is empty')
        object.__setattr__(self, 'path', path)

    def to_mapping(self) -> dict[str, str]:
        return {'path': self.path, 'reason': self.reason}

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> ScopeRequest:
        for field in ('path', 'reason'):
            if field not in mapping:
                raise InvalidScopeRequest(field, f'{field} is missing from {dict(mapping)!r}')
        return cls(mapping['path'], mapping['reason'])


def pending_scope_paths(
    requests: Iterable[ScopeRequest], plan_files: Iterable[str]
) -> tuple[str, ...]:
    """Requested paths not yet declared, de-duplicated in request order."""
    declared = {_normalized_path(entry) for entry in plan_files}
    pending: dict[str, None] = {}
    for request in requests:
        if request.path not in declared:
            pending.setdefault(request.path)
    return tuple(pending)


@dataclass(frozen=True)
class ScopeDriftReport:
    """What one writer boundary did with the scope requests it found."""

    reason: str
    role: str
    iteration: int
    requested: tuple[str, ...]
    acquired: tuple[str, ...]
    contested: tuple[tuple[str, str], ...]

    def event_data(self) -> dict[str, Any]:
        return {
            'reason': self.reason,
            'role': self.role,
            'iteration': self.iteration,
            'requested': list(self.requested),
            'acquired': list(self.acquired),
            'contested': dict(self.contested),
        }


def build_scope_drift_report(
    requested: Iterable[str],
    contested: Iterable[Blocker],
    *,
    lock_depth: int,
    role: str,
    iteration: int,
    task_id: str,
) -> ScopeDriftReport:
    """Split *requested* into acquired paths and contested ``(path, holder)`` pairs.

    A path is contested when its lock key matches a blocker's module; the
    holder recorded is the lowest blocker in :class:`Blocker` order. Every
    other path, including one that derives no lock key, counts as acquired.
    """
    holder_by_module: dict[str, str] = {}
    for blocker in sorted(contested):
        holder_by_module.setdefault(blocker.module, blocker.owner)
    paths = tuple(requested)
    acquired: list[str] = []
    contested_paths: list[tuple[str, str]] = []
    for path in paths:
        holders = [
            holder_by_module[key]
            for key in derive_modules([path], lock_depth, task_id=task_id)
            if key in holder_by_module
        ]
        if holders:
            contested_paths.append((path, holders[0]))
        else:
            acquired.append(path)
    return ScopeDriftReport(
        reason=(
            IMPLEMENT_BLAST_RADIUS_LOCK_CONFLICT if contested_paths else IMPLEMENT_SCOPE_EXPANDED
        ),
        role=role,
        iteration=iteration,
        requested=paths,
        acquired=tuple(acquired),
        contested=tuple(contested_paths),
    )
