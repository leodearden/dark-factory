"""Done-task memories that name a path the task branch added and then removed before merging.

Git decides which paths never landed (``services/unlanded_branch_paths.py``).
This module matches those paths in the done task's Mem0 memories and emits one
advisory Stage-1 flag per task; Stage 2 acts on it. Task 5626, after
solar_challenge task 107's memory described a test file its branch deleted
before the merge.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from fused_memory.models.reconciliation import EventType, ReconciliationEvent
from fused_memory.services.unlanded_branch_paths import (
    ProbeOutcome,
    UnlandedPath,
    UnlandedPathsProbe,
    probe_unlanded_paths,
)

logger = logging.getLogger(__name__)

UNLANDED_PATH_FLAG_TYPE = 'memory_cites_unlanded_path'
UNLANDED_PATH_FLAG_CATEGORY = 'memory_stale'

_SHORT_SHA_LENGTH = 10

#: One memory that names unlanded paths: its id, and the paths it names.
Offender = tuple[str, tuple[UnlandedPath, ...]]


@dataclass(frozen=True)
class DoneTransition:
    task_id: str
    commit: str | None


def _own_merge_commit(payload: dict) -> str | None:
    """The task's own merge commit; a found_on_main commit may be another task's merge."""
    provenance = payload.get('done_provenance')
    if not isinstance(provenance, dict) or provenance.get('kind') != 'merged':
        return None
    commit = provenance.get('commit')
    return commit if isinstance(commit, str) and commit else None


def _is_done_transition(event: ReconciliationEvent) -> bool:
    return (
        event.type == EventType.task_status_changed
        and event.payload.get('new_status') == 'done'
        and event.payload.get('task_id') is not None
    )


def select_done_transitions(events: Sequence[ReconciliationEvent]) -> tuple[DoneTransition, ...]:
    """One transition per task that went done, carrying its LAST event's own merge commit."""
    latest: dict[str, DoneTransition] = {}
    for event in filter(_is_done_transition, events):
        task_id = str(event.payload['task_id'])
        latest[task_id] = DoneTransition(task_id=task_id, commit=_own_merge_commit(event.payload))
    return tuple(latest.values())


def mentions_path(content: str, path: str) -> bool:
    """True iff *content* names *path* itself, not a longer filename or a deeper path."""
    return re.search(r'(?<![\w.-])' + re.escape(path) + r'(?![\w/-])', content) is not None


def _short(sha: str) -> str:
    return sha[:_SHORT_SHA_LENGTH]


def _describe_path(unlanded: UnlandedPath) -> str:
    commits = ', '.join(_short(sha) for sha in unlanded.branch_commits)
    return f'{unlanded.path} (touched by branch commits {commits})'


def _describe_offender(offender: Offender) -> str:
    memory_id, paths = offender
    return f'memory {memory_id} names ' + '; '.join(_describe_path(p) for p in paths)


def build_unlanded_path_flag(task_id: object, probe: UnlandedPathsProbe, offenders: Sequence[Offender]) -> dict:
    """The FINDING_ITEM_SCHEMA-shaped flag for one done task. Pure."""
    tid = str(task_id)
    merge = _short(probe.merge_commit)
    description = (
        f'Task {tid} is done by merge commit {merge}. Git shows paths that commits '
        f'on its branch touched and that are absent from both the tree and the net diff of {merge}: '
        + ' | '.join(_describe_offender(o) for o in offenders)
        + '. That is evidence the memory may present a never-landed artifact as delivered.'
    )
    return {
        'description': description,
        'severity': 'moderate',
        'actionable': True,
        'task_id': tid,
        'flag_type': UNLANDED_PATH_FLAG_TYPE,
        'category': UNLANDED_PATH_FLAG_CATEGORY,
        'suggested_action': (
            'Read each cited memory. If it presents a listed path as part of the '
            "task's delivered work, amend it in place with update_memory (keep the "
            'memory id) to record that the branch removed that path before merge '
            f'{merge}. If the memory already records the removal, dismiss this flag.'
        ),
        'cited_memories': [{'memory_id': memory_id, 'store': 'mem0'} for memory_id, _ in offenders],
    }


@dataclass(frozen=True)
class UnlandedPathCheckResult:
    flags: tuple[dict, ...]
    checked: int
    skipped: int
    errors: int

    @classmethod
    def empty(cls) -> UnlandedPathCheckResult:
        return cls(flags=(), checked=0, skipped=0, errors=0)

    def stats(self) -> dict[str, int]:
        return {
            'unlanded_path_check_checked': self.checked,
            'unlanded_path_check_skipped': self.skipped,
            'unlanded_path_check_errors': self.errors,
            'unlanded_path_check_flags_emitted': len(self.flags),
        }

    def __add__(self, other: UnlandedPathCheckResult) -> UnlandedPathCheckResult:
        return UnlandedPathCheckResult(
            flags=self.flags + other.flags,
            checked=self.checked + other.checked,
            skipped=self.skipped + other.skipped,
            errors=self.errors + other.errors,
        )


_SKIPPED = UnlandedPathCheckResult(flags=(), checked=0, skipped=1, errors=0)
_ERRORED = UnlandedPathCheckResult(flags=(), checked=0, skipped=0, errors=1)


def _memory_ids(rows: object) -> tuple[str, ...]:
    if not isinstance(rows, list):
        return ()
    return tuple(str(row['id']) for row in rows if isinstance(row, dict) and row.get('id') is not None)


async def _memory_content(memory_service, project_id: str, memory_id: str) -> str | None:
    memory = await memory_service.get_memory_by_id(project_id=project_id, memory_id=memory_id)
    content = memory.get('content') if isinstance(memory, dict) else None
    return content if isinstance(content, str) else None


async def _offending_memories(
    memory_service, project_id: str, task_id: str, probe: UnlandedPathsProbe, log: logging.Logger,
) -> tuple[tuple[Offender, ...], int]:
    """The task's memories that name an unlanded path, and how many reads failed."""
    try:
        rows = await memory_service.get_memories_by_metadata(project_id=project_id, filters={'task_id': task_id})
    except Exception:
        log.exception('unlanded_path_check: listing memories of task %s (merge %s) failed', task_id, probe.merge_commit)
        return (), 1
    offenders: list[Offender] = []
    errors = 0
    for memory_id in _memory_ids(rows):
        try:
            content = await _memory_content(memory_service, project_id, memory_id)
        except Exception:
            log.exception('unlanded_path_check: reading memory %s of task %s failed', memory_id, task_id)
            errors += 1
            continue
        if content is None:
            continue
        named = tuple(p for p in probe.unlanded if mentions_path(content, p.path))
        if named:
            offenders.append((memory_id, named))
    return tuple(offenders), errors


async def _check_transition(
    memory_service, project_id: str, project_root: str, transition: DoneTransition, log: logging.Logger,
) -> UnlandedPathCheckResult:
    if transition.commit is None:
        return _SKIPPED
    probe = await probe_unlanded_paths(project_root, transition.commit)
    if probe.outcome is ProbeOutcome.UNKNOWN:
        log.warning('unlanded_path_check: git could not analyze task %s merge %s', transition.task_id, transition.commit)
        return _ERRORED
    if probe.outcome is ProbeOutcome.NOT_A_MERGE:
        return _SKIPPED
    if not probe.unlanded:
        return UnlandedPathCheckResult(flags=(), checked=1, skipped=0, errors=0)
    offenders, errors = await _offending_memories(memory_service, project_id, transition.task_id, probe, log)
    flags = (build_unlanded_path_flag(transition.task_id, probe, offenders),) if offenders else ()
    return UnlandedPathCheckResult(flags=flags, checked=1, skipped=0, errors=errors)


async def check_unlanded_path_claims(
    memory_service,
    project_id: str,
    project_root: str,
    events: Sequence[ReconciliationEvent],
    *,
    log: logging.Logger = logger,
) -> UnlandedPathCheckResult:
    """Flag each task that went done in *events* whose Mem0 memories name a path its branch never landed.

    Total: a per-transition failure is logged and tallied into ``errors``, and
    only cancellation-class exceptions propagate, so callers need no guard.
    """
    result = UnlandedPathCheckResult.empty()
    for transition in select_done_transitions(events):
        try:
            result += await _check_transition(memory_service, project_id, project_root, transition, log)
        except Exception:
            log.exception(
                'unlanded_path_check: task %s (commit %s) failed', transition.task_id, transition.commit,
            )
            result += _ERRORED
    return result
