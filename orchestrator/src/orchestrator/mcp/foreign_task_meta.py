"""Finds the paths a plan's prose names inside another lane's or task's
``.task-meta`` directory, writes no lane of the planning task can make
(``orchestrator/src/orchestrator/agents/write_set.py::WriteSet.is_foreign_task_meta``),
so plan-tools can report them to the architect at authoring time.

It flags rather than refuses, because prose cannot tell a read from a write.
Rationale: ``docs/task-authoring.md`` §9 "Superseding another task's plan".
"""
from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from orchestrator.agents.write_set import WriteSet, write_set_of
from orchestrator.artifacts import TaskArtifacts

logger = logging.getLogger(__name__)

FOREIGN_TASK_META_KEY = 'foreign_task_meta'

FOREIGN_TASK_META_NOTE = (
    "These paths are inside another lane's or task's .task-meta directory. "
    'No lane of this task can write them: the OS sandbox grants only '
    'orchestrator/src/orchestrator/agents/write_set.py::WriteSet.writable_paths, '
    "whose only .task-meta entries are this dispatch's own lane and task dirs, "
    "and an escalation's granted_files does not widen it. Reading them is fine. "
    'A step that WRITES one cannot be implemented, so move the write out of the '
    'plan. To make another task plan afresh, a handler resolves an escalation '
    "for that task with resolve_issue(action='restart', recut=True), which "
    'moves its .task-meta aside; for any other such write, call '
    'escalate_blocker for a handler-side run. See docs/task-authoring.md §9 '
    '"Superseding another task\'s plan".'
)

_ABSOLUTE_PATH_TOKEN = re.compile(r'/[^\s`\'"()<>\[\]{},;|*]+')


def foreign_task_meta_paths(text: str, write_set: WriteSet) -> tuple[str, ...]:
    """The absolute paths *text* names inside another lane's or task's
    ``.task-meta`` dir, as written, deduped in first-seen order."""
    kept: list[str] = []
    for match in _ABSOLUTE_PATH_TOKEN.finditer(text):
        token = match.group().rstrip('.:')
        try:
            if write_set.is_foreign_task_meta(Path(token)):
                kept.append(token)
        except (OSError, RuntimeError, ValueError):
            continue
    return tuple(dict.fromkeys(kept))


def step_findings(artifacts: TaskArtifacts, entries: Iterable[object]) -> dict[str, list[str]]:
    """Step or prerequisite id -> the foreign paths its description names.

    Empty when no write set can be derived for *artifacts*: the check is
    advisory, so it fails open.
    """
    try:
        write_set = write_set_of(artifacts)
    except ValueError as exc:
        logger.info('foreign .task-meta check skipped for %s: %s', artifacts.worktree, exc)
        return {}
    findings: dict[str, list[str]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get('description'), str):
            continue
        paths = foreign_task_meta_paths(entry['description'], write_set)
        if paths:
            findings[str(entry.get('id'))] = list(paths)
    return findings


def annotate(response: dict[str, Any], findings: Mapping[str, list[str]]) -> dict[str, Any]:
    """Attach *findings* to a plan-tools *response*; the key is absent when
    there are none."""
    if findings:
        response[FOREIGN_TASK_META_KEY] = {'paths': dict(findings), 'note': FOREIGN_TASK_META_NOTE}
    return response
