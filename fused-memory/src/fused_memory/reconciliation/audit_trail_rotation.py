"""Bound a long-running audit-trail task instead of letting it accrete forever.

``docs/task-authoring.md`` §10 is the normative rule; this module is the
harness side of it.  It plans a rotation of one task (a pure function of the
task and the clock) and executes the plan over two injected ports: an archive
that must read back byte-identical before anything leaves the task, and a
compare-and-swap commit.

Import-light by design (stdlib + ``shared``), like ``consolidation_gate.py``:
the interceptor imports this module, and the memory service is duck-typed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from shared.task_statuses import TERMINAL

__all__ = [
    'ROTATE_THRESHOLD_BYTES',
    'ROTATE_TARGET_BYTES',
    'HISTORY_KEEP',
    'HISTORY_MAX',
    'ARCHIVE_MAX_BYTES',
    'ROLLUP_KEY',
    'RotationPlan',
    'plan_rotation',
    'task_fingerprint',
    'task_payload_bytes',
]

# docs §10 "The threshold": rotate above 20,000 B, rotate down to <=10,000 B.
ROTATE_THRESHOLD_BYTES = 20_000
ROTATE_TARGET_BYTES = 10_000
# The cap of the autopilot_video 654 / gate 657 hand rotation.
HISTORY_KEEP = 5
# Trim from 2x down to 1x, so an archive memory is written once per HISTORY_KEEP cycles.
HISTORY_MAX = 2 * HISTORY_KEEP
# text-embedding-3-small takes 8,191 tokens; mem0 embeds the whole archive (infer=False),
# so budget a conservative 2 bytes/token.
ARCHIVE_MAX_BYTES = 16_000
ROLLUP_KEY = 'audit_trail_rotation'

_TEXT_COLUMNS = ('title', 'description', 'details')


def task_payload_bytes(task: Mapping[str, Any]) -> int:
    """Whole-task payload size as docs §10 measures it, in UTF-8 bytes."""
    text_bytes = sum(len((task.get(column) or '').encode()) for column in _TEXT_COLUMNS)
    metadata = task.get('metadata')
    metadata_bytes = 0 if metadata is None else len(json.dumps(metadata).encode())
    return text_bytes + metadata_bytes


def task_fingerprint(task: Mapping[str, Any]) -> str:
    """Compare-and-swap token: changes whenever any column a rotation reads changes."""
    digest = hashlib.sha256()
    for column in (*_TEXT_COLUMNS, 'status'):
        digest.update((task.get(column) or '').encode())
        digest.update(b'\0')
    digest.update(json.dumps(task.get('metadata'), sort_keys=True).encode())
    return digest.hexdigest()


@dataclass(frozen=True)
class RotationPlan:
    task_id: str
    rotated_at: str


def plan_rotation(task: Mapping[str, Any], *, now: datetime) -> RotationPlan | None:
    """Plan a rotation of ``task``, or None when there is nothing to bound."""
    if task.get('status') in TERMINAL:
        return None
    if not isinstance(task.get('metadata'), dict):
        return None
    return None
