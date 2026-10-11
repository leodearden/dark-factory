"""Where a project's stores live under its root, and which roots are known.

Contract: plans/review-all-s14-shared-leaves-prd.md (decisions 1, 2 and 5).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

__all__ = [
    'ESCALATION_QUEUE_RELPATH',
    'KNOWN_PROJECT_ROOTS_ENV',
    'StrPath',
    'escalation_queue_dir',
    'known_project_roots',
    'orchestrator_data_dir',
    'park_eviction_requests_db',
    'runs_db',
    'scheduler_overrides_db',
    'scheduler_state_json',
    'tasks_db',
]

ESCALATION_QUEUE_RELPATH = 'data/escalations'
KNOWN_PROJECT_ROOTS_ENV = 'DASHBOARD_KNOWN_PROJECT_ROOTS'

StrPath = str | os.PathLike[str]


def escalation_queue_dir(root: StrPath) -> Path:
    return Path(root) / ESCALATION_QUEUE_RELPATH


def orchestrator_data_dir(root: StrPath) -> Path:
    return Path(root) / 'data' / 'orchestrator'


def runs_db(root: StrPath) -> Path:
    return orchestrator_data_dir(root) / 'runs.db'


def scheduler_state_json(root: StrPath) -> Path:
    return orchestrator_data_dir(root) / 'scheduler_state.json'


def scheduler_overrides_db(root: StrPath) -> Path:
    return orchestrator_data_dir(root) / 'scheduler_overrides.db'


def park_eviction_requests_db(root: StrPath) -> Path:
    return orchestrator_data_dir(root) / 'park_eviction_requests.db'


def tasks_db(root: StrPath) -> Path:
    return Path(root) / '.taskmaster' / 'tasks' / 'tasks.db'


def known_project_roots(environ: Mapping[str, str] = os.environ) -> tuple[Path, ...]:
    entries = environ.get(KNOWN_PROJECT_ROOTS_ENV, '').split(',')
    return tuple(Path(entry.strip()) for entry in entries if entry.strip())
