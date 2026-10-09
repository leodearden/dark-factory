"""Where a deterministic task's ``before_done`` script runs.

Contract: ``docs/task-authoring.md`` §5. Must agree with the submit-time check
``fused-memory/src/fused_memory/middleware/deterministic_task_guard.py::_validate_before_done``,
which validates ``project_root / script``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BeforeDonePaths:
    script: Path
    cwd: Path

    def __post_init__(self) -> None:
        for field in ('script', 'cwd'):
            value = getattr(self, field)
            if not value.is_absolute():
                raise ValueError(
                    f'BeforeDonePaths.{field} must be absolute; got {value!r}',
                )


def resolve_before_done_paths(
    before_done: Mapping[str, Any], project_root: Path,
) -> BeforeDonePaths:
    cwd = before_done.get('cwd')
    return BeforeDonePaths(
        script=project_root / before_done['script'],
        cwd=project_root / cwd if cwd else project_root,
    )
