"""Where a deterministic task's ``before_done`` script runs.

Contract: ``docs/task-authoring.md`` §5. Lives in ``shared`` so the run-time
executor (``orchestrator/src/orchestrator/deterministic_runner.py``) and the
submit-time check
(``fused-memory/src/fused_memory/middleware/deterministic_task_guard.py::_validate_before_done``,
which validates ``script_base / script``) can read one definition. Both choose
``script_base`` through ``resolve_script_base``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shared.task_metadata import script_root_kind_conflict


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


class ScriptRootRefused(ValueError):
    """``before_done.script_root`` names no base this record may use: an unknown
    value, or ``'factory'`` on a non-predicate."""


def resolve_script_base(
    before_done: Mapping[str, Any],
    project_root: Path,
    factory_root_resolver: Callable[[], Path | None],
) -> Path | None:
    """The directory ``before_done['script']`` resolves under.

    ``project_root`` for ``script_root='project'`` (the default). For
    ``'factory'``, whatever ``factory_root_resolver()`` returns: the
    dark-factory root, or ``None`` when it finds none. The resolver runs only
    for ``'factory'``, and only once ``script_root_kind_conflict`` admits the
    pairing; otherwise this raises ``ScriptRootRefused``.
    """
    script_root = before_done.get('script_root', 'project')
    if script_root == 'project':
        return project_root
    if script_root != 'factory':
        raise ScriptRootRefused(
            f"before_done.script_root={script_root!r} is neither 'project' nor 'factory'.",
        )
    conflict = script_root_kind_conflict(script_root, before_done.get('kind') or 'deploy')
    if conflict is not None:
        raise ScriptRootRefused(f'before_done.{conflict}')
    return factory_root_resolver()


def resolve_before_done_paths(
    before_done: Mapping[str, Any], project_root: Path, *, script_base: Path | None = None,
) -> BeforeDonePaths:
    """``script`` joins ``script_base`` (``project_root`` when omitted) even when
    ``cwd`` is set: the file the submit guard validated is the file that runs.
    ``cwd`` always joins ``project_root``."""
    base = script_base if script_base is not None else project_root
    cwd = before_done.get('cwd')
    return BeforeDonePaths(
        script=base / before_done['script'],
        cwd=project_root / cwd if cwd else project_root,
    )
