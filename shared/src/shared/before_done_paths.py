"""Where a deterministic task's ``before_done`` script runs.

Contract: ``docs/task-authoring.md`` §5. Lives in ``shared`` so the run-time
executor (``orchestrator/src/orchestrator/deterministic_runner.py``) and the
submit-time check
(``fused-memory/src/fused_memory/middleware/deterministic_task_guard.py::_validate_before_done``)
read one definition: each makes one call to ``resolve_before_done_paths`` and
maps the ``BeforeDonePathsRefused`` subclasses it raises to its own
diagnostics.
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
    script_base: Path
    cwd: Path

    def __post_init__(self) -> None:
        for field in ('script', 'script_base', 'cwd'):
            value = getattr(self, field)
            if not value.is_absolute():
                raise ValueError(
                    f'BeforeDonePaths.{field} must be absolute; got {value!r}',
                )


class BeforeDonePathsRefused(ValueError):
    """before_done names no script location this process can resolve."""


class ScriptRootRefused(BeforeDonePathsRefused):
    """``before_done.script_root`` names no base this record may use: an unknown
    value, or ``'factory'`` on a non-predicate."""


class CwdRefused(BeforeDonePathsRefused):
    """``before_done.cwd`` is present but not a string path."""


class FactoryRootUnresolved(BeforeDonePathsRefused):
    """``before_done.script_root='factory'`` but the resolver found no
    dark-factory root. The message names no process; each caller adds its own
    remedy."""

    def __init__(self, script: str) -> None:
        self.script = script
        super().__init__(
            "before_done.script_root='factory' but the dark-factory root could not "
            f'be resolved, so script {script!r} cannot be located.',
        )


def _script_base(
    before_done: Mapping[str, Any],
    project_root: Path,
    factory_root_resolver: Callable[[], Path | None],
) -> Path:
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
    factory_root = factory_root_resolver()
    if factory_root is None:
        raise FactoryRootUnresolved(before_done['script'])
    return factory_root


def _before_done_cwd(before_done: Mapping[str, Any]) -> str | None:
    cwd = before_done.get('cwd')
    if cwd is not None and not isinstance(cwd, str):
        raise CwdRefused(
            f'before_done.cwd must be a string path when present (got {type(cwd).__name__!r}).',
        )
    return cwd


def resolve_before_done_paths(
    before_done: Mapping[str, Any],
    project_root: Path,
    factory_root_resolver: Callable[[], Path | None],
) -> BeforeDonePaths:
    """The script, its base and the cwd a ``before_done`` record names.

    The base is ``project_root`` for ``script_root='project'`` (the default)
    and ``factory_root_resolver()`` for ``'factory'``; the resolver runs only
    once ``script_root_kind_conflict`` admits that pairing. ``script`` joins
    the base even when ``cwd`` is set, so the file the submit guard validated
    is the file that runs; ``cwd`` always joins ``project_root``.

    Raises ``BeforeDonePathsRefused``, checking in this order:
    ``CwdRefused`` for a non-string cwd, ``ScriptRootRefused`` for a base the
    record may not use, ``FactoryRootUnresolved`` when the resolver finds no
    dark-factory root.
    """
    cwd = _before_done_cwd(before_done)
    base = _script_base(before_done, project_root, factory_root_resolver)
    return BeforeDonePaths(
        script=base / before_done['script'],
        script_base=base,
        cwd=project_root / cwd if cwd else project_root,
    )
