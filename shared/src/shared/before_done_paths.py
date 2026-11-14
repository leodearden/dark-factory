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
from enum import StrEnum
from pathlib import Path
from typing import Any

from shared.task_metadata import script_root_kind_conflict


class ScriptRoot(StrEnum):
    """``before_done.script_root``: which directory the script joins."""

    PROJECT = 'project'
    FACTORY = 'factory'

    @property
    def base_name(self) -> str:
        """How a diagnostic names this root's directory."""
        match self:
            case ScriptRoot.PROJECT:
                return 'project_root'
            case ScriptRoot.FACTORY:
                return 'the dark-factory root'


@dataclass(frozen=True)
class BeforeDonePaths:
    script: Path
    script_root: ScriptRoot
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
    """before_done names a script or cwd this process cannot resolve."""


class ScriptRootRefused(BeforeDonePathsRefused):
    """``before_done.script_root`` names no base this record may use: an unknown
    value, or ``'factory'`` on a non-predicate."""


class ScriptRefused(BeforeDonePathsRefused):
    """``before_done.script`` is absent, not a string, or blank."""


class CwdRefused(BeforeDonePathsRefused):
    """``before_done.cwd`` is present but not a string path."""


class FactoryRootUnresolved(BeforeDonePathsRefused):
    """``before_done.script_root='factory'`` but the resolver found no
    dark-factory root. The message names no process; each caller adds its own
    remedy."""

    def __init__(self, script: str) -> None:
        super().__init__(
            "before_done.script_root='factory' but the dark-factory root could not "
            f'be resolved, so script {script!r} cannot be located.',
        )


def _script_root(before_done: Mapping[str, Any]) -> ScriptRoot:
    script_root = before_done.get('script_root', ScriptRoot.PROJECT)
    try:
        return ScriptRoot(script_root)
    except ValueError:
        raise ScriptRootRefused(
            f'before_done.script_root={script_root!r} is neither '
            f'{ScriptRoot.PROJECT.value!r} nor {ScriptRoot.FACTORY.value!r}.',
        ) from None


def _before_done_script(before_done: Mapping[str, Any]) -> str:
    script = before_done.get('script')
    if not isinstance(script, str) or not script.strip():
        raise ScriptRefused(
            'before_done.script is required and must be a non-empty string path '
            f'(got {type(script).__name__} {script!r}).',
        )
    return script


def _script_base(
    before_done: Mapping[str, Any],
    script: str,
    script_root: ScriptRoot,
    project_root: Path,
    factory_root_resolver: Callable[[], Path | None],
) -> Path:
    if script_root is ScriptRoot.PROJECT:
        return project_root
    conflict = script_root_kind_conflict(script_root, before_done.get('kind') or 'deploy')
    if conflict is not None:
        raise ScriptRootRefused(f'before_done.{conflict}')
    factory_root = factory_root_resolver()
    if factory_root is None:
        raise FactoryRootUnresolved(script)
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
    ``ScriptRefused`` for a missing, non-string or blank script,
    ``CwdRefused`` for a non-string cwd, ``ScriptRootRefused`` for a base the
    record may not use, ``FactoryRootUnresolved`` when the resolver finds no
    dark-factory root.
    """
    script = _before_done_script(before_done)
    cwd = _before_done_cwd(before_done)
    script_root = _script_root(before_done)
    base = _script_base(before_done, script, script_root, project_root, factory_root_resolver)
    return BeforeDonePaths(
        script=base / script,
        script_root=script_root,
        script_base=base,
        cwd=project_root / cwd if cwd else project_root,
    )
