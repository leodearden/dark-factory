"""The per-project frame the deferral CLIs share: the projects to run, and one report block each.

A CLI supplies its own result type and that result's body lines. A project with no task
store, or whose store did not answer, is an UnreadableProject, rendered the same way by
every CLI. Contract: plans/deferral-flip-condition-prd.md §9 ε.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol, TextIO, TypeVar

import deferral_store
import orchestrator_projects
from orchestrator_projects import ProjectTarget
from shared.project_layout import tasks_db

__all__ = [
    'UnreadableProject',
    'ids',
    'render',
    'targets_with_problems_reported',
    'unknown_project',
]


@dataclass(frozen=True)
class UnreadableProject:
    """A project that was not read: *unknown* when it holds no task store, else its store did not answer."""

    target: ProjectTarget
    reason: str
    unknown: bool = False

    @property
    def line(self) -> str:
        return f'{"UNKNOWN PROJECT" if self.unknown else "UNREADABLE"}: {self.reason}'


def unknown_project(target: ProjectTarget) -> UnreadableProject | None:
    """*target* as an unknown project when it holds no task store, else None."""
    if deferral_store.known_project(target.root):
        return None
    return UnreadableProject(target, f'no task store at {tasks_db(target.root)}', unknown=True)


class _ProjectResult(Protocol):
    @property
    def target(self) -> ProjectTarget: ...


_Result = TypeVar('_Result', bound=_ProjectResult)


def ids(task_ids: Iterable[int]) -> str:
    return ', '.join(str(task_id) for task_id in sorted(task_ids))


def render(
    results: Iterable[_Result | UnreadableProject], body: Callable[[_Result], Iterable[str]], *, out: TextIO
) -> None:
    """One block per project: its root and sweep, then *body*'s lines, or why it was not read."""
    for result in results:
        print(f'{result.target.root} — {result.target.sweep_label}', file=out)
        lines = [result.line] if isinstance(result, UnreadableProject) else body(result)
        for line in lines:
            print(f'  {line}', file=out)


def targets_with_problems_reported(
    args: argparse.Namespace, *, tool: str, verb: str
) -> tuple[ProjectTarget, ...]:
    """The projects *args* name; each unit that could not be read, and an empty run set, go to stderr."""
    targets, scan = orchestrator_projects.targets_from_args(args)
    for problem in scan.problems:
        print(f'{tool}: {problem}', file=sys.stderr)
    if not targets:
        print(
            f'{tool}: no project to {verb}: no --project-root, and no unit under {args.units_dir} names one',
            file=sys.stderr,
        )
    return targets
