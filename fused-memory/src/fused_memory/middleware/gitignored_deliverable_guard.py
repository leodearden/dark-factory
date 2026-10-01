"""Gitignored-deliverable lint guard (task 3611).

Declaration-only lint at the fused-memory ``submit_task`` boundary: flags —
or, when explicitly enforced, hard-rejects — a ``task_kind='normal'``
submission from ANY caller whose declared ``metadata.files`` are ALL
gitignored in the target project.

## Why such a task is undeliverable

A normal task is delivered as a commit. ``confirm_plan`` only checks that
``plan.files`` is declared and non-empty, so the plan stage passes. But the
merge-time plan-files-touched gate (``OutcomeKind.plan_files_not_touched``)
and ``done_provenance`` both need the work backed by a real commit on main,
and a gitignored path can never be committed. The usual right routing is
``task_kind='deterministic'``.

## Exemptions (no finding)

- ``task_kind`` is not ``'normal'``.
- ``metadata.execution_class`` is a non-``'code_tdd'`` member of
  :data:`fused_memory.reconciliation.recon_self_model.EXECUTION_CLASSES`.
- ``metadata.cross_repo`` is truthy — a HAND-SET task-3004 marker. The
  interceptor stamps its own marker only after this boundary has run.
- No non-blank declared files (``files=[]`` is the defer-to-architect value).
- At least one declared file is committable.
- git cannot answer (non-git root, a path outside the repo, timeout): the
  probe returns ``None`` and the guard fails open.

## Placement

One call site in ``tools.py::submit_task``, just before
``task_interceptor.submit_task``. The curator and ``planning_mode`` paths
split inside the interceptor, so that one placement covers both.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    'make_gitignore_probe',
]

_GIT_PROBE_TIMEOUT_SECS = 10.0

GitignoreProbe = Callable[[Sequence[str]], frozenset[str] | None]


def make_gitignore_probe(project_root: str | Path) -> GitignoreProbe:
    """Build ``probe(paths)`` answering which *paths* are gitignored under *project_root*.

    The probe returns the ignored subset, each path echoed exactly as given,
    or ``None`` when git could not answer — never raises.
    """

    def probe(paths: Sequence[str]) -> frozenset[str] | None:
        if not paths:
            return frozenset()
        try:
            # No --no-index: the index must be consulted so a tracked path
            # that matches an ignore rule still counts as committable.
            result = subprocess.run(
                ['git', 'check-ignore', '-z', '--stdin'],
                cwd=project_root,
                input=''.join(p + '\0' for p in paths),
                capture_output=True,
                text=True,
                timeout=_GIT_PROBE_TIMEOUT_SECS,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning(
                'gitignored_deliverable_guard: git check-ignore failed under %s '
                '— probe fails open: %s',
                project_root, exc,
            )
            return None
        if result.returncode not in (0, 1):
            logger.warning(
                'gitignored_deliverable_guard: git check-ignore exited %d under %s '
                '— probe fails open: %s',
                result.returncode, project_root, result.stderr.strip(),
            )
            return None
        return frozenset(p for p in result.stdout.split('\0') if p)

    return probe
