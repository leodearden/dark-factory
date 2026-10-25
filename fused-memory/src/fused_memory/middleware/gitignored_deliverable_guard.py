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
import os
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fused_memory.middleware.lock_charter_guard import extract_files
from fused_memory.middleware.metadata_dict import raw_metadata_dict
from fused_memory.reconciliation.recon_self_model import EXECUTION_CLASSES

logger = logging.getLogger(__name__)

__all__ = [
    'GitignoredDeliverableFinding',
    'gitignored_deliverable_enforced',
    'gitignored_deliverable_finding',
    'gitignored_deliverable_reject',
    'gitignored_deliverable_warning',
    'log_gitignored_deliverable_flagged',
    'make_gitignore_probe',
]

_GIT_PROBE_TIMEOUT_SECS = 10.0

_ENFORCE_ENV_VAR = 'FUSED_GITIGNORED_DELIVERABLE_ENFORCE'
_TRUTHY_ENV_VALUES: frozenset[str] = frozenset({'1', 'true', 'yes', 'on'})

_ROUTING_HINT = (
    "This is likely task_kind='deterministic' work, which has no commit "
    'requirement: point metadata.before_done at a committed script for a '
    'scripted action, or set metadata.always_escalates=True for a human gate. '
    'If a code deliverable really is intended, declare at least one '
    'committable path in metadata.files.'
)

_EXEMPT_EXECUTION_CLASSES: frozenset[str] = frozenset(
    c for c in EXECUTION_CLASSES if c != 'code_tdd'
)

GitignoreProbe = Callable[[Sequence[str]], frozenset[str] | None]


@dataclass(frozen=True)
class GitignoredDeliverableFinding:
    """Every declared deliverable is gitignored; ``ignored_paths`` in declaration order."""

    ignored_paths: tuple[str, ...]


def gitignored_deliverable_finding(
    *,
    task_kind: str,
    metadata: str | dict[str, Any] | None,
    probe: GitignoreProbe,
) -> GitignoredDeliverableFinding | None:
    """Return a finding when every declared ``metadata.files`` path is gitignored.

    See the module docstring for the exemptions. *probe* is consulted last,
    and only with the non-blank declared paths, so an exempt submission
    never spawns git.
    """
    if task_kind != 'normal':
        return None
    parsed = raw_metadata_dict(metadata, source='gitignored_deliverable_guard')
    if parsed.get('execution_class') in _EXEMPT_EXECUTION_CLASSES:
        return None
    if parsed.get('cross_repo'):
        return None
    declared = [f.strip() for f in extract_files(parsed) if f.strip()]
    if not declared:
        return None
    ignored = probe(declared)
    if ignored is None:
        return None
    if not all(p in ignored for p in declared):
        return None
    return GitignoredDeliverableFinding(ignored_paths=tuple(declared))


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


def _undeliverable_detail(finding: GitignoredDeliverableFinding) -> str:
    paths = ', '.join(finding.ignored_paths)
    return (
        'Every declared metadata.files path is gitignored in the target '
        f"project ({paths}), so a task_kind='normal' task can never produce a "
        'commit for it. confirm_plan only requires a non-empty plan.files '
        'declaration, but the merge-time plan-files-touched gate '
        '(OutcomeKind.plan_files_not_touched) and done_provenance require the '
        'work to be backed by a real commit on main, which a gitignored path '
        'can never be.'
    )


def gitignored_deliverable_reject(finding: GitignoredDeliverableFinding) -> dict[str, Any]:
    """Hard-reject payload, used when :func:`gitignored_deliverable_enforced` is ``True``."""
    return {
        'error': _undeliverable_detail(finding),
        'error_type': 'ValidationError',
        'hint': _ROUTING_HINT,
    }


def gitignored_deliverable_warning(finding: GitignoredDeliverableFinding) -> dict[str, Any]:
    """Non-blocking advisory to merge into a successful submit result."""
    return {
        'gitignored_deliverable_warning': {
            'ignored_paths': list(finding.ignored_paths),
            'detail': _undeliverable_detail(finding),
            'hint': (
                f'{_ROUTING_HINT} Set {_ENFORCE_ENV_VAR}=1 to hard-reject such '
                'submissions instead of warning.'
            ),
        },
    }


def log_gitignored_deliverable_flagged(finding: GitignoredDeliverableFinding) -> None:
    """Log the census line whose rate is the signal for flipping the enforce switch.

    Call it only for an ACCEPTED filing, so the census counts tasks that
    actually landed rather than submissions the interceptor went on to reject.
    """
    logger.warning(
        'gitignored_deliverable_lint.flagged task_kind=normal paths=%s',
        ','.join(finding.ignored_paths),
    )


def gitignored_deliverable_enforced() -> bool:
    """``True`` only when the enforce env var holds a recognised truthy value."""
    return os.environ.get(_ENFORCE_ENV_VAR, '').strip().lower() in _TRUTHY_ENV_VALUES
