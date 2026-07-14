"""orchestrator.delivered_checks — the delivered-check runner (task delta,
capability-delivered-checks PRD: plans/capability-delivered-checks-prd.md).

Evaluates a single ``metadata.delivered_checks`` entry (PRD §Contract;
schema defined by ``shared.capability_manifest.DeliveredCheckMeta``, task
alpha) and returns a :class:`DeliveredCheckResult`. Two kinds:

- ``grep`` — evaluated against the COMMITTED tree at *ref* (default
  ``'main'``) via ``git -C <project_root> grep -E <pattern> <ref>``. This
  is the PRIMARY kind: it reads exactly what's on ``main``, immune to
  working-checkout dirtiness.
- ``script`` — evaluated against the WORKING CHECKOUT (PRD Open-Q 2
  DECIDED: a documented approximation, not a temp-tree materialization of
  *ref*) via ``<project_root>/<script> <args>``, bounded by
  ``timeout_secs``. The escape hatch for capabilities that can't be
  expressed as a grep pattern.

``Scheduler._compute_delivered_check_cache`` (scheduler.py) is the sole
caller in production; both the git subprocess runner and the resolved
``ref`` are caller-supplied so tests never need a real git repo.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from shared.capability_manifest import DeliveredCheckMeta

from orchestrator import git_ops

logger = logging.getLogger(__name__)

__all__ = ['DeliveredCheckResult', 'run_delivered_check']

# (returncode, stdout, stderr) — matches orchestrator.git_ops._run's shape.
_Runner = Callable[..., Awaitable[tuple[int, str, str]]]


class DeliveredCheckResult(Enum):
    """Outcome of evaluating a single delivered-check descriptor."""

    #: The capability is verifiably present (grep matched / expect=absent's
    #: no-match / script exited 0).
    DELIVERED = 'delivered'
    #: The check ran cleanly and definitively did NOT find the capability.
    FAILED = 'failed'
    #: The check could not be evaluated at all (git error, malformed
    #: descriptor, script timeout/spawn failure) — fail-safe: never treated
    #: as a definitive DELIVERED or FAILED result by callers.
    ERRORED = 'errored'


async def run_delivered_check(
    check: dict[str, Any],
    *,
    project_root: str | Path,
    ref: str = 'main',
    runner: _Runner = git_ops._run,
) -> DeliveredCheckResult:
    """Evaluate a single delivered-check descriptor. Never raises.

    *check* is a raw ``metadata.delivered_checks`` entry dict (as stored on
    a task record) — defensively re-validated here via
    :class:`shared.capability_manifest.DeliveredCheckMeta` so a malformed
    entry degrades to :attr:`DeliveredCheckResult.ERRORED` rather than
    raising. *ref* is only consulted by the ``grep`` kind (the ``script``
    kind always runs against the working checkout — see the module
    docstring). *runner* is the injected subprocess seam
    (``(argv, **kwargs) -> (returncode, stdout, stderr)``), defaulting to
    :func:`orchestrator.git_ops._run`.
    """
    try:
        meta = DeliveredCheckMeta(**check)
    except (ValidationError, TypeError):
        logger.warning(
            'run_delivered_check: malformed check descriptor %r', check, exc_info=True
        )
        return DeliveredCheckResult.ERRORED

    try:
        if meta.kind == 'grep':
            return await _run_grep_check(meta, project_root=project_root, ref=ref, runner=runner)
        return await _run_script_check(meta, project_root=project_root, runner=runner)
    except Exception:
        logger.warning(
            'run_delivered_check: runner raised evaluating check %r', check, exc_info=True
        )
        return DeliveredCheckResult.ERRORED


async def _run_grep_check(
    meta: DeliveredCheckMeta,
    *,
    project_root: str | Path,
    ref: str,
    runner: _Runner,
) -> DeliveredCheckResult:
    """``git -C <project_root> grep -E <pattern> <ref> [-- <paths...>]``.

    rc==0 (match) / rc==1 (no match) are both valid outcomes; rc>=2 is a
    git error (ERRORED). Which of match/no-match is DELIVERED depends on
    ``meta.expect``: ``'present'`` wants a match, ``'absent'`` wants no
    match.
    """
    argv = ['git', '-C', str(project_root), 'grep', '-E', meta.pattern, ref]
    if meta.paths:
        argv.append('--')
        argv.extend(meta.paths)
    rc, _out, _err = await runner(argv)
    if rc >= 2:
        return DeliveredCheckResult.ERRORED
    matched = rc == 0
    delivered = matched if meta.expect == 'present' else not matched
    return DeliveredCheckResult.DELIVERED if delivered else DeliveredCheckResult.FAILED


async def _run_script_check(
    meta: DeliveredCheckMeta,
    *,
    project_root: str | Path,
    runner: _Runner,
) -> DeliveredCheckResult:
    """``<project_root>/<script> <args>``, cwd=project_root, bounded by
    ``meta.timeout_secs`` via ``asyncio.wait_for``.

    PRD Open-Q 2 DECIDED: this is a documented WORKING-CHECKOUT
    approximation — unlike the grep kind (which reads the exact committed
    tree at *ref*), the script runs against whatever is currently checked
    out at *project_root*. Acceptable because grep is the PRIMARY kind;
    script is the escape hatch for non-greppable capabilities. Any
    exception the runner raises (``asyncio.TimeoutError``/``TimeoutError``
    from the outer guard, ``OSError`` for a missing/non-executable script)
    propagates to :func:`run_delivered_check`'s catch-all, which maps it to
    :attr:`DeliveredCheckResult.ERRORED`.
    """
    script_path = str(Path(project_root) / meta.script)
    argv = [script_path, *meta.args]
    assert meta.timeout_secs is not None  # enforced by the script cross-field validator
    rc, _out, _err = await asyncio.wait_for(
        runner(argv, cwd=Path(project_root)), timeout=meta.timeout_secs
    )
    return DeliveredCheckResult.DELIVERED if rc == 0 else DeliveredCheckResult.FAILED
