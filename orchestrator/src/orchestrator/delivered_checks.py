"""orchestrator.delivered_checks — the delivered-check runner (task delta,
capability-delivered-checks PRD: plans/capability-delivered-checks-prd.md).

Evaluates a single ``metadata.delivered_checks`` entry (PRD §Contract;
schema defined by ``shared.capability_manifest.DeliveredCheckMeta``, task
alpha) and returns a :class:`DeliveredCheckResult`. Two kinds:

- ``grep`` — evaluated against the COMMITTED tree at *ref* (default
  ``'main'``) via ``git -C <project_root> grep -E -e <pattern> <ref>``.
  This is the PRIMARY kind: it reads exactly what's on ``main``, immune to
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
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError
from shared.capability_manifest import DeliveredCheckMeta

from orchestrator import git_ops

logger = logging.getLogger(__name__)

__all__ = [
    'DeliveredCheckResult',
    'DeliveredChecksVerdict',
    'run_delivered_check',
    'verify_delivered_checks_on_main',
]

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


@dataclass(frozen=True)
class DeliveredChecksVerdict:
    """Aggregate outcome of running a whole ``metadata.delivered_checks``
    list against a single resolved ``main`` SHA (task 2794).

    Returned by :func:`verify_delivered_checks_on_main`. ``outcome`` collapses
    the per-check :class:`DeliveredCheckResult`\\s with the precedence
    ``all_delivered`` > ``failed`` > ``errored`` (see that function's
    docstring). ``main_sha`` echoes the SHA the checks were evaluated at (so a
    caller's WARNING can name it verbatim); ``failed_check`` carries the FIRST
    FAILED descriptor when ``outcome == 'failed'`` (``None`` otherwise), so the
    caller can name which capability was provably absent.
    """

    outcome: Literal['all_delivered', 'failed', 'errored']
    main_sha: str | None = None
    failed_check: dict[str, Any] | None = None


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
    """``git -C <project_root> grep -E -e <pattern> <ref> [-- <paths...>]``.

    rc==0 (match) / rc==1 (no match) are both valid outcomes; rc>=2 is a
    git error (ERRORED). Which of match/no-match is DELIVERED depends on
    ``meta.expect``: ``'present'`` wants a match, ``'absent'`` wants no
    match.

    The explicit ``-e`` separator (reviewer_comprehensive amendment) keeps
    a pattern beginning with ``'-'`` from being parsed as a ``git grep``
    option instead of the search pattern — without it, such a pattern
    would fail with a git error (rc>=2, ERRORED) rather than being used
    literally.
    """
    argv = ['git', '-C', str(project_root), 'grep', '-E', '-e', meta.pattern, ref]
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

    On timeout, ``asyncio.wait_for`` cancels the ``runner`` coroutine;
    ``orchestrator.git_ops._run`` kills and reaps its spawned subprocess in
    that case (task 2608), so a script that hangs past ``timeout_secs`` no
    longer leaks an orphaned child process.
    """
    assert meta.script is not None  # enforced by the script cross-field validator
    assert meta.timeout_secs is not None  # enforced by the script cross-field validator
    script_path = str(Path(project_root) / meta.script)
    argv = [script_path, *meta.args]
    rc, _out, _err = await asyncio.wait_for(
        runner(argv, cwd=Path(project_root)), timeout=meta.timeout_secs
    )
    return DeliveredCheckResult.DELIVERED if rc == 0 else DeliveredCheckResult.FAILED


async def verify_delivered_checks_on_main(
    checks: list[dict[str, Any]],
    *,
    project_root: str | Path,
    main_sha: str,
    check_timeout_secs: float,
    runner: _Runner = git_ops._run,
) -> DeliveredChecksVerdict:
    """Run a whole ``metadata.delivered_checks`` list against ``main_sha`` and
    collapse the per-check results into one :class:`DeliveredChecksVerdict`.
    Never raises.

    This is the SHARED delivered-capability ground-truth guard: given a task's
    declared capability checks and the SHA of ``main`` those checks should be
    evaluated at, it answers "is the deliverable actually present on ``main``?"
    — the layer above git-level attribution and effect-present guards, which
    prove only that *a* merge advanced ``main``, not that *this* task's
    declared capability survived to it.

    Each check runs via :func:`run_delivered_check` (reused verbatim — NO
    second check runner is forked) with ``ref=main_sha`` and the injected
    *runner*, each bounded by ``asyncio.wait_for(timeout=check_timeout_secs)``.
    The checks are evaluated CONCURRENTLY (``asyncio.gather``): they are
    independent and this reconcile call site carries no per-tick fan-out budget
    (unlike ``Scheduler._compute_delivered_check_cache``, which stays sequential
    under ``max_checks_per_tick``), so wall-clock is the slowest single check
    rather than the sum. ``gather`` preserves input order, so the FAILED
    precedence collapse below is order-independent of scheduling.
    A hung check (the timeout-less grep kind; defense-in-depth for scripts,
    which also carry their own ``timeout_secs``) raises ``TimeoutError``, which
    is mapped to :attr:`DeliveredCheckResult.ERRORED` — the same fail-safe
    downstream handling as a runner error. ``run_delivered_check`` itself never
    raises (a malformed descriptor degrades to ERRORED), so this aggregation
    cannot raise either.

    Results aggregate with the SAME precedence as
    ``Scheduler._compute_delivered_check_cache`` (scheduler.py 3150-3211):

    - every check DELIVERED (and at least one check ran) -> ``all_delivered``;
    - else any check FAILED -> ``failed`` (carrying the FIRST FAILED
      descriptor). A definitive absence drives the clean recovery
      (re-dispatch) and must NOT be masked into a fail-safe no-op by an
      unrelated ERRORED check;
    - else (some ERRORED, none FAILED, or an empty ``checks`` list) ->
      ``errored`` (fail-safe wait — the checks could not be evaluated, so make
      no claim either way).

    *main_sha* is a required caller-resolved parameter (not fetched here) so
    the aggregation stays pure and unit-testable with a fake runner and no git,
    and so the verdict can echo the exact SHA the checks ran against.

    The current production caller is the reconcile-sweep
    ``found_on_main``/MARK_DONE_WITH_PROVENANCE arm in
    ``Harness._reconcile_one_stranded`` (harness.py, task 2794). The other
    mark-done-on-main stamp sites SHOULD adopt this same guard so a hollow-done
    can never be stamped from any of them: the harness pre-dispatch
    ``found_on_main`` check (~harness.py:8033), ``TaskWorkflow._recover_before_execute``
    in workflow.py (pre-EXECUTE recovery — a different module/class a Harness
    method could not serve, which is why this is a module-level function), and
    the harness coalesce ``redrive_member`` path (~harness.py:779).
    """
    async def _run_one(
        check: dict[str, Any],
    ) -> tuple[dict[str, Any], DeliveredCheckResult]:
        try:
            result = await asyncio.wait_for(
                run_delivered_check(
                    check, project_root=project_root, ref=main_sha, runner=runner
                ),
                timeout=check_timeout_secs,
            )
        except TimeoutError:
            # Fail-safe (mirror scheduler.py 3116-3129): a hung check maps to
            # ERRORED — same downstream handling as a runner error.
            logger.warning(
                'verify_delivered_checks_on_main: check %r exceeded '
                'check_timeout_secs=%s at main@%s — treating as ERRORED '
                '(fail-safe)',
                check.get('name') if isinstance(check, dict) else check,
                check_timeout_secs,
                main_sha,
            )
            result = DeliveredCheckResult.ERRORED
        return (check, result)

    # Evaluate the checks CONCURRENTLY: they are independent and this reconcile
    # call site carries no per-tick fan-out budget (unlike
    # Scheduler._compute_delivered_check_cache, kept sequential under
    # max_checks_per_tick), so wall-clock is the slowest single check rather
    # than the sum of all of them. asyncio.gather PRESERVES input order, so the
    # first-FAILED precedence collapse below is byte-identical to a sequential
    # run; each thunk still bounds its own check with asyncio.wait_for and maps
    # TimeoutError -> ERRORED, and run_delivered_check never raises — so gather
    # cannot raise either (an empty checks list -> gather() -> []).
    results: list[tuple[dict[str, Any], DeliveredCheckResult]] = list(
        await asyncio.gather(*(_run_one(check) for check in checks))
    )

    if results and all(r is DeliveredCheckResult.DELIVERED for _c, r in results):
        return DeliveredChecksVerdict(outcome='all_delivered', main_sha=main_sha)

    failed = next(
        (c for c, r in results if r is DeliveredCheckResult.FAILED), None
    )
    if failed is not None:
        return DeliveredChecksVerdict(
            outcome='failed', main_sha=main_sha, failed_check=failed
        )

    # Some ERRORED and none FAILED (or no checks ran at all) — fail-safe wait.
    return DeliveredChecksVerdict(outcome='errored', main_sha=main_sha)
