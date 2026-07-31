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
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError
from shared.capability_manifest import DeliveredCheckMeta

from orchestrator import git_ops

#: Module alias for :mod:`orchestrator.git_ops`.
#: ``gate_mark_done_on_delivered_checks`` takes a parameter named ``git_ops``
#: (the attribute name every adopting seam already uses for its own handle),
#: which shadows the module inside that function's body. The alias keeps the
#: module reachable for the default-runner binding without renaming it in the
#: two pre-existing functions below.
git_ops_module = git_ops

logger = logging.getLogger(__name__)

__all__ = [
    'DeliveredCheckResult',
    'DeliveredChecksBlock',
    'DeliveredChecksVerdict',
    'gate_mark_done_on_delivered_checks',
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

    Production callers reach this function through exactly ONE intermediary —
    :func:`gate_mark_done_on_delivered_checks` — so the mark-done DECISION
    (main-SHA resolution, verdict collapse, fail-safe arms, operator WARNING)
    has a single implementation rather than one copy per stamp seam. Task 3057
    routed every attribution-shaped ``mark_done`` seam through it:

    - ``Harness._reconcile_one_stranded`` — the reconcile-sweep
      ``found_on_main``/MARK_DONE_WITH_PROVENANCE arm (originally task 2794's
      inline block, since folded onto the shared helper);
    - ``Harness._already_landed_dispatch_gate`` — all three evidence arms
      (ancestry / merge-marker / content-equivalent);
    - ``Harness.build_train_callback_factory``'s ``mark_member_done`` and
      ``redrive_member`` closures (train-member flip; coalesce-derail redrive);
    - ``TaskWorkflow._finalise_recovery_done`` — the single funnel for all six
      recovery stamps (3x journal ``kind='merged'`` + 3x fallback
      ``kind='found_on_main'``);
    - ``TaskWorkflow._handle_already_done_report`` — the architect's
      "already done" claim;
    - ``TaskWorkflow._on_architect_merge_done`` — the architect-desync
      merge-landed flip;
    - ``TaskWorkflow``'s inline ``_mark_member_done`` closure and the
      solo-passer arm of ``TaskWorkflow._attribute_train_failure``;
    - ``merge_queue.reconcile_landed_row`` — the RC-2 crash-window journal
      recovery.

    Those seams span three modules and both free functions and methods, which
    is why the guard is a module-level function rather than a Harness method.
    Deliberately NOT routed: ``TaskWorkflow``'s own post-merge success stamp,
    where the workflow performed the merge itself and holds its own merge SHA —
    guarding it would convert an attribution guard into a fleet-wide
    merge-blocking quality gate (task 3057 design decision 2; a follow-up task
    holds that decision open for a human).

    Sites are named by SYMBOL only, never by line number: the numeric anchors
    this paragraph used to carry had all gone stale within two months.
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


@dataclass(frozen=True)
class DeliveredChecksBlock:
    """"Do NOT stamp this task done here" — the result of
    :func:`gate_mark_done_on_delivered_checks` when a mark-done must be
    withheld (task 3057).

    A ``None`` return from that helper means "mark-done may proceed"; a
    ``DeliveredChecksBlock`` means the caller MUST NOT stamp, and should take
    its OWN existing "no landing evidence" path instead.

    ``reason`` distinguishes the one case that is a DEFINITIVE absence from the
    two that are merely unknown:

    - ``'failed'`` — the checks ran cleanly and the declared capability is
      provably NOT on main. The only reason that may drive an ACTIVE recovery
      (e.g. the stranded arm's revert-to-pending for re-dispatch), because it
      is the only one that actually knows something.
    - ``'errored'`` — the checks could not be evaluated (git error, malformed
      descriptor, timeout). Fail-safe: makes no claim either way.
    - ``'main_sha_unresolved'`` — there was no ``main`` ref to evaluate
      against at all. Also fail-safe.

    Callers deliberately collapse all three into the same recovery except where
    task 2794 already distinguished them: the alternative (a fail-safe "wait and
    retry" for ``errored``) can WEDGE a task permanently, since a malformed
    descriptor ERRORs forever — and a wedged task is a worse outcome than an
    extra dispatch of an already-landed one. The invariant being defended is
    "never stamp a hollow done"; degrading toward re-delivery always satisfies
    it, degrading toward waiting does not always terminate.

    ``main_sha`` echoes the SHA the checks were evaluated at (``None`` when it
    could not be resolved); ``failed_check`` carries the first FAILED descriptor
    when ``reason == 'failed'`` (``None`` otherwise).
    """

    reason: Literal['failed', 'errored', 'main_sha_unresolved']
    main_sha: str | None = None
    failed_check: dict[str, Any] | None = None


async def gate_mark_done_on_delivered_checks(
    task_id: str,
    metadata: Mapping[str, Any] | None,
    *,
    project_root: str | Path,
    check_timeout_secs: float,
    site: str,
    git_ops: Any = None,
    main_sha: str | None = None,
    enabled: bool = True,
    log: logging.Logger = logger,
    runner: _Runner = git_ops_module._run,
) -> DeliveredChecksBlock | None:
    """The ONE shared mark-done decision for every attribution-shaped stamp
    seam (task 3057). Never raises.

    Returns ``None`` when the caller MAY stamp done — either because the guard
    is inert (disabled, or the task declares no ``delivered_checks``) or because
    the declared capability is verifiably present on ``main``. Returns a
    :class:`DeliveredChecksBlock` otherwise, having ALREADY emitted the operator
    WARNING on *log*.

    Why one helper rather than one guard per seam: this defect class's history
    is a chain of point fixes (1087 -> 1091 -> 1180 -> 2372 -> 2500 -> 2648 ->
    2787 -> 2794), each re-opened by a seam the previous fix did not cover — and
    during task 3057's own queue wait a BRAND-NEW unguarded ``found_on_main``
    seam was added by task 3031. Eleven hand-copied blocks would be eleven
    independent drift surfaces; the next refactor only has to miss one.

    Contract notes for adopting seams:

    - *log* is a PARAMETER, not this module's logger, so each seam keeps its own
      module logger and its existing caplog-addressable name (task 2794's
      stranded-arm assertions on ``logger='orchestrator.harness'`` still hold
      verbatim after the fold).
    - *site* is a short label naming the seam; it appears in every WARNING so an
      operator can tell WHICH of the eleven withheld a stamp.
    - *main_sha*, when supplied, short-circuits SHA resolution entirely. That is
      a correctness requirement, not an optimization: the two seams that pass it
      (``merge_queue.reconcile_landed_row`` and
      ``TaskWorkflow._handle_already_done_report``) have ALREADY made an
      ancestry decision against that exact SHA, and re-resolving could audit a
      DIFFERENT (newer) ``main`` than the decision being gated.
    - the evaluation itself is delegated to
      :func:`verify_delivered_checks_on_main` VERBATIM — no second check runner
      is forked, so grep/script semantics, the per-check timeout and the
      malformed-descriptor degradation are shared rather than re-derived.
    - the call is still wrapped in ``try/except`` as defence in depth: that
      function documents "Never raises", but a guard must never be ABLE to crash
      a mark-done path, so a broken contract degrades to an ``errored`` block.
    """
    checks = (metadata or {}).get('delivered_checks')
    if not checks or not enabled:
        # Inert, with ZERO I/O: check-less tasks keep their exact pre-guard
        # behavior, and the `enabled` kill switch is single-sourced HERE so one
        # hot reload of delivered_checks.enabled disarms all eleven seams at
        # once. Every seam FORWARDS the flag rather than short-circuiting
        # locally, which is what makes that true.
        return None

    sha = main_sha
    if not sha:
        if git_ops is None:
            log.warning(
                'Delivered-checks guard [%s]: task %s carries delivered_checks '
                'but neither a git_ops handle nor a pre-resolved main SHA was '
                'supplied — cannot verify the capability, NOT marking done '
                '(fail-safe)',
                site, task_id,
            )
            return DeliveredChecksBlock('main_sha_unresolved')
        try:
            sha = await git_ops.get_main_sha()
        except Exception:
            log.warning(
                'Delivered-checks guard [%s]: task %s carries delivered_checks '
                'but the main SHA could not be resolved (get_main_sha raised) '
                '— NOT marking done (fail-safe), will retry',
                site, task_id, exc_info=True,
            )
            return DeliveredChecksBlock('main_sha_unresolved')
        if not sha:
            log.warning(
                'Delivered-checks guard [%s]: task %s carries delivered_checks '
                'but get_main_sha() returned empty — NOT marking done '
                '(fail-safe), will retry',
                site, task_id,
            )
            return DeliveredChecksBlock('main_sha_unresolved')

    try:
        verdict = await verify_delivered_checks_on_main(
            checks,
            project_root=project_root,
            main_sha=sha,
            check_timeout_secs=check_timeout_secs,
            runner=runner,
        )
    except Exception:
        log.warning(
            'Delivered-checks guard [%s]: task %s — verify_delivered_checks_on_main '
            'raised at main@%s despite its never-raises contract; NOT marking '
            'done (fail-safe)',
            site, task_id, sha, exc_info=True,
        )
        return DeliveredChecksBlock('errored', sha)

    if verdict.outcome == 'failed':
        failed_check = verdict.failed_check or {}
        is_grep = failed_check.get('kind') == 'grep'
        log.warning(
            'Delivered-checks guard [%s]: task %s delivered-check %r (%s=%r) is '
            'absent from main@%s — declared capability not present, NOT marking '
            'done',
            site, task_id, failed_check.get('name'),
            'pattern' if is_grep else 'script',
            failed_check.get('pattern') if is_grep else failed_check.get('script'),
            sha,
        )
        return DeliveredChecksBlock('failed', sha, verdict.failed_check)

    if verdict.outcome == 'errored':
        # Some check could not be evaluated and none FAILED — make no claim
        # either way.
        log.warning(
            'Delivered-checks guard [%s]: task %s delivered-checks could not be '
            'evaluated at main@%s (errored) — NOT marking done (fail-safe)',
            site, task_id, sha,
        )
        return DeliveredChecksBlock('errored', sha)

    # outcome == 'all_delivered' — the declared capability is present on main;
    # the caller's mark_done proceeds, unchanged.
    return None
