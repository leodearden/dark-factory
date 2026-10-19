"""One recorder for every flake call site (PRD task ε; offline-lane PRD D6).

`plans/flake-ledger-prd.md` §5.8 — the topology rule this module exists
to encode:

  * the DISCRIMINATOR (``verify.confirm_isolated_rerun_verdict``) runs wherever the
    WORKTREE is — local host or remote runner — because that is the only place the
    failing tests can actually be re-run;
  * the RECORDER runs on the DISPATCHER, because that is the only place an
    ``EventStore``, an escalation queue, the project root, the merge SHA and the
    task id all exist at once.

Before ε those two roles were fused inside ``verify.apply_merge_flake_suppression``,
which meant every side-effect fired on the host that happened to own the worktree.
On the remote path that host has no event store, no escalation queue, and its own
process-local streak counter — so the ``merge_flake_suppressed`` fact was dropped,
no ledger row was ever written (there was no ledger call at all), and INV-4's storm
detector silently reset on process exit, disarming it exactly where load — and
therefore the flake rate this PRD exists to measure — is highest.

Splitting them makes the side-effects unconditional BY CONSTRUCTION rather than
dependent on which host ran the verify: the producer merely ATTACHES a
``FlakeSuppression`` to the ``VerifyResult``, the observation rides the wire home,
and this module records it.

Since ε0 (task 6934) a suppression's debt is not filed here.  Each suppressed test gets
an item on the ledger's due list in the same transaction as its occurrence row
(``flake_ledger.enqueue_debt``), and ``orchestrator/src/orchestrator/flake_debt_drain.py``
files the owner afterwards (PRD §5.9) and raises any ``regressed_after_resolution``
escalation, so no network call is left on the observation path.

Since ε1 (task 6864) :func:`record_call_site_observations` is the one body every call
site records through, and :data:`_CALL_SITES_WITH_LIVE_SIGNALS` names the call sites
whose observations do more than write the rows and the due items.  The merge-gate entry
unwraps its ``VerifyResult`` and delegates.

Import discipline: this module imports ``flake_ledger`` (which depends on ``shared``
alone) at runtime and NOTHING else from ``orchestrator``.
``VerifyResult`` is a ``TYPE_CHECKING``-only annotation and ``EventType`` /
``Escalation`` are imported lazily inside their functions, so ``flake_recorder``
never imports ``verify`` or ``event_store`` at runtime and cannot participate in an
import cycle with either.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from escalation.categories import EscalationCategory

from orchestrator.flake_ledger import (
    DebtContext,
    DueItem,
    FlakeCallSite,
    FlakeSuppression,
    FlakeVerdict,
    context_at_call_site,
    debt_context_to_json,
    enqueue_debt,
    ledger_db_path,
    named_tests,
)

if TYPE_CHECKING:
    from orchestrator.event_store import EventStore
    from orchestrator.verify import VerifyResult

logger = logging.getLogger(__name__)


#: The call sites whose suppressions emit ``merge_flake_suppressed`` and bump the INV-4
#: storm streak: the merge gate's alone.  The offline lane gates no merge, so a lane clear
#: masks nothing (offline-lane PRD D6, call 1).
_CALL_SITES_WITH_LIVE_SIGNALS: Final = frozenset({FlakeCallSite.merge_gate})


def _emit_merge_flake_suppressed(
    event_store: EventStore | None,
    task_id: str | None,
    merge_sha: str,
    node_ids: list[str],
) -> None:
    """Emit the INV-2 structured suppression fact. None-safe (skips on None).

    ``EventType`` is imported lazily so ``flake_recorder`` has no runtime import
    of ``event_store`` at all — see the module docstring's import-discipline note.
    The lazy import is what keeps this module's runtime dependency set to
    ``flake_ledger`` alone, by construction rather than by convention.
    """
    if event_store is None:
        return
    from orchestrator.event_store import EventType  # noqa: PLC0415 — lazy, avoid cycle

    event_store.emit(
        EventType.merge_flake_suppressed,
        task_id=task_id,
        data={
            'node_ids': node_ids,
            'merge_sha': merge_sha,
            'measured_at': datetime.now(UTC).isoformat(),
        },
    )


#: Module-global suppression counter (INV-4 storm detector). Bumped ONLY on a
#: suppression; reset to 0 only once the window (threshold) is reached and the
#: storm escalation decision is made. A clean, non-suppressed merge-verify does
#: NOT reset it, so this is a CUMULATIVE count of suppressions since the last
#: reset — NOT a count of back-to-back (consecutive) merges. A count-window
#: detector; time-windowing is a sanctioned PRD §9 follow-up.
_merge_flake_suppression_streak = 0

#: Suppressions per window before the born-at-L2 storm escalation fires. A
#: tunable (PRD §9): chronic suppression means α is repeatedly masking reds —
#: a fleet-health "someone must look now" condition.
_MERGE_FLAKE_SUPPRESSION_STREAK_THRESHOLD = 5

#: Fixed dedup sentinel task_id for the storm escalation — the signal is a
#: global fleet-health condition, not tied to any one merge task.
_MERGE_FLAKE_SUPPRESSION_STORM_SENTINEL = 'merge-flake-suppression-storm'

def _bump_suppression_streak_and_maybe_escalate(
    escalation_queue: Any, task_id: str | None, merge_sha: str,
) -> None:
    """Advance the suppression streak; file a born-at-L2 storm escalation at
    the threshold, then reset the counter (INV-4).

    Modeled on ``merge_queue._alarm_verify_worktree_contention``: a born-at-L2
    escalation (``severity='critical'``, ``level=2``,
    ``agent_role='orchestrator-merge-flake-monitor'`` — the ``orchestrator-``
    prefix marks it a harness sentinel so the escalation server never downgrades
    the critical severity) that routes straight to a human, bypassing the
    auto-watcher. Deduped on a fixed open-L2 sentinel task_id so a persistent
    storm files at most one open critical per window.

    The window resets to 0 whenever the threshold is reached — on submit, on a
    dedup-skip, AND on a ``None`` queue — so the counter can never grow
    unbounded and each fresh window makes an independent escalation decision.
    None-safe: with no queue there is nothing to file into, so it resets and
    returns.  After task ε the remote worktree host no longer reaches this
    function at all — recording happens on the dispatcher, which HAS a queue —
    so a ``None`` queue now means only a CLI or test caller, not the
    CPU-starvation target this gate addresses (the α scope fence).

    ONE MERGE CAN CONTRIBUTE MORE THAN ONE UNIT TO THE WINDOW, and that is
    deliberate: the unit is a SUPPRESSION (one red masked), never a merge.  Two
    routes reach two units for a single merge SHA, both of which really did mask
    two separate reds:

    * a REMOTE green that suppressed, then cross-checked by the local trust
      anchor (``merge_queue._run_post_merge_verify``'s task-2822 branch) whose
      own gate also suppressed — two independent gate runs on two different
      hosts.  Recording both is what makes ``FlakeSuppression.runner`` able to
      answer θ's class-3 question (same tests suppressed on BOTH hosts ⇒ the
      SUITE, not the host);
    * a suppressed scoped red whose attempt was then superseded by an
      infra-transient retry — the superseded observation is still an
      observation (§5.5).

    Before ε the remote leg emitted nothing at all on the first route, so that
    path effectively bumped once per merge; the count is now higher there
    because it is now COMPLETE, not because it is double-counting.  Anything
    reading this counter as "merges since reset" is misreading it — read the
    ``flake_occurrence`` rows, which carry ``merge_sha`` and can be grouped.
    """
    global _merge_flake_suppression_streak
    _merge_flake_suppression_streak += 1
    if _merge_flake_suppression_streak < _MERGE_FLAKE_SUPPRESSION_STREAK_THRESHOLD:
        return

    # Window reached: make the escalation decision once, then reset regardless.
    _merge_flake_suppression_streak = 0
    if escalation_queue is None:
        return

    from escalation.models import Escalation  # noqa: PLC0415 — local, escalation optional dep

    sentinel = _MERGE_FLAKE_SUPPRESSION_STORM_SENTINEL
    # Dedup: don't re-alarm while an open L2 already exists for the storm
    # sentinel (has_open_l1 is hardcoded to level=1, so get_by_task is used).
    if escalation_queue.get_by_task(sentinel, status='pending', level=2):
        return

    summary = (
        'Merge-verify flake-suppression storm: the isolated-rerun-confirm gate '
        f'has suppressed {_MERGE_FLAKE_SUPPRESSION_STREAK_THRESHOLD} merge-verify '
        'reds since the last reset'
    )
    detail = (
        f'The role=merge isolated-rerun-confirm gate (observed by '
        f'verify.confirm_isolated_rerun_verdict, recorded by '
        f'flake_recorder.record_merge_flake_suppression) has suppressed '
        f'{_MERGE_FLAKE_SUPPRESSION_STREAK_THRESHOLD} merge-verify failures as '
        f'CPU-starvation flakes since the counter was last reset — a CUMULATIVE '
        f'count, NOT necessarily back-to-back merges (a clean merge-verify does '
        f'not reset the counter). Most recent merge SHA: {merge_sha}, task_id: '
        f'{task_id}. Each suppression means a merge-verify red passed on isolated '
        're-run — but a sustained rate of suppressions indicates either chronic '
        'host CPU starvation or a genuinely flaky test that is being repeatedly '
        'masked. Investigate before the gate hides a real regression.'
    )
    esc = Escalation(
        id=escalation_queue.make_id(sentinel),
        task_id=sentinel,
        agent_role='orchestrator-merge-flake-monitor',
        severity='critical',
        level=2,
        category=EscalationCategory.MERGE_FLAKE_SUPPRESSION_STORM,
        summary=summary,
        detail=detail,
        suggested_action=(
            'Inspect merge-flake-suppressed events (EventType.merge_flake_suppressed) '
            'and host CPU load. Confirm the suppressed tests are load flakes, not a '
            'masked regression; if a specific test is chronically flaky, de-flake or '
            'quarantine it.'
        ),
    )
    escalation_queue.submit(esc)


@dataclass(frozen=True)
class _Guard:
    """Runs one side-effect under its OWN catch-all, so a failure costs exactly that
    signal, and names the observation in the one line that reports the loss."""

    call_site: str
    head_sha: str
    observing_task_id: str | None

    def __call__(self, what: str, fn: Callable[[], object]) -> None:
        try:
            fn()
        except Exception:
            self.lost(what)

    def lost(self, what: str) -> None:
        """The one place a lost signal is reported; call it from an ``except`` block."""
        logger.warning(
            'flake_recorder: failed to record a flake observation [%s] (call_site=%s, '
            'head_sha=%s, task_id=%s); the merge/verify is unaffected',
            what,
            self.call_site,
            self.head_sha,
            self.observing_task_id,
            exc_info=True,
        )


@dataclass(frozen=True)
class _DueItems:
    """The due items one call owes: those riding each suppression's own write, in
    suppression order, and those of eligible tests no suppression names."""

    per_suppression: tuple[tuple[DueItem, ...], ...]
    unnamed: tuple[DueItem, ...]

    @classmethod
    def none(cls, suppressions: int) -> _DueItems:
        return cls(per_suppression=((),) * suppressions, unnamed=())


def _due_item(
    test_id: str,
    ctx: DebtContext | None,
    *,
    observed_at: str,
    call_site: str,
    head: str | None,
    observing_task_id: str | None,
) -> DueItem:
    return DueItem(
        test_id=test_id,
        observed_at=observed_at,
        call_site=call_site,
        head=head,
        observing_task_id=observing_task_id,
        context=_context_text(test_id, context_at_call_site(call_site, ctx)),
    )


def _context_text(test_id: str, ctx: DebtContext | None) -> str:
    """*ctx* as due-list text; ``'{}'`` (logged) for one that will not encode: the owner
    is still owed, only its wording is lost."""
    if ctx is None:
        return '{}'
    try:
        return debt_context_to_json(ctx)
    except Exception:
        logger.warning(
            'flake_recorder: could not encode the debt context for %s; enqueuing it '
            'without one',
            test_id,
            exc_info=True,
        )
        return '{}'


def _due_items(
    suppressions: Sequence[FlakeSuppression],
    debt_eligible: Mapping[str, DebtContext | None],
    *,
    call_site: FlakeCallSite,
    head: str | None,
    observed_at: str | None,
    observing_task_id: str | None,
) -> _DueItems:
    """One item per debt-eligible test.  A test a suppression names takes that
    suppression's instant and call site, so its key is its row's; an unnamed one takes
    the call's, and is skipped (logged) when the call has no instant."""
    named = [named_tests(s) for s in suppressions]
    per_suppression: list[list[DueItem]] = [[] for _ in suppressions]
    unnamed: list[DueItem] = []
    for test_id, ctx in debt_eligible.items():
        index = next((i for i, tests in enumerate(named) if test_id in tests), None)
        if index is not None:
            s = suppressions[index]
            per_suppression[index].append(
                _due_item(
                    test_id, ctx,
                    observed_at=s.observed_at, call_site=s.call_site,
                    head=head, observing_task_id=observing_task_id,
                ),
            )
        elif observed_at is None:
            logger.warning(
                'flake_recorder: %s is debt-eligible at call_site=%s but no suppression names '
                'it and the call has no observed_at; skipping it rather than stamping the '
                'wall clock',
                test_id,
                call_site,
            )
        else:
            unnamed.append(
                _due_item(
                    test_id, ctx,
                    observed_at=observed_at, call_site=call_site,
                    head=head, observing_task_id=observing_task_id,
                ),
            )
    return _DueItems(tuple(tuple(items) for items in per_suppression), tuple(unnamed))


def _write_ledger(
    db_path: Path,
    project_id: str,
    suppressions: Sequence[FlakeSuppression],
    due: _DueItems,
    *,
    head: str | None,
    observing_task_id: str | None,
    guard: _Guard,
) -> None:
    """Each suppression's rows with its own items, in one transaction and under one guard
    each; then the unnamed items."""
    for s, items in zip(suppressions, due.per_suppression, strict=True):
        guard(
            'ledger',
            lambda s=s, items=items: enqueue_debt(
                db_path, project_id, items,
                occurrence=s, merge_sha=head, task_id=observing_task_id,
            ),
        )
    if due.unnamed:
        guard('ledger', lambda: enqueue_debt(db_path, project_id, due.unnamed))


def _is_suppression(s: FlakeSuppression, guard: _Guard) -> bool:
    """Whether *s* is ``passes_in_isolation``, by ``==``; ``False`` (logged) for a verdict
    that cannot be read, since a bogus item or live signal is worse than a missing one."""
    try:
        return s.verdict == FlakeVerdict.passes_in_isolation
    except Exception:
        guard.lost('verdict')
        return False


def _fire_live_signals(
    suppressions: Sequence[FlakeSuppression],
    *,
    head_sha: str,
    observing_task_id: str | None,
    event_store: EventStore | None,
    escalation_queue: Any,
    guard: _Guard,
) -> None:
    """The ``merge_flake_suppressed`` fact and the storm-streak bump, per suppression."""
    for s in suppressions:
        if not _is_suppression(s, guard):
            continue
        guard(
            'event',
            lambda s=s: _emit_merge_flake_suppressed(
                event_store, observing_task_id, head_sha, list(s.test_ids),
            ),
        )
        # Its OWN guard, after the emit: the storm detector must survive a broken event
        # store rather than being taken down with it.
        guard(
            'streak',
            lambda: _bump_suppression_streak_and_maybe_escalate(
                escalation_queue, observing_task_id, head_sha,
            ),
        )


async def record_call_site_observations(
    suppressions: Sequence[FlakeSuppression],
    *,
    call_site: FlakeCallSite,
    project_root: Path,
    project_id: str,
    head_sha: str,
    observed_at: str | None = None,
    observing_task_id: str | None = None,
    debt_eligible: Mapping[str, DebtContext | None],
    event_store: EventStore | None = None,
    escalation_queue: Any = None,
) -> None:
    """Record what one call site observed at *head_sha* (offline-lane PRD §5.8).

    Every suppression's ``flake_occurrence`` rows are written, whatever its verdict
    (§5.5: the observation, not the remedy).  Every test in *debt_eligible* gets one item
    on the ledger's due list carrying its context, in the same transaction as the rows
    of the suppression that names it; which tests are eligible is the caller's call,
    never derived from a verdict here.  Then, for a call site in
    :data:`_CALL_SITES_WITH_LIVE_SIGNALS`, each ``passes_in_isolation`` suppression
    emits its fact and bumps the storm streak.

    ORDER IS THE CONTRACT: the durable rows and items first, the in-process signals
    second, so a crash mid-call loses the recoverable half.  ``==``, never ``is``, on the
    verdict, which may be a wire-rebuilt string.  Each effect has its own guard, so one
    failure costs that signal alone, and the function NEVER RAISES (B12): a lost
    measurement must not fail a verify or stall the merge queue.
    """
    guard = _Guard(call_site, head_sha, observing_task_id)
    head = head_sha or None
    due = _DueItems.none(len(suppressions))
    try:
        due = _due_items(
            suppressions, debt_eligible,
            call_site=call_site, head=head, observed_at=observed_at,
            observing_task_id=observing_task_id,
        )
    except Exception:
        logger.warning(
            'flake_recorder: could not derive the due items at call_site=%s; writing the '
            'rows without them',
            call_site,
            exc_info=True,
        )
    _write_ledger(
        ledger_db_path(project_root), project_id, suppressions, due,
        head=head, observing_task_id=observing_task_id, guard=guard,
    )
    if call_site in _CALL_SITES_WITH_LIVE_SIGNALS:
        _fire_live_signals(
            suppressions,
            head_sha=head_sha, observing_task_id=observing_task_id,
            event_store=event_store, escalation_queue=escalation_queue, guard=guard,
        )


async def record_merge_flake_suppression(
    result: VerifyResult,
    *,
    project_root: Path,
    project_id: str,
    merge_sha: str,
    task_id: str | None,
    event_store: EventStore | None = None,
    escalation_queue: Any = None,
) -> None:
    """Record the merge gate's flake observation *result* carries, through
    :func:`record_call_site_observations`.

    It unwraps the ``VerifyResult`` carrier and supplies the merge gate's half of the
    contract: the merge sha as the head, the task being merged as the observing task,
    and, on a ``passes_in_isolation`` observation only, every test it names as
    debt-eligible with no context.  A suppression is the only merge-gate observation
    that puts a test in the ledger: a confirmed red is a bug, not a flake, and an
    unconfirmable observation names no test to own.
    """
    s = getattr(result, 'flake_suppression', None)
    if s is None:
        # B13 — new dispatcher, OLD remote: the wire payload simply has no
        # `flake_suppression` key, so the field defaults to None.  That is a
        # degradation (an un-upgraded runner), not an observation: recording a
        # sentinel row for it would put fiction in the evidence trail.
        return

    suppressed = _is_suppression(s, _Guard(FlakeCallSite.merge_gate, merge_sha, task_id))
    debt_eligible: dict[str, DebtContext | None] = (
        {test_id: None for test_id in named_tests(s)} if suppressed else {}
    )
    await record_call_site_observations(
        (s,),
        call_site=FlakeCallSite.merge_gate,
        project_root=project_root,
        project_id=project_id,
        head_sha=merge_sha,
        observing_task_id=task_id,
        debt_eligible=debt_eligible,
        event_store=event_store,
        escalation_queue=escalation_queue,
    )
