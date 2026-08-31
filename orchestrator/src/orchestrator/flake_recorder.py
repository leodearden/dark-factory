"""Dispatcher-side RECORDER for merge-gate flake observations (PRD task ε).

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

Task ζ added a FOURTH side-effect on the same seam — the ``flake_debt`` row, whose
``owner_task_id`` names a de-flake task filed AT WRITE TIME.  That is where PRD
§5.9's invariant is enforced ("any test in the flaky ledger has a non-terminal
de-flake task explicitly responsible both for fixing the root defect and for
removing the test from the ledger"), and it belongs here for the same topology
reason the other three do: the dispatcher is the only host that has a task client.
The COUPLING RULE that comes with it is binding — the ledger READS task status and
WRITES it only as part of the initial filing, never marking a task done, blocked or
reprioritised.

Import discipline: this module imports ``flake_ledger`` (which depends only on
``shared.sqlite_sync_base``) at runtime and NOTHING else from ``orchestrator``.
``VerifyResult`` is a ``TYPE_CHECKING``-only annotation and ``EventType`` /
``Escalation`` are imported lazily inside their functions, so ``flake_recorder``
never imports ``verify`` or ``event_store`` at runtime and cannot participate in an
import cycle with either.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from orchestrator.flake_ledger import (
    FlakeVerdict,
    ledger_db_path,
    open_debt,
    record_flake_occurrence,
)

if TYPE_CHECKING:
    from orchestrator.event_store import EventStore
    from orchestrator.verify import VerifyResult

logger = logging.getLogger(__name__)


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
        category='merge_flake_suppression_storm',
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


async def record_merge_flake_suppression(
    result: VerifyResult,
    *,
    project_root: Path,
    project_id: str,
    merge_sha: str,
    task_id: str | None,
    event_store: EventStore | None = None,
    escalation_queue: Any = None,
    task_client: Any = None,
) -> None:
    """Record the flake observation *result* carries — ε's job, plus ζ's fourth effect.

    Four side-effects, on two different triggers:

    * the durable ``flake_occurrence`` ledger row(s) — on EVERY carried verdict,
      including ``fails_in_isolation`` and ``unconfirmable``.  §5.5: record the
      OBSERVATION, not the remedy.  The non-suppressing verdicts are what make θ's
      health checks computable at all — an unconfirmable RATE needs its numerator,
      and a suppression RATE needs the confirmed-red denominator — which is why
      ``record_flake_occurrence`` counts an unconfirmable observation under its
      ``UNKNOWN_TEST_ID`` sentinel rather than dropping it;
    * the ``merge_flake_suppressed`` fact, the INV-4 storm-streak bump, and the
      ``flake_debt`` row — on ``passes_in_isolation`` ONLY.  The first two are
      byte-identical to the trigger condition they had inline before ε; nothing about
      WHEN they fire changes here, only WHERE.  The debt row joins them on the same
      trigger because §5.9's invariant is about tests that are IN the ledger, and only
      a suppression puts one there: a confirmed red is a bug, not a flake, and an
      unconfirmable observation names no test to own.

    THE FOURTH EFFECT (task ζ, PRD §5.9): each carried ``test_id`` gets
    ``flake_ledger.open_debt``, which enforces at WRITE TIME that the test has a
    non-terminal de-flake task responsible BOTH for fixing the root defect AND for
    removing the test from the ledger.  Enforcing it here — rather than auditing it
    afterwards — is what makes the invariant self-maintaining.  The COUPLING RULE it
    inherits is binding: the ledger READS task status and writes it only as part of the
    initial filing, never marking a task done, blocked or reprioritised.  With no
    ``task_client`` wired (the CLI, and ``merge_queue.reverify_member_solo``, which
    holds no worker to read one from) NO debt row is opened at all, and those callers
    stay byte-identical to their pre-ζ behaviour.  ``open_debt`` itself would write the
    row unowned — it has to, being safe for any direct caller — but ι renders an
    ownerless row as an invariant BREACH, and a caller that structurally cannot own
    anything is a configuration, not a breach; manufacturing rows here would swamp the
    one surface that exists to show a filing which was attempted and failed.

    "STRUCTURALLY cannot own" is a claim about SCOPE and is worth checking before it is
    repeated: it holds for a caller with no worker in scope, and it does NOT hold merely
    because a call site currently passes nothing.  The train pipeline
    (``merge_queue._do_train_merge``) was such a site — it read five other fields off
    the worker it is handed — and it now passes the handle; treating "unthreaded" as
    "incapable" is what let §5.9 go unenforced there while a comment asserted the
    opposite reason.

    ORDER IS THE CONTRACT, not an incidental sequence — local/durable first (the
    occurrence rows), then the in-process live signals (the event, the streak), then the
    NETWORK-BOUND filing LAST:

    * the occurrence row outlives this process and is the evidence θ reads, so if
      anything is lost to a crash mid-call, lose the recoverable half;
    * the filing dispatches an MCP tool, so it is by far the likeliest of the four to
      fail or hang, and INV-4's fixed-sentinel storm escape
      (:func:`_bump_suppression_streak_and_maybe_escalate`) is the one signal whose
      entire job is to fire when α is masking too much.  Bumping the streak BEFORE the
      client is ever touched makes the escape armed independently of the fail-soft path
      — otherwise a systemic outage of the filing path would switch off the alarm for
      exactly that outage.

    ζ's write-time filing rides that same escape, deliberately: a storm of DISTINCT
    tests each acquiring their own de-flake task is a systemic signal that the per-test
    dedup (one owner per ``test_id``) cannot bound, and the streak covers it until task
    θ's dedicated class-3 systemic counter lands.

    ``==``, never ``is``, on the verdict.  The observation may have been rebuilt from
    JSON by ``flake_ledger.flake_suppression_from_wire``, and an unrecognised
    vocabulary string is deliberately PRESERVED there rather than coerced — so an
    identity test would skip the emit and the bump for exactly the remote
    suppressions ε exists to make visible.  (``StrEnum`` makes ``==`` correct for
    both the member and its wire spelling.)

    NEVER RAISES (B12).  A ledger write, an event emit, an escalation submit or a task
    filing failing is a lost measurement; letting it propagate would fail a VERIFY, or
    stall the merge queue, over bookkeeping.

    The four side-effects are INDEPENDENTLY guarded — one ``try`` each — so losing one
    really does not cost the others.  A single shared ``try`` made that claim false by
    ordering alone: an ``event_store.emit`` that raised (a locked or closed sqlite
    store) would skip straight to the catch-all and the streak bump would never run, so
    a broken event store silently disarmed the INV-4 storm detector — the one signal
    whose whole job is to fire when something is going wrong.

    None-safe on all three collaborators: the CLI and any storeless caller still
    contribute the durable row.
    """
    s = getattr(result, 'flake_suppression', None)
    if s is None:
        # B13 — new dispatcher, OLD remote: the wire payload simply has no
        # `flake_suppression` key, so the field defaults to None.  That is a
        # degradation (an un-upgraded runner), not an observation: recording a
        # sentinel row for it would put fiction in the evidence trail.
        return

    def _lost(what: str) -> None:
        """The one place a lost signal is reported, shared by both guards so the sync
        and async effects are indistinguishable in the log."""
        logger.warning(
            'flake_recorder: failed to record merge flake suppression [%s] '
            '(merge_sha=%s, task_id=%s); the merge/verify is unaffected',
            what,
            merge_sha,
            task_id,
            exc_info=True,
        )

    def _guarded(what: str, fn: Callable[[], None]) -> None:
        """Run one side-effect under its OWN catch-all, so a failure costs exactly
        that signal.  Also guards the verdict read itself: `s` is an un-validated
        wire object, so even `s.verdict` can fail on a malformed payload."""
        try:
            fn()
        except Exception:
            _lost(what)

    async def _guarded_async(what: str, fn: Callable[[], Awaitable[object]]) -> None:
        """The async sibling of :func:`_guarded` — identical per-signal isolation, for
        an effect that must be awaited.

        Deliberately a separate function rather than teaching ``_guarded`` to sniff its
        return value for awaitability: that would leave every existing sync call site
        one ``inspect.isawaitable`` miss away from silently dropping a coroutine, which
        is a lost measurement that logs NOTHING — the exact failure mode this module's
        independently-guarded discipline exists to make impossible.
        """
        try:
            await fn()
        except Exception:
            _lost(what)

    _guarded(
        'ledger',
        lambda: record_flake_occurrence(
            ledger_db_path(project_root),
            project_id,
            s,
            merge_sha=merge_sha or None,
            task_id=task_id,
        ),
    )

    # Read the verdict once, under its own guard, so a malformed observation cannot
    # decide the two live signals by raising.  Defaults to False = "not a suppression",
    # which is the fail-safe reading: a bogus event and a phantom streak bump would be
    # worse than a missing one.
    suppressed = False
    try:
        suppressed = s.verdict == FlakeVerdict.passes_in_isolation
    except Exception:
        logger.warning(
            'flake_recorder: unreadable verdict on the carried observation '
            '(merge_sha=%s, task_id=%s); skipping the event and the streak bump',
            merge_sha,
            task_id,
            exc_info=True,
        )

    if suppressed:
        _guarded(
            'event',
            lambda: _emit_merge_flake_suppressed(
                event_store, task_id, merge_sha, list(s.test_ids),
            ),
        )
        # Its OWN guard, and deliberately AFTER the emit rather than inside it: the
        # storm detector is the escape hatch for "α is masking too much", so it must
        # survive a broken event store rather than being taken down with it.
        _guarded(
            'streak',
            lambda: _bump_suppression_streak_and_maybe_escalate(
                escalation_queue, task_id, merge_sha,
            ),
        )
        # LAST, and that ordering is the contract (see the docstring): the streak above
        # is now armed, so the NETWORK-BOUND filing below — the likeliest of the four to
        # fail or hang — cannot disarm INV-4's escape by failing.
        #
        # ONE GUARD PER TEST, not one for the batch: a two-test observation is two
        # independent defects, and a filing that fails for one must not cost the other
        # its owner.  `UNKNOWN_TEST_ID` is deliberately NOT filtered here — `open_debt`
        # refuses the sentinel itself, so that policy stays in exactly one place.
        #
        # UNWIRED IS NOT A BREACH.  With no `task_client` this opens NO debt row at all,
        # rather than opening one unowned.  `open_debt` would happily write the row and
        # log the missing client (it must — it is safe for any caller, including a
        # direct CLI one), but ι renders an ownerless row as
        # `*** NO OWNER (invariant breach) ***`, and a caller that STRUCTURALLY cannot
        # own anything is a configuration, not a breach.  Manufacturing rows here would
        # swamp that surface with non-breaches and destroy its ability to show the one
        # thing it exists for: a filing that was attempted and FAILED.  It also keeps
        # the two `_run_post_merge_verify` callers that thread no client byte-identical
        # to their pre-ζ behaviour.
        if task_client is None:
            return
        carried: list[str] = []
        _guarded('debt-test-ids', lambda: carried.extend(s.test_ids))
        for carried_test_id in carried:
            await _guarded_async(
                'debt',
                lambda tid=carried_test_id: open_debt(
                    ledger_db_path(project_root),
                    project_id,
                    tid,
                    task_client=task_client,
                ),
            )
