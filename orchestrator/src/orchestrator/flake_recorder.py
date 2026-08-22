"""Dispatcher-side RECORDER for merge-gate flake observations (PRD task ε).

`plans/cpu-load-robust-verify-prd.md` §5.8 — the topology rule this module exists
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

Splitting them makes the three side-effects unconditional BY CONSTRUCTION rather
than dependent on which host ran the verify: the producer merely ATTACHES a
``FlakeSuppression`` to the ``VerifyResult``, the observation rides the wire home,
and this module records it.

Import discipline: this module imports ``flake_ledger`` (which depends only on
``shared.sqlite_sync_base``) at runtime and NOTHING else from ``orchestrator``.
``VerifyResult`` is a ``TYPE_CHECKING``-only annotation and ``EventType`` /
``Escalation`` are imported lazily inside their functions, so ``flake_recorder``
never imports ``verify`` or ``event_store`` at runtime and cannot participate in an
import cycle with either.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from orchestrator.flake_ledger import (
    FlakeVerdict,
    ledger_db_path,
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


def record_merge_flake_suppression(
    result: VerifyResult,
    *,
    project_root: Path,
    project_id: str,
    merge_sha: str,
    task_id: str | None,
    event_store: EventStore | None = None,
    escalation_queue: Any = None,
) -> None:
    """Record the flake observation *result* carries — the whole of task ε's job.

    Three side-effects, on two different triggers:

    * the durable ``flake_occurrence`` ledger row(s) — on EVERY carried verdict,
      including ``fails_in_isolation`` and ``unconfirmable``.  §5.5: record the
      OBSERVATION, not the remedy.  The non-suppressing verdicts are what make θ's
      health checks computable at all — an unconfirmable RATE needs its numerator,
      and a suppression RATE needs the confirmed-red denominator — which is why
      ``record_flake_occurrence`` counts an unconfirmable observation under its
      ``UNKNOWN_TEST_ID`` sentinel rather than dropping it;
    * the ``merge_flake_suppressed`` fact and the INV-4 storm-streak bump — on
      ``passes_in_isolation`` ONLY, byte-identical to the trigger condition these
      two had inline before ε.  Nothing about WHEN they fire changes here; only
      WHERE.

    Order is deliberate: the durable row first, then the in-process signals.  The
    row outlives this process and is the evidence θ reads; the event and the streak
    are live telemetry.  If anything is going to be lost to a crash mid-call, lose
    the recoverable half.

    ``==``, never ``is``, on the verdict.  The observation may have been rebuilt from
    JSON by ``flake_ledger.flake_suppression_from_wire``, and an unrecognised
    vocabulary string is deliberately PRESERVED there rather than coerced — so an
    identity test would skip the emit and the bump for exactly the remote
    suppressions ε exists to make visible.  (``StrEnum`` makes ``==`` correct for
    both the member and its wire spelling.)

    NEVER RAISES (B12).  A ledger write, an event emit or an escalation submit
    failing is a lost measurement; letting it propagate would fail a VERIFY, or
    stall the merge queue, over bookkeeping.  The catch-all is the outer boundary —
    ``record_flake_occurrence`` has its own B12 guard inside it, so a broken ledger
    degrades to a warning and the event and streak still fire, which is why losing
    one signal here does not cost the other two.

    None-safe on both stores: the CLI and any storeless caller still contribute the
    durable row.
    """
    s = getattr(result, 'flake_suppression', None)
    if s is None:
        # B13 — new dispatcher, OLD remote: the wire payload simply has no
        # `flake_suppression` key, so the field defaults to None.  That is a
        # degradation (an un-upgraded runner), not an observation: recording a
        # sentinel row for it would put fiction in the evidence trail.
        return

    try:
        record_flake_occurrence(
            ledger_db_path(project_root),
            project_id,
            s,
            merge_sha=merge_sha or None,
            task_id=task_id,
        )

        if s.verdict == FlakeVerdict.passes_in_isolation:
            _emit_merge_flake_suppressed(
                event_store, task_id, merge_sha, list(s.test_ids),
            )
            _bump_suppression_streak_and_maybe_escalate(
                escalation_queue, task_id, merge_sha,
            )
    except Exception:
        logger.warning(
            'flake_recorder: failed to record merge flake suppression '
            '(merge_sha=%s, task_id=%s); the merge/verify is unaffected',
            merge_sha,
            task_id,
            exc_info=True,
        )
