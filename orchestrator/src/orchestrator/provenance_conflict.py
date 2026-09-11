"""Shared sink for ``done_evidence_stale`` rejections (task 2677).

The found_on_main provenance-integrity gate (task 2674's
``_check_reopen_freshness`` in fused-memory's ``TaskInterceptor``) rejects a
``set_task_status(done)`` write whose evidence commit predates the task's
most recent ``reopen_at`` stamp. ``orchestrator.scheduler.set_task_status``
classifies that rejection into a typed ``StaleEvidenceRejection`` (see
scheduler.py). This module is the CONSUMER of that exception: it converts a
rejection into a deduped, born-at-L2 ``provenance_conflict`` escalation
(INV-4: storm escape is ``dedupe_count``, never log spam) plus an in-memory
"terminal-for-this-tick" memo so the four orchestrator done-writers (dispatch
gate, stranded sweep, landed-outbox reconcile, coalesce re-drive) do not
re-attempt the same rejected write on every dispatch tick.

INV-5: one helper, not N copies — every done-writer site shares a single
``ProvenanceConflictSink`` instance constructed by the harness (see
plan.json design_decisions).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from escalation.dedupe import DedupeConfig, content_fingerprint_key, submit_or_dedupe
from escalation.models import Escalation

if TYPE_CHECKING:
    # Type-only: avoids a hard runtime import cycle with orchestrator.scheduler
    # (mirrors the TYPE_CHECKING-gated EscalationQueue import in scheduler.py).
    from escalation.queue import EscalationQueue

    from orchestrator.scheduler import StaleEvidenceRejection

logger = logging.getLogger(__name__)

__all__ = [
    'PROVENANCE_CONFLICT_AGENT_ROLE',
    'PROVENANCE_CONFLICT_CATEGORY',
    'ProvenanceConflictSink',
    'stale_evidence_fingerprint',
]

# Free-form category — escalation/server.CATEGORIES is inert/unvalidated
# (see plan.json design_decisions), so no escalation/ package change is
# needed to introduce this category.
PROVENANCE_CONFLICT_CATEGORY = 'provenance_conflict'

# Born-at-L2 sentinel role (mirrors deterministic_runner's
# 'orchestrator-deterministic' pattern) — keeps severity='urgent'/level=2
# past the escalation server's downgrade gate.
PROVENANCE_CONFLICT_AGENT_ROLE = 'orchestrator-provenance-conflict'

# Sentinel distinguishing "caller omitted reopen_at" (skip that invalidation
# arm of should_skip) from "caller explicitly passed None" — mirrors
# scheduler.py's _CLAIMANT_ABSENT pattern.
_UNKNOWN_REOPEN = object()


def stale_evidence_fingerprint(task_id: str, evidence_commit: str) -> str:
    """Return the dedupe fingerprint for a stale-evidence rejection.

    Keyed on ``(task_id, evidence_commit)`` — a repeat rejection for the
    SAME evidence commit folds into the same pending escalation via
    ``content_fingerprint_key`` (dedupe_count increments); a DIFFERENT
    evidence commit (e.g. a later writer's attempt after a new reopen)
    files a distinct record.
    """
    return f'{PROVENANCE_CONFLICT_CATEGORY}:{task_id}:{evidence_commit}'


def _dedupe_config() -> DedupeConfig:
    """DedupeConfig for provenance-conflict folding: unbounded window, fingerprint key.

    Mirrors ``DedupeConfig.for_recon()`` — an unbounded window so a
    recurring conflict over hours/days always folds into the same parent,
    and ``content_fingerprint_key`` so folding is keyed on
    ``dedupe_fingerprint`` (task_id + evidence_commit) rather than the
    summary-prefix default.
    """
    return DedupeConfig(
        infra_dedupe_enabled=True,
        infra_dedupe_window_secs=float('inf'),
        infra_dedupe_categories=(PROVENANCE_CONFLICT_CATEGORY,),
        key_fn=content_fingerprint_key,
    )


def _parse_reopen_at(value: object) -> datetime | None:
    """Best-effort parse of a ``reopen_at`` value to an aware UTC datetime.

    ``should_skip``'s memo comparison receives ``reopen_at`` strings from
    two different code paths: the memo'd side originates from the
    fused-memory interceptor's rejection payload (``StaleEvidenceRejection.
    reopen_at``), and the caller-supplied side is typically a live
    ``metadata.get('reopen_at')`` read (e.g. harness.py's dispatch-gate and
    stranded-sweep call sites). Both are stamped from the same underlying
    ``metadata.reopen_at`` field, so raw string equality is exact today —
    but this parses both sides as a defensive guard against any future
    ISO-8601 formatting drift between the two read paths (trailing ``Z``
    vs. ``+00:00``, microsecond precision, …). Returns ``None`` for
    anything not confidently parseable (non-string, empty, malformed).
    """
    if not isinstance(value, str) or not value:
        return None
    text = value[:-1] + '+00:00' if value.endswith('Z') else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _reopen_at_matches(caller_value: object, memo_value: str) -> bool:
    """True when *caller_value* denotes the same instant as *memo_value*.

    Tries exact string equality first (the common, expected case — see
    ``_parse_reopen_at``). Falls back to a parsed-instant comparison so
    ``should_skip`` is not defeated by incidental ISO-8601 formatting
    drift between the memo'd and caller-supplied ``reopen_at`` strings. An
    unparseable *caller_value* keeps the (already negative) raw-string
    verdict — never less safe than a plain ``==``.
    """
    if caller_value == memo_value:
        return True
    parsed_caller = _parse_reopen_at(caller_value)
    if parsed_caller is None:
        return False
    return parsed_caller == _parse_reopen_at(memo_value)


class ProvenanceConflictSink:
    """Records ``done_evidence_stale`` rejections; gates per-tick re-attempts.

    Protection layers (see plan.json design_decisions):
    - ``should_skip`` — an in-memory per-task memo giving "no per-tick
      retry" (terminal-for-this-tick). Lost on restart.
    - ``record`` — the durable disk fingerprint dedup (``submit_or_dedupe``)
      giving "exactly one escalation / dedupe_count on repeats", including
      across a restart that clears the memo, or from a different writer
      site.
    - ``arbitration_pending`` — the durable READ side (task 3534), deciding
      "is this task still bound by an arbitration?" from the store's pending
      records rather than the memo, so it holds on a process that did not
      itself file the conflict. It shares ``should_skip``'s invalidation
      semantics: a resolved escalation leaves ``get_by_task(...,
      status='pending')`` (and is skipped defensively even if an unfiltered
      read hands one over anyway), and a moved-on ``reopen_at`` releases
      too. A released hold does NOT mean "flip to done" — the escalation is
      still pending, so the caller's any-level pin veto then catches it and
      the task dispatches instead.
    - ``note_hold_observed`` / ``clear_hold_observed`` — log-streak
      bookkeeping for that hold. Unlike a veto that ABSTAINS (the task
      dispatches and stops being a candidate), a hold re-fires on every tick
      until the arbitration ends, so its caller logs loudly only on a
      ``(task_id, reopen_at)`` TRANSITION and quietly on the repeats (INV-4:
      storm escape is ``dedupe_count``, never log spam).

    ``escalation_queue`` is a late-bindable public attribute: the harness
    constructs this sink before its ``EscalationQueue`` exists (the merge
    worker is built first) and sets ``.escalation_queue`` once the queue is
    created. Every method is None-safe: with no queue, ``record`` memoizes
    in-process only and returns ``None`` instead of raising.
    """

    def __init__(self, escalation_queue: EscalationQueue | None = None) -> None:
        self.escalation_queue = escalation_queue
        # task_id -> (reopen_at, evidence_commit, escalation_id | None)
        self._memo: dict[str, tuple[str, str, str | None]] = {}
        # task_id -> the reopen_at whose hold was last LOGGED loudly (None
        # when the caller had none); see note_hold_observed/clear_hold_observed.
        self._hold_logged: dict[str, str | None] = {}

    def record(
        self,
        *,
        task_id: str,
        evidence_commit: str,
        evidence_committed_at: str,
        reopen_at: str,
        agent_id: str,
        gate_source: str,
    ) -> str | None:
        """File (or fold) a born-at-L2 ``provenance_conflict`` escalation.

        Returns the escalation id (the parent's id on a dedupe fold), or
        ``None`` when no queue is bound yet — the memo is still updated
        (with a ``None`` escalation id) so ``should_skip`` still gates the
        in-process retry even before the queue is late-bound.
        """
        if self.escalation_queue is None:
            # task 2677 amendment (reviewer_comprehensive
            # robustness_silent_degradation): a rejection can land here
            # before the harness late-binds .escalation_queue (the merge
            # worker's run loop can start before that bind happens — see
            # plan.json design_decisions). Memoizing with escalation_id=None
            # is bounded rather than permanently silent: _escalation_pending
            # treats a None escalation_id as "still blocking" ONLY while
            # self.escalation_queue remains None (see below). The instant a
            # queue is later bound, it invalidates that memo — the next
            # should_skip caller stops skipping, retries the write, and that
            # retry's fresh rejection reaches record() again with a queue
            # now bound, filing a real escalation. Until the queue is bound,
            # log at WARNING so the interim window stays observable rather
            # than a silent no-op (loud-over-silent-degradation norm).
            logger.warning(
                'ProvenanceConflictSink.record: no escalation_queue bound yet '
                '— memoizing task %s (evidence %s, gate_source=%s) in-process '
                'only; should_skip will gate this task until the queue is '
                'late-bound, at which point the next attempt self-heals into '
                'a real escalation',
                task_id, evidence_commit, gate_source,
            )
            self._memo[task_id] = (reopen_at, evidence_commit, None)
            return None

        detail: dict[str, Any] = {
            'task_id': task_id,
            'evidence_commit': evidence_commit,
            'evidence_committed_at': evidence_committed_at,
            'reopen_at': reopen_at,
            'agent_id': agent_id,
            'gate_source': gate_source,
        }
        esc = Escalation(
            id=self.escalation_queue.make_id(task_id),
            task_id=task_id,
            agent_role=PROVENANCE_CONFLICT_AGENT_ROLE,
            severity='urgent',
            category=PROVENANCE_CONFLICT_CATEGORY,
            summary=(
                f'Stale done-evidence for task {task_id}: evidence '
                f'{evidence_commit} predates reopen_at {reopen_at}'
            )[:200],
            detail=json.dumps(detail, sort_keys=True),
            level=2,
            dedupe_fingerprint=stale_evidence_fingerprint(task_id, evidence_commit),
        )
        result = submit_or_dedupe(self.escalation_queue, esc, _dedupe_config())
        escalation_id = str(result['id'])
        self._memo[task_id] = (reopen_at, evidence_commit, escalation_id)
        return escalation_id

    def record_from_rejection(
        self, exc: StaleEvidenceRejection, *, gate_source: str,
    ) -> str | None:
        """Adapt a caught ``StaleEvidenceRejection`` into ``record(...)``."""
        return self.record(
            task_id=exc.task_id,
            evidence_commit=exc.evidence_commit,
            evidence_committed_at=exc.evidence_committed_at,
            reopen_at=exc.reopen_at,
            agent_id=exc.agent_id,
            gate_source=gate_source,
        )

    def should_skip(self, task_id: str, *, reopen_at: str | object = _UNKNOWN_REOPEN) -> bool:
        """True while a conflict is memo'd for *task_id* — the caller must not retry.

        Invalidates (returns False) when:
        - no rejection has been memo'd for this task, OR
        - *reopen_at* is given and differs from the memo'd value (the task
          was reopened again since the conflict was recorded — a fresh
          write attempt is warranted). Compared via ``_reopen_at_matches``
          (exact string match, with a parsed-instant fallback that guards
          against incidental ISO-8601 formatting drift between the memo'd
          and caller-supplied strings — see that helper), OR
        - the recorded escalation is no longer pending (an operator
          resolved the arbitration), OR
        - the memo was recorded with no escalation filed (queue unbound at
          record() time) and a queue is now bound — see
          ``_escalation_pending``'s handling of ``escalation_id is None``.

        Passing no ``reopen_at`` (the ``_UNKNOWN_REOPEN`` sentinel) skips
        that invalidation arm — used by callers that do not have the
        task's current metadata handy.

        A memo whose escalation is no longer pending, or whose *reopen_at*
        no longer matches the caller-supplied value, is pruned from
        ``self._memo`` before returning False (task 2677 amendments,
        reviewer_comprehensive resource_management) — otherwise the entry
        would linger for the rest of the process lifetime, growing
        ``self._memo`` monotonically with the number of distinct tasks
        that ever hit a stale-evidence conflict. A *reopen_at* mismatch is
        pruned immediately rather than left for the next ``record()`` to
        overwrite: the common self-heal case is that the retried write,
        now carrying a fresh ``reopen_at``, succeeds outright — in which
        case ``record()`` is never called again and a left-in-place entry
        would never be overwritten.
        """
        memo = self._memo.get(task_id)
        if memo is None:
            return False
        memo_reopen_at, _evidence_commit, escalation_id = memo
        if reopen_at is not _UNKNOWN_REOPEN and not _reopen_at_matches(reopen_at, memo_reopen_at):
            del self._memo[task_id]
            return False
        if self._escalation_pending(escalation_id):
            return True
        del self._memo[task_id]
        return False

    def arbitration_pending(
        self,
        pending_records: Iterable[Any] | None,
        *,
        reopen_at: str | object = _UNKNOWN_REOPEN,
    ) -> bool:
        """True while a pending ``provenance_conflict`` record binds this task.

        The restart-DURABLE twin of :meth:`should_skip`: decided from the
        escalation store's own records instead of ``self._memo``, so it
        survives the two cases the memo cannot cover — a cold memo (fleet
        redeploy / watchdog restart) and a conflict filed by a different
        writer site (``merge_queue.py``'s two ``ProvenanceConflictSink``
        call sites).  ``should_skip`` remains the zero-I/O in-process fast
        path; this is the correctness backstop underneath it.

        Takes ALREADY-READ records rather than doing its own ``get_by_task``
        so the caller pays for exactly one store read.
        ``pending_records=None`` (an unreadable store) returns ``False``:
        that disposition belongs to the caller, which already treats a failed
        read as ``store_unavailable`` (see the already-landed dispatch gate's
        try/except).

        PRECONDITION, named in the parameter: pass the rows of
        ``get_by_task(task_id, status='pending')``.  A resolved record must
        RELEASE the hold — that is how an operator ends an arbitration, and it
        is the same release arm ``should_skip`` gets from
        ``_escalation_pending``'s ``esc.status == 'pending'`` check.  Records
        whose ``status`` is present and not ``'pending'`` are therefore
        skipped defensively here too, so a caller that hands over an
        unfiltered ``get_by_task(task_id)`` (which also scans the archive
        tier) cannot turn a long-resolved conflict into a permanent
        withhold — precisely the wedge the reopen_at arm below exists to
        avoid.  A record with no ``status`` attribute at all is treated as
        pending, keeping the ambiguity-holds asymmetry below.

        Shares :meth:`should_skip`'s reopen_at INVALIDATION ARM, comparing
        the caller's value against the one ``record()`` serialised into the
        escalation's ``detail`` JSON.  That arm is load-bearing, not
        incidental — ``merge_queue.py`` names it as the self-heal for a task
        reopened after its conflict was filed — so a hold that ignored it
        would trade the cold-memo bug for a permanent wedge (neither
        dispatched nor flipped until a human resolved the L2).  Passing no
        ``reopen_at`` (the ``_UNKNOWN_REOPEN`` sentinel) skips the arm,
        exactly as it does for the memo, so the two layers invalidate on ONE
        shared signal rather than two subtly different ones.

        Deliberate asymmetry: EVERY ambiguity — caller passed no
        ``reopen_at``, ``detail`` missing or unparseable, recorded value
        absent or not a string — resolves to HOLD.  Releasing is the action
        that lets a contested task move, so it must require positive
        evidence that the arbitration is stale; the absence of evidence is
        never enough.  ``_reopen_at_matches`` (not ``==``) performs the
        comparison, so incidental ISO-8601 spelling drift between the stored
        and caller-supplied strings cannot masquerade as a moved-on value.
        """
        if not pending_records:
            return False
        for record in pending_records:
            if getattr(record, 'category', None) != PROVENANCE_CONFLICT_CATEGORY:
                continue
            if getattr(record, 'status', 'pending') != 'pending':
                # Defence in depth for the PENDING precondition above: only
                # a still-open arbitration binds the task, so a resolved /
                # dismissed row released it (mirrors _escalation_pending's
                # `esc.status == 'pending'`).  Cheap enough to be worth not
                # depending on every present and future caller having
                # remembered `status='pending'` on its read.
                continue
            if reopen_at is _UNKNOWN_REOPEN:
                return True
            try:
                recorded = json.loads(
                    getattr(record, 'detail', None) or '{}',
                ).get('reopen_at')
            except (TypeError, ValueError, AttributeError):
                return True
            if not isinstance(recorded, str) or _reopen_at_matches(reopen_at, recorded):
                return True
        return False

    def note_hold_observed(
        self, task_id: str, *, reopen_at: str | object = _UNKNOWN_REOPEN,
    ) -> bool:
        """Register an :meth:`arbitration_pending` hold; True on a TRANSITION.

        INV-4 ("storm escape is ``dedupe_count``, never log spam") applied to
        the hold's log line.  The any-level pin veto does not need this — it
        ABSTAINS, so the task dispatches and stops being a gate candidate —
        but a hold repeats on EVERY dispatch tick for as long as the
        arbitration stands, and on a cold-memo process it short-circuits
        before ``record()`` and so never re-warms anything that would quiet
        it.  A caller logging it unconditionally at INFO would emit one line
        per scheduler tick, indefinitely, until a human resolved the L2.

        So the caller logs LOUDLY (INFO) only when this returns True — the
        first observation of a given ``(task_id, reopen_at)`` — and quietly
        (DEBUG) on the repeats.  A moved-on ``reopen_at`` is a genuinely new
        hold state and is loud again, which is exactly the transition an
        operator wants to see.

        Advisory per-process state, deliberately: losing it on a restart
        costs one extra INFO line per contested task, which is the right side
        to fail on.  :meth:`clear_hold_observed` prunes it when the hold
        releases, so it stays bounded by the count of CURRENTLY-held tasks
        rather than every task ever contested in this process (the same
        resource_management discipline ``should_skip``'s memo carries).
        """
        key = reopen_at if isinstance(reopen_at, str) else None
        if task_id in self._hold_logged and self._hold_logged[task_id] == key:
            return False
        self._hold_logged[task_id] = key
        return True

    def clear_hold_observed(self, task_id: str) -> None:
        """Drop *task_id*'s hold-log state — the arbitration no longer binds.

        Called by a caller whose :meth:`arbitration_pending` came back False,
        so a hold that returns later announces itself loudly again instead of
        being silently folded into a stale streak, and so
        :meth:`note_hold_observed`'s dict cannot grow monotonically.
        """
        self._hold_logged.pop(task_id, None)

    def _escalation_pending(self, escalation_id: str | None) -> bool:
        """True when *escalation_id* is still open (or unverifiable).

        ``escalation_id is None`` means ``record()`` memoized before a
        queue was bound (its no-queue branch). WHILE the queue remains
        unbound, that is treated as "still blocking" — best-effort
        in-memory-only mode, matching the None-queue-safe contract: we
        cannot verify or file an escalation without a queue, so we do not
        prematurely allow a retry storm. But the moment a queue IS bound,
        a ``None`` escalation_id memo no longer reflects reality — treat it
        as no-longer-pending (return False) so ``should_skip`` stops
        gating and the next caller retries the write. That retry's fresh
        rejection reaches ``record()`` again, now with a queue bound, and
        files the real escalation (task 2677 amendment,
        reviewer_comprehensive robustness_silent_degradation — previously
        this returned True unconditionally, silently gating the task for
        the rest of the process lifetime with no escalation ever filed).

        Deliberately uncached (task 2677 amendment, reviewer_comprehensive
        efficiency): ``self.escalation_queue.get(...)`` performs a disk
        read on every call that finds a live memo, so a task under active
        arbitration incurs one read per dispatch tick / stranded-sweep
        cycle for as long as its conflict stays open. Reviewed and left
        as-is — call volume is bounded by the (rare) count of
        concurrently-contested tasks, this is not a hot path, and a
        TTL/tick-based cache would trade that minor, bounded I/O cost for
        a real risk: staleness in the one signal (an operator's
        resolution) this check exists to observe promptly.
        """
        if escalation_id is None:
            return self.escalation_queue is None
        if self.escalation_queue is None:
            return True
        esc = self.escalation_queue.get(escalation_id)
        return esc is not None and esc.status == 'pending'
