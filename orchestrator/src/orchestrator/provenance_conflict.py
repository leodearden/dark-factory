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

    Two-layer protection (see plan.json design_decisions):
    - ``should_skip`` — an in-memory per-task memo giving "no per-tick
      retry" (terminal-for-this-tick). Lost on restart.
    - ``record`` — the durable disk fingerprint dedup (``submit_or_dedupe``)
      giving "exactly one escalation / dedupe_count on repeats", including
      across a restart that clears the memo, or from a different writer
      site.

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

        A memo whose escalation is no longer pending is pruned from
        ``self._memo`` before returning False (task 2677 amendment,
        reviewer_comprehensive resource_management) — otherwise a resolved
        conflict's entry would linger for the rest of the process
        lifetime, growing ``self._memo`` monotonically with the number of
        distinct tasks that ever hit a stale-evidence conflict. A memo
        invalidated by a *reopen_at* mismatch is left in place: it is
        still the most recent record() write and will be overwritten (not
        duplicated) by the next one.
        """
        memo = self._memo.get(task_id)
        if memo is None:
            return False
        memo_reopen_at, _evidence_commit, escalation_id = memo
        if reopen_at is not _UNKNOWN_REOPEN and not _reopen_at_matches(reopen_at, memo_reopen_at):
            return False
        if self._escalation_pending(escalation_id):
            return True
        del self._memo[task_id]
        return False

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
        """
        if escalation_id is None:
            return self.escalation_queue is None
        if self.escalation_queue is None:
            return True
        esc = self.escalation_queue.get(escalation_id)
        return esc is not None and esc.status == 'pending'
