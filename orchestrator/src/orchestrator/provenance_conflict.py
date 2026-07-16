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
from typing import TYPE_CHECKING, Any

from escalation.dedupe import DedupeConfig, content_fingerprint_key, submit_or_dedupe
from escalation.models import Escalation

if TYPE_CHECKING:
    # Type-only: avoids a hard runtime import cycle with orchestrator.scheduler
    # (mirrors the TYPE_CHECKING-gated EscalationQueue import in scheduler.py).
    from escalation.queue import EscalationQueue

    from orchestrator.scheduler import StaleEvidenceRejection

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

    def __init__(self, escalation_queue: 'EscalationQueue | None' = None) -> None:
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
        self, exc: 'StaleEvidenceRejection', *, gate_source: str,
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
          write attempt is warranted), OR
        - the recorded escalation is no longer pending (an operator
          resolved the arbitration).

        Passing no ``reopen_at`` (the ``_UNKNOWN_REOPEN`` sentinel) skips
        that invalidation arm — used by callers that do not have the
        task's current metadata handy.
        """
        memo = self._memo.get(task_id)
        if memo is None:
            return False
        memo_reopen_at, _evidence_commit, escalation_id = memo
        if reopen_at is not _UNKNOWN_REOPEN and reopen_at != memo_reopen_at:
            return False
        return self._escalation_pending(escalation_id)

    def _escalation_pending(self, escalation_id: str | None) -> bool:
        """True when *escalation_id* is still open (or unverifiable).

        ``None`` (no queue was bound at record() time) is treated as "still
        blocking" — best-effort in-memory-only mode, matching the
        None-queue-safe contract: we cannot verify resolution without a
        queue, so we do not prematurely allow a retry storm.
        """
        if escalation_id is None or self.escalation_queue is None:
            return True
        esc = self.escalation_queue.get(escalation_id)
        return esc is not None and esc.status == 'pending'
