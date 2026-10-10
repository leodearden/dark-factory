"""Storm escapes for Hook A's entity standing decisions (tasks 2896 and 2943).

Hook A (``flag_dedup.py::filter_entity_standing_decisions``) drops Stage-1
flags that an ACTIVE standing decision has already adjudicated.  A filter whose
whole job is dropping findings must not do it unaccountably, so two escapes
file (or fold onto) one L1 escalation per standing decision, both in
``CATEGORY_STANDING_DECISION_STORM``:

- the per-cycle escape, :func:`maybe_escalate_suppression_storm`, for a
  decision that suppressed more than N flags in one cycle;
- the streak escape, for a decision that suppressed flags in at least K
  consecutive full cycles and more than N of them across its last K.
  :func:`update_suppression_streaks` advances that state on the recon ledger
  and returns a verdict per decision, and
  :func:`maybe_escalate_suppression_streak` files the ones that escalate.

Both read Hook A's ``EntityStandingSuppressionResult``; Hook A never depends
on this module.  The wiring is
``stages/memory_consolidator.py::MemoryConsolidator.run`` and the design is
``plans/stage1-entity-standing-decision-prd.md`` §Storm escapes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from fused_memory.reconciliation.flag_dedup import EntityStandingSuppressionResult
from fused_memory.reconciliation.recon_ledger import SuppressionStreakRow
from fused_memory.reconciliation.standing_decision_constants import (
    CATEGORY_STANDING_DECISION_STORM,
    STANDING_DECISION_TTL_DAYS,
    SUPPRESSION_STORM_THRESHOLD_PER_CYCLE,
    SUPPRESSION_STREAK_THRESHOLD_CYCLES,
    SUPPRESSION_STREAK_VOLUME_THRESHOLD,
)

# Optional escalation dependency: the reconciliation package must import
# cleanly where the escalation package is not installed, and both filers no-op
# when the name is None.
try:
    from escalation.dedupe import file_or_fold_l1  # type: ignore[import-untyped]
except ImportError:
    file_or_fold_l1 = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


#: Finding-category component of the storm escalation's dedupe fingerprint —
#: the second axis of ``compute_content_fingerprint``, distinguishing this
#: finding from any other that might one day share the storm category.
_STORM_FINDING_CATEGORY: str = 'entity_standing_decision_suppression_storm'

#: The streak arm's finding category (task 2943).  It shares the storm's
#: escalation category, so this distinct second fingerprint axis is what stops a
#: streak record and a per-cycle storm record for the same entity folding into
#: each other: a flood in one cycle and a persistent drain are different
#: diagnoses, and folding either into the other would hide it for any entity
#: that had ever tripped the other escape.
_STREAK_FINDING_CATEGORY: str = 'entity_standing_decision_suppression_streak'

_AGENT_ROLE: str = 'reconciliation-stage1'
_LOG_LABEL: str = 'standing-decision storm escape'


async def maybe_escalate_suppression_storm(
    escalation_queue: Any,
    project_id: str,
    run_id: str,
    result: EntityStandingSuppressionResult,
    *,
    threshold: int = SUPPRESSION_STORM_THRESHOLD_PER_CYCLE,
) -> list[str]:
    """File (or fold) a "storm escape" L1 escalation per over-active standing decision.

    For each ``(entity_uuid, count)`` in ``result.suppressed_by_decision`` whose
    *count* exceeds *threshold* (strict ``>``), submit one
    ``Escalation(level=1, severity='blocking',
    category=CATEGORY_STANDING_DECISION_STORM,
    agent_role='reconciliation-stage1', ...)`` naming the entity, its grounds,
    and the per-cycle count. An active decision hiding a flood of flags in one
    cycle is a signal it may be over-broad or the entity's situation changed —
    worth a human look.

    **Filed through** ``escalation/dedupe.py::file_or_fold_l1``, NOT gated on
    ``has_open_l1`` (task 3522). Stage 1 re-evaluates every cycle, so a decision
    that storms once tends to storm every cycle — exactly the recurring-detector
    shape for which the sibling gate-backlog path retired the ``has_open_l1``
    skip: that skip suppressed every cycle after the first, so ``dedupe_count``
    stayed pinned at 0 and the operator saw no difference between one storm and
    forty. Folding instead keeps ONE pending record per entity and increments
    ``dedupe_count`` on it, which is the steward's recurrence / triage-order
    signal; the fold key and window are that helper's. Accepted cost, as on the
    gate-backlog path: a folded record keeps the PARENT's summary, so the count
    named there is the FIRST breach's, while ``dedupe_count`` carries how often
    it has recurred since.

    Best-effort throughout: returns ``[]`` immediately when the ``escalation``
    package is unavailable; any per-decision fingerprint/id-gen/construction/
    submit/fold failure logs WARNING and excludes that entity from the returned
    list. Returns the entity_uuids that received a NEW record this cycle — folds
    are excluded, so the list means "new filings", with recurrence carried by
    ``dedupe_count`` and the fold INFO log.

    Its cross-cycle sibling, :func:`maybe_escalate_suppression_streak`, files
    the same category for a decision whose streak has reached K and that
    suppressed more than N flags in total across its last K cycles.
    """
    if file_or_fold_l1 is None:
        return []

    escalated: list[str] = []
    for entity_uuid, count in result.suppressed_by_decision.items():
        if count <= threshold:
            continue
        grounds = result.grounds_by_decision.get(entity_uuid, 'unknown')
        summary = (
            f'Standing decision for entity {entity_uuid} suppressed {count} recon '
            f'flag(s) in a single cycle (> {threshold})'
        )
        detail = '\n'.join([
            f'project_id: {project_id}',
            f'run_id: {run_id}',
            f'entity_uuid: {entity_uuid}',
            f'grounds: {grounds}',
            f'suppressed_this_cycle: {count}',
            f'threshold: {threshold}',
        ])
        if file_or_fold_l1(
            escalation_queue,
            project_id=project_id,
            subject=entity_uuid,
            category=CATEGORY_STANDING_DECISION_STORM,
            finding_category=_STORM_FINDING_CATEGORY,
            agent_role=_AGENT_ROLE,
            summary=summary,
            detail=detail,
            log=logger,
            log_label=_LOG_LABEL,
        ):
            escalated.append(entity_uuid)
    return escalated


@dataclass(frozen=True)
class SuppressionStreakUpdate:
    """One standing decision's suppression streak after this cycle (task 2943).

    ``streak`` counts the consecutive full cycles, ending with this one, in
    which the decision on ``(entity_uuid, grounds)`` suppressed at least one
    flag; 0 records a reset.  ``window_suppressed`` is the number of flags it
    suppressed across the last K cycles of that streak, counting only cycles
    at or under the per-cycle N.  ``escalate`` is ``streak >= threshold and
    window_suppressed > volume_threshold``.
    """

    entity_uuid: str
    grounds: str
    streak: int
    window_suppressed: int
    escalate: bool


def _advanced_streak(
    prior: SuppressionStreakRow | None, count: int, run_id: str, window_cycles: int
) -> tuple[int, tuple[int, ...]]:
    """The ``(streak, window)`` of a decision that suppressed *count* flags this cycle.

    It extends *prior*'s streak by one and appends *count* to its window,
    keeping the last *window_cycles* counts.  When *prior* was last written by
    this *run_id* (a replayed cycle) both are held as stored.
    """
    if prior is None:
        return 1, (count,)
    if run_id and prior.last_run_id == run_id:
        return prior.streak, prior.recent_counts
    return prior.streak + 1, (*prior.recent_counts, count)[-window_cycles:]


def _next_suppression_streaks(
    stored: dict[tuple[str, str], SuppressionStreakRow],
    result: EntityStandingSuppressionResult,
    run_id: str,
    window_cycles: int,
) -> dict[tuple[str, str], tuple[int, tuple[int, ...]]]:
    """The ``(entity_uuid, grounds) -> (streak, window)`` rows this cycle must write.

    Every suppressing decision advances (:func:`_advanced_streak`).  A stored
    non-zero streak whose decision suppressed nothing resets to ``(0, ())``;
    a stored 0 needs no write.  Pure, sync, no I/O.
    """
    next_rows: dict[tuple[str, str], tuple[int, tuple[int, ...]]] = {}
    for entity_uuid, count in result.suppressed_by_decision.items():
        key = (entity_uuid, result.grounds_by_decision.get(entity_uuid, ''))
        next_rows[key] = _advanced_streak(stored.get(key), count, run_id, window_cycles)
    for key, prior in stored.items():
        if key not in next_rows and prior.streak:
            next_rows[key] = (0, ())
    return next_rows


def _utc_now(now: str | None) -> datetime:
    """*now* as a UTC ``datetime``, or the current time when it is ``None``.

    Raises ``ValueError`` naming *now* when it is not ISO-8601 or carries no
    UTC offset.
    """
    if now is None:
        return datetime.now(UTC)
    try:
        parsed = datetime.fromisoformat(now)
    except ValueError as exc:
        raise ValueError(f'now must be an ISO-8601 timestamp (got {now!r})') from exc
    if parsed.tzinfo is None:
        raise ValueError(f'now must carry a UTC offset (got {now!r})')
    return parsed.astimezone(UTC)


async def update_suppression_streaks(
    memory_service: Any,
    project_id: str,
    run_id: str,
    result: EntityStandingSuppressionResult,
    *,
    now: str | None = None,
    threshold: int = SUPPRESSION_STREAK_THRESHOLD_CYCLES,
    volume_threshold: int = SUPPRESSION_STREAK_VOLUME_THRESHOLD,
    per_cycle_threshold: int = SUPPRESSION_STORM_THRESHOLD_PER_CYCLE,
) -> list[SuppressionStreakUpdate]:
    """Advance every standing decision's cross-cycle suppression streak by one cycle.

    The persistence half of the storm escape's streak arm; filing is
    :func:`maybe_escalate_suppression_streak`.  Each decision in
    ``result.suppressed_by_decision`` extends its streak by one and records
    this cycle's suppression count in a window of its last *threshold* cycles.
    Every stored non-zero streak whose decision suppressed nothing this cycle
    resets to 0 with an empty window.  A row already at 0 is left alone, so a
    decision that has gone quiet is written once and then ages out through
    ``gc()``'s ``expires_at`` arm.

    A decision escalates when its streak has reached *threshold* (K) and the
    flags it suppressed across that window total more than *volume_threshold*
    (N).  The window is bounded because a decision that works suppresses its
    re-derived complaint about once per cycle: its window then totals K <= N,
    so it never files however long its streak runs.  A cycle over
    *per_cycle_threshold* stays in the window but is left out of the total:
    the per-cycle escape already reported it, and one flood must not page both
    escapes in the same cycle.  That only holds while *per_cycle_threshold* is
    the *threshold* :func:`maybe_escalate_suppression_storm` runs with, so a
    caller tuning one passes the same value to both.  The verdict is
    re-evaluated every cycle, so a decision that drops back to its
    steady-state rate stops escalating without a reset.

    Replaying a cycle is idempotent: a row whose stored ``last_run_id`` equals
    *run_id* is re-written without incrementing, so a re-entered stage cannot
    reach *threshold* before the drain has persisted for that many real cycles.

    Best-effort throughout, and an unread ledger is never read as a quiet
    cycle.  A *result* with ``suppression_evaluated`` False returns ``[]``
    before any I/O: Hook A fell open, so its empty ``suppressed_by_decision``
    is not evidence that any decision went quiet.  No ``recon_ledger`` returns
    ``[]`` (logged DEBUG).  A failed streak read returns ``[]`` and writes
    NOTHING (logged WARNING), because without the prior counts every write
    would clobber an established streak.  A failed write costs that one
    entity its update (logged WARNING, excluded from the return), not the
    cycle.  The ledger decodes a malformed stored row as streak 0 or an empty
    window (``ReconLedgerStore.list_suppression_streaks``).  A lost window can
    only under-count, so it may delay a filing by up to K cycles but never
    cause one.

    Every write refreshes ``expires_at`` to *now* plus
    ``STANDING_DECISION_TTL_DAYS``, stored in UTC and spelled as
    ``isoformat()`` like the standing-decision writer, so ``gc()``'s TEXT
    comparison stays valid.  *now* is a timezone-aware ISO-8601 string and
    defaults to the current time.  A malformed or offset-less *now* raises
    ``ValueError`` before any I/O: it is a caller's programming error, which
    the best-effort contract does not cover.

    Returns one :class:`SuppressionStreakUpdate` per row written, sorted by
    ``(entity_uuid, grounds)``.
    """
    now_dt = _utc_now(now)
    if not result.suppression_evaluated:
        return []

    ledger = getattr(memory_service, 'recon_ledger', None)
    if ledger is None:
        logger.debug(
            'update_suppression_streaks: no recon_ledger on memory_service for '
            'project %s; suppression streaks not advanced this cycle',
            project_id,
        )
        return []

    try:
        rows = await ledger.list_suppression_streaks(project_id)
    except Exception as e:
        logger.warning(
            'update_suppression_streaks: recon_ledger.list_suppression_streaks '
            'failed for project %s: %s (best-effort — writing no streak this '
            'cycle so established streaks survive intact)',
            project_id,
            e,
            exc_info=True,
        )
        return []
    stored = {(row.entity_uuid.lower(), row.grounds): row for row in rows}

    expires_at = (now_dt + timedelta(days=STANDING_DECISION_TTL_DAYS)).isoformat()
    updates: list[SuppressionStreakUpdate] = []
    for (entity_uuid, grounds), (streak, window) in sorted(
        _next_suppression_streaks(stored, result, run_id, window_cycles=threshold).items()
    ):
        try:
            await ledger.upsert_suppression_streak(
                project_id=project_id,
                entity_uuid=entity_uuid,
                grounds=grounds,
                streak=streak,
                recent_counts=window,
                last_run_id=run_id,
                updated_at=now_dt.isoformat(),
                expires_at=expires_at,
            )
        except Exception as e:
            logger.warning(
                'update_suppression_streaks: failed to write streak=%d for '
                'entity_uuid=%s grounds=%s in project %s: %s',
                streak,
                entity_uuid,
                grounds,
                project_id,
                e,
                exc_info=True,
            )
            continue
        window_suppressed = sum(count for count in window if count <= per_cycle_threshold)
        updates.append(
            SuppressionStreakUpdate(
                entity_uuid=entity_uuid,
                grounds=grounds,
                streak=streak,
                window_suppressed=window_suppressed,
                escalate=streak >= threshold and window_suppressed > volume_threshold,
            )
        )
    return updates


async def maybe_escalate_suppression_streak(
    escalation_queue: Any,
    project_id: str,
    run_id: str,
    updates: list[SuppressionStreakUpdate],
    *,
    threshold: int = SUPPRESSION_STREAK_THRESHOLD_CYCLES,
    volume_threshold: int = SUPPRESSION_STREAK_VOLUME_THRESHOLD,
) -> list[str]:
    """File (or fold) a storm-escape L1 per standing decision whose streak of at
    least K cycles suppressed more than N flags across its last K.

    The filing half of the storm escape's streak arm (task 2943); the streak
    and its window are advanced by :func:`update_suppression_streaks`, whose
    ``escalate`` verdict selects which *updates* file.  *threshold* and
    *volume_threshold* are the K and N that verdict was measured against, named
    in the record.  Category, record shape, fold and best-effort contract are
    those of :func:`maybe_escalate_suppression_storm`; only the finding
    category (:data:`_STREAK_FINDING_CATEGORY`) differs, so the two escapes
    never fold into each other.

    It files on every cycle whose window volume stays above N, and each such
    filing folds onto the pending parent, incrementing its ``dedupe_count``.  A
    decision that falls back to its steady-state rate stops filing on the next
    cycle without any reset.  Escalating still does not reset the streak,
    because restarting the clock would hide a drain that is still running.  A
    resolved record therefore re-mints only while such a drain continues, the
    INV-4 behaviour the per-cycle sibling shares.  A folded record keeps the
    PARENT's summary, so the volume named there is the first filing's.

    Returns the entity_uuids that received a NEW record this cycle.
    """
    if file_or_fold_l1 is None:
        return []

    escalated: list[str] = []
    for update in updates:
        if not update.escalate:
            continue
        summary = (
            f'Standing decision for entity {update.entity_uuid} suppressed '
            f'{update.window_suppressed} recon flag(s) across its last {threshold} '
            f'consecutive cycles (> {volume_threshold})'
        )
        detail = '\n'.join([
            f'project_id: {project_id}',
            f'run_id: {run_id}',
            f'entity_uuid: {update.entity_uuid}',
            f'grounds: {update.grounds}',
            f'streak: {update.streak}',
            f'threshold: {threshold}',
            f'suppressed_in_window: {update.window_suppressed}',
            f'volume_threshold: {volume_threshold}',
        ])
        if file_or_fold_l1(
            escalation_queue,
            project_id=project_id,
            subject=update.entity_uuid,
            category=CATEGORY_STANDING_DECISION_STORM,
            finding_category=_STREAK_FINDING_CATEGORY,
            agent_role=_AGENT_ROLE,
            summary=summary,
            detail=detail,
            log=logger,
            log_label=_LOG_LABEL,
        ):
            escalated.append(update.entity_uuid)
    return escalated
