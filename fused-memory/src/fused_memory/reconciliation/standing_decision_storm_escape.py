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

# Optional escalation dependency (mirrors stage1_stall_detector's pattern): the
# reconciliation package must import cleanly even where the escalation package
# is not installed.  Both filers no-op when Escalation is None.
#
# ONE combined block, deliberately (same reasoning as stage1_stall_detector's):
# all five names bind or fail together, so any ONE identity check suffices at
# RUNTIME.  Every name is still listed in the guards because only an identity
# check on the name itself narrows an optionally-imported symbol for the type
# checker.
try:
    from escalation.dedupe import (  # type: ignore[import-untyped]
        DedupeConfig,
        compute_content_fingerprint,
        content_fingerprint_key,
        submit_or_dedupe,
    )
    from escalation.models import Escalation  # type: ignore[import-untyped]
except ImportError:
    Escalation = None  # type: ignore[assignment,misc]
    DedupeConfig = None  # type: ignore[assignment,misc]
    compute_content_fingerprint = None  # type: ignore[assignment]
    content_fingerprint_key = None  # type: ignore[assignment]
    submit_or_dedupe = None  # type: ignore[assignment]

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


def _storm_dedupe_config() -> Any:
    """The fold config both storm-escape filers share, or ``None`` when the
    escalation package is unavailable.

    The window is UNBOUNDED so a decision that keeps tripping an escape for days
    still folds into its original parent.
    """
    # Any ONE identity check suffices at runtime (all five names bind or fail
    # together in the module's single import block); each is named so a test
    # that nulls one of them still disables filing.
    if (
        Escalation is None
        or DedupeConfig is None
        or compute_content_fingerprint is None
        or content_fingerprint_key is None
        or submit_or_dedupe is None
    ):
        return None
    return DedupeConfig(
        infra_dedupe_enabled=True,
        infra_dedupe_window_secs=float('inf'),
        infra_dedupe_categories=(CATEGORY_STANDING_DECISION_STORM,),
        key_fn=content_fingerprint_key,
    )


def _file_or_fold_storm(
    escalation_queue: Any,
    project_id: str,
    entity_uuid: str,
    finding_category: str,
    summary: str,
    detail: str,
    config: Any,
) -> bool:
    """Submit one storm-escape L1 for *entity_uuid*; return True iff a NEW record
    was minted.

    Folds on the ``(category, finding_category, project:entity)`` content
    fingerprint (why: :func:`maybe_escalate_suppression_storm`).  Everything
    that can fail is inside the try, so a fingerprint, id-gen, constructor,
    submit or fold failure costs this entity its filing and is logged WARNING —
    never the rest of the cycle.  A fold logs INFO, so the recurrence is
    visible in the log stream and not only as a counter on disk.

    ``preservation_specimen_guard.py::_file_or_fold`` is a second copy of this
    helper and of :func:`_storm_dedupe_config`; task 5466 owns moving both
    into ``escalation/dedupe.py``.
    """
    try:
        # Unreachable through the callers, which check the whole import block
        # via _storm_dedupe_config; restated so this helper enforces its own
        # precondition as a logged WARNING rather than an AttributeError.
        if Escalation is None or compute_content_fingerprint is None or submit_or_dedupe is None:
            raise RuntimeError('escalation package unavailable')
        fingerprint = compute_content_fingerprint(
            CATEGORY_STANDING_DECISION_STORM,
            finding_category,
            [f'{project_id}:{entity_uuid}'],
        )
        # Fail closed rather than file with a falsy key: find_dedupe_parent
        # short-circuits on one, so the record would silently become a second
        # visible pending record for this entity every cycle. Unreachable via
        # today's sha256 callee — this guards a future change to it.
        if not fingerprint:
            raise ValueError(
                f'empty dedupe_fingerprint for {finding_category} entity_uuid={entity_uuid}'
            )
        esc = Escalation(
            id=escalation_queue.make_id(entity_uuid),
            # The entity is the subject, so it occupies task_id: that is the
            # key get_by_task/has_open_l1 read a storm record back by, and it
            # is what makes the record greppable per entity. It is NOT the
            # fold key — dedupe_fingerprint below is — so the two cannot
            # drift apart the way a task_id-keyed guard once did.
            task_id=entity_uuid,
            agent_role='reconciliation-stage1',
            severity='blocking',
            category=CATEGORY_STANDING_DECISION_STORM,
            summary=summary,
            detail=detail,
            level=1,
            dedupe_fingerprint=fingerprint,
        )
        outcome = submit_or_dedupe(escalation_queue, esc, config)
    except Exception as exc:
        logger.warning(
            'standing-decision storm escape (%s): failed to escalate entity_uuid=%s '
            '(fingerprint, id-gen, construction, submit, or fold): %s',
            finding_category,
            entity_uuid,
            exc,
            extra={'project_id': project_id},
        )
        return False

    if outcome.get('status') == 'dedup_skipped':
        logger.info(
            'standing-decision storm escape (%s): entity_uuid=%s folded into '
            'parent_id=%s (child_id=%s) — the condition is recurring',
            finding_category,
            entity_uuid,
            outcome.get('parent_id'),
            outcome.get('child_id'),
            extra={'project_id': project_id},
        )
        return False
    # Tested with != rather than == 'queued' so observed_submit_response's
    # auto-resolved/dismissed branch (a record WAS minted) still counts.
    return True


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

    **Filed through** :func:`escalation.dedupe.submit_or_dedupe`, NOT gated on
    ``has_open_l1`` (task 3522). Stage 1 re-evaluates every cycle, so a decision
    that storms once tends to storm every cycle — exactly the recurring-detector
    shape for which the sibling gate-backlog path retired the ``has_open_l1``
    skip: that skip suppressed every cycle after the first, so ``dedupe_count``
    stayed pinned at 0 and the operator saw no difference between one storm and
    forty. Folding instead keeps ONE pending record per entity and increments
    ``dedupe_count`` on it, which is the steward's recurrence / triage-order
    signal. The fold key is the ``(category, finding_category, project:entity)``
    content fingerprint :func:`_file_or_fold_storm` stamps — deliberately NOT the count or run_id,
    which drift every cycle and would mint a fresh record per breach. The window
    is UNBOUNDED so a decision storming for days still folds into its original
    parent. Accepted cost, as on the gate-backlog path: a folded record keeps the
    PARENT's summary, so the count named there is the FIRST breach's, while
    ``dedupe_count`` carries how often it has recurred since.

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
    config = _storm_dedupe_config()
    if config is None:
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
        if _file_or_fold_storm(
            escalation_queue,
            project_id,
            entity_uuid,
            _STORM_FINDING_CATEGORY,
            summary,
            detail,
            config,
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


async def update_suppression_streaks(
    memory_service: Any,
    project_id: str,
    run_id: str,
    result: EntityStandingSuppressionResult,
    *,
    now: str | None = None,
    threshold: int = SUPPRESSION_STREAK_THRESHOLD_CYCLES,
    volume_threshold: int = SUPPRESSION_STREAK_VOLUME_THRESHOLD,
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
    so it never files however long its streak runs.  A cycle over the
    per-cycle N stays in the window but is left out of the total: the per-cycle
    escape already reported it, and one flood must not page both escapes in
    the same cycle.  The verdict is re-evaluated every cycle, so a decision
    that drops back to its steady-state rate stops escalating without a reset.

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
    ``STANDING_DECISION_TTL_DAYS``, spelled as ``isoformat()`` like the
    standing-decision writer so ``gc()``'s TEXT comparison stays valid.
    *now* is an ISO-8601 string defaulting to the current UTC time.

    Returns one :class:`SuppressionStreakUpdate` per row written, sorted by
    ``(entity_uuid, grounds)``.
    """
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

    now_dt = datetime.now(UTC) if now is None else datetime.fromisoformat(now)
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
        window_suppressed = sum(
            count for count in window if count <= SUPPRESSION_STORM_THRESHOLD_PER_CYCLE
        )
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
    config = _storm_dedupe_config()
    if config is None:
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
        if _file_or_fold_storm(
            escalation_queue,
            project_id,
            update.entity_uuid,
            _STREAK_FINDING_CATEGORY,
            summary,
            detail,
            config,
        ):
            escalated.append(update.entity_uuid)
    return escalated
