"""Reconciliation freshness pre-check — task 2417.

Stage 1 (MemoryConsolidator) and Stage 2 (TaskKnowledgeSync) remediation
passes currently re-run a full LLM-based re-investigation over every
actionable finding, even when a finding is a *cross-project scope-correction*
thread whose underlying subject task hasn't moved since the last time it was
consolidated.  Incident: autopilot_video Stage-2 remediation re-investigated
the dark_factory:2405 ``done_provenance``-kind scope-correction thread via a
fresh full LLM pass, but every cited fact was unchanged from a consolidated
snapshot written earlier the same cycle-day.

This module provides a cheap, deterministic pre-check that runs BEFORE the
LLM stages are invoked (wired into
:meth:`ReconciliationHarness._run_remediation_pass`).  For each
cross-project scope-correction finding, it reads the most recent prior
freshness snapshot (a keyed Mem0 memory this module itself owns — see
:data:`CONSOLIDATED_SCOPE_KIND`), issues a single ``get_task`` call on the
finding's primary (usually foreign) subject task, and compares
``(status, updatedAt, description-fingerprint)``.  When all three are
unchanged, the finding is skipped from re-derivation and a lightweight
'still blocked, no change' marker is written instead; otherwise the finding
is kept for full re-investigation and the snapshot is (re)written to record
the subject's current state.

Fail-open throughout: an unknown foreign project, a ``get_task`` failure, a
Mem0 read failure, or any other unexpected error routes the finding back to
re-investigation.  A finding is only ever skipped on a POSITIVE freshness
confirmation — a false skip would silently drop a genuinely-changed thread
from remediation, which is far worse than a redundant LLM pass.

Single-subject caveat: that "positive freshness confirmation" is a
confirmation of the finding's PRIMARY subject only (:func:`select_primary_subject`
— the first foreign ``cited_tasks`` entry, or the first entry when none are
foreign).  A finding that cites MULTIPLE foreign tasks is skipped whenever
that one primary subject is unchanged, even if a different, non-primary
cited task has in fact changed.  This is a deliberate scope decision (one
``get_task`` call per finding, per the task 2417 plan's design decisions),
not a claim that every cited fact was re-verified.  Widening this pre-check
to check freshness across every cited task — at the cost of one
``get_task`` call per cited task instead of one per finding — is a
reasonable follow-up if multi-subject scope-correction findings prove
common in practice, but is out of scope here.

Consecutive-skip cap: a ``(task_ref, flag_key)`` pair is skipped at most
``max_consecutive_skips - 1`` cycles in a row (see
:data:`DEFAULT_MAX_CONSECUTIVE_SKIPS`); the cycle after that is always
forced back into ``to_reinvestigate`` — and the streak reset — even though
the subject is still unchanged.  Without this cap a genuinely-stranded
cross-project thread would be skipped by this pre-check forever, and since
``ReconciliationHarness``'s Stage-3 persistence-gated escalation
(``_INTEGRITY_FINDING_RECURRENCE_THRESHOLD``) only ever evaluates a finding
when the CURRENT remediation pass produces its own fresh ``integrity_check``
stage report, an indefinite skip would silently suppress escalation of a
stranded issue forever — a project "loud failure over silent degradation"
norm violation, not merely a missed optimization.

Mirrors :meth:`ReconciliationHarness._reconcile_status_correction` /
``_delete_status_correction_memories``'s read-compare-supersede,
add-then-delete pool-cap pattern, and reuses
:func:`fused_memory.reconciliation.flag_dedup._content_fingerprint` for a
deterministic description fingerprint (avoids storing full description text
in the snapshot payload).

This module has no imports from ``harness`` or ``stages/`` (other than the
pure ``flag_dedup`` fingerprint helper) — callers inject ``memory_service``,
``taskmaster``, and a ``resolve_project_root`` callable, keeping it
decoupled and unit-testable without a real harness.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, NamedTuple

from fused_memory.reconciliation.flag_dedup import _content_fingerprint

if TYPE_CHECKING:
    from fused_memory.backends.task_backend_protocol import TaskBackendProtocol
    from fused_memory.services.memory_service import MemoryService

logger = logging.getLogger(__name__)

CONSOLIDATED_SCOPE_KIND: str = 'consolidated_scope_correction'
"""Mem0 metadata ``kind`` tag for a scope-correction freshness snapshot."""

SCOPE_FRESHNESS_SOURCE: str = 'stage_scope_freshness'
"""``_source`` audit tag used for every scope_freshness memory write."""

DEFAULT_MAX_CONSECUTIVE_SKIPS: int = 4
"""Default cap on consecutive freshness-confirmed skips for one
``(task_ref, flag_key)`` pair before :func:`precheck_scope_correction_freshness`
forces a real re-investigation even though the subject is still unchanged —
see the module docstring's "Consecutive-skip cap" section (task 2417
amendment: reviewer finding ``robustness_silent_degradation``).

Set to match ``ReconciliationHarness._INTEGRITY_FINDING_RECURRENCE_THRESHOLD``'s
current value as a plain literal — NOT imported, since this module
intentionally has no ``harness``/``stages`` import (see module docstring).
Callers that care about staying in lockstep with the live threshold (e.g.
the harness itself) should pass ``max_consecutive_skips`` explicitly rather
than relying on this default.
"""


def is_cross_project_scope_correction(finding: dict[str, Any] | None, project_id: str) -> bool:
    """Return True iff *finding* is a cross-project scope-correction thread.

    True iff BOTH:
      - ``finding.get('flag_type') == 'cross_project'`` OR
        ``finding.get('category') == 'cross_project_routing'``, AND
      - at least one ``cited_tasks`` entry is a dict with a truthy
        ``'project_id'`` that differs from the running *project_id*.

    Tolerates a non-dict *finding*, and a missing/None/non-list
    ``cited_tasks`` (or non-dict entries within it) — returns False rather
    than raising.  Pure, sync, no I/O.
    """
    if not isinstance(finding, dict):
        return False

    flag_type = finding.get('flag_type')
    category = finding.get('category')
    if flag_type != 'cross_project' and category != 'cross_project_routing':
        return False

    cited_tasks = finding.get('cited_tasks')
    if not isinstance(cited_tasks, list):
        return False

    for cited in cited_tasks:
        if not isinstance(cited, dict):
            continue
        cited_project_id = cited.get('project_id')
        if cited_project_id and cited_project_id != project_id:
            return True

    return False


def select_primary_subject(
    finding: dict[str, Any], project_id: str,
) -> tuple[str, str] | None:
    """Pick the finding's primary subject task as ``(subject_project_id, subject_task_id)``.

    Prefers the FIRST ``cited_tasks`` entry whose ``project_id`` differs from
    the running *project_id* (the foreign subject a scope-correction finding
    is usually about).  Falls back to the first structurally-valid entry
    (truthy ``project_id`` + non-None ``task_id``) when none are foreign.
    Returns None when ``cited_tasks`` is missing/empty, or contains no
    structurally-valid entry.  ``task_id`` is coerced to ``str``.  Pure,
    sync, no I/O.

    This single subject is the ONLY one freshness is ever tracked against —
    see the module docstring's "Single-subject caveat" for what that means
    for a finding citing more than one foreign task.
    """
    if not isinstance(finding, dict):
        return None
    cited_tasks = finding.get('cited_tasks')
    if not isinstance(cited_tasks, list) or not cited_tasks:
        return None

    fallback: tuple[str, str] | None = None
    for cited in cited_tasks:
        if not isinstance(cited, dict):
            continue
        cited_project_id = cited.get('project_id')
        cited_task_id = cited.get('task_id')
        if not cited_project_id or cited_task_id is None:
            continue
        if fallback is None:
            fallback = (str(cited_project_id), str(cited_task_id))
        if cited_project_id != project_id:
            return (str(cited_project_id), str(cited_task_id))

    return fallback


def compute_scope_signature(finding: dict[str, Any], project_id: str) -> tuple[str, str] | None:
    """Derive the freshness-snapshot key ``(task_ref, flag_key)`` for *finding*.

    ``task_ref`` is the project-qualified subject reference
    (``f'{subject_project_id}:{subject_task_id}'``) from
    :func:`select_primary_subject`; ``flag_key`` is
    ``finding['flag_type']`` or ``finding['category']`` or ``''``.  Returns
    None when :func:`select_primary_subject` finds no usable subject.  Pure,
    sync, no I/O.
    """
    subject = select_primary_subject(finding, project_id)
    if subject is None:
        return None
    subject_project_id, subject_task_id = subject
    task_ref = f'{subject_project_id}:{subject_task_id}'
    flag_key = str(finding.get('flag_type') or finding.get('category') or '')
    return (task_ref, flag_key)


def build_scope_snapshot_metadata(
    *,
    task_ref: str,
    flag_key: str,
    subject_project_id: str,
    subject_task_id: str,
    status: str | None,
    updated_at: str | None,
    description: str | None,
    run_id: str,
    snapshot_at: str,
    no_change: bool = False,
    skip_streak: int = 0,
) -> dict[str, Any]:
    """Build the canonical scope-freshness-snapshot Mem0 metadata payload.

    Keyed by ``task_id=task_ref`` (project-qualified subject reference, e.g.
    ``'dark_factory:2405'``) + ``flag_type=flag_key`` — the pair
    :func:`precheck_scope_correction_freshness` queries back on the next
    cycle via ``get_memories_by_metadata``.  ``subject_description_fingerprint``
    reuses :func:`fused_memory.reconciliation.flag_dedup._content_fingerprint`
    (deterministic SHA-256 of the normalized description) rather than storing
    the full description text.

    ``no_change`` is only included (as ``True``) when explicitly requested —
    set on the lightweight 'still blocked, no change' marker written when a
    finding is skipped; omitted (not merely ``False``) on a snapshot
    recording a first-sight or changed subject.  ``skip_streak`` (always
    included, default 0) is the number of CONSECUTIVE freshness-confirmed
    skips this snapshot represents — 0 on a first-sight/changed/forced
    snapshot, incremented by the caller on each consecutive 'no change'
    marker; see the module docstring's "Consecutive-skip cap".  Pure, sync,
    no I/O.
    """
    metadata: dict[str, Any] = {
        'kind': CONSOLIDATED_SCOPE_KIND,
        'source': SCOPE_FRESHNESS_SOURCE,
        'task_id': task_ref,
        'flag_type': flag_key,
        'subject_project_id': subject_project_id,
        'subject_task_id': subject_task_id,
        'subject_status': status,
        'subject_updated_at': updated_at,
        'subject_description_fingerprint': _content_fingerprint(description or ''),
        'run_id': run_id,
        'snapshot_at': snapshot_at,
        'skip_streak': skip_streak,
    }
    if no_change:
        metadata['no_change'] = True
    return metadata


def _extract_task_fields(resp: Any) -> tuple[str | None, str | None, str | None, dict]:
    """Normalize a taskmaster ``get_task`` response to its four freshness fields.

    Mirrors :func:`fused_memory.reconciliation.targeted._extract_task`'s
    ``'data'``-envelope unwrap: accepts either the flat sqlite-backend shape
    (``{'status', 'updatedAt', 'description', 'metadata', ...}``) or a
    ``{'data': {...}}``-enveloped shape (live-Taskmaster / MCP-proxied).
    Tolerates a non-dict *resp* (or non-dict ``metadata``) by returning
    ``(None, None, None, {})`` / ``metadata={}`` rather than raising.  Pure,
    sync, no I/O.

    Returns:
        ``(status, updated_at, description, metadata)``.
    """
    task = resp
    if isinstance(resp, dict) and isinstance(resp.get('data'), dict):
        task = resp['data']
    if not isinstance(task, dict):
        return (None, None, None, {})

    metadata = task.get('metadata')
    if not isinstance(metadata, dict):
        metadata = {}
    return (task.get('status'), task.get('updatedAt'), task.get('description'), metadata)


def snapshot_is_fresh(snapshot_metadata: dict[str, Any], live_task: Any) -> bool:
    """Return True iff *live_task* is unchanged from *snapshot_metadata*.

    Requires ALL THREE of live ``status``, ``updatedAt``, and the
    content-fingerprint of ``description`` (via
    :func:`fused_memory.reconciliation.flag_dedup._content_fingerprint`) to
    equal the snapshot's ``subject_status``, ``subject_updated_at``, and
    ``subject_description_fingerprint`` respectively.  ``updatedAt`` is the
    load-bearing signal — sqlite_task_backend always advances it on any
    write, even a no-op — the other two are corroborating.

    Fail-safe: returns False (not fresh — i.e. keep for re-investigation)
    when *snapshot_metadata* is not a dict, or is missing any of the three
    ``subject_*`` keys it needs to compare against.  Pure, sync, no I/O.
    """
    if not isinstance(snapshot_metadata, dict):
        return False
    required_keys = (
        'subject_status', 'subject_updated_at', 'subject_description_fingerprint',
    )
    if any(key not in snapshot_metadata for key in required_keys):
        return False

    status, updated_at, description, _metadata = _extract_task_fields(live_task)
    if status != snapshot_metadata['subject_status']:
        return False
    if updated_at != snapshot_metadata['subject_updated_at']:
        return False
    return _content_fingerprint(description or '') == snapshot_metadata['subject_description_fingerprint']


async def _pool_cap_scope_snapshots(
    memory_service: MemoryService,
    prior_memories: list[dict[str, Any]],
    project_id: str,
    task_ref: str,
) -> None:
    """Best-effort delete every memory in *prior_memories*.

    Called immediately after a fresh snapshot/marker has already been
    written for the same ``(task_ref, flag_key)``, so the pool is bounded
    back down to that single just-written memory regardless of delete
    outcomes here — mirrors
    :meth:`ReconciliationHarness._delete_status_correction_memories`'s
    add-then-delete, per-item-exception-swallowing pattern. A no-op when
    *prior_memories* is empty (bootstrap: nothing to delete). Never raises.
    """
    for prior in prior_memories:
        try:
            await memory_service.delete_memory(
                memory_id=prior['id'], store='mem0', project_id=project_id,
            )
        except Exception as exc:
            logger.warning(
                'reconciliation.scope_freshness_delete_failed',
                extra={
                    'project_id': project_id,
                    'task_ref': task_ref,
                    'memory_id': prior.get('id'),
                    'error': str(exc),
                },
            )


class ScopeFreshnessResult(NamedTuple):
    """Result of :func:`precheck_scope_correction_freshness`.

    Attributes:
        to_reinvestigate: Findings to feed into the LLM stages — every
            non-scope-correction finding, every scope-correction finding with
            no usable/resolvable subject, every scope-correction finding
            whose PRIMARY subject (see :func:`select_primary_subject` and
            the module docstring's "Single-subject caveat") is new or has
            changed since the last snapshot, AND every scope-correction
            finding forced back for re-investigation because its
            consecutive freshness-confirmed-skip streak reached
            ``max_consecutive_skips`` (see "Consecutive-skip cap").
            Order-preserving relative to the input ``findings`` list.
        skipped: Cross-project scope-correction findings whose PRIMARY
            subject was confirmed unchanged since the last consolidated
            snapshot, and whose consecutive-skip streak was still under
            ``max_consecutive_skips`` — dropped from this cycle's LLM
            re-derivation.
        stats: Counters for logging/observability —
            ``scope_freshness_candidates`` (cross-project scope-correction
            findings with a usable subject and a resolvable project root —
            i.e. ones this pre-check actually attempted a live comparison
            for), ``scope_freshness_reinvestigated`` (every finding kept in
            ``to_reinvestigate`` — scope-correction candidates AND
            plain pass-through findings alike: non-scope-correction, no
            usable subject/signature, unresolvable root, per-finding error,
            genuinely changed, and cap-forced), ``scope_freshness_skipped``,
            and ``scope_freshness_forced_reinvestigation`` (the subset of
            ``scope_freshness_reinvestigated`` that was forced back by the
            consecutive-skip cap rather than genuinely new/changed).

            Two exact identities hold ALWAYS (task 2417 amendment — reviewer
            finding observability: a prior version of this pre-check left
            two kept-finding branches uncounted, so ``reinvestigated`` could
            be less than ``len(to_reinvestigate)``):
            ``scope_freshness_reinvestigated == len(to_reinvestigate)`` and
            ``scope_freshness_skipped == len(skipped)``.
            ``scope_freshness_candidates`` is a subset count — only
            scope-correction findings with a resolvable root are
            "candidates" — so it relates to the other two only as an
            inequality: every skipped finding is necessarily a candidate,
            but not every candidate that was kept is a non-candidate, so
            ``scope_freshness_candidates <= scope_freshness_reinvestigated +
            scope_freshness_skipped``, not equality.
    """

    to_reinvestigate: list[dict[str, Any]]
    skipped: list[dict[str, Any]]
    stats: dict[str, int]


async def precheck_scope_correction_freshness(
    *,
    memory_service: MemoryService,
    taskmaster: TaskBackendProtocol,
    project_id: str,
    resolve_project_root: Callable[[str], str | None],
    run_id: str,
    findings: list[dict[str, Any]],
    max_consecutive_skips: int = DEFAULT_MAX_CONSECUTIVE_SKIPS,
) -> ScopeFreshnessResult:
    """Filter *findings*, skipping re-derivation of unchanged scope-correction threads.

    For each finding: non-scope-correction findings pass through untouched.
    Cross-project scope-correction findings (see
    :func:`is_cross_project_scope_correction`) with a usable subject are
    checked against the most recent prior freshness snapshot for
    ``(task_ref, flag_key)``; a single ``taskmaster.get_task`` call reads the
    subject's live state.  When unchanged since the snapshot (see
    :func:`snapshot_is_fresh`) AND the pair's consecutive-skip streak is
    still under *max_consecutive_skips*, the finding is skipped and a
    lightweight 'still blocked, no change' marker is written (recording the
    bumped streak); otherwise — first sight, the subject changed, OR the
    streak just reached *max_consecutive_skips* — the finding is kept and
    the snapshot is (re)written to record the subject's current state with
    the streak reset to 0.  See the module docstring's "Consecutive-skip
    cap" and "Single-subject caveat" sections.

    Fail-open: an unresolvable foreign project root, a ``get_task`` failure,
    a Mem0 read/write failure, or any other unexpected per-finding error
    keeps that finding in ``to_reinvestigate`` rather than raising — see the
    module docstring.  A catastrophic failure outside the per-finding guard
    (e.g. a bug in the pure helpers) falls all the way back to treating
    EVERY input finding as needing re-investigation.  Never raises.
    """
    safe_findings = list(findings) if isinstance(findings, list) else []
    to_reinvestigate: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    stats: dict[str, int] = {
        'scope_freshness_candidates': 0,
        'scope_freshness_reinvestigated': 0,
        'scope_freshness_skipped': 0,
        'scope_freshness_forced_reinvestigation': 0,
    }

    try:
        for finding in safe_findings:
            if not is_cross_project_scope_correction(finding, project_id):
                to_reinvestigate.append(finding)
                # task 2417 amendment — reviewer finding observability: every
                # finding kept in to_reinvestigate increments this counter so
                # scope_freshness_reinvestigated == len(to_reinvestigate)
                # always (not just for scope-correction candidates), matching
                # the ScopeFreshnessResult stats docstring.
                stats['scope_freshness_reinvestigated'] += 1
                continue

            signature = compute_scope_signature(finding, project_id)
            subject = select_primary_subject(finding, project_id)
            if signature is None or subject is None:
                to_reinvestigate.append(finding)
                stats['scope_freshness_reinvestigated'] += 1
                continue
            task_ref, flag_key = signature
            subject_project_id, subject_task_id = subject

            try:
                subject_root = resolve_project_root(subject_project_id)
                if not subject_root:
                    to_reinvestigate.append(finding)
                    stats['scope_freshness_reinvestigated'] += 1
                    continue
                # A "candidate" (per ScopeFreshnessResult.stats docstring) is
                # a finding with BOTH a usable subject AND a resolvable
                # project root — i.e. one we actually attempt a live
                # get_task/snapshot comparison for.  Counted here, after the
                # resolvable-root guard above, not before it (task 2417
                # amendment — reviewer finding observability_stats_inconsistency:
                # counting an unresolvable-root finding as a candidate
                # contradicted this docstring).
                stats['scope_freshness_candidates'] += 1

                prior_memories = await memory_service.get_memories_by_metadata(
                    project_id=project_id,
                    filters={
                        'kind': CONSOLIDATED_SCOPE_KIND,
                        'task_id': task_ref,
                        'flag_type': flag_key,
                    },
                )
                live_task = await taskmaster.get_task(
                    task_id=subject_task_id, project_root=subject_root,
                )
                status, updated_at, description, _live_metadata = _extract_task_fields(
                    live_task,
                )
                snapshot_at = datetime.now(UTC).isoformat()

                if prior_memories:
                    for _prior in prior_memories:
                        _created = _prior.get('created_at')
                        if _created is not None and not isinstance(_created, str):
                            # Loud, not silent: a non-str created_at still sorts
                            # (via the str() coercion below) so this can never
                            # crash the precheck, but a Mem0 backend shape drift
                            # here would otherwise permanently and silently
                            # defeat correct latest-snapshot selection every
                            # cycle (task 2417 amendment — reviewer finding
                            # robustness: max() sort-key type assumption).
                            logger.warning(
                                'reconciliation.scope_freshness_created_at_type_drift',
                                extra={
                                    'project_id': project_id,
                                    'task_ref': task_ref,
                                    'memory_id': _prior.get('id'),
                                    'created_at_type': type(_created).__name__,
                                },
                            )
                    latest_prior = max(prior_memories, key=lambda m: str(m.get('created_at') or ''))
                else:
                    latest_prior = None

                if latest_prior is not None and snapshot_is_fresh(
                    latest_prior.get('metadata') or {}, live_task,
                ):
                    prior_metadata = latest_prior.get('metadata') or {}
                    prior_skip_streak = prior_metadata.get('skip_streak')
                    if not isinstance(prior_skip_streak, int):
                        prior_skip_streak = 0
                    new_skip_streak = prior_skip_streak + 1

                    if new_skip_streak < max_consecutive_skips:
                        # Unchanged since the last snapshot, and still under
                        # the consecutive-skip cap: skip re-derivation and
                        # write a lightweight 'still blocked, no change'
                        # marker instead, recording the bumped streak.
                        no_change_metadata = build_scope_snapshot_metadata(
                            task_ref=task_ref, flag_key=flag_key,
                            subject_project_id=subject_project_id,
                            subject_task_id=subject_task_id,
                            status=status, updated_at=updated_at, description=description,
                            run_id=run_id, snapshot_at=snapshot_at, no_change=True,
                            skip_streak=new_skip_streak,
                        )
                        await memory_service.add_memory(
                            content=(
                                f'Scope-correction freshness check for {task_ref} '
                                f'(flag_type={flag_key!r}): still blocked, no change.'
                            ),
                            category='observations_and_summaries',
                            project_id=project_id,
                            metadata=no_change_metadata,
                            _source=SCOPE_FRESHNESS_SOURCE,
                        )
                        await _pool_cap_scope_snapshots(
                            memory_service, prior_memories, project_id, task_ref,
                        )
                        skipped.append(finding)
                        stats['scope_freshness_skipped'] += 1
                        logger.info(
                            'reconciliation.scope_freshness_skipped',
                            extra={
                                'project_id': project_id, 'task_ref': task_ref,
                                'flag_type': flag_key, 'skip_streak': new_skip_streak,
                            },
                        )
                        continue

                    # Unchanged, but the consecutive-skip cap is reached:
                    # force a real re-investigation anyway so a
                    # genuinely-stranded thread can never be silently skipped
                    # past Stage 3's persistence-escalation window forever
                    # (task 2417 amendment — reviewer finding
                    # robustness_silent_degradation; see module docstring's
                    # "Consecutive-skip cap").  Falls through to the same
                    # snapshot-rewrite-and-keep code below as a genuinely
                    # changed subject, resetting skip_streak to 0.
                    stats['scope_freshness_forced_reinvestigation'] += 1
                    logger.info(
                        'reconciliation.scope_freshness_forced_reinvestigation',
                        extra={
                            'project_id': project_id, 'task_ref': task_ref,
                            'flag_type': flag_key, 'skip_streak': new_skip_streak,
                            'max_consecutive_skips': max_consecutive_skips,
                        },
                    )

                # First sight (no prior snapshot), the subject changed since
                # the last snapshot, or the consecutive-skip cap was just
                # reached: keep for re-investigation and (re)write the
                # snapshot recording the subject's current state (skip_streak
                # reset to 0) so the NEXT cycle has something to compare
                # against.
                fresh_metadata = build_scope_snapshot_metadata(
                    task_ref=task_ref, flag_key=flag_key,
                    subject_project_id=subject_project_id,
                    subject_task_id=subject_task_id,
                    status=status, updated_at=updated_at, description=description,
                    run_id=run_id, snapshot_at=snapshot_at,
                )
                await memory_service.add_memory(
                    content=(
                        f'Scope-correction freshness snapshot for {task_ref} '
                        f'(flag_type={flag_key!r}).'
                    ),
                    category='observations_and_summaries',
                    project_id=project_id,
                    metadata=fresh_metadata,
                    _source=SCOPE_FRESHNESS_SOURCE,
                )
                await _pool_cap_scope_snapshots(memory_service, prior_memories, project_id, task_ref)
                to_reinvestigate.append(finding)
                stats['scope_freshness_reinvestigated'] += 1
            except Exception as exc:
                # Fail-open: an unresolvable root, a get_task/Mem0 read
                # failure, or any other per-finding surprise must never
                # drop — let alone silently skip — a genuinely-changed
                # thread. Keep it for full re-investigation instead.
                logger.warning(
                    'reconciliation.scope_freshness_precheck_failed',
                    extra={
                        'project_id': project_id,
                        'task_ref': task_ref,
                        'flag_type': flag_key,
                        'error': str(exc),
                    },
                )
                # Every finding must land in exactly one of
                # to_reinvestigate/skipped AND increment the matching stat,
                # so scope_freshness_reinvestigated == len(to_reinvestigate)
                # holds even on a failure path (task 2417 amendment —
                # reviewer finding observability_stats_inconsistency: this
                # branch previously appended to to_reinvestigate without
                # incrementing any counter).
                stats['scope_freshness_reinvestigated'] += 1
                to_reinvestigate.append(finding)
    except Exception as exc:
        # Catastrophic failure outside the per-finding guard (e.g. a pure
        # helper misbehaving) — fail open for the WHOLE batch rather than
        # risk returning a partial/inconsistent result.
        logger.warning(
            'reconciliation.scope_freshness_precheck_catastrophic_failure',
            extra={'project_id': project_id, 'error': str(exc)},
        )
        return ScopeFreshnessResult(
            to_reinvestigate=list(safe_findings),
            skipped=[],
            stats={
                'scope_freshness_candidates': 0,
                'scope_freshness_reinvestigated': 0,
                'scope_freshness_skipped': 0,
                'scope_freshness_forced_reinvestigation': 0,
            },
        )

    return ScopeFreshnessResult(to_reinvestigate=to_reinvestigate, skipped=skipped, stats=stats)
