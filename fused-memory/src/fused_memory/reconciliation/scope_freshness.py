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


def is_cross_project_scope_correction(finding: dict[str, Any], project_id: str) -> bool:
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
    recording a first-sight or changed subject.  Pure, sync, no I/O.
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
    if _content_fingerprint(description or '') != snapshot_metadata['subject_description_fingerprint']:
        return False
    return True


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
            no usable/resolvable subject, and every scope-correction finding
            whose subject is new or has changed since the last snapshot.
            Order-preserving relative to the input ``findings`` list.
        skipped: Cross-project scope-correction findings confirmed unchanged
            since the last consolidated snapshot — dropped from this cycle's
            LLM re-derivation.
        stats: Counters for logging/observability —
            ``scope_freshness_candidates`` (cross-project scope-correction
            findings with a usable subject and a resolvable project root),
            ``scope_freshness_reinvestigated``, ``scope_freshness_skipped``.
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
) -> ScopeFreshnessResult:
    """Filter *findings*, skipping re-derivation of unchanged scope-correction threads.

    For each finding: non-scope-correction findings pass through untouched.
    Cross-project scope-correction findings (see
    :func:`is_cross_project_scope_correction`) with a usable subject are
    checked against the most recent prior freshness snapshot for
    ``(task_ref, flag_key)``; a single ``taskmaster.get_task`` call reads the
    subject's live state.  When unchanged since the snapshot (see
    :func:`snapshot_is_fresh`), the finding is skipped and a lightweight
    'still blocked, no change' marker is written; otherwise (first sight, or
    the subject changed) the finding is kept and the snapshot is (re)written
    to record the subject's current state.

    Fail-open: an unresolvable foreign project root, a ``get_task`` failure,
    a Mem0 read/write failure, or any other unexpected per-finding error
    keeps that finding in ``to_reinvestigate`` rather than raising — see the
    module docstring.  Never raises.
    """
    safe_findings = list(findings) if isinstance(findings, list) else []
    to_reinvestigate: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    stats: dict[str, int] = {
        'scope_freshness_candidates': 0,
        'scope_freshness_reinvestigated': 0,
        'scope_freshness_skipped': 0,
    }

    for finding in safe_findings:
        if not is_cross_project_scope_correction(finding, project_id):
            to_reinvestigate.append(finding)
            continue

        signature = compute_scope_signature(finding, project_id)
        subject = select_primary_subject(finding, project_id)
        if signature is None or subject is None:
            to_reinvestigate.append(finding)
            continue
        task_ref, flag_key = signature
        subject_project_id, subject_task_id = subject
        stats['scope_freshness_candidates'] += 1

        subject_root = resolve_project_root(subject_project_id)
        if not subject_root:
            to_reinvestigate.append(finding)
            stats['scope_freshness_reinvestigated'] += 1
            continue

        prior_memories = await memory_service.get_memories_by_metadata(
            project_id=project_id,
            filters={
                'kind': CONSOLIDATED_SCOPE_KIND,
                'task_id': task_ref,
                'flag_type': flag_key,
            },
        )
        live_task = await taskmaster.get_task(task_id=subject_task_id, project_root=subject_root)
        status, updated_at, description, _live_metadata = _extract_task_fields(live_task)
        snapshot_at = datetime.now(UTC).isoformat()

        latest_prior = (
            max(prior_memories, key=lambda m: m.get('created_at') or '')
            if prior_memories else None
        )

        if latest_prior is not None and snapshot_is_fresh(
            latest_prior.get('metadata') or {}, live_task,
        ):
            # Unchanged since the last snapshot: skip re-derivation and write
            # a lightweight 'still blocked, no change' marker instead.
            skipped.append(finding)
            stats['scope_freshness_skipped'] += 1
            no_change_metadata = build_scope_snapshot_metadata(
                task_ref=task_ref, flag_key=flag_key,
                subject_project_id=subject_project_id, subject_task_id=subject_task_id,
                status=status, updated_at=updated_at, description=description,
                run_id=run_id, snapshot_at=snapshot_at, no_change=True,
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
            await _pool_cap_scope_snapshots(memory_service, prior_memories, project_id, task_ref)
            logger.info(
                'reconciliation.scope_freshness_skipped',
                extra={'project_id': project_id, 'task_ref': task_ref, 'flag_type': flag_key},
            )
            continue

        # First sight (no prior snapshot), or the subject changed since the
        # last snapshot: keep for re-investigation and (re)write the
        # snapshot recording the subject's current state so the NEXT cycle
        # has something to compare against.
        to_reinvestigate.append(finding)
        stats['scope_freshness_reinvestigated'] += 1
        fresh_metadata = build_scope_snapshot_metadata(
            task_ref=task_ref, flag_key=flag_key,
            subject_project_id=subject_project_id, subject_task_id=subject_task_id,
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

    return ScopeFreshnessResult(to_reinvestigate=to_reinvestigate, skipped=skipped, stats=stats)
