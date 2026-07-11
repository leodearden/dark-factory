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
from typing import Any

logger = logging.getLogger(__name__)


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
