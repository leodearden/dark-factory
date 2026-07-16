#!/usr/bin/env python3
"""One-shot correction: reviewed disposition of the found_on_main provenance
audit backlog (task 2648 dry-run, 2026-07-16).

Motivation
----------
``audit_found_on_main_provenance.py`` (task 2645) swept every
``found_on_main``-provenance done task and classified it via a fixed
precedence ladder (see that module's docstring). Task 2648's dry-run against
the live backlog surfaced 66 found_on_main tasks: 16 ``ok``, and 50 flagged
(27 ``misattributed``, 21 ``unverifiable``, 1 ``reverted``, 1
``deliverable_absent``). Task 2500's hardening fix already closes the
*live* misattribution defect class going forward — this script is the
retrospective, reviewed cleanup of the pre-existing backlog.

Reviewed dispositions
----------------------
Two records got individual, human-reviewed treatment (see
``REOPEN_DISPOSITIONS`` / ``BENIGN_DISPOSITIONS`` below for the full
evidence text):

  - **Task 1175** (``reverted``) — a GENUINE false completion. Its declared
    deliverable is absent from main, and its cited commit is an empty
    "Merge task/1175 into main" marker that never carried it. Prior
    reconciliation reopens silently failed to persist the done -> pending
    flip (metadata.reopen_* fields were written but ``status`` never
    changed). This script reopens it, with a post-write read-back verify so
    a silent non-persist is caught loudly rather than repeating that
    history.
  - **Task 2273** (``deliverable_absent``) — a BENIGN false positive. It is
    a deterministic pure-gate task (``task_kind=deterministic``,
    ``always_escalates=true``) whose declared files are sibling-produced
    reference artifacts; its own ``done_provenance.note`` already documents
    that the live migration landed under sibling task 2456. No reopen —
    annotated as reviewed-benign.

Every other flagged task (the historical ``misattributed``/``unverifiable``
bulk, overwhelmingly the known-benign sibling-merge-citation pattern
predating task 2500's fix) is annotated ``presumed_benign_historical``,
carrying the audit's own classification reasons forward — a documented,
reviewed disposition rather than a silent skip. An ``ok`` verdict is left
completely untouched.

This script NEVER mutates ``done_provenance`` or ``files`` — every
annotation is a ``metadata_mode='merge'`` (the default) write of a single
``x_provenance_audit`` key, which preserves every sibling metadata field.

Usage
-----
  # Dry run (default): print JSON summary, touch nothing.
  python scripts/correct_found_on_main_backlog.py --project-root /path/to/project

  # Apply: reopen task 1175 (with persistence verification) and annotate
  # every other flagged task's metadata.x_provenance_audit.
  python scripts/correct_found_on_main_backlog.py --project-root /path/to/project --apply
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger('correct_found_on_main_backlog')

# ---------------------------------------------------------------------------
# Reviewed disposition data (task 2667 architect findings, spot-checked by
# the implementer against the live backlog at step-14) — see module
# docstring for the full rationale summary.
# ---------------------------------------------------------------------------

REOPEN_DISPOSITIONS: dict[str, str] = {
    '1175': (
        "Declared deliverable fused-memory/tests/test_delete_memory_truncated_uuid.py "
        'is absent from main (confirmed via git ls-files). Its cited commit 614137480e '
        'is itself the degenerate "Merge task/1175 into main" marker (self-citing, so '
        'not a misattribution) — an empty found_on_main landing that never carried the '
        'test. Prior reconciliation reopens (runs 8c949e50, dfb0eb95) wrote '
        'metadata.reopen_* fields but the top-level status silently failed to flip from '
        "done to pending — task 1175's own metadata.reopen_reason documents this. "
        'Reviewed disposition: genuine false completion, reopen to pending with a '
        'post-write read-back verify (task 2667 audit backlog, task 2648 dry-run '
        '2026-07-16).'
    ),
}

BENIGN_DISPOSITIONS: dict[str, str] = {
    '2273': (
        'Declared deliverable fused-memory/scripts/migrate_cross_graph_leak.py IS '
        'present on main (confirmed via git ls-files) — the deliverable_absent verdict '
        'is a false positive. This is a deterministic pure-gate task '
        '(task_kind=deterministic, always_escalates=true) whose declared files are '
        "sibling-produced reference artifacts, and whose own done_provenance.note "
        'already documents that the live migration landed under sibling task 2456. '
        'Reviewed disposition: benign — expected shape for a pure-gate task carrying '
        'reference files, not a genuine false completion (task 2667 audit backlog, '
        'task 2648 dry-run 2026-07-16).'
    ),
}

# Action / label constants — shared vocabulary between plan_corrections and
# apply_corrections, and between this module and its tests.
ACTION_REOPEN = 'reopen'
ACTION_ANNOTATE = 'annotate'

LABEL_REOPENED = 'reopened'
LABEL_REVIEWED_BENIGN = 'reviewed_benign'
LABEL_PRESUMED_BENIGN_HISTORICAL = 'presumed_benign_historical'


@dataclass
class Correction:
    """A single planned correction for one found_on_main-provenance task.

    ``action`` is either ``'reopen'`` (task 1175's regression-guarded
    done -> pending flip) or ``'annotate'`` (a non-destructive
    ``x_provenance_audit`` metadata merge-write).

    ``reasons`` always starts with the audit's own ``classify()`` reasons
    for this task (``report['tasks'][i]['reasons']``); when the task_id has
    an individually-reviewed disposition (present in ``REOPEN_DISPOSITIONS``
    or ``BENIGN_DISPOSITIONS``), that evidence string is appended as one
    more reason — so the persisted annotation documents both what the audit
    flagged AND why it was reviewed-dispositioned. For the unreviewed
    default-fallback case, ``reasons`` is exactly the audit's reasons list,
    untouched.

    ``reopen_reason`` is populated only for ``action == 'reopen'`` (the
    same evidence string as ``reasons[-1]`` in that case) — kept as an
    explicit field because the apply layer's reopen audit-trail annotation
    surfaces it under its own ``reopen_reason`` key.

    ``ref`` is the audited git ref this correction was planned against
    (threaded from ``report['ref']`` — ``apply_corrections`` never sees the
    report itself, only the already-planned corrections).
    """

    task_id: str
    action: str
    label: str
    ref: str
    reasons: list[str] = field(default_factory=list)
    reopen_reason: str | None = None


# ---------------------------------------------------------------------------
# Planning (pure)
# ---------------------------------------------------------------------------

def plan_corrections(report: dict[str, Any]) -> list[Correction]:
    """Route every non-``ok`` task in *report* to a reviewed :class:`Correction`.

    *report* is shaped like ``audit_found_on_main_provenance.build_audit_report``'s
    return value (``{'ref': ..., 'tasks': [{'task_id', 'verdict', 'commit',
    'commit_subject', 'reasons'}, ...], ...}``) — this function is pure and
    import-free; it never touches git or the task backend.

    Routing (first match wins):
      1. ``verdict == 'ok'`` — skipped, no Correction.
      2. ``task_id in REOPEN_DISPOSITIONS`` — ``action='reopen'``,
         ``label=LABEL_REOPENED``.
      3. ``task_id in BENIGN_DISPOSITIONS`` — ``action='annotate'``,
         ``label=LABEL_REVIEWED_BENIGN``.
      4. Everything else (every other non-``ok`` verdict, including
         ``unverifiable`` — which the audit tool's own ``--apply`` leaves
         alone, but task 2667 explicitly scopes into this backlog cleanup)
         — ``action='annotate'``, ``label=LABEL_PRESUMED_BENIGN_HISTORICAL``.
    """
    ref = report.get('ref', 'main')
    corrections: list[Correction] = []
    for detail in report.get('tasks', []):
        verdict = detail.get('verdict')
        if verdict == 'ok':
            continue
        task_id = str(detail['task_id'])
        audit_reasons = list(detail.get('reasons', []))

        if task_id in REOPEN_DISPOSITIONS:
            evidence = REOPEN_DISPOSITIONS[task_id]
            corrections.append(Correction(
                task_id=task_id, action=ACTION_REOPEN, label=LABEL_REOPENED, ref=ref,
                reasons=[*audit_reasons, evidence], reopen_reason=evidence,
            ))
            continue

        if task_id in BENIGN_DISPOSITIONS:
            evidence = BENIGN_DISPOSITIONS[task_id]
            corrections.append(Correction(
                task_id=task_id, action=ACTION_ANNOTATE, label=LABEL_REVIEWED_BENIGN,
                ref=ref, reasons=[*audit_reasons, evidence],
            ))
            continue

        corrections.append(Correction(
            task_id=task_id, action=ACTION_ANNOTATE,
            label=LABEL_PRESUMED_BENIGN_HISTORICAL, ref=ref, reasons=audit_reasons,
        ))

    return corrections


# ---------------------------------------------------------------------------
# Apply layer
# ---------------------------------------------------------------------------

async def apply_corrections(
    backend: Any,
    project_root: str,
    corrections: list[Correction],
    *,
    apply: bool,
    tag: str | None = None,
) -> dict[str, Any]:
    """Apply (or dry-run report) every planned *corrections* against *backend*.

    ``apply=False`` (the default, safe posture) performs ZERO backend
    calls — every reopen/annotate is purely reported under ``planned`` so a
    dry run can never mutate anything, mirroring every other
    ``fused-memory/scripts/`` remediation tool's dry-run contract.

    ``apply=True`` executes each correction with per-op isolation (added in
    later steps of this module's build-out) so one failure never aborts the
    batch.

    Returns a summary dict:
      - ``dry_run``: ``not apply``, echoed back.
      - ``planned``: ``[{'task_id', 'action', 'label'}, ...]`` — the full
        intended-action list, always populated (dry-run or apply alike).
      - ``reopened`` / ``annotated`` / ``errors``: running counters — all 0
        in dry-run, since nothing was actually attempted.
      - ``reopen_failed``: task ids whose reopen request did not persist
        (the task-1175-class silent-write-failure regression guard) —
        empty in dry-run.
      - ``needs_human_review``: task ids of every ``annotate`` correction,
        computed straight from the input plan — independent of
        apply/dry-run or of whether the write itself later succeeds.
    """
    planned = [
        {'task_id': c.task_id, 'action': c.action, 'label': c.label}
        for c in corrections
    ]
    needs_human_review = [c.task_id for c in corrections if c.action == ACTION_ANNOTATE]

    summary: dict[str, Any] = {
        'dry_run': not apply,
        'planned': planned,
        'reopened': 0,
        'annotated': 0,
        'errors': 0,
        'reopen_failed': [],
        'needs_human_review': needs_human_review,
    }

    if not apply:
        return summary

    for correction in corrections:
        if correction.action == ACTION_ANNOTATE:
            await _apply_annotate(backend, project_root, correction, tag)
            summary['annotated'] += 1
        elif correction.action == ACTION_REOPEN:
            reopened = await _apply_reopen(backend, project_root, correction, tag)
            if reopened:
                summary['reopened'] += 1
            else:
                # Loud, not silent: a reopen that did not verifiably
                # persist is recorded as a failure and counted as an
                # error, but the batch is NOT aborted — the next
                # correction still gets a chance to apply.
                summary['reopen_failed'].append(correction.task_id)
                summary['errors'] += 1
            # Per-op error isolation (try/except around each correction so
            # one raised exception never aborts the batch) lands in a
            # later step of this task's step sequence.

    return summary


def _annotation_metadata(correction: Correction, *, extra: dict[str, Any] | None = None) -> str:
    """Build the JSON-encoded ``{'x_provenance_audit': {...}}`` metadata patch.

    Always carries ``label``/``reasons``/``ref``/``audited_at``; *extra*
    (e.g. ``reopen_reason`` for the reopen audit trail) is merged in on top.
    Freshly stamps ``audited_at`` on every call — this is a write-time
    fact, never carried over from the Correction itself.
    """
    annotation: dict[str, Any] = {
        'label': correction.label,
        'reasons': correction.reasons,
        'ref': correction.ref,
        'audited_at': datetime.now(UTC).isoformat(),
    }
    if extra:
        annotation.update(extra)
    return json.dumps({'x_provenance_audit': annotation})


async def _apply_annotate(
    backend: Any, project_root: str, correction: Correction, tag: str | None,
) -> None:
    """Merge-write *correction*'s reviewed disposition into metadata.x_provenance_audit.

    Uses ``update_task``'s DEFAULT metadata_mode (``'merge'``) —
    ``metadata_mode`` is never passed, let alone set to ``'replace'`` — so
    ``done_provenance``/``files`` and every sibling metadata key survive
    untouched. A corrective annotation must never destroy the very
    provenance record it documents.
    """
    await backend.update_task(
        correction.task_id, project_root,
        metadata=_annotation_metadata(correction),
        tag=tag,
    )
    logger.info('Annotated task %s (label=%s)', correction.task_id, correction.label)


async def _apply_reopen(
    backend: Any, project_root: str, correction: Correction, tag: str | None,
) -> bool:
    """Flip *correction*'s task ``done -> pending`` and verify it persisted.

    Task 1175's own metadata documents that prior reconciliation reopens
    wrote ``metadata.reopen_*`` fields but the top-level ``status`` silently
    never flipped — so a bare ``set_task_status`` call is not trusted on its
    own. Success requires BOTH:

      1. ``set_task_status`` did not return the typed
         :class:`StatusWriteNotPersistedResult` failure DTO
         (``result.get('success') is False`` or
         ``result.get('error') == 'status_write_not_persisted'``).
      2. An independent ``get_task`` re-read confirms ``status == 'pending'``
         — a belt-and-suspenders check against exactly the class of silent
         non-persistence task 1175 suffered.

    Only on a verified success is the audit-trail annotation written (the
    same non-destructive ``x_provenance_audit`` merge path as
    :func:`_apply_annotate`, with ``reopen_reason`` merged in) and ``True``
    returned.

    On any not-persisted outcome — the not-persisted DTO OR a re-read that
    still shows a non-``'pending'`` status — this logs loudly at ERROR
    (never silently), skips the audit annotation entirely (a corrective
    annotation must not claim a reopen that didn't happen), and returns
    ``False`` without raising. The caller is responsible for
    ``reopen_failed``/``errors`` accounting and for continuing the batch.
    """
    result = await backend.set_task_status(correction.task_id, 'pending', project_root, tag)
    write_persisted = (
        result.get('success') is not False
        and result.get('error') != 'status_write_not_persisted'
    )

    fresh = await backend.get_task(correction.task_id, project_root, tag)
    reread_confirms_pending = fresh.get('status') == 'pending'

    if not (write_persisted and reread_confirms_pending):
        actual_status = result.get('actual_status', fresh.get('status'))
        logger.error(
            'Reopen for task %s did NOT persist (set_task_status '
            "reported error=%r, get_task re-read status=%r) - recording as "
            'reopen_failed_to_persist and skipping the audit annotation.',
            correction.task_id, result.get('error'), actual_status,
        )
        return False

    await backend.update_task(
        correction.task_id, project_root,
        metadata=_annotation_metadata(
            correction, extra={'reopen_reason': correction.reopen_reason},
        ),
        tag=tag,
    )
    logger.info('Reopened task %s (done -> pending)', correction.task_id)
    return True


# NOTE: the CLI entry point (main/_run, step 14) lands in a later commit of
# this same task's step sequence.
