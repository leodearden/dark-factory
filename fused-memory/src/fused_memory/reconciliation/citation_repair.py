"""Cross-run citation repair for findings owned by a completed run (task 3065).

This module owns the **third half** of the invariant ``citation_verifier``
declares: *a cited memory id must resolve*. Its two halves both act on the
CURRENT run — ``verify_cited_memories`` drops phantoms from the in-flight
report, and the ``repoint_*`` helpers rewrite live task metadata before a
delete. Neither can touch a finding whose owning run already closed, and that
is exactly the case this module handles: re-point (or drop) a
confirmed-dangling citation on a finding filed by a **prior, already-completed**
run.

**Why the recon-report tools cannot serve this.** Every ``cite_*`` tool resolves
through ``ReconReportState._resolve_entry(run_id)``, which requires the run to
have a currently ACTIVE stage, and ``_resolve_finding`` keys strictly on the
caller's own ``run_id``. Relaxing either would not be enough: a closed run's
report state does not survive to be resolved against. ``recon_report_state_ttl_seconds``
defaults to 300s, ``ReconReportState.tick`` evicts each completed entry past the
TTL, and at run quiescence it calls ``ReconReportStore.delete_run`` — so both the
in-process entry and its shadow SQLite rows are gone minutes after the run ends.

The reconciliation journal's ``runs.stage_reports`` blob is therefore the ONLY
durable home of a closed run's findings, and it is what this module reads and
rewrites, through the journal's existing ``get_run`` / ``update_run_stage_reports``
accessors — no new table, no new SQL.

**The repair can only ever fix provenance, never rewrite a live claim.** The
victim citation must be CONFIRMED absent and the replacement must resolve; a
raised backend read is *unknown*, not *absent*, and never licenses a mutation.
Those gates are what keep this from being a provenance-falsification surface —
the worst it can do is retarget a claim that already had no backing. It reuses
``citation_verifier``'s lookup primitive and its found/None/raised branching so
the two halves cannot disagree about what a backend timeout means.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from fused_memory.models.reconciliation import StageReport

logger = logging.getLogger(__name__)

__all__ = ['build_citation_repair_record', 'repair_memory_citation']

# The provenance key appended to a repaired finding. Deliberate sibling of the
# ``citation_failures`` key ``verify_cited_memories`` writes: a reader of any
# finding sees both "this claim lost its backing" and "this claim's backing was
# re-pointed", in the same shape.
CITATION_REPAIRS_KEY = 'citation_repairs'

# The only store this path can corroborate against. ``get_memory_by_id`` is a
# Mem0/Qdrant point read, so a graphiti id would resolve to None and be
# false-flagged as dangling — the same hazard ``verify_cited_memories``
# documents as its reason for skipping non-mem0 entries.
SUPPORTED_STORE = 'mem0'

# --------------------------------------------------------------------------- #
# Structured error branches (INV-2: structured facts, never a bare failure)
# --------------------------------------------------------------------------- #

_ERR_INVALID_UUID_SHAPE: dict[str, str] = {
    'error': 'invalid_uuid_shape',
    'error_type': 'ReconReportInvalidUuid',
}

_ERR_UNSUPPORTED_STORE: dict[str, str] = {
    'error': 'unsupported_store',
    'error_type': 'ReconCitationUnsupportedStore',
}

_ERR_TARGET_RUN_NOT_FOUND: dict[str, str] = {
    'error': 'target_run_not_found',
    'error_type': 'ReconCitationTargetRunNotFound',
}

# Reuses the recon-report spelling on purpose: a consumer that already branches
# on ``finding_unknown`` from cite_memory sees one vocabulary, not two.
_ERR_FINDING_UNKNOWN: dict[str, str] = {
    'error': 'finding_unknown',
    'error_type': 'ReconReportFindingUnknown',
}

_ERR_CITATION_NOT_PRESENT: dict[str, str] = {
    'error': 'citation_not_present',
    'error_type': 'ReconCitationNotPresent',
}

_ERR_CITATION_NOT_DANGLING: dict[str, str] = {
    'error': 'citation_not_dangling',
    'error_type': 'ReconCitationNotDangling',
}

_ERR_REPLACEMENT_NOT_FOUND: dict[str, str] = {
    'error': 'replacement_not_found',
    'error_type': 'ReconCitationReplacementNotFound',
}

_ERR_VERIFICATION_ERROR: dict[str, str] = {
    'error': 'verification_error',
    'error_type': 'ReconCitationVerificationError',
}

_ERR_RUN_STILL_LIVE: dict[str, str] = {
    'error': 'run_still_live',
    'error_type': 'ReconCitationRunStillLive',
}


def build_citation_repair_record(
    memory_id: str,
    replacement_memory_id: str | None,
    store: str,
    repaired_by: str,
    reason: str = 'memory_not_found',
) -> dict[str, Any]:
    """Build the one provenance record a repair appends to a finding.

    Declared in exactly one function, mirroring
    ``citation_verifier.build_citation_tombstone``: a durable rewrite of a
    historical audit record must say what it changed and why, or the repaired
    blob becomes indistinguishable from a report that never carried the dangling
    id. ``replacement_memory_id`` is None for a drop-only repair.
    """
    return {
        'memory_id': memory_id,
        'replacement_memory_id': replacement_memory_id,
        'store': store,
        'reason': reason,
        'repaired_by': repaired_by,
        'repaired_at': datetime.now(UTC).isoformat(),
    }


def _fingerprint_from_record(record: Any) -> dict[str, Any]:
    """The mem0 ``metadata_fingerprint`` shape, from a ``get_memory_by_id`` record.

    Same ``{category, agent_id, created_at}`` triple ``MemoryService.get_memory``
    returns for a mem0 citation, so a repaired citation is shaped exactly like
    one ``cite_memory`` would have written. Built from the record the
    corroboration read already fetched rather than a second service call.
    """
    payload = (record or {}).get('metadata') or {}
    return {
        'category': payload.get('category'),
        'agent_id': payload.get('agent_id'),
        'created_at': payload.get('created_at'),
    }


def _find_finding(
    run: Any, finding_id: str
) -> tuple[str, Any, dict[str, Any]] | None:
    """Locate ``finding_id`` across EVERY stage's ``items_flagged``.

    Cross-stage because the caller knows the finding id, not which stage filed
    it. Any ``stage_reports`` value that is not a ``StageReport`` — the raw
    ``_error`` / ``_resume`` entries ``journal.get_run`` deliberately keeps
    as-is — is inert here rather than a crash.
    """
    for stage_name, report in run.stage_reports.items():
        if not isinstance(report, StageReport):
            continue
        for finding in report.items_flagged:
            if isinstance(finding, dict) and finding.get('finding_id') == finding_id:
                return stage_name, report, finding
    return None


def _verification_error(memory_id: str, role: str, exc: BaseException) -> dict[str, Any]:
    """The verdict for a RAISED lookup: unknown, never 'absent'.

    Mirrors ``verify_cited_memories``'s third branch. A raised read leaves the
    id's existence undetermined, and acting on 'undetermined' is precisely the
    silent-fail this path forbids — so it is reported with the raised type as a
    structured fact and the caller decides, rather than propagating (which would
    crash a stage) or collapsing into a repair.
    """
    logger.warning(
        'citation_repair: %s lookup for %s raised %s — refusing to act on an '
        'undetermined citation',
        role,
        memory_id,
        type(exc).__name__,
    )
    return _ERR_VERIFICATION_ERROR | {
        'memory_id': memory_id,
        'role': role,
        'exception_type': type(exc).__name__,
        'exception_message': str(exc),
    }


def _is_citation_of(entry: Any, memory_id: str) -> bool:
    """True when ``entry`` is a mem0 citation of ``memory_id``.

    Case-insensitive on the id for the same reason ``recon_report._UUID_RE``
    carries ``re.IGNORECASE``: neither Graphiti/Neo4j nor mem0 normalises UUID
    case on read-back, so a case-differing stored id is the same citation.
    """
    return (
        isinstance(entry, dict)
        and entry.get('store') == SUPPORTED_STORE
        and isinstance(entry.get('memory_id'), str)
        and entry['memory_id'].lower() == memory_id.lower()
    )


async def repair_memory_citation(
    journal: Any,
    memory_service: Any,
    *,
    target_run_id: str,
    finding_id: str,
    memory_id: str,
    store: str,
    replacement_memory_id: str | None,
    repaired_by: str,
    live_run_ids: frozenset[str] = frozenset(),
    apply: bool = True,
) -> dict[str, Any]:
    """Re-point (or drop) a dangling citation on a completed run's finding.

    ``target_run_id`` is the run that OWNS the finding — deliberately NOT the
    caller's own ``run_id``, which the ``ReconReportState`` wrapper keeps for
    its unchanged ``_resolve_entry`` contract and for stamping ``repaired_by``.

    Returns a structured dict: ``{'status': 'repaired'|'dry_run', ...}`` on
    success, or one of the ``_ERR_*`` branches. Never raises for a backend
    read failure — that is reported as ``verification_error``.

    ``apply=False`` computes the whole outcome, INCLUDING both corroboration
    reads, and returns it without writing, so the operator script's dry-run and
    the MCP tool traverse one code path (INV-5) and a dry-run tells the operator
    whether the gates pass before anything is written.
    """
    run = await journal.get_run(target_run_id)

    located = _find_finding(run, finding_id)
    stage_name, _report, finding = located

    # ── Corroboration (INV-3: corroborate before acting) ──────────────────
    # Ordered so NO journal write can happen unless every gate passes. Both
    # reads run even for apply=False, so a dry-run tells the operator whether
    # the gates hold before anything is written.
    try:
        victim_record = await memory_service.get_memory_by_id(run.project_id, memory_id)
    except Exception as exc:
        return _verification_error(memory_id, 'victim', exc)
    if victim_record:
        # Still alive — this is a valid claim, not a dangling one. Refusing here
        # is what makes the path structurally incapable of retargeting live
        # provenance; the worst it can ever do is re-point a claim that already
        # had no backing.
        return _ERR_CITATION_NOT_DANGLING | {
            'target_run_id': target_run_id,
            'finding_id': finding_id,
            'memory_id': memory_id,
            'hint': (
                f'{memory_id} still resolves in mem0 for project '
                f'{run.project_id!r}; only a CONFIRMED-absent citation may be '
                'repaired.'
            ),
        }

    replacement_record = None
    if replacement_memory_id is not None:
        try:
            replacement_record = await memory_service.get_memory_by_id(
                run.project_id, replacement_memory_id
            )
        except Exception as exc:
            return _verification_error(replacement_memory_id, 'replacement', exc)
        if not replacement_record:
            # A repair that installed an unresolvable id would make this tool a
            # generator of the defect it exists to fix.
            return _ERR_REPLACEMENT_NOT_FOUND | {
                'target_run_id': target_run_id,
                'finding_id': finding_id,
                'replacement_memory_id': replacement_memory_id,
                'hint': (
                    f'{replacement_memory_id} does not resolve in mem0 for '
                    f'project {run.project_id!r}; omit it to DROP the dangling '
                    'citation instead of re-pointing it.'
                ),
            }

    cited = finding.get('cited_memories') or []
    kept = [
        entry
        for entry in cited
        if not _is_citation_of(entry, memory_id)
    ]
    removed_count = len(cited) - len(kept)
    if replacement_memory_id is not None:
        kept.append(
            {
                'memory_id': replacement_memory_id,
                'store': store,
                'metadata_fingerprint': _fingerprint_from_record(replacement_record),
            }
        )
    finding['cited_memories'] = kept
    finding.setdefault(CITATION_REPAIRS_KEY, []).append(
        build_citation_repair_record(
            memory_id, replacement_memory_id, store, repaired_by
        )
    )

    await journal.update_run_stage_reports(target_run_id, run.stage_reports)
    logger.info(
        'citation_repair: run=%s stage=%s finding=%s removed=%s (x%d) -> replacement=%s by=%s',
        target_run_id,
        stage_name,
        finding_id,
        memory_id,
        removed_count,
        replacement_memory_id,
        repaired_by,
    )
    return {
        'status': 'repaired',
        'target_run_id': target_run_id,
        'project_id': run.project_id,
        'stage': stage_name,
        'finding_id': finding_id,
        'removed_memory_id': memory_id,
        'removed_count': removed_count,
        'replacement_memory_id': replacement_memory_id,
        'deduped': False,
        'store': store,
        'cited_memories': kept,
    }
