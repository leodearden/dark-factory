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
the worst it can do is retarget a claim that already had no backing, and (for an
in-agent caller) only within its own project: the journal is shared across every
project the process reconciles, so ``caller_project_id`` confines the repair to
runs the caller actually owns. It reuses ``citation_verifier``'s lookup primitive and its found/None/raised branching so
the two halves cannot disagree about what a backend timeout means.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from fused_memory.models.reconciliation import RunStatus, StageReport

# Imported, not re-declared: ``recon_report._UUID_RE`` and
# ``citation_verifier._CANONICAL_UUID_RE`` already exist, and a third copy of the
# same gate is exactly the lockstep duplication INV-5 forbids.
from fused_memory.reconciliation.citation_verifier import is_concrete_memory_id

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

# The run statuses a repair may touch — an ALLOWLIST, deliberately inverted from
# the "refuse status == 'running'" check this replaces, because the two failure
# directions are asymmetric: wrongly PERMITTING yields a write that reports
# ``status: repaired`` and is then silently overwritten (what the run_still_live
# hint itself calls worse than a clean refusal), while wrongly REFUSING yields a
# loud, recoverable error. So a status absent from the enum-of-today must land on
# the refusing side by default.
#
# These four are genuinely terminal: nothing re-adopts them. ``interrupted`` is
# excluded precisely because something does — ``journal.get_interrupted_runs()``
# is ``SELECT * FROM runs WHERE status = 'interrupted'``, the startup
# adopt-and-resume pass re-claims exactly those runs, and the resumed cycle
# rewrites the whole stage_reports blob from its own loaded copy.
REPAIRABLE_RUN_STATUSES = frozenset(
    {
        RunStatus.completed,
        RunStatus.failed,
        RunStatus.rolled_back,
        RunStatus.circuit_breaker,
    }
)

# Compared on raw ``.value`` strings so the gate holds whether ``run.status``
# arrives as a coerced ``RunStatus`` or as a bare ``str`` off the journal row.
# Every member's name happens to equal its value today, so StrEnum hashing would
# coincide — the gate deliberately does not rest on that coincidence.
_REPAIRABLE_STATUS_VALUES = frozenset(status.value for status in REPAIRABLE_RUN_STATUSES)

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

# Cross-project isolation. ``reconciliation.data_dir`` holds ONE journal for
# every project this process reconciles, so ``get_run(target_run_id)`` will
# happily resolve a run owned by someone else. Without this gate a Stage-1/
# Stage-2 agent reconciling project A could rewrite the durable audit record of
# a completed run owned by project B — and issue B-scoped Mem0 point reads to do
# it. Every other citation path scopes its backend read to the CALLER's own
# project (``cite_memory`` uses ``finding_entry.project_id``); this keeps the
# repair path in that same containment story.
_ERR_PROJECT_MISMATCH: dict[str, str] = {
    'error': 'project_mismatch',
    'error_type': 'ReconCitationProjectMismatch',
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


def _run_still_live_hint(
    target_run_id: str, status_value: str, in_process: bool
) -> str:
    """Say WHICH of the two live-run reasons refused this repair.

    The reasons are independent — a non-terminal row status, and an in-process
    recon-report entry — and an operator can only act on the refusal if the
    message names the one that actually fired.
    """
    parts: list[str] = []
    if status_value not in _REPAIRABLE_STATUS_VALUES:
        parts.append(
            f'run {target_run_id} is in status {status_value!r}, which is not one '
            'of the terminal statuses a repair may touch '
            f'({sorted(_REPAIRABLE_STATUS_VALUES)}); its stage_reports blob is '
            'rewritten wholesale from a writer\'s own loaded copy at each stage '
            'end, so a journal-side repair would be silently clobbered.'
        )
        if status_value == RunStatus.interrupted.value:
            parts.append(
                "An 'interrupted' run is RESUMABLE, not finished: the startup "
                'adopt-and-resume pass re-claims exactly these runs '
                "(get_interrupted_runs() is SELECT * FROM runs WHERE status = "
                "'interrupted'), and the resumed cycle sets status = running IN "
                "MEMORY ONLY — the row stays 'interrupted' on disk — while "
                'rewriting stage_reports at each stage boundary. Retry once the '
                'run reaches a terminal status.'
            )
    if in_process:
        parts.append(
            f'run {target_run_id} is still held by the in-process recon-report '
            'state, so the harness may yet rewrite its stage_reports. Within a '
            'live run the ordinary cite_memory / delete_finding path already '
            'reaches the finding (_resolve_finding is cross-stage).'
        )
    return ' '.join(parts)


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
    caller_project_id: str | None = None,
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

    ``caller_project_id`` is the project the CALLER is scoped to. When supplied
    (the MCP tool always supplies it, from its own ``_resolve_entry(run_id)``
    entry) a target run owned by any other project is refused with
    ``project_mismatch`` BEFORE any Mem0 read or journal write, so an in-agent
    caller can never reach across the shared journal into another project's
    audit record. ``None`` is the deliberate operator bypass: the
    ``repair_recon_citation`` script runs out-of-band with no owning project and
    is the intended path for a cross-project correction.
    """
    # ── Shape gates (no I/O; cheapest refusals first) ─────────────────────
    for role, candidate in (
        ('memory_id', memory_id),
        ('replacement_memory_id', replacement_memory_id),
    ):
        if candidate is None:
            continue  # drop-only repair: no replacement to shape-check
        if not is_concrete_memory_id(candidate):
            return _ERR_INVALID_UUID_SHAPE | {
                'field': role,
                'value': candidate,
                'hint': (
                    'must be a canonical 36-char UUID; a truncated prefix is '
                    'not an id, and looking one up would return not-found and '
                    'read as a confirmed-absent citation.'
                ),
            }

    if store != SUPPORTED_STORE:
        return _ERR_UNSUPPORTED_STORE | {
            'store': store,
            'hint': (
                f'only store={SUPPORTED_STORE!r} can be repaired: corroboration '
                'goes through get_memory_by_id, a Mem0/Qdrant point read, which '
                'returns not-found for every graphiti id and would therefore '
                'classify valid graph provenance as dangling. Verifying a graph '
                'citation needs a different primitive (get_edge / '
                'get_entity_by_uuid).'
            ),
        }

    # ── Resolution gates ──────────────────────────────────────────────────
    run = await journal.get_run(target_run_id)
    if run is None:
        return _ERR_TARGET_RUN_NOT_FOUND | {'target_run_id': target_run_id}

    # Cross-project isolation, checked before the liveness gate and therefore
    # before any Mem0 read: the corroboration reads below are issued in
    # ``run.project_id``'s scope, so a mismatch must refuse BEFORE them, not
    # merely before the write.
    if caller_project_id is not None and run.project_id != caller_project_id:
        return _ERR_PROJECT_MISMATCH | {
            'target_run_id': target_run_id,
            'caller_project_id': caller_project_id,
            'target_project_id': run.project_id,
            'hint': (
                f'run {target_run_id} is owned by project '
                f'{run.project_id!r}, but this caller is scoped to '
                f'{caller_project_id!r}. The reconciliation journal is shared '
                'across projects; a repair may only touch a run owned by the '
                "caller's own project. A genuine cross-project correction goes "
                'through the out-of-band repair_recon_citation operator script.'
            ),
        }

    # Terminal runs only — checked before any Mem0 read. Two halves because the
    # journal row and the in-process view can disagree: a row can already read
    # 'completed' while ReconReportState is still writing its entries through.
    status_value = str(run.status)
    in_process = target_run_id in live_run_ids
    if status_value not in _REPAIRABLE_STATUS_VALUES or in_process:
        return _ERR_RUN_STILL_LIVE | {
            'target_run_id': target_run_id,
            'status': status_value,
            'hint': _run_still_live_hint(target_run_id, status_value, in_process),
        }

    located = _find_finding(run, finding_id)
    if located is None:
        return _ERR_FINDING_UNKNOWN | {
            'target_run_id': target_run_id,
            'finding_id': finding_id,
        }
    stage_name, _report, finding = located

    if not any(
        _is_citation_of(entry, memory_id)
        for entry in finding.get('cited_memories') or []
    ):
        # Also the idempotency verdict: a second identical repair lands here
        # rather than appending a further provenance record.
        return _ERR_CITATION_NOT_PRESENT | {
            'target_run_id': target_run_id,
            'stage': stage_name,
            'finding_id': finding_id,
            'memory_id': memory_id,
        }

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

    # ── Compute the new citation list (no mutation yet) ───────────────────
    # The mutation is confined to cited_memories plus the appended provenance
    # record — never the finding roster, stats or summary — so the run's
    # flagged_count and every downstream stat stay exactly as originally
    # reported, and complete()'s cached response and the judge's stat
    # verification cannot be invalidated after the fact.
    cited = finding.get('cited_memories') or []
    kept = [entry for entry in cited if not _is_citation_of(entry, memory_id)]
    removed_count = len(cited) - len(kept)
    deduped = False
    if replacement_memory_id is not None:
        if any(_is_citation_of(entry, replacement_memory_id) for entry in kept):
            # Already cited — appending again would leave the finding claiming
            # the same evidence twice. The repair is still real: the dangling
            # entry is gone.
            deduped = True
        else:
            kept.append(
                {
                    'memory_id': replacement_memory_id,
                    'store': store,
                    'metadata_fingerprint': _fingerprint_from_record(replacement_record),
                }
            )

    outcome = {
        'status': 'repaired' if apply else 'dry_run',
        'target_run_id': target_run_id,
        'project_id': run.project_id,
        'stage': stage_name,
        'finding_id': finding_id,
        'removed_memory_id': memory_id,
        'removed_count': removed_count,
        'replacement_memory_id': replacement_memory_id,
        'deduped': deduped,
        'store': store,
        'cited_memories': kept,
    }
    if not apply:
        # Everything above already ran — including both corroboration reads —
        # so the operator learns whether the gates hold before anything is
        # written. Only the mutation and the write are skipped, which is what
        # makes this ONE code path rather than a second implementation of
        # "which entries would be removed" that must stay in lockstep (INV-5).
        return outcome

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
    return outcome
