"""The git-authority landing tier, extracted from the escalation MCP server.

PRD ``docs/prds/landed-not-done-recovery.md`` label ζ (task 4887).  Everything
here lived as 4-space-indent siblings inside
``escalation/src/escalation/server.py::create_server``, reachable only by
building a whole MCP server — which is why the periodic landed reconciler that
PRD's leaf η (task 4651) needs could not reuse the fully-guarded verdict
``merge_status`` computes and throws away.

**Where this lives** resolves that PRD's Open question 4 in favour of its own
recommendation: ``orchestrator/pyproject.toml`` already declares ``escalation``
as a workspace dependency (and not the reverse), so η — which lives in
``orchestrator`` — imports this module STATICALLY, with no new runtime
indirection.  ``shared/`` was rejected because it would place a runtime
reverse-import of ``orchestrator.landing_evidence`` inside the package every
other package depends on: strictly worse layering than the status quo, since
the reverse import would then sit in the most widely-imported package rather
than in the one that already tolerates it.

**Two import rules this module must keep.**  It must never import
``escalation.server`` — server.py imports this, and a cycle would turn a
graceful degradation into a server-construction failure.  And its
``orchestrator.landing_evidence`` import is RUNTIME-ONLY, made inside the
fire-safe wrapper in :func:`probe_landing`: ``escalation/pyproject.toml``
declares no orchestrator dependency, and the import resolves at runtime only
because the escalation server is hosted inside the orchestrator process.
Hoisting it to module level would convert that degradation into an import-time
crash.  ``shared.*`` imports are intra-workspace and unconstrained.

**On the two non-landing outcomes.**  ``merge_status`` today renders
:attr:`GitAuthorityOutcome.landed_unconfirmed` and
:attr:`GitAuthorityOutcome.no_signal` IDENTICALLY — as the bare Tier-4
``unknown``.  The distinction is materialized on the verdict rather than on
the wire because the ``merge_status`` response vocabulary belongs to task
**4831** (``plans/merge-status-durable-non-landed-prd.md`` D1/D2), which
consumes this module and maps the two onto its epistemic states.  Do not add
a response field here to carry it.
"""
from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import Any

from shared.merge_state import MergeState

logger = logging.getLogger(__name__)


class GitAuthorityOutcome(enum.StrEnum):
    """What git authority was able to establish about a branch's landing.

    A ``StrEnum`` rather than bare strings so a consumer switches on a closed
    set, and for consistency with the two enums this code sits between —
    ``shared.merge_state.MergeState`` above it and
    ``orchestrator.landing_evidence.LandingReason`` below.

    ``landed_unconfirmed`` and ``no_signal`` are DIFFERENT PROPOSITIONS and
    that is the whole reason both exist: the first means git positively
    established a landing (the branch is an ancestor of main, or a merge
    marker was found) but attribution or effect-survival could not be
    confirmed; the second means nothing was established at all.  Collapsing
    them is exactly the conflation task 4831 exists to undo.
    """

    found_on_main = 'found_on_main'
    landed_unconfirmed = 'landed_unconfirmed'
    no_signal = 'no_signal'


def found_on_main_response(request_id: str | None, merge_sha: str) -> dict[str, Any]:
    """Build the git-authority Tier-3.5 done/found_on_main response.

    ``merge_sha`` is a commit ON MAIN on both resolution paths, with one
    explicit exception stated below (task 3103):

    - **Live-branch path** (``is_ancestor`` hit): the citation commit
      discovered by ``validate_landing_evidence`` — a commit on main
      whose subject cites the task.
    - **Deleted-branch path** (``find_merge_marker`` hit): the
      merge-commit SHA found on main via ``git log``.

    Both are effect-present-checked against current main HEAD before
    being returned, so ``merge_sha`` is safe to record as provenance
    as-is.  (Before task 3103 the live-branch path returned the *branch
    tip*, which for a ``--no-ff`` merge is a distinct commit that is not
    on main's first-parent chain — callers were told to prefer the
    deleted-branch path's value.  That caveat no longer applies.)

    **The one exception — ``git.commit_citation_pattern == ''``.**  That
    is the documented per-project opt-out for projects with no citation
    convention (config.py; ``find_task_citation_commit`` honours it by
    returning None for everything, so running the gate would reject
    unconditionally and turn this tier into dead code).  On that setting
    the live-branch path skips the citation gate entirely and
    ``merge_sha`` is the raw BRANCH TIP, neither citation-discovered nor
    effect-present-checked — i.e. exactly the pre-3103 ``--no-ff`` wart,
    deliberately retained as the price of the opt-out (the degeneracy
    guard still applies).  Do not read the paragraphs above as
    unconditional: on such a project a caller stamping ``merge_sha`` as
    provenance is recording a branch tip, and a reverted landing is
    indistinguishable from a live one (review #4).  The opt-out is
    ``''`` only; ``None`` means "use the built-in default pattern" and
    keeps the full guarantee.  Both SKILL.md runbooks carry the same
    exception.
    """
    return {
        'state': MergeState.done,
        'request_id': request_id,
        'generation': 1,
        'kind': 'found_on_main',
        'merge_sha': merge_sha,
        'outcome': 'found_on_main',
    }


@dataclass(frozen=True)
class TaskMetadataResult:
    """Task metadata PLUS whether the fetch that produced it actually worked.

    ``metadata == {}`` alone is not proof that a task carries no metadata —
    it is equally what a missing harness, an absent scheduler or a raising
    ``get_task`` produces.  Read it TOGETHER with ``unavailable``, which is
    True only when the fetch FAILED.

    Copies the shape of
    ``orchestrator/src/orchestrator/task_ground_truth.py::TruthReport``
    (``escalation_store_unavailable``), and its normative source
    ``escalation/src/escalation/pins.py::classify_pins``
    (``store_unavailable``), down to defaulting the flag False so
    hand-construction of the healthy case stays ergonomic.

    The distinction exists for the PRD's leaf η (task 4651), whose writer
    must treat "could not read metadata" as "cannot verify the degeneracy
    guard" rather than as "no degeneracy" — the collapse today's
    ``{}``-on-everything forces.  ``merge_status`` and ``merge_request``
    both deliberately still read only ``.metadata``.
    """

    metadata: dict[str, Any]
    unavailable: bool = False


async def task_metadata(harness: Any, tid: str, *, site: str) -> TaskMetadataResult:
    """Best-effort task metadata for the git-authority guards (task 3103).

    Returns ``metadata == {}`` on EVERY failure mode — no harness, no
    ``scheduler`` attribute, ``get_task`` raising, or a None/metadata-less
    task — and never raises.  A scheduler fault must degrade a single guard,
    not swallow the whole probe.

    The ``{}`` is qualified by :attr:`TaskMetadataResult.unavailable`, which
    separates a FAULT from a FACT: the first three modes above set it True
    ("we could not read"), while a healthy read that finds no metadata — or
    finds no such task — leaves it False ("there is genuinely nothing
    there").  "No such task" is deliberately a fact, not a fault: the
    discriminant is READ SUCCESS, because a present-but-metadata-less record
    is equally unable to answer the degeneracy question yet is plainly not a
    fault, and flagging an absent record True would make ``unavailable``
    mean a fault OR a legitimately-absent task — the very conflation it
    exists to remove.

    ``{}`` deliberately FAILS OPEN out of the degeneracy check.  On the
    ``merge_status`` path it then falls THROUGH to the citation gate,
    which is git-only and needs no task metadata; on the
    ``merge_request`` fast path there is no citation gate, so the block
    simply reverts to its pre-3103 ancestry/patch-id behaviour.  Either
    way this is exact parity with the harness, which treats an absent or
    non-40-hex ``branch_base_sha`` as "no degeneracy signal" rather than
    as grounds to reject: a metadata fault must never fabricate a
    confident answer, and must never hard-fail a genuinely merged branch.

    Args:
        harness: The orchestrator harness, duck-typed — only
            ``harness.scheduler.get_task`` is ever touched, and ``None`` is
            a supported value.  It was a ``create_server`` capture with no
            module-level fallback, so extraction makes it an explicit
            parameter.
        tid: Bare task id (no ``task/`` prefix).  Both callers derive it
            from the branch ref they resolved the tip from, so the
            metadata and the tip always describe the same branch.
        site: The calling tool (``'merge_status'`` / ``'merge_request'``),
            interpolated into the degradation warning.  Without it a
            scheduler fault on the SUBMIT path was logged as a
            merge_status failure, so an operator grepping for a
            submit-path degradation would not find it (review #3).
    """
    if harness is None:
        return TaskMetadataResult(metadata={}, unavailable=True)
    scheduler = getattr(harness, 'scheduler', None)
    if scheduler is None:
        return TaskMetadataResult(metadata={}, unavailable=True)
    try:
        task = await scheduler.get_task(tid)
    except Exception:
        logger.warning(
            '%s: scheduler.get_task(%s) failed — proceeding without task '
            'metadata (degeneracy check skipped)',
            site, tid, exc_info=True,
        )
        return TaskMetadataResult(metadata={}, unavailable=True)
    if not task:
        return TaskMetadataResult(metadata={})
    return TaskMetadataResult(metadata=task.get('metadata') or {})
