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
reverse-import of ``orchestrator.merge_lane.landing_evidence`` inside the package every
other package depends on: strictly worse layering than the status quo, since
the reverse import would then sit in the most widely-imported package rather
than in the one that already tolerates it.

**Two import rules this module must keep.**  It must never import
``escalation.server`` — server.py imports this, and a cycle would turn a
graceful degradation into a server-construction failure.  And its
``orchestrator.merge_lane.landing_evidence`` imports are RUNTIME-ONLY, made inside
function bodies reached from the fire-safe wrapper in :func:`probe_landing`:
``escalation/pyproject.toml`` declares no orchestrator dependency, and the
import resolves at runtime only because the escalation server is hosted inside
the orchestrator process.  Hoisting either to module level would convert that
degradation into an import-time crash.  ``shared.*`` imports are intra-workspace
and unconstrained.

**On the three non-landing outcomes.**  This module reports WHAT git
established — :attr:`GitAuthorityOutcome.landed_unconfirmed`,
:attr:`GitAuthorityOutcome.no_signal` or
:attr:`GitAuthorityOutcome.probe_failed` — and never how a consumer renders
it.  ``merge_status``'s mapping onto its epistemic states lives in
``escalation/src/escalation/server.py::merge_status``.  Do not add a response
field here to carry it.
"""
from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import Any

from shared.branch_names import bare_queued_branch_id, canonical_queued_branch_name
from shared.merge_state import MergeState
from shared.task_read import TaskAbsent, TaskFound

logger = logging.getLogger(__name__)


class GitAuthorityOutcome(enum.StrEnum):
    """What git authority was able to establish about a branch's landing.

    A ``StrEnum`` rather than bare strings so a consumer switches on a closed
    set, and for consistency with the two enums this code sits between —
    ``shared.merge_state.MergeState`` above it and
    ``orchestrator.merge_lane.landing_evidence.LandingReason`` below.

    ``landed_unconfirmed``, ``no_signal`` and ``probe_failed`` are DIFFERENT
    PROPOSITIONS and that is the whole reason all three exist: the first
    means git positively established a landing (the branch is an ancestor of
    main, a merge marker was found, or every branch commit is
    patch-equivalent on main) but attribution or effect-survival could not
    be confirmed; the second means the probe ran and established
    nothing; the third means the probe raised, so nothing could be
    established either way.

    NOT to be confused with the unrelated ``'no_signal'`` that
    ``orchestrator.merge_lane.landing_evidence`` writes into
    ``probe['delivered_checks_outcome']``, which means "the task's declared
    delivered-checks produced no confirming signal" — a statement about one
    sub-probe, not about whether a landing was established.  The two never
    share an object (a verdict here carries ``reason``/``evidence_sha``, not
    ``probe``), so always read a bare ``no_signal`` with its owner attached.
    """

    found_on_main = 'found_on_main'
    landed_unconfirmed = 'landed_unconfirmed'
    no_signal = 'no_signal'
    probe_failed = 'probe_failed'


class GitAuthorityArm(enum.StrEnum):
    """Which git signal a verdict was reached through.

    A ``StrEnum`` for the same reason :class:`GitAuthorityOutcome` is one
    (heuristic 12, structured data instead of meaningful strings): ``arm`` is
    a closed three-member set a consumer switches on, not free text.
    ``ancestor`` — the branch ref still exists and its tip is an ancestor of
    main; ``marker`` — the ref is gone and a merge marker for it was found on
    main; ``patch_id`` — the ref still exists, its tip is NOT an ancestor of
    main, and
    ``orchestrator/src/orchestrator/merge_lane/landing_evidence.py::branch_work_landed``
    decided the verdict (patch-id containment, behind its degeneracy and
    empty-net-diff guards).  That last is the shape a worker-rebased solo
    landing leaves: ``GitOps.advance_main``'s rebased path retains the ref at
    its pre-rebase tip.  The arms carry different evidence and different
    guards, which is why the verdict names the one it came from.
    """

    ancestor = 'ancestor'
    marker = 'marker'
    patch_id = 'patch_id'


def found_on_main_response(request_id: str | None, merge_sha: str) -> dict[str, Any]:
    """Build the git-authority Tier-3.5 done/found_on_main response.

    ``merge_sha`` is a commit ON MAIN on all three resolution paths, with one
    explicit exception — Exception 1 below (task 3103); Exception 2
    weakens a different guarantee and leaves this one intact:

    - **Live-branch path** (``is_ancestor`` hit): the citation commit
      discovered by ``validate_landing_evidence`` — a commit on main
      whose subject cites the task.
    - **Deleted-branch path** (``find_merge_marker`` hit): the
      merge-commit SHA found on main via ``git log``.
    - **Patch-id path** (a live branch that is NOT an ancestor of main,
      every commit patch-equivalent on main — the worker-rebased landing):
      ``branch_work_landed``'s ``evidence_sha``, the most recent commit on
      main whose subject cites the task by the project's
      ``git.commit_citation_pattern`` (none on a ``''`` opt-out), else the
      branch tip's patch-id twin on main.

    The first two are effect-present-checked against current main HEAD
    before being returned.  The patch-id path is NOT, by
    ``docs/prds/landed-not-done-recovery.md`` D1's non-decay contract: it
    is checked by containment instead, and a later revert does not unland
    it.  So ``merge_sha`` is safe to record as provenance as-is on every
    path — EXCEPT in the two cases below, which
    :attr:`GitAuthorityVerdict.merge_sha_fully_guarded` answers as one
    question.  Neither exception applies to the patch-id path: it never
    consults the delivered checks, and on a ``''`` opt-out its sha is still
    a commit on main carrying the tip's diff.  (Before task 3103 the
    live-branch path returned the *branch tip*, which for a ``--no-ff``
    merge is a distinct commit that is not
    on main's first-parent chain — callers were told to prefer the
    deleted-branch path's value.  That caveat no longer applies.)

    **Exception 1 — ``git.commit_citation_pattern == ''``.**  That
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
    indistinguishable from a live one (review #4).  A caller detects that
    case from :attr:`GitAuthorityVerdict.citation_gate_skipped` rather
    than by inspecting config.  The opt-out is ``''`` only; ``None`` means
    "use the built-in default pattern" and keeps the full guarantee.
    Both SKILL.md runbooks carry the same exception.

    **Exception 2 — a delivered-checks rescue.**  ``merge_sha`` is still a
    commit ON MAIN here (this exception does NOT reach the where-the-sha-
    came-from paragraphs above), but the EFFECT-PRESENT guarantee is gone:
    ``validate_landing_evidence``'s second accept path
    (``orchestrator/src/orchestrator/landing_evidence.py``, both modes)
    sits INSIDE the ``commit_effect_present_in_main(...) is False`` branch
    and accepts anyway when the three-leg delivered-checks differential
    confirms.  So ``merge_sha`` may name a landing that was later REVERTED
    — the exact task-1175 shape FIX 1' exists to catch.  It is NARROW: it
    requires (i) the task to declare a non-empty
    ``metadata.delivered_checks``, (ii) the fleet-wide
    ``delivered_checks.enabled`` config to be on, and (iii) the
    differential to actually confirm.  A task declaring no checks passes
    ``[]``, which is falsy, so the path stays unreachable and its
    behaviour is unchanged.  A caller detects the case from
    :attr:`GitAuthorityVerdict.merge_sha_fully_guarded` — preferred, it
    covers both exceptions — or from
    :attr:`GitAuthorityVerdict.rescued_by_delivered_checks` for this one
    specifically, rather than by inspecting task metadata or config.
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
    it is equally what a missing harness, an absent scheduler, a raising
    ``read_task`` or a failed read produces.  Read it TOGETHER with
    ``unavailable``, which is True only when the fetch FAILED.

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
    ``scheduler`` attribute, ``read_task`` raising, ``read_task`` reporting a
    failed read (a fused-memory error reply, a dispatch fault, an unreadable
    envelope), or an absent or metadata-less task — and never raises.  A
    scheduler fault must degrade a single guard, not swallow the whole probe.

    The ``{}`` is qualified by :attr:`TaskMetadataResult.unavailable`, which
    separates a FAULT from a FACT: the first four modes above set it True
    ("we could not read"), while a healthy read that finds no metadata — or
    finds no such task (``TaskAbsent``) — leaves it False ("there is
    genuinely nothing there").  "No such task" is deliberately a fact, not a
    fault: the discriminant is READ SUCCESS, because a present-but-metadata-less record
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
            ``harness.scheduler.read_task`` is ever touched, and ``None`` is
            a supported value.  It was a ``create_server`` capture with no
            module-level fallback, so extraction makes it an explicit
            parameter.
        tid: Bare task id (no ``task/`` prefix).  Both callers derive it
            from the branch ref they resolved the tip from, so the
            metadata and the tip always describe the same branch.
        site: The calling tool or writer (``'merge_status'`` /
            ``'merge_request'``), interpolated into the degradation
            warning.  Without it a scheduler fault on the SUBMIT path was
            logged as a merge_status failure, so an operator grepping for a
            submit-path degradation would not find it (review #3).  Why a
            read failed is logged once, by its owner
            ``orchestrator/src/orchestrator/scheduler.py::Scheduler.read_task``;
            this warning adds only the site and what it skipped.
    """
    if harness is None:
        return TaskMetadataResult(metadata={}, unavailable=True)
    scheduler = getattr(harness, 'scheduler', None)
    if scheduler is None:
        return TaskMetadataResult(metadata={}, unavailable=True)
    try:
        read = await scheduler.read_task(tid)
    except Exception:
        logger.warning(
            '%s: scheduler.read_task(%s) raised — proceeding without task '
            'metadata (degeneracy check skipped)',
            site, tid, exc_info=True,
        )
        return TaskMetadataResult(metadata={}, unavailable=True)
    if isinstance(read, TaskAbsent):
        return TaskMetadataResult(metadata={})
    if isinstance(read, TaskFound):
        return TaskMetadataResult(metadata=read.task.get('metadata') or {})
    logger.warning(
        '%s: task %s metadata unavailable — proceeding without it '
        '(degeneracy check skipped)',
        site, tid,
    )
    return TaskMetadataResult(metadata={}, unavailable=True)


@dataclass(frozen=True)
class GitAuthorityVerdict:
    """What git authority established about one branch, and on what evidence.

    ``landed_unconfirmed`` and ``no_signal`` are the whole reason this is a
    verdict rather than a boolean.  ``landed_unconfirmed`` means git
    POSITIVELY ESTABLISHED that this branch's work is on main — the branch
    is an ancestor of main, a merge marker for it was found, or every
    branch commit is patch-equivalent on main — and only ATTRIBUTION or
    EFFECT-SURVIVAL could not be confirmed.  ``no_signal``
    means no landing was established at all.  A caller that cannot tell them
    apart cannot tell "we found the landing but could not verify it" from
    "we found nothing".

    ``probe_failed`` is a third proposition again: the probe raised, so it
    could not look at all.  How ``merge_status`` renders each outcome lives
    in ``escalation/src/escalation/server.py::merge_status``; this dataclass
    is that tool's INPUT, and leaf η's (task 4651).

    ``merge_sha`` is populated on ``found_on_main`` ONLY — it is the sha a
    caller may stamp as provenance, WITH TWO EXCEPTIONS that this dataclass
    carries structurally rather than leaving to prose.  They are
    INDEPENDENT, and they differ in KIND:

    - ``citation_gate_skipped`` — the project opted out of citation checking
      (``git.commit_citation_pattern == ''``) and ``merge_sha`` is the raw
      BRANCH TIP: not a commit discovered on main, and not
      effect-present-checked.
    - ``rescued_by_delivered_checks`` — ``merge_sha`` IS a genuine commit on
      main, but its EFFECT is absent at main HEAD and the accept came from
      the delivered-checks differential instead; i.e. the task-1175
      reverted-landing shape the FIX 1' effect-present guard exists to
      catch.

    A provenance writer should ask :attr:`merge_sha_fully_guarded` rather
    than enumerate these flags, so a third weakening does not silently
    break it.  Do NOT infer either case from ``evidence_sha`` being None,
    which is a coincidence of one path and not a documented signal.
    :func:`found_on_main_response` states both caveats once.

    ``evidence_sha`` is the weaker thing: the commit the probe was reasoning
    ABOUT, and it is present on rejects too.

    ``reason`` stays a plain ``str | None`` carrying the landing verdict's
    own spelling rather than
    ``orchestrator.merge_lane.landing_evidence.LandingReason``, because importing that
    enum would force exactly the static reverse import this module exists to
    avoid.  ``LandingReason`` is a genuine ``str`` subclass, so the spelling
    round-trips and JSON-encodes as itself.
    """

    outcome: GitAuthorityOutcome
    merge_sha: str | None = None
    arm: GitAuthorityArm | None = None
    reason: str | None = None
    evidence_sha: str | None = None
    metadata_unavailable: bool = False
    citation_gate_skipped: bool = False
    rescued_by_delivered_checks: bool = False

    @property
    def merge_sha_fully_guarded(self) -> bool:
        """May a caller stamp :attr:`merge_sha` as provenance as-is?

        The ONE question a provenance writer has, asked once here instead
        of requiring every consumer to know the current list of weakenings
        (heuristic 11) — a future third weakening updates this expression
        rather than silently breaking every caller.

        False on ``landed_unconfirmed``, ``no_signal`` and ``probe_failed``
        because there is no sha to stamp at all; that is NOT a claim that the
        branch did not land.

        A ``patch_id``-arm ``found_on_main`` is fully guarded: that path is
        checked by containment rather than effect survival, which is
        ``docs/prds/landed-not-done-recovery.md`` D1's contract and not a
        weakening.
        """
        return (
            self.outcome is GitAuthorityOutcome.found_on_main
            and not self.citation_gate_skipped
            and not self.rescued_by_delivered_checks
        )


async def _evidence_verdict(
    git_ops: Any,
    tid: str,
    full_branch: str,
    arm: GitAuthorityArm,
    fetch: TaskMetadataResult,
    *,
    branch_tip_sha: str | None,
    candidate_sha: str | None = None,
    pattern_template: str | None = None,
) -> GitAuthorityVerdict:
    """Run the landing-evidence gate for one arm and render its verdict.

    :func:`_ancestor_arm` and :func:`_marker_arm` end identically once their
    own guard has passed — validate, then accept or report the landing as
    unconfirmed — so that sequence lives here ONCE (heuristic 11) and the arms differ
    only in the ``arm`` they label and the evidence keywords they pass.
    ``candidate_sha=None`` selects ``validate_landing_evidence``'s DISCOVERY
    mode, where a commit on main must positively cite the task (FIX 2) AND
    its effect must still be present at main HEAD (FIX 1', the task-1175
    reverted-landing guard); passing a marker selects CANDIDATE mode, where
    the marker's subject match already establishes attribution so only the
    effect-present guard remains.  Neither arm escalates on reject — this
    mirrors the harness ancestor arm's silent-False, and ``merge_status`` is
    a read-only probe with no write side.

    ``delivered_checks`` is THREE-STATE (task 4498): ``None`` means "this
    call site is unwired", ``[]`` means "wired, and this task declares no
    checks".  Both arms ARE wired, so ``or []`` never degrades to ``None`` —
    the capstone (task 4500) reads ``probe['delivered_checks_state']`` to
    find sites that regressed to the default.  ``[]`` is also the fail-safe
    direction: the checks are consulted ONLY on the ``effect_absent`` reject
    path and can only ever UPGRADE a rejection to an acceptance, so an empty
    list simply leaves that second accept path unreachable — exactly today's
    behaviour, and a failed metadata fetch (which yields ``{}``, hence
    ``[]``) cannot fabricate a confident ``done``.  For a task that DOES
    declare checks the wiring is a real change: this is the first production
    call site ever to pass them, so that second accept path is LIVE here,
    and an accept may now carry an effect that is ABSENT at main HEAD.
    :attr:`GitAuthorityVerdict.rescued_by_delivered_checks` is what reports
    it.  The awkward cell — a
    FAILED fetch reported as 'none_declared' rather than as genuinely-empty
    — is resolved OUT OF BAND by :attr:`TaskMetadataResult.unavailable`, not
    by abusing the third state: the parameter has no fourth value, and
    sending ``None`` here would trade a mild probe-label inaccuracy for a
    false "unwired" claim in operator-facing prose.

    ``fetch`` is taken whole rather than as a bare ``metadata`` dict so the
    "``{}`` is not proof" invariant travels with the data, and so the
    unavailability flag reaches the verdict without a second parameter.

    The runtime-only reverse import is repeated here rather than threaded in
    from :func:`probe_landing` — see the module docstring for why it must
    stay inside a function body.  Resolving it per call is also what lets
    ``escalation/tests/test_merge_status_git_authority.py`` observe the call
    contract by patching ``orchestrator.merge_lane.landing_evidence``, the DEFINING
    module, rather than this one.
    """
    from orchestrator.merge_lane.landing_evidence import (
        validate_landing_evidence,
    )
    verdict = await validate_landing_evidence(
        git_ops, tid, full_branch,
        branch_tip_sha=branch_tip_sha,
        candidate_sha=candidate_sha,
        pattern_template=pattern_template,
        delivered_checks=fetch.metadata.get('delivered_checks') or [],
    )
    # Plain `str`, never LandingReason — see GitAuthorityVerdict.reason.
    reason = None if verdict.reason is None else str(verdict.reason)
    # `accepted` implies a non-None evidence_sha (see LandingEvidenceVerdict),
    # but assert it explicitly: found_on_main_response's merge_sha is a hard
    # `str`, and a contract violation must degrade to landed_unconfirmed
    # rather than emit a `done` with a null sha.
    if verdict.accepted and verdict.evidence_sha is not None:
        # EQUALITY, never truthiness: 'confirmed' is the only accept-path
        # spelling that means "the effect-present guard REJECTED and the
        # three-leg differential rescued it".  On a fully-guarded accept the
        # key is ABSENT, and 'no_signal'/'disabled'/the error case all return
        # False from the differential and therefore reach _reject, never here
        # — so a truthiness test would read 'no_signal' as a rescue.  See
        # orchestrator/src/orchestrator/landing_evidence.py::_delivered_checks_differential.
        return GitAuthorityVerdict(
            outcome=GitAuthorityOutcome.found_on_main,
            merge_sha=verdict.evidence_sha, arm=arm, reason=reason,
            evidence_sha=verdict.evidence_sha,
            metadata_unavailable=fetch.unavailable,
            rescued_by_delivered_checks=(
                (verdict.probe or {}).get('delivered_checks_outcome') == 'confirmed'
            ),
        )
    # NOT no_signal.  git POSITIVELY established this branch's work is on
    # main — its tip is an ancestor of main, or a marker survived the
    # predates-this-incarnation veto — and only attribution or
    # effect-survival failed.  evidence_sha comes from probe['citation']
    # because a REJECTED verdict carries evidence_sha is None STRUCTURALLY
    # — landing_evidence.py::_reject_verdict, "a rejected verdict carrying
    # a sha is an invalid state".
    return GitAuthorityVerdict(
        outcome=GitAuthorityOutcome.landed_unconfirmed, arm=arm, reason=reason,
        evidence_sha=(verdict.probe or {}).get('citation'),
        metadata_unavailable=fetch.unavailable,
    )


async def _marker_arm(
    git_ops: Any,
    tid: str,
    full_branch: str,
    fetch: TaskMetadataResult,
    *,
    marker: str,
) -> GitAuthorityVerdict:
    """Judge a merge marker found on main for a deleted branch ref.

    Predates-this-incarnation veto (task 3103, mirroring the harness marker
    arm): when the branch was deleted and recreated under the SAME task id, a
    marker older than this incarnation's ``branch_base_sha`` is a previous
    run's merge, so it establishes nothing about this one.  Otherwise the
    marker is the candidate :func:`_evidence_verdict` checks.
    """
    from orchestrator.merge_lane.landing_evidence import is_valid_sha_40
    branch_base_sha = fetch.metadata.get('branch_base_sha')
    # Validity is checked first so a missing or malformed base never reaches
    # is_ancestor as an argument.
    if is_valid_sha_40(branch_base_sha) and await git_ops.is_ancestor(
        marker, branch_base_sha,
    ):
        return GitAuthorityVerdict(
            outcome=GitAuthorityOutcome.no_signal, metadata_unavailable=fetch.unavailable,
        )
    return await _evidence_verdict(
        git_ops, tid, full_branch, GitAuthorityArm.marker, fetch,
        branch_tip_sha=None, candidate_sha=marker,
    )


async def _ancestor_arm(
    git_ops: Any,
    tid: str,
    full_branch: str,
    fetch: TaskMetadataResult,
    *,
    branch_tip_sha: str,
    pattern: str | None,
) -> GitAuthorityVerdict:
    """Judge a live branch whose tip is an ancestor of main but not main's tip.

    The degeneracy guard (task 3103) runs FIRST and independently of the
    citation gate.  A tip still equal to the recorded ``branch_base_sha``
    means no commit was ever pushed past the creation point: the branch is
    parked at an OLD main commit, which passes both dispatch checks, and a
    citing commit on main would otherwise stamp a confident ``done`` against
    a commit holding none of the task's work (reify 5493).  It judges the
    SAME tip the dispatch's ``is_ancestor`` ran on (task 3103 review #2), so a
    warm-lane reseed cannot split the two observations.

    *pattern* is ``git.commit_citation_pattern``.  ``''`` is the documented
    per-project opt-out: ``find_task_citation_commit`` then returns None for
    everything, so running the gate would reject unconditionally and turn
    this arm into dead code.  The opt-out instead returns the raw BRANCH TIP
    with ``citation_gate_skipped`` set (see :func:`found_on_main_response`,
    Exception 1).  ``None`` means "use the built-in default pattern" and is
    NOT the opt-out.
    """
    from orchestrator.merge_lane.landing_evidence import branch_is_degenerate
    if await branch_is_degenerate(
        git_ops, full_branch, fetch.metadata, branch_tip_sha=branch_tip_sha,
    ):
        return GitAuthorityVerdict(
            outcome=GitAuthorityOutcome.no_signal, metadata_unavailable=fetch.unavailable,
        )
    if pattern == '':
        return GitAuthorityVerdict(
            outcome=GitAuthorityOutcome.found_on_main, merge_sha=branch_tip_sha,
            arm=GitAuthorityArm.ancestor, citation_gate_skipped=True,
            metadata_unavailable=fetch.unavailable,
        )
    return await _evidence_verdict(
        git_ops, tid, full_branch, GitAuthorityArm.ancestor, fetch,
        branch_tip_sha=branch_tip_sha, pattern_template=pattern,
    )


async def _patch_id_arm(
    git_ops: Any,
    tid: str,
    full_branch: str,
    fetch: TaskMetadataResult,
    *,
    branch_tip_sha: str,
    pattern: str | None,
    site: str,
) -> GitAuthorityVerdict:
    """Decide the patch-id arm through ``branch_work_landed`` and render its verdict.

    The producer owns the whole decision and its normative guard order
    (``orchestrator/src/orchestrator/merge_lane/landing_evidence.py::branch_work_landed``);
    this only maps its ``LandingReason`` onto a :class:`GitAuthorityOutcome`:
    ``no_attribution`` (containment proven, nothing attributable) is
    ``landed_unconfirmed``; ``git_error`` (the detector could not look) is
    ``probe_failed``, never a negative; every other reject establishes no
    landing.  The mapping stays inside the function because a module-level
    table keyed by ``LandingReason`` would hoist the reverse import, which is
    runtime-only for the same reason as :func:`_evidence_verdict`'s.  No
    escalation queue is passed: ``None`` is the producer's documented
    supported shape, and this tier is a read-only probe.  *pattern* is
    forwarded so this arm and :func:`_ancestor_arm` apply one citation policy.
    """
    from orchestrator.merge_lane.landing_evidence import LandingReason, branch_work_landed
    verdict = await branch_work_landed(
        git_ops, tid, full_branch, branch_tip_sha=branch_tip_sha, metadata=fetch.metadata,
        pattern_template=pattern,
    )
    reason = str(verdict.reason)
    if verdict.accepted and verdict.evidence_sha is not None:
        return GitAuthorityVerdict(
            outcome=GitAuthorityOutcome.found_on_main,
            merge_sha=verdict.evidence_sha, arm=GitAuthorityArm.patch_id, reason=reason,
            evidence_sha=verdict.evidence_sha, metadata_unavailable=fetch.unavailable,
        )
    if verdict.reason == LandingReason.no_attribution:
        outcome = GitAuthorityOutcome.landed_unconfirmed
    elif verdict.reason == LandingReason.git_error:
        logger.warning(
            '%s: patch-id arm could not answer for %s (git_error at stage %s) '
            '— returning probe_failed',
            site, full_branch, verdict.probe.get('git_error_stage'),
        )
        outcome = GitAuthorityOutcome.probe_failed
    else:
        outcome = GitAuthorityOutcome.no_signal
    return GitAuthorityVerdict(
        outcome=outcome, arm=GitAuthorityArm.patch_id,
        reason=reason, metadata_unavailable=fetch.unavailable,
    )


async def probe_landing(
    git_ops: Any,
    key: str,
    *,
    orch_config: Any,
    harness: Any,
    site: str = 'merge_status',
) -> GitAuthorityVerdict:
    """Ask git whether *key*'s branch landed on main, and how confidently.

    *key* is a branch name or a bare task id; the ref is derived through
    ``canonical_queued_branch_name`` and the task id through
    ``bare_queued_branch_id``.  Returns a VERDICT, not an MCP
    response — which is what lets a periodic writer (PRD leaf η, task 4651)
    reuse the fully-guarded decision ``merge_status`` computes and throws
    away.  It is also why ``request_id`` is not a parameter: it was only
    ever used to build the response, so moving the rendering out to
    :func:`found_on_main_response` removed it from this body entirely.

    Three arms, one per :class:`GitAuthorityArm`: a gone ref goes to the
    marker search (:func:`_marker_arm`); a live ref not at main's tip goes to
    :func:`_ancestor_arm` when its tip is an ancestor of main, and otherwise
    to :func:`_patch_id_arm`, whose guards (degeneracy, empty net diff) and
    containment check all belong to
    ``orchestrator/src/orchestrator/merge_lane/landing_evidence.py::branch_work_landed``.
    A ref sitting at exactly main's tip is ``no_signal``.  Every arm that
    reads task metadata is handed it already fetched, so the fetch's
    ``unavailable`` flag reaches the verdict even when the arm then raises.

    ``orch_config`` and ``harness`` are KEYWORD-ONLY because all three
    collaborators are typed ``Any`` and are therefore interchangeable to a
    type checker: a transposed pair would raise inside the blanket
    ``except`` below and degrade to ``probe_failed`` on every call, i.e. a
    fresh consumer that can never answer.  Naming them at the call site is
    what makes that mistake unwritable.

    ``site`` labels the CALLER in every degradation this probe logs, both
    its own and ``task_metadata``'s (task 3103 review #3: an unlabelled
    fault is invisible to an operator grepping for the tool they actually
    invoked).  ``merge_status`` takes the default; every other consumer —
    leaf η's periodic writer first — passes its own.

    FIRE-SAFE: every failure — including an ImportError from an arm's
    runtime reverse import — degrades to ``probe_failed`` and is logged; this
    never raises.  The wrapper lives HERE rather than at the call site so
    every consumer inherits the fire-safety instead of re-adding it.  A
    clean miss is ``no_signal``, never ``probe_failed``.
    """
    metadata_unavailable = False

    def _verdict(outcome: GitAuthorityOutcome, **fields: Any) -> GitAuthorityVerdict:
        return GitAuthorityVerdict(
            outcome=outcome, metadata_unavailable=metadata_unavailable, **fields,
        )

    try:
        git_config = orch_config.git
        full_branch = canonical_queued_branch_name(key, git_config.branch_prefix)
        tip = await git_ops.resolve_branch_sha(full_branch)
        main_tip = await git_ops.resolve_branch_sha(git_config.main_branch)
        tid = bare_queued_branch_id(key, git_config.branch_prefix)
        if tip is None:
            # Branch ref gone — the canonical 4352 deleted-branch shape.
            # find_merge_marker gates on branch existence internally.
            marker = await git_ops.find_merge_marker(full_branch)
            if marker is None:
                return _verdict(GitAuthorityOutcome.no_signal)
            fetch = await task_metadata(harness, tid, site=site)
            metadata_unavailable = fetch.unavailable
            return await _marker_arm(git_ops, tid, full_branch, fetch, marker=marker)
        if tip == main_tip:
            # A commit is its own ancestor, so a branch at exactly main's HEAD
            # passes is_ancestor while nothing has been merged.
            return _verdict(GitAuthorityOutcome.no_signal)
        tip_is_ancestor = await git_ops.is_ancestor(tip, git_config.main_branch)
        fetch = await task_metadata(harness, tid, site=site)
        metadata_unavailable = fetch.unavailable
        if tip_is_ancestor:
            return await _ancestor_arm(
                git_ops, tid, full_branch, fetch,
                branch_tip_sha=tip, pattern=git_config.commit_citation_pattern,
            )
        return await _patch_id_arm(
            git_ops, tid, full_branch, fetch, branch_tip_sha=tip,
            pattern=git_config.commit_citation_pattern, site=site,
        )
    except Exception:
        logger.warning(
            '%s: git-authority probe failed for %s — returning probe_failed',
            site, key, exc_info=True,
        )
        return _verdict(GitAuthorityOutcome.probe_failed)
