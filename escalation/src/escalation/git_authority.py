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
``orchestrator.landing_evidence`` imports are RUNTIME-ONLY, made inside
function bodies reached from the fire-safe wrapper in :func:`probe_landing`:
``escalation/pyproject.toml`` declares no orchestrator dependency, and the
import resolves at runtime only because the escalation server is hosted inside
the orchestrator process.  Hoisting either to module level would convert that
degradation into an import-time crash.  ``shared.*`` imports are intra-workspace
and unconstrained.

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

from shared.branch_names import canonical_queued_branch_name
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

    NOT to be confused with the unrelated ``'no_signal'`` that
    ``orchestrator.landing_evidence`` writes into
    ``probe['delivered_checks_outcome']``, which means "the task's declared
    delivered-checks produced no confirming signal" — a statement about one
    sub-probe, not about whether a landing was established.  The two never
    share an object (a verdict here carries ``reason``/``evidence_sha``, not
    ``probe``), so always read a bare ``no_signal`` with its owner attached.
    """

    found_on_main = 'found_on_main'
    landed_unconfirmed = 'landed_unconfirmed'
    no_signal = 'no_signal'


class GitAuthorityArm(enum.StrEnum):
    """Which git signal a verdict was reached through.

    A ``StrEnum`` for the same reason :class:`GitAuthorityOutcome` is one
    (heuristic 12, structured data instead of meaningful strings): ``arm`` is
    a closed two-member set a consumer switches on, not free text.
    ``ancestor`` — the branch ref still exists and its tip is an ancestor of
    main; ``marker`` — the ref is gone and a merge marker for it was found on
    main.  The two arms carry different evidence and different guards, which
    is why the verdict names the one it came from.
    """

    ancestor = 'ancestor'
    marker = 'marker'


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
    indistinguishable from a live one (review #4).  A caller detects that
    case from :attr:`GitAuthorityVerdict.citation_gate_skipped` rather
    than by inspecting config.  The opt-out is ``''`` only; ``None`` means
    "use the built-in default pattern" and keeps the full guarantee.
    Both SKILL.md runbooks carry the same exception.
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
        site: The calling tool or writer (``'merge_status'`` /
            ``'merge_request'``), interpolated into the degradation
            warning.  Without it a scheduler fault on the SUBMIT path was
            logged as a merge_status failure, so an operator grepping for a
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


@dataclass(frozen=True)
class GitAuthorityVerdict:
    """What git authority established about one branch, and on what evidence.

    ``landed_unconfirmed`` and ``no_signal`` are the whole reason this is a
    verdict rather than a boolean.  ``landed_unconfirmed`` means git
    POSITIVELY ESTABLISHED that this branch's work is on main — the branch
    is an ancestor of main, or a merge marker for it was found — and only
    ATTRIBUTION or EFFECT-SURVIVAL could not be confirmed.  ``no_signal``
    means no landing was established at all.  A caller that cannot tell them
    apart cannot tell "we found the landing but could not verify it" from
    "we found nothing".

    ``merge_status`` renders BOTH identically today, as the bare Tier-4
    ``unknown``.  That is deliberate and behaviour-preserving: the
    ``merge_status`` response vocabulary belongs to task **4831**
    (``plans/merge-status-durable-non-landed-prd.md`` label β), which maps
    these outcomes onto its epistemic states and reason codes.  This
    dataclass is that task's INPUT, and leaf η's (task 4651).

    ``merge_sha`` is populated on ``found_on_main`` ONLY — it is the sha a
    caller may stamp as provenance, WITH ONE EXCEPTION that this dataclass
    carries structurally rather than leaving to prose: when
    ``citation_gate_skipped`` is True the project has opted out of citation
    checking (``git.commit_citation_pattern == ''``) and ``merge_sha`` is
    the raw BRANCH TIP — not a commit discovered on main, and not
    effect-present-checked, so a reverted landing is indistinguishable from
    a live one.  A caller stamping provenance must branch on that flag; do
    NOT infer the case from ``evidence_sha`` being None, which is a
    coincidence of that path and not a documented signal.
    :func:`found_on_main_response` states the full caveat once.

    ``evidence_sha`` is the weaker thing: the commit the probe was reasoning
    ABOUT, and it is present on rejects too.

    ``reason`` stays a plain ``str | None`` carrying the landing verdict's
    own spelling rather than
    ``orchestrator.landing_evidence.LandingReason``, because importing that
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

    Both arms of :func:`probe_landing` end identically once their own guard
    has passed — validate, then accept or report the landing as unconfirmed
    — so that sequence lives here ONCE (heuristic 11) and the arms differ
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
    ``[]``) cannot fabricate a confident ``done``.  The awkward cell — a
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
    contract by patching ``orchestrator.landing_evidence``, the DEFINING
    module, rather than this one.
    """
    from orchestrator.landing_evidence import (  # type: ignore[reportMissingImports]
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
    # `str`, and a contract violation must degrade to the Tier-4 unknown
    # rather than emit a `done` with a null sha.
    if verdict.accepted and verdict.evidence_sha is not None:
        return GitAuthorityVerdict(
            outcome=GitAuthorityOutcome.found_on_main,
            merge_sha=verdict.evidence_sha, arm=arm, reason=reason,
            evidence_sha=verdict.evidence_sha,
            metadata_unavailable=fetch.unavailable,
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


async def probe_landing(
    git_ops: Any,
    key: str,
    *,
    orch_config: Any,
    harness: Any,
    site: str = 'merge_status',
) -> GitAuthorityVerdict:
    """Ask git whether *key*'s branch landed on main, and how confidently.

    *key* is a branch name or a bare task id; it is normalised through
    ``canonical_queued_branch_name``.  Returns a VERDICT, not an MCP
    response — which is what lets a periodic writer (PRD leaf η, task 4651)
    reuse the fully-guarded decision ``merge_status`` computes and throws
    away.  It is also why ``request_id`` is not a parameter: it was only
    ever used to build the response, so moving the rendering out to
    :func:`found_on_main_response` removed it from this body entirely.

    ``orch_config`` and ``harness`` are KEYWORD-ONLY because all three
    collaborators are typed ``Any`` and are therefore interchangeable to a
    type checker: a transposed pair would raise inside the blanket
    ``except`` below and degrade SILENTLY to ``no_signal``, i.e. a fresh
    consumer reporting that nothing has ever landed.  Naming them at the
    call site is what makes that mistake unwritable.

    ``site`` labels the CALLER in every degradation this probe logs, both
    its own and ``task_metadata``'s (task 3103 review #3: an unlabelled
    fault is invisible to an operator grepping for the tool they actually
    invoked).  ``merge_status`` takes the default; every other consumer —
    leaf η's periodic writer first — passes its own.

    FIRE-SAFE: every failure — including an ImportError from the runtime
    reverse import below — degrades to ``no_signal`` and is logged; this
    never raises.  The wrapper lives HERE rather than at the call site so
    every consumer inherits the fire-safety instead of re-adding it.
    """
    metadata_unavailable = False

    def _verdict(outcome: GitAuthorityOutcome, **fields: Any) -> GitAuthorityVerdict:
        return GitAuthorityVerdict(
            outcome=outcome, metadata_unavailable=metadata_unavailable, **fields,
        )

    try:
        prefix = orch_config.git.branch_prefix
        full_branch = canonical_queued_branch_name(key, prefix)
        tip = await git_ops.resolve_branch_sha(full_branch)
        main_tip = await git_ops.resolve_branch_sha(orch_config.git.main_branch)
        tid = full_branch.removeprefix(prefix)
        # Runtime-only reverse import: orchestrator depends on escalation,
        # not vice versa, so this lazy import deliberately avoids a static
        # cycle (same shape as server.py:1423 / :2049 / :2148).  It resolves
        # at runtime because the escalation server is hosted inside the
        # orchestrator process.  An ImportError is an Exception and therefore
        # already degrades to the honest Tier-4 unknown via the wrapper below.
        from orchestrator.landing_evidence import (  # type: ignore[reportMissingImports]
            branch_is_degenerate,
            is_valid_sha_40,
        )
        if (tip is not None and tip != main_tip
                and await git_ops.is_ancestor(tip, orch_config.git.main_branch)):
            # Live branch is already an ancestor of main (normal merged case).
            # tip != main_tip guards against the no-op case: a branch sitting at
            # exactly main's HEAD satisfies is_ancestor trivially (a commit is
            # its own ancestor) but nothing has been merged.
            # Degeneracy guard (task 3103): a tip still equal to the
            # recorded branch_base_sha proves ZERO commits were ever
            # pushed beyond the creation point.  Such a branch is parked
            # at an OLD main commit, which makes it an ancestor of main
            # AND distinct from main_tip — both conjuncts above pass — so
            # without this guard the tier stamps a confident `done`
            # against a commit containing none of the task's work.  A
            # degenerate branch falls through to the honest Tier-4
            # unknown.  Runs FIRST and independently of the citation gate:
            # a degenerate branch whose task DOES have a citing commit on
            # main (reify 5493) is caught only by this ordering.
            # branch_tip_sha=tip: the probe judges degeneracy against the
            # SAME tip the is_ancestor check above just ran on, instead of
            # re-reading the ref (review #2) — one subprocess fewer, and no
            # window for a warm-lane reseed to split the two observations.
            fetch = await task_metadata(harness, tid, site=site)
            metadata_unavailable = fetch.unavailable
            if not await branch_is_degenerate(
                git_ops, full_branch, fetch.metadata, branch_tip_sha=tip,
            ):
                # Citation gate.  Read the pattern off orch_config.git for
                # consistency with the adjacent .main_branch / .branch_prefix
                # reads (same object as git_ops.config in production).
                pattern = orch_config.git.commit_citation_pattern
                if pattern == '':
                    # Documented per-project opt-out (config.py
                    # commit_citation_pattern): '' disables the citation
                    # check entirely for projects without citation
                    # conventions, and find_task_citation_commit honours it
                    # by returning None for EVERYTHING.  Running the gate
                    # here would therefore reject unconditionally and turn
                    # Tier 3.5 into dead code rather than merely un-gated —
                    # a silent capability loss for an explicit opt-in.
                    # Note: None means "use the built-in
                    # DEFAULT_COMMIT_CITATION_PATTERN" and is NOT the
                    # opt-out.  The degeneracy guard above still applies.
                    # merge_sha is therefore the raw BRANCH TIP, and
                    # citation_gate_skipped is what tells a caller so —
                    # see GitAuthorityVerdict and found_on_main_response.
                    return _verdict(
                        GitAuthorityOutcome.found_on_main, merge_sha=tip,
                        arm=GitAuthorityArm.ancestor, citation_gate_skipped=True,
                    )
                return await _evidence_verdict(
                    git_ops, tid, full_branch, GitAuthorityArm.ancestor, fetch,
                    branch_tip_sha=tip, pattern_template=pattern,
                )
        elif tip is None:
            # Branch ref gone — the canonical 4352 deleted-branch shape.
            # find_merge_marker internally gates on branch existence so it only
            # fires when the ref is gone (consistent with the cheaper-common-path
            # ordering: cheaper is_ancestor check first, find_merge_marker only
            # when the branch has been deleted).
            marker = await git_ops.find_merge_marker(full_branch)
            if marker is not None:
                fetch = await task_metadata(harness, tid, site=site)
                metadata_unavailable = fetch.unavailable
                branch_base_sha = fetch.metadata.get('branch_base_sha')
                # Predates-this-incarnation veto (task 3103, mirroring
                # the harness marker arm): the branch was deleted and
                # recreated under the SAME task id, so a marker older
                # than this incarnation's base attributes a previous
                # run's merge to the current task.  is_valid_sha_40 sits
                # on the LEFT of the `and` so a missing or malformed
                # base never reaches is_ancestor with a bad argument.
                if not (
                    is_valid_sha_40(branch_base_sha)
                    and await git_ops.is_ancestor(marker, branch_base_sha)
                ):
                    return await _evidence_verdict(
                        git_ops, tid, full_branch, GitAuthorityArm.marker, fetch,
                        branch_tip_sha=None, candidate_sha=marker,
                    )
    except Exception:
        logger.warning(
            '%s: git-authority probe failed for %s — returning no_signal',
            site, key, exc_info=True,
        )
    return _verdict(GitAuthorityOutcome.no_signal)
