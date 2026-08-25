"""Shared landing-evidence helper for already-landed re-derivation sites (task 2678).

Five always-on sites re-derive "has this task's work already landed on
main?" from live git state before stamping a task ``done``:

  1. ``Harness._already_landed_dispatch_gate``'s ancestry path
  2. ``Harness._already_landed_dispatch_gate``'s branch-deleted merge-marker path
  3. ``Harness._already_landed_dispatch_gate``'s content-equivalence fallback
  4. the stranded-in-progress sweep (``Harness._reconcile_one_stranded``)
  5. ``SpeculativeMergeWorker._redrive_coalesce_members`` (coalesce re-drive)

Prior to task 2678 each site inlined its own subset of two primitives landed
by task 2675 (dep δ): ``git_ops.find_task_citation_commit`` (FIX 2,
subject-anchored citation discovery) and ``git_ops.commit_effect_present_in_main``
(FIX 1', merge second-parent/octopus-aware effect-still-present check —
since task 3116 a threshold LINE-SURVIVAL test, not byte-identity) — an
inline-per-site shape that let two of the five sites (the merge-marker path
and the coalesce re-drive) ship WITHOUT the effect-present guard at all (the
task-1175 clobber: a reverted merge still read as a genuine landing) and let
two more lean on a silent ``x or <fallback-sha>`` expression that fabricated
provenance when discovery came up empty.

This module is the single, INV-5 extraction point: ONE async function,
:func:`validate_landing_evidence`, that both ``harness.py`` (×4 call sites)
and ``merge_queue.py`` (×1 call site) delegate to.

**Module-level, not a method** — a standalone function taking ``git_ops`` as
its first parameter, deliberately NOT a ``GitOps`` method and NOT a
``Harness`` method:

- ``harness.py`` already imports from ``merge_queue.py``; a helper living in
  either module risks an import cycle from the other. A standalone module
  both simply import (duck-typing ``git_ops``) has none.
- Existing gate-wiring tests construct ``h.git_ops = MagicMock()`` and stub
  its sub-methods (``find_task_citation_commit`` / ``is_ancestor`` /
  ``commit_effect_present_in_main``, and optionally
  ``describe_commit_effect_in_main`` for the task-3116 divergence
  diagnostics). A ``GitOps`` *method* named
  ``validate_landing_evidence`` would auto-mock under that MagicMock and
  silently bypass the real logic under test; a module-level function that
  merely CALLS those same (already-stubbed) sub-methods keeps exercising the
  real, shared decision logic.

**Pure / read-only** — this function never marks a task done, never
escalates, and never mutates git or task state. It returns a frozen
:class:`LandingVerdict` describing whether the evidence is
attributable and effect-present; each call site owns its own stamp-vs-
escalate-vs-revert action, which differs per site (the dispatch gate returns
a bool, the sweep reverts to pending, the coalesce re-drive calls
``redrive_member``).

**THE ONE CARVE-OUT, and it is narrow on purpose** (task 4647): the G7 storm
escape — :class:`LandingTally` and
:func:`file_landing_git_error_storm_escalation` — DOES write to the escalation
queue.  It is a claim about DETECTOR HEALTH, never about a task: it still
never marks a task done, never changes any task's status, and never mutates
git.  It is filed against a synthetic sentinel id rather than a real task
precisely so it cannot be read as a hold on one, which is what keeps the
promise above intact for task state.  The carve-out exists because a landing
detector fails SILENTLY by construction — every verdict a broken one produces
rejects, so an unreadable repo is indistinguishable from a repo with nothing
landed in it — and a purity rule that forbids saying so buys purity by
guaranteeing the failure goes unnoticed.  It is rate-gated, deduped and
kill-switched (``recovery_emission.landing_git_error_escalation_enabled``) so
the write stays one alarm per storm.

**Two modes**, selected by whether ``candidate_sha`` is given:

- **DISCOVERY** (``candidate_sha=None``) — the branch ref is live: discover
  a citation via ``find_task_citation_commit``, classify the effect anchor
  with ``is_ancestor(citation, branch)`` (True → an in-branch work commit,
  anchor on ``branch_tip_sha``; False → this branch's own no-ff merge commit
  OR a divergent/realigned branch ref, anchor on the citation itself), then
  apply the FIX 1' effect-present guard to that anchor. **Task 2870 removed
  the former FIX 2 bidirectional lineage HARD-REJECT** (a citation an
  ancestor of the branch in NEITHER direction used to reject as
  ``lineage_mismatch``): every DISCOVERY caller already establishes the
  branch's content is on main before delegating here, and
  ``find_task_citation_commit`` greps ``git log main`` so its citation is by
  construction reachable from main — branch-ref ancestry is NOT the landing
  authority, so a stale/divergent/realigned branch tip that fails both
  directions is still a genuine landing (reify esc-5252-9); hard-rejecting
  it caused a close↔refile ping-pong that never self-resolved. The FIX 1'
  effect-present guard stays the real gate. The ``branch_tip_sha`` anchor
  for an in-branch work-commit citation handles a stale intermediate commit
  (the branch's actual final state is its tip); the citation anchor for a
  no-ff merge commit is NOT a no-op — task 2675 made
  ``commit_effect_present_in_main`` merge-aware, so it diffs each of the
  merge commit's non-first parents' content against current main HEAD,
  correctly catching a reverted no-ff merge (the task-1175 shape). Used by
  the ancestry path, the content-equivalence fallback, and the coalesce
  re-drive.
- **CANDIDATE** (``candidate_sha`` given) — attribution was already
  established by the caller (a merge-marker subject match, or a stranded-
  sweep ground-truth report): skip citation discovery and the lineage guard
  entirely, and apply ONLY the FIX 1' effect-present guard to
  ``candidate_sha``. Used by the merge-marker path and the stranded-
  in-progress sweep.

A second shared predicate lives here for the same INV-5 reason (task 3103):
:func:`branch_is_degenerate` (with its :func:`is_valid_sha_40` helper) answers
"did this branch ever advance past its recorded creation point?" — the #1226
degeneracy signal that must run BEFORE :func:`validate_landing_evidence` on
every already-landed re-derivation. Both were previously private to
``harness.py``, which left the escalation server's ``merge_status`` Tier-3.5
and ``merge_request`` fast-path unguarded against a branch parked at an OLD
main commit: such a branch IS an ancestor of main and is NOT at main's tip,
so an ancestry-only check answers a confident ``done`` against a commit
containing none of the task's work. Consumers: ``harness.py`` (×3 sites, via
the ``Harness._branch_is_degenerate`` delegation), ``escalation/server.py``
``merge_status`` (both git-authority arms) and ``merge_request``'s
``already_merged`` fast-path.

Also shared here (INV-5): :func:`format_unattributed_landing_detail` renders
a rejected verdict into a human-facing ``(summary, detail)`` pair, and
:func:`file_unattributed_landing_escalation` is the dedup-guarded L1 filing
boilerplate (queue-None guard, ``has_open_l1`` dedup, ``Escalation``
construction) that both ``Harness._file_unattributed_landing_escalation``
and ``SpeculativeMergeWorker._file_unattributed_landing_escalation``
delegate to — the two differ only in the ``agent_role`` they pass.
"""

from __future__ import annotations

import collections
import enum
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, TypeGuard

if TYPE_CHECKING:
    from collections.abc import Callable

    from escalation.queue import EscalationQueue

    from orchestrator.config import RecoveryEmissionConfig
    from orchestrator.git_ops import GitOps

logger = logging.getLogger(__name__)

__all__ = [
    'LANDING_GIT_ERROR_STORM_CATEGORY',
    'LANDING_GIT_ERROR_STORM_SENTINEL',
    'LANDING_TALLY',
    'LandingEvidenceVerdict',
    'LandingMethod',
    'LandingReason',
    'LandingTally',
    'LandingVerdict',
    'branch_is_degenerate',
    'branch_work_landed',
    'file_landing_git_error_storm_escalation',
    'file_unattributed_landing_escalation',
    'format_unattributed_landing_detail',
    'is_valid_sha_40',
    'validate_landing_evidence',
]


class LandingReason(enum.StrEnum):
    """The closed vocabulary of landing-evidence verdict codes.

    Genuine ``str`` members (mirrors ``orchestrator.recovery_emission``'s
    ``RecoverySite`` / ``LeaveReason`` and ``escalation.pins.PinClass``) so
    equality against a plain string holds without an explicit ``.value``, a
    member resolves as a plain-string dict key — which
    :data:`_REASON_EXPLANATIONS` depends on — and a member JSON-encodes as its
    spelling.

    Values are written out LITERALLY rather than produced by ``enum.auto()``.
    That is not style: this task's ``metadata.delivered_checks`` greps this
    module for the literal ``'no_attribution'``, so an ``auto()`` declaration
    would build an identical runtime vocabulary and silently fail the gate.

    ONE vocabulary, not two.  The first six members are the PRD Contract's
    closed set, emitted by :func:`branch_work_landed`.  The last three are the
    PRE-CONTRACT spelling :func:`validate_landing_evidence` still emits, kept
    here rather than in a separate legacy enum because a second vocabulary
    would be two authorities that must be kept in step forever, and because a
    legacy code reaching a formatter that cannot explain it renders
    ``'Unrecognized reason code: ...'`` into an L1 body a human reads.  Leaf
    epsilon retires the legacy three when it repoints the consumers.
    """

    #: Accepted — the branch's work is attributably present on main.
    landed = 'landed'
    #: Rejected — a landing marker exists but the branch's NET contribution is
    #: empty, so the task delivered nothing (the task-1175 shape).
    no_op_landing = 'no_op_landing'
    #: Rejected — the branch's work is genuinely absent from main.
    not_landed = 'not_landed'
    #: Rejected — the work IS on main but no main-reachable commit could be
    #: attributed to this task, so there is nothing to anchor provenance on.
    #: The Contract's rename of the legacy ``no_citation``.
    no_attribution = 'no_attribution'
    #: Rejected — a provisioning-only branch whose tip never advanced past its
    #: recorded ``branch_base_sha`` (#1226).  Patch-id-contained in main by
    #: construction, which is exactly why it needs its own code.
    degenerate_branch = 'degenerate_branch'
    #: Rejected — git could not answer.  A FAIL-SOFT DEGRADATION and the one
    #: code whose repetition means THE DETECTOR is broken rather than the task
    #: being unlanded.  No consumer may collapse it into ``not_landed``.
    git_error = 'git_error'

    #: Legacy (``validate_landing_evidence`` accept).  Epsilon retires it.
    ok = 'ok'
    #: Legacy (``validate_landing_evidence`` DISCOVERY miss).  The Contract's
    #: spelling is :attr:`no_attribution`; epsilon flips the emitted value.
    no_citation = 'no_citation'
    #: Legacy (``validate_landing_evidence`` survival reject, FIX 1').
    effect_absent = 'effect_absent'


class LandingMethod(enum.StrEnum):
    """The closed vocabulary of HOW a landing was attributed.

    Genuine ``str`` members, for the same reasons as :class:`LandingReason`.

    This is the explicit mode/policy discriminator the PRD's epsilon bullet
    demands in place of two separate functions: it maps one-to-one onto the
    three production attribution paths, so any consumer can read which policy
    decided a verdict — in particular whether it came from the NON-DECAYING
    patch-id contract or from the legacy effect-present policy epsilon is
    retiring at the landing-detection sites.
    """

    #: :func:`branch_work_landed` — ``git cherry`` patch-id equivalence.  The
    #: non-decaying path.
    patch_id = 'patch_id'
    #: :func:`validate_landing_evidence` CANDIDATE mode — attribution was
    #: already established by the caller's merge-marker subject match.
    merge_marker = 'merge_marker'
    #: :func:`validate_landing_evidence` DISCOVERY mode — attribution by
    #: ``find_task_citation_commit`` subject-anchored citation discovery.
    citation = 'citation'
    #: No attribution path ran: the verdict was hand-constructed rather than
    #: produced by this module (the shape several gate-wiring test files
    #: build).  The default, so existing four-keyword constructions are
    #: unchanged.
    unspecified = 'unspecified'


#: Category used for BOTH the storm filing and its ``has_open_l1`` dedup
#: filter.  Its OWN category, deliberately NOT ``'provenance_unattributed'``:
#: category scoping is load-bearing after the task-4105 incident (see
#: :func:`file_unattributed_landing_escalation`), and a detector-health alarm
#: sharing a category with a provenance defect would let each silently
#: suppress the other.
LANDING_GIT_ERROR_STORM_CATEGORY = 'landing_detector_git_error'

#: The SYNTHETIC task id the storm alarm is filed against.
#:
#: Never a real task id, for two reasons.  A storm spans every task the sweep
#: touched, so any single id would be arbitrary — the alarm is a claim about
#: the DETECTOR, not about one task's work.  And an open L1 on a real task is
#: read by the recovery predicates as a hold, so filing there would deepen the
#: very strand the alarm reports.  Same shape and same reasoning as
#: ``recovery_emission.RECOVERY_VETO_STREAK_SENTINEL_PREFIX``.
LANDING_GIT_ERROR_STORM_SENTINEL = '__landing_git_error_storm__'

#: agent_role stamped on the storm alarm.
_GIT_ERROR_STORM_ROLE = 'orchestrator-landing-evidence'


class LandingTally:
    """A monotonic per-reason count of the verdicts this module has produced.

    **Why a counter at all.** ``git_error`` is the one landing reason whose
    REPETITION means the DETECTOR is broken rather than the task being
    unlanded, and a broken detector is SILENT BY CONSTRUCTION: every verdict
    it produces rejects, so a repo whose git is unreadable looks exactly like
    a repo with nothing landed in it.  Nothing downstream can tell those apart
    from a single verdict — only the RATE can.  This is the object that makes
    that rate observable, and :func:`file_landing_git_error_storm_escalation`
    is the escape hatch it feeds.

    **Two different questions, deliberately kept apart.**

    - :meth:`snapshot` answers "what has this detector said, ever".  It is
      MONOTONIC: no count ever decreases, so an operator reading it late still
      sees what happened early.
    - :meth:`git_error_count_in_window` answers "is it failing RIGHT NOW".  It
      slides, because a latched alarm that never clears is one that gets
      ignored rather than fixed.

    Collapsing the two would break whichever question lost.

    The count is keyed from the :class:`LandingReason` ENUM rather than from a
    literal list, so a reason added to the vocabulary is tallied the day it
    ships.  A reason that is not a member is dropped LOUDLY (a WARNING) rather
    than silently widening the keyspace — the closed keyspace is what lets a
    reader trust that a zero row means "never happened" rather than "never
    counted".

    **Deliberately NOT thread-safe.**  The orchestrator runs a single event
    loop and every caller of this module is awaited on it, so a lock here
    would buy nothing and would only add a way for the counter to deadlock a
    recovery sweep.  If a second loop ever charges it, the counts drift by at
    most the interleaved increments — a telemetry inaccuracy, never a wrong
    verdict, because nothing reads the tally to DECIDE a landing.
    """

    #: The trailing span the ``git_error`` rate is measured over.  One hour,
    #: matching the ``landing_git_error_rate_per_hour`` config leaf's units.
    DEFAULT_WINDOW_SECS = 3600.0

    #: Hard cap on retained stamps, so a pathological storm cannot grow the
    #: deque without bound between trims.  Far above any threshold an operator
    #: would set; it exists as a memory backstop, not as a policy.
    DEFAULT_MAX_STAMPS = 4096

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        window_secs: float = DEFAULT_WINDOW_SECS,
        max_stamps: int = DEFAULT_MAX_STAMPS,
    ) -> None:
        #: Injectable so the window can be driven deterministically in tests.
        #: ``time.monotonic`` and not ``time.time``: an NTP step backwards
        #: would otherwise appear as a burst of stamps inside the window.
        self._clock = clock
        self.window_secs = window_secs
        self._counts: collections.Counter[LandingReason] = collections.Counter(
            dict.fromkeys(LandingReason, 0),
        )
        self._git_error_stamps: collections.deque[float] = collections.deque(
            maxlen=max_stamps,
        )

    def record(self, reason: LandingReason) -> None:
        """Charge one verdict.  Called for EVERY verdict, accepted or not."""
        try:
            key = LandingReason(reason)
        except ValueError:
            logger.warning(
                'landing tally: unrecognised reason %r was not counted; the '
                'tally keyspace is closed over LandingReason by design',
                reason,
            )
            return
        self._counts[key] += 1
        if key is LandingReason.git_error:
            self._git_error_stamps.append(self._clock())
            self._trim()

    def _trim(self) -> None:
        cutoff = self._clock() - self.window_secs
        while self._git_error_stamps and self._git_error_stamps[0] <= cutoff:
            self._git_error_stamps.popleft()

    def git_error_count_in_window(self) -> int:
        """How many ``git_error`` verdicts fall in the trailing window."""
        self._trim()
        return len(self._git_error_stamps)

    def snapshot(self) -> dict[LandingReason, int]:
        """A COPY of the cumulative per-reason counts.

        A copy so a caller that mutates what it got back — an escalation body
        builder, say — cannot corrupt the counter it is describing.
        """
        return dict(self._counts)

    def render(self) -> str:
        """One grep-friendly line, every reason present including the zeros.

        Fixed shape on purpose: a row that disappears when its count is zero
        makes ``grep`` answer "no such reason" and "never happened"
        identically.
        """
        return ' '.join(f'{reason}={self._counts[reason]}' for reason in LandingReason)


#: The process-lifetime tally every :func:`branch_work_landed` verdict charges.
#:
#: Module-level rather than per-call because the question it answers — "is this
#: detector failing repeatedly?" — is not a property of any one call.  Tests
#: replace this attribute with a fresh instance for isolation.
LANDING_TALLY = LandingTally()


def _resolve_recovery_emission(
    recovery_emission: RecoveryEmissionConfig | None,
) -> RecoveryEmissionConfig:
    """The caller's live config submodel, or the shipped defaults.

    Lazily imported so this module stays importable without pulling in the
    config stack, and constructed from ``RecoveryEmissionConfig`` rather than
    from local literals so the thresholds have exactly ONE authority.  A
    second copy of ``10`` here would drift from the stanza the operator edits.
    """
    if recovery_emission is not None:
        return recovery_emission
    from orchestrator.config import RecoveryEmissionConfig as _Config  # noqa: PLC0415

    return _Config()


def _observe_landing_verdict(
    verdict: LandingVerdict,
    *,
    escalation_queue: EscalationQueue | None = None,
    recovery_emission: RecoveryEmissionConfig | None = None,
) -> None:
    """Tally, log, and rate-gate the storm alarm for one verdict.

    Wholly best-effort: :func:`branch_work_landed` promises never to raise, and
    an alarm that cannot be recorded must not be allowed to destroy the verdict
    it was describing.  A recovery sweep that dies because its own telemetry
    failed stops recovering every OTHER task in the same pass.
    """
    try:
        tally = LANDING_TALLY
        tally.record(verdict.reason)
        # Logged EVERY pass, at INFO: a counter nobody can read without a
        # dashboard is a second silence layered on the first.
        logger.info('landing tally (cumulative, per reason): %s', tally.render())
        if verdict.reason is not LandingReason.git_error:
            return
        config = _resolve_recovery_emission(recovery_emission)
        if not config.landing_git_error_escalation_enabled:
            return
        file_landing_git_error_storm_escalation(
            escalation_queue,
            tally=tally,
            rate_per_hour=config.landing_git_error_rate_per_hour,
        )
    except Exception:
        logger.warning(
            'landing tally/storm-escape failed (non-fatal); the verdict is '
            'unaffected',
            exc_info=True,
        )


def is_valid_sha_40(s: object) -> TypeGuard[str]:
    """Return True iff *s* is a well-formed 40-char lowercase hex SHA.

    Used to validate ``branch_base_sha`` values read from task metadata
    before comparing them against live git output.  Any non-conforming
    value is treated as missing so the reconciler falls through to the
    existing citation-grep guard rather than making a bogus comparison.
    """
    return (
        isinstance(s, str)
        and len(s) == 40
        and all(c in '0123456789abcdef' for c in s)
    )


async def branch_is_degenerate(
    git_ops: GitOps, branch: str, metadata: dict[str, Any],
    *, branch_tip_sha: str | None = None,
) -> bool:
    """Return True iff the branch is a provisioning-only degenerate branch.

    A branch is degenerate when its live tip SHA equals the recorded
    branch_base_sha (#1226), meaning zero commits were ever pushed beyond
    the creation point.  Such a branch is still an ancestor of main and is
    still distinct from main's tip, so every ancestry-based already-landed
    check needs this predicate to avoid attributing a foreign commit's
    content to the task.

    Returns False when:
    - branch_base_sha is absent or not a valid 40-hex SHA (backward compat
      for pre-#1226 tasks or tasks whose metadata write failed transiently);
    - resolve_branch_sha returns None (branch ref vanished mid-sweep —
      treat as non-degenerate so the caller falls through to escalate); or
    - the live tip has advanced past the recorded base SHA.

    The metadata check deliberately precedes the git call: an absent or
    malformed base must cost nothing and must never reach a comparison
    against live git output.

    Args:
        git_ops: A ``GitOps`` instance (or a duck-typed stand-in exposing
            ``resolve_branch_sha``).  Not consulted when *branch_tip_sha*
            is supplied.
        branch: The task's branch name (e.g. ``f'task/{task_id}'``).
        metadata: The task's metadata dict, read for ``branch_base_sha``.
        branch_tip_sha: An already-resolved tip for *branch*.  When given,
            it is used verbatim and ``resolve_branch_sha`` is NOT called.
            This is not merely a saved subprocess: a caller that already
            ran an ancestry (or patch-id) check against a tip it observed
            must judge degeneracy against that SAME tip.  Re-reading the
            ref here would let a concurrent warm-lane reseed land between
            the two reads, so a branch whose ancestry was checked at SHA A
            could be judged degenerate/non-degenerate at SHA B and the
            resulting verdict would be anchored on evidence validated for
            a tip that is no longer current (task 3103 review).  Pass
            ``None`` (the default) only when you have not resolved the ref
            yourself — the harness call sites do exactly that.
    """
    branch_base_sha = metadata.get('branch_base_sha')
    if not is_valid_sha_40(branch_base_sha):
        return False
    if branch_tip_sha is None:
        branch_tip_sha = await git_ops.resolve_branch_sha(branch)
    return branch_tip_sha is not None and branch_tip_sha == branch_base_sha


@dataclass(frozen=True)
class LandingVerdict:
    """The verdict of this module's landing-evidence producers.

    ONE verdict type for the whole producer family — :func:`branch_work_landed`
    and :func:`validate_landing_evidence` both return this — so a consumer
    never has to know which produced the verdict in order to read it, and there
    is no second shape that must be kept in step forever.  Formerly named
    ``LandingEvidenceVerdict``; that spelling survives as a module-level alias
    (below) and remains valid.

    Attributes:
        accepted: Whether the evidence is attributable AND effect-present.
        evidence_sha: The commit sha to anchor provenance on when
            ``accepted`` is True (the discovered citation, or
            ``candidate_sha`` in CANDIDATE mode); ``None`` when rejected.
        reason: Machine-readable code — ``'ok'`` when accepted, else one of
            ``'no_citation'`` (DISCOVERY only — no commit on main cites the
            task) or ``'effect_absent'`` (FIX 1': the evidence sha's effect
            is not present at current main HEAD). (Task 2870 removed the
            former ``'lineage_mismatch'`` DISCOVERY reject — a citation
            unreachable from the branch in both directions is now accepted;
            see :func:`validate_landing_evidence`.)
        probe: Structured facts about the check — ``task_id``, ``branch``,
            ``branch_tip_sha``, ``citation`` (the discovered citation or the
            candidate), ``effect_check_sha`` (the sha the effect-present
            guard actually ran against), and ``reason`` — so a caller can
            build a structured-facts escalation without prose-parsing.

            On an ``'effect_absent'`` REJECT ONLY, four further keys carry
            the divergence diagnostics (task 3116; absent on every accept and
            on a ``'no_citation'`` reject):

            - ``diverged_paths`` — the touched paths that no longer match
              main HEAD, as a list.  ``None`` means "could not be
              determined" (the probe itself failed); ``[]`` means
              "determined, and empty".  The two are deliberately DISTINCT —
              collapsing them would render an unprobeable ``git_ops``
              stand-in as a clean no-divergence result.
            - ``effect_failure`` — the probe's structured failure code (see
              :class:`~orchestrator.git_ops.CommitEffectProbe`), or ``None``
              when the re-probe found the effect PRESENT, which means main
              HEAD advanced between the decision and the probe.
            - ``effect_anchor_sha`` — the commit the divergence comparison
              actually ran against (a merge citation is judged against a
              parent, not against ``effect_check_sha`` itself).
            - ``effect_probe_error`` — present only when the probe raised;
              the ``repr`` of the exception.  Recorded rather than swallowed
              so the escalation states plainly that the paths are unknown.
        method: WHICH attribution path produced this verdict — the explicit
            mode discriminator (see :class:`LandingMethod`).
            ``LandingMethod.unspecified`` means no path ran: the verdict was
            hand-constructed rather than produced by this module.  LAST and
            defaulted on purpose, so every existing positional and
            four-keyword construction is unchanged.
    """

    accepted: bool
    evidence_sha: str | None
    reason: LandingReason
    probe: dict[str, Any]
    method: LandingMethod = LandingMethod.unspecified


#: Backward-compatible alias for the pre-rename spelling.  ``harness.py`` and
#: ``merge_queue.py`` import this name and use it as an annotation, and four
#: sibling test files construct it by keyword; the alias keeps every one of
#: them working with zero edits.  Deliberately an ALIAS and not a second
#: dataclass or a subclass — two verdict types that must agree forever is
#: exactly the lockstep duplication this task exists to avoid.
LandingEvidenceVerdict = LandingVerdict


async def _record_effect_divergence(
    git_ops: GitOps, effect_check_sha: str, probe: dict[str, Any],
) -> None:
    """Enrich *probe* with WHICH paths diverged, on the effect_absent reject
    path only (task 3116).

    Awaits ``git_ops.describe_commit_effect_in_main(effect_check_sha)`` and
    writes ``diverged_paths`` / ``effect_failure`` / ``effect_anchor_sha``,
    plus the part-(b) SURVIVAL facts that actually decided the verdict
    (``aggregate_survival`` and its ``added_lines_total`` denominator, the
    worst guarded file and its ratio, the three thresholds applied, and any
    ``vacuous_paths``).

    Carrying the survival numbers is not decoration.  Since part (b),
    ``diverged_paths`` is an explicitly DEMOTED diagnostic — it no longer
    decides anything — so an escalation that printed only the paths would
    show the reader everything except the measurement the rejection was
    actually based on, and invite exactly the "it says diverged, so it was
    reverted" leap this task exists to stop.  The thresholds ride along so a
    later retune is visible in the output instead of silently changing what
    the same numbers mean.

    Called ONLY after the boolean ``commit_effect_present_in_main`` has
    already rejected, and never on the accept path — the decision is the
    bool's, this is only its explanation.  The extra git work is free in
    practice because both calls hit the same ``(commit_sha, main_sha)`` memo.

    Diagnostic-only, so it must NEVER break a gate decision: any exception is
    contained.  But it is NOT swallowed — the failure is recorded into
    ``probe``, which is rendered verbatim into the escalation a human reads,
    and logged at WARNING with a traceback.  That is the repo's
    structured-facts-at-failure / no-silent-fail-soft invariant applied to
    this try/except.  The realistic trigger is a duck-typed ``git_ops``
    stand-in predating this method (the shape seven gate-wiring test files
    construct), which raises AttributeError here.

    On failure ``diverged_paths`` is set to None — "could not be determined",
    deliberately distinct from ``[]`` — so a formatter can never render an
    unprobeable stand-in as a clean no-divergence result.
    """
    try:
        result = await git_ops.describe_commit_effect_in_main(effect_check_sha)
        probe['diverged_paths'] = list(result.diverged_paths)
        probe['effect_failure'] = result.failure
        probe['effect_anchor_sha'] = result.anchor_sha
        probe['aggregate_survival'] = result.aggregate_survival
        probe['added_lines_total'] = result.added_lines_total
        probe['worst_guarded_path'] = result.worst_guarded_path
        probe['worst_guarded_survival'] = result.worst_guarded_survival
        probe['aggregate_threshold'] = result.aggregate_threshold
        probe['per_file_threshold'] = result.per_file_threshold
        probe['per_file_min_added_lines'] = result.per_file_min_added_lines
        probe['vacuous_paths'] = list(result.vacuous_paths)
    except Exception as exc:
        probe['diverged_paths'] = None
        probe['effect_probe_error'] = repr(exc)
        logger.warning(
            'describe_commit_effect_in_main failed for %s — the effect_absent '
            'escalation will report that diverged paths could not be determined',
            effect_check_sha, exc_info=True,
        )


#: Per-check differential verdicts recorded in
#: ``probe['delivered_checks_legs']`` (task 3116).  ``'made_true_by_this_commit'``
#: is the only one that confirms; every other value is explicitly NO SIGNAL,
#: never evidence against a landing.
_DIFFERENTIAL_CONFIRMED = 'made_true_by_this_commit'


def _main_ref(git_ops: GitOps) -> str:
    """The project's main BRANCH NAME, honouring ``config.main_branch``.

    ``main`` is a configurable Pydantic field, not a constant, and a project
    that sets it to anything else would otherwise have the differential's
    third leg permanently dead: an unresolvable ref makes
    ``run_delivered_check`` return ERRORED, which degrades to no_signal and
    silently declines every upgrade, with no diagnostic saying why (amendment
    pass, review finding).

    The BRANCH NAME, deliberately not a sha resolved here — see
    :func:`_delivered_checks_differential`'s docstring for why the third leg
    must ask about HEAD *now*.

    Falls back to ``'main'`` for a duck-typed ``git_ops`` stand-in carrying no
    config (the shape several gate-wiring test files construct), which is the
    pre-amendment behaviour rather than a crash.
    """
    ref = getattr(getattr(git_ops, 'config', None), 'main_branch', None)
    return ref if isinstance(ref, str) and ref else 'main'


async def _differential_parent_ref(
    git_ops: GitOps, effect_check_sha: str, *, anchor_is_branch_tip: bool,
) -> str:
    """The ref that stands for "before this landing" in the differential.

    ``<sha>^1`` for a MERGE-COMMIT or citation anchor: the first parent is
    main-before-the-merge, exactly the pre-merge baseline the three-leg
    sequence needs.

    NOT ``^1`` for a BRANCH-TIP anchor, which DISCOVERY mode selects whenever
    the citation is an in-branch work commit.  There ``tip^1`` is the branch's
    own previous work commit, so the differential would ask "did the branch's
    LAST commit make this true?" instead of "did this BRANCH make it true?".
    For any multi-commit branch whose deliverable landed in an earlier commit
    the parent leg then reads DELIVERED, the sequence fails, and the second
    accept path silently never fires — for precisely the multi-commit shape
    most likely to have tripped the survival heuristic in the first place
    (amendment pass, review finding).  The branch's FORK POINT from main is
    the honest baseline there.

    Falls back to ``^1`` when the fork point cannot be resolved — a stand-in
    git_ops without :meth:`~orchestrator.git_ops.GitOps.merge_base_with_main`,
    a deleted branch, any git failure.  That is the pre-amendment behaviour:
    a weaker baseline that can only cost an upgrade, never manufacture one.
    """
    parent_ref = f'{effect_check_sha}^1'
    if not anchor_is_branch_tip:
        return parent_ref
    resolver = getattr(git_ops, 'merge_base_with_main', None)
    if resolver is None:
        return parent_ref
    try:
        fork_point = await resolver(effect_check_sha)
    except Exception:
        logger.warning(
            'fork-point resolution failed for %s — the delivered-checks '
            'differential falls back to its first parent',
            effect_check_sha, exc_info=True,
        )
        return parent_ref
    return fork_point if isinstance(fork_point, str) and fork_point else parent_ref


async def _delivered_checks_differential(
    git_ops: GitOps,
    effect_check_sha: str,
    delivered_checks: list[dict[str, Any]],
    probe: dict[str, Any],
    *,
    anchor_is_branch_tip: bool = False,
) -> bool:
    """The SECOND accept path: did *effect_check_sha* MAKE a declared
    capability true? (task 3116 part b.)

    Threshold line survival (``commit_effect_present_in_main``) recovers most
    of the false-positive class, but it remains a heuristic over line sets.
    This is orthogonal POSITIVE evidence, and it is the task's OWN declared
    ground truth: run each ``metadata.delivered_checks`` entry at three refs —

        PARENT        the pre-branch baseline: ``<sha>^1`` for a merge-commit
                      or citation anchor, the branch's FORK POINT for a
                      branch-tip anchor (see :func:`_differential_parent_ref`)
        ``<sha>``     the citation itself
        MAIN          current HEAD, named by ``config.main_branch``

    — and confirm when a check was FALSE at the parent, TRUE at the citation
    and TRUE now.  That sequence is the whole signal: a check that merely
    passes at main proves nothing, because the capability may have arrived by
    any other route or have been true all along.  Only the three legs together
    say THIS commit made it true, which no amount of line-set decay can argue
    with.

    **UPGRADE-ONLY.**  This is called from inside the ``effect_absent`` reject
    branch and nowhere else, so it is structurally incapable of rejecting a
    landing that survival accepted — the rule holds by construction rather
    than by care.  That matters because the stored checks are NOT trustworthy
    input: they rot when a path is renamed (erroring forever), they are
    sometimes written broad enough to have been true before the merge, and 2
    of the 458 live entries are ``script`` kind, which takes no ref at all.
    Every one of those degrades to NO SIGNAL.

    ``expect`` is honoured ONE LAYER DOWN and must not be re-applied here.
    :func:`~orchestrator.delivered_checks.run_delivered_check` already
    resolves it (for ``expect='absent'`` it is NO-match that means DELIVERED),
    so the leg pattern reads uniformly for both expects.  Inverting again for
    the 43 stored ``expect='absent'`` checks would read a capability a LATER
    commit removed as proof that this one delivered it — the exact
    double-inversion this docstring exists to prevent.

    The main leg deliberately names the BRANCH (``config.main_branch``, via
    :func:`_main_ref`) rather than a sha resolved here: it asks whether the
    capability holds at HEAD *now*, which is the question, and it matches
    ``run_delivered_check``'s own contract (grep is evaluated against the
    committed tree at *ref*).  The first two legs are immutable history, so a
    HEAD advance mid-differential can only change the freshness of the third,
    not the "this commit made it true" core.

    KILL SWITCH.  Honours ``config.delivered_checks.enabled``, the same flag
    ``Harness._delivered_checks_withhold`` honours for the mark-done gate.
    Without it, disabling the delivered-checks feature would still leave this
    consumer of it running task-declared checks (amendment pass, review
    finding).  A stand-in git_ops carrying no config reads as ENABLED, which
    is the pre-amendment behaviour.

    LAZY import (``# noqa: PLC0415``), mirroring
    :func:`file_unattributed_landing_escalation`'s ``escalation.models``
    import: this module deliberately imports nothing from ``orchestrator`` at
    module level (see the module docstring) because
    ``orchestrator.delivered_checks`` imports ``orchestrator.git_ops``, and a
    module-level edge here would re-create the harness<->merge_queue cycle
    this module was extracted to avoid.

    Writes ``delivered_checks_legs`` (every check's three-leg outcome) and
    ``delivered_checks_outcome`` (``'confirmed'`` / ``'no_signal'``) into
    *probe*, so the escalation can say "capability X was made true by this
    merge" or state plainly that there was no signal.  Returns True only on a
    confirmation; any exception is contained, recorded in
    ``delivered_checks_error`` and logged, and declines the upgrade — the
    fail-safe direction for an accept path is to not accept.
    """
    checks_config = getattr(getattr(git_ops, 'config', None), 'delivered_checks', None)
    if not getattr(checks_config, 'enabled', True):
        # Switched off fleet-wide. Recorded, never silent: an operator who
        # disabled the feature must be able to see in the escalation that the
        # second accept path was not merely unlucky.
        probe['delivered_checks_legs'] = []
        probe['delivered_checks_outcome'] = 'disabled'
        return False

    from orchestrator.delivered_checks import (  # noqa: PLC0415
        DeliveredCheckResult,
        run_delivered_check,
    )

    main_ref = _main_ref(git_ops)
    legs: list[dict[str, Any]] = []
    confirmed = False
    try:
        parent_ref = await _differential_parent_ref(
            git_ops, effect_check_sha, anchor_is_branch_tip=anchor_is_branch_tip,
        )
        probe['delivered_checks_parent_ref'] = parent_ref
        for check in delivered_checks:
            kind = check.get('kind')
            if kind != 'grep':
                # A script check execs against the LIVE CHECKOUT and takes no
                # ref, so it cannot express a differential at all: running it
                # three times would answer identically each time.  Recorded as
                # an explicit carve-out rather than silently dropped.
                legs.append({
                    'name': check.get('name'),
                    'kind': kind,
                    'verdict': (
                        'script_kind_no_signal' if kind == 'script'
                        else 'unsupported_kind_no_signal'
                    ),
                })
                continue
            results = {}
            for label, ref in (
                ('parent', parent_ref),
                ('citation', effect_check_sha),
                ('main', main_ref),
            ):
                results[label] = await run_delivered_check(
                    check, project_root=git_ops.project_root, ref=ref,
                )
            made_true = (
                results['parent'] is DeliveredCheckResult.FAILED
                and results['citation'] is DeliveredCheckResult.DELIVERED
                and results['main'] is DeliveredCheckResult.DELIVERED
            )
            confirmed = confirmed or made_true
            legs.append({
                'name': check.get('name'),
                'kind': kind,
                'expect': check.get('expect'),
                'parent': results['parent'].value,
                'citation': results['citation'].value,
                'main': results['main'].value,
                'verdict': _DIFFERENTIAL_CONFIRMED if made_true else 'no_signal',
            })
    except Exception as exc:
        # Contained but NOT swallowed (structured-facts-at-failure): the whole
        # differential declines rather than upgrading on partial evidence.
        confirmed = False
        probe['delivered_checks_error'] = repr(exc)
        logger.warning(
            'delivered-checks differential failed for %s — the landing will '
            'be reported as unattributed with no differential signal',
            effect_check_sha, exc_info=True,
        )
    probe['delivered_checks_legs'] = legs
    probe['delivered_checks_outcome'] = 'confirmed' if confirmed else 'no_signal'
    return confirmed


async def _resolve_main_reachable_evidence(
    git_ops: GitOps, task_id: str, head: str, probe: dict[str, Any],
) -> str | None:
    """Resolve a MAIN-REACHABLE commit carrying *task_id*'s work, or None.

    Contract invariant 3 in one place: the sha :func:`branch_work_landed`
    anchors provenance on must be a commit reachable from main that carries
    this task's work.  Two tiers, tried in order, and NO third tier:

    1. **Citation** — ``find_task_citation_commit`` walks
       ``config.main_branch``'s own history, so anything it returns is
       main-reachable by construction; no extra reachability check is needed
       (and none is done, which is what keeps this leg off the decaying
       path-set predicates — see :func:`branch_work_landed`).
    2. **Patch-id equivalent** — ``find_equivalent_commit`` over
       ``merge-base(main, head)..HEAD`` in ``project_root``, whose HEAD is
       main.  The map it builds is keyed on ``git patch-id --stable``, so the
       sha it returns is the commit on main whose diff is equivalent to
       *head*'s — the replayed twin a rebase landing produces.  It carries its
       own refuse-to-guess posture (an ambiguous patch-id, or an ambiguous
       subject in its tier-2 fallback, yields ``None`` rather than an
       arbitrary pick), which is inherited here deliberately.

    Returns ``None`` when NEITHER tier resolves.  The caller turns that into
    :attr:`LandingReason.no_attribution` rather than accepting: anchoring on
    the branch tip (which in the rebase-landing shape is not on main at all)
    or on main's current tip (which is whatever landed most recently, from any
    task) would FABRICATE provenance — the exact defect the citation guard was
    introduced to prevent, re-created one layer down.

    *head* is used as the equivalence target as-is.  A tip that carries no
    diff of its own — an empty commit, or a merge commit, whose ``git log -p``
    output git suppresses by default — simply fails to resolve and lands on
    ``no_attribution``; it is never guessed around.  (The sync-merge tip
    shape, boundary row B2, is handled explicitly by a later step of this
    task; until then it degrades to a refusal, not to a wrong answer.)

    Every decision is recorded in *probe* under ``evidence_source`` so a
    reader of the escalation can see WHICH tier answered, or which one
    declined.
    """
    citation = await git_ops.find_task_citation_commit(task_id)
    if citation:
        probe['citation'] = citation
        probe['evidence_source'] = 'citation'
        return citation

    base = await git_ops.merge_base_with_main(head)
    probe['merge_base'] = base
    if base is None:
        # No fork point means no range to search for a replayed twin. Not an
        # answer about the task — recorded as its own source so it is not read
        # as "searched and found nothing".
        probe['evidence_source'] = 'merge_base_unresolved'
        return None

    equivalent = await git_ops.find_equivalent_commit(git_ops.project_root, base, head)
    probe['evidence_source'] = (
        'patch_id_equivalent' if equivalent else 'no_equivalent_on_main'
    )
    return equivalent


async def _unresolvable_endpoints(git_ops: GitOps, *refs: str) -> list[str]:
    """Which of *refs* do NOT resolve to a commit in this repo right now.

    The health re-probe behind the fail-open disambiguation in
    :func:`branch_work_landed`.  ``merge_queue.patch_content_contained``
    returns ``False`` both for "the work is genuinely not there" and for
    "``git cherry`` failed", so the only way to tell them apart at the call
    site is to ask git a question whose failure is unambiguous.

    Implemented as ``is_ancestor(ref, ref)``: ``git merge-base --is-ancestor
    X X`` succeeds for every commit (a commit is its own ancestor) and fails
    for anything that does not resolve, so the rc alone answers "does this
    ref exist?" without a second ref-resolution authority in the module and
    without :meth:`resolve_branch_sha`'s ``refs/heads/`` restriction — the
    endpoints here are a raw sha and a branch NAME, and only one of them is a
    local branch.

    Returns the refs that failed, in the order given, so the caller can
    record WHICH endpoint was unreadable rather than only that one was.
    """
    unresolvable: list[str] = []
    for ref in refs:
        if not await git_ops.is_ancestor(ref, ref):
            unresolvable.append(ref)
    return unresolvable


class _NoOpQuestion(NamedTuple):
    """Which two commit-ishes the no-op guard should diff, if any.

    Three distinguishable outcomes, and collapsing any pair of them
    re-introduces a bug this task exists to remove:

    - ``left``/``right`` both set — ask ``left..right``.
    - both ``None`` with ``indeterminate=False`` — the question is
      INAPPLICABLE (no baseline exists for this shape).  The guard is skipped
      and the verdict is decided by the arms after it.
    - both ``None`` with ``indeterminate=True`` — git could not answer.  That
      is a broken DETECTOR, never a statement about the task, so the caller
      maps it to ``git_error``.
    """

    left: str | None
    right: str | None
    indeterminate: bool


async def _no_op_question(
    git_ops: GitOps,
    upstream: str,
    head: str,
    metadata: dict[str, Any] | None,
    probe: dict[str, Any],
) -> _NoOpQuestion:
    """Resolve the baseline the no-op guard measures *head*'s contribution from.

    A LADDER, tried in order, because no single formula is correct for every
    landing shape.  Which rung answered is recorded in
    ``probe['no_op_baseline']``; a reader of the escalation can then see what
    the emptiness claim was measured against, which the bare boolean cannot
    say.

    1. **The recorded ``branch_base_sha``** — the fork point the orchestrator
       wrote down when it CREATED the branch, and the only rung that stays
       correct for a COALESCED landing: a merge train brings several tasks in
       at once, so the train merge's own diff is non-empty even when this
       branch contributed nothing to it.  Rungs 2 and 3 would both read that
       as "not a no-op".  Validated with :func:`is_valid_sha_40` and required
       to be an ancestor of *head*, so a malformed, foreign or
       rebase-stale value falls through instead of making ``git diff`` fail.
    2. **``merge-base(upstream, head)``** — the honest fork point while the
       branch is still OUTSIDE ``upstream``'s history (the unlanded and
       rebase-landing shapes).
    3. **The landing merge's own contribution** — used exactly when rung 2
       degenerates, i.e. when *head* is ALREADY an ancestor of *upstream* and
       the merge base is therefore *head* itself.  See
       :meth:`~orchestrator.git_ops.GitOps.landing_merge_for`: taking rung 2's
       answer at face value here would report EVERY merged landing as a no-op
       and re-dispatch it forever.

    When rung 3 finds no merge either — a fast-forward landing, or a branch
    parked at an old ``upstream`` commit with no metadata — there is no
    baseline at all and the question is declared INAPPLICABLE rather than
    answered.  That is a deliberate, recorded degradation: the guard goes
    quiet and the later arms decide, so the cost is a no-op landing that must
    be caught by attribution instead.  Supplying ``branch_base_sha`` (rung 1)
    removes it, which is why every orchestrator-dispatched caller has one.
    """
    recorded = (metadata or {}).get('branch_base_sha')
    if is_valid_sha_40(recorded) and await git_ops.is_ancestor(recorded, head):
        probe['no_op_baseline'] = 'recorded_branch_base'
        return _NoOpQuestion(recorded, head, False)

    fork_point = await git_ops.merge_base_with_main(head)
    probe['no_op_fork_point'] = fork_point
    if fork_point is None:
        # Two disconnected histories, an unreadable object or a locked repo.
        # NOT "the branch has content" and NOT "the branch delivered nothing".
        probe['no_op_baseline'] = 'indeterminate'
        return _NoOpQuestion(None, None, True)
    if fork_point != head:
        probe['no_op_baseline'] = 'merge_base'
        return _NoOpQuestion(fork_point, head, False)

    landing_merge = await git_ops.landing_merge_for(head, upstream)
    if landing_merge is not None:
        probe['no_op_baseline'] = 'landing_merge'
        probe['no_op_landing_merge'] = landing_merge
        return _NoOpQuestion(f'{landing_merge}^1', landing_merge, False)

    probe['no_op_baseline'] = 'unavailable'
    return _NoOpQuestion(None, None, False)


async def branch_work_landed(
    git_ops: GitOps,
    task_id: str,
    branch: str,
    *,
    branch_tip_sha: str | None,
    metadata: dict[str, Any] | None = None,
    escalation_queue: EscalationQueue | None = None,
    recovery_emission: RecoveryEmissionConfig | None = None,
) -> LandingVerdict:
    """Has *branch*'s work landed on main, by PATCH-ID equivalence?

    The PRD "landed-not-done-recovery" Contract's producer, and the
    NON-DECAYING counterpart to :func:`validate_landing_evidence`.  Both
    return the same :class:`LandingVerdict`; this one sets
    :attr:`LandingMethod.patch_id` so a consumer can read which policy
    decided.

    **Why a second producer rather than a fix to the first.** The existing
    landing-detection policy asks two questions that DECAY: "does a commit on
    main cite this task?" (a rebase landing rewrites the shas and drops the
    citation, so there is nothing to cite) and "is the cited commit's effect
    still present at main HEAD?" (later commits touching the same paths erode
    line survival past the 0.98/0.90 thresholds).  Patch-id equivalence asks
    instead "is an equivalent patch anywhere in main's history?", which no
    later commit can un-answer.

    **Attribution is by ``git cherry``**, via the EXISTING production helper
    ``merge_queue.patch_content_contained`` rather than a second local
    implementation — two patch-id authorities in one repo is precisely the
    lockstep duplication this task exists to remove.  The import is
    FUNCTION-SCOPED because ``merge_queue.py`` imports from THIS module at
    module level, so a top-level reverse import would be a cycle; that is the
    same idiom this module already uses for ``orchestrator.delivered_checks``
    and ``escalation.models``, and that ``merge_queue`` itself uses for its
    main-health fingerprint.

    **Mind the argument order at that call**: the helper's signature is
    ``patch_content_contained(head, upstream, git_ops)`` but it shells
    ``git cherry <upstream> <head>``.  Passing them the way the shell command
    reads would invert the question into "is main's history contained in the
    branch?", which a freshly-created branch answers YES to.

    ``git cherry`` SKIPS merge commits (measured, not assumed — see
    ``TestBoundaryFixtures.test_git_cherry_skips_merge_commits``), which is what
    makes a sync-merge tip safe here: a branch that pulled main in to resolve
    a conflict has a merge commit at its tip carrying main's own history, and
    a containment check that counted it would report a landed branch as
    unlanded.  The containment question is therefore already restricted to the
    branch's OWN non-merge commits, with no extra filtering needed.

    **THE NON-DECAY INVARIANT — the PRD's headline, and it may not be
    waived.** Once this function has reported a branch landed, NO subsequent
    commit on main may change that verdict unless the work is GENUINELY
    REMOVED from main's history.  Later commits that touch, rewrite or churn
    the very same paths must not weaken it, and neither must a post-hoc
    ``git revert`` (which ADDS an inverse commit and leaves the original
    patch-ids in place).  Its regression pin is
    ``test_branch_work_landed.py``'s ``TestB1NonDecay``, which re-runs this
    function after EVERY one of five same-path churn commits — sampling the
    whole sequence, because "never decayed" and "decayed and recovered" are
    indistinguishable from a single end-state check.

    It is the headline invariant because the legacy policy's false negative is
    MONOTONIC: each later commit touching those paths erodes line survival
    further, so every detection attempt is strictly less likely to succeed
    than the one before it.  A stranded task therefore becomes progressively
    LESS recoverable the longer it goes unnoticed, and past some point is not
    recoverable at all — the failure mode that stranded tasks 3103 and 3916.
    A merely-flaky detector is an annoyance; a monotonically-decaying one
    converts a transient miss into a permanent loss.

    Concretely, that forbids two things in this function's body, and both are
    pinned at a zero call count by ``TestB2SyncMergeTip.
    test_neither_decaying_predicate_is_ever_awaited`` across every boundary
    row: ``git_ops.branch_content_in_main`` (byte-identity of the touched
    paths against main RIGHT NOW) and ``git_ops.commit_effect_present_in_main``
    / ``describe_commit_effect_in_main`` (line survival against main HEAD).
    Both answer questions about main's CURRENT state, so both decay by
    construction — including on the evidence-resolution path, where a
    reachability check must never be implemented by diffing a path set.

    The one construction that DOES flip the verdict is a genuine rewind: if
    the commits are no longer in main's history, their patch-ids are genuinely
    absent and the answer is ``not_landed``.  That keeps the invariant
    conditional rather than vacuous.

    **THE ORDERING RULE, and it is NORMATIVE — not defensive.** The arms run
    in exactly this order and may not be reordered::

        git preflight -> degenerate_branch -> no_op_landing -> patch-id
                      -> landed / not_landed

    Both guards describe states in which the patch-id arm would confidently
    ACCEPT, which is why they must precede it rather than merely accompany it:

    - a **no-op landing** really was merged, so every one of its commits is
      patch-id-present in main — and its net contribution is nonetheless
      empty.  ``git cherry`` is answering truthfully; the question it answers
      is just not the one that decides whether anything shipped.
    - a **degenerate branch** is patch-id-contained BY CONSTRUCTION: it is
      parked at an old main commit and contributes no commits of its own, so
      containment holds vacuously.

    Run in the other order, the producer does not merely mis-report — it
    attributes a FOREIGN commit's content to the task and stamps provenance
    on it, with full confidence.  That is strictly worse than any false
    negative here, because a false negative re-dispatches a task while a
    false positive closes one that never delivered.  The ordering is pinned
    mechanically by ``test_branch_work_landed.py``'s ``TestOrderingRule``,
    which asserts what never runs rather than only what the verdict says.

    **``git_error`` IS A FAIL-SOFT DEGRADATION, AND NO CONSUMER MAY COLLAPSE
    IT INTO ``not_landed``.** The two codes answer different questions.
    ``not_landed`` is a claim ABOUT THE TASK — the work is not on main, so
    dispatch it.  ``git_error`` is a claim about the DETECTOR — it could not
    look, and says nothing whatsoever about the task.  A repo lock, a corrupt
    object or an unresolvable ref silently reading as "not landed" re-dispatches
    a task whose work is already on main, and keeps re-dispatching it, because
    re-running the check does not fix whatever broke it.  That is the defect
    this PRD exists to fix, so producing it here would be strictly worse than
    shipping nothing.  A consumer that cannot act on ``git_error`` must
    escalate or retry — never treat it as a negative.

    Every git failure therefore reaches the caller as ``git_error`` with the
    failing operation named in ``probe['git_error_stage']``, which is
    structured facts at failure rather than a bare code:

    - ``resolve_branch_sha`` — the branch ref did not resolve.
    - ``no_op_baseline`` — no fork point could be computed (see
      :func:`_no_op_question`; distinct from "no baseline EXISTS for this
      shape", which is recorded as ``no_op_baseline == 'unavailable'`` and is
      not an error).
    - ``net_diff_is_empty`` — the tri-state primitive returned ``None``.
    - ``patch_id_containment`` — the fail-open disambiguation below.
    - ``unexpected_exception`` — with ``probe['exception']`` carrying the
      repr and a traceback logged at WARNING.

    **The fail-open disambiguation.** ``patch_content_contained`` swallows
    ``rc != 0`` into ``False`` for its own caller, which then falls through to
    a full merge attempt and is therefore safe under a wrong ``False``.  Here
    the same ``False`` would mean "this task never landed", so it is re-probed
    instead of believed: both endpoints were resolved before the call, so if
    either fails to re-resolve afterwards the answer came from a broken repo.
    Which branch was taken is recorded in ``probe['containment_recheck']``
    (``'healthy'`` / ``'unhealthy'``) and ``probe['containment_unresolvable']``,
    so a reader can see a genuine negative distinguished from an unreadable
    repo without re-running git.  Re-implementing ``git cherry`` locally would
    have avoided the disambiguation and created a SECOND patch-id authority in
    the same repo, which is the duplication this task exists to remove.

    Two legs remain fail-soft rather than fail-closed, recorded here so the
    residue is known rather than discovered: :func:`branch_is_degenerate`
    returns a plain ``bool`` and so cannot report its own git failures (a
    failure there reads as "not degenerate" and the later arms decide), and
    the evidence-resolution tiers degrade to ``no_attribution`` rather than
    ``git_error`` — a refusal that escalates for a human rather than one that
    re-dispatches, with ``probe['evidence_source']`` naming which tier
    declined.

    It NEVER RAISES.  Whatever happens, a verdict comes back — it never
    accepts on doubt and never propagates.

    Pure and read-only, exactly as :func:`validate_landing_evidence` is: it
    never stamps a task done, never escalates and never mutates git or task
    state.  The caller owns the action.

    Args:
        git_ops: A ``GitOps`` instance, or a duck-typed stand-in exposing
            ``resolve_branch_sha`` / ``find_task_citation_commit`` /
            ``merge_base_with_main`` / ``find_equivalent_commit`` and a
            ``project_root``.
        task_id: Bare task id (no ``task/`` prefix).
        branch: The task's branch name (e.g. ``f'task/{task_id}'``).
        branch_tip_sha: An already-resolved tip for *branch*, or ``None`` to
            resolve it here.  REQUIRED-BY-KEYWORD rather than defaulted, so a
            caller must state which it means: a caller that already anchored
            other checks on a tip it observed must pass that SAME tip, or a
            concurrent warm-lane reseed between the two reads would anchor
            this verdict on a tip that is no longer current (the hazard
            :func:`branch_is_degenerate` documents from the task 3103 review).
        metadata: The task's metadata dict.  A documented WIDENING of the
            Contract's sketched signature — boundary row B6 needs
            ``branch_base_sha``, which the sketched four-argument form cannot
            supply.  Consumed by the degenerate-branch guard.
        escalation_queue: The queue the ``git_error`` storm-escape L1 is filed
            on (see :func:`file_landing_git_error_storm_escalation`).
            ``None`` means "no queue available", which is a SUPPORTED shape
            and not a degraded one — every verdict this function returns is
            correct without one; a missing queue costs the storm alarm and
            nothing else.
        recovery_emission: The live ``config.recovery_emission`` submodel,
            supplying the storm gate's rate and kill switch.  Pass the
            OrchestratorConfig's own submodel (a reference, not a copy) so the
            green-tier hot-reload of those leaves takes effect without a
            restart.  ``None`` falls back to the shipped defaults, which is
            what the bare-harness construction sites want.

    Returns:
        A :class:`LandingVerdict` with ``method`` set to
        :attr:`LandingMethod.patch_id`.
    """
    upstream = _main_ref(git_ops)
    probe: dict[str, Any] = {
        'task_id': task_id,
        'branch': branch,
        'branch_tip_sha': branch_tip_sha,
        'upstream_ref': upstream,
        'citation': None,
        # Recorded up front, not only on the accept: WHICH producer answered
        # is a property of the call, and a consumer reading a rejected verdict
        # needs to know it came from the patch-id policy and not the legacy
        # effect-present one.
        'method': LandingMethod.patch_id,
    }

    def _finish(verdict: LandingVerdict) -> LandingVerdict:
        # EVERY exit routes through here — including the except-clause reject
        # below — so the tally can never under-count the reason that matters
        # most.  Wholly best-effort inside; it can never break the verdict.
        _observe_landing_verdict(
            verdict,
            escalation_queue=escalation_queue,
            recovery_emission=recovery_emission,
        )
        return verdict

    def _reject(reason: LandingReason) -> LandingVerdict:
        probe['reason'] = reason
        return _finish(LandingVerdict(
            accepted=False, evidence_sha=None, reason=reason,
            probe=dict(probe), method=LandingMethod.patch_id,
        ))

    def _accept(evidence_sha: str) -> LandingVerdict:
        probe['reason'] = LandingReason.landed
        return _finish(LandingVerdict(
            accepted=True, evidence_sha=evidence_sha, reason=LandingReason.landed,
            probe=dict(probe), method=LandingMethod.patch_id,
        ))

    try:
        head = branch_tip_sha
        if head is None:
            head = await git_ops.resolve_branch_sha(branch)
            probe['branch_tip_sha'] = head
        if head is None:
            # An unresolvable ref is a broken DETECTOR, not an unlanded task.
            probe['git_error_stage'] = 'resolve_branch_sha'
            return _reject(LandingReason.git_error)

        # GUARD 1 — degenerate branch.  Passing the ALREADY-RESOLVED tip is
        # mandatory, not an optimisation: re-reading the ref would let a
        # concurrent warm-lane reseed land between the two reads, so the verdict
        # could be anchored on a tip that is no longer the one the later arms
        # judge (task 3103 review; see branch_is_degenerate's own docstring).
        degenerate = await branch_is_degenerate(
            git_ops, branch, metadata or {}, branch_tip_sha=head,
        )
        probe['degenerate'] = degenerate
        if degenerate:
            return _reject(LandingReason.degenerate_branch)

        # GUARD 2 — no-op landing.  The BASELINE comes from a ladder rather than
        # from `upstream` directly (see _no_op_question): once the branch has
        # merged, merge-base(upstream, head) IS head, so asking the question
        # against `upstream` reports every landed branch as a no-op.
        question = await _no_op_question(git_ops, upstream, head, metadata, probe)
        if question.indeterminate:
            probe['git_error_stage'] = 'no_op_baseline'
            return _reject(LandingReason.git_error)
        if question.left is not None and question.right is not None:
            # The probe out-parameter carries the measured commit's parent shas
            # and merge-base into the verdict, so an escalation body can show
            # whether it is a merge without re-running git.
            net_empty = await git_ops.net_diff_is_empty(
                question.left, question.right, probe=probe,
            )
            probe['net_diff_is_empty'] = net_empty
            if net_empty is None:
                # TRI-STATE, and the third state is NOT "not a no-op": collapsing
                # it would launder a broken merge-base or an unreadable commit
                # into a statement about the task.  Fully classified in a later
                # step.
                probe['git_error_stage'] = 'net_diff_is_empty'
                return _reject(LandingReason.git_error)
            if net_empty:
                return _reject(LandingReason.no_op_landing)

        # Function-scoped: merge_queue.py imports from this module at module
        # level, so importing it back at module level would close an import cycle.
        from orchestrator.merge_queue import patch_content_contained  # noqa: PLC0415

        contained = await patch_content_contained(head, upstream, git_ops)
        probe['patch_id_contained'] = contained
        if not contained:
            # THE FAIL-OPEN DISAMBIGUATION.  patch_content_contained deliberately
            # swallows `rc != 0` into False for ITS caller, which falls through to
            # a full merge attempt and is therefore safe under a wrong False.  The
            # same False here would mean "this task never landed", so it is
            # re-probed rather than believed: both endpoints were resolved before
            # the call, so if either fails to re-resolve NOW the containment answer
            # came from a broken repo and not from the task.
            unresolvable = await _unresolvable_endpoints(git_ops, head, upstream)
            probe['containment_unresolvable'] = unresolvable
            probe['containment_recheck'] = 'unhealthy' if unresolvable else 'healthy'
            if unresolvable:
                probe['git_error_stage'] = 'patch_id_containment'
                return _reject(LandingReason.git_error)
            return _reject(LandingReason.not_landed)

        evidence_sha = await _resolve_main_reachable_evidence(git_ops, task_id, head, probe)
        if evidence_sha is None:
            return _reject(LandingReason.no_attribution)
        return _accept(evidence_sha)
    except Exception as exc:
        # CONTAINED but not SWALLOWED, the same discipline as
        # _record_effect_divergence and _delivered_checks_differential above.
        # Every caller is a RECOVERY path — a stranded-task sweep, a dispatch
        # gate, a merge-status query — so an exception escaping here does not
        # fail one check, it stops every OTHER task in the same sweep from
        # being recovered.  The repr goes into the probe (structured facts for
        # the escalation body) and a traceback goes to the log, so a contained
        # failure is never mistakable for a clean negative.
        logger.warning(
            'branch_work_landed: unexpected failure for task %s on %s; '
            'returning git_error rather than a claim about the task',
            task_id, branch, exc_info=True,
        )
        probe['git_error_stage'] = 'unexpected_exception'
        probe['exception'] = repr(exc)
        return _reject(LandingReason.git_error)


async def validate_landing_evidence(
    git_ops: GitOps,
    task_id: str,
    branch: str,
    *,
    branch_tip_sha: str | None,
    candidate_sha: str | None = None,
    pattern_template: str | None = None,
    delivered_checks: list[dict[str, Any]] | None = None,
) -> LandingVerdict:
    """Validate already-landed evidence for *task_id* on *branch*.

    See the module docstring for the DISCOVERY (``candidate_sha=None``) vs
    CANDIDATE (``candidate_sha`` given) mode split.

    Args:
        git_ops: A ``GitOps`` instance (or a duck-typed stand-in exposing
            ``find_task_citation_commit`` / ``is_ancestor`` /
            ``commit_effect_present_in_main``, plus
            ``describe_commit_effect_in_main`` for the effect_absent
            divergence diagnostics — a stand-in lacking that last method
            still decides identically; the failure is recorded in
            ``probe['effect_probe_error']``).
        task_id: Bare task id (no ``task/`` prefix).
        branch: The task's branch name (e.g. ``f'task/{task_id}'``), used
            for the FIX 2 lineage guard in DISCOVERY mode. Not consulted in
            CANDIDATE mode (attribution is already established).
        branch_tip_sha: The branch's current tip sha, used as the
            effect-present anchor in DISCOVERY mode when the citation is an
            in-branch work commit. May be ``None`` in CANDIDATE mode (the
            branch may no longer exist).
        candidate_sha: When given, switches to CANDIDATE mode: skip
            discovery/lineage and apply the effect-present guard to this
            sha only.
        pattern_template: Optional override forwarded to
            ``find_task_citation_commit`` (DISCOVERY mode only).
        delivered_checks: The task's ``metadata.delivered_checks`` list,
            enabling the SECOND accept path (task 3116; see
            :func:`_delivered_checks_differential`). THREE-STATE, and the
            states are not interchangeable: ``None`` means NOT SUPPLIED — a
            call site that has not been wired yet — while ``[]`` means
            supplied by a task that declares no checks. Both are recorded in
            ``probe['delivered_checks_state']`` so a permanently-unwired site
            is distinguishable from a task with nothing to check; collapsing
            them would leave the capstone task that makes this parameter
            required with no signal to act on. Consulted ONLY on the
            ``effect_absent`` reject path, so it can upgrade a rejection to an
            acceptance and can NEVER do the reverse (b6: the interface stays
            binary — omitting this parameter reproduces the pre-task behaviour
            exactly).

            STAGING, NAMED SO IT CANNOT GO PERMANENT (amendment pass, review
            finding — as shipped by task 3116 NO production call site passes
            this, so the whole second accept path is unreachable and the only
            live effect is the escalation text "this call site is unwired").
            The call sites are owned by other tasks and are deliberately not
            edited here — task 3116 holds no lock on them:

              task 4496  harness.py x4 (three ``_already_landed_dispatch_gate``
                         arms + ``_reconcile_one_stranded``)
              task 4497  merge_queue.py x1 (the coalesce re-drive)
              task 4498  escalation/server.py x2 (``merge_status``'s
                         git-authority arms)
              task 4500  CAPSTONE — flips this parameter to REQUIRED and
                         keyword-only once all seven are wired, so no future
                         caller can inherit the default silently

            If ``delivered_checks_state == 'unwired'`` is still appearing in
            escalations after 4500 has landed, that is the bug: one of the
            seven sites regressed to the default.

    Returns:
        A :class:`LandingVerdict`.
    """
    probe: dict[str, Any] = {
        'task_id': task_id,
        'branch': branch,
        'branch_tip_sha': branch_tip_sha,
        'citation': None,
        'effect_check_sha': None,
        # Supply state, recorded unconditionally — wiring is a property of the
        # CALL SITE, not of the outcome, so an accepted verdict must show it
        # too.  Otherwise the only way to learn a site is unwired is to wait
        # for it to reject.
        'delivered_checks_state': (
            'unwired' if delivered_checks is None
            else 'evaluated' if delivered_checks
            else 'none_declared'
        ),
    }

    def _reject(reason: str) -> LandingVerdict:
        probe['reason'] = reason
        return LandingVerdict(
            accepted=False, evidence_sha=None, reason=reason, probe=dict(probe),
        )

    def _accept(evidence_sha: str) -> LandingVerdict:
        probe['reason'] = 'ok'
        return LandingVerdict(
            accepted=True, evidence_sha=evidence_sha, reason='ok', probe=dict(probe),
        )

    if candidate_sha is not None:
        # CANDIDATE mode: attribution is already established by the caller
        # (a merge-marker subject match, or a stranded-sweep ground-truth
        # report) — skip discovery and the FIX 2 lineage guard entirely and
        # apply ONLY the FIX 1' effect-present guard to candidate_sha.
        probe['citation'] = candidate_sha
        probe['effect_check_sha'] = candidate_sha
        if not await git_ops.commit_effect_present_in_main(candidate_sha):
            await _record_effect_divergence(git_ops, candidate_sha, probe)
            # SECOND ACCEPT PATH, and it lives INSIDE the reject branch on
            # purpose: a differential can only ever rescue a rejection, never
            # produce one (task 3116).
            if delivered_checks and await _delivered_checks_differential(
                git_ops, candidate_sha, delivered_checks, probe,
            ):
                return _accept(candidate_sha)
            return _reject('effect_absent')
        return _accept(candidate_sha)

    citation = await git_ops.find_task_citation_commit(
        task_id, pattern_template=pattern_template,
    )
    if citation is None:
        return _reject('no_citation')
    probe['citation'] = citation

    # Citation-anchor selection (task 2870, esc-5252-9; formerly the FIX 2
    # bidirectional lineage HARD-REJECT). is_ancestor(citation, branch) still
    # classifies the anchor for the FIX 1' effect-present guard below — True
    # for a WORK commit ON the branch (anchor on the branch tip, which may
    # have advanced past this intermediate commit); False for this branch's
    # OWN no-ff merge commit OR a divergent/realigned branch ref (anchor on
    # the citation itself). A citation unreachable from the branch in BOTH
    # directions is NO LONGER rejected as 'lineage_mismatch': every DISCOVERY
    # caller has already established the branch's content is on main before
    # delegating here (the dispatch-gate ancestry path, the
    # content-equivalence fallback, and the coalesce re-drive), and
    # find_task_citation_commit greps `git log main` so its citation is by
    # construction reachable from main — branch-ref ancestry is NOT the
    # landing authority. A stale/divergent/realigned branch tip that fails
    # both is_ancestor directions is still a genuine landing; hard-rejecting
    # it caused a close<->refile ping-pong (the auto-watcher closes the
    # benign L1, the gate re-opens it) that never self-resolved. The FIX 1'
    # effect-present guard below stays the real gate against a reverted
    # (task-1175) landing.
    citation_on_branch = await git_ops.is_ancestor(citation, branch)

    # FIX 1' effect-present guard (task 2500/2675): ancestry alone doesn't
    # mean the effect survives at HEAD — a later commit on main may have
    # REVERTED or rewritten the lines the citation ADDED. Task 3116 part b
    # replaced byte-identity with line-survival here, so ordinary additive
    # later evolution of those same paths no longer reads as a revert.
    # Anchor on the branch
    # TIP for an in-branch work commit (it may be a stale intermediate
    # commit); anchor on the citation itself for a no-ff merge commit (task
    # 2675 made this a REAL check — it diffs each non-first parent's content
    # against current main HEAD, not a no-op; see
    # commit_effect_present_in_main), for a divergent/realigned branch ref
    # (task 2870 — citation_on_branch False, the branch tip is not
    # authoritative), or in the defensive case a DISCOVERY caller omitted
    # branch_tip_sha despite citation_on_branch.
    effect_check_sha = citation
    anchor_is_branch_tip = False
    if citation_on_branch and branch_tip_sha is not None:
        effect_check_sha = branch_tip_sha
        anchor_is_branch_tip = True
    probe['effect_check_sha'] = effect_check_sha
    if not await git_ops.commit_effect_present_in_main(effect_check_sha):
        await _record_effect_divergence(git_ops, effect_check_sha, probe)
        # SECOND ACCEPT PATH — anchored on the sha the survival check actually
        # ran against, so both modes ask the same question.  Inside the reject
        # branch, so it is structurally upgrade-only (task 3116).  The
        # branch-tip flag travels with it because it decides what "before this
        # landing" means: a tip's ^1 is the branch's own previous commit, not
        # a pre-branch baseline (see _differential_parent_ref).
        if delivered_checks and await _delivered_checks_differential(
            git_ops, effect_check_sha, delivered_checks, probe,
            anchor_is_branch_tip=anchor_is_branch_tip,
        ):
            return _accept(citation)
        return _reject('effect_absent')

    return _accept(citation)


#: Operator-facing prose for every :class:`LandingReason` member.
#:
#: COMPLETE by contract, machine-checked in
#: test_landing_contract_vocabulary.py by iterating the enum: a member with no
#: entry here makes :func:`format_unattributed_landing_detail` render the
#: literal ``'Unrecognized reason code: ...'`` into the L1 body a human reads.
_REASON_EXPLANATIONS: dict[str, str] = {
    'landed': (
        "The branch's work is present on main by PATCH-ID equivalence and a "
        'main-reachable commit was attributed to this task. Accepted.'
    ),
    'no_op_landing': (
        'A landing marker for this task is on main, but the branch '
        'contributes NO NET CHANGE relative to where it forked — its own '
        'commits cancel out (added then removed, or reverted within the '
        'branch). The merge is genuine; the deliverable is not. Stamping '
        'this done would record a task as delivered when nothing shipped '
        '(the task-1175 shape), so it is rejected on purpose. Check whether '
        'the work was backed out mid-branch or landed on a different branch.'
    ),
    'not_landed': (
        "The branch's commits are genuinely absent from main: no commit on "
        'main carries an equivalent patch. This is the ordinary negative — '
        'the task has not landed and should be dispatched normally.'
    ),
    'no_attribution': (
        'The work IS on main, but no main-reachable commit could be '
        'attributed to THIS task — no commit cites it and no equivalent '
        'commit resolved. There is no sha to anchor provenance on, and this '
        'producer refuses to guess: anchoring on the branch tip (not on '
        'main) or on main\'s current tip would fabricate provenance, which '
        'is the defect the citation guard exists to prevent. The Contract\'s '
        'rename of the legacy "no_citation".'
    ),
    'degenerate_branch': (
        'The branch never advanced past its recorded branch_base_sha '
        '(#1226): zero commits were ever pushed beyond the creation point. '
        'Such a branch is patch-id-contained in main BY CONSTRUCTION, so '
        'every containment check reads it as a confident landing of a '
        "foreign commit's content. Rejected before any attribution runs. "
        'The task has no work to attribute; investigate why the branch is '
        'empty rather than why the check failed.'
    ),
    'git_error': (
        'Git could not answer. This is a FAIL-SOFT DEGRADATION of the '
        'DETECTOR, not a statement about the task: a repo lock, a corrupt '
        'object, an unresolvable ref or an unreadable merge-base all land '
        'here. It must NEVER be read as "not landed" — a git failure '
        'silently reading as unlanded re-dispatches an already-landed task '
        'on every tick forever, which is the defect this machinery exists '
        'to fix rather than an instance of it. See probe.git_error_stage '
        'for which operation failed. If this code repeats, the detector is '
        'broken; fix the repo or the ref, do not re-dispatch the task.'
    ),
    'ok': (
        'The evidence is attributable and its effect is present at main '
        'HEAD. Accepted. (The pre-Contract spelling of an accept; '
        ':attr:`LandingReason.landed` is the Contract\'s.)'
    ),
    'no_citation': (
        'No commit on main cites this task (find_task_citation_commit found '
        'nothing) — there is no positive evidence to attribute a landing to.'
    ),
    'effect_absent': (
        "The evidence commit's own effect did not SURVIVE to current main "
        "HEAD (FIX 1', task 2500/2675, survival semantics from task 3116) "
        '— too few of the lines it added are still present at main. This '
        'is no longer the old byte-identity question ("has anyone touched '
        'these paths since?"), so ordinary additive evolution — a later '
        'task appending to the same file, a co-touched hot file, an '
        'unrelated edit a merge integrated cleanly — does NOT land here. '
        'What does land here is a genuine revert of the deliverable, a '
        'heavy rewrite that replaced most of those lines (a refactor, a '
        'reformat, an extraction), or a revert paired with unrelated '
        'additions in the same files; this check cannot separate those '
        'from one another. Judge the diverged paths below against what '
        "this task actually delivered: if the task's deliverable is still "
        'recognisably present at main, this is a rewrite reading as a '
        'revert rather than a real one.'
    ),
}


#: Operator-facing prose for every :class:`LandingMethod` member — WHICH
#: attribution path decided.  Complete by contract, machine-checked the same
#: way :data:`_REASON_EXPLANATIONS` is.
_METHOD_EXPLANATIONS: dict[str, str] = {
    'patch_id': (
        'Attributed by patch-id equivalence (`git cherry`): every commit the '
        'branch contributes is already present in main as an equivalent '
        'patch. This is the NON-DECAYING path — later commits touching the '
        'same paths cannot make an equivalent patch stop existing in '
        "main's history."
    ),
    'merge_marker': (
        'Attribution was established by the CALLER (a merge-marker subject '
        'match or a stranded-sweep ground-truth report) and this module only '
        'applied the effect-present guard to the supplied candidate — '
        'validate_landing_evidence CANDIDATE mode.'
    ),
    'citation': (
        "Attributed by subject-anchored citation discovery over main's "
        'history (find_task_citation_commit) — validate_landing_evidence '
        'DISCOVERY mode. Depends on a commit on main naming the task, so a '
        'rebase landing that dropped the citation is invisible to it.'
    ),
    'unspecified': (
        'No attribution path ran: this verdict was hand-constructed rather '
        'than produced by this module (the shape several gate-wiring tests '
        'build). Not a landing claim.'
    ),
}


def format_unattributed_landing_detail(
    task_id: str, branch: str, verdict: LandingVerdict,
) -> tuple[str, str]:
    """Render a rejected :class:`LandingVerdict` as (summary, detail).

    Shared (INV-5) across every escalating call site — the harness
    ``_file_unattributed_landing_escalation`` helper and the merge_queue
    coalesce re-drive escalation — so the human-facing message is
    single-sourced rather than five independently-drifting inline f-strings.

    Args:
        task_id: The task id the escalation is filed for.
        branch: The branch the evidence check ran against.
        verdict: A rejected verdict (``accepted`` False); the reason and
            probe are rendered into the detail text regardless of value,
            but this is intended to be called only on rejection.

    Returns:
        A ``(summary, detail)`` tuple — ``summary`` is a one-line, ``[:200]``-
        safe string suitable for ``Escalation.summary``; ``detail`` is a
        multi-line block for ``Escalation.detail``.
    """
    explanation = _REASON_EXPLANATIONS.get(
        verdict.reason, f'Unrecognized reason code: {verdict.reason}',
    )
    divergence_block, summary_fragment = _render_effect_divergence(verdict)
    differential_block = _render_delivered_checks_differential(verdict)
    summary = (
        f'Task {task_id}: landing evidence on branch {branch!r} could not '
        f'be attributed ({verdict.reason}){summary_fragment}'
    )[:200]
    detail = (
        f'validate_landing_evidence rejected the landing evidence for task '
        f'{task_id} on branch {branch!r}.\n\n'
        f'reason: {verdict.reason}\n'
        f'{explanation}\n\n'
        f'{divergence_block}'
        f'{differential_block}'
        f'probe: {verdict.probe}\n\n'
        'The task was NOT marked done. It is left pending (or flipped to '
        'pending by the coalesce re-drive, from merge-deferred), which means '
        'it will be DISPATCHED TO AN AGENT on the next dispatch tick — a '
        'full plan/verify/review cycle, not a cheap idempotent re-check. If '
        'this landing is genuine, that dispatch is pure waste and will '
        'REPEAT every tick, because this condition does not heal on its own. '
        'Investigate why attribution/effect-survival failed (e.g. a heavy '
        'rewrite of the touched paths, a branch-alias landing, a genuine '
        'reverted merge, an unattributed commit, or a missing task-citing '
        'commit); resolve this escalation once confirmed.'
    )
    return summary, detail


def _survival_lines(probe: dict[str, Any]) -> list[str]:
    """Render the part-(b) survival MEASUREMENT that decided the verdict.

    Returns [] for a probe that carries no survival keys at all — a legacy
    probe, or one written before this was threaded through — so an older
    caller still renders cleanly rather than printing "None/None".

    This is the block that answers "why was this rejected?".  The diverged
    paths above it are a demoted diagnostic; these numbers are the decision.
    The denominator is printed with the ratio on purpose: 2/3 lines
    surviving and 2000/3000 are the same ratio and warrant very different
    confidence, and a bare "0.67" invites over-trusting a 3-line sample.
    Thresholds are printed beside the values they gate so the reader can see
    WHICH arm failed without reading the source, and so a later retune shows
    up in the escalation instead of silently changing what the number means.
    """
    if 'aggregate_survival' not in probe:
        return []

    lines: list[str] = []
    aggregate = probe.get('aggregate_survival')
    total = probe.get('added_lines_total') or 0
    threshold = probe.get('aggregate_threshold')

    if aggregate is None:
        lines.append(
            'survival: not measured — no touched path contributed any added '
            'lines (all vacuous), or resolution failed before the '
            'measurement stage'
        )
    else:
        survived = round(aggregate * total)
        verdict_word = (
            'BELOW threshold' if threshold is not None and aggregate < threshold
            else 'at or above threshold'
        )
        gate = '' if threshold is None else f', threshold {threshold}'
        lines.append(
            f'survival (aggregate): {aggregate:.4f} — about {survived} of '
            f'{total} added lines still present at main{gate} '
            f'[{verdict_word}]'
        )

    worst_path = probe.get('worst_guarded_path')
    worst = probe.get('worst_guarded_survival')
    if worst_path and worst is not None:
        per_file = probe.get('per_file_threshold')
        floor = probe.get('per_file_min_added_lines')
        gate = '' if per_file is None else f', threshold {per_file}'
        scope = '' if floor is None else f' (guard applies at >= {floor} added lines)'
        state = (
            'BELOW threshold' if per_file is not None and worst < per_file
            else 'at or above threshold'
        )
        lines.append(
            f'worst guarded file: {worst_path} at {worst:.4f}{gate} '
            f'[{state}]{scope}'
        )

    vacuous = probe.get('vacuous_paths')
    if vacuous:
        lines.append(
            'vacuous paths (zero added lines — decided by deletion/rename/'
            'blob comparison, not by line survival):'
        )
        lines.extend(f'  - {path}' for path in vacuous)
    return lines


def _render_effect_divergence(
    verdict: LandingVerdict,
) -> tuple[str, str]:
    """Render the effect_absent divergence diagnostics (task 3116).

    Returns a ``(detail_block, summary_fragment)`` pair; both are empty
    strings for any reason other than ``'effect_absent'`` and for a legacy
    ``probe={}`` that predates the diagnostics, so a caller constructing
    either still renders cleanly.

    The labelled block is the point: the raw ``probe: {...}`` dict repr
    already contained the paths, but nothing pointed a reader at them.
    Naming them under a header is the one line that resolves the
    "is this a revert or just skew?" question the reason prose now poses.

    The paths are followed by :func:`_survival_lines`, the part-(b)
    MEASUREMENT that actually decided the verdict.  Order is deliberate and
    the header says so: paths first because they are the most legible line
    in the escalation, but explicitly labelled a diagnostic, because since
    the survival semantics they no longer decide anything.  Printing them
    alone would show the reader everything except the basis of the
    rejection.
    """
    if verdict.reason != 'effect_absent':
        return '', ''
    probe = verdict.probe
    if 'diverged_paths' not in probe:
        return '', ''

    paths = probe['diverged_paths']
    if paths is None:
        error = probe.get('effect_probe_error', '<no detail recorded>')
        return (
            f'diverged paths could not be determined: {error}\n\n'
        ), ''

    if not paths:
        failure = probe.get('effect_failure')
        if failure:
            # Still emit the survival block: on a structural failure it says
            # "not measured", which is the fact the reader needs — silence
            # here would read as "measured and fine".
            tail = _survival_lines(probe)
            suffix = ('\n' + '\n'.join(tail)) if tail else ''
            return (
                'no path divergence recorded — the effect check failed '
                f'structurally: {failure}{suffix}\n\n'
            ), ''
        # The decision said absent but the re-probe says present: main HEAD
        # advanced between the two calls.  Render the race rather than let
        # the escalation silently contradict itself.
        return (
            'no path divergence recorded — the re-probe found the effect '
            'present; main HEAD may have advanced between the decision and '
            'this probe.\n\n'
        ), ''

    lines = [
        'diverged paths (touched by the evidence commit, no longer '
        'matching main HEAD — a DIAGNOSTIC; since the survival semantics '
        'these do not decide the verdict):',
    ]
    lines.extend(f'  - {path}' for path in paths)
    anchor = probe.get('effect_anchor_sha')
    if anchor:
        lines.append(f'effect anchor: {anchor}')
    failure = probe.get('effect_failure')
    if failure:
        lines.append(f'effect failure: {failure}')
    lines.extend(_survival_lines(probe))
    block = '\n'.join(lines) + '\n\n'

    fragment = f'; diverged: {paths[0]}'
    if len(paths) > 1:
        fragment += f' +{len(paths) - 1} more'
    return block, fragment


def _render_delivered_checks_differential(
    verdict: LandingVerdict,
) -> str:
    """Render the delivered-checks differential outcome (task 3116 part b).

    Returns a detail block, or ``''`` for a legacy ``probe`` predating the
    key so a caller constructing one still renders cleanly.

    The UNWIRED state is named EXPLICITLY, and separately from
    ``none_declared``.  Omitting it would hide precisely the degradation this
    second accept path introduces: a call site nobody wired reads, in an
    escalation, exactly like a task that declares nothing to check — and the
    two are fixed by different people.  An unwired site is an orchestrator
    gap; a task with no checks is a task-authoring gap.
    """
    state = verdict.probe.get('delivered_checks_state')
    if state is None:
        return ''
    if state == 'unwired':
        return (
            'delivered-checks differential: NOT RUN — this call site is '
            'unwired (it passes no delivered_checks), so the second accept '
            'path could not be attempted. If this landing is genuine, wiring '
            "the site is what would let the task's own declared capability "
            'checks confirm it.\n\n'
        )
    if state == 'none_declared':
        return (
            'delivered-checks differential: NOT RUN — this task declares '
            'no delivered_checks, so there is no capability to confirm it '
            'delivered. Declaring one would give this landing a second, '
            'independent way to be accepted.\n\n'
        )

    outcome = verdict.probe.get('delivered_checks_outcome')
    if outcome is None:
        # Checks were supplied but the differential never ran — the survival
        # check accepted, so there was nothing to rescue.
        return (
            'delivered-checks differential: not consulted (the effect check '
            'did not reject).\n\n'
        )
    if outcome == 'disabled':
        return (
            'delivered-checks differential: NOT RUN — delivered_checks.enabled '
            'is false in this project config, so the second accept path is '
            'switched off (the same kill switch the mark-done delivered-check '
            'gate honours). This landing was decided by line survival alone.'
            '\n\n'
        )
    lines = [f'delivered-checks differential: {outcome}']
    parent_ref = verdict.probe.get('delivered_checks_parent_ref')
    if parent_ref:
        lines.append(f'  pre-landing baseline (the parent leg): {parent_ref}')
    for leg in verdict.probe.get('delivered_checks_legs') or []:
        name = leg.get('name')
        if 'parent' not in leg:
            lines.append(f'  - {name}: {leg.get("verdict")}')
            continue
        lines.append(
            f'  - {name}: parent={leg.get("parent")} '
            f'citation={leg.get("citation")} main={leg.get("main")} '
            f'-> {leg.get("verdict")}'
        )
    error = verdict.probe.get('delivered_checks_error')
    if error:
        lines.append(f'  differential error: {error}')
    lines.append(
        '  (a check must FAIL at the parent and PASS at both the citation '
        'and main to confirm — a static pass at main proves nothing)'
    )
    return '\n'.join(lines) + '\n\n'


def file_unattributed_landing_escalation(
    escalation_queue: EscalationQueue | None,
    task_id: str,
    branch: str,
    verdict: LandingVerdict,
    *,
    agent_role: str,
) -> None:
    """Best-effort, dedup-guarded L1 escalation for unattributable landing
    evidence (task 2678, INV-5; extracted in the amendment pass — review
    finding: ``Harness._file_unattributed_landing_escalation`` and
    ``SpeculativeMergeWorker._file_unattributed_landing_escalation`` were a
    near-verbatim copy of this filing boilerplate, differing only in
    ``agent_role``).

    Called by every site that found a positive landing signal (a merge
    marker, branch content equivalent to main, or an on-main coalesce
    member) but whose :func:`validate_landing_evidence` verdict came back
    rejected — an unattributed or effect-absent landing that must not be
    silently stamped done (the task-1175 clobber this task closes).
    Escalate-instead-of-stamp is deliberately non-status-blocking: the
    caller leaves the task/member row pending (or flips it to pending) and
    it is re-evaluated next tick; the caller's own open-L1 veto naturally
    suppresses reprocessing while this L1 stays open — no separate status
    transition happens here.

    Best-effort (a no-op when *escalation_queue* is None, e.g. bare-harness
    or bare-worker unit tests) and deduped via ``has_open_l1`` so repeated
    ticks re-observing the same unattributable evidence don't stack
    duplicate L1s — one open escalation per task PER CATEGORY at a time.

    **The category scoping is load-bearing** (task 3116). This call used to
    pass a bare ``task_id``, and ``category=None`` matches ANY open L1 on the
    task — a two-way blindfold in which an unrelated pending escalation (a
    ``task_failure``, say) silently suppressed a ``provenance_unattributed``
    filing, so a provenance defect hid behind an escalation that had nothing
    to do with it. Observed live on task 4105. The narrower dedup
    deliberately accepts slightly higher L1 volume — a task can now hold one
    open L1 per category rather than one overall — in exchange for not hiding
    provenance defects behind unrelated escalations. Do not widen it back.

    Args:
        escalation_queue: The caller's ``EscalationQueue``, or ``None``.
        task_id: The task (or coalesce member) id the escalation is filed for.
        branch: The branch the evidence check ran against.
        verdict: A rejected :class:`LandingVerdict`.
        agent_role: The filing caller's role, e.g. ``'harness-reconcile'``
            or ``'orchestrator-merge-worker'`` — the only thing that
            distinguishes the two call sites.
    """
    if not escalation_queue:
        return
    try:
        if escalation_queue.has_open_l1(
            task_id, category='provenance_unattributed',
        ):
            return
        from escalation.models import Escalation  # noqa: PLC0415

        summary, detail = format_unattributed_landing_detail(task_id, branch, verdict)
        esc = Escalation(
            id=escalation_queue.make_id(task_id),
            task_id=task_id,
            agent_role=agent_role,
            severity='blocking',
            category='provenance_unattributed',
            summary=summary,
            detail=detail,
            suggested_action='investigate_unattributed_landing_evidence',
            level=1,
        )
        escalation_queue.submit(esc)
        logger.warning(
            'Filed provenance_unattributed escalation %s for task %s '
            '(branch %s, reason %s, agent_role %s)',
            esc.id, task_id, branch, verdict.reason, agent_role,
        )
    except Exception:
        logger.warning(
            'Failed to file provenance_unattributed escalation for task %s '
            '(branch %s) — continuing without it',
            task_id, branch, exc_info=True,
        )


def file_landing_git_error_storm_escalation(
    escalation_queue: EscalationQueue | None,
    *,
    tally: LandingTally,
    rate_per_hour: int,
    agent_role: str = _GIT_ERROR_STORM_ROLE,
) -> bool:
    """Best-effort, rate-gated, dedup-guarded L1 for a ``git_error`` STORM.

    The G7 storm escape (task 4647).  ``git_error`` is a FAIL-SOFT
    DEGRADATION: one of them is a transient — a repo lock, a ref that lost a
    race — and says nothing about any task.  A STREAM of them says the
    detector itself is broken, and a broken landing detector is silent by
    construction, because every verdict it produces rejects and a rejecting
    detector is indistinguishable from a repo with nothing landed in it.  This
    is the one thing that makes that silence audible.

    **Strict exceedance.**  The configured rate is quiet; only ``> rate`` in
    the trailing window files.  That keeps the shipped default a ceiling an
    operator can reason about rather than a fence they trip over.

    **Exactly one alarm per storm.**  The dedup is ``has_open_l1`` under
    :data:`LANDING_GIT_ERROR_STORM_CATEGORY`, so re-observing the same storm
    on the next sweep re-files nothing.  One L1 per verdict would BE the storm
    this function exists to report.

    **Category-scoped, never a bare id.**  ``category=None`` matches ANY open
    L1, so an unrelated escalation would silently suppress this filing and a
    broken detector would hide behind whatever else happened to be open — the
    task-4105 two-way blindfold documented at
    :func:`file_unattributed_landing_escalation`.  Do not widen it.

    **Filed against a synthetic sentinel**, never a real task — see
    :data:`LANDING_GIT_ERROR_STORM_SENTINEL` for why filing it against a task
    would deepen the very strand it reports.

    Best-effort in the same sense as its sibling filer: a ``None`` queue is a
    supported shape (bare-harness and bare-worker unit tests construct exactly
    that), and any failure to build or submit the alarm is logged and
    swallowed.  Every verdict the producer returns is correct without it.

    Args:
        escalation_queue: The caller's ``EscalationQueue``, or ``None``.
        tally: The :class:`LandingTally` whose window and counts are reported.
        rate_per_hour: ``recovery_emission.landing_git_error_rate_per_hour``.
        agent_role: The filing role stamped on the alarm.

    Returns:
        True iff an escalation was actually submitted.  Reported rather than
        merely applied so a caller (and a test) can tell "suppressed by the
        dedup" from "below the rate" from "filed".
    """
    observed = tally.git_error_count_in_window()
    if observed <= rate_per_hour:
        return False
    if not escalation_queue:
        return False
    try:
        if escalation_queue.has_open_l1(
            LANDING_GIT_ERROR_STORM_SENTINEL,
            category=LANDING_GIT_ERROR_STORM_CATEGORY,
        ):
            return False
        from escalation.models import Escalation  # noqa: PLC0415

        window_hours = tally.window_secs / 3600.0
        summary = (
            f'Landing detector produced {observed} git_error verdicts in the '
            f'trailing {window_hours:.0f} hour (threshold {rate_per_hour}) — '
            'THE DETECTOR IS BROKEN, not the tasks unlanded'
        )
        detail = (
            f'Observed git_error verdicts: {observed}\n'
            f'Window: the trailing {window_hours:.0f} hour '
            f'({tally.window_secs:.0f}s)\n'
            f'Threshold (strict exceedance): {rate_per_hour} per hour '
            '(recovery_emission.landing_git_error_rate_per_hour)\n'
            f'Cumulative tally by reason: {tally.render()}\n'
            '\n'
            'WHAT THIS IS NOT: these are NOT not_landed verdicts.  '
            'not_landed is a claim ABOUT A TASK — its work is not on main, so '
            'dispatch it.  git_error is a claim about the DETECTOR — it could '
            'not look, and says nothing whatsoever about any task.  Do not '
            'read the rejections above as "these tasks never landed"; that '
            'inference, made by a human or by code, is the exact defect the '
            'landed-not-done-recovery PRD exists to prevent.\n'
            '\n'
            'WHY IT MATTERS: a repo lock, a corrupt object or an unresolvable '
            'ref reading as "not landed" re-dispatches tasks whose work is '
            'already on main, and keeps re-dispatching them, because '
            're-running the check does not fix whatever broke it.  While this '
            'holds, every landing verdict from this detector should be treated '
            'as unknown rather than negative.\n'
            '\n'
            'WHERE TO LOOK: the failing stage is named per verdict in '
            "probe['git_error_stage'] — resolve_branch_sha, no_op_baseline, "
            'net_diff_is_empty, patch_id_containment or unexpected_exception '
            "(the last also carries probe['exception']).  Grep the "
            "orchestrator log for 'landing tally' to see the per-reason "
            'counts over time.\n'
            '\n'
            f'This alarm is filed against a SYNTHETIC sentinel task id '
            f'({LANDING_GIT_ERROR_STORM_SENTINEL}), never against a real '
            'task: a storm spans every task the sweep touched, and an open L1 '
            'on a real task is read by the recovery predicates as a hold, so '
            'filing it there would deepen the very strand it reports.'
        )
        esc = Escalation(
            id=escalation_queue.make_id(LANDING_GIT_ERROR_STORM_SENTINEL),
            task_id=LANDING_GIT_ERROR_STORM_SENTINEL,
            agent_role=agent_role,
            severity='blocking',
            category=LANDING_GIT_ERROR_STORM_CATEGORY,
            summary=summary,
            detail=detail,
            suggested_action=(
                'Investigate git health in project_root and the task '
                'worktrees (index locks, corrupt objects, unresolvable refs) '
                'before acting on ANY recent landing verdict.  If the storm '
                'is understood and the alarm is the noisy part, retune or '
                'silence it live via the green-tier config leaves '
                'recovery_emission.{landing_git_error_rate_per_hour,'
                'landing_git_error_escalation_enabled} — no fleet restart '
                'needed.'
            ),
            level=1,
        )
        escalation_queue.submit(esc)
    except Exception:
        logger.warning(
            'Failed to file %s escalation for the landing detector '
            '(%d git_error verdicts observed) — continuing without it',
            LANDING_GIT_ERROR_STORM_CATEGORY, observed, exc_info=True,
        )
        return False
    logger.warning(
        'Filed %s escalation %s — %d git_error landing verdicts in the '
        'trailing %.0fs (threshold %d/hour); the DETECTOR is broken, not the '
        'tasks unlanded',
        LANDING_GIT_ERROR_STORM_CATEGORY, esc.id, observed,
        tally.window_secs, rate_per_hour,
    )
    return True
