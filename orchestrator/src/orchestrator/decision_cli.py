"""orchestrator.decision_cli — the decision verbs of the fleet registry CLI, served at ``session_registry.py <verb>``.

write-decision, close-decision, reopen-decision, reap-decisions and
migrate-decision-projects: their subparsers, their runners, and the decision
half of the CLI's exit-code policy (run_verb).

It imports only the stdlib, this checkout's stdlib-only ``shared`` leaves
(``shared.project_id``) and its sibling ``decision_registry``, whose names it
uses module-qualified so a test patch on that module reaches every runner. It
never imports ``session_registry``: in a bare-shell (tier-1) run
session_registry is ``__main__``, so an upward import would load a second copy
of it. The filer's record slug is session identity, so it arrives as
run_verb's injected ``filing_record_slug``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

# Reach this checkout's stdlib-only shared leaves from a bare shell; rationale in
# test_session_registry.py::TestStdlibOnlySelfContainment.
if (
    _CHECKOUT_SHARED_SRC := str(Path(__file__).resolve().parents[3] / 'shared' / 'src')
) not in sys.path and Path(_CHECKOUT_SHARED_SRC).is_dir():
    sys.path.append(_CHECKOUT_SHARED_SRC)

from shared.project_id import fold_project_token

# Imported by the spelling this module was loaded by: relative as
# orchestrator.decision_cli, bare as a tier-1 sibling or __main__.
# TYPE_CHECKING points the type checker at the package arm.
if TYPE_CHECKING or __package__:
    from . import decision_registry
else:
    import decision_registry

logger = logging.getLogger(__name__)

DECISION_VERBS: frozenset[str] = frozenset(
    {'write-decision', 'close-decision', 'reopen-decision', 'reap-decisions', 'migrate-decision-projects'}
)


def _read_decision_or_none(decision_id: str) -> decision_registry.DecisionRecord | None:
    """The record at *decision_id*, or None when it is absent, unreadable or corrupt."""
    try:
        return decision_registry.DecisionRecord.from_json(decision_registry.decision_path_for_id(decision_id).read_text())
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


@contextlib.contextmanager
def _locked_filing_target(
    canonical_project: str, local_id: str
) -> Iterator[tuple[str, decision_registry.DecisionRecord | None]]:
    """Lock and read the record a write-decision filing of *local_id* lands on.

    That is the project-qualified id, unless only a bare-keyed record filed
    before task 4835 exists at *local_id* for the SAME project: the filing
    then continues that record in place, keeping its custody and any
    held-closed state. Migration is lazy and ids are never rewritten, as in
    migrate_decision_project_tokens. Locks go qualified-then-legacy, and the
    legacy one only when its file exists, so a fresh filing leaves no orphan
    sidecar. See plans/4835-decision-plumbing-decisions.md.
    """
    qualified = decision_registry.qualify_decision_id(canonical_project, local_id)
    with decision_registry.decision_id_lock(qualified):
        if (
            local_id != qualified
            and not decision_registry.decision_path_for_id(qualified).exists()
            and decision_registry.decision_path_for_id(local_id).exists()
        ):
            with decision_registry.decision_id_lock(local_id):
                legacy = _read_decision_or_none(local_id)
                if (
                    legacy is not None
                    and fold_project_token(legacy.project) == canonical_project
                ):
                    yield local_id, legacy
                else:
                    yield qualified, None
            return
        yield qualified, _read_decision_or_none(qualified)


def _run_write_decision(
    local_id: str,
    project: str,
    text: str,
    task_id: str | None,
    escalation_id: str | None,
    session_id: str | None,
    severity: str = '',
    escalations_dir: str = '',
    *,
    record_slug: str = '',
) -> None:
    """Run the ``write-decision`` verb (Fleet Cockpit C8: park-to-registry).

    Files an OPEN DecisionRecord from a watcher's park / tell-the-human
    moment, IN ADDITION to its in-session note / afk-digest line, so the
    cockpit decision queue (C5b) becomes the primary return-triage surface.
    ``state`` is left at its DecisionState.OPEN default (a watcher only ever
    files open decisions; state transitions are the cockpit's job via
    update_decision_state). Root resolves via $CLAUDE_FLEET_ROOT, same as
    every other verb.

    ``severity`` threads the parked escalation's severity (info|blocking|
    critical|urgent) onto the record so the cockpit decision queue can
    weight this ask (Fleet Cockpit F7 fix 1); defaults to '' when the
    caller doesn't supply one.

    ``record_slug`` is best-effort enrichment derived from ``$CLAUDE_PID``'s
    pid pointer (session_registry.py::resolve_own_record_slug), never typed
    by the caller and never a reason to refuse or delay a filing: an
    unresolvable one is filed as '', is announced by a WARNING at dispatch
    (session_registry.py::_resolve_filing_record_slug, run_verb's injected
    filing_record_slug), and is visible downstream as an unresolved link.

    ``project`` is stored NORMALIZED (see shared.project_id.fold_project_token,
    task 3807), so ``df``, ``dark-factory`` and ``Dark_Factory`` all persist as
    the one canonical ``dark_factory`` and ``reap-decisions`` can no longer
    miss a partition. Only ``DecisionRecord.project`` is folded on write:
    ``SessionRecord.project`` stays raw because it feeds
    ``session_registry.py::build_session_slug``, and the cockpit folds both record kinds at its
    read boundary (``cockpit/src/cockpit/registry_reader.py``), so any other
    reader comparing a raw ``SessionRecord.project`` must fold it itself.
    This NORMALIZES an unrecognized spelling rather than
    REJECTING it, deliberately: (1) this verb is fail-soft by contract --
    run_verb returns 0 and write_decision never raises -- so a rejection
    would silently DROP the decision, which is precisely the
    fail-into-invisibility harm the whole verb exists to prevent; a human's
    parked question would vanish instead of appearing under a slightly-off
    token. (2) An allowlist would mean a genuinely new project cannot file a
    decision until someone ships a code change, turning a data-hygiene rule
    into an availability dependency. A rewrite is logged at WARNING so it
    stays LOUD rather than silent.

    ``local_id`` (``--id``) is project-LOCAL (task 4835): the record is
    filed under ``qualify_decision_id(canonical project, local_id)``, so the
    same escalation id in two projects lands on two rows, while both queues
    of ONE project still share a row (the MODE-2 collapse below). A
    same-project record filed before then under the bare id is continued in
    place (_locked_filing_target), and the printed id is the one landed on.

    ``escalations_dir`` names the escalation QUEUE *escalation_id* belongs
    to, and is stored NORMALIZED (see normalize_escalations_dir). A watcher
    passes the SAME queue dir it later reaps with, so this fleet-global
    decision can be joined back to the right per-queue escalation-id
    namespace: ids are unique only within one queue, and a project may run
    several (task 3528). It is MANDATORY (task 3559), on two layers: the
    parser marks it ``required=True`` so an OMITTED flag exits 2, and this
    verb additionally refuses a stamp that NORMALIZES to nothing -- which
    argparse cannot see, since ``--escalations-dir ''`` satisfies a required
    flag. An unstamped fleet-global record is cross-queue-ambiguous, and is
    exactly the legacy population task 3640 had to back-fill; it must not be
    allowed to regrow through this verb.

    ``UNKNOWN_QUEUE`` is likewise refused HERE, even though it normalizes
    non-empty. It is a back-fill-only sentinel (set_decision_escalations_dir,
    task 3640) meaning "investigated, could not determine", and every reaper
    refuses a decision so stamped -- so a record filed with it could only
    ever be closed by hand, which is strictly worse than the '' record this
    verb already refuses. A watcher always knows its own queue by
    construction (it is the same dir it passes to reap-decisions), so there
    is no legitimate write-path caller.

    UPSERT, NOT A BLIND OVERWRITE (task 3559). Against an existing record at
    the same id, three cases are told apart:

    - DIFFERENT project -- an id COLLISION, not one gate seen twice. Since
      task 4835 qualifies the id by project, this is reachable only when the
      qualified id is itself held by another project's record: a legacy
      hand-prefixed id such as the live ``recon-esc-7459-1`` held by project
      reify. REFUSED (loud, fail-soft, nothing written): merging would hide
      this ask inside the other project's row and overwriting would delete
      that row. Scoped to an OPEN incumbent -- a closed row in another
      project is a question already dealt with, so a filing there starts a
      new ask.
    - DIFFERENT queue stamp -- the second watcher observing the same human
      gate through another queue (the observed esc-5914-1 MODE-2 shape), so
      the filing is folded in via merge_decision_enrichment rather than
      clobbering or downgrading what the first watcher wrote. Also scoped to
      an OPEN incumbent, and for the same reason: ACROSS queues the
      ``esc-<taskid>-<n>`` namespaces genuinely collide, so a filing against
      a closed row may be an unrelated NEW ask.
    - MATCHING queue stamp -- the SAME watcher re-filing across a restart,
      which both SKILL.md files promise is idempotent, so its whole view
      lands (including fields going down or empty). ``filed_at``,
      ``manual_boost`` AND ``state`` are held back, as CUSTODY: a restart is
      not news about queue age, says nothing about the operator's C5 boost
      (set_manual_boost's field), and is not a disposition
      (update_decision_state's). Same custody set merge_decision_enrichment
      keeps -- custody does not depend on which queue re-filed.

      THIS ARM IS NOT SCOPED TO AN OPEN RECORD (task 3872), unlike the two
      above. Within ONE queue an ``esc-<taskid>-<n>`` id is unique (task
      3528's premise), so a same-project same-queue re-file is definitively
      the SAME gate the human already answered or dropped -- identity is
      certain here in a way it is not on the cross-queue axis. The harm it
      removes is concrete and unbounded: both watcher SKILLs tell an agent to
      re-file its stable id on EVERY restart while an item stays parked, and
      reap_answered_decisions skips a non-open decision, so without this an
      operator's C5b dismissal is undone on the next restart and again
      forever -- the drop action is inert for exactly the class of row it
      exists for. Holding the state back is announced with a WARNING (below),
      since this is the one place the verb declines to do what the filer
      asked.

    Everything else keeps today's full overwrite: no existing record (the
    normal first-filing case), an unreadable/corrupt one, or a non-open one
    reached on either QUEUE-axis exception above -- a different project, or a
    different queue stamp (including the legacy unstamped population task
    3640 is draining, whose '' stamp matches no real queue). In each of those
    the incumbent cannot be shown to be the same gate, so a new filing starts
    a new ask, boost and age included.

    On success, prints the filed record's id (mirrors `launching` printing
    the record dir and `lease-claim` printing `decision=`) so the caller can
    cross-link it into its in-session note or afk-digest line. write_decision
    is itself self-guarding fail-soft (never raises), so a fault there is
    silently skipped here rather than printed as a false confirmation.

    Concurrency NOTE (see DecisionRecord's docstring): the read -> merge ->
    write span above is serialized per-decision-id via decision_id_lock (a
    stable ``<id>.json.lock`` sidecar), so two watchers filing the SAME id
    from two queues at the same moment cannot drop either one's
    contribution. That matters more here than for the field setters: this is
    the only mutator that may CREATE the record, so an unserialized span
    races on a path where nothing exists on disk yet.

    Intentionally omits DecisionRecord's ``options`` field: C8 watchers only
    ever file plain open/text decisions, and the peer `update_decision_state`
    / `set_manual_boost` helpers already own post-filing mutation (state,
    boost). If a watcher ever needs to offer candidate answers up front, add
    an optional repeatable ``--option`` flag here rather than widening this
    verb's other args.
    """
    # CRITICAL: both stamp guards live HERE, in the CLI verb, and NOT in
    # write_decision() -- do not "helpfully" push them down into the writer.
    # write_decision() is the single atomic path the C8 watcher and the C5
    # cockpit share by PRD design (fleet-cockpit-prd.md:177-179), and it has
    # direct in-repo callers that legitimately write queue-less records:
    # cockpit/tests/test_app.py, and tests/scripts/test_backfill_decision_
    # queue_stamp.py, which must be able to CONSTRUCT the unstamped legacy
    # population in order to test back-filling it. The contract being
    # enforced is "how a WATCHER invokes write-decision", which is this
    # verb's boundary, not the writer's.
    stamp = decision_registry.normalize_escalations_dir(escalations_dir)
    if not stamp:
        logger.error(
            'write-decision refusing to file %s: --escalations-dir is mandatory and '
            'normalized to nothing. A DecisionRecord is fleet-global but an '
            'esc-<taskid>-<n> id is unique only within one queue, so a queue-less '
            'record is cross-queue-ambiguous. Pass the SAME queue dir you reap with.',
            local_id,
        )
        return
    if stamp == decision_registry.UNKNOWN_QUEUE:
        logger.error(
            'write-decision refusing to file %s: --escalations-dir %s is a '
            'back-fill-only sentinel (set_decision_escalations_dir, task 3640) and no '
            'reaper will ever close a decision stamped with it -- only a human could. '
            'A watcher knows its own queue by construction; pass the real queue dir '
            'you also reap with.',
            local_id,
            decision_registry.UNKNOWN_QUEUE,
        )
        return

    canonical_project = fold_project_token(project)
    if canonical_project != project:
        logger.warning(
            'write-decision: normalized --project %r -> %r', project, canonical_project
        )

    # Advisory only (task 3813). Placed AFTER canonicalization so the hint
    # reflects the token that will actually be STORED, and BEFORE the record
    # is built so it fires even if a later step fails. It must not alter
    # canonical_project, gate the filing, or change the return code.
    # action='file' because this path CREATES rather than matches: the reap
    # wording ("matches no decisions") would be false one line below, where
    # a row is filed under exactly this token. The filing consequence is the
    # worse of the two -- that row lands in a bucket no documented reap
    # scopes to and can never auto-close -- so it is the one worth naming.
    declined_hint = decision_registry.declined_project_token_hint(canonical_project, action='file')
    if declined_hint is not None:
        logger.warning('write-decision: %s', declined_hint)

    # WHY THIS IS NOT ROUTED THROUGH _mutate_decision, despite sharing its
    # read-modify-write shape: _mutate_decision is a STRICT read-modify-write
    # -- it does DecisionRecord.from_json(path.read_text()) and fail-softs to
    # None when the file is absent. Here, absent is the NORMAL case (the first
    # filing of a new decision), so delegating would make CREATING a record
    # impossible. This verb is an UPSERT: read-or-construct, merge, write.
    # What genuinely must not diverge are the invariants that helper's
    # docstring identifies as load-bearing, so they are mirrored verbatim:
    # its exact (OSError, json.JSONDecodeError, KeyError, TypeError,
    # ValueError) read/parse/write fault set, and (step-14) the lock sitting
    # INSIDE the try/except so an acquisition fault is absorbed rather than
    # raised at a watcher.
    # The lock is taken HERE, after the stamp guards, so a refused
    # invocation never creates an orphan `<id>.json.lock` sidecar for an id
    # that will never be written (the ORPHAN SIDECARS caveat in
    # decision_id_lock's own docstring), and INSIDE the try/except so a
    # lock-acquisition fault is absorbed rather than raised at a watcher.
    try:
        with _locked_filing_target(canonical_project, local_id) as (decision_id, existing):
            incoming = decision_registry.DecisionRecord(
                id=decision_id,
                project=canonical_project,
                text=text,
                filed_at=datetime.now(UTC).isoformat(),
                task_id=task_id,
                escalation_id=escalation_id,
                session_id=session_id,
                severity=severity,
                escalations_dir=stamp,
                record_slug=record_slug,
            )
            record = incoming
            if existing is not None:
                if fold_project_token(existing.project) != canonical_project:
                    # SAME id, DIFFERENT project: an id COLLISION, not a
                    # MODE-2 collapse. The id is project-qualified (task
                    # 4835), so this is reachable only when the qualified id
                    # is held by a LEGACY hand-prefixed record of another
                    # project (the live 'recon-esc-7459-1' held by reify is
                    # the shape). Two projects also always have different
                    # queue dirs, so without this guard such a collision
                    # lands in the enrichment branch below and gets folded
                    # into the OTHER project's row -- one cockpit row
                    # claiming to be project A's ask while B's human gate is
                    # invisible and unreapable by B's reaper.
                    #
                    # BOTH sides go through fold_project_token (task
                    # 3807), exactly as the reap-decisions project axis does.
                    # A raw compare here would read one project's own
                    # historical spellings as a cross-project collision --
                    # a watcher passing `--project df` against a stored
                    # `dark_factory` row (this verb now stamps the canonical
                    # form) would be REFUSED, turning the partition bug 3807
                    # removes into a hard filing failure. Normalizing only
                    # ever merges spellings of the SAME project, so the
                    # collision this guard exists to catch still trips.
                    #
                    # Refuse rather than overwrite: overwriting would DELETE a
                    # live row (with any operator boost and state on it),
                    # which is the clobber this whole task exists to stop, and
                    # leaves exactly one of the two gates visible either way.
                    # Refusing is the non-destructive, deterministic choice --
                    # the same first-writer-wins policy _merge_queue_and_
                    # escalation_id applies to a conflicting queue stamp.
                    #
                    # STILL scoped to an OPEN incumbent after task 3872, on
                    # the QUEUE axis's own reasoning: two projects always run
                    # different queue dirs, so a cross-project collision is by
                    # construction a CROSS-queue filing, where the
                    # esc-<taskid>-<n> namespaces genuinely collide and a
                    # closed row cannot be shown to be the same gate. A filing
                    # there is a new ask, so `record` stays `incoming` and it
                    # keeps today's full overwrite.
                    if existing.state == decision_registry.DecisionState.OPEN:
                        logger.error(
                            'write-decision refusing to file %s for project %s: an OPEN '
                            'decision already exists at that project-qualified id for a '
                            'DIFFERENT project (%s) -- a legacy hand-prefixed id filed '
                            'before ids were qualified by project. This is an id '
                            'COLLISION, not a MODE-2 cross-queue collapse of one human '
                            'gate -- merging would hide this ask inside the other '
                            'project\'s cockpit row and overwriting would delete that '
                            'row, so neither is safe. The existing row is left intact '
                            'and THIS ask did not reach the cockpit; it is still '
                            'carried by the in-session note / afk-digest line this '
                            'filing accompanies. Re-file it under a different --id.',
                            decision_id,
                            project,
                            existing.project,
                        )
                        return
                elif decision_registry.normalize_escalations_dir(existing.escalations_dir) == stamp:
                    # A MATCHING stamp is the SAME watcher re-filing across a
                    # restart, which both SKILL.md files promise is a plain
                    # idempotent overwrite -- text, severity and task_id are
                    # its own to revise, including downwards -- with the
                    # CUSTODY fields (filed_at, manual_boost, state) held
                    # back. merge_same_queue_refile carries the reasoning.
                    #
                    # NOT scoped to an OPEN record (task 3872), unlike the two
                    # arms around it. This is the ONE axis on which identity
                    # is knowable: within a single queue an esc-<taskid>-<n>
                    # id is unique (task 3528's premise), so a same-project
                    # same-queue re-file is the same gate the human already
                    # answered or dropped -- modulo the counter-loss re-mint
                    # hole merge_same_queue_refile names, which the WARNING
                    # just below is the backstop for. Re-opening it would
                    # make an operator's C5b dismissal impossible to ever make
                    # stick -- a watcher re-files on EVERY restart while an
                    # item stays parked, and reap_answered_decisions skips a
                    # non-open record, so nothing would ever close it again.
                    record = decision_registry.merge_same_queue_refile(existing, incoming)
                    if existing.state != decision_registry.DecisionState.OPEN:
                        # THE ONE PLACE THIS VERB DECLINES WHAT THE FILER
                        # ASKED FOR: the filing carries state=open (the
                        # default above) and the row stays closed. Loud, per
                        # this repo's loud-over-silent-degradation norm, and
                        # because the watcher SKILLs' own rule is to
                        # ADJUDICATE such a divergence rather than assume the
                        # re-file landed -- which needs it to be visible.
                        #
                        # Guarded on non-open so the COMMON path stays quiet:
                        # a same-queue re-file against an open record is every
                        # watcher restart for every still-parked item, nothing
                        # is declined there, and a warning on each one is how
                        # this one gets tuned out.
                        #
                        # stdout is deliberately untouched (the id is still
                        # printed below): both SKILLs document "no id on
                        # stdout means your filing did not land" as the
                        # did-it-work signal, and this filing DID land.
                        logger.warning(
                            'write-decision kept decision %s in state %r rather than '
                            're-opening it: this re-file came from the SAME queue (%s) '
                            'as the record it matched, so it is the same gate a human '
                            'already answered or dropped, not a new ask. Your text, '
                            'severity and ids DID land, but the row stays CLOSED and '
                            'will NOT reappear in the cockpit decision queue, which '
                            'shows only state=open rows. Re-opening it on a re-file would '
                            'make an operator\'s cockpit disposition impossible to ever '
                            'make stick, since a watcher re-files its stable id on every '
                            'restart while an item stays parked. ADJUDICATE this rather '
                            'than re-filing blindly: if this SAME gate genuinely needs a '
                            'human again, re-open the row in place with '
                            '`session_registry.py reopen-decision --id %s --project %s '
                            '--escalations-dir %s`; file under a NEW id only for a '
                            'genuinely DIFFERENT ask.',
                            decision_id,
                            str(existing.state),
                            stamp,
                            decision_id,
                            canonical_project,
                            stamp,
                        )
                elif existing.state == decision_registry.DecisionState.OPEN:
                    # A DIFFERENT queue filing against a live record: this is
                    # the MODE-2 cross-queue collapse, so enrich rather than
                    # overwrite.
                    record = decision_registry.merge_decision_enrichment(existing, incoming)
                # else: a DIFFERENT queue filing against a NON-open record --
                # including the legacy unstamped ('' stamp) population task
                # 3640 is draining, which matches no real queue. Across queues
                # the id namespaces collide, so this may be an unrelated NEW
                # ask; holding it closed would make a live gate invisible,
                # which is the fail-CLOSED direction _run_reap_decisions'
                # docstring rules out. `record` stays `incoming`: today's full
                # overwrite, unchanged.

            if decision_registry.write_decision(record):
                print(record.id)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        logger.error(
            'write-decision: failed to file %s for project %s',
            local_id,
            canonical_project,
            exc_info=True,
        )


def _reap_scope_hint(reaper_project: str, *, expect_matches: bool) -> str | None:
    """The one advisory reap-decisions logs: the declined-alias special case first, else the opt-in census."""
    declined = decision_registry.declined_project_token_hint(reaper_project, action='reap')
    if declined is not None:
        return declined
    if expect_matches:
        census = decision_registry.project_token_census(decision_registry.list_decisions())
        return decision_registry.unmatched_project_token_hint(reaper_project, census)
    return None


def _run_reap_decisions(project: str, escalations_dir: str, *, expect_matches: bool = False) -> None:
    """Run the ``reap-decisions`` verb (Fleet Cockpit C8: close-on-resolve driver).

    Builds the production ``escalation_status`` closure for
    reap_answered_decisions. The (fleet-global decisions <-> per-queue
    escalations) join is scoped on TWO axes, and a decision failing either
    one is left alone (returns None -- unresolved):

    1. PROJECT. DecisionRecords are fleet-global
       (``~/.claude/fleet/decisions/``) but escalations are per-project
       (``<project_root>/data/escalations``), so a decision belonging to a
       DIFFERENT project is skipped. Scoping this way keeps the join correct
       without fragile project_id -> path auto-discovery. The match is on the
       CANONICAL token (see fold_project_token, task 3807), with BOTH
       sides normalized -- the reaper's own ``--project`` AND the decision's
       stored value -- so a single run covers every historical spelling of
       one project. A raw compare silently PARTITIONED one project's
       decisions: measured 2026-08-06, 41 OPEN dark-factory decisions split
       22/17/2 across ``dark_factory``/``df``/``dark-factory``, each
       partition invisible to a reap scoped to either of the others, forcing
       an L2 session to run this verb once per token. That
       run-once-per-token workaround is now removed. Normalization only ever
       WIDENS the axis-1 match to spellings of the SAME project; it never
       widens ACROSS projects (``solar_challenge`` vs
       ``solar_challenge_platform`` are different project roots and are
       guarded from merging), so the fail-OPEN framing below is intact.
       A ``--project`` naming a DECLINED alias target now WARNS (task 3813,
       see PROJECT_TOKEN_ALIASES_DECLINED), so an operator passing a
       config-declared token that matches zero rows learns it immediately
       instead of reading a silent no-op as "nothing to reap". The general
       case is opt-in: with *expect_matches* (``--expect-matches``, task
       4835) a token no registry record carries in ANY state warns and
       names the tokens that do. It is not always on because by count alone
       a never-filed project cannot be told apart from a mismatched one, and
       a watcher reaps every Main Loop cycle. Both are advisory, one line
       per invocation (_reap_scope_hint): neither changes the axis, the
       scoping, or what gets closed.
    2. QUEUE (task 3528). An escalation id (``esc-<taskid>-<n>``) is unique
       only WITHIN one queue, and a project can run several: dark_factory
       runs ``data/escalations`` (orchestrator) and
       ``data/reconciliation/escalations`` (recon watcher) over the SAME id
       namespace. Project scoping alone therefore let either watcher's
       reaper resolve the OTHER watcher's decisions against its own queue --
       observed with ``esc-3036-1``, where an unrelated resolved
       orchestrator escalation silently closed a still-pending recon
       blocking gate, which then sat invisible in the cockpit for ~7 days
       (15 ids were resolved in both queues at time of measurement). So a
       decision stamped with a queue OTHER than *escalations_dir* is
       skipped. Both sides go through normalize_escalations_dir, including
       the decision's own stored value, so equivalent spellings match and a
       hand-written record is compared honestly rather than fail-open.

    A decision carrying NO queue (``escalations_dir == ''`` -- every record
    filed before that field existed) falls back to axis 1 alone, keeping
    today's exact behaviour for the existing population rather than changing
    it under the human; the two watchers stamp the queue on newly-filed
    decisions, so the unprotected set only shrinks. Task 3640 then BACK-FILLED
    the pre-existing open population, so that fallback set is drained rather
    than merely shrinking. Otherwise the closure defers to
    read_escalation_status against *escalations_dir*. Root resolves via
    $CLAUDE_FLEET_ROOT, same as every other verb.

    UNKNOWN QUEUE (task 3640). A decision stamped ``UNKNOWN_QUEUE`` is
    refused by name: its owning queue was investigated and could not be
    determined, so NO reaper may close it and it stays a visible cockpit row
    until a human closes it. Honestly: the axis-2 compare above would already
    refuse it, since it refuses ANY truthy stamp that is not this reaper's
    queue -- so the by-name guard changes no outcome today. It is here so the
    policy is INTENTIONAL and named rather than an accident of string
    inequality that a future refactor of that compare could silently remove,
    and because it closes the one degenerate case the compare genuinely does
    not cover: a reaper invoked with the sentinel as its OWN
    ``--escalations-dir`` makes both sides compare EQUAL, and the record then
    survives only because read_escalation_status happens to find no queue at
    that bogus relative path.

    Both guards are fail-OPEN: on any doubt the decision stays OPEN and
    visible in the cockpit queue, matching reap_answered_decisions' contract
    that any status not in DECISION_CLOSE_MAP leaves a decision OPEN. The
    asymmetry of harm is the reason -- an over-held decision is a
    human-triageable row, while a falsely closed one is invisible.

    A watcher runs this once per Main Loop cycle (SKILL.md "Closing parked
    decisions on resolve") so a decision it (or anyone -- /unblock, an L2
    cascade, the human) files closes once its escalation is actually
    resolved/dismissed. Prints one ``f'{id} {new_state}'`` line per closed
    decision, mirroring `write-decision` printing the filed id.
    """
    reaper_dir = decision_registry.normalize_escalations_dir(escalations_dir)
    reaper_project = fold_project_token(project)

    # Advisory only (tasks 3813, 4835). Deliberately OUTSIDE _status, so it
    # fires ONCE per invocation rather than once per record scanned -- a
    # watcher runs this every Main Loop cycle and a per-record line would
    # flood its log. It must not touch reaper_project, neither scoping axis,
    # nor what gets closed: both guards below stay fail-OPEN as documented.
    scope_hint = _reap_scope_hint(reaper_project, expect_matches=expect_matches)
    if scope_hint is not None:
        logger.warning('reap-decisions: %s', scope_hint)

    def _status(decision: decision_registry.DecisionRecord) -> str | None:
        # Axis 1: normalize the decision's OWN stored token at compare time
        # too, not just at write time -- for the same reason axis 2 already
        # does. A raw compare fails open for any record write-decision did
        # not produce (hand-repaired, migrated between checkouts, filed by an
        # older build), which is exactly the 173 legacy records the migration
        # verb exists to repair; comparing honestly means the fix holds even
        # for a record the migration has not reached yet.
        if fold_project_token(decision.project) != reaper_project:
            return None
        if decision.escalation_id is None:
            # Belt-and-suspenders, not reachable via reap_answered_decisions
            # today: it already skips any decision with a falsy
            # escalation_id before ever calling this closure. Kept so this
            # narrows `str | None` -> `str` for the read_escalation_status
            # call below (typed to take a `str`) and stays correct if this
            # closure is ever invoked from elsewhere.
            return None
        # Axis 2: normalize the decision's OWN stored value at compare time
        # too, not just at write time -- a raw compare would fail open for
        # any record write-decision did not produce (hand-repaired, migrated
        # between checkouts, or written by a future caller), and a false
        # NON-match is how silent divergence creeps back in.
        decision_dir = decision_registry.normalize_escalations_dir(decision.escalations_dir)
        # Task 3640: refuse an undeterminable-queue record BY NAME, before the
        # inequality compare below -- see the UNKNOWN QUEUE paragraph above
        # for why this deliberately-redundant-today guard exists.
        if decision_dir == decision_registry.UNKNOWN_QUEUE:
            return None
        if decision_dir and decision_dir != reaper_dir:
            return None
        # RAW escalations_dir here, deliberately: only the comparison needs a
        # canonical spelling; the read keeps using exactly what the caller
        # passed.
        return decision_registry.read_escalation_status(escalations_dir, decision.escalation_id)

    for reaped in decision_registry.reap_answered_decisions(escalation_status=_status):
        print(f'{reaped.id} {reaped.new_state}')


def _run_close_decision(
    decision_id: str,
    state: str,
    evidence: str,
    root: str | None,
    *,
    expected_project: str,
    expected_escalations_dir: str,
) -> int:
    """Run the ``close-decision`` verb, whose failure, like ``reopen-decision``'s, is a non-zero exit.

    Its caller is an agent executing a pre-built apply payload
    (``scripts/sitting/payloads.py::close_decision_argv``), not
    spawn-claude.sh, so a refusal or an unreadable record must be seen rather
    than read as success. Prints the closed record's id on success.
    """
    try:
        record = decision_registry.close_decision_with_evidence(
            decision_id,
            state,
            evidence,
            root=root,
            expected_project=expected_project,
            expected_escalations_dir=expected_escalations_dir,
        )
    except decision_registry.DecisionCloseRefused as exc:
        print(f'close-decision refused: {exc}', file=sys.stderr)
        return 1
    if record is None:
        print(f'close-decision: {decision_id} has no readable record to close (see the ERROR log)', file=sys.stderr)
        return 1
    print(record.id)
    return 0


def _run_reopen_decision(
    decision_id: str,
    root: str | None,
    *,
    expected_project: str,
    expected_escalations_dir: str,
) -> int:
    """Run the ``reopen-decision`` verb; like close-decision, a refusal or an unreadable record exits non-zero.

    Its caller is a human or agent adjudicating a held-closed row, who must
    see a refusal rather than read it as success. Prints the record's id on
    success, plus a stderr WARNING when the linked escalation is already
    terminal in the stamped queue: the next reap-decisions cycle then closes
    the row again, so the reopen alone cannot keep it open.
    """
    try:
        record = decision_registry.reopen_decision(
            decision_id,
            root=root,
            expected_project=expected_project,
            expected_escalations_dir=expected_escalations_dir,
        )
    except decision_registry.DecisionReopenRefused as exc:
        print(f'reopen-decision refused: {exc}', file=sys.stderr)
        return 1
    if record is None:
        print(f'reopen-decision: {decision_id} has no readable record to reopen (see the ERROR log)', file=sys.stderr)
        return 1
    print(record.id)
    closing_status = _reapable_escalation_status(record)
    if closing_status is not None:
        print(
            f'reopen-decision: WARNING {record.id} is open again, but its escalation '
            f'{record.escalation_id} is already {closing_status!r} in {record.escalations_dir}, so the '
            'next reap-decisions cycle closes it again. A gate that needs a human again needs a new '
            'escalation, filed under its own --id.',
            file=sys.stderr,
        )
    return 0


def _reapable_escalation_status(record: decision_registry.DecisionRecord) -> str | None:
    """The terminal status the reaper would close *record* on, read from its stamped queue; else None.

    The reaper's own join (_run_reap_decisions._status) for a record whose
    queue is known. None when the record has no escalation id, is unstamped
    or stamped UNKNOWN_QUEUE (no reaper can be named), or its escalation is
    not in DECISION_CLOSE_MAP.
    """
    queue = decision_registry.normalize_escalations_dir(record.escalations_dir)
    if not record.escalation_id or queue in ('', decision_registry.UNKNOWN_QUEUE):
        return None
    status = decision_registry.read_escalation_status(queue, record.escalation_id)
    return status if status in decision_registry.DECISION_CLOSE_MAP else None


def _run_migrate_decision_projects(dry_run: bool) -> None:
    """Run the ``migrate-decision-projects`` verb (task 3807).

    The one-shot backfill for DecisionRecords filed BEFORE ``write-decision``
    canonicalized ``--project`` (see migrate_decision_project_tokens for the
    preservation invariants -- ``state`` and ``filed_at`` are carried
    through, so an already-answered row is never reopened, and the record's
    id/filename is never rewritten).

    It is IDEMPOTENT, so a re-run once the fleet is clean prints nothing.
    That is what makes it safe to keep permanently rather than delete after
    first use: it doubles as the repair tool for a hand-edited or
    externally-written record, with no need to reason about whether it has
    already run. Run it with ``--dry-run`` first to preview.

    Prints one ``f'{id} {old_project} -> {new_project}'`` line per migrated
    record, mirroring `reap-decisions` printing one line per closed decision.
    Root resolves via $CLAUDE_FLEET_ROOT, same as every other verb.
    """
    for migrated in decision_registry.migrate_decision_project_tokens(dry_run=dry_run):
        print(f'{migrated.id} {migrated.old_project} -> {migrated.new_project}')


def _add_record_identity_args(verb_parser: argparse.ArgumentParser) -> None:
    """The compare-and-swap expectations close-decision and reopen-decision name a record by."""
    verb_parser.add_argument(
        '--project',
        required=True,
        help=(
            "compare-and-swap expectation of the record's CURRENT project, not a stamp "
            "like write-decision's: a record at --id whose folded project differs is refused"
        ),
    )
    verb_parser.add_argument(
        '--escalations-dir',
        required=True,
        help=(
            "compare-and-swap expectation of the record's CURRENT queue stamp, not a stamp "
            "like write-decision's, so '' is a legal expectation for a legacy unstamped record; "
            'a record at --id whose normalized stamp differs is refused'
        ),
    )



def add_verb_parsers(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the DECISION_VERBS subparsers on the CLI's *sub*."""
    write_decision_p = sub.add_parser(
        'write-decision',
        help='file an OPEN DecisionRecord (Fleet Cockpit C8: park-to-registry)',
    )
    write_decision_p.add_argument(
        '--id',
        required=True,
        help=(
            "this decision's id LOCAL to --project (usually the escalation id); "
            'filed fleet-unique as <project>-<id>, which is printed'
        ),
    )
    write_decision_p.add_argument(
        '--project',
        required=True,
        help=(
            "this decision's project; stored NORMALIZED to one canonical token "
            '(see fold_project_token), so df/dark-factory/Dark_Factory all '
            'persist as dark_factory'
        ),
    )
    write_decision_p.add_argument('--text', required=True, help='the decision/question text')
    write_decision_p.add_argument('--task-id', default=None)
    write_decision_p.add_argument('--escalation-id', default=None)
    write_decision_p.add_argument('--session-id', default=None)
    write_decision_p.add_argument(
        '--severity',
        default='',
        help='escalation severity to weight this ask (info|blocking|critical|urgent)',
    )
    write_decision_p.add_argument(
        '--escalations-dir',
        required=True,
        help=(
            "MANDATORY: the escalation queue dir this decision's --escalation-id "
            'belongs to; scopes the reaper (see reap-decisions). Required because a '
            'DecisionRecord is fleet-global while an esc-<taskid>-<n> id is unique '
            'only WITHIN one queue (task 3528), so an unstamped record is '
            'cross-queue-ambiguous -- the legacy population task 3640 back-filled '
            'out of. Pass the SAME queue dir you later reap with.'
        ),
    )

    close_decision_p = sub.add_parser(
        'close-decision',
        help='close a decision to answered|dropped, quoting the deciding evidence (task 5376)',
    )
    close_decision_p.add_argument('--id', required=True, help="the decision's id")
    close_decision_p.add_argument('--state', required=True, help='answered or dropped')
    close_decision_p.add_argument('--evidence', required=True, help='the deciding evidence, verbatim')
    _add_record_identity_args(close_decision_p)
    close_decision_p.add_argument('--root', default=None, help='fleet root (default: fleet_root())')

    reopen_decision_p = sub.add_parser(
        'reopen-decision',
        help='re-open a closed decision in place, clearing its closing evidence (task 4835)',
    )
    reopen_decision_p.add_argument(
        '--id', required=True, help='the record id as stored/printed by write-decision'
    )
    _add_record_identity_args(reopen_decision_p)
    reopen_decision_p.add_argument('--root', default=None, help='fleet root (default: fleet_root())')

    # NOTE: --escalations-dir is required on BOTH halves of the file/reap
    # pair. reap-decisions has always required it; write-decision joined it
    # in task 3559, so the two are symmetric rather than each inventing a
    # convention.
    reap_decisions_p = sub.add_parser(
        'reap-decisions',
        help='close OPEN decisions whose escalation has resolved/dismissed (Fleet Cockpit C8)',
    )
    reap_decisions_p.add_argument(
        '--project',
        required=True,
        help=(
            'project to scope the reap to; matched on the canonical token '
            '(see fold_project_token), so ONE run closes every '
            'historical spelling of that project'
        ),
    )
    reap_decisions_p.add_argument('--escalations-dir', required=True)
    reap_decisions_p.add_argument(
        '--expect-matches',
        action='store_true',
        help=(
            'opt-in: warn when no registry record, in any state, carries the folded '
            '--project token, naming the tokens that do; for diagnosis and manual runs, '
            'not the per-cycle Main Loop reap'
        ),
    )

    migrate_projects_p = sub.add_parser(
        'migrate-decision-projects',
        help='canonicalize DecisionRecord.project onto one token per project',
    )
    migrate_projects_p.add_argument(
        '--dry-run',
        action='store_true',
        help='print what would be migrated without writing anything',
    )


def run_verb(args: argparse.Namespace, *, filing_record_slug: Callable[[str, str], str]) -> int:
    """Run the decision verb *args* names and return the CLI's exit code for it.

    write-decision, reap-decisions and migrate-decision-projects are
    fail-soft: a fault in their core logic is logged loudly (stderr, via the
    standard logging machinery) and swallowed, so a registry fault can never
    change the exit code of the bash caller. write-decision resolves its
    record slug through *filing_record_slug* (local id, raw --project) inside
    that same span.

    That contract has always covered runtime FAULTS only -- never a
    malformed INVOCATION. argparse's own required-argument path raises
    SystemExit(2) from session_registry.main's parse_args(), before this
    function is reached, and has done so since the parser existed:
    `write-decision` with no --id/--project/--text exits 2 today.
    Task 3559 deliberately put --escalations-dir in that same louder class
    rather than swallowing it, because a queue-less DecisionRecord is
    cross-queue-ambiguous and silently filing one is worse than a hard stop.
    A runtime refusal (e.g. an explicitly EMPTY stamp, which argparse cannot
    distinguish from a supplied one) stays fail-soft: ERROR log, nothing
    written, nothing printed, rc 0.

    ``close-decision`` and ``reopen-decision`` are the two exceptions, and
    they run outside the swallowing try/except: see _run_close_decision and
    _run_reopen_decision for why their refusals exit non-zero.
    """
    if args.verb not in DECISION_VERBS:
        raise ValueError(f'{args.verb!r} is not a decision verb; expected one of {sorted(DECISION_VERBS)}')
    if args.verb == 'close-decision':
        return _run_close_decision(
            args.id,
            args.state,
            args.evidence,
            args.root,
            expected_project=args.project,
            expected_escalations_dir=args.escalations_dir,
        )
    if args.verb == 'reopen-decision':
        return _run_reopen_decision(
            args.id,
            args.root,
            expected_project=args.project,
            expected_escalations_dir=args.escalations_dir,
        )

    try:
        if args.verb == 'write-decision':
            _run_write_decision(
                args.id,
                args.project,
                args.text,
                args.task_id,
                args.escalation_id,
                args.session_id,
                args.severity,
                args.escalations_dir,
                record_slug=filing_record_slug(args.id, args.project),
            )
        elif args.verb == 'reap-decisions':
            _run_reap_decisions(
                args.project, args.escalations_dir, expect_matches=args.expect_matches
            )
        elif args.verb == 'migrate-decision-projects':
            _run_migrate_decision_projects(args.dry_run)
    except Exception:
        logger.error('session_registry %s failed', args.verb, exc_info=True)
    return 0
