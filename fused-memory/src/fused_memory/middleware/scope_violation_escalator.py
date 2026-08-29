"""Route path-scope-guard outcomes to the project's escalation queue.

When :mod:`fused_memory.middleware.path_scope_guard` finds that a candidate
task cites paths owned by another project, whatever happens next is easy for
everyone to miss: an MCP caller may or may not surface it; an LLM agent may
not retry; and the operator never sees that a misroute was attempted.  This
escalator writes a parallel ``scope_violation`` escalation so the operator's
queue surfaces the misroute even when the calling agent never reports it.

Since task 2206 (commit 0204e25fa5) the guard produces TWO outcomes, and
:meth:`ScopeViolationEscalator.report_rejection` files an escalation for both
— the ``advisory=`` keyword selects which one the record describes:

* **FILES-certain rejection** (``advisory=False``) — an exact
  ``metadata.files`` owner mismatch.  Hard reject: NO task is created and the
  structured ``DarkFactoryPathScopeViolation`` error dict is returned to the
  caller.
* **PROSE-only advisory** (``advisory=True``) — a regex-over-prose heuristic
  hit with no files-level mismatch.  Nothing is blocked: the submission
  proceeds, carrying ``metadata.possible_scope_mismatch``.

Since task 3123 a THIRD event is reported, by
:meth:`ScopeViolationEscalator.report_routing_override`:

* **ROUTING OVERRIDE bypass** — the caller supplied a non-blank
  ``routing_override_reason``, so BOTH outcomes above were skipped
  entirely.  Nothing is blocked, the submission proceeds, and the record
  is purely an audit trail.  It exists because a bypass with no
  operator-visible record is indistinguishable from no bypass at all — the
  parameter is caller-supplied over a public MCP surface, validated only as
  "non-blank after stripping", and its only prior trace was a
  ``logger.warning`` (which ``journalctl -p warning`` does NOT match, so it
  was not a usable audit trail either).

The distinction is load-bearing rather than cosmetic (task 3119): this
escalation is read by operators and rendered into agent briefings, so an
advisory described in rejection wording reports a rejection that never
happened — and tells the reader to resubmit work that was never blocked.
The advisory wording may not over-claim in the other direction either
(task 4159): ``report_rejection`` is called from the ``submit_task``
PHASE-1 guard, before ``tm.add_task`` and before the submission has been
resolved at all, so it cannot state that a task exists.  The stamp above
reaches a task only when one is actually CREATED from the submission — a
candidate that is dropped, or folded into an existing task, never carries
it (``_execute_combine`` does not propagate it to a combine target).
The same bound binds the override record: it is written from that same
PHASE-1 seam, so it too reports only that nothing was blocked, never that a
task exists.  All three records are severity ``info`` — the FLOOR of the
``blocking|info|critical|urgent`` vocabulary in ``escalation.models``, so the
wording is what distinguishes them, not the severity.

Design mirrors :class:`fused_memory.middleware.curator_escalator.CuratorEscalator`:

* Defensive import of the optional ``escalation`` workspace package.  When
  the package is missing (minimal envs, tests without escalation infra),
  ``report_rejection`` becomes a logged no-op — the guard's own outcome is
  unaffected (on the FILES-certain path the error dict is still returned;
  on the advisory path the submission still proceeds and still carries the
  stamp), so escalation is purely additive.
* Per-project ``EscalationQueue`` cache keyed by ``project_root``.
* Escalations land in ``{project_root}/data/escalations`` — the *filing*
  project's queue (the place the agent was operating against), regardless
  of outcome.  This matches the existing esc-2240-series scope_violation
  pattern referenced in task 1088.

Burst control: :meth:`report_budget_misconfig` applies a per-project dedup
window (``_BUDGET_MISCONFIG_DEDUP_WINDOW_SECS``) so a sustained per-call
budget exhaustion files ONE escalation per window rather than flooding the
operator queue.  :meth:`report_rejection` folds recurring identical events
(same filing project + matched paths + suggested owner + outcome mode) into a
single on-disk pending parent via ``escalation.dedupe.submit_or_dedupe`` — the
first occurrence still escalates, but re-proposals of the same misroute
(e.g. a daily reconciliation consolidation round) increment the parent's
``dedupe_count`` instead of filing a fresh escalation (task 2946).  The two
outcome modes fold INDEPENDENTLY so a pending record of one can never absorb
the other and report it with the wrong outcome — see
``_ADVISORY_FINGERPRINT_TOKEN`` for how, and why only one mode contributes a
token.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue  # type: ignore[import-untyped]

# Mirror the defensive-import pattern from curator_escalator.py:50-55 so
# the escalator silently no-ops when the escalation package is unavailable
# (minimal CI / unit-test envs, deployments that haven't installed it yet).
try:
    from escalation.dedupe import (  # type: ignore[import-untyped]
        DedupeConfig,
        compute_content_fingerprint,
        content_fingerprint_key,
        submit_or_dedupe,
    )
    from escalation.models import Escalation  # type: ignore[import-untyped]
    from escalation.queue import EscalationQueue  # type: ignore[import-untyped,no-redef]
    HAS_ESCALATION = True
except ImportError:  # pragma: no cover — exercised only in minimal envs
    HAS_ESCALATION = False

logger = logging.getLogger(__name__)


_QUEUE_DIRNAME: str = 'data/escalations'

# Anchor task_id used by ``EscalationQueue.make_id`` so the resulting
# escalation IDs (e.g. ``esc-task-path-guard-37``) are easily greppable.
_ANCHOR_TASK_ID: str = 'task-path-guard'

_AGENT_ROLE: str = 'fused-memory/path-guard'
_CATEGORY: str = 'scope_violation'

# Dedup discriminator putting the outcome mode into the content fingerprint, so
# an advisory and a FILES-certain rejection over the same paths never fold into
# one parent whose wording would then mislabel the other (task 3119).
#
# Appended on the advisory branch ONLY, and the asymmetry is the whole point:
# compute_content_fingerprint sorts and \x1f-joins affected_ids before hashing,
# so a 'mode:rejection' counterpart would change the REJECTION digest too and
# stop any parent that is still pending across this change from folding.  The
# rejection composition is therefore left byte-identical to pre-task-3119, and
# only the newly-introduced mode carries a token.
_ADVISORY_FINGERPRINT_TOKEN: str = 'mode:advisory'

# ``suggested_action`` for the PROSE-only advisory.  Deliberately NOT
# ``resubmit_to_<project>``: ``orchestrator/agents/briefing.py`` renders
# suggested_action verbatim into an agent's briefing, so a resubmit
# instruction on a submission that was never blocked reads as a directive to
# redo work that already landed (task 3119).
_ADVISORY_SUGGESTED_ACTION: str = 'no_action_advisory_only'

# Operator-facing outcome-mode labels for this module's log lines.  Named
# constants rather than inline literals so the unavailable-package debug line
# and the terminal queued warning are guaranteed to emit the SAME token — an
# operator greps one mode across both, and the two lines used to be
# indistinguishable by mode at all (task 4159).
_ADVISORY_MODE_LABEL: str = 'advisory'
_REJECTION_MODE_LABEL: str = 'rejection'

# --- Routing-override audit record (task 3123) -----------------------------
# Distinct anchor from _ANCHOR_TASK_ID so the resulting ids
# (esc-task-path-guard-override-N) answer "how often is the bypass used?" with
# a single grep, without having to filter the rejection/advisory records that
# share the scope_violation category.
_OVERRIDE_ANCHOR_TASK_ID: str = 'task-path-guard-override'

# Nothing was blocked on the override path.  Any ``resubmit_to_<project>``
# string here would be rendered verbatim into an agent briefing (task 3119)
# and read as an order to redo landed work.
#
# The CONSUMER exists, and it is deliberately NOT keyed on this string.  Every
# record in this family carries a SYNTHETIC anchor as ``task_id``, so the
# generic ``scope_violation`` recipe (``update_task`` on metadata.modules, then
# ``resolve_issue(action='resume')``) would target a task that does not exist
# and no-op, leaving the record pending forever.  Task 3465 closed that
# (``skills/escalation-watcher-auto/SKILL.md``, landed on main as ``e560568c4c``
# / ``80337b2161``): an audit-only branch keyed on ``category ==
# 'scope_violation'`` AND (``agent_role == 'fused-memory/path-guard'`` OR an id
# starting with ``esc-task-path-guard``) closes these with ``close_only`` /
# ``resolution_class='benign'`` plus a mandatory one-line operator digest, and a
# drain-step-5 carve-out stops a fresh triage stamp from skipping them.  The
# override record matches BOTH halves of that discriminator — ``_submit`` stamps
# ``_AGENT_ROLE``, and _OVERRIDE_ANCHOR_TASK_ID yields ids reading
# ``esc-task-path-guard-override-N`` — so it is consumed, not stranded.  That
# branch matches on structural fields, never on ``suggested_action``, so this
# string stays prose for a human reader (same rule as _ADVISORY_SUGGESTED_ACTION).
#
# Residual, and a follow-up rather than a defect here: that branch's mode table
# names only ``rejection`` and ``advisory``, both under _ANCHOR_TASK_ID, so an
# override record is closed benign in the ``unrecognised-anchor`` digest bucket
# rather than under a mode of its own.  The census is emitted either way;
# naming the third mode is a SKILL.md change outside this module.
_OVERRIDE_SUGGESTED_ACTION: str = 'review_override_justification'

# Dedup discriminator for the override mode.  TWO composition points, both
# load-bearing:
#
# 1. INDEPENDENCE.  Same asymmetry rule as _ADVISORY_FINGERPRINT_TOKEN:
#    compute_content_fingerprint sorts and \x1f-joins affected_ids before
#    hashing, so adding a token to THIS mode alone leaves the rejection and
#    advisory digests byte-identical and keeps any of their still-pending
#    parents folding across this change.  The three modes therefore cannot
#    fold into each other, so a pending record of one can never absorb another
#    and describe it with the wrong outcome (task 3119, now three-way).
#
# 2. THE REASON IS IN THE FINGERPRINT — unlike report_rejection, which folds on
#    matched paths alone.  There the reason is the GUARD's own verdict, so the
#    paths fully identify the event.  Here the reason is CALLER-SUPPLIED and is
#    the very thing being audited: two overrides over the same paths with
#    different justifications are two different claims an operator may judge
#    differently.  Including it gets both ends right — an automated filer
#    looping on one fixed justification files ONE record whose dedupe_count
#    climbs (the flood shape that would otherwise get this reverted), while a
#    genuinely new justification surfaces as a new record.  The reason is
#    STRIPPED before hashing so the fold is stable across entry points
#    (submit_task strips at task_interceptor.py; direct _path_guard_or_skip
#    callers do not).
#
#    Residual, deliberately accepted: a filer that varies its reason string on
#    every call defeats the fold by design.  That produces a visible burst of
#    distinct records, which is itself the signal — the failure mode being
#    fixed here is silence, not volume.
_OVERRIDE_FINGERPRINT_TOKEN: str = 'mode:routing_override'

# ``summary`` is the one-line field every operator view and agent briefing
# renders, and ``reason`` is unbounded caller-supplied free text arriving over
# a public MCP surface.  Cap it there; ``detail`` keeps the (generously
# bounded — see below) string, and detail is the field the audit actually
# needs.
_OVERRIDE_SUMMARY_REASON_MAX: int = 120

# ``detail`` is ALSO rendered verbatim into agent briefings
# (orchestrator/agents/briefing.py) and lands on disk, so "detail keeps the
# full reason" cannot mean UNBOUNDED — that would let a caller push arbitrary
# free text over a public MCP surface straight into a briefing.  Generous
# enough that no honest justification is ever clipped, finite enough that a
# runaway one cannot bloat a briefing; and when it DOES clip it says so, with
# the original length, rather than silently truncating the very text being
# audited.  The dedup fingerprint keeps hashing the FULL reason, so two claims
# that differ only past the cap still fold apart.
_OVERRIDE_DETAIL_REASON_MAX: int = 4000

# ``matched_paths`` is caller-supplied too — on the FILES-certain side it is
# derived directly from ``metadata.files`` — so the same "summary is a
# ONE-LINE field" rationale that bounds the reason has to bound the path list:
# a submission declaring 500 foreign files must not render a multi-kilobyte
# summary.  Shared by every scope_violation-family record.  ``detail`` always
# carries the full list, and the dedup fingerprint hashes ``matched_paths``
# directly rather than this rendering, so capping it cannot change how
# anything folds.
_SUMMARY_PATHS_MAX: int = 5


def _summary_paths(matched_paths: tuple[str, ...]) -> str:
    """Render *matched_paths* bounded, for a one-line escalation ``summary``.

    Returns ``''`` for an empty tuple so each call site can pick its own
    placeholder wording.  Over-cap lists are elided with an explicit
    ``(+K more)`` marker — the summary says what it dropped rather than
    reading as a complete list.
    """
    if not matched_paths:
        return ''
    head = matched_paths[:_SUMMARY_PATHS_MAX]
    rendered = ', '.join(head)
    remaining = len(matched_paths) - len(head)
    if remaining > 0:
        rendered += f' (+{remaining} more)'
    return rendered

# Budget-misconfig escalation constants — deliberately distinct from the
# scope_violation family so operators can immediately tell these apart.
_BUDGET_MISCONFIG_ANCHOR_TASK_ID: str = 'adjudicator-budget-defect'
_BUDGET_MISCONFIG_AGENT_ROLE: str = 'fused-memory/path-scope-adjudicator'
_BUDGET_MISCONFIG_CATEGORY: str = 'adjudicator_config_defect'

# Burst-suppression dedup window (mirrors curator_escalator._ZOT_DEDUP_WINDOW_SECS).
# A sustained per-call budget exhaustion files ONE escalation per window, not one
# per hit, so the operator queue doesn't flood.  Intentionally short — the condition
# resolves only via a config change, so we want one reminder per window, not silence.
_BUDGET_MISCONFIG_DEDUP_WINDOW_SECS: float = 300.0


class ScopeViolationEscalator:
    """Escalation helper for path-scope guard events.

    Handles three distinct reported events across two categories:

    * **scope_violation** (:meth:`report_rejection`) — filed for BOTH
      task-creation outcomes the path-scope guard produces (module docstring has
      the taxonomy); the ``advisory=`` keyword selects which one the record
      describes.  Recurring identical events fold into one pending parent
      (content-fingerprint dedup, unbounded window), the two modes folding
      independently.  Disable via the ``scope_violation_dedupe_enabled=False``
      constructor escape hatch to restore legacy one-escalation-per-call
      behavior.
    * **scope_violation / routing override** (:meth:`report_routing_override`)
      — filed on the BYPASS path, where a caller-supplied
      ``routing_override_reason`` skipped the guard outright.  NOTHING was
      blocked; the submission proceeded and this is an audit record, filed
      because a bypass that leaves no operator-visible trace is
      indistinguishable from no bypass at all.  Like the other two modes it is
      written from the PHASE-1 seam, so it never claims a task exists.  Uses
      its own anchor task id so the ids are independently greppable, and folds
      independently of the two ``report_rejection`` modes.
    * **adjudicator_config_defect** (:meth:`report_budget_misconfig`) — filed
      when the path-scope adjudicator's LLM call returns ``error_max_budget_usd``
      with ``cost_usd > 0``, indicating the per-call budget is too low for the
      call shape (deterministic config defect, not a transient hang).
      Burst-suppressed to one escalation per project per dedup window.

    The class intentionally hosts both categories: it already owns the defensive
    ``HAS_ESCALATION`` import, the per-project ``EscalationQueue`` cache, and the
    never-raise submit contract that both callers need.
    """

    def __init__(
        self,
        budget_misconfig_dedup_window_secs: float = _BUDGET_MISCONFIG_DEDUP_WINDOW_SECS,
        scope_violation_dedupe_enabled: bool = True,
    ) -> None:
        self._queues: dict[str, EscalationQueue] = {}
        self._budget_misconfig_dedup_window_secs = budget_misconfig_dedup_window_secs
        # project_id → monotonic timestamp of the last submitted budget-misconfig
        # escalation.  Prevents a sustained per-call exhaustion from flooding the
        # operator queue with one entry per adjudicator hit.
        self._budget_misconfig_last_submitted: dict[str, float] = {}
        # Escape hatch (mirrors budget_misconfig_dedup_window_secs above): set
        # False to restore legacy no-fold behavior for report_rejection, e.g.
        # if content-fingerprint dedup ever needs to be disabled in the field.
        self._scope_violation_dedupe_enabled = scope_violation_dedupe_enabled

    def _queue_for(self, project_root: str) -> EscalationQueue | None:
        """Return (cached) :class:`EscalationQueue` for *project_root*.

        Returns ``None`` when the escalation package is unavailable so
        callers can silently skip without conditional plumbing.
        """
        if not HAS_ESCALATION:
            return None
        q = self._queues.get(project_root)
        if q is None:
            q = EscalationQueue(Path(project_root) / _QUEUE_DIRNAME)
            self._queues[project_root] = q
        return q

    def _submit(
        self,
        queue,
        *,
        anchor_task_id: str,
        project_id: str,
        candidate_title: str,
        summary: str,
        detail: str,
        suggested_action: str,
        fingerprint_ids: list[str],
    ) -> str | None:
        """Build and file ONE ``scope_violation``-family record; never raises.

        Owns everything the family's three modes (FILES-certain rejection,
        PROSE advisory, routing-override audit) share and must NOT diverge on:
        the ``Escalation`` envelope (``severity='info'`` — the family's floor,
        module docstring — plus ``agent_role``, ``category`` and ``level``),
        the content fingerprint, the unbounded-window ``DedupeConfig``, and the
        swallow-and-log contract.

        Keeping the dedup policy in ONE place is load-bearing rather than
        tidy: the modes stay disjoint only because each contributes its own
        discriminator token to an OTHERWISE IDENTICAL composition (see
        :data:`_ADVISORY_FINGERPRINT_TOKEN`), and three hand-maintained copies
        of that composition could drift into folding each other's records —
        exactly the mislabelling defect task 3119 fixed.

        Callers pass only what genuinely differs per mode: the anchor, the
        operator-facing wording, the ``suggested_action``, and the
        pre-discriminated fingerprint ids (sorted here, so call sites need not
        remember to).

        ``Escalation`` construction is INSIDE the guarded block deliberately:
        every mode is additive to an outcome that already stands (a rejection
        already rejected, a submission already allowed), so a malformed
        payload must degrade to "no escalation", never to an exception out of
        the guard.

        Returns the escalation id — which on a fold is the EXISTING parent's,
        not a freshly-minted one — or ``None`` if anything failed.
        """
        try:
            esc = Escalation(  # type: ignore[possibly-unbound]
                id=queue.make_id(anchor_task_id),
                task_id=anchor_task_id,
                agent_role=_AGENT_ROLE,
                severity='info',
                category=_CATEGORY,
                summary=summary,
                detail=detail,
                suggested_action=suggested_action,
                level=1,
                dedupe_fingerprint=compute_content_fingerprint(  # type: ignore[possibly-unbound]
                    'scope_violation',
                    'path_guard_misroute',
                    affected_ids=sorted(fingerprint_ids),
                ),
            )
            config = DedupeConfig(  # type: ignore[possibly-unbound]
                infra_dedupe_enabled=self._scope_violation_dedupe_enabled,
                infra_dedupe_window_secs=float('inf'),
                infra_dedupe_categories=(_CATEGORY,),
                key_fn=content_fingerprint_key,  # type: ignore[possibly-unbound]
            )
            return submit_or_dedupe(queue, esc, config)['id']  # type: ignore[possibly-unbound]
        except Exception:
            # Queue I/O failure must not propagate — the guard's own outcome
            # stands either way, the operator just doesn't get the heads-up.
            # Mirrors curator_escalator's tolerance so a broken filesystem
            # can't break task creation.
            logger.exception(
                'scope_violation_escalator: failed to submit escalation '
                'for project %s (anchor=%s, candidate=%r)',
                project_id, anchor_task_id, candidate_title[:80],
            )
            return None

    def report_rejection(
        self,
        *,
        project_root: str,
        project_id: str,
        candidate_title: str,
        matched_paths: tuple[str, ...],
        suggested_project: str | None,
        suggested_root: str | None = None,
        llm_reason: str | None = None,
        advisory: bool = False,
    ) -> str | None:
        """File a ``scope_violation`` escalation for EITHER guard outcome.

        Despite the name (kept for call-site/test-double back-compat), this
        method covers BOTH outcomes described in the module docstring, selected
        by *advisory*: a FILES-certain hard rejection (default — wording tells
        the operator to resubmit to the owner) or a PROSE-only advisory
        (nothing was blocked, so the wording says only that, and
        ``suggested_action`` is ``no_action_advisory_only``).

        The advisory wording deliberately does NOT say a task exists
        (task 4159).  This method is reached from the ``submit_task``
        PHASE-1 path guard, which returns before ``tm.add_task`` is called
        and before the submission has been resolved — so at the moment this
        record is written the outcome is genuinely unknown, and the
        ``metadata.possible_scope_mismatch`` stamp reaches a task only when
        one is actually created from the submission (``_execute_combine``
        does not propagate it to a combine target).  The wording names no
        resolver and no timing either: the ordinary ticket path is resolved
        asynchronously by the curator, but ``planning_mode`` bypasses the
        curator and creates the task synchronously.

        Returns the escalation id when one was filed, ``None`` otherwise
        (escalation package missing, queue write failed, etc.).  Never
        raises — escalation is additive to the guard's own outcome, so a
        queue write failure must not turn a guard rejection into a guard
        exception (nor break creation on the advisory path).

        Routes through :func:`escalation.dedupe.submit_or_dedupe` keyed on a
        content fingerprint over the misroute *shape* (filing project_id +
        sorted matched_paths + suggested_project + the *advisory* mode), with
        an unbounded dedup window.  A recurring identical misroute (e.g. the
        same reconciliation consolidation candidate re-proposed every round)
        therefore folds into the first pending escalation — this method then
        returns the EXISTING parent id, not a freshly-minted one — until a
        human resolves it, at which point a later recurrence re-escalates.
        Advisories and rejections fold INDEPENDENTLY, so a pending record of one
        can never swallow the other and report it with the wrong outcome.

        Sync because :meth:`escalation.queue.EscalationQueue.submit` is a
        synchronous filesystem write (atomic ``rename``); no await needed,
        and keeping this sync lets the existing sync ``_path_guard_or_skip``
        in :class:`fused_memory.middleware.task_interceptor.TaskInterceptor`
        call it without changing the call-site signature.

        Args:
            llm_reason: When set (Stage-2 adjudicator returned reject/uncertain
                or failed), this string is the LLM's stated reason (or a
                fail-safe marker).  Appended to the escalation detail so the
                operator can see why the LLM judged this a genuine/uncertain
                misroute.  None (default) preserves the existing detail format
                for callers that don't use the Stage-2 adjudicator.
            advisory: Select the PROSE-only advisory wording (see above).
                Default ``False`` — the FILES-certain rejection wording,
                byte-identical to the pre-task-3119 output.
        """
        # One label for every operator-facing line this call emits.
        mode = _ADVISORY_MODE_LABEL if advisory else _REJECTION_MODE_LABEL
        queue = self._queue_for(project_root)
        if queue is None:
            logger.debug(
                'scope_violation_escalator: escalation package unavailable; '
                '%s of %r in project %r will not be escalated',
                mode, candidate_title[:80], project_id,
            )
            return None

        # Bounded rendering — matched_paths is caller-supplied and summary is a
        # one-line field (see _summary_paths).  detail below keeps the full list.
        paths_str = _summary_paths(matched_paths) or '<none>'
        target = suggested_project or '<unknown — multiple or no owner>'
        if advisory:
            suggested_action = _ADVISORY_SUGGESTED_ACTION
        else:
            suggested_action = (
                f'resubmit_to_{suggested_project}' if suggested_project else 'manual_route'
            )

        # Labels are outcome-NEUTRAL ('filing_*', not 'rejecting_*'): detail is
        # rendered verbatim into agent briefings, and on the advisory path
        # nothing was rejected (task 3119).  Safe to relabel — detail is not an
        # input to the dedup fingerprint.
        detail_lines = [
            f'candidate_title={candidate_title!r}',
            f'filing_project_id={project_id!r}',
            f'filing_project_root={project_root!r}',
            f'matched_paths={list(matched_paths)}',
            f'suggested_project={suggested_project!r}',
        ]
        if suggested_root:
            detail_lines.append(f'suggested_project_root={suggested_root!r}')
        if llm_reason is not None:
            detail_lines.append(f'llm_adjudicator_reason={llm_reason!r}')
        # Only the CLOSING PROSE differs between the two modes — the
        # structured routing context above is what the operator acts on and is
        # identical either way.
        detail_lines.append('')
        if advisory:
            # Claims ONLY what is established at guard time (task 4159).  This
            # fires from submit_task phase-1 — before tm.add_task, before the
            # submission has been resolved at all — so any assertion that a
            # task exists would be unverified here, and false if the candidate
            # is dropped or folded into an existing task.  The prose states
            # only that epistemic fact: it names no RESOLVER and no timing,
            # because both differ by path.  The ordinary ticket path is
            # resolved asynchronously by the curator, but planning_mode
            # bypasses the curator and creates the task synchronously, and
            # that kwarg is not read until well after this guard runs — so
            # "queued for curation" / "resolved asynchronously" would be the
            # same class of unverified claim in a new direction.  The stamp
            # sentence is conditional on a task actually being CREATED:
            # _execute_combine merges only curator_* keys onto the target, so
            # a combine target never receives this candidate's
            # possible_scope_mismatch.
            detail_lines.append(
                'A task creation request cited paths that look like they belong '
                'to another project, based on a heuristic scan of its prose '
                '(title/description/details) only.  The submission was NOT '
                'blocked, and no resubmission is needed.  This record is filed '
                'at the submission guard, BEFORE the submission has been '
                'resolved, so it does not establish that a task exists: the '
                'submission may result in a new task, be folded into an '
                'existing one, or be dropped.  Only a task newly created from '
                'this submission carries the match as '
                'metadata.possible_scope_mismatch — a candidate that is '
                'dropped, or folded into an existing task, does not.  '
                'suggested_project above is a POSSIBLE owner, not a verdict: '
                'review and reroute ONLY if a task does result and the '
                'attribution above is actually correct.',
            )
        else:
            detail_lines.append(
                'A task creation request was rejected because its text or files '
                'reference paths owned by another project.  See suggested_project '
                'above for the intended target; resubmit there or, if no clear '
                'owner is known, route the task manually.',
            )
        detail = '\n'.join(detail_lines)

        esc_id = self._submit(
            queue,
            anchor_task_id=_ANCHOR_TASK_ID,
            project_id=project_id,
            candidate_title=candidate_title,
            summary=(
                # No creation claim: this is written in submit_task
                # phase-1, before the submission has been resolved
                # (task 4159).  briefing.py renders summary verbatim into
                # an agent briefing, so an unverified claim here misleads
                # exactly as the detail's did.
                f'Path-scope ADVISORY: submission not blocked, outcome '
                f'not yet resolved, cites {paths_str} '
                f'(possible owner: {target})'
                if advisory else
                f'Misrouted task rejected: cites {paths_str} '
                f'(suggested target: {target})'
            ),
            detail=detail,
            suggested_action=suggested_action,
            fingerprint_ids=[
                *matched_paths,
                f'suggested:{suggested_project or "none"}',
                f'project:{project_id}',
                # Advisory-only by design — see _ADVISORY_FINGERPRINT_TOKEN.
                *([_ADVISORY_FINGERPRINT_TOKEN] if advisory else []),
            ],
        )
        if esc_id is None:
            return None

        # Mode is interpolated into the EXISTING single line, after the id:
        # 'scope_violation_escalator: queued <id>' is the greppable anchor
        # shared with the sibling escalators, so neither the prefix nor the
        # one-line-per-submit shape may change (task 4159).
        logger.warning(
            'scope_violation_escalator: queued %s (%s) for project %s '
            '(candidate=%r, suggested=%s)',
            esc_id, mode, project_id, candidate_title[:80], target,
        )
        return esc_id

    def report_routing_override(
        self,
        *,
        project_root: str,
        project_id: str,
        candidate_title: str,
        reason: str,
        matched_paths: tuple[str, ...],
        suggested_project: str | None,
        suggested_root: str | None = None,
    ) -> str | None:
        """File a ``scope_violation`` AUDIT record for a routing-override bypass.

        Unlike :meth:`report_rejection`, this fires on the BYPASS path: the
        caller supplied a non-blank ``routing_override_reason``, so ALL of the
        module taxonomy's outcomes were skipped and NOTHING was blocked.  Like
        the advisory (task 4159) this record is written from the ``submit_task``
        PHASE-1 guard, before the submission has been resolved, so it reports
        the BYPASS and never claims a task exists — the emitted detail says so
        explicitly, and a test pins it.  The record exists so the bypass is
        visible somewhere an operator actually reads — before task 3123 the
        only trace was a ``logger.warning``, which made a bypass operationally
        indistinguishable from no bypass at all.

        *matched_paths* / *suggested_project* describe what the guard WOULD
        have flagged, computed by the caller for reporting only; both may be
        empty, and an override whose verdicts came back clean is deliberately
        still recorded (it is the evidence that the parameter was reached for
        unnecessarily).

        Both caller-supplied free-text inputs are BOUNDED before rendering,
        because ``summary`` and ``detail`` are each rendered verbatim into
        operator views and agent briefings: *reason* is clipped hard in the
        summary (:data:`_OVERRIDE_SUMMARY_REASON_MAX`) and generously in the
        detail (:data:`_OVERRIDE_DETAIL_REASON_MAX`, with an explicit
        truncation marker naming the original length), and *matched_paths* is
        elided in the summary (:func:`_summary_paths`) while the detail keeps
        the full list.  The dedup fingerprint hashes the UNCLIPPED values, so
        no cap can make two distinct claims fold together.

        Returns the escalation id when one was filed, ``None`` otherwise
        (escalation package missing, queue write failed).  Never raises — the
        submission it describes has already been allowed, so a queue failure
        must not convert an allowed submission into an exception.
        """
        queue = self._queue_for(project_root)
        if queue is None:
            logger.debug(
                'scope_violation_escalator: escalation package unavailable; '
                'routing override of %r in project %r will not be escalated',
                candidate_title[:80], project_id,
            )
            return None

        reason = (reason or '').strip()
        # Bounded rendering for the one-line summary; detail keeps the full
        # list (and, up to a generous cap, the full reason).
        paths_str = _summary_paths(matched_paths) or '<nothing>'

        # detail is rendered verbatim into agent briefings too, so "verbatim"
        # is bounded — loudly, with the dropped length named, so the audit
        # never quietly misrepresents the text it is auditing.
        detail_reason = reason
        if len(reason) > _OVERRIDE_DETAIL_REASON_MAX:
            detail_reason = (
                reason[:_OVERRIDE_DETAIL_REASON_MAX]
                + f'…[truncated: reason was {len(reason)} chars]'
            )

        detail_lines = [
            'routing_override=True',
            f'routing_override_reason={detail_reason!r}',
            f'candidate_title={candidate_title!r}',
            f'filing_project_id={project_id!r}',
            f'filing_project_root={project_root!r}',
            f'would_have_matched_paths={list(matched_paths)}',
            f'would_have_suggested_project={suggested_project!r}',
        ]
        if suggested_root:
            detail_lines.append(f'would_have_suggested_project_root={suggested_root!r}')
        detail_lines.append('')
        detail_lines.append(
            'The path-scope guards were DELIBERATELY BYPASSED by a '
            'caller-supplied routing_override_reason on this submission.  '
            'The guards did NOT block it: it proceeded to task creation, and '
            'this record is an AUDIT TRAIL, not a rejection — nothing here '
            'asks for a resubmission.  (Deliberately not a claim that the '
            'task now exists: this record is filed from the guard, which runs '
            'BEFORE creation and never observes its outcome.)  The paths listed '
            'above are what the guard WOULD have flagged had the override been '
            'absent (an empty list means the guard would have allowed the '
            'submission anyway).  The reason above is the CALLER\'S OWN '
            'assertion — it is validated only as non-blank and is never '
            'cross-checked against the registry or the claimed paths.  Review '
            'is needed ONLY if that assertion looks wrong for this submission.',
        )
        detail = '\n'.join(detail_lines)

        summary_reason = reason[:_OVERRIDE_SUMMARY_REASON_MAX]
        if len(reason) > _OVERRIDE_SUMMARY_REASON_MAX:
            summary_reason += '…'

        # severity/level/category/agent_role and the whole dedup policy live in
        # _submit — shared with report_rejection so the three modes cannot
        # drift into folding each other's records.  'info' like the rest of the
        # family: this is an audit trail, not a page, and the legitimate
        # self-referential filings it exists to record must not wake an operator.
        esc_id = self._submit(
            queue,
            anchor_task_id=_OVERRIDE_ANCHOR_TASK_ID,
            project_id=project_id,
            candidate_title=candidate_title,
            summary=(
                f'Path-guard ROUTING OVERRIDE used in {project_id}: '
                f'guards skipped (would have flagged: {paths_str}) '
                f'— reason: {summary_reason!r}'
            ),
            detail=detail,
            suggested_action=_OVERRIDE_SUGGESTED_ACTION,
            fingerprint_ids=[
                *matched_paths,
                f'suggested:{suggested_project or "none"}',
                f'project:{project_id}',
                # Keeps this mode's digest disjoint from the rejection
                # and advisory digests — see _OVERRIDE_FINGERPRINT_TOKEN.
                _OVERRIDE_FINGERPRINT_TOKEN,
                # Unlike report_rejection, the CALLER's reason is part
                # of the fingerprint — see _OVERRIDE_FINGERPRINT_TOKEN
                # for why.  Already stripped above, so the fold is
                # identical whichever entry point supplied it.  The FULL
                # reason, not the detail-truncated one: two distinct claims
                # sharing a 4000-char prefix are still two distinct claims.
                f'reason:{reason}',
            ],
        )
        if esc_id is None:
            return None

        logger.warning(
            'scope_violation_escalator: queued routing-override audit %s for '
            'project %s (candidate=%r, reason=%r, would_have_matched=%s)',
            esc_id, project_id, candidate_title[:80], summary_reason, paths_str,
        )
        return esc_id

    def report_budget_misconfig(
        self,
        *,
        project_root: str,
        project_id: str,
        cost_usd: float,
        turns: int,
        max_budget_usd: float,
        model: str,
    ) -> str | None:
        """File an ``adjudicator_config_defect`` escalation for a budget-misconfig hit.

        Called when the path-scope adjudicator's LLM call returns
        ``error_max_budget_usd`` with ``cost_usd > 0`` — a DETERMINISTIC config
        defect: the per-call budget (``config.path_scope_adjudicator.max_budget_usd``)
        is too low for the call shape.  This is NOT a transient hang.

        Returns the escalation id when one was filed, ``None`` otherwise
        (escalation package missing, within dedup window, queue write failed, etc.).
        Never raises — escalation is purely additive; the adjudicator still returns
        its fail-safe verdict regardless of whether the escalation succeeds.

        Burst-suppressed: a second call for the same *project_id* within
        ``budget_misconfig_dedup_window_secs`` logs and returns ``None`` so that a
        sustained per-call exhaustion files ONE escalation per window, not one per hit.
        """
        # Burst-suppression dedup (mirrors curator_escalator._zot_last_submitted).
        now_mono = time.monotonic()
        last = self._budget_misconfig_last_submitted.get(project_id)
        if last is not None and (now_mono - last) < self._budget_misconfig_dedup_window_secs:
            logger.info(
                'scope_violation_escalator: deduplicating budget-misconfig escalation '
                'for project %s (last submitted %.1fs ago < dedup window %.0fs)',
                project_id,
                now_mono - last,
                self._budget_misconfig_dedup_window_secs,
            )
            return None

        queue = self._queue_for(project_root)
        if queue is None:
            logger.debug(
                'scope_violation_escalator: escalation package unavailable; '
                'budget-misconfig for project %r will not be escalated',
                project_id,
            )
            return None

        detail_lines = [
            f'project_id={project_id!r}',
            f'project_root={project_root!r}',
            f'cost_usd={cost_usd}',
            f'turns={turns}',
            f'max_budget_usd={max_budget_usd}',
            f'model={model!r}',
            '',
            'The path-scope adjudicator LLM call returned error_max_budget_usd with '
            'cost_usd > 0, indicating the per-call budget is too low for the call '
            'shape.  This is a DETERMINISTIC CONFIG DEFECT — not a transient hang.',
            '',
            'FIX: raise config.path_scope_adjudicator.max_budget_usd above '
            f'{max_budget_usd} (current value) so the adjudicator call can complete.',
        ]
        detail = '\n'.join(detail_lines)

        try:
            esc = Escalation(  # type: ignore[possibly-unbound]
                id=queue.make_id(_BUDGET_MISCONFIG_ANCHOR_TASK_ID),
                task_id=_BUDGET_MISCONFIG_ANCHOR_TASK_ID,
                agent_role=_BUDGET_MISCONFIG_AGENT_ROLE,
                severity='blocking',
                category=_BUDGET_MISCONFIG_CATEGORY,
                summary=(
                    f'Adjudicator budget misconfiguration: error_max_budget_usd '
                    f'cost_usd={cost_usd} turns={turns} '
                    f'(max_budget_usd={max_budget_usd} is too low)'
                ),
                detail=detail,
                suggested_action=(
                    f'raise path_scope_adjudicator.max_budget_usd above {max_budget_usd}'
                ),
                level=1,
            )
            esc_id = queue.submit(esc)
        except Exception:
            logger.exception(
                'scope_violation_escalator: failed to submit budget-misconfig escalation '
                'for project %s (cost_usd=%s, max_budget_usd=%s)',
                project_id, cost_usd, max_budget_usd,
            )
            return None

        # Record the submission timestamp AFTER a successful submit only.
        self._budget_misconfig_last_submitted[project_id] = now_mono

        logger.warning(
            'scope_violation_escalator: queued budget-misconfig escalation %s '
            'for project %s (cost_usd=%s, turns=%d, max_budget_usd=%s, model=%r)',
            esc_id, project_id, cost_usd, turns, max_budget_usd, model,
        )
        return esc_id
