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
  hit with no files-level mismatch.  Nothing is blocked: the task IS created
  and stamped with ``metadata.possible_scope_mismatch``.

The distinction is load-bearing rather than cosmetic (task 3119): this
escalation is read by operators and rendered into agent briefings, so an
advisory described in rejection wording reports that a task was rejected when
it in fact exists — and tells the reader to resubmit work that already
landed.  Both outcomes are severity ``info`` — the FLOOR of the
``blocking|info|critical|urgent`` vocabulary in ``escalation.models``, so the
wording is what distinguishes them, not the severity.

Design mirrors :class:`fused_memory.middleware.curator_escalator.CuratorEscalator`:

* Defensive import of the optional ``escalation`` workspace package.  When
  the package is missing (minimal envs, tests without escalation infra),
  ``report_rejection`` becomes a logged no-op — the guard's own outcome is
  unaffected (on the FILES-certain path the error dict is still returned;
  on the advisory path the task is still created and stamped), so
  escalation is purely additive.
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

    Handles two distinct categories:

    * **scope_violation** (:meth:`report_rejection`) — filed for BOTH
      task-creation outcomes the path-scope guard produces (module docstring has
      the taxonomy); the ``advisory=`` keyword selects which one the record
      describes.  Recurring identical events fold into one pending parent
      (content-fingerprint dedup, unbounded window), the two modes folding
      independently.  Disable via the ``scope_violation_dedupe_enabled=False``
      constructor escape hatch to restore legacy one-escalation-per-call
      behavior.
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
        the operator to resubmit to the owner) or a PROSE-only advisory (the
        task WAS created and stamped, so the wording says so and
        ``suggested_action`` is ``no_action_advisory_only``).

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
        queue = self._queue_for(project_root)
        if queue is None:
            logger.debug(
                'scope_violation_escalator: escalation package unavailable; '
                '%s of %r in project %r will not be escalated',
                'advisory' if advisory else 'rejection',
                candidate_title[:80], project_id,
            )
            return None

        paths_str = ', '.join(matched_paths) or '<none>'
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
            # curator picks drop/combine/create/refuse — so any assertion that
            # a task exists would be unverified here and false on a drop or a
            # combine.  The stamp sentence is conditional on the CREATE outcome
            # specifically: _execute_combine merges only curator_* keys onto
            # the target, so a combine target never receives this candidate's
            # possible_scope_mismatch.  Nor can this say "queued for curation"
            # as an absolute — planning_mode bypasses the curator entirely, and
            # that kwarg is not read until well after this guard runs.
            detail_lines.append(
                'A task creation request cited paths that look like they belong '
                'to another project, based on a heuristic scan of its prose '
                '(title/description/details) only.  The submission was NOT '
                'blocked, and no resubmission is needed.  This record is filed '
                'at the submission guard, BEFORE the submission has been '
                'resolved, so it does not establish that a task exists: '
                'resolution happens asynchronously and may create a new task, '
                'combine this candidate into an existing one, or drop it.  '
                'Only a task the curator CREATES carries the match as '
                'metadata.possible_scope_mismatch — a candidate that is '
                'dropped, or combined into an existing task, does not.  '
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

        try:
            esc = Escalation(  # type: ignore[possibly-unbound]
                id=queue.make_id(_ANCHOR_TASK_ID),
                task_id=_ANCHOR_TASK_ID,
                agent_role=_AGENT_ROLE,
                # 'info' for BOTH modes: already the severity floor (module docstring).
                severity='info',
                category=_CATEGORY,
                summary=(
                    f'Path-scope ADVISORY: task CREATED and stamped, '
                    f'cites {paths_str} (possible owner: {target})'
                    if advisory else
                    f'Misrouted task rejected: cites {paths_str} '
                    f'(suggested target: {target})'
                ),
                detail=detail,
                suggested_action=suggested_action,
                level=1,
                dedupe_fingerprint=compute_content_fingerprint(  # type: ignore[possibly-unbound]
                    'scope_violation',
                    'path_guard_misroute',
                    affected_ids=sorted([
                        *matched_paths,
                        f'suggested:{suggested_project or "none"}',
                        f'project:{project_id}',
                        # Advisory-only by design — see _ADVISORY_FINGERPRINT_TOKEN.
                        *([_ADVISORY_FINGERPRINT_TOKEN] if advisory else []),
                    ]),
                ),
            )
            config = DedupeConfig(  # type: ignore[possibly-unbound]
                infra_dedupe_enabled=self._scope_violation_dedupe_enabled,
                infra_dedupe_window_secs=float('inf'),
                infra_dedupe_categories=(_CATEGORY,),
                key_fn=content_fingerprint_key,  # type: ignore[possibly-unbound]
            )
            esc_id = submit_or_dedupe(queue, esc, config)['id']  # type: ignore[possibly-unbound]
        except Exception:
            # Queue I/O failure must not propagate — the guard's own outcome
            # stands either way, the operator just doesn't get the heads-up.
            # Mirrors curator_escalator's tolerance so a broken filesystem
            # can't break task creation.
            logger.exception(
                'scope_violation_escalator: failed to submit escalation '
                'for project %s (candidate=%r)',
                project_id, candidate_title[:80],
            )
            return None

        logger.warning(
            'scope_violation_escalator: queued %s for project %s '
            '(candidate=%r, suggested=%s)',
            esc_id, project_id, candidate_title[:80], target,
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
