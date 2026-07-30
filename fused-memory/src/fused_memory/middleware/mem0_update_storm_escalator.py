"""Route in-place ``update_memory`` content-amend storms to the operator queue.

Task 3088. In-place amendment is a SILENT-REWRITE primitive: it changes what a
record says while preserving its Qdrant point id, so an agent stuck in a rewrite
loop leaves behind no new records, no id churn and no failed calls — nothing a
downstream reader could notice. INV-4 (``storm-escape-required``) therefore
requires an unconditional escape hatch: ``MemoryService`` counts content-amend
calls per ``agent_id`` in a rolling window
(:class:`fused_memory.server.storm_counter.StormCounter`) and, on a burst, calls
this module to make the condition reach a human.

Design mirrors :mod:`fused_memory.middleware.scope_violation_escalator` — the
house pattern for exactly this job:

* **Defensive import** of the optional ``escalation`` workspace package. When it
  is missing (minimal envs, tests without escalation infra) the report is a
  logged no-op and the triggering ``update_memory`` call still succeeds. The
  alarm observes; it never blocks the write.
* **Per-project ``EscalationQueue`` cache** keyed on ``project_root``, landing in
  ``{project_root}/data/escalations`` — the same queue an operator already
  watches for scope-violation and curator escalations.
* **Burst folding via** :func:`escalation.dedupe.submit_or_dedupe` with an
  unbounded window, so a sustained storm from one agent folds into ONE pending
  parent (incrementing ``dedupe_count``) rather than paging once per breach.

WHY A CLASS, not the module-function shape of the newer siblings
:func:`fused_memory.server.markup_tripwire.emit_markup_storm_escalation` and
:func:`fused_memory.middleware.candidate_key_escalation.emit_residual_candidate_key_escalation`:
both of those dedupe by scanning for an already-OPEN escalation for the project,
which folds every storm in a project into one entry and so cannot attribute a
burst to the offending ``agent_id``. Attribution is the whole point here — the
operator's first question is *which agent is looping*. A ``(category, agent_id)``
content fingerprint answers that while still folding the repeat breaches, so this
keeps decision doc §4's ratified class shape and its state (the queue cache).

RESOLVING ``project_id`` → ``project_root``. ``update_memory`` carries only a
``project_id`` and ``MemoryService`` holds no filesystem root, so the
``{project_id: project_root}`` snapshot that ``server/main.py`` already builds
with ``build_known_projects_map`` (and already hands to ``ReconciliationHarness``
and ``TicketJanitor``) is INJECTED here via :meth:`set_known_projects`. That is
pure data with no lifetime coupling to any conditionally-constructed component,
which is what lets this escalator be built unconditionally in
``MemoryService.__init__`` — the alarm must not vanish when
``config.reconciliation.enabled`` is false, since that is exactly the degraded
configuration where an unattended rewrite loop is least likely to be noticed any
other way.

When the root cannot be resolved (setter never called, or the id is absent from
the map) this logs a structured WARN carrying the agent, the count and the
window — so the signal stays recoverable from logs — and submits nothing.
Falling back to ``config.taskmaster.project_root`` is explicitly FORBIDDEN: it
defaults to ``'.'``, so the fallback would file into the server's cwd, where no
operator is watching, and report success while doing it.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue  # type: ignore[import-untyped]

# Mirror the defensive-import pattern from scope_violation_escalator.py:50-61 so
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

# Anchor task_id used by ``EscalationQueue.make_id`` so the resulting escalation
# ids (e.g. ``esc-mem0-update-storm-3``) are greppable and distinct from the
# path-guard's ``task-path-guard`` and the migration's ``candidate-key-migration``
# series.
_ANCHOR_TASK_ID: str = 'mem0-update-storm'

_AGENT_ROLE: str = 'fused-memory/mem0-update-guard'
_CATEGORY: str = 'mem0_in_place_update_storm'

# Finding-category component of the dedupe fingerprint. Decision doc §4 spells
# the fingerprint ``compute_content_fingerprint('mem0_in_place_update_storm',
# agent_id)``; the real signature is ``(escalation_category, finding_category,
# affected_ids, description='')`` with ``affected_ids`` REQUIRED, so the two-arg
# spelling is not a callable form. The identity it names is preserved exactly:
# the fingerprint is over the category and the agent and nothing else, with the
# agent carried in ``affected_ids`` (the field whose values ``compute_content_
# fingerprint`` sorts and hashes) and this constant naming the finding.
_FINDING_CATEGORY: str = 'content_amend_storm'


class Mem0UpdateStormEscalator:
    """File ``mem0_in_place_update_storm`` escalations for update_memory bursts.

    Constructed zero-arg (``Mem0UpdateStormEscalator()``) inside
    ``MemoryService.__init__``; the ``{project_id: project_root}`` map arrives
    afterwards via :meth:`set_known_projects` because the map is built later in
    server startup than the service is.

    Holds one cached :class:`escalation.queue.EscalationQueue` per
    ``project_root``. State is process-local and per-instance, so nothing bleeds
    between servers or between tests.
    """

    def __init__(self, known_projects: Mapping[str, str] | None = None) -> None:
        self._queues: dict[str, EscalationQueue] = {}
        # Snapshot-by-copy: the caller's map must not be able to mutate ours out
        # from under a later resolution.
        self._known_projects: dict[str, str] = dict(known_projects or {})

    def set_known_projects(self, known_projects: Mapping[str, str] | None) -> None:
        """Inject the ``{project_id: project_root}`` registry snapshot.

        Follows the ``MemoryService.set_event_buffer`` / ``set_write_journal``
        injection shape. Passing ``None`` clears the map, restoring the
        unresolvable-root no-op posture rather than leaving a stale registry in
        place.
        """
        self._known_projects = dict(known_projects or {})

    def _queue_for(self, project_root: str) -> EscalationQueue | None:
        """Return the (cached) :class:`EscalationQueue` for *project_root*.

        Returns ``None`` when the escalation package is unavailable so callers
        can skip without conditional plumbing.
        """
        if not HAS_ESCALATION:
            return None
        q = self._queues.get(project_root)
        if q is None:
            q = EscalationQueue(Path(project_root) / _QUEUE_DIRNAME)
            self._queues[project_root] = q
        return q

    def report_storm(
        self,
        *,
        project_id: str,
        agent_id: str,
        count: int,
        threshold: int,
        window_seconds: float,
    ) -> str | None:
        """File (or fold into) a storm escalation for *agent_id*.

        Returns the escalation id — freshly minted, or the existing pending
        parent's id when this breach folded into it — or ``None`` when nothing
        was filed (escalation package absent, project_root unresolvable, queue
        write failed).

        NEVER raises. This runs after a successful ``update_memory`` write: a
        queue failure must not turn a completed write into a tool exception, and
        the storm counter is a monitoring alarm, not a rate limiter — crossing
        the threshold never rejects the write that crossed it.

        Sync, like :meth:`ScopeViolationEscalator.report_rejection`, because
        ``EscalationQueue.submit`` is a synchronous atomic-rename filesystem
        write; keeping it sync lets the async service call it without awaiting.
        """
        project_root = self._known_projects.get(project_id)
        if not project_root:
            # Structured WARN, never a guess at the root — see the module
            # docstring on why config.taskmaster.project_root is forbidden here.
            logger.warning(
                'mem0_update_storm_escalator: cannot resolve project_root for '
                'project_id=%r (%d known project(s)); NOT escalating the '
                'update_memory content-amend storm from agent_id=%r '
                '(count=%d, threshold=%d, window_seconds=%s). '
                'Wire MemoryService.set_known_projects(build_known_projects_map(...)) '
                'at server startup to restore this alarm.',
                project_id, len(self._known_projects), agent_id,
                count, threshold, window_seconds,
            )
            return None

        queue = self._queue_for(project_root)
        if queue is None:
            logger.warning(
                'mem0_update_storm_escalator: escalation package unavailable; '
                'update_memory content-amend storm from agent_id=%r in project '
                '%r (count=%d, threshold=%d, window_seconds=%s) will not be '
                'escalated',
                agent_id, project_id, count, threshold, window_seconds,
            )
            return None

        detail = '\n'.join([
            f'project_id={project_id!r}',
            f'project_root={project_root!r}',
            f'agent_id={agent_id!r}',
            f'count={count}',
            f'threshold={threshold}',
            f'window_seconds={window_seconds}',
            '',
            f'Agent {agent_id!r} issued {count} in-place update_memory '
            f'content amendments within {window_seconds}s, at or above the '
            f'storm threshold of {threshold}.',
            '',
            'In-place amendment preserves the record id, so a runaway rewrite '
            'loop leaves no new records, no id churn and no failed calls — this '
            'alarm is the only signal it produces.  Check whether the agent is '
            'looping over the same record ids; if so, stop it, then inspect the '
            'affected records (the write journal records every amendment with '
            'its memory_id and a truncated before/after).',
            '',
            'The writes were NOT blocked: this counter is a monitoring alarm, '
            'not a rate limiter, so a legitimate large consolidation cycle is '
            'never failed mid-run over its own success count.',
            '',
            'To retune: mem0_update.storm_threshold / '
            'mem0_update.storm_window_seconds (both green-tier hot-reloadable). '
            'To stop amendments outright: mem0_update.enabled=false.',
        ])

        try:
            esc = Escalation(  # type: ignore[possibly-unbound]
                id=queue.make_id(_ANCHOR_TASK_ID),
                task_id=_ANCHOR_TASK_ID,
                agent_role=_AGENT_ROLE,
                severity='blocking',
                category=_CATEGORY,
                summary=(
                    f'update_memory content-amend storm: agent {agent_id!r} '
                    f'amended {count} records in {window_seconds}s '
                    f'(threshold {threshold})'
                ),
                detail=detail,
                suggested_action='investigate_runaway_amendment_loop',
                level=1,
                # Over (category, agent_id) ONLY.  Deliberately NOT over count
                # or window: those change on every breach, so including them
                # would mint a fresh escalation per recount and defeat the very
                # folding this fingerprint exists to provide.
                dedupe_fingerprint=compute_content_fingerprint(  # type: ignore[possibly-unbound]
                    _CATEGORY,
                    _FINDING_CATEGORY,
                    affected_ids=[f'agent:{agent_id}'],
                ),
            )
            config = DedupeConfig(  # type: ignore[possibly-unbound]
                infra_dedupe_enabled=True,
                infra_dedupe_window_secs=float('inf'),
                infra_dedupe_categories=(_CATEGORY,),
                key_fn=content_fingerprint_key,  # type: ignore[possibly-unbound]
            )
            esc_id = submit_or_dedupe(queue, esc, config)['id']  # type: ignore[possibly-unbound]
        except Exception:
            # Queue I/O failure must not propagate — the amendment already
            # landed, and turning a completed write into an exception would be
            # strictly worse than losing the alarm.  Mirrors
            # scope_violation_escalator's tolerance.
            logger.exception(
                'mem0_update_storm_escalator: failed to submit storm escalation '
                'for project %s (agent_id=%r, count=%d)',
                project_id, agent_id, count,
            )
            return None

        logger.warning(
            'mem0_update_storm_escalator: queued %s for project %s '
            '(agent_id=%r, count=%d, threshold=%d, window_seconds=%s)',
            esc_id, project_id, agent_id, count, threshold, window_seconds,
        )
        return esc_id
