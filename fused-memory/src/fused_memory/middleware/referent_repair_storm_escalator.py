"""File a repair-storm escalation when the referent repair pass keeps firing.

The INV-4 storm escape for the write-time referent REPAIR sub-pass (task 3672,
PRD leaf eta of ``plans/memory-referent-fidelity-prd.md``).  This is the FIRE
half of the alarm; the counter half is
``MemoryService._referent_repair_streaks``, which counts CONSECUTIVE episodes
whose repair pass moved at least one edge endpoint and resets on any pass that
looked and found nothing to move.

WHY A SUSTAINED STREAK IS ANOMALOUS BY CONSTRUCTION.  The PRD's measured base
rate for a conflated edge endpoint is ~0.22% of live task-mentioning edges —
a couple of edges in a thousand, arriving at random.  Ten consecutive episodes
each needing a repair is not that distribution: it means the SCANNER
(``utils/canonical_labels``) has started matching text it should not, or the
RESOLVER (``utils/referent_resolution``) has started producing the wrong
target, and every episode is now arriving mis-attributed.  A repair pass that
silently absorbed that would present a regression as steady-state health —
the graph would keep being corrected, correctly, forever, while nobody learned
that the thing producing the errors had broken.  That is exactly the
silent-degradation shape the no-silent-fail-soft invariant rules out.

THE ESCALATION IS THE ALARM, NOT A HALT.  Repairs CONTINUE while it fires:
this function is never consulted before a write, never returns a
keep-going/stop verdict, and is not a rate limiter.  A repair is the correct
action on a conflated endpoint whether or not the rate is anomalous, and
halting on the alarm would leave a growing pile of known-wrong edges in the
graph to trade a legible bug for an illegible one.  ``suggested_action`` says
so explicitly, because an operator who assumes writes are parked will
mis-triage the urgency in both directions.

WHY THE MODULE-FUNCTION SHAPE.  fused-memory has two live escalator shapes.
:mod:`fused_memory.middleware.mem0_update_storm_escalator` is a CLASS holding
per-agent counters, because its operator question is "which agent is looping"
and folding two agents' storms together would destroy the attribution that
makes it actionable.  Here attribution is per-PROJECT: the streak is keyed by
``group_id`` (== ``project_id``), the escalation queue is the project's own
``data/escalations``, and a project's storm has exactly one cause at a time.
Folding every breach in a project into one entry is therefore not a loss of
fidelity, it is the correct grain — so this copies the newer module-function
shape of :mod:`fused_memory.middleware.candidate_key_escalation` instead.

NEVER RAISES, for that module's stated reason and one more: it is called from
the live memory-write path (via ``asyncio.to_thread`` off
``MemoryService._repair_episode_referents``), where a raise would fail an
already-committed episode's reconcile chain because the COMPLAINT ABOUT the
write failed.  The repairs it is complaining about have already landed by the
time this runs; escalation is purely additive.  That contract is now KEPT IN
ONE PLACE: the defensive optional-package import, the guarded queue, the
dedupe-fold and the never-raise submit all live in
:mod:`fused_memory.middleware._folded_escalation`, which this module calls.
What stays here is this alarm's own identity (``_ANCHOR_TASK_ID`` and friends)
and its own evidence rendering.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

from fused_memory.middleware._folded_escalation import file_folded_escalation

logger = logging.getLogger(__name__)

#: Anchor task_id handed to ``file_folded_escalation``, which threads it through
#: both ``EscalationQueue.make_id`` and the fold lookup — so the resulting ids
#: (``esc-referent-repair-storm-1``, ...) are greppable and distinct from the
#: other fused-memory series — and, more importantly, so
#: ``get_by_task(_ANCHOR_TASK_ID, status='pending')`` is a stable per-project
#: lookup for "is this alarm already open".
#:
#: STAYS A CONSTANT OF THIS MODULE even though the filer body is now shared: a
#: filer deduping against an anchor somebody else keeps open never files again,
#: and that silence is indistinguishable from health.  The pairwise regression
#: that catches a colliding rename
#: (``tests/test_folded_escalation.py::TestNoTwoFilersShareAnAnchor``) reads
#: every filer's anchor FROM ITS OWN HOME, which only works while this lives
#: here.
_ANCHOR_TASK_ID: str = 'referent-repair-storm'

_AGENT_ROLE: str = 'fused-memory/referent-repair-guard'
_CATEGORY: str = 'referent_repair_storm'

#: How many structured repair records the detail carries. The evidence is the
#: point (INV-2), but an escalation detail is read by a human, and a pathological
#: episode could carry hundreds; the truncation is REPORTED rather than silent
#: so the reader never mistakes a window for the whole.
_MAX_RECORDS_IN_DETAIL: int = 20


def emit_referent_repair_storm_escalation(
    project_root: str,
    *,
    project_id: str,
    streak: int,
    threshold: int,
    repairs: int,
    records: Sequence[dict[str, Any]],
) -> str | None:
    """File a ``referent_repair_storm`` escalation for *project_id*.

    Called from ``MemoryService._repair_episode_referents`` (through
    ``asyncio.to_thread`` — this does blocking filesystem I/O) once a project's
    consecutive-repair streak reaches *threshold*.

    Args:
        project_root: The affected project's root; the escalation lands in that
            project's OWN ``data/escalations`` queue, resolved by the caller
            from ``MemoryService._known_projects``. Never defaulted to the
            server cwd, where no operator watches.
        project_id: The graph/group whose repairs are storming.
        streak: Consecutive episodes that needed a repair, at the moment of the
            breach. Carried rather than recomputed, so the alarm reports the
            reading that actually tripped it.
        threshold: The policy constant that was breached, so the detail is
            self-describing when the constant later changes.
        repairs: Endpoints moved by THIS episode's pass — the width of the
            latest breach, alongside the streak's depth.
        records: ``ReferentRepair.to_dict()`` payloads for the latest pass: the
            INV-2 structured evidence, shipped rather than summarized.

    Returns the escalation id — freshly filed, or the id of the already-open
    escalation for this project when one exists — or ``None`` when neither is
    possible (the ``escalation`` package is unavailable, or the queue write
    failed). NEVER raises.
    """
    shown = list(records[:_MAX_RECORDS_IN_DETAIL])
    omitted = len(records) - len(shown)
    try:
        records_json = json.dumps(shown, indent=2, sort_keys=True, default=str)
    except Exception:  # pragma: no cover — defensive; records are plain dicts
        records_json = repr(shown)

    detail_lines = [
        f'project_id={project_id!r}',
        f'project_root={project_root!r}',
        f'streak={streak} (consecutive episodes whose repair pass moved at '
        f'least one edge endpoint)',
        f'threshold={threshold}',
        f'repairs={repairs} (endpoints moved by the episode that tripped this)',
        '',
        'The write-time referent repair pass (PRD leaf eta) has repaired an '
        'edge endpoint in every one of the last '
        f'{streak} episodes for this project. The measured base rate for a '
        'conflated endpoint is ~0.22% of live task-mentioning edges, so a '
        'streak this long is not that distribution: it means the SCANNER or '
        'the RESOLVER has regressed and episodes are now arriving '
        'mis-attributed. The repairs themselves are correct and have already '
        'landed; what needs a human is the thing producing the errors.',
        '',
        'REPAIRS WERE NOT HALTED and continue while this is open — this alarm '
        'is not a rate limiter. Resolving it does not require stopping writes.',
        '',
        f'Structured repair records from the tripping episode ({len(records)} '
        f'total{f", {omitted} omitted below" if omitted else ""}):',
        records_json,
    ]

    # The filer skeleton — defensive import, guarded queue, the DEDUPE-FOLD on
    # `_ANCHOR_TASK_ID`, and the never-raise submit — lives in
    # `middleware/_folded_escalation`.  `_ANCHOR_TASK_ID` stays a constant of
    # THIS module and is passed in explicitly: the helper has no default for it,
    # because a filer that dedupes against an anchor somebody else keeps open
    # goes permanently silent and that silence reads exactly like health.
    #
    # The fold itself is why the anchor is per-project rather than per-episode:
    # once a project is storming, EVERY subsequent episode breaches the
    # threshold again, and filing per breach would bury the operator queue.
    return file_folded_escalation(
        project_root,
        anchor_task_id=_ANCHOR_TASK_ID,
        agent_role=_AGENT_ROLE,
        category=_CATEGORY,
        severity='blocking',
        summary=(
            f'referent repair storm in {project_id}: {streak} consecutive '
            f'episodes needed an edge-endpoint repair (threshold {threshold})'
        ),
        detail='\n'.join(detail_lines),
        suggested_action=(
            'Audit the referent scanner (fused_memory/utils/canonical_labels.py) '
            'and the resolver (fused_memory/utils/referent_resolution.py) for a '
            'regression that mis-attributes edge endpoints; '
            'MemoryService.referent_repair_counts() reports the live '
            'per-project streaks and the process-lifetime repaired / '
            'flagged_unrepairable / failed totals. Repairs were NOT halted '
            'and continue while this is open.'
        ),
        logger=logger,
        log_label='referent_repair_storm',
        context=(
            f'project_id={project_id!r} streak={streak} threshold={threshold} '
            f'{repairs} repair(s) this episode; repairs continue'
        ),
        level=1,
    )
