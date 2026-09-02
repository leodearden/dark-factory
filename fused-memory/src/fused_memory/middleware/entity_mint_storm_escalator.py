"""File an ``entity_mint_storm`` escalation when MCP node-minting bursts.

The INV-4 storm escape for the ``ensure_entity_node`` MCP tool (task 4932).
This is the FIRE half of the alarm; the counter half is
``MemoryService._entity_mint_storm_counters``, a per-``agent_id``
:class:`shared.storm_counter.StormCounter` recording one event per MINT.

WHY THE MINT BRANCH IS THE ONE THAT NEEDS AN ALARM. Of the tool's four
outcomes, three create nothing: a resolve returns an existing uuid and makes no
backend call at all, and the ambiguous-name and lock-busy refusals write
nothing by construction. Only a mint adds a node — and NOTHING SWEEPS ORPHAN
MINTED NODES. A caller stuck in a mint loop therefore leaves behind a growing
pile of junk identity nodes, each of which is individually well-formed, each of
which passed every gate, and none of which any downstream reader would flag.
That is the silent-degradation shape the no-silent-fail-soft invariant rules
out, so the burst has to reach a human by itself.

WHY A NEW MODULE RATHER THAN A REUSE OF
:class:`fused_memory.middleware.mem0_update_storm_escalator.Mem0UpdateStormEscalator`.
That class hardcodes ``_CATEGORY = 'mem0_in_place_update_storm'`` and a detail
body written entirely about in-place content amendment — the record id that
does not churn, the ``operation='update_memory'`` journal rows to diff, the
``mem0_update.*`` retune knobs. Reusing it would file an operator signal whose
every concrete instruction points at the wrong subsystem: an operator paged for
a mint storm would go read update_memory rows that do not exist and flip a
mem0_update knob that would not stop it. A misleading alarm is worse than a
generic one, so this is its own category with its own body. What IS reused is
the shape — the module docstring conventions, the defensive import, the
unbounded-window dedupe fold — not the copy.

WHY THE MODULE-FUNCTION SHAPE, matching
:mod:`fused_memory.middleware.referent_repair_storm_escalator` and
:mod:`fused_memory.middleware.candidate_key_escalation` rather than the older
class shape. This holds no state: ``project_root`` arrives as an explicit
argument, resolved by the CALLER from ``MemoryService._known_projects``, so
there is no queue cache to own and no ``set_known_projects`` lifecycle to keep
in sync. Attribution is preserved where the class shape earned it — the dedupe
fingerprint carries ``agent_id``, so two agents storming at once produce two
escalations and the operator's first question ("which caller is looping") is
answered by the entry itself.

THE ALARM ESCALATES, IT NEVER BLOCKS. This function is never consulted before a
mint, returns no keep-going/stop verdict, and is not a rate limiter. The mint it
is complaining about has ALREADY LANDED by the time this runs — the counter is
recorded after the identity lock is released. Blocking on the threshold would
fail a legitimate large repair batch mid-run over its own success count, and
would do it precisely when the batch is going well. The detail body says so in
as many words, because an operator who assumes minting is parked will
mis-triage the urgency in both directions.

NEVER RAISES, for that reason and one more: it is dispatched through
``asyncio.to_thread`` off the live MCP request path, where a raise would turn a
completed, already-committed mint into a tool exception because the COMPLAINT
about it failed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue  # type: ignore[import-untyped]

# Defensive import of the optional `escalation` workspace package, mirroring
# every sibling escalator. When it is missing (minimal envs, CI without the
# escalation infra) the alarm degrades to a logged no-op and the mint that
# triggered it still succeeds.
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
except ImportError: # pragma: no cover — exercised only in minimal envs
    HAS_ESCALATION = False

logger = logging.getLogger(__name__)

_QUEUE_DIRNAME: str = 'data/escalations'

# Anchor task_id for ``EscalationQueue.make_id``, so the resulting ids
# (e.g. ``esc-entity-mint-storm-3``) are greppable and distinct from the
# sibling alarms' ``mem0-update-storm`` and ``referent-repair-storm`` series.
_ANCHOR_TASK_ID: str = 'entity-mint-storm'

_AGENT_ROLE: str = 'fused-memory/entity-mint-guard'
_CATEGORY: str = 'entity_mint_storm'

# Finding-category component of the dedupe fingerprint.
_FINDING_CATEGORY: str = 'entity_mint_burst'


def emit_entity_mint_storm_escalation(
    project_root,
    *,
    project_id,
    agent_id,
    count,
    threshold,
    window_seconds,
):
    """File (or fold into) an ``entity_mint_storm`` escalation for *agent_id*.

    Called from ``MemoryService._record_entity_mint`` through
    ``asyncio.to_thread`` — ``EscalationQueue.submit`` is a synchronous
    atomic-rename filesystem write, and doing it on the event loop would stall
    every other in-flight MCP request.

    Args:
        project_root: The affected project's root; the escalation lands in that
            project's OWN ``data/escalations`` queue. Resolved by the caller
            from ``MemoryService._known_projects`` and never defaulted to the
            server cwd, where no operator watches.
        project_id: The graph/group being minted into.
        agent_id: The SELF-REPORTED caller that is storming, or
            ``'<unattributed>'``. Carried in the dedupe fingerprint so two
            agents storming at once produce two escalations rather than one
            entry that names neither.
        count: Mints inside the window at the moment of the breach. Carried
            rather than recomputed, so the alarm reports the reading that
            actually tripped it.
        threshold: The ``entity_mint.storm_threshold`` value that was breached,
            so the detail stays self-describing after an operator retunes it.
        window_seconds: The ``entity_mint.storm_window_seconds`` value in force,
            for the same reason.

    Returns the escalation id — freshly filed, or the pending parent's id when
    this breach folded into it — or ``None`` when nothing was filed (the
    ``escalation`` package is unavailable, or the queue write failed). NEVER
    raises.
    """
    if not HAS_ESCALATION:
        logger.warning(
            'entity_mint_storm: escalation package unavailable; agent_id=%r '
            'minted %d Entity node(s) in project_id=%r within %ss '
            '(threshold %s) and that burst will NOT be escalated. Minting '
            'continues.',
            agent_id, count, project_id, window_seconds, threshold,
        )
        return None

    try:
        queue = EscalationQueue(Path(project_root) / _QUEUE_DIRNAME)
    except Exception:
        # Constructing the queue creates its directory; a read-only or missing
        # project_root must not turn an alarm into a crash on the write path.
        logger.exception(
            'entity_mint_storm: could not open the escalation queue at '
            'project_root=%r; the mint burst from agent_id=%r in project_id=%r '
            '(count=%s) goes unescalated',
            project_root, agent_id, project_id, count,
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
        f'Agent {agent_id!r} MINTED {count} Graphiti Entity node(s) via the '
        f'ensure_entity_node MCP tool within {window_seconds}s, at or above '
        f'the storm threshold of {threshold}.',
        '',
        'Only MINTS are counted here. A resolve (the name already existed) '
        'makes no backend call at all, and the ambiguous-name and lock-busy '
        'refusals write nothing — so this count is exactly the number of new '
        'nodes that entered the identity graph, not tool traffic.',
        '',
        'WHY THAT MATTERS: nothing sweeps orphan minted nodes. Each node here '
        'is individually well-formed and passed every gate, so a mint loop '
        'produces no other signal — this alarm is it. Check whether the caller '
        'is looping over the same referent, and whether the nodes it created '
        'are edgeless (a still-edgeless minted node can be removed with '
        'delete_entity at force=False; once an edge has been reassigned onto '
        'one, reversal is no longer clean).',
        '',
        'EVIDENCE TRAIL: the write journal carries one '
        "`operation='ensure_entity_node'` row per call, whose params hold the "
        'name and summary and whose result_summary distinguishes MINTED from '
        'RESOLVED from each refusal — enough to reconstruct exactly what the '
        'loop created and what it merely looked up.',
        '',
        'THE MINTS WERE NOT BLOCKED and minting continues while this is open: '
        'this counter is a monitoring alarm, not a rate limiter, so a '
        'legitimate large repair batch is never failed mid-run over its own '
        'success count. Resolving this does not require stopping writes.',
        '',
        'To retune: entity_mint.storm_threshold / '
        'entity_mint.storm_window_seconds. To stop minting outright: '
        'entity_mint.enabled=false, which denies EVERY caller on the very next '
        'call. All three are green-tier hot-reloadable via reload_config — no '
        'restart is needed for any of them.',
    ])

    try:
        esc = Escalation( # type: ignore[possibly-unbound]
            id=queue.make_id(_ANCHOR_TASK_ID),
            task_id=_ANCHOR_TASK_ID,
            agent_role=_AGENT_ROLE,
            severity='blocking',
            category=_CATEGORY,
            summary=(
                f'entity mint storm: agent {agent_id!r} minted {count} Entity '
                f'node(s) in {project_id} within {window_seconds}s '
                f'(threshold {threshold})'
            ),
            detail=detail,
            suggested_action=(
                'Investigate the runaway minting loop, then audit the nodes it '
                'created for orphans; entity_mint.enabled=false stops all '
                'minting immediately without a restart. Mints were NOT halted '
                'and continue while this is open.'
            ),
            # BORN AT L1, matching all four sibling fused-memory escalators.
            # The L0-routes-to-the-steward rule governs an escalation filed BY A
            # DISPATCHED AGENT about its own task. This one is filed by a
            # background server process under the synthetic anchor
            # `_ANCHOR_TASK_ID`, which is never dispatched and so never has a
            # steward — an L0 entry here would have no consumer at all, and
            # would merely be delayed by `orphan_l0_timeout_secs` before
            # `HarnessRunner._reap_orphan_l0_escalations` promoted it to exactly
            # where L1 puts it immediately.
            level=1,
            # Over (category, finding_category, agent_id) ONLY. Deliberately NOT
            # over count or window_seconds: both change on every breach, so
            # including them would mint a fresh escalation per recount and
            # defeat the very folding this fingerprint exists to provide.
            dedupe_fingerprint=compute_content_fingerprint( # type: ignore[possibly-unbound]
                _CATEGORY,
                _FINDING_CATEGORY,
                affected_ids=[f'agent:{agent_id}'],
            ),
        )
        config = DedupeConfig( # type: ignore[possibly-unbound]
            infra_dedupe_enabled=True,
            # UNBOUNDED window: a sustained storm folds into ONE pending parent
            # (incrementing dedupe_count) rather than paging once per breach.
            infra_dedupe_window_secs=float('inf'),
            infra_dedupe_categories=(_CATEGORY,),
            key_fn=content_fingerprint_key, # type: ignore[possibly-unbound]
        )
        esc_id = submit_or_dedupe(queue, esc, config)['id'] # type: ignore[possibly-unbound]
    except Exception:
        # The mints this is complaining about have already committed; a queue
        # I/O failure must cost the operator a heads-up, never the write.
        logger.exception(
            'entity_mint_storm: failed to submit the alarm for project_id=%r '
            '(agent_id=%r, count=%s)',
            project_id, agent_id, count,
        )
        return None

    logger.warning(
        'entity_mint_storm: queued %s for project_id=%r — agent_id=%r minted '
        '%s Entity node(s) within %ss (threshold %s). Minting continues.',
        esc_id, project_id, agent_id, count, window_seconds, threshold,
    )
    return esc_id
