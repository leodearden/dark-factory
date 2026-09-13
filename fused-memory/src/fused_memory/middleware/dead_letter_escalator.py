"""Push a ``durable_write_dead_letter`` escalation when a queued write is lost.

Task 3583. The FIRE half of the signal whose seam is
``DurableWriteQueue._notify_dead_letter`` and whose caller is
``MemoryService._report_queue_dead_letter``.

WHAT THIS CLOSES. A durably-queued write is ACCEPTED synchronously and executed
later: ``add_episode`` returns ``status='queued'``, and ``add_memory``'s
Graphiti leg appends ``SourceStore.graphiti`` to ``stores_written`` the moment
the item is enqueued. Both are true statements about durable acceptance. But
when the item later exhausts its attempts, the caller who acted on that success
is told nothing, ever — the whole premise of the queue is that an accepted
enqueue eventually lands, and a dead-letter is that premise being broken.

WHY THIS RECORD IS THE ONLY EVIDENCE THAT SURVIVES. Every pull-side surface
reads the LIVE ``write_queue`` table: ``get_stats``'s ``dead_by_operation``,
``get_queue_stats``, and ``reconciliation/queue_health.py``'s aggregate
``dead_count``. Routine ``delete_dead_letters`` cleanup sweeps those rows and
every one of those counters goes back to zero with them — that sweep had
already erased 26 of the 28 rows by the time anyone looked at the esc-3561-3
investigation, which is how 28 permanently-failed writes across five projects
went unnoticed for three and a half months. An escalation survives the sweep,
which is why this module must not depend on anything conditionally constructed
and must not be reachable only from a reconciliation sweep.

NO THRESHOLD, deliberately, unlike the ``mem0_update`` and ``entity_mint``
storm alarms this otherwise copies. Those count a burst because a single event
is unremarkable; here ONE permanently-lost durable write is already the event
worth paging on. The investigation's writes died at roughly one every few days
across five projects — any storm threshold tuned to suppress noise would have
suppressed all 28. Page volume is bounded by the dedupe FOLD instead: a
sustained failure of one operation in one project with one error class stays
ONE escalation with an incrementing ``dedupe_count``, so a genuine storm cannot
flood the queue. No config knob either, for the same reason the markup
tripwire kept its thresholds as module constants — the only operator gain would
be the ability to switch off the alarm that existed only by accident for three
and a half months.

WHY THE MODULE-FUNCTION SHAPE, matching
:mod:`fused_memory.middleware.entity_mint_storm_escalator` rather than the
older ``Mem0UpdateStormEscalator`` class. This holds no state: ``project_root``
arrives as an explicit argument, resolved by the CALLER from
``MemoryService._known_projects``, so there is no queue cache to own and no
``set_known_projects`` lifecycle to keep in sync. The ``EscalationQueue`` is
built FRESH per call inside its own ``try/except`` because constructing it
CREATES its directory — a read-only root raises at construction, and a cache
would defer that raise to an arbitrary later call.

NEVER RAISES. This runs after the queue has already committed the item's dead
state, from inside ``_process_item``; turning a lost alarm into an exception
would cost the worker as well as the signal. Every failure mode degrades to a
log line plus ``None``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue  # type: ignore[import-untyped]

try:
    from escalation.models import Escalation  # type: ignore[import-untyped]
    from escalation.queue import EscalationQueue  # type: ignore[import-untyped,no-redef]
    HAS_ESCALATION = True
except ImportError:
    HAS_ESCALATION = False

logger = logging.getLogger(__name__)

_QUEUE_DIRNAME = 'data/escalations'

_ANCHOR_TASK_ID = 'durable-write-dead-letter'

_AGENT_ROLE = 'fused-memory/dead-letter-guard'
_CATEGORY = 'durable_write_dead_letter'

_FINDING_CATEGORY = 'queue_dead_letter'


def emit_dead_letter_escalation(
    project_root,
    *,
    project_id,
    operation,
    group_id,
    item_id,
    attempts,
    error,
    post_execute,
    content_preview,
    write_op_id,
):
    """File (or fold into) a ``durable_write_dead_letter`` escalation.

    Called from ``MemoryService._report_queue_dead_letter`` through
    ``asyncio.to_thread`` — ``EscalationQueue.submit`` is a synchronous
    fsync-flushed filesystem write, and this hook runs on the event loop inside
    the durable queue's worker, so calling it directly would stall the pool.

    Args:
        project_root: The affected project's root; the escalation lands in that
            project's OWN ``data/escalations`` queue. Resolved by the caller
            from ``MemoryService._known_projects`` and never defaulted to the
            server cwd, where no operator watches.
        project_id: The project whose write was lost.
        operation: The durable-queue operation that died (``add_episode``,
            ``add_memory_graphiti``, ``mem0_classify_and_add``, ...). The
            operator's first question, and part of the dedupe fold.
        group_id: The queue group the item sat in — the graphiti group for a
            Graphiti write, ``mem0_{project_id}`` for a Mem0 one.
        item_id: The ``write_queue`` row id, for as long as the row survives.
        attempts: The committed attempt count the item died on.
        error: The reported error, still carrying ``POST_EXECUTE_DEAD_PREFIX``
            when applicable; the fold keys on its CLASS only.
        post_execute: True when the backend write LANDED and only the
            post-execute work kept failing, so a blind replay DUPLICATES it.
        content_preview: A prefix of what was being written, for identifying
            the lost content once the queue row is swept.
        write_op_id: The ``write_ops`` join key, or None for an operation that
            carries none (``mem0_classify_and_add``, ``replay_from_store``).

    Returns the escalation id — freshly filed, or the pending parent's id when
    this death folded into it — or ``None`` when nothing was filed (the
    ``escalation`` package is unavailable, or the queue write failed). NEVER
    raises.
    """
    if not HAS_ESCALATION:
        logger.warning(
            'durable_write_dead_letter: escalation package unavailable; a '
            'queued %r write for project_id=%r (group_id=%r, item_id=%s) was '
            'permanently abandoned after %s attempts and will NOT be '
            'escalated. error=%r',
            operation, project_id, group_id, item_id, attempts, error,
        )
        return None

    try:
        queue = EscalationQueue(Path(project_root) / _QUEUE_DIRNAME)
    except Exception:
        # Constructing the queue creates its directory; a read-only or missing
        # project_root must not turn a lost alarm into a crash in the worker.
        logger.exception(
            'durable_write_dead_letter: could not open the escalation queue at '
            'project_root=%r; the abandoned %r write for project_id=%r '
            '(item_id=%s) goes unescalated',
            project_root, operation, project_id, item_id,
        )
        return None

    detail = '\n'.join([
        f'project_id={project_id!r}',
        f'project_root={project_root!r}',
        f'operation={operation!r}',
        f'group_id={group_id!r}',
        f'queue_item_id={item_id}',
        f'attempts={attempts}',
        f'write_op_id={write_op_id!r}',
        f'post_execute={post_execute}',
        f'error={error!r}',
    ])

    try:
        esc = Escalation( # type: ignore[possibly-unbound]
            id=queue.make_id(_ANCHOR_TASK_ID),
            task_id=_ANCHOR_TASK_ID,
            agent_role=_AGENT_ROLE,
            severity='blocking',
            category=_CATEGORY,
            summary=(
                f'durable write permanently lost: {operation} in {project_id} '
                f'died after {attempts} attempt(s)'
            ),
            detail=detail,
            suggested_action=(
                'Establish whether the backend write landed before deciding on '
                'a replay; see the record body.'
            ),
            # BORN AT L1, matching every sibling fused-memory escalator. The
            # L0-routes-to-the-steward rule governs an escalation filed BY A
            # DISPATCHED AGENT about its own task. This one is filed by a
            # background server process under a synthetic anchor that is never
            # dispatched and so never has a steward — an L0 entry would have no
            # consumer at all, and would merely wait out `orphan_l0_timeout_secs`
            # before being promoted to exactly where L1 puts it immediately.
            level=1,
        )
        esc_id = queue.submit(esc)
    except Exception:
        # The item is already committed dead; a queue I/O failure must cost the
        # operator a heads-up, never the worker that was draining the group.
        logger.exception(
            'durable_write_dead_letter: failed to submit the alarm for a lost '
            '%r write in project_id=%r (item_id=%s)',
            operation, project_id, item_id,
        )
        return None

    logger.warning(
        'durable_write_dead_letter: queued %s — a %r write for project_id=%r '
        '(group_id=%r, item_id=%s) was permanently abandoned after %s '
        'attempts. error=%r',
        esc_id, operation, project_id, group_id, item_id, attempts, error,
    )
    return esc_id
