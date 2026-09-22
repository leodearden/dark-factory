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

WHAT THE CALLER WAS TOLD is carried explicitly, via ``_REPORTED_TO_CALLER``,
keyed on the durable-queue OPERATION name and gated on the evidence that
operation's claim rests on. The two enqueue sites that report a synchronous
success make DIFFERENT claims — ``add_episode`` returns ``status='queued'``,
``add_memory`` returns ``stores_written`` containing graphiti — and an alarm
that misreported which one was made would not be triageable. Neither is the
operation name alone always evidence that a claim was made at all: see
``_Claim.requires_write_op_id`` for the operation with a second, caller-less
producer. A new enqueue site that reports success synchronously must add an
entry there; the default is deliberately neutral rather than optimistic, so a
forgotten entry, or a producer that made no claim, under-claims instead of
inventing a caller to warn.

NEVER RAISES. This runs after the queue has already committed the item's dead
state, from inside ``_process_item``; turning a lost alarm into an exception
would cost the worker as well as the signal. Every failure mode degrades to a
log line plus ``None``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue  # type: ignore[import-untyped]

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
except ImportError:
    HAS_ESCALATION = False

logger = logging.getLogger(__name__)

_QUEUE_DIRNAME = 'data/escalations'

_ANCHOR_TASK_ID = 'durable-write-dead-letter'

_AGENT_ROLE = 'fused-memory/dead-letter-guard'
_CATEGORY = 'durable_write_dead_letter'

_FINDING_CATEGORY = 'queue_dead_letter'

# Matches the bound `write_journal.py::log_write_op` already applies to the
# content it records, so the two records truncate the same way.
_PREVIEW_CHARS = 200


@dataclass(frozen=True)
class _Claim:
    """What a caller was synchronously told, and the evidence it rests on.

    ``requires_write_op_id`` is here because an operation can have MORE THAN
    ONE producer and only some of them face a caller, so the operation NAME
    alone is not always evidence that anybody was told anything.
    ``add_memory_graphiti`` is the case that forces it: ``add_memory`` mints a
    ``_write_op_id`` for every item it enqueues and returns ``stores_written``
    to a live caller, while ``MemoryService.replay_from_store`` enqueues the
    same operation from a background loop with no caller, no synchronous claim
    and deliberately no ``_write_op_id``. Ungated, a replayed death would name
    an ``add_memory`` caller who never existed — the exact invention the
    neutral default below exists to prevent.
    """

    text: str
    requires_write_op_id: bool = False


# What the CALLER was synchronously told, keyed on the durable-queue OPERATION
# name. This is what makes the record say not merely "a write died" but "a
# caller acted on a success that will never be true" — the difference between
# an alarm an operator can triage and one they cannot.
#
# A new enqueue site that reports success synchronously must add an entry here,
# and must set `requires_write_op_id` when it is not that operation's only
# producer.
_REPORTED_TO_CALLER = {
    'add_episode': _Claim(
        "add_episode returned status='queued' and an episode_id, so the caller "
        'was told the write had been durably accepted and would land'
    ),
    'add_memory_graphiti': _Claim(
        'add_memory returned stores_written containing graphiti at enqueue '
        'time — the caller was told this write LANDED, not that it was queued',
        requires_write_op_id=True,
    ),
}

# Deliberately NEUTRAL rather than optimistic, so a forgotten entry — or a
# producer whose evidence is absent — UNDER-claims. An operator wrongly told a
# caller was lied to would go hunting for a caller to warn and find none; the
# reverse error merely under-reports.
_REPORTED_TO_CALLER_DEFAULT = (
    'no synchronous success was reported to any caller for this operation'
)


def _reported_to_caller(
    operation: str, write_op_id: str | None, caller_reference: str | None,
) -> str:
    """The one-line statement of what the caller was promised."""
    claim = _REPORTED_TO_CALLER.get(operation)
    if claim is None or (claim.requires_write_op_id and write_op_id is None):
        text = _REPORTED_TO_CALLER_DEFAULT
    else:
        text = claim.text
    if caller_reference:
        return f'{text} (id handed to the caller: {caller_reference})'
    return text


def _error_class(error: str | None) -> str:
    """The exception CLASS name out of a queue-reported error string.

    ``_handle_failure`` writes ``f'{type(exc).__name__}: {exc}'``, optionally
    behind ``POST_EXECUTE_DEAD_PREFIX``. That prefix is imported from
    ``services.durable_queue`` rather than restated here: it is one constant
    with one owner, and a local copy would silently stop stripping the day the
    wording changed.

    The import is DEFERRED into this function to break an import cycle, not as
    a style choice. At module scope it reaches ``services/__init__``, which
    eagerly imports ``MemoryService``, which imports this module back — so
    whenever this module is imported FIRST (collecting
    ``tests/middleware/`` alone does exactly that) the service layer finds it
    half-initialized and ``emit_dead_letter_escalation`` undefined. By call
    time both modules are fully loaded.

    Falls back to ``'unknown'`` for ``None`` or an unparseable message, which
    folds those deaths together rather than dropping them.
    """
    if not error:
        return 'unknown'
    from fused_memory.services.durable_queue import POST_EXECUTE_DEAD_PREFIX

    text = error
    if text.startswith(POST_EXECUTE_DEAD_PREFIX):
        text = text[len(POST_EXECUTE_DEAD_PREFIX):]
    head, sep, _ = text.partition(': ')
    if not sep or not head or any(c.isspace() for c in head):
        return 'unknown'
    return head


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
    caller_reference=None,
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
        caller_reference: The id the caller was HANDED and is still holding
            (``add_episode``'s correlation id), so an operator can tie this
            alarm back to the call that was told the write had succeeded.

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

    # EVERY remaining statement is inside this one `try`, so the NEVER RAISES
    # contract is structural rather than an argument about which of these
    # expressions can throw. The fields being rendered are whatever JSON a
    # queue row happened to hold, and none of them is type-checked upstream.
    try:
        # `str()` for exactly that reason: `content_preview` reaches here from
        # `payload.get('content') or payload.get('fact_text')`, so a non-str
        # body must cost the record its legibility, never the alarm itself.
        # The 200-char bound is the one `log_write_op` already applies to
        # `params={'content': content[:200]}` — an escalation queue an operator
        # reads must not grow a full episode body per entry.
        preview = str(content_preview or '')[:_PREVIEW_CHARS]

        if post_execute:
            remediation = [
                'THE BACKEND WRITE LANDED. `dead` means the queue gave up, NOT '
                'that the write never happened: the registered callback runs '
                'AFTER the backend write returned, so what kept failing here was '
                'the post-execute work. Replaying this item DUPLICATES the write.',
                '',
                'Before any replay, confirm what landed: look up the `backend_ops` '
                'row joined on this `write_op_id`. Do NOT join on '
                "`backend_ops.operation` — it is the literal 'add_episode' for "
                'both the add_episode and the add_memory_graphiti path, so it '
                'cannot distinguish them and will match the wrong write.',
                '',
                'Once the landed write is confirmed, `delete_dead_letters` is the '
                'correct disposition for this item; `replay_dead_letters` is not.',
            ]
        else:
            remediation = [
                'The backend write did not land: `_execute_write` itself failed, '
                'so nothing was written and there is nothing to duplicate.',
                '',
                'Once the underlying cause is fixed, `replay_dead_letters` is the '
                'safe remediation and will re-run this write. Use '
                '`delete_dead_letters` only when the item is known unrecoverable '
                '— it destroys the only remaining copy of the content above.',
            ]

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
            f'content_preview={preview!r}',
            'reported_to_caller='
            f'{_reported_to_caller(operation, write_op_id, caller_reference)!r}',
            '',
            f'A durably-queued {operation!r} write for project {project_id!r} '
            f'exhausted its attempts and was PERMANENTLY ABANDONED after '
            f'{attempts} attempt(s). Nothing will retry it.',
            '',
            *remediation,
            '',
            'ONE RECORD PER (project, operation, error class). `dedupe_count` is '
            'how many writes died this way, not how many times one write was '
            'retried — the fields above describe the FIRST death, and the folded '
            'children carry the rest.',
            '',
            'DURABLE RECORD: `write_ops.terminal_status` / '
            '`write_ops.terminal_error` (task 3582) carry this outcome per write, '
            'readable via `WriteJournal.get_write_op` for anyone holding the '
            '`write_op_id` above. The LIVE queue counters — `dead_by_operation` on '
            'get_queue_stats / get_status, and the aggregate `dead_count` in '
            'reconciliation/queue_health.py — corroborate this while the row '
            'exists, and go to zero the moment `delete_dead_letters` sweeps it. '
            'This escalation is what survives that sweep.',
        ])

        esc = Escalation( # type: ignore[possibly-unbound]
            id=queue.make_id(_ANCHOR_TASK_ID),
            task_id=_ANCHOR_TASK_ID,
            agent_role=_AGENT_ROLE,
            severity='blocking',
            category=_CATEGORY,
            summary=(
                f'durable write permanently lost: {operation} in {project_id} '
                f'died after {attempts} attempt(s) — {_error_class(error)}'
            ),
            detail=detail,
            suggested_action=(
                'Establish whether the backend write LANDED before deciding on '
                'a replay: read the post_execute field first, because a blind '
                'replay of a post-execute death duplicates the write. Then fix '
                'the underlying cause and replay, or delete the item if it is '
                'unrecoverable.'
            ),
            # BORN AT L1, matching every sibling fused-memory escalator. The
            # L0-routes-to-the-steward rule governs an escalation filed BY A
            # DISPATCHED AGENT about its own task. This one is filed by a
            # background server process under a synthetic anchor that is never
            # dispatched and so never has a steward — an L0 entry would have no
            # consumer at all, and would merely wait out `orphan_l0_timeout_secs`
            # before being promoted to exactly where L1 puts it immediately.
            level=1,
            # Over (category, finding_category, project, operation, error
            # class) ONLY. Deliberately NOT over item_id, attempts or
            # content_preview: all three change on EVERY death, so including
            # any of them would mint a fresh escalation per death and defeat
            # the very folding this fingerprint exists to provide. The three
            # that are here are exactly the triple an operator needs to tell
            # "the same failure again" from "a new failure mode" — which is
            # also why this is not the per-project anchor scan
            # `referent_repair_storm_escalator` uses, a shape that folds every
            # event in a project into one entry and can attribute none of them.
            dedupe_fingerprint=compute_content_fingerprint(  # type: ignore[possibly-unbound]
                _CATEGORY,
                _FINDING_CATEGORY,
                affected_ids=[
                    f'project:{project_id}',
                    f'operation:{operation}',
                    # The CLASS, never the message: the esc-3561-3 errors were
                    # all `node <uuid> not found` with a different uuid per
                    # write, so a message-keyed fingerprint would have minted
                    # 28 separate escalations.
                    f'error:{_error_class(error)}',
                ],
            ),
        )
        config = DedupeConfig(  # type: ignore[possibly-unbound]
            infra_dedupe_enabled=True,
            # UNBOUNDED window: these deaths arrive days apart (esc-3561-3 ran
            # for three and a half months), so any finite window would page
            # again for what is still the same failure.
            infra_dedupe_window_secs=float('inf'),
            infra_dedupe_categories=(_CATEGORY,),
            key_fn=content_fingerprint_key,  # type: ignore[possibly-unbound]
        )
        esc_id = submit_or_dedupe(queue, esc, config)['id']  # type: ignore[possibly-unbound]
    except Exception:
        # The item is already committed dead; a queue I/O failure must cost the
        # operator a heads-up, never the worker that was draining the group.
        logger.exception(
            'durable_write_dead_letter: failed to build or submit the alarm '
            'for a lost %r write in project_id=%r (item_id=%s)',
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
