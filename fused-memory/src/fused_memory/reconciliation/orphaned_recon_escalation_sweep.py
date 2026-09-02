"""Terminal-subject recon-escalation reaper — DETECTION only (task 3052).

THE DEFECT.  ``stage1_stall_detector.py::maybe_escalate_stalled_gate_backlog``
files a ``reconciliation_stale_gate_backlog`` L1 for a gate task, but only
while that task is selected by
``stage1_stall_detector.py::extract_stalled_gate_backlog_task_ids``, which
requires ``task['status'] == 'blocked'``.  The moment the subject goes
terminal — ``done``/``cancelled`` — or disappears from the task store, the
record's whole premise ("a human decision is still awaited on this blocked
gate") is false.  It cannot ever re-file, and nothing closes it, so it sits
pending forever.  The sibling category
``reconciliation_stale_human_operator`` (``::maybe_escalate_stalled_tasks``)
has the identical lifecycle and the identical defect.

WHY THIS MODULE ONLY DETECTS.  The A7b escalation-closure contract, stated
verbatim in the comment above ``reconciliation/harness.py::_RECON_DEDUP_CONFIG``:
"The reconciliation harness NEVER calls queue.resolve() ON THE RECON
ESCALATION QUEUE ... The watcher session (port 8103) is the sole closer of
recon escalations."  So this module computes the reap set and emits a Stage-1
flag naming it; the CLOSE is performed by the watcher session, or by an
operator running ``fused-memory/scripts/derive_orphaned_recon_escalations.py
--apply``.  Keeping detection and repair in different actors is the same
discipline ``cli_stage_runner.py`` applies when it denies Stage 3
``repair_memory_citation``.

WHY NOT THE ORCHESTRATOR'S EXISTING SWEEP.  ``orchestrator/harness.py``
already auto-closes a terminal-subject escalation with
``resolution_class='moot-terminal-subject'`` — but it is structurally blind to
these records on two independent grounds.  It reads the orchestrator's OWN
``<project_root>/data/escalations`` queue, not
``config.escalation_queue_dir``; and it returns early on
``getattr(esc, 'level', None) != 2`` BEFORE
``config.escalation_revalidation_allowlist`` is ever consulted, whereas every
reconciliation-filed stale record is born at ``level=1`` (re-verified live
2026-09-02: 124 of 124 pending ``reconciliation_stale_gate_backlog`` records
are ``level=1``).  Widening that allowlist therefore cannot reach them, which
is why the detection lives here instead.

WHY REAPING IS SANCTIONED, NOT A POLICY CHANGE.
``skills/recon-escalation-watcher/SKILL.md``'s playbook row for this category
makes PARK the default — correctly, because resolving a still-``blocked``
subject's record re-arms the filing rule and produces measured churn
(``esc-650-1`` -> ``esc-650-2`` in ~4h).  But the same row already carves out
the exit: "**Resolve only** when the underlying task will genuinely stop
qualifying for re-selection."  A terminal or absent subject is provably that
case — selection requires ``status == 'blocked'`` — so a reap here cannot
churn.  The gap this module closes is that nobody COMPUTED which pending
records fall in that already-sanctioned branch.

Design decisions (captured in plan.json):

- The classifier is compared against EACH RECORD'S OWN project's task store,
  resolved through ``BaseStage.known_projects``.  A record whose
  ``project_id`` cannot be parsed, or that names a project absent from that
  map, is counted ``unresolvable`` and is NEVER called an orphan — classifying
  a foreign record against the querying project's census would tell the sole
  closer to resolve a record whose subject may still be legitimately
  ``blocked``.
- The per-project census is CROSS-TAG-COMPLETE (``list_tags`` then one
  ``get_statuses_fresh`` per tag, merged).  ``get_statuses_fresh`` defaults to
  a single tag — see
  ``backends/task_backend_protocol.py::list_tags`` — so a single untagged read
  would classify a subject living in another tag as ``missing`` and reap a
  possibly-still-``blocked`` record.
- ``get_statuses_fresh`` rather than ``get_statuses``: it opens its own
  short-lived autocommit connection per call and so can never be pinned to a
  stale WAL read-snapshot (task 2388), the same reason
  ``harness.py::_fetch_task_count_census`` chose it.
- Best-effort, and fail-SAFE in ONE direction: an errored read is never
  evidence of terminality.  A false ``terminal`` hands the sole closer a live
  record; a missed detection merely waits for the next cycle.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# The two recon stale families this reaper covers.  Both are pending L1
# records filed per-subject-task by Stage 1 whose premise a terminal subject
# genuinely moots, both write ``project_id:`` as their first detail line, and
# ``skills/recon-escalation-watcher/SKILL.md`` already treats their playbook
# rows identically ("Same aging/park shape as reconciliation_stale_gate_backlog
# above").  ``reconciliation_stale_human_operator`` has ZERO pending records
# today (live census 2026-09-02), so its inclusion is pure future-proofing —
# which is why an emitted flag always names the record's own category, so the
# watcher lands on the right playbook row rather than assuming gate-backlog.
REAPABLE_STALE_CATEGORIES: frozenset[str] = frozenset({
    'reconciliation_stale_gate_backlog',
    'reconciliation_stale_human_operator',
})

# Subject statuses from which a task can never return to ``blocked`` and so
# can never re-qualify for re-selection by
# ``stage1_stall_detector.py::extract_stalled_gate_backlog_task_ids``.
# ``deferred`` is deliberately NOT here: a deferred task can be un-deferred
# back into ``blocked``, so reaping its record would re-arm the filing rule.
TERMINAL_TASK_STATUSES: frozenset[str] = frozenset({'done', 'cancelled'})

# Stage-1 flag identity.  Together with the str-coerced subject ``task_id``
# these form ``flag_dedup.compute_flag_signature``'s key, so the flag earns a
# ``stage1_flag_marker`` recurrence row and honours explicit suppression
# instead of re-emitting unmarked every cycle.
ORPHANED_ESCALATION_FLAG_TYPE = 'orphaned_recon_escalation'

# A member of ``cli_stage_runner.FINDING_ITEM_SCHEMA``'s nine-value category
# enum.  The defect is a disagreement between two DISTINCT stores — the recon
# escalation queue (a JSON queue directory) and the subject project's task
# store — which is exactly what ``cross_store_inconsistency`` names.  The
# curator-gate sweep's ``task_memory_mismatch`` would be wrong here: no memory
# is involved at all, so copying it would mislead a reader grepping by
# category.
ORPHANED_ESCALATION_FLAG_CATEGORY = 'cross_store_inconsistency'

# The detail-block key both producers write.  Compared case-sensitively and
# anchored to the start of a stripped line so a ``project_id`` mention inside
# a free-text ``description:`` line cannot be mistaken for the field.
_PROJECT_ID_DETAIL_KEY = 'project_id:'


def escalation_project_id(esc):
    """Return the subject ``project_id`` parsed out of *esc*'s detail block.

    DELIBERATE INV-2 EXCEPTION.  ``escalation.models.Escalation`` has no
    ``project_id`` field (verified: zero occurrences in
    ``escalation/src/escalation/models.py``), so there is no structured fact
    to read and the value must be recovered from prose.  Adding the field
    would help only FUTURE records; the entire population this reaper exists
    to clear is the records already on disk, which would still need parsing.
    This function is therefore the SINGLE owner of that parse — the in-cycle
    sweep and ``scripts/derive_orphaned_recon_escalations.py`` both call it,
    so the rule cannot drift into two copies that disagree.

    The line is written by both producers as the FIRST entry of their
    ``detail_parts`` list —
    ``stage1_stall_detector.py::maybe_escalate_stalled_gate_backlog`` and
    ``stage1_stall_detector.py::maybe_escalate_stalled_tasks`` — but position
    is not relied upon here: the first line whose stripped form starts with
    ``project_id:`` wins.  Empirical basis for treating the parse as total: 0
    of 124 live pending records fail it, across both observed detail vintages
    (``age_hours_at_filing:`` and the older ``age_hours:`` shape still carried
    by ``esc-5943-1``).

    Splits on the FIRST colon only, so a value that itself contains a colon is
    returned whole rather than silently truncated (a truncated id would miss
    ``known_projects`` and be counted ``unresolvable`` — fail-safe, but an
    avoidable recall loss).

    Returns ``None`` when *esc* has no readable ``detail``, or its detail
    carries no ``project_id:`` line, or the parsed value is empty.  NEVER
    raises: detail is deserialised from JSON on disk, so a malformed record
    must degrade to ``unresolvable`` rather than abort the sweep for every
    other record.

    Pure: no I/O, no side effects.
    """
    detail = getattr(esc, 'detail', None)
    if not isinstance(detail, str):
        return None
    for raw_line in detail.splitlines():
        line = raw_line.strip()
        if not line.startswith(_PROJECT_ID_DETAIL_KEY):
            continue
        value = line[len(_PROJECT_ID_DETAIL_KEY):].strip()
        return value or None
    return None


def select_reapable_escalations(escalations):
    """Return the elements of *escalations* this reaper may consider, in order.

    All three conditions must hold; widening any of them would hand the sole
    closer records it has no sanction to close:

    - ``category in REAPABLE_STALE_CATEGORIES`` — other categories (notably
      ``recon_integrity_issue``) have a different lifecycle in which a
      terminal subject does not moot the record.
    - ``status == 'pending'`` — an already-resolved or dismissed record is
      closed; re-closing it is at best a no-op and at worst re-archives it.
    - ``level == 1`` — the level every reconciliation-filed stale record is
      born at.  An L2 record of this category would have been promoted by a
      human-facing path (``promote_to_l2``) and belongs to the orchestrator's
      own revalidation sweep, so this reaper stays out of it.

    A non-``Escalation`` element (or one missing any of these attributes) is
    skipped rather than raising — one malformed row must not cost the whole
    selection.

    Pure: no I/O, no side effects.  Empty input returns ``[]``.
    """
    selected = []
    for esc in escalations:
        if getattr(esc, 'category', None) not in REAPABLE_STALE_CATEGORIES:
            continue
        if getattr(esc, 'status', None) != 'pending':
            continue
        if getattr(esc, 'level', None) != 1:
            continue
        selected.append(esc)
    return selected


def classify_orphan(esc, statuses):
    """Classify *esc* against *statuses*, its own project's status census.

    Returns one of:

    - ``'terminal'`` — the subject's status is in ``TERMINAL_TASK_STATUSES``.
      The record is moot and provably cannot churn if closed: selection
      requires ``status == 'blocked'``.
    - ``'missing'`` — the subject has no row in the census at all.  The
      caller MUST have built that census cross-tag-complete (see
      ``_project_status_census``), because "absent from a single tag" is not
      "absent from the store", and reaping on the weaker signal could close a
      record whose subject is still ``blocked`` in another tag.
    - ``'live'`` — any other status, including ``blocked`` (the state that
      re-qualifies the subject for re-selection) and ``deferred`` (which can
      return to ``blocked``).  A ``'live'`` record is never flagged.

    The id lookup is ``str``-coerced on BOTH sides: censuses come back
    ``{id_str: status_str}`` but task ids arrive as ints on some paths, and an
    un-coerced lookup would report a ``done`` subject as ``missing`` — the
    reap decision would coincide, but the EVIDENCE handed to the closer would
    be false, which is exactly the failure the sibling sweeps' "not proof"
    discipline exists to prevent.

    *statuses* must be a successfully-read census.  An errored or partial read
    must never reach here: it would render as ``missing`` for every record.
    The async orchestrator enforces that by classifying nothing for a project
    whose census read failed.

    Pure: no I/O, no side effects.
    """
    tid = str(getattr(esc, 'task_id', None))
    by_str = {str(k): v for k, v in statuses.items()}
    if tid not in by_str:
        return 'missing'
    if by_str[tid] in TERMINAL_TASK_STATUSES:
        return 'terminal'
    return 'live'
