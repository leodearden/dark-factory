#!/usr/bin/env python3
"""One-shot backfill: re-anchor LEGACY gate-backlog escalation summaries.

Motivation
----------
Task 3520 replaced the gate-backlog escalation's RELATIVE-age summary
(``'Gate task 166 has awaited a human decision for 48.7h'``) with an absolute
anchor (``'... since <ISO> (past the 48h gate-backlog threshold)'``), because
the relative phrasing goes stale the moment the record is filed: a compact
drain projects ``summary`` and drops ``detail``
(``_COMPACT_ESCALATION_FIELDS``, ``escalation/src/escalation/server.py``), so a
steward triaging a gate that has now been open 400h still reads ``48.7h``.

Records filed BEFORE 3520 keep that stale summary forever.  ``attach_dedupe_child``
deliberately never rewrites ``summary``/``detail``, and task 3522's
``gate_backlog_fingerprint_key`` keeps those legacy records alive as fold
targets precisely so they are NOT superseded — so the stale text is permanent
until something rewrites it in place.  That is what this script does, and it is
a separate, independently reviewable operator action by design (see
``gate_backlog_fingerprint_key``'s "ACCEPTED COST" paragraph and task 3522's
plan.json design_decisions).

Selection
---------
A record is rewritten iff it is a PENDING ``reconciliation_stale_gate_backlog``
whose summary matches the pre-3520 shape (``LEGACY_SUMMARY_RE``).  Selection is
deliberately NOT ``dedupe_fingerprint is None`` and NOT a hardcoded count:
measured on the live queue, of 132 pending gate-backlog records 78 are
unstamped but only 67 carry the old summary.  The other 11 were filed in the
window between 3520 landing and 3522's stamp landing — their summary is ALREADY
correct and rewriting them would be a pointless mutation of live production
records.  Gating on the summary shape also buys idempotence for free: after a
rewrite the summary no longer matches, so a second ``--apply`` is a structural
no-op.

Usage
-----
  # Dry run (default): print a JSON report, touch nothing.
  python scripts/backfill_gate_backlog_summaries.py

  # Override the queue directory (default: ./data/reconciliation/escalations).
  python scripts/backfill_gate_backlog_summaries.py --queue-dir /path/to/queue

  # Commit the rewrites.  OPERATOR ACTION — see below.
  python scripts/backfill_gate_backlog_summaries.py --apply

``--apply`` mutates live production escalation records that are NOT under git,
so it is neither visible in a diff nor revertable by a merge-lane rollback.  It
is deliberately an operator step, not something the implementing branch runs:
review the dry-run report first, confirm ``skipped_fingerprint_drift`` is 0 and
``legacy_total`` matches the population you expect, then re-run with ``--apply``.

Safety properties
-----------------
- Dry-run is the default — no writes occur unless ``--apply`` is passed.
- Idempotent: after a rewrite the summary no longer matches
  ``LEGACY_SUMMARY_RE``, so a second ``--apply`` selects nothing.
- Only ``summary`` and ``detail`` are ever assigned.  ``dedupe_fingerprint``,
  ``dedupe_count``, ``dedupe_children``, ``level``, ``severity``, ``status``,
  ``timestamp`` and ``updated_at`` are all left exactly as found.
- Every read-modify-write happens inside ``escalation_id_lock``, with the record
  re-read and re-checked INSIDE the lock, so a Stage-1 fold landing concurrently
  is never reverted.
- The rewrite is fingerprint-preserving by MACHINE CHECK, not by argument:
  ``plan_rewrites`` fails closed on any record whose
  ``gate_backlog_fingerprint_key`` would change.
"""

from __future__ import annotations

import re

from escalation.models import Escalation

GATE_BACKLOG_CATEGORY = 'reconciliation_stale_gate_backlog'
"""``Escalation.category`` this backfill acts on — the emitter's
``stage1_stall_detector._GATE_BACKLOG_ESCALATION_CATEGORY``."""

# The PRE-3520 summary shape, fully anchored at both ends.  Anchoring is what
# makes selection safe: an unanchored pattern would also match a post-3520
# summary that happened to embed the phrase, and (worse) would re-select a
# record this script had already rewritten.
LEGACY_SUMMARY_RE = re.compile(
    r'^Gate task (?P<task_id>\S+) has awaited a human decision for [0-9.]+h$'
)

# Named couplings to the emitter's ``detail_parts`` in
# ``fused_memory/reconciliation/stage1_stall_detector.py::maybe_escalate_stalled_gate_backlog``
# — kept as module constants so the coupling is named rather than inlined, in
# the style of ``escalation/src/escalation/dedupe.py::_GATE_BACKLOG_DETAIL_PROJECT_PREFIX``.
# A change to those emitted lines must update these.
_GATE_ESCALATED_AT_PREFIX = 'gate_escalated_at: '
_LEGACY_AGE_PREFIX = 'age_hours: '
_CANONICAL_AGE_PREFIX = 'age_hours_at_filing: '


def is_legacy_gate_backlog_record(esc: Escalation) -> bool:
    """True iff *esc* is a PENDING gate-backlog record with a pre-3520 summary.

    The predicate reads the record's own summary shape and nothing else — not
    ``dedupe_fingerprint`` (78 unstamped records, but only 67 stale summaries)
    and not a count.  Because a rewritten record no longer matches, this
    predicate is also the script's idempotence guarantee.
    """
    return (
        esc.category == GATE_BACKLOG_CATEGORY
        and esc.status == 'pending'
        and LEGACY_SUMMARY_RE.match(esc.summary or '') is not None
    )
