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

import copy
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime

from escalation.dedupe import gate_backlog_fingerprint_key
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from fused_memory.reconciliation.stage1_stall_detector import (
    STAGE1_GATE_BACKLOG_STALL_THRESHOLD_SECS,
)

logger = logging.getLogger(__name__)

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


def extract_gate_escalated_at(detail: str) -> str | None:
    """Recover the ``gate_escalated_at`` anchor from a record's *detail*, VERBATIM.

    Returns the remainder of the FIRST line starting with
    ``_GATE_ESCALATED_AT_PREFIX``, with only a trailing ``\\r`` (from a CRLF
    detail) dropped — mirroring ``gate_backlog_fingerprint_key``'s handling of
    the ``project_id: `` remainder, and for the same reason: turning a parse
    ambiguity into a silently DIFFERENT value is the worse failure.

    The ISO parse below is a validity GATE and NEVER a reformatter.  The emitter
    interpolated ``metadata['gate_escalated_at']`` unmodified into both this
    detail line and the same-cycle summary, so the verbatim remainder is
    byte-for-byte the string the post-3520 emitter would have written.
    Re-serialising through ``datetime.isoformat()`` would normalise a ``Z``
    suffix and microsecond precision and produce a summary the emitter never
    emits — breaking the byte-parity the emitter-parity test pins.  The ``Z``
    normalisation below is therefore applied to a THROWAWAY COPY, used only to
    decide validity.

    Fails CLOSED — returns ``None``, never a guess — when the line is absent or
    its value is empty, the literal token ``None`` (the emitter's unguarded
    ``f'gate_escalated_at: {gate_escalated_at}'`` renders a missing stamp that
    way), or unparseable as ISO-8601.  A ``None`` here is not an error: it
    selects ``rebuild_summary``'s threshold-only fallback branch, exactly as the
    emitter's own ``age_hours is not None`` condition does.

    Matching is on a LINE-START prefix, not a substring search: a record's
    multi-line ``description:`` block can itself quote a ``gate_escalated_at: ``
    line, and a blob-wide search would let that decoy displace the real one.
    """
    for line in (detail or '').split('\n'):
        if not line.startswith(_GATE_ESCALATED_AT_PREFIX):
            continue
        value = line[len(_GATE_ESCALATED_AT_PREFIX):].rstrip('\r')
        if not value or value == 'None':
            return None
        try:
            # Throwaway parse for validity only — the RETURNED value is the
            # untouched verbatim string above.
            datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError:
            return None
        return value
    return None


def rebuild_summary(
    task_id: str,
    gate_escalated_at: str | None,
    threshold_secs: float = STAGE1_GATE_BACKLOG_STALL_THRESHOLD_SECS,
) -> str:
    """Render the post-3520 summary for *task_id*, exactly as the emitter would.

    Source of truth:
    ``fused_memory/reconciliation/stage1_stall_detector.py::maybe_escalate_stalled_gate_backlog``.
    The two branches below mirror its two summary branches, and the branch
    condition mirrors its ``age_hours is not None`` guard: that guard holds only
    when the stamp parsed, so a record whose anchor cannot be recovered falls to
    the threshold-only phrasing rather than having one guessed for it.

    *threshold_secs* defaults to the emitter's own module constant rather than a
    hardcoded 48 so the two cannot drift.  A comment cannot enforce that, though
    — what actually keeps this in sync is the emitter-parity test, which mints a
    record by CALLING the emitter and asserts this function reproduces its
    summary byte-for-byte.

    *gate_escalated_at* is interpolated VERBATIM (see
    ``extract_gate_escalated_at``): it is already the exact string the emitter
    would have written.
    """
    if gate_escalated_at is not None:
        return (
            f'Gate task {task_id} has awaited a human decision since '
            f'{gate_escalated_at} (past the {threshold_secs / 3600:.0f}h '
            f'gate-backlog threshold)'
        )
    return (
        f'Gate task {task_id} has awaited a human decision beyond the '
        f'{threshold_secs / 3600:.0f}h gate-backlog threshold'
    )


def rebuild_detail(detail: str) -> str:
    """Rename the legacy ``age_hours: `` key to ``age_hours_at_filing: ``.

    Renames the FIRST such line's KEY only.  The value remainder, the line's
    position, the line count, and EVERY other line are byte-identical — line 0
    above all, because ``escalation/src/escalation/dedupe.py::gate_backlog_fingerprint_key``
    recovers a legacy record's entire fold identity from
    ``detail.split('\\n', 1)[0]``.  Disturbing that line would make the record a
    permanently non-folding parent that mints a duplicate every Stage-1 cycle.
    (That is not left to inspection: ``plan_rewrites`` machine-checks the
    fingerprint across the rewrite and fails closed.)

    The value is renamed rather than recomputed: post-3520 the emitter writes
    ``age_hours_at_filing``, a FILING-TIME forensic value that goes stale on
    purpose, so the number this record already carries is the correct one — the
    old key merely misnamed it as if it were current.

    Matching is a line-start prefix test with an explicit
    ``_CANONICAL_AGE_PREFIX`` exclusion.  ``'age_hours_at_filing: 1.0'`` also
    starts with ``'age_hours'``, so without the exclusion an already-normalised
    line would be re-prefixed into ``age_hours_at_filing_at_filing:``; with it,
    the already-normalised case is a true no-op and the whole function is
    idempotent.  Returns *detail* unchanged when no legacy line exists.
    """
    lines = (detail or '').split('\n')
    for i, line in enumerate(lines):
        if line.startswith(_CANONICAL_AGE_PREFIX) or not line.startswith(_LEGACY_AGE_PREFIX):
            continue
        lines[i] = _CANONICAL_AGE_PREFIX + line[len(_LEGACY_AGE_PREFIX):]
        return '\n'.join(lines)
    return detail


@dataclass
class Rewrite:
    """One planned summary/detail rewrite, with both the before and the after."""

    escalation_id: str
    task_id: str
    old_summary: str
    new_summary: str
    old_detail: str
    new_detail: str
    anchored: bool  # False = the anchor was unrecoverable; fallback summary used


@dataclass
class BackfillPlan:
    """The full set of planned rewrites plus the counters the report projects."""

    rewrites: list[Rewrite] = field(default_factory=list)
    pending_total: int = 0
    legacy_total: int = 0
    anchored: int = 0
    fallback: int = 0
    skipped_fingerprint_drift: int = 0


def plan_rewrites(pending: list[Escalation]) -> BackfillPlan:
    """Decide what to rewrite.  PURE — no I/O, no mutation of *pending*.

    Purity is what makes the dry-run path total BY CONSTRUCTION rather than by a
    flag check: ``run()`` can build and report a plan without ``apply_rewrites``
    ever being reachable, so there is no branch in which a dry run could write.

    Every emitted rewrite is FINGERPRINT-PRESERVING BY MACHINE CHECK.  Before
    emitting, the record's ``gate_backlog_fingerprint_key`` is recomputed on a
    copy carrying the new detail and compared with the original's; any
    difference — or a ``None`` on either side — logs a WARNING, skips the record
    and increments ``skipped_fingerprint_drift``.

    Why a check and not a comment: renaming a key on a LATER line cannot disturb
    ``detail``'s line 0, which is that key's only recovery site — but "cannot" is
    an argument, and a future edit to ``rebuild_detail`` (or a record with an
    unexpected line ordering) could break it silently.  The failure mode is a
    permanently non-folding parent that mints a duplicate every Stage-1 cycle,
    so the guard FAILS CLOSED: a skipped record simply keeps its stale summary,
    which is strictly recoverable.  This mirrors the surrounding code's
    discipline — ``gate_backlog_fingerprint_key`` returns ``None`` rather than
    guessing an identity, and ``stage1_stall_detector`` raises on an empty
    fingerprint rather than filing anyway.
    """
    plan = BackfillPlan(pending_total=len(pending))
    for esc in pending:
        if not is_legacy_gate_backlog_record(esc):
            continue
        plan.legacy_total += 1

        anchor = extract_gate_escalated_at(esc.detail)
        new_summary = rebuild_summary(esc.task_id, anchor)
        new_detail = rebuild_detail(esc.detail)

        # Shallow copy carrying only the new detail: the key is derived from
        # (dedupe_fingerprint, category, task_id, detail line 0), so this is the
        # exact record-shape the fold path will see after the rewrite lands.
        after = copy.copy(esc)
        after.detail = new_detail
        key_before = gate_backlog_fingerprint_key(esc)
        key_after = gate_backlog_fingerprint_key(after)
        if key_before is None or key_after is None or key_before != key_after:
            logger.warning(
                'Skipping %s: gate_backlog_fingerprint_key would change across the '
                'rewrite (before=%r after=%r) — refusing to turn it into a '
                'non-folding parent',
                esc.id, key_before, key_after,
            )
            plan.skipped_fingerprint_drift += 1
            continue

        if anchor is not None:
            plan.anchored += 1
        else:
            plan.fallback += 1
        plan.rewrites.append(
            Rewrite(
                escalation_id=esc.id,
                task_id=esc.task_id,
                old_summary=esc.summary,
                new_summary=new_summary,
                old_detail=esc.detail,
                new_detail=new_detail,
                anchored=anchor is not None,
            )
        )
    return plan


def apply_rewrites(queue: EscalationQueue, plan: BackfillPlan) -> dict:
    """Persist *plan*'s rewrites.  Assigns ``summary`` and ``detail``, nothing else.

    The record is read from the queue ROOT only — never the archive — mirroring
    ``attach_dedupe_child``'s ``if not path.exists(): return None``, so a record
    a steward resolved between planning and applying is skipped with no mutation
    and counted under ``skipped_missing``.

    NO OTHER FIELD IS ASSIGNED, and ``updated_at`` is deliberately NOT bumped.
    ``updated_at`` is the "last-substantive-change marker"
    (``escalation/src/escalation/models.py``) that the watcher's stamp-then-skip
    protocol reads via its ``updated_at > triaged_at`` re-verify rule.
    ``attach_dedupe_child`` bumps it because a fold genuinely changes the
    record's substance; this backfill changes only the RENDERING of a fact the
    record already asserted — the gate, and the instant it was escalated, are
    identical before and after.  Bumping it across the whole backlog in one pass
    would mass-invalidate existing triage stamps and force a full re-drain.

    Returns a counters dict.
    """
    rewritten = 0
    skipped_missing = 0
    for rewrite in plan.rewrites:
        path = queue.queue_dir / f'{rewrite.escalation_id}.json'
        if not path.exists():
            logger.info(
                'Skipping %s: no longer in the queue root (resolved/archived '
                'between planning and applying)', rewrite.escalation_id,
            )
            skipped_missing += 1
            continue
        esc = Escalation.from_json(path.read_text())
        esc.summary = rewrite.new_summary
        esc.detail = rewrite.new_detail
        queue._rewrite(rewrite.escalation_id, esc)
        rewritten += 1
    return {'rewritten': rewritten, 'skipped_missing': skipped_missing}
