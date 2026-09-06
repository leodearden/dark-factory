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
whose summary matches the pre-3520 shape (``LEGACY_SUMMARY_RE``).

Selection is deliberately NOT ``dedupe_fingerprint is None`` and NOT a hardcoded
count.  MEASURED on the live queue (2026-09-06), of 132 pending gate-backlog
records:

===  ==========================================  =========================
  n  shape                                       disposition
===  ==========================================  =========================
 67  pre-3520 summary, ``age_hours:``, unstamped  REWRITTEN by this script
 11  3520-anchored, ``age_hours_at_filing:``,     out of scope
     unstamped
 54  3520-anchored, stamped                       out of scope
===  ==========================================  =========================

67 + 11 = 78 is the unstamped population — which is why "unstamped" is the wrong
gate.  Those 11 were filed in the window between 3520 landing and 3522's stamp
landing, so their summary is ALREADY correct and rewriting them would be a
pointless mutation of live production records.  A hardcoded 67 would rot too: it
is a snapshot that can only shrink as stewards drain the queue, not an invariant.

Gating on the summary shape also buys idempotence for free: after a rewrite the
summary no longer matches, so a second ``--apply`` is a structural no-op.

Usage
-----
  # Dry run (default): print a JSON report, touch nothing.
  python scripts/backfill_gate_backlog_summaries.py

  # Override the queue directory (default: ./data/reconciliation/escalations).
  python scripts/backfill_gate_backlog_summaries.py --queue-dir /path/to/queue

  # Commit the rewrites.  OPERATOR ACTION — see below.
  python scripts/backfill_gate_backlog_summaries.py --apply

Operator procedure for ``--apply``
----------------------------------
``--apply`` mutates live production escalation records that are NOT under git,
so it is neither visible in a diff nor revertable by a merge-lane rollback.  It
is deliberately an operator step, and was NOT run by the branch that added this
script (task 4314).

1. Dry-run first and read the report::

     python scripts/backfill_gate_backlog_summaries.py \
         --queue-dir /home/leo/src/dark-factory/data/reconciliation/escalations

2. Confirm ``skipped_fingerprint_drift`` is **0**.  Any non-zero value means a
   record's fold identity would move across the rewrite; investigate that record
   before proceeding rather than applying the rest.  This is machine-signalled
   as well as reported: the process exits **3** when either drift counter is
   non-zero, so a wrapper or ``&&`` chain cannot read a drifting run as success.
3. Confirm ``legacy_total`` is in the range you expect.  It shrinks over time as
   stewards drain the backlog; it should never GROW, because nothing files the
   pre-3520 shape any more.  A growing count means the emitter regressed.
4. Re-run the identical command with ``--apply`` appended.  Check
   ``skipped_error`` is 0; a non-zero count names unreadable records in the log
   and is safe to resolve by fixing them and re-running (the pass is idempotent).
5. Re-run the dry run once more.  ``legacy_total`` must now be 0 — the
   idempotence check.

Exit codes: ``0`` success, ``1`` unusable ``--queue-dir``, ``2`` argparse usage
error, ``3`` at least one record skipped for fingerprint drift.

The rewrite is safe to run while reconciliation is live: every record is
mutated under its own ``escalation_id_lock``, so a concurrent Stage-1 fold is
never reverted.  There is no need to halt the scheduler or drain the queue.

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
- The rewrite is fingerprint-preserving by MACHINE CHECK, not by argument, on
  BOTH sides: ``plan_rewrites`` checks the planned detail and ``apply_rewrites``
  re-checks the bytes it is actually about to write (recomputed inside the lock
  from the fresh record).  Either failing closed leaves the record with its
  stale summary, so a record can never be turned into a non-folding parent that
  mints a duplicate every cycle.
- The dry-run path is read-only BY CONSTRUCTION, not by a flag check inside the
  writer: ``plan_rewrites`` is pure, and ``run()`` simply does not reach
  ``apply_rewrites`` unless ``apply`` is True.  ``run()`` also refuses a
  ``--queue-dir`` that does not already exist, so a dry run against a typo'd
  path errors out instead of silently CREATING it (which
  ``EscalationQueue.__init__`` would otherwise do) and reporting a 0-of-0
  backfill that reads as complete.
- One unreadable record costs one record, not the run: every per-record apply is
  fault-isolated and counted under ``skipped_error``, so the operator keeps the
  report for everything that did land.
- Records outside the selection — the 11 already-anchored unstamped and the 54
  stamped ones above, and every non-gate-backlog category — are byte-identical
  across an ``--apply`` run.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from escalation.dedupe import gate_backlog_fingerprint_key
from escalation.models import Escalation
from escalation.queue import EscalationQueue, escalation_id_lock

from fused_memory.reconciliation.stage1_stall_detector import (
    STAGE1_GATE_BACKLOG_STALL_THRESHOLD_SECS,
)

logger = logging.getLogger(__name__)

GATE_BACKLOG_CATEGORY = 'reconciliation_stale_gate_backlog'
"""``Escalation.category`` this backfill acts on.

Deliberately a LITERAL rather than an import of the emitter's
``fused_memory/reconciliation/stage1_stall_detector.py::_GATE_BACKLOG_ESCALATION_CATEGORY``
— unlike ``STAGE1_GATE_BACKLOG_STALL_THRESHOLD_SECS``, which IS imported.  The
asymmetry is deliberate: the threshold is rendered into text this script
WRITES, so it must track the emitter or the rewritten summaries diverge from
new filings; the category is matched against records already ON DISK, whose
category string was frozen at filing time.  Importing it would mean a future
change to the emitter's value silently retargets this backfill AWAY from the
very records it exists to fix, reporting ``legacy_total: 0`` as if the work
were done.

The coupling is therefore pinned by a TEST rather than by an import:
``test_category_constant_matches_the_emitter`` asserts this literal equals
``_GATE_BACKLOG_ESCALATION_CATEGORY``, so a divergence fails the suite loudly
and a human decides which side moved."""

# The PRE-3520 summary shape, fully anchored at both ends.  Anchoring is what
# makes selection safe: an unanchored pattern would also match a post-3520
# summary that happened to embed the phrase, and (worse) would re-select a
# record this script had already rewritten.
#
# The ``task_id`` group is READ, not decorative: is_legacy_gate_backlog_record
# requires it to equal ``esc.task_id`` (see there).
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

    The final clause makes the REGEX and the REWRITE agree by construction.
    ``rebuild_summary`` re-renders the summary around ``esc.task_id``, while the
    text being replaced names the id captured here; the emitter always writes
    the same value into both (``Escalation(task_id=task_id, ...)`` interpolating
    the same ``task_id`` into its summary), and all 67 live legacy records were
    measured to agree.  A record where they DISAGREE is therefore not something
    this emitter produced, and rewriting it would silently change which gate the
    text refers to — so it fails closed and keeps its stale summary, which a
    human can still read correctly.
    """
    if esc.category != GATE_BACKLOG_CATEGORY or esc.status != 'pending':
        return False
    match = LEGACY_SUMMARY_RE.match(esc.summary or '')
    if match is None:
        return False
    return match.group('task_id') == esc.task_id


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


def fingerprint_preserved(esc: Escalation, new_detail: str) -> bool:
    """True iff rewriting *esc*'s detail to *new_detail* keeps its fold identity.

    ``escalation/src/escalation/dedupe.py::gate_backlog_fingerprint_key``
    recovers a legacy record's entire fold identity from
    ``detail.split('\\n', 1)[0]`` plus ``task_id``/``category``, so a rewrite
    that disturbed line 0 would turn the record into a permanently non-folding
    parent that mints a duplicate every Stage-1 cycle.

    Fails CLOSED — ``False`` on any difference, and also on a ``None`` key on
    either side, since an unrecoverable identity is exactly the state a fold
    cannot survive.  A ``False`` costs one record its rewrite (it keeps its
    stale summary, a strictly recoverable outcome); a missing check costs a
    duplicate escalation every cycle forever.

    Called from BOTH sides on purpose: ``plan_rewrites`` checks the PLANNED
    detail, and ``apply_rewrites`` re-checks the bytes it is ACTUALLY about to
    write — which are recomputed inside the lock from a possibly-newer record,
    so the planned check does not cover them.
    """
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
        return False
    return True


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

        if not fingerprint_preserved(esc, new_detail):
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

    Every per-record body is independently fault-isolated: ONE unreadable or
    corrupt record costs that record, not the run.  ``queue.get_pending()``
    already tolerates such records on the planning side (via
    ``read_escalation_for_scan``); without the same tolerance here, a record
    that became corrupt between planning and applying would abort the pass and
    the operator would lose the report for everything already rewritten.

    Returns a counters dict.
    """
    rewritten = 0
    skipped_missing = 0
    skipped_no_longer_legacy = 0
    skipped_fingerprint_drift_at_apply = 0
    skipped_error = 0
    for rewrite in plan.rewrites:
        try:
            # Lock-scoped read-modify-write.  These records are LIVE fold targets:
            # every Stage-1 cycle folds repeats into them via attach_dedupe_child,
            # which mutates dedupe_count/dedupe_children/severity/updated_at.  A
            # lock-free read-then-write would silently REVERT a fold that landed in
            # between — i.e. it would touch exactly the fields this backfill is
            # forbidden to touch.  escalation/src/escalation/sweep.py is the
            # in-repo precedent for an external module importing the exported lock
            # and doing raw lock-scoped file work.
            with escalation_id_lock(queue.queue_dir, rewrite.escalation_id):
                path = queue.queue_dir / f'{rewrite.escalation_id}.json'
                if not path.exists():
                    logger.info(
                        'Skipping %s: no longer in the queue root (resolved/archived '
                        'between planning and applying)', rewrite.escalation_id,
                    )
                    skipped_missing += 1
                    continue
                esc = Escalation.from_json(path.read_text())

                # Re-check INSIDE the lock, closing the TOCTOU window exactly as
                # sweep._relocate_terminal re-checks its target.  A record that
                # stopped being legacy since planning (a concurrent rewrite, or a
                # resolution flipping status) must be left alone, not overwritten.
                if not is_legacy_gate_backlog_record(esc):
                    logger.info(
                        'Skipping %s: no longer matches the legacy shape at apply time',
                        rewrite.escalation_id,
                    )
                    skipped_no_longer_legacy += 1
                    continue

                # Rebuild from the FRESHLY READ record rather than reusing the
                # planned strings, so a detail change that landed since planning is
                # carried forward instead of being clobbered by stale text.
                new_summary = rebuild_summary(
                    esc.task_id, extract_gate_escalated_at(esc.detail)
                )
                new_detail = rebuild_detail(esc.detail)

                # Re-run the fingerprint guard on the bytes ACTUALLY being
                # written.  plan_rewrites checked the PLANNED detail, but the
                # detail above is recomputed from a possibly-newer record, so
                # the planned check does not cover it — and the design's stated
                # principle is that fingerprint preservation is a CHECK, not an
                # argument.  Same helper, same fail-closed outcome.
                if not fingerprint_preserved(esc, new_detail):
                    skipped_fingerprint_drift_at_apply += 1
                    continue

                esc.summary = new_summary
                esc.detail = new_detail
                # ONLY the two assignments above.  updated_at stays as found — see
                # this function's docstring.
                #
                # queue._rewrite, not queue.submit(): escalation_id_lock opens a
                # FRESH fd per call and flock is per-open-file-description, so
                # submit()'s own escalation_id_lock would take a second LOCK_EX on a
                # second fd and SELF-DEADLOCK inside ours (measured).  _rewrite is
                # the same unlocked atomic writer attach_dedupe_child calls inside
                # its own lock.
                queue._rewrite(rewrite.escalation_id, esc)
                rewritten += 1
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # One bad record costs one record, not the run: the remaining
            # rewrites still land and the operator still gets a report.  Safe to
            # simply continue because the script is idempotent — a re-run picks
            # this record up again once whatever broke it is fixed.  (Catching
            # ValueError covers json.JSONDecodeError, which subclasses it.)
            logger.warning(
                'Skipping %s: unreadable or unwritable at apply time (%s: %s)',
                rewrite.escalation_id, type(exc).__name__, exc,
            )
            skipped_error += 1
    return {
        'rewritten': rewritten,
        'skipped_missing': skipped_missing,
        'skipped_no_longer_legacy': skipped_no_longer_legacy,
        'skipped_fingerprint_drift_at_apply': skipped_fingerprint_drift_at_apply,
        'skipped_error': skipped_error,
    }


def run(queue_dir: str | Path, *, apply: bool = False) -> dict:
    """Plan the backfill for *queue_dir* and, only when *apply*, execute it.

    Returns a report dict.  When ``apply`` is False (dry-run, the default) the
    ``apply_rewrites`` call is not reached at all — ``plan_rewrites`` is pure, so
    a dry run is read-only by construction rather than by a flag check inside
    the writer.

    *queue_dir* is validated BEFORE the queue is constructed, because
    ``EscalationQueue.__init__`` does ``mkdir(parents=True, exist_ok=True)``
    (``escalation/src/escalation/queue.py``): a typo'd path would otherwise be
    CREATED as an empty directory and report ``pending_total: 0,
    legacy_total: 0`` — indistinguishable from a finished backfill, which would
    also make the operator procedure's step-5 idempotence check vacuously pass.
    It would additionally make a "read-only" dry run mutate the filesystem.
    Raises ``FileNotFoundError`` / ``NotADirectoryError``; ``main`` turns either
    into a non-zero exit naming the path.
    """
    path = Path(queue_dir)
    if not path.exists():
        raise FileNotFoundError(
            f'queue directory does not exist: {path} — refusing to create it, '
            'because an empty queue reports legacy_total: 0 and would read as a '
            'completed backfill'
        )
    if not path.is_dir():
        raise NotADirectoryError(f'queue path is not a directory: {path}')

    queue = EscalationQueue(path)
    plan = plan_rewrites(queue.get_pending())

    report: dict = {
        'queue_dir': str(queue_dir),
        'pending_total': plan.pending_total,
        'legacy_total': plan.legacy_total,
        'anchored': plan.anchored,
        'fallback': plan.fallback,
        'skipped_fingerprint_drift': plan.skipped_fingerprint_drift,
        'dry_run': not apply,
    }

    if apply:
        report.update(apply_rewrites(queue, plan))

    return report


EXIT_OK = 0
EXIT_BAD_QUEUE_DIR = 1
EXIT_FINGERPRINT_DRIFT = 3
"""``main`` exit codes.  ``2`` is deliberately unused: argparse already exits 2
on a usage error, and an operator reading ``$?`` must be able to tell a
mistyped flag from a record whose fold identity would move."""


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Returns an exit code.

    ``EXIT_FINGERPRINT_DRIFT`` (3) rather than 0 whenever any
    ``skipped_fingerprint_drift`` counter is non-zero.  That counter is the
    script's designated stop-and-investigate signal, and reporting it only as a
    JSON field and a WARNING log line would let any wrapper, cron or ``&&``
    chain read a drifting run as success.  ``skipped_missing`` /
    ``skipped_no_longer_legacy`` / ``skipped_error`` are deliberately NOT
    escalated to an exit code: the first two are benign by design (a steward
    resolved or a cycle rewrote the record) and the third is self-healing on a
    re-run, whereas drift means the rewrite itself is unsafe.
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s %(message)s')

    parser = argparse.ArgumentParser(
        description=(
            'Backfill: re-anchor legacy (pre-3520) reconciliation_stale_gate_backlog '
            'summaries onto the absolute `since <ISO>` form.'
        ),
    )
    parser.add_argument(
        '--queue-dir',
        default='./data/reconciliation/escalations',
        help='Path to the escalation queue directory (default: ./data/reconciliation/escalations).',
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        default=False,
        help=(
            'Perform writes. Without this flag the script is a dry run. '
            'OPERATOR ACTION: --apply mutates live escalation records that are '
            'not under git, so review the dry-run report first (expect '
            'skipped_fingerprint_drift: 0 — a non-zero count exits 3).'
        ),
    )
    args = parser.parse_args(argv)

    try:
        report = run(args.queue_dir, apply=args.apply)
    except OSError as exc:
        logger.error('--queue-dir is unusable: %s', exc)
        return EXIT_BAD_QUEUE_DIR

    print(json.dumps(report, indent=2, default=str))

    # Printed FIRST, then the exit code: the operator needs the report in order
    # to act on the signal it is being told to investigate.
    drift = report['skipped_fingerprint_drift'] + report.get(
        'skipped_fingerprint_drift_at_apply', 0
    )
    if drift:
        logger.error(
            '%d record(s) skipped because the rewrite would move their '
            'gate_backlog_fingerprint_key — investigate before re-running '
            '(exit %d)', drift, EXIT_FINGERPRINT_DRIFT,
        )
        return EXIT_FINGERPRINT_DRIFT
    return EXIT_OK


if __name__ == '__main__':
    sys.exit(main())
