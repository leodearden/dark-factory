#!/usr/bin/env python3
"""Census every task record carrying ``metadata.files_tagged_at``, and classify it.

READ-ONLY / REPORT-ONLY: this module and its CLI never mutate a task record,
an event record, or a plan artifact. Every corpus connection it opens is a
read-only SQLite URI (``sqlite3.connect(f"file:{path}?mode=ro", uri=True)``),
the identical spelling as ``audit_wiped_metadata_files.load_task_records``,
so the sweep is structurally incapable of writing to the live WAL databases
the six running orchestrators hold open. REMEDIATION IS A SEPARATE, REVIEWED
FOLLOW-UP — nothing here repairs, backfills or clears a stamp.

WHAT THIS IS FOR. PRD task epsilon of ``plans/module-tagger-retirement-prd.md``
(decision 3). The retired module tagger stamped ``metadata.files_tagged_at``
onto task records alongside a GUESSED file scope. This census enumerates every
surviving stamp across all six project corpora and classifies each on three
axes, so its two consumers can join a machine-readable candidate set instead of
re-deriving one from prose:

  * DF 3113 P4a — the forward fix for the DONE-path ``metadata.files`` wipe.
  * DF 3427 — the debris repair pipeline.

Both already speak the audit's ``merge_finalized`` wipe-signature vocabulary,
so every census row carries that verdict too, keyed to the same task ids.

CORPUS ACCESS — WHY tasks.db, READ-ONLY (task 4525, esc-4525-1).
Task 4525's text asks for "the fused-memory read path or each project's own
data/orchestrator/runs.db opened readonly; NEVER raw .taskmaster/tasks.db".
That is unsatisfiable as written, and the same task's reuse mandate points the
other way. Measured at plan time:

  * runs.db carries NO task metadata at all (tables: account_events, events,
    invocations, runs, scheduler_state, sqlite_sequence, task_results — there
    is no metadata column), so it cannot answer "which records carry
    metadata.files_tagged_at". The stamp lives ONLY in ``tasks.metadata``.
  * The MCP read path is impractical at 12,323 tasks across six corpora:
    ``get_statuses`` carries no metadata, and ``get_tasks`` at that size
    exceeds the documented MCP transport limit.
  * The prescribed reuse target IS the mode=ro tasks.db pattern. Four
    in-production sweep scripts already read it that way through
    ``_task_db_scan`` Tier 1, and ``load_task_records`` documents verbatim
    that mode=ro "is structurally incapable of mutating live task records even
    while fused-memory holds the same file open in WAL mode".

The hazard that clause guards — writing behind the TaskInterceptor's back — is
structurally excluded by mode=ro. Filed as esc-4525-1 (non-blocking
design_concern) asking the PRD owner to correct the task and PRD wording.

IMPORT-RESOLUTION CONTRACT — read before moving this file.
This module MUST stay a flat sibling in ``scripts/``, and must NEVER be invoked
via ``python -m``. Its CLI tests drive ``main()`` by shelling out
(``subprocess.run([sys.executable, <script path>, ...])``), and
``tests/scripts/conftest.py``'s sys.path insertion does not reach those child
processes: they resolve ``import audit_wiped_metadata_files`` and
``import _task_db_scan`` solely because a DIRECTLY-EXECUTED script places its
own directory at ``sys.path[0]``. Either change breaks every CLI test at once.
Same contract, same reasons, as ``_task_db_scan.py:93-103``.
"""
from __future__ import annotations

from typing import NamedTuple

# The terminal allowlist is IMPORTED, never re-spelled. The census exists to
# tell the repair which non-terminal records it is currently blind to (the
# repair skips them via SKIP_NOT_TERMINAL), so the two notions of "terminal"
# must be the same object or the census would be answering a question the
# repair is not asking.
from repair_wiped_metadata_files import TERMINAL_STATUSES

# ---------------------------------------------------------------------------
# The classification vocabulary.
#
# These six strings are the artifact's PUBLIC contract: DF 3113 P4a and DF 3427
# read them out of the committed JSON, where they cannot see this module's
# constants. tests/scripts/test_census_tagger_debris.py pins each literal so a
# rename fails CI rather than silently producing an unjoinable artifact.
# ---------------------------------------------------------------------------

# Axis 1 — status. Whether the record is still live work.
STATUS_TERMINAL = "terminal"
STATUS_NON_TERMINAL = "non_terminal"

# Axis 2 — reconciliation. Whether a real scope derivation ever SUPERSEDED the
# tagger's guess. A reconciled record is no longer a victim; a never-reconciled
# one still has the guess as its live scope.
RECONCILED = "plan_reconciled"
NEVER_RECONCILED = "never_reconciled"

# Axis 3 — wipe signature. Whether an authoritative scope EXISTED BEFORE the
# tagger stamped over it (the damaging case), or the stamp was the first scope
# this record ever had.
POST_WIPE_OVERWRITE = "post_wipe_overwrite"
NO_PRIOR_SCOPE = "no_prior_scope"


class ScopeEvent(NamedTuple):
    """One timestamped event asserting a real file/module scope for a task.

    ``fidelity`` carries the audit's own FIDELITY_* label, which stays
    load-bearing here for the same reason it is there: a lock-level scope is a
    MODULE set, not a plan.files list, and the two must never be conflated by a
    downstream repair. ``file_count`` is recorded rather than the paths
    themselves — the census classifies WHEN a scope existed, not what it was,
    and carrying full path lists for every event would bloat the artifact
    without informing any of the three axes.
    """

    timestamp: str
    event_type: str
    event_id: int
    fidelity: str
    file_count: int


class ScopeEvidence(NamedTuple):
    """The deciding event behind one classification axis, or all-None.

    ALWAYS present on a classification, even when nothing was found: a record
    whose axis is undecided still carries these three keys set to None. A
    MISSING key in the artifact would be indistinguishable from a serializer
    bug, whereas an explicit null says "looked, found nothing" (INV-2 —
    no classification is a prose-only claim).
    """

    event_type: str | None
    event_id: int | None
    timestamp: str | None


_NO_EVIDENCE = ScopeEvidence(event_type=None, event_id=None, timestamp=None)


class Classification(NamedTuple):
    """The three-axis verdict for one stamped record, plus its evidence."""

    status_class: str
    reconciliation: str
    wipe_signature: str
    reconciled_by: ScopeEvidence
    preceded_by: ScopeEvidence


class CensusRecord(NamedTuple):
    """One stamped task record as the census reports it.

    ``merge_signature`` is the audit's OWN ``classify_wipe_signature`` verdict
    over this task's ``merge_finalized`` history — a second, independently
    derived piece of evidence in a vocabulary DF 3113 P4a and DF 3427 already
    consume, so they can join this artifact to their existing candidate sets
    without a translation layer. It is deliberately NOT the same field as
    ``wipe_signature``, which is this census's own axis-3 label.
    """

    project_id: str
    tag: str
    task_id: int
    status: str
    files_tagged_at: str
    status_class: str
    reconciliation: str
    wipe_signature: str
    reconciled_by: ScopeEvidence
    preceded_by: ScopeEvidence
    metadata_files: tuple[str, ...]
    merge_signature: str


def _evidence(event: ScopeEvent) -> ScopeEvidence:
    return ScopeEvidence(
        event_type=event.event_type,
        event_id=event.event_id,
        timestamp=event.timestamp,
    )


def classify_record(
    files_tagged_at: str,
    status: str,
    scope_events: list[ScopeEvent],
) -> Classification:
    """Classify one stamped record on all three axes. PURE — no I/O.

    *files_tagged_at* is the record's stamp; *scope_events* are every scope
    event observed for that task, in any order.

    TIMESTAMP COMPARISON IS STRICT (``>`` / ``<``), DELIBERATELY. An event
    bearing exactly the stamp's instant is evidence of NEITHER reconciliation
    nor overwrite: the two writes are not ordered with respect to each other at
    equal timestamps, and picking an order would be a guess presented as a
    measurement. Both live columns are ISO-8601 strings with a timezone
    (measured on ``events.timestamp`` and on ``metadata.files_tagged_at``),
    written by the same process family at the same offset, so a plain string
    compare is total and correct.

    Evidence selection: the EARLIEST post-stamp event decides reconciliation
    (the first thing that superseded the guess), and the LATEST pre-stamp event
    decides the overwrite (the most recent authoritative scope the stamp wrote
    over). Both are the closest event to the stamp on their side, so the
    evidence names the write that actually bracketed it.

    The two axes are INDEPENDENT: a record can have been stamped over a prior
    scope AND later reconciled. Collapsing them would lose exactly the
    distinction the repair pipeline needs.
    """
    status_class = STATUS_TERMINAL if status in TERMINAL_STATUSES else STATUS_NON_TERMINAL

    after = [event for event in scope_events if event.timestamp > files_tagged_at]
    before = [event for event in scope_events if event.timestamp < files_tagged_at]

    if after:
        reconciliation = RECONCILED
        reconciled_by = _evidence(min(after, key=lambda event: event.timestamp))
    else:
        reconciliation = NEVER_RECONCILED
        reconciled_by = _NO_EVIDENCE

    if before:
        wipe_signature = POST_WIPE_OVERWRITE
        preceded_by = _evidence(max(before, key=lambda event: event.timestamp))
    else:
        wipe_signature = NO_PRIOR_SCOPE
        preceded_by = _NO_EVIDENCE

    return Classification(
        status_class=status_class,
        reconciliation=reconciliation,
        wipe_signature=wipe_signature,
        reconciled_by=reconciled_by,
        preceded_by=preceded_by,
    )
