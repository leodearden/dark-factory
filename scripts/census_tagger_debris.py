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

import json
import sqlite3
from collections.abc import Collection
from pathlib import Path
from typing import NamedTuple

# Tier 1 — the ONE tasks.db locator. Never a hand-rolled path.
from _task_db_scan import tasks_db_path

# The audit owns the defensive JSON decoders. ``_coerce_file_list`` is
# imported rather than re-implemented so the census's notion of "this
# record's current scope" is BYTE-IDENTICAL to the audit's and the repair's
# — the three scripts can then never disagree about the same record (INV-5).
# Precedent for importing audit internals including underscore names:
# repair_wiped_metadata_files.py:65-74.
from audit_wiped_metadata_files import (
    _EVENT_PLAN_SOURCES,
    FIDELITY_LOCK_LEVEL,
    NO_MERGE_EVENT,
    _coerce_file_list,
    _decode_files,
    classify_wipe_signature,
    load_merge_signatures,
    runs_db_path,
)

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


# ---------------------------------------------------------------------------
# The scope-event lens.
#
# DERIVED from the audit's table, never re-spelled (INV-5). The LENS DEFINITION
# — which event types carry a scope, under which payload key, at what fidelity
# — is the audit's to own, so a future edit there is inherited here
# AUTOMATICALLY. tests/scripts/test_census_tagger_debris.py holds that line by
# OBJECT IDENTITY on each shared entry: a copied literal would build a distinct
# tuple and fail, where a derived table shares the very objects.
#
# The census adds ONE entry the audit has no use for. Task 4525 names "a
# set_to_plan/phase_skipped{plan_files} event or real lock set" as the
# reconciliation signal, and a lock_acquired payload is
# {"modules": [...], "priority": ...} — module paths, so LOCK_LEVEL fidelity,
# the audit's own label for "this is a module set, never a plan.files list".
_LOCK_ACQUIRED_SOURCE = ("modules", "lock_acquired_event", FIDELITY_LOCK_LEVEL)
SCOPE_EVENT_SOURCES = {**_EVENT_PLAN_SOURCES, "lock_acquired": _LOCK_ACQUIRED_SOURCE}

# The conflict-with-nothing FALLBACK lock renders its module set as this single
# sentinel. Measured in the live event log: tasks 4514 and 4521 hold
# modules == ["task-4514"] / ["task-4521"], while task 4525's real derived lock
# lists actual paths. Counting the sentinel as a scope assertion would classify
# every tagger-era dispatch that derived NOTHING as plan-reconciled — inverting
# the classification for exactly the population this census exists to find.
_SYNTHETIC_LOCK_PREFIX = "task-"


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


class StampedRecord(NamedTuple):
    """One task record carrying ``metadata.files_tagged_at``, as stored.

    ``metadata_files`` is the record's CURRENT scope — for a never-reconciled
    record that is still the tagger's guess, which is the whole reason it is
    reported. An empty tuple is a real and common shape (the reify-5632 case),
    not an error.
    """

    tag: str
    task_id: int
    status: str
    files_tagged_at: str
    metadata_files: tuple[str, ...]


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


# ---------------------------------------------------------------------------
# Corpus readers. Every connection below is mode=ro; see the module docstring.
# ---------------------------------------------------------------------------


def _connect_readonly(path: str) -> sqlite3.Connection:
    """Open *path* through a read-only SQLite URI.

    The IDENTICAL spelling as audit_wiped_metadata_files.load_task_records:492.
    Kept as a single named helper so every corpus connection this module opens
    is provably the same one, and so the "cannot write" property is testable
    against the opener itself rather than re-asserted per call site.
    """
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def load_stamped_records(tasks_db_path: str) -> dict[tuple[str, int], StampedRecord]:
    """Load every task carrying a truthy ``metadata.files_tagged_at``.

    Keyed by the full ``(tag, id)`` primary key: the live corpora use a single
    ``master`` tag, but the schema permits the same numeric id under two tags
    and collapsing them would silently merge two distinct tasks.

    A row whose ``metadata`` is NULL, malformed JSON, a non-dict payload, or
    carries no/empty ``files_tagged_at`` is SKIPPED rather than raising: one
    corrupt row must never abort a sweep over 12,000+ records. That is the same
    defensive posture ``_decode_files`` takes on the audit side.

    Only the stamp itself is decoded here, not the scope — the scope goes
    through the audit's imported ``_coerce_file_list`` so a wrong-typed or
    empty ``files`` degrades to an empty tuple exactly as it does for the audit
    and the repair.
    """
    records: dict[tuple[str, int], StampedRecord] = {}
    conn = _connect_readonly(tasks_db_path)
    try:
        cursor = conn.execute("SELECT tag, id, status, metadata FROM tasks")
        for tag, task_id, status, metadata in cursor:
            if not metadata or not isinstance(metadata, (str, bytes)):
                continue
            try:
                payload = json.loads(metadata)
            except (ValueError, TypeError):
                continue
            if not isinstance(payload, dict):
                continue
            stamp = payload.get("files_tagged_at")
            if not stamp or not isinstance(stamp, str):
                continue
            records[(tag, task_id)] = StampedRecord(
                tag=tag,
                task_id=task_id,
                status=status,
                files_tagged_at=stamp,
                metadata_files=_coerce_file_list(payload.get("files")),
            )
    finally:
        conn.close()
    return records


def _real_lock_modules(modules: tuple[str, ...], task_id: str) -> tuple[str, ...]:
    """Drop the synthetic ``task-<id>`` fallback sentinel from a lock's modules.

    Only the sentinel for THIS task is dropped, matched exactly — a real module
    path that merely starts with ``task-`` is left alone. A lock carrying the
    sentinel alongside real paths is still a real lock; only the sentinel goes.
    """
    sentinel = f"{_SYNTHETIC_LOCK_PREFIX}{task_id}"
    return tuple(module for module in modules if module != sentinel)


def load_scope_events(
    runs_db_path: str, task_ids: Collection[str]
) -> dict[str, list[ScopeEvent]]:
    """Load every timestamped scope event for *task_ids*, keyed by task id.

    WHY THE AUDIT'S OWN READER COULD NOT BE CALLED HERE.
    ``audit_wiped_metadata_files.load_plan_files_from_events`` answers a
    different question: it collapses each task to ONE highest-fidelity
    ``PlanFilesRecord`` and selects only ``id, task_id, event_type, data`` —
    it never reads ``timestamp`` at all. Every classification this census makes
    is a comparison of an event's timestamp against ``metadata.files_tagged_at``,
    so a collapsed, un-timestamped record cannot express "did a scope event
    postdate the stamp". What differs is the QUERY and the retention policy,
    which is genuinely new behaviour; the LENS stays single-sourced above.

    Filtered to *task_ids* in SQL rather than in Python: dark_factory's event
    log alone carries 21,345 ``lock_acquired`` rows, and decoding all of them to
    then discard almost all would make a six-corpus sweep needlessly expensive.
    An empty *task_ids* returns ``{}`` — never "all events".

    ``ORDER BY id`` is emission order, so each task's list is chronological as
    the event log recorded it. Rows with a NULL ``task_id``, malformed/NULL
    ``data``, a non-dict payload, a missing key, or an empty/wrong-typed file
    list are SKIPPED rather than raising: an unusable row is not a scope
    assertion, and one corrupt payload must not abort the sweep.
    """
    wanted = {str(task_id) for task_id in task_ids}
    if not wanted:
        return {}

    events: dict[str, list[ScopeEvent]] = {}
    type_slots = ", ".join("?" for _ in SCOPE_EVENT_SOURCES)
    id_slots = ", ".join("?" for _ in wanted)
    ordered_ids = sorted(wanted)
    conn = _connect_readonly(runs_db_path)
    try:
        cursor = conn.execute(
            "SELECT id, timestamp, task_id, event_type, data FROM events "
            f"WHERE event_type IN ({type_slots}) AND task_id IN ({id_slots}) "
            "ORDER BY id",
            (*SCOPE_EVENT_SOURCES, *ordered_ids),
        )
        for event_id, timestamp, task_id, event_type, data in cursor:
            if not task_id or not timestamp:
                continue
            key, _source, fidelity = SCOPE_EVENT_SOURCES[event_type]
            files = _decode_files(data, key)
            if event_type == "lock_acquired":
                files = _real_lock_modules(files, str(task_id))
            if not files:
                continue
            events.setdefault(str(task_id), []).append(
                ScopeEvent(
                    timestamp=timestamp,
                    event_type=event_type,
                    event_id=event_id,
                    fidelity=fidelity,
                    file_count=len(files),
                )
            )
    finally:
        conn.close()
    return events


# ---------------------------------------------------------------------------
# The census itself.
# ---------------------------------------------------------------------------


class CensusCoverage(NamedTuple):
    """How much of one project the census could actually see.

    ALWAYS reported, including for a project with zero stamped records:
    "found nothing" and "looked at nothing" are different claims, and this
    block is the only thing that distinguishes them.

    ``event_log_read`` False means the reconciliation and wipe-signature axes
    are UNKNOWN for every record in this project, not measured to be clean.
    Presenting them as clean would be exactly the no-silent-fail-soft violation
    in docs/legibility/design-invariants.md.
    """

    project_root: str
    project_id: str
    total_tasks: int
    stamped_records: int
    event_log_read: bool
    event_log_path: str


class ProjectCensus(NamedTuple):
    """One project's census: what was found, and what could be seen."""

    project_root: str
    project_id: str
    records: list[CensusRecord]
    coverage: CensusCoverage


def project_id_for(project_root: str) -> str:
    """``/home/leo/src/solar-challenge-platform`` -> ``solar_challenge_platform``.

    The six corpora spell their project ids with underscores where the
    directory uses hyphens. The artifact must key on the id spelling its
    consumers already use, not on the directory name.
    """
    return Path(project_root).name.replace("-", "_")


def _record_sort_key(record: CensusRecord) -> tuple[str, int, str]:
    """Sort by (tag, NUMERIC task id) so 100 follows 20 rather than preceding it.

    Same shape and same fallback as audit_wiped_metadata_files._candidate_sort_key
    :573-578 — a non-numeric id sorts first under its own string rather than
    raising, so one odd id cannot abort a whole sweep's rendering.
    """
    try:
        return (record.tag, int(record.task_id), "")
    except (TypeError, ValueError):
        return (record.tag, 0, str(record.task_id))


def census_project(project_root: str) -> ProjectCensus:
    """Census one project root for surviving ``metadata.files_tagged_at`` stamps.

    Reads the task store and the event log, both READ-ONLY, classifies every
    stamped record on all three axes, and attaches the audit's own
    ``merge_finalized`` verdict as correlating evidence.

    A missing or unreadable runs.db is TOLERATED but never hidden: every record
    still appears, degraded to NEVER_RECONCILED / NO_PRIOR_SCOPE /
    NO_MERGE_EVENT, and ``coverage.event_log_read`` goes False so a consumer
    can see the axes were unknown rather than measured clean.

    A tasks.db error PROPAGATES. That is the contract
    ``_task_db_scan.sweep_project_roots`` documents at :444-472 — exactly one
    result per root, or raise — and its exit-3 gate rests on the resulting
    ``len(audits) + len(unreadable) == len(roots)`` equality. Returning a
    partial census here would silently re-open the false green exit 3 closes.
    """
    project_id = project_id_for(project_root)
    tasks_db = tasks_db_path(project_root)

    conn = _connect_readonly(str(tasks_db))
    try:
        (total_tasks,) = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()
    finally:
        conn.close()
    stamped = load_stamped_records(str(tasks_db))

    # runs_db_path is IMPORTED from the audit, which already resolves
    # <root>/data/orchestrator/runs.db. That is how the 0-byte data/runs.db
    # decoy task 4525 warns about is avoided — by reuse, not by a new constant
    # this module would have to keep correct on its own.
    runs_db = runs_db_path(project_root)
    scope_events: dict[str, list[ScopeEvent]] = {}
    merge_signatures: dict[str, list[dict]] = {}
    event_log_read = False
    if runs_db.exists():
        try:
            scope_events = load_scope_events(
                str(runs_db), {str(record.task_id) for record in stamped.values()}
            )
            merge_signatures = load_merge_signatures(str(runs_db))
            event_log_read = True
        except sqlite3.Error:
            # Deliberately NOT re-raised: an unreadable EVENT log costs two
            # axes, while an unreadable TASK store costs the whole population.
            # The coverage flag below is what keeps the difference visible.
            scope_events = {}
            merge_signatures = {}

    records: list[CensusRecord] = []
    for record in stamped.values():
        events = scope_events.get(str(record.task_id), [])
        verdict = classify_record(record.files_tagged_at, record.status, events)
        records.append(
            CensusRecord(
                project_id=project_id,
                tag=record.tag,
                task_id=record.task_id,
                status=record.status,
                files_tagged_at=record.files_tagged_at,
                status_class=verdict.status_class,
                reconciliation=verdict.reconciliation,
                wipe_signature=verdict.wipe_signature,
                reconciled_by=verdict.reconciled_by,
                preceded_by=verdict.preceded_by,
                metadata_files=record.metadata_files,
                merge_signature=(
                    classify_wipe_signature(merge_signatures[str(record.task_id)])
                    if str(record.task_id) in merge_signatures
                    else NO_MERGE_EVENT
                ),
            )
        )
    records.sort(key=_record_sort_key)

    return ProjectCensus(
        project_root=project_root,
        project_id=project_id,
        records=records,
        coverage=CensusCoverage(
            project_root=project_root,
            project_id=project_id,
            total_tasks=total_tasks,
            stamped_records=len(records),
            event_log_read=event_log_read,
            event_log_path=str(runs_db),
        ),
    )
