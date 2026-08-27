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

import argparse
import contextlib
import json
import os
import sqlite3
import sys
import tempfile
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import NamedTuple

# Tier 1 — the ONE tasks.db locator. Tier 3 — the warn-and-continue per-root
# loop and the shared no-root message. run_audit_cli is deliberately NOT
# adopted; the exit-ladder block near main() records why.
# discover_project_roots comes second-hand through the audit's documented
# re-export, the same route repair_wiped_metadata_files.py takes.
from _task_db_scan import (
    NO_PROJECT_ROOT_RESOLVED_MESSAGE,
    sweep_project_roots,
    tasks_db_path,
)

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
    discover_project_roots,
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
# tagger's guess.
#
# THREE VALUES, NOT TWO — and the correction is the whole point of schema v2.
# v1 counted ANY post-stamp ``lock_acquired`` as proof the guess had been
# superseded. It is not. ``Scheduler._get_modules``
# (orchestrator/src/orchestrator/scheduler.py:8797-8801) computes the locked
# module set as ``derive_modules(metadata["files"], depth)`` — STRAIGHT FROM
# ``metadata.files``, which for a never-reconciled tagger-stamped record IS the
# tagger's guess. A post-stamp lock is therefore an ECHO of the guess, not an
# independent scope assertion, for exactly the population this census exists to
# flag. Measured on the v1 artifact: 278 of 286 plan_reconciled verdicts (97%)
# rested on a lock alone, and 267 of 507 records were called reconciled while
# still carrying a non-empty ``metadata.files``.
#
#   plan_reconciled  — a set_to_plan / phase_skipped{plan_files} event
#                      postdates the stamp. A genuine PLAN-derived assertion,
#                      and the audit module's own lens. The strong signal.
#   lock_reconciled  — no plan event, but a post-stamp lock names at least one
#                      module the record's own ``metadata.files`` cannot
#                      explain, so it was not a re-derivation of the guess.
#                      Real but WEAKER evidence, given its own label so a
#                      consumer can filter it rather than having to trust it.
#   never_reconciled — nothing superseded the guess. It is still this record's
#                      live scope.
RECONCILED = "plan_reconciled"
LOCK_RECONCILED = "lock_reconciled"
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
    downstream repair.

    ``files`` HOLDS THE PATHS, and ``file_count`` is derived from that same
    list at every construction site so the two cannot drift. v1 kept only the
    count, on the reasoning that the census classifies WHEN a scope existed
    rather than what it was; schema v2 makes the paths load-bearing, because
    the axis-2 echo test asks whether a lock's modules are merely a
    re-derivation of the record's own ``metadata.files``, which is a question
    about the paths themselves. THE ARTIFACT STILL EMITS ONLY ``file_count``:
    the paths stay in memory, so the no-path-bloat property that reasoning
    protected survives and no JSON row grows.
    """

    timestamp: str
    event_type: str
    event_id: int
    fidelity: str
    file_count: int
    files: tuple[str, ...]


class ScopeEvidence(NamedTuple):
    """The deciding event behind one classification axis, or all-None.

    ALWAYS present on a classification, even when nothing was found: a record
    whose axis is undecided still carries these four keys set to None. A
    MISSING key in the artifact would be indistinguishable from a serializer
    bug, whereas an explicit null says "looked, found nothing" (INV-2 —
    no classification is a prose-only claim).

    ``fidelity`` (schema v2) is the deciding event's own FIDELITY_* label, so a
    consumer can tell a file-level assertion from a lock-level one WITHOUT
    re-deriving it from ``event_type``. The two are not interchangeable: a
    module path must never be written back into ``metadata.files``.
    """

    event_type: str | None
    event_id: int | None
    timestamp: str | None
    fidelity: str | None


_NO_EVIDENCE = ScopeEvidence(
    event_type=None, event_id=None, timestamp=None, fidelity=None
)


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
        fidelity=event.fidelity,
    )


def _path_parts(path: str) -> tuple[str, ...]:
    """Split a file or module path into its non-empty '/'-separated components.

    Module paths from a lock render with a trailing slash (``"scripts/"``) and
    file paths do not, so both sides are normalised here rather than at each
    comparison.
    """
    return tuple(part for part in path.split("/") if part)


def _module_is_explained_by(module: str, files: tuple[str, ...]) -> bool:
    """True when *module* is a component-wise PATH PREFIX of some entry in *files*.

    THIS IS THE DEPTH-INVARIANT OF ``derive_modules``. ``files_to_modules``
    truncates each path to ``depth`` components, so EVERY module it can emit is
    a prefix of some entry in the file list it was given — whatever ``depth``
    happened to be. Testing the prefix relation rather than
    ``derive_modules(files, depth) == modules`` is what keeps this correct
    without knowing the depth: measured at plan time, ``lock_depth`` is
    12/10/4/4/3/unset(2) across the six corpora AND changed mid-tagger-era
    (dark-factory 4->12, reify 4->10), so an equality-at-one-depth test would
    have silently regressed the first time an operator retuned it.

    Component-wise, never ``str.startswith``: ``"a/bcd/e.py".startswith("a/b")``
    is True and meaningless, and taking it as a match would call an unrelated
    module an echo and downgrade a genuine reconciliation to never_reconciled.
    """
    parts = _path_parts(module)
    if not parts:
        return False
    # A slice longer than the candidate simply yields the whole candidate,
    # which cannot equal the longer *parts* — so a module DEEPER than the file
    # it is compared against falls out as unexplained without a length guard.
    return any(_path_parts(candidate)[: len(parts)] == parts for candidate in files)


def _lock_echoes_guess(event: ScopeEvent, metadata_files: tuple[str, ...]) -> bool:
    """True when *event*'s module set is fully explained by the record's own files.

    An echo asserts nothing the tagger did not already assert, so it is NOT
    evidence that the guess was superseded. See the axis-2 vocabulary block.
    """
    if not event.files:
        # Not a scope assertion at all. Treated as an echo — i.e. as NO
        # evidence of reconciliation — because the alternative would credit an
        # empty module set with superseding the guess. load_scope_events
        # already drops these rows; this is a defensive floor, not a live path.
        return True
    if not metadata_files:
        # THE reify-5632 SHAPE (metadata.files == []). With no guess on the
        # record, NOTHING can explain the lock's modules, so it cannot be a
        # re-derivation of the guess. Pinned so an empty guess can never
        # degenerate into a false echo via a vacuous all().
        return False
    return all(_module_is_explained_by(module, metadata_files) for module in event.files)


def classify_record(
    files_tagged_at: str,
    status: str,
    scope_events: list[ScopeEvent],
    metadata_files: tuple[str, ...],
) -> Classification:
    """Classify one stamped record on all three axes. PURE — no I/O.

    *files_tagged_at* is the record's stamp; *scope_events* are every scope
    event observed for that task, in any order; *metadata_files* is that
    record's CURRENT ``metadata.files`` — for a never-reconciled record, the
    tagger's surviving guess.

    *metadata_files* IS REQUIRED, deliberately, with no default. Defaulting it
    to ``()`` would mean "assume this record carries no guess", under which
    every post-stamp lock reads as genuine and the record is reported as
    already reconciled — the exact false-repaired verdict schema v2 exists to
    correct. A caller that forgets it fails loudly at the call site instead.

    TIMESTAMP COMPARISON IS STRICT (``>`` / ``<``), DELIBERATELY. An event
    bearing exactly the stamp's instant is evidence of NEITHER reconciliation
    nor overwrite: the two writes are not ordered with respect to each other at
    equal timestamps, and picking an order would be a guess presented as a
    measurement. Both live columns are ISO-8601 strings with a timezone
    (measured on ``events.timestamp`` and on ``metadata.files_tagged_at``),
    written by the same process family at the same offset, so a plain string
    compare is total and correct.

    AXIS 2 IS PARTITIONED, NOT MERELY ORDERED. Post-stamp events split into
    PLAN-level (the audit's own ``_EVENT_PLAN_SOURCES`` — set_to_plan and
    phase_skipped{plan_files}) and LOCK-level (lock_acquired). A plan event is
    a genuine plan-derived assertion and always wins, EVEN IF a lock postdates
    the stamp earlier: the stronger signal must never be masked by the weaker
    one. A lock counts only if it survives ``_lock_echoes_guess``, and then
    under its own weaker label. Plan events are never echo-filtered — they are
    assertions, not re-derivations of ``metadata.files``.

    Evidence selection: the EARLIEST qualifying post-stamp event decides
    reconciliation (the first thing that superseded the guess), and the LATEST
    pre-stamp event decides the overwrite (the most recent authoritative scope
    the stamp wrote over). Both are the closest event to the stamp on their
    side, so the evidence names the write that actually bracketed it.

    The two axes are INDEPENDENT: a record can have been stamped over a prior
    scope AND later reconciled. Collapsing them would lose exactly the
    distinction the repair pipeline needs.
    """
    status_class = STATUS_TERMINAL if status in TERMINAL_STATUSES else STATUS_NON_TERMINAL

    after = [event for event in scope_events if event.timestamp > files_tagged_at]
    before = [event for event in scope_events if event.timestamp < files_tagged_at]

    plan_after = [event for event in after if event.event_type in _EVENT_PLAN_SOURCES]
    lock_after = [
        event
        for event in after
        if event.event_type not in _EVENT_PLAN_SOURCES
        and not _lock_echoes_guess(event, metadata_files)
    ]

    if plan_after:
        reconciliation = RECONCILED
        reconciled_by = _evidence(min(plan_after, key=lambda event: event.timestamp))
    elif lock_after:
        reconciliation = LOCK_RECONCILED
        reconciled_by = _evidence(min(lock_after, key=lambda event: event.timestamp))
    else:
        reconciliation = NEVER_RECONCILED
        reconciled_by = _NO_EVIDENCE

    # AXIS 3 IS DELIBERATELY NOT ECHO-FILTERED, and this asymmetry is the
    # correct one: a PRE-stamp lock proves a file-derived scope existed BEFORE
    # the tagger stamped, so it cannot be an echo of a guess that did not yet
    # exist. Applying the axis-2 filter here "for consistency" would erase
    # exactly the wipe signal this census exists to surface.
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

    THE EVENT AND MERGE JOINS CANNOT HONOUR THAT KEY, and ``census_project``
    says so at the lookup sites: ``events`` has no ``tag`` column at all (its
    columns are id, timestamp, run_id, task_id, event_type, phase, role, data,
    cost_usd, duration_ms), so a scope event and a merge verdict can only be
    keyed by numeric id. The full key is therefore load-bearing HERE — it keeps
    two same-id records DISTINCT ROWS in the artifact — and presentational
    there: two such rows would receive the same event list. Unreachable on
    today's single-``master``-tag corpora, and it would take a tag column on
    ``events`` to make it fixable rather than merely documented.

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
                    files=files,
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


def _sort_key(record: CensusRecord, lead: str) -> tuple[str, int, str]:
    """Sort by (*lead*, NUMERIC task id) so 100 follows 20 rather than preceding it.

    ONE helper for BOTH orderings, parameterized by the leading field. A
    ProjectCensus orders its records by ``tag`` (a single project, possibly
    several tags); the cross-project report orders by ``project_id``. The two
    differ in nothing else, and the near-identical copies this replaces invited
    drift in the FALLBACK — the part that actually carries the risk.

    Same shape and same fallback as audit_wiped_metadata_files._candidate_sort_key
    :573-578 — a non-numeric id sorts first under its own string rather than
    raising, so one odd id cannot abort a whole sweep's rendering.
    """
    try:
        return (lead, int(record.task_id), "")
    except (TypeError, ValueError):
        return (lead, 0, str(record.task_id))


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
        # KEYED BY NUMERIC ID, NOT BY (tag, id) — deliberately, and not an
        # oversight against load_stamped_records' emphatic full-key docstring:
        # the ``events`` table carries NO tag column, so neither this lookup nor
        # the merge_signature one below can be disambiguated by tag. Two records
        # sharing an id under different tags stay distinct ROWS (the stamped
        # dict is keyed on the full primary key) but would share an event list
        # and a merge verdict. Not reachable on today's single-``master``-tag
        # corpora; fixing it would take a tag column on ``events``.
        events = scope_events.get(str(record.task_id), [])
        verdict = classify_record(
            record.files_tagged_at, record.status, events, record.metadata_files
        )
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
    records.sort(key=lambda record: _sort_key(record, record.tag))

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


# ---------------------------------------------------------------------------
# The report.
#
# Bump this and record what changed right here, the way
# census_memory_metadata.py:446-458 does, so a consumer reading an older
# artifact can tell what it is looking at.
#
# v1 — the initial shape: schema_version, params, projects (per-project totals
#      plus per-axis and eight-cell counts), records (the COMPLETE population,
#      never truncated), coverage.
#
# v2 — THE AXIS-2 CORRECTION. v1's defect in one line: it counted ANY
#      post-stamp lock_acquired as proof the tagger's guess had been
#      superseded, but the scheduler derives a lock's module set FROM
#      metadata.files, so for a never-reconciled record that lock is an echo of
#      the guess and v1 reported the majority of live victims as repaired.
#      What changed:
#        * axis 2 gained a third value, ``lock_reconciled``. Per-project
#          ``reconciliation`` blocks gain that key and ``cells`` goes 8 -> 12.
#        * ``plan_reconciled`` NARROWED to genuine plan-derived assertions
#          (set_to_plan / phase_skipped{plan_files}) — the audit's own lens.
#        * post-stamp locks are ECHO-FILTERED against the record's own
#          ``metadata.files``; a surviving one classifies ``lock_reconciled``,
#          never ``plan_reconciled``.
#        * every evidence object (``reconciled_by`` / ``preceded_by``) gained
#          a ``fidelity`` key, present-and-null when the axis is undecided.
#      DF 3113 P4a and DF 3427 MUST re-read: a record they saw as
#      plan_reconciled under v1 may be lock_reconciled or never_reconciled
#      here, and the never_reconciled population is materially larger.
#      Axis 1, axis 3 and every other key are unchanged.
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 2

# The vocabulary, ordered. Iterating THIS rather than the observed data is what
# makes a zero-valued cell present rather than absent — a missing key must
# never be readable as a zero.
_STATUS_CLASSES = (STATUS_TERMINAL, STATUS_NON_TERMINAL)
_RECONCILIATIONS = (RECONCILED, LOCK_RECONCILED, NEVER_RECONCILED)
_WIPE_SIGNATURES = (POST_WIPE_OVERWRITE, NO_PRIOR_SCOPE)

# THE ARTIFACT DELIBERATELY CARRIES NO generated_at, timestamp OR SHA, and a
# test walks the built structure to keep it that way. All three committed
# plans/*.json artifacts in this repo follow the same convention: with a clock
# read in the file every regeneration would diff dirty, destroying exactly the
# signal task 4525 asks for — "re-running the script reproduces the counts",
# checkable by `git diff --exit-code`. Provenance is the params block's
# regen_command instead. Please do not "fix" this back in.
_REGEN_COMMAND_HEAD = "python scripts/census_tagger_debris.py"


def _cell_key(record: CensusRecord) -> str:
    return f"{record.status_class}|{record.reconciliation}|{record.wipe_signature}"


def _record_to_dict(record: CensusRecord) -> dict:
    return {
        "project_id": record.project_id,
        "tag": record.tag,
        "task_id": record.task_id,
        "status": record.status,
        "files_tagged_at": record.files_tagged_at,
        "status_class": record.status_class,
        "reconciliation": record.reconciliation,
        "wipe_signature": record.wipe_signature,
        "reconciled_by": record.reconciled_by._asdict(),
        "preceded_by": record.preceded_by._asdict(),
        "merge_signature": record.merge_signature,
        "metadata_files": list(record.metadata_files),
    }


def build_report(
    censuses: list[ProjectCensus], unreadable: Sequence[str] = ()
) -> dict:
    """Assemble the committed artifact's JSON structure. PURE — no I/O, no clock.

    Ordering is total and derived from the data, never from the caller's
    argument order: projects sort by ``project_id``, records by
    ``(project_id, numeric task id)``. Passing the same censuses in a different
    order therefore yields a byte-identical dump, which is what lets the
    artifact be regenerated and diffed as a reproducibility check.

    The ``records`` array is the COMPLETE population and is never truncated;
    only the markdown twin is capped.

    *unreadable* is the roots ``sweep_project_roots`` had to SKIP. They are
    recorded in the coverage block so the incompleteness survives in the
    artifact itself, not only on a stderr line nobody kept.
    """
    ordered = sorted(censuses, key=lambda census: census.project_id)

    projects: dict[str, dict] = {}
    for census in ordered:
        # Annotated dict[str, int] rather than left to inference: the
        # vocabulary constants are string LITERALS, so an unannotated
        # dict.fromkeys narrows the key type to those literals and then
        # rejects the plain `str` a CensusRecord field carries.
        cells: dict[str, int] = {
            f"{status}|{reconciliation}|{signature}": 0
            for status in _STATUS_CLASSES
            for reconciliation in _RECONCILIATIONS
            for signature in _WIPE_SIGNATURES
        }
        status_counts: dict[str, int] = dict.fromkeys(_STATUS_CLASSES, 0)
        reconciliation_counts: dict[str, int] = dict.fromkeys(_RECONCILIATIONS, 0)
        signature_counts: dict[str, int] = dict.fromkeys(_WIPE_SIGNATURES, 0)
        for record in census.records:
            status_counts[record.status_class] += 1
            reconciliation_counts[record.reconciliation] += 1
            signature_counts[record.wipe_signature] += 1
            cells[_cell_key(record)] += 1

        projects[census.project_id] = {
            "project_root": census.project_root,
            "total_tasks": census.coverage.total_tasks,
            "stamped_records": census.coverage.stamped_records,
            "event_log_read": census.coverage.event_log_read,
            "status_class": status_counts,
            "reconciliation": reconciliation_counts,
            "wipe_signature": signature_counts,
            "cells": cells,
        }

    # Re-sorted under a DIFFERENT leading field: census_project already ordered
    # each project's records by (tag, id) for its own ProjectCensus.records
    # consumers, and the artifact needs (project_id, id) across all of them. The
    # per-project sort is therefore not redundant work this could drop — it is a
    # separate ordering with a separate consumer.
    records = sorted(
        (record for census in ordered for record in census.records),
        key=lambda record: _sort_key(record, record.project_id),
    )
    roots = [census.project_root for census in ordered]

    return {
        "schema_version": SCHEMA_VERSION,
        "params": {
            "project_roots": roots,
            "stamp_key": "metadata.files_tagged_at",
            "classification": {
                "status_class": list(_STATUS_CLASSES),
                "reconciliation": list(_RECONCILIATIONS),
                "wipe_signature": list(_WIPE_SIGNATURES),
            },
            "consumers": ["dark_factory 3113 P4a", "dark_factory 3427"],
            "regen_command": " ".join(
                [_REGEN_COMMAND_HEAD, *(f"--project-root {root}" for root in roots)]
            ),
        },
        "projects": projects,
        "records": [_record_to_dict(record) for record in records],
        "coverage": {
            "projects_swept": len(ordered),
            "projects_without_event_log": [
                census.project_id for census in ordered if not census.coverage.event_log_read
            ],
            "projects_skipped_unreadable": list(unreadable),
            "total_tasks": sum(census.coverage.total_tasks for census in ordered),
            "stamped_records": sum(census.coverage.stamped_records for census in ordered),
        },
    }


# ---------------------------------------------------------------------------
# The artifact pair.
#
# Resolved __file__-relatively, never as a hardcoded absolute path, so a copy
# of this script running from a task worktree writes into ITS OWN tree rather
# than the main checkout. Same reasoning and same form as
# repair_wiped_metadata_files.py:79-82 (tasks 2881/2882).
# ---------------------------------------------------------------------------

_PLANS_DIR = Path(__file__).resolve().parent.parent / "plans"
DEFAULT_JSON_OUT = _PLANS_DIR / "module-tagger-debris-census.json"
DEFAULT_MD_OUT = _PLANS_DIR / "module-tagger-debris-census.md"

# The markdown records table is CAPPED; the JSON is not. The cap is a render
# limit only, and the markdown says so at the top so a reader can never mistake
# a truncated table for the whole population.
_MARKDOWN_RECORD_CAP = 60


def render_markdown(report: dict) -> str:
    """Render the readable twin of *report*. PURE — no I/O, no clock.

    Leads with the per-project counts an operator actually reads, then the
    live-victim cells, then a capped sample of records, and closes with the
    regeneration command. Deterministic: the same report renders byte-identical
    every time, which is what keeps a re-run's `git diff` clean.
    """
    projects = report["projects"]
    coverage = report["coverage"]
    params = report["params"]

    lines: list[str] = [
        "# Tagger-debris census",
        "",
        "Every task record still carrying `metadata.files_tagged_at` — the stamp the",
        "retired module tagger left behind — across all six project corpora, classified",
        "on three axes for the repair pipeline.",
        "",
        f"Consumers: {', '.join(params['consumers'])}.",
        "",
        "**`module-tagger-debris-census.json` is the complete record.** This markdown is",
        f"its readable twin and caps the record table at {_MARKDOWN_RECORD_CAP} rows; the JSON",
        "never truncates. Neither file carries a generation timestamp, deliberately, so",
        "re-running the command below and diffing is a meaningful reproducibility check.",
        "",
        "## Classification vocabulary",
        "",
        f"- **status_class** — `{STATUS_TERMINAL}` (status in {{done, cancelled}}) vs `{STATUS_NON_TERMINAL}`.",
        f"- **reconciliation** — `{RECONCILED}` if a genuine plan-derived assertion",
        "  (a `set_to_plan` or `phase_skipped` event) postdates the stamp, meaning the",
        f"  tagger's guess was superseded; `{LOCK_RECONCILED}` if only a `lock_acquired`",
        "  event does, and that lock named at least one module the record's own",
        f"  `metadata.files` cannot explain; `{NEVER_RECONCILED}` if neither does, meaning",
        "  the guess is still this record's live scope. Read the caveat below before",
        f"  treating `{LOCK_RECONCILED}` as repaired.",
        f"- **wipe_signature** — `{POST_WIPE_OVERWRITE}` if an authoritative scope event predates",
        f"  the stamp (the tagger stamped over it); `{NO_PRIOR_SCOPE}` otherwise.",
        "- **merge_signature** — the audit's own `merge_finalized` verdict",
        "  (`audit_wiped_metadata_files.classify_wipe_signature`), carried as correlating",
        "  evidence in the vocabulary both consumers already speak.",
        "",
        f"### Why `{LOCK_RECONCILED}` is a weaker signal than `{RECONCILED}`",
        "",
        "A `lock_acquired` event's module set is **derived from `metadata.files`** by the",
        "scheduler — `Scheduler._get_modules` computes it as",
        "`derive_modules(metadata['files'], depth)` — so a lock is",
        "**not an independent scope derivation**. For a record still carrying the tagger's",
        "guess, the lock is an ECHO of that guess and proves nothing about it.",
        "",
        "This census therefore discounts any post-stamp lock whose modules are fully",
        "explained by the record's own `metadata.files`, and reports the rest as",
        f"`{LOCK_RECONCILED}` rather than folding them into `{RECONCILED}`.",
        "",
        f"A record counted `{LOCK_RECONCILED}` **may still be carrying the tagger's guess**",
        "as its live scope: all that is known is that some lock named a module the guess",
        f"cannot account for. Only `{RECONCILED}` reflects a genuine plan-derived assertion.",
        f"A consumer must **decide for itself** whether to treat `{LOCK_RECONCILED}` records",
        "as repaired — the class is reported separately precisely so that choice is",
        "available rather than made here.",
        "",
        "## Per-project counts",
        "",
        "| project | total tasks | stamped | terminal | non-terminal | plan reconciled "
        "| lock reconciled | never reconciled | post-wipe overwrite | event log |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for project_id in sorted(projects):
        block = projects[project_id]
        lines.append(
            f"| {project_id} | {block['total_tasks']} | {block['stamped_records']} "
            f"| {block['status_class'][STATUS_TERMINAL]} "
            f"| {block['status_class'][STATUS_NON_TERMINAL]} "
            f"| {block['reconciliation'][RECONCILED]} "
            f"| {block['reconciliation'][LOCK_RECONCILED]} "
            f"| {block['reconciliation'][NEVER_RECONCILED]} "
            f"| {block['wipe_signature'][POST_WIPE_OVERWRITE]} "
            f"| {'read' if block['event_log_read'] else 'UNREADABLE'} |"
        )

    lines += [
        "",
        "## Three-axis cells",
        "",
        "Every cell is emitted even at zero: a missing count must never be readable as",
        "a zero. The strict live-victim cell for the repair pipeline is",
        f"`{STATUS_NON_TERMINAL}|{NEVER_RECONCILED}|{POST_WIPE_OVERWRITE}` — live work whose",
        "scope was overwritten and never superseded.",
        "",
        f"`{STATUS_NON_TERMINAL}|{LOCK_RECONCILED}|{POST_WIPE_OVERWRITE}` is the SECOND cell a",
        "repair must consider: those records may still be carrying the guess (see the",
        "caveat above). Whether they belong in the population is the consumer's call —",
        "which is why the two cells are counted apart rather than merged.",
        "",
        "| project | cell | count |",
        "| --- | --- | ---: |",
    ]
    for project_id in sorted(projects):
        for cell in sorted(projects[project_id]["cells"]):
            lines.append(f"| {project_id} | `{cell}` | {projects[project_id]['cells'][cell]} |")

    lines += [
        "",
        "## Coverage",
        "",
        f"- projects swept: {coverage['projects_swept']}",
        f"- tasks examined: {coverage['total_tasks']}",
        f"- stamped records: {coverage['stamped_records']}",
    ]
    unreadable = coverage["projects_without_event_log"]
    if unreadable:
        lines += [
            f"- **event log UNREADABLE for: {', '.join(unreadable)}**. For those projects the",
            "  reconciliation and wipe_signature axes are UNKNOWN, not measured clean — every",
            f"  record there is reported as `{NEVER_RECONCILED}`/`{NO_PRIOR_SCOPE}` because no",
            "  scope event could be read, not because none exists.",
        ]
    else:
        lines.append("- event log read for every swept project (no coverage shortfall)")

    records = report["records"]
    lines += [
        "",
        f"## Records (showing {min(len(records), _MARKDOWN_RECORD_CAP)} of {len(records)})",
        "",
        "| project | task | status | status_class | reconciliation | wipe_signature | merge_signature | files_tagged_at |",
        "| --- | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for record in records[:_MARKDOWN_RECORD_CAP]:
        lines.append(
            f"| {record['project_id']} | {record['task_id']} | {record['status']} "
            f"| {record['status_class']} | {record['reconciliation']} "
            f"| {record['wipe_signature']} | {record['merge_signature']} "
            f"| {record['files_tagged_at']} |"
        )

    lines += [
        "",
        "## Regenerate",
        "",
        "```",
        params["regen_command"],
        "```",
        "",
        "Read-only: every corpus connection is a `mode=ro` SQLite URI.",
    ]
    return "\n".join(lines) + "\n"


def _atomic_write_text(path: Path, text: str) -> None:
    """Write *text* to *path* via a same-directory tempfile plus os.replace.

    Mirrors bake_off_storage_shape._atomic_write_text:2590-2606. A reader can
    never observe a half-written artifact, and a failed write leaves the
    previous file intact rather than truncated.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(suffix=".tmp", prefix=f"{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def write_artifacts(report: dict, json_path: Path, md_path: Path) -> tuple[Path, Path]:
    """Write the JSON and its markdown twin, atomically and in that order.

    THE MARKDOWN IS RENDERED BEFORE EITHER DESTINATION IS TOUCHED. If rendering
    raises, both existing files survive byte-for-byte, so a stale .md can never
    accompany a fresh .json. Same property, same shape, as
    bake_off_storage_shape.write_artifacts.

    ``sort_keys=False`` is deliberate: key ORDER carries meaning here, with
    schema_version leading so a reader sees the version before anything it
    would have to interpret under that version.
    """
    markdown = render_markdown(report)
    _atomic_write_text(json_path, json.dumps(report, indent=2, sort_keys=False) + "\n")
    _atomic_write_text(md_path, markdown)
    return json_path, md_path


# ---------------------------------------------------------------------------
# CLI.
#
# WHY THIS SCRIPT KEEPS ITS OWN EXIT LADDER instead of routing through
# _task_db_scan.run_audit_cli, whose Tier-3 skeleton it otherwise adopts.
#
# run_audit_cli returns AUDIT_EXIT_FINDINGS=1 whenever its is_dirty predicate
# fires, meaning "the read-only sweep found something dirty". THE CENSUS ALWAYS
# FINDS RECORDS — 507 measured across the six corpora at plan time — so routing
# through it would make task 4525's mandated user-observable signal,
# "re-running the script reproduces the counts (exit 0)", structurally
# unreachable. That is the identical exit-1 SEMANTIC collision
# _task_db_scan.py:66-72 already records as one of the four reasons
# repair_wiped_metadata_files.py keeps its own ladder. run_audit_cli
# additionally prints its rendered report unconditionally and has nowhere to
# write artifacts.
#
# sweep_project_roots carries none of that and IS adopted: it is a pure
# synchronous warn-and-continue loop whose one-result-per-root-or-raise
# contract census_project satisfies exactly.
#
# 0/2/3 stay in NUMERIC LOCKSTEP with AUDIT_EXIT_*, enforced by a test that
# imports those constants, so renumbering either copy fails CI instead of
# drifting silently. 1 deliberately diverges in MEANING, as described above.
# ---------------------------------------------------------------------------

EXIT_OK = 0                  # swept; artifacts written (or --check agreed)
EXIT_STALE = 1               # --check: the committed artifact does not match a fresh sweep
EXIT_NO_ROOT = 2             # no project root resolved to a readable tasks.db
EXIT_NOTHING_SCANNED = 3     # roots resolved but EVERY one failed to read

_EPILOG = """\
Read-only: every corpus connection is a mode=ro SQLite URI, so this script is
structurally incapable of mutating a task record, an event record or a plan.

Exit codes: 0 swept ok; 1 --check found the artifact stale; 2 no project root
resolved to a readable tasks.db; 3 roots resolved but every one was unreadable
(NOT a clean run - nothing was examined, and no artifact is written).

--check WARNING: it compares the committed artifact against a LIVE, continuously
drifting corpus. Six orchestrators mutate these databases, and a stamped task
becoming plan-reconciled is a NORMAL event, not a defect. Never wire --check
into a verify or CI gate: it would go red on ordinary progress.
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Census every task record carrying metadata.files_tagged_at.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--project-root", dest="project_roots", action="append",
        help="Project root to sweep; repeatable. Defaults to the discovered roots.",
    )
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT), help="JSON artifact path.")
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT), help="Markdown artifact path.")
    parser.add_argument(
        "--check", action="store_true",
        help="Compare against the existing artifact WITHOUT writing. See the --check warning above.",
    )
    parser.add_argument(
        "--json", action="store_true", dest="json_stdout",
        help="Print the report to stdout instead of writing the artifacts.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Sweep the roots, classify every stamp, and publish the artifact pair."""
    args = _build_parser().parse_args(argv)

    roots = discover_project_roots(args.project_roots)
    if not roots:
        print(NO_PROJECT_ROOT_RESOLVED_MESSAGE, file=sys.stderr)
        return EXIT_NO_ROOT

    censuses, unreadable = sweep_project_roots(roots, census_project)
    if not censuses:
        # Every root failed. NOT a clean run: nothing was examined, so no
        # artifact is written — replacing real findings with an empty file
        # would record "no debris" as a measurement nobody made.
        print(
            f"error: {len(unreadable)} project root(s) resolved but NONE could be "
            "read; nothing was examined. This is NOT a clean run and no artifact "
            "was written.",
            file=sys.stderr,
        )
        return EXIT_NOTHING_SCANNED

    report = build_report(censuses, unreadable=unreadable)

    if args.json_stdout:
        print(json.dumps(report, indent=2, sort_keys=False))
        return EXIT_OK

    json_path = Path(args.json_out)
    md_path = Path(args.md_out)

    if args.check:
        expected = json.dumps(report, indent=2, sort_keys=False) + "\n"
        if not json_path.exists():
            print(f"stale: {json_path} does not exist", file=sys.stderr)
            return EXIT_STALE
        actual = json_path.read_text(encoding="utf-8")
        if actual != expected:
            print(
                f"stale: {json_path} does not match a fresh sweep "
                f"({len(actual)} bytes on disk vs {len(expected)} freshly rendered). "
                "The corpus drifts continuously; regenerate rather than treating "
                "this as a defect.",
                file=sys.stderr,
            )
            return EXIT_STALE
        return EXIT_OK

    write_artifacts(report, json_path, md_path)
    print(
        f"wrote {json_path} and {md_path}: "
        f"{report['coverage']['stamped_records']} stamped record(s) across "
        f"{report['coverage']['projects_swept']} project(s)"
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
