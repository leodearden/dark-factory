#!/usr/bin/env python3
"""Audit the blast radius of the DONE-path ``metadata.files`` wipe.

READ-ONLY / REPORT-ONLY: this module and its CLI never mutate a task record,
an event record, or a plan artifact. Every database connection it opens is a
read-only SQLite URI (``sqlite3.connect(f"file:{path}?mode=ro", uri=True)``),
so the sweep is structurally incapable of writing to the live 128MB WAL
databases the running orchestrator holds open. Plan files on disk are only
ever read. REMEDIATION IS A SEPARATE, REVIEWED FOLLOW-UP — backfilling
``metadata.files`` from this report is never done by this script.

Background (task 3146): ``TaskWorkflow._reconcile_metadata_files_for_done``
(orchestrator/src/orchestrator/workflow.py:2001-2021) contains an
``elif self._merge_sha: ... else: files = []`` ladder. A task that reaches
the DONE path without a merge sha therefore has its ``metadata.files``
BLANKED rather than left alone. This script enumerates every task whose plan
declared a non-empty file scope but whose ``metadata.files`` is now empty.

It deliberately reports an OBSERVABLE SUBSET, never "the damaged population":
most tasks have no recoverable plan.files at all, so the report always
carries a COVERAGE block naming how many tasks had no plan signal whatsoever.
Presenting a partial scan as complete would be exactly the
no-silent-fail-soft violation in docs/legibility/design-invariants.md.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import NamedTuple

# Tier 1 (tasks.db discovery) and Tier 3 (audit-script CLI plumbing: the roots
# loop, the warn-and-continue skip, the exit-code ladder and the two reporting
# layout primitives), shared with audit_combine_gate_marker_loss.py as of task
# 3616. Tier 2 is the LEAK-SCANNER skeleton and does not apply here — it sweeps
# db paths and accumulates matches, not one audit per project root.
#
# This module deliberately keeps its own format_report/format_json/_build_parser:
# its --min-fidelity filter, its object-shaped JSON, its epilog wording and its
# COVERAGE caveat/labels are genuinely different behaviour, not duplication.
# Exit 3 is NOT among those differences any more — it is Tier 3's, shared.
#
# The noqa below sits on the ONE name that needs it rather than on the
# statement header: a header noqa blankets every name in the block, so a name
# that later went dead here would be invisible to lint. Only
# discover_project_roots is unused BY THIS MODULE — it is re-exported because
# scripts/tests/test_audit_wiped_metadata_files.py imports it from here.
from _task_db_scan import (
    discover_project_roots,  # noqa: F401  (re-exported for this script's tests)
    format_coverage_block,
    format_kv_line,
    run_audit_cli,
    tasks_db_path,
)

# The canonical meta-root directory name, per config.py:1343 TASK_META_DIRNAME
# and artifacts.py:287 meta_root_for. It is a SIBLING of the worktrees inside
# the worktree base, not a child of any worktree.
TASK_META_DIRNAME = ".task-meta"


class TaskRecord(NamedTuple):
    """One task's audit-relevant state, as stored in tasks.db.

    ``metadata_files`` is the CURRENT ``metadata.files`` list — an empty
    tuple is the wipe signature this audit looks for (in combination with a
    non-empty recovered plan scope).
    """

    tag: str
    task_id: int
    status: str
    metadata_files: tuple[str, ...]


def _coerce_file_list(value: object) -> tuple[str, ...]:
    """Coerce a raw JSON ``files`` value into a tuple of non-empty paths.

    Anything that is not a list (None, a bare string, a dict, a number)
    degrades to an empty tuple rather than raising — a wrong-typed ``files``
    is corrupt data to be reported as "no scope", not a reason to abort a
    3000-row sweep. Non-string and empty entries inside an otherwise-valid
    list are dropped so downstream string formatting can never blow up on a
    stray ``None``.
    """
    if not isinstance(value, list):
        return ()
    return tuple(entry for entry in value if isinstance(entry, str) and entry)


def _decode_files(raw: object, key: str = "files") -> tuple[str, ...]:
    """Decode a JSON blob and pull *key* out of it as a file tuple.

    Degrades to an empty tuple for NULL, malformed JSON, or a payload that
    decodes to anything other than a dict.
    """
    if not raw or not isinstance(raw, (str, bytes)):
        return ()
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        return ()
    if not isinstance(payload, dict):
        return ()
    return _coerce_file_list(payload.get(key))


# ---------------------------------------------------------------------------
# Recovered plan scope.
#
# FIDELITY IS LOAD-BEARING, NOT DECORATION. A FILE_LEVEL record is a faithful
# plan.files list and could in principle be backfilled verbatim. A LOCK_LEVEL
# record is the lock-level MODULE set and must NEVER be presented as if it
# were plan.files — writing a module path into metadata.files would corrupt
# any downstream repair. The label makes that distinction machine-checkable
# instead of relying on a reader knowing the emit-site history.
# ---------------------------------------------------------------------------

FIDELITY_FILE_LEVEL = "file_level"
FIDELITY_LOCK_LEVEL = "lock_level"


class PlanFilesRecord(NamedTuple):
    """A task's recovered plan scope, with where it came from and how faithful.

    ``fidelity`` is one of :data:`FIDELITY_FILE_LEVEL` (a real plan.files
    list) or :data:`FIDELITY_LOCK_LEVEL` (a lock-level module projection).
    """

    files: tuple[str, ...]
    source: str
    fidelity: str


# Event types carrying a recoverable plan scope, mapped to
# (payload key, source label, fidelity).
#
#   phase_skipped.plan_files  (workflow.py:4275, :4447) — TRUE file-level
#       plan.files, snapshotted at the moment revalidation/SIMPLE_TASK
#       planning skipped a phase.
#   set_to_plan.files  (scheduler.py:6987-6994) — DELIBERATELY LOCK-LEVEL.
#       event_store.py:77-82 and scheduler.py:6982-6984 both state that this
#       payload carries `needed`, the lock-level module set, and NOT the
#       file-level persist set — the emit site keeps it that way on purpose
#       to preserve the reify zeta-gate contract. It therefore reaches more
#       tasks than any file-level source but its paths are MODULE paths.
#       Tagged LOCK_LEVEL so it can only ever be used as a "this task
#       declared a non-empty scope" presence signal.
_EVENT_PLAN_SOURCES = {
    "phase_skipped": ("plan_files", "phase_skipped_event", FIDELITY_FILE_LEVEL),
    "set_to_plan": ("files", "set_to_plan_event", FIDELITY_LOCK_LEVEL),
}

# Higher rank wins. Compared before recency, so a file-level record beats a
# lock-level one regardless of which event row is newer.
_FIDELITY_RANK = {FIDELITY_LOCK_LEVEL: 0, FIDELITY_FILE_LEVEL: 1}


# ---------------------------------------------------------------------------
# Wipe signatures.
#
# WHY THIS IS STATE-AWARE AND PER-TASK rather than the naive per-EVENT
# "merge_finalized with merge_sha=null". Measured read-only over all 1622
# merge_finalized events in the live runs.db, grouped by (state, sha is null):
#
#     done            1159    0 null
#     blocked          382  365 null
#     already_merged    21   21 null
#     superseded        19   19 null
#     unknown_branch    17   17 null
#     conflict          14   14 null
#     abandoned          7    7 null
#     wip_halted         3    3 null
#
# A null merge_sha is OVERWHELMINGLY a merge that FAILED (blocked/conflict)
# and was later retried successfully — it never took the workflow to DONE, so
# it never called _reconcile_metadata_files_for_done with _merge_sha=None and
# never wiped anything. A bare merge_sha-is-null test therefore over-reports
# by roughly 20x and would drown the real population.
#
# state='done' carries a non-null sha in 1159/1159 cases, so the "done + null
# sha" event the task description posits does not occur in this DB; the actual
# DONE-with-no-sha shortcut is state='already_merged' (21 events, all
# null-sha). 'done' is nonetheless kept in the DONE-reaching set below: if the
# shape ever does occur it IS the wipe, and excluding it would silently miss
# the very case the audit exists to find.
# ---------------------------------------------------------------------------

# At least one merge_finalized reached a DONE state carrying no merge sha —
# the wipe as such.
CONFIRMED_NULL_SHA_DONE_PATH = "confirmed_null_sha_done_path"
# merge_finalized events exist but NONE ever carried a non-null merge sha.
NO_SUCCESSFUL_MERGE_SHA = "no_successful_merge_sha"
# A null-sha row exists AND the same task also obtained a REAL merge sha — a
# failed attempt that was retried and landed. NOT the null-sha wipe. The claim
# is COEXISTENCE, not ordering: emission order is not relied on, so the label
# stays true whichever row came first.
CONTRADICTED_REAL_MERGE_SHA = "contradicted_real_merge_sha"
# Every merge_finalized row carried a real merge sha; there is NO null-sha row
# at all. Kept DISTINCT from CONTRADICTED_REAL_MERGE_SHA because the latter
# asserts a null-sha row exists, which would be factually false here.
#
# NOT an exoneration: _reconcile_metadata_files_for_done also blanks files on
# its `if err is not None: files = []` branch (workflow.py:2001-2006), which
# runs WITH a real merge sha present. A clean-sha task whose metadata.files is
# empty is therefore still a live wipe candidate and is reported as one — it is
# never filed under "contradicted", which would tell an operator not to repair
# it.
CLEAN_MERGE_SHA = "clean_merge_sha"
# No merge_finalized at all. UNKNOWN, not clean: found_on_main recovery and
# eval mode both reach DONE without emitting one.
NO_MERGE_EVENT = "no_merge_event"

# merge_finalized states that mean the workflow PROCEEDED TO DONE. A null sha
# on one of these is the wipe; a null sha on any other state (blocked,
# conflict, superseded, ...) is a failed attempt.
_DONE_REACHING_STATES = frozenset({"done", "already_merged"})


def classify_wipe_signature(merge_finalized_payloads: list) -> str:
    """Classify one task's merge history into a wipe signature.

    *merge_finalized_payloads* is that task's decoded ``merge_finalized``
    payloads (merge_queue.py:3830-3844) in emission order. Non-dict entries
    are ignored rather than raising, so one corrupt row cannot abort a sweep
    over thousands of tasks.

    Returns one of :data:`CONFIRMED_NULL_SHA_DONE_PATH`,
    :data:`NO_SUCCESSFUL_MERGE_SHA`, :data:`CONTRADICTED_REAL_MERGE_SHA`,
    :data:`CLEAN_MERGE_SHA`, or :data:`NO_MERGE_EVENT`. See the block comment
    above for the live-DB measurement that justifies the state-aware rule.

    Null-sha and real-sha observations are tracked SEPARATELY so a history
    that only ever carried real shas is labelled :data:`CLEAN_MERGE_SHA`
    rather than "contradicted" — the latter asserts a null-sha row exists,
    and mislabelling would tell an operator not to repair a task the err
    branch (workflow.py:2001-2006) may genuinely have wiped.
    """
    saw_event = False
    saw_real_sha = False
    saw_null_sha = False
    for payload in merge_finalized_payloads:
        if not isinstance(payload, dict):
            continue
        saw_event = True
        state = payload.get("state")
        merge_sha = payload.get("merge_sha")
        if not merge_sha:
            # A DONE-reaching state with no sha IS the wipe, and outranks
            # every other observation for this task.
            if state in _DONE_REACHING_STATES:
                return CONFIRMED_NULL_SHA_DONE_PATH
            saw_null_sha = True
            continue
        saw_real_sha = True

    if not saw_event:
        return NO_MERGE_EVENT
    if saw_real_sha:
        return CONTRADICTED_REAL_MERGE_SHA if saw_null_sha else CLEAN_MERGE_SHA
    return NO_SUCCESSFUL_MERGE_SHA


def load_merge_signatures(runs_db_path: str) -> dict[str, list[dict]]:
    """Group every ``merge_finalized`` payload in *runs_db_path* by task id.

    One read-only pass across ALL runs (no run_id filter — a task's merge
    history spans retries in different runs), in ascending ``id`` order so
    each task's list is in emission order. Rows with a NULL ``task_id`` or a
    malformed/NULL/non-dict payload are skipped.
    """
    signatures: dict[str, list[dict]] = {}
    conn = sqlite3.connect(f"file:{runs_db_path}?mode=ro", uri=True)
    try:
        cursor = conn.execute(
            "SELECT id, task_id, data FROM events "
            "WHERE event_type = 'merge_finalized' ORDER BY id"
        )
        for _event_id, task_id, data in cursor:
            if not task_id or not data:
                continue
            try:
                payload = json.loads(data)
            except (ValueError, TypeError):
                continue
            if not isinstance(payload, dict):
                continue
            signatures.setdefault(str(task_id), []).append(payload)
    finally:
        conn.close()
    return signatures


# The ONE statement of source precedence, most authoritative first. Every
# source label the loaders can emit appears here, so a merge is always total.
#
#   meta_root_plan_json      — the canonical persisted plan artifact.
#   legacy_worktree_plan_json — the same artifact at its pre-relocation path.
#   phase_skipped_event      — a durable EVENT SNAPSHOT of plan.files; equally
#                              file-level, but a snapshot taken at one moment
#                              rather than the plan as finally persisted, so
#                              it loses to either disk artifact.
#   set_to_plan_event        — lock-level module projection; last resort.
_SOURCE_PRECEDENCE = (
    "meta_root_plan_json",
    "legacy_worktree_plan_json",
    "phase_skipped_event",
    "set_to_plan_event",
)


def _source_rank(record: PlanFilesRecord) -> int:
    """Rank *record* by source; lower is more authoritative.

    An unrecognised source sorts last rather than raising, so adding a new
    loader can never make the merge crash — it just ranks below every known
    source until it is added to :data:`_SOURCE_PRECEDENCE`.
    """
    try:
        return _SOURCE_PRECEDENCE.index(record.source)
    except ValueError:
        return len(_SOURCE_PRECEDENCE)


def merge_plan_file_sources(
    disk_records: dict[str, PlanFilesRecord],
    event_records: dict[str, PlanFilesRecord],
) -> dict[str, PlanFilesRecord]:
    """Merge the disk-borne and event-borne plan scopes by source precedence.

    Returns a FRESH dict covering the union of both inputs' task ids; neither
    argument is mutated. Where both carry a record for one task, the one whose
    source ranks higher in :data:`_SOURCE_PRECEDENCE` wins.
    """
    merged: dict[str, PlanFilesRecord] = {}
    for records in (disk_records, event_records):
        for task_id, record in records.items():
            existing = merged.get(task_id)
            if existing is None or _source_rank(record) < _source_rank(existing):
                merged[task_id] = record
    return merged


def _read_plan_file(plan_path: Path, dir_name: str) -> tuple[str, tuple[str, ...]] | None:
    """Read one plan.json, returning ``(task_id_key, files)`` or None.

    The key is the plan's OWN ``task_id`` field when present, falling back to
    *dir_name* — the caller-supplied lane/meta-root entry name. The caller must
    pass it explicitly because the two layouts nest the plan at different
    depths (``<base>/.task-meta/<name>/plan.json`` vs
    ``<base>/<name>/.task/plan.json``), so deriving it from ``plan_path``
    would yield the literal ``.task`` for every legacy plan and collapse them
    all onto one key. This is the same self-identification
    ``_find_lane_by_plan_task_id`` relies on (git_ops.py:6793-6845), which is
    robust to a plan sitting in a pooled lane directory whose name is not the
    task id.

    Returns None for anything unusable: a malformed/unreadable file, a
    dangling symlink, a plan.json that is a directory, a payload that is not
    a dict, or a missing/empty/wrong-typed ``files`` list. Every failure is
    a skip, never an exception — one corrupt plan among hundreds must not
    abort the sweep.
    """
    try:
        payload = json.loads(plan_path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    files = _coerce_file_list(payload.get("files"))
    if not files:
        return None
    raw_task_id = payload.get("task_id")
    if raw_task_id is None or raw_task_id == "":
        key = dir_name
    else:
        key = str(raw_task_id)
    return key, files


def load_plan_files_from_disk(worktree_base: str) -> dict[str, PlanFilesRecord]:
    """Recover plan scope from BOTH on-disk plan locations under *worktree_base*.

    New-then-old, the same precedence git_ops.py:6833-6835 and
    harness.py:2721-2735 already use:

      1. ``<base>/.task-meta/<name>/plan.json`` — the CANONICAL artifact
         (config.py:1343, artifacts.py:287). ``cleanup_worktree``
         (git_ops.py:11216-11310) removes the worktree but never the
         meta-root, and crash recovery explicitly skips ``.task-meta``
         (harness.py:2958-2975), so these plans outlive their worktrees.
      2. ``<base>/<name>/plan.json`` under ``.task/`` — the LEGACY location.
         In the current layout this path is only an absolute SYMLINK to (1)
         (artifacts.py:354-386), so it usually resolves to a task already
         recorded from the meta-root and is dropped as a lower-precedence
         duplicate. Pre-relocation plans, however, exist ONLY here, so it
         cannot be skipped.

    A missing *worktree_base* returns an empty dict.
    """
    records: dict[str, PlanFilesRecord] = {}
    base = Path(worktree_base)

    def _ingest(plan_path: Path, dir_name: str, source: str) -> None:
        read = _read_plan_file(plan_path, dir_name)
        if read is None:
            return
        key, files = read
        # New-then-old: a meta-root hit is never overwritten by the legacy
        # scan (which, in the symlink layout, is the very same artifact).
        if key in records:
            return
        records[key] = PlanFilesRecord(
            files=files, source=source, fidelity=FIDELITY_FILE_LEVEL
        )

    meta_root = base / TASK_META_DIRNAME
    try:
        meta_entries = sorted(meta_root.iterdir())
    except OSError:
        meta_entries = []
    for entry in meta_entries:
        _ingest(entry / "plan.json", entry.name, "meta_root_plan_json")

    try:
        base_entries = sorted(base.iterdir())
    except OSError:
        base_entries = []
    for entry in base_entries:
        # The meta-root is a sibling of the worktrees, not a worktree — a
        # <base>/.task-meta/.task/plan.json must not be mis-scanned as a lane.
        if entry.name == TASK_META_DIRNAME:
            continue
        _ingest(entry / ".task" / "plan.json", entry.name, "legacy_worktree_plan_json")

    return records


def load_plan_files_from_events(runs_db_path: str) -> dict[str, PlanFilesRecord]:
    """Recover plan scope per task from durable event payloads in *runs_db_path*.

    One read-only pass in ascending ``id`` order. Per task, keeps the
    highest-FIDELITY record, breaking ties by recency (later row wins) — so a
    file-level ``phase_skipped`` snapshot always beats a lock-level
    ``set_to_plan`` one no matter which was emitted last.

    Rows with a NULL ``task_id``, malformed/NULL ``data``, a missing key, or
    an empty/wrong-typed file list are SKIPPED — never allowed to overwrite an
    earlier real signal with an empty one. Keys are event ``task_id`` values,
    which are TEXT in the live schema.
    """
    records: dict[str, PlanFilesRecord] = {}
    placeholders = ", ".join("?" for _ in _EVENT_PLAN_SOURCES)
    conn = sqlite3.connect(f"file:{runs_db_path}?mode=ro", uri=True)
    try:
        cursor = conn.execute(
            "SELECT id, task_id, event_type, data FROM events "
            f"WHERE event_type IN ({placeholders}) ORDER BY id",
            tuple(_EVENT_PLAN_SOURCES),
        )
        for _event_id, task_id, event_type, data in cursor:
            if not task_id:
                continue
            key, source, fidelity = _EVENT_PLAN_SOURCES[event_type]
            files = _decode_files(data, key)
            if not files:
                continue
            existing = records.get(str(task_id))
            if existing is not None and (
                _FIDELITY_RANK[existing.fidelity] > _FIDELITY_RANK[fidelity]
            ):
                continue
            records[str(task_id)] = PlanFilesRecord(
                files=files, source=source, fidelity=fidelity
            )
    finally:
        conn.close()
    return records


def load_task_records(tasks_db_path: str) -> dict[tuple[str, int], TaskRecord]:
    """Load every task from *tasks_db_path*, keyed by ``(tag, id)``.

    Keyed by the full ``(tag, id)`` primary key — the live DB uses a single
    ``master`` tag, but the schema permits the same numeric id under two tags
    and collapsing them would silently merge two distinct tasks.

    Opens the database via a read-only URI (``mode=ro``) so the load is
    structurally incapable of mutating live task records even while
    fused-memory holds the same file open in WAL mode.
    """
    records: dict[tuple[str, int], TaskRecord] = {}
    conn = sqlite3.connect(f"file:{tasks_db_path}?mode=ro", uri=True)
    try:
        cursor = conn.execute("SELECT tag, id, status, metadata FROM tasks")
        for tag, task_id, status, metadata in cursor:
            records[(tag, task_id)] = TaskRecord(
                tag=tag,
                task_id=task_id,
                status=status,
                metadata_files=_decode_files(metadata),
            )
    finally:
        conn.close()
    return records


# ---------------------------------------------------------------------------
# The audit itself.
# ---------------------------------------------------------------------------


class WipeCandidate(NamedTuple):
    """One task whose plan declared a scope but whose metadata.files is empty.

    ``plan_files_fidelity`` decides whether ``plan_files`` may be consumed
    verbatim: only :data:`FIDELITY_FILE_LEVEL` entries are real plan.files.
    A :data:`FIDELITY_LOCK_LEVEL` entry holds MODULE paths and must never be
    written into metadata.files as-is.
    """

    tag: str
    task_id: int
    status: str
    plan_files: tuple[str, ...]
    plan_files_source: str
    plan_files_fidelity: str
    wipe_signature: str


class AuditCoverage(NamedTuple):
    """How much of the task population the audit could actually see.

    ALWAYS reported, including when zero candidates are found. Most tasks
    have no recoverable plan.files at all, so the candidate list is an
    OBSERVABLE SUBSET, never the damaged population. Presenting it as
    complete would be a no-silent-fail-soft violation
    (docs/legibility/design-invariants.md).

    The three signal tiers are counted over TASKS (not over plan records) and
    partition ``total_tasks`` exactly:
    ``file_level + lock_level_only + without_plan_signal == total_tasks``.
    """

    project_root: str
    total_tasks: int
    tasks_with_file_level_signal: int
    tasks_with_lock_level_signal_only: int
    tasks_without_plan_signal: int
    plan_records_without_task: int


class ProjectAudit(NamedTuple):
    """One project's audit result: what was found, and what could be seen."""

    project_root: str
    candidates: list[WipeCandidate]
    coverage: AuditCoverage


def runs_db_path(project_root: str) -> Path:
    """``<root>/data/orchestrator/runs.db`` (harness.py:1877; no config key).

    NOTE: runs.db lives in the MAIN checkout, not in a task worktree.
    """
    return Path(project_root) / "data" / "orchestrator" / "runs.db"


def worktree_base_path(project_root: str) -> Path:
    """``<root>/.worktrees`` — the worktree base (config.py:1410 worktree_dir)."""
    return Path(project_root) / ".worktrees"


def _candidate_sort_key(candidate: WipeCandidate) -> tuple[str, int, str]:
    """Sort by (tag, NUMERIC task id) so 100 follows 20 rather than preceding it."""
    try:
        return (candidate.tag, int(candidate.task_id), "")
    except (TypeError, ValueError):
        return (candidate.tag, 0, str(candidate.task_id))


def audit_project(project_root: str) -> ProjectAudit:
    """Audit one project root for the DONE-path ``metadata.files`` wipe.

    Reads the task store, both on-disk plan locations and the event log, all
    READ-ONLY, merges the plan sources by precedence, joins on task id, and
    reports every task that declared a non-empty plan scope but now has an
    empty ``metadata.files`` — plus the coverage block naming how much of the
    population was invisible.

    A missing runs.db is tolerated: every signature degrades to
    :data:`NO_MERGE_EVENT` rather than aborting the audit.
    """
    task_records = load_task_records(str(tasks_db_path(project_root)))

    disk_records = load_plan_files_from_disk(str(worktree_base_path(project_root)))
    runs_db = runs_db_path(project_root)
    if runs_db.exists():
        event_records = load_plan_files_from_events(str(runs_db))
        merge_signatures = load_merge_signatures(str(runs_db))
    else:
        event_records = {}
        merge_signatures = {}
    plan_records = merge_plan_file_sources(disk_records, event_records)

    # Plan records are keyed by bare task id; tasks are keyed by (tag, id).
    # The live DB uses a single 'master' tag, so match by id across all tags.
    tasks_by_id: dict[str, list[TaskRecord]] = {}
    for record in task_records.values():
        tasks_by_id.setdefault(str(record.task_id), []).append(record)

    candidates: list[WipeCandidate] = []
    file_level = 0
    lock_level_only = 0
    matched_plan_ids: set[str] = set()

    for task_id, plan in plan_records.items():
        matching = tasks_by_id.get(task_id)
        if not matching:
            # A plan/event signal whose task is absent from tasks.db (removed,
            # or a different project's id). Counted, never silently dropped.
            continue
        matched_plan_ids.add(task_id)

        for record in matching:
            # Counted PER MATCHED TASK, not per plan record. One plan record is
            # keyed by a bare task id, so when the same numeric id exists under
            # two tags it matches two DISTINCT tasks; counting it once would
            # leave the second task in the `total - file_level - lock_level`
            # remainder and report it as having "NO plan signal at all" when it
            # had one. The coverage block is the honesty artifact here, so an
            # off-by-N in it undercuts the whole report.
            if plan.fidelity == FIDELITY_FILE_LEVEL:
                file_level += 1
            else:
                lock_level_only += 1

            if record.metadata_files:
                continue  # files survived — nothing was wiped
            candidates.append(
                WipeCandidate(
                    tag=record.tag,
                    task_id=record.task_id,
                    status=record.status,
                    plan_files=plan.files,
                    plan_files_source=plan.source,
                    plan_files_fidelity=plan.fidelity,
                    wipe_signature=classify_wipe_signature(
                        merge_signatures.get(task_id, [])
                    ),
                )
            )

    candidates.sort(key=_candidate_sort_key)
    coverage = AuditCoverage(
        project_root=project_root,
        total_tasks=len(task_records),
        tasks_with_file_level_signal=file_level,
        tasks_with_lock_level_signal_only=lock_level_only,
        tasks_without_plan_signal=len(task_records) - file_level - lock_level_only,
        plan_records_without_task=len(plan_records) - len(matched_plan_ids),
    )
    return ProjectAudit(
        project_root=project_root, candidates=candidates, coverage=coverage
    )


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------

_LOCK_LEVEL_CAVEAT = (
    "    CAVEAT: the paths above are LOCK-LEVEL module paths (from "
    "set_to_plan.files), NOT verbatim plan.files entries. They evidence that "
    "a non-empty scope was declared; they must not be backfilled into "
    "metadata.files as-is."
)


_COVERAGE_CAVEAT = (
    "  COVERAGE (the candidate list above is an OBSERVABLE SUBSET, not the",
    "  full damaged population — tasks with no recoverable plan scope are",
    "  UNKNOWN, neither clean nor damaged):",
)


def _format_candidate_line(candidate: WipeCandidate) -> str:
    # signature/source/fidelity are deliberately shorter than the
    # wipe_signature/plan_files_source/plan_files_fidelity attributes they
    # carry, so the keys are spelled here, not derived from the field names.
    return format_kv_line([
        ("task_id", candidate.task_id),
        ("tag", candidate.tag),
        ("status", candidate.status),
        ("signature", candidate.wipe_signature),
        ("source", candidate.plan_files_source),
        ("fidelity", candidate.plan_files_fidelity),
        ("plan_files", len(candidate.plan_files)),
    ])


def _format_coverage(coverage: AuditCoverage) -> list[str]:
    """Render the always-printed coverage block.

    Never omitted, never abbreviated when there are no candidates: the whole
    point is that the candidate list is an observable subset, and a reader
    must be told the size of the unobservable remainder.

    Only the ALIGNMENT is shared with audit_combine_gate_marker_loss.py (via
    format_coverage_block); the caveat prose above and the labels below are
    this script's, and say what this script could not see.
    """
    return format_coverage_block(
        _COVERAGE_CAVEAT,
        [
            ("total tasks scanned:", coverage.total_tasks),
            ("with a file-level plan signal:", coverage.tasks_with_file_level_signal),
            ("with only a lock-level signal:", coverage.tasks_with_lock_level_signal_only),
            ("with NO plan signal at all:", coverage.tasks_without_plan_signal),
            ("plan records with no such task:", coverage.plan_records_without_task),
        ],
    )


def format_report(audits: list[ProjectAudit]) -> str:
    """Render *audits* as a grouped, human-readable report.

    Reportable candidates and CONTRADICTED ones are printed in SEPARATE
    sections: a contradicted candidate has BOTH a null-sha row and a real
    merge sha, i.e. a failed attempt that was retried and landed, so it is not
    the null-sha wipe and must not be consumed by a repair job alongside the
    rest. Note that :data:`CLEAN_MERGE_SHA` candidates are deliberately NOT
    contradicted — see that constant.

    The coverage block is emitted for every project, INCLUDING projects with
    zero candidates.

    The trailing summary counts BOTH figures, because ``main()`` exits 1 on
    any candidate including contradicted ones: a summary that counted only the
    reportable ones would contradict the exit code on a contradicted-only run.
    """
    lines: list[str] = []
    total_confirmed = 0
    total_contradicted = 0
    for audit in audits:
        lines.append(f"{audit.project_root}:")

        contradicted = [
            c for c in audit.candidates
            if c.wipe_signature == CONTRADICTED_REAL_MERGE_SHA
        ]
        confirmed = [
            c for c in audit.candidates
            if c.wipe_signature != CONTRADICTED_REAL_MERGE_SHA
        ]
        total_confirmed += len(confirmed)
        total_contradicted += len(contradicted)

        if confirmed:
            lines.append(f"  -- candidates ({len(confirmed)}) --")
            for candidate in confirmed:
                lines.append(_format_candidate_line(candidate))
                if candidate.plan_files_fidelity == FIDELITY_LOCK_LEVEL:
                    lines.append(_LOCK_LEVEL_CAVEAT)
        else:
            lines.append("  -- candidates (0) --")

        if contradicted:
            lines.append(
                f"  -- contradicted ({len(contradicted)}): a null-sha row and a "
                "REAL merge sha coexist for these tasks (a failed merge attempt "
                "that was retried and landed), so the null-sha DONE path was "
                "not what emptied them. NOT confirmed wipes — do not repair. --"
            )
            for candidate in contradicted:
                lines.append(_format_candidate_line(candidate))
                if candidate.plan_files_fidelity == FIDELITY_LOCK_LEVEL:
                    lines.append(_LOCK_LEVEL_CAVEAT)

        lines.extend(_format_coverage(audit.coverage))

    lines.append(
        f"{total_confirmed + total_contradicted} candidate(s) across "
        f"{len(audits)} project(s): {total_confirmed} reportable, "
        f"{total_contradicted} contradicted (not wipes)"
    )
    return "\n".join(lines)


def format_json(audits: list[ProjectAudit]) -> str:
    """Render *audits* as a JSON OBJECT carrying full file lists and coverage.

    An object rather than a bare array so the coverage counts travel with the
    candidates — a consumer that received only an array could not tell a clean
    project from an unobservable one. File lists are never truncated, so the
    output can be fed directly to a follow-up repair task.
    """
    return json.dumps(
        {
            "projects": [
                {
                    "project_root": audit.project_root,
                    "coverage": audit.coverage._asdict(),
                    "candidates": [
                        {**c._asdict(), "plan_files": list(c.plan_files)}
                        for c in audit.candidates
                    ],
                }
                for audit in audits
            ]
        }
    )


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

# --min-fidelity values, loosest last.
_MIN_FIDELITY_CHOICES = {
    "file-level": FIDELITY_FILE_LEVEL,
    "lock-level": FIDELITY_LOCK_LEVEL,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "READ-ONLY audit of the DONE-path metadata.files wipe: reports "
            "every task whose plan declared a non-empty file scope but whose "
            "metadata.files is now empty. Reporting only -- never mutates a "
            "task, event, or plan record. Remediation is a separate, "
            "individually-reviewed follow-up."
        ),
        epilog=(
            "exit codes: 0 = audited, no candidates; 1 = at least one "
            "candidate found; 2 = no project root resolved to a readable "
            "tasks.db; 3 = roots resolved but every one failed to audit, so "
            "NOTHING was scanned (never treat 3 as a clean run)."
        ),
    )
    parser.add_argument(
        "--project-root", dest="project_roots", action="append",
        help=(
            "Project root to audit (resolves <root>/.taskmaster/tasks/tasks.db, "
            "<root>/data/orchestrator/runs.db and <root>/.worktrees). "
            "May be repeated."
        ),
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit a JSON object (full untruncated file lists) instead of a report.",
    )
    parser.add_argument(
        "--min-fidelity", choices=sorted(_MIN_FIDELITY_CHOICES), default="file-level",
        help=(
            "Lowest plan-scope fidelity to report (default: %(default)s). "
            "Lock-level candidates are OPT-IN because their paths are module "
            "paths, not plan.files entries, and must never silently enter a "
            "repair feed."
        ),
    )
    return parser


def _passes_min_fidelity(candidate: WipeCandidate, min_fidelity: str) -> bool:
    if min_fidelity == "lock-level":
        return True
    return candidate.plan_files_fidelity == FIDELITY_FILE_LEVEL


def _audit_root(root: str, args: argparse.Namespace) -> ProjectAudit:
    """Audit ONE project root, applying the per-root ``--min-fidelity`` filter.

    Raises ``sqlite3.Error`` for an unreadable project, which is what
    :func:`_task_db_scan.sweep_project_roots` turns into a warn-and-skip; every
    other exception propagates. Returns exactly one audit, per that function's
    one-audit-per-root contract.
    """
    audit = audit_project(root)
    filtered = [
        c for c in audit.candidates
        if _passes_min_fidelity(c, args.min_fidelity)
    ]
    return audit._replace(candidates=filtered)


def _render(audits: list[ProjectAudit], args: argparse.Namespace) -> str:
    return format_json(audits) if args.json else format_report(audits)


def _is_dirty(audits: list[ProjectAudit]) -> bool:
    return any(a.candidates for a in audits)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Exit codes: 0 = audited, no candidates; 1 = at least one candidate found;
    2 = no project root resolved to a readable tasks.db; 3 = roots resolved
    but EVERY one of them failed to audit, so NOTHING was scanned.

    3 exists because 0 would otherwise be returned for two opposite outcomes —
    "audited everything, found nothing" and "audited nothing at all" — and a
    CI/cron consumer reading only the exit code would take a total failure for
    a clean run. That is exactly the silent fail-soft this module refuses to do
    (see the module docstring).

    A single unreadable project (a corrupt/locked tasks.db) does NOT abort the
    sweep: it is logged to stderr and skipped so every other project is still
    audited, and a trailing warning states that the results are incomplete.

    The roots loop, the warn-and-continue skip and the exit-code ladder are
    :func:`_task_db_scan.run_audit_cli` (Tier 3, task 3616), shared with
    audit_combine_gate_marker_loss.py. What stays here is what genuinely
    differs: this script's parser and epilog, its ``--min-fidelity`` per-root
    filter (:func:`_audit_root`), its object-shaped JSON and its report.
    """
    return run_audit_cli(
        argv,
        parser=_build_parser(),
        audit_fn=_audit_root,
        render=_render,
        is_dirty=_is_dirty,
    )


if __name__ == "__main__":
    sys.exit(main())
