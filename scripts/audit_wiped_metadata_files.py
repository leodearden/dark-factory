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

import json
import sqlite3
from pathlib import Path
from typing import NamedTuple

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


def _read_plan_file(plan_path: Path) -> tuple[str, tuple[str, ...]] | None:
    """Read one plan.json, returning ``(task_id_key, files)`` or None.

    The key is the plan's OWN ``task_id`` field when present, falling back to
    the caller-supplied directory name — the same self-identification
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
        key = plan_path.parent.name
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

    def _ingest(plan_path: Path, source: str) -> None:
        read = _read_plan_file(plan_path)
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
        _ingest(entry / "plan.json", "meta_root_plan_json")

    try:
        base_entries = sorted(base.iterdir())
    except OSError:
        base_entries = []
    for entry in base_entries:
        # The meta-root is a sibling of the worktrees, not a worktree — a
        # <base>/.task-meta/.task/plan.json must not be mis-scanned as a lane.
        if entry.name == TASK_META_DIRNAME:
            continue
        _ingest(entry / ".task" / "plan.json", "legacy_worktree_plan_json")

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
