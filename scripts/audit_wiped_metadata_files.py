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
from typing import NamedTuple


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
