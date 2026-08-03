#!/usr/bin/env python3
"""Audit the blast radius of the curator-combine ``metadata`` wipe.

READ-ONLY / REPORT-ONLY: this module and its CLI never mutate a task record,
a ticket record, or a manifest file. Every database connection it opens is a
read-only SQLite URI (``sqlite3.connect(f"file:{path}?mode=ro", uri=True)``),
so the sweep is structurally incapable of writing to the live WAL databases
the running orchestrator holds open. Manifest YAML on disk is only ever read.
There is no ``--apply`` flag and no MCP client is ever constructed.
REMEDIATION IS A SEPARATE, REVIEWED FOLLOW-UP — backfilling a lost key from
this report is never done by this script (the audit/repair split of tasks
3146 / 3329).

Background (task 3591): the curator's combine path
(fused-memory/src/fused_memory/server/task_interceptor.py:2100) writes exactly
``{'curator_action': 'combine', 'curator_justification', 'combined_at'}`` and
passes ``metadata_mode='replace'``, so every OTHER key the surviving task
carried is DROPPED rather than merged. The load-bearing casualty is
``metadata.delivered_checks``: orchestrator/src/orchestrator/delivered_checks.py
defines both ``gate_mark_done_on_delivered_checks`` and
``verify_delivered_checks_on_main`` off that key, so wiping it SILENTLY REMOVES
a mark-done gate — the task still closes, just without the check that was
supposed to hold it. ``prd_path``/``prd_task_label`` (provenance) and
``task_kind`` (dispatch) are the other consumers. This script enumerates every
task bearing the combine signature whose pre-combine metadata can be
reconstructed from an independent source, and reports the keys that are gone.

WHAT A LIVE HIT MEANS, MEASURED RATHER THAN ASSUMED. As of main tip
9cec63e10e (2026-08-03) the combine path STILL reads ``metadata_mode='replace'``
(task_interceptor.py:2130); task 3446's fix is unmerged, on branch ``task/3446``
(commits 556e720c3d, 8750bbaaaa). So a NON-TERMINAL finding from this script
means the fix HAS NOT LANDED YET — not that it regressed. Once 3446 merges,
that reading inverts and a fresh live hit becomes a regression signal.

EPISTEMIC HONESTY. Findings are ``ticket-evidenced``, never certain. The
creating ticket is a SUBMIT-TIME snapshot, so it cannot see a key legitimately
added between submit and combine; that is why this script reports key LOSS
(``expected_keys - live_keys``) and never value drift. Most combine targets
have no reconstructable pre-combine state at all, so the report always carries
a COVERAGE block naming how many targets had no comparison source whatsoever.
Presenting a partial sweep as complete would be exactly the
no-silent-fail-soft violation in docs/legibility/design-invariants.md.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import NamedTuple

# Tier 1 (tasks.db discovery) ONLY, imported as a flat sibling. This module
# deliberately keeps its own format_report/format_json/_build_parser/main, for
# the same reasons audit_wiped_metadata_files.py does: a fourth exit code
# (3 = roots resolved but every one failed to audit, kept distinct from 0 per
# docs/legibility/design-invariants.md's no-silent-fail-soft rule) and
# object-shaped rather than array-shaped JSON.
#
# IMPORT-RESOLUTION CONTRACT: _task_db_scan.py must stay a flat sibling in
# scripts/, and this script must NEVER be invoked via `python -m` — the CLI
# tests shell out to the script path and resolve this import solely because a
# DIRECTLY-EXECUTED script puts its own directory at sys.path[0].
from _task_db_scan import discover_project_roots, tasks_db_path  # noqa: F401

# The curator verdict this audit is about. The sibling verdict is 'create',
# which files a NEW task and wipes nothing.
CURATOR_ACTION_COMBINE = "combine"


class CombineTarget(NamedTuple):
    """One task bearing the curator-combine signature, as stored in tasks.db.

    ``metadata_keys`` is the CURRENT live key set. On an unremediated combine
    target it is exactly the three keys the combine path writes
    (``curator_action``, ``curator_justification``, ``combined_at``); every
    key a comparison source expected but that is ABSENT from this tuple is a
    finding.

    A NamedTuple rather than a dataclass, following the precedent: ``_asdict()``
    feeds the JSON writer and ``_replace()`` feeds filtering.
    """

    tag: str
    task_id: int
    status: str
    metadata_keys: tuple[str, ...]


def _decode_metadata(raw: object) -> dict:
    """Decode a raw ``metadata`` blob into a dict, degrading to ``{}``.

    Mirrors :func:`audit_wiped_metadata_files._decode_files`. Degrades for
    NULL, an empty string, malformed JSON, or a payload that decodes to
    anything other than a dict (a list, a bare scalar, ``null``). A corrupt
    metadata blob is data to be skipped, never a reason to abort a sweep over
    thousands of tasks.
    """
    if not raw or not isinstance(raw, (str, bytes)):
        return {}
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def load_combine_targets(tasks_db_path: str) -> dict[tuple[str, int], CombineTarget]:
    """Load every curator-combined task from *tasks_db_path*, keyed by ``(tag, id)``.

    Selects only rows whose decoded ``metadata.curator_action`` is the string
    ``'combine'`` — a wrong-typed value is corrupt data and is skipped, and
    the sibling ``'create'`` verdict wipes nothing so it is not a target.

    Keyed by the full ``(tag, id)`` primary key — the live DB uses a single
    ``master`` tag, but the schema permits the same numeric id under two tags
    and collapsing them would silently merge two distinct tasks.

    Opens the database via a read-only URI (``mode=ro``) so the load is
    structurally incapable of mutating live task records even while
    fused-memory holds the same file open in WAL mode. Closed in a
    ``try/finally`` and never a ``with`` block — a sqlite3 ``with`` is a
    TRANSACTION, not a close.
    """
    targets: dict[tuple[str, int], CombineTarget] = {}
    conn = sqlite3.connect(f"file:{tasks_db_path}?mode=ro", uri=True)
    try:
        cursor = conn.execute("SELECT tag, id, status, metadata FROM tasks")
        for tag, task_id, status, metadata in cursor:
            payload = _decode_metadata(metadata)
            if payload.get("curator_action") != CURATOR_ACTION_COMBINE:
                continue
            targets[(tag, task_id)] = CombineTarget(
                tag=tag,
                task_id=task_id,
                status=status,
                metadata_keys=tuple(payload),
            )
    finally:
        conn.close()
    return targets


# ---------------------------------------------------------------------------
# Comparison source (1) — the creating ticket.
#
# The curator files every task through a reconciliation ticket, and the
# ticket's candidate payload is a SUBMIT-TIME SNAPSHOT of the metadata the
# task was created with. A status='created' row is therefore the closest thing
# to a pre-combine record of what the task's metadata held.
# ---------------------------------------------------------------------------

# Ticket status meaning "the curator actually filed this task". The other
# statuses ('pending', 'combined', 'dropped', ...) never produced the task
# whose metadata this audit is reconstructing.
TICKET_STATUS_CREATED = "created"


def tickets_db_path(project_root: str) -> Path:
    """``<root>/data/reconciliation/tickets.db`` — the curator's ticket store.

    NOTE: tickets.db lives in the MAIN checkout, not in a task worktree.
    """
    return Path(project_root) / "data" / "reconciliation" / "tickets.db"


def load_ticket_expectations(tickets_db_path: str, project_id: str) -> dict[str, dict]:
    """Load submit-time metadata per task id from *tickets_db_path*.

    Returns ``{str(task_id): submit_metadata_dict}`` built from each created
    ticket's ``candidate_json['metadata']``.

    THE ``project_id`` PREDICATE IS LOAD-BEARING — DO NOT DROP IT AS
    REDUNDANT. Measured read-only on the live store: task_id ``'3157'`` has a
    ``project_id='reify'`` created-row sitting right next to a
    ``project_id='dark_factory'`` row for the same id. A task_id-only query
    would silently import ANOTHER PROJECT's submit payload and then report
    every dark_factory key that payload lacks as LOST — a wall of confident
    false positives. Bound as a parameter, never string-formatted.

    ``task_id`` is a TEXT column, so results are keyed by ``str(task_id)``;
    callers joining from tasks.db's INTEGER id must convert. A row is skipped
    (never raised on) when its ``task_id`` is NULL, its ``candidate_json`` is
    malformed or decodes to a non-dict, or its ``metadata`` is absent or not a
    dict — one corrupt ticket cannot abort a sweep.

    Returns ``{}`` rather than raising when the file is absent; the caller
    records that as a coverage fact, not as a clean result.
    """
    if not Path(tickets_db_path).exists():
        return {}

    expectations: dict[str, dict] = {}
    conn = sqlite3.connect(f"file:{tickets_db_path}?mode=ro", uri=True)
    try:
        cursor = conn.execute(
            "SELECT task_id, candidate_json FROM tickets "
            "WHERE project_id = ? AND status = ?",
            (project_id, TICKET_STATUS_CREATED),
        )
        for task_id, candidate_json in cursor:
            if task_id is None:
                continue
            candidate = _decode_metadata(candidate_json)
            metadata = candidate.get("metadata")
            if not isinstance(metadata, dict):
                continue
            expectations[str(task_id)] = metadata
    finally:
        conn.close()
    return expectations
