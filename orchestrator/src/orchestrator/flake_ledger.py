"""Flake ledger — occurrence trail and debt set (plans/flake-ledger-prd.md, task α).

Two tables in the project's existing ``data/orchestrator/runs.db``:

- ``flake_occurrence`` — append-only; MANY rows per test.  The evidence trail: every
  time a discriminator judged a failing test, what it concluded, where it ran, and
  what the host pressure was at that moment.
- ``flake_debt`` — ONE row per test.  The set the invariant governs: a test that has
  been suppressed owes a de-flake task, and the row carries the ledger-owned clock
  (``opened_at``/``resolved_at``) that ``.taskmaster/tasks/tasks.db`` cannot supply
  (§5.4 — that store has no ``created_at``, and ``planning_mode=True`` bypasses the
  curator ticket store entirely).

Living in the shared ``orchestrator`` package, on a DB every client project already
has, is what hands this facility to all 8 projects with ZERO provisioning: the tables
appear on first write via ``CREATE TABLE IF NOT EXISTS``, with no migration to sequence
(§5.2).

TWO BINDING CONTRACTS, both machine-checked:

1. **Writes go only through this API** — ``record_flake_occurrence`` / ``open_debt`` /
   ``resolve_debt``, never raw SQL at a call site (§5.2).  Four call sites hand-rolling
   INSERTs is exactly how the two merge gates would drift into different notions of the
   verdict vocabulary.
2. **No public entry point ever raises** (§8.3, boundary row B12).  The merge path has
   no ``VerifyInfraError`` handler, so an uncaught raise here stalls the merge queue —
   a ledger failure must never fail a verify or a merge.  Every entry point degrades to
   an honest value (``None`` / ``[]``) and logs LOUDLY with ``exc_info``; it never fails
   silently.  This mirrors ``chronic_flake``'s catch-all-defensive contract.

RETENTION — a named, accepted position, not an oversight (§5.2, §11 Q1).  ``flake_debt``
is bounded by construction (one row per test); resolved rows are retained DELIBERATELY
because the recurrence trigger reads them, so deleting one would silently disarm it.
``flake_occurrence`` is append-only and UNPRUNED, bounded exactly the way ``events`` is
— i.e. it isn't.  That is consistent with the rest of ``runs.db``, where nothing is
pruned.  Do NOT add bespoke pruning for one table here; revisit repo-wide when ``events``
retention is addressed.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from shared.sqlite_sync_base import apply_full_durability_pragmas_sync

logger = logging.getLogger(__name__)

# Sentinel test_id for an `unconfirmable` observation that identified no node-ids.
# The observation is still COUNTED — θ's class-1 health check is an `unconfirmable`
# RATE, so dropping the row because no node-ids were resolved would make "could not
# even determine which tests failed" invisible, reproducing the exact blindness this
# PRD exists to end.  The angle-bracket form cannot collide with a real pytest
# node-id.  It must NEVER be passed to `open_debt`: a sentinel names no test, so it
# can own no de-flake task.
UNKNOWN_TEST_ID = '<unknown>'

# PRD §5.3 verbatim.  Additive `CREATE TABLE IF NOT EXISTS` with no `PRAGMA
# user_version` ladder — the event_store.py:23-41 / run_store.py:18-76 idiom, and
# what makes the ledger a well-behaved fifth owner of an existing runs.db.
_SCHEMA = """\
CREATE TABLE IF NOT EXISTS flake_occurrence (   -- append-only; many rows per test; the evidence trail
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    observed_at     TEXT    NOT NULL,   -- ISO-8601 UTC, LEDGER-OWNED (§5.4)
    test_id         TEXT    NOT NULL,   -- pytest node-id, or script-suite test name
    project_id      TEXT    NOT NULL,   -- denormalised: the dashboard aggregates across runs.db files
    verdict         TEXT    NOT NULL,   -- passes_in_isolation | fails_in_isolation | unconfirmable
    call_site       TEXT    NOT NULL,   -- merge_gate | main_probe | chronic_marker
    runner          TEXT,               -- 'local' | remote host name — WHERE the discriminator ran
    merge_sha       TEXT,
    task_id         TEXT,
    psi_cpu_some10  REAL,               -- host pressure AT observation (shared.psi), NULL if read_ok=False
    detail          TEXT    DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS flake_debt (         -- ONE row per test; the set the invariant governs
    test_id                 TEXT PRIMARY KEY,   -- runs.db is per-project, so test_id alone is the key
    project_id              TEXT NOT NULL,
    opened_at               TEXT NOT NULL,      -- LEDGER-OWNED
    resolved_at             TEXT,               -- NULL while open
    owner_task_id           TEXT,               -- the non-terminal de-flake task (§5.5)
    open_count              INTEGER NOT NULL DEFAULT 1,
    prior_resolved_at       TEXT,               -- previous cycle — feeds the recurrence trigger
    prior_resolving_commit  TEXT,               -- cited verbatim in the regressed_after_resolution L2
    last_occurrence_at      TEXT NOT NULL
);

-- §8.3's idempotency key, enforced declaratively: a replayed merge-path write must
-- land one row, not two.
CREATE UNIQUE INDEX IF NOT EXISTS idx_flake_occurrence_dedup
    ON flake_occurrence(test_id, observed_at, call_site);

-- Serves the windowed rate reads (unconfirmable rate, suppressions-in-window).
CREATE INDEX IF NOT EXISTS idx_flake_occurrence_observed
    ON flake_occurrence(observed_at);
"""


class FlakeVerdict(StrEnum):
    """What a discriminator concluded about a failing test (PRD §8).

    §5.5: the vocabulary names the OBSERVATION, never the remedy — ``passes_in_isolation``,
    never ``flaky_test: true``.  One discriminator serves both merge gates, so this is
    what keeps them from drifting into different notions of "passes in isolation".
    """

    passes_in_isolation = 'passes_in_isolation'  # re-ran clean, isolated + serial → suppressible
    fails_in_isolation = 'fails_in_isolation'  # re-ran and failed → a real red
    unconfirmable = 'unconfirmable'  # could not map node-ids / could not re-run


@dataclass(frozen=True)
class FlakeSuppression:
    """Discriminator output. Produced WHEREVER the worktree is (local or remote);
    consumed ONLY on the dispatcher. Rides VerifyResult across the wire."""

    verdict: FlakeVerdict
    test_ids: tuple[str, ...]  # node-ids examined — EMPTY is legal only for `unconfirmable`
    observed_at: str  # ISO-8601 UTC, stamped by the DISCRIMINATOR at observation
    call_site: str  # 'merge_gate' | 'main_probe' | 'chronic_marker'
    runner: str  # 'local' | remote host name — WHERE the re-run ran
    psi_cpu_some10: float | None  # shared.psi at observation; None when read_ok is False
    unconfirmable_reason: str | None  # populated iff verdict is unconfirmable


@dataclass(frozen=True)
class FlakeOccurrenceRow:
    """One ``flake_occurrence`` row — a single test's observation at a single moment.

    A ``FlakeSuppression`` naming N tests fans out to N of these.
    """

    id: int
    observed_at: str
    test_id: str
    project_id: str
    verdict: str
    call_site: str
    runner: str | None
    merge_sha: str | None
    task_id: str | None
    psi_cpu_some10: float | None
    detail: str


def ledger_db_path(project_root: Path) -> Path:
    """The project's ``runs.db``.

    One spelling of a literal that is hand-built at 5+ sites today (harness.py:2119 and
    friends) with no shared helper, so the ledger's consumers do not each re-derive it.
    """
    return project_root / 'data' / 'orchestrator' / 'runs.db'


def _connect(db_path: Path) -> sqlite3.Connection:
    """Open a ledger connection with the full durability pragma triad applied.

    Byte-identical to the two existing ``runs.db`` owners (event_store.py:512,
    run_store.py:90) — no new pragma policy and no second durability story.  The 5s
    ``busy_timeout`` is what absorbs concurrent merge-lane contention, in place of a
    hand-rolled lock.
    """
    conn = sqlite3.connect(str(db_path))
    apply_full_durability_pragmas_sync(conn, busy_timeout_ms=5000)
    return conn


def ensure_schema(db_path: Path) -> None:
    """Create both ledger tables and their indexes if absent.  Additive and idempotent.

    Called by every public entry point rather than once in a constructor: the API is
    free functions taking a ``db_path`` (§8.3) because the recorder is called from
    stateless merge-path sites and a CLI, neither of which holds a long-lived store
    object.  ``CREATE TABLE IF NOT EXISTS`` is cheap and merge-path call frequency is a
    handful per hour, so this is deliberately NOT memoized — a per-path cache would go
    stale the moment the DB file is replaced or removed.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(db_path)
    try:
        conn.executescript(_SCHEMA)
    finally:
        conn.close()


def record_flake_occurrence(
    db_path: Path,
    project_id: str,
    s: FlakeSuppression,
    *,
    merge_sha: str | None,
    task_id: str | None,
) -> None:
    """Append one ``flake_occurrence`` row per test named in *s* (PRD §8.3).

    ``observed_at`` comes from *s* — the DISCRIMINATOR stamps it at observation, which
    is both semantically right for the remote path (the observation happens on the
    remote host; the write happens later on the dispatcher) and mechanically required:
    §8.3's idempotency key is ``(test_id, observed_at, call_site)``, and a write-time
    stamp would make every retry a distinct row.
    """
    ensure_schema(db_path)
    detail = (
        json.dumps({'unconfirmable_reason': s.unconfirmable_reason})
        if s.unconfirmable_reason
        else '{}'
    )
    # Columns are named explicitly, never a positional `INSERT INTO t VALUES (...)`:
    # run_store.py:53-58's ALTER-parity note — a migrated DB and a fresh one must both
    # accept the same statement.
    rows = [
        (
            s.observed_at,
            test_id,
            project_id,
            s.verdict.value,
            s.call_site,
            s.runner,
            merge_sha,
            task_id,
            s.psi_cpu_some10,  # bound directly, so None becomes SQL NULL, never 0.0
            detail,
        )
        for test_id in s.test_ids
    ]
    conn = _connect(db_path)
    try:
        conn.executemany(
            'INSERT INTO flake_occurrence '
            '(observed_at, test_id, project_id, verdict, call_site, runner, '
            ' merge_sha, task_id, psi_cpu_some10, detail) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def _to_occurrence_row(row: sqlite3.Row) -> FlakeOccurrenceRow:
    return FlakeOccurrenceRow(
        id=row['id'],
        observed_at=row['observed_at'],
        test_id=row['test_id'],
        project_id=row['project_id'],
        verdict=row['verdict'],
        call_site=row['call_site'],
        runner=row['runner'],
        merge_sha=row['merge_sha'],
        task_id=row['task_id'],
        psi_cpu_some10=row['psi_cpu_some10'],
        detail=row['detail'],
    )


def read_occurrences(db_path: Path) -> list[FlakeOccurrenceRow]:
    """Every recorded occurrence, oldest first."""
    ensure_schema(db_path)
    conn = _connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute('SELECT * FROM flake_occurrence ORDER BY id').fetchall()
    finally:
        conn.close()
    return [_to_occurrence_row(r) for r in rows]
