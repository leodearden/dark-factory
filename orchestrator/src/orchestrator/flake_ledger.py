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
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True)
class DebtRow:
    """The single ``flake_debt`` row for one test — its current cycle plus the
    prior-cycle fields that feed η's recurrence trigger."""

    test_id: str
    project_id: str
    opened_at: str
    resolved_at: str | None
    owner_task_id: str | None
    open_count: int
    prior_resolved_at: str | None
    prior_resolving_commit: str | None
    last_occurrence_at: str


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
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = _connect(db_path)
        try:
            conn.executescript(_SCHEMA)
        finally:
            conn.close()
    except Exception:
        logger.warning('flake_ledger: schema init failed for %s', db_path, exc_info=True)


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
    # §8: EMPTY test_ids is legal only for `unconfirmable`.  An unconfirmable
    # observation that resolved no node-ids is still COUNTED, under the sentinel —
    # θ's class-1 health check is an unconfirmable RATE, so dropping the row would
    # make "could not even determine which tests failed" invisible.  UNKNOWN_TEST_ID
    # must never reach `open_debt`: it names no test, so it can own no de-flake task.
    if s.test_ids:
        test_ids = s.test_ids
    elif s.verdict is FlakeVerdict.unconfirmable:
        test_ids = (UNKNOWN_TEST_ID,)
    else:
        # A confirmed verdict about zero tests is meaningless.  Degrade loudly rather
        # than accept it: a sentinel row here would corrupt the denominator θ's
        # unconfirmable rate divides by.
        logger.warning(
            'flake_ledger: %s verdict with empty test_ids at call_site=%s; dropping '
            '(a confirmed verdict about zero tests is meaningless)',
            s.verdict.value,
            s.call_site,
        )
        return

    try:
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
            for test_id in test_ids
        ]
        conn = _connect(db_path)
        try:
            # OR IGNORE, against the `idx_flake_occurrence_dedup` UNIQUE index, is §8.3's
            # idempotency key `(test_id, observed_at, call_site)` enforced declaratively.
            # Deliberately NOT left to the catch-all: a plain INSERT would (1) abort the
            # rest of the batch on the first duplicate, silently LOSING the genuinely new
            # rows beside it, and (2) turn every legitimate merge-path retry into a loud
            # warning, training operators to ignore the log line B12 relies on. One
            # executemany in one transaction, so the batch still lands atomically.
            conn.executemany(
                'INSERT OR IGNORE INTO flake_occurrence '
                '(observed_at, test_id, project_id, verdict, call_site, runner, '
                ' merge_sha, task_id, psi_cpu_some10, detail) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                rows,
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        logger.warning(
            'flake_ledger: failed to record occurrence for %d test(s) at call_site=%s',
            len(test_ids),
            s.call_site,
            exc_info=True,
        )
        return None


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


def read_occurrences(
    db_path: Path,
    *,
    test_id: str | None = None,
    since: str | None = None,
) -> list[FlakeOccurrenceRow]:
    """Recorded occurrences, oldest first, optionally filtered.

    Args:
        test_id: Only this test's observations — the recurrence chain for one test.
        since: Only observations at or after this ISO-8601 UTC stamp.  INCLUSIVE at
            the boundary.  The comparison is lexicographic on a TEXT column, which is
            sound because ISO-8601 UTC strings with a fixed offset sort in the same
            order as they do chronologically; ``idx_flake_occurrence_observed`` serves
            the window scan.

    Ordering is part of the contract, not incidental: ``observed_at`` then ``id``, so
    the sequence reads chronologically even though rows arrive out of order (a remote
    observation is written on the dispatcher after a local one that followed it).
    """
    try:
        ensure_schema(db_path)
        # Values are BOUND, never interpolated — a test_id is an arbitrary pytest node-id
        # arriving from a remote host.
        clauses: list[str] = []
        params: list[str] = []
        if test_id is not None:
            clauses.append('test_id = ?')
            params.append(test_id)
        if since is not None:
            clauses.append('observed_at >= ?')
            params.append(since)
        where = f' WHERE {" AND ".join(clauses)}' if clauses else ''

        conn = _connect(db_path)
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f'SELECT * FROM flake_occurrence{where} ORDER BY observed_at, id', params
            ).fetchall()
        finally:
            conn.close()
        return [_to_occurrence_row(r) for r in rows]
    except Exception:
        logger.warning(
            'flake_ledger: failed to read occurrences from %s',
            db_path,
            exc_info=True,
        )
        return []


def _to_debt_row(row: sqlite3.Row) -> DebtRow:
    return DebtRow(
        test_id=row['test_id'],
        project_id=row['project_id'],
        opened_at=row['opened_at'],
        resolved_at=row['resolved_at'],
        owner_task_id=row['owner_task_id'],
        open_count=row['open_count'],
        prior_resolved_at=row['prior_resolved_at'],
        prior_resolving_commit=row['prior_resolving_commit'],
        last_occurrence_at=row['last_occurrence_at'],
    )


def read_debt(db_path: Path, test_id: str) -> DebtRow | None:
    """The debt row for *test_id*, or ``None`` if the test has never been suppressed.

    A primary-key lookup — which is exactly the shape §5.3 says makes η's recurrence
    trigger a lookup rather than a scan.  RESOLVED rows are returned too: they are
    retained deliberately (§5.2) because the recurrence trigger reads them.
    """
    try:
        ensure_schema(db_path)
        conn = _connect(db_path)
        try:
            conn.row_factory = sqlite3.Row
            row = conn.execute('SELECT * FROM flake_debt WHERE test_id = ?', (test_id,)).fetchone()
        finally:
            conn.close()
        return _to_debt_row(row) if row is not None else None
    except Exception:
        logger.warning(
            'flake_ledger: failed to read debt for test_id=%s',
            test_id,
            exc_info=True,
        )
        return None


async def open_debt(
    db_path: Path,
    project_id: str,
    test_id: str,
    *,
    task_client: Any = None,
    now: datetime | None = None,
) -> DebtRow | None:
    """Open (or advance) the single ``flake_debt`` row for *test_id* (PRD §8.3).

    Returns the resulting row, or ``None`` if the ledger was unavailable — the never-
    raises invariant and a ``-> DebtRow`` return type are in direct conflict, so the
    honest degrade is ``None`` rather than a fabricated row, and consumers must handle
    ledger unavailability explicitly.

    ``task_client`` is accepted for SIGNATURE STABILITY and is UNUSED in α.  Task ζ
    adds the invariant enforcement here — re-corroborate ``owner_task_id``'s live
    status and file a de-flake task if none is non-terminal — and declaring the
    parameter (and the ``async`` colour) now means ζ never has to churn ``await`` at
    ε's merge-path call sites.  The coupling rule ζ inherits: the ledger READS task
    status but never WRITES it, except for the initial filing.

    ``UNKNOWN_TEST_ID`` must never be passed here: a sentinel names no test, so it can
    own no de-flake task.

    *now* defaults to the current UTC time; it is injectable so §5.4's ledger-owned
    timestamps are deterministically testable.
    """
    try:
        stamp = (now or datetime.now(UTC)).isoformat()
        ensure_schema(db_path)
        conn = _connect(db_path)
        try:
            # ONE statement, deliberately.  SQL evaluates every SET right-hand side against
            # the PRE-UPDATE row, so all three CASE guards observe the OLD `resolved_at`
            # even though the same clause sets it to NULL — which is what lets a re-entry
            # (was resolved) and an ordinary repeat (still open) be distinguished without a
            # prior SELECT.  Verified empirically on SQLite 3.50.4: open_count=2,
            # prior_resolved_at=the old resolved_at, resolved_at=NULL, opened_at advanced.
            #
            # Splitting this into SELECT-then-UPDATE would reintroduce exactly the
            # read-modify-write race §5.1 chose SQLite to avoid, inside the ledger itself:
            # two merge lanes suppressing the same test could interleave and lose a cycle.
            #
            # `prior_resolving_commit` is deliberately NOT touched — `resolve_debt` already
            # wrote it, and it must survive the re-open verbatim for η's
            # regressed_after_resolution citation.
            conn.execute(
                'INSERT INTO flake_debt (test_id, project_id, opened_at, open_count, '
                ' last_occurrence_at) '
                'VALUES (?, ?, ?, 1, ?) '
                'ON CONFLICT(test_id) DO UPDATE SET '
                '    last_occurrence_at = excluded.last_occurrence_at, '
                '    open_count        = open_count '
                '                        + (CASE WHEN resolved_at IS NOT NULL THEN 1 ELSE 0 END), '
                '    opened_at         = CASE WHEN resolved_at IS NOT NULL '
                '                        THEN excluded.opened_at ELSE opened_at END, '
                '    prior_resolved_at = CASE WHEN resolved_at IS NOT NULL '
                '                        THEN resolved_at ELSE prior_resolved_at END, '
                '    resolved_at       = NULL',
                (test_id, project_id, stamp, stamp),
            )
            conn.commit()
        finally:
            conn.close()
        return read_debt(db_path, test_id)
    except Exception:
        logger.warning(
            'flake_ledger: failed to open debt for test_id=%s (project_id=%s)',
            test_id,
            project_id,
            exc_info=True,
        )
        return None


async def resolve_debt(
    db_path: Path,
    project_id: str,
    test_id: str,
    *,
    resolving_commit: str | None,
    now: datetime | None = None,
) -> None:
    """Close *test_id*'s current debt cycle (PRD §8.3).  Called when the owning task
    goes terminal.

    The row is RETAINED, not deleted (§5.2) — η's recurrence trigger reads resolved
    rows, so reaping one would silently disarm it.  Resolution's observable effect is
    that the test leaves :func:`list_open_debt` while :func:`read_debt` still finds it.

    ``prior_resolving_commit`` is written HERE, not on the next re-open: after a
    resolution there is no "current" cycle for it to describe, and writing it now is
    what lets step-18's re-entry carry it forward untouched for η's
    ``regressed_after_resolution`` citation.

    The lookup keys on ``test_id`` ALONE — §5.3: runs.db is per-project, so test_id is
    the primary key.  *project_id* is accepted per the §8.3 signature and used only in
    log messages; it is NOT a dropped filter.

    A zero-rowcount UPDATE (no debt for this test) is a legitimate no-op, not an error.
    ``async`` for the same forward-compat reason as :func:`open_debt` — η adds the
    recurrence escalation inside this function.
    """
    try:
        stamp = (now or datetime.now(UTC)).isoformat()
        ensure_schema(db_path)
        conn = _connect(db_path)
        try:
            conn.execute(
                'UPDATE flake_debt SET resolved_at = ?, prior_resolving_commit = ? '
                'WHERE test_id = ?',
                (stamp, resolving_commit, test_id),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        logger.warning(
            'flake_ledger: failed to resolve debt for test_id=%s (project_id=%s)',
            test_id,
            project_id,
            exc_info=True,
        )
        return None


def list_open_debt(db_path: Path) -> list[DebtRow]:
    """Every test currently in debt, oldest first.

    Ordering is part of the contract (``opened_at`` then ``test_id``) because ι prints
    this list.
    """
    try:
        ensure_schema(db_path)
        conn = _connect(db_path)
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                'SELECT * FROM flake_debt WHERE resolved_at IS NULL ORDER BY opened_at, test_id'
            ).fetchall()
        finally:
            conn.close()
        return [_to_debt_row(r) for r in rows]
    except Exception:
        logger.warning(
            'flake_ledger: failed to list open debt from %s',
            db_path,
            exc_info=True,
        )
        return []
